#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


from typing import Tuple, Union, Callable, Optional
from functools import partial
from math import inf


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch import Tensor
from torch.nn.attention import SDPBackend
import torch.nn.functional as F


# # Utils

# In[2]:


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-4,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class FlashNormLinear(nn.Linear):
    def __init__(self, in_features, out_features, eps=1e-6, bias=True, rank=None):
        super().__init__(in_features, out_features, bias)
        self.rms_weight = nn.Parameter(torch.ones(in_features))
        self.eps = eps
        _ = rank  # Not used for now

    def forward(self, x: Tensor) -> Tensor:
        # RMS normalize
        ms = (x**2).mean(dim=-1, keepdim=True) + self.eps
        x = x / (ms**0.5)
        # Fuse scaling into weight
        scaled_weight = self.weight * self.rms_weight.unsqueeze(0)
        return F.linear(x, scaled_weight, self.bias)


class FlashMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Only replace fc1 with FlashNormLinear
        self.fc1 = FlashNormLinear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)  # includes RMSNorm internally
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LinearGroupedLowRank(nn.Module): 
    def __init__(self, d, t): 
        super().__init__()
        self.dense = nn.Linear(d, d)
        self.act = nn.GELU() 
        self.grouped = nn.Conv1d(d, t*d, 1, groups=t)  # [B, d, N] -> [B, t*d, N]

    def forward(self, x):    # x: [B, N, d]
        x = self.dense(x)   # [B, N, d]
        x = self.act(x)     # [B, N, d]
        x = x.transpose(1, 2)  # [B, d, N]
        x = self.grouped(x)    # [B, t*d, N]
        x = x.transpose(2, 1)  # [B, N, t*d]
        return x

class ContextAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        bank_size: int = 16,
        bank_depth = 1,
        num_heads: int = 6,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim must be divisible by num_heads"
        self.dim = dim
        self.n_h = num_heads
        self.h_d = dim // num_heads
        self.bank_size = bank_size
        self.bank_depth = bank_depth
        self.bank = nn.Parameter(torch.randn(1, bank_depth, bank_size, num_heads, self.h_d))
        
        self.proj_x = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.cls_to_w = nn.Sequential(nn.Linear(dim, dim),
                                      nn.GELU(),
                                      nn.Linear(dim, bank_depth * num_heads))
        self.bank_norm = norm_layer(dim) 
        self.norm_ctx = norm_layer(dim)
        self.proj_ctx = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_out = nn.Linear(dim, dim, bias=proj_bias)

        self.attn_drop = attn_drop
        self.out_drop = nn.Dropout(proj_drop)
        
    def sdpa(self, q, k, v, gqa=False):
        dropout_p = self.attn_drop if self.training else 0
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, enable_gqa=gqa)

    def _forward(self, x):
        B, N, D = x.shape 
        K, M, H, d = self.bank_depth, self.bank_size, self.n_h, self.h_d

        x_q, x_k, x_v = self.proj_x(x).view(B, N, 3, H, d).permute(2, 0, 3, 1, 4) # 3[B, H, N, d]
        w = torch.softmax(self.cls_to_w(x[:, 0]).view(B, K, H), dim=1)
        w = w.view(B, K, 1, H, 1)
        bank = self.bank.expand(B, -1, -1, -1, -1)
        weighted_bank = (bank * w).sum(dim=1).view(B, M, D)
        Q = self.bank_norm(weighted_bank).view(B, M, H, d).transpose(1, 2)
        
        ctx_attn = self.sdpa(Q, x_k, x_v) # [B, H, M * T, d]
        ctx_norm = self.norm_ctx(ctx_attn.transpose(1, 2).reshape(B, M, D)) # [B, MT, D]
        ctx_k = self.proj_ctx(ctx_norm).view(B, M, H, d).transpose(1, 2) # [B, H/G, GMT, d]
        ctx_v = ctx_norm.view(B, M, H, d).transpose(1, 2) # [B, H/G, GMT, d]
            
        x_attn = self.sdpa(x_q, ctx_k, ctx_v, gqa = True) # [B, H, N, d]
        x_attn = x_attn.transpose(1, 2).reshape(B, N, D) # [B, N, D]
        return self.out_drop(self.proj_out(x_attn)) # [B, N, D]
    
    def forward(self, x: Tensor, threshold=None) -> Tensor:
        forw_fn = self._forward
        
        if threshold is None:
            print(
                "Warning: SDP kernel threshold is set to None, using default forward method"
            )
            return forw_fn(x)

        if x.size(1) > threshold:
            sdp_kernel = SDPBackend.FLASH_ATTENTION
        else:
            sdp_kernel = SDPBackend.EFFICIENT_ATTENTION

        with torch.nn.attention.sdpa_kernel(sdp_kernel):
            return forw_fn(x)


# # Block
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        layerscale: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        flash_mlp: bool = False,
        bank_size=64,
        bank_depth=1,
        sdp_threshold=None,
    ) -> None:
        super().__init__()
        self.sdp_threshold = sdp_threshold
        self.norm1 = norm_layer(dim) if not flash_mlp else nn.Identity()
        self.attn = ContextAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            bank_size=bank_size,
            bank_depth=bank_depth
        )
        self.ls1 = (
            LayerScale(dim, init_values=layerscale) if layerscale else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim) if not flash_mlp else nn.Identity()

        ffn_layer = FlashMlp if flash_mlp else Mlp
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=layerscale) if layerscale else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path


    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), threshold=self.sdp_threshold))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path2(ffn_residual_func(x))  
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x

def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)

# # Patch Embed

# In[7]:


# # Patch emb
def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int), x
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 96,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.n_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x


# # Vit

# In[ ]:


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module)
    return module


def init_weights_vit_timm(module: nn.Module):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


class ContextMoeViTv1(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=True,
        layerscale=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        token_drop=0,
        n_registers=0,
        bank_size=64,
        bank_depth=1,
        flash_mlp=False,
        return_cls_only=True,
        sdp_threshold=inf,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            layerscale (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
        """
        super().__init__()
        assert bank_size >= 1, "bank_size needs to be 1 or more"
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.p_token_drop = token_drop
        self.n_registers = bank_size - 1
        self.bank_size = bank_size
        self.return_cls_only = return_cls_only
        _ = n_registers

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.n_patches = self.patch_embed.n_patches
        self.tok_cls = nn.Parameter(torch.zeros(1, 1 + self.n_registers, embed_dim))
        num_pos_emb = 1 + self.n_registers + self.n_patches
        self.tok_pos_emb = nn.Parameter(torch.zeros(1, num_pos_emb, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        blocks_list = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                layerscale=layerscale,
                bank_size=bank_size,
                bank_depth=bank_depth,
                flash_mlp=flash_mlp,
                sdp_threshold=sdp_threshold,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.tok_pos_emb, std=0.02)
        nn.init.normal_(self.tok_cls, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def prepare_tokens(self, x):
        with torch.profiler.record_function("Patch Embed"):
            x = self.patch_embed(x)
        with torch.profiler.record_function("prepare Tokens"):
            cls_token = self.tok_cls.expand(x.size(0), -1, -1)
            x = torch.cat((cls_token, x), dim=1) + self.tok_pos_emb
        with torch.profiler.record_function("Token Drop"):
            x = self.token_drop(x)
        return x

    def token_drop(self, x):
        if not self.p_token_drop or not self.training:
            return x
        patch_idx = 1 + self.n_registers 
        non_patches, patches = x[:, :patch_idx, :], x[:, patch_idx:, :]
        num_keep = int((1 - self.p_token_drop) * self.n_patches)
        r = torch.rand(x.size(0), self.n_patches, device=x.device)
        batch_perms = torch.argsort(r, dim=1)[:, :num_keep]  # [B, num_keep]
        batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1)  # [B, 1]
        patches = patches[batch_idx, batch_perms]  # [B, num_keep, D]
        return torch.cat([non_patches, patches], dim=1)  # [B,1+num_keep,D]

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        with torch.profiler.record_function("Final Norm"):
            out = self.norm(x)
        return out[:, 0, :] if self.return_cls_only else out


# # END
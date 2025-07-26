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


class CLSControlledFusion(nn.Module):
    def __init__(self, dim, query_bank, out_norm, hidden_mult=4):
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_mult * dim
        self.weight_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.query_bank = query_bank
        self.out_norm = out_norm

    def forward(self, cls_token, ctx_mem, bank_mem, B):
        bank = self.query_bank.expand(B, -1, -1)
        weights = F.softmax(self.weight_proj(cls_token), dim=-1)
        w = weights.unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]
        sources = torch.stack([bank, ctx_mem, bank_mem], dim=1)  # [B, 3, M, D]
        return self.out_norm((w * sources).sum(dim=1))


class ContextAttentionMeM(nn.Module):
    def __init__(
        self,
        dim: int,
        bank_size: int,
        num_heads: int = 6,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        bank_memory=True,
        bank_cls_condition=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.D = D = dim
        self.n_h = num_heads
        self.h_D = dim // num_heads
        self.M = bank_size
        self.sdpa = F.scaled_dot_product_attention

        self.proj_x = nn.Linear(D, 3 * D, bias=qkv_bias)
        self.out_x = nn.Linear(D, D, bias=proj_bias)

        if bank_cls_condition:
            assert bank_memory is True
        self.query_bank = nn.Parameter(torch.zeros(1, bank_size, D))
        self.bank_cls_condition = bank_cls_condition
        self.bank_memory = bank_memory
        self.bank_norm = norm_layer(D)
        if bank_cls_condition:
            self.bank_fusion = CLSControlledFusion(D, self.query_bank, self.bank_norm)

        self.ctx_norm = norm_layer(D)
        t = 3 if bank_memory else 2
        self.proj_ctx = nn.Linear(D, t * D, bias=qkv_bias)

        self.attn_drop_v = attn_drop
        self.out_drop = nn.Dropout(proj_drop)

        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.query_bank, std=0.02)

    def sdpa_w_reshape(self, q, k, v):
        B, _, N, _ = q.size()
        attn = self.sdpa(q, k, v, dropout_p=self.attn_drop_v)
        return attn.transpose(1, 2).reshape(B, N, -1)

    def get_bank_query(self, B, mem=None):
        if mem is None or mem is True:
            return self.query_bank.expand(B, -1, -1)
        return self.bank_norm(self.query_bank + mem)

    def __forward(self, x, mem=None) -> Tensor:
        (B, N, _), M = x.size(), self.M

        with torch.profiler.record_function("Proj: x -> QKV"):
            x_proj = torch.chunk(self.proj_x(x), 3, dim=-1)
            xQ, xK, xV = [y.view(B, N, self.n_h, -1).transpose(1, 2) for y in x_proj]

        with torch.profiler.record_function("Get Bank Query"):
            bankQ = self.get_bank_query(B, mem)
            bankQ = bankQ.view(B, self.M, self.n_h, -1).transpose(1, 2)  # [B, H, M, d]

        with torch.profiler.record_function("Ca: X <- Bank"):
            ctx = self.sdpa_w_reshape(bankQ, xK, xV)  # [B, M, d] O(NMD)

        with torch.profiler.record_function("Proj: ctx -> KV & Mem"):
            *ctx_kv, mem_out = torch.chunk(self.proj_ctx(self.ctx_norm(ctx)), 3, dim=-1)
            ctxK, ctxV = [y.view(B, M, self.n_h, -1).transpose(1, 2) for y in ctx_kv]

        with torch.profiler.record_function("Ca: X <- ctx"):
            x_attn = self.sdpa_w_reshape(xQ, ctxK, ctxV)  # [B, N, D] O(NMD)

        with torch.profiler.record_function("Proj: X -> out"):
            x_out = self.out_drop(self.out_x(x_attn))  # [B, N, D] O(ND^2)

        return x_out, mem_out

    def _forward(self, x, mem) -> Tensor:
        (B, N, _), M = x.size(), self.M
        ctx_mem, bank_mem = mem
        bank_out = ctx_out = None

        with torch.profiler.record_function("Proj: x -> QKV"):
            x_proj = torch.chunk(self.proj_x(x), 3, dim=-1)
            xQ, xK, xV = [y.view(B, N, self.n_h, -1).transpose(1, 2) for y in x_proj]

        with torch.profiler.record_function("Get Bank Query"):
            if not self.bank_memory or ctx_mem is True:
                bankQ = self.bank_norm(self.query_bank.expand(B, -1, -1))
                if self.bank_cls_condition:
                    bank_out = bankQ
            elif not self.bank_cls_condition:
                bankQ = self.bank_norm(self.query_bank + ctx_mem)  # [B, M, D]
            else:
                bankQ = bank_out = self.bank_fusion(x[:, 0, :], ctx_mem, bank_mem, B)
            bankQ = bankQ.view(B, self.M, self.n_h, -1).transpose(1, 2)  # [B, H, M, D]

        with torch.profiler.record_function("Ca: X <- Bank"):
            ctx = self.sdpa_w_reshape(bankQ, xK, xV)  # [B, M, D] O(NMD)

        with torch.profiler.record_function("Proj: ctx -> KV & Mem"):
            chunk = 3 if self.bank_memory else 2
            ctx_kv = torch.chunk(self.proj_ctx(self.ctx_norm(ctx)), chunk, dim=-1)
            if self.bank_memory:
                *ctx_kv, ctx_out = ctx_kv
            ctxK, ctxV = [y.view(B, M, self.n_h, -1).transpose(1, 2) for y in ctx_kv]

        with torch.profiler.record_function("Ca: X <- ctx"):
            x_attn = self.sdpa_w_reshape(xQ, ctxK, ctxV)  # [B, N, D] O(NMD)

        with torch.profiler.record_function("Proj: X -> out"):
            x_out = self.out_drop(self.out_x(x_attn))  # [B, N, D] O(ND^2)

        return x_out, (ctx_out, bank_out)

    def forward(self, x: Tensor, mem=None, threshold=None) -> Tensor:
        if threshold is None:
            print(
                "Warning: SDP kernel threshold is set to None, using default forward method"
            )
            return self._forward(x, mem)

        if x.size(1) > threshold:
            sdp_kernel = SDPBackend.FLASH_ATTENTION
        else:
            sdp_kernel = SDPBackend.EFFICIENT_ATTENTION

        with torch.nn.attention.sdpa_kernel(sdp_kernel):
            return self._forward(x, mem)


# # Block

# In[6]:


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
        bank_memory=True,
        bank_cls_condition=False,
        sdp_threshold=None,
    ) -> None:
        super().__init__()
        self.sdp_threshold = sdp_threshold
        self.norm1 = norm_layer(dim) if not flash_mlp else nn.Identity()
        self.attn = ContextAttentionMeM(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            bank_size=bank_size,
            bank_memory=bank_memory,
            bank_cls_condition=bank_cls_condition,
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

    def attn_residual_func(self, x: Tensor, mem) -> Tensor:
        x_out, mem_out = self.attn(self.norm1(x), mem, threshold=self.sdp_threshold)
        return self.ls1(x_out), mem_out

    def ffn_residual_func(self, x: Tensor) -> Tensor:
        return self.ls2(self.mlp(self.norm2(x)))

    def forward(self, x, mem=None) -> Tensor:
        if self.training and self.sample_drop_ratio > 0.1:
            assert False, "not implemented yet"
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=self.attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=self.ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x_out, mem_out = self.attn_residual_func(x, mem)
            x = x + self.drop_path1(x_out)
            x = x + self.drop_path2(self.ffn_residual_func(x))
        else:
            x_out, mem_out = self.attn_residual_func(x, mem)
            x = x + self.ffn_residual_func(x + x_out)
        return x, mem_out


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


class ContextViTv4MeM(nn.Module):
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
        flash_mlp=False,
        return_cls_only=True,
        bank_memory=True,
        bank_cls_condition=False,
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
        assert bank_size >= 1, "n_C needs to be 1 or more"
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.p_token_drop = token_drop
        self.n_registers = n_registers
        self.return_cls_only = return_cls_only
        self.bank_memory = bank_memory

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
                bank_size=bank_size,
                layerscale=layerscale,
                flash_mlp=flash_mlp,
                bank_memory=bank_memory,
                bank_cls_condition=bank_cls_condition,
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
            cls_token = self.tok_cls.expand(x.shape[0], -1, -1)
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
        mem = (True, True) if self.bank_memory else (None, None)
        for blk in self.blocks:
            x, mem = blk(x, mem)
        with torch.profiler.record_function("Final Norm"):
            out = self.norm(x)
        return out[:, 0, :] if self.return_cls_only else out


# # END
#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


from typing import Tuple, Union, Callable, Optional
from functools import partial


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
        self.gamma.no_wd = True

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


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


class ContextAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_registers: int,           # including CLS
        num_prototypes: int = 128,    # K
        num_heads: int = 6,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        num_banks: int = 4,           # M (banks)
        ema_momentum: float = 0.995,   # EMA for centroids
        noise_kw: dict = None,     # exploration noise on sims (train only)
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.D, self.H, self.d = dim, num_heads, dim // num_heads
        self.K, self.R, self.M = num_prototypes, num_registers, num_banks

        # --- Q banks: [M, K, D] (one bank selected per sample) ---
        self.Q_banks = nn.Parameter(torch.randn(self.M, self.K, self.D))

        self.proj_q   = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_ctx = nn.Linear(dim, 2 * dim, bias=proj_bias)
        self.proj_out = nn.Linear(dim, dim,      bias=proj_bias)

        self.attn_drop = attn_drop
        self.out_drop  = nn.Dropout(proj_drop)
        
        if noise_kw is None:
            self.noise_min = self.noise_max = noise_scale = 0
            self.noise_steps = 1
        else:
             self.noise_max = noise_scale = noise_kw["max"]
             self.noise_min = noise_kw["min"]
             self.noise_steps = noise_kw["steps"]

        self.ema_m = ema_momentum
        with torch.no_grad():
            A = torch.randn(self.D, self.M, dtype=torch.float32)
            Q, _ = torch.linalg.qr(A, mode="reduced")      # [D, M]
            C = Q.t().contiguous()                         # [M, D]
            self.register_buffer("centroids", F.normalize(C, dim=-1))     # fp32
            self.register_buffer("sum_buf",  torch.zeros(self.M, self.D)) # fp32
            self.register_buffer("cnt_buf",  torch.zeros(self.M))         # fp32
            self.register_buffer("noise_scale",  torch.tensor(noise_scale))         # fp32
            self.register_buffer("noise_step",  torch.tensor(0, dtype=torch.int64))
            trunc_normal_(self.Q_banks, std=0.02)
            
    @torch.no_grad()
    def noise_schedule(self):
        S = max(1, int(self.noise_steps))                   # avoid div-by-zero
        s = min(int(self.noise_step), S)                    # clamp
        t = s / S                                           # 0..1 progress
        val = self.noise_min + (self.noise_max - self.noise_min) * (1.0 - t)
        self.noise_scale.copy_(self.noise_scale.new_tensor(val))
        self.noise_step.add_(1)
        
        
    @torch.no_grad()
    def ema_update(self, cls_n, idx):
        self.sum_buf.zero_().index_add_(0, idx, cls_n)      # [M, D]
        ones = torch.ones_like(idx, dtype=self.sum_buf.dtype)
        self.cnt_buf.zero_().index_add_(0, idx, ones)       # [M]
        used = self.cnt_buf > 0
        if used.any():
            mean = self.sum_buf[used] / self.cnt_buf[used].unsqueeze(1)
            upd  = F.normalize(self.ema_m * self.centroids[used] + (1 - self.ema_m) * mean, dim=-1)
            self.centroids[used].copy_(upd)
            
    @torch.no_grad()
    def route(self, cls):
        cls_n = F.normalize(cls, dim=-1)                          # dtype = activations (bf16/fp16/fp32)
        cent  = self.centroids.to(cls_n.dtype)                    # cast view for matmul
        sims  = cls_n @ cent.t()                                  # stays in activation dtype
        if self.training:  
            sims_prime = sims + self.noise_scale.to(sims.dtype) * torch.randn_like(sims)
        else:
            sims_prime = sims
        idx = sims_prime.argmax(dim=-1)
        return cls_n.detach().to(torch.float32), idx              # fp32 for EMA

    def sdpa(self, q, k, v):
        p = self.attn_drop if self.training else 0.0
        return F.scaled_dot_product_attention(q, k, v, dropout_p=p)

    def forward(self, x):
        """
        x: [B, N, D] with token order [CLS, REG_1..REG_R, PATCH_1..PATCH_P]
        """
        B, N, D = x.shape
        K, H, d, R = self.K, self.H, self.d, self.R
        xreg = x[:, :R, :]          # [B, R, D]
        xp   = x[:, R:, :]          # [B, P, D]
        P = xp.size(1)
        assert R + P == N

        cls_n, idx = self.route(x[:, 0, :])                               # [B]
        q_ctx      = self.Q_banks.index_select(0, idx)                    # [B, K, D]
        ctx_p      = F.softmax(q_ctx @ xp.transpose(1, 2), dim=-1) @ xp   # [B, K, D]
        ctx        = torch.cat([xreg, ctx_p], dim=1)                      # [B, R+K, D]
        ctx_kv     = self.proj_ctx(ctx).reshape(B, R+K, 2, H, d).permute(2, 0, 3, 1, 4)
        k, v       = ctx_kv[0], ctx_kv[1]                                 # [B, H, R+K, d]
        q          = self.proj_q(x).view(B, N, H, d).transpose(1, 2).contiguous() # [B,H,N,d]
        y          = self.sdpa(q, k, v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        return       self.out_drop(self.proj_out(y)), cls_n, idx          # [B, N, D]
    
# # Block

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - float(drop_prob)              # compile-time constant
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape, dtype=torch.float32).bernoulli_(keep).div_(keep).to(x.dtype)
    return x * mask


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
        num_registers: int,
        num_prototypes=64,
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
        num_banks: int = 4,           # M (banks)
        ema_momentum: float = 0.995,   # EMA for centroids
        noise_kw: dict = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ContextAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_prototypes=num_prototypes,
            num_registers=num_registers,
            num_banks=num_banks,
            ema_momentum=ema_momentum,
            noise_kw=noise_kw
        )
        self.ls1 = (
            LayerScale(dim, init_values=layerscale) if layerscale else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # precompute attention branch once
        y_attn, *cache = self.attn(self.norm1(x))   # (attn_out, cls_n, idx)
        attn_out = self.ls1(y_attn)                     # [B,N,D]

        # FFN residual func stays as-is
        def ffn_residual_func(u: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(u)))

        if self.training and self.sample_drop_ratio > 0.1:
            # pass a const-returning func (ignores its input)
            assert False, "not implemented yet (ignore this for now bc i use 0.1 as default)"
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=lambda _: attn_out,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_out)
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_out
            x = x + ffn_residual_func(x)

        return x, cache

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


class ContextViTv39(nn.Module):
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
        num_prototypes=16,
        return_cls_only=True,
        sdp_kernel=SDPBackend.EFFICIENT_ATTENTION,
        num_banks: int = 4,           # M (banks)
        ema_momentum: float = 0.95,   # EMA for centroids
        noise_kw: dict = None,
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
        assert num_prototypes >= 1, "num_prototypes needs to be 1 or more"
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.p_token_drop = token_drop
        self.n_registers = n_registers
        self.num_prototypes = num_prototypes
        self.return_cls_only = return_cls_only
        self.sdp_kernel = sdp_kernel

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.n_patches = self.patch_embed.n_patches
        self.tok_regs = nn.Parameter(torch.zeros(1, 1 + self.n_registers, embed_dim))
        num_tokens = 1 + self.n_registers + self.n_patches
        self.tok_pos_emb = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            dpr[-1] = drop_path_rate

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
                num_prototypes=num_prototypes-(n_registers+1),
                num_registers=n_registers+1,
                num_banks=num_banks,
                ema_momentum=ema_momentum,
                noise_kw=noise_kw
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.tok_pos_emb, std=0.02)
        nn.init.normal_(self.tok_regs, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def prepare_tokens(self, x):
        with torch.profiler.record_function("Patch Embed"):
            x = self.patch_embed(x)
        with torch.profiler.record_function("prepare Tokens"):
            reg_tokens = self.tok_regs.expand(x.size(0), -1, -1)
            x = torch.cat((reg_tokens, x), dim=1) + self.tok_pos_emb
        with torch.profiler.record_function("Token Drop"):
            x = self.token_drop(x)
        return x
    
    def update(self, cache):
        for i, blk in enumerate(self.blocks):
            cls_n, idx = cache[i]
            blk.attn.ema_update(cls_n, idx)
            blk.attn.noise_schedule()
            

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
        with torch.nn.attention.sdpa_kernel(self.sdp_kernel):
            caches = []
            for blk in self.blocks:
                x, cache = blk(x)
                if self.training:
                    caches.append(cache)
        
        x = x[:, 0, :] if self.return_cls_only else x
        with torch.profiler.record_function("Final Norm"):
            out = self.norm(x)
        return (out, caches) if self.training else out


# # END
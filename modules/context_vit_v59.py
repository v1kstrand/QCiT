#!/usr/bin/env pythonphi0


from typing import Tuple, Union, Callable, Optional
from functools import partial
import math
from contextlib import contextmanager


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch import Tensor
from torch.nn.attention import SDPBackend
import torch.nn.functional as F


def no_wd(m: nn.Module) -> None:
    m.no_wd = True
    return m

def no_init(m: nn.Module) -> None:
    m.no_init = True
    return m


# FFN


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

class CPB2D(nn.Module):
    """
    Continuous Position Bias over tile-tied U anchors.

    Parent must set before forward():
      - self.P_pos        : [P, 1, 3]  query patch positions [qx, qy, 0]
      - self.tile_centers : [T, 1, 3]  per-tile centers [cx, cy, 0] (3rd channel must be 0)
      - self.u_pos        : nn.Parameter [1, U, 3]  per-U (Δcx, Δcy, s), tied across tiles
      - self.R            : int  number of special/register tokens

    Forward returns:
      - bias: [H, N, K] (batch-broadcastable), where N=R+P and K=R+(T*U)
    """
    def __init__(self, R: int, H: int, hidden: int = 32):
        super().__init__()
        self.R = int(R)
        self.H = int(H)
        # 3-dim in: [φ(dx), φ(dy), -s_k] → per-head bias
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, H),
        )
        self.reset_parameters()

        # to be populated by the parent model
        self.P_pos: torch.Tensor | None = None        # [P, 1, 3]
        self.tile_centers: torch.Tensor | None = None # [T, 1, 3]
        self.u_pos: nn.Parameter | None = None        # [1, U, 3]

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
            
    def set_dtype(self, dtype: torch.dtype):
        self.P_pos = self.P_pos.to(dtype)
        self.tile_centers = self.tile_centers.to(dtype)
        self.u_pos = self.u_pos.to(dtype)

    def forward(self, dtype: torch.dtype) -> torch.Tensor:
        P = self.P_pos.size(0)
        T = self.tile_centers.size(0)
        U = self.u_pos.size(1)
        R, H = self.R, self.H
        N, K_ctx = R + P, T * U
        K = R + K_ctx
        
        self.set_dtype(dtype)

        # --- per-tile, per-U anchors (tied across tiles), shape → [1, K_ctx, 3]
        K_ctx_pos = (self.tile_centers + self.u_pos).reshape(1, K_ctx, 3)

        # --- relative features for patch→ctx block
        delta = (self.P_pos - K_ctx_pos).to(self.P_pos.dtype)   # [P, K_ctx, 3]
        delta_xy, s_feat = torch.split(delta, (2, 1), -1)            # [P, K_ctx, 2]
        phi_xy   = torch.sign(delta_xy) * torch.log1p(delta_xy.abs())   
        feats_ctx = torch.cat([phi_xy, -s_feat], dim=-1)          # [P, K_ctx, 3]    
        out_ctx = self.mlp(feats_ctx)  # [P, K_ctx, H]# [P, K_ctx, 3]

        # --- MLP on ctx block, then scatter into full [N,K,H]
        out = torch.empty(N, K, H, device=out_ctx.device, dtype=out_ctx.dtype)
        out[:R, :, :].zero_()      # zero rows for registers
        out[:, :R, :].zero_()      # zero cols for registers
        out[R:, R:, :].copy_(out_ctx)  # fill the big patch→ctx block
        return out.permute(2, 0, 1).contiguous() # H, N, K


class ContextAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        num_heads: int = 6,
        num_regs: int = 1,  # first R tokens are [CLS/REG...]
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        proj_bias: bool = True,
        tile_comp_size: int = 1,
        tile_dim = 1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.H = num_heads
        self.d = dim // num_heads
        
        self.N = num_tokens
        self.R = num_regs
        self.P = num_tokens - num_regs
        
        self.S = int(self.P**0.5) # grid size (S x S)
        self.td = tile_dim # tile dim
        self.ts = ts = tile_dim ** 2 # tile size
        assert self.P % ts == 0 and self.S % self.td == 0
        self.T = self.P // ts # number of tiles
        self.U = tile_comp_size

        self.logit = no_wd(nn.Linear(dim, self.U, bias=False))
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_out = nn.Linear(dim, dim, bias=proj_bias)
        
        self.cpb_mlp = CPB2D(num_regs, num_heads)

        self.attn_drop = attn_drop
        self.out_drop = nn.Dropout(proj_drop)
        self.return_cache = False

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        H, d, R = self.H, self.d, self.R
        U, S, T, td, ts = self.U, self.S, self.T, self.td, self.ts

        patch = x[:, R:, :]  # [B,P,D]
        patch = patch.view(B, S // td, td, S // td, td, D)  # [B,S/td,td,S/td,td,D]
        tiled = patch.permute(0, 1, 3, 2, 4, 5).reshape(B, T, ts, D)  # [B, T, ts, D]

        scores = self.logit(tiled)  # [B, k^, U]
        w = F.softmax(scores.transpose(-1, -2).float(), dim=-1).to(scores.dtype)  # [B,T, U,ts]
        out = torch.matmul(w, tiled)  # [B, T, U, D] 
        ctx_learn = out.reshape(B, T*U, D)  # [B, T*U, D]

        # prepend registers back
        ctx = torch.cat([x[:, :R, :], ctx_learn], dim=1)  # [B, K, D]
        K = ctx.size(1)

        # keys/values from pooled contexts
        q = self.proj_q(x).view(B, N, H, d).transpose(1, 2)  # [B,H,N,d]
        
        kv = self.proj_kv(ctx).reshape(B, K, 2, H, d).permute(2, 0, 3, 1, 4)
        k, v = torch.unbind(kv, dim=0)

        # SDPA
        x_attn = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=self.cpb_mlp(x.dtype),
            dropout_p=self.attn_drop if self.training else 0.0
        )  # [B,H,N,d]
        

        out = self.out_drop(self.proj_out(x_attn.transpose(1, 2).reshape(B, N, D)))
        return out, None
    
# Block
class ResidualAdd(nn.Module):
    """
    y = x + res * (gamma * scale)
    where scale is per-sample DropPath factor in {0, 1/keep}.
    """

    def __init__(self, dim: int, drop_prob: float = 0.0, ls_init: float = None):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.gamma = (
            nn.Parameter(torch.full((dim,), ls_init)) if ls_init is not None else 1
        )

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x + res * self.gamma

        keep = 1.0 - self.drop_prob
        shape = (res.shape[0],) + (1,) * (res.ndim - 1)
        scale = torch.empty(shape, device=res.device).bernoulli_(keep)
        scale = scale.to(res.dtype) / keep
        return x + res * (self.gamma * scale)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ckw: dict,
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
        block_idx=-1,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        attn_kw = dict(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.attn = ContextAttention(**attn_kw, **ckw)
        self.residual_add_attn = ResidualAdd(dim, drop_path, layerscale)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.residual_add_ffn = ResidualAdd(dim, drop_path, layerscale)

    def forward(self, x: Tensor):
        x_attn, cache = self.attn(self.norm1(x))
        x = self.residual_add_attn(x, x_attn)
        x_ffn = self.mlp(self.norm2(x))
        x = self.residual_add_ffn(x, x_ffn)
        return x, cache


# # Patch Embed
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

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


# # Vit
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
        if not hasattr(module, "no_init"):
            trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif hasattr(module, "reset_parameters"):
        module.reset_parameters()


class ContextViTv59(nn.Module):
    def __init__(
        self,
        ckw,
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
        return_cls_only=True,
        sdp_kernel=SDPBackend.EFFICIENT_ATTENTION,
        debug_compile=False
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
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.p_token_drop = token_drop
        self.n_registers = n_registers
        self.return_cls_only = return_cls_only
        self.sdp_kernel = sdp_kernel
        self.ckw = ckw

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        ckw["num_regs"] = self.n_registers + 1
        self.n_patches = self.patch_embed.n_patches
        self.tok_regs = nn.Parameter(torch.zeros(1, 1 + self.n_registers, embed_dim))
        ckw["num_tokens"] = num_tokens = 1 + self.n_registers + self.n_patches
        self.tok_pos_emb = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-5)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        print(f"INFO: Drop Path Rates: {[round(n, 4) for n in dpr]}")

        blocks_list = [
            Block(
                ckw=ckw,
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
                block_idx=i,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.init_weights()
        if debug_compile:
            self.blocks[0].compile(backend="inductor", fullgraph=True, dynamic=False)
        
    def init_weights(self):
        trunc_normal_(self.tok_pos_emb, std=0.02)
        nn.init.normal_(self.tok_regs, std=1e-6)
        named_apply(init_weights_vit_timm, self)
        P, td, U = self.n_patches, self.ckw["tile_dim"], self.ckw["tile_comp_size"]
        P_pos, tile_centers, u_pos = self.make_cpb_pos_tables(P, td, U)
        self.register_buffer("P_pos", P_pos, persistent=False)
        self.register_buffer("tile_centers", tile_centers, persistent=False)
        for blk in self.blocks:
            blk.attn.cpb_mlp.P_pos = self.P_pos
            blk.attn.cpb_mlp.tile_centers = self.tile_centers
            blk.attn.cpb_mlp.u_pos = nn.Parameter(u_pos)
        
        
    @torch.no_grad()
    def make_cpb_pos_tables(
        self,
        P: int,              # patch grid side (P = S*S)
        td: int,             # tile side (tile = td x td)
        U: int,              # pooled tokens per tile
        device=None,
        dtype=torch.float32,
        init_u_offset=0.0,   # optional small init for Δcx,Δcy (e.g., 0.0 or 1e-3)
        init_u_spread=0.0,   # initial log-spread s (0.0 = neutral)
    ):
        """
        Returns:
        P_pos        : [P, 1, 3] with [qx, qy, 0]
        tile_centers : [T, 1, 3] with [cx, cy, 0]
        u_pos        : nn.Parameter [1, U, 3] with [Δcx, Δcy, s], tied across tiles
        """
        S = int(P ** 0.5)
        assert S > 0 and td > 0 and U > 0
        assert S % td == 0, "S must be divisible by td"
        
        Sh, Sw = S // td, S // td
        T = Sh * Sw
        ts = td * td

        device = next(self.parameters()).device

        # --- Patch coords in [-1, 1]
        ys = torch.linspace(-1.0, 1.0, S, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, S, device=device, dtype=dtype)
        GY, GX = torch.meshgrid(ys, xs, indexing="ij")
        Qx = GX.reshape(-1)  # [P]
        Qy = GY.reshape(-1)

        P_pos = torch.zeros(P, 1, 3, device=device, dtype=dtype)
        P_pos[:, 0, 0] = Qx
        P_pos[:, 0, 1] = Qy
        # last channel stays 0

        # --- Tile centers by uniform mean of td×td blocks
        subGX = GX.view(Sh, td, Sw, td).permute(0, 2, 1, 3).reshape(T, ts)
        subGY = GY.view(Sh, td, Sw, td).permute(0, 2, 1, 3).reshape(T, ts)
        Cx = subGX.mean(dim=1)  # [T]
        Cy = subGY.mean(dim=1)  # [T]

        tile_centers = torch.zeros(T, 1, 3, device=device, dtype=dtype)
        tile_centers[:, 0, 0] = Cx
        tile_centers[:, 0, 1] = Cy
        # last channel stays 0 (important so spread is unaffected)

        # --- Per-U learnable anchor (Δcx, Δcy, s), shared across tiles
        u_pos = nn.Parameter(torch.zeros(1, U, 3, device=device, dtype=dtype))
        if init_u_offset != 0.0:
            u_pos.data[:, :, 0:2].fill_(init_u_offset)
        if init_u_spread != 0.0:
            u_pos.data[:, :, 2].fill_(init_u_spread)

        return P_pos, tile_centers, u_pos
        
    @contextmanager
    def return_caches(self):
        prev = []
        try:
            for b in self.blocks:
                prev.append(b.attn.return_cache)
                b.attn.return_cache = True
            yield
        finally:
            for b, v in zip(self.blocks, prev):
                b.attn.return_cache = v

    def prepare_tokens(self, x):
        with torch.profiler.record_function("Patch Embed"):
            x = self.patch_embed(x)
        with torch.profiler.record_function("prepare Tokens"):
            reg_tokens = self.tok_regs.expand(x.size(0), -1, -1)
            x = torch.cat((reg_tokens, x), dim=1) #+ self.tok_pos_emb
        with torch.profiler.record_function("Token Drop"):
            x = self.token_drop(x)
        return x

    def token_drop(self, x):
        if not self.p_token_drop or not self.training:
            return x
        assert False, "not implemented, p_token_drop must be 0"
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
                caches.append(cache)

        x = x[:, 0, :] if self.return_cls_only else x
        with torch.profiler.record_function("Final Norm"):
            out = self.norm(x)
        return (out, caches) if self.training else out


# # END
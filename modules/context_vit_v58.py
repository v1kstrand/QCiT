#!/usr/bin/env python
# coding: utf-8


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
from torch.nn.attention.flex_attention import flex_attention


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

def soft_clip01(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """
    Smooth clamp to [0, 1] using a softplus-based approximation of clamp.
    Keeps the output in x.dtype (important under BF16 autocast).
    """
    out = (F.softplus(beta * x) - F.softplus(beta * (x - 1.0))) / beta
    return out.to(x.dtype)


class ContextAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        num_heads: int = 6,
        num_regs: int = 1,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        proj_bias: bool = True,
        C_dim: int = 3,
        tile_comp_size: int = 1,   # U
        tile_dim: int = 1,         # td
        mlp_hidden: int = 8,
        gate_init: float = 0.05,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.H = H = num_heads
        self.d = dim // num_heads

        self.N = num_tokens
        self.R = num_regs
        self.P = num_tokens - num_regs

        self.S  = int(self.P ** 0.5)
        self.td = tile_dim
        self.ts = tile_dim ** 2
        assert self.P % self.ts == 0 and self.S % self.td == 0
        self.T = self.P // self.ts
        self.U = tile_comp_size
        self.c = C_dim
        self.K = self.T * self.U + self.R  # total keys

        # projections & pooling (params live in FP32 as master)
        self.logit   = no_wd(nn.Linear(dim, self.U, bias=False))
        self.w_to_C  = no_wd(nn.Linear(self.ts, C_dim, bias=False))
        self.proj_q  = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_out = nn.Linear(dim, dim, bias=proj_bias)

        assert attn_drop == 0.0, "attn_drop not implemented for flex_attention"
        self.out_drop = nn.Dropout(proj_drop)

        # φ table flattened to [N*K, 2] (buffer stays FP32 master; cast per-forward)
        feats = self._build_feats(P=self.P, R=self.R, S=self.S, td=self.td, U=self.U)
        self.register_buffer("feats", feats, persistent=False)

        # GPU int64 index buffers (move with .to(device))
        self.register_buffer("K_t",     torch.tensor(self.K, dtype=torch.int64), persistent=False)
        self.register_buffer("R_t",     torch.tensor(self.R, dtype=torch.int64), persistent=False)
        self.register_buffer("zero_idx", torch.tensor(0,     dtype=torch.int64), persistent=False)

        # tiny per-head MLP (FP32 master)
        self.W1 = nn.Parameter(torch.randn(H, mlp_hidden, 2 + self.c) * 1e-2)
        self.b1 = nn.Parameter(torch.zeros(H, mlp_hidden))
        self.W2 = nn.Parameter(torch.randn(H, mlp_hidden) * 1e-2)

        # per-head gate α (logit) in FP32 master
        a = math.log(gate_init / (1 - gate_init + 1e-8) + 1e-8)
        self.alpha = nn.Parameter(torch.full((H,), float(a)))
        self.return_cache = False

    def reset_parameters(self):
        nn.init.trunc_normal_(self.W1, std=0.02)
        nn.init.trunc_normal_(self.W2, std=0.02)

    @torch.no_grad()
    def _build_feats(self, P: int, R: int, S: int, td: int, U: int) -> torch.Tensor:
        ts = td * td
        T = P // ts
        K_ctx = T * U
        N = R + P
        K = R + K_ctx

        ys = torch.linspace(-1.0, 1.0, S)
        xs = torch.linspace(-1.0, 1.0, S)
        GY, GX = torch.meshgrid(ys, xs, indexing="ij")
        Qx = GX.reshape(-1)
        Qy = GY.reshape(-1)

        Sh = S // td; Sw = S // td
        subGX = GX.view(Sh, td, Sw, td).permute(0, 2, 1, 3).reshape(T, ts)
        subGY = GY.view(Sh, td, Sw, td).permute(0, 2, 1, 3).reshape(T, ts)
        Cx = subGX.mean(dim=1)
        Cy = subGY.mean(dim=1)
        Kx = Cx.repeat_interleave(U)
        Ky = Cy.repeat_interleave(U)

        dx = Qx[:, None] - Kx[None, :]
        dy = Qy[:, None] - Ky[None, :]
        phix = torch.sign(dx) * torch.log1p(dx.abs())
        phiy = torch.sign(dy) * torch.log1p(dy.abs())

        feats = torch.zeros(N, K, 2)
        feats[R:R+P, R:R+K_ctx, 0] = phix
        feats[R:R+P, R:R+K_ctx, 1] = phiy
        return feats.reshape(N * K, 2).contiguous()

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        H, d, R = self.H, self.d, self.R
        U, S, T, td, ts, K = self.U, self.S, self.T, self.td, self.ts, self.K

        # pool per tile
        patch = x[:, R:, :]
        patch = patch.view(B, S // td, td, S // td, td, D)
        tiled = patch.permute(0, 1, 3, 2, 4, 5).reshape(B, T, ts, D)

        scores = self.logit(tiled)  # [B,T,ts,U]
        w = F.softmax(scores.transpose(-1, -2).float(), dim=-1).to(scores.dtype)  # [B,T,U,ts]
        ctx_learn = torch.matmul(w, tiled).reshape(B, T * U, D)
        ctx = torch.cat([x[:, :R, :], ctx_learn], dim=1)
        assert ctx.size(1) == self.K

        # Q/K/V
        q = self.proj_q(x).view(B, N, H, d).transpose(1, 2)
        kv = self.proj_kv(ctx).reshape(B, K, 2, H, d).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # per-(b,k_ctx) descriptors (detach to avoid direct loss path)
        C = soft_clip01(self.w_to_C(w.view(B, T * U, ts).detach()))  # likely BF16 under autocast

        """# ---------- FP32 master -> BF16 views for flex path ----------
        q_dtype    = q.dtype  # typically torch.bfloat16 under autocast
        W1_bf16    = self.W1.to(q_dtype)
        b1_bf16    = self.b1.to(q_dtype)
        W2_bf16    = self.W2.to(q_dtype)
        feats_bf16 = self.feats.to(q_dtype)        # [N*K, 2]
        C_bf16     = C.to(q_dtype)                 # [B, K_ctx, c]

        # pre-split first-layer weights once (avoid slice inside closure)
        W1_phi_bf16 = W1_bf16[..., :2]             # [H, HID, 2]
        W1_c_bf16   = W1_bf16[..., 2:]             # [H, HID, c]

        # fold α into W2 once (removes an extra mul inside closure)
        a_bf16      = torch.sigmoid(self.alpha).to(q_dtype)     # [H]
        W2_scaled   = a_bf16.view(-1, 1) * W2_bf16              # [H, HID]

        K_t, R_t, zero_idx = self.K_t, self.R_t, self.zero_idx  # int64 buffers

        def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
            # gate (bool) and numeric view for the final multiply
            reg_bool   = ((q_idx >= R_t) & (kv_idx >= R_t))
            reg_gate_f = reg_bool.to(score.dtype)

            # φ(q,k) from flattened table
            lin = (q_idx * K_t + kv_idx).view(1)
            phi = feats_bf16.index_select(0, lin).squeeze(0)            # [2]

            # C[b, k_ctx, :] with k_ctx = kv_idx - R (guarded for regs)
            k_ctx = kv_idx - R_t
            k_sel = torch.where(reg_bool, k_ctx, zero_idx).view(1)      # int64
            cvec  = C_bf16.index_select(0, b_idx.view(1)).squeeze(0) \
                        .index_select(0, k_sel).squeeze(0)             # [c]

            # per-head params (single read each)
            h1   = h_idx.view(1)
            w1p  = W1_phi_bf16.index_select(0, h1).squeeze(0)           # [HID, 2]
            w1c  = W1_c_bf16.index_select(0, h1).squeeze(0)             # [HID, c]
            bb1  = b1_bf16.index_select(0, h1).squeeze(0)               # [HID]
            w2s  = W2_scaled.index_select(0, h1).squeeze(0)             # [HID]

            # tiny MLP: two matvecs + bias, then a single dot
            hid  = torch.relu(torch.mv(w1p, phi) + torch.mv(w1c, cvec) + bb1)  # [HID]
            bias = torch.dot(w2s, hid)                                         # scalar

            return score + bias * reg_gate_f"""


        def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
            # trivial pointwise op that leaves score unchanged
            return score 

        x_attn = flex_attention(q, k, v, score_mod=score_mod)
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


class ContextViTv58(nn.Module):
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
        
    def init_weights(self):
        trunc_normal_(self.tok_pos_emb, std=0.02)
        nn.init.normal_(self.tok_regs, std=1e-6)
        named_apply(init_weights_vit_timm, self)
        
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
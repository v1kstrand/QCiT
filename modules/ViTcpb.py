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
from torch.nn.attention.flex_attention import flex_attention  # PyTorch >= 2.5

def no_wd(m: nn.Module) -> None:
    m.no_wd = True
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


class FlexAttentionCPB(nn.Module):
    def __init__(self, H: int, hidden: int = 32):
        super().__init__()
        # Tiny MLP: R^2 -> R^H
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, H, bias=False),
        )
        self.H = H
        #self.gamma = nn.Parameter(torch.zeros(H)) * 1e-4
        

    def forward(self):
        """Return a callable used by flex_attention to inject CPB."""
        
        bt = self.mlp(self.rel_table)  # [L, H]
        idx = self.idx_table  # [N,N], {-1} U [0..L-1
        #mu_q, mu_k = mu.split(self.H, dim=1)  # [B,H,N] each
        #gam_sig = torch.sigmoid(self.gamma)  # [H]
        def score_mod(score, b, h, q, kv):
            has_bias = (q >= self.r_cutoff) & (kv >= self.r_cutoff)
            l2 = idx[q, kv]   # [...], in [0..L]
            bias = bt[l2, h]
            #muq = mu_q[b, h, q]
            #muk = mu_k[b, h, kv]
            #w_gate = gam_sig[h] * (muq + muk)
            #return score + w_gate * bias
            return score + has_bias * bias
        return score_mod
    
    def reset_parameters(self):
        self.register_buffer("r_cutoff", torch.tensor(self.R, dtype=torch.long), persistent=False)
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)  # start near zero bias



class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.cpb_mlp = FlexAttentionCPB(num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.return_cache = False

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        H, D  = self.num_heads, self.head_dim

        # 1) project to QKV and split heads
        #    → [B, N, 3⋅H⋅D]
        #    → [B, 3, H, N, D]
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]   # each is [B, H, N, D]

        # 2) scaled‐dot‐product‐attention with dropout
        #    (dropout is only active when self.training==True)
        attn_out = flex_attention(q, k, v, score_mod=self.cpb_mlp())

        # 3) re‐assemble heads → [B, N, C]
        attn_out = (attn_out
                    .transpose(1, 2)        # [B, N, H, D]
                    .reshape(B, N, C))      # [B, N, H⋅D]

        # 4) output projection + dropout
        x = self.proj(attn_out)
        x = self.proj_drop(x)
        return x, None
    
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
        self.attn = Attention(**attn_kw, **ckw)
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
        make_2tuple = lambda x: x if isinstance(x, tuple) else (x, x)
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
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif hasattr(module, "reset_parameters"):
        module.reset_parameters()


class ViTcpb(nn.Module):
    def __init__(
        self,
        ckw=None,
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
        ckw = ckw or {}
        self.ckw = ckw

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.n_patches = self.patch_embed.n_patches
        self.R = self.n_registers + 1
        self.N = self.R + self.n_patches
        self.tok_regs = nn.Parameter(torch.zeros(1, self.R, embed_dim))
        #self.tok_pos_emb = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

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
        
        nn.init.normal_(self.tok_regs, std=0.02)
        for blk in self.blocks: 
            blk.attn.cpb_mlp.R = self.R
        named_apply(init_weights_vit_timm, self)
        self.make_cpb_pos_tables(self.N, self.R, num_heads)
        for blk in self.blocks:
            blk.attn.cpb_mlp.rel_table = self.rel_table.cuda()
            blk.attn.cpb_mlp.idx_table = self.idx_table.cuda()
        
        self.debug_compile = debug_compile
        
        if debug_compile:
            self.blocks[0].compile(backend="inductor", fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")

    @torch.no_grad()
    def make_cpb_pos_tables(self,N,R,H):
        """
        Returns:
        P_pos        : [P, 1, 3] with [qx, qy, 0]
        tile_centers : [T, 1, 3] with [cx, cy, 0]
        u_pos        : nn.Parameter [1, U, 3] with [Δcx, Δcy, s], tied across tiles
        """
        assert 0 <= R < N
        P = N - R
        S = int(P**0.5)
        assert S * S == P
        self.N, self.R, self.P, self.S, self.H = N, R, P, S, H

        # Unique rel offsets: [L,2], L = (2S-1)^2
        rng = torch.arange(-(S - 1), S, dtype=torch.float32)
        dY, dX = torch.meshgrid(rng, rng, indexing="ij")
        rel = torch.stack([dY / max(S - 1, 1), dX / max(S - 1, 1)], -1).reshape(-1, 2)
        rel_table = torch.sign(rel) * torch.log1p(rel.abs())
        self.register_buffer("rel_table", rel_table.cuda(), persistent=False)  # [L,2]

        # Precompute index table: {-1} (specials) or [0..L-1] for window<->window
        L = rel.shape[0]
        yy = torch.arange(S)
        xx = torch.arange(S)
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        flat = torch.stack([Y, X], 0).flatten(1)  # [2,P]
        d = flat[:, :, None] - flat[:, None, :]  # [2,P,P]
        d = d.permute(1, 2, 0).contiguous()  # [P,P,2]
        d[:, :, 0] += S - 1
        d[:, :, 1] += S - 1
        d[:, :, 0] *= 2 * S - 1
        l_idx = d.sum(-1).to(torch.long)  # [P,P] in [0..L-1]

        idx = torch.full((N, N), 0, dtype=torch.long)  # sentinel -1
        idx[R:, R:] = l_idx 
        self.register_buffer("idx_table", idx.cuda(), persistent=False)  # [N,N], {-1 U [0..L-1]}
        
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
                if self.debug_compile:
                    break

        x = x[:, 0, :] if self.return_cls_only else x
        with torch.profiler.record_function("Final Norm"):
            out = self.norm(x)
        return (out, caches) if self.training else out


# # END
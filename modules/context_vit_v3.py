#!/usr/bin/env python
# coding: utf-8

# # Imports 


from typing import Tuple, Union, Callable, Optional
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch import Tensor
import torch.nn.functional as F

from copy import deepcopy


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        norm_layer=nn.Identity,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.norm_layer = norm_layer(in_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_layer(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiMlp(nn.Module):
    def __init__(self, mlp_base, k, n_ctx):
        super().__init__()
        assert n_ctx % k == 0, "n_ctx must be divisible by k"
        self.mlps = nn.ModuleList([deepcopy(mlp_base) for _ in range(1 + k)])
        self.k = k
        self.n_ctx = n_ctx

    def forward(self, x):
        ctx, patches = x[:, : self.n_ctx], x[:, self.n_ctx :]

        seq = []
        ctx_chunks = torch.chunk(ctx, self.k, 1)  # chunk ctx into K parts
        for i, ctx_chunk in enumerate(ctx_chunks):
            seq.append(self.mlps[i](ctx_chunk))
        seq.append(self.mlps[self.k](patches))

        return torch.cat(seq, dim=1)

class LinearContextAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_ctx: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        act_layer=nn.Identity,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.n_ctx = n_ctx
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sdpa = F.scaled_dot_product_attention

        # — Proj —
        self.p_to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.ctx_to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.agg_to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # — Dropout —
        self.attn_drop_v = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        # — Norms —
        self.norm_p = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.norm_agg = nn.LayerNorm(dim)

        # - Act -
        if not isinstance(act_layer, tuple):
            self.act_ctx = act_layer()
            self.act_p = act_layer()
        else:
            self.act_ctx = act_layer[0]()
            self.act_p = act_layer[1]()

        # — Out proj —
        self.to_out_ctx = nn.Linear(dim, dim, bias=proj_bias)
        self.to_out_p = nn.Linear(dim, dim, bias=proj_bias)

    def proj_qkv(self, x: Tensor, W: nn.Linear, seq_len: int, n: int) -> Tensor:
        y = W(x).view(x.size(0), seq_len, n, self.num_heads, self.head_dim)
        y = y.permute(2, 0, 3, 1, 4)  # [n, B, H, seq_len, d]
        return y[0] if n == 1 else (y[0], y[1]) if n == 2 else (y[0], y[1], y[2])

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape  # x: [B, M+N, D]. S = [Ctx, Cls, Reg, Patches]
        M = self.n_ctx  # Ctx, Agg
        N = S - M  # Cls + Reg + Patches

        # 1) split out tokens
        ctx, patches = x[:, : self.n_ctx], x[:, self.n_ctx :]

        # 2) patch QKV proj
        Q_p, K_p, V_p = self.proj_qkv(self.norm_p(patches), self.p_to_qkv, N, 3)

        # 3) Ctx ← Patches  (M × N)
        Q_ctx = self.proj_qkv(self.norm_ctx(ctx), self.ctx_to_q, M, 1)
        att_agg = self.sdpa(Q_ctx, K_p, V_p, dropout_p=self.attn_drop_v)
        att_agg = att_agg.transpose(1, 2).reshape(B, M, D)
        ctx = self.proj_drop(self.act_ctx(self.to_out_ctx(att_agg)))

        # 4) Patches ← agg  (N × M)
        K_agg, V_agg = self.proj_qkv(self.norm_agg(att_agg), self.agg_to_kv, M, 2)
        att_p = self.sdpa(Q_p, K_agg, V_agg, dropout_p=self.attn_drop_v)
        att_p = att_p.transpose(1, 2).reshape(B, N, D)
        patches = self.proj_drop(self.act_p(self.to_out_p(att_p)))

        # 5) re-pack and return
        return torch.cat([ctx, patches], dim=1)


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


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        n_ctx: int,
        k_ctx: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        layerscale=True,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        attn_act=nn.Identity,
    ) -> None:
        super().__init__()
        # self.norm1 = norm_layer(dim) # Included in attn
        # self.norm2 = norm_layer(dim) # Included in mlp

        self.attn = LinearContextAttention(
            dim,
            num_heads=num_heads,
            n_ctx=n_ctx,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            act_layer=attn_act,
        )
        self.ls1 = LayerScale(dim, init_values=1e-4) if layerscale else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            norm_layer=norm_layer,
        )

        if k_ctx == 0:
            self.mlp = mlp
        else:
            self.mlp = MultiMlp(mlp, k_ctx, n_ctx)

        self.ls2 = LayerScale(dim, init_values=1e-5) if layerscale else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(x))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(x))

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

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops



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


class LinearContextViTv3(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=True,
        layerscale=False,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        token_drop=0,
        n_registers=0,
        n_ctx=32,
        k_ctx=0,
        attn_act=nn.Identity,
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
        assert n_ctx >= 1, "n_ctx needs to be 1 or more"
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.p_token_drop = token_drop
        self.n_registers = n_registers
        self.n_ctx = n_ctx

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1 + self.n_registers, embed_dim))
        self.ctx_tokens = nn.Parameter(torch.zeros(1, self.n_ctx, embed_dim))
        num_pos_emb = self.n_ctx + 1 + self.n_registers + self.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_pos_emb, embed_dim))

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
                ffn_layer=Mlp,
                layerscale=layerscale,
                n_ctx=n_ctx,
                k_ctx=k_ctx,
                attn_act=attn_act,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.ctx_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def prepare_tokens(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        ctx_tokens = self.ctx_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((ctx_tokens, cls_token, x), dim=1)
        return self.token_drop(x + self.pos_embed)

    def token_drop(self, x):
        if not self.p_token_drop or not self.training:
            return x
        patch_idx = 1 + self.n_registers + self.n_ctx
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
        return self.norm(x)[:, self.n_ctx :]



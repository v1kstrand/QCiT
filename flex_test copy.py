import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention 
from torch.nn.attention import SDPBackend


class FlexAttentionCPB(nn.Module):
    def __init__(self, N: int, R: int, H: int = 6, hidden: int = 32):
        super().__init__()
        # Two-layer MLP that learns per-head content-position bias (CPB) weights.
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, H, bias=False),
        )
        # Gate that scales the learned CPB contribution on a head-by-head basis.
        self.gamma = nn.Parameter(torch.zeros(H))
        self.H = H
        self.init_tables(N, R)
        # Store R as a tensor so score_mod can build a branch-free sentinel mask.
        self.register_buffer("r_cutoff", torch.tensor(R, dtype=torch.long), persistent=False)

    def init_tables(self, N: int, R: int) -> None:
        """Precompute geometry-dependent lookup tables used by score_mod."""
        assert 0 <= R < N
        P = N - R
        S = int(P**0.5)
        assert S * S == P  # the non-sentinel tokens must form a perfect square grid
        self.N, self.R, self.P, self.S = N, R, P, S

        # Build a table of normalized 2-D relative offsets between tile centers.
        rng = torch.arange(-(S - 1), S, dtype=torch.float32)
        dY, dX = torch.meshgrid(rng, rng, indexing="ij")
        rel = torch.stack([dY / max(S - 1, 1), dX / max(S - 1, 1)], dim=-1).reshape(-1, 2)
        rel_table = torch.sign(rel) * torch.log1p(rel.abs())
        self.register_buffer("rel_table", rel_table, persistent=False)  # [L, 2]

        # Precompute an index lookup that maps window pairs to rows in rel_table.
        L = rel.shape[0]
        yy = torch.arange(S)
        xx = torch.arange(S)
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        flat = torch.stack([Y, X], 0).flatten(1)  # [2, P]
        d = flat[:, :, None] - flat[:, None, :]   # [2, P, P]: all pairwise offsets
        d = d.permute(1, 2, 0).contiguous()       # [P, P, 2]
        d[:, :, 0] += S - 1
        d[:, :, 1] += S - 1
        d[:, :, 0] *= 2 * S - 1
        l_idx = d.sum(-1).to(torch.long)          # [P, P] in [0, L-1]

        # idx_table keeps a 0-based index for real tiles and -1 for the sentinel prefix.
        idx = torch.full((N, N), -1, dtype=torch.long)
        idx[R:, R:] = l_idx 
        self.register_buffer("idx_table", idx, persistent=False)  # [N, N], {0 U [1..L]}

    def _score_mod(self, mu: torch.Tensor):
        """Return a callable used by flex_attention to inject CPB."""
        bt = self.mlp(self.rel_table)
        idx = self.idx_table
        mu_q, mu_k = mu.unbind(2)
        gam_sig = torch.sigmoid(self.gamma)

        def score_mod(score, b, h, q, kv):
            has_bias = (q >= self.r_cutoff) & (kv >= self.r_cutoff)
            l2 = idx[q, kv] 
            bias = bt[l2, h]
            muq = mu_q[b, h, q]
            muk = mu_k[b, h, kv]
            w_gate = gam_sig[h] * (muq + muk)
            return score + has_bias.to(score.dtype) * w_gate * bias

        return score_mod

    def forward(self, q, k, v, mu):
        return flex_attention(q, k, v, score_mod=self._score_mod(mu))



def main() -> None:
    device = torch.device("cuda")

    B, N, R, d, H = 100, 200, 4, 64, 6
    flex = FlexAttentionCPB(N, R, H).to(device)
    flex.compile(dynamic=False, mode="max-autotune-no-cudagraphs") 

    q = torch.randn(B, H, N, d, device=device)
    k = torch.randn(B, H, N, d, device=device)
    v = torch.randn(B, H, N, d, device=device)
    mu = torch.randn(B, H, 2, N, device=device)

    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = flex(q, k, v, mu)
            out.norm().backward()
            print("Done")


if __name__ == "__main__":
    main()

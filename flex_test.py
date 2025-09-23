import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention  # PyTorch >= 2.5
from torch.nn.attention import SDPBackend


class FlexCPB2DIdxSentinel(nn.Module):
    def __init__(self, N: int, R: int, H: int, hidden: int = 32):
        super().__init__()
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
        self.register_buffer("rel_table", rel_table, persistent=False)  # [L,2]

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

        idx = torch.full((N, N), -1, dtype=torch.long)  # sentinel -1
        idx[R:, R:] = l_idx
        self.register_buffer("idx_table", idx, persistent=False)  # [N,N], {-1 U [0..L-1]}

        # Tiny MLP: R^2 -> R^H
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, H, bias=False),
        )
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)  # start near zero bias

        self.gamma = nn.Parameter(torch.randn(H))

    def _score_mod(self, bias_table_ext: torch.Tensor, mu: torch.Tensor):
        """Return a callable used by flex_attention to inject CPB."""
        idx = self.idx_table  # [N,N], {-1} U [0..L-1]
        w = bias_table_ext.reshape(-1)  # [(L+1)*H] flattened

        # Pre-normalize the conditioning signal per head so the callback sees bounded values.
        mu = mu.to(torch.float32)
        mu = torch.sign(mu) * torch.log1p(mu.abs())
        mu_q, mu_k = mu.split(self.H, dim=1)  # [B,H,N] each
        mu_q_flat = mu_q.reshape(-1)  # [B*H*N]
        mu_k_flat = mu_k.reshape(-1)

        gam_sig = torch.sigmoid(self.gamma)  # [H]

        def score_mod(score, b, h, q_idx, kv_idx):
            l2 = (idx[q_idx, kv_idx] + 1).to(torch.long)  # [...], in [0..L]
            h64 = h.to(torch.long)

            idx_flat = l2 * self.H + h64
            bias = w.gather(0, idx_flat).to(score.dtype)

            b64 = b.to(torch.long)
            q64 = q_idx.to(torch.long)
            kv64 = kv_idx.to(torch.long)
            base = (b64 * self.H + h64) * self.N
            i_q = base + q64
            i_k = base + kv64

            muq = mu_q_flat.gather(0, i_q).to(score.dtype)
            muk = mu_k_flat.gather(0, i_k).to(score.dtype)
            g = muq + muk

            gam_h = gam_sig.gather(0, h64).to(score.dtype)
            w_gate = gam_h * g
            return score + w_gate * bias

        return score_mod

    def forward(self, q, k, v, mu):
        bt = self.mlp(self.rel_table)  # [L, H]
        zeros_row = bt.new_zeros(1, bt.size(1))  # [1, H], no CPB sentinel
        bt_ext = torch.cat([zeros_row, bt], dim=0)  # [L+1, H], index 0 = no-CPB
        return flex_attention(q, k, v, score_mod=self._score_mod(bt_ext, mu))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("flex_test.py requires a CUDA-capable build of PyTorch.")

    device = torch.device("cuda")

    cpb_mlp = FlexCPB2DIdxSentinel(200, 4, 6).to(device)
    cpb_mlp = cpb_mlp.compile(backend="inductor", fullgraph=True, dynamic=False)

    B, N, D, d, H = 100, 200, 384, 64, 6
    q = torch.randn(B, H, N, d, device=device)
    k = torch.randn(B, H, N, d, device=device)
    v = torch.randn(B, H, N, d, device=device)
    mu = torch.randn(B, H * 2, N, device=device)

    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = cpb_mlp(q, k, v, mu)
            out.norm().backward()


if __name__ == "__main__":
    main()

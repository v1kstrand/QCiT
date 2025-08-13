import math, numpy as np, torch, matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_fn_idx(caches, M: int = None, block_labels=None, normalize: bool = False,
            compact: bool = False, sort: str = "id"):
    """
    Plot per-block routing assignments.
    - If M is given: use exactly M bins (0..M-1).
    - If M is None and compact=False: infer M = max(idx)+1 (may be huge).
    - If compact=True: show only banks that appeared (unique idx), label with bank IDs.
      `sort` in {"id","count"} controls x-order in compact mode.
    """
    idx_list, max_m = [], 0
    for _, idx in caches:
        idx_np = idx.detach().to("cpu").view(-1).numpy() if isinstance(idx, torch.Tensor) else np.asarray(idx).reshape(-1)
        idx_list.append(idx_np)
        if idx_np.size: max_m = max(max_m, int(idx_np.max()) + 1)

    if not compact:
        if M is None: M = max_m if max_m > 0 else 1

    B = len(idx_list)
    ncols = min(4, B); nrows = math.ceil(B / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows), squeeze=False)
    eps = 1e-12

    for b, idx_np in enumerate(idx_list):
        r, c = divmod(b, ncols)
        ax = axes[r][c]
        title = f"Block {b}" if not block_labels else str(block_labels[b])

        if compact:
            uniq, counts = np.unique(idx_np, return_counts=True)
            if sort == "count":
                order = np.argsort(-counts)
                uniq, counts = uniq[order], counts[order]
            vals = counts.astype(np.float64)
            total = vals.sum() if vals.size else 1.0
            y = vals / total if normalize else vals
            ax.bar(np.arange(len(uniq)), y)
            ax.set_xticks(np.arange(len(uniq)))
            ax.set_xticklabels([str(int(u)) for u in uniq], rotation=0)
            ax.set_xlabel("bank id (seen)")
            if normalize:
                H = -np.sum((vals/total + eps) * np.log(vals/total + eps))
                top = (vals.max()/total) if total > 0 else 0.0
                ax.set_ylabel("proportion"); ax.set_ylim(0, 1.0)
                ax.set_title(f"{title} | H={H:.2f}, top={top:.2f}")
            else:
                ax.set_ylabel("count"); ax.set_title(f"{title} | n={int(total)}")
        else:
            cnt = np.bincount(idx_np, minlength=M).astype(np.float64)
            total = cnt.sum() if cnt.sum() > 0 else 1.0
            y = cnt / total if normalize else cnt
            ax.bar(np.arange(M), y)
            ax.set_xlabel("bank m"); ax.set_xlim(-0.5, M-0.5)
            if normalize:
                H = -np.sum((cnt/total + eps) * np.log(cnt/total + eps))
                top = (cnt.max()/total) if total > 0 else 0.0
                ax.set_ylabel("proportion"); ax.set_ylim(0, 1.0)
                ax.set_title(f"{title} | H={H:.2f}, top={top:.2f}")
            else:
                ax.set_ylabel("count"); ax.set_title(f"{title} | n={int(total)}")

        if (not compact) and M > 20:
            ax.set_xticks(np.arange(0, M, max(1, M // 10)))

    for k in range(B, nrows*ncols):
        r, c = divmod(k, ncols); axes[r][c].axis("off")

    fig.tight_layout()
    return fig

def plot_fn_sim(
    caches,
    k: int = 10,                 # rows to sample per block
    seed: int = 0,               # RNG seed
    ncols: int | None = None,    # heatmaps per row; None → auto
    share_scale: bool = True,    # same color scale for all blocks
    block_labels=None,
    title: str | None = None,
    annotate: bool = True,       # <— draw value text in each cell
    fmt: str = ".2f",            # number format for annotations
    ann_fontsize: int = 8,       # annotation font size
):
    """
    Assumes: for _, _, sim in caches: sim has shape [B, M] (raw CLS·centroid sims).
    For each block, sample k rows (without replacement), plot one heatmap per block
    in a grid (rows = sampled CLS, cols = banks). Optionally annotate each cell.
    """
    rng = np.random.default_rng(seed)
    sims_list, picked = [], []

    # collect sampled matrices per block
    for _, _, sim in caches:
        sims = sim.detach().to("cpu").numpy() if isinstance(sim, torch.Tensor) else np.asarray(sim)
        B, M = sims.shape
        k_b = min(k, B)
        idx = rng.choice(B, size=k_b, replace=False)
        picked.append(idx)
        sims_list.append(sims[idx])  # [k_b, M]

    num_blocks = len(sims_list)
    M = sims_list[0].shape[1] if num_blocks > 0 else 0

    # grid shape
    if ncols is None:
        ncols = min(num_blocks, max(1, int(math.ceil(math.sqrt(num_blocks)))))
    nrows = int(math.ceil(num_blocks / ncols))

    # shared color scale (optional)
    if share_scale and num_blocks > 0:
        all_min = min(s.min() for s in sims_list)
        all_max = max(s.max() for s in sims_list)
        norm = Normalize(vmin=all_min, vmax=all_max)
    else:
        norm = None

    # labels
    if block_labels is None:
        block_labels = [f"blk {i}" for i in range(num_blocks)]

    # figure size
    fig_w = max(6, ncols * max(3.0, M * 0.20))
    fig_h = max(4, nrows * max(2.5, k * 0.25))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    ims = []
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i >= num_blocks:
            ax.axis("off")
            continue

        mat = sims_list[i]
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", norm=norm)
        ims.append(im)

        # annotate each cell with its value
        if annotate:
            # establish normalization for contrast decision
            if norm is not None:
                to01 = lambda v: float(norm(v))
            else:
                vmin, vmax = im.get_clim()
                denom = (vmax - vmin) if (vmax > vmin) else 1.0
                to01 = lambda v: float((v - vmin) / denom)

            for rr in range(mat.shape[0]):
                for cc in range(mat.shape[1]):
                    val = mat[rr, cc]
                    # dark text on light background, light text on dark
                    tcol = "black" if to01(val) > 0.5 else "white"
                    ax.text(
                        cc, rr, f"{val:{fmt}}",
                        ha="center", va="center",
                        fontsize=ann_fontsize, color=tcol
                    )

        ax.set_title(block_labels[i])
        # only bottom row shows x ticks
        if r == nrows - 1:
            ax.set_xlabel("bank id")
            ax.set_xticks(range(M))
            ax.set_xticklabels([str(j) for j in range(M)], rotation=0)
        else:
            ax.set_xticks([])
        # only first col shows y ticks
        if c == 0:
            ax.set_ylabel("sampled CLS")
        else:
            ax.set_yticks([])

    # shared colorbar
    if ims:
        if norm is not None:
            sm = ScalarMappable(norm=norm, cmap=ims[0].get_cmap()); sm.set_array([])
            fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, label="raw sim (CLS · centroid)")
        else:
            for im in ims:
                fig.colorbar(im, ax=im.axes, fraction=0.046, pad=0.04, label="raw sim")

    if title:
        fig.suptitle(title, y=0.995)
    
    return fig




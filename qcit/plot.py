import math, numpy as np, torch, matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_fn_idx(caches, i, M: int = None, block_labels=None, normalize: bool = False,
            compact: bool = False, sort: str = "id"):
    """
    Plot per-block routing assignments.
    - If M is given: use exactly M bins (0..M-1).
    - If M is None and compact=False: infer M = max(idx)+1 (may be huge).
    - If compact=True: show only banks that appeared (unique idx), label with bank IDs.
      `sort` in {"id","count"} controls x-order in compact mode.
    """
    idx_list, max_m = [], 0
    for c in caches:
        if c is None:
            continue
        idx = c[i]
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
    i, 
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
    for c in caches:
        if c is None:
            continue
        sim = c[i]
        sims = sim.detach().float().to("cpu").numpy() if isinstance(sim, torch.Tensor) else np.asarray(sim)
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

def plot_BKN_heatmap(caches, ci, num_samples: int = 20):
    """
    caches: list of cache-like objects where each supports indexing `cache[ci]`
            and returns a tensor of shape [B, K, N] with softmax rows along N.
    ci: index used to select a tensor from each cache.
    num_samples: Number of distinct batch indices (B) to sample per cache (max 10 by default).
    seed: Optional random seed for reproducibility. Applied independently per cache.
    """    
    k = sum(c is not None for c in caches)
    fig, axes = plt.subplots(k, 1, figsize=(10, max(3, k * 3)))
    if k == 0:
        print("Found no caches to plot.")
        return
    if k == 1:
        axes = [axes]  # ensure iterable
        
    idx = 0
    for cache in caches:
        if cache is None:
            continue
        ax = axes[idx]
        
        x = cache[ci]  # [B,K,N]
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B,K,N], got {tuple(x.shape)}")
        B, K, N = x.shape

        num = min(B, num_samples)
        b_indices = torch.randperm(B)[:num]
        k_indices = torch.randint(low=0, high=K, size=(num,), device=x.device)

        rows, labels = [], []
        for i in range(num):
            b = int(b_indices[i].item())
            k = int(k_indices[i].item())
            rows.append(x[b, k].detach().float().cpu().unsqueeze(0))
            labels.append(f"b={b}, k={k}")
        heat = torch.cat(rows, dim=0).numpy()  # [num, N]

        im = ax.imshow(heat, aspect="auto", interpolation="nearest")
        ax.set_xlabel("N (class index)")
        ax.set_ylabel("Sampled (b, k)")
        ax.set_yticks(np.arange(num))
        ax.set_yticklabels(labels)
        ax.set_title(f"[cache {idx}] Sampled rows: {num}×{N} (B={B}, K={K})")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Vals")
        idx += 1

    plt.tight_layout()
    return fig

def K_over_B_heatmap(caches, ci, b_index: int | None = None, height_per_10rows: float = 1.0):
    """
    For each cache in `caches`, select a single batch index `b` (random if not given),
    and plot a heatmap of all K rows (in order) for that b: heat = x[b, :, :]  [K, N].

    Args:
        caches: list where each element supports `cache[ci] -> Tensor[B,K,N]` (rows over N sum to 1).
        ci:     index used to select a tensor from each cache.
        b_index: optional fixed batch index to use for all caches; if None, choose a random b per cache.
        height_per_10rows: vertical figure size per 10 rows of K (scales each subplot).
    Returns:
        matplotlib.figure.Figure
    """
    valid_caches = [c for c in caches if c is not None]
    k = len(valid_caches)
    if k == 0:
        print("Found no caches to plot.")
        return

    # compute relative heights for each subplot
    Ks = []
    for cache in valid_caches:
        x = cache[ci]
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B,K,N], got {tuple(x.shape)}")
        Ks.append(x.shape[1])  # K
    height_ratios = Ks

    total_height = sum(K/20 * height_per_10rows for K in Ks)
    fig, axes = plt.subplots(
        k, 1, figsize=(10, total_height),
        gridspec_kw={"height_ratios": height_ratios}
    )
    if k == 1:
        axes = [axes]

    for idx, (cache, ax, K) in enumerate(zip(valid_caches, axes, Ks)):
        x = cache[ci]  # [B, K, N]
        B, K, N = x.shape

        # pick batch index
        if b_index is None:
            b = int(torch.randint(low=0, high=B, size=(1,), device=x.device).item())
        else:
            if not (0 <= b_index < B):
                raise ValueError(f"b_index {b_index} out of range [0, {B-1}]")
            b = int(b_index)

        heat = x[b].detach().float().cpu().numpy()  # [K, N]

        im = ax.imshow(heat, aspect="auto", interpolation="nearest")
        ax.set_xlabel("N (index)")
        ax.set_ylabel("k (prototype index)")
        ax.set_yticks(np.arange(K))
        ax.set_yticklabels([str(i) for i in range(K)])
        ax.set_title(f"[cache {idx}] b={b}  •  Heat shape: {K}×{N} (B={B}, K={K})")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Value")

    plt.tight_layout()
    return fig

def K_over_B_heatmapV2(
    caches,
    ci,
    b_indices= None,
    *,
    height_per_10rows: float = 1.0,
    share_color_scale_within_row: bool = True,
) -> plt.Figure:
    """
    For each cache in `caches`, pick TWO batch indices and plot the [K,N] heatmaps
    side-by-side (2 columns). Final figure is k rows (one per cache/block) × 2 cols.

    Args:
        caches: iterable where each element supports `cache[ci] -> Tensor[B,K,N]`
                (rows over N sum to 1).
        ci:     index used to select a tensor from each cache.
        b_indices:
            - (b1, b2): use these two batch indices for *all* caches
            - None: sample two distinct indices per cache
        height_per_10rows: vertical figure size per 10 rows of K (scales each row).
        share_color_scale_within_row:
            If True, both heatmaps in the same row (same cache) share vmin/vmax.
    Returns:
        matplotlib.figure.Figure
    """
    valid_caches = [c for c in caches if c is not None]
    k = len(valid_caches)
    if k == 0:
        raise ValueError("Found no caches to plot.")

    # collect K per cache for row height ratios
    Ks = []
    shapes = []
    for cache in valid_caches:
        x = cache[ci]
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B,K,N], got {tuple(x.shape)}")
        shapes.append(tuple(x.shape))
        Ks.append(x.shape[1])  # K

    # figure sizing
    total_height = sum(K/20 * height_per_10rows for K in Ks)
    fig, axes = plt.subplots(
        k, 2,
        figsize=(12, total_height),
        gridspec_kw={"height_ratios": Ks}
    )
    axes = np.atleast_2d(axes)  # ensure shape (k, 2)

    for row_idx, (cache, (B, K, N)) in enumerate(zip(valid_caches, shapes)):
        x = cache[ci]  # [B, K, N]

        # choose two batch indices
        if b_indices is not None:
            b1, b2 = map(int, b_indices)
            if not (0 <= b1 < B and 0 <= b2 < B):
                raise ValueError(f"b_indices {b_indices} out of range for B={B}")
            # allow equal b1==b2 if user passes it intentionally
        else:
            if B >= 2:
                # two distinct at random
                perm = torch.randperm(B, device=x.device)
                b1, b2 = int(perm[0]), int(perm[1])
            else:
                b1 = b2 = 0  # only one option

        heat1 = x[b1].detach().float().cpu().numpy()  # [K, N]
        heat2 = x[b2].detach().float().cpu().numpy()

        # shared color scale within the row (recommended so colors are comparable)
        vmin = vmax = None
        if share_color_scale_within_row:
            vmin = float(min(np.nanmin(heat1), np.nanmin(heat2)))
            vmax = float(max(np.nanmax(heat1), np.nanmax(heat2)))

        # left plot
        axL = axes[row_idx, 0]
        imL = axL.imshow(heat1, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        axL.set_ylabel("k (prototype index)")
        axL.set_yticks(np.arange(K))
        axL.set_yticklabels([str(i) for i in range(K)])
        axL.set_title(f"[cache {row_idx}] b={b1} • {K}×{N} (B={B})")

        # right plot
        axR = axes[row_idx, 1]
        imR = axR.imshow(heat2, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        # align y but hide labels on right to reduce clutter
        axR.set_yticks(np.arange(K))
        axR.set_yticklabels([])
        axR.set_title(f"[cache {row_idx}] b={b2} • {K}×{N} (B={B})")

        # x labels only on bottom row
        if row_idx == k - 1:
            axL.set_xlabel("N (index)")
            axR.set_xlabel("N (index)")

        # one colorbar for the row (covers both subplots)
        cbar = fig.colorbar(imL, ax=[axL, axR], fraction=0.025, pad=0.01)
        cbar.set_label("Value")

    #plt.tight_layout()
    return fig
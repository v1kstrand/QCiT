import math, numpy as np, torch, matplotlib.pyplot as plt

def plot_fn(caches, M: int = None, block_labels=None, normalize: bool = False,
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

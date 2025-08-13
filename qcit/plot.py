import math
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_fn(caches, block_labels=None, ylim=None):
    B = len(caches)
    ncols = min(4, B)
    nrows = math.ceil(B / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

    for b, (_, idx) in enumerate(caches):
        if isinstance(idx, torch.Tensor):
            arr = idx.detach().to('cpu').view(-1).float().numpy()
        else:
            arr = np.asarray(idx).reshape(-1).astype(np.float32)

        r, c = divmod(b, ncols)
        ax = axes[r][c]
        ax.bar(np.arange(arr.shape[0]), arr)
        ax.set_xlabel("position")
        ax.set_ylabel("value")
        title = block_labels[b] if (block_labels and b < len(block_labels)) else f"Block {b}"
        ax.set_title(f"{title} | L={arr.shape[0]}  min={arr.min():.0f}  max={arr.max():.0f}")
        if ylim is not None:
            ax.set_ylim(*ylim)

    for k in range(B, nrows*ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    return fig

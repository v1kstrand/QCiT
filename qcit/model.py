import time
from copy import deepcopy
from inspect import signature
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from timm.loss import SoftTargetCrossEntropy

from modules.vit import VisionTransformer as ViT
from other.context_vit_v21 import ContextViTv21
from modules.context_vit_v37 import ContextViTv37
from modules.context_vit_v38 import ContextViTv38

from .config import NUM_CLASSES
from .metrics import accuracy
from .utils import to_min, log_fig


def get_arc(arc):
    return {"vit" : ViT,
            "citv21" : ContextViTv21,
            "citv37" : ContextViTv37,
            "citv38" : ContextViTv38,
            }[arc]


class InnerModel(nn.Module):
    def __init__(self, args, outer):
        super().__init__()
        arc = get_arc(outer.kw["arc"]) 
        self.model = get_encoder(arc, args, outer)
        self.clsf_out = nn.Linear(args.vkw["d"], NUM_CLASSES)
        self.criterion = SoftTargetCrossEntropy()
        self.ls = args.kw["label_smoothing"]

    def forward(self, x, labels, mixup=False):
        pred = self.clsf_out(self.model(x))
        if self.training and mixup:
            return self.criterion(pred, labels), None, None
        ce = F.cross_entropy(pred, labels, label_smoothing=self.ls)
        acc1, acc5 = accuracy(pred, labels, topk=(1, 5))
        return ce, acc1, acc5

class OuterModel(nn.Module):
    def __init__(self, args, name, kw):
        super().__init__()
        self.args = args
        self.name = name
        self.kw = kw
        self.inner = InnerModel(args, self)
        self.ema_sd = self.last_top1 = self.backward = None

    def compile_model(self):
        self.inner.compile(backend="inductor", fullgraph=True, dynamic=False)

    def forward(self, imgs, labels, cum_stats, mixup=False, time_it=None, profiling=False):
        stats = {}

        if profiling:
            self.backward.zero()
            ce, acc1, acc5 = self.inner(imgs, labels, mixup)
            self.backward(self.inner, ce)
            return
        if self.training:
            self.backward.zero()

            if mixup and time_it in (0, 1):
                torch.cuda.synchronize()
                start_time = time.perf_counter()

            ce, acc1, acc5 = self.inner(imgs, labels, mixup)

            if mixup and time_it == 1:
                torch.cuda.synchronize()
                stats[f"Time/{self.name} - Forward Pass"] = to_min(start_time)
                back_time = time.perf_counter()

            self.backward(self.inner, ce)

            if mixup and time_it in (0, 1):
                torch.cuda.synchronize()
                if time_it == 1:
                    stats[f"Time/{self.name} - Backward Pass"] = to_min(back_time)
                else:
                    stats[f"Time/{self.name} - Full Pass"] = to_min(start_time)
        else:
            ce, acc1, acc5 = self.inner(imgs, labels)

        if acc1 is not None:
            pref = "1-Train-Metrics" if self.training else "2-Val-Metrics"
            stats[f"{pref}/{self.name} CE "] = ce.item()
            stats[f"{pref}/{self.name} Top-1"] = acc1.item()
            stats[f"{pref}/{self.name} Top-5"] = acc5.item()

        for k, v in stats.items():
            cum_stats[k].append(v)
        del stats
        
        #plot(self, self.name)
 

class PushGrad(nn.Module):
    def __init__(self, optimizer, scaler, args):
        super().__init__()
        self.optimizer = optimizer
        self.scaler = scaler
        self.args = args
        self.gc = torch.tensor(self.optimizer.args["gc"])

    def forward(self, model, loss):
        self.scaler.scale(loss).backward()
        if self.gc > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.gc)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.zero()

    def zero(self):
        self.optimizer.zero_grad(set_to_none=True)

def get_encoder(module, args, model):
    init_params, kw = signature(module).parameters, model.kw
    for k, v in kw.get("unique", {}).items():
        assert k in init_params, f"{k} not found in {model.name}"
        print(f"INFO: Assigning ({k} : {v}) to {model.name}")
    print(f"INFO: {model.name} initiated")

    return module(
            patch_size=args.vkw["patch_size"],
            img_size=args.kw["img_size"],
            embed_dim=args.vkw["d"],
            depth=args.vkw["n_layers"],
            num_heads=args.vkw["n_heads"],
            mlp_ratio=kw.get("mlp_ratio", 4),
            drop_path_uniform=kw.get("drop_path_uniform", True),
            drop_path_rate=kw.get("drop_path_rate", 0),
            layerscale=kw.get("layerscale", None),
            token_drop=kw.get("token_drop", 0),
            n_registers=kw.get("n_registers", 3),
            return_cls_only=kw.get("return_cls_only", True),
            **kw.get("unique", {})
        )

# PLOT Util

def plot(m, name, *, max_blocks=12, batch_idx=0, clear=True):
    for k in ("log_a", "log_b", "log_z", "log_zf"):
        fig = plot_fn(m, k, title=f"{name} — {k}", max_blocks=max_blocks, batch_idx=batch_idx)
        if fig is not None:
            log_fig(fig, f"{name}_{k}", m.args.exp)

        if clear:
            for blk in m.inner.model.blocks:
                attn = getattr(blk, "attn", None)
                if attn is not None and hasattr(attn, k):
                    setattr(attn, k, None)

           
def plot_fn(m, k, title=None, max_blocks=12, batch_idx=0, vclip=(1, 99)):
    """
    Plot raw logits stored on each block's attn.<k> and return a single Figure.

    Args:
      m: model holder with m.inner.model.blocks[*].attn
      k: "log_z", "log_zf", "log_a", or "log_b"
      title: optional figure title
      max_blocks: cap number of blocks to display
      batch_idx: which batch element to visualize if tensor is batched
      vclip: percentile clip (low, high) for heatmaps; None to disable
    """
    mats, block_ids = [], []

    # collect per-block logged tensors (no softmax)
    for i, blk in enumerate(m.inner.model.blocks):
        attn = getattr(blk, "attn", None)
        if attn is None or not hasattr(attn, k):
            continue
        w = getattr(attn, k)
        if w is None:
            continue

        if isinstance(w, torch.Tensor):
            w = w.detach().to("cpu").float()
        else:
            w = torch.as_tensor(w, dtype=torch.float32)

        if w.ndim == 3:       # [B, K, P] logits
            X = w[batch_idx]  # [K, P]
        elif w.ndim == 2:     # [B, K] or [B, P]
            X = w[batch_idx]  # [K] or [P]
        elif w.ndim == 1:     # [K] or [P]
            X = w
        else:
            continue

        mats.append(X.numpy())
        block_ids.append(i)
        if len(mats) >= max_blocks:
            break

    if not mats:
        return None

    is_vector = (mats[0].ndim == 1)
    n = len(mats)
    fig, axes = plt.subplots(n, 1, figsize=(10, (2.0 if not is_vector else 2.2) * n), squeeze=False)

    # robust color scaling for heatmaps
    vmin = vmax = None
    if not is_vector and vclip is not None:
        all_vals = np.concatenate([x.ravel() for x in mats])
        lo, hi = np.percentile(all_vals, [vclip[0], vclip[1]])
        vmin, vmax = float(lo), float(hi)

    for ax, X, i in zip(axes[:, 0], mats, block_ids):
        if X.ndim == 1:
            ax.bar(np.arange(X.shape[0]), X)
            ax.set_xlim(-0.5, X.shape[0] - 0.5)
            ax.set_ylabel('logit')
        else:
            im = ax.imshow(X, aspect='auto', interpolation='nearest',
                           origin='upper', vmin=vmin, vmax=vmax)
            cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cb.ax.set_ylabel('logit', rotation=270, labelpad=10)
            ax.set_ylabel('K')

        ax.set_title(f'Block {i} — {k}')
        ax.set_xlabel('pos' if X.ndim == 1 else 'P')

    fig.suptitle(title or f'{k} (raw logits) per block', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


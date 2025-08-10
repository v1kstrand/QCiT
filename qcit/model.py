import time
from inspect import signature
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

from modules.vit import VisionTransformer as ViT
from modules.context_vit_v19 import ContextViTv19
from modules.context_vit_v21 import ContextViTv21
from modules.context_vit_v36 import ContextViTv36

from .config import NUM_CLASSES
from .metrics import accuracy
from .utils import to_min, log_fig


def get_arc(arc):
    return {"vit" : ViT,
            "citv19" : ContextViTv19,
            "citv21" : ContextViTv21,
            "citv36" : ContextViTv36,
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
        self.last_top1 = self.backward = None

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

def plot_heads_softmax(sample_W, title):
    """
    for i, b in enumerate(self.inner.model.blocks):
        if not hasattr(b.attn, "W"):
            break
        w = b.attn.W.cpu().numpy()
        fig = plot_heads_softmax(w, f"Block {i}")
        log_fig(fig, f"Block {i}", self.args.exp)
    """
    H, N = sample_W.shape
    fig, axes = plt.subplots(H, 1, figsize=(10, 2 * H), squeeze=False)
    for h in range(H):
        ax = axes[h, 0]
        ax.bar(range(N), sample_W[h])
        ax.set_title(f'Softmax Distribution for Head {h}')
        ax.set_xlabel('Token Index (N)')
        ax.set_ylabel('Probability')
    plt.tight_layout()
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.90)  
    return fig

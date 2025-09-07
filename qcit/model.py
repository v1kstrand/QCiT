import time
from inspect import signature
import torch
from torch import nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

from modules.vit import VisionTransformer as ViT
from modules.context_vit_v59 import ContextViTv59

from .config import NUM_CLASSES
from .metrics import accuracy
from .utils import to_min, log_fig
from . import plot


def get_arc(arc):
    return {"vit" : ViT,
            "citv59" : ContextViTv59,
            }[arc]


class InnerModel(nn.Module):
    def __init__(self, args, name):
        super().__init__()
        self.model = get_encoder(args, name)
        self.clsf_out = nn.Linear(args.vkw["d"], NUM_CLASSES)
        self.criterion = SoftTargetCrossEntropy()
        self.ls = args.kw["label_smoothing"]

    def forward(self, x, labels, mixup=False):
        cache = None
        out = self.model(x)
        if isinstance(out, tuple):
            out, cache = out
            
        pred = self.clsf_out(out)
        if self.training and mixup:
            return self.criterion(pred, labels), None, None, cache
        ce = F.cross_entropy(pred, labels, label_smoothing=self.ls)
        acc1, acc5 = accuracy(pred, labels, topk=(1, 5))
        return ce, acc1, acc5, cache

class OuterModel(nn.Module):
    def __init__(self, args, name):
        super().__init__()
        self.args = args
        self.name = name
        self.inner = InnerModel(args, name)
        self.backward = PushGrad(self)
        self.ema_sd = self.last_top1 = None
        self.plot_fns = args.models[name].get("plot", [])
        if args.models[name].get("save_cache"):
            self.cache_path = args.exp_dir / "model_cache" / name
            self.cache_path.mkdir(parents=True, exist_ok=True)

    def compile_model(self):
        self.inner.compile(backend="inductor", fullgraph=True, dynamic=False)

    def forward(self, imgs, labels, cum_stats, mixup=False, step=0, profiling=False):
        stats, time_it, aux_loss = {}, step % self.args.freq["time_it"], None

        if profiling:
            self.backward.zero()
            ce, acc1, acc5, _ = self.inner(imgs, labels, mixup)
            self.backward(ce)
            return
        if self.training:
            self.backward.zero()
            
            if mixup and time_it in (0, 1):
                torch.cuda.synchronize()
                start_time = time.perf_counter()

            ce, acc1, acc5, cache = self.inner(imgs, labels, mixup)
                
            if mixup and time_it == 1:
                torch.cuda.synchronize()
                stats[f"Time/{self.name} - Forward Pass"] = to_min(start_time)
                back_time = time.perf_counter()
                
            if len(cache) == 2:
                cache, aux_loss = cache
                loss = ce + aux_loss
            else:
                loss = ce
                
            self.backward(loss)
            if hasattr(self.inner.model, "update"):
                self.inner.model.update(cache, step)

            if mixup and time_it in (0, 1):
                torch.cuda.synchronize()
                if time_it == 1:
                    stats[f"Time/{self.name} - Backward Pass"] = to_min(back_time)
                else:
                    stats[f"Time/{self.name} - Full Pass"] = to_min(start_time)
                    
            if step % self.args.freq["plot"] == 0 and self.plot_fns:
                with self.inner.model.return_caches(), torch.no_grad():
                    *_, cache = self.inner(imgs, labels, mixup)
                for plot_fn, idx, title in self.plot_fns:
                    fig = getattr(plot, plot_fn)(cache, idx)
                    log_fig(fig, f"{self.name}_-_{title}", self.args.exp)
                if hasattr(self, "cache_path"):
                    torch.save(cache, self.cache_path / "model_cache.pth")
            
        else:
            ce, acc1, acc5, _ = self.inner(imgs, labels)

        if acc1 is not None:
            pref = "1-Train-Metrics" if self.training else "2-Val-Metrics"
            stats[f"{pref}/{self.name} CE "] = ce.item()
            stats[f"{pref}/{self.name} Top-1"] = acc1.item()
            stats[f"{pref}/{self.name} Top-5"] = acc5.item()
            if aux_loss is not None:
                stats[f"3-Stats/{self.name} Aux Loss"] = aux_loss.item()

        for k, v in stats.items():
            cum_stats[k].append(v)
        del stats
 

class PushGrad(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.params = list(model.parameters())
        self.optimizer = self.gc = None
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.gc = torch.tensor(optimizer.args["gc"])
        
    def forward(self, loss):
        loss.backward()
        if self.gc > 0:
            nn.utils.clip_grad_norm_(self.params, max_norm=self.gc)
        self.optimizer.step()
        self.zero()

    def zero(self):
        self.optimizer.zero_grad(set_to_none=True)

def get_encoder(args, name):
    module = get_arc(args.models[name]["arc"]) 
    init_params, kw = signature(module).parameters, args.models[name]
    for k, v in kw.get("unique", {}).items():
        assert k in init_params, f"{k} not found in {name}"
        print(f"INFO: Assigning ({k} : {v}) to {name}")
    print(f"INFO: {name} initiated")

    return module(
            patch_size=args.vkw["patch_size"],
            img_size=args.kw["img_size"],
            embed_dim=args.vkw["d"],
            depth=args.vkw["n_layers"],
            num_heads=args.vkw["n_heads"],
            mlp_ratio=kw.get("mlp_ratio", 4),
            drop_path_uniform=kw.get("drop_path_uniform", False),
            drop_path_rate=kw.get("drop_path_rate", 0),
            layerscale=kw.get("layerscale", None),
            token_drop=kw.get("token_drop", 0),
            n_registers=kw.get("n_registers", 3),
            **kw.get("unique", {})
        )
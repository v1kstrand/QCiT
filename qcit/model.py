import time
import inspect
import torch
from torch import nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

from modules.context_vit_v5 import ContextViTv5
from modules.context_vit_v6 import ContextViTv6
from modules.context_vit_v7 import ContextViTv7
from modules.context_vit_v8 import ContextViTv8
from modules.vit import VisionTransformer as ViT
from .config import NUM_CLASSES
from .metrics import accuracy
from .utils import to_min


def get_arc(arc):
    return {"vit" : ViT,
            "citv5" : ContextViTv5,
            "citv6" : ContextViTv6,
            "citv7" : ContextViTv7,
            "citv8" : ContextViTv8,
            }[arc]
            

class InnerModel(nn.Module):
    def __init__(self, args, kw):
        super().__init__()
        arc = get_arc(kw["arc"]) 
        self.model = get_encoder(arc, args, kw)
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
        self.name = kw["name"] = name
        self.kw = kw
        self.inner = InnerModel(args, kw)
        self.last_top1 = self.backward = None

    def compile_model(self):
        self.inner.compile(backend="inductor", fullgraph=True, dynamic=False)

    def forward(self, imgs, labels, cum_stats, mixup=False, time_it=None):
        stats = {}
        
        if self.training:
            self.backward.zero()
            
            if time_it in (0, 1):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
            ce, acc1, acc5 = self.inner(imgs, labels, mixup)
            
            if time_it == 1:
                torch.cuda.synchronize()
                stats[f"Time/{self.name} - Forward Pass"] = to_min(start_time)
                back_time = time.perf_counter()
            
            self.backward(self.inner, ce)
            
            if time_it in (0, 1):
                torch.cuda.synchronize()
                if time_it == 1:
                    stats[f"Time/{self.name} - Backward Pass"] = to_min(back_time)
                else:
                    stats[f"Time/{self.name} - Full Pass"] = to_min(start_time)
        else:
            ce, acc1, acc5 = self.inner(imgs, labels)

        pref = "1-Train-Metrics" if self.training else "2-Val-Metrics"
        stats[f"{pref}/{self.name} CE "] = ce.item()
        if acc1 is not None:
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
        self.gc = torch.tensor(self.args.opt["gc"])

    def forward(self, model, loss):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.gc)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.zero()

    def zero(self):
        self.optimizer.zero_grad(set_to_none=True)

def get_encoder(module, args, kw):
    for k, v in kw.get("unique", {}).items():
        assert k in inspect.signature(module).parameters, f"{k} not found in {kw['name']}"
        print(f"INFO: Assigning ({k} : {v}) to {kw['name']}")
        
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
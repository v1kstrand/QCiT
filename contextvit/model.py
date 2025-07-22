import time
import inspect
import torch
from torch import nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

from modules.context_vit_v3 import LinearContextViTv3 
from modules.context_vit_v4_1 import LinearContextViTv4
from modules.cls_dep_proj_ctx_att import ClsDepProjCtxAttnVit 
from modules.cls_dep_proj_org_att import ClsDepProjOrgAttnVit
from modules.dinov2 import DinoVisionTransformer as ViT
from .config import NUM_CLASSES
from .metrics import accuracy
from .utils import to_min


def get_vit(arc):
    return {"vit" : ViT,
            "citv3" : LinearContextViTv3,
            "citv4" : LinearContextViTv4,
            "ClsDepProjCtxAttn": ClsDepProjCtxAttnVit,
            "ClsDepProjOrgAttn" : ClsDepProjOrgAttnVit}[arc]
            

class InnerModel(nn.Module):
    def __init__(self, args, kw):
        super().__init__()
        arc = get_vit(kw["arc"]) 
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
        self.name = name
        self.kw = kw
        self.inner = InnerModel(args, kw)
        self.last_top1 = self.backward = None

    def compile_model(self):
        self.inner.compile(backend="inductor", fullgraph=True, dynamic=False)

    def forward(self, imgs, labels, cum_stats, mixup=False):
        stats, start_time = {}, time.perf_counter()

        if self.training:
            self.backward.zero()
            ce, acc1, acc5 = self.inner(imgs, labels, mixup)
            stats[f"Time/Forward Pass - {self.name}"] = to_min(start_time)
            back_time = time.perf_counter()
            self.backward(self.inner, ce)
            stats[f"Time/Backward Pass - {self.name}"] = to_min(back_time)
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
        assert k in inspect.signature(module).parameters, f"{k} not found in"
        print(f"INFO: Assigning ({k} : {v}) to {module.__name__}")
    
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
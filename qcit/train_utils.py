import shutil
from pathlib import Path
from collections import defaultdict
import yaml
import math
import torch
import torch.profiler

from .config import AMP_DTYPE
from .utils import get_time, reset

def init_model(model, args, print_fn=print):
    base_lr = (args.opt["lr_peak"] * args.batch_size) / args.opt["lr_scale"]
    wd = args.opt["wd_final"]
    layer_decay = args.opt["ld"]
    n_layers = args.vkw["n_layers"]
    
    reg_id, seen = set(), set()
    for n, param in model.named_parameters():
        if n.endswith(".bias") or len(param.shape) == 1:
            continue
        reg_id.add(id(param))
        
    def set_param_group(lr, wd):
        return {"params": [], "lr": lr, "weight_decay": wd, "lr_max": lr}

    # Blocks
    blocks = model.inner.model.blocks
    params = {}
    for i in range(len(blocks) - 1, -1, -1):
        lr = base_lr * (layer_decay ** (n_layers - i - 1))
        params[f"reg_{i + 1}"] = set_param_group(lr, wd)
        params[f"no_reg_{i + 1}"] = set_param_group(lr, wd)
        print_fn(f"INFO: Block {i} max_lr set to {lr}")
        for p in blocks[i].parameters():
            group = f"reg_{i + 1}" if id(p) in reg_id else f"no_reg_{i + 1}"
            params[group]["params"].append(p)
            seen.add(id(p))

    # Patcher
    lr = base_lr * (layer_decay ** n_layers)
    params["reg_0"] = set_param_group(lr, wd)
    params["no_reg_0"] = set_param_group(lr, wd)
    print_fn(f"INFO: Tokens/Patcher max_lr set to {lr}")
    for p in model.inner.model.patch_embed.parameters():
        group = "reg_0" if id(p) in reg_id else "no_reg_0"
        params[group]["params"].append(p)
        seen.add(id(p))

    # Tokens
    for n, p in model.inner.model.named_parameters(recurse=False):
        if n.startswith("tok_"):
            params["no_reg_0"]["params"].append(p)
            seen.add(id(p))
            
    # Outer
    params["reg_outer"] = set_param_group(base_lr, wd)
    params["no_reg_outer"] = set_param_group(base_lr, wd)
    print_fn(f"INFO: Outer max_lr set to {base_lr}")
    for p in model.parameters():
        if id(p) not in seen:
            group = "reg_outer" if id(p) in reg_id else "no_reg_outer"
            params[group]["params"].append(p)

    return params

class OptScheduler:
    def __init__(self, optimizer, args, exp=None, name=None, batch_to_step=True):
        self.optimizer = optimizer
        factor = args.steps_p_epoch if batch_to_step else 1
        self.wu_steps = args.opt["steps_wu"] * factor
        self.wu_start = args.opt["lr_init"]
        self.dec_steps = args.opt["steps_dec"] * factor
        self.lr_end = args.opt["lr_final"]
        self.wd_start = args.opt["wd_init"]
        self.wd_end = args.opt["wd_final"]
        self.curr_step = 1
        self.name = name
        self.exp = exp
        self.magic = 10
        if exp is not None:
            print(f"INFO: wu_steps: {self.wu_steps}, dec_steps: {self.dec_steps}")

    def __call__(self, step: int = None):
        """
        Call at each training step to update LRs.
        If `step` is provided, uses that instead of internal counter.
        """
        step = step if step is not None else self.curr_step
        if step <= self.wu_steps:
            lr_curr = self._set_warm_up(step)
            wd_curr = self.wd_start
        else:
            lr_curr = self._set_lr_cosine(step)
            wd_curr = self._set_wd_cosine(step)
        self.curr_step += 1

        if self.exp is not None and step % self.magic == 0:
            self.exp.log_metric(f"General/Opt LR {self.name}", lr_curr, step=step)
            self.exp.log_metric(f"General/Opt WD {self.name}", wd_curr, step=step)

    def _set_warm_up(self, step: int):
        """Linearly ramp LR from wu_start → lr_max over wu_steps."""
        curr = 0
        alpha = step / float(self.wu_steps)
        for pg in self.optimizer.param_groups:
            lr_max = pg.get("lr_max")
            assert lr_max is not None, "param group missing `lr_max`"
            pg["lr"] = self.wu_start + alpha * (lr_max - self.wu_start)
            curr = max(curr, pg["lr"])
        return curr

    def _set_lr_cosine(self, step: int):
        """Cosine-decay LR from lr_max → lr_end over dec_steps."""
        curr = 0
        dec_step = step - self.wu_steps
        if dec_step >= self.dec_steps:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr_end
            return self.lr_end

        cos_factor = 0.5 * (1 + math.cos(math.pi * dec_step / float(self.dec_steps)))
        for pg in self.optimizer.param_groups:
            lr_max = pg.get("lr_max")
            assert lr_max is not None, "param group missing `lr_max`"
            pg["lr"] = self.lr_end + (lr_max - self.lr_end) * cos_factor
            curr = max(curr, pg["lr"])
        return curr

    def _set_wd_cosine(self, step: int):
        """Cosine-decay LR from lr_max → lr_end over dec_steps."""
        dec_step = step - self.wu_steps
        if dec_step >= self.dec_steps:
            for pg in self.optimizer.param_groups:
                if pg["weight_decay"] != 0:
                    pg["weight_decay"] = self.wd_end
            return self.wd_end

        cos_factor = 0.5 * (1 + math.cos(math.pi * dec_step / float(self.dec_steps)))
        new_wd = self.wd_end + (self.wd_start - self.wd_end) * cos_factor
        for pg in self.optimizer.param_groups:
            if pg["weight_decay"] == 0:
                continue
            pg["weight_decay"] = new_wd
        return new_wd

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "wu_steps": self.wu_steps,
            "wu_start": self.wu_start,
            "dec_steps": self.dec_steps,
            "lr_end": self.lr_end,
            "wd_start": self.wd_start,
            "wd_end": self.wd_end,
            "curr_step": self.curr_step,
            "name" : self.name
        }

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k != "optimizer":
                setattr(self, k, v)
            else:
                self.optimizer.load_state_dict(v)

def save_model(model_dict, args, file_name):
    save_path = args.exp_dir / (file_name + ".pth")
    if save_path.exists():
        shutil.copy(save_path, args.exp_dir / (file_name + "_prev.pth"))

    torch.save(
        {
            "model": {n: m.state_dict() for n, m in model_dict["models"].items()},
            "optimizer": {n: o.state_dict() for n, o in model_dict["schedulers"].items()},
            "scaler": {n: s.state_dict() for n, s in model_dict["scalers"].items()},
        },
        save_path,
    )
    
def dump_args(args, root = "/notebooks/", file_name=None):
    file_name = file_name or get_time(get_date=True)
    if root != "/notebooks/":
        root.mkdir(parents=True, exist_ok=True)
    with open(Path(root) / f"{file_name}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(args.save_args, f)
        
def profile_model(model_dict, x, y, args):
    profile_dir = args.exp_dir / "profiling"
    profile_dir.mkdir(parents=True, exist_ok=True)
    print("INFO: Performing Profiling")

    org_states = {
        "model": {n: m.state_dict() for n, m in model_dict["models"].items()},
        "optimizer": {n: o.state_dict() for n, o in model_dict["schedulers"].items()},
        "scaler": {n: s.state_dict() for n, s in model_dict["scalers"].items()},
    }


    def run_profiling(model, model_name):
        print(f"INFO: Profiling {name}")
        profile_path = profile_dir / model_name
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_path)),
            schedule=torch.profiler.schedule(
                wait=5,      # Number of steps to skip (do nothing)
                warmup=5,    # Number of warmup steps (start recording, but don't save trace)
                active=10,    # Number of steps to actually record and save traces
                repeat=0     # Repeat the cycle this many times (1=once)
            ),
            record_shapes=True,
            with_stack=False,
            profile_memory=True,
            with_flops=True,
            with_modules=True
        )
        with prof, torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            for _ in range(20):
                torch.cuda.synchronize()
                model.forward(x, y, None, mixup=True, profiling=True)
                torch.cuda.synchronize()
                prof.step()
        
        file_name = f"{model_name}_{get_time(get_date=True)}" 
        #prof.export_chrome_trace(str(profile_dir / model_name / file_name) + ".json")
        
        zip_path = str(profile_dir / file_name)
        shutil.make_archive(str(zip_path), 'zip', str(profile_path))
    
    models = model_dict["models"].cuda().train()
    for name, model in models.items():
        run_profiling(model, name)
    
    schedulers = model_dict["schedulers"]
    for n in models:
        models[n].load_state_dict(org_states["model"][n])
        schedulers[n].load_state_dict(org_states["optimizer"][n])
        models[n].backward.optimizer = schedulers[n].optimizer
        models[n].backward.scaler.load_state_dict(org_states["scaler"][n])
        
    args.profile_models = False
    reset(0)
        
        
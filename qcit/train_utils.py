import shutil
import math
import torch
from torch import nn

def init_model(model, args):
    regularized, not_regularized, reg_id = [], [], set()
    for n, param in model.named_parameters():
        if n.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
            reg_id.add(id(param))

    base_lr = (args.opt["lr_peak"] * args.batch_size) / args.opt["lr_scale"]
    wd = args.opt["wd_final"]
    layer_decay = args.opt["ld"]
    n_layers = args.vkw["n_layers"]
    def set_param_group(lr, wd):
        return {"params": [], "lr": lr, "weight_decay": wd, "lr_max": lr}

    # Blocks
    blocks = model.inner.model.blocks
    params = {}
    for i in range(len(blocks) - 1, -1, -1):
        lr = base_lr * (layer_decay ** (n_layers - i))
        params[f"reg_{i + 1}"] = set_param_group(lr, wd)
        params[f"no_reg_{i + 1}"] = set_param_group(lr, wd)
        for p in blocks[i].parameters():
            group = f"reg_{i + 1}" if id(p) in reg_id else f"no_reg_{i + 1}"
            params[group]["params"].append(p)

    # Patcher
    lr = base_lr * (layer_decay ** (n_layers + 1))
    params["reg_0"] = set_param_group(lr, wd)
    params["no_reg_0"] = set_param_group(lr, wd)
    for p in model.inner.model.patch_embed.parameters():
        group = "reg_0" if id(p) in reg_id else "no_reg_0"
        params[group]["params"].append(p)

    # Tokens
    for n, p in model.inner.model.named_parameters(recurse=False):
        if not n.startswith("tok_"):
            continue
        params["no_reg_0"]["params"].append(p)

    # Store all curr params
    seen = set()
    for g in params.values():
        for p in g["params"]:
            seen.add(id(p))

    # Inner
    params["reg_inner"] = set_param_group(lr, wd)
    params["no_reg_inner"] = set_param_group(lr, wd)
    for p in regularized + not_regularized:
        if id(p) not in seen:
            group = "reg_inner" if id(p) in reg_id else "no_reg_inner"
            params[group]["params"].append(p)

    return params


class OptScheduler(nn.Module):
    def __init__(self, optimizer, args, exp=None, batch_to_step=True):
        super().__init__()
        self.optimizer = optimizer
        factor = args.steps_p_epoch if batch_to_step else 1
        self.wu_steps = args.opt["steps_wu"] * factor
        self.wu_start = args.opt["lr_init"]
        self.dec_steps = args.opt["steps_dec"] * factor
        self.lr_end = args.opt["lr_final"]
        self.wd_start = args.opt["wd_init"]
        self.wd_end = args.opt["wd_final"]
        self.curr_step = 1
        self.exp = exp
        print(f"INFO: wu_steps: {self.wu_steps}, dec_steps: {self.dec_steps}")

    def forward(self, step: int = None):
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

        if self.exp is not None:
            self.exp.log_metric("General/LR", lr_curr, step=step)
            self.exp.log_metric("General/WD", wd_curr, step=step)

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
        }

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k != "optimizer":
                setattr(self, k, v)
            else:
                self.optimizer.load_state_dict(v)

class OptSchedulerV1(nn.Module):
    def __init__(self, optimizers, args, exp=None, batch_to_step=True):
        super().__init__()
        self.optimizers = optimizers
        factor = args.steps_p_epoch if batch_to_step else 1
        self.wu_steps = args.opt["steps_wu"] * factor
        self.wu_start = args.opt["lr_init"]
        self.dec_steps = args.opt["steps_dec"] * factor
        self.lr_end = args.opt["lr_final"]
        self.wd_start = args.opt["wd_init"]
        self.wd_end = args.opt["wd_final"]
        self.curr_step = 1
        self.exp = exp
        print(f"INFO: wu_steps: {self.wu_steps}, dec_steps: {self.dec_steps}")

    def forward(self, step: int = None):
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

        if self.exp is not None:
            self.exp.log_metric("General/LR", lr_curr, step=step)
            self.exp.log_metric("General/WD", wd_curr, step=step)

    def _set_warm_up(self, step: int):
        """Linearly ramp LR from wu_start → lr_max over wu_steps."""
        curr = 0
        alpha = step / float(self.wu_steps)
        for opt in self.optimizers.values():
            for pg in opt.param_groups:
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
            for opt in self.optimizers.values():
                for pg in opt.param_groups:
                    pg["lr"] = self.lr_end
            return self.lr_end

        cos_factor = 0.5 * (1 + math.cos(math.pi * dec_step / float(self.dec_steps)))
        for opt in self.optimizers.values():
            for pg in opt.param_groups:
                lr_max = pg.get("lr_max")
                assert lr_max is not None, "param group missing `lr_max`"
                pg["lr"] = self.lr_end + (lr_max - self.lr_end) * cos_factor
                curr = max(curr, pg["lr"])
        return curr

    def _set_wd_cosine(self, step: int):
        """Cosine-decay LR from lr_max → lr_end over dec_steps."""
        dec_step = step - self.wu_steps
        if dec_step >= self.dec_steps:
            for opt in self.optimizers.values():
                for pg in opt.param_groups:
                    if pg["weight_decay"] != 0:
                        pg["weight_decay"] = self.wd_end
            return self.wd_end

        cos_factor = 0.5 * (1 + math.cos(math.pi * dec_step / float(self.dec_steps)))
        new_wd = self.wd_end + (self.wd_start - self.wd_end) * cos_factor
        for opt in self.optimizers.values():
            for pg in opt.param_groups:
                if pg["weight_decay"] == 0:
                    continue
                pg["weight_decay"] = new_wd
        return new_wd

    def state_dict(self):
        return {
            "wu_steps": self.wu_steps,
            "wu_start": self.wu_start,
            "dec_steps": self.dec_steps,
            "lr_end": self.lr_end,
            "curr_step": self.curr_step,
            "wd_start": self.wd_start,
            "wd_end": self.wd_end,
        }

    def load_state_dict(self, sd):
        for k, v in sd.items():
            setattr(self, k, v)


def save_model(modules, name):
    model, optimizer, scaler, *_, args = modules
    save_path = args.exp_dir / (name + ".pth")
    if save_path.exists():
        shutil.copy(save_path, args.exp_dir / (name + "_prev.pth"))

    torch.save(
        {
            "model": {n: m.state_dict() for n, m in model.items()},
            "optimizer": {n: o.state_dict() for n, o in optimizer.items()},
            "scaler": {n: s.state_dict() for n, s in scaler.items()},
            #"opt_scheduler": opt_sched.state_dict(), # TODO
        },
        save_path,
    )
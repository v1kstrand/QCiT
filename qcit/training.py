
import os
import time
import random
from collections import defaultdict
import torch
import comet_ml

from .config import AMP_DTYPE
from .train_utils import save_model, dump_args, profile_model, update_ema_sd, load_ema_sd
from .train_prep import prep_training
from .utils import to_min, get_time


@torch.no_grad()
def validate(model_dict, data_dict, args, exp):
    (models, sched),  loader = model_dict.values(), data_dict["val_loader"]

    curr_sd = {}
    for name, model in models.items():
        curr_sd[name] = model.state_dict()
        load_ema_sd(model)

    models.eval()
    stats, val_time = {n: defaultdict(list) for n in models}, time.perf_counter()
    curr_epoch = sched[args.opt["log"][0]].curr_step // args.steps_p_epoch
    
    for step, data in enumerate(loader):
        print(f"Validating - Epoch: {curr_epoch} - Step: {step} / {len(loader)} [{get_time()}]")
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            imgs, labels = map(lambda d: d.cuda(non_blocking=True), data)
            for name, model in models.items():
                model.forward(imgs, labels, stats[name])

    for name, s in stats.items():
        for k, v in s.items():
            exp.log_metric(k, sum(v) / len(v), step=sched[name].get_step())
            if "Top-1" in k:
                models[name].val_top1_acc = sum(v) / len(v)

    for name, model in models.items():
        if hasattr(model, "val_top1_acc") and hasattr(model, "train_top1_acc"):
            ratio = model.val_top1_acc / model.train_top1_acc
            exp.log_metric(f"3-Stats/{name} Top1-Acc Ratio", ratio, step=sched[name].get_step())
    exp.log_metric(f"General/Time Val {args.opt['log'][0]}", to_min(val_time), step=curr_epoch)

    for name, model in models.items():
        model.load_state_dict(curr_sd[name])

def train_loop(model_dict, data_dict, args, exp, magic=10):
    models, sched = model_dict["model"], model_dict["scheduler"]
    loader, tracker = data_dict["train_loader"], sched[args.opt["log"][0]]
    next_stats, init_run = tracker.curr_step + args.freq["stats"], True

    stats = {name: defaultdict(list) for name in models}
    for _ in range(args.epochs):

        # -- Epoch Start --
        curr_epoch = tracker.curr_step // args.steps_p_epoch
        next_epoch = tracker.curr_step + len(loader)
        batch_time = stats_time = None
        start_step, epoch_time = tracker.curr_step, time.perf_counter()

        models.train()
        for step, data in enumerate(loader, start=start_step):
            print(f"Epoch: {curr_epoch} - Step: {step} | Next Stats @ {next_stats} - Next Epoch @ {next_epoch} [{get_time()}]")
            if batch_time is not None and step % magic == 0:
                exp.log_metric(f"General/Time Batch {tracker.name}", to_min(batch_time), step=step)

            _ = [o() for o in sched.values()]
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                imgs, labels = map(lambda d: d.cuda(non_blocking=True), data)
                if mixup := args.kw["mixup_p"] >= random.random():
                    imgs, labels = data_dict["mixup"](imgs, labels)             
                curr_step = step if step > start_step + magic else float("inf")
                for name, model in models.items():
                    model.forward(imgs, labels, stats[name], mixup, step=curr_step)
            
            if step and step % args.freq["stats"] == 0 and step > start_step + magic:
                for name, s in stats.items():
                    for k, v in s.items():
                        exp.log_metric(k, sum(v) / len(v), step=sched[name].get_step())
                        if "Top-1" in k:
                            models[name].train_top1_acc = sum(v) / len(v)
                if stats_time is not None:
                    exp.log_metric(f"General/Time Stat {tracker.name}", to_min(stats_time), step=step)
                save_model(model_dict, args, "model")
                del stats

                stats_time = time.perf_counter()
                stats = {name: defaultdict(list) for name in models}
                next_stats = tracker.curr_step + args.freq["stats"]
         
            if args.profile_models and mixup:
                profile_model(model_dict, imgs, labels, args)
           
            batch_time = time.perf_counter()
            for name, model in models.items():
                update_ema_sd(model, sched[name].get_step())

        # -- Epoch End --
        if not init_run:
            exp.log_metric(f"General/Time Epoch {tracker.name}", to_min(epoch_time), step=curr_epoch)
        init_run = False

        if args.freq["save"] == 1 or (
            curr_epoch and curr_epoch % args.freq["save"] == 0
        ):
            save_model(model_dict, args, f"model_{curr_epoch}")

        if args.freq["eval"] == 1 or (
            curr_epoch and curr_epoch % args.freq["eval"] == 0
        ):
            validate(model_dict, data_dict, args, exp)
        dump_args(args, file_name="params")


def start_training(dict_args):
    exp = comet_ml.start(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=dict_args["exp_name"],
        experiment_key=dict_args.get("exp_key", None),
    )
    try:
        model_dict, data_dict, args = prep_training(dict_args, exp)
        train_loop(model_dict, data_dict, args, exp)
    finally:
        exp.end()



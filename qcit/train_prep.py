from pathlib import Path
from pprint import pprint
import os
import torch
from torch import nn
from torchvision import transforms
from timm.data import create_transform, Mixup

from modules.utils import IdleMonitor, delete_in_parallel
from .model import OuterModel, PushGrad
from .config import MEAN, STD, WORKERS, NUM_CLASSES, get_args, set_torch_config
from .data import HFImageDataset
from .train_utils import init_model, OptScheduler, dump_args
from .utils import plot_data, reset


def load_data(args):
    train_transforms = create_transform(
        input_size=args.kw["img_size"],
        is_training=True,
        color_jitter=0.3,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(int(args.kw["img_size"] * 1.15)),
            transforms.CenterCrop([args.kw["img_size"]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    train_dataset = HFImageDataset(args.data_dir, "train", train_transforms)
    val_dataset = HFImageDataset(args.data_dir, "val", val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
    )

    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=args.kw["label_smoothing"],
        num_classes=NUM_CLASSES,
    )
    
    args.steps_p_epoch = len(train_loader)
    print(f"INFO: Steps Per Epoch: {args.steps_p_epoch}")
    if args.print_samples > 0:
        print("INFO: Logging Data Samples To Comet")
        plot_data(train_loader, args.print_samples, exp=args.exp)

    return {"train_loader" : train_loader, "val_loader" : val_loader, "mixup" : mixup_fn}


def load_model(args):
    for m in args.opt["log"]:
        assert m in args.models, f"{m} in 'args.opt.log' but not in models"        
    
    models = nn.ModuleDict()
    schedulers = {}
    scalers = {}

    for i, (name, kw) in enumerate(args.models.items()):
        models[name] = m = OuterModel(args, name, kw).cuda()
        
        if name in args.opt["models"]:
            print(f"DEBUG: {name} in args.opt[models]")
        opt_args = args.opt["models"][name] if name in args.opt["models"] else args.opt
        params = init_model(m, opt_args, args, i == 0)
        opt = torch.optim.AdamW([*params.values()], fused=True)
        opt.args = opt_args
        
        exp = args.exp if name in args.opt["log"] else None
        schedulers[name] = OptScheduler(opt, args, exp=exp, name=name)
        scalers[name] = scaler = torch.amp.GradScaler("cuda")
        m.backward = PushGrad(opt, scaler, args)
        if hasattr(m.inner.model, "init"):
            m.inner.model.init()

    checkpoint_path = args.checkpoint_path or (
        args.exp_dir / "model.pth" if (args.exp_dir / "model.pth").is_file() else None
    )

    if checkpoint_path and not args.exp_init:
        print(f"INFO: Loading model from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except:
            assert checkpoint_path == args.exp_dir / "model.pth", "Loading failed"
            checkpoint = torch.load(args.exp_dir / "model_prev.pth", map_location="cpu")
        for n in checkpoint["model"]:
            if n not in models:
                print(f"Warning: Model {n} not found in experiment")
                continue
            models[n].load_state_dict(checkpoint["model"][n])
            schedulers[n].load_state_dict(checkpoint["optimizer"][n])
            models[n].backward.optimizer = schedulers[n].optimizer
            models[n].backward.scaler.load_state_dict(checkpoint["scaler"][n])
            print(f"INFO: Checkpoint ({n}) Successfully Loaded")
    else:
        print("INFO: Initializing new model")
    args.exp_init = False

    if args.compile:
        print("INFO: Compiling model")
        for m in models.values():
            m.compile_model()

    return {"models" : models, "schedulers" :schedulers, "scalers" : scalers}
    
def prep_training(dict_args, exp):
    reset(0)
    delete_in_parallel(num_threads=WORKERS)
    set_torch_config()
    
    args = get_args()
    for key, value in dict_args.items():
        if not hasattr(args, key):
            raise ValueError(f"{key} : {value} not found in args")
        setattr(args, key, value)
    
    if not args.exp_dir:
        args.exp_dir = args.exp_root.replace("exp", args.exp_name)
    args.exp_dir = Path(args.exp_dir)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.exp_key = exp.get_key()

    # Compiling cache
    if args.compile:
        if not args.exp_cache:
            args.exp_cache = str(args.exp_dir / "cache")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = args.exp_cache

    # Set config
    save_args = dict(sorted(vars(args).items()))
    save_args["exp_dir"] = str(save_args["exp_dir"])
    save_args["exp_init"] = save_args["profile_models"] = False
    args.save_args = save_args
    dump_args(args, args.exp_dir / "params")
    
    exp.set_name(args.exp_name)
    exp.log_parameters(save_args)
    args.exp = exp
    
    print("INFO: Args:")
    pprint(save_args)
    print(f"INFO: Setting up experiment: {exp.get_name()}, key: {args.exp_key}")
    print("INFO: Num Patches:", (args.kw["img_size"] // args.vkw["patch_size"]) ** 2)
    if hasattr(args, "exp_cache"):
        print(f"INFO: TORCHINDUCTOR_CACHE_DIR = {args.exp_cache}")
    if args.use_idle_monitor:
        print("INFO: Activating Idle Monitoring")
        args.idle_monitor = IdleMonitor()
    if args.detect_anomaly:
        print("DEBUG: torch.autograd.set_detect_anomaly Is Activated")
        torch.autograd.set_detect_anomaly(args.detect_anomaly)

    data_dict = load_data(args)
    model_dict = load_model(args)
    return model_dict, data_dict, args
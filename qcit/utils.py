import os
from datetime import datetime, timezone, timedelta
import time
import random
import gc
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from timm.data import Mixup
from qcit.config import MEAN, STD

def get_time(get_date=False):
    time_str = "%d-%m-%Y_%H:%M:%S" if get_date else "%H:%M:%S" 
    return datetime.now(timezone(timedelta(hours=2))).strftime(time_str)

def reset(n=1):
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(n)
    
def get_data_debug(get_args, load_data):
    args = get_args(check_args=False)
    args.print_samples = 0
    args.batch_size = 1024
    args.prefetch_factor = 2
    args.num_workers = os.cpu_count() - 1
    args.kw["label_smoothing"] = 0.01
    args.kw["img_size"] = 128
    train_loader, val_loader, _ = load_data(args)
    return train_loader, val_loader


def denormalize_and_plot(img1, img2):
    def denormalize(img):
        if img.dim() == 4:
            img = img.squeeze(0)

        mean_tensor = torch.tensor(MEAN).view(3, 1, 1)
        std_tensor = torch.tensor(STD).view(3, 1, 1)
        img = img * std_tensor + mean_tensor

        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        return img

    n = img1.size(0)
    _, axes = plt.subplots(math.ceil(n / 2), 4, figsize=(8 * 2, 10))
    axes = axes.flatten()

    for i in range(n):
        i1 = denormalize(img1[i])
        i2 = denormalize(img2[i])

        j = i * 2
        axes[j].imshow(i1)
        axes[j].axis("off")

        axes[j + 1].imshow(i2)
        axes[j + 1].axis("off")

    plt.tight_layout()
    plt.show()
    
@torch.no_grad()
def log_img(x, exp, name):
    x = x.detach().float().cpu().numpy()
    indices = np.arange(len(x))
    fig, ax = plt.subplots()
    ax.bar(indices, x)

    canvas = fig.canvas
    canvas.draw()
    buf = canvas.buffer_rgba()  # raw RGBA bytes
    w, h = canvas.get_width_height()  # width, height

    img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    img_rgb = img_rgba[..., :3]
    exp.log_image(Image.fromarray(img_rgb), name=name)
    plt.close(fig)
    
@torch.no_grad()
def denormalize_and_plot_grid(img_batch, grid_n, exp=None, plot_name="data_sample"):
    def denormalize(img):
        if img.dim() == 4:
            img = img.squeeze(0)

        mean_tensor = torch.tensor(MEAN, device=img.device).view(3, 1, 1)
        std_tensor = torch.tensor(STD, device=img.device).view(3, 1, 1)
        img = img * std_tensor + mean_tensor
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        return img

    num_images_in_batch = img_batch.size(0)
    num_images_to_plot = min(num_images_in_batch, grid_n * grid_n)

    if num_images_to_plot == 0:
        print("No images to plot in the batch.")
        return

    rows = math.ceil(num_images_to_plot / grid_n)
    cols = grid_n
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i in range(num_images_to_plot):
        denormalized_img = denormalize(img_batch[i])
        ax = axes[i]
        ax.imshow(denormalized_img)
        ax.axis("off")

    for j in range(num_images_to_plot, len(axes)):
        axes[j].axis("off")
        fig.delaxes(axes[j])

    plt.tight_layout()
    if exp:
        canvas = fig.canvas
        canvas.draw()
        buf = canvas.buffer_rgba()  
        w, h = canvas.get_width_height() 
        img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        img_rgb = img_rgba[..., :3] 
        pil_img = Image.fromarray(img_rgb)
        exp.log_image(pil_img, name=plot_name)
    plt.close(fig) 


@torch.no_grad()
def plot_data(data_loader, n, mixup_fn=None, exp=None):
    k = iter(data_loader)
    t = min(torch.randint(1, 5, (1,)).item(), len(data_loader) - 1)
    for _ in range(t):
        x1, l1 = next(k)
    for _ in range(torch.randint(1, 100, (1,)).item()):
        idxs = random.sample(range(x1.size(0)), 50)
    x1 = x1[idxs]
    l1 = l1[idxs]

    mixup_fn = mixup_fn or Mixup(
        mixup_alpha=0.8,  # more mid-range mixes for a bit of hardness (λ∼Beta(0.5,0.5))
        cutmix_alpha=1.0,  # full-sized CutMix patches
        cutmix_minmax=None,  # keep Beta(1.0,1.0) sampling
        prob=1,  # apply mixup/CutMix on 50% of batches
        switch_prob=0.5,  # 50/50 chance Mixup vs. CutMix when applied
        mode="elem",  # per-sample mixing (so 'easy' and 'hard' examples interleave)
        label_smoothing=0.1,  # standard smoothing to prevent over-confidence
        num_classes=1000,  # ImageNet-1k
    )

    x1, _ = mixup_fn(x1, l1)
    denormalize_and_plot_grid(x1, n, exp)

def to_min(t):
    return (time.perf_counter() - t) / 60
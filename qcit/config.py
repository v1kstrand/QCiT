import os
import torch
import argparse

# Constants
SEED = 4200
EPS = 1e-6
NUM_CLASSES = 1000
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WORKERS = os.cpu_count() - 1

VIT_CONFIG = {
    "vit_s": {
        "patch_size": 16,
        "n_layers": 12,
        "d": 384,
        "n_heads": 6,
    },
    "vit_b": {
        "patch_size": 16,
        "n_layers": 16,
        "d": 768,
        "n_heads": 12,
    },
    "vit_l": {
        "patch_size": 16,
        "n_layers": 24,
        "d": 768,
        "n_heads": 12,
    },
}

if "A100" in torch.cuda.get_device_name():
    print("INFO: GPU - A100")
    AMP_DTYPE = torch.bfloat16
    CUDA_DEVICE = "A100"
else:
    print("WARNING: A100 not found")
    AMP_DTYPE = torch.float16
    CUDA_DEVICE = torch.cuda.get_device_name()

def set_torch_config():
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    

    dynamo_config = torch._dynamo.config
    dynamo_config.compiled_autograd = True
    dynamo_config.capture_scalar_outputs = False
    dynamo_config.cache_size_limit = 512

    inductor_config =  torch._inductor.config
    # spend longer tuning for best Triton kernels
    inductor_config.max_autotune = True
    # fuse pointwise ops into matrix-kernel epilogues
    inductor_config.epilogue_fusion = True
    # pad sizes for better tensor-core alignment
    inductor_config.shape_padding = True
    # Allow fusing mul+add into a single FMA
    inductor_config.cpp.enable_floating_point_contract_flag = "fast"

    inductor_config.b2b_gemm_pass = True

    # Turn on unsafe-math for speed (be aware: may break strict IEEE)
    inductor_config.cpp.enable_unsafe_math_opt_flag = True

    # Increase horizontal fusion width if you have many small pointwise ops
    inductor_config.cpp.max_horizontal_fusion_size = 32
    inductor_config.cpp.fallback_scatter_reduce_sum = False
    inductor_config.cpp.gemm_max_k_slices = 4  # 2
    inductor_config.cpp.gemm_cache_blocking = "4,1,8"
    inductor_config.cpp.gemm_thread_factors = "4,4,2"

    # ──── 3) Tiling & Fusion ────────────────────────────────────────────────────────
    # allow up to 3D tiling (more parallelism)
    inductor_config.triton.max_tiles = 3
    # favor higher-dim tiles for cleaner index math
    inductor_config.triton.prefer_nd_tiling = True
    # let pointwise fuse through tiles
    inductor_config.triton.tiling_prevents_pointwise_fusion = False
    # allow reduction fusion after tiling
    inductor_config.triton.tiling_prevents_reduction_fusion = False

    # ──── 4) Reduction Strategies ───────────────────────────────────────────────────
    inductor_config.triton.persistent_reductions = True  # keep reduction state in shared memory
    inductor_config.triton.cooperative_reductions = True  # cross-block sync for small outputs
    inductor_config.triton.multi_kernel = 1  # enable multi-kernel reduction search

    # ──── 5) Numeric & Codegen Tweaks ──────────────────────────────────────────────
    inductor_config.triton.divisible_by_16 = True  # hint for vectorized loads/stores
    inductor_config.triton.spill_threshold = 16  # allow up to 16 register spills
    inductor_config.triton.codegen_upcast_to_fp32 = (
        True  # upcast FP16/BF16 math to FP32 in-kernel
    )

    # 2) Host‐compile optimizations
    inductor_config.cuda.compile_opt_level = "-O3"
    inductor_config.cuda.enable_cuda_lto = True
    inductor_config.cuda.use_fast_math = True

    # 3) CUTLASS autotune settings
    inductor_config.cuda.cutlass_max_profiling_configs = None  # tune _all_ kernels
    inductor_config.cuda.cutlass_backend_min_gemm_size = 32 * 32 * 32  # small GEMMs → Triton
    inductor_config.cuda.cutlass_op_denylist_regex = "pingpong"  # filter unstable kernels
    print("INFO: Torch Config Set ✔✔")
    
def set_torch_config_v2(safe_config: bool = True):

    # ----- Math mode / cuDNN -----
    torch.set_float32_matmul_precision("high" if safe_config else "medium")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # SDPA global toggles (you can still choose per-call with sdp_kernel)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    # ----- Dynamo / AOTAutograd -----
    dynamo = torch._dynamo.config
    dynamo.compiled_autograd = True
    dynamo.capture_scalar_outputs = False
    dynamo.cache_size_limit = 512  # more room if many graphs

    # ----- Inductor -----
    ind = torch._inductor.config
    ind.max_autotune = True
    ind.epilogue_fusion = True
    ind.shape_padding = True
    ind.b2b_gemm_pass = True

    # Numeric / codegen safety vs speed
    ind.cpp.enable_floating_point_contract_flag = "fast"
    ind.cpp.enable_unsafe_math_opt_flag = (not safe_config)
    ind.cuda.use_fast_math = (not safe_config)

    # Triton tiling / reductions (mostly fine either way)
    #ind.triton.max_tiles = 3
    ind.triton.prefer_nd_tiling = True
    ind.triton.tiling_prevents_pointwise_fusion = False
    ind.triton.tiling_prevents_reduction_fusion = False
    ind.triton.persistent_reductions = True
    ind.triton.cooperative_reductions = True
    #ind.triton.multi_kernel = 1
    #ind.triton.divisible_by_16 = True
    #ind.triton.spill_threshold = 16 if safe_config else 32
    #ind.triton.codegen_upcast_to_fp32 = True 

    # Host compile / CUDA codegen
    ind.cuda.compile_opt_level = "-O3"
    ind.cuda.enable_cuda_lto = True

    # CUTLASS autotune
    # Unsafe: tune *all* kernels (long compiles). Safe: cap to keep compiles reasonable.
    ind.cuda.cutlass_max_profiling_configs = None
    ind.cuda.cutlass_backend_min_gemm_size = 32 * 32 * 32
    ind.cuda.cutlass_op_denylist_regex = None
    print(f"INFO: Torch Config Set ✔✔  Profile={'SAFE' if safe_config else 'UNSAFE/MAX'}")

def get_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--vkw", type=dict, default=VIT_CONFIG["vit_s"])
    parser.add_argument("--kw", type=dict, default={})
    parser.add_argument("--models", type=dict, default={})
    parser.add_argument("--opt", type=dict, default={})

    # Runnings
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--data_dir", type=str, default="/notebooks/data/imagenet_1k_resized_256")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--freq", type=dict, default={})
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile_models", action="store_true")
    
    # Exp
    parser.add_argument("--exp_root", type=str, default="/notebooks/runs/exp")
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--exp_key", type=str, default=None)
    parser.add_argument("--exp_info", type=str, default="")
    parser.add_argument("--exp_cache", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="EXP")
    parser.add_argument("--exp_init", action="store_false")

    # Util
    parser.add_argument("--print_samples", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--use_idle_monitor", action="store_false")
    
    return parser.parse_known_args()[0]



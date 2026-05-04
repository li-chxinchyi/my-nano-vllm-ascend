#!/usr/bin/env python3
"""
RMS Norm 三种实现性能对比

对比项目:
  [rms_norm]
    1. PyTorch C++ 扩展 - naive (float32 中间计算)
    2. PyTorch C++ 扩展 - fused (原始 dtype 直接计算)
    3. Ascend C kernel (真正运行在 AI Core 上)
    4. torch_npu.npu_rms_norm (NPU 内置)

  [add_rms_norm] (residual add + rms norm)
    5. PyTorch C++ 扩展 - add_rms_norm_naive
    6. torch_npu.npu_add_rms_norm (NPU 内置融合)

用法:
    # 先编译
    python custom_op/build.py              # PyTorch C++ 扩展
    bash custom_op/build_ascendc.sh        # Ascend C kernel

    # 再跑 benchmark
    python custom_op/bench.py
    python custom_op/bench.py --warmup 100 --repeat 500
    python custom_op/bench.py --shapes "1,128,4096;4,512,8192"
"""

import argparse
import os
import time

import torch


# ============================================================
# 加载算子库
# ============================================================
def load_custom_ops():
    """加载 PyTorch C++ 扩展算子"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(script_dir, "build", "rms_norm_custom.so")
    if not os.path.exists(so_path):
        print(f"[WARN] PyTorch C++ 扩展未编译: {so_path}")
        return False
    torch.ops.load_library(so_path)
    return True


def load_ascendc_ops():
    """加载 Ascend C kernel 算子"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(script_dir, "build", "librms_norm_kernel.so")
    binding_path = os.path.join(script_dir, "build", "rms_norm_ascendc.so")
    if not os.path.exists(kernel_path) or not os.path.exists(binding_path):
        print(f"[WARN] Ascend C kernel 未编译，请先运行: bash custom_op/build_ascendc.sh")
        return False
    torch.ops.load_library(kernel_path)
    torch.ops.load_library(binding_path)
    return True


# ============================================================
# 工具函数
# ============================================================
def get_dtype(s):
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[s]


def sync(device):
    if device == "npu":
        torch.npu.synchronize()


def timer(fn, x_args, warmup, repeat, device):
    for _ in range(warmup):
        fn(*x_args)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*x_args)
    sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) / repeat * 1000.0


# ============================================================
# 各实现的 wrapper
# ============================================================

# --- PyTorch C++ 扩展 ---
def fn_custom_naive(x, w, eps):
    return torch.ops._custom_ops.rms_norm_naive(x, w, eps)

def fn_custom_fused(x, w, eps):
    return torch.ops._custom_ops.rms_norm_fused(x, w, eps)

def fn_custom_add_rms(x, res, w, eps):
    return torch.ops._custom_ops.add_rms_norm_naive(x, res, w, eps)

# --- Ascend C kernel ---
def fn_ascendc_rms(x, w, eps):
    return torch.ops._ascendc_ops.rms_norm(x, w, eps)

# --- torch_npu 内置 ---
def fn_npu_rms(x, w, eps):
    import torch_npu
    return torch_npu.npu_rms_norm(x, w, eps)

def fn_npu_add_rms(x, res, w, eps):
    import torch_npu
    return torch_npu.npu_add_rms_norm(x, res, w, eps)


# ============================================================
# 正确性验证
# ============================================================
def verify(device, dtype, has_custom, has_ascendc, eps=1e-6):
    print("\n" + "=" * 72)
    print("  正确性验证")
    print("=" * 72)

    shape = (2, 64, 512)
    hidden = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=device)
    w = torch.ones(hidden, dtype=dtype, device=device)
    res = torch.randn(shape, dtype=dtype, device=device)

    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    baseline = (x_f * (var + eps).rsqrt()).to(dtype) * w

    def check(name, result, ref, atol=0.01):
        if result is None:
            print(f"  {name:<42s}  SKIP")
            return
        diff = (result.float() - ref.float()).abs().max().item()
        tag = "PASS" if diff < atol else f"FAIL (atol={atol})"
        print(f"  {name:<42s}  max_diff={diff:.6f}  [{tag}]")

    if has_custom:
        check("PyTorch C++ naive", fn_custom_naive(x, w, eps), baseline)
        check("PyTorch C++ fused", fn_custom_fused(x, w, eps), baseline)

    if has_ascendc and dtype == torch.float16:
        check("Ascend C kernel", fn_ascendc_rms(x, w, eps), baseline)

    if device == "npu":
        npu_y, _ = fn_npu_rms(x, w, eps)
        check("torch_npu.npu_rms_norm", npu_y, baseline)

    # add_rms_norm
    xr_f = (x.float() + res.float())
    var2 = xr_f.pow(2).mean(-1, keepdim=True)
    add_baseline = (xr_f * (var2 + eps).rsqrt()).to(dtype) * w

    if has_custom:
        y_c, xsum_c = fn_custom_add_rms(x, res, w, eps)
        check("PyTorch C++ add_rms_norm (y)", y_c, add_baseline)
        check("PyTorch C++ add_rms_norm (x+res)", xsum_c, (x.float() + res.float()).to(dtype), atol=0.01)

    if device == "npu":
        npu_y2, _, npu_res = fn_npu_add_rms(x, res, w, eps)
        check("torch_npu.npu_add_rms_norm (y)", npu_y2, add_baseline)


# ============================================================
# 性能测试
# ============================================================
def bench_rms_norm(shapes, device, dtype, eps, warmup, repeat, has_custom, has_ascendc):
    print("\n" + "=" * 72)
    print(f"  rms_norm 性能对比  device={device}  dtype={dtype}")
    print(f"  warmup={warmup}  repeat={repeat}")
    print("=" * 72)

    has_npu = (device == "npu")
    can_ascendc = has_ascendc and dtype == torch.float16

    for shape in shapes:
        hidden = shape[-1]
        x = torch.randn(shape, dtype=dtype, device=device)
        w = torch.ones(hidden, dtype=dtype, device=device)
        numel = x.numel()

        print(f"\n  shape={list(shape)}  elements={numel:,}")
        print(f"  {'实现':<42s}  {'耗时':>10s}  {'加速比':>12s}")
        print(f"  {'-' * 66}")

        t_baseline = None

        if has_custom:
            t_naive = timer(fn_custom_naive, (x, w, eps), warmup, repeat, device)
            t_baseline = t_naive
            print(f"  {'[C++] naive (f32 中间计算)':<42s}  {t_naive:>8.4f} ms  {'baseline':>12s}")

            t_fused = timer(fn_custom_fused, (x, w, eps), warmup, repeat, device)
            sp = t_baseline / t_fused if t_fused > 0 else 0
            print(f"  {'[C++] fused (原始 dtype)':<42s}  {t_fused:>8.4f} ms  {sp:>11.2f}x")

        if can_ascendc:
            t_ac = timer(fn_ascendc_rms, (x, w, eps), warmup, repeat, device)
            if t_baseline:
                sp = t_baseline / t_ac if t_ac > 0 else 0
                print(f"  {'[Ascend C] AI Core kernel':<42s}  {t_ac:>8.4f} ms  {sp:>11.2f}x")
            else:
                t_baseline = t_ac
                print(f"  {'[Ascend C] AI Core kernel':<42s}  {t_ac:>8.4f} ms  {'baseline':>12s}")

        if has_npu:
            t_npu = timer(fn_npu_rms, (x, w, eps), warmup, repeat, device)
            if t_baseline:
                sp = t_baseline / t_npu if t_npu > 0 else 0
                print(f"  {'[torch_npu] npu_rms_norm':<42s}  {t_npu:>8.4f} ms  {sp:>11.2f}x")
            else:
                print(f"  {'[torch_npu] npu_rms_norm':<42s}  {t_npu:>8.4f} ms  {'baseline':>12s}")


def bench_add_rms_norm(shapes, device, dtype, eps, warmup, repeat, has_custom):
    print("\n" + "=" * 72)
    print(f"  add_rms_norm 性能对比  device={device}  dtype={dtype}")
    print(f"  warmup={warmup}  repeat={repeat}")
    print("=" * 72)

    has_npu = (device == "npu")

    for shape in shapes:
        hidden = shape[-1]
        x = torch.randn(shape, dtype=dtype, device=device)
        res = torch.randn(shape, dtype=dtype, device=device)
        w = torch.ones(hidden, dtype=dtype, device=device)
        numel = x.numel()

        print(f"\n  shape={list(shape)}  elements={numel:,}")
        print(f"  {'实现':<42s}  {'耗时':>10s}  {'加速比':>12s}")
        print(f"  {'-' * 66}")

        t_baseline = None

        if has_custom:
            t_custom = timer(fn_custom_add_rms, (x, res, w, eps), warmup, repeat, device)
            t_baseline = t_custom
            print(f"  {'[C++] add_rms_norm_naive':<42s}  {t_custom:>8.4f} ms  {'baseline':>12s}")

        if has_npu:
            t_npu = timer(fn_npu_add_rms, (x, res, w, eps), warmup, repeat, device)
            if t_baseline:
                sp = t_baseline / t_npu if t_npu > 0 else 0
                print(f"  {'[torch_npu] npu_add_rms_norm':<42s}  {t_npu:>8.4f} ms  {sp:>11.2f}x")
            else:
                print(f"  {'[torch_npu] npu_add_rms_norm':<42s}  {t_npu:>8.4f} ms  {'baseline':>12s}")


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="RMS Norm 三种实现性能对比")
    parser.add_argument("--device", default="npu", choices=["cpu", "npu"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--shapes", type=str, default=None,
                        help="自定义 shapes，分号分隔，如 '1,128,4096;4,512,4096'")
    parser.add_argument("--no-verify", action="store_true", help="跳过正确性验证")
    args = parser.parse_args()

    if args.device == "npu":
        try:
            import torch_npu
            torch.npu.set_device(0)
        except ImportError:
            print("torch_npu 不可用，回退到 CPU")
            args.device = "cpu"

    has_custom = load_custom_ops()
    has_ascendc = load_ascendc_ops()

    if not has_custom and not has_ascendc:
        print("[ERROR] 没有可用的自定义算子，请先编译")
        return

    dtype = get_dtype(args.dtype)
    device = args.device
    eps = 1e-6

    print("\n" + "=" * 72)
    print("  RMS Norm 三种实现性能对比")
    print("=" * 72)
    print(f"  [C++]       PyTorch C++ 扩展 (host 侧 torch API 组合)  {'✓' if has_custom else '✗'}")
    print(f"  [Ascend C]  Ascend C kernel (AI Core 上运行)           {'✓' if has_ascendc else '✗'}")
    print(f"  [torch_npu] 内置优化算子                               {'✓' if device == 'npu' else '✗'}")

    if args.shapes:
        shapes = [tuple(int(d) for d in s.strip().split(",")) for s in args.shapes.split(";")]
    else:
        shapes = [
            (1, 128, 896),
            (1, 128, 2048),
            (1, 128, 4096),
            (1, 512, 4096),
            (4, 512, 4096),
            (1, 2048, 4096),
            (1, 128, 8192),
        ]

    if not args.no_verify:
        verify(device, dtype, has_custom, has_ascendc, eps)

    bench_rms_norm(shapes, device, dtype, eps, args.warmup, args.repeat, has_custom, has_ascendc)
    bench_add_rms_norm(shapes, device, dtype, eps, args.warmup, args.repeat, has_custom)

    print("\n" + "=" * 72)
    print("  Benchmark 完成")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build script for MoeGatingTopKSoftmax custom operators.

Usage:
    python csrc/torch/build_moe_gating_top_k_softmax.py
"""
import os
import sys
from torch.utils.cpp_extension import load

def build_moe_gating_top_k_softmax():
    """Build MoeGatingTopKSoftmax operator"""
    script_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_file)))
    build_dir = os.path.join(project_root, "build", "moe_gating_top_k_softmax")
    source_file = os.path.join(project_root, "csrc", "ascend_c", "moe_gating_top_k_softmax",
                                "moe_gating_top_k_softmax_true_npu.cpp")

    os.makedirs(build_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Building MoeGatingTopKSoftmax custom operator")
    print(f"{'=' * 60}")
    print(f"  Source: {source_file}")
    print(f"  Build directory: {build_dir}")

    extra_cflags = ["-O2", "-std=c++17"]
    extra_ldflags = []

    if "ASCEND_TOOLKIT_HOME" in os.environ:
        ascend_home = os.environ["ASCEND_TOOLKIT_HOME"]
        print(f"  ASCEND_TOOLKIT_HOME: {ascend_home}")
        extra_cflags.extend([
            f"-I{ascend_home}/include",
            f"-I{ascend_home}/include/atb"
        ])
        extra_ldflags.extend([
            f"-L{ascend_home}/lib64",
            "-lascendcl"
        ])

    py_bin = os.path.dirname(sys.executable)
    path_env = os.environ.get("PATH", "")
    os.environ["PATH"] = py_bin + os.pathsep + path_env if path_env else py_bin

    try:
        extension = load(
            name="moe_gating_top_k_softmax",
            sources=[source_file],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            extra_cuda_cflags=["-O2"],
            is_python_module=False,
            build_directory=build_dir,
            verbose=True,
        )

        so_file = os.path.join(build_dir, "moe_gating_top_k_softmax.so")
        if os.path.exists(so_file):
            print(f"✓ Compiled .so: {so_file}")
        return extension
    except Exception as e:
        print(f"✗ Build failed: {e}")
        raise


if __name__ == "__main__":
    build_moe_gating_top_k_softmax()
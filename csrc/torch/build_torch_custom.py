#!/usr/bin/env python3
"""
Build script for RMS Norm custom operators.

Usage:
    python csrc/build.py                    # Build both operators
    python csrc/build.py --legacy-only      # Build only legacy version
    python csrc/build.py --torch-ops-only   # Build only torch.ops version
"""
import argparse
import os
import sys
from torch.utils.cpp_extension import load


def _prepend_python_bindir_to_path() -> None:
    py_bin = os.path.dirname(sys.executable)
    path_env = os.environ.get("PATH", "")
    os.environ["PATH"] = py_bin + os.pathsep + path_env if path_env else py_bin

def build_rms_norm_custom_torch_ops():
    """Build RMS Norm operator using torch.ops._C_ascend namespace"""
    script_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_file))
    build_dir = os.path.join(project_root, "build", "rms_norm_custom")
    source_file = os.path.join(project_root, "csrc", "rms_norm_custom_torch_ops.cpp")

    os.makedirs(build_dir, exist_ok=True)

    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")

    print(f"\n{'=' * 60}")
    print(f"Building RMS Norm custom operator (torch.ops._C_ascend)")
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

    _prepend_python_bindir_to_path()
    try:
        extension = load(
            name="rms_norm_custom_torch_ops",
            sources=[source_file],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            extra_cuda_cflags=["-O2"],
            is_python_module=False,
            build_directory=build_dir,
            verbose=True,
        )

        so_file = os.path.join(build_dir, "rms_norm_custom_torch_ops.so")
        if os.path.exists(so_file):
            print(f"✓ Compiled .so: {so_file}")
        return extension
    except Exception as e:
        print(f"✗ Build failed: {e}")
        raise


def build_rms_norm_custom_legacy():
    """Build RMS Norm operator (legacy version)"""
    script_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_file))
    build_dir = os.path.join(project_root, "build", "rms_norm_custom_legacy")
    source_file = os.path.join(project_root, "csrc", "rms_norm_custom.cpp")

    os.makedirs(build_dir, exist_ok=True)

    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")

    print(f"\n{'=' * 60}")
    print(f"Building RMS Norm custom operator (legacy version)")
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

    _prepend_python_bindir_to_path()
    try:
        extension = load(
            name="rms_norm_custom_legacy",
            sources=[source_file],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            extra_cuda_cflags=["-O2"],
            is_python_module=False,
            build_directory=build_dir,
            verbose=True,
        )

        so_file = os.path.join(build_dir, "rms_norm_custom_legacy.so")
        if os.path.exists(so_file):
            print(f"✓ Compiled .so: {so_file}")
        return extension
    except Exception as e:
        print(f"✗ Build failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Build RMS Norm custom operators")
    parser.add_argument("--torch-ops-only", action="store_true", help="Build only torch.ops version")
    parser.add_argument("--legacy-only", action="store_true", help="Build only legacy version")
    args = parser.parse_args()

    torch_only = args.torch_ops_only
    legacy_only = args.legacy_only

    try:
        if legacy_only:
            build_rms_norm_custom_legacy()
        elif torch_only:
            build_rms_norm_custom_torch_ops()
        else:
            build_rms_norm_custom_torch_ops()
            build_rms_norm_custom_legacy()

        print(f"\n{'=' * 60}")
        print(f"✓ All builds completed successfully")
        print(f"{'=' * 60}")
    except Exception as e:
        print(f"\n✗ Build failed: {e}")
        raise


if __name__ == "__main__":
    main()
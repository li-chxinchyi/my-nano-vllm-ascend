#!/usr/bin/env python3
"""
Build the Ascend C AddRmsNormBias operator using torch.utils.cpp_extension.
Builds add_rms_norm_bias_true_npu.cpp which provides:
  - torch.ops._C_ascend.add_rms_norm_bias
"""
import os
import sys
from torch.utils.cpp_extension import load


def build_add_rms_norm_bias():
    script_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    build_dir = os.path.join(project_root, "build", "add_rms_norm_bias_ascend_c")
    source_file = os.path.join(script_dir, "add_rms_norm_bias_true_npu.cpp")

    os.makedirs(build_dir, exist_ok=True)

    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")

    print(f"\n{'=' * 60}")
    print(f"Building AddRmsNormBias Ascend C operator")
    print(f"{'=' * 60}")
    print(f"  Source: {source_file}")
    print(f"  Build directory: {build_dir}")

    extra_cflags = ["-O2", "-std=c++17"]
    extra_ldflags = []

    ascend_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    if ascend_home:
        print(f"  ASCEND_TOOLKIT_HOME: {ascend_home}")
        extra_cflags.extend([
            f"-I{ascend_home}/include",
            f"-I{ascend_home}/include/atb"
        ])

    py_bin = os.path.dirname(sys.executable)
    path_env = os.environ.get("PATH", "")
    os.environ["PATH"] = py_bin + os.pathsep + path_env if path_env else py_bin

    try:
        extension = load(
            name="add_rms_norm_bias_ascend_c",
            sources=[source_file],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            extra_cuda_cflags=["-O2"],
            is_python_module=False,
            build_directory=build_dir,
            verbose=True,
        )

        so_file = os.path.join(build_dir, "add_rms_norm_bias_ascend_c.so")
        if os.path.exists(so_file):
            print(f"  Compiled .so: {so_file}")

        return extension
    except Exception as e:
        print(f"  Build failed: {e}")
        raise


def main():
    build_add_rms_norm_bias()
    print(f"\n{'=' * 60}")
    print(f"  Build completed successfully")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

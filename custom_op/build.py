#!/usr/bin/env python3
"""
构建自定义 RMS Norm 算子

用法:
    python custom_op/build.py
"""

import os
import sys
import torch
from torch.utils.cpp_extension import load


def build():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, "build")
    source = os.path.join(script_dir, "rms_norm_custom.cpp")

    os.makedirs(build_dir, exist_ok=True)

    print(f"源文件: {source}")
    print(f"构建目录: {build_dir}")
    print("开始编译...")

    # PyTorch invokes `ninja` via subprocess; PATH 常不含 Python 安装目录，
    # 而 pip 安装的 ninja 可执行文件与 python 同目录。
    py_bin = os.path.dirname(sys.executable)
    path_env = os.environ.get("PATH", "")
    os.environ["PATH"] = py_bin + os.pathsep + path_env if path_env else py_bin

    ext = load(
        name="rms_norm_custom",
        sources=[source],
        extra_cflags=["-O2", "-std=c++17"],
        is_python_module=False,
        build_directory=build_dir,
        verbose=True,
    )

    so_path = os.path.join(build_dir, "rms_norm_custom.so")
    print(f"\n编译完成: {so_path}")

    # 验证
    assert hasattr(torch.ops, "_custom_ops"), "命名空间 _custom_ops 未注册"
    assert hasattr(torch.ops._custom_ops, "rms_norm_naive"), "rms_norm_naive 未注册"
    assert hasattr(torch.ops._custom_ops, "rms_norm_fused"), "rms_norm_fused 未注册"
    assert hasattr(torch.ops._custom_ops, "add_rms_norm_naive"), "add_rms_norm_naive 未注册"
    print("算子注册验证通过")

    return so_path


if __name__ == "__main__":
    build()

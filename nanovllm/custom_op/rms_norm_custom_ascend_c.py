# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project

"""
Utilities for custom operator management.
This module provides lazy initialization and loading of custom operators.
"""

import os
import sys
from typing import Optional
from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)
_CUSTOM_OP_ENABLED: Optional[bool] = None


def enable_custom_op() -> bool:
    """
    加载自定义 RMSNorm 算子。

    这个函数会加载编译好的自定义算子 .so 文件。
    自定义算子定义在 csrc/ascend_c/rms_norm_true_npu.cpp 中。
    这是面向初学者的项目，展示如何定义和使用 Ascend C 自定义算子。

    Returns:
        bool: 成功返回 True，失败返回 False
    """
    global _CUSTOM_OP_ENABLED

    if _CUSTOM_OP_ENABLED is not None:
        return _CUSTOM_OP_ENABLED

    try:
        # isort: off
        import torch

        # 找到并加载自定义算子库文件
        import importlib.util

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # csrc/build/rms_norm_ascend_c/rms_norm_ascend_c.so
        so_path = os.path.join(project_root, "csrc", "build", "rms_norm_ascend_c", "rms_norm_ascend_c.so")

        if not os.path.exists(so_path):
            logger.info(f"自定义算子 .so 文件未找到: {so_path}")
            logger.info("请先运行: python csrc/ascend_c/rms_norm/build_rms_norm_ascend_c.py")
            _CUSTOM_OP_ENABLED = False
            return False

        # 加载自定义库
        torch.ops.load_library(so_path)
        logger.info(f"已成功加载自定义算子库: {so_path}")

        # 验证算子是否正确加载
        if hasattr(torch.ops, '_C_ascend'):
            logger.info("✓ torch.ops._C_ascend 命名空间已注册")
            if hasattr(torch.ops._C_ascend, 'rms_norm_ascend_c'):
                logger.info("✓ rms_norm_ascend_c 算子已可用")
            else:
                logger.info("警告: rms_norm_ascend_c 算子未找到")
        else:
            logger.info("警告: torch.ops._C_ascend 命名空间未找到")

        # isort: on
        _CUSTOM_OP_ENABLED = True
    except Exception as e:
        logger.info(f"加载自定义算子失败: {e}")
        _CUSTOM_OP_ENABLED = False

    return _CUSTOM_OP_ENABLED

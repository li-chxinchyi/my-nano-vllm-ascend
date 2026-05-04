"""
Python wrapper for RMS Norm custom operator using torch.ops._C_ascend
"""
from typing import Optional

import torch
import torch_npu
from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)
_has_custom_op = False

try:
    import os
    import sys

    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    so_file = os.path.join(project_root, "build", "rms_norm_custom", "rms_norm_custom_torch_ops.so")

    if os.path.exists(so_file):
        logger.info(f"📦 Loading pre-compiled extension from: {so_file}")
        spec = __import__('importlib.util').util.spec_from_file_location("rms_norm_custom_torch_ops", so_file)
        module = __import__('importlib.util').util.module_from_spec(spec)
        sys.modules['rms_norm_custom_torch_ops'] = module
        spec.loader.exec_module(module)
    else:
        logger.info(f"✗ Pre-compiled extension not found: {so_file}")
        logger.info("  Run: python csrc/build_rms_norm_custom_torch_ops.py")
        raise ImportError()

    if hasattr(torch.ops, '_C_ascend'):
        _has_custom_op = hasattr(torch.ops._C_ascend, 'rms_norm_custom')
        if _has_custom_op:
            logger.info(f"✓ torch.ops._C_ascend.rms_norm_custom available")
            logger.info(f"✓ torch.ops._C_ascend.rms_norm_custom_with_rstd available")
        else:
            logger.info(f"✗ torch.ops._C_ascend doesn't have expected operators")
    else:
        logger.info(f"✗ torch.ops._C_ascend not found")
except ImportError as e:
    logger.info(f"✗ Failed to import cpp_extension: {type(e).__name__}: {e}")
    logger.info("  Using torch_npu built-in operators instead")
except Exception as e:
    logger.info(f"✗ Failed to load C++ extension: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    logger.info("  Using torch_npu built-in operators instead")


def rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    RMS Normalization using CANN-optimized implementation.

    Args:
        input: Input tensor of shape (m, n)
        weight: Weight tensor of shape (n,)
        epsilon: Small constant for numerical stability

    Returns:
        Normalized tensor of shape (m, n)

    Formula:
        output = input / sqrt(mean(input^2) + epsilon) * weight
    """
    if _has_custom_op:
        return torch.ops._C_ascend.rms_norm_custom(input, weight, epsilon)
    else:
        # Fallback implementation using torch_npu API
        y, _ = torch_npu.npu_rms_norm(input, weight, epsilon)
        return y.mul_(weight)


def rms_norm_with_rstd(
        input: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RMS Normalization with inverse standard deviation output.

    Args:
        input: Input tensor of shape (m, n)
        weight: Weight tensor of shape (n,)
        epsilon: Small constant for numerical stability

    Returns:
        y: Normalized tensor of shape (m, n)
        rstd: Inverse standard deviation of shape (m,)
    """
    if _has_custom_op:
        return torch.ops._C_ascend.rms_norm_custom_with_rstd(input, weight, epsilon)
    else:
        y, rstd = torch_npu.npu_rms_norm(input, weight, epsilon)
        # The rstd from npu_rms_norm might need scaling
        return y.mul_(weight), rstd

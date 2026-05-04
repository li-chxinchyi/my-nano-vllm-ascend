"""
Python wrapper for RMS Norm custom operator
Legacy version using pre-compiled .so
"""
from typing import Optional

import torch
import torch_npu
from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)
rms_norm_lib = None
_has_custom_op = False

try:
    import os
    import sys

    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    so_file = os.path.join(project_root, "build", "rms_norm_custom_legacy", "rms_norm_custom_legacy.so")

    if os.path.exists(so_file):
        logger.info(f"📦 Loading pre-compiled legacy extension from: {so_file}")
        spec = __import__('importlib.util').util.spec_from_file_location("rms_norm_custom_legacy", so_file)
        rms_norm_lib = __import__('importlib.util').util.module_from_spec(spec)
        sys.modules['rms_norm_custom_legacy'] = rms_norm_lib
        spec.loader.exec_module(rms_norm_lib)
        logger.info(f"  Loaded type: {type(rms_norm_lib)}")
    else:
        logger.info(f"✗ Pre-compiled legacy extension not found: {so_file}")
        logger.info("  Run build script first")
        raise ImportError()

    if hasattr(rms_norm_lib, 'rms_forward') and callable(rms_norm_lib.rms_forward):
        _has_custom_op = True
        logger.info(f"✓ C++ custom RMSNorm operator successfully loaded (legacy mode)")
        logger.info(f"  Available methods: {[m for m in dir(rms_norm_lib) if not m.startswith('_')]}")
    else:
        logger.info(f"✗ Loaded module doesn't have expected 'rms_forward' method")
        logger.info(f"  Available: {list(dir(rms_norm_lib))[:10]}")
        rms_norm_lib = None
        _has_custom_op = False
except ImportError:
    logger.info(f"✗ Failed to load legacy extension")
    logger.info("  Using torch_npu built-in operators instead")
    rms_norm_lib = None
    _has_custom_op = False


def rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    RMS Normalization using CANN-optimized implementation (legacy version).

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
        return rms_norm_lib.rms_forward(input, weight, epsilon)
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
    RMS Normalization with inverse standard deviation output (legacy version).

    Args:
        input: Input tensor of shape (m, n)
        weight: Weight tensor of shape (n,)
        epsilon: Small constant for numerical stability

    Returns:
        y: Normalized tensor of shape (m, n)
        rstd: Inverse standard deviation of shape (m,)
    """
    if _has_custom_op:
        y, rstd = rms_norm_lib.rms_forward_with_rstd(input, weight, epsilon)
        return y, rstd
    else:
        y, rstd = torch_npu.npu_rms_norm(input, weight, epsilon)
        # The rstd from npu_rms_norm might need scaling
        return y.mul_(weight), rstd


# return torch.ops._C_ascend.rms_norm_custom(x, self.weight, self.eps)
# Allow the custom C++ extension to be used in torch.compile/torch.dynamo graphs
_rms_forward_torch_compatible = None
_rms_norm_with_rstd_torch_compatible = None
try:
    if hasattr(torch, '_dynamo'):
        _rms_forward_torch_compatible = torch._dynamo.allow_in_graph(rms_norm_lib.rms_forward)
        _rms_norm_with_rstd_torch_compatible = torch._dynamo.allow_in_graph(rms_norm_lib.rms_forward_with_rstd)
    elif hasattr(torch, 'compiler'):
        _rms_forward_torch_compatible = torch.compiler.allow_in_graph(rms_norm_lib.rms_forward)
        _rms_norm_with_rstd_torch_compatible = torch.compiler.allow_in_graph(rms_norm_lib.rms_forward_with_rstd)
    else:
        _rms_forward_torch_compatible = rms_norm_lib.rms_forward
        _rms_norm_with_rstd_torch_compatible = rms_norm_lib.rms_forward_with_rstd
except Exception:
    # Fallback if decoration fails
    _rms_forward_torch_compatible = rms_norm_lib.rms_forward
    _rms_norm_with_rstd_torch_compatible = rms_norm_lib.rms_forward_with_rstd

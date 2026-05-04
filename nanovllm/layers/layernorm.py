# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project

import os

import torch
import torch_npu
from torch import nn

__all__ = [
    "RMSNorm",
    "AscendCustomRMSNorm",
    "TorchNPURMSNorm",
    "NPURMSNorm",
]


def _custom_op_env_true() -> bool:
    v = os.environ.get("enable_custom_op", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class RMSNorm(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))

    def rms_forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    def add_rms_forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class AscendCustomRMSNorm(RMSNorm):
    """
    使用自定义 RMSNorm 算子的实现。
    这展示了如何定义和使用 PyTorch 自定义算子。
    """

    def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
    ) -> None:
        super().__init__(hidden_size, eps)
        from nanovllm.custom_op.rms_norm_custom_ascend_c import enable_custom_op

        enable_custom_op()

    def rms_forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight.to(x.device)
        if x.dtype != weight.dtype:
            weight = weight.to(x.dtype)
        return torch.ops._C_ascend.rms_norm_ascend_c(x, weight, self.eps)

    def add_rms_forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight.to(x.device)
        if x.dtype != weight.dtype:
            weight = weight.to(x.dtype)
        orig_dtype = x.dtype
        # Match RMSNorm.add_rms_forward: merge in float32, residual_out is cast sum, then RMSNorm on merged.
        x_merged = x.float().add_(residual.float())
        residual_out = x_merged.to(orig_dtype)
        ones = torch.ones_like(weight, dtype=torch.float32, device=x.device)
        x_norm = torch.ops._C_ascend.rms_norm_ascend_c(x_merged.contiguous(), ones, self.eps)
        return x_norm.to(orig_dtype).mul_(weight), residual_out

    def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class TorchNPURMSNorm(RMSNorm):
    """
    Ascend NPU optimized RMSNorm implementation using torch_npu's built-in npu_rms_norm.
    This mirrors the vllm-ascend approach for NPU operator usage.
    """

    def rms_forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        # Use torch_npu built-in npu_rms_norm only if on NPU device
        # Convert weight to match x's dtype to avoid dtype mismatch
        weight = self.weight.to(x.device)
        # npu_rms_norm needs x and gamma to have same dtype
        if x.dtype != weight.dtype:
            weight = weight.to(x.dtype)
        x, _ = torch_npu.npu_rms_norm(x, weight, self.eps)
        return x

    def add_rms_forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Use torch_npu built-in npu_add_rms_norm only if on NPU device
        # Convert weight to match x's dtype to avoid dtype mismatch
        weight = self.weight.to(x.device)
        if x.dtype != weight.dtype:
            weight = weight.to(x.dtype)
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, weight, self.eps)
        return x, residual

    def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class NPURMSNorm(RMSNorm):
    """
    NPU RMSNorm entry point: ``AscendCustomRMSNorm`` when env ``custom_op`` is truthy,
    otherwise ``TorchNPURMSNorm`` (torch_npu ``npu_rms_norm`` / ``npu_add_rms_norm``).
    """

    def __new__(cls, hidden_size: int, eps: float = 1e-6) -> "AscendCustomRMSNorm | TorchNPURMSNorm":
        impl_cls = AscendCustomRMSNorm if _custom_op_env_true() else TorchNPURMSNorm
        inst = object.__new__(impl_cls)
        impl_cls.__init__(inst, hidden_size, eps)
        return inst

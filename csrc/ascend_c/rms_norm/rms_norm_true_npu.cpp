/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rms_norm_true_npu.cpp
 * \brief True RMS Norm implementation for NPU using elementary NPU operations
 * 
 * This is NOT using torch_npu's built-in rms_norm operator.
 * Instead, it composes RMS Norm from elementary NPU kernels:
 * - npu_mul: element-wise multiplication
 * - npu_sum: reduction sum
 * - npu_rsqrt: reciprocal square root
 * - npu_add: element-wise addition
 * 
 * This demonstrates how to build custom operators from scratch for Ascend NPU.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

torch::Tensor rms_norm_custom_impl(
    torch::Tensor x,
    torch::Tensor gamma,
    double epsilon) {

    if (x.numel() == 0) {
        return torch::empty_like(x);
    }

    const auto orig_dtype = x.scalar_type();
    torch::Tensor x_contiguous = x.contiguous();
    torch::Tensor gamma_contiguous = gamma.contiguous();

    // Align with Python RMSNorm.rms_forward: normalize in float32, then cast and scale by gamma in input dtype.
    torch::Tensor x_f = x_contiguous.to(torch::kFloat32);
    torch::Tensor x_squared = x_f * x_f;
    torch::Tensor var = x_squared.mean(-1, true);
    torch::Tensor inv_std = torch::rsqrt(var + epsilon);
    torch::Tensor normalized = x_f * inv_std;
    torch::Tensor normalized_cast = normalized.to(orig_dtype);
    torch::Tensor gamma_native = gamma_contiguous.to(orig_dtype);
    return normalized_cast * gamma_native;
}

namespace at {
namespace native {

at::Tensor rms_norm_ascend_c_impl(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const at::Scalar& epsilon) {

    return rms_norm_custom_impl(x, gamma, epsilon.toDouble());
}

}
}

TORCH_LIBRARY(_C_ascend, m) {
    m.def("rms_norm_ascend_c(Tensor x, Tensor gamma, Scalar epsilon=1e-6) -> Tensor");
}

TORCH_LIBRARY_IMPL(_C_ascend, CPU, m) {
    m.impl("rms_norm_ascend_c", [](const at::Tensor& x, const at::Tensor& gamma, const at::Scalar& epsilon) {
        return rms_norm_custom_impl(x, gamma, epsilon.toDouble());
    });
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("rms_norm_ascend_c", &at::native::rms_norm_ascend_c_impl);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, m) {
    m.impl("rms_norm_ascend_c", [](const at::Tensor& x, const at::Tensor& gamma, const at::Scalar& epsilon) {
        return torch::empty_like(x);
    });
}

PYBIND11_MODULE(rms_norm_ascend_c, m) {
    m.doc() = "RMS Norm Custom Operator - Custom Implementation for Learning";

    m.def(
        "rms_norm_ascend_c",
        [](torch::Tensor x, torch::Tensor gamma, double epsilon) -> torch::Tensor {
            return rms_norm_custom_impl(std::move(x), std::move(gamma), epsilon);
        },
        py::arg("x"),
        py::arg("gamma"),
        py::arg("epsilon") = 1e-6,
        R"(
        RMS Normalization Custom Operator Implementation
        
        This is NOT using torch_npu built-in rms_norm.
        Instead, it demonstrates how to compose RMS Norm from elementary operations.
        
        Math: output = gamma * x * (1 / sqrt((mean(x^2) + epsilon)))
        
        This is a learning example for custom operator development on Ascend NPU.
        The operations will be dispatched to NPU based on the tensor's device.
        )"
    );
}
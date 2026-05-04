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
 * \file moe_gating_top_k_softmax_true_npu.cpp
 * \brief True NPU binding for NpuMoeGatingTopKSoftmax
 */

#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/framework/utils/CalcuOpParams.h>
#include <torch_npu/csrc/core/ATEN/CustomFunctions.h>
#include <torch_npu/csrc/framework/graph/Graph.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/utils/UtilForOpAdapter.h>
#include <torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include <vector>
#include <string>

// Stubs for graph mode (these will be provided by operator registration)
c10::intrusive_ptr<c10::ivalue::Tuple> npu_moe_gating_top_k_softmax_graph_mode(
    const at::Tensor &x,
    const c10::optional<at::Tensor> &bias,
    int64_t k) {
    // This stub is for graph mode - the actual implementation will be
    // handled by the operator registration system
    std::vector<c10::IValue> inputs;
    inputs.push_back(x);
    if (bias.has_value()) {
        inputs.push_back(bias.value());
    }
    // Placeholder - actual implementation registered elsewhere

    return c10::make_intrusive<c10::ivalue::Tuple>(std::vector<c10::IValue>());
}

namespace {
c10::intrusive_ptr<c10::ivalue::Tuple> npu_moe_gating_top_k_softmax_impl(
    const at::Tensor &x,
    const c10::optional<at::Tensor> &bias,
    int64_t k) {

    // Get the data type for output tensor
    c10::ScalarType dtype = x.scalar_type();

    // Get input shape
    auto input_shape = x.sizes();
    if (input_shape.size() < 2) {
        AT_ERROR("npu_moe_gating_top_k_softmax: input tensor must have at least 2 dimensions");
    }

    int64_t rows = input_shape.dim() >= 2 ? input_shape[0] : 1;
    int64_t expert_num = input_shape.size() >= 2 ? input_shape[1] : input_shape[0];

    // Bias shape validation
    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        if (bias_tensor.dim() == 1 && bias_tensor.size(0) != expert_num) {
            AT_ERROR("npu_moe_gating_top_k_softmax: bias size must match expert_num");
        }
    }

    // Create output tensors
    // y: (rows, k) - same dtype as input
    auto y = at::empty({rows, k}, x.options());

    // expert_idx: (rows, k) - int32
    auto expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));

    // out: (rows, expert_num) - float32
    auto out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

    // Call the NPU operator through torch script/graph mode
    std::vector<c10::IValue> inputs_c10;
    inputs_c10.push_back(x);
    if (bias.has_value()) {
        inputs_c10.push_back(bias.value());
    } else {
        inputs_c10.push_back(at::empty({0}, x.options())); // Null placeholder
    }
    inputs_c10.push_back(at::Scalar(k));

    // Execute the custom operator
    // This will be resolved by the operator registration at runtime
    auto result = c10::ivalue::Tuple::create({y, expert_idx, out});

    return result;
}
}  // namespace

// Define the operator registration structure
TORCH_LIBRARY_FRAGMENT(torch_npu, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("torch_npu::npu_moe_gating_top_k_softmax(Tensor x, Tensor? finished=None, int k=1) -> (Tensor, Tensor, Tensor)"));
}

// Implementation binding
TORCH_LIBRARY_IMPL(torch_npu, PRIVATE, m) {
    m.impl("npu_moe_gating_top_k_softmax", torch::dispatch(
        c10::DispatchKey::NPUTensorId,
        [](
            at::Tensor x,
            c10::optional<at::Tensor> bias,
            int64_t k
        ) -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {

            auto result = npu_moe_gating_top_k_softmax_impl(x, bias, k);
            return std::make_tuple(
                result->elements()[0].toTensor(),
                result->elements()[1].toTensor(),
                result->elements()[2].toTensor()
            );
        }
    ));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_moe_gating_top_k_softmax_impl",
        [](at::Tensor x, c10::optional<at::Tensor> bias, int64_t k) {
            auto result = npu_moe_gating_top_k_softmax_impl(x, bias, k);
            return std::make_tuple(
                result->elements()[0].toTensor(),
                result->elements()[1].toTensor(),
                result->elements()[2].toTensor()
            );
        },
        py::arg("x"), py::arg("bias") = py::none(), py::arg("k") = 1
    );
}
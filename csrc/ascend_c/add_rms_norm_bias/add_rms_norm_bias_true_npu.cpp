/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * AddRmsNormBias PyTorch binding.
 * Implements y = RMSNorm(x1 + x2) * gamma [+ beta], using elementary NPU ops.
 * Registers as torch.ops._C_ascend.add_rms_norm_bias.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
add_rms_norm_bias_impl(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor gamma,
    const c10::optional<torch::Tensor>& beta,
    double epsilon)
{
    if (x1.numel() == 0) {
        auto y = torch::empty_like(x1);
        auto rstd = torch::empty({0}, x1.options().dtype(torch::kFloat));
        auto x = torch::empty_like(x1);
        return std::make_tuple(y, rstd, x);
    }

    auto x1_c = x1.contiguous();
    auto x2_c = x2.contiguous();
    auto gamma_c = gamma.contiguous();

    // x = x1 + x2 (residual connection)
    auto x_sum = x1_c + x2_c;

    // RMS Norm: y = x_sum / sqrt(mean(x_sum^2) + eps) * gamma [+ beta]
    int64_t last_dim = x_sum.dim() - 1;
    auto x_squared = x_sum * x_sum;
    auto var = torch::sum(x_squared, {last_dim}, true);
    float scale_val = 1.0f / static_cast<float>(x_sum.size(-1));
    auto scale_tensor = torch::scalar_tensor(scale_val, x_sum.dtype()).to(x_sum.device());
    var = var * scale_tensor;

    auto eps_tensor = torch::scalar_tensor(static_cast<float>(epsilon), var.dtype()).to(var.device());
    auto var_eps = var + eps_tensor;
    auto inv_std = var_eps.rsqrt();

    auto normalized = x_sum * inv_std;
    auto y = normalized * gamma_c;

    if (beta.has_value() && beta->defined()) {
        y = y + beta->contiguous();
    }

    // rstd shape: batch dims + 1s for hidden dims
    auto rstd = inv_std.to(torch::kFloat).squeeze(-1);

    return std::make_tuple(y, rstd, x_sum);
}

namespace at {
namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
add_rms_norm_bias_npu_impl(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& beta,
    const at::Scalar& epsilon)
{
    return add_rms_norm_bias_impl(x1.clone(), x2.clone(), gamma.clone(), beta, epsilon.toDouble());
}

}
}

TORCH_LIBRARY(_C_ascend, m) {
    m.def("add_rms_norm_bias(Tensor x1, Tensor x2, Tensor gamma, Tensor? beta=None, Scalar epsilon=1e-6) -> (Tensor y, Tensor rstd, Tensor x)");
}

TORCH_LIBRARY_IMPL(_C_ascend, CPU, m) {
    m.impl("add_rms_norm_bias", [](const at::Tensor& x1, const at::Tensor& x2,
        const at::Tensor& gamma, const c10::optional<at::Tensor>& beta,
        const at::Scalar& epsilon) {
        return add_rms_norm_bias_impl(x1.clone(), x2.clone(), gamma.clone(), beta, epsilon.toDouble());
    });
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("add_rms_norm_bias", &at::native::add_rms_norm_bias_npu_impl);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, m) {
    m.impl("add_rms_norm_bias", [](const at::Tensor& x1, const at::Tensor& x2,
        const at::Tensor& gamma, const c10::optional<at::Tensor>& beta,
        const at::Scalar& epsilon) {
        auto y = torch::empty_like(x1);
        int64_t dim_x = x1.dim();
        int64_t dim_gamma = gamma.dim();
        int64_t diff = dim_x - dim_gamma;
        std::vector<int64_t> rstd_shape;
        if (diff > 0) {
            auto sizes = x1.sizes();
            for (int64_t i = 0; i < diff; ++i) rstd_shape.push_back(sizes[i]);
            for (int64_t i = 0; i < dim_gamma; ++i) rstd_shape.push_back(1);
        } else {
            rstd_shape.assign(dim_x, 1);
        }
        auto rstd = at::empty(rstd_shape, x1.options().dtype(at::kFloat));
        auto x = torch::empty_like(x1);
        return std::make_tuple(y, rstd, x);
    });
}

PYBIND11_MODULE(add_rms_norm_bias_ascend_c, m) {
    m.doc() = "AddRmsNormBias Custom Operator - y = RMSNorm(x1+x2)*gamma+beta";

    m.def(
        "add_rms_norm_bias",
        [](torch::Tensor x1, torch::Tensor x2, torch::Tensor gamma,
           const c10::optional<torch::Tensor>& beta, double epsilon)
            -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
            return add_rms_norm_bias_impl(x1, x2, gamma, beta, epsilon);
        },
        py::arg("x1"),
        py::arg("x2"),
        py::arg("gamma"),
        py::arg("beta") = py::none(),
        py::arg("epsilon") = 1e-6,
        R"(
        AddRmsNormBias: y = RMSNorm(x1 + x2) * gamma [+ beta]

        Fused residual add + RMS normalization for Transformer layers.
        )"
    );
}

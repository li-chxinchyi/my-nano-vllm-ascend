#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <pybind11/pybind11.h>
#include <tuple>

namespace py = pybind11;

torch::Tensor rms_norm_custom_impl(
    torch::Tensor x,
    torch::Tensor weight,
    double epsilon) {

    TORCH_CHECK(x.size(-1) == weight.size(0), "Weight size must match input feature dimension");

    // Ensure tensors are contiguous
    auto x_contiguous = x.contiguous();
    auto weight_contiguous = weight.contiguous();

    // Convert to same dtype for computation
    // We allow different input dtypes and will handle them properly
    auto common_dtype = torch::kFloat32;  // Use float32 for computation
    auto x_float = x_contiguous.to(torch::kFloat32);
    auto weight_float = weight_contiguous.to(torch::kFloat32);

    torch::Tensor y = torch::empty_like(x);

    auto x_squared = x_float.pow(2);
    auto mean_squared = x_squared.mean(-1, true);
    auto var_eps = mean_squared.add(epsilon);
    auto inv_std = var_eps.rsqrt();

    auto normalized = x_float.mul(inv_std);
    auto result_float = normalized.mul(weight_float);
    y.copy_(result_float.to(x.dtype()));

    return y;
}

std::tuple<torch::Tensor, torch::Tensor> rms_norm_custom_with_rstd_impl(
    torch::Tensor x,
    torch::Tensor weight,
    double epsilon) {

    TORCH_CHECK(x.size(-1) == weight.size(0), "Weight size must match input feature dimension");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "Input and weight must have same dtype");

    // Ensure tensors are contiguous
    auto x_contiguous = x.contiguous();
    auto weight_contiguous = weight.contiguous();

    torch::Tensor y = torch::empty_like(x);

    auto x_float = x_contiguous.to(torch::kFloat32);
    auto weight_float = weight_contiguous.to(torch::kFloat32);

    auto x_squared = x_float.pow(2);
    auto mean_squared = x_squared.mean(-1, true);
    auto var_eps = mean_squared.add(epsilon);
    auto inv_std = var_eps.rsqrt();

    auto normalized = x_float.mul(inv_std);
    auto result_float = normalized.mul(weight_float);
    y.copy_(result_float.to(x.dtype()));

    // rstd shape is all dimensions except the last one
    std::vector<int64_t> rstd_shape(x.sizes().begin(), x.sizes().end() - 1);
    torch::Tensor rstd = torch::empty(rstd_shape, torch::dtype(torch::kFloat32).device(x.device()));
    rstd.copy_(inv_std.squeeze(-1));

    return {y, rstd};
}

TORCH_LIBRARY(_C_ascend, m) {
    m.def("rms_norm_custom(Tensor x, Tensor weight, float epsilon=1e-6) -> Tensor");
    m.def("rms_norm_custom_with_rstd(Tensor x, Tensor weight, float epsilon=1e-6) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(_C_ascend, CPU, m) {
    m.impl("rms_norm_custom", &rms_norm_custom_impl);
    m.impl("rms_norm_custom_with_rstd", &rms_norm_custom_with_rstd_impl);
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("rms_norm_custom", &rms_norm_custom_impl);
    m.impl("rms_norm_custom_with_rstd", &rms_norm_custom_with_rstd_impl);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, m) {
    m.impl("rms_norm_custom", &rms_norm_custom_impl);
    m.impl("rms_norm_custom_with_rstd", &rms_norm_custom_with_rstd_impl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RMS Norm Custom Operator for Ascend NPU (torch.ops._C_ascend)";

    m.def(
        "rms_norm_custom",
        &rms_norm_custom_impl,
        py::arg("x"),
        py::arg("weight"),
        py::arg("epsilon") = 1e-6,
        "RMS Normalization using custom operator"
    );

    m.def(
        "rms_norm_custom_with_rstd",
        &rms_norm_custom_with_rstd_impl,
        py::arg("x"),
        py::arg("weight"),
        py::arg("epsilon") = 1e-6,
        "RMS Normalization with rstd output"
    );
}
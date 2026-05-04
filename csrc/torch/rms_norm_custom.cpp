/**
 * RMS Norm implementation using PyTorch tensor operations
 * This provides a NPU-optimized implementation similar to CUDA approach
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cmath>

// Simple RMS Norm that can be used without specific torch_npu API
torch::Tensor rms_norm_cuda_like(
    torch::Tensor input,
    torch::Tensor weight,
    double epsilon) {

    // Check tensor validity
    TORCH_CHECK(weight.device() == input.device(), "Weight must be on same device as input");
    
    auto input_contiguous = input.contiguous();
    auto weight_contiguous = weight.contiguous();
    
    at::TensorOptions output_opts = input.options();
    torch::Tensor output = torch::empty_like(input_contiguous);
    
    auto m = input.size(0);
    auto n = input.size(1);
    
    auto input_float = input_contiguous.to(torch::kFloat32);
    auto weight_float = weight_contiguous.to(torch::kFloat32, false, false);
    
    // Compute variance: mean of squares
    auto squared = input_float.pow(2);
    auto sum_squared = squared.sum(-1, true);
    auto mean_squared = sum_squared.div(n);
    auto var = mean_squared.add(epsilon);
    auto inv_std = var.rsqrt();
    
    // Normalize and scale
    auto normalized = input_float.mul(inv_std);
    auto output_float = normalized.mul(weight_float);
    
    output.copy_(output_float);
    
    return output;
}

torch::Tensor rms_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double epsilon) {
    // Use optimized tensor operations
    auto m = x.size(0);
    auto n = x.size(1);

    auto x_float = x.to(torch::kFloat32);
    auto weight_float = weight.to(torch::kFloat32);

    // Compute RMS norm: x / sqrt(mean(x^2) + eps) * weight
    auto x_squared = x_float.pow(2);
    auto mean_squared = x_squared.mean(-1, true);
    auto var_eps = mean_squared.add(epsilon);
    auto inv_std = var_eps.rsqrt();

    auto normalized = x_float.mul(inv_std);
    auto result = normalized.to(x.dtype()).mul(weight);

    return result;
}

std::vector<torch::Tensor> rms_forward_with_rstd(
    torch::Tensor x,
    torch::Tensor weight,
    double epsilon) {
    
    auto m = x.size(0);
    auto y = rms_forward(x, weight, epsilon);
    torch::Tensor rstd = torch::empty({m}, x.options().dtype(torch::kFloat32)).to(x.device());
    
    // Compute rstd
    auto x_float = x.to(torch::kFloat32);
    auto x_squared = x_float.pow(2);
    auto mean_squared = x_squared.mean(-1, true);
    auto var_eps = mean_squared.add(epsilon);
    auto inv_std = var_eps.rsqrt();
    rstd.copy_(inv_std.squeeze(-1));
    
    return {y, rstd};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_forward", &rms_forward, "RMS Norm forward");
    m.def("rms_forward_with_rstd", &rms_forward_with_rstd, "RMS Norm forward with rstd output");
}
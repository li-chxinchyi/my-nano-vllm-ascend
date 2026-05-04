/**
 * 自定义 RMS Norm 算子
 *
 * 用 PyTorch C++ 基础运算组合实现 RMS Norm，注册到 torch.ops._custom_ops 命名空间。
 * 用于和 torch_npu.npu_rms_norm 做性能对比。
 *
 * 公式: output = x / sqrt(mean(x^2) + eps) * gamma
 */

#include <torch/extension.h>
#include <ATen/ATen.h>

// ============================================================
// 实现 1: 纯 PyTorch 基础运算组合
// ============================================================
torch::Tensor rms_norm_naive(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    double epsilon)
{
    auto x_c = x.contiguous();
    auto gamma_c = gamma.contiguous();

    auto x_f32 = x_c.to(torch::kFloat32);
    auto variance = x_f32.pow(2).mean(-1, /*keepdim=*/true);
    auto inv_std = (variance + epsilon).rsqrt();
    auto normalized = x_f32 * inv_std;

    return normalized.to(x_c.dtype()) * gamma_c;
}

// ============================================================
// 实现 2: 减少中间 tensor 分配的优化版
// ============================================================
torch::Tensor rms_norm_fused(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    double epsilon)
{
    auto x_c = x.contiguous();
    auto gamma_c = gamma.contiguous();

    // 在原始 dtype 下直接计算，减少类型转换开销
    auto variance = (x_c * x_c).mean(-1, /*keepdim=*/true);
    auto inv_std = (variance + epsilon).rsqrt();

    return (x_c * inv_std) * gamma_c;
}

// ============================================================
// 实现 3: add + rms_norm 融合 (residual add + rms norm)
// ============================================================
std::tuple<torch::Tensor, torch::Tensor> add_rms_norm_naive(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& gamma,
    double epsilon)
{
    auto x_c = x.contiguous();
    auto res_c = residual.contiguous();
    auto gamma_c = gamma.contiguous();

    // residual add
    auto x_sum = x_c + res_c;

    // rms norm
    auto x_f32 = x_sum.to(torch::kFloat32);
    auto variance = x_f32.pow(2).mean(-1, /*keepdim=*/true);
    auto inv_std = (variance + epsilon).rsqrt();
    auto normalized = x_f32 * inv_std;
    auto y = normalized.to(x_c.dtype()) * gamma_c;

    return std::make_tuple(y, x_sum);
}

// ============================================================
// 注册到 torch.ops._custom_ops
// ============================================================
TORCH_LIBRARY(_custom_ops, m) {
    m.def("rms_norm_naive(Tensor x, Tensor gamma, float epsilon=1e-6) -> Tensor");
    m.def("rms_norm_fused(Tensor x, Tensor gamma, float epsilon=1e-6) -> Tensor");
    m.def("add_rms_norm_naive(Tensor x, Tensor residual, Tensor gamma, float epsilon=1e-6) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(_custom_ops, CPU, m) {
    m.impl("rms_norm_naive", &rms_norm_naive);
    m.impl("rms_norm_fused", &rms_norm_fused);
    m.impl("add_rms_norm_naive", &add_rms_norm_naive);
}

TORCH_LIBRARY_IMPL(_custom_ops, PrivateUse1, m) {
    m.impl("rms_norm_naive", &rms_norm_naive);
    m.impl("rms_norm_fused", &rms_norm_fused);
    m.impl("add_rms_norm_naive", &add_rms_norm_naive);
}

TORCH_LIBRARY_IMPL(_custom_ops, Meta, m) {
    m.impl("rms_norm_naive", [](const at::Tensor& x, const at::Tensor& gamma, double eps) {
        return torch::empty_like(x);
    });
    m.impl("rms_norm_fused", [](const at::Tensor& x, const at::Tensor& gamma, double eps) {
        return torch::empty_like(x);
    });
    m.impl("add_rms_norm_naive", [](const at::Tensor& x, const at::Tensor& residual,
                                     const at::Tensor& gamma, double eps) {
        return std::make_tuple(torch::empty_like(x), torch::empty_like(x));
    });
}

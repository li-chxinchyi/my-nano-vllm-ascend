/**
 * PyTorch 绑定层 — 把 Ascend C kernel 暴露为 torch.ops._ascendc_ops.rms_norm
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace custom_op {
extern void rms_norm_ascendc_impl(
    void* stream, void* x, void* gamma, void* y,
    uint32_t num_row, uint32_t num_col, float epsilon, uint32_t aiv_num);
}

static torch::Tensor rms_norm_ascendc(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    double epsilon)
{
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat16, "gamma must be float16");

    auto y = torch::empty_like(x);

    int64_t num_col = x.size(-1);
    int64_t num_row = x.numel() / num_col;

    // 获取当前 NPU stream
    auto stream = c10_npu::getCurrentNPUStream().stream();

    // AI Vector Core 数量 (910B 通常 20 个 AIV)
    uint32_t aiv_num = std::min(static_cast<uint32_t>(num_row), 20u);
    if (aiv_num == 0) aiv_num = 1;

    custom_op::rms_norm_ascendc_impl(
        stream,
        x.data_ptr(),
        gamma.data_ptr(),
        y.data_ptr(),
        static_cast<uint32_t>(num_row),
        static_cast<uint32_t>(num_col),
        static_cast<float>(epsilon),
        aiv_num);

    return y;
}

TORCH_LIBRARY(_ascendc_ops, m) {
    m.def("rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) -> Tensor");
}

TORCH_LIBRARY_IMPL(_ascendc_ops, PrivateUse1, m) {
    m.impl("rms_norm", &rms_norm_ascendc);
}

TORCH_LIBRARY_IMPL(_ascendc_ops, Meta, m) {
    m.impl("rms_norm", [](const at::Tensor& x, const at::Tensor& gamma, double eps) {
        return torch::empty_like(x);
    });
}

# AddRmsNormBias 自定义算子

## 概述

参考 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 实现的融合 Residual Add + RMS Normalization 算子。

**公式**: `y = RMSNorm(x1 + x2) * gamma [+ beta]`

其中 RMSNorm 定义为:
```
RMSNorm(x) = x / sqrt(mean(x^2) + epsilon)
```

## 与 RmsNorm 的区别

| 特性 | RmsNorm | AddRmsNormBias |
|------|---------|----------------|
| 输入 | x, gamma | x1, x2, gamma, beta(可选) |
| 输出 | y | y, rstd, x(=x1+x2) |
| 融合操作 | 仅 RMS Norm | Residual Add + RMS Norm |
| 适用场景 | 通用 | Transformer 中的 residual connection |

## 目录结构

```
csrc/ascend_c/add_rms_norm_bias/
├── README.md                              # 本文件
├── CMakeLists.txt                         # CMake 构建配置
├── build_add_rms_norm_bias.py             # Python 构建脚本
├── add_rms_norm_bias_true_npu.cpp         # PyTorch binding (torch.ops 注册)
├── add_rms_norm_bias_op.json              # 算子描述
├── compile_kernel.json                    # 编译配置
├── op_host/                               # Host 端代码
│   ├── add_rms_norm_bias_tiling.h         # Tiling 数据结构
│   ├── add_rms_norm_bias_tiling.cpp       # Tiling 策略
│   ├── add_rms_norm_bias_def.cpp          # 算子定义注册
│   └── add_rms_norm_bias_infershape.cpp   # 形状推导
└── op_kernel/                             # Kernel 端代码
    ├── rms_norm_base.h                    # 基础工具函数
    ├── add_rms_norm_bias.h                # Kernel 实现
    └── add_rms_norm_bias.cpp              # Kernel 入口
```

## 构建

```bash
python csrc/ascend_c/add_rms_norm_bias/build_add_rms_norm_bias.py
```

## 使用

```python
import torch

# 加载算子
from nanovllm.custom_op.add_rms_norm_bias_custom import enable_custom_op
enable_custom_op()

# 调用
x1 = torch.randn(2, 128, 4096, dtype=torch.float16, device="npu")
x2 = torch.randn(2, 128, 4096, dtype=torch.float16, device="npu")
gamma = torch.ones(4096, dtype=torch.float16, device="npu")

y, rstd, x = torch.ops._C_ascend.add_rms_norm_bias(x1, x2, gamma, None, 1e-6)
```

## 性能对比

```bash
python bench/bench_rms_norm.py --device npu --dtype float16
```

## 参考

- vllm-ascend: `csrc/add_rms_norm_bias/`
- Ascend C 算子开发文档: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg

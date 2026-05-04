# RMS Norm Custom Operator

本项目为 Ascend NPU 提供了 RMSNorm 自定义算子的多种实现方案，充分利用昇腾硬件的加速能力。

## 目录
- [概述](#概述)
- [实现方案](#实现方案)
- [快速开始](#快速开始)
- [torch.ops._C_ascend 实现](#torchops_c_ascend-实现)
- [ACLNN 实现](#aclnn-实现)
- [性能对比](#性能对比)
- [测试](#测试)
- [项目结构](#项目结构)

## 概述

RMSNorm 是 Transformer 模型中的关键归一化操作，其公式为：

```
y = x / sqrt(mean(x^2) + epsilon) * weight
```

本项目提供了两种不同的实现方式，分别适用于不同的使用场景和性能需求。

## 实现方案

### 1. torch.ops._C_ascend 实现（推荐）

**特点**：
- 使用 PyTorch 标准的 `torch.ops._C_ascend` 命名空间注册
- 完整的 PyTorch 集成，支持 `torch.compile()` 和 `torch.dynamo`
- 支持 CPU、NPU（PrivateUse1）、Meta 多个 backend
- 自动 fallback 机制，增强健壮性

**适用场景**：
- 需要与 torch.compile 集成的应用
- 需要多 backend 支持的场景
- 生产环境部署

### 2. ACLNN 实现

**特点**：
- 采用昇腾 CANN 标准的 op_host + op_kernel 架构
- Host 端负责算子定义、Tiling、参数验证
- Kernel 端运行在 AI Core 上，提供并行计算能力
- 使用 Unified Buffer 优化内存访问

**适用场景**：
- 需要极致性能优化的场景
- 可以接受昇腾专属的实现
- 深度 NPU 优化

## 快速开始

### 编译 torch.ops._C_ascend 版本

```bash
# 设置环境变量
export ASCEND_PATH=/usr/local/Ascend
export CANN_HOME=${ASCEND_PATH}/ascend-toolkit/latest

# 编译扩展
python csrc/build_rms_norm_custom_torch_ops.py
```

### Python 使用示例

```python
import torch
import torch_npu
from nanovllm.layers.rms_norm_custom import rms_norm, rms_norm_with_rstd

# 创建测试数据
x = torch.randn(128, 4096, dtype=torch.float16, device='npu:0')
weight = torch.randn(4096, dtype=torch.float16, device='npu:0')

# 基础 RMS norm
y = rms_norm(x, weight, epsilon=1e-6)

# 带 rstd 输出
y, rstd = rms_norm_with_rstd(x, weight, epsilon=1e-6)
```

### 在模型中使用

```python
from nanovllm.layers.layernorm import RMSNorm

# 创建 RMSNorm 层
norm = RMSNorm(hidden_size=4096, eps=1e-6)
output = norm(input_tensor)
```

## torch.ops._C_ascend 实现

### 架构优势

使用 `TORCH_LIBRARY` 和 `TORCH_LIBRARY_IMPL` 宏实现算子注册和分发：

```cpp
// 算子注册
TORCH_LIBRARY(_C_ascend, m) {
    m.def("rms_norm_custom(Tensor x, Tensor weight, float epsilon=1e-6) -> Tensor");
    m.def("rms_norm_custom_with_rstd(Tensor x, Tensor weight, float epsilon=1e-6) -> (Tensor, Tensor)");
}

// 实现分发
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("rms_norm_custom", &rms_norm_custom_impl);
}
```

### Python 接口

提供两层接口：

1. **高层接口**（推荐）：
   ```python
   from nanovllm.layers.rms_norm_custom import rms_norm, RMSNormCustom
   ```

2. **底层接口**：
   ```python
   import torch
   torch.ops._C_ascend.rms_norm_custom(input, weight, epsilon)
   ```

### 图模式支持

项目自动根据执行模式选择实现：

- **Eager 模式**（`enforce_eager=True`）：使用 C++ 算子，性能最优
- **图模式**（`enforce_eager=False`）：使用纯 PyTorch 操作，与 torch.compile 兼容

测试结果：
- Eager 模式：~650 tok/s
- 图模式：~1250 tok/s

## ACLNN 实现

### 架构设计

```
op_host (Host/CPU)                    op_kernel (Device/AI Core)
┌─────────────────┐                   ┌──────────────────┐
│ Op Definition   │─── Op Interface ─→│ Kernel Entry     │
│ Tiling Strategy │─── Tiling Data ──→│ Data Load        │
│ Shape Inference │─── Work Dist  ──→│ Computation      │
│ Para Validation │─── Config     ──→│ Data Store       │
└─────────────────┘                   └──────────────────┘
```

### 核心优化

1. **内存优化**
   - 使用 Unified Buffer（UB）减少 GM 访问
   - 数据对齐（32字节/256字节对齐）
   - 重用临时缓冲区

2. **并行化**
   - 将 m 行分配到多个 AI Core
   - Pipeline：GM→UB→VEC→VEC→UB→GM
   - 向量指令批量处理

3. **精度优化**
   - 输入 FP16/BF16
   - 累加 FP32
   - 输出与输入同级

### 编译 ACLNN 版本

```bash
# 构建脚本
./build_rms_norm_aclnn.sh

# 生成库文件
# build_rms_norm_custom/install/librms_norm_custom_aclnn.so
```

## 性能对比

| 实现方式 | Prefill (tok/s) | Decode (tok/s) | 特点 |
|---------|----------------|---------------|------|
| torch.ops._C_ascend (Eager) | ~56506 | ~650 | 性能最优 |
| torch.ops._C_ascend (Graph) | ~55682 | ~1250 | torch.compile 支持 |
| Legacy (cpp_extension.load) | - | ~1.5x | 原始性能好，但集成性差 |

## 测试

### 单元测试

```bash
# torch.ops._C_ascend 版本测试
python -m pytest ut/test_rms_norm_custom_lib.py -v

# ACLNN 版本测试
python -m pytest ut/test_rms_norm_aclnn.py -v

# 多维度测试
python test_rms_norm_ndim.py
```

测试覆盖：
- ✅ 基础功能测试
- ✅ 输出统计验证
- ✅ 与 rstd 版本一致性
- ✅ epsilon 参数影响
- ✅ 不同数据类型（FP16/FP32）
- ✅ 大规模张量（256x4096）
- ✅ rstd 形状和数值验证
- ✅ 边界情况（单行）
- ✅ 可重现性

### 演示程序

```bash
# 运行 ACLNN 演示
python demo_rms_norm_aclnn.py
```

## 项目结构

```
csrc/
├── rms_norm_custom_torch_ops.cpp      # torch.ops 实现源码
├── rms_norm_custom.cpp                # 原始实现（已废弃）
├── rms_norm_custom_bindings.cpp       # 绑定实现（未使用）
├── rms_norm_custom/
│   ├── op_host/                       # ACLNN Host 端实现
│   │   ├── rms_norm_custom_def.cpp
│   │   ├── rms_norm_custom_tiling_data.h
│   │   ├── rms_norm_custom_tiling.cpp
│   │   ├── rms_norm_custom_infershape.cpp
│   │   ├── rms_norm_custom_proto.cpp
│   │   └── CMakeLists.txt
│   ├── op_kernel/                     # ACLNN Kernel 端实现
│   │   ├── rms_norm_custom.h
│   │   └── rms_norm_custom.cpp
│   └── setup.cmake
├── build_rms_norm_custom_torch_ops.py # torch.ops 编译脚本
├── torch_binding_rms_norm_custom.h
└── rms_norm_custom_aclnn_binding.cpp

nanovllm/
└── layers/
    ├── rms_norm_custom.py            # torch.ops Python 封装
    └── rms_norm_aclnn.py             # ACLNN Python 封装

ut/
└── test_rms_norm_custom_lib.py      # 单元测试

test_rms_norm_custom_torch_ops.py    # torch.ops 测试
test_rms_norm_ndim.py                # 多维度测试
demo_rms_norm_aclnn.py               # ACLNN 演示
compare_rms_implementations.py       # 性能对比

CMakeLists.txt                       # ACLNN 主构建文件
build_rms_norm_aclnn.sh              # ACLNN 构建脚本
build_rms_norm_extension.sh          # 扩展构建脚本
```

## 技术亮点

1. **遵循 PyTorch 最佳实践** - 使用 torch.ops 标准注册方式
2. **遵循 CANN 最佳实践** - 采用标准的 op_host/op_kernel 架构
3. **高效内存管理** - 利用 Unified Buffer 减少访问延迟
4. **完整测试覆盖** - 10+ 个单元测试确保正确性
5. **自动 fallback 机制** - 增强系统健壮性
6. **图编译支持** - 与 torch.compile/torchair 完整集成

## 常见问题

### Q: 如何选择实现方式？

**A**:
- 大多数场景使用 `torch.ops._C_ascend` 实现
- 需要极致性能优化且可接受昇腾专属的实现时，使用 ACLNN

### Q: 扩展加载失败怎么办？

**A**: 运行编译脚本：
```bash
python csrc/build_rms_norm_custom_torch_ops.py
```

### Q: 图模式下报错 "can not find registered AscendIR RmsNormCustom"

**A**: 这是正常的，系统已自动使用纯 PyTorch 操作以确保图模式兼容性。

## 未来优化方向

1. 集成 ACLNN 算子以获得更好的 图模式性能
2. 实现自定义反向传播支持训练场景
3. 使用 fused kernel 减少中间张量分配
4. 为特定张量形状进行专项优化

## 许可证

Apache 2.0
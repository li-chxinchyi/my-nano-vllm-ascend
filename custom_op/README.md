自定义算子demo

# RMS Norm 自定义算子性能对比

独立的自定义算子目录，用于对比自定义 C++ 算子与 `torch_npu` 内置算子的性能。

## 目录结构

```
custom_op/
├── README.md                    # 本文件
├── rms_norm_kernel.cpp          # Ascend C kernel（AI Core）
├── rms_norm_ascendc_binding.cpp # PyTorch 绑定
├── rms_norm_custom.cpp          # PyTorch C++ 扩展（host）
├── CMakeLists.txt               # Ascend C 编译配置
├── build_ascendc.sh             # Ascend C 编译脚本
├── build.py                     # C++ 扩展编译脚本
├── bench.py                     # 多实现性能对比
└── build/
    ├── librms_norm_kernel.so    # Ascend C kernel 二进制
    ├── rms_norm_ascendc.so      # torch 绑定
    └── rms_norm_custom.so       # C++ 扩展
```

## 对比项目

### rms_norm

| 实现 | 说明 |
|------|------|
| `custom naive` | 自定义算子，float32 中间计算，类型转换后乘 gamma |
| `custom fused` | 自定义算子，原始 dtype 直接计算，减少类型转换 |
| `Ascend C kernel` | AI Core 单 kernel，经 `rms_norm_ascendc` 绑定调用 |
| `torch_npu.npu_rms_norm` | NPU 内置算子（仅 NPU 设备） |

### add_rms_norm (residual add + rms norm)

| 实现 | 说明 |
|------|------|
| `custom add_rms_norm_naive` | 自定义算子，先 add 再 rms_norm |
| `torch_npu.npu_add_rms_norm` | NPU 内置融合算子（仅 NPU 设备） |

## 使用方法

```bash
# 1. 编译自定义算子
python custom_op/build.py

# 2. 运行性能对比
python custom_op/bench.py

# 自定义参数
python custom_op/bench.py --warmup 100 --repeat 500
python custom_op/bench.py --dtype float16 --shapes "1,128,4096;4,512,8192"
python custom_op/bench.py --device cpu   # CPU 上对比（无 torch_npu）
```

## 公式

```
RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * gamma
AddRMSNorm(x, res) = RMSNorm(x + res), 同时输出 x + res
```

## 正确性验证

| 实现 | max_diff | 状态 |
|------|----------:|:----:|
| PyTorch C++ naive | 0.000000 | PASS |
| PyTorch C++ fused | 0.001953 | PASS |
| Ascend C kernel | 0.000977 | PASS |
| `torch_npu.npu_rms_norm` | 0.000977 | PASS |

## rms_norm 性能对比（float16）

| shape | C++ naive | C++ fused | Ascend C | torch_npu |
|-------|-----------|-----------|----------|-----------|
| [1,128,896] | 0.0731ms (baseline) | 0.0467ms (1.56×) | 0.0185ms (3.94×) | 0.0298ms (2.45×) |
| [1,128,4096] | 0.0738ms | 0.0480ms (1.54×) | 0.0182ms (4.06×) | 0.0302ms (2.44×) |
| [1,512,4096] | 0.0925ms | 0.0638ms (1.45×) | 0.0229ms (4.03×) | 0.0303ms (3.05×) |
| [4,512,4096] | 0.1482ms | 0.0937ms (1.58×) | 0.0804ms (1.84×) | 0.0333ms (4.45×) |
| [1,2048,4096] | 0.1470ms | 0.0926ms (1.59×) | 0.0804ms (1.83×) | 0.0335ms (4.39×) |
| [1,128,8192] | 0.0788ms | 0.0541ms (1.46×) | 0.0181ms (4.35×) | 0.0309ms (2.55×) |

## 关键发现

- **小/中等规模（行数 ≤ 512）**：Ascend C kernel 相对 `torch_npu` 约 **1.5×**，launch 开销更小，直接在 AI Core 执行，无 aclnn 框架额外开销。
- **大规模（行数 ≥ 2048）**：`torch_npu` 更快，tiling 更成熟，多核并行与 UB 管理更好；当前 Ascend C 实现逐行串行，行数很大时并行度不足。

## 三种实现的层次

1. **C++ 扩展（最慢）**：PyTorch 算子组合，多次 launch，开销大。
2. **Ascend C kernel（中间）**：在 AI Core 上单 kernel 完成计算，小规模往往最优。
3. **`torch_npu` 内置（大规模最优）**：华为深度优化，tiling/多核调度更成熟。

## 算子源码对应关系（rms_norm）

| 实现 | 文件 | 入口/函数 |
|------|------|-----------|
| C++ naive | `rms_norm_custom.cpp` | `rms_norm_naive` |
| C++ fused | `rms_norm_custom.cpp` | `rms_norm_fused` |
| Ascend C | `rms_norm_kernel.cpp` | `rms_norm_ascendc_kernel` |
| Ascend C 绑定 | `rms_norm_ascendc_binding.cpp` | PyTorch 侧调用 |


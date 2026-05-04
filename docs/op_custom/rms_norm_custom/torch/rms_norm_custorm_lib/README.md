# RMSNorm 自定义算子

RMSNorm (Root Mean Square Normalization) 自定义算子实现，适用于 nano-vllm-ascend 项目在昇腾 NPU 上的高性能推理。

## 核心特性

✅ **NPU 优化** - 使用昇腾 NPU 硬件加速，支持 FP16, FP32, BF16 数据类型  
✅ **PyTorch 集成** - 遵循 torch.utils.cpp_extension.load() 模式，与 torch.compile 完全兼容  
✅ **高性能** - 融合算子减少内存访问，相比 torch_npu builtin 性能提升约 15-30%  
✅ **自动降级** - 如果 C++ 扩展加载失败，自动使用 torch_npu.npu_rms_norm

## 实现架构

本项目提供两种实现方式：

### 方式一：自定义 C++ 扩展（推荐）

使用 `torch.utils.cpp_extension.load()` 模式单文件实现：
- **实现文件**：`csrc/rms_norm_custom.cpp`
- **Python 包装**：`nanovllm/layers/rms_norm_custom.py`
- **特点**：单文件、易维护、JIT 编译、自动降级

### 方式二：传统 CANN 算子（高级定制）

完整 CANN 算子开发pipeline：
- **实现目录**：`csrc/rms_norm_bias/`
- **特点**：完整 CANN pipeline、深度优化、需 CMake 编译

## 项目结构

```
rms_norm_custom/
├── torch/                          # PyTorch风格实现
│   └── rms_norm_custorm_lib/       # 当前目录
│       └── README.md
└── ascend_c/                       # CANN算子实现
    └── rms_norm/                   # 完整CANN算子
```

## 核心文件

### 代码实现
- **`csrc/rms_norm_custom.cpp`** - C++ 实现，使用 pybind11 绑定
- **`nanovllm/layers/rms_norm_custom.py`** - Python 接口封装
- **`nanovllm/layers/layernorm.py`** - 集成的 RMSNorm 层实现

### 测试文件
- **`test/test_rms_norm_custom_lib.py`** - 完整测试套件
- **`example/example_rms_norm_lib_integration.py`** - 集成示例和基准测试

## 快速开始

### 1. 基本使用

```python
import torch
import torch_npu
from nanovllm.layers.rms_norm_custom import rms_norm, RMSNormCustom

device = torch.device("npu:0" if torch.npu.is_available() else "cpu")

# 方法1: 直接函数调用
x = torch.randn(32, 4096, device=device)
weight = torch.ones(4096, device=device)
output = rms_norm(x, weight, epsilon=1e-6)

# 方法2: 使用层类
norm = RMSNormCustom(hidden_size=4096, eps=1e-6).to(device)
output = norm(x)
```

### 2. 运行测试

```bash
python test/test_rms_norm_custom_lib.py
```

### 3. 集成示例

```bash
python example/example_rms_norm_lib_integration.py
```

## API 参考

### rms_norm(input, weight, epsilon=1e-6)

RMS Normalization 函数

**参数:**
- `input` (torch.Tensor): 输入张量，shape 为 (..., hidden_size)
- `weight` (torch.Tensor): 权重张量，shape 为 (hidden_size,)
- `epsilon` (float): 数值稳定性参数，默认 1e-6

**返回:**
- `torch.Tensor`: 标准化后的张量，shape 与输入相同

**公式:**
```
output = input / sqrt(mean(input²) + epsilon) * weight
```

### rms_norm_with_rstd(input, weight, epsilon=1e-6)

RMS Normalization（包含逆标准差输出）

**参数:**
- `input` (torch.Tensor): 输入张量
- `weight` (torch.Tensor): 权重张量
- `epsilon` (float): 数值稳定性参数

**返回:**
- `tuple[torch.Tensor, torch.Tensor]`: (标准化张量, 逆标准差)

### class RMSNormCustom

RMSNorm 层类

**构造函数:**
```python
RMSNormCustom(hidden_size: int, eps: float = 1e-6)
```

**方法:**
- `forward(x)` - 应用 RMS 归一化
- `forward_with_rstd(x)` - 返回归一化结果和逆标准差

## 关键修复

### 问题1: C++ 扩展加载失败 ✅ 已修复

**根本原因**: `torch.utils.cpp_extension.load()` 使用 `is_python_module=False` 时返回 `.so` 文件路径而非模块对象。

**解决方案**: 使用 `importlib.util.spec_from_file_location()` 正确加载：

```python
so_path = load(name="rms_norm_custom", sources=[source_file], ...)
spec = importlib.util.spec_from_file_location("rms_norm_custom", so_path)
rms_norm_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rms_norm_lib)
```

### 问题2: Dynamo 编译器不兼容 ✅ 已修复

**根本原因**: PyTorch Dynamo 编译器无法跟踪 pybind11 绑定的 C++ 扩展函数。

**解决方案**: 使用 `torch._dynamo.allow_in_graph()` 或 `torch.compiler.allow_in_graph()` 装饰器：

```python
try:
    if hasattr(torch, '_dynamo'):
        _rms_forward_torch_compatible = torch._dynamo.allow_in_graph(rms_norm_lib.rms_forward)
    elif hasattr(torch, 'compiler'):
        _rms_forward_torch_compatible = torch.compiler.allow_in_graph(rms_norm_lib.rms_forward)
    else:
        _rms_forward_torch_compatible = rms_norm_lib.rms_forward
except Exception:
    _rms_forward_torch_compatible = rms_norm_lib.rms_forward
```

## 性能对比

| 实现 | 平均时间 | 吞吐量 | 状态 |
|------|---------|--------|------|
| Python 纯实现 | 0.055 ms | 2.37e+09 elements/sec | Baseline |
| torch_npu builtin | 0.065 ms | 2.02e+09 elements/sec | Reference |
| **自定义 C++ 算子** | **0.047 ms** | **2.80e+09 elements/sec** | **最优** ✅ |

## 与现有代码集成

在 `nanovllm/layers/layernorm.py` 中已有集成：

```python
from nanovllm.layers.rms_norm_custom import rms_norm as _rms_norm_impl

class RMSNormOptimized(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm_impl(x, self.weight, self.eps)
```

### 与 torch.compile 配合

```python
import torch
from nanovllm.layers import RMSNorm

norm = RMSNorm(hidden_size=4096, eps=1e-6)
norm.weight.data = norm.weight.data.to('npu')

# 编译模型（Dynamo 会自动处理自定义算子）
compiled_norm = torch.compile(norm)
output = compiled_norm(x)
```

## 验证步骤

### 1. 单算子测试

```bash
python test/test_rms_norm_custom_lib.py
```

预期输出:
```
✓ C++ custom RMSNorm operator successfully loaded
✓ RMS Norm correctness test passed
✓ torch.compile works
```

### 2. 集成对比测试

```bash
python example/example_rms_norm_lib_integration.py
```

预期输出:
```
Correctness Verification:
Python   vs Native  : max diff = 4.7684e-07 ✓ PASS
Python   vs Hybrid  : max diff = 0.0000e+00 ✓ PASS
Native   vs Hybrid  : max diff = 4.7684e-07 ✓ PASS
```

### 3. 模型推理测试

```bash
python example/example.py
```

预期:
- 不出现 Dynamo 错误
- 模型正常推理
- 性能与预期相当或更好

## 兼容性

### PyTorch 版本
- ✅ PyTorch 2.9.0+ (已测试)
- ✅ PyTorch 2.0+ (理论上支持)

### torch_npu 版本
- ✅ torch_npu 2.9.0 (已测试)

### 编译器
- GCC 7.0+
- Ninja 1.13.0+

## 注意事项

1. **首次编译**: 首次运行时会编译 C++ 扩展，需要几秒时间
2. **编译缓存**: 编译结果缓存在 `build/rms_norm_custom/` 目录
3. **降级机制**: 如果 C++ 扩展加载失败，自动使用 `torch_npu.npu_rms_norm`
4. **LSP 警告**: 编辑器中的 LSP 警告是静态分析误报，不影响运行

## 常见问题

### Q: 为什么启动时需要编译？

A: 使用 `torch.utils.cpp_extension.load()` 首次导入时会自动编译 C++ 扩展。编译结果会被缓存，后续启动会很快。

### Q: 如何确认自定义算子被使用？

A: 查看日志输出：
```
✓ C++ custom RMSNorm operator successfully loaded
Available methods: ['rms_forward', 'rms_forward_with_rstd']
```
另外，通过性能对比可以确认：自定义算子应比 torch_npu builtin 更快。

### Q: torch.compile 会报错吗？

A: 不会。已通过 `torch._dynamo.allow_in_graph()` 装饰器解决了兼容性问题。

### Q: 如何排查问题？

A:
1. 检查日志中的加载信息
2. 运行测试套件：`python test/test_rms_norm_custom_lib.py`
3. 验证性能和正确性：`python example/example_rms_norm_lib_integration.py`

## 参考资料

- [PyTorch Custom C++ Operators](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [torch.compile 兼容性](https://pytorch.org/docs/stable/compile.html)
- [torch_npu 文档](https://torch-npu.readthedocs.io/)
- [CANN Operator Development Guide](https://www.hiascend.com/document)

## 许可证

Apache-2.0 (见项目根目录 LICENSE 文件)
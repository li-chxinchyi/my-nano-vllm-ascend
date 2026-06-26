# LongBench Evaluation Script

LongBench评测脚本用于测试模型精度和推理性能。

## 目录结构

```
eval/
├── eval_longbench.py          # 主测试脚本
├── metrics/
│   ├── accuracy.py            # 精度指标计算
│   └── performance.py         # 性能指标计算
├── utils/
│   ├── data_loader.py         # LongBench数据加载器
│   └── report_generator.py    # 报告生成器
└── requirements.txt           # 依赖列表
```

## 依赖安装

脚本会自动安装缺失的依赖，也可以手动安装：

```bash
pip install -r eval/requirements.txt
```

## 支持的LongBench任务

| 任务名称 | 类型 | 语言 | 评测指标 |
|---------|------|------|---------|
| longbook_qa_eng | 问答 | 英文 | F1 Score |
| longbook_qa_chn | 问答 | 中文 | F1 Score |
| longbook_summ_eng | 摘要 | 英文 | ROUGE-L |
| longbook_summ_chn | 摘要 | 中文 | ROUGE-L |
| longbook_choice_eng | 选择题 | 英文 | Accuracy |
| longbook_choice_chn | 选择题 | 中文 | Accuracy |

## 使用方法

### 基本用法

```bash
python eval/eval_longbench.py \
    --model /path/to/model \
    --tasks longbook_qa_eng,longbook_summ_eng \
    --max_samples 100 \
    --output_dir ./eval_results
```

### 完整参数

```bash
python eval/eval_longbench.py \
    --model /path/to/model \
    --tasks longbook_qa_eng,longbook_summ_eng,longbook_choice_eng \
    --max_samples 100 \
    --output_dir ./eval_results \
    --tensor_parallel_size 1 \
    --max_tokens 512 \
    --temperature 0.6 \
    --max_model_len 4096 \
    --enforce_eager \
    --hccl_port 3456 \
    --max_num_seqs 4
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|-------|
| --model | 模型路径（必需） | - |
| --tasks | 任务列表（逗号分隔） | longbook_qa_eng,longbook_summ_eng |
| --max_samples | 每个任务的样本数 | 100 |
| --output_dir | 结果输出目录 | ./eval_results |
| --tensor_parallel_size | 张量并行数 | 1 |
| --max_tokens | 最大生成token数 | 512 |
| --temperature | 采样温度 | 0.6 |
| --max_model_len | 模型最大长度 | 4096 |
| --enforce_eager | 使用eager模式 | False |
| --hccl_port | HCCL端口 | 3456 |
| --max_num_seqs | 每批最大序列数 | 4 |
| --skip_dependencies | 跳过依赖安装 | False |

## 输出结果

评测完成后会在 `output_dir` 下生成时间戳目录，包含：

```
eval_results/
└── 20250126_143025/
    ├── config.json              # 测试配置
    ├── results.json             # 详细结果
    ├── summary.json             # 汇总统计
    ├── performance_chart.png    # 性能图表
    ├── accuracy_chart.png       # 精度图表
    ├── latency_distribution.png # 延迟分布图
    └── report.md                # Markdown报告
```

### 性能指标

- **TTFT**: Time To First Token（首个token生成时间）
- **TPOT**: Time Per Output Token（每token生成时间）
- **Throughput**: tokens/s 吞吐量
- **Latency**: 端到端延迟
- **Peak Memory**: 峰值显存占用

### 精度指标

- **F1 Score**: 用于问答任务
- **ROUGE-L**: 用于摘要任务
- **Accuracy**: 用于选择题任务

## 示例输出

```
Evaluation completed!
Results saved to: ./eval_results/20250126_143025
Overall summary:
  - Total samples: 300
  - Total time: 1234.56s
  - Avg TTFT: 123.45ms
  - Avg Throughput: 64.2 tok/s

Task longbook_qa_eng:
  - Avg F1 Score: 0.456
  - Avg TTFT: 115.23ms
  - Avg Throughput: 68.5 tok/s

Task longbook_summ_eng:
  - Avg ROUGE-L: 0.234
  - Avg TTFT: 145.67ms
  - Avg Throughput: 55.1 tok/s
```

## 注意事项

1. **数据集下载**: 首次运行会自动从HuggingFace下载LongBench数据集（约2GB）
2. **环境要求**: 需要NPU环境（或GPU环境）支持
3. **显存需求**: 根据模型大小和max_tokens调整参数
4. **数据集缓存**: 默认缓存到 `~/.cache/longbench`

## 与example目录代码的对比

本评测脚本参考了example目录的代码风格：
- 使用相同的LLM引擎初始化方式（参考example/example.py）
- 使用相同的SamplingParams参数（参考bench/bench.py）
- 使用相同的时间测量方法（参考bench/serving_bench.py）
- 使用相同的进度显示方式（使用tqdm）
# nano-vllm-ascend V1 服务部署文档

## 目录
- [快速开始](#快速开始)
- [服务部署命令](#服务部署命令)
- [部署参数详情](#部署参数详情)
- [API接口说明](#api接口说明)
- [Chat Completion 接口](#chat-completion-接口)
- [Completion 接口](#completion-接口)
- [最佳实践](#最佳实践)

## 快速开始

### 最小化部署

```bash
# 启动服务（使用默认参数）
python nanovllm/v1/run_api_server.py --model /path/to/model
```

### 后台运行

```bash
# 后台运行并输出到日志
nohup python nanovllm/v1/run_api_server.py \
  --model /path/to/model \
  --port 8000 \
  > online_infer.log 2>&1 &

# 查看日志
tail -f online_infer.log
```

### 推荐生产环境部署

```bash
python nanovllm/v1/run_api_server.py \
  --model /path/to/model \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 256 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --log-level info
```

## 服务部署命令

### 基本语法

```bash
python nanovllm/v1/run_api_server.py [参数...]
```

### 示例命令

#### 1. 单卡部署（推荐用于中小模型）

```bash
python nanovllm/v1/run_api_server.py \
  --model /path/to/Qwen3-0.6B \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9
```

#### 2. 多卡部署（用于大模型）

```bash
python nanovllm/v1/run_api_server.py \
  --model /path/to/large-model \
  --port 8000 \
  --tensor-parallel-size 4 \
  --hccl-port 28000 \
  --gpu-memory-utilization 0.85
```

#### 3. 调试模式部署

```bash
python nanovllm/v1/run_api_server.py \
  --model /path/to/model \
  --log-level debug \
  --port 8000
```

#### 4. 生产环境部署（后台运行）

```bash
nohup python nanovllm/v1/run_api_server.py \
  --model /path/to/model \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 256 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --log-level info \
  > /var/log/nano-vllm/online_infer.log 2>&1 &
```

#### 5. 自定义模型名称

```bash
python nanovllm/v1/run_api_server.py \
  --model /mnt/models/Qwen3-0.6B-Instruct \
  --served-model-name qwen-0.6b-instruct \
  --port 8000
```

## 部署参数详情

### 必需参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `--model` | str | 模型路径（必需） | `/mnt/models/Qwen3-0.6B/` |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 | 推荐值 |
|------|------|--------|------|--------|
| `--host` | str | `0.0.0.0` | 服务绑定的主机地址 | `0.0.0.0`（所有接口）<br>`127.0.0.1`（本地） |
| `--port` | int | `8000` | 服务监听端口 | `8000`、`8080`、`9000` |
| `--served-model-name` | str | `None` | API中暴露的模型名称 | 自定义友好名称 |
| `--max-num-batched-tokens` | int | `16384` | 批处理最大token数 | `8192` (小模型)<br>`16384` (中模型)<br>`32768` (大模型) |
| `--max-num-seqs` | int | `256` | 最大并发序列数 | `128` (小卡)<br>`256` (中卡)<br>`512` (大卡) |
| `--max-model-len` | int | `4096` | 模型最大序列长度 | `2048`、`4096`、`8192`、`16384` |
| `--gpu-memory-utilization` | float | `0.9` | GPU内存使用率 (0-1) | `0.85` (保守)<br>`0.9` (标准)<br>`0.95` (激进) |
| `--tensor-parallel-size` | int | `1` | 张量并行大小（卡数） | `1` (单卡)<br>`2` (2卡)<br>`4` (4卡)<br>`8` (8卡) |
| `--hccl-port` | int | `28000` | HCCL通信端口 | `28000`、`29500` |
| `--enforce-eager` | flag | `True` | 强制eager模式执行 | 保持默认 |
| `--log-level` | str | `info` | 日志级别 | `debug`、`info`、`warning`、`error` |

### 参数详解

#### `--host`
- **用途**：指定服务监听的IP地址
- **场景**：
  - `0.0.0.0`：允许所有客户端访问，生产环境推荐
  - `127.0.0.1`：仅本地访问，开发测试环境
- **注意事项**：确保防火墙规则允许外部访问

#### `--port`
- **用途**：指定服务监听的TCP端口
- **注意事项**：
  - 确保端口未被占用
  - 端口号需小于65535
  - 常用端口：8000、8080、9000

#### `--max-num-batched-tokens`
- **用途**：控制单次推理的批处理token数量
- **影响**：
  - 值越大，吞吐量越高
  - 但会增加内存占用和延迟
- **调优建议**：
  - 根据GPU显存调整
  - 910C (32GB): 16384-32768
  - 910B (64GB): 32768-65536

#### `--max-num-seqs`
- **用途**：最大并发请求数
- **影响**：
  - 值越大，并发能力越强
  - 每个序列会占用固定KV缓存空间
- **调优建议**：
  - 根据max_model_len和显存计算
  - 公式：`max_num_seqs * max_model_len * kv_cache_size < 可用显存`

#### `--max-model-len`
- **用途**：请求的最大序列长度（输入+输出）
- **注意事项**：
  - 不能超过模型支持的最大长度
  - Qwen3-0.6B: 支持8192
  - Qwen3-32B: 支持32768
- **建议**：根据业务需求设置，一般4096足够

#### `--gpu-memory-utilization`
- **用途**：控制GPU显存使用率
- **调优建议**：
  - `0.85`：保守，留足余量，避免OOM
  - `0.9`：标准，推荐用于生产环境
  - `0.95`：激进，充分利用显存，可能有OOM风险
- **注意事项**：超出会导致CUDA OOM错误

#### `--tensor-parallel-size`
- **用途**：张量并行度，即使用的NPU卡数
- **要求**：
  - 值必须等于实际使用的NPU卡数
  - 需要配置HCCL通信
- **场景**：
  - `1`：单卡部署，适合中小模型
  - `2-8`：多卡部署，适合大模型或高吞吐场景

#### `--hccl-port`
- **用途**：HCCL（华为集合通信）端口
- **注意事项**：
  - 多卡部署时必须设置
  - 确保端口未被占用
  - 所有卡使用相同端口

#### `--enforce-eager`
- **用途**：强制使用eager模式而非图模式
- **当前状态**：默认启用
- **影响**：
  - Eager模式：灵活，但性能较低
  - 图模式：高性能，但需要编译

## API接口说明

### 基础URL

```
http://<host>:<port>/v1
```

### 可用端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 获取可用模型列表 |
| `/v1/completions` | POST | 文本补全接口 |
| `/v1/chat/completions` | POST | 对话补全接口 |

### 响应格式

所有API响应使用JSON格式，遵循OpenAI API规范。

## Chat Completion 接口

### 端点

```
POST /v1/chat/completions
```

### 请求参数

#### 基本参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | str | 是 | - | 模型名称，使用启动时设置的路径或自定义名称 |
| `messages` | List[Dict] | 是 | - | 对话消息列表，每个消息包含role和content |

#### 采样参数

| 参数 | 类型 | 必需 | 默认值 | 说明 | 范围 |
|------|------|------|--------|------|------|
| `temperature` | float | 否 | 0.7 | 采样温度，控制随机性 | 0.0-2.0<br>0.0=确定性<br>1.0=标准随机性 |
| `top_p` | float | 否 | 1.0 | 核采样概率 | 0.0-1.0 |
| `top_k` | int | 否 | -1 | Top-K采样，-1表示关闭 | -1或正整数 |
| `max_tokens` | int | 否 | 64 | 最大生成token数 | 正整数 |
| `n` | int | 否 | 1 | 每个prompt生成n个回复 | 正整数 |

#### 停止参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `stop` | str 或 List[str] | 否 | [] | 停止序列，遇到这些字符串停止生成 | 例如：`\n`、`["\n", "###"]` |
| `stop_token_ids` | List[int] | 否 | [] | 停止token ID列表 | `[151643, 151645]` |
| `ignore_eos` | bool | 否 | False | 是否忽略EOS token | True=生成到max_tokens<br>False=遇到EOS停止 |

#### 高级参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `stream` | bool | 否 | False | 是否流式输出 | True=流式<br>False=一次性返回 |
| `logit_bias` | Dict[str, float] | 否 | None | 偏置特定token | `{"50256": -100}` |
| `presence_penalty` | float | 否 | 0.0 | 存在惩罚，降低重复话题 | -2.0到2.0 |
| `frequency_penalty` | float | 否 | 0.0 | 频率惩罚，降低重复词汇 | -2.0到2.0 |
| `repetition_penalty` | float | 否 | 1.0 | 重复惩罚 | >1.0降低重复 |
| `min_p` | float | 否 | 0.0 | 最小概率阈值 | 0.0-1.0 |
| `length_penalty` | float | 否 | 1.0 | 长度惩罚 | <1.0偏短<br>>1.0偏长 |

#### 高级功能参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `echo` | bool | 否 | False | 是否回显输入 | True=输入+输出 |
| `add_generation_prompt` | bool | 否 | True | 是否添加生成提示符 | True=标准格式<br>False=原始格式 |
| `skip_special_tokens` | bool | 否 | True | 是否跳过特殊token | True=标准<br>False=保留 |
| `spaces_between_special_tokens` | bool | 否 | True | 特殊token是否加空格 | True=标准<br>False=紧凑 |

#### 算法参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `use_beam_search` | bool | 否 | False | 是否使用束搜索 | True=高质量但慢<br>False=标准采样 |
| `best_of` | int | 否 | None | 束搜索的候选数 | 正整数 |

#### 其他参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user` | str | 否 | None | 用户标识符 | 用于监控 |
| `include_stop_str_in_output` | bool | 否 | False | 返回值是否包含停止字符串 | True=包含<br>False=不包含 |

### 消息格式

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
    {"role": "user", "content": "什么是人工智能？"}
  ]
}
```

可用的角色：
- `system`: 系统提示词，设定助手行为
- `user`: 用户消息
- `assistant`: 助手回复
- `developer`: 开发者提示（某些模型支持）

### 请求示例

#### 示例1：基础对话请求

```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [
      {"role": "user", "content": "什么是AI？"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### 示例2：流式输出

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "messages": [
      {"role": "user", "content": "给我讲个故事"}
    ],
    "max_tokens": 200,
    "stream": true
  }'
```

#### 示例3：多轮对话

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "messages": [
      {"role": "system", "content": "你是一个专业的程序员"},
      {"role": "user", "content": "Python怎么写Hello World？"},
      {"role": "assistant", "content": "很简单：\n\n```python\nprint(\"Hello World\")\n```"},
      {"role": "user", "content": "那Java呢？"}
    ],
    "max_tokens": 100,
    "temperature": 0.5
  }'
```

#### 示例4：使用停止序列

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "messages": [
      {"role": "user", "content": "列出3种水果"}
    ],
    "max_tokens": 100,
    "stop": ["\n", ";"]
  }'
```

#### 示例5：低温度，确定性输出

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "messages": [
      {"role": "user", "content": "计算 1+1"}
    ],
    "max_tokens": 50,
    "temperature": 0.1
  }'
```

### 响应格式

#### 非流式响应

```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "/path/to/model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "AI是人工智能的缩写..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

#### 流式响应

流式响应以Server-Sent Events (SSE)格式返回：

```
data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1234567890, "model": "/path/to/model", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1234567890, "model": "/path/to/model", "choices": [{"index": 0, "delta": {"content": "AI"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1234567890, "model": "/path/to/model", "choices": [{"index": 0, "delta": {"content": "是"}, "finish_reason": null}]}

...

data: [DONE]
```

## Completion 接口

### 端点

```
POST /v1/completions
```

### 请求参数

#### 基本参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | str | 是 | - | 模型名称 |
| `prompt` | str 或 List[str] 或 List[int] 或 List[List[int]] | 是 | - | 输入提示词，可以是字符串、字符串列表或token id列表 |

#### 采样参数

| 参数 | 类型 | 必需 | 默认值 | 说明 | 范围 |
|------|------|------|--------|------|------|
| `temperature` | float | 否 | 1.0 | 采样温度 | 0.0-2.0 |
| `top_p` | float | 否 | 1.0 | 核采样概率 | 0.0-1.0 |
| `top_k` | int | 否 | -1 | Top-K采样 | -1或正整数 |
| `max_tokens` | int | 否 | 16 | 最大生成token数 | 正整数 |
| `n` | int | 否 | 1 | 每个prompt生成n个回复 | 正整数 |

#### 停止参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `stop` | str 或 List[str] | 否 | [] | 停止序列 | 对应Chat接口参数 |
| `stop_token_ids` | List[int] | 否 | [] | 停止token ID列表 | 对应Chat接口参数 |
| `ignore_eos` | bool | 否 | False | 是否忽略EOS token | 对应Chat接口参数 |

#### 高级参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `stream` | bool | 否 | False | 是否流式输出 | True=流式<br>False=一次性返回 |
| `logprobs` | int | 否 | None | 返回的logprobs数量 | 0-5，None=不返回 |
| `echo` | bool | 否 | False | 是否回显提示词 | True=回显<br>False=只输出生成 |
| `suffix` | str | 否 | None | 插入到生成内容后的后缀 | 用于补全固定格式 |
| `logit_bias` | Dict[str, float] | 否 | None | 偏置特定token | 对应Chat接口参数 |
| `presence_penalty` | float | 否 | 0.0 | 存在惩罚 | -2.0到2.0 |
| `frequency_penalty` | float | 否 | 0.0 | 频率惩罚 | -2.0到2.0 |
| `repetition_penalty` | float | 否 | 1.0 | 重复惩罚 | >1.0降低重复 |
| `min_p` | float | 否 | 0.0 | 最小概率阈值 | 0.0-1.0 |
| `length_penalty` | float | 否 | 1.0 | 长度惩罚 | <1.0偏短<br>>1.0偏长 |

#### 高级功能参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `skip_special_tokens` | bool | 否 | True | 是否跳过特殊token | 对应Chat接口参数 |
| `spaces_between_special_tokens` | bool | 否 | True | 特殊token是否加空格 | 对应Chat接口参数 |
| `include_stop_str_in_output` | bool | 否 | False | 返回值是否包含停止字符串 | 对应Chat接口参数 |

#### 算法参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `use_beam_search` | bool | 否 | False | 是否使用束搜索 | 对应Chat接口参数 |
| `best_of` | int | 否 | None | 束搜索的候选数 | 正整数 |

#### 其他参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user` | str | 否 | None | 用户标识符 | 用于监控 |

### 请求示例

#### 示例1：基础文本补全

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "prompt": "人工智能是",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

#### 示例2：代码补全

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "prompt": "def factorial(n):\n    ",
    "max_tokens": 100,
    "temperature": 0.2,
    "stop": ["\n\n", "def"],
    "echo": false
  }'
```

#### 示例3：批量处理（多个提示词）

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "prompt": [
      "人工智能的定义是",
      "机器学习包括",
      "深度学习的优势是"
    ],
    "max_tokens": 30,
    "temperature": 0.7
  }'
```

#### 示例4：使用stop参数控制输出

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "prompt": "列举三种水果：",
    "max_tokens": 100,
    "stop": ["\n\n", "***"],
    "temperature": 0.7
  }'
```

#### 示例5：回显输入（echo=true）

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "prompt": "Once upon a time",
    "max_tokens": 20,
    "echo": true
  }'
```

#### 示例6：使用后缀（suffix）

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "prompt": "Question: What is 2+2?",
    "suffix": "\nAnswer: 4",
    "max_tokens": 50
  }'
```

#### 示例7：流式输出

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "prompt": "写一首关于春天的诗",
    "max_tokens": 200,
    "stream": true
  }'
```

#### 示例8：使用Token ID输入

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "prompt": [104997, 104994, 99344],
    "max_tokens": 50
  }'
```

#### 示例9：返回logprobs

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "logprobs": 5
  }'
```

#### 示例10：惩罚和重复控制

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "prompt": "Write a short story:",
    "max_tokens": 100,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.5,
    "repetition_penalty": 1.2,
    "temperature": 0.8
  }'
```

### 响应格式

#### 非流式响应

```json
{
  "id": "cmpl-1234567890",
  "object": "text_completion",
  "created": 1234567890,
  "model": "/path/to/model",
  "choices": [
    {
      "index": 0,
      "text": "人工智能是计算机科学的一个分支...",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 42,
    "total_tokens": 50
  }
}
```

**含logprobs的响应**：

```json
{
  "id": "cmpl-1234567890",
  "object": "text_completion",
  "created": 1234567890,
  "model": "/path/to/model",
  "choices": [
    {
      "index": 0,
      "text": "Paris",
      "logprobs": {
        "text_offset": [0],
        "token_logprobs": [-0.1234],
        "tokens": ["Paris"],
        "top_logprobs": [
          {
            "Paris": -0.1234,
            "paris": -2.3456,
            "Paris,": -3.4567
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 1,
    "total_tokens": 8
  }
}
```

#### 流式响应

```
data: {"id": "cmpl-123", "object": "text_completion", "created": 1234567890, "model": "/path/to/model", "choices": [{"index": 0, "text": "人", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-123", "object": "text_completion", "created": 1234567890, "model": "/path/to/model", "choices": [{"index": 0, "text": "工", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-123", "object": "text_completion", "created": 1234567890, "model": "/path/to/model", "choices": [{"index": 0, "text": "智", "logprobs": null, "finish_reason": null}]}

...

data: [DONE]
```

## 最佳实践

### 部署调优建议

#### 1. 根据模型选择参数

**Qwen3-0.6B (6亿参数)**
```bash
python nanovllm/v1/run_api_server.py \
  --model /path/to/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 256 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

**Qwen3-32B (320亿参数)**
```bash
python nanovllm/v1/run_api_server.py \
  --model /path/to/Qwen3-32B \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 128 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

#### 2. 根据场景选择参数

**低延迟场景（实时对话）**
```bash
python nanovllm/v1/run_api_server.py \
  --model /path/to/model \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

**高吞吐场景（批处理）**
```bash
python nanovllm/v1/run_api_server.py \
  --model /path/to/model \
  --max-num-seqs 512 \
  --max-num-batched-tokens 32768 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

### API使用建议

#### 1. 温度参数选择

| 场景 | 推荐温度 | 说明 |
|------|---------|------|
| 代码生成 | 0.1-0.3 | 低温度，确定性高 |
| 事实问答 | 0.3-0.5 | 中低温度，准确优先 |
| 创意写作 | 0.7-1.0 | 中高温度，创意优先 |
| 角色扮演 | 1.0-1.5 | 高温度，多样性高 |

#### 2. Max Tokens选择

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 简短回答 | 32-64 | 一句话回答 |
| 详细解释 | 128-256 | 完整段落 |
| 长文本生成 | 512-1024 | 文章、故事 |
| 代码生成 | 128-512 | 取决于代码长度 |

#### 3. 流式vs非流式

| 场景 | 推荐模式 | 说明 |
|------|---------|------|
| 实时对话 | 流式 | 用户体验好 |
| 批处理 | 非流式 | 吞吐量高 |
| 长文本生成 | 流式 | 避免超时 |
| API集成 | 非流式 | 实现简单 |

#### 4. 性能优化技巧

1. **设置合理的stop序列**：避免不必要的token生成
2. **使用适当的max_tokens**：不要设置过大的值
3. **流式输出减少延迟感知**：即使总时间相同，用户体验更好
4. **batch请求**：对于批量任务，一次发送多个prompt

#### 5. 错误处理

常见错误及解决方案：

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| CUDA OOM | GPU内存不足 | 降低gpu_memory_utilization<br>减少max_num_seqs或max_model_len |
| 超时 | 请求时间过长 | 检查是否模型过载<br>增加max_num_seqs |
| 503服务不可用 | 服务崩溃 | 查看infer.log日志<br>检查NPU状态 |
| 参数验证错误 | 请求参数不合法 | 检查参数类型和范围 |

### 监控和日志

#### 查看服务状态

```bash
# 检查服务是否运行
ps aux | grep run_api_server

# 查看实时日志
tail -f online_infer.log

# 查看NPU使用情况
npu-smi info

# 查看端口监听
netstat -tlnp | grep 8000
```

#### 健康检查

```bash
# 检查models端点
curl http://localhost:8000/v1/models
```

### 故障排查

#### 服务无法启动

1. 检查模型路径是否正确
2. 检查NPU是否可用：`npu-smi info`
3. 查看infer.log中的错误信息

#### 请求无响应

1. 检查服务是否正常运行
2. 检查网络连接
3. 查看请求参数是否正确
4. 检查infer.log是否有错误

#### 响应质量差

1. 使用Instruction-tuned模型（如Qwen3-0.6B-Instruct）
2. 调整temperature参数
3. 添加合适的system prompt
4. 检查chat template是否正确

## 附录

### A. 完整参数对照表

#### 服务启动参数

| 参数名称 | 短参数 | 类型 | 默认值 | 说明 |
|----------|--------|------|--------|------|
| --host | - | str | 0.0.0.0 | 监听地址 |
| --port | - | int | 8000 | 监听端口 |
| --model | - | str | (必需) | 模型路径 |
| --served-model-name | - | str | None | API中显示的模型名 |
| --max-num-batched-tokens | - | int | 16384 | 批处理token数 |
| --max-num-seqs | - | int | 256 | 最大并发数 |
| --max-model-len | - | int | 4096 | 最大序列长度 |
| --gpu-memory-utilization | - | float | 0.9 | 显存使用率 |
| --tensor-parallel-size | - | int | 1 | 张量并行大小 |
| --hccl-port | - | int | 28000 | HCCL端口 |
| --enforce-eager | - | flag | True | 强制eager模式 |
| --log-level | - | str | info | 日志级别 |

#### Chat Completion 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | str | 必需 | 模型名称 |
| messages | List[Dict] | 必需 | 对话消息列表 |
| temperature | float | 0.7 | 采样温度 (0.0-2.0) |
| top_p | float | 1.0 | 核采样 (0.0-1.0) |
| top_k | int | -1 | Top-K采样 |
| max_tokens | int | 64 | 最大生成token数 |
| n | int | 1 | 生成候选数 |
| stop | str/List[str] | [] | 停止序列 |
| stop_token_ids | List[int] | [] | 停止token IDs |
| ignore_eos | bool | False | 忽略EOS |
| stream | bool | False | 流式输出 |
| presence_penalty | float | 0.0 | 存在惩罚 (-2.0 到 2.0) |
| frequency_penalty | float | 0.0 | 频率惩罚 (-2.0 到 2.0) |
| repetition_penalty | float | 1.0 | 重复惩罚 |
| min_p | float | 0.0 | 最小概率 (0.0-1.0) |
| length_penalty | float | 1.0 | 长度惩罚 |
| echo | bool | False | 回显输入 |
| add_generation_prompt | bool | True | 添加生成提示 |
| skip_special_tokens | bool | True | 跳过特殊token |
| spaces_between_special_tokens | bool | True | 特殊token空格 |
| include_stop_str_in_output | bool | False | 包含停止字符串 |
| use_beam_search | bool | False | 束搜索 |
| best_of | int | None | 束搜索候选数 |
| logit_bias | Dict | None | Token偏置 |
| user | str | None | 用户标识 |

#### Completion 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | str | 必需 | 模型名称 |
| prompt | str/List/List[int] | 必需 | 输入提示词 |
| suffix | str | None | 后缀 |
| max_tokens | int | 16 | 最大生成token数 |
| temperature | float | 1.0 | 采样温度 (0.0-2.0) |
| top_p | float | 1.0 | 核采样 (0.0-1.0) |
| top_k | int | -1 | Top-K采样 |
| n | int | 1 | 生成候选数 |
| stream | bool | False | 流式输出 |
| logprobs | int | None | logprobs数量 |
| echo | bool | False | 回显输入 |
| stop | str/List[str] | [] | 停止序列 |
| stop_token_ids | List[int] | [] | 停止token IDs |
| ignore_eos | bool | False | 忽略EOS |
| presence_penalty | float | 0.0 | 存在惩罚 (-2.0 到 2.0) |
| frequency_penalty | float | 0.0 | 频率惩罚 (-2.0 到 2.0) |
| repetition_penalty | float | 1.0 | 重复惩罚 |
| min_p | float | 0.0 | 最小概率 (0.0-1.0) |
| length_penalty | float | 1.0 | 长度惩罚 |
| skip_special_tokens | bool | True | 跳过特殊token |
| spaces_between_special_tokens | bool | True | 特殊token空格 |
| include_stop_str_in_output | bool | False | 包含停止字符串 |
| use_beam_search | bool | False | 束搜索 |
| best_of | int | None | 束搜索候选数 |
| logit_bias | Dict | None | Token偏置 |
| user | str | None | 用户标识 |

### B. 常用模型配置

#### Qwen3-0.6B

```json
{
  "推荐配置": {
    "tensor_parallel_size": 1,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 256,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9
  },
  "支持的最大长度": 8192
}
```

#### Qwen3-32B

```json
{
  "推荐配置": {
    "tensor_parallel_size": 4,
    "max_num_batched_tokens": 32768,
    "max_num_seqs": 128,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85
  },
  "支持的最大长度": 32768
}
```

### C. 环境变量

| 环境变量 | 说明 |
|----------|------|
| HCCL_PORT | HCCL通信端口（可通过--hccl-port设置） |

### D. 参考资源

- OpenAI API文档：https://platform.openai.com/docs/api-reference
- vLLM文档：https://docs.vllm.ai/

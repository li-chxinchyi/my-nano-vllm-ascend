# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project

"""Benchmark script for nano-vllm multimodal inference with local assets."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable

from PIL import Image
from nanovllm import LLM, SamplingParams

_BENCH_ROOT = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR = os.path.join(_BENCH_ROOT, "asset")
_DEFAULT_NAMES = (
    "L0.png", "X0.png", "X1.png", "X2.png", "X3.png", "X4.png", "X5.png", "X6.png", "X7.png",
)
DEFAULT_IMAGE_PATHS: tuple[str, ...] = tuple(
    os.path.join(_ASSET_DIR, name) for name in _DEFAULT_NAMES
)

DEFAULT_PROMPT = (
    "Please describe the scene in the image and highlight the primary objects."
)


def load_local_image(path: str) -> Image.Image:
    """从本地磁盘加载并预处理图片"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到图片文件: {path}")
    # 使用 Image.open 直接读取本地文件，不需要 io.BytesIO
    return Image.open(path).convert("RGB")


@dataclass
class BenchmarkResult:
    num_requests: int
    total_prompt_tokens: int
    total_generated_tokens: int
    latency: float

    @property
    def tok_per_sec(self) -> float:
        """计算每秒生成的 Token 数 (Throughput)"""
        if self.latency <= 0:
            return 0.0
        return self.total_generated_tokens / self.latency


def build_requests(
        processor,
        image_paths: Iterable[str],
        prompt: str,
) -> list[dict]:
    """构建多模态推理请求列表"""
    requests = []
    for path in image_paths:
        image = load_local_image(path)

        # 应用对话模板
        chat_prompt = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        requests.append(
            {
                "text": chat_prompt,
                "images": [image],
                "meta": {"path": path},  # 记录路径以便追踪
            }
        )
    return requests


def run_benchmark(
        model_path: str,
        max_new_tokens: int,
        temperature: float,
        image_paths: Iterable[str],
) -> BenchmarkResult:
    """执行 Benchmark 测试"""

    # 初始化 Nano-vLLM 引擎
    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1, max_num_seqs=1)

    # 延迟加载 transformers 减少启动开销
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)

    # 准备数据
    requests = build_requests(processor, image_paths, DEFAULT_PROMPT)
    num_requests = len(requests)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    sampling_params_list = [sampling_params] * num_requests

    print("正在进行预热 (Warm-up)...")
    llm.generate_multimodal(
        [requests[0]],
        sampling_params_list[0],
        processor,
        use_tqdm=False,
    )

    print(f"开始推理测试 (共 {num_requests} 个请求)...")
    start = time.perf_counter()
    outputs = llm.generate_multimodal(
        requests,
        sampling_params_list,
        processor,
        use_tqdm=False,
    )
    latency = time.perf_counter() - start

    # 统计生成 Token 数量
    total_generated = sum(len(item["token_ids"]) for item in outputs)

    prompt_token_lengths = 0
    for req in requests:
        encoded = processor(
            text=[req["text"]],
            images=req["images"],
            return_tensors="pt",
        )
        prompt_token_lengths += encoded["input_ids"].shape[-1]

    return BenchmarkResult(
        num_requests=num_requests,
        total_prompt_tokens=prompt_token_lengths,
        total_generated_tokens=total_generated,
        latency=latency,
    )


def main() -> None:
    model_path = os.environ.get("NANO_VLLM_BENCH_MODEL", "/data/model/Qwen3-VL-2B-Instruct")

    result = run_benchmark(
        model_path=model_path,
        max_new_tokens=1024,
        temperature=0.7,
        image_paths=DEFAULT_IMAGE_PATHS,
    )

    print("\n" + "=" * 40)
    print("=== nano-vllm multimodal benchmark ===")
    print(f"Model Path        : {model_path}")
    print(f"Requests          : {result.num_requests}")
    print(f"Prompt tokens     : {result.total_prompt_tokens}")
    print(f"Generated tokens  : {result.total_generated_tokens}")
    print(f"Total Latency     : {result.latency:.2f}s")
    print(f"Throughput        : {result.tok_per_sec:.2f} tok/s")
    print("=" * 40)


if __name__ == "__main__":
    main()

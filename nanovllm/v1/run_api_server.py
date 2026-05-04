#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
v1_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root))  # 项目根目录
sys.path.insert(0, str(v1_root))  # v1 根目录

from nanovllm.v1.entrypoints.openai.api_server import app
from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="nano-vLLM-Ascend V1 OpenAI-Compatible API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument("--served-model-name", type=str, default=None,
                        help="Model name to expose in API")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384,
                        help="Max number of batched tokens")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="Max number of sequences")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model sequence length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization (0-1)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--hccl-port", type=int, default=28000,
                        help="HCCL port for distributed initialization")
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="Force eager mode execution")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Log level")

    args = parser.parse_args()

    import os
    os.environ['HCCL_PORT'] = str(args.hccl_port)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

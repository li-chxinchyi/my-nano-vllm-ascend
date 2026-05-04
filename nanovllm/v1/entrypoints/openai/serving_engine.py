import asyncio
from typing import Optional
from nanovllm.sampling_params import SamplingParams

# 导入 v1 模块的 AsyncLLMEngine
from nanovllm.v1.engine import AsyncLLMEngine


class OpenAIServing:
    """Base class for OpenAI API serving"""

    def __init__(
            self,
            engine: AsyncLLMEngine,
            served_model: list[str],
            response_role: str = "assistant"):
        self.engine = engine
        self.served_model = served_model
        self.response_role = response_role

    async def check_model(self, model) -> None:
        if model not in self.served_model:
            raise ValueError(f"Model {model} not found. Available models: {self.served_model}")

    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        await self.engine.abort(request_id)
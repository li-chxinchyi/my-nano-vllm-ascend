import asyncio
import time
from fastapi import Request
from typing import AsyncGenerator, Callable, List, Optional

from .protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    LogProbs,
    UsageInfo,
)
from .serving_engine import OpenAIServing

# 导入 v1 内部模块
from nanovllm.v1.core.outputs import RequestOutput


async def completion_stream_generator(
        request: CompletionRequest,
        raw_request: Request,
        on_abort,
        result_generator: AsyncGenerator[RequestOutput, None],
        create_logprobs_fn: Callable,
        request_id: str,
        created_time: int,
        model_name: str,
) -> AsyncGenerator[str, None]:
    previous_texts = [""]
    previous_num_tokens = [0]
    has_echoed = [False]

    async for res in result_generator:

        if raw_request is not None and await raw_request.is_disconnected():
            await on_abort(request_id)
            raise StopAsyncIteration()

        for output in res.outputs:
            i = output.index

            if request.echo and request.max_tokens == 0:
                delta_text = res.prompt
                delta_token_ids = res.prompt_token_ids
                has_echoed[i] = True
            elif request.echo and request.max_tokens > 0 and not has_echoed[i]:
                delta_text = output.text
                delta_token_ids = output.token_ids
                has_echoed[i] = True
            else:
                delta_text = output.text[len(previous_texts[i]):]
                delta_token_ids = output.token_ids[previous_num_tokens[i]:]

            if request.logprobs is not None:
                logprobs = create_logprobs_fn(
                    token_ids=delta_token_ids,
                    num_output_top_logprobs=request.logprobs,
                    initial_text_offset=len(previous_texts[i]),
                )
            else:
                logprobs = None

            previous_texts[i] = output.text
            previous_num_tokens[i] = len(output.token_ids)
            finish_reason = output.finish_reason
            response_json = CompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[
                    CompletionResponseStreamChoice(
                        index=i,
                        text=delta_text,
                        logprobs=logprobs,
                        finish_reason=finish_reason,
                    )
                ]
            ).model_dump_json(exclude_unset=True)
            yield f"data: {response_json}\n\n"

            if output.finish_reason is not None:
                logprobs = LogProbs() if request.logprobs is not None else None
                prompt_tokens = len(res.prompt_token_ids)
                completion_tokens = len(output.token_ids)
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                response_json = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        CompletionResponseStreamChoice(
                            index=i,
                            text="",
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                        )
                    ],
                    usage=final_usage,
                ).model_dump_json(exclude_unset=True)
                yield f"data: {response_json}\n\n"

    yield "data: [DONE]\n\n"


class OpenAIServingCompletion(OpenAIServing):
    """OpenAI API completion endpoint"""

    def __init__(
            self,
            engine,
            served_model: list[str],
            response_role: str = "assistant"):
        super().__init__(engine, served_model, response_role)

    def create_logprobs_fn(
            self,
            token_ids: List[int],
            num_output_top_logprobs: int,
            initial_text_offset: int,
    ) -> LogProbs:
        return LogProbs(
            text_offset=[initial_text_offset],
            token_logprobs=[None] * len(token_ids),
            tokens=[f"token_{tid}" for tid in token_ids],
            top_logprobs=[None] * len(token_ids) if num_output_top_logprobs is None
            else [{} for _ in range(len(token_ids))],
        )

    async def create_completion(self, request: CompletionRequest) -> str | AsyncGenerator[str, None]:
        await self.check_model(request.model)

        request_id = f"cmpl-{time.time()}"
        created_time = int(time.time())

        def on_abort(request_id: str):
            return self.engine.abort(request_id)

        try:
            if request.stream:
                result_generator = self.engine.generate(
                    prompt=request.prompt,
                    sampling_params=request.to_sampling_params(),
                    request_id=request_id,
                )

                stream = completion_stream_generator(
                    request=request,
                    raw_request=None,
                    on_abort=on_abort,
                    result_generator=result_generator,
                    create_logprobs_fn=self.create_logprobs_fn,
                    request_id=request_id,
                    created_time=created_time,
                    model_name=request.model,
                )
                return stream
            else:
                result = []
                async for res in self.engine.generate(
                    prompt=request.prompt,
                    sampling_params=request.to_sampling_params(),
                    request_id=request_id,
                ):
                    result.append(res)

                if not result:
                    raise ValueError("No result generated")

                final_output = result[-1]
                choices = []
                for output in final_output.outputs:
                    logprobs = None
                    if request.logprobs is not None:
                        logprobs = self.create_logprobs_fn(
                            token_ids=output.token_ids,
                            num_output_top_logprobs=request.logprobs,
                            initial_text_offset=0,
                        )

                    choices.append(
                        CompletionResponseChoice(
                            index=output.index,
                            text=output.text,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                        )
                    )

                usage = UsageInfo(
                    prompt_tokens=len(final_output.prompt_token_ids),
                    total_tokens=len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids),
                    completion_tokens=len(final_output.outputs[0].token_ids),
                )

                response = CompletionResponse(
                    id=request_id,
                    created=created_time,
                    model=request.model,
                    choices=choices,
                    usage=usage,
                )

                return response.model_dump_json()
        except Exception as e:
            raise e
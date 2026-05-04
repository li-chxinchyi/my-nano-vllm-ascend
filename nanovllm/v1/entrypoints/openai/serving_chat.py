import time
from typing import AsyncGenerator
from fastapi import Request

from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)
from .serving_engine import OpenAIServing

from nanovllm.v1.core.outputs import RequestOutput


async def chat_completion_stream_generator(
        request: ChatCompletionRequest,
        raw_request: Request,
        on_abort,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        created_time: int,
        model_name: str,
        response_role: str,
) -> AsyncGenerator[str, None]:

    chunk_object_type = "chat.completion.chunk"

    for i in range(request.n):
        choice_data = ChatCompletionResponseStreamChoice(
            index=i, delta=DeltaMessage(role=response_role), finish_reason=None)
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            object=chunk_object_type,
            created=created_time,
            choices=[choice_data],
            model=model_name)
        data = chunk.model_dump_json(exclude_unset=True)
        yield f"data: {data}\n\n"

    if request.echo:
        last_msg_content = ""
        if request.messages and isinstance(
                request.messages, list) and request.messages[-1].get(
                    "content") and request.messages[-1].get(
                        "role") == response_role:
            last_msg_content = request.messages[-1]["content"]
        if last_msg_content:
            for i in range(request.n):
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(content=last_msg_content),
                    finish_reason=None)
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice_data],
                    model=model_name)
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

    previous_texts = [""] * request.n
    previous_num_tokens = [0] * request.n
    finish_reason_sent = [False] * request.n
    async for res in result_generator:
        if raw_request is not None and await raw_request.is_disconnected():
            await on_abort(request_id)
            raise StopAsyncIteration()

        for output in res.outputs:
            i = output.index

            if finish_reason_sent[i]:
                continue

            delta_text = output.text[len(previous_texts[i]):]
            previous_texts[i] = output.text
            previous_num_tokens[i] = len(output.token_ids)

            if output.finish_reason is None:
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(content=delta_text),
                    finish_reason=None)
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice_data],
                    model=model_name)
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"
            else:
                prompt_tokens = len(res.prompt_token_ids)
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=previous_num_tokens[i],
                    total_tokens=prompt_tokens + previous_num_tokens[i],
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(content=delta_text),
                    finish_reason=output.finish_reason)
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice_data],
                    model=model_name)
                if final_usage is not None:
                    chunk.usage = final_usage
                data = chunk.model_dump_json(exclude_unset=True,
                                             exclude_none=True)
                yield f"data: {data}\n\n"
                finish_reason_sent[i] = True

    yield "data: [DONE]\n\n"


async def chat_completion_full_generator(
        request: ChatCompletionRequest,
        raw_request: Request,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        response_role: str,
        model_name: str,
):
    final_res: RequestOutput = None

    async for res in result_generator:
        if raw_request is not None and await raw_request.is_disconnected():
            return ErrorResponse(
                object="error",
                message="Client disconnected",
                type="client_disconnected",
                code=400,
            )
        final_res = res

    assert final_res is not None

    choices = []
    for output in final_res.outputs:
        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=ChatMessage(role=response_role, content=output.text),
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    if request.echo:
        last_msg_content = ""
        if request.messages and isinstance(
                request.messages, list) and request.messages[-1].get(
                    "content") and request.messages[-1].get(
                        "role") == response_role:
            last_msg_content = request.messages[-1]["content"]

        for choice in choices:
            full_message = last_msg_content + choice.message.content
            choice.message.content = full_message

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=choices,
        usage=usage,
    )

    return response


class OpenAIServingChat(OpenAIServing):

    def __init__(
            self,
            engine,
            served_model: list[str],
            response_role: str = "assistant"):
        super().__init__(engine, served_model, response_role)

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1].role

    async def create_chat_completion(
            self, request: ChatCompletionRequest, raw_request: Request
    ):
        await self.check_model(request.model)

        request_id = f"chatcmpl-{time.time()}"
        created_time = int(time.time())

        def on_abort(request_id: str):
            return self.engine.abort(request_id)

        try:
            if request.stream:
                result_generator = self.engine.generate(
                    prompt=None,
                    sampling_params=request.to_sampling_params(),
                    request_id=request_id,
                    messages=request.messages,
                )

                role = self.get_chat_request_role(request)
                stream = chat_completion_stream_generator(
                    request=request,
                    raw_request=raw_request,
                    on_abort=on_abort,
                    result_generator=result_generator,
                    request_id=request_id,
                    created_time=created_time,
                    model_name=request.model,
                    response_role=role,
                )
                return stream
            else:
                result = []
                async for res in self.engine.generate(
                    prompt=None,
                    sampling_params=request.to_sampling_params(),
                    request_id=request_id,
                    messages=request.messages,
                ):
                    result.append(res)

                if not result:
                    raise ValueError("No result generated")

                final_output = result[-1]
                role = self.get_chat_request_role(request)
                usage = UsageInfo(
                    prompt_tokens=len(final_output.prompt_token_ids),
                    total_tokens=len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids),
                    completion_tokens=len(final_output.outputs[0].token_ids),
                )

                choices = []
                for output in final_output.outputs:
                    choice_data = ChatCompletionResponseChoice(
                        index=output.index,
                        message=ChatMessage(role=role, content=output.text),
                        finish_reason=output.finish_reason,
                    )
                    choices.append(choice_data)

                response = ChatCompletionResponse(
                    id=request_id,
                    created=created_time,
                    model=request.model,
                    choices=choices,
                    usage=usage,
                )

                return response.model_dump_json()
        except Exception as e:
            raise e
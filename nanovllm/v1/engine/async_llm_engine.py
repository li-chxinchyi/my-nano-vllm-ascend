import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

# 确保主版本模块可以导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.utils.logger import init_logger

# v1 内部模块使用相对导入
from ..core.outputs import CompletionOutput, RequestOutput

logger = init_logger(__name__)


class AsyncEngineDeadError(RuntimeError):
    pass


class AsyncStream:
    """A stream of RequestOutputs for a request that can be iterated over asynchronously."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: RequestOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[tuple[AsyncStream, dict]] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._request_streams

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def propagate_exception(
            self,
            exc: Exception,
            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
        else:
            for stream in self._request_streams.values():
                stream.put(exc)

    def process_request_output(
            self,
            request_output: RequestOutput,
            *,
            verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

    def add_request(self, request_id: str, **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
            request_id].finished:
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        self.new_requests_event.clear()

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.
    """

    def __init__(
            self,
            *args,
            log_requests: bool = True,
            max_log_len: Optional[int] = None,
            start_engine_loop: bool = True,
            **kwargs) -> None:
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = LLMEngine(*args, **kwargs)
        self.background_loop = None
        self._background_loop_unshielded = None
        self.start_engine_loop = start_engine_loop
        self._request_tracker = RequestTracker()
        self._seq_id_to_request_id: Dict[int, str] = {}
        logger.debug(f"AsyncLLMEngine.__init__ called, start_engine_loop={start_engine_loop}")

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and not self.background_loop.done())

    def start_background_loop(self) -> None:
        """Start the background loop."""
        logger.debug("start_background_loop called")
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self._request_tracker.init_event()

        logger.debug("Creating background loop task")
        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            lambda task: self._handle_task_exception(task, exc=None))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)
        logger.debug("Background loop started")

    def _handle_task_exception(self, task, exc):
        logger.debug(f"_handle_task_exception called, task.cancelled={task.cancelled()}, exc={exc}")
        if not task.cancelled():
            if exc:
                logger.debug(f"Propagating exception: {exc}")
                self._request_tracker.propagate_exception(exc)
            else:
                logger.debug("Getting task.result()")
                task.result()

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""
        try:
            logger.debug("engine_step: starting")
            new_requests, finished_requests = (
                self._request_tracker.get_new_and_finished_requests())

            for new_request in new_requests:
                request_id = new_request.pop("request_id")
                sampling_params = new_request.pop("sampling_params")
                prompt = new_request.pop("prompt", None)
                prompt_token_ids = new_request.pop("prompt_token_ids", None)

                if prompt_token_ids is not None:
                    prompt = prompt_token_ids

                images = new_request.pop("images", None)
                pixel_values = new_request.pop("pixel_values", None)
                image_grid_thw = new_request.pop("image_grid_thw", None)
                vision_counts = new_request.pop("vision_counts", None)
                vision_placeholders = new_request.pop("vision_placeholders", None)

                self.engine.add_request(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    images=images,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    vision_counts=vision_counts,
                    vision_placeholders=vision_placeholders,
                )

                for seq in self.engine.scheduler.waiting:
                    if seq.request_id == request_id and seq.seq_id not in self._seq_id_to_request_id:
                        self._seq_id_to_request_id[seq.seq_id] = request_id
                        logger.debug(f"Mapped seq_id {seq.seq_id} to request_id {request_id}")
                        break

            if finished_requests:
                for request_id in finished_requests:
                    self.engine.abort_request(request_id)

            logger.debug("engine_step: calling engine.step()")
            outputs, num_tokens = self.engine.step()
            logger.debug(f"engine_step returned {len(outputs)} outputs: {outputs}")

            request_outputs = []

            logger.debug(f"engine_step: processing {len(outputs)} outputs")
            for seq_id, completion_token_ids, prompt_len, cache_tokens in outputs:
                completion_token_ids_list = list(completion_token_ids) if not isinstance(completion_token_ids,
                                                                                         list) else completion_token_ids

                seq = self._find_sequence_by_id(seq_id)

                if seq:
                    logger.debug(
                        f"Found seq {seq_id}, is_finished={seq.is_finished}, num_completion_tokens={seq.num_completion_tokens}")
                    generated_text = self.engine.tokenizer.decode(completion_token_ids_list)
                    if seq.is_finished or seq.status.name == "FINISHED":
                        completion_output = CompletionOutput(
                            index=0,
                            text=generated_text,
                            token_ids=completion_token_ids_list,
                            cumulative_logprob=0.0,
                            logprobs=None,
                            finish_reason=seq.finish_reason.name.lower() if seq.finish_reason else "length",
                        )
                        request_output = RequestOutput(
                            request_id=seq.request_id,
                            prompt="",
                            prompt_token_ids=list(seq.prompt_token_ids),
                            prompt_logprobs=None,
                            outputs=[completion_output],
                            finished=True,
                        )
                        request_outputs.append(request_output)
                        logger.debug(f"Added finished request_output for {seq_id}")
                    elif seq.num_completion_tokens > 0:
                        completion_output = CompletionOutput(
                            index=0,
                            text=generated_text,
                            token_ids=completion_token_ids_list,
                            cumulative_logprob=0.0,
                            logprobs=None,
                            finish_reason=None,
                        )
                        request_output = RequestOutput(
                            request_id=seq.request_id,
                            prompt="",
                            prompt_token_ids=prompt_token_ids if 'prompt_token_ids' in locals() else list(
                                seq.prompt_token_ids)[:prompt_len],
                            prompt_logprobs=None,
                            outputs=[completion_output],
                            finished=False,
                        )
                        request_outputs.append(request_output)
                        logger.debug(f"Added running request_output for {seq_id}")
                else:
                    request_id = self._seq_id_to_request_id.get(seq_id, str(seq_id))
                    logger.debug(
                        f"Could not find seq {seq_id} in scheduler (likely finished), request_id={request_id}, got completion_token_ids: {completion_token_ids_list[:5]}")
                    generated_text = self.engine.tokenizer.decode(completion_token_ids_list)
                    completion_output = CompletionOutput(
                        index=0,
                        text=generated_text,
                        token_ids=completion_token_ids_list,
                        cumulative_logprob=0.0,
                        logprobs=None,
                        finish_reason="length",
                    )
                    request_output = RequestOutput(
                        request_id=request_id,
                        prompt="",
                        prompt_token_ids=completion_token_ids_list[:prompt_len] if prompt_len > 0 else [],
                        prompt_logprobs=None,
                        outputs=[completion_output],
                        finished=True,
                    )
                    request_outputs.append(request_output)
                    logger.debug(f"Added finished request_output for missing seq {seq_id}, request_id={request_id}")

            for request_output in request_outputs:
                self._request_tracker.process_request_output(
                    request_output, verbose=self.log_requests)

            return len(request_outputs) > 0
        except Exception as e:
            logger.error(f"Error in engine_step: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _find_sequence_by_id(self, seq_id):
        """Find a sequence by its seq_id in the scheduler."""
        for seq in self.engine.scheduler.waiting:
            if seq.seq_id == seq_id:
                return seq
        for seq in self.engine.scheduler.running:
            if seq.seq_id == seq_id:
                return seq
        return None

    async def run_engine_loop(self):
        logger.debug("run_engine_loop started")
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                logger.debug("Waiting for new requests...")
                await self._request_tracker.wait_for_new_requests()
                logger.debug("Got new requests!")
            has_requests_in_progress = await self.engine_step()
            await asyncio.sleep(0)

    def add_request(
            self,
            request_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: Optional[float] = None,
    ) -> AsyncStream:
        logger.debug(f"add_request called for {request_id}, is_running={self.is_running}")
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.max_log_len]
            logger.info(f"Received request {request_id}: "
                        f"prompt: {shortened_prompt!r}, "
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {shortened_token_ids}.")

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running.")

        if arrival_time is None:
            arrival_time = time.time()

        if prompt_token_ids is None:
            if prompt is None:
                raise ValueError("Either prompt or prompt_token_ids must be provided.")
            prompt_token_ids = self.engine.tokenizer.encode(prompt)

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
        )

        return stream

    async def generate(
            self,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            messages: Optional[List[Dict]] = None,
    ):
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the request.
        """
        arrival_time = time.monotonic()

        logger.debug(f"generate called, prompt={prompt}, messages={messages}")

        if messages is not None:
            try:
                if hasattr(self.engine.tokenizer, 'apply_chat_template'):
                    prompt = self.engine.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    logger.debug(f"Converted messages to prompt: {prompt[:200]}...")
                else:
                    prompt = str(messages)
                    logger.debug("tokenizer has no apply_chat_template, using str(messages)")
            except Exception as e:
                logger.debug(f"Error applying chat template: {e}")
                prompt = str(messages)

        try:
            stream = self.add_request(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time,
            )

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running.")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.log_requests)

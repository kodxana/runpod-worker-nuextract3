"""Lazy direct-Transformers NuExtract3 runtime."""

from __future__ import annotations

import json
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    MAX_CONTEXT_TOKENS,
    MIN_GPU_MEMORY_BYTES,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
    Settings,
    require_model_id,
)
from .errors import WorkerError
from .model_manifest import BAKED_MODEL_PATH, MODEL_FILES, VERIFIED_LFS_FILES
from .schema import (
    Request,
    parse_json_output,
    split_reasoning,
    validate_structured_output,
    validate_template,
)


class _CancellationCriteria:
    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, *_: Any, **__: Any) -> bool:
        return self.event.is_set()


@dataclass(frozen=True)
class InferenceResult:
    value: dict[str, Any] | str
    reasoning: str | None
    input_tokens: int
    generated_tokens: int
    finish_reason: str
    duration_ms: int
    preprocess_ms: int = 0
    generation_queue_ms: int = 0
    generation_ms: int = 0
    postprocess_ms: int = 0
    generated_tokens_per_second: float = 0.0
    gpu_peak_allocated_bytes: int = 0
    gpu_peak_reserved_bytes: int = 0


def _complete_snapshot(path: Path) -> bool:
    if not path.is_dir() or not all((path / name).is_file() for name in MODEL_FILES):
        return False
    return all((path / name).stat().st_size == size for name, size, _ in VERIFIED_LFS_FILES)


def resolve_model_snapshot(
    *,
    baked_model_path: str | Path = BAKED_MODEL_PATH,
) -> Path:
    """Return the complete model snapshot baked into the image."""

    require_model_id()
    model_path = Path(baked_model_path)
    if not _complete_snapshot(model_path):
        raise RuntimeError(f"Baked NuExtract3 snapshot is incomplete or invalid at {model_path}")
    return model_path


class TransformersRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any = None
        self.processor: Any = None
        self.device = "cuda:0"
        self._torch: Any = None
        self._processor_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._gpu_failed = False

    def load(self) -> None:
        require_model_id()
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(f"transformers must be exactly {TRANSFORMERS_VERSION}")
        if torch.__version__.split("+", 1)[0] != TORCH_VERSION:
            raise RuntimeError(f"torch must be exactly {TORCH_VERSION}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("A GPU with native bfloat16 support is required")
        if torch.cuda.get_device_properties(0).total_memory < MIN_GPU_MEMORY_BYTES:
            raise RuntimeError("At least 24 GB of GPU memory is required")

        model_path = resolve_model_snapshot()
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=False,
        ).eval()

        parameter_devices = {parameter.device.type for parameter in model.parameters()}
        if parameter_devices != {"cuda"}:
            raise RuntimeError("All model parameters must reside on CUDA")
        self._assert_fast_path(model)

        self._torch = torch
        self.processor = processor
        self.model = model

    @staticmethod
    def _assert_fast_path(model: Any) -> None:
        layers = model.model.language_model.layers
        linear_attention = next(
            (layer.linear_attn for layer in layers if hasattr(layer, "linear_attn")),
            None,
        )
        if linear_attention is None:
            raise RuntimeError("Model has no linear-attention layers")
        chunk_module = getattr(linear_attention.chunk_gated_delta_rule, "__module__", "")
        recurrent_module = getattr(linear_attention.recurrent_gated_delta_rule, "__module__", "")
        conv_module = getattr(linear_attention.causal_conv1d_fn, "__module__", "")
        if not chunk_module.startswith("fla.") or not recurrent_module.startswith("fla."):
            raise RuntimeError("Flash Linear Attention fast path is unavailable")
        if not conv_module.startswith("causal_conv1d"):
            raise RuntimeError("causal-conv1d fast path is unavailable")

    def infer(
        self,
        request: Request,
        message_items: list[dict[str, Any]],
        cancellation_event: threading.Event | None = None,
    ) -> InferenceResult:
        if self.model is None or self.processor is None or self._torch is None:
            raise RuntimeError("Model runtime is not loaded")
        torch = self._torch
        started = time.monotonic()
        inputs: Any = None
        generated: Any = None
        generated_ids: Any = None
        try:
            messages = [{"role": "user", "content": message_items}]
            chat_kwargs: dict[str, Any] = {"enable_thinking": request.generation.thinking}
            if request.mode == "structured":
                chat_kwargs["template"] = json.dumps(
                    request.template, ensure_ascii=False, separators=(",", ":")
                )
                if request.instructions:
                    chat_kwargs["instructions"] = request.instructions
            else:
                chat_kwargs["mode"] = request.mode

            preprocess_started = time.monotonic()
            with self._processor_lock:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    **chat_kwargs,
                )
            input_tokens = int(inputs["input_ids"].shape[1])
            if input_tokens + request.generation.max_new_tokens > MAX_CONTEXT_TOKENS:
                raise WorkerError(
                    "CONTEXT_LIMIT_EXCEEDED",
                    f"Processed input plus max_new_tokens exceeds {MAX_CONTEXT_TOKENS} tokens.",
                )
            preprocess_ms = round((time.monotonic() - preprocess_started) * 1_000)

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": request.generation.max_new_tokens,
                "max_time": self.settings.generation_timeout_seconds,
                "return_dict_in_generate": False,
                "use_cache": True,
            }
            if request.generation.temperature > 0:
                generation_kwargs.update(
                    do_sample=True,
                    temperature=request.generation.temperature,
                    top_p=request.generation.top_p,
                    top_k=request.generation.top_k,
                )
            else:
                generation_kwargs["do_sample"] = False
            if cancellation_event is not None:
                generation_kwargs["stopping_criteria"] = [_CancellationCriteria(cancellation_event)]

            queue_started = time.monotonic()
            self._acquire_generation(cancellation_event)
            try:
                generation_queue_ms = round((time.monotonic() - queue_started) * 1_000)
                if cancellation_event is not None and cancellation_event.is_set():
                    raise WorkerError("JOB_CANCELLED", "Job was cancelled.")
                if self._gpu_failed:
                    raise WorkerError(
                        "GPU_RUNTIME_UNAVAILABLE",
                        "GPU runtime is unavailable after an earlier fatal error.",
                        retryable=True,
                        refresh_worker=True,
                    )
                try:
                    if request.generation.temperature > 0:
                        torch.manual_seed(request.generation.seed)
                        torch.cuda.manual_seed_all(request.generation.seed)
                    torch.cuda.reset_peak_memory_stats()
                    generation_started = time.monotonic()
                    inputs = inputs.to(self.device)
                    with torch.inference_mode():
                        generated = self.model.generate(**inputs, **generation_kwargs)
                    generated_ids = generated[:, input_tokens:].cpu()
                    generation_seconds = time.monotonic() - generation_started
                    gpu_peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
                    gpu_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
                    generated = None
                    inputs = None
                except torch.OutOfMemoryError as exc:
                    generated_ids = None
                    generated = None
                    inputs = None
                    self._mark_gpu_failed(torch)
                    raise WorkerError(
                        "GPU_OOM",
                        "GPU ran out of memory while processing the bounded request.",
                        retryable=True,
                        refresh_worker=True,
                    ) from exc
                except Exception as exc:
                    generated_ids = None
                    generated = None
                    inputs = None
                    self._mark_gpu_failed(torch)
                    raise WorkerError(
                        "GPU_RUNTIME_ERROR",
                        "GPU inference failed and the worker must be refreshed.",
                        retryable=True,
                        refresh_worker=True,
                    ) from exc
            finally:
                self._generation_lock.release()

            generation_ms = round(generation_seconds * 1_000)
            generated_tokens = int(generated_ids.shape[1])
            if cancellation_event is not None and cancellation_event.is_set():
                raise WorkerError("JOB_CANCELLED", "Job was cancelled.")
            if generated_tokens == 0:
                raise WorkerError("MODEL_OUTPUT_EMPTY", "Model returned no output.")

            eos = self.model.generation_config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            last_token = int(generated_ids[0, -1].item())
            if last_token in eos_ids:
                finish_reason = "eos"
            elif generated_tokens >= request.generation.max_new_tokens:
                raise WorkerError("MODEL_OUTPUT_TRUNCATED", "Model output reached max_new_tokens.")
            else:
                raise WorkerError(
                    "MODEL_TIMEOUT", "Model generation exceeded its time limit.", retryable=True
                )

            postprocess_started = time.monotonic()
            with self._processor_lock:
                decoded = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            reasoning, answer = split_reasoning(decoded, request.generation.thinking)
            if not answer:
                raise WorkerError("MODEL_OUTPUT_EMPTY", "Model returned no answer.")

            if request.mode == "structured":
                value: dict[str, Any] | str = parse_json_output(answer)
                validate_structured_output(value, request.template)
            elif request.mode == "template-generation":
                value = parse_json_output(answer)
                try:
                    validate_template(value)
                except WorkerError as exc:
                    raise WorkerError(
                        "MODEL_OUTPUT_SCHEMA_MISMATCH",
                        "Model output is not a valid NuExtract template.",
                    ) from exc
            else:
                value = answer

            return InferenceResult(
                value=value,
                reasoning=reasoning if request.generation.return_reasoning else None,
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
                finish_reason=finish_reason,
                duration_ms=round((time.monotonic() - started) * 1_000),
                preprocess_ms=preprocess_ms,
                generation_queue_ms=generation_queue_ms,
                generation_ms=generation_ms,
                postprocess_ms=round((time.monotonic() - postprocess_started) * 1_000),
                generated_tokens_per_second=(
                    round(generated_tokens / generation_seconds, 3)
                    if generation_seconds > 0
                    else 0.0
                ),
                gpu_peak_allocated_bytes=gpu_peak_allocated_bytes,
                gpu_peak_reserved_bytes=gpu_peak_reserved_bytes,
            )
        except WorkerError:
            raise
        finally:
            del generated_ids
            del generated
            del inputs

    def _mark_gpu_failed(self, torch: Any) -> None:
        self._gpu_failed = True
        with suppress(Exception):
            torch.cuda.empty_cache()

    def _acquire_generation(self, cancellation_event: threading.Event | None) -> None:
        if cancellation_event is None:
            self._generation_lock.acquire()
            return
        while not cancellation_event.is_set():
            if self._generation_lock.acquire(timeout=0.1):
                return
        raise WorkerError("JOB_CANCELLED", "Job was cancelled.")

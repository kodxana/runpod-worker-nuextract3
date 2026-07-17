from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import nuextract_worker.runtime as runtime_module
from nuextract_worker.config import MODEL_ID, Settings, require_model_id
from nuextract_worker.errors import WorkerError
from nuextract_worker.model_manifest import MODEL_FILES, VERIFIED_LFS_FILES
from nuextract_worker.runtime import TransformersRuntime, resolve_model_snapshot
from nuextract_worker.schema import validate_request


def _complete_snapshot(root: Path) -> Path:
    snapshot = root / "nuextract3"
    snapshot.mkdir(parents=True)
    for name in MODEL_FILES:
        (snapshot / name).write_text("fixture")
    return snapshot


def test_snapshot_resolution_uses_the_complete_baked_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "VERIFIED_LFS_FILES",
        tuple((name, len("fixture"), digest) for name, _, digest in VERIFIED_LFS_FILES),
    )
    snapshot = _complete_snapshot(tmp_path)

    result = resolve_model_snapshot(baked_model_path=snapshot)
    assert result == snapshot


def test_snapshot_resolution_rejects_wrong_model_and_incomplete_bake(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "other/model")
    with pytest.raises(RuntimeError, match="Configured model"):
        resolve_model_snapshot(baked_model_path=tmp_path)

    monkeypatch.setenv("MODEL_NAME", MODEL_ID)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        resolve_model_snapshot(baked_model_path=incomplete)


def test_fixed_model_is_default_with_legacy_override_validation(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)
    assert require_model_id() == MODEL_ID

    monkeypatch.setenv("MODEL_ID", MODEL_ID)
    assert require_model_id() == MODEL_ID

    monkeypatch.setenv("MODEL_NAME", "other/model")
    with pytest.raises(RuntimeError, match="same model"):
        require_model_id()


def _function_from(module: str):
    def function():
        return None

    function.__module__ = module
    return function


def _fast_path_model(chunk="fla.ops", recurrent="fla.ops", conv="causal_conv1d"):
    linear = SimpleNamespace(
        chunk_gated_delta_rule=_function_from(chunk),
        recurrent_gated_delta_rule=_function_from(recurrent),
        causal_conv1d_fn=_function_from(conv),
    )
    layers = [SimpleNamespace(linear_attn=linear)]
    return SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))


def test_fast_path_assertion_checks_fla_and_causal_conv() -> None:
    TransformersRuntime._assert_fast_path(_fast_path_model())
    with pytest.raises(RuntimeError, match="Flash Linear Attention"):
        TransformersRuntime._assert_fast_path(_fast_path_model(chunk="transformers.fallback"))
    with pytest.raises(RuntimeError, match="causal-conv1d"):
        TransformersRuntime._assert_fast_path(_fast_path_model(conv="transformers.fallback"))
    with pytest.raises(RuntimeError, match="no linear-attention"):
        TransformersRuntime._assert_fast_path(
            SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[])))
        )


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Tensor:
    def __init__(self, tokens):
        self.tokens = list(tokens)

    @property
    def shape(self):
        return (1, len(self.tokens))

    def __getitem__(self, key):
        row, column = key
        if isinstance(column, slice):
            return _Tensor(self.tokens[column])
        return _Scalar(self.tokens[column])

    def cpu(self):
        return self


class _Batch(dict):
    def to(self, device):
        self.device = device
        return self


class _Cuda:
    def __init__(self):
        self.emptied = False

    def manual_seed_all(self, _seed):
        return None

    def empty_cache(self):
        self.emptied = True

    def synchronize(self):
        return None

    def reset_peak_memory_stats(self):
        return None

    def max_memory_allocated(self):
        return 9_000

    def max_memory_reserved(self):
        return 10_000


class _Torch:
    class OutOfMemoryError(Exception):
        pass

    def __init__(self):
        self.cuda = _Cuda()
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return None

    def inference_mode(self):
        return nullcontext()


class _Processor:
    def __init__(self, decoded, input_tokens=3):
        self.decoded = decoded
        self.input_tokens = input_tokens
        self.chat_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_kwargs = (messages, kwargs)
        return _Batch(input_ids=_Tensor(range(self.input_tokens)))

    def batch_decode(self, generated_ids, **kwargs):
        self.generated_ids = generated_ids
        return [self.decoded]


class _Model:
    def __init__(
        self,
        torch,
        generated=(7, 99),
        input_tokens=3,
        oom=False,
        runtime_error=False,
        other_error=False,
        delay=0.0,
    ):
        self.torch = torch
        self.generated = generated
        self.input_tokens = input_tokens
        self.oom = oom
        self.runtime_error = runtime_error
        self.other_error = other_error
        self.delay = delay
        self.generation_started = threading.Event()
        self.active_generations = 0
        self.max_active_generations = 0
        self._tracking_lock = threading.Lock()
        self.generation_config = SimpleNamespace(eos_token_id=99)

    def generate(self, **kwargs):
        self.kwargs = kwargs
        with self._tracking_lock:
            self.active_generations += 1
            self.max_active_generations = max(self.max_active_generations, self.active_generations)
        try:
            self.generation_started.set()
            if self.oom:
                raise self.torch.OutOfMemoryError("oom")
            if self.runtime_error:
                raise RuntimeError("cuda launch failed")
            if self.other_error:
                raise ValueError("triton compilation failed")
            time.sleep(self.delay)
            return _Tensor([0] * self.input_tokens + list(self.generated))
        finally:
            with self._tracking_lock:
                self.active_generations -= 1


def _loaded_runtime(
    decoded,
    *,
    generated=(7, 99),
    input_tokens=3,
    oom=False,
    runtime_error=False,
    other_error=False,
    delay=0.0,
):
    torch = _Torch()
    runtime = TransformersRuntime(Settings())
    runtime._torch = torch
    runtime.processor = _Processor(decoded, input_tokens)
    runtime.model = _Model(
        torch,
        generated,
        input_tokens,
        oom,
        runtime_error,
        other_error,
        delay,
    )
    return runtime


def _structured_request(**generation):
    value = {
        "schema_version": "1",
        "mode": "structured",
        "sources": [{"type": "text", "text": "total 12.5"}],
        "template": {"total": "number"},
    }
    if generation:
        value["generation"] = generation
    return validate_request(value)


def test_infer_builds_native_chat_and_validates_structured_output() -> None:
    runtime = _loaded_runtime('{"total":12.5}')
    request = _structured_request()

    result = runtime.infer(request, [{"type": "text", "text": "total 12.5"}])

    assert result.value == {"total": 12.5}
    assert result.generated_tokens == 2
    assert result.finish_reason == "eos"
    _, kwargs = runtime.processor.chat_kwargs
    assert kwargs["enable_thinking"] is False
    assert kwargs["template"] == '{"total":"number"}'
    assert runtime.model.kwargs["do_sample"] is False
    assert "temperature" not in runtime.model.kwargs
    assert "top_p" not in runtime.model.kwargs
    assert "top_k" not in runtime.model.kwargs
    assert runtime.model.kwargs["return_dict_in_generate"] is False
    assert result.gpu_peak_allocated_bytes == 9_000
    assert result.gpu_peak_reserved_bytes == 10_000
    assert result.generated_tokens_per_second > 0


def test_infer_splits_reasoning_and_uses_bounded_sampling() -> None:
    runtime = _loaded_runtime('work here\n</think>\n{"total":12.5}')
    request = _structured_request(
        thinking=True,
        return_reasoning=True,
        max_new_tokens=32,
    )

    result = runtime.infer(request, [{"type": "text", "text": "source"}])
    assert result.reasoning == "work here"
    assert runtime.model.kwargs["do_sample"] is True
    assert runtime.model.kwargs["temperature"] == 0.6
    assert runtime.model.kwargs["top_p"] == 1.0
    assert runtime.model.kwargs["top_k"] == 0
    assert runtime.model.kwargs["max_new_tokens"] == 32
    assert runtime._torch.seed == 0


def test_infer_uses_per_job_sampling_controls() -> None:
    runtime = _loaded_runtime('{"total":12.5}')
    request = _structured_request(
        temperature=0.25,
        top_p=0.85,
        top_k=40,
        seed=1234,
    )
    runtime.infer(request, [])

    assert runtime.model.kwargs["do_sample"] is True
    assert runtime.model.kwargs["temperature"] == 0.25
    assert runtime.model.kwargs["top_p"] == 0.85
    assert runtime.model.kwargs["top_k"] == 40
    assert runtime._torch.seed == 1234


def test_infer_installs_cooperative_cancellation_criteria() -> None:
    cancellation_event = threading.Event()
    runtime = _loaded_runtime('{"total":12.5}')
    runtime.infer(_structured_request(), [], cancellation_event)

    criterion = runtime.model.kwargs["stopping_criteria"][0]
    assert criterion() is False
    cancellation_event.set()
    assert criterion() is True


def test_generated_template_errors_are_model_output_errors() -> None:
    runtime = _loaded_runtime('{"field":"unsupported-type"}')
    request = validate_request(
        {
            "schema_version": "1",
            "mode": "template-generation",
            "sources": [{"type": "text", "text": "rental"}],
        }
    )

    with pytest.raises(WorkerError) as caught:
        runtime.infer(request, [{"type": "text", "text": "rental"}])
    assert caught.value.code == "MODEL_OUTPUT_SCHEMA_MISMATCH"


def test_infer_rejects_context_overflow_and_truncation(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "MAX_CONTEXT_TOKENS", 4)
    runtime = _loaded_runtime('{"total":12.5}', input_tokens=4)
    with pytest.raises(WorkerError) as caught:
        runtime.infer(_structured_request(max_new_tokens=1), [])
    assert caught.value.code == "CONTEXT_LIMIT_EXCEEDED"

    monkeypatch.setattr(runtime_module, "MAX_CONTEXT_TOKENS", 32_768)
    runtime = _loaded_runtime('{"total":12.5}', generated=(7, 8))
    with pytest.raises(WorkerError) as caught:
        runtime.infer(_structured_request(max_new_tokens=2), [])
    assert caught.value.code == "MODEL_OUTPUT_TRUNCATED"


def test_infer_maps_gpu_oom_to_refreshable_error() -> None:
    runtime = _loaded_runtime("unused", oom=True)
    with pytest.raises(WorkerError) as caught:
        runtime.infer(_structured_request(), [])
    assert caught.value.code == "GPU_OOM"
    assert caught.value.retryable is True
    assert caught.value.refresh_worker is True
    assert runtime._torch.cuda.emptied is True

    runtime.model.oom = False
    with pytest.raises(WorkerError) as unavailable:
        runtime.infer(_structured_request(), [])
    assert unavailable.value.code == "GPU_RUNTIME_UNAVAILABLE"
    assert unavailable.value.refresh_worker is True


def test_infer_marks_runtime_error_as_fatal() -> None:
    runtime = _loaded_runtime("unused", runtime_error=True)
    with pytest.raises(WorkerError) as caught:
        runtime.infer(_structured_request(), [])
    assert caught.value.code == "GPU_RUNTIME_ERROR"
    assert caught.value.retryable is True
    assert caught.value.refresh_worker is True

    runtime.model.runtime_error = False
    with pytest.raises(WorkerError) as unavailable:
        runtime.infer(_structured_request(), [])
    assert unavailable.value.code == "GPU_RUNTIME_UNAVAILABLE"


def test_infer_marks_non_runtime_gpu_exception_as_fatal() -> None:
    runtime = _loaded_runtime("unused", other_error=True)
    with pytest.raises(WorkerError) as caught:
        runtime.infer(_structured_request(), [])
    assert caught.value.code == "GPU_RUNTIME_ERROR"
    assert caught.value.refresh_worker is True


def test_concurrent_inference_keeps_exactly_one_gpu_generation() -> None:
    runtime = _loaded_runtime('{"total":12.5}', delay=0.01)
    request = _structured_request()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: runtime.infer(request, []), range(2)))

    assert [result.value for result in results] == [{"total": 12.5}, {"total": 12.5}]
    assert runtime.model.max_active_generations == 1
    assert any(result.generation_queue_ms > 0 for result in results)


def test_cancelled_inference_stops_waiting_for_gpu_owner() -> None:
    runtime = _loaded_runtime('{"total":12.5}', delay=0.5)
    request = _structured_request()
    cancellation_event = threading.Event()

    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(runtime.infer, request, [])
        assert runtime.model.generation_started.wait(1)
        waiting = executor.submit(runtime.infer, request, [], cancellation_event)
        time.sleep(0.02)
        cancellation_event.set()
        with pytest.raises(WorkerError) as caught:
            waiting.result(timeout=0.3)
        assert caught.value.code == "JOB_CANCELLED"
        assert active.result(timeout=1).value == {"total": 12.5}

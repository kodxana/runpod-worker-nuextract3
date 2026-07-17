from __future__ import annotations

import json

import pytest

import nuextract_worker.worker as worker_module
from nuextract_worker.config import MODEL_ID, MODEL_REVISION, Settings
from nuextract_worker.errors import WorkerError
from nuextract_worker.media import ResolvedDocument
from nuextract_worker.runtime import InferenceResult
from nuextract_worker.worker import Worker, _two_concurrency


class _Image:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Resolver:
    def __init__(self, document):
        self.document = document

    def resolve(self, request, temp_dir):
        self.request = request
        self.temp_dir = temp_dir
        return self.document


class _Runtime:
    def __init__(self, value=None, error=None):
        self.value = {"total": 12.5} if value is None else value
        self.error = error

    def infer(self, request, items, cancellation_event=None):
        self.cancellation_event = cancellation_event
        if self.error:
            raise self.error
        return InferenceResult(
            value=self.value,
            reasoning=None,
            input_tokens=10,
            generated_tokens=5,
            finish_reason="eos",
            duration_ms=12,
        )


class _Uploader:
    def upload(self, **kwargs):
        self.kwargs = kwargs
        return {
            "delivery": "s3",
            "bucket": kwargs["config"].bucket,
            "key": "result.json",
            "content_type": kwargs["content_type"],
            "bytes": len(kwargs["body"]),
            "sha256": kwargs["sha256"],
        }


def _job(*, mode="structured", output=None):
    request = {
        "schema_version": "1",
        "mode": mode,
        "sources": [{"type": "text", "text": "total 12.5"}],
    }
    if mode == "structured":
        request["template"] = {"total": "number"}
    if output is not None:
        request["output"] = output
    return {"id": "job-123", "input": request}


def _document(image=None):
    images = [] if image is None else [image]
    items = [{"type": "text", "text": "total 12.5"}]
    return ResolvedDocument(items, images, 0, 0, 0)


def _decode_error(response):
    return json.loads(response["error"])


def test_worker_returns_integrity_usage_and_closes_document() -> None:
    image = _Image()
    document = _document(image)
    resolver = _Resolver(document)
    progress = []
    worker = Worker(
        _Runtime(),
        Settings(),
        media_factory=lambda: resolver,
        progress=lambda job, message: progress.append(message),
    )

    response = worker(_job())

    assert response["model"] == {"id": MODEL_ID, "revision": MODEL_REVISION}
    assert response["result"]["data"] == {"total": 12.5}
    assert response["result"]["bytes"] == len(b'{"total":12.5}')
    assert response["usage"]["images"] == 1
    assert response["usage"]["input_tokens"] == 10
    assert response["usage"]["generated_tokens_per_second"] == 0.0
    assert response["usage"]["gpu_peak_allocated_bytes"] == 0
    assert image.closed is True
    assert progress == [
        "validating request",
        "loading document sources",
        "running inference",
        "preparing result",
        "completed",
    ]


def test_markdown_uses_text_result() -> None:
    worker = Worker(
        _Runtime(value="# Heading"),
        Settings(),
        media_factory=lambda: _Resolver(_document()),
    )
    response = worker(_job(mode="markdown"))
    assert response["result"]["text"] == "# Heading"
    assert response["result"]["content_type"] == "text/markdown; charset=utf-8"


def test_markdown_rejects_invalid_unicode_model_output() -> None:
    worker = Worker(
        _Runtime(value="invalid \ud800"),
        Settings(),
        media_factory=lambda: _Resolver(_document()),
    )
    assert _decode_error(worker(_job(mode="markdown")))["code"] == ("MODEL_OUTPUT_INVALID_TEXT")


def test_invalid_and_unhandled_errors_are_safe() -> None:
    worker = Worker(_Runtime(), Settings())
    invalid = _decode_error(worker({"id": "job", "input": {"secret": "do not return"}}))
    assert invalid["code"] == "INVALID_REQUEST"
    assert "secret" not in invalid["message"]

    worker = Worker(
        _Runtime(error=RuntimeError("private traceback value")),
        Settings(),
        media_factory=lambda: _Resolver(_document()),
    )
    internal = _decode_error(worker(_job()))
    assert internal["code"] == "INTERNAL_ERROR"
    assert internal["retryable"] is True
    assert "private" not in internal["message"]


def test_cancelled_job_stops_before_source_resolution() -> None:
    import threading

    cancellation_event = threading.Event()
    cancellation_event.set()
    resolver_called = False

    def media_factory():
        nonlocal resolver_called
        resolver_called = True
        return _Resolver(_document())

    response = Worker(_Runtime(), Settings(), media_factory=media_factory)(
        _job(), cancellation_event
    )
    assert _decode_error(response)["code"] == "JOB_CANCELLED"
    assert resolver_called is False


def test_worker_error_preserves_retry_and_refresh_flags() -> None:
    worker = Worker(
        _Runtime(
            error=WorkerError(
                "GPU_OOM",
                "GPU ran out of memory.",
                retryable=True,
                refresh_worker=True,
            )
        ),
        Settings(),
        media_factory=lambda: _Resolver(_document()),
    )
    response = worker(_job())
    error = _decode_error(response)
    assert error["code"] == "GPU_OOM"
    assert error["retryable"] is True
    assert response["refresh_worker"] is True


def test_s3_delivery_uses_top_level_runpod_config() -> None:
    uploader = _Uploader()
    worker = Worker(
        _Runtime(),
        Settings(),
        media_factory=lambda: _Resolver(_document()),
        uploader=uploader,
    )
    job = _job(output={"delivery": "s3", "presign_ttl_seconds": 600})
    job["s3Config"] = {
        "accessId": "access",
        "accessSecret": "secret",
        "bucketName": "bucket-name",
        "endpointUrl": "https://s3.example.com",
    }

    response = worker(job)
    assert response["result"]["delivery"] == "s3"
    assert uploader.kwargs["presign_ttl_seconds"] == 600
    assert uploader.kwargs["config"].bucket == "bucket-name"


@pytest.mark.parametrize(
    ("delivery", "code"),
    [("inline", "INLINE_RESULT_TOO_LARGE"), ("auto", "S3_CONFIG_REQUIRED")],
)
def test_oversized_delivery_has_no_unbounded_inline_fallback(delivery, code, monkeypatch) -> None:
    monkeypatch.setattr(worker_module, "MAX_INLINE_RESULT_BYTES", 5)
    worker = Worker(
        _Runtime(value={"value": "large"}),
        Settings(),
        media_factory=lambda: _Resolver(_document()),
    )
    response = worker(_job(output={"delivery": delivery}))
    assert _decode_error(response)["code"] == code


def test_progress_failures_do_not_fail_the_job() -> None:
    def broken_progress(job, message):
        raise RuntimeError("progress transport failed")

    worker = Worker(
        _Runtime(),
        Settings(),
        media_factory=lambda: _Resolver(_document()),
        progress=broken_progress,
    )
    assert worker(_job())["result"]["delivery"] == "inline"


def test_concurrency_is_bounded_at_two() -> None:
    assert _two_concurrency(1) == 2
    assert _two_concurrency(100) == 2

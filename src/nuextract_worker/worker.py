"""Queue handler orchestration and safe response construction."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any

from .config import MAX_INLINE_RESULT_BYTES, MODEL_ID, MODEL_REVISION, Settings
from .errors import WorkerError
from .media import MediaResolver, ResolvedDocument
from .runtime import InferenceResult
from .schema import Request, validate_request
from .storage import S3Uploader, resolve_s3_config

logger = logging.getLogger(__name__)
Progress = Callable[[dict[str, Any], str], None]


class Worker:
    def __init__(
        self,
        runtime: Any,
        settings: Settings,
        *,
        media_factory: Callable[[], MediaResolver] | None = None,
        uploader: S3Uploader | None = None,
        progress: Progress | None = None,
    ) -> None:
        self.runtime = runtime
        self.settings = settings
        self.media_factory = media_factory or (lambda: MediaResolver(settings))
        self.uploader = uploader or S3Uploader(settings)
        self.progress = progress

    def __call__(
        self,
        job: dict[str, Any],
        cancellation_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        job_id = job.get("id") if isinstance(job, dict) else None
        request_id = job_id if isinstance(job_id, str) and job_id else "unknown"
        document: ResolvedDocument | None = None
        try:
            if not isinstance(job, dict) or "input" not in job:
                raise WorkerError("INVALID_REQUEST", "Runpod job must contain input.")
            self._raise_if_cancelled(cancellation_event)
            self._progress(job, "validating request")
            request = validate_request(job["input"], self.settings)

            with tempfile.TemporaryDirectory(prefix="nuextract3-") as temp_dir:
                self._raise_if_cancelled(cancellation_event)
                self._progress(job, "loading document sources")
                document = self.media_factory().resolve(request, temp_dir)
                self._raise_if_cancelled(cancellation_event)
                self._progress(job, "running inference")
                inference = self.runtime.infer(
                    request,
                    document.items,
                    cancellation_event=cancellation_event,
                )
                self._raise_if_cancelled(cancellation_event)
                self._progress(job, "preparing result")
                response = self._success_response(
                    job=job,
                    request_id=request_id,
                    request=request,
                    document=document,
                    inference=inference,
                    started_at=started,
                    cancellation_event=cancellation_event,
                )
            self._raise_if_cancelled(cancellation_event)
            self._progress(job, "completed")
            return response
        except WorkerError as exc:
            logger.warning("job=%s code=%s", request_id, exc.code)
            return self._error_response(request_id, exc)
        except Exception as exc:
            logger.exception("job=%s unhandled_error=%s", request_id, type(exc).__name__)
            return self._error_response(
                request_id,
                WorkerError(
                    "INTERNAL_ERROR", "The worker could not complete the request.", retryable=True
                ),
            )
        finally:
            if document is not None:
                document.close()

    def _success_response(
        self,
        *,
        job: dict[str, Any],
        request_id: str,
        request: Request,
        document: ResolvedDocument,
        inference: InferenceResult,
        started_at: float,
        cancellation_event: threading.Event | None,
    ) -> dict[str, Any]:
        body, content_type = self._result_bytes(request, inference)
        digest = hashlib.sha256(body).hexdigest()

        if request.output.delivery == "s3" or (
            request.output.delivery == "auto" and len(body) > MAX_INLINE_RESULT_BYTES
        ):
            s3_config = resolve_s3_config(job)
            if s3_config is None:
                raise WorkerError(
                    "S3_CONFIG_REQUIRED",
                    "S3 output was requested but no S3 configuration is available.",
                )
            self._raise_if_cancelled(cancellation_event)
            result = self.uploader.upload(
                config=s3_config,
                job_id=request_id,
                mode=request.mode,
                body=body,
                content_type=content_type,
                sha256=digest,
                presign_ttl_seconds=request.output.presign_ttl_seconds,
            )
        else:
            if len(body) > MAX_INLINE_RESULT_BYTES:
                raise WorkerError(
                    "INLINE_RESULT_TOO_LARGE", "Inline result exceeds the size limit."
                )
            result = {
                "delivery": "inline",
                "content_type": content_type,
                "bytes": len(body),
                "sha256": digest,
            }
            if request.mode == "markdown":
                result["text"] = inference.value
            else:
                result["data"] = inference.value

        total_duration_ms = round((time.monotonic() - started_at) * 1_000)
        response: dict[str, Any] = {
            "schema_version": "1",
            "request_id": request_id,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "mode": request.mode,
            "result": result,
            "usage": {
                "input_tokens": inference.input_tokens,
                "generated_tokens": inference.generated_tokens,
                "images": len(document.images),
                "pdf_pages": document.pdf_pages,
                "source_bytes": document.source_bytes,
                "rendered_pixels": document.rendered_pixels,
                "finish_reason": inference.finish_reason,
                "inference_ms": inference.duration_ms,
                "preprocess_ms": inference.preprocess_ms,
                "generation_queue_ms": inference.generation_queue_ms,
                "generation_ms": inference.generation_ms,
                "postprocess_ms": inference.postprocess_ms,
                "generated_tokens_per_second": inference.generated_tokens_per_second,
                "gpu_peak_allocated_bytes": inference.gpu_peak_allocated_bytes,
                "gpu_peak_reserved_bytes": inference.gpu_peak_reserved_bytes,
                "duration_ms": total_duration_ms,
            },
        }
        if inference.reasoning is not None:
            response["reasoning"] = inference.reasoning
        return response

    @staticmethod
    def _result_bytes(request: Request, inference: InferenceResult) -> tuple[bytes, str]:
        try:
            if request.mode == "markdown":
                if not isinstance(inference.value, str):
                    raise WorkerError("INTERNAL_ERROR", "Markdown result has an invalid type.")
                return inference.value.encode("utf-8"), "text/markdown; charset=utf-8"
            return (
                json.dumps(
                    inference.value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8"),
                "application/json",
            )
        except UnicodeError as exc:
            raise WorkerError(
                "MODEL_OUTPUT_INVALID_TEXT",
                "Model output contains an invalid Unicode scalar value.",
            ) from exc

    @staticmethod
    def _error_response(request_id: str, error: WorkerError) -> dict[str, Any]:
        payload = json.dumps(
            {
                "schema_version": "1",
                "request_id": request_id,
                "code": error.code,
                "message": error.message,
                "retryable": error.retryable,
            },
            separators=(",", ":"),
        )
        response: dict[str, Any] = {"error": payload}
        if error.refresh_worker:
            response["refresh_worker"] = True
        return response

    def _progress(self, job: dict[str, Any], message: str) -> None:
        if self.progress is None:
            return
        try:
            self.progress(job, message)
        except Exception as exc:
            logger.warning("progress_update_failed=%s", type(exc).__name__)

    @staticmethod
    def _raise_if_cancelled(cancellation_event: threading.Event | None) -> None:
        if cancellation_event is not None and cancellation_event.is_set():
            raise WorkerError("JOB_CANCELLED", "Job was cancelled.")


def _two_concurrency(_: int) -> int:
    return 2

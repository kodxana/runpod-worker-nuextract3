"""Executable entry point required by Runpod Hub."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from nuextract_worker.config import Settings
from nuextract_worker.runtime import TransformersRuntime
from nuextract_worker.worker import Worker, _two_concurrency

_ACTIVE_WORKER: Worker | None = None


async def handler(job: dict[str, Any]) -> dict[str, Any]:
    if _ACTIVE_WORKER is None:
        raise RuntimeError("Worker runtime is not initialized")
    cancellation_event = threading.Event()
    worker_task = asyncio.create_task(asyncio.to_thread(_ACTIVE_WORKER, job, cancellation_event))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(worker_task)
            break
        except asyncio.CancelledError:
            if worker_task.cancelled():
                raise
            cancelled = True
            cancellation_event.set()
    if cancelled:
        raise asyncio.CancelledError
    return result


def main() -> None:
    global _ACTIVE_WORKER

    import runpod

    settings = Settings.from_env()
    runtime = TransformersRuntime(settings)
    runtime.load()
    _ACTIVE_WORKER = Worker(
        runtime,
        settings,
        progress=lambda job, message: runpod.serverless.progress_update(job, message),
    )
    runpod.serverless.start({"handler": handler, "concurrency_modifier": _two_concurrency})


if __name__ == "__main__":
    main()

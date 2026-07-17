"""Safe errors that may cross the Runpod job boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(eq=False)
class WorkerError(Exception):
    code: str
    message: str
    retryable: bool = False
    refresh_worker: bool = False

    def __str__(self) -> str:
        return self.message

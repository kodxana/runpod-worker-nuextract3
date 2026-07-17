"""Fixed model identity and bounded worker settings."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .model_manifest import MODEL_ID as MODEL_ID
from .model_manifest import MODEL_REVISION as MODEL_REVISION

TRANSFORMERS_VERSION = "5.5.4"
TORCH_VERSION = "2.10.0"

SUPPORTED_MEDIA_TYPES = frozenset({"application/pdf", "image/jpeg", "image/png", "image/webp"})

MAX_SOURCES = 8
MAX_TEXT_CHARS = 200_000
MAX_TEMPLATE_BYTES = 65_536
MAX_INSTRUCTIONS_CHARS = 16_384
MAX_TEMPLATE_DEPTH = 12
MAX_TEMPLATE_NODES = 1_000
MAX_ENUM_VALUES = 100
MAX_ENUM_VALUE_CHARS = 512

MAX_CONTEXT_TOKENS = 32_768
MAX_NEW_TOKENS = 4_096
DEFAULT_MAX_NEW_TOKENS = 1_024

MAX_BASE64_BYTES = 7_000_000
MAX_BASE64_CHARS = ((MAX_BASE64_BYTES + 2) // 3) * 4
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_DOWNLOAD_SECONDS = 120.0
MAX_RAW_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_DIMENSION = 8_192
MAX_IMAGE_PIXELS = 4_194_304
MAX_TOTAL_IMAGE_PIXELS = 24_000_000
MAX_PDF_PAGES = 6
PDF_DPI = 170

MAX_MODEL_OUTPUT_BYTES = 1_000_000
MAX_INLINE_RESULT_BYTES = 1_000_000
MIN_GPU_MEMORY_BYTES = 22 * 1024**3


def _host_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip().lower().rstrip(".") for item in value.split(",") if item.strip())


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_float(
    name: str,
    default: float | None,
    minimum: float,
    maximum: float,
) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


@dataclass(frozen=True)
class Settings:
    """Deployment settings that do not weaken fixed resource limits."""

    source_host_allowlist: tuple[str, ...] = ()
    s3_endpoint_host_allowlist: tuple[str, ...] = ()
    generation_timeout_seconds: int = 600
    download_connect_timeout_seconds: int = 5
    download_read_timeout_seconds: int = 30
    default_thinking: bool = False
    default_return_reasoning: bool = False
    default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    default_temperature: float | None = None
    default_top_p: float = 1.0
    default_top_k: int = 0
    default_seed: int = 0

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            source_host_allowlist=_host_list(os.environ.get("SOURCE_HOST_ALLOWLIST")),
            s3_endpoint_host_allowlist=_host_list(os.environ.get("S3_ENDPOINT_HOST_ALLOWLIST")),
            generation_timeout_seconds=_bounded_int("GENERATION_TIMEOUT_SECONDS", 600, 30, 900),
            download_connect_timeout_seconds=_bounded_int(
                "DOWNLOAD_CONNECT_TIMEOUT_SECONDS", 5, 1, 30
            ),
            download_read_timeout_seconds=_bounded_int("DOWNLOAD_READ_TIMEOUT_SECONDS", 30, 1, 120),
            default_thinking=_boolean("NUEXTRACT_DEFAULT_THINKING", False),
            default_return_reasoning=_boolean("NUEXTRACT_DEFAULT_RETURN_REASONING", False),
            default_max_new_tokens=_bounded_int(
                "NUEXTRACT_DEFAULT_MAX_NEW_TOKENS",
                DEFAULT_MAX_NEW_TOKENS,
                1,
                MAX_NEW_TOKENS,
            ),
            default_temperature=_bounded_float("NUEXTRACT_DEFAULT_TEMPERATURE", None, 0.0, 2.0),
            default_top_p=_bounded_float("NUEXTRACT_DEFAULT_TOP_P", 1.0, 0.0, 1.0),
            default_top_k=_bounded_int("NUEXTRACT_DEFAULT_TOP_K", 0, 0, 100),
            default_seed=_bounded_int("NUEXTRACT_DEFAULT_SEED", 0, 0, 2_147_483_647),
        )
        if settings.default_return_reasoning and not settings.default_thinking:
            raise RuntimeError(
                "NUEXTRACT_DEFAULT_RETURN_REASONING requires NUEXTRACT_DEFAULT_THINKING=true"
            )
        if settings.default_top_p <= 0:
            raise RuntimeError("NUEXTRACT_DEFAULT_TOP_P must be greater than 0")
        return settings


def require_model_id(value: str | None = None) -> str:
    """Reject an incompatible legacy model override when one is configured."""

    model_name = os.environ.get("MODEL_NAME")
    legacy_model_id = os.environ.get("MODEL_ID")
    if model_name and legacy_model_id and model_name != legacy_model_id:
        raise RuntimeError("MODEL_NAME and MODEL_ID must identify the same model")
    selected = value if value is not None else model_name or legacy_model_id or MODEL_ID
    if selected != MODEL_ID:
        raise RuntimeError(f"Configured model must be exactly {MODEL_ID}")
    return selected

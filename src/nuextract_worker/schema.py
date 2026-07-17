"""Strict request, template, and model-output validation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from .config import (
    DEFAULT_MAX_NEW_TOKENS,
    MAX_BASE64_CHARS,
    MAX_ENUM_VALUE_CHARS,
    MAX_ENUM_VALUES,
    MAX_INSTRUCTIONS_CHARS,
    MAX_MODEL_OUTPUT_BYTES,
    MAX_NEW_TOKENS,
    MAX_PDF_PAGES,
    MAX_SOURCES,
    MAX_TEMPLATE_BYTES,
    MAX_TEMPLATE_DEPTH,
    MAX_TEMPLATE_NODES,
    MAX_TEXT_CHARS,
    SUPPORTED_MEDIA_TYPES,
    Settings,
)
from .errors import WorkerError

NUEXTRACT_TYPES = frozenset(
    {
        "bic",
        "boolean",
        "country",
        "currency",
        "date",
        "date-time",
        "duration",
        "email-address",
        "iban",
        "integer",
        "language",
        "language-tag",
        "number",
        "phone-number",
        "region:AU",
        "region:BR",
        "region:CA",
        "region:DE",
        "region:ES",
        "region:FR",
        "region:GB",
        "region:IE",
        "region:IT",
        "region:JP",
        "region:KR",
        "region:MX",
        "region:PT",
        "region:US",
        "script",
        "string",
        "time",
        "unit-code",
        "url",
        "verbatim-string",
    }
)

PAGE_RANGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["start", "end"],
    "properties": {
        "start": {"type": "integer", "minimum": 1, "maximum": 100_000},
        "end": {"type": "integer", "minimum": 1, "maximum": 100_000},
    },
}

REQUEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "mode", "sources"],
    "properties": {
        "schema_version": {"const": "1"},
        "mode": {"enum": ["structured", "markdown", "template-generation"]},
        "sources": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_SOURCES,
            "items": {
                "oneOf": [
                    {"$ref": "#/$defs/text"},
                    {"$ref": "#/$defs/url"},
                    {"$ref": "#/$defs/base64"},
                ]
            },
        },
        "template": {"type": "object"},
        "instructions": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_INSTRUCTIONS_CHARS,
        },
        "generation": {"$ref": "#/$defs/generation"},
        "output": {"$ref": "#/$defs/output"},
    },
    "$defs": {
        "text": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "text"],
            "properties": {
                "type": {"const": "text"},
                "text": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
            },
        },
        "url": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "url", "media_type"],
            "properties": {
                "type": {"const": "url"},
                "url": {"type": "string", "minLength": 1, "maxLength": 4_096},
                "media_type": {"enum": sorted(SUPPORTED_MEDIA_TYPES)},
                "pages": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PDF_PAGES,
                    "items": PAGE_RANGE_SCHEMA,
                },
            },
        },
        "base64": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "data", "media_type"],
            "properties": {
                "type": {"const": "base64"},
                "data": {"type": "string", "minLength": 1, "maxLength": MAX_BASE64_CHARS},
                "media_type": {"enum": sorted(SUPPORTED_MEDIA_TYPES)},
                "pages": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PDF_PAGES,
                    "items": PAGE_RANGE_SCHEMA,
                },
            },
        },
        "generation": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "thinking": {"type": "boolean"},
                "return_reasoning": {"type": "boolean"},
                "max_new_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_NEW_TOKENS,
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 2,
                },
                "top_p": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                },
                "seed": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2_147_483_647,
                },
            },
        },
        "output": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "delivery": {"enum": ["auto", "inline", "s3"]},
                "presign_ttl_seconds": {
                    "type": "integer",
                    "oneOf": [{"const": 0}, {"minimum": 60, "maximum": 3_600}],
                },
            },
        },
    },
}

_VALIDATOR = Draft202012Validator(REQUEST_SCHEMA)
_FENCE_RE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n?(.*?)\r?\n?```[ \t]*\Z", re.DOTALL | re.IGNORECASE
)


@dataclass(frozen=True)
class GenerationOptions:
    thinking: bool = False
    return_reasoning: bool = False
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 0


@dataclass(frozen=True)
class OutputOptions:
    delivery: str = "auto"
    presign_ttl_seconds: int = 0


@dataclass(frozen=True)
class Request:
    mode: str
    sources: tuple[dict[str, Any], ...]
    template: dict[str, Any] | None
    instructions: str | None
    generation: GenerationOptions
    output: OutputOptions


def _request_error(message: str) -> WorkerError:
    return WorkerError("INVALID_REQUEST", message)


def validate_request(value: Any, settings: Settings | None = None) -> Request:
    errors = sorted(_VALIDATOR.iter_errors(value), key=lambda item: list(item.absolute_path))
    if errors:
        path = "/" + "/".join(str(part) for part in errors[0].absolute_path)
        raise _request_error(f"Request does not match the schema at {path or '/'}.")
    try:
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except UnicodeError as exc:
        raise _request_error("Request strings must contain valid Unicode scalar values.") from exc

    mode = value["mode"]
    template = value.get("template")
    instructions = value.get("instructions")

    if mode == "structured":
        if template is None:
            raise _request_error("structured mode requires template.")
        validate_template(template)
    elif template is not None:
        raise _request_error("template is only valid in structured mode.")

    if instructions is not None and mode != "structured":
        raise _request_error("instructions are only valid in structured mode.")

    defaults = settings or Settings()
    generation_value = value.get("generation", {})
    thinking = generation_value.get("thinking", defaults.default_thinking)
    default_temperature = defaults.default_temperature
    if default_temperature is None:
        default_temperature = 0.6 if thinking else 0.0
    temperature = generation_value.get("temperature", default_temperature)
    top_p = generation_value.get("top_p", defaults.default_top_p)
    if not math.isfinite(temperature) or not math.isfinite(top_p):
        raise _request_error("Generation values must be finite numbers.")
    generation = GenerationOptions(
        thinking=thinking,
        return_reasoning=generation_value.get(
            "return_reasoning", defaults.default_return_reasoning
        ),
        max_new_tokens=generation_value.get("max_new_tokens", defaults.default_max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=generation_value.get("top_k", defaults.default_top_k),
        seed=generation_value.get("seed", defaults.default_seed),
    )
    if generation.return_reasoning and not generation.thinking:
        raise _request_error("return_reasoning requires thinking=true.")
    if mode == "template-generation" and generation.thinking:
        raise _request_error("thinking is not supported in template-generation mode.")

    text_chars = 0
    for source in value["sources"]:
        if source["type"] == "text":
            text_chars += len(source["text"])
            continue
        has_pages = "pages" in source
        if source["media_type"] != "application/pdf" and has_pages:
            raise _request_error("pages is only valid for PDF sources.")
        if has_pages:
            _validate_page_ranges(source["pages"])
    if text_chars > MAX_TEXT_CHARS:
        raise _request_error(f"Combined text exceeds {MAX_TEXT_CHARS} characters.")

    output_value = value.get("output", {})
    output = OutputOptions(
        delivery=output_value.get("delivery", "auto"),
        presign_ttl_seconds=output_value.get("presign_ttl_seconds", 0),
    )

    return Request(
        mode=mode,
        sources=tuple(value["sources"]),
        template=template,
        instructions=instructions,
        generation=generation,
        output=output,
    )


def _validate_page_ranges(ranges: list[dict[str, int]]) -> None:
    previous_end = 0
    selected = 0
    for page_range in ranges:
        start = page_range["start"]
        end = page_range["end"]
        if end < start:
            raise _request_error("Each PDF page range must have end >= start.")
        if start <= previous_end:
            raise _request_error("PDF page ranges must be sorted and non-overlapping.")
        selected += end - start + 1
        previous_end = end
    if selected > MAX_PDF_PAGES:
        raise _request_error(f"At most {MAX_PDF_PAGES} PDF pages may be selected.")


def validate_template(template: Any) -> None:
    try:
        serialized = json.dumps(
            template, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        serialized_bytes = serialized.encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise _request_error("template must contain only JSON values.") from exc
    if len(serialized_bytes) > MAX_TEMPLATE_BYTES:
        raise _request_error(f"template exceeds {MAX_TEMPLATE_BYTES} bytes.")

    nodes = 0

    def visit(node: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_TEMPLATE_NODES:
            raise _request_error(f"template exceeds {MAX_TEMPLATE_NODES} nodes.")
        if depth > MAX_TEMPLATE_DEPTH:
            raise _request_error(f"template exceeds depth {MAX_TEMPLATE_DEPTH}.")

        if isinstance(node, str):
            if node not in NUEXTRACT_TYPES:
                raise _request_error("template string leaves must use a supported NuExtract type.")
            return
        if isinstance(node, dict):
            if not node:
                raise _request_error("template objects must not be empty.")
            for key, child in node.items():
                if not key or len(key) > 512:
                    raise _request_error("template keys must contain 1 to 512 characters.")
                visit(child, depth + 1)
            return
        if isinstance(node, list):
            if not node:
                raise _request_error("template arrays and enums must not be empty.")
            if len(node) > 1:
                _validate_enum(node)
                return
            child = node[0]
            if isinstance(child, list):
                _validate_enum(child)
                return
            visit(child, depth + 1)
            return
        raise _request_error("template leaves must be types, arrays, enums, or objects.")

    visit(template, 1)


def _validate_enum(values: list[Any]) -> None:
    if not values or len(values) > MAX_ENUM_VALUES:
        raise _request_error(f"enums must contain 1 to {MAX_ENUM_VALUES} values.")
    if not all(
        isinstance(value, str) and 0 < len(value) <= MAX_ENUM_VALUE_CHARS for value in values
    ):
        raise _request_error("enum values must be non-empty strings of bounded length.")
    if len(set(values)) != len(values):
        raise _request_error("enum values must be unique.")


def split_reasoning(text: str, thinking: bool) -> tuple[str | None, str]:
    value = text.strip()
    if not thinking:
        return None, value
    if "</think>" not in value:
        raise WorkerError(
            "MODEL_OUTPUT_REASONING_INVALID",
            "Model reasoning did not contain a closing delimiter.",
        )
    before, _, after = value.partition("</think>")
    reasoning = before.removeprefix("<think>").strip()
    return reasoning or None, after.strip()


def parse_json_output(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        raw_bytes = raw.encode("utf-8")
    except UnicodeError as exc:
        raise WorkerError(
            "MODEL_OUTPUT_INVALID_JSON", "Model output is not valid UTF-8 text."
        ) from exc
    if len(raw_bytes) > MAX_MODEL_OUTPUT_BYTES:
        raise WorkerError("MODEL_OUTPUT_TOO_LARGE", "Model output exceeds the JSON size limit.")
    fence = _FENCE_RE.fullmatch(raw)
    if fence:
        raw = fence.group(1).strip()

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WorkerError(
                    "MODEL_OUTPUT_INVALID_JSON", "Model output contains duplicate JSON keys."
                )
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise WorkerError(
            "MODEL_OUTPUT_INVALID_JSON", "Model output contains a non-standard JSON number."
        )

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise WorkerError(
                "MODEL_OUTPUT_INVALID_JSON",
                "Model output contains a non-finite JSON number.",
            )
        return parsed

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except WorkerError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise WorkerError("MODEL_OUTPUT_INVALID_JSON", "Model output is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise WorkerError("MODEL_OUTPUT_INVALID_JSON", "Model output must be a JSON object.")
    try:
        json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except UnicodeError as exc:
        raise WorkerError(
            "MODEL_OUTPUT_INVALID_JSON",
            "Model output contains an invalid Unicode scalar value.",
        ) from exc
    return parsed


def validate_structured_output(value: Any, template: Any) -> None:
    """Validate JSON shape and primitive types against a NuExtract template."""

    def visit(output: Any, spec: Any, *, allow_null: bool = True) -> None:
        if output is None:
            if allow_null:
                return
            raise WorkerError(
                "MODEL_OUTPUT_SCHEMA_MISMATCH",
                "Model output arrays must not contain null elements.",
            )
        if isinstance(spec, dict):
            if not isinstance(output, dict) or set(output) != set(spec):
                raise WorkerError(
                    "MODEL_OUTPUT_SCHEMA_MISMATCH", "Model output does not match the template keys."
                )
            for key, child in spec.items():
                visit(output[key], child)
            return
        if isinstance(spec, str):
            valid = _primitive_matches(output, spec)
            if not valid:
                raise WorkerError(
                    "MODEL_OUTPUT_SCHEMA_MISMATCH",
                    "Model output contains an invalid primitive type.",
                )
            return
        if isinstance(spec, list) and len(spec) > 1:
            if output not in spec:
                raise WorkerError(
                    "MODEL_OUTPUT_SCHEMA_MISMATCH", "Model output contains an invalid enum value."
                )
            return
        if isinstance(spec, list) and len(spec) == 1:
            child = spec[0]
            if not isinstance(output, list):
                raise WorkerError(
                    "MODEL_OUTPUT_SCHEMA_MISMATCH", "Model output contains an invalid array."
                )
            if isinstance(child, list):
                if any(item not in child for item in output):
                    raise WorkerError(
                        "MODEL_OUTPUT_SCHEMA_MISMATCH",
                        "Model output contains an invalid multi-enum value.",
                    )
                return
            for item in output:
                visit(item, child, allow_null=False)
            return
        raise WorkerError(
            "MODEL_OUTPUT_SCHEMA_MISMATCH", "Model output does not match the template."
        )

    visit(value, template)


def _primitive_matches(value: Any, kind: str) -> bool:
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and math.isfinite(value)
        )
    return isinstance(value, str)

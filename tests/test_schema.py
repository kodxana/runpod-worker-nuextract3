from __future__ import annotations

import pytest

from nuextract_worker.config import Settings
from nuextract_worker.errors import WorkerError
from nuextract_worker.schema import (
    REQUEST_SCHEMA,
    parse_json_output,
    split_reasoning,
    validate_request,
    validate_structured_output,
    validate_template,
)


def _request(**overrides):
    value = {
        "schema_version": "1",
        "mode": "structured",
        "sources": [{"type": "text", "text": "An invoice"}],
        "template": {"total": "number"},
    }
    value.update(overrides)
    return value


def _assert_error(code: str, function, *args) -> WorkerError:
    with pytest.raises(WorkerError) as caught:
        function(*args)
    assert caught.value.code == code
    return caught.value


def test_valid_structured_request_is_normalized() -> None:
    request = validate_request(
        _request(
            template={
                "name": "verbatim-string",
                "kind": ["invoice", "receipt"],
                "tags": [["urgent", "paid"]],
                "items": [{"quantity": "integer"}],
            },
            instructions="Return the printed value.",
            generation={
                "thinking": True,
                "return_reasoning": True,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 20,
                "seed": 42,
            },
            output={"delivery": "inline", "presign_ttl_seconds": 0},
        )
    )

    assert request.mode == "structured"
    assert request.generation.thinking is True
    assert request.generation.return_reasoning is True
    assert request.generation.max_new_tokens == 256
    assert request.generation.temperature == 0.7
    assert request.generation.top_p == 0.9
    assert request.generation.top_k == 20
    assert request.generation.seed == 42
    assert request.output.delivery == "inline"


def test_generation_defaults_preserve_greedy_and_thinking_modes() -> None:
    greedy = validate_request(_request())
    assert greedy.generation.temperature == 0.0
    assert greedy.generation.top_p == 1.0
    assert greedy.generation.top_k == 0
    assert greedy.generation.seed == 0

    thinking = validate_request(_request(generation={"thinking": True}))
    assert thinking.generation.temperature == 0.6


def test_job_schema_exposes_bounded_generation_controls() -> None:
    generation = REQUEST_SCHEMA["$defs"]["generation"]
    assert generation["additionalProperties"] is False
    assert set(generation["properties"]) == {
        "thinking",
        "return_reasoning",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
    }


def test_job_generation_values_override_endpoint_defaults() -> None:
    settings = Settings(
        default_thinking=True,
        default_return_reasoning=True,
        default_max_new_tokens=2048,
        default_temperature=0.7,
        default_top_p=0.85,
        default_top_k=40,
        default_seed=1234,
    )
    endpoint_request = validate_request(_request(), settings)
    assert endpoint_request.generation.thinking is True
    assert endpoint_request.generation.return_reasoning is True
    assert endpoint_request.generation.max_new_tokens == 2048
    assert endpoint_request.generation.temperature == 0.7
    assert endpoint_request.generation.top_p == 0.85
    assert endpoint_request.generation.top_k == 40
    assert endpoint_request.generation.seed == 1234

    job_request = validate_request(
        _request(
            generation={
                "thinking": False,
                "return_reasoning": False,
                "max_new_tokens": 64,
                "temperature": 0,
                "top_p": 1,
                "top_k": 0,
                "seed": 9,
            }
        ),
        settings,
    )
    assert job_request.generation.thinking is False
    assert job_request.generation.return_reasoning is False
    assert job_request.generation.max_new_tokens == 64
    assert job_request.generation.temperature == 0
    assert job_request.generation.top_p == 1
    assert job_request.generation.top_k == 0
    assert job_request.generation.seed == 9


@pytest.mark.parametrize(
    "value",
    [
        _request(extra=True),
        _request(schema_version="2"),
        _request(sources=[]),
        _request(sources=[{"type": "text", "text": "x", "url": "https://example.com"}]),
        _request(generation={"unknown": True}),
        _request(output={"delivery": "filesystem"}),
    ],
)
def test_request_schema_rejects_unknown_or_invalid_values(value) -> None:
    _assert_error("INVALID_REQUEST", validate_request, value)


@pytest.mark.parametrize(
    "generation",
    [
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"temperature": float("nan")},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_p": float("inf")},
        {"top_k": -1},
        {"top_k": 101},
        {"seed": -1},
        {"seed": 2_147_483_648},
    ],
)
def test_generation_controls_are_bounded_and_finite(generation) -> None:
    _assert_error("INVALID_REQUEST", validate_request, _request(generation=generation))


def test_request_rejects_invalid_unicode_scalars() -> None:
    _assert_error(
        "INVALID_REQUEST",
        validate_request,
        _request(sources=[{"type": "text", "text": "invalid \ud800"}]),
    )


@pytest.mark.parametrize(
    "value",
    [
        _request(mode="structured", template=None),
        _request(mode="markdown", template={"x": "string"}),
        _request(mode="markdown", template=None, instructions="not allowed"),
        _request(mode="template-generation", template=None, generation={"thinking": True}),
        _request(
            generation={"thinking": False, "return_reasoning": True},
        ),
    ],
)
def test_mode_specific_rules(value) -> None:
    if value.get("template") is None:
        value.pop("template", None)
    _assert_error("INVALID_REQUEST", validate_request, value)


def test_pdf_ranges_must_be_sorted_non_overlapping_and_bounded() -> None:
    base = {
        "schema_version": "1",
        "mode": "markdown",
        "sources": [
            {
                "type": "url",
                "url": "https://example.com/a.pdf",
                "media_type": "application/pdf",
                "pages": [{"start": 3, "end": 4}, {"start": 2, "end": 2}],
            }
        ],
    }
    _assert_error("INVALID_REQUEST", validate_request, base)

    base["sources"][0]["pages"] = [{"start": 1, "end": 7}]
    _assert_error("INVALID_REQUEST", validate_request, base)

    base["sources"][0]["pages"] = [{"start": 1, "end": 2}]
    request = validate_request(base)
    assert request.sources[0]["pages"] == [{"start": 1, "end": 2}]


def test_pages_are_only_valid_for_pdfs() -> None:
    value = {
        "schema_version": "1",
        "mode": "markdown",
        "sources": [
            {
                "type": "url",
                "url": "https://example.com/a.png",
                "media_type": "image/png",
                "pages": [{"start": 1, "end": 1}],
            }
        ],
    }
    _assert_error("INVALID_REQUEST", validate_request, value)


@pytest.mark.parametrize(
    "template",
    [
        {},
        {"field": "unsupported"},
        {"field": []},
        {"field": ["same", "same"]},
        {"field": [1, 2]},
        {"field": None},
    ],
)
def test_invalid_templates_are_rejected(template) -> None:
    _assert_error("INVALID_REQUEST", validate_template, template)


def test_all_template_constructors_are_accepted() -> None:
    validate_template(
        {
            "scalar": "date-time",
            "array": ["string"],
            "objects": [{"value": "number"}],
            "enum": ["one", "two"],
            "multi_enum": [["A", "B"]],
        }
    )


def test_json_output_accepts_fence_and_rejects_unsafe_forms() -> None:
    assert parse_json_output('```json\n{"ok":true}\n```') == {"ok": True}
    _assert_error("MODEL_OUTPUT_INVALID_JSON", parse_json_output, '{"x":1,"x":2}')
    _assert_error("MODEL_OUTPUT_INVALID_JSON", parse_json_output, '{"x":NaN}')
    _assert_error("MODEL_OUTPUT_INVALID_JSON", parse_json_output, '{"x":1e999}')
    _assert_error("MODEL_OUTPUT_INVALID_JSON", parse_json_output, "[]")
    _assert_error("MODEL_OUTPUT_INVALID_JSON", parse_json_output, 'prefix {"x": 1}')
    _assert_error("MODEL_OUTPUT_INVALID_JSON", parse_json_output, '{"x":' + "9" * 5_000 + "}")
    _assert_error("MODEL_OUTPUT_INVALID_JSON", parse_json_output, '{"x":"\\ud800"}')


def test_reasoning_is_split_only_at_the_model_delimiter() -> None:
    assert split_reasoning('analysis\n</think>\n{"x":1}', True) == (
        "analysis",
        '{"x":1}',
    )
    assert split_reasoning("# Markdown", False) == (None, "# Markdown")
    _assert_error("MODEL_OUTPUT_REASONING_INVALID", split_reasoning, "unfinished", True)


def test_structured_output_matches_template_shape_and_types() -> None:
    template = {
        "total": "number",
        "paid": "boolean",
        "kind": ["invoice", "receipt"],
        "tags": [["A", "B"]],
        "items": [{"quantity": "integer", "name": "string"}],
    }
    output = {
        "total": 12.5,
        "paid": True,
        "kind": "invoice",
        "tags": ["A", "B"],
        "items": [{"quantity": 2, "name": "pen"}],
    }
    validate_structured_output(output, template)
    validate_structured_output({key: None for key in template}, template)

    bad = dict(output, paid=1)
    _assert_error("MODEL_OUTPUT_SCHEMA_MISMATCH", validate_structured_output, bad, template)
    bad = dict(output, kind="other")
    _assert_error("MODEL_OUTPUT_SCHEMA_MISMATCH", validate_structured_output, bad, template)
    bad = dict(output)
    bad.pop("total")
    _assert_error("MODEL_OUTPUT_SCHEMA_MISMATCH", validate_structured_output, bad, template)
    bad = dict(output, items=[None])
    _assert_error("MODEL_OUTPUT_SCHEMA_MISMATCH", validate_structured_output, bad, template)


def test_number_validation_does_not_overflow_on_large_json_integers() -> None:
    validate_structured_output({"number": 10**10_000}, {"number": "number"})

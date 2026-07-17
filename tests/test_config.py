from __future__ import annotations

import pytest

from nuextract_worker.config import Settings


def test_endpoint_environment_overrides_generation_defaults(monkeypatch) -> None:
    values = {
        "NUEXTRACT_DEFAULT_THINKING": "true",
        "NUEXTRACT_DEFAULT_RETURN_REASONING": "1",
        "NUEXTRACT_DEFAULT_MAX_NEW_TOKENS": "2048",
        "NUEXTRACT_DEFAULT_TEMPERATURE": "0.7",
        "NUEXTRACT_DEFAULT_TOP_P": "0.85",
        "NUEXTRACT_DEFAULT_TOP_K": "40",
        "NUEXTRACT_DEFAULT_SEED": "1234",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings.from_env()
    assert settings.default_thinking is True
    assert settings.default_return_reasoning is True
    assert settings.default_max_new_tokens == 2048
    assert settings.default_temperature == 0.7
    assert settings.default_top_p == 0.85
    assert settings.default_top_k == 40
    assert settings.default_seed == 1234


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NUEXTRACT_DEFAULT_THINKING", "sometimes"),
        ("NUEXTRACT_DEFAULT_MAX_NEW_TOKENS", "0"),
        ("NUEXTRACT_DEFAULT_TEMPERATURE", "nan"),
        ("NUEXTRACT_DEFAULT_TEMPERATURE", "2.1"),
        ("NUEXTRACT_DEFAULT_TOP_P", "0"),
        ("NUEXTRACT_DEFAULT_TOP_K", "101"),
        ("NUEXTRACT_DEFAULT_SEED", "-1"),
    ],
)
def test_endpoint_generation_defaults_are_strictly_bounded(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        Settings.from_env()


def test_reasoning_default_requires_thinking_default(monkeypatch) -> None:
    monkeypatch.setenv("NUEXTRACT_DEFAULT_RETURN_REASONING", "true")
    with pytest.raises(RuntimeError, match="requires NUEXTRACT_DEFAULT_THINKING"):
        Settings.from_env()

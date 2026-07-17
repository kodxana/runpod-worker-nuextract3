from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from nuextract_worker.model_manifest import (
    MODEL_FILES,
    MODEL_ID,
    MODEL_LICENSE_FILE,
    MODEL_REVISION,
    MODEL_WEIGHT_FILES,
)

ROOT = Path(__file__).parents[1]


def _load_downloader():
    spec = importlib.util.spec_from_file_location("download_model", ROOT / "download_model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load download_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


downloader = _load_downloader()


def _payloads() -> dict[str, bytes]:
    payloads = {name: f"fixture:{name}".encode() for name in (*MODEL_FILES, MODEL_LICENSE_FILE)}
    indexed_size = sum(len(payloads[name]) for name in MODEL_WEIGHT_FILES)
    payloads["model.safetensors.index.json"] = json.dumps(
        {
            "metadata": {"total_size": indexed_size},
            "weight_map": {
                "model.weight": "model.safetensors",
                "mtp.weight": "model_mtp.safetensors",
            },
        }
    ).encode()
    return payloads


def _use_fixture_hashes(monkeypatch, payloads: dict[str, bytes]) -> None:
    monkeypatch.setattr(
        downloader,
        "MODEL_WEIGHT_BYTES",
        sum(len(payloads[name]) for name in MODEL_WEIGHT_FILES),
    )
    monkeypatch.setattr(
        downloader,
        "VERIFIED_LFS_FILES",
        tuple(
            (name, len(payloads[name]), hashlib.sha256(payloads[name]).hexdigest())
            for name in ("model.safetensors", "model_mtp.safetensors", "tokenizer.json")
        ),
    )


def test_download_model_uses_exact_revision_and_removes_hub_metadata(tmp_path, monkeypatch) -> None:
    payloads = _payloads()
    _use_fixture_hashes(monkeypatch, payloads)
    calls = []

    def snapshot_download(**kwargs) -> str:
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        for name, payload in payloads.items():
            (destination / name).write_bytes(payload)
        (destination / ".cache/huggingface").mkdir(parents=True)
        return str(destination)

    destination = tmp_path / "model"
    assert (
        downloader.download_model(destination, snapshot_download=snapshot_download) == destination
    )
    assert calls == [
        {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_dir": str(destination),
            "allow_patterns": [*MODEL_FILES, MODEL_LICENSE_FILE],
        }
    ]
    assert not (destination / ".cache").exists()


def test_model_verification_rejects_an_altered_artifact(tmp_path, monkeypatch) -> None:
    payloads = _payloads()
    _use_fixture_hashes(monkeypatch, payloads)
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
    (tmp_path / "model_mtp.safetensors").write_bytes(b"altered-payload")

    with pytest.raises(RuntimeError, match="model_mtp.safetensors size"):
        downloader.verify_snapshot(tmp_path)

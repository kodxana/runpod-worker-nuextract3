from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from nuextract_worker.config import MODEL_ID, MODEL_REVISION
from nuextract_worker.model_manifest import (
    BAKED_MODEL_PATH,
    MODEL_FILES,
    MODEL_WEIGHT_BYTES,
    MODEL_WEIGHT_FILES,
    VERIFIED_LFS_FILES,
)

ROOT = Path(__file__).parents[1]


def test_hub_metadata_uses_the_baked_model_without_a_selector() -> None:
    hub = json.loads((ROOT / ".runpod/hub.json").read_text())
    assert hub["type"] == "serverless"
    assert hub["category"] == "language"
    assert "iconUrl" not in hub
    assert hub["config"]["containerDiskInGb"] == 50
    assert hub["config"]["gpuCount"] == 1
    assert set(hub["config"]["allowedCudaVersions"]) == {"12.8", "12.9", "13.0"}
    assert set(hub["config"]["gpuIds"].split(",")) == {
        "AMPERE_24",
        "ADA_24",
        "AMPERE_48",
        "ADA_48_PRO",
        "AMPERE_80",
        "ADA_80_PRO",
        "HOPPER_141",
    }

    assert "env" not in hub["config"]


def test_hub_smoke_test_uses_a_supported_gpu_without_model_environment() -> None:
    tests = json.loads((ROOT / ".runpod/tests.json").read_text())
    assert tests["tests"][0]["input"]["schema_version"] == "1"
    assert tests["tests"][0]["timeout"] == 900_000
    assert tests["config"]["gpuTypeId"] == "NVIDIA GeForce RTX 4090"
    assert tests["config"]["gpuCount"] == 1
    assert "env" not in tests["config"]


def test_docker_and_native_dependency_pins_are_immutable() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    requirements = (ROOT / "requirements.txt").read_text()

    assert (
        "FROM docker.io/pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:"
        "b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
    ) in dockerfile
    assert 'CMD ["python", "-u", "handler.py"]' in dockerfile
    assert "EXPOSE" not in dockerfile
    assert "snapshot_download" not in dockerfile
    assert "python /app/download_model.py" in dockerfile
    assert "HF_HOME=/tmp/huggingface" in dockerfile
    assert "rm -rf /tmp/huggingface" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "/runpod-volume" not in dockerfile
    assert "torch==2.10.0" in requirements
    assert "torchvision==0.25.0" in requirements
    assert "transformers==5.5.4" in requirements
    assert "runpod==1.10.1" in requirements
    assert "flash-linear-attention[cuda]==0.5.1" in requirements
    assert "sha256=c16c1c48d4fa63415cc797e02d69f97248c57c04627d99e394d5bb0ef266e288" in requirements
    requirement_lines = [
        line for line in requirements.splitlines() if line and not line.startswith("--")
    ]
    assert all("==" in line or " @ " in line for line in requirement_lines)


def test_model_identity_and_revision_are_not_floating() -> None:
    assert MODEL_ID == "numind/NuExtract3"
    assert MODEL_REVISION == "2e9fca82ee641e6bb6e1f5d905241e994be27a07"
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert MODEL_REVISION in notices
    assert "pypdfium2 5.12.0" in notices
    assert "PyMuPDF" not in notices


def test_baked_model_manifest_contains_every_indexed_weight_shard() -> None:
    assert BAKED_MODEL_PATH == "/opt/models/nuextract3"
    assert "model.safetensors" in MODEL_FILES
    assert "model_mtp.safetensors" in MODEL_FILES
    weights = {name: (size, digest) for name, size, digest in VERIFIED_LFS_FILES}
    assert weights["model.safetensors"] == (
        9_078_620_504,
        "aca0a9d61da5df4fa4b1475b68c0a7205e5f8f5f20beb5055fde0622991f9ed7",
    )
    assert weights["model_mtp.safetensors"] == (
        241_200_704,
        "7f993d7b896c6d3c72ee66fd446b28bcf316d5f5ce4a0427c0442dfe461cbe1b",
    )
    assert MODEL_WEIGHT_FILES == ("model.safetensors", "model_mtp.safetensors")
    assert sum(weights[name][0] for name in MODEL_WEIGHT_FILES) == MODEL_WEIGHT_BYTES


def test_importing_handler_does_not_import_gpu_or_network_stacks() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    script = """
import inspect
import sys
import handler
assert inspect.iscoroutinefunction(handler.handler)
for name in ('torch', 'transformers', 'runpod', 'PIL', 'pypdfium2', 'httpx', 'boto3'):
    assert name not in sys.modules, name
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def test_worker_startup_loads_before_queue_and_pipelines_two_jobs() -> None:
    source = (ROOT / "handler.py").read_text()
    assert "async def handler(" in source
    assert "asyncio.to_thread(_ACTIVE_WORKER, job, cancellation_event)" in source
    assert "await asyncio.shield(worker_task)" in source
    assert source.index("runtime.load()") < source.index("runpod.serverless.start(")
    assert '"handler": handler' in source
    assert '"concurrency_modifier": _two_concurrency' in source


def test_handler_cancellation_waits_for_thread_cleanup() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    script = """
import asyncio
import threading
import time
import handler

started = threading.Event()
finished = threading.Event()

class ActiveWorker:
    def __call__(self, job, cancellation_event):
        started.set()
        assert cancellation_event.wait(2)
        time.sleep(0.02)
        finished.set()
        return {}

async def check():
    handler._ACTIVE_WORKER = ActiveWorker()
    task = asyncio.create_task(handler.handler({"id": "cancelled"}))
    while not started.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=2)
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("handler did not preserve cancellation")
    assert finished.is_set()

asyncio.run(check())
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        timeout=10,
    )

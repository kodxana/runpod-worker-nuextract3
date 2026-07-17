"""Download and verify the model files baked into the worker image."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

from nuextract_worker.model_manifest import (
    BAKED_MODEL_PATH,
    MODEL_FILES,
    MODEL_ID,
    MODEL_LICENSE_FILE,
    MODEL_REVISION,
    MODEL_WEIGHT_BYTES,
    MODEL_WEIGHT_FILES,
    VERIFIED_LFS_FILES,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: Path) -> None:
    required_files = (*MODEL_FILES, MODEL_LICENSE_FILE)
    missing = [name for name in required_files if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Pinned snapshot is missing: {', '.join(missing)}")

    linked = [name for name in required_files if (path / name).is_symlink()]
    if linked:
        raise RuntimeError(f"Baked model files must not be symbolic links: {', '.join(linked)}")

    try:
        index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        indexed_size = index["metadata"]["total_size"]
        indexed_files = set(index["weight_map"].values())
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("Pinned model weight index is invalid") from exc
    if indexed_size != MODEL_WEIGHT_BYTES:
        raise RuntimeError(
            f"Unexpected indexed weight size: expected {MODEL_WEIGHT_BYTES}, got {indexed_size}"
        )
    if indexed_files != set(MODEL_WEIGHT_FILES):
        raise RuntimeError(
            f"Unexpected indexed weight files: expected {MODEL_WEIGHT_FILES}, got {indexed_files}"
        )

    for filename, expected_size, expected_hash in VERIFIED_LFS_FILES:
        artifact = path / filename
        actual_size = artifact.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Unexpected {filename} size: expected {expected_size}, got {actual_size}"
            )
        actual_hash = sha256(artifact)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Unexpected {filename} SHA-256: expected {expected_hash}, got {actual_hash}"
            )


def download_model(
    destination: Path = Path(BAKED_MODEL_PATH),
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> Path:
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hub_snapshot_download

        snapshot_download = hub_snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(destination),
        allow_patterns=[*MODEL_FILES, MODEL_LICENSE_FILE],
    )
    verify_snapshot(destination)
    shutil.rmtree(destination / ".cache", ignore_errors=True)
    return destination


def main() -> None:
    destination = download_model()
    print(f"Verified {MODEL_ID}@{MODEL_REVISION} in {destination}")


if __name__ == "__main__":
    main()

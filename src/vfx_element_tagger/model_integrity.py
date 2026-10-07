"""Verify the immutable, locally tested inference snapshots before loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


LOCK_PATH = Path(__file__).parent / "data" / "model-lock.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(key: str, directory: Path) -> dict:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    spec = lock["models"].get(key)
    if not spec:
        raise ValueError(f"No frozen snapshot for {key}; experimental models are not v1-verified")
    for name, expected in spec["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Unsafe path in model lock")
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError(f"Missing or wrong-size model file: {key}/{name}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Model checksum mismatch: {key}/{name}")
    actual_weights = {str(path.relative_to(directory)) for path in directory.rglob("*.safetensors")}
    expected_weights = {name for name in spec["files"] if name.endswith(".safetensors")}
    if actual_weights != expected_weights:
        raise ValueError(f"Unexpected model weight files for {key}")
    return spec

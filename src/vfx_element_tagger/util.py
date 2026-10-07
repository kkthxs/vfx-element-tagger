from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for part in parts:
        hasher.update(str(part).encode("utf-8", "ignore"))
        hasher.update(b"\0")
    return f"{prefix}_{hasher.hexdigest()[:length]}"


def file_digest(path: Path, sample_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    hasher = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        if size <= sample_size * 3:
            while chunk := handle.read(1024 * 256):
                hasher.update(chunk)
        else:
            offsets = [0, max(0, size // 2 - sample_size // 2), max(0, size - sample_size)]
            for offset in offsets:
                handle.seek(offset)
                hasher.update(handle.read(sample_size))
    hasher.update(str(size).encode("ascii"))
    return hasher.hexdigest()


def parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def run_json(command: list[str], timeout: int = 30) -> tuple[dict[str, Any] | None, str | None]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return None, completed.stderr.strip() or completed.stdout.strip()
    try:
        return json.loads(completed.stdout), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def run_command(command: list[str], timeout: int = 120) -> tuple[bool, str | None]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if completed.returncode != 0:
        return False, completed.stderr.strip() or completed.stdout.strip()
    return True, None


def tool_exists(name: str) -> bool:
    return shutil.which(name) is not None


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_root(name: str, preserve_identifiers: bool = False) -> str:
    stem = Path(name).stem.lower()
    if not preserve_identifiers:
        stem = re.sub(r"(?<![a-z])\d{2,8}$", "", stem)
    stem = re.sub(r"[\._-]?(master|proxy|preview|prores|h264|h265|exr|dpx)$", "", stem)
    stem = re.sub(r"[\._-]?(480p|720p|1080p|2k|4k|6k|8k|uhd|hd|sd)$", "", stem)
    if not preserve_identifiers:
        stem = re.sub(r"[\._-]+v\d{1,4}$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or Path(name).stem.lower()


def now_ms() -> int:
    return int(time.time() * 1000)


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(number)} {unit}"
            return f"{number:.1f} {unit}"
        number /= 1024


def iter_files(root: Path, excluded_dirs: Iterable[Path]) -> Iterable[Path]:
    excluded_resolved = {path.resolve() for path in excluded_dirs}
    for current_root, dirs, files in os.walk(root):
        current = Path(current_root).resolve()
        dirs[:] = [
            item
            for item in dirs
            if (current / item).resolve() not in excluded_resolved
            and item not in {".git", "__pycache__", ".venv", ".uv"}
        ]
        for filename in files:
            yield Path(current_root) / filename

"""Relocatable workstation paths, with explicit environment overrides."""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_settings() -> dict:
    path = Path(os.environ.get("VFX_TAGGER_CONFIG", PROJECT_ROOT / ".vfx-tagger.json"))
    return json.loads(path.read_text()) if path.is_file() else {}


def configured_path(key: str, environment: str, fallback: Path | Callable[[], Path]) -> Path:
    value = os.environ.get(environment) or project_settings().get(key)
    if not value:
        value = fallback() if callable(fallback) else fallback
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def models_dir() -> Path:
    return configured_path("models_dir", "VFX_TAGGER_MODELS_DIR",
                           lambda: Path.home() / ".cache/vfx-element-tagger/models")


def library_path() -> Path:
    return configured_path("library", "VFX_TAGGER_LIBRARY",
                           PROJECT_ROOT / ".cache/vfx-element-tagger/library.json")


def cache_dir() -> Path:
    return configured_path("cache_dir", "VFX_TAGGER_CACHE_DIR",
                           PROJECT_ROOT / ".cache/vfx-element-tagger")

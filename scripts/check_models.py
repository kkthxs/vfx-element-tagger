#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vfx_element_tagger.settings import models_dir
from vfx_element_tagger.model_integrity import verify_snapshot
DEFAULT_MANIFEST = models_dir() / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local analysis model snapshots and MLX loaders.")
    parser.add_argument("--models-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-only", action="store_true", help="Check SHA-256 files without importing MLX")
    args = parser.parse_args()

    if not args.verify_only:
        from mlx_vlm import load  # type: ignore[import-not-found]

    payload = json.loads(args.models_manifest.expanduser().read_text(encoding="utf-8"))
    failed = 0
    for key, entry in payload.get("models", {}).items():
        if entry.get("role") not in {"primary", "challenger"}:
            continue
        path = Path(entry["path"])
        missing = [name for name in ("config.json", "tokenizer_config.json") if not (path / name).exists()]
        safetensors = list(path.glob("*.safetensors"))
        if missing or not safetensors:
            print(f"FAIL {key}: missing={missing}, safetensors={len(safetensors)}")
            failed += 1
            continue
        try:
            verified = verify_snapshot(key, path)
            if args.verify_only:
                print(f"OK {key}: revision={verified['revision']}, SHA-256 verified")
                continue
            model, processor = load(str(path), lazy=True)
            model_type = getattr(getattr(model, "config", None), "model_type", type(model).__name__)
            print(
                f"OK {key}: model_type={model_type}, weights={len(safetensors)}, "
                f"processor={type(processor).__name__}"
            )
            del model, processor
            gc.collect()
        except Exception as exc:
            print(f"FAIL {key}: {type(exc).__name__}: {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

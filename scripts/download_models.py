#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vfx_element_tagger.settings import models_dir as configured_models_dir
from vfx_element_tagger.model_integrity import LOCK_PATH, verify_snapshot

MODELS = {
    "analysis_primary": {
        "repo_id": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "backend": "mlx_video",
        "role": "primary",
    },
    "video_embedding": {
        "repo_id": "mlx-community/Qwen3-VL-Embedding-2B-6bit",
        "backend": "qwen_vl_embedding",
        "role": "embedding",
    },
    "challenger_marlin": {
        "repo_id": "NemoStation/Marlin-2B-MLX-8bit",
        "backend": "mlx_video",
        "role": "challenger",
    },
    "challenger_minicpm": {
        "repo_id": "mlx-community/MiniCPM-o-4_5-4bit",
        "backend": "mlx_video",
        "role": "challenger",
    },
}

DEFAULT_MODELS = ["analysis_primary"]
CHALLENGERS = ["challenger_minicpm", "challenger_marlin"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights for VFX Element Tagger.")
    parser.add_argument(
        "--models-dir",
        default=str(configured_models_dir()),
        help="Directory where model snapshots should be stored (outside synced project folders).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(MODELS),
        default=None,
        help="Model keys to download.",
    )
    parser.add_argument(
        "--with-challengers",
        action="store_true",
        help="Also download both lazy challenger models (substantially more disk space).",
    )
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    selected = list(args.only or DEFAULT_MODELS)
    frozen = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["models"]
    if any(key not in frozen for key in selected):
        parser.error("Only the frozen primary/challenger models are supported by the v1 downloader")
    if args.with_challengers:
        selected.extend(key for key in CHALLENGERS if key not in selected)

    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = models_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest.update({
        "downloaded_at": datetime.now(tz=timezone.utc).isoformat(),
        "models_dir": str(models_dir),
        "analysis_stack": {
            "primary": "analysis_primary",
            "embedding": "video_embedding",
            "challengers": CHALLENGERS,
        },
    })
    manifest.setdefault("models", {})

    for key in selected:
        spec = MODELS[key]
        repo_id = spec["repo_id"]
        local_dir = models_dir / repo_id.replace("/", "__")
        print(f"Downloading {repo_id} -> {local_dir}", flush=True)
        try:
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                revision=frozen[key]["revision"],
            )
        except Exception as exc:
            # Marlin is currently gated.  Record the unavailable challenger and
            # continue so an accessible model does not get discarded with it.
            if spec.get("role") != "challenger":
                raise
            manifest.setdefault("unavailable_models", {})[key] = {
                **spec,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"Unavailable {repo_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        manifest["models"][key] = {**spec, "path": str(Path(path).resolve())}
        local_patches = _repair_known_model_metadata(key, Path(path))
        if local_patches:
            manifest["models"][key]["local_patches"] = local_patches
        verified = verify_snapshot(key, Path(path))
        manifest["models"][key]["revision"] = verified["revision"]
        manifest["models"][key]["verified_sha256"] = True
        (manifest.get("unavailable_models") or {}).pop(key, None)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}", flush=True)
    return 0


def _repair_known_model_metadata(key: str, model_path: Path) -> list[str]:
    """Apply narrow compatibility fixes to upstream conversion metadata.

    Marlin's current MLX snapshot labels its Qwen3.5 visual tower
    ``qwen3_5_vision`` while mlx-vlm's Qwen3.5 VisionModel accepts
    ``qwen3_5``.  The weights and architecture are unchanged; this only aligns
    the nested config discriminator used by the loader.
    """

    if key != "challenger_marlin":
        return []
    config_path = model_path / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    vision = payload.get("vision_config") or {}
    if vision.get("model_type") == "qwen3_5":
        return ["vision_config.model_type:qwen3_5_vision->qwen3_5"]
    if vision.get("model_type") != "qwen3_5_vision":
        return []
    vision["model_type"] = "qwen3_5"
    payload["vision_config"] = vision
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return ["vision_config.model_type:qwen3_5_vision->qwen3_5"]


if __name__ == "__main__":
    raise SystemExit(main())

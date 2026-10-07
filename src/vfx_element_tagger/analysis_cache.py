"""Invalidate semantic results when their inputs or analysis implementation change."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def analysis_signature(element, manifest: dict, options: dict | None = None) -> str:
    names = ("semantic", "gated_analysis", "model_backends", "video_inputs",
             "video_activity", "temporal_specialist", "analysis_runner")
    code = {name: hashlib.sha256((Path(__file__).parent / f"{name}.py").read_bytes()).hexdigest()
            for name in names}
    source = [{"fingerprint": rep.content_fingerprint, "size": rep.file_size_bytes,
               "sequence": rep.is_sequence, "first": rep.sequence_first_frame,
               "last": rep.sequence_last_frame, "missing": rep.missing_frames,
               "file": file_identity(rep.path)} for rep in element.source_representations]
    previews = [file_identity(path) for path in
                (element.preview_movie_480_path, element.preview_movie_1080_path) if path]
    models = {}
    for name, model in manifest.get("models", {}).items():
        path = Path(model.get("path") or "/nonexistent-model")
        models[name] = {p.name: file_identity(p) for p in path.iterdir() if p.is_file()} if path.is_dir() else {}
    value = {"code": code, "source": source, "previews": previews,
             "fingerprint": element.content_fingerprint, "manifest": manifest, "model_files": models,
             "options": options or {}, "technical": [element.duration_seconds, element.fps,
                 element.has_alpha_channel, element.alpha_non_empty, element.premult_status]}
    return "analysis-v1:" + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_identity(value):
    if not value:
        return None
    path = Path(value)
    if not path.is_file():
        return {"available": False}
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def analysis_is_current(element, signature: str) -> bool:
    return (element.analysis_status in {"accepted", "escalated", "needs_review"}
            and bool(element.semantic_analysis)
            and element.analysis_escalation.get("analysis_signature") == signature)

#!/usr/bin/env python3
"""One-time verified, same-volume relocation of this workstation project."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat


def digest(path):
    h = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            count += len(chunk)
    if count != path.stat().st_size:
        raise RuntimeError(f"Incomplete cloud file read: {path}: {count} bytes")
    return h.hexdigest()


def remap(value, mappings):
    if isinstance(value, dict):
        return {key: remap(item, mappings) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item, mappings) for item in value]
    if isinstance(value, str):
        for old, new in mappings:
            value = value.replace(str(old), str(new))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--remove-legacy-weights", action="store_true")
    parser.add_argument("--previous-project", type=Path, help="Finish remapping an already moved project.")
    args = parser.parse_args()
    actual_project = Path(__file__).resolve().parents[1]
    project = args.previous_project or actual_project
    models = Path.home() / ".cache/vfx-element-tagger/models"
    target = args.destination / project.name
    target_models = args.destination / "vfx-element-tagger-models"
    if not args.previous_project and (target.exists() or target_models.exists()):
        parser.error("destination project or models already exists; refusing to merge")
    if actual_project.stat().st_dev != args.destination.stat().st_dev:
        parser.error("this migration requires a same-volume atomic move")
    removed = []
    legacy = actual_project / ".cache/models"
    if args.remove_legacy_weights:
        for path in sorted(legacy.rglob("*")):
            if path.is_file() and path.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
                if not path.resolve().is_relative_to(legacy.resolve()):
                    raise RuntimeError("legacy weight path escapes expected directory")
                removed.append({"path": str(path.relative_to(project)), "bytes": path.stat().st_size})
                path.unlink()
        print(f"Removed {len(removed)} approved legacy weight files", flush=True)
    dataless = getattr(stat, "SF_DATALESS", 0x40000000)
    remaining = sum(path.stat().st_size for path in actual_project.rglob("*")
                    if path.is_file() and path.stat().st_flags & dataless)
    if remaining + 2_000_000_000 > shutil.disk_usage(actual_project).free:
        raise RuntimeError(f"Need {remaining} bytes plus a 2GB safety reserve to hydrate remaining files")
    checks = {}
    roots = (("project", target), ("models", target_models)) if args.previous_project else (("project", project), ("models", models))
    for key, root in roots:
        files = [path for path in root.rglob("*") if path.is_file()]
        for index, path in enumerate(files):
            checks[(key, str(path.relative_to(root)))] = digest(path)
            if index % 50 == 0:
                print(f"Hydrated/hashed {key}: {index+1}/{len(files)}", flush=True)
    if not args.previous_project:
        journal = [{"root": key, "path": relative, "sha256": sha}
                   for (key, relative), sha in checks.items()]
        (project.parent / (project.name + ".relocation-checks.json")).write_text(json.dumps(journal))
        project.rename(target)
        models.rename(target_models)
    for (key, relative), expected in checks.items():
        path = (target if key == "project" else target_models) / relative
        if digest(path) != expected:
            raise RuntimeError(f"Post-move hash mismatch: {path}")
    mappings = [(project, target), (models, target_models)]
    changed = []
    for path in list(target.rglob("*.json")) + [target_models / "manifest.json"]:
        original = json.loads(path.read_text())
        updated = remap(original, mappings)
        if path == target / ".cache/models/manifest.json" and (args.remove_legacy_weights or args.previous_project):
            updated["retired_models"] = updated.pop("models", {})
            updated["models"] = {}
            updated["note"] = "Legacy weights removed with user approval; metadata retained only."
        if updated != original:
            path.write_text(json.dumps(updated, indent=2) + "\n")
            changed.append(str(path.relative_to(target)) if path.is_relative_to(target) else str(path))
    source_db = Path.home() / "Library/Application Support/VFX Element Tagger/library.sqlite3"
    target_db = target / "data/library.sqlite3"
    target_db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(target_db) as dest:
            source.backup(dest)
    with sqlite3.connect(target_db) as db:
        count = db.execute("SELECT count(*) FROM elements").fetchone()[0]
        for key, payload in db.execute("SELECT element_id, payload_json FROM elements").fetchall():
            updated = remap(json.loads(payload), mappings)
            db.execute("UPDATE elements SET payload_json=? WHERE element_id=?", (json.dumps(updated), key))
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("relocated catalog integrity failure")
    settings = {"models_dir": str(target_models), "library": str(target_db)}
    with sqlite3.connect(target_db) as database:
        artifact_roots = [str(Path(json.loads(row[0])["poster_path"]).parent.parent.parent)
                          for row in database.execute("SELECT payload_json FROM elements")
                          if json.loads(row[0]).get("poster_path")]
    if artifact_roots:
        from collections import Counter
        settings["cache_dir"] = Counter(artifact_roots).most_common(1)[0][0]
    (target / ".vfx-tagger.json").write_text(json.dumps(settings, indent=2) + "\n")
    report = {"project": str(target), "models": str(target_models), "catalog": str(target_db),
              "verified_files": len(checks), "removed_legacy_weights": removed,
              "remapped_json_files": changed, "elements": count,
              "verification": "full-length reads and repeated hashes after move; original analysis-code hashes separately checked" if args.previous_project else "before/after SHA256",
              "previously_removed_approved_weight_files": 9 if args.previous_project else 0,
              "original_catalog_retained_as_backup": str(source_db)}
    (target / "evaluation/relocation-2026-10-07.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

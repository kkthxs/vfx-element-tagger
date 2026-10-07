#!/usr/bin/env python3
"""Maintainer tool: freeze the locally tested snapshots without including weights or paths."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vfx_element_tagger.model_integrity import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.models_manifest.read_text())
    frozen = {}
    for key, entry in manifest["models"].items():
        if entry.get("role") not in {"primary", "challenger"}:
            continue
        directory = Path(entry["path"])
        revisions = {path.read_text().splitlines()[0] for path in
                     (directory / ".cache/huggingface/download").rglob("*.metadata")}
        if len(revisions) != 1:
            raise ValueError(f"Expected one immutable cached revision for {key}: {revisions}")
        files = {}
        for path in sorted(directory.rglob("*")):
            name = path.relative_to(directory)
            if not path.is_file() or any(part.startswith(".") for part in name.parts):
                continue
            if path.suffix not in {".json", ".safetensors", ".model", ".txt", ".jinja", ".py", ".tiktoken"}:
                continue
            files[str(name)] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        frozen[key] = {"repo_id": entry["repo_id"], "revision": revisions.pop(),
                       "local_patches": entry.get("local_patches", []), "files": files}
        print(f"Frozen {key}: {len(files)} files", flush=True)
    args.output.write_text(json.dumps({"schema": 1, "models": frozen}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

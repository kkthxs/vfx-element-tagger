#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vfx_element_tagger.store import LibraryStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a VFX catalog between portable JSON and transactional SQLite."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing target after writing a sibling backup.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    target = args.target.expanduser().resolve()
    if source == target:
        parser.error("source and target must be different paths")
    if not source.exists():
        parser.error(f"source catalog does not exist: {source}")
    if target.exists() and not args.replace:
        parser.error(f"target already exists: {target}; use --replace")
    if target.exists():
        backup = target.with_name(f"{target.name}.pre-migration-backup")
        LibraryStore(target).backup(backup)
        print(f"Backed up existing target: {backup}")
        target.unlink()
        for sidecar_suffix in ("-wal", "-shm"):
            target.with_name(target.name + sidecar_suffix).unlink(missing_ok=True)

    source_store = LibraryStore(source)
    elements = source_store.load()
    target_store = LibraryStore(target)
    target_store.save(elements)
    reloaded = target_store.load()
    if [item.to_dict() for item in reloaded] != [item.to_dict() for item in elements]:
        raise RuntimeError("migration verification failed: target records differ from source")
    print(
        f"Migrated {len(elements)} elements: {source.name} -> {target.name}; "
        f"backend={'sqlite' if target_store.is_sqlite else 'json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

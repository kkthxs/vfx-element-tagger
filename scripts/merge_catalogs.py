#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vfx_element_tagger.catalog_merge import merge_catalogs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely merge one VFX catalog into another and relocate its web artifacts."
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--no-copy-artifacts", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    result = merge_catalogs(
        args.target,
        args.source,
        copy_artifacts=not args.no_copy_artifacts,
        replace_existing=args.replace_existing,
        create_backup=not args.no_backup,
    )
    print(
        f"added={result.added} replaced={result.replaced} skipped={result.skipped} "
        f"total={result.total}"
    )
    if result.backup_path:
        print(f"backup={result.backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

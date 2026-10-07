#!/usr/bin/env python3
"""Serve the local library, including browser-controlled import and analysis jobs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vfx_element_tagger.settings import cache_dir, library_path
from vfx_element_tagger.web import QCServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=library_path())
    parser.add_argument("--cache-dir", type=Path, default=cache_dir())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true", help="Label the interface as synthetic demo / curated data")
    parser.add_argument("--models-manifest", type=Path, help="Local analysis model manifest")
    args = parser.parse_args()
    if args.library.exists() and not args.library.is_file():
        parser.error(f"catalog is not a file: {args.library}")
    try:
        QCServer(args.host, args.port, args.library.resolve(), args.cache_dir.resolve(),
                 demo=args.demo, models_manifest=args.models_manifest).serve_forever()
    except (OSError, ValueError) as exc:
        parser.error(f"could not serve catalog: {exc}; choose another --port if occupied")


if __name__ == "__main__":
    main()

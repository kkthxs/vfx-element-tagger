from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from .pipeline import run_ingest
from .web import QCServer
from .settings import library_path as configured_library_path
from .settings import cache_dir as configured_cache_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest VFX elements and open the local QC view.")
    parser.add_argument("--folder", required=True, help="Folder containing movies or image sequences.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for artifacts and library data.",
    )
    parser.add_argument(
        "--library",
        default=None,
        help=(
            "Path to a .sqlite3 operational catalog or portable .json catalog. "
            "Defaults to the configured library, or <cache-dir>/library.json for an explicit cache."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="QC server bind host.")
    parser.add_argument("--port", type=int, default=8765, help="QC server port.")
    parser.add_argument("--default-fps", type=float, default=24.0, help="FPS for image sequences.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--no-server", action="store_true", help="Ingest only; do not start the QC server.")
    parser.add_argument("--no-artifacts", action="store_true", help="Skip ffmpeg poster/preview generation.")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Rebuild discovered elements even when their source fingerprints already exist in the library.",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Remove existing library elements that are not discovered under --folder.",
    )
    args = parser.parse_args(argv)

    folder = Path(args.folder).resolve()
    if not folder.exists():
        parser.error(f"Folder does not exist: {folder}")
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else configured_cache_dir().resolve()
    library_path = (Path(args.library).resolve() if args.library else
                    configured_library_path() if args.cache_dir is None else
                    cache_dir / "library.json")

    result = run_ingest(
        folder=folder,
        cache_dir=cache_dir,
        library_path=library_path,
        artifacts_enabled=not args.no_artifacts,
        default_fps=args.default_fps,
        force_recompute=args.force_recompute,
        prune_missing=args.prune_missing,
    )
    print(f"Library now has {len(result.elements)} elements after scanning {result.references_seen} media references")
    print(
        "Incremental ingest: "
        f"{result.processed_elements} processed, "
        f"{result.reused_elements} reused, "
        f"{result.preserved_unseen_elements} preserved outside this scan"
    )
    print(f"Library: {result.library_path}")
    print(f"Processing log: {result.log_path}")

    if args.no_server:
        return 0

    url = f"http://{args.host}:{args.port}"
    if not args.no_open:
        webbrowser.open(url)
    QCServer(
        host=args.host,
        port=args.port,
        library_path=library_path,
        cache_dir=cache_dir,
        artifacts_enabled=not args.no_artifacts,
    ).serve_forever()
    return 0

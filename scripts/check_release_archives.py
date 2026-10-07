#!/usr/bin/env python3
"""Check built release contents and optionally remove source-archive owner metadata."""
from __future__ import annotations

import argparse
from copy import copy
import gzip
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import zipfile

from export_public_release import SENSITIVE


FORBIDDEN_PARTS = {".git", ".cache", "test_assets", "models", "private", "screenshots"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".in", ".cfg"}
METADATA_NAMES = {"LICENSE", "NOTICE", "PKG-INFO", "METADATA", "WHEEL", "RECORD",
                  ".python-version", ".gitignore"}


def check_entry(name: str, payload: bytes) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or FORBIDDEN_PARTS.intersection(path.parts):
        raise ValueError(f"Unsafe/private archive path: {name}")
    if path.suffix not in TEXT_SUFFIXES and path.name not in METADATA_NAMES:
        raise ValueError(f"Unexpected non-source file: {name}")
    if path.suffix == ".json" and path.name not in {
        "model-lock.json", "PUBLICATION-MANIFEST.json", ".vfx-tagger.example.json"
    }:
        raise ValueError(f"Unexpected catalog/configuration JSON: {name}")
    if path.name in {"stress_test_analysis.py", "prepare_article_catalog.py", ".vfx-tagger.json"}:
        raise ValueError(f"Private development file: {name}")
    text = payload.decode("utf-8")
    if path.name != "export_public_release.py" and SENSITIVE.search(text):
        raise ValueError(f"Machine path or credential pattern: {name}")


def normalize_sdist(path: Path) -> None:
    """Mechanically rewrite generated tar headers, preserving all file payloads."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".normalized-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(path, "r:gz") as source, temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target:
                    for original in source.getmembers():
                        if not original.isfile() and not original.isdir():
                            raise ValueError(f"Unexpected link or special entry: {original.name}")
                        member = copy(original)
                        member.uid = member.gid = 0
                        member.uname = member.gname = ""
                        member.mtime = 0
                        member.pax_headers = {}
                        if original.isfile():
                            with source.extractfile(original) as stream:
                                target.addfile(member, stream)
                        else:
                            target.addfile(member)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def check_archive(path: Path) -> int:
    count = 0
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            if archive.testzip():
                raise ValueError("Wheel CRC check failed")
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                if entry.extra:
                    raise ValueError(f"Unexpected wheel extra metadata: {entry.filename}")
                check_entry(entry.filename, archive.read(entry))
                count += 1
    else:
        with tarfile.open(path, "r:gz") as archive:
            for entry in archive.getmembers():
                if entry.uid or entry.gid or entry.uname or entry.gname:
                    raise ValueError("Source archive contains machine owner metadata; normalize it first")
                if entry.isdir():
                    continue
                if not entry.isfile():
                    raise ValueError(f"Unexpected archive link: {entry.name}")
                check_entry(entry.name, archive.extractfile(entry).read())
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--normalize-sdist", action="store_true",
                        help="Rewrite generated source tar headers without changing file content")
    args = parser.parse_args()
    archives = sorted([*args.dist.glob("*.whl"), *args.dist.glob("*.tar.gz")])
    if not archives:
        parser.error("No wheel or source archive found")
    try:
        for archive in archives:
            if args.normalize_sdist and archive.name.endswith(".tar.gz"):
                normalize_sdist(archive)
            print(f"PASS {archive.name}: {check_archive(archive)} source/metadata files")
    except (OSError, ValueError, UnicodeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

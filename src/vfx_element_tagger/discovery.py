from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .util import iter_files, normalize_root, stable_id


SIDECAR_FILENAME = ".proxy-chains.json"


MOVIE_EXTENSIONS = {
    ".mov",
    ".mp4",
    ".m4v",
    ".mkv",
    ".avi",
    ".webm",
    ".mpg",
    ".mpeg",
}

IMAGE_EXTENSIONS = {
    ".exr",
    ".dpx",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tga",
}

# A frame number can be the entire stem (``0001.HR.exr``), can use any of
# the common separators, and can be followed by a compound suffix.  Treating
# only ``name.0001.exr`` as a sequence made real vendor libraries explode into
# hundreds of single-frame elements.
FRAME_RE = re.compile(
    r"^(?:(?P<prefix>.*?)(?P<separator>[._-]))?"
    r"(?P<frame>\d{2,8})(?P<suffix>(?:\.[^.]+)+)$"
)


@dataclass(frozen=True)
class MediaReference:
    reference_id: str
    kind: str
    path: Path
    original_path: Path
    files: list[Path]
    format_hint: str
    prefix: str | None = None
    first_frame: int | None = None
    last_frame: int | None = None
    frame_padding: int | None = None
    missing_frames: list[int] | None = None
    filename_root: str | None = None


def discover_media(folder: Path, cache_dir: Path | None = None) -> list[MediaReference]:
    folder = folder.resolve()
    excluded = [cache_dir.resolve()] if cache_dir else []
    image_groups: dict[tuple[Path, str, str, str, int], list[tuple[int, Path]]] = {}
    single_images: list[Path] = []
    movies: list[Path] = []

    for path in iter_files(folder, excluded):
        suffix = path.suffix.lower()
        if suffix in MOVIE_EXTENSIONS:
            movies.append(path)
            continue
        if suffix not in IMAGE_EXTENSIONS:
            continue
        match = FRAME_RE.match(path.name)
        if match:
            frame_text = match.group("frame")
            key = (
                path.parent,
                match.group("prefix") or "",
                match.group("separator") or "",
                match.group("suffix"),
                len(frame_text),
            )
            image_groups.setdefault(key, []).append((int(frame_text), path))
        else:
            single_images.append(path)

    sequence_files: set[Path] = set()
    references: list[MediaReference] = []

    for (parent, prefix, separator, suffix, padding), frames in sorted(image_groups.items()):
        frames.sort(key=lambda item: item[0])
        if len(frames) < 2:
            single_images.extend(path for _, path in frames)
            continue
        numbers = [number for number, _ in frames]
        first_frame = numbers[0]
        last_frame = numbers[-1]
        expected = set(range(first_frame, last_frame + 1))
        missing = sorted(expected.difference(numbers))
        files = [path for _, path in frames]
        sequence_files.update(files)
        pattern = parent / f"{prefix}{separator}%0{padding}d{suffix}"
        root = _sequence_filename_root(parent, prefix)
        references.append(
            MediaReference(
                reference_id=stable_id(
                    "ref", parent, prefix, separator, suffix, first_frame, last_frame
                ),
                kind="sequence",
                path=pattern,
                original_path=parent,
                files=files,
                format_hint=f"{Path(suffix).suffix.lower().lstrip('.')}_sequence",
                prefix=prefix,
                first_frame=first_frame,
                last_frame=last_frame,
                frame_padding=padding,
                missing_frames=missing,
                filename_root=root,
            )
        )

    for path in sorted(single_images):
        if path in sequence_files:
            continue
        references.append(
            MediaReference(
                reference_id=stable_id("ref", path),
                kind="single_image",
                path=path,
                original_path=path,
                files=[path],
                format_hint="single_image",
                filename_root=normalize_root(path.name, preserve_identifiers=True),
            )
        )

    for path in sorted(movies):
        references.append(
            MediaReference(
                reference_id=stable_id("ref", path),
                kind="movie",
                path=path,
                original_path=path,
                files=[path],
                format_hint=_movie_format_hint(path),
                filename_root=normalize_root(path.name, preserve_identifiers=True),
            )
        )

    return sorted(references, key=lambda item: str(item.path))


def _sequence_filename_root(parent: Path, prefix: str) -> str:
    if prefix:
        return normalize_root(prefix, preserve_identifiers=True)
    # Numeric-only sequences are common below version folders.  Use both the
    # asset and version directories so otherwise-identical ``0001.exr`` files
    # do not all acquire the same empty filename identity.
    return normalize_root(
        f"{parent.parent.name}_{parent.name}", preserve_identifiers=True
    )


@dataclass
class SidecarManifest:
    """A `.proxy-chains.json` directive parsed from disk.

    Sidecar format::

        {
          "version": 1,
          "groups": [
            {"id": "optional-stable-id",
             "members": ["relative/path/movie.mov", "subdir/master.0001.exr"]}
          ],
          "separate": [["a.mov", "b.mov"]]
        }

    `groups[*].members` are paths relative to the sidecar file's parent
    directory. Members that do not resolve to an existing file on disk are
    dropped (with a warning to stderr). `separate` is an optional list of
    path-tuples that must NOT end up in the same Element even if proxy-chain
    inference would otherwise merge them.
    """

    path: Path
    groups: list[list[Path]] = field(default_factory=list)
    separate: list[list[Path]] = field(default_factory=list)


def discover_sidecar_manifests(
    folder: Path,
    excluded: Iterable[Path] | None = None,
) -> list[SidecarManifest]:
    """Walk `folder` for `.proxy-chains.json` files and parse them.

    Member paths are resolved relative to each sidecar's parent directory.
    Entries pointing to nonexistent files are dropped and logged to stderr.
    """

    folder = folder.resolve()
    excluded_paths = list(excluded) if excluded else []
    manifests: list[SidecarManifest] = []
    for path in iter_files(folder, excluded_paths):
        if path.name != SIDECAR_FILENAME:
            continue
        manifest = _parse_sidecar(path)
        if manifest is not None:
            manifests.append(manifest)
    return manifests


def _parse_sidecar(path: Path) -> SidecarManifest | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"[sidecar] failed to read {path}: {exc}\n")
        return None

    parent = path.parent.resolve()
    groups: list[list[Path]] = []
    for group in payload.get("groups") or []:
        members = group.get("members") if isinstance(group, dict) else None
        if not members:
            continue
        resolved = _resolve_member_paths(members, parent, path)
        if len(resolved) >= 2:
            groups.append(resolved)

    separate: list[list[Path]] = []
    for tuple_entry in payload.get("separate") or []:
        if not isinstance(tuple_entry, (list, tuple)):
            continue
        resolved = _resolve_member_paths(list(tuple_entry), parent, path)
        if len(resolved) >= 2:
            separate.append(resolved)

    return SidecarManifest(path=path.resolve(), groups=groups, separate=separate)


def _resolve_member_paths(
    members: list[str],
    parent: Path,
    sidecar_path: Path,
) -> list[Path]:
    resolved: list[Path] = []
    for member in members:
        if not isinstance(member, str):
            continue
        candidate = (parent / member).resolve()
        if candidate.exists():
            resolved.append(candidate)
        else:
            sys.stderr.write(
                f"[sidecar] {sidecar_path}: member '{member}' does not exist on disk; skipping\n"
            )
    return resolved


def _movie_format_hint(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "mov":
        return "mov"
    if suffix == "mp4":
        return "mp4"
    return suffix or "movie"

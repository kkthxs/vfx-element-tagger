from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .models import Element
from .store import LibraryStore


@dataclass(frozen=True)
class CatalogMergeResult:
    added: int
    replaced: int
    skipped: int
    total: int
    backup_path: Path | None


def merge_catalogs(
    target_path: Path,
    source_path: Path,
    *,
    copy_artifacts: bool = True,
    replace_existing: bool = False,
    create_backup: bool = True,
) -> CatalogMergeResult:
    """Merge source records into a catalog while preserving target review data.

    Element IDs and content fingerprints are both treated as duplicate guards.
    Existing target records win unless replacement is explicitly requested.
    """

    target_path = target_path.expanduser().resolve()
    source_path = source_path.expanduser().resolve()
    target_store = LibraryStore(target_path)
    source_store = LibraryStore(source_path)
    target = target_store.load()
    source = source_store.load()

    target_by_id = {element.element_id: index for index, element in enumerate(target)}
    target_fingerprints = {
        element.content_fingerprint: element.element_id
        for element in target
        if element.content_fingerprint
    }
    additions: list[Element] = []
    replacements: list[tuple[int, Element]] = []
    skipped = 0

    for source_element in source:
        element = Element.from_dict(source_element.to_dict())
        existing_index = target_by_id.get(element.element_id)
        fingerprint_owner = (
            target_fingerprints.get(element.content_fingerprint)
            if element.content_fingerprint
            else None
        )
        if existing_index is not None:
            if not replace_existing:
                skipped += 1
                continue
            replacements.append((existing_index, element))
            continue
        if fingerprint_owner is not None:
            skipped += 1
            continue
        additions.append(element)
        target_by_id[element.element_id] = len(target) + len(additions) - 1
        if element.content_fingerprint:
            target_fingerprints[element.content_fingerprint] = element.element_id

    if not additions and not replacements:
        return CatalogMergeResult(0, 0, skipped, len(target), None)

    if copy_artifacts:
        for element in [*additions, *(item for _index, item in replacements)]:
            _relocate_artifacts(element, source_path.parent, target_path.parent)

    backup_path = None
    if create_backup and target_path.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_path = target_path.with_name(
            f"{target_path.stem}.pre-merge-{stamp}{target_path.suffix}"
        )
        target_store.backup(backup_path)

    for index, element in replacements:
        target[index] = element
    target.extend(additions)
    target_store.save(target)
    return CatalogMergeResult(
        added=len(additions),
        replaced=len(replacements),
        skipped=skipped,
        total=len(target),
        backup_path=backup_path,
    )


def _relocate_artifacts(element: Element, source_cache: Path, target_cache: Path) -> None:
    source_dir = source_cache / "artifacts" / element.element_id
    target_dir = target_cache / "artifacts" / element.element_id
    if not source_dir.exists():
        raise FileNotFoundError(f"artifact directory is missing for {element.element_id}: {source_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    def relocated(value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value).expanduser().resolve()
        try:
            relative = path.relative_to(source_cache.resolve())
        except ValueError:
            return value
        return str((target_cache / relative).resolve())

    element.poster_path = relocated(element.poster_path)
    element.preview_movie_480_path = relocated(element.preview_movie_480_path)
    element.preview_movie_1080_path = relocated(element.preview_movie_1080_path)

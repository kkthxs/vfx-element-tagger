from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.catalog_merge import merge_catalogs  # noqa: E402
from vfx_element_tagger.models import Element, SourceRepresentation  # noqa: E402
from vfx_element_tagger.store import LibraryStore  # noqa: E402


class CatalogMergeTests(unittest.TestCase):
    def test_merge_preserves_target_records_deduplicates_and_moves_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_path = root / "target" / "library.json"
            source_path = root / "source" / "library.json"
            existing = _element("existing", "fingerprint-a", root / "target")
            existing.human_overrides = {"reviewed": True}
            duplicate = _element("duplicate", "fingerprint-a", root / "source")
            incoming = _element("incoming", "fingerprint-b", root / "source")
            _write_artifacts(root / "source", incoming.element_id)
            LibraryStore(target_path).save([existing])
            LibraryStore(source_path).save([duplicate, incoming])

            result = merge_catalogs(target_path, source_path)
            merged = LibraryStore(target_path).load()

            self.assertEqual((result.added, result.replaced, result.skipped, result.total), (1, 0, 1, 2))
            self.assertIsNotNone(result.backup_path)
            self.assertTrue(result.backup_path.exists())
            self.assertEqual(merged[0].human_overrides, {"reviewed": True})
            self.assertEqual([element.element_id for element in merged], ["existing", "incoming"])
            self.assertTrue(Path(merged[1].poster_path).exists())
            self.assertEqual(
                Path(merged[1].poster_path).parent.resolve(),
                (root / "target" / "artifacts" / "incoming").resolve(),
            )

    def test_sqlite_merge_backup_keeps_the_database_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_path = root / "target" / "library.sqlite3"
            source_path = root / "source" / "library.sqlite3"
            LibraryStore(target_path).save([_element("existing", "fingerprint-a", root / "target")])
            incoming = _element("incoming", "fingerprint-b", root / "source")
            _write_artifacts(root / "source", incoming.element_id)
            LibraryStore(source_path).save([incoming])

            result = merge_catalogs(target_path, source_path)

            self.assertIsNotNone(result.backup_path)
            self.assertEqual(result.backup_path.suffix, ".sqlite3")


def _element(element_id: str, fingerprint: str, cache_dir: Path) -> Element:
    representation = SourceRepresentation(
        representation_id=f"rep-{element_id}",
        path=f"/{element_id}.mov",
        original_path=f"/{element_id}.mov",
        format="mov",
        is_primary=True,
        content_fingerprint=fingerprint,
    )
    artifact_dir = cache_dir / "artifacts" / element_id
    return Element(
        element_id=element_id,
        source_representations=[representation],
        primary_representation_id=representation.representation_id,
        content_fingerprint=fingerprint,
        poster_path=str(artifact_dir / "poster.jpg"),
        preview_movie_480_path=str(artifact_dir / "preview_480.mp4"),
        preview_movie_1080_path=str(artifact_dir / "preview_1080.mp4"),
    )


def _write_artifacts(cache_dir: Path, element_id: str) -> None:
    artifact_dir = cache_dir / "artifacts" / element_id
    artifact_dir.mkdir(parents=True)
    for name in ("poster.jpg", "preview_480.mp4", "preview_1080.mp4"):
        (artifact_dir / name).write_bytes(b"test")


if __name__ == "__main__":
    unittest.main()

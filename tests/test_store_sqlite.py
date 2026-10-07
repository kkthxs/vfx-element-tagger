from __future__ import annotations

import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.models import Element, SourceRepresentation  # noqa: E402
from vfx_element_tagger.store import LibraryStore  # noqa: E402


class SQLiteStoreTests(unittest.TestCase):
    def test_sqlite_round_trip_and_single_element_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LibraryStore(Path(tmp) / "library.sqlite3")
            first = _element("first", "smoke")
            second = _element("second", "fire")
            store.save([first, second])
            revision = store.revision()

            first.primary_family = "dust"
            first.semantic_analysis["primary_family"] = "dust"
            store.save_element(first)

            loaded = store.load()
            self.assertEqual([item.element_id for item in loaded], ["first", "second"])
            self.assertEqual(loaded[0].primary_family, "dust")
            self.assertNotEqual(store.revision(), revision)

    def test_full_save_removes_missing_records_and_preserves_ratings_for_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LibraryStore(Path(tmp) / "library.sqlite3")
            first = _element("first", "smoke")
            second = _element("second", "fire")
            store.save([first, second])
            store.rate_element("first", "person-a", 5, "Alex")

            store.save([first])

            self.assertEqual([item.element_id for item in store.load()], ["first"])
            self.assertEqual(store.rating_summary("first")["rating_average"], 5.0)

    def test_individual_ratings_are_idempotent_and_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LibraryStore(Path(tmp) / "library.sqlite3")
            store.save([_element("rated", "spark")])
            store.rate_element("rated", "person-a", 5, "Alex")
            store.rate_element("rated", "person-b", 3, "Sam")
            store.rate_element("rated", "person-a", 4, "Alex")

            summary = store.rating_summary("rated", "person-a")

            self.assertEqual(summary["rating_average"], 3.5)
            self.assertEqual(summary["rating_count"], 2)
            self.assertEqual(summary["user_rating"], 4)
            self.assertEqual(summary["rating_distribution"], {"1": 0, "2": 0, "3": 1, "4": 1, "5": 0})

    def test_concurrent_people_can_rate_one_element(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LibraryStore(Path(tmp) / "library.sqlite3")
            store.save([_element("rated", "particle")])

            def rate(index: int) -> None:
                store.rate_element(
                    "rated",
                    f"person-{index}",
                    index % 5 + 1,
                    f"Person {index}",
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(rate, range(40)))

            summary = store.rating_summary("rated")
            self.assertEqual(summary["rating_count"], 40)
            self.assertEqual(summary["rating_average"], 3.0)
            self.assertEqual(summary["rating_distribution"], {str(star): 8 for star in range(1, 6)})


def _element(element_id: str, family: str) -> Element:
    representation = SourceRepresentation(
        representation_id=f"rep-{element_id}",
        path=f"/{element_id}.mov",
        original_path=f"/{element_id}.mov",
        format="mov",
        is_primary=True,
        content_fingerprint=f"fingerprint-{element_id}",
    )
    return Element(
        element_id=element_id,
        source_representations=[representation],
        primary_representation_id=representation.representation_id,
        primary_family=family,
        semantic_analysis={
            "primary_family": family,
            "effect_type": f"{family}_element",
            "summary": f"A {family} element for testing.",
        },
        analysis_status="accepted",
    )


if __name__ == "__main__":
    unittest.main()

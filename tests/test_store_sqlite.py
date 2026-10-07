from __future__ import annotations

import sys
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.models import Element, ProcessingEvent, SourceRepresentation  # noqa: E402
from vfx_element_tagger.store import LibraryStore  # noqa: E402


class SQLiteStoreTests(unittest.TestCase):
    def test_operations_close_connections_including_readonly_backups_and_json_ratings(self):
        connect = sqlite3.connect
        connections = []

        def tracked_connect(*args, **kwargs):
            connection = connect(*args, **kwargs)
            connections.append(connection)
            return connection

        with tempfile.TemporaryDirectory() as tmp, \
             patch("vfx_element_tagger.store.sqlite3.connect", side_effect=tracked_connect):
            for suffix in (".sqlite3", ".json"):
                with self.subTest(suffix=suffix):
                    store = LibraryStore(Path(tmp) / ("library" + suffix))
                    store.save([_element("first", "smoke")])
                    store.save_element(_element("second", "fire"))
                    self.assertEqual(len(store.load()), 2)
                    self.assertIsNotNone(store.load_one("first"))
                    self.assertTrue(store.revision())
                    store.append_events([ProcessingEvent("first", "test", "success")])
                    store.rate_element("first", "artist", 5)
                    self.assertEqual(store.rating_summary("first")["rating_count"], 1)
                    with self.assertRaises(KeyError):
                        store.rate_element("missing", "artist", 5)
                    backup = store.backup(Path(tmp) / ("backup" + suffix))
                    self.assertEqual(len(LibraryStore.load_readonly(backup)), 2)
            self.assertTrue(connections)
            for connection in connections:
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

    def test_failed_full_save_rolls_back_and_closes_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LibraryStore(Path(tmp) / "library.sqlite3")
            first = _element("first", "smoke")
            store.save([first])
            revision = store.revision()
            connection = store._connect()
            with patch.object(store, "_connect", return_value=connection), \
                 self.assertRaises(sqlite3.IntegrityError):
                store.save([first, first])
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")
            self.assertEqual(store.revision(), revision)
            self.assertEqual([item.element_id for item in store.load()], ["first"])

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

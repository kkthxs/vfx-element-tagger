from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vfx_element_tagger.analysis_cache import analysis_is_current, analysis_signature
from vfx_element_tagger.models import Element, SourceRepresentation
from analyze_gated import _select_elements, _apply_outcome
from types import SimpleNamespace
from vfx_element_tagger.store import LibraryStore
import sqlite3


class AnalysisCacheTests(unittest.TestCase):
    def test_readonly_loading_does_not_create_missing_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.sqlite3"
            with self.assertRaises(sqlite3.OperationalError):
                LibraryStore.load_readonly(path)
            self.assertFalse(path.exists())

    def element(self):
        return Element("test", [SourceRepresentation("rep", "/absent.mov", "/absent.mov", "mov")], "rep")

    def test_signature_changes_with_source_prompt_settings_and_models(self):
        element = self.element()
        original = analysis_signature(element, {"primary": "one"}, {"max_tokens": 100})
        self.assertEqual(original, analysis_signature(element, {"primary": "one"}, {"max_tokens": 100}))
        self.assertNotEqual(original, analysis_signature(element, {"primary": "two"}, {"max_tokens": 100}))
        self.assertNotEqual(original, analysis_signature(element, {"primary": "one"}, {"max_tokens": 200}))
        element.primary().content_fingerprint = "new-content"
        self.assertNotEqual(original, analysis_signature(element, {"primary": "one"}, {"max_tokens": 100}))

    def test_modified_proxy_invalidates_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proxy.mp4"
            path.write_bytes(b"old")
            element = self.element()
            element.preview_movie_480_path = str(path)
            before = analysis_signature(element, {})
            path.write_bytes(b"new-content")
            self.assertNotEqual(before, analysis_signature(element, {}))

    def test_matching_review_result_skipped_but_force_and_stale_recomputed(self):
        element = self.element()
        element.semantic_analysis = {"summary": "result"}
        element.analysis_status = "needs_review"
        element.analysis_escalation = {"analysis_signature": "same"}
        self.assertTrue(analysis_is_current(element, "same"))
        self.assertEqual(_select_elements([element], set(), False, 5, signatures={"test": "same"}), [])
        self.assertEqual(_select_elements([element], set(), True, 5, signatures={"test": "same"}), [element])
        self.assertEqual(_select_elements([element], set(), False, 5, signatures={"test": "new"}), [element])

    def test_refresh_drops_old_guesses_but_preserves_human_fields_and_history(self):
        element = self.element()
        element.content_facets = {"old": "fire"}
        element.motion_facets = {"old": "falling"}
        element.human_overrides = {"fields": {"motion.speed": "slow"}}
        element.analysis_escalation = {"human_review_history": [{"note": "artist"}]}
        outcome = SimpleNamespace(analysis={"primary_family": "plate", "summary": "A man turns.",
                                            "motion": {"speed": "fast"}},
                                  confidence=.9, status="needs_review", model_version="new",
                                  schema_version="new", escalation={}, candidates=[])
        _apply_outcome(element, outcome, {})
        self.assertEqual(element.semantic_analysis["motion"]["speed"], "slow")
        self.assertNotIn("old", element.content_facets)
        self.assertNotIn("old", element.motion_facets)
        self.assertEqual(element.analysis_escalation["human_review_history"], [{"note": "artist"}])

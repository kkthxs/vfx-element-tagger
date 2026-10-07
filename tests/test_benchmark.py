from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from vfx_element_tagger.models import Element, SourceRepresentation
from prepare_benchmark import annotation_queue
from evaluate_ground_truth import _checks


class BenchmarkTests(unittest.TestCase):
    def element(self, key, family=None):
        return Element(key, [SourceRepresentation(key, f"/{key}.mov", f"/{key}.mov", "mov")], key,
                       source_family_id=family, semantic_analysis={"summary": "Secret model answer"})

    def test_holdout_excludes_tuned_groups_and_model_answers(self):
        known = self.element("known", "family")
        duplicate = self.element("duplicate", "family")
        heldout = self.element("heldout")
        payload = annotation_queue([known, duplicate, heldout], ["known"])
        self.assertEqual(payload["selected_groups"], 1)
        self.assertEqual(payload["assets"][0]["element_id"], "heldout")
        self.assertNotIn("Secret model answer", str(payload))
        self.assertEqual(payload["labeled_assets"], 0)

    def test_range_evaluation_rejects_trim_that_drops_flash(self):
        element = self.element("flash")
        element.semantic_analysis["action_timing"] = {"optimal_start_seconds": 11, "optimal_end_seconds": 15}
        constraint = {"type": "range_covers", "start_seconds": 8.8, "end_seconds": 9.1}
        self.assertFalse(_checks(element, constraint)[0][0])
        element.semantic_analysis["action_timing"]["optimal_start_seconds"] = 8.2
        self.assertTrue(_checks(element, constraint)[0][0])

    def test_overlapping_identifiers_exclude_transitive_duplicates(self):
        known = self.element("known", "family")
        bridge = self.element("bridge", "family")
        duplicate = self.element("duplicate", "other-family")
        bridge.near_duplicate_group_id = duplicate.near_duplicate_group_id = "near-match"
        heldout = self.element("heldout")
        payload = annotation_queue([duplicate, heldout, known, bridge], ["known"])
        self.assertEqual(payload["eligible_groups"], 1)
        self.assertEqual(payload["assets"][0]["element_id"], "heldout")

    def test_shared_fingerprint_groups_distinct_families(self):
        first = self.element("first", "family-one")
        second = self.element("second", "family-two")
        first.content_fingerprint = second.content_fingerprint = "same-content"
        payload = annotation_queue([second, first])
        self.assertEqual(payload["selected_groups"], 1)
        self.assertEqual(payload, annotation_queue([first, second]))

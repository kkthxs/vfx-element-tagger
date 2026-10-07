from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ground_truth_evaluator", ROOT / "scripts/evaluate_ground_truth.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_only_selected_affirmative_claims_count_as_visual_evidence(self):
        class Record:
            primary_family = "dust"
            secondary_families = []
            semantic_analysis = {
                "summary": "No smoke is visible, but fire rises.",
                "uncertainty": ["It is unclear whether it travels toward camera."],
                "search_text": "smoke toward camera",
                "usability": {"best_uses": ["smoke reference"]},
            }

        record = Record()
        constraint = {"type": "text_all", "concepts": [["smoke"], ["toward camera"], ["fire"]]}
        self.assertEqual([ok for ok, _ in evaluation._checks(record, constraint)], [False, False, True])
        self.assertTrue(all(ok for ok, _ in evaluation._checks(record, constraint, legacy_text=True)))
        record.semantic_analysis = deepcopy(record.semantic_analysis)
        record.semantic_analysis["motion"] = {"depth_motion": "toward_camera"}
        self.assertTrue(evaluation._checks(record, constraint)[1][0])

    def test_word_boundaries_and_negation_do_not_turn_absence_into_presence(self):
        for text in ("No smoke.", "Smoke is not visible.", "Without smoke or sparks.", "Perhaps smoke."):
            self.assertFalse(evaluation._term_present(text, "smoke"), text)
        self.assertFalse(evaluation._term_present("firearm", "fire"))
        self.assertTrue(evaluation._term_present("no debris, but smoke billows", "smoke"))
        self.assertTrue(evaluation._term_present("The cloud dissipates.", "dissipating"))
        self.assertTrue(evaluation._term_present("burst_then_dissipate", "dissipating"))
        self.assertFalse(evaluation._term_present("The cloud is not dissipating.", "dissipates"))
        self.assertEqual(evaluation._constraint_size({"type": "text_all", "concepts": [["a"], ["b"]]}), 2)

    def test_missing_assets_count_as_failures_and_cli_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "truth.json"
            truth.write_text(json.dumps({"assets": {"missing": {"constraints": [
                {"type": "family_any", "values": ["smoke"]},
                {"type": "text_all", "concepts": [["smoke"], ["rising"]]},
            ]}}}))
            output = io.StringIO()
            with patch.object(evaluation.LibraryStore, "load", return_value=[]), \
                 patch.object(sys, "argv", ["evaluate", "--library", str(Path(tmp)/"empty.json"),
                                             "--ground-truth", str(truth), "--json"]), \
                 redirect_stdout(output):
                code = evaluation.main()
            result = json.loads(output.getvalue())
            self.assertEqual(code, 1)
            self.assertEqual((result["passed"], result["total"]), (0, 3))
            self.assertEqual(result["coverage"], {"expected_assets": 1, "found_assets": 0})


if __name__ == "__main__":
    unittest.main()

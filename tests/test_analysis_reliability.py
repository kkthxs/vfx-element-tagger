from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vfx_element_tagger.semantic import analyses_agree, assess_analysis, normalize_analysis
from vfx_element_tagger import settings


def valid_analysis():
    return {
        "summary": "A single bright dust burst expands and falls against black.",
        "primary_family": "dust", "secondary_families": ["debris"], "effect_type": "dust_hit",
        "overall_confidence": 0.96,
        "composition": {"viewpoint": "front", "shot_scale": "wide", "edge_contact": ["none"],
                        "frame_coverage": "medium", "instance_count": "single"},
        "appearance": {"colour": ["brown"], "density": "medium", "backing": "black"},
        "motion": {"temporal_arc": "burst_then_dissipate", "speed": "fast", "direction": "right",
                   "depth_motion": "in_plane", "event_count": "single", "character": ["expanding"],
                   "loopability": "not_loopable"},
        "usability": {"best_uses": ["dust impact"]},
        "evidence": [{"timestamp_seconds": 0.1, "observation": "Dust expands."}],
        "field_confidence": {"family": 0.96, "motion": 0.94}, "uncertainty": [],
    }


class AnalysisReliabilityTests(unittest.TestCase):
    def test_gate_rejects_invalid_family_and_low_field_confidence(self):
        baseline = valid_analysis()
        self.assertTrue(assess_analysis(baseline).accepted)
        for key, value in (("primary_family", "invented"),
                           ("field_confidence", {"family": 0.01})):
            changed = {**baseline, key: value}
            self.assertFalse(assess_analysis(changed).accepted)

    def test_consensus_compares_motion_counts_and_framing(self):
        baseline = valid_analysis()
        for section, field, value in (("motion", "direction", "left"),
                                      ("motion", "depth_motion", "toward_camera"),
                                      ("motion", "event_count", "multiple"),
                                      ("composition", "instance_count", "multiple"),
                                      ("composition", "edge_contact", ["bottom"])):
            changed = deepcopy(baseline)
            changed[section][field] = value
            self.assertFalse(analyses_agree(baseline, changed), field)

    def test_unknown_moving_fields_and_low_information_stillness_are_reviewed(self):
        value = valid_analysis()
        value["motion"]["direction"] = "unknown"
        self.assertFalse(assess_analysis(value).accepted)
        value["motion"].update(speed="still", temporal_arc="static")
        self.assertFalse(assess_analysis(value, {"proxy_information_limited": True}).accepted)

    def test_unknown_framing_cannot_be_accepted_at_full_confidence(self):
        value = valid_analysis()
        value["overall_confidence"] = 1
        value["composition"]["viewpoint"] = "unknown"
        self.assertFalse(assess_analysis(value).accepted)

    def test_nonfinite_confidence_and_invalid_times_are_not_laundered(self):
        for number in (float("nan"), float("inf"), True):
            value = valid_analysis()
            value["field_confidence"]["motion"] = number
            self.assertFalse(assess_analysis(normalize_analysis(value)).accepted)
        for timestamp in (-1, None, float("nan"), "not a time"):
            value = valid_analysis()
            value["evidence"][0]["timestamp_seconds"] = timestamp
            self.assertFalse(assess_analysis(normalize_analysis(value)).accepted)

    def test_paths_support_relative_settings_and_environment_override(self):
        with patch.object(settings, "project_settings", return_value={"models_dir": "local-models"}), \
             patch.dict(settings.os.environ, {}, clear=True), \
             patch.object(settings.Path, "home", side_effect=RuntimeError("No home directory")):
            self.assertEqual(settings.models_dir(), settings.PROJECT_ROOT / "local-models")
            override = settings.PROJECT_ROOT.parent / "override-models"
            with patch.dict(settings.os.environ, {"VFX_TAGGER_MODELS_DIR": str(override)}):
                self.assertEqual(settings.models_dir(), override)

    def test_models_default_resolves_home_only_when_needed(self):
        home = settings.PROJECT_ROOT.parent / "example-home"
        with patch.object(settings, "project_settings", return_value={}), \
             patch.dict(settings.os.environ, {}, clear=True), \
             patch.object(settings.Path, "home", return_value=home) as resolve_home:
            self.assertEqual(settings.models_dir(), home / ".cache/vfx-element-tagger/models")
            resolve_home.assert_called_once_with()

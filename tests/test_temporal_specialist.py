from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.temporal_specialist import (  # noqa: E402
    detail_specialist_reasons,
    merge_detail_specialist,
    parse_detail_specialist,
    merge_temporal_specialist,
    parse_temporal_specialist,
    specialist_reasons,
)


class TemporalSpecialistTests(unittest.TestCase):
    def test_semantic_burst_routes_specialist_with_generic_filename(self):
        analysis = _wall_analysis()
        analysis["effect_type"] = "rising_smoke_plume"
        analysis["motion"]["temporal_arc"] = "burst_then_dissipate"
        reasons = specialist_reasons(analysis, {"source_filename_hint": "asset.mov"})
        self.assertIn("impact_late_phase_unresolved", reasons)

    def test_proxy_detail_risk_triggers_source_resolution_specialist(self) -> None:
        analysis = _wall_analysis()
        analysis["primary_family"] = "particle"

        reasons = detail_specialist_reasons(
            analysis,
            {"fine_detail_likelihood": 0.75, "high_resolution_recommended": True},
        )

        self.assertIn("proxy_fine_detail_risk", reasons)
        self.assertIn("delicate_family_requires_detail_check", reasons)

    def test_high_confidence_detail_adds_texture_material_and_motion(self) -> None:
        analysis = _wall_analysis()
        specialist = {
            "detail_observation": "Fine sparks and grains fall from the plume.",
            "observed_materials": ["dust", "spark", "particle"],
            "texture": ["fine"],
            "density": "light",
            "instance_count": "continuous_many",
            "motion_character": ["falling"],
            "temporal_arc": "burst_then_dissipate",
            "confidence": 0.9,
        }

        merged, refinements = merge_detail_specialist(analysis, specialist)

        self.assertIn("spark", merged["secondary_families"])
        self.assertIn("fine", merged["appearance"]["texture"])
        self.assertIn("falling", merged["motion"]["character"])
        self.assertTrue(any(item["method"].startswith("activity-focused") for item in refinements))

    def test_source_detail_can_expand_but_not_shrink_proxy_timing(self) -> None:
        analysis = _wall_analysis()
        analysis["action_timing"] = {
            "available": True,
            "visible_start_seconds": 0.0,
            "main_action_start_seconds": 0.0,
            "peak_seconds": 0.5,
            "main_action_end_seconds": 1.0,
            "visible_end_seconds": 1.2,
            "optimal_start_seconds": 0.0,
            "optimal_end_seconds": 1.5,
            "optimal_duration_seconds": 1.5,
            "pattern": "burst",
            "event_segments": [],
        }
        specialist = {
            "confidence": 0.94,
            "timing": {
                "visible_start_seconds": 0.0,
                "main_action_start_seconds": 0.2,
                "peak_seconds": 0.8,
                "main_action_end_seconds": 5.0,
                "visible_end_seconds": 7.0,
                "confidence": 0.93,
            },
        }

        merged, refinements = merge_detail_specialist(
            analysis,
            specialist,
            allow_timing_expansion=True,
            source_fps=24.0,
            source_frame_count=360,
            duration_seconds=15.0,
        )

        timing = merged["action_timing"]
        self.assertEqual(timing["peak_seconds"], 0.5)
        self.assertEqual(timing["visible_end_seconds"], 7.0)
        self.assertGreater(timing["optimal_end_seconds"], 1.5)
        self.assertEqual(timing["optimal_end_frame"], 131)
        self.assertTrue(any(item["field"] == "action_timing" for item in refinements))

    def test_detail_parser_normalizes_structured_output(self) -> None:
        parsed = parse_detail_specialist(
            '{"detail_observation":"Fine embers", "observed_materials":["Spark"], '
            '"texture":["Fine"], "motion_character":["Falling"], "confidence":0.9}'
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["observed_materials"], ["spark"])
        self.assertEqual(parsed["motion_character"], ["falling"])

    def test_unresolved_dust_hit_triggers_specialist(self) -> None:
        analysis = _wall_analysis()

        reasons = specialist_reasons(
            analysis,
            {"source_filename_hint": "Wall_Hit_Front_1.mov"},
        )

        self.assertIn("impact_material_unresolved", reasons)
        self.assertIn("impact_depth_unresolved", reasons)
        self.assertIn("impact_late_phase_unresolved", reasons)

    def test_compound_dusthit_filename_triggers_specialist(self) -> None:
        reasons = specialist_reasons(
            _wall_analysis(),
            {"source_filename_hint": "DustHit-002.mov"},
        )

        self.assertIn("impact_depth_unresolved", reasons)

    def test_nonimpact_electricity_does_not_use_impact_specialist(self) -> None:
        analysis = _wall_analysis()
        analysis["primary_family"] = "electricity"

        self.assertEqual(
            specialist_reasons(analysis, {"source_filename_hint": "Lightning_Front.mov"}),
            [],
        )

    def test_high_confidence_specialist_merges_material_and_motion_not_edges(self) -> None:
        analysis = _wall_analysis()
        analysis["motion"]["depth_motion"] = "unknown"
        specialist = {
            "timeline_summary": "Initial radial expansion followed by falling fragments.",
            "observed_materials": ["dust", "smoke", "debris"],
            "initial_screen_motion": "outward radial expansion",
            "depth_motion": "toward_camera",
            "later_motion": "downward settling",
            "motion_phases": ["initial impact", "fragment fall", "dust settling"],
            "confidence": 0.85,
            "uncertainty": [],
            "edge_contact": ["bottom"],
        }

        merged, refinements = merge_temporal_specialist(analysis, specialist)

        self.assertIn("debris", merged["secondary_families"])
        self.assertEqual(merged["motion"]["depth_motion"], "toward_camera")
        self.assertIn("falling", merged["motion"]["character"])
        self.assertEqual(merged["composition"]["edge_contact"], ["none"])
        self.assertTrue(any(item["field"] == "search_text" for item in refinements))

    def test_specialist_does_not_override_in_plane_depth_without_corroboration(self) -> None:
        analysis = _wall_analysis()
        analysis["effect_type"] = "ground_dust_hit"
        specialist = {
            "timeline_summary": "Radial impact followed by settling.",
            "observed_materials": ["dust", "particle"],
            "depth_motion": "away_from_camera",
            "later_motion": "particles fall downward",
            "motion_phases": ["impact", "falling particles"],
            "confidence": 0.95,
        }

        merged, _refinements = merge_temporal_specialist(analysis, specialist)

        self.assertEqual(merged["motion"]["depth_motion"], "in_plane")
        self.assertEqual(merged["temporal_detail"]["depth_motion"], "unknown")
        self.assertIn("debris", merged["secondary_families"])

    def test_low_confidence_specialist_is_retained_but_not_merged(self) -> None:
        analysis = _wall_analysis()
        merged, refinements = merge_temporal_specialist(
            analysis,
            {"observed_materials": ["debris"], "confidence": 0.5},
        )

        self.assertEqual(merged, analysis)
        self.assertEqual(refinements, [])

    def test_parser_normalizes_depth_motion_phrase(self) -> None:
        parsed = parse_temporal_specialist(
            '{"timeline_summary":"Impact", "observed_materials":["Dust"], '
            '"depth_motion":"expansion toward camera", "confidence":0.85}'
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["depth_motion"], "toward_camera")


def _wall_analysis() -> dict:
    return {
        "summary": "A rising dust plume expands against black.",
        "search_text": "rising dust smoke plume",
        "primary_family": "dust",
        "secondary_families": ["smoke"],
        "effect_type": "rising_dust_cloud",
        "composition": {"edge_contact": ["none"]},
        "appearance": {"backing": "black"},
        "motion": {
            "depth_motion": "in_plane",
            "character": ["rising", "expanding"],
        },
    }


if __name__ == "__main__":
    unittest.main()

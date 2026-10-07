from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.gated_analysis import (  # noqa: E402
    GatedSequenceAnalyzer,
    sampling_plan,
    ultra_sampling_plan,
)
from vfx_element_tagger.semantic import (  # noqa: E402
    apply_deterministic_refinements,
    assess_analysis,
    parse_analysis,
)


class FakeBackend:
    def __init__(self, model_id: str, outputs: list[str]):
        self.model_id = model_id
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def analyze(self, video_path, prompt, fps, max_frames, max_tokens):
        self.calls.append({"fps": fps, "max_frames": max_frames, "prompt": prompt})
        return self.outputs.pop(0)


class FakeImageBackend(FakeBackend):
    def __init__(self, model_id: str, outputs: list[str], image_outputs: list[str]):
        super().__init__(model_id, outputs)
        self.image_outputs = list(image_outputs)
        self.image_calls: list[dict] = []

    def analyze_image(self, image, prompt, max_tokens):
        self.image_calls.append({"image": image, "prompt": prompt, "max_tokens": max_tokens})
        return self.image_outputs.pop(0)


class GatedAnalysisTests(unittest.TestCase):
    def test_low_information_needs_independent_corroboration(self):
        primary = FakeBackend("primary", [json.dumps(_analysis(confidence=0.96))])
        result = GatedSequenceAnalyzer(primary, []).analyze(Path("/missing.mp4"), 1,
                    activity={"proxy_information_limited": True})
        self.assertEqual(result.status, "needs_review")
        self.assertIn("low_information_without_independent_corroboration",
                      result.escalation["final_assessment"]["issues"])

    def test_selected_challenger_does_not_compare_only_with_itself(self):
        weak = _analysis(confidence=0.6)
        changed = _analysis(confidence=0.95)
        changed["motion"]["direction"] = "left"
        primary = FakeBackend("primary", [json.dumps(weak), json.dumps(weak), "{}"])
        challenger = FakeBackend("other", [json.dumps(changed)])
        result = GatedSequenceAnalyzer(primary, [challenger]).analyze(Path("/missing.mp4"), 1, activity={})
        self.assertTrue(result.escalation["candidate_disagreement"])
        self.assertEqual(result.status, "needs_review")

    def test_confident_primary_does_not_load_challenger(self) -> None:
        primary = FakeBackend("primary", [json.dumps(_analysis(confidence=0.91))])
        challenger = FakeBackend("challenger", [json.dumps(_analysis(confidence=0.95))])
        analyzer = GatedSequenceAnalyzer(primary, [challenger])

        with tempfile.TemporaryDirectory() as tmp:
            result = analyzer.analyze(Path(tmp) / "preview.mp4", 4.0, activity={})

        self.assertEqual(result.status, "accepted")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(challenger.calls), 0)
        self.assertFalse(result.escalation["triggered"])

    def test_uncertainty_uses_more_frames_and_challenger(self) -> None:
        weak = _analysis(confidence=0.41)
        weak["motion"]["speed"] = "still"
        dense = _analysis(confidence=0.86)
        challenger_result = _analysis(confidence=0.89)
        primary = FakeBackend(
            "primary",
            [json.dumps(weak), json.dumps(dense)],
        )
        challenger = FakeBackend("challenger", [json.dumps(challenger_result)])
        analyzer = GatedSequenceAnalyzer(primary, [challenger])

        with tempfile.TemporaryDirectory() as tmp:
            result = analyzer.analyze(
                Path(tmp) / "preview.mp4",
                4.0,
                activity={"temporal_change": 0.08, "transient_likelihood": 0.7},
            )

        self.assertEqual(result.status, "escalated")
        self.assertEqual(len(primary.calls), 2)
        self.assertEqual(len(challenger.calls), 1)
        self.assertGreater(primary.calls[1]["max_frames"], primary.calls[0]["max_frames"])
        self.assertGreater(primary.calls[1]["fps"], primary.calls[0]["fps"])
        self.assertEqual(result.escalation["challengers_attempted"], ["challenger"])

    def test_empty_parsed_challenger_does_not_create_disagreement(self) -> None:
        primary = FakeBackend(
            "primary",
            [json.dumps(_analysis(confidence=0.4)), json.dumps(_analysis(confidence=0.95))],
        )
        challenger = FakeBackend("challenger", ["{}"])
        analyzer = GatedSequenceAnalyzer(primary, [challenger])

        with tempfile.TemporaryDirectory() as tmp:
            result = analyzer.analyze(Path(tmp) / "preview.mp4", 1.0, activity={})

        self.assertEqual(result.status, "escalated")
        self.assertFalse(result.escalation["candidate_disagreement"])

    def test_ambiguous_atmospheric_material_uses_focused_specialist(self) -> None:
        value = _analysis(confidence=0.93)
        value["primary_family"] = "smoke"
        value["secondary_families"] = ["atmosphere", "snow"]
        value["effect_type"] = "rising_smoke_plume"
        value["summary"] = "A pale airborne plume rolls along a dark mountain ridge."
        specialist = {
            "recommended_primary_family": "atmosphere",
            "recommended_effect_type": "billowing_snow",
            "observed_materials": ["snow"],
            "morphology": "billowing",
            "revised_summary": "Billowing wind-blown snow drifts along a dark mountain ridge.",
            "search_terms": ["billowing snow", "mountain spindrift"],
            "evidence": ["fine granular snow lifts from the ridge"],
            "confidence": 0.94,
            "uncertainty": [],
        }
        primary = FakeImageBackend("primary", [json.dumps(value)], [json.dumps(specialist)])
        analyzer = GatedSequenceAnalyzer(primary, [])

        with tempfile.TemporaryDirectory() as tmp, patch(
            "vfx_element_tagger.gated_analysis.build_focused_storyboard",
            return_value=(object(), {"timestamps_seconds": [0.0, 0.5, 1.0]}),
        ):
            result = analyzer.analyze(Path(tmp) / "preview.mp4", 1.0, activity={})

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.analysis["primary_family"], "atmosphere")
        self.assertEqual(result.analysis["effect_type"], "billowing_snow")
        self.assertTrue(result.escalation["material_specialist"]["resolved"])
        self.assertIn("mountain spindrift", result.analysis["search_text"])

    def test_severely_weak_dense_result_uses_ultra_sampling_tier(self) -> None:
        primary = FakeBackend(
            "primary",
            [
                json.dumps(_analysis(confidence=0.35)),
                json.dumps(_analysis(confidence=0.42)),
                json.dumps(_analysis(confidence=0.92)),
            ],
        )
        analyzer = GatedSequenceAnalyzer(primary, [])

        with tempfile.TemporaryDirectory() as tmp:
            result = analyzer.analyze(Path(tmp) / "preview.mp4", 20.0, activity={})

        self.assertTrue(result.escalation["ultra_triggered"])
        self.assertEqual(len(primary.calls), 3)
        self.assertEqual(result.escalation["selected_role"], "primary_ultra")
        self.assertGreater(primary.calls[2]["max_frames"], primary.calls[1]["max_frames"])
        self.assertGreater(primary.calls[2]["fps"], primary.calls[1]["fps"])

    def test_narrow_action_window_forces_dense_confirmation(self) -> None:
        value = _analysis(confidence=0.94)
        value["evidence"] = [
            {"timestamp_seconds": 0.0, "observation": "idle handle"},
            {"timestamp_seconds": 15.0, "observation": "brief burst"},
            {"timestamp_seconds": 29.0, "observation": "idle tail"},
        ]
        primary = FakeBackend("primary", [json.dumps(value), json.dumps(value)])
        analyzer = GatedSequenceAnalyzer(primary, [])
        activity = {
            "duration_seconds": 30.0,
            "action_timing": {
                "available": True,
                "coverage_fraction": 0.03,
                "optimal_duration_seconds": 1.0,
                "event_segments": [{"start_seconds": 14.5, "end_seconds": 15.5}],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = analyzer.analyze(Path(tmp) / "preview.mp4", 30.0, activity=activity)

        self.assertEqual(len(primary.calls), 2)
        self.assertEqual(result.status, "escalated")
        self.assertIn(
            "activity_focus:narrow_action_window",
            result.escalation["sampling_escalation_reasons"],
        )

    def test_fine_detail_risk_uses_activity_focused_high_resolution_stills(self) -> None:
        detail = {
            "detail_observation": "Fine sparks split into small falling embers.",
            "observed_materials": ["electricity", "spark", "particle"],
            "texture": ["fine", "filament"],
            "density": "light",
            "instance_count": "continuous_many",
            "motion_character": ["branching", "falling", "flickering"],
            "temporal_arc": "instantaneous",
            "confidence": 0.93,
            "uncertainty": [],
        }
        primary = FakeImageBackend(
            "primary",
            [json.dumps(_analysis(confidence=0.94))],
            [json.dumps(detail)],
        )
        analyzer = GatedSequenceAnalyzer(primary, [])
        activity = {
            "fine_detail_likelihood": 0.82,
            "high_resolution_recommended": True,
            "analysis_focus": {"recommended_timestamps_seconds": [0.0, 0.2, 0.8, 1.0]},
        }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "vfx_element_tagger.gated_analysis.build_focused_storyboard",
            return_value=(object(), {"frame_count": 16, "selection": "activity_peaks_with_uniform_anchors"}),
        ):
            result = analyzer.analyze(
                Path(tmp) / "preview.mp4",
                1.0,
                activity=activity,
                detail_video_path=Path(tmp) / "source.mov",
            )

        self.assertEqual(len(primary.image_calls), 1)
        self.assertIn("spark", result.analysis["secondary_families"])
        self.assertIn("fine", result.analysis["appearance"]["texture"])
        self.assertTrue(result.escalation["high_resolution_detail_specialist"]["triggered"])

    def test_activity_conflict_rejects_still_label(self) -> None:
        value = _analysis(confidence=0.92)
        value["motion"]["speed"] = "still"
        value["motion"]["temporal_arc"] = "static"

        assessment = assess_analysis(
            value,
            {"temporal_change": 0.08, "transient_likelihood": 0.8},
        )

        self.assertFalse(assessment.accepted)
        self.assertTrue(any(issue.startswith("activity_conflict") for issue in assessment.issues))

    def test_coloured_effect_is_not_forced_into_plate_by_chroma(self) -> None:
        value = _analysis(confidence=0.95)

        assessment = assess_analysis(
            value,
            {"key_screen_likely": "green_screen"},
        )

        self.assertTrue(assessment.accepted)
        self.assertFalse(any(issue.startswith("key_screen_conflict") for issue in assessment.issues))

    def test_verified_key_plate_does_not_review_unknown_lighting(self) -> None:
        value = _analysis(confidence=0.95)
        value["primary_family"] = "plate"
        value["effect_type"] = "greenscreen_performance_plate"
        value["appearance"]["backing"] = "green_screen"
        value["appearance"]["lighting"] = "unknown"
        value["uncertainty"] = ["Lighting is unknown on the green screen plate."]

        assessment = assess_analysis(value, {"key_screen_likely": "green_screen"})

        self.assertTrue(assessment.accepted)
        self.assertNotIn("model_declared_field_uncertainty:lighting", assessment.issues)

    def test_screen_plate_subtype_requires_matching_declared_backing(self) -> None:
        value = _analysis(confidence=0.95)
        value["primary_family"] = "plate"
        value["effect_type"] = "greenscreen_performance_plate"
        value["appearance"]["backing"] = "environment"

        assessment = assess_analysis(value, {})

        self.assertFalse(assessment.accepted)
        self.assertIn(
            "screen_subtype_conflict:expected_green_screen_got_environment",
            assessment.issues,
        )

    def test_confident_production_plate_does_not_escalate_for_bursty_human_motion(self) -> None:
        value = _analysis(confidence=0.95)
        value["primary_family"] = "plate"
        value["appearance"]["backing"] = "green_screen"
        value["motion"]["temporal_arc"] = "continuous"
        value["motion"]["loopability"] = "unknown"
        value["evidence"] = [
            {"timestamp_seconds": 0.0, "observation": "performer starts centered"},
            {"timestamp_seconds": 15.0, "observation": "performer turns"},
            {"timestamp_seconds": 29.0, "observation": "performer returns to a pose"},
        ]
        primary = FakeBackend("primary", [json.dumps(value)])
        challenger = FakeBackend("challenger", [json.dumps(value)])
        analyzer = GatedSequenceAnalyzer(primary, [challenger])

        with tempfile.TemporaryDirectory() as tmp:
            result = analyzer.analyze(
                Path(tmp) / "preview.mp4",
                30.0,
                activity={
                    "duration_seconds": 30.0,
                    "transient_likelihood": 0.9,
                    "key_screen_likely": "green_screen",
                },
            )

        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.analysis["motion"]["loopability"], "not_loopable")
        self.assertEqual(len(challenger.calls), 0)

    def test_measured_multiple_windows_reject_single_event_count(self) -> None:
        value = _analysis(confidence=0.95)
        value["motion"]["event_count"] = "single"

        assessment = assess_analysis(
            value,
            {
                "action_timing": {
                    "available": True,
                    "pattern": "multiple_events",
                    "confidence": 0.9,
                }
            },
        )

        self.assertIn(
            "activity_conflict:measured_multiple_windows_but_event_count_single",
            assessment.issues,
        )

    def test_one_shot_explosion_cannot_claim_possible_loop(self) -> None:
        value = _analysis(confidence=0.95)
        value["primary_family"] = "explosion"
        value["motion"]["temporal_arc"] = "build_peak_decay"
        value["motion"]["loopability"] = "possible_loop"

        assessment = assess_analysis(value)

        self.assertIn(
            "semantic_conflict:one_shot_explosion_marked_loopable",
            assessment.issues,
        )

    def test_long_clip_requires_temporally_distributed_evidence(self) -> None:
        value = _analysis(confidence=0.9)

        assessment = assess_analysis(value, {"duration_seconds": 9.0})

        self.assertFalse(assessment.accepted)
        self.assertIn("insufficient_temporal_evidence", assessment.issues)

    def test_short_measured_event_uses_post_action_evidence_horizon(self) -> None:
        value = _analysis(confidence=0.9)
        value["evidence"] = [
            {"timestamp_seconds": 0.0, "observation": "particles appear"},
            {"timestamp_seconds": 0.8, "observation": "particles expand"},
            {"timestamp_seconds": 2.0, "observation": "black post-action handle"},
        ]

        assessment = assess_analysis(
            value,
            {
                "duration_seconds": 15.0,
                "action_timing": {
                    "available": True,
                    "coverage_fraction": 0.05,
                    "visible_end_seconds": 1.0,
                },
            },
        )

        self.assertTrue(assessment.accepted)
        self.assertNotIn("insufficient_temporal_evidence", assessment.issues)

    def test_long_decay_is_not_forced_to_be_a_transient(self) -> None:
        value = _analysis(confidence=0.9)
        value["motion"]["temporal_arc"] = "already_active_decay"
        value["motion"]["character"] = ["rising", "contracting"]
        value["evidence"] = [
            {"timestamp_seconds": 0.0, "observation": "strong fire"},
            {"timestamp_seconds": 10.0, "observation": "reduced fire"},
            {"timestamp_seconds": 19.0, "observation": "embers"},
        ]

        assessment = assess_analysis(
            value,
            {"duration_seconds": 19.4, "transient_likelihood": 0.8},
        )

        self.assertFalse(
            any(issue == "activity_conflict:transient_character_missing" for issue in assessment.issues)
        )

    def test_filename_impact_cue_triggers_visual_recheck_without_forcing_label(self) -> None:
        value = _analysis(confidence=0.9)
        value["summary"] = "A grey plume rises and disperses against black."
        value["search_text"] = "rising grey smoke plume"
        value["primary_family"] = "smoke"
        value["secondary_families"] = ["dust"]
        value["effect_type"] = "rising_smoke_plume"
        value["usability"]["best_uses"] = ["smoke layer", "atmosphere"]

        assessment = assess_analysis(
            value,
            context={"source_filename_hint": "Wall_Hit_Front_1.mov"},
        )

        self.assertFalse(assessment.accepted)
        self.assertIn(
            "filename_visual_disagreement:impact_cue_missing_from_analysis",
            assessment.issues,
        )

    def test_black_backed_edge_claim_is_checked_against_measured_border(self) -> None:
        value = _analysis(confidence=0.95)

        assessment = assess_analysis(
            value,
            activity={"bright_border_contact": ["top"]},
        )

        self.assertFalse(assessment.accepted)
        self.assertIn(
            "measured_edge_conflict:claimed_bottom_without_border_pixels",
            assessment.issues,
        )

    def test_measured_edge_refinement_can_resolve_review_status(self) -> None:
        value = _analysis(confidence=0.95)
        primary = FakeBackend("primary", [json.dumps(value)] * 3)
        analyzer = GatedSequenceAnalyzer(primary, [])

        with tempfile.TemporaryDirectory() as tmp:
            result = analyzer.analyze(
                Path(tmp) / "preview.mp4",
                None,
                activity={"bright_border_contact": ["top"]},
            )

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.analysis["composition"]["edge_contact"], ["top"])
        self.assertTrue(result.escalation["final_assessment"]["accepted"])

    def test_deterministic_refinement_uses_measured_alpha_plate_border(self) -> None:
        value = _analysis(confidence=0.95)
        value["appearance"]["backing"] = "transparent"

        refined, evidence = apply_deterministic_refinements(
            value,
            activity={"bright_border_contact": ["top"]},
            context={"has_alpha_channel": True},
        )

        self.assertEqual(refined["composition"]["edge_contact"], ["top"])
        self.assertEqual(evidence[0]["method"], "measured_bright_border_contact")

    def test_alpha_matte_input_does_not_become_literal_appearance(self) -> None:
        value = _analysis(confidence=0.95)
        value["appearance"].update(
            {
                "backing": "black",
                "colour": ["white"],
                "brightness": "bright",
                "lighting": "self_luminous",
            }
        )
        value["summary"] = (
            "A flock of small, white, bird-like silhouettes drifts against a black background."
        )
        value["search_text"] = "white birds black background, flocking birds"

        refined, evidence = apply_deterministic_refinements(
            value,
            context={"semantic_input_representation": "grayscale_alpha_matte"},
        )

        self.assertEqual(refined["appearance"]["backing"], "transparent")
        self.assertEqual(refined["appearance"]["colour"], ["unknown"])
        self.assertEqual(refined["appearance"]["brightness"], "unknown")
        self.assertNotIn("white", refined["summary"].lower())
        self.assertIn("transparent background", refined["summary"].lower())
        self.assertTrue(
            refined["search_text"].startswith(
                "birds on a transparent background, flocking birds"
            )
        )
        self.assertTrue(any(item["method"] == "alpha_matte_input_guard" for item in evidence))

    def test_continuous_alpha_subject_uses_its_visible_range_not_a_false_burst(self) -> None:
        value = _analysis(confidence=0.95)
        value["primary_family"] = "bird"
        value["motion"]["temporal_arc"] = "continuous"

        refined, _evidence = apply_deterministic_refinements(
            value,
            activity={
                "duration_seconds": 9.13,
                "source_fps": 24.0,
                "source_frame_count": 219,
                "action_timing": {
                    "available": True,
                    "visible_start_seconds": 0.0,
                    "main_action_start_seconds": 0.23,
                    "peak_seconds": 0.42,
                    "main_action_end_seconds": 0.65,
                    "visible_end_seconds": 7.02,
                    "optimal_start_seconds": 0.0,
                    "optimal_end_seconds": 0.9,
                    "optimal_duration_seconds": 0.9,
                    "coverage_fraction": 0.046,
                    "pattern": "burst",
                    "event_segments": [],
                    "confidence": 0.79,
                },
            },
        )

        timing = refined["action_timing"]
        self.assertEqual(timing["pattern"], "continuous")
        self.assertEqual(timing["optimal_start_seconds"], 0.0)
        self.assertEqual(timing["optimal_end_seconds"], 7.02)
        self.assertEqual(timing["optimal_end_frame"], 168)

    def test_deterministic_refinement_detects_top_lit_non_emissive_volume(self) -> None:
        value = _analysis(confidence=0.95)
        value["primary_family"] = "cloud"
        value["appearance"]["lighting"] = "unknown"

        refined, evidence = apply_deterministic_refinements(
            value,
            activity={"bright_border_contact": [], "vertical_luminance_bias": 0.018},
        )

        self.assertEqual(refined["appearance"]["lighting"], "top_lit")
        self.assertTrue(any(item["field"] == "appearance.lighting" for item in evidence))

    def test_branching_electricity_gets_fork_search_expansion(self) -> None:
        refined, evidence = apply_deterministic_refinements(_analysis(confidence=0.95))

        self.assertIn("multiple branches", refined["search_text"])
        self.assertTrue(any(item["field"] == "search_text" for item in evidence))

    def test_transient_explosion_loopability_is_canonicalized(self) -> None:
        value = _analysis(confidence=0.95)
        value["primary_family"] = "explosion"
        value["motion"]["temporal_arc"] = "build_peak_decay"
        value["motion"]["loopability"] = "possible_loop"
        value["motion"]["event_count"] = "single"

        refined, evidence = apply_deterministic_refinements(value)

        self.assertEqual(refined["motion"]["loopability"], "one_shot")
        self.assertTrue(any(item["field"] == "motion.loopability" for item in evidence))

    def test_activity_windows_do_not_overwrite_performance_take_count(self) -> None:
        value = _analysis(confidence=0.95)
        value["primary_family"] = "plate"
        value["appearance"]["backing"] = "green_screen"
        value["motion"]["temporal_arc"] = "continuous"
        value["motion"]["loopability"] = "unknown"
        value["motion"]["event_count"] = "single"

        refined, evidence = apply_deterministic_refinements(
            value,
            activity={
                "action_timing": {
                    "available": True,
                    "pattern": "multiple_events",
                    "event_segments": [],
                },
                "transient_likelihood": 0.9,
            },
        )
        assessment = assess_analysis(
            refined,
            activity={"transient_likelihood": 0.9},
        )

        self.assertEqual(refined["motion"]["loopability"], "not_loopable")
        self.assertEqual(refined["motion"]["event_count"], "single")
        self.assertFalse(
            any(issue == "activity_conflict:transient_signal_but_nontransient_arc" for issue in assessment.issues)
        )
        self.assertFalse(any(item["field"] == "motion.event_count" for item in evidence))

    def test_unstructured_challenger_evidence_is_adjudicated(self) -> None:
        weak = _analysis(confidence=0.4)
        dense = _analysis(confidence=0.84)
        adjudicated = _analysis(confidence=0.9)
        adjudicated["primary_family"] = "bird"
        adjudicated["effect_type"] = "circling_bird_flock"
        primary = FakeBackend(
            "primary",
            [json.dumps(weak), json.dumps(dense), json.dumps(adjudicated)],
        )
        challenger = FakeBackend(
            "dense-captioner",
            ["Scene: A flock of birds circles beneath an overcast sky. Events: birds turn."],
        )
        analyzer = GatedSequenceAnalyzer(primary, [challenger])

        with tempfile.TemporaryDirectory() as tmp:
            result = analyzer.analyze(Path(tmp) / "preview.mp4", 8.0, activity={})

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.analysis["primary_family"], "bird")
        self.assertEqual(result.escalation["selected_role"], "adjudicator")
        self.assertEqual(len(primary.calls), 3)
        self.assertEqual(len(challenger.calls), 1)
        self.assertIn("raw_evidence", primary.calls[2]["prompt"])

    def test_parser_accepts_fenced_json_and_normalizes_tokens(self) -> None:
        value = _analysis(confidence=0.84)
        value["primary_family"] = "Muzzle Flash"

        parsed = parse_analysis("```json\n" + json.dumps(value) + "\n```")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["primary_family"], "muzzle_flash")
        self.assertEqual(parsed["composition"]["instance_count"], "multiple")
        self.assertEqual(
            parsed["composition"]["notable_objects"][0],
            {"object": "street light", "location": "lower_right", "role": "foreground"},
        )
        self.assertEqual(parsed["motion"]["depth_motion"], "in_plane")
        self.assertEqual(parsed["motion"]["event_count"], "multiple")

    def test_duration_drives_whole_sequence_sampling_budget(self) -> None:
        self.assertEqual(sampling_plan(4.0).fps, 8.0)
        self.assertEqual(sampling_plan(12.0).max_frames, 64)
        self.assertEqual(sampling_plan(30.0).max_frames, 96)
        self.assertGreater(
            ultra_sampling_plan(30.0).max_frames,
            sampling_plan(30.0).max_frames,
        )


def _analysis(confidence: float) -> dict:
    return {
        "summary": "A fast branching lightning strike flashes vertically through the centre.",
        "search_text": "fast branching electrical lightning strike central vertical one shot",
        "primary_family": "electricity",
        "secondary_families": ["light"],
        "effect_type": "branching_lightning_strike",
        "composition": {
            "viewpoint": "front",
            "shot_scale": "wide",
            "origin": "top centre",
            "direction": "top to bottom",
            "edge_contact": ["top", "bottom"],
            "frame_coverage": "large",
            "spatial_distribution": "central",
            "instance_count": "multiple",
            "notable_objects": [
                {"object": "street light", "location": "lower right", "role": "foreground"}
            ],
        },
        "appearance": {
            "colour": ["white", "blue"],
            "brightness": "bright",
            "density": "light",
            "texture": ["filament"],
            "lighting": "self_luminous",
            "backing": "black",
        },
        "motion": {
            "temporal_arc": "instantaneous",
            "onset": "instant",
            "speed": "very_fast",
            "direction": "top to bottom",
            "character": ["branching", "flickering"],
            "depth_motion": "in_plane",
            "event_count": "multiple",
            "expansion": "slight",
            "loopability": "one_shot",
        },
        "usability": {
            "best_uses": ["lightning strike", "energy impact"],
            "isolation": "isolated",
            "cleanup_notes": ["touches top and bottom frame edges"],
            "layering": "foreground",
        },
        "evidence": [
            {"timestamp_seconds": 0.2, "observation": "A bright branching bolt appears."}
        ],
        "uncertainty": [],
        "field_confidence": {
            "family": confidence,
            "composition": confidence,
            "appearance": confidence,
            "motion": confidence,
            "usability": confidence,
        },
        "overall_confidence": confidence,
    }


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.models import Element, SourceRepresentation
from vfx_element_tagger.review import (
    apply_saved_human_overrides,
    build_review_context,
    review_element,
)
from vfx_element_tagger.review_ui import review_html


class ReviewTests(unittest.TestCase):
    def test_context_turns_uncertainty_and_candidate_disagreement_into_choices(self) -> None:
        element = _element()
        element.semantic_analysis["uncertainty"] = ["Rain vs. mist is ambiguous at this scale."]
        element.analysis_escalation = {
            "candidate_disagreement": True,
            "final_assessment": {"issues": ["model_declared_material_uncertainty"]},
            "selected_role": "challenger",
        }
        alternate = dict(element.semantic_analysis)
        alternate["primary_family"] = "atmosphere"
        alternate["effect_type"] = "falling_mist"
        element.analysis_candidates = [
            {
                "role": "challenger",
                "model": "local/challenger",
                "confidence": 0.88,
                "result": alternate,
            }
        ]

        context = build_review_context(element)

        paths = {field["path"] for field in context["review_fields"]}
        family = next(field for field in context["review_fields"] if field["path"] == "primary_family")
        subtype = next(field for field in context["review_fields"] if field["path"] == "effect_type")
        self.assertIn("effect_type", paths)
        self.assertEqual([option["value"] for option in family["options"]], ["rain", "atmosphere"])
        self.assertIn("bird", family["vocabulary"])
        self.assertIn("falling_rain", subtype["vocabulary"])
        self.assertIn("flocking_birds", subtype["vocabulary_by_family"]["bird"])
        self.assertTrue(any("unsure" in reason for reason in context["review_reasons"]))

    def test_review_persists_normalized_human_overrides_and_accepts_element(self) -> None:
        element = _element()

        record = review_element(
            element,
            {
                "primary_family": "Atmospheric Rain",
                "effect_type": "Fine Falling Mist",
                "composition.edge_contact": ["Top Edge", "Bottom Edge"],
            },
            note="Artist confirmed against the moving preview.",
        )

        self.assertEqual(element.analysis_status, "accepted")
        self.assertEqual(element.primary_family, "atmospheric_rain")
        self.assertEqual(element.semantic_analysis["effect_type"], "fine_falling_mist")
        self.assertEqual(
            element.semantic_analysis["composition"]["edge_contact"],
            ["top_edge", "bottom_edge"],
        )
        self.assertEqual(element.human_overrides["fields"]["primary_family"], "atmospheric_rain")
        self.assertEqual(element.human_overrides["review_note"], "Artist confirmed against the moving preview.")
        self.assertTrue(element.provenance["semantic_analysis"]["human_reviewed"])
        self.assertEqual(record["previous_values"]["primary_family"], "rain")
        self.assertIn("fine falling mist", element.caption)
        self.assertIn("atmospheric rain", element.semantic_analysis["search_text"])
        self.assertEqual(len(element.caption_embed), 64)
        self.assertIn("human-reviewed semantic refresh", element.embedding_model_version)

    def test_saved_overrides_win_over_later_model_values(self) -> None:
        element = _element()
        element.human_overrides = {
            "fields": {"primary_family": "snow", "effect_type": "wide_falling_snow"}
        }
        element.semantic_analysis["primary_family"] = "rain"
        element.semantic_analysis["effect_type"] = "falling_rain"

        apply_saved_human_overrides(element)

        self.assertEqual(element.primary_family, "snow")
        self.assertEqual(element.semantic_analysis["effect_type"], "wide_falling_snow")
        self.assertEqual(element.qwen_tags["effect_type"], "wide_falling_snow")

    def test_changed_primary_removes_duplicate_and_infers_material_secondary(self) -> None:
        element = _element()
        element.semantic_analysis["primary_family"] = "smoke"
        element.semantic_analysis["secondary_families"] = ["atmosphere"]
        element.semantic_analysis["effect_type"] = "billowing_smoke_plume"

        review_element(
            element,
            {"primary_family": "atmosphere", "effect_type": "billowing_snow"},
        )

        self.assertEqual(element.secondary_families, ["snow"])
        self.assertEqual(element.semantic_analysis["secondary_families"], ["snow"])

    def test_review_rejects_unapproved_field(self) -> None:
        element = _element()

        with self.assertRaisesRegex(ValueError, "not reviewable"):
            review_element(element, {"technical.fps": 30})

    def test_review_page_has_queue_uncertainty_choices_and_custom_input(self) -> None:
        html = review_html()

        self.assertIn("REVIEW QUEUE", html)
        self.assertIn("What needs confirmation", html)
        self.assertIn("Custom value", html)
        self.assertIn("Library vocabulary", html)
        self.assertIn("Choose a predefined value", html)
        self.assertIn("Custom value — last resort", html)
        self.assertIn("Confirm choices & mark reviewed", html)
        self.assertIn("frames ${start}–${end}", html)
        self.assertIn('data-preview-channel="alpha"', html)
        self.assertIn("Source on disk", html)
        self.assertIn('aria-label="Copy source file path"', html)
        self.assertIn("/api/review/", html)


def _element() -> Element:
    representation = SourceRepresentation(
        representation_id="rep",
        path="/library/DistantRain-001.mov",
        original_path="/library/DistantRain-001.mov",
        format="mov",
        is_primary=True,
    )
    return Element(
        element_id="rain-1",
        source_representations=[representation],
        primary_representation_id="rep",
        primary_family="rain",
        category="rain",
        analysis_status="needs_review",
        analysis_confidence=0.85,
        semantic_analysis={
            "primary_family": "rain",
            "secondary_families": ["atmosphere"],
            "effect_type": "falling_rain",
            "summary": "Fine rain falls across a dark frame.",
            "composition": {
                "viewpoint": "front",
                "shot_scale": "wide",
                "edge_contact": ["top", "bottom"],
            },
            "motion": {
                "temporal_arc": "continuous",
                "event_count": "continuous_many",
                "loopability": "possible_loop",
            },
            "uncertainty": [],
        },
    )


if __name__ == "__main__":
    unittest.main()

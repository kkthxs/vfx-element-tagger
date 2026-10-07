from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.models import Element, SourceRepresentation
from vfx_element_tagger.web import _element_summary, _parse_byte_range
from vfx_element_tagger.web_ui import index_html


class WebUITests(unittest.TestCase):
    def test_summary_exposes_artist_taxonomy_and_timing(self) -> None:
        representation = SourceRepresentation(
            representation_id="rep",
            path="/library/Ground_Dust_Hit.mov",
            original_path="/library/Ground_Dust_Hit.mov",
            format="mov",
            is_primary=True,
        )
        element = Element(
            element_id="dust-1",
            source_representations=[representation],
            primary_representation_id="rep",
            primary_family="dust",
            secondary_families=["debris"],
            semantic_analysis={
                "summary": "A fast ground dust hit spreads outwards.",
                "effect_type": "ground_dust_hit",
                "composition": {"viewpoint": "front", "edge_contact": ["bottom"]},
                "motion": {"speed": "fast"},
                "action_timing": {"available": True, "optimal_start_seconds": 1.0},
            },
            analysis_status="accepted",
            analysis_confidence=0.93,
        )

        summary = _element_summary(element, Path("/tmp/cache"))

        self.assertEqual(summary["name"], "Ground_Dust_Hit")
        self.assertEqual(summary["primary_family"], "dust")
        self.assertEqual(summary["effect_type"], "ground_dust_hit")
        self.assertEqual(summary["secondary_families"], ["debris"])
        self.assertEqual(summary["composition"]["edge_contact"], ["bottom"])
        self.assertEqual(summary["action_timing"]["optimal_start_seconds"], 1.0)
        self.assertEqual(summary["analysis_status"], "accepted")

    def test_unanalysed_snow_gets_a_provisional_browse_family(self) -> None:
        representation = SourceRepresentation(
            representation_id="rep",
            path="/library/Falling_Snow_Wide.mov",
            original_path="/library/Falling_Snow_Wide.mov",
            format="mov",
            is_primary=True,
        )
        element = Element(
            element_id="snow-1",
            source_representations=[representation],
            primary_representation_id="rep",
            category="unknown",
            primary_family="unknown",
            analysis_status="unanalysed",
        )

        summary = _element_summary(element, Path("/tmp/cache"))

        self.assertEqual(summary["primary_family"], "snow")
        self.assertEqual(summary["analysis_status"], "unanalysed")

    def test_summary_advertises_alpha_preview_only_for_visible_alpha(self) -> None:
        representation = SourceRepresentation(
            representation_id="rep",
            path="/library/Birdies_Transparent_B.mov",
            original_path="/library/Birdies_Transparent_B.mov",
            format="mov",
            is_primary=True,
        )
        element = Element(
            element_id="birdies alpha",
            source_representations=[representation],
            primary_representation_id="rep",
            has_alpha_channel=True,
            alpha_non_empty=True,
        )

        summary = _element_summary(element, Path("/tmp/cache"))

        self.assertEqual(summary["alpha_preview_url"], "/api/alpha-preview/birdies%20alpha")
        self.assertTrue(summary["alpha_non_empty"])
        element.alpha_non_empty = False
        self.assertIsNone(_element_summary(element, Path("/tmp/cache"))["alpha_preview_url"])

    def test_interface_contains_preview_taxonomy_and_artist_metadata_sections(self) -> None:
        html = index_html([])

        self.assertIn('id="family-list"', html)
        self.assertIn('data-testid="element-card"', html)
        self.assertIn("Hover preview", html)
        self.assertIn('data-preview-channel="alpha"', html)
        self.assertIn("Show the grayscale alpha matte", html)
        self.assertIn("Source on disk", html)
        self.assertIn('aria-label="Copy source file path"', html)
        self.assertIn("Path copied to clipboard", html)
        self.assertIn("Organisational tags", html)
        self.assertIn("Suggested action range", html)
        self.assertIn("Timing not artist-verified", html)
        self.assertIn("Model confidence (uncalibrated)", html)
        self.assertIn("Frames ${frames.start}–${frames.end}", html)
        self.assertIn("Seconds ${formatSeconds", html)
        self.assertIn("Technical metadata", html)
        self.assertIn('href="/review"', html)
        self.assertIn('id="review-count"', html)
        self.assertIn('id="sort-filter"', html)
        self.assertIn("Artist quality rating", html)
        self.assertIn("data-rating", html)
        self.assertIn("Show more", html)

    def test_preview_byte_ranges_support_open_ended_and_suffix_requests(self) -> None:
        self.assertEqual(_parse_byte_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(_parse_byte_range("bytes=90-", 100), (90, 99))
        self.assertEqual(_parse_byte_range("bytes=-12", 100), (88, 99))

    def test_preview_byte_ranges_reject_invalid_requests(self) -> None:
        for value in ("items=0-1", "bytes=100-120", "bytes=20-10", "bytes=0-1,3-4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _parse_byte_range(value, 100)


if __name__ == "__main__":
    unittest.main()

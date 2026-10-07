from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.category_resolver import (  # noqa: E402
    resolve_caption_category,
    resolve_initial_category,
)


class CategoryResolverTests(unittest.TestCase):
    def test_low_confidence_key_screen_initial_category_becomes_plate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            poster = Path(tmp) / "poster.jpg"
            pixels = np.zeros((90, 160, 3), dtype=np.uint8)
            pixels[:, :] = [25, 190, 35]
            Image.fromarray(pixels).save(poster)

            category, confidence, provenance = resolve_initial_category("lens_effect", 0.43, poster)

        self.assertEqual(category, "plate")
        self.assertGreaterEqual(confidence, 0.6)
        self.assertEqual(provenance["siglip_top_category"], "lens_effect")

    def test_caption_can_correct_bad_category_to_plate(self) -> None:
        correction = resolve_caption_category(
            "lens_effect",
            0.7,
            "A male performer turns in a wide green screen studio with clean chroma key backing.",
            {"subject": {"value": "person"}, "backing": {"value": "green_screen"}},
        )

        self.assertIsNotNone(correction)
        assert correction is not None
        self.assertEqual(correction[0], "plate")
        self.assertEqual(correction[2]["previous_category"], "lens_effect")

    def test_keyable_effect_without_subject_does_not_become_plate(self) -> None:
        correction = resolve_caption_category(
            "dust",
            0.7,
            "A keyable dust burst against a green screen backing. The subject is centrally framed.",
            {"backing": {"value": "green_screen"}, "keyability": {"value": "keyable"}},
        )

        self.assertIsNone(correction)

    def test_blue_screen_operator_with_rpg_corrects_to_plate(self) -> None:
        correction = resolve_caption_category(
            "practical_light",
            0.45,
            "A wide blue screen plate of an operator firing an RPG rocket launcher with smoke backblast.",
            {"backing": {"value": "blue_screen"}},
        )

        self.assertIsNotNone(correction)
        assert correction is not None
        self.assertEqual(correction[0], "plate")
        self.assertEqual(correction[2]["previous_category"], "practical_light")


if __name__ == "__main__":
    unittest.main()

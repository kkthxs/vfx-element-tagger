from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.artifacts import generate_alpha_preview
from vfx_element_tagger.models import SourceRepresentation


class AlphaPreviewTests(unittest.TestCase):
    def test_generates_high_quality_grayscale_matte_and_reuses_cache(self) -> None:
        representation = SourceRepresentation(
            representation_id="rep",
            path="/library/Birdies_Transparent_B.mov",
            original_path="/library/Birdies_Transparent_B.mov",
            format="mov",
            is_primary=True,
        )
        commands: list[list[str]] = []

        def fake_run(command: list[str], timeout: int):
            commands.append(command)
            Path(command[-1]).write_bytes(b"alpha-preview")
            return True, ""

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with (
                patch("vfx_element_tagger.artifacts.tool_exists", return_value=True),
                patch("vfx_element_tagger.artifacts.run_command", side_effect=fake_run),
            ):
                first = generate_alpha_preview("birdies", representation, cache_dir)
                second = generate_alpha_preview("birdies", representation, cache_dir)

            self.assertEqual(first, second)
            self.assertEqual(len(commands), 1)
            self.assertIn("alphaextract", commands[0][commands[0].index("-vf") + 1])
            self.assertIn("1080", commands[0][commands[0].index("-vf") + 1])
            self.assertEqual(commands[0][commands[0].index("-crf") + 1], "12")
            self.assertTrue(Path(first or "").exists())

    def test_returns_none_when_ffmpeg_is_unavailable(self) -> None:
        representation = SourceRepresentation(
            representation_id="rep",
            path="/library/element.mov",
            original_path="/library/element.mov",
            format="mov",
            is_primary=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("vfx_element_tagger.artifacts.tool_exists", return_value=False):
                self.assertIsNone(
                    generate_alpha_preview("element", representation, Path(tmp))
                )


if __name__ == "__main__":
    unittest.main()

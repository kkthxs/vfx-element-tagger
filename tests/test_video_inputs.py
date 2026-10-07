from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vfx_element_tagger.video_inputs import multimodal_positions, spatial_budget, timestamp_video_tokens


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|video_pad|>": 9, "<|vision_start|>": 7, "<|vision_end|>": 8}[token]

    def encode(self, text, add_special_tokens=False):
        return [100 + round(float(text.split()[0][1:]) * 1000)]


class VideoInputsTests(unittest.TestCase):
    def test_timestamp_groups_use_actual_nonuniform_times(self):
        ids, grids, times = timestamp_video_tokens(
            [1, 7] + [9]*12 + [8, 2], Tokenizer(), [3, 4, 4], [0, 0.1, 1.2, 2.4, 9.9])
        for actual, expected in zip(times, [0.05, 1.8, 9.9]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(grids, [[1, 4, 4]]*3)
        self.assertEqual(ids.count(9), 12)
        self.assertEqual(ids.count(7), 3)
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 2)

    def test_timestamp_alignment_fails_closed(self):
        for ids, grid, times in [([9]*3, [1, 4, 4], [0, 1]),
                                 ([9]*4, [1, 4, 4], [0, 1, 2]),
                                 ([9, 9, 2, 9, 9], [1, 4, 4], [0, 1])]:
            with self.subTest(ids=ids, times=times), self.assertRaises(ValueError):
                timestamp_video_tokens(ids, Tokenizer(), grid, times)

    def test_rope_keeps_separate_video_blocks_and_spatial_coordinates(self):
        row = [1, 7, 9, 9, 9, 9, 8, 5, 7, 9, 9, 9, 9, 8]
        positions, deltas = multimodal_positions([row], [], [[1, 4, 4]]*2, 10, 9)
        self.assertEqual(positions[0][0][2:6], [2, 2, 2, 2])
        self.assertEqual(positions[1][0][2:6], [2, 2, 3, 3])
        self.assertEqual(positions[2][0][2:6], [2, 3, 2, 3])
        self.assertEqual(positions[0][0][9], 7)
        self.assertEqual(deltas, [[-4]])

    def test_rope_rejects_mismatched_visual_blocks(self):
        with self.assertRaises(ValueError):
            multimodal_positions([[9, 9, 2]], [], [[1, 4, 4]], 10, 9)

    def test_spatial_budget_preserves_detail_and_bounds_visual_tokens(self):
        for width, height in [(1920, 1080), (1080, 1920), (4096, 512), (320, 240)]:
            for frames in [4, 64, 128, 256]:
                with self.subTest(width=width, frames=frames):
                    w, h, n = spatial_budget(width, height, frames)
                    self.assertEqual(w % 32, 0)
                    self.assertEqual(h % 32, 0)
                    self.assertGreaterEqual(min(w, h), min(384, min(width, height)//32*32))
                    self.assertLessEqual(((n+1)//2)*(w//32)*(h//32), 16384)
                    self.assertLessEqual(n, frames)

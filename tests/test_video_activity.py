from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.video_activity import (  # noqa: E402
    _key_screen_choice,
    add_frame_ranges,
    derive_action_timing,
    profile_video,
    recommend_analysis_timestamps,
    foreground_change_fraction,
)


class ActionTimingTests(unittest.TestCase):
    def test_sparse_moving_particles_are_not_diluted_by_black_backing(self):
        import numpy as np
        first = np.zeros((1080, 1920), dtype=np.uint8)
        second = first.copy()
        first[100:104, 100:104] = 80
        second[110:114, 100:104] = 80
        self.assertGreater(foreground_change_fraction(first, second), 0.9)
        self.assertEqual(foreground_change_fraction(first, first), 0)

    def test_brief_flash_survives_longer_later_action(self):
        timestamps = [i/30 for i in range(600)]
        signal = [0.0]*600
        signal[267] = 0.9
        signal[360:450] = [0.6]*90
        timing = derive_action_timing(timestamps, signal, 20)
        self.assertLess(timing["optimal_start_seconds"], 8.9)
        self.assertGreater(timing["optimal_end_seconds"], 14)
        self.assertTrue(timing["review_required"])

    def test_key_screen_requires_broad_or_border_dominant_chroma(self) -> None:
        self.assertIsNone(_key_screen_choice(0.0, 0.178, 0.0, 0.25))
        self.assertEqual(_key_screen_choice(0.15, 0.0, 0.82, 0.0), "green_screen")
        self.assertEqual(_key_screen_choice(0.92, 0.02, 0.95, 0.04), "green_screen")

    def test_activity_focus_retains_uniform_context_around_peak_frames(self) -> None:
        timestamps = [float(index) for index in range(21)]
        signal = [0.0] * 21
        signal[10] = 1.0
        timing = derive_action_timing(timestamps, signal, 20.0)

        focus = recommend_analysis_timestamps(timestamps, signal, 20.0, timing, limit=16)

        for anchor in (0.0, 5.0, 10.0, 15.0, 20.0):
            self.assertIn(anchor, focus)
        self.assertLessEqual(len(focus), 16)

    def test_default_profile_measures_every_proxy_frame(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.avi"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                12.0,
                (96, 64),
            )
            if not writer.isOpened():
                self.skipTest("MJPG test writer is unavailable")
            for index in range(12):
                frame = np.zeros((64, 96, 3), dtype=np.uint8)
                cv2.circle(frame, (8 + index * 6, 32), 3, (255, 255, 255), -1)
                writer.write(frame)
            writer.release()

            profile = profile_video(path, target_dimension=64)

        self.assertTrue(profile["available"])
        self.assertEqual(profile["profiling_mode"], "every_proxy_frame")
        self.assertEqual(profile["profiled_frames"], 12)
        self.assertEqual(profile["action_timing"]["sample_count"], 12)
        self.assertTrue(profile["analysis_focus"]["uniform_anchors_included"])

    def test_quick_middle_burst_gets_tight_optimal_range(self) -> None:
        timestamps = [index * 0.3 for index in range(101)]
        signal = [0.0] * 101
        signal[49:53] = [0.2, 1.0, 0.8, 0.2]

        timing = derive_action_timing(timestamps, signal, 30.0)

        self.assertEqual(timing["pattern"], "burst")
        self.assertLess(timing["optimal_start_seconds"], 15.0)
        self.assertGreater(timing["optimal_end_seconds"], 15.0)
        self.assertLess(timing["optimal_duration_seconds"], 4.0)

    def test_continuous_signal_uses_full_clip(self) -> None:
        timestamps = [float(index) for index in range(11)]
        signal = [0.5] * 11

        timing = derive_action_timing(timestamps, signal, 10.0)

        self.assertEqual(timing["pattern"], "continuous")
        self.assertEqual(timing["optimal_start_seconds"], 0.0)
        self.assertEqual(timing["optimal_end_seconds"], 10.0)

    def test_separated_bursts_are_retained_as_multiple_events(self) -> None:
        timestamps = [float(index) for index in range(21)]
        signal = [0.0] * 21
        signal[4:7] = [0.3, 1.0, 0.3]
        signal[14:17] = [0.3, 0.9, 0.3]

        timing = derive_action_timing(timestamps, signal, 20.0)

        self.assertEqual(timing["pattern"], "multiple_events")
        self.assertEqual(len(timing["event_segments"]), 2)

    def test_action_range_includes_zero_based_inclusive_source_frames(self) -> None:
        timing = derive_action_timing(
            [index * 0.5 for index in range(21)],
            [0.0] * 8 + [0.2, 1.0, 0.2] + [0.0] * 10,
            10.0,
        )

        timing = add_frame_ranges(timing, 25.0, 250)

        self.assertEqual(timing["frame_numbering"], "zero_based_inclusive")
        self.assertEqual(timing["source_fps"], 25.0)
        self.assertGreaterEqual(timing["optimal_start_frame"], 0)
        self.assertLessEqual(timing["optimal_end_frame"], 249)
        self.assertEqual(
            timing["optimal_duration_frames"],
            timing["optimal_end_frame"] - timing["optimal_start_frame"] + 1,
        )
        self.assertIn("start_frame", timing["event_segments"][0])


if __name__ == "__main__":
    unittest.main()

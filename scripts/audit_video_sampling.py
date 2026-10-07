#!/usr/bin/env python3
"""Inspect the frames and time markers actually presented to the installed VLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vfx_element_tagger.video_inputs import sample_video, TimestampedVideoProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from mlx_vlm.utils import load_config, load_processor, prepare_inputs
    from mlx_vlm.prompt_utils import apply_chat_template
    import mlx_vlm.models.qwen3_5  # Registers the installed processor compatibility adapter.

    processor = load_processor(str(args.model_path), add_detokenizer=False)
    config = load_config(args.model_path)
    reports = []
    for fps, cap in [(4.0, 64), (8.0, 128)]:
        video, sampling = sample_video(args.video, fps, cap)
        effective_fps = len(video) * sampling["source_fps"] / sampling["source_frame_count"]
        prompt = apply_chat_template(processor, config, "Describe the video.",
                                     video=[str(args.video)], fps=effective_fps, enable_thinking=False)
        vp = processor.video_processor
        budget = vp.max_pixels
        try:
            vp.max_pixels = len(video)*video.shape[2]*video.shape[3]
            inputs = prepare_inputs(TimestampedVideoProcessor(processor, sampling), prompts=prompt,
                                    videos=[video], fps=effective_fps, add_special_tokens=True)
        finally:
            vp.max_pixels = budget
        ids = inputs["input_ids"].tolist()[0]
        text = processor.tokenizer.decode(ids)
        times = [float(t) for t in re.findall(r"<([0-9.]+) seconds>", text)]
        grid = inputs.get("video_grid_thw")
        report = {**sampling, "requested_fps": fps, "max_frames": cap, "decoded_frames": len(video),
                  "effective_fps": effective_fps, "input_tokens": len(ids),
                  "video_grid_thw": grid.tolist() if hasattr(grid, "tolist") else grid,
                  "time_marker_count": len(times), "first_timestamp": min(times) if times else None,
                  "last_timestamp": max(times) if times else None,
                  "expected_duration": len(video) / effective_fps}
        reports.append(report)
        print(json.dumps(report), flush=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()

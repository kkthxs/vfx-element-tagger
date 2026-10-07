"""Exact video sampling and timestamp alignment for local Qwen video inputs."""
from __future__ import annotations

import math

VIDEO_INPUT_VERSION = "timestamped-native-video-1"


def spatial_budget(width, height, frame_count, max_tokens=16_384):
    aspect = width / max(height, 1)
    minimum = min(384, width, height)
    maximum = min(480, width, height)
    for short in range(max(32, int(maximum)//32*32), max(0, int(minimum)//32*32-1), -32):
        h = short if aspect >= 1 else max(32, round(short/aspect/32)*32)
        w = max(32, round(short*aspect/32)*32) if aspect >= 1 else short
        tokens = math.ceil(frame_count/2) * (h//16) * (w//16) // 4
        if tokens <= max_tokens:
            return w, h, frame_count
    short = max(32, int(minimum)//32*32)
    h = short if aspect >= 1 else max(32, round(short/aspect/32)*32)
    w = max(32, round(short*aspect/32)*32) if aspect >= 1 else short
    count = max(2, min(frame_count, max_tokens * 2 // max(1, (h//16)*(w//16)//4)))
    return w, h, count//2*2


def sample_video(path, fps, max_frames, focus_times=()):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total < 1 or source_fps <= 0:
            raise ValueError("Cannot ground video timestamps without frame count and FPS")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = min(total, max_frames, max(4, int(math.ceil(total/source_fps*fps))))
        width, height, count = spatial_budget(width, height, count)
        count = min(total, count)
        # Uniform anchors always cover both ends; measured peaks add attention,
        # not semantic labels. Never approximate source time from requested FPS.
        focus = sorted({max(0, min(total-1, round(float(t)*source_fps))) for t in focus_times})
        if len(focus) > count//3:
            focus = [focus[i] for i in np.linspace(0, len(focus)-1, max(1, count//3)).round().astype(int)]
        indices = set(np.linspace(0, total-1, max(1, count-len(focus))).round().astype(int))
        indices.update(focus)
        frames, actual = [], []
        for index in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Video decode failed at source frame {index}")
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            actual.append(int(index))
        video = np.stack(frames).transpose(0, 3, 1, 2)
        report = {"version": VIDEO_INPUT_VERSION, "source_fps": source_fps,
                  "source_frame_count": total, "sampled_frame_indices": actual,
                  "timestamps_seconds": [round(i/source_fps, 6) for i in actual],
                  "decoded_frames": len(actual), "width": width, "height": height,
                  "requested_fps": fps, "requested_max_frames": max_frames}
        return video, report
    finally:
        cap.release()


def timestamp_video_tokens(ids, tokenizer, grid, timestamps, temporal_patch_size=2, merge_size=2):
    video_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    groups, h, w = map(int, grid)
    per_group = h*w//(merge_size**2)
    positions = [i for i, token in enumerate(ids) if token == video_id]
    if len(positions) != groups*per_group or not positions:
        raise ValueError("Video tokens do not match the processed visual grid")
    first, last = positions[0], positions[-1]
    if positions != list(range(first, last+1)):
        raise ValueError("Unexpected pre-expanded video layout; refusing ambiguous time alignment")
    if len(timestamps) > groups*temporal_patch_size or len(timestamps) <= (groups-1)*temporal_patch_size:
        raise ValueError("Timestamp count does not match temporal patch grouping")
    if first > 0 and ids[first-1] == start_id:
        first -= 1
    if last+1 < len(ids) and ids[last+1] == end_id:
        last += 1
    replacement, group_times = [], []
    padded = list(timestamps) + [timestamps[-1]] * (groups*temporal_patch_size-len(timestamps))
    for i in range(groups):
        times = padded[i*temporal_patch_size:(i+1)*temporal_patch_size]
        seconds = sum(times)/len(times)
        group_times.append(seconds)
        replacement.extend(tokenizer.encode(f"<{seconds:.3f} seconds>", add_special_tokens=False))
        replacement.extend([start_id] + [video_id]*per_group + [end_id])
    return ids[:first] + replacement + ids[last+1:], [[1, h, w] for _ in range(groups)], group_times


class TimestampedVideoProcessor:
    def __init__(self, processor, report):
        self.processor = processor
        self.report = report

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        import mlx.core as mx

        inputs = self.processor(images=images, text=text, videos=videos, **kwargs)
        if videos is None:
            return inputs
        rows = inputs["input_ids"].tolist()
        grids = inputs["video_grid_thw"].tolist()
        if len(rows) != 1 or len(grids) != 1:
            raise ValueError("Timestamp adapter currently supports one video per request")
        vp = self.processor.video_processor
        ids, grids, times = timestamp_video_tokens(
            rows[0], self.processor.tokenizer, grids[0], self.report["timestamps_seconds"],
            vp.temporal_patch_size, vp.merge_size,
        )
        inputs["input_ids"] = mx.array([ids], dtype=mx.int32)
        inputs["attention_mask"] = mx.ones((1, len(ids)), dtype=mx.int32)
        inputs["video_grid_thw"] = mx.array(grids, dtype=mx.int32)
        if "mm_token_type_ids" in inputs:
            vid = self.processor.video_token_id
            inputs["mm_token_type_ids"] = mx.array([[2 if token == vid else 0 for token in ids]])
        self.report.update(timestamp_markers=len(times), temporal_group_timestamps=times,
                           visual_tokens=sum(g[1]*g[2]//vp.merge_size**2 for g in grids),
                           video_grid_thw=grids)
        return inputs


def multimodal_positions(rows, image_grids, video_grids, image_id, video_id, merge_size=2, masks=None):
    """Build mRoPE positions without the installed adapter's summed-index bug."""
    positions = [[[1]*len(row) for row in rows] for _ in range(3)]
    grid_offsets = {image_id: 0, video_id: 0}
    grids = {image_id: image_grids, video_id: video_grids}
    deltas = []
    for batch, row in enumerate(rows):
        active = [i for i in range(len(row)) if masks is None or masks[batch][i]]
        offset, current = 0, 0
        while offset < len(active):
            token = row[active[offset]]
            if token not in grids:
                for axis in range(3):
                    positions[axis][batch][active[offset]] = current
                offset += 1
                current += 1
                continue
            available = grids[token]
            index = grid_offsets[token]
            if index >= len(available):
                raise ValueError("Missing visual grid for mRoPE token block")
            t, h, w = map(int, available[index])
            h, w = h//merge_size, w//merge_size
            length = t*h*w
            block = active[offset:offset+length]
            if len(block) != length or any(row[i] != token for i in block):
                raise ValueError("Visual token block and mRoPE grid differ")
            for j, col in enumerate(block):
                for axis, coordinate in enumerate((j//(h*w), (j//w)%h, j%w)):
                    positions[axis][batch][col] = current + coordinate
            current += max(t, h, w)
            offset += length
            grid_offsets[token] += 1
        deltas.append([current-len(row)])
    return positions, deltas


def patch_qwen_rope(model):
    import types
    import mlx.core as mx

    language = model.language_model
    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        config = self.config
        positions, deltas = multimodal_positions(
            input_ids.tolist(), [] if image_grid_thw is None else image_grid_thw.tolist(),
            [] if video_grid_thw is None else video_grid_thw.tolist(),
            config.image_token_id, config.video_token_id, config.vision_config.spatial_merge_size,
            None if attention_mask is None else attention_mask.tolist(),
        )
        return mx.array(positions, dtype=mx.int32), mx.array(deltas, dtype=mx.int32)
    language.get_rope_index = types.MethodType(get_rope_index, language)

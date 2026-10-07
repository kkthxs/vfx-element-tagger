from __future__ import annotations

import math
from pathlib import Path
from typing import Any


ACTIVITY_PROFILE_VERSION = "sequential-proxy-activity-0.8"
ACTION_TIMING_VERSION = "per-frame-action-range-0.4"


def profile_video(
    path: Path,
    max_samples: int | None = None,
    *,
    max_profile_frames: int = 30_000,
    target_dimension: int = 192,
) -> dict[str, Any]:
    """Cheap sequential activity profiling for a low-resolution proxy.

    Every decoded proxy frame is measured for normal clips. Very long sources are
    still decoded sequentially but expensive pixel measurements are strided to a
    bounded ``max_profile_frames``. Only scalar signals are retained, so memory use
    does not grow with frame resolution. ``max_samples`` remains as a backwards-
    compatible explicit cap; the default no longer skips through only 96 frames.

    This deliberately avoids optical flow and semantic labels. Sparse, flickering
    and non-rigid VFX routinely violate optical-flow assumptions; simple change,
    occupancy and edge statistics are safer attention signals for the VLM gate.
    """

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return {"version": ACTIVITY_PROFILE_VERSION, "available": False, "reason": "opencv_missing"}

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"version": ACTIVITY_PROFILE_VERSION, "available": False, "reason": "decode_failed"}
    try:
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            return {"version": ACTIVITY_PROFILE_VERSION, "available": False, "reason": "no_frames"}
        requested_cap = max_samples if max_samples is not None else max_profile_frames
        requested_cap = max(2, int(requested_cap))
        profile_stride = max(1, int(math.ceil(frame_count / requested_cap)))
        previous_grey = None
        previous_detail = None
        foreground_changes: list[float] = []
        changes: list[float] = []
        edge_occupancy: list[float] = []
        luminance: list[float] = []
        foreground_occupancy: list[float] = []
        green_screen_occupancy: list[float] = []
        blue_screen_occupancy: list[float] = []
        green_screen_border_occupancy: list[float] = []
        blue_screen_border_occupancy: list[float] = []
        sample_timestamps: list[float] = []
        vertical_luminance_bias: list[float] = []
        border_hits = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        decoded_frames = 0
        profiled_frames = 0
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frames += 1
            should_profile = frame_index % profile_stride == 0 or frame_index == frame_count - 1
            if not should_profile:
                frame_index += 1
                continue
            height, width = frame.shape[:2]
            detail = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_detail is not None and previous_detail.shape == detail.shape:
                foreground_changes.append(foreground_change_fraction(previous_detail, detail))
            previous_detail = detail
            scale = min(1.0, float(target_dimension) / max(height, width, 1))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(2, int(width * scale)), max(2, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            blue, green, red = cv2.split(frame.astype(np.float32))
            green_screen = (
                (green > 24.0)
                & (green >= red + 8.0)
                & (green >= blue + 5.0)
            )
            blue_screen = (
                (blue > 24.0)
                & (blue >= red + 8.0)
                & (blue >= green + 5.0)
            )
            green_screen_occupancy.append(float(green_screen.mean()))
            blue_screen_occupancy.append(float(blue_screen.mean()))
            border_width = max(1, min(green_screen.shape) // 32)
            for mask, target in (
                (green_screen, green_screen_border_occupancy),
                (blue_screen, blue_screen_border_occupancy),
            ):
                border_pixels = np.concatenate(
                    (
                        mask[:border_width, :].reshape(-1),
                        mask[-border_width:, :].reshape(-1),
                        mask[:, :border_width].reshape(-1),
                        mask[:, -border_width:].reshape(-1),
                    )
                )
                target.append(float(border_pixels.mean()))
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_grey is not None:
                changes.append(float(cv2.absdiff(previous_grey, grey).mean() / 255.0))
            previous_grey = grey
            luminance.append(float(grey.mean() / 255.0))
            foreground = grey > 12
            foreground_occupancy.append(float(foreground.mean()))
            sample_timestamps.append(
                float(frame_index / fps) if fps > 0 else float(profiled_frames)
            )
            if int(foreground.sum()) > 100:
                y_positions = np.indices(grey.shape)[0] / max(1, grey.shape[0])
                mask_centroid = float(y_positions[foreground].mean())
                weighted_centroid = float(
                    (y_positions * grey).sum() / max(float(grey.sum()), 1.0)
                )
                vertical_luminance_bias.append(mask_centroid - weighted_centroid)
            edges = cv2.Canny(grey, 40, 120)
            edge_occupancy.append(float((edges > 0).mean()))
            border_strips = {
                "top": grey[:3, :],
                "bottom": grey[-3:, :],
                "left": grey[:, :3],
                "right": grey[:, -3:],
            }
            for edge_name, strip in border_strips.items():
                if float((strip > 12).mean()) > 0.001:
                    border_hits[edge_name] += 1
            profiled_frames += 1
            frame_index += 1

        if profiled_frames < 2 or not changes:
            return {"version": ACTIVITY_PROFILE_VERSION, "available": False, "reason": "too_few_decoded_frames"}

        change_array = np.asarray(changes, dtype=np.float32)
        mean_change = float(change_array.mean())
        peak_change = float(change_array.max())
        p90_change = float(np.percentile(change_array, 90))
        median_change = float(np.median(change_array))
        change_mad = float(np.median(np.abs(change_array - median_change)))
        p75_change = float(np.percentile(change_array, 75))
        p25_change = float(np.percentile(change_array, 25))
        activity_threshold = max(
            0.0015,
            median_change + max(0.0015, change_mad * 3.0),
            p75_change + (p75_change - p25_change) * 1.5,
        )
        active_fraction = float((change_array > activity_threshold).mean())
        peak_ratio = peak_change / max(mean_change, 0.002)
        transient_likelihood = min(1.0, max(0.0, (peak_ratio - 1.4) / 4.0))
        if active_fraction < 0.35 and peak_change > max(0.008, activity_threshold * 2.0):
            transient_likelihood = max(transient_likelihood, 0.65)
        motion_detected = bool(
            p90_change >= max(0.003, median_change * 1.8)
            or peak_change >= max(0.008, activity_threshold * 1.5)
            or mean_change >= 0.004
        )
        foreground_change_p90 = float(np.percentile(foreground_changes, 90)) if foreground_changes else 0.0
        motion_detected = motion_detected or foreground_change_p90 >= 0.12

        duration = (frame_count / fps) if fps > 0 else None
        green_screen_fraction = float(np.median(green_screen_occupancy))
        blue_screen_fraction = float(np.median(blue_screen_occupancy))
        green_screen_border_fraction = float(np.median(green_screen_border_occupancy))
        blue_screen_border_fraction = float(np.median(blue_screen_border_occupancy))
        key_screen_likely = _key_screen_choice(
            green_screen_fraction,
            blue_screen_fraction,
            green_screen_border_fraction,
            blue_screen_border_fraction,
        )

        if key_screen_likely:
            action_signal = [changes[0]]
            action_signal.extend(
                (changes[index - 1] + changes[index]) / 2.0
                for index in range(1, len(changes))
            )
            action_signal.append(changes[-1])
            action_signal_source = "temporal_change_key_screen"
        elif float(np.mean(luminance)) < 0.35:
            action_signal = foreground_occupancy
            action_signal_source = "foreground_occupancy"
        else:
            action_signal = [changes[0]]
            action_signal.extend(
                (changes[index - 1] + changes[index]) / 2.0
                for index in range(1, len(changes))
            )
            action_signal.append(changes[-1])
            action_signal_source = "temporal_change"
        action_timing = derive_action_timing(
            sample_timestamps,
            action_signal,
            float(duration or sample_timestamps[-1]),
            signal_source=action_signal_source,
        )
        action_timing = add_frame_ranges(action_timing, fps, frame_count)
        focus_timestamps = recommend_analysis_timestamps(
            sample_timestamps,
            action_signal,
            float(duration or sample_timestamps[-1]),
            action_timing,
        )
        median_foreground = float(np.median(foreground_occupancy))
        peak_foreground = float(max(foreground_occupancy))
        mean_edges = float(np.mean(edge_occupancy))
        p90_edges = float(np.percentile(np.asarray(edge_occupancy), 90))
        peak_edges = float(max(edge_occupancy))
        sparse_score = max(0.0, min(1.0, (0.20 - median_foreground) / 0.20))
        edge_presence = max(0.0, min(1.0, p90_edges / 0.025))
        edge_to_area = max(
            0.0,
            min(1.0, (mean_edges / max(median_foreground, 0.01)) / 0.55),
        )
        change_presence = max(0.0, min(1.0, p90_change / 0.018))
        fine_detail_likelihood = (
            sparse_score * 0.35
            + edge_presence * 0.25
            + edge_to_area * 0.20
            + change_presence * 0.20
        )
        if key_screen_likely:
            fine_detail_likelihood *= 0.35
        mean_luminance = float(np.mean(luminance))
        proxy_information_limited = bool(
            not key_screen_likely
            and mean_luminance < 0.002
            and peak_foreground < 0.03
            and (
                peak_foreground > 0.00002
                or peak_edges > 0.00002
                or peak_change > 0.00015
            )
        )
        high_resolution_recommended = bool(
            not key_screen_likely
            and (
                (
                    fine_detail_likelihood >= 0.62
                    and (p90_edges >= 0.001 or peak_change >= 0.004)
                )
                or proxy_information_limited
            )
        )
        detail_reasons = []
        if sparse_score >= 0.65:
            detail_reasons.append("sparse_foreground")
        if edge_to_area >= 0.55 and p90_edges >= 0.001:
            detail_reasons.append("fine_edge_structure")
        if transient_likelihood >= 0.65 and active_fraction <= 0.35:
            detail_reasons.append("brief_transient")
        if proxy_information_limited:
            detail_reasons.append("proxy_signal_near_resolution_floor")
        border_contact_fraction = {
            edge_name: round(hit_count / profiled_frames, 4)
            for edge_name, hit_count in border_hits.items()
        }
        bright_border_contact = [
            edge_name
            for edge_name, fraction in border_contact_fraction.items()
            if fraction >= 0.05
        ]
        return {
            "version": ACTIVITY_PROFILE_VERSION,
            "available": True,
            "profiling_mode": "every_proxy_frame" if profile_stride == 1 else "sequential_strided",
            "decoded_frames": decoded_frames,
            "profiled_frames": profiled_frames,
            "profile_stride": profile_stride,
            "frame_coverage_fraction": round(profiled_frames / max(frame_count, 1), 6),
            "target_dimension": target_dimension,
            "source_frame_count": frame_count,
            "source_fps": round(fps, 4) if fps > 0 else None,
            "duration_seconds": round(duration, 4) if duration else None,
            "temporal_change": round(mean_change, 6),
            "peak_change": round(peak_change, 6),
            "p90_change": round(p90_change, 6),
            "activity_threshold": round(activity_threshold, 6),
            "peak_ratio": round(peak_ratio, 4),
            "active_fraction": round(active_fraction, 4),
            "motion_detected": motion_detected,
            "foreground_relative_change_p90": round(foreground_change_p90, 6),
            "transient_likelihood": round(transient_likelihood, 4),
            "edge_activity": round(mean_edges, 6),
            "p90_edge_activity": round(p90_edges, 6),
            "peak_edge_activity": round(peak_edges, 6),
            "median_foreground_occupancy": round(median_foreground, 6),
            "peak_foreground_occupancy": round(peak_foreground, 6),
            "fine_detail_likelihood": round(fine_detail_likelihood, 4),
            "proxy_information_limited": proxy_information_limited,
            "high_resolution_recommended": high_resolution_recommended,
            "detail_reasons": detail_reasons,
            "bright_border_contact": bright_border_contact,
            "border_contact_fraction": border_contact_fraction,
            "mean_luminance": round(mean_luminance, 6),
            "green_screen_fraction": round(green_screen_fraction, 6),
            "blue_screen_fraction": round(blue_screen_fraction, 6),
            "green_screen_border_fraction": round(green_screen_border_fraction, 6),
            "blue_screen_border_fraction": round(blue_screen_border_fraction, 6),
            "key_screen_likely": key_screen_likely,
            "vertical_luminance_bias": (
                round(float(np.median(vertical_luminance_bias)), 6)
                if vertical_luminance_bias
                else None
            ),
            "action_timing": action_timing,
            "analysis_focus": {
                "recommended_timestamps_seconds": focus_timestamps,
                "uniform_anchors_included": True,
                "source": "activity_peaks_with_uniform_anchors",
            },
        }
    finally:
        capture.release()


def foreground_change_fraction(previous, current) -> float:
    """Avoid diluting moving bright particles into a mostly black whole frame."""
    import numpy as np

    active = (previous > 12) | (current > 12)
    count = int(active.sum())
    if count < 16:
        return 0.0
    changed = np.abs(previous.astype(np.int16) - current.astype(np.int16)) > 8
    changed_count = int((changed & active).sum())
    return changed_count / count if changed_count >= 12 else 0.0


def _key_screen_choice(
    green_fraction: float,
    blue_fraction: float,
    green_border_fraction: float = 0.0,
    blue_border_fraction: float = 0.0,
) -> str | None:
    """Require broad or border-dominant chroma coverage before calling a key screen."""

    def convincing(fraction: float, border_fraction: float) -> bool:
        return fraction >= 0.12 and (fraction >= 0.35 or border_fraction >= 0.60)

    if convincing(green_fraction, green_border_fraction) and green_fraction >= blue_fraction * 1.2:
        return "green_screen"
    if convincing(blue_fraction, blue_border_fraction) and blue_fraction >= green_fraction * 1.2:
        return "blue_screen"
    return None


def derive_action_timing(
    timestamps: list[float],
    signal: list[float],
    duration_seconds: float,
    signal_source: str = "foreground_occupancy",
) -> dict[str, Any]:
    """Derive visible, main-action and optimal editorial ranges from a sampled signal."""

    count = min(len(timestamps), len(signal))
    if count < 2 or duration_seconds <= 0:
        return {
            "version": ACTION_TIMING_VERSION,
            "available": False,
            "reason": "insufficient_samples",
        }
    times = [max(0.0, float(value)) for value in timestamps[:count]]
    values = [max(0.0, float(value)) for value in signal[:count]]
    sample_interval = max(
        1e-3,
        sum(max(0.0, right - left) for left, right in zip(times, times[1:]))
        / max(1, count - 1),
    )
    smoothing_radius = max(1, min(6, int(round(0.12 / sample_interval))))
    smoothed = [
        sum(
            values[
                max(0, index - smoothing_radius) : min(count, index + smoothing_radius + 1)
            ]
        )
        / len(
            values[
                max(0, index - smoothing_radius) : min(count, index + smoothing_radius + 1)
            ]
        )
        for index in range(count)
    ]
    peak_value = max(smoothed)
    if peak_value <= 1e-7:
        return {
            "version": ACTION_TIMING_VERSION,
            "available": False,
            "reason": "no_measurable_action",
        }

    ordered = sorted(smoothed)
    baseline = ordered[max(0, int((len(ordered) - 1) * 0.10))]
    dynamic = max(0.0, peak_value - baseline)
    if dynamic <= peak_value * 0.08:
        visible_mask = [True] * count
        main_mask = [True] * count
    else:
        visible_threshold = baseline + dynamic * 0.08
        main_threshold = baseline + dynamic * 0.30
        visible_mask = [value >= visible_threshold for value in smoothed]
        main_mask = [smooth >= main_threshold or raw >= main_threshold
                     for smooth, raw in zip(smoothed, values)]
    gap_samples = max(1, min(8, int(round(0.15 / sample_interval))))
    main_mask = _fill_short_gaps(main_mask, gap_samples)
    visible_mask = _fill_short_gaps(visible_mask, gap_samples)

    segments = _segments(main_mask)
    if not segments:
        peak_index = smoothed.index(peak_value)
        segments = [(peak_index, peak_index)]
    # A brief flash can matter more than a long, high-energy smoke tail.
    # Keep each threshold-crossing window instead of ranking by total energy.
    qualified = segments
    first_main = min(start for start, _end in qualified)
    last_main = max(end for _start, end in qualified)
    visible_indices = [index for index, active in enumerate(visible_mask) if active]
    first_visible = min(visible_indices) if visible_indices else first_main
    last_visible = max(visible_indices) if visible_indices else last_main
    peak_index = smoothed.index(peak_value)

    def start_time(index: int) -> float:
        return max(0.0, min(duration_seconds, times[index] - sample_interval * 0.5))

    def end_time(index: int) -> float:
        return max(0.0, min(duration_seconds, times[index] + sample_interval * 0.5))

    main_start = start_time(first_main)
    main_end = end_time(last_main)
    visible_start = start_time(first_visible)
    visible_end = end_time(last_visible)
    handle = max(0.25, min(1.0, sample_interval * 1.5))
    optimal_start = max(0.0, main_start - handle)
    optimal_end = min(duration_seconds, main_end + handle)
    coverage = max(0.0, main_end - main_start) / max(duration_seconds, 1e-6)
    if len(qualified) > 1:
        pattern = "multiple_events"
    elif (
        coverage >= 0.75
        and visible_start <= sample_interval
        and visible_end >= duration_seconds - sample_interval
    ):
        pattern = "continuous"
    elif coverage <= 0.35:
        pattern = "burst"
    else:
        pattern = "sustained"
    contrast = dynamic / max(peak_value, 1e-6)
    confidence = min(0.98, 0.62 + contrast * 0.34)
    if signal_source.startswith("temporal_change"):
        confidence = min(confidence, 0.78)

    event_segments = []
    for start, end in qualified:
        local_peak = max(range(start, end + 1), key=lambda index: smoothed[index])
        event_segments.append(
            {
                "start_seconds": round(start_time(start), 4),
                "peak_seconds": round(min(duration_seconds, times[local_peak]), 4),
                "end_seconds": round(end_time(end), 4),
            }
        )
    return {
        "version": ACTION_TIMING_VERSION,
        "available": True,
        "verification": "activity_suggestion",
        "review_required": True,
        "visible_start_seconds": round(visible_start, 4),
        "main_action_start_seconds": round(main_start, 4),
        "peak_seconds": round(min(duration_seconds, times[peak_index]), 4),
        "main_action_end_seconds": round(main_end, 4),
        "visible_end_seconds": round(visible_end, 4),
        "optimal_start_seconds": round(optimal_start, 4),
        "optimal_end_seconds": round(optimal_end, 4),
        "optimal_duration_seconds": round(max(0.0, optimal_end - optimal_start), 4),
        "coverage_fraction": round(coverage, 4),
        "pattern": pattern,
        "event_segments": event_segments,
        "signal_source": signal_source,
        "sample_count": count,
        "confidence": round(confidence, 4),
    }


def recommend_analysis_timestamps(
    timestamps: list[float],
    signal: list[float],
    duration_seconds: float,
    action_timing: dict[str, Any] | None = None,
    limit: int = 16,
) -> list[float]:
    """Blend activity peaks with uniform anchors without letting motion dominate coverage."""

    count = min(len(timestamps), len(signal))
    if count < 1 or duration_seconds <= 0 or limit < 1:
        return []
    times = [max(0.0, min(duration_seconds, float(value))) for value in timestamps[:count]]
    values = [max(0.0, float(value)) for value in signal[:count]]
    chosen: list[float] = []

    def add(value: Any) -> None:
        try:
            timestamp = max(0.0, min(duration_seconds, float(value)))
        except (TypeError, ValueError):
            return
        separation = max(0.04, duration_seconds / max(limit * 8.0, 1.0))
        if all(abs(existing - timestamp) >= separation for existing in chosen):
            chosen.append(timestamp)

    # These anchors protect idle context, handles and secondary events from an
    # attention signal dominated by one energetic burst.
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        add(duration_seconds * fraction)

    timing = action_timing or {}
    for field in (
        "visible_start_seconds",
        "main_action_start_seconds",
        "peak_seconds",
        "main_action_end_seconds",
        "visible_end_seconds",
    ):
        add(timing.get(field))
    for segment in timing.get("event_segments") or []:
        if not isinstance(segment, dict):
            continue
        add(segment.get("start_seconds"))
        add(segment.get("peak_seconds"))
        add(segment.get("end_seconds"))

    for index in sorted(range(count), key=lambda item: values[item], reverse=True):
        if len(chosen) >= limit:
            break
        add(times[index])
    return [round(value, 4) for value in sorted(chosen[:limit])]


def add_frame_ranges(
    timing: dict[str, Any],
    source_fps: float | None,
    source_frame_count: int | None = None,
) -> dict[str, Any]:
    """Add zero-based inclusive source-frame ranges to a seconds-based timing record."""

    result = dict(timing)
    fps = float(source_fps or 0.0)
    if not result.get("available") or fps <= 0:
        return result
    frame_count = max(0, int(source_frame_count or 0))
    last_frame = frame_count - 1 if frame_count > 0 else None

    def clip(value: int) -> int:
        value = max(0, value)
        return min(value, last_frame) if last_frame is not None else value

    def start_frame(seconds: Any) -> int:
        return clip(int(math.floor(max(0.0, float(seconds or 0.0)) * fps + 1e-8)))

    def end_frame(seconds: Any) -> int:
        value = int(math.ceil(max(0.0, float(seconds or 0.0)) * fps - 1e-8)) - 1
        return clip(max(0, value))

    def peak_frame(seconds: Any) -> int:
        return clip(int(round(max(0.0, float(seconds or 0.0)) * fps)))

    result.update(
        {
            "version": ACTION_TIMING_VERSION,
            "frame_numbering": "zero_based_inclusive",
            "source_fps": round(fps, 6),
            "source_frame_count": frame_count or None,
            "visible_start_frame": start_frame(result.get("visible_start_seconds")),
            "main_action_start_frame": start_frame(result.get("main_action_start_seconds")),
            "peak_frame": peak_frame(result.get("peak_seconds")),
            "main_action_end_frame": end_frame(result.get("main_action_end_seconds")),
            "visible_end_frame": end_frame(result.get("visible_end_seconds")),
            "optimal_start_frame": start_frame(result.get("optimal_start_seconds")),
            "optimal_end_frame": end_frame(result.get("optimal_end_seconds")),
        }
    )
    result["optimal_duration_frames"] = max(
        0,
        int(result["optimal_end_frame"]) - int(result["optimal_start_frame"]) + 1,
    )
    enriched_segments = []
    for segment in result.get("event_segments") or []:
        if not isinstance(segment, dict):
            continue
        enriched_segments.append(
            {
                **segment,
                "start_frame": start_frame(segment.get("start_seconds")),
                "peak_frame": peak_frame(segment.get("peak_seconds")),
                "end_frame": end_frame(segment.get("end_seconds")),
            }
        )
    result["event_segments"] = enriched_segments
    return result


def _fill_single_sample_gaps(mask: list[bool]) -> list[bool]:
    return _fill_short_gaps(mask, 1)


def _fill_short_gaps(mask: list[bool], max_gap: int) -> list[bool]:
    result = list(mask)
    index = 1
    while index < len(result) - 1:
        if result[index] or not result[index - 1]:
            index += 1
            continue
        end = index
        while end < len(result) and not result[end]:
            end += 1
        if end < len(result) and end - index <= max_gap:
            for gap_index in range(index, end):
                result[gap_index] = True
        index = max(end, index + 1)
    return result


def _segments(mask: list[bool]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(mask) - 1):
            end = index if active and index == len(mask) - 1 else index - 1
            result.append((start, end))
            start = None
    return result

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .discovery import MediaReference
from .models import SourceRepresentation
from .util import (
    file_digest,
    parse_fraction,
    run_command,
    run_json,
    safe_float,
    safe_int,
    stable_id,
    tool_exists,
)


ALPHA_PROBE_VERSION = "ffmpeg_alpha_probe"
ALPHA_PROBE_CONFIDENCE = 0.85
ALPHA_HEURISTIC_SOURCE = "filename_heuristic"
ALPHA_HEURISTIC_CONFIDENCE = 0.3
ALPHA_FIELDS = (
    "has_alpha_channel",
    "alpha_non_empty",
    "alpha_is_binary",
    "alpha_is_soft",
    "premult_status",
    "over_black_likely",
    "keyable_bg_likely",
)


ALPHA_PIX_FMTS = {
    "rgba",
    "argb",
    "bgra",
    "abgr",
    "yuva420p",
    "yuva422p",
    "yuva444p",
    "gbrap",
    "gbrap10le",
    "gbrap12le",
    "gbrap16le",
}

LOSSLESS_CODECS = {"prores", "dnxhd", "dnxhr", "ffv1", "huffyuv", "utvideo"}


def build_source_representation(reference: MediaReference, default_fps: float = 24.0) -> SourceRepresentation:
    probe_target = reference.files[0] if reference.files else reference.path
    metadata = _probe(probe_target)
    file_size = sum(path.stat().st_size for path in reference.files if path.exists())
    created_at = None
    if reference.files:
        created_at = _created_at(reference.files[0])

    stream = metadata.get("stream", {})
    format_data = metadata.get("format", {})
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate"))
    if reference.kind == "sequence":
        fps = fps or default_fps
        duration = len(reference.files) / fps if fps else None
    else:
        duration = safe_float(format_data.get("duration")) or safe_float(stream.get("duration"))

    representation = SourceRepresentation(
        representation_id=stable_id("rep", reference.reference_id, reference.path),
        path=str(reference.path.resolve()),
        original_path=str(reference.original_path.resolve()),
        format=reference.format_hint,
        codec=stream.get("codec_name"),
        container=format_data.get("format_name"),
        width=safe_int(stream.get("width")),
        height=safe_int(stream.get("height")),
        fps=fps,
        duration_seconds=duration,
        bit_depth=_bit_depth(stream),
        color_space=_color_space(stream),
        is_sequence=reference.kind == "sequence",
        sequence_first_frame=reference.first_frame,
        sequence_last_frame=reference.last_frame,
        sequence_frame_padding=reference.frame_padding,
        missing_frames=reference.missing_frames or [],
        file_size_bytes=file_size,
        created_at=created_at,
        content_fingerprint=_content_fingerprint(reference, duration),
        filename_root=reference.filename_root,
        technical_metadata=_technical_metadata(reference, stream, format_data),
    )
    return representation


def alpha_cluster(representation: SourceRepresentation) -> dict[str, object]:
    """Return the seven Element-level alpha fields plus an `alpha_provenance` dict.

    Tries a real pixel-level probe via ffmpeg + Pillow first. Falls back to a
    format/extension heuristic if ffmpeg, Pillow, or numpy are unavailable or
    the probe fails.
    """

    probe_source: str | None
    if representation.is_sequence and representation.sequence_first_frame is not None:
        # Pass the ffmpeg-style pattern directly so the sequence demuxer can iterate it.
        probe_source = representation.path
    else:
        probe_source = _probe_path_for_representation(representation)

    stats: dict[str, object] | None = None
    if probe_source:
        source_has_alpha = _source_has_alpha_channel(representation)
        stats = _alpha_stats(
            probe_source,
            representation.is_sequence,
            representation.sequence_first_frame,
            representation.sequence_frame_padding,
            source_has_alpha,
        )

    if stats is not None:
        cluster = {
            "has_alpha_channel": bool(stats.get("has_alpha_channel", False)),
            "alpha_non_empty": bool(stats.get("alpha_non_empty", False)),
            "alpha_is_binary": bool(stats.get("alpha_is_binary", False)),
            "alpha_is_soft": bool(stats.get("alpha_is_soft", False)),
            "premult_status": str(stats.get("premult_status", "unknown")),
            "over_black_likely": bool(stats.get("over_black_likely", False)),
            "keyable_bg_likely": bool(stats.get("keyable_bg_likely", False)),
        }
        source = str(stats.get("_source", ALPHA_PROBE_VERSION))
        confidence = float(stats.get("_confidence", ALPHA_PROBE_CONFIDENCE))
    else:
        cluster = _alpha_filename_heuristic(representation)
        source = ALPHA_HEURISTIC_SOURCE
        confidence = ALPHA_HEURISTIC_CONFIDENCE

    cluster["alpha_provenance"] = {
        name: {"source": source, "confidence": confidence} for name in ALPHA_FIELDS
    }
    return cluster


def _alpha_filename_heuristic(representation: SourceRepresentation) -> dict[str, object]:
    fmt = (representation.format or "").lower()
    has_alpha = fmt.startswith(("png", "exr"))
    if not has_alpha and tool_exists("ffprobe"):
        target = _probe_path_for_representation(representation)
        if target:
            data = _probe(Path(target))
            pix_fmt = data.get("stream", {}).get("pix_fmt")
            has_alpha = bool(pix_fmt in ALPHA_PIX_FMTS or (pix_fmt and "alpha" in pix_fmt))
    return {
        "has_alpha_channel": has_alpha,
        "alpha_non_empty": has_alpha,
        "alpha_is_binary": False,
        "alpha_is_soft": has_alpha,
        "premult_status": "unknown",
        "over_black_likely": not has_alpha,
        "keyable_bg_likely": False,
    }


def _alpha_stats(
    path: str,
    is_sequence: bool,
    first_frame: int | None,
    padding: int | None,
    source_has_alpha: bool,
) -> dict[str, object] | None:
    """Extract a representative frame via ffmpeg and analyse alpha statistics.

    Returns None if ffmpeg, Pillow, or the probe itself is unavailable / failed.
    On success returns a dict with the seven alpha fields plus `_source` and
    `_confidence` keys describing how the probe ran.
    """

    if not tool_exists("ffmpeg"):
        return None

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None

    tmp_handle, tmp_path = tempfile.mkstemp(suffix=".png", prefix="vfx_alpha_")
    os.close(tmp_handle)
    try:
        if is_sequence and first_frame is not None:
            command = [
                "ffmpeg",
                "-y",
                "-framerate",
                "24",
                "-start_number",
                str(first_frame),
                "-i",
                path,
                "-frames:v",
                "1",
                "-vframes",
                "1",
                "-vf",
                "scale=256:-2",
                "-pix_fmt",
                "rgba",
                tmp_path,
            ]
        else:
            duration = _probe_duration(Path(path))
            seek = max(0.0, (duration or 0.0) / 2.0)
            command = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{seek:.3f}",
                "-i",
                path,
                "-frames:v",
                "1",
                "-vframes",
                "1",
                "-vf",
                "scale=256:-2",
                "-pix_fmt",
                "rgba",
                tmp_path,
            ]
        ok, _err = run_command(command, timeout=30)
        if not ok or not Path(tmp_path).exists() or Path(tmp_path).stat().st_size == 0:
            return None

        try:
            with Image.open(tmp_path) as img:
                rgba = img.convert("RGBA")
                rgba.load()
        except Exception:
            return None

        try:
            return _analyse_rgba(rgba, source_has_alpha)
        except Exception:
            return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _analyse_rgba(image, source_has_alpha: bool) -> dict[str, object]:  # pragma: no cover - exercised via ffmpeg path
    from PIL import Image  # type: ignore[import-not-found]

    width, height = image.size
    total_pixels = max(1, width * height)

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None  # type: ignore[assignment]

    if np is not None:
        arr = np.asarray(image, dtype=np.uint8)
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        a = arr[..., 3]
        alpha_min = int(a.min())
        alpha_max = int(a.max())
        alpha_mean = float(a.mean())
        alpha_non_empty = bool(alpha_max > 0 and alpha_min < 255)
        binary_count = int(((a == 0) | (a == 255)).sum())
        alpha_is_binary = (binary_count / total_pixels) > 0.95
        alpha_is_soft = alpha_non_empty and not alpha_is_binary

        # Premult heuristic: sample partially transparent pixels.
        partial_mask = (a > 0) & (a < 250)
        partial_count = int(partial_mask.sum())
        if partial_count < 50:
            premult_status = "unknown"
        else:
            max_rgb = np.maximum(np.maximum(r, g), b)
            cmp_alpha = a.astype(np.int16) + 4
            within = (max_rgb.astype(np.int16) <= cmp_alpha) & partial_mask
            premult_ratio = int(within.sum()) / partial_count
            premult_status = "premultiplied" if premult_ratio > 0.70 else "straight"

        # over_black_likely: no alpha doing work AND luminance heavily on the low end.
        luminance = (0.2126 * r.astype(np.float32) + 0.7152 * g.astype(np.float32) + 0.0722 * b.astype(np.float32))
        dark_ratio = float((luminance <= 32).sum()) / total_pixels
        alpha_empty = (alpha_max == 0) or (not alpha_non_empty)
        over_black_likely = bool(alpha_empty and dark_ratio > 0.70)

        # keyable_bg_likely: examine the 1-pixel border in HSV.
        border_pixels = _border_pixels_np(r, g, b, np)
        keyable_bg_likely = _keyable_from_border_np(border_pixels, np)

        return {
            "has_alpha_channel": source_has_alpha,
            "alpha_non_empty": bool(source_has_alpha and alpha_non_empty),
            "alpha_is_binary": bool(source_has_alpha and alpha_is_binary and alpha_non_empty),
            "alpha_is_soft": bool(source_has_alpha and alpha_is_soft),
            "premult_status": premult_status if source_has_alpha else "unknown",
            "over_black_likely": over_black_likely,
            "keyable_bg_likely": keyable_bg_likely,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "alpha_mean": alpha_mean,
            "_source": ALPHA_PROBE_VERSION,
            "_confidence": ALPHA_PROBE_CONFIDENCE,
        }

    # numpy-free fallback using Image.histogram() and getdata().
    alpha_channel = image.getchannel("A")
    hist = alpha_channel.histogram()  # 256 bins
    if len(hist) < 256:
        hist = list(hist) + [0] * (256 - len(hist))
    alpha_min = next((i for i, v in enumerate(hist) if v > 0), 0)
    alpha_max = next((255 - i for i, v in enumerate(reversed(hist)) if v > 0), 0)
    alpha_sum = sum(i * v for i, v in enumerate(hist))
    alpha_mean = alpha_sum / total_pixels
    alpha_non_empty = alpha_max > 0 and alpha_min < 255
    binary_ratio = (hist[0] + hist[255]) / total_pixels
    alpha_is_binary = binary_ratio > 0.95
    alpha_is_soft = alpha_non_empty and not alpha_is_binary

    # Sample-based premult heuristic without numpy.
    pixels = list(image.getdata())
    partial = [(r, g, b, a) for (r, g, b, a) in pixels if 0 < a < 250]
    if len(partial) < 50:
        premult_status = "unknown"
    else:
        within = sum(1 for (r, g, b, a) in partial if max(r, g, b) <= a + 4)
        premult_status = "premultiplied" if (within / len(partial)) > 0.70 else "straight"

    luminance_values = [0.2126 * r + 0.7152 * g + 0.0722 * b for (r, g, b, _a) in pixels]
    dark_ratio = sum(1 for lv in luminance_values if lv <= 32) / total_pixels
    alpha_empty = alpha_max == 0 or not alpha_non_empty
    over_black_likely = alpha_empty and dark_ratio > 0.70

    border = _border_pixels_list(pixels, width, height)
    keyable_bg_likely = _keyable_from_border_list(border)

    return {
        "has_alpha_channel": source_has_alpha,
        "alpha_non_empty": bool(source_has_alpha and alpha_non_empty),
        "alpha_is_binary": bool(source_has_alpha and alpha_is_binary and alpha_non_empty),
        "alpha_is_soft": bool(source_has_alpha and alpha_is_soft),
        "premult_status": premult_status if source_has_alpha else "unknown",
        "over_black_likely": over_black_likely,
        "keyable_bg_likely": keyable_bg_likely,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_mean": alpha_mean,
        "_source": ALPHA_PROBE_VERSION,
        "_confidence": ALPHA_PROBE_CONFIDENCE,
    }


def _source_has_alpha_channel(representation: SourceRepresentation) -> bool:
    target = _probe_path_for_representation(representation)
    if target:
        data = _probe(Path(target))
        pix_fmt = str(data.get("stream", {}).get("pix_fmt") or "").lower()
        if pix_fmt:
            return _pix_fmt_has_alpha(pix_fmt)

    # Conservative fallback when ffprobe cannot answer. These formats commonly
    # carry alpha, but the pixel-level probe still decides whether it is empty,
    # binary, or soft.
    fmt = (representation.format or "").lower()
    return fmt.startswith(("png", "exr"))


def _pix_fmt_has_alpha(pix_fmt: str) -> bool:
    return (
        pix_fmt in ALPHA_PIX_FMTS
        or pix_fmt.startswith(("rgba", "argb", "bgra", "abgr", "yuva", "gbrap", "ya"))
        or "alpha" in pix_fmt
    )


def _border_pixels_np(r, g, b, np):  # pragma: no cover - exercised via ffmpeg path
    top_r, top_g, top_b = r[0, :], g[0, :], b[0, :]
    bot_r, bot_g, bot_b = r[-1, :], g[-1, :], b[-1, :]
    left_r, left_g, left_b = r[:, 0], g[:, 0], b[:, 0]
    right_r, right_g, right_b = r[:, -1], g[:, -1], b[:, -1]
    rs = np.concatenate([top_r, bot_r, left_r, right_r])
    gs = np.concatenate([top_g, bot_g, left_g, right_g])
    bs = np.concatenate([top_b, bot_b, left_b, right_b])
    return rs, gs, bs


def _keyable_from_border_np(border, np) -> bool:  # pragma: no cover - exercised via ffmpeg path
    rs, gs, bs = border
    total = max(1, int(rs.shape[0]))
    near_white = int(((rs > 240) & (gs > 240) & (bs > 240)).sum())
    near_black = int(((rs < 8) & (gs < 8) & (bs < 8)).sum())
    if near_white / total > 0.60 or near_black / total > 0.60:
        return True

    # HSV conversion (vectorised) using PIL on small border strip is awkward;
    # convert per-pixel via numpy for speed.
    r = rs.astype(np.float32) / 255.0
    g = gs.astype(np.float32) / 255.0
    b = bs.astype(np.float32) / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    value = cmax
    saturation = np.where(cmax == 0, 0.0, delta / np.where(cmax == 0, 1.0, cmax))
    # Hue in degrees [0, 360).
    hue = np.zeros_like(cmax)
    mask = delta > 0
    # Compute three branches.
    rmax = mask & (cmax == r)
    gmax = mask & (cmax == g) & (~rmax)
    bmax = mask & (cmax == b) & (~rmax) & (~gmax)
    hue = np.where(rmax, ((g - b) / np.where(delta == 0, 1.0, delta)) % 6, hue)
    hue = np.where(gmax, ((b - r) / np.where(delta == 0, 1.0, delta)) + 2, hue)
    hue = np.where(bmax, ((r - g) / np.where(delta == 0, 1.0, delta)) + 4, hue)
    hue = (hue * 60.0) % 360.0

    saturated = (saturation > 0.6) & (value > 0.05)
    if int(saturated.sum()) < max(20, int(0.20 * total)):
        return False
    sat_hues = hue[saturated]
    # Find a 30-degree window containing >60% of saturated border pixels.
    if sat_hues.size == 0:
        return False
    # Use a coarse histogram of 12 bins (30 deg each) on the circle.
    bins = np.zeros(12, dtype=np.int64)
    idx = (sat_hues // 30).astype(np.int64) % 12
    for i in range(12):
        bins[i] = int((idx == i).sum())
    # Sliding pair sum (covers any 30-degree window aligned to a bin) — single bin already covers 30 deg.
    max_bin = int(bins.max())
    return (max_bin / int(saturated.sum())) > 0.60


def _border_pixels_list(pixels, width, height):
    border = []
    if height == 0 or width == 0:
        return border
    for x in range(width):
        border.append(pixels[x])
        border.append(pixels[(height - 1) * width + x])
    for y in range(1, height - 1):
        border.append(pixels[y * width])
        border.append(pixels[y * width + (width - 1)])
    return border


def _keyable_from_border_list(border) -> bool:
    total = max(1, len(border))
    near_white = sum(1 for (r, g, b, _a) in border if r > 240 and g > 240 and b > 240)
    near_black = sum(1 for (r, g, b, _a) in border if r < 8 and g < 8 and b < 8)
    if near_white / total > 0.60 or near_black / total > 0.60:
        return True

    bins = [0] * 12
    saturated_count = 0
    for (r, g, b, _a) in border:
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
        cmax = max(rf, gf, bf)
        cmin = min(rf, gf, bf)
        delta = cmax - cmin
        value = cmax
        saturation = 0.0 if cmax == 0 else delta / cmax
        if saturation <= 0.6 or value <= 0.05 or delta == 0:
            continue
        if cmax == rf:
            hue = ((gf - bf) / delta) % 6
        elif cmax == gf:
            hue = ((bf - rf) / delta) + 2
        else:
            hue = ((rf - gf) / delta) + 4
        hue = (hue * 60.0) % 360.0
        bins[int(hue // 30) % 12] += 1
        saturated_count += 1

    if saturated_count < max(20, int(0.20 * total)):
        return False
    return (max(bins) / saturated_count) > 0.60


def _probe_duration(path: Path) -> float | None:
    data = _probe(path)
    return safe_float(data.get("format", {}).get("duration")) or safe_float(
        data.get("stream", {}).get("duration")
    )


def _probe(path: Path) -> dict[str, object]:
    if not tool_exists("ffprobe") or not path.exists():
        return {"stream": {}, "format": {}}
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    data, _ = run_json(command)
    if not data:
        return {"stream": {}, "format": {}}
    streams = data.get("streams") or []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    return {"stream": video_stream, "format": data.get("format", {})}


def _bit_depth(stream: dict[str, object]) -> int | None:
    for key in ("bits_per_raw_sample", "bits_per_sample"):
        value = safe_int(stream.get(key))
        if value:
            return value
    pix_fmt = str(stream.get("pix_fmt") or "")
    for value in (32, 16, 12, 10, 8):
        if str(value) in pix_fmt:
            return value
    return None


def _color_space(stream: dict[str, object]) -> str | None:
    for key in ("color_space", "color_transfer", "color_primaries"):
        value = stream.get(key)
        if value and value != "unknown":
            return str(value)
    return None


def _created_at(path: Path) -> str:
    timestamp = path.stat().st_mtime
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _technical_metadata(
    reference: MediaReference,
    stream: dict[str, Any],
    format_data: dict[str, Any],
) -> dict[str, Any]:
    """Collect technical truth without asking a vision-language model."""

    result: dict[str, Any] = {
        "metadata_reader": "ffprobe",
        "pixel_format": stream.get("pix_fmt"),
        "codec_profile": stream.get("profile"),
        "codec_tag": stream.get("codec_tag_string"),
        "field_order": stream.get("field_order"),
        "sample_aspect_ratio": stream.get("sample_aspect_ratio"),
        "display_aspect_ratio": stream.get("display_aspect_ratio"),
        "colour_range": stream.get("color_range"),
        "colour_primaries": stream.get("color_primaries"),
        "colour_transfer": stream.get("color_transfer"),
        "colour_matrix": stream.get("color_space"),
        "chroma_location": stream.get("chroma_location"),
        "time_base": stream.get("time_base"),
        "start_time": stream.get("start_time"),
        "stream_tags": stream.get("tags") or {},
        "container_tags": format_data.get("tags") or {},
    }
    result = {key: value for key, value in result.items() if _metadata_value_present(value)}
    if reference.files and reference.files[0].suffix.lower() in IMAGE_EXTENSIONS_FOR_OIIO:
        result.update(_oiio_metadata(reference.files[0]))
    return result


IMAGE_EXTENSIONS_FOR_OIIO = {".exr", ".dpx", ".tif", ".tiff", ".png"}


def _oiio_metadata(path: Path) -> dict[str, Any]:
    try:
        import OpenImageIO as oiio  # type: ignore[import-not-found]
    except ImportError:
        return {"image_metadata_reader": "ffprobe_fallback"}

    image_input = None
    try:
        image_input = oiio.ImageInput.open(str(path))
        if image_input is None:
            return {"image_metadata_reader": "oiio_open_failed"}
        spec = image_input.spec()
        channel_names = [str(item) for item in (getattr(spec, "channelnames", None) or [])]
        channel_formats = [
            str(item) for item in (getattr(spec, "channelformats", None) or [])
        ]
        attributes: dict[str, Any] = {}
        for parameter in getattr(spec, "extra_attribs", None) or []:
            name = str(getattr(parameter, "name", "") or "")
            value = getattr(parameter, "value", None)
            if name and _json_scalar(value):
                attributes[name] = value

        alpha_value = getattr(spec, "alpha_channel", -1)
        z_value = getattr(spec, "z_channel", -1)
        alpha_index = int(alpha_value) if alpha_value is not None else -1
        z_index = int(z_value) if z_value is not None else -1
        result: dict[str, Any] = {
            "image_metadata_reader": "OpenImageIO",
            "channel_names": channel_names,
            "channel_formats": channel_formats,
            "alpha_channel": channel_names[alpha_index]
            if 0 <= alpha_index < len(channel_names)
            else None,
            "z_channel": channel_names[z_index] if 0 <= z_index < len(channel_names) else None,
            "deep": bool(getattr(spec, "deep", False)),
            "data_window": {
                "x": int(getattr(spec, "x", 0)),
                "y": int(getattr(spec, "y", 0)),
                "width": int(getattr(spec, "width", 0)),
                "height": int(getattr(spec, "height", 0)),
            },
            "display_window": {
                "x": int(getattr(spec, "full_x", 0)),
                "y": int(getattr(spec, "full_y", 0)),
                "width": int(getattr(spec, "full_width", 0)),
                "height": int(getattr(spec, "full_height", 0)),
            },
            "tile_width": int(getattr(spec, "tile_width", 0)),
            "tile_height": int(getattr(spec, "tile_height", 0)),
            "attributes": attributes,
        }
        return {key: value for key, value in result.items() if _metadata_value_present(value)}
    except Exception as exc:
        return {
            "image_metadata_reader": "oiio_error",
            "image_metadata_error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if image_input is not None:
            try:
                image_input.close()
            except Exception:
                pass


def _json_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return len(value) <= 64 and all(
            item is None or isinstance(item, (str, int, float, bool)) for item in value
        )
    return False


def _metadata_value_present(value: Any) -> bool:
    return value is not None and value != "" and value != "unknown"


def _content_fingerprint(reference: MediaReference, duration: float | None) -> str:
    if reference.kind == "sequence" and reference.files:
        sample_paths = [
            reference.files[0],
            reference.files[len(reference.files) // 2],
            reference.files[-1],
        ]
        parts = [file_digest(path, sample_size=128 * 1024) for path in sample_paths if path.exists()]
        return stable_id("fingerprint", *parts, len(reference.files), duration)
    if reference.files:
        return file_digest(reference.files[0])
    return stable_id("fingerprint", reference.path, duration)


def _probe_path_for_representation(representation: SourceRepresentation) -> str | None:
    path = Path(representation.path)
    if path.exists():
        return str(path)
    if representation.is_sequence and representation.sequence_first_frame is not None:
        token = f"%0{representation.sequence_frame_padding or 4}d"
        candidate = representation.path.replace(token, f"{representation.sequence_first_frame:0{representation.sequence_frame_padding or 4}d}")
        if Path(candidate).exists():
            return candidate
    return None

from __future__ import annotations

import functools
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .models import SourceRepresentation
from .util import run_command, stable_id, tool_exists

try:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    _PHASH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when deps are absent
    np = None  # type: ignore
    Image = None  # type: ignore
    _PHASH_AVAILABLE = False

try:  # optional fast DCT
    from scipy.fft import dct as _scipy_dct  # type: ignore

    _SCIPY_DCT_AVAILABLE = True
except Exception:  # pragma: no cover
    _scipy_dct = None  # type: ignore
    _SCIPY_DCT_AVAILABLE = False


ARTIFACT_VERSION = "ffmpeg-tonemap-0.1.6"
ALPHA_PREVIEW_VERSION = "ffmpeg-alpha-preview-0.1.0"

_PHASH_TIMEOUT_SECONDS = 5.0


def generate_artifacts(
    element_id: str,
    primary: SourceRepresentation,
    cache_dir: Path,
    enabled: bool = True,
) -> dict[str, str | None]:
    artifact_dir = cache_dir / "artifacts" / element_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    poster = artifact_dir / "poster.jpg"
    preview_480 = artifact_dir / "preview_480.mp4"
    preview_1080 = artifact_dir / "preview_1080.mp4"

    if not enabled or not tool_exists("ffmpeg"):
        placeholder = artifact_dir / "poster.svg"
        _write_placeholder_svg(placeholder, primary)
        return {
            "poster_path": str(placeholder),
            "preview_movie_480_path": None,
            "preview_movie_1080_path": None,
        }

    poster_ok = _extract_poster(primary, poster)
    preview_480_ok = _extract_preview(primary, preview_480, 480)
    preview_1080_ok = _extract_preview(primary, preview_1080, 1080)

    if not poster_ok:
        placeholder = artifact_dir / "poster.svg"
        _write_placeholder_svg(placeholder, primary)
        poster_path: str | None = str(placeholder)
    else:
        poster_path = str(poster)

    return {
        "poster_path": poster_path,
        "preview_movie_480_path": str(preview_480) if preview_480_ok else None,
        "preview_movie_1080_path": str(preview_1080) if preview_1080_ok else None,
    }


def generate_alpha_preview(
    element_id: str,
    primary: SourceRepresentation,
    cache_dir: Path,
    enabled: bool = True,
    long_edge: int = 1080,
) -> str | None:
    """Build and cache a grayscale movie of the source alpha channel.

    Alpha previews are deliberately generated on demand instead of alongside every
    RGB proxy. Large catalogues therefore pay the extra decode/storage cost only for
    mattes an artist actually inspects. A temporary output is atomically promoted so
    concurrent HTTP range requests never see a partially written movie.
    """

    if not enabled or not tool_exists("ffmpeg"):
        return None

    artifact_dir = cache_dir / "artifacts" / element_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = artifact_dir / "preview_alpha.mp4"
    if output.exists() and output.stat().st_size > 0:
        return str(output)

    temporary = artifact_dir / "preview_alpha.build.mp4"
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        return None

    command = ["ffmpeg", "-y"]
    is_still = primary.format == "single_image"
    if primary.is_sequence:
        command.extend(_sequence_input_args(primary))
    elif is_still:
        command.extend(
            ["-loop", "1", "-framerate", f"{primary.fps or 24.0:.6g}", "-i", primary.path]
        )
    else:
        command.extend(["-i", primary.path])

    command.extend(
        [
            "-vf",
            (
                "alphaextract,"
                f"scale='if(gte(iw,ih),{long_edge},-2)':"
                f"'if(gte(ih,iw),{long_edge},-2)',format=yuv420p"
            ),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "12",
            "-movflags",
            "+faststart",
        ]
    )
    if is_still:
        command.extend(["-t", "1"])
    command.append(str(temporary))

    ok, _ = run_command(command, timeout=480)
    try:
        if ok and temporary.exists() and temporary.stat().st_size > 0:
            temporary.replace(output)
            return str(output)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return None


def generate_representation_phash(
    representation: SourceRepresentation,
    cache_dir: Path,
    enabled: bool = True,
) -> str | None:
    if not enabled or not tool_exists("ffmpeg"):
        return None
    phash_dir = cache_dir / "representation_phash"
    phash_dir.mkdir(parents=True, exist_ok=True)
    poster = phash_dir / f"{representation.representation_id}.jpg"
    if not poster.exists() and not _extract_poster(representation, poster):
        return None
    return perceptual_hash_from_artifact(poster)


def perceptual_hash_from_artifact(path: str | None, fallback: str | None = None) -> str | None:
    if not path:
        return fallback
    file_path = Path(path)
    if not file_path.exists():
        return fallback
    try:
        if file_path.stat().st_size == 0:
            return fallback
    except OSError:
        return fallback

    if _PHASH_AVAILABLE:
        start = time.monotonic()
        try:
            phash_hex = _compute_dct_phash(file_path)
            if phash_hex and (time.monotonic() - start) <= _PHASH_TIMEOUT_SECONDS:
                return f"phash_{phash_hex}"
        except Exception:
            pass

    try:
        data = file_path.read_bytes()
    except OSError:
        return fallback
    return stable_id("phash", data[:128 * 1024], len(data), length=16)


def _compute_dct_phash(path: Path) -> str | None:
    """64-bit DCT pHash: grayscale -> 32x32 -> DCT-II -> top-left 8x8 -> median-threshold bits."""
    if not _PHASH_AVAILABLE:
        return None
    with Image.open(path) as image:
        gray = image.convert("L").resize((32, 32), Image.LANCZOS)
    array = np.asarray(gray, dtype=np.float64)

    if _SCIPY_DCT_AVAILABLE:
        dct_full = _scipy_dct(_scipy_dct(array, axis=0, norm="ortho"), axis=1, norm="ortho")
    else:
        basis = _dct_basis(32)
        dct_full = basis @ array @ basis.T

    block = dct_full[:8, :8]
    flat = block.flatten()
    median = float(np.median(flat[1:]))
    bits = (flat > median).astype(np.uint8)

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


@functools.lru_cache(maxsize=4)
def _dct_basis(n: int) -> "np.ndarray":  # type: ignore[name-defined]
    indices = np.arange(n)
    k = indices.reshape(-1, 1)
    i = indices.reshape(1, -1)
    basis = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    scale = np.full((n, 1), np.sqrt(2.0 / n))
    scale[0, 0] = np.sqrt(1.0 / n)
    return basis * scale


def _is_exr_source(primary: SourceRepresentation) -> bool:
    fmt = (primary.format or "").lower()
    codec = (primary.codec or "").lower()
    if fmt.startswith("exr") or codec == "exr":
        return True
    for candidate in (primary.path, primary.original_path):
        if candidate and candidate.lower().endswith(".exr"):
            return True
    return False


@functools.lru_cache(maxsize=1)
def _tone_map_strategy() -> str:
    if tool_exists("oiiotool"):
        return "oiiotool"
    if tool_exists("ffmpeg") and _ffmpeg_has_zscale():
        return "ffmpeg_zscale"
    return "ffmpeg_basic"


@functools.lru_cache(maxsize=1)
def _ffmpeg_has_zscale() -> bool:
    try:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    return any("zscale" in line for line in completed.stdout.splitlines())


def _log_strategy(stage: str, primary: SourceRepresentation, strategy: str) -> None:
    print(
        f"[artifacts] {stage} strategy={strategy} path={primary.path}",
        file=sys.stderr,
        flush=True,
    )


def _exr_middle_frame_path(primary: SourceRepresentation) -> Path | None:
    if not primary.is_sequence:
        candidate = Path(primary.path)
        return candidate if candidate.suffix.lower() == ".exr" and candidate.exists() else None
    first = primary.sequence_first_frame
    last = primary.sequence_last_frame
    padding = primary.sequence_frame_padding or 4
    if first is None or last is None:
        return None
    middle = (first + last) // 2
    pattern_path = Path(primary.path)
    parent = pattern_path.parent
    name = pattern_path.name
    # path is typically a printf-style pattern like prefix.%04d.exr; derive the prefix/ext.
    if "%" in name:
        prefix, _, rest = name.partition("%")
        _, _, ext_with_dot = rest.partition("d")
        ext = ext_with_dot if ext_with_dot.startswith(".") else f".{ext_with_dot}"
    else:
        # Best-effort: split on the last dot-padded number block.
        stem = pattern_path.stem
        ext = pattern_path.suffix or ".exr"
        prefix = stem.rsplit(".", 1)[0] + "."
    frame_name = f"{prefix}{middle:0{padding}d}{ext}"
    candidate = parent / frame_name
    return candidate if candidate.exists() else None


def _exr_zscale_filter_prefix() -> str:
    return (
        "zscale=tin=linear:t=linear,tonemap=hable,"
        "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
    )


def _extract_poster(primary: SourceRepresentation, output: Path) -> bool:
    is_exr = _is_exr_source(primary)
    strategy = _tone_map_strategy() if is_exr else "ffmpeg_basic"
    if is_exr:
        _log_strategy("poster", primary, strategy)

    if is_exr and strategy == "oiiotool":
        frame = _exr_middle_frame_path(primary)
        if frame is not None:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                ok_cc, _ = run_command(
                    [
                        "oiiotool",
                        str(frame),
                        "--colorconvert",
                        "linear",
                        "sRGB",
                        "--ch",
                        "R,G,B",
                        "-o",
                        str(tmp_path),
                    ],
                    timeout=120,
                )
                if ok_cc and tmp_path.exists():
                    command = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(tmp_path),
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale='if(gte(iw,ih),1280,-2)':'if(gte(ih,iw),1280,-2)'",
                        "-q:v",
                        "3",
                        str(output),
                    ]
                    ok, _ = run_command(command, timeout=120)
                    if ok and output.exists():
                        return True
            finally:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        # fall through to ffmpeg path

    command = ["ffmpeg", "-y"]
    if primary.is_sequence:
        command.extend(_sequence_input_args(primary))
    else:
        seek = max(0.0, (primary.duration_seconds or 0.0) / 2.0)
        command.extend(["-ss", f"{seek:.3f}", "-i", primary.path])

    vf_prefix = _exr_zscale_filter_prefix() if (is_exr and strategy == "ffmpeg_zscale") else ""
    command.extend(
        [
            "-frames:v",
            "1",
            "-vf",
            f"{vf_prefix}scale='if(gte(iw,ih),1280,-2)':'if(gte(ih,iw),1280,-2)'",
            "-q:v",
            "3",
            str(output),
        ]
    )
    ok, _ = run_command(command, timeout=120)
    return ok and output.exists()


def _extract_preview(primary: SourceRepresentation, output: Path, long_edge: int) -> bool:
    is_exr = _is_exr_source(primary)
    # For previews we cannot afford oiiotool per-frame; fall back to ffmpeg's zscale tonemap when available.
    if is_exr:
        strategy = _tone_map_strategy()
        if strategy == "oiiotool":
            strategy = "ffmpeg_zscale" if _ffmpeg_has_zscale() else "ffmpeg_basic"
        _log_strategy("preview", primary, strategy)
    else:
        strategy = "ffmpeg_basic"

    command = ["ffmpeg", "-y"]
    is_still = primary.format == "single_image"
    if primary.is_sequence:
        command.extend(_sequence_input_args(primary))
    elif is_still:
        command.extend(["-loop", "1", "-framerate", f"{primary.fps or 24.0:.6g}", "-i", primary.path])
    else:
        command.extend(["-i", primary.path])

    vf_prefix = _exr_zscale_filter_prefix() if (is_exr and strategy == "ffmpeg_zscale") else ""
    command.extend(
        [
            "-vf",
            f"{vf_prefix}scale='if(gte(iw,ih),{long_edge},-2)':'if(gte(ih,iw),{long_edge},-2)'",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if is_still:
        command.extend(["-t", "1"])
    command.append(str(output))
    ok, _ = run_command(command, timeout=240)
    return ok and output.exists()


def _sequence_input_args(primary: SourceRepresentation) -> list[str]:
    fps = primary.fps or 24.0
    args = ["-framerate", f"{fps:.6g}"]
    if primary.sequence_first_frame is not None:
        args.extend(["-start_number", str(primary.sequence_first_frame)])
    args.extend(["-i", primary.path])
    return args


def _write_placeholder_svg(path: Path, primary: SourceRepresentation) -> None:
    width = primary.width or 1280
    height = primary.height or 720
    label = Path(primary.original_path).name
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#171a1f"/>
  <rect x="24" y="24" width="{max(1, width - 48)}" height="{max(1, height - 48)}" fill="none" stroke="#4b5563" stroke-width="2"/>
  <text x="50%" y="47%" fill="#e5e7eb" font-family="system-ui, sans-serif" font-size="32" text-anchor="middle">Preview unavailable</text>
  <text x="50%" y="55%" fill="#9ca3af" font-family="system-ui, sans-serif" font-size="20" text-anchor="middle">{_escape_xml(label)}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

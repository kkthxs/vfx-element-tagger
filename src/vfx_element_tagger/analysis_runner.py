#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


from vfx_element_tagger.gated_analysis import GatedSequenceAnalyzer  # noqa: E402
from vfx_element_tagger.model_backends import load_analysis_stack  # noqa: E402
from vfx_element_tagger.models import Element  # noqa: E402
from vfx_element_tagger.review import apply_saved_human_overrides  # noqa: E402
from vfx_element_tagger.store import LibraryStore  # noqa: E402
from vfx_element_tagger.video_activity import profile_video  # noqa: E402
from vfx_element_tagger.settings import library_path, models_dir
from vfx_element_tagger.analysis_cache import analysis_signature, analysis_is_current
from vfx_element_tagger.catalog_lock import catalog_writer


DEFAULT_LIBRARY = library_path()
DEFAULT_MANIFEST = models_dir() / "manifest.json"


def main(argv=None, *, progress=None, should_stop=None, writer_locked=False) -> int:
    parser = argparse.ArgumentParser(
        description="Run uncertainty-gated, whole-duration semantic analysis on a limited catalog slice."
    )
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--models-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum elements to analyse. Defaults to a deliberately small five-element batch.",
    )
    parser.add_argument("--element-id", action="append", default=[])
    parser.add_argument(
        "--selection",
        choices=("diverse", "sequential"),
        default="diverse",
        help="How to choose a limited batch when explicit element IDs are not supplied.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.88)
    parser.add_argument("--max-challengers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--force", action="store_true", help="Recompute selected elements even when their analysis signature matches.")
    parser.add_argument("--no-challengers", action="store_true")
    parser.add_argument("--allow-auto-accept", action="store_true",
                        help="Experimental: accept gate-approved predictions without artist review.")
    parser.add_argument("--dry-run", action="store_true", help="List pending/stale elements without loading models or changing the catalog.")
    args = parser.parse_args(argv)

    if not args.dry_run and not writer_locked:
        with catalog_writer(args.library):
            return _run(args, parser, progress, should_stop)
    return _run(args, parser, progress, should_stop)


def _run(args, parser, progress=None, should_stop=None) -> int:

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    manifest_path = args.models_manifest.expanduser().resolve()
    if not manifest_path.exists():
        parser.error(f"model manifest not found: {manifest_path}")

    catalog_path = args.library.expanduser().resolve()
    if not catalog_path.is_file():
        parser.error(f"catalog not found: {catalog_path}; ingest assets first")
    store = None if args.dry_run else LibraryStore(catalog_path)
    elements = store.load() if store is not None else LibraryStore.load_readonly(catalog_path)
    primary, challengers, manifest = load_analysis_stack(manifest_path)
    if args.no_challengers:
        challengers = []
    analyzer = GatedSequenceAnalyzer(
        primary,
        challengers,
        confidence_threshold=args.confidence_threshold,
        max_challengers=args.max_challengers,
        max_tokens=args.max_tokens,
    )

    options = {"confidence_threshold": args.confidence_threshold, "max_challengers": args.max_challengers,
               "max_tokens": args.max_tokens, "no_challengers": args.no_challengers,
               "allow_auto_accept": args.allow_auto_accept}
    signatures = {element.element_id: analysis_signature(element, manifest, options) for element in elements}
    selected = _select_elements(elements, set(args.element_id), args.force, args.limit,
                                args.selection, signatures)
    if not selected:
        if progress:
            progress({"phase": "analysis", "completed": 0, "total": 0,
                      "message": "No pending or stale elements."})
        print("No matching pending or stale elements.")
        return 0
    print(
        f"Selected {len(selected)} of {len(elements)} elements; primary={primary.model_id}; "
        f"challengers={[item.model_id for item in challengers]}",
        flush=True,
    )
    if args.dry_run:
        for element in selected:
            reason = "forced" if args.force else "pending" if not element.semantic_analysis else "stale"
            print(f"{element.element_id}: {reason} ({Path(element.primary().original_path).name})")
        print("Dry run: no inference and no catalog writes.")
        return 0

    failed = 0
    if progress:
        progress({"phase": "analysis", "completed": 0, "total": len(selected)})
    for index, element in enumerate(selected, start=1):
        if should_stop and should_stop():
            return 130
        if progress:
            progress({"phase": "analysis", "completed": index - 1, "total": len(selected),
                      "current": Path(element.primary().original_path).name})
        rgb_video_path = _video_path(element)
        if rgb_video_path is None:
            element.analysis_status = "failed"
            element.analysis_escalation = {"error": "no preview movie is available"}
            failed += 1
            print(f"[{index}/{len(selected)}] {element.element_id}: skipped (no preview movie)")
            if not args.dry_run:
                store.save_element(element)
            if progress:
                progress({"completed": index, "errors": failed})
            continue

        activity_path = _activity_video_path(element) or rgb_video_path
        rgb_activity = profile_video(activity_path)
        video_path = rgb_video_path
        detail_video_path = _detail_video_path(element) or rgb_video_path
        activity = rgb_activity
        selected_activity_path = activity_path
        semantic_input = "rgb_proxy"
        alpha_path = _alpha_preview_path(element)
        if alpha_path is not None:
            alpha_activity = profile_video(alpha_path)
            if _prefer_alpha_semantic_input(element, rgb_activity, alpha_activity):
                video_path = alpha_path
                detail_video_path = alpha_path
                activity = alpha_activity
                selected_activity_path = alpha_path
                semantic_input = "grayscale_alpha_matte"
        print(
            f"[{index}/{len(selected)}] {element.element_id}: {video_path.name} "
            f"activity={activity.get('temporal_change', 'unavailable')} "
            f"profile={activity.get('profiled_frames', 0)}/{activity.get('source_frame_count', 0)} "
            f"via={selected_activity_path.name} "
            f"semantic_input={semantic_input}",
            flush=True,
        )
        try:
            outcome = analyzer.analyze(
                video_path,
                element.duration_seconds,
                technical_context=_technical_context(
                    element,
                    activity,
                    semantic_input=semantic_input,
                    rgb_activity=rgb_activity,
                ),
                activity=activity,
                detail_video_path=detail_video_path,
            )
            _apply_outcome(element, outcome, activity, allow_auto_accept=args.allow_auto_accept)
            element.analysis_escalation["analysis_signature"] = signatures[element.element_id]
            print(
                f"  -> {element.primary_family}/{element.semantic_analysis.get('effect_type')} "
                f"confidence={element.analysis_confidence:.2f} status={element.analysis_status} "
                f"passes={len(element.analysis_candidates)}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            element.analysis_status = "failed"
            element.analysis_escalation = {
                "error": f"{type(exc).__name__}: {exc}",
                "activity_profile": activity,
            }
            print(f"  -> failed: {type(exc).__name__}: {exc}", flush=True)
        if not args.dry_run:
            store.save_element(element)
        if progress:
            progress({"completed": index, "errors": failed})

    if args.dry_run:
        print("Dry run: catalog was not written.")
    print(f"Finished {len(selected)} elements; failures={failed}.")
    return 1 if failed else 0


def _select_elements(
    elements: list[Element],
    requested_ids: set[str],
    force: bool,
    limit: int,
    selection: str = "diverse",
    signatures: dict[str, str] | None = None,
) -> list[Element]:
    candidates: list[Element] = []
    for element in elements:
        if requested_ids and element.element_id not in requested_ids:
            continue
        if not force and signatures is not None and analysis_is_current(element, signatures[element.element_id]):
            continue
        candidates.append(element)
    if requested_ids or selection == "sequential":
        return candidates[:limit]
    return _diverse_subset(candidates, limit)


def _diverse_subset(candidates: list[Element], limit: int) -> list[Element]:
    remaining = sorted(candidates, key=lambda item: item.element_id)
    selected: list[Element] = []
    seen: set[str] = set()
    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda item: (
                len(_selection_features(item).difference(seen)),
                item.is_sequence,
                item.duration_seconds or 0.0,
                item.element_id,
            ),
        )
        selected.append(best)
        seen.update(_selection_features(best))
        remaining.remove(best)
    return selected


def _selection_features(element: Element) -> set[str]:
    duration = element.duration_seconds or 0.0
    duration_bin = "short" if duration <= 5 else "medium" if duration <= 15 else "long"
    width = element.width or 0
    resolution_bin = "uhd" if width >= 3800 else "2k" if width >= 2000 else "hd_or_less"
    primary = element.primary()
    return {
        f"kind:{'sequence' if element.is_sequence else 'file'}",
        f"duration:{duration_bin}",
        f"alpha:{element.has_alpha_channel}",
        f"format:{primary.format}",
        f"resolution:{resolution_bin}",
        f"fallback_family:{element.category}",
    }


def _video_path(element: Element) -> Path | None:
    for value in (element.preview_movie_1080_path, element.preview_movie_480_path):
        if value and Path(value).exists():
            return Path(value).resolve()
    primary = element.primary()
    path = Path(primary.path)
    if not primary.is_sequence and path.suffix.lower() in {".mov", ".mp4", ".m4v", ".mkv", ".webm"}:
        return path.resolve() if path.exists() else None
    return None


def _activity_video_path(element: Element) -> Path | None:
    for value in (element.preview_movie_480_path, element.preview_movie_1080_path):
        if value and Path(value).exists():
            return Path(value).resolve()
    return _video_path(element)


def _detail_video_path(element: Element) -> Path | None:
    primary = element.primary()
    for value in (primary.path, primary.original_path):
        path = Path(value)
        if (
            not primary.is_sequence
            and path.suffix.lower() in {".mov", ".mp4", ".m4v", ".mkv", ".webm"}
            and path.exists()
        ):
            return path.resolve()
    return _video_path(element)


def _alpha_preview_path(element: Element) -> Path | None:
    for value in (element.preview_movie_480_path, element.preview_movie_1080_path, element.poster_path):
        if not value:
            continue
        candidate = Path(value).resolve().parent / "preview_alpha.mp4"
        if candidate.exists():
            return candidate
    return None


def _prefer_alpha_semantic_input(
    element: Element,
    rgb_activity: dict,
    alpha_activity: dict,
) -> bool:
    if not element.has_alpha_channel or not element.alpha_non_empty:
        return False
    rgb_change = float(rgb_activity.get("temporal_change") or 0.0)
    alpha_change = float(alpha_activity.get("temporal_change") or 0.0)
    fine_detail = float(alpha_activity.get("fine_detail_likelihood") or 0.0)
    return (
        rgb_change <= 0.0001
        and alpha_change >= max(0.0001, rgb_change * 2.0)
        and fine_detail >= 0.35
    )


def _technical_context(
    element: Element,
    activity: dict | None = None,
    *,
    semantic_input: str = "rgb_proxy",
    rgb_activity: dict | None = None,
) -> dict:
    primary = element.primary()
    context = {
        "element_id": element.element_id,
        "duration_seconds": element.duration_seconds,
        "width": element.width,
        "height": element.height,
        "fps": element.fps,
        "bit_depth": element.bit_depth,
        "colour_space": element.color_space,
        "has_alpha_channel": element.has_alpha_channel,
        "alpha_non_empty": element.alpha_non_empty,
        "alpha_is_soft": element.alpha_is_soft,
        "premult_status": element.premult_status,
        "over_black_likely": element.over_black_likely,
        "keyable_bg_likely": element.keyable_bg_likely,
        "is_sequence": element.is_sequence,
        "has_missing_frames": element.has_missing_frames,
        "source_format": primary.format,
        "source_filename_hint": Path(primary.original_path).name,
        "source_technical_metadata": primary.technical_metadata,
        "semantic_input_representation": semantic_input,
    }
    if semantic_input == "grayscale_alpha_matte":
        context["semantic_input_note"] = (
            "The supplied video is the source alpha matte because RGB contrast was insufficient. "
            "White and grey indicate opacity, not literal colour or lighting. Identify silhouettes, "
            "instances and motion from the matte; treat colour and brightness as unknown."
        )
        context["rgb_proxy_temporal_change"] = (rgb_activity or {}).get("temporal_change")
    activity = activity or {}
    if activity.get("key_screen_likely"):
        context.update(
            {
                "measured_key_screen": activity.get("key_screen_likely"),
                "measured_green_screen_fraction": activity.get("green_screen_fraction"),
                "measured_blue_screen_fraction": activity.get("blue_screen_fraction"),
            }
        )
    return context


def _apply_outcome(element: Element, outcome, activity: dict, *, allow_auto_accept: bool = False) -> None:
    analysis = outcome.analysis
    element.semantic_analysis = analysis
    element.primary_family = str(analysis.get("primary_family") or "unknown")
    element.secondary_families = list(analysis.get("secondary_families") or [])
    element.analysis_candidates = outcome.candidates
    element.analysis_confidence = outcome.confidence
    element.analysis_status = outcome.status if allow_auto_accept or outcome.status == "failed" else "needs_review"
    element.analysis_model_version = outcome.model_version
    element.analysis_schema_version = outcome.schema_version
    history = {key: value for key, value in element.analysis_escalation.items()
               if key in {"human_review", "human_review_history"}}
    element.analysis_escalation = {**history, **outcome.escalation, "activity_profile": activity}
    element.analysis_escalation["gate_status"] = outcome.status
    element.analysis_escalation["human_review_required"] = not allow_auto_accept
    # Previous category-specific guesses and caption vectors describe an older
    # result. Do not let them silently contradict the authoritative semantic layer.
    element.content_facets = {}
    element.motion_facets = {}
    element.motion_warnings = []
    element.caption_embed = []
    element.video_embed = []

    # Populate compatibility fields so the current browser/search remains useful
    # during the v0.2 migration.  The structured object above is authoritative.
    element.category = element.primary_family
    element.category_confidence = element.analysis_confidence
    element.caption = str(analysis.get("summary") or "")
    element.captioning_model_version = f"{outcome.model_version} gated-native-video"
    element.qwen_tags = {
        "effect_type": analysis.get("effect_type"),
        "secondary_families": element.secondary_families,
        "search_text": analysis.get("search_text"),
    }
    element.provenance["semantic_analysis"] = {
        "source": "uncertainty_gated_native_video",
        "model": outcome.model_version,
        "schema": outcome.schema_version,
        "confidence": outcome.confidence,
        "status": element.analysis_status,
    }
    # Artist-confirmed fields outrank refreshed model output. New model issues
    # may still return the element to review, but its confirmed tags survive.
    apply_saved_human_overrides(element)


if __name__ == "__main__":
    raise SystemExit(main())

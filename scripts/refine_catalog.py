#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vfx_element_tagger.semantic import (  # noqa: E402
    SEMANTIC_SCHEMA_VERSION,
    apply_deterministic_refinements,
    assess_analysis,
    normalize_analysis,
)
from vfx_element_tagger.review import apply_override_fields  # noqa: E402
from vfx_element_tagger.store import LibraryStore  # noqa: E402
from vfx_element_tagger.temporal_specialist import merge_temporal_specialist  # noqa: E402
from vfx_element_tagger.video_activity import profile_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reapply deterministic measurements without rerunning semantic models."
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--element-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    store = LibraryStore(args.library.resolve())
    elements = store.load()
    requested = set(args.element_id)
    selected = [
        element
        for element in elements
        if (not requested or element.element_id in requested) and element.semantic_analysis
    ][: max(0, args.limit)]

    changed = 0
    for element in selected:
        video_path = _video_path(element)
        if video_path is None:
            print(f"{element.element_id}: skipped (no preview movie)")
            continue
        activity = profile_video(video_path)
        temporal_record = element.analysis_escalation.get("temporal_specialist") or {}
        specialist_result = temporal_record.get("result") if isinstance(temporal_record, dict) else None
        selected_candidate = _selected_candidate_analysis(element) if specialist_result else None
        source_analysis = selected_candidate or element.semantic_analysis
        normalized = normalize_analysis(source_analysis)
        refined, refinements = apply_deterministic_refinements(
            normalized,
            activity,
            _technical_context(element, activity),
        )
        if isinstance(specialist_result, dict):
            refined, temporal_refinements = merge_temporal_specialist(refined, specialist_result)
        else:
            temporal_refinements = []
        override_fields = (element.human_overrides or {}).get("fields") or {}
        if isinstance(override_fields, dict):
            apply_override_fields(refined, override_fields)
        assessment = assess_analysis(refined, activity, context=_technical_context(element, activity))
        candidate_disagreement = bool(element.analysis_escalation.get("candidate_disagreement"))
        if not assessment.accepted:
            element.analysis_status = "needs_review"
            if "semantic_analysis" in element.provenance:
                element.provenance["semantic_analysis"]["status"] = "needs_review"
        adjudicated_resolution = (
            element.analysis_escalation.get("selected_role") == "adjudicator"
        )
        if assessment.accepted and element.analysis_status == "needs_review" and (
            not candidate_disagreement or adjudicated_resolution
        ):
            element.analysis_status = "escalated"
            if "semantic_analysis" in element.provenance:
                element.provenance["semantic_analysis"]["status"] = "escalated"
        schema_changed = refined != element.semantic_analysis
        if schema_changed or refinements or temporal_refinements:
            changed += 1
            element.semantic_analysis = refined
            element.analysis_schema_version = SEMANTIC_SCHEMA_VERSION
            element.primary_family = str(refined.get("primary_family") or element.primary_family)
            element.secondary_families = list(refined.get("secondary_families") or [])
            element.caption = str(refined.get("summary") or element.caption)
            element.qwen_tags = {
                "effect_type": refined.get("effect_type"),
                "secondary_families": element.secondary_families,
                "search_text": refined.get("search_text"),
            }
        previous = list(element.analysis_escalation.get("deterministic_refinements") or [])
        if isinstance(temporal_record, dict) and temporal_record.get("triggered"):
            temporal_record = {**temporal_record, "refinements": temporal_refinements}
        element.analysis_escalation = {
            **element.analysis_escalation,
            "activity_profile": activity,
            "deterministic_refinements": previous + refinements,
            "temporal_specialist": temporal_record,
            "final_assessment": {
                "accepted": assessment.accepted,
                "score": assessment.score,
                "confidence": assessment.confidence,
                "completeness": assessment.completeness,
                "issues": assessment.issues,
            },
        }
        print(
            f"{element.element_id}: normalized={schema_changed} "
            f"refinements={len(refinements) + len(temporal_refinements)} "
            f"fields={[item['field'] for item in refinements + temporal_refinements]}"
        )

    if not args.dry_run:
        if store.is_sqlite:
            for element in selected:
                store.save_element(element)
        else:
            store.save(elements)
    print(f"Processed {len(selected)} elements; changed={changed}; dry_run={args.dry_run}.")
    return 0


def _video_path(element) -> Path | None:
    for value in (element.preview_movie_480_path, element.preview_movie_1080_path):
        if value and Path(value).exists():
            return Path(value).resolve()
    primary = element.primary()
    path = Path(primary.path)
    if not primary.is_sequence and path.exists():
        return path.resolve()
    return None


def _technical_context(element, activity: dict | None = None) -> dict:
    primary = element.primary()
    context = {
        "has_alpha_channel": element.has_alpha_channel,
        "over_black_likely": element.over_black_likely,
        "source_filename_hint": Path(primary.original_path).name,
    }
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


def _selected_candidate_analysis(element) -> dict | None:
    selected_role = element.analysis_escalation.get("selected_role")
    selected_model = element.analysis_escalation.get("selected_model")
    matches = [
        candidate
        for candidate in element.analysis_candidates
        if candidate.get("role") == selected_role
        and candidate.get("model") == selected_model
        and isinstance(candidate.get("result"), dict)
    ]
    if not matches:
        return None
    best = max(matches, key=lambda candidate: float(candidate.get("score") or 0.0))
    return best.get("result")


if __name__ == "__main__":
    raise SystemExit(main())

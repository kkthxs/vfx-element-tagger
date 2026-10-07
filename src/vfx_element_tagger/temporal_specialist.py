from __future__ import annotations

from copy import deepcopy
import json
import re
from pathlib import Path
from typing import Any

from .video_activity import add_frame_ranges


TEMPORAL_SPECIALIST_VERSION = "impact-storyboard-0.1"
DETAIL_SPECIALIST_VERSION = "activity-focused-high-detail-0.1"
MATERIAL_SPECIALIST_VERSION = "ambiguous-atmospheric-material-0.1"

_DELICATE_FAMILIES = {
    "debris",
    "electricity",
    "energy",
    "magic",
    "particle",
    "projectile",
    "rain",
    "snow",
    "spark",
}
_VFX_MATERIALS = {
    "blood",
    "cloud",
    "debris",
    "dust",
    "electricity",
    "energy",
    "fire",
    "fog",
    "light",
    "magic",
    "particle",
    "rain",
    "smoke",
    "snow",
    "spark",
    "steam",
    "water",
}
_AMBIGUOUS_ATMOSPHERIC_FAMILIES = {
    "atmosphere",
    "cloud",
    "dust",
    "fog",
    "rain",
    "smoke",
    "snow",
    "steam",
    "weather",
}


def specialist_reasons(
    analysis: dict[str, Any], context: dict[str, Any] | None = None
) -> list[str]:
    context = context or {}
    filename = str(context.get("source_filename_hint") or "").lower()
    family = str(analysis.get("primary_family") or "")
    secondary = {_token(item) for item in analysis.get("secondary_families") or []}
    motion = analysis.get("motion") if isinstance(analysis.get("motion"), dict) else {}
    characters = {_token(item) for item in motion.get("character") or []}
    depth_motion = _token(motion.get("depth_motion")) or "unknown"

    filename_token = _token(filename)
    subtype = _token(analysis.get("effect_type"))
    impact_hint = ("hit" in filename_token or "impact" in filename_token
                   or any(term in subtype.split("_") for term in ("hit", "impact", "burst", "strike"))
                   or _token(motion.get("temporal_arc")) in {"instantaneous", "burst_then_dissipate"}
                   or "impact" in characters)
    impact_family = family in {"dust", "smoke", "debris", "explosion"}
    missing_material = "debris" not in {family, *secondary}
    missing_depth = depth_motion in {"unknown", "in_plane"}
    missing_fall = not characters.intersection({"falling", "settling"})
    if impact_hint and impact_family and (missing_material or missing_depth or missing_fall):
        reasons = []
        if missing_material:
            reasons.append("impact_material_unresolved")
        if missing_depth:
            reasons.append("impact_depth_unresolved")
        if missing_fall:
            reasons.append("impact_late_phase_unresolved")
        return reasons
    return []


def impact_storyboard_prompt(analysis: dict[str, Any], frame_count: int) -> str:
    known_families = [
        analysis.get("primary_family"),
        *(analysis.get("secondary_families") or []),
    ]
    return f"""You are a VFX temporal motion specialist.
The supplied image is a numbered chronological contact sheet of {frame_count} frames from one
complete isolated element. Determine whether it is an impact or an emission rather than assuming
an impact. Read panels left-to-right, then top-to-bottom. The general
analysis provisionally sees these material families: {json.dumps(known_families)}.

Compare the visible silhouette and particles across every panel. Identify:
- material families actually visible, especially dust, smoke, and discrete debris;
- the initial impact direction and radial expansion;
- whether the effect expands toward camera, away from camera, or only in the image plane;
- whether fragments, particles, or dust later fall downward or settle;
- distinct early, peak, and late motion phases.

Do not use a filename. Do not invent a wall, floor, or source object that is not visible.
Do not judge frame-edge contact; that field is measured separately. Return concise JSON only:
{{
  "timeline_summary": "phase-aware description",
  "observed_materials": ["dust|smoke|debris|particle|fire|other"],
  "initial_screen_motion": "short phrase",
  "depth_motion": "toward_camera|away_from_camera|in_plane|mixed|unknown",
  "later_motion": "short phrase",
  "motion_phases": ["ordered phase phrases"],
  "confidence": 0.0,
  "uncertainty": ["genuine ambiguity only"]
}}"""


def parse_temporal_specialist(raw: str) -> dict[str, Any] | None:
    value = _json_object(raw)
    if value is None:
        return None
    confidence = _number(value.get("confidence")) or 0.0
    return {
        "timeline_summary": _text(value.get("timeline_summary")),
        "observed_materials": _tokens(value.get("observed_materials")),
        "initial_screen_motion": _text(value.get("initial_screen_motion")),
        "depth_motion": _depth_motion(value.get("depth_motion")),
        "later_motion": _text(value.get("later_motion")),
        "motion_phases": _texts(value.get("motion_phases")),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "uncertainty": _texts(value.get("uncertainty")),
    }


def merge_temporal_specialist(
    analysis: dict[str, Any], specialist: dict[str, Any], threshold: float = 0.82
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    merged = deepcopy(analysis)
    confidence = _number(specialist.get("confidence")) or 0.0
    if confidence < threshold:
        return merged, []

    refinements: list[dict[str, Any]] = []
    materials = set(_tokens(specialist.get("observed_materials")))
    secondary = list(merged.get("secondary_families") or [])
    primary = str(merged.get("primary_family") or "")
    phase_text = " ".join(
        [
            _text(specialist.get("timeline_summary")),
            _text(specialist.get("initial_screen_motion")),
            _text(specialist.get("later_motion")),
            *_texts(specialist.get("motion_phases")),
        ]
    ).lower()
    impact_type = _token(merged.get("effect_type"))
    derived_debris = (
        "particle" in materials
        and primary == "dust"
        and any(term in impact_type for term in ("hit", "impact", "burst"))
        and any(term in phase_text for term in ("fall", "settle", "downward", "fragment"))
    )
    if ("debris" in materials or derived_debris) and primary != "debris" and "debris" not in secondary:
        secondary.append("debris")
        merged["secondary_families"] = secondary
        refinements.append(
            {
                "field": "secondary_families",
                "method": TEMPORAL_SPECIALIST_VERSION,
                "added": "debris",
                "derived_from": "impact particles with a falling/settling phase"
                if derived_debris and "debris" not in materials
                else None,
            }
        )

    motion = merged.setdefault("motion", {})
    specialist_depth = _depth_motion(specialist.get("depth_motion"))
    previous_depth = _token(motion.get("depth_motion")) or "unknown"
    accepted_depth = "unknown"
    if specialist_depth in {"toward_camera", "away_from_camera", "mixed"} and previous_depth == "unknown":
        motion["depth_motion"] = specialist_depth
        accepted_depth = specialist_depth
        refinements.append(
            {
                "field": "motion.depth_motion",
                "method": TEMPORAL_SPECIALIST_VERSION,
                "previous": previous_depth,
                "value": specialist_depth,
            }
        )

    characters = list(motion.get("character") or [])
    additions = []
    if any(term in phase_text for term in ("fall", "downward", "settle", "drop")):
        additions.append("falling")
    if any(term in phase_text for term in ("impact", "hit", "initial flash")):
        additions.append("impact")
    if any(term in phase_text for term in ("expand", "outward", "radial")):
        additions.append("expanding")
    for character in additions:
        if character not in characters:
            characters.append(character)
    if additions:
        motion["character"] = characters
        refinements.append(
            {
                "field": "motion.character",
                "method": TEMPORAL_SPECIALIST_VERSION,
                "added": additions,
            }
        )

    accepted_materials = sorted(materials.union({"debris"} if derived_debris else set()))
    temporal_detail = {
        "timeline_summary": _text(specialist.get("timeline_summary")),
        "observed_materials": accepted_materials,
        "initial_screen_motion": _text(specialist.get("initial_screen_motion")),
        "depth_motion": accepted_depth,
        "later_motion": _text(specialist.get("later_motion")),
        "motion_phases": _texts(specialist.get("motion_phases")),
        "confidence": confidence,
        "uncertainty": _texts(specialist.get("uncertainty")),
        "source": TEMPORAL_SPECIALIST_VERSION,
    }
    merged["temporal_detail"] = temporal_detail
    search_addition = " ".join(
        item
        for item in (
            _text(specialist.get("timeline_summary")),
            "debris" if "debris" in accepted_materials else "",
            accepted_depth if accepted_depth != "unknown" else "",
            _text(specialist.get("later_motion")),
        )
        if item
    )
    if search_addition:
        search_text = str(merged.get("search_text") or "")
        merged["search_text"] = f"{search_text}, {search_addition}".strip(", ")
        refinements.append(
            {
                "field": "search_text",
                "method": TEMPORAL_SPECIALIST_VERSION,
                "added": search_addition,
            }
        )
    return merged, refinements


def detail_specialist_reasons(
    analysis: dict[str, Any],
    activity: dict[str, Any] | None = None,
) -> list[str]:
    """Decide whether source-resolution stills can resolve proxy-scale ambiguity."""

    activity = activity or {}
    fine_detail = _number(activity.get("fine_detail_likelihood")) or 0.0
    family = _token(analysis.get("primary_family"))
    secondary = set(_tokens(analysis.get("secondary_families")))
    semantic_text = json.dumps(analysis, sort_keys=True).lower().replace("_", " ")
    delicate_terms = {
        "particle",
        "particles",
        "spark",
        "sparks",
        "ember",
        "embers",
        "grain",
        "grains",
        "droplet",
        "droplets",
        "snow",
        "rain",
        "filament",
        "fine debris",
    }
    reasons: list[str] = []
    if bool(activity.get("high_resolution_recommended")):
        reasons.append("proxy_fine_detail_risk")
    if fine_detail >= 0.42 and {family, *secondary}.intersection(_DELICATE_FAMILIES):
        reasons.append("delicate_family_requires_detail_check")
    if fine_detail >= 0.48 and any(term in semantic_text for term in delicate_terms):
        reasons.append("semantic_fine_structure_requires_detail_check")
    timing = activity.get("action_timing")
    duration = _number(activity.get("duration_seconds")) or 0.0
    if (
        isinstance(timing, dict)
        and timing.get("available")
        and duration >= 12.0
        and (_number(timing.get("coverage_fraction")) or 1.0) <= 0.08
    ):
        reasons.append("brief_action_requires_focused_storyboard")
    return list(dict.fromkeys(reasons))


def material_specialist_reasons(
    analysis: dict[str, Any],
    activity: dict[str, Any] | None = None,
    *,
    candidate_disagreement: bool = False,
) -> list[str]:
    """Route only materially ambiguous atmospheric elements to a focused pass."""

    activity = activity or {}
    family = _token(analysis.get("primary_family"))
    secondary = set(_tokens(analysis.get("secondary_families")))
    candidates = {family, *secondary}.intersection(_AMBIGUOUS_ATMOSPHERIC_FAMILIES)
    if family not in _AMBIGUOUS_ATMOSPHERIC_FAMILIES or len(candidates) < 2:
        return []
    reasons: list[str] = []
    confidence = _number(analysis.get("overall_confidence")) or 0.0
    fine_detail = _number(activity.get("fine_detail_likelihood")) or 0.0
    if candidate_disagreement:
        reasons.append("atmospheric_candidate_disagreement")
    if confidence < 0.90:
        reasons.append("low_confidence_atmospheric_material")
    if fine_detail >= 0.25:
        reasons.append("proxy_scale_material_ambiguity")
    if "snow" in candidates and family != "snow":
        reasons.append("snow_vs_airborne_volume_ambiguity")
    return list(dict.fromkeys(reasons))


def material_storyboard_prompt(
    analysis: dict[str, Any],
    frame_count: int,
    timestamps_seconds: list[float] | None = None,
) -> str:
    return f"""You are a senior VFX atmospheric-material specialist.
The supplied image is a numbered chronological contact sheet of {frame_count} enlarged frames
selected from one complete clip. Read panels left-to-right, then top-to-bottom. Timestamps are:
{json.dumps(timestamps_seconds or [])}
The general video pass proposed:
{json.dumps(analysis, sort_keys=True, default=str)}

Resolve only the dominant visible material and artist-facing library shelf. Compare fine particle
structure, gravity, turbulent roll, dissipation, terrain interaction and motion across panels.
Distinguish wind-blown snow or spindrift from smoke, steam, dust, fog and cloud. Do not call an
effect an avalanche merely because a dark ridge or slope is visible; reserve avalanche for a
clearly sliding mass. Use `atmosphere` for integrated airborne spindrift or snow haze whose useful
identity is its atmospheric shape, and `snow` for snowfall, flakes or a discrete snow burst.
Do not use the filename, and do not change framing or edge-contact fields.

Return concise JSON only:
{{
  "recommended_primary_family": "atmosphere|cloud|dust|fog|rain|smoke|snow|steam|weather|unknown",
  "recommended_effect_type": "short reusable snake_case subtype",
  "observed_materials": ["snow|smoke|steam|dust|fog|cloud|rain|water|other"],
  "morphology": "wispy|billowing|plume|trail|sheet|falling|burst|mixed|unknown",
  "revised_summary": "one precise sentence with no unsupported material",
  "search_terms": ["concise artist search phrases"],
  "evidence": ["visible material cue"],
  "confidence": 0.0,
  "uncertainty": ["genuine ambiguity only"]
}}"""


def parse_material_specialist(raw: str) -> dict[str, Any] | None:
    value = _json_object(raw)
    if value is None:
        return None
    family = _token(value.get("recommended_primary_family")) or "unknown"
    if family not in _AMBIGUOUS_ATMOSPHERIC_FAMILIES:
        family = "unknown"
    confidence = _number(value.get("confidence")) or 0.0
    return {
        "recommended_primary_family": family,
        "recommended_effect_type": _token(value.get("recommended_effect_type")) or "unknown",
        "observed_materials": _tokens(value.get("observed_materials")),
        "morphology": _token(value.get("morphology")) or "unknown",
        "revised_summary": _text(value.get("revised_summary")),
        "search_terms": _texts(value.get("search_terms")),
        "evidence": _texts(value.get("evidence")),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "uncertainty": _texts(value.get("uncertainty")),
        "source": MATERIAL_SPECIALIST_VERSION,
    }


def merge_material_specialist(
    analysis: dict[str, Any],
    specialist: dict[str, Any],
    threshold: float = 0.88,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    merged = deepcopy(analysis)
    confidence = _number(specialist.get("confidence")) or 0.0
    family = _token(specialist.get("recommended_primary_family")) or "unknown"
    effect_type = _token(specialist.get("recommended_effect_type")) or "unknown"
    if confidence < threshold or family not in _AMBIGUOUS_ATMOSPHERIC_FAMILIES:
        return merged, []

    refinements: list[dict[str, Any]] = []
    previous_family = _token(merged.get("primary_family")) or "unknown"
    if family != previous_family:
        merged["primary_family"] = family
        refinements.append(
            {
                "field": "primary_family",
                "method": MATERIAL_SPECIALIST_VERSION,
                "previous": previous_family,
                "value": family,
            }
        )
    previous_effect = _token(merged.get("effect_type")) or "unknown"
    if effect_type != "unknown" and effect_type != previous_effect:
        merged["effect_type"] = effect_type
        refinements.append(
            {
                "field": "effect_type",
                "method": MATERIAL_SPECIALIST_VERSION,
                "previous": previous_effect,
                "value": effect_type,
            }
        )

    observed = [
        value
        for value in _tokens(specialist.get("observed_materials"))
        if value in _VFX_MATERIALS and value != family
    ]
    secondary = []
    for value in [*observed, *_tokens(merged.get("secondary_families"))]:
        if value in {"", "unknown", family, previous_family} or value in secondary:
            continue
        secondary.append(value)
    if secondary != list(merged.get("secondary_families") or []):
        merged["secondary_families"] = secondary
        refinements.append(
            {
                "field": "secondary_families",
                "method": MATERIAL_SPECIALIST_VERSION,
                "value": secondary,
            }
        )

    revised_summary = _text(specialist.get("revised_summary"))
    if len(revised_summary.split()) >= 6 and revised_summary != str(merged.get("summary") or ""):
        merged["summary"] = revised_summary
        refinements.append(
            {
                "field": "summary",
                "method": MATERIAL_SPECIALIST_VERSION,
                "value": revised_summary,
            }
        )
    search_terms = _texts(specialist.get("search_terms"))
    search_text = " ".join(
        value
        for value in (
            revised_summary,
            family.replace("_", " "),
            effect_type.replace("_", " ") if effect_type != "unknown" else "",
            *search_terms,
        )
        if value
    )
    if search_text:
        merged["search_text"] = search_text
        refinements.append(
            {
                "field": "search_text",
                "method": MATERIAL_SPECIALIST_VERSION,
                "value": search_text,
            }
        )
    merged["material_detail"] = deepcopy(specialist)
    return merged, refinements


def detail_storyboard_prompt(
    analysis: dict[str, Any],
    frame_count: int,
    timestamps_seconds: list[float] | None = None,
) -> str:
    return f"""You are a senior VFX element detail specialist.
The supplied image is a numbered chronological contact sheet of {frame_count} frames selected
from one complete element. It mixes full-duration anchors with frames around measured activity
peaks. Read panels left-to-right, then top-to-bottom. The panel timestamps in that order are:
{json.dumps(timestamps_seconds or [])}
A lower-resolution video pass proposed:
{json.dumps(analysis, sort_keys=True, default=str)}

Use the enlarged frames only to resolve fine visual structure that a proxy may lose: particles,
sparks, embers, droplets, grains, hairline electrical branches, small fragments, density and
texture. Also compare panels to identify the visible motion of those details. Do not infer alpha,
resolution, FPS, colour space or premultiplication. Do not use a filename. Do not overturn a
well-supported broad family merely because a few secondary particles are present.

Return concise JSON only:
{{
  "detail_observation": "precise source-resolution visual finding",
  "observed_materials": ["visible VFX material families"],
  "texture": ["fine|billowing|chunky|filament|turbulent|smooth|other"],
  "density": "wispy|light|medium|dense|opaque|mixed|unknown",
  "instance_count": "single|multiple|continuous_many|unknown",
  "motion_character": ["rising|falling|expanding|contracting|drifting|flickering|branching|turbulent|impact|other"],
  "temporal_arc": "instantaneous|build_peak_decay|already_active_decay|continuous|burst_then_dissipate|looped|static|other|unknown",
  "timing": {{
    "visible_start_seconds": 0.0,
    "main_action_start_seconds": 0.0,
    "peak_seconds": 0.0,
    "main_action_end_seconds": 0.0,
    "visible_end_seconds": 0.0,
    "confidence": 0.0
  }},
  "confidence": 0.0,
  "uncertainty": ["genuine ambiguity only"]
}}"""


def parse_detail_specialist(raw: str) -> dict[str, Any] | None:
    value = _json_object(raw)
    if value is None:
        return None
    confidence = _number(value.get("confidence")) or 0.0
    return {
        "detail_observation": _text(value.get("detail_observation")),
        "observed_materials": _tokens(value.get("observed_materials")),
        "texture": _tokens(value.get("texture")),
        "density": _token(value.get("density")) or "unknown",
        "instance_count": _token(value.get("instance_count")) or "unknown",
        "motion_character": _tokens(value.get("motion_character")),
        "temporal_arc": _token(value.get("temporal_arc")) or "unknown",
        "timing": _detail_timing(value.get("timing")),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "uncertainty": _texts(value.get("uncertainty")),
    }


def merge_detail_specialist(
    analysis: dict[str, Any],
    specialist: dict[str, Any],
    threshold: float = 0.84,
    *,
    allow_timing_expansion: bool = False,
    source_fps: float | None = None,
    source_frame_count: int | None = None,
    duration_seconds: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Conservatively merge only high-confidence, detail-scale observations."""

    merged = deepcopy(analysis)
    confidence = _number(specialist.get("confidence")) or 0.0
    if confidence < threshold:
        return merged, []
    refinements: list[dict[str, Any]] = []

    primary = _token(merged.get("primary_family"))
    secondary = list(_tokens(merged.get("secondary_families")))
    added_materials = []
    for material in _tokens(specialist.get("observed_materials")):
        if material not in _VFX_MATERIALS or material == primary or material in secondary:
            continue
        secondary.append(material)
        added_materials.append(material)
    if added_materials:
        merged["secondary_families"] = secondary
        refinements.append(
            {
                "field": "secondary_families",
                "method": DETAIL_SPECIALIST_VERSION,
                "added": added_materials,
            }
        )

    appearance = merged.setdefault("appearance", {})
    existing_texture = [item for item in _tokens(appearance.get("texture")) if item != "unknown"]
    valid_textures = {"fine", "billowing", "chunky", "filament", "turbulent", "smooth", "other"}
    added_textures = [
        item
        for item in _tokens(specialist.get("texture"))
        if item in valid_textures and item not in existing_texture
    ]
    if added_textures:
        appearance["texture"] = existing_texture + added_textures
        refinements.append(
            {
                "field": "appearance.texture",
                "method": DETAIL_SPECIALIST_VERSION,
                "added": added_textures,
            }
        )
    density = _token(specialist.get("density")) or "unknown"
    if _token(appearance.get("density")) in {"", "unknown"} and density != "unknown":
        appearance["density"] = density
        refinements.append(
            {"field": "appearance.density", "method": DETAIL_SPECIALIST_VERSION, "value": density}
        )

    composition = merged.setdefault("composition", {})
    instance_count = _token(specialist.get("instance_count")) or "unknown"
    if _token(composition.get("instance_count")) in {"", "unknown"} and instance_count != "unknown":
        composition["instance_count"] = instance_count
        refinements.append(
            {
                "field": "composition.instance_count",
                "method": DETAIL_SPECIALIST_VERSION,
                "value": instance_count,
            }
        )

    motion = merged.setdefault("motion", {})
    existing_characters = [item for item in _tokens(motion.get("character")) if item != "unknown"]
    allowed_characters = {
        "rising",
        "falling",
        "expanding",
        "contracting",
        "drifting",
        "flickering",
        "branching",
        "turbulent",
        "impact",
        "other",
    }
    added_characters = [
        item
        for item in _tokens(specialist.get("motion_character"))
        if item in allowed_characters and item not in existing_characters
    ]
    if added_characters:
        motion["character"] = existing_characters + added_characters
        refinements.append(
            {
                "field": "motion.character",
                "method": DETAIL_SPECIALIST_VERSION,
                "added": added_characters,
            }
        )
    temporal_arc = _token(specialist.get("temporal_arc")) or "unknown"
    if _token(motion.get("temporal_arc")) in {"", "unknown", "other"} and temporal_arc not in {
        "unknown",
        "other",
    }:
        motion["temporal_arc"] = temporal_arc
        refinements.append(
            {"field": "motion.temporal_arc", "method": DETAIL_SPECIALIST_VERSION, "value": temporal_arc}
        )

    detail_observation = _text(specialist.get("detail_observation"))
    search_text = _text(merged.get("search_text"))
    if detail_observation and detail_observation.lower() not in search_text.lower():
        merged["search_text"] = f"{search_text}, {detail_observation}".strip(", ")
        refinements.append(
            {
                "field": "search_text",
                "method": DETAIL_SPECIALIST_VERSION,
                "added": detail_observation,
            }
        )
    if allow_timing_expansion:
        previous_timing = merged.get("action_timing")
        specialist_timing = specialist.get("timing")
        expanded_timing = _expanded_detail_timing(
            previous_timing if isinstance(previous_timing, dict) else {},
            specialist_timing if isinstance(specialist_timing, dict) else {},
            source_fps,
            source_frame_count,
            duration_seconds,
        )
        if expanded_timing is not None:
            merged["action_timing"] = expanded_timing
            refinements.append(
                {
                    "field": "action_timing",
                    "method": DETAIL_SPECIALIST_VERSION,
                    "previous_optimal_end_seconds": (previous_timing or {}).get(
                        "optimal_end_seconds"
                    ),
                    "value": expanded_timing,
                }
            )
    return merged, refinements


def _detail_timing(value: Any) -> dict[str, Any]:
    timing = value if isinstance(value, dict) else {}
    fields = (
        "visible_start_seconds",
        "main_action_start_seconds",
        "peak_seconds",
        "main_action_end_seconds",
        "visible_end_seconds",
    )
    result = {field: _number(timing.get(field)) for field in fields}
    confidence = _number(timing.get("confidence")) or 0.0
    result["confidence"] = round(max(0.0, min(1.0, confidence)), 4)
    return result


def _expanded_detail_timing(
    previous: dict[str, Any],
    specialist: dict[str, Any],
    source_fps: float | None,
    source_frame_count: int | None,
    duration_seconds: float | None,
) -> dict[str, Any] | None:
    confidence = _number(specialist.get("confidence")) or 0.0
    duration = _number(duration_seconds) or 0.0
    if confidence < 0.88 or duration <= 0 or not previous.get("available"):
        return None
    ordered_fields = (
        "visible_start_seconds",
        "main_action_start_seconds",
        "peak_seconds",
        "main_action_end_seconds",
        "visible_end_seconds",
    )
    proposed = [_number(specialist.get(field)) for field in ordered_fields]
    if any(value is None for value in proposed):
        return None
    values = [max(0.0, min(duration, float(value))) for value in proposed if value is not None]
    if values != sorted(values):
        return None

    previous_visible_end = _number(previous.get("visible_end_seconds")) or 0.0
    proposed_visible_end = values[4]
    minimum_extension = max(0.35, duration * 0.025)
    if proposed_visible_end < previous_visible_end + minimum_extension:
        return None

    previous_main_start = _number(previous.get("main_action_start_seconds")) or values[1]
    previous_peak = _number(previous.get("peak_seconds"))
    previous_main_end = _number(previous.get("main_action_end_seconds")) or values[3]
    main_start = min(previous_main_start, values[1])
    main_end = max(previous_main_end, values[3])
    peak = previous_peak if previous_peak is not None else values[2]
    peak = max(main_start, min(main_end, peak))
    visible_start = min(
        _number(previous.get("visible_start_seconds")) or values[0],
        values[0],
    )
    visible_end = max(previous_visible_end, proposed_visible_end)
    optimal_start = min(
        _number(previous.get("optimal_start_seconds")) or main_start,
        max(0.0, main_start - 0.25),
    )
    optimal_end = max(
        _number(previous.get("optimal_end_seconds")) or main_end,
        min(duration, main_end + 0.5),
    )
    expanded = {
        **previous,
        "version": "source-resolution-storyboard-action-range-0.1",
        "available": True,
        "visible_start_seconds": round(visible_start, 4),
        "main_action_start_seconds": round(main_start, 4),
        "peak_seconds": round(peak, 4),
        "main_action_end_seconds": round(main_end, 4),
        "visible_end_seconds": round(visible_end, 4),
        "optimal_start_seconds": round(optimal_start, 4),
        "optimal_end_seconds": round(optimal_end, 4),
        "optimal_duration_seconds": round(max(0.0, optimal_end - optimal_start), 4),
        "coverage_fraction": round(max(0.0, main_end - main_start) / duration, 4),
        "signal_source": "high_resolution_storyboard_extension",
        "confidence": round(min(0.95, confidence), 4),
        "event_segments": [
            {
                "start_seconds": round(main_start, 4),
                "peak_seconds": round(peak, 4),
                "end_seconds": round(main_end, 4),
            }
        ],
    }
    return add_frame_ranges(expanded, source_fps, source_frame_count)


def build_storyboard(video_path: Path, frame_count: int = 16):
    storyboard, _metadata = _build_storyboard(video_path, frame_count=frame_count)
    return storyboard


def build_focused_storyboard(
    video_path: Path,
    activity: dict[str, Any] | None = None,
    frame_count: int = 16,
):
    focus = (activity or {}).get("analysis_focus") or {}
    timestamps = focus.get("recommended_timestamps_seconds") or []
    return _build_storyboard(
        video_path,
        frame_count=frame_count,
        timestamps=timestamps,
        panel_width=480,
        panel_height=270,
    )


def _build_storyboard(
    video_path: Path,
    frame_count: int = 16,
    timestamps: list[float] | None = None,
    panel_width: int = 320,
    panel_height: int = 180,
):
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    capture = cv2.VideoCapture(str(video_path))
    try:
        source_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        source_width = max(0, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        source_height = max(0, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        indices: list[int] = []
        for timestamp in timestamps or []:
            if source_fps <= 0:
                continue
            index = max(0, min(source_frames - 1, int(round(float(timestamp) * source_fps))))
            if index not in indices:
                indices.append(index)
        for index in np.linspace(0, source_frames - 1, frame_count).round().astype(int):
            value = int(index)
            if value not in indices:
                indices.append(value)
            if len(indices) >= frame_count:
                break
        indices = sorted(indices[:frame_count])
        panels = []
        decoded_timestamps = []
        for panel_number, frame_index in enumerate(indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = min(panel_width / max(1, width), panel_height / max(1, height))
            resized = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            y = (panel_height - resized.shape[0]) // 2
            x = (panel_width - resized.shape[1]) // 2
            panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            timestamp = frame_index / source_fps if source_fps > 0 else 0.0
            decoded_timestamps.append(round(timestamp, 4))
            label = f"{panel_number:02d}  {timestamp:.2f}s"
            cv2.rectangle(panel, (0, 0), (116, 22), (0, 0, 0), -1)
            cv2.putText(
                panel,
                label,
                (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        if not panels:
            raise ValueError(f"No storyboard frames decoded from {video_path}")
        while len(panels) < frame_count:
            panels.append(np.zeros_like(panels[0]))
        rows = [np.hstack(panels[index : index + 4]) for index in range(0, frame_count, 4)]
        sheet = np.vstack(rows)
        return Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)), {
            "frame_count": len(decoded_timestamps),
            "timestamps_seconds": decoded_timestamps,
            "source_width": source_width or None,
            "source_height": source_height or None,
            "source_fps": round(source_fps, 4) if source_fps > 0 else None,
            "panel_width": panel_width,
            "panel_height": panel_height,
            "selection": "activity_peaks_with_uniform_anchors" if timestamps else "uniform",
        }
    finally:
        capture.release()


def _json_object(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for start in [index for index, char in enumerate(stripped) if char == "{"]:
        try:
            value, _end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _depth_motion(value: Any) -> str:
    token = _token(value)
    if "toward" in token or "towards" in token:
        return "toward_camera"
    if "away" in token:
        return "away_from_camera"
    if token in {"in_plane", "mixed", "unknown"}:
        return token
    return "unknown"


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _tokens(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [token for item in values if (token := _token(item))]


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _texts(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [text for item in values if (text := _text(item))]


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

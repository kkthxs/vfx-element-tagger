from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .video_activity import add_frame_ranges


SEMANTIC_SCHEMA_VERSION = "vfx-semantic-0.5"

VFX_FAMILIES = {
    "animal",
    "atmosphere",
    "blood",
    "bird",
    "cloud",
    "creature",
    "crowd",
    "debris",
    "distortion",
    "dust",
    "electricity",
    "energy",
    "environment",
    "explosion",
    "fire",
    "fluid",
    "fog",
    "light",
    "lens_effect",
    "magic",
    "muzzle_flash",
    "particle",
    "plate",
    "projectile",
    "rain",
    "smoke",
    "snow",
    "spark",
    "steam",
    "trail",
    "vegetation",
    "water",
    "weather",
    "unknown",
    "other",
}


@dataclass(frozen=True)
class AnalysisAssessment:
    accepted: bool
    score: float
    confidence: float
    completeness: float
    issues: list[str]


def analysis_prompt(technical_context: dict[str, Any] | None = None) -> str:
    technical = json.dumps(technical_context or {}, sort_keys=True, default=str)
    return f"""You are a senior VFX element librarian analysing one complete moving element.
Watch the whole supplied sequence, including onset, peak action, dissipation, and any loop seam.
Describe the element in vocabulary a compositor, FX artist, or editor would actually search for.

Important rules:
- This is an element, not necessarily a normal photographed scene. Bright sparse pixels can be
  lightning, sparks, energy, muzzle flash, or magic; dark translucent motion can still be smoke.
- Effects are often compound. Choose one primary family and retain every visually meaningful
  secondary family. Never force an explosion-with-debris into only smoke.
- Treat green-screen and blue-screen shoot footage as production plates, not as an isolated
  elemental effect. Use `plate` as the primary family, describe the performer count, wardrobe,
  props and complete action, and use a specific subtype such as `greenscreen_performance_plate`.
  Record the backing, studio equipment or crew outside it, spill/lighting/keying limitations,
  body or prop edge contacts, and whether useful idle handles surround the main performance.
  For plates, `event_count` describes distinct performance takes or action beats, not the number
  of performers. Describe red staining, gore makeup, costume decoration, or damage objectively;
  if the cause is visually ambiguous, retain both useful search interpretations in uncertainty.
- `primary_family` must be one single value from: animal, atmosphere, bird, blood, cloud,
  creature, crowd, debris, distortion, dust, electricity, energy, environment, explosion,
  fire, fluid, fog, light, lens_effect, magic, muzzle_flash, particle, plate, projectile,
  rain, smoke, snow, spark, steam, trail, vegetation, water, weather, other, unknown.
- Judge temporal behaviour from the video, not from one representative frame. Distinguish true
  translation from flicker, growth, turbulence, emission, impact, expansion, and dissipation.
- State framing in useful terms: viewpoint, scale, origin, dominant direction, edges touched,
  and frame coverage. Mention practical usability and cleanup limitations.
- Count distinct simultaneous instances (for example one bolt versus several) and record notable
  secondary objects or occluders with their screen location and cleanup relevance.
- `event_count` is separate: count distinct emissions/strikes/bursts across the whole timeline,
  even when only one instance is visible at any particular moment.
- Separate 2D screen direction from depth motion toward/away from camera. For animals and crowds,
  distinguish circling/orbiting/flocking from simple one-direction translation.
- `edge_contact` means visible effect pixels actually touch or cross that image boundary. Being
  merely near the lower/upper/side region is not edge contact. Check more than one frame.
- Do not infer resolution, FPS, alpha, colour space, bit depth, or premultiplication. Those are
  measured separately. Deterministic context is supplied only to help interpret the media.
- When technical context says `semantic_input_representation` is `grayscale_alpha_matte`, the
  supplied pixels describe opacity because the RGB proxy was too low-contrast. Use the matte to
  identify silhouettes, instance count and motion. Do not report matte white/grey/black as the
  element's literal colour, brightness, backing or lighting; use `unknown` for unavailable RGB
  appearance while still making the strongest supported semantic family judgment.
- Use `unknown` when evidence is genuinely absent. Give timestamped evidence for key judgments.
- For moving clips longer than 3 seconds, give at least early, middle, and late evidence points.
  For shorter moving clips, give at least early and late evidence. Timestamps must be within duration.
- Return JSON only. No markdown fence and no commentary.

Technical context: measured metadata is authoritative. `source_filename_hint` is untrusted and
may be useful, generic, or wrong; use it only to notice a possible conflict worth re-checking:
{technical}

Return exactly this shape, using snake_case values:
{{
  "summary": "one precise natural-language sentence",
  "search_text": "compact synonym-rich search description",
  "primary_family": "one broad VFX family",
  "secondary_families": ["zero or more additional families"],
  "effect_type": "specific subtype such as ground_dust_hit or branching_lightning_strike",
  "composition": {{
    "viewpoint": "front|side|top_down|low_angle|oblique|unknown",
    "shot_scale": "macro|close|medium|wide|very_wide|unknown",
    "origin": "where the action begins in frame",
    "direction": "dominant spatial direction or no_dominant_direction",
    "edge_contact": ["top|bottom|left|right|none"],
    "frame_coverage": "small|medium|large|full_frame",
    "spatial_distribution": "central|off_centre|grounded|overhead|scattered|full_frame|unknown",
    "instance_count": "single|multiple|continuous_many|unknown",
    "notable_objects": [{{"object": "secondary object", "location": "screen region", "role": "foreground|midground|background|occluder|reference"}}]
  }},
  "appearance": {{
    "colour": ["dominant colours"],
    "brightness": "dark|mid|bright|mixed|unknown",
    "density": "wispy|light|medium|dense|opaque|mixed|unknown",
    "texture": ["fine|billowing|chunky|filament|turbulent|smooth|other"],
    "lighting": "self_luminous|top_lit|bottom_lit|side_lit|front_lit|backlit|ambient|mixed|unknown",
    "backing": "transparent|black|white|green_screen|blue_screen|checkerboard|environment|unknown"
  }},
  "motion": {{
    "temporal_arc": "instantaneous|build_peak_decay|already_active_decay|continuous|burst_then_dissipate|looped|static|other",
    "onset": "instant|fast|gradual|already_active|unknown",
    "speed": "still|slow|medium|fast|very_fast|mixed",
    "direction": "motion direction or no_dominant_direction",
    "character": ["rising|falling|expanding|contracting|drifting|circling|orbiting|flocking|flickering|branching|turbulent|impact|walking|turning|gesturing|posing|speaking|interaction|other"],
    "depth_motion": "toward_camera|away_from_camera|in_plane|mixed|unknown",
    "event_count": "single|multiple|continuous_many|unknown",
    "expansion": "none|slight|strong|contracting|mixed|unknown",
    "loopability": "clean_loop|possible_loop|one_shot|not_loopable|unknown"
  }},
  "usability": {{
    "best_uses": ["short practical use phrases"],
    "isolation": "isolated|mostly_isolated|integrated_plate|unknown",
    "cleanup_notes": ["edge crops, backing, occlusion, or other limitations"],
    "layering": "foreground|midground|background|flexible|unknown"
  }},
  "evidence": [{{"timestamp_seconds": 0.0, "observation": "specific visible evidence"}}],
  "uncertainty": ["only genuine ambiguities"],
  "field_confidence": {{"family": 0.0, "composition": 0.0, "appearance": 0.0, "motion": 0.0, "usability": 0.0}},
  "overall_confidence": 0.0
}}"""


def correction_prompt(
    previous: dict[str, Any] | None,
    issues: list[str],
    technical_context: dict[str, Any] | None = None,
) -> str:
    return (
        analysis_prompt(technical_context)
        + "\n\nThis is an uncertainty-resolution pass with denser temporal sampling. "
        "Re-watch the complete element and correct the specific weaknesses below.\n"
        + "Issues: "
        + json.dumps(issues)
        + "\nPrevious candidate (evidence only, not authority):\n"
        + json.dumps(previous or {}, sort_keys=True, default=str)
    )


def adjudication_prompt(
    candidates: list[dict[str, Any]],
    technical_context: dict[str, Any] | None = None,
) -> str:
    compact = [
        {
            "model": candidate.get("model"),
            "role": candidate.get("role"),
            "result": candidate.get("result"),
            "issues": candidate.get("issues"),
            "raw_evidence": candidate.get("raw_excerpt")
            if not candidate.get("result")
            else None,
        }
        for candidate in candidates
    ]
    return (
        analysis_prompt(technical_context)
        + "\n\nAct as adjudicator. Independently watch the complete element, then reconcile the "
        "candidate analyses below. Prefer visible timestamped evidence over consensus. Preserve "
        "compound secondary families when they are present, and do not average incompatible labels.\n"
        + json.dumps(compact, sort_keys=True, default=str)
    )


def parse_analysis(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    starts = [0] if stripped.startswith("{") else []
    starts.extend(index for index, char in enumerate(stripped) if char == "{")
    for start in dict.fromkeys(starts):
        try:
            value, _end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return normalize_analysis(value)
    return None


def normalize_analysis(value: dict[str, Any]) -> dict[str, Any]:
    composition = _mapping(value.get("composition"))
    appearance = _mapping(value.get("appearance"))
    motion = _mapping(value.get("motion"))
    usability = _mapping(value.get("usability"))
    confidence = _number(value.get("overall_confidence"))
    if confidence is None:
        confidence = _number(value.get("confidence")) or 0.0

    result: dict[str, Any] = {
        "summary": _text(value.get("summary")),
        "search_text": _text(value.get("search_text")),
        "primary_family": _token(value.get("primary_family") or value.get("category")) or "unknown",
        "secondary_families": _tokens(value.get("secondary_families")),
        "effect_type": _token(value.get("effect_type")) or "unknown",
        "composition": {
            "viewpoint": _token(composition.get("viewpoint")) or "unknown",
            "shot_scale": _token(composition.get("shot_scale")) or "unknown",
            "origin": _text(composition.get("origin")) or "unknown",
            "direction": _text(composition.get("direction")) or "unknown",
            "edge_contact": _tokens(composition.get("edge_contact")) or ["unknown"],
            "frame_coverage": _token(composition.get("frame_coverage")) or "unknown",
            "spatial_distribution": _token(composition.get("spatial_distribution")) or "unknown",
            "instance_count": _token(composition.get("instance_count")) or "unknown",
            "notable_objects": _notable_objects(composition.get("notable_objects")),
        },
        "appearance": {
            "colour": _tokens(appearance.get("colour") or appearance.get("color")) or ["unknown"],
            "brightness": _token(appearance.get("brightness")) or "unknown",
            "density": _token(appearance.get("density")) or "unknown",
            "texture": _tokens(appearance.get("texture")) or ["unknown"],
            "lighting": _token(appearance.get("lighting")) or "unknown",
            "backing": _token(appearance.get("backing")) or "unknown",
        },
        "motion": {
            "temporal_arc": _token(motion.get("temporal_arc")) or "unknown",
            "onset": _token(motion.get("onset")) or "unknown",
            "speed": _token(motion.get("speed")) or "unknown",
            "direction": _text(motion.get("direction")) or "unknown",
            "character": _tokens(motion.get("character")) or ["unknown"],
            "depth_motion": _token(motion.get("depth_motion")) or "unknown",
            "event_count": _token(motion.get("event_count")) or "unknown",
            "expansion": _token(motion.get("expansion")) or "unknown",
            "loopability": _token(motion.get("loopability")) or "unknown",
        },
        "usability": {
            "best_uses": _texts(usability.get("best_uses")),
            "isolation": _token(usability.get("isolation")) or "unknown",
            "cleanup_notes": _texts(usability.get("cleanup_notes")),
            "layering": _token(usability.get("layering")) or "unknown",
        },
        "evidence": _evidence(value.get("evidence")),
        "uncertainty": _texts(value.get("uncertainty")),
        "temporal_detail": _temporal_detail(value.get("temporal_detail")),
        "action_timing": _action_timing(value.get("action_timing")),
        "field_confidence": _confidence_map(value.get("field_confidence")),
        "overall_confidence": round(max(0.0, min(1.0, confidence)), 4),
    }
    result["secondary_families"] = [
        family for family in result["secondary_families"] if family != result["primary_family"]
    ]
    return result


def assess_analysis(
    result: dict[str, Any] | None,
    activity: dict[str, Any] | None = None,
    confidence_threshold: float = 0.88,
    context: dict[str, Any] | None = None,
) -> AnalysisAssessment:
    if result is None:
        return AnalysisAssessment(False, 0.0, 0.0, 0.0, ["parse_failure"])

    issues: list[str] = []
    confidence = _number(result.get("overall_confidence")) or 0.0
    required = {
        "summary": result.get("summary"),
        "primary_family": result.get("primary_family"),
        "effect_type": result.get("effect_type"),
        "composition.viewpoint": _path(result, "composition", "viewpoint"),
        "composition.shot_scale": _path(result, "composition", "shot_scale"),
        "composition.edge_contact": _path(result, "composition", "edge_contact"),
        "composition.frame_coverage": _path(result, "composition", "frame_coverage"),
        "appearance.colour": _path(result, "appearance", "colour"),
        "appearance.density": _path(result, "appearance", "density"),
        "motion.temporal_arc": _path(result, "motion", "temporal_arc"),
        "motion.speed": _path(result, "motion", "speed"),
        "motion.direction": _path(result, "motion", "direction"),
        "motion.character": _path(result, "motion", "character"),
        "motion.loopability": _path(result, "motion", "loopability"),
        "usability.best_uses": _path(result, "usability", "best_uses"),
        "evidence": result.get("evidence"),
    }
    known = 0
    for name, item in required.items():
        if _known(item):
            known += 1
        else:
            issues.append(f"missing_or_unknown:{name}")
    completeness = known / len(required)

    family = str(result.get("primary_family") or "unknown")
    if family not in VFX_FAMILIES:
        issues.append(f"nonstandard_primary_family:{family}")
    if len(str(result.get("summary") or "").split()) < 6:
        issues.append("summary_too_vague")
    if confidence < confidence_threshold:
        issues.append(f"low_confidence:{confidence:.2f}")
    for field, value in _mapping(result.get("field_confidence")).items():
        number = _number(value)
        if number is None or number < 0.70:
            issues.append(f"low_field_confidence:{field}:{number if number is not None else 'invalid'}")
    if completeness < 0.72:
        issues.append(f"low_completeness:{completeness:.2f}")

    activity = activity or {}
    measured_key_screen = str(activity.get("key_screen_likely") or "")
    measured_backing = str(_path(result, "appearance", "backing") or "unknown")
    effect_type = str(result.get("effect_type") or "unknown")
    expected_screen = (
        "green_screen"
        if effect_type.startswith("greenscreen_")
        else "blue_screen"
        if effect_type.startswith("bluescreen_")
        else ""
    )
    if expected_screen and measured_backing != expected_screen:
        issues.append(
            f"screen_subtype_conflict:expected_{expected_screen}_got_{measured_backing}"
        )
    if family == "plate" and measured_key_screen in {"green_screen", "blue_screen"}:
        if measured_backing != measured_key_screen:
            issues.append(
                f"key_screen_conflict:expected_{measured_key_screen}_got_{measured_backing}"
            )
    temporal_change = _number(activity.get("temporal_change")) or 0.0
    motion_detected = bool(activity.get("motion_detected")) or temporal_change >= 0.025
    transient = _number(activity.get("transient_likelihood")) or 0.0
    duration = _number(activity.get("duration_seconds")) or 0.0
    speed = str(_path(result, "motion", "speed") or "unknown")
    arc = str(_path(result, "motion", "temporal_arc") or "unknown")
    if arc not in {"static", "unknown"}:
        for field in ("speed", "direction"):
            if not _known(_path(result, "motion", field)):
                issues.append(f"unverified_core_motion:{field}")
    for field in ("viewpoint", "shot_scale"):
        if not _known(_path(result, "composition", field)):
            issues.append(f"unverified_core_framing:{field}")
    if activity.get("proxy_information_limited") and (speed == "still" or arc == "static"):
        issues.append("activity_conflict:low_information_cannot_verify_stillness")
    characters = set(_tokens(_path(result, "motion", "character")))
    if motion_detected and speed == "still":
        issues.append("activity_conflict:measured_change_but_speed_still")
    if family != "plate" and transient >= 0.55 and arc in {"static", "continuous", "looped"}:
        issues.append("activity_conflict:transient_signal_but_nontransient_arc")
    if 0 < duration <= 6.0 and transient >= 0.65 and not characters.intersection(
        {"impact", "flickering", "expanding", "branching", "emitting", "bursting"}
    ):
        issues.append("activity_conflict:transient_character_missing")
    measured_timing = activity.get("action_timing")
    event_count = str(_path(result, "motion", "event_count") or "unknown")
    if (
        isinstance(measured_timing, dict)
        and measured_timing.get("available")
        and measured_timing.get("pattern") == "multiple_events"
        and (_number(measured_timing.get("confidence")) or 0.0) >= 0.75
        and event_count == "single"
    ):
        issues.append("activity_conflict:measured_multiple_windows_but_event_count_single")

    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
    evidence_times = [
        _number(item.get("timestamp_seconds")) or 0.0
        for item in evidence
        if isinstance(item, dict)
    ]
    if any(_number(item.get("timestamp_seconds")) is None for item in evidence if isinstance(item, dict)):
        issues.append("evidence_timestamp_invalid")
    if any(timestamp < 0 for timestamp in evidence_times):
        issues.append("evidence_timestamp_out_of_range")
    required_evidence = 3 if duration > 3.0 else 2 if duration > 1.5 else 1
    evidence_horizon = duration * 0.55
    if isinstance(measured_timing, dict) and measured_timing.get("available"):
        coverage = _number(measured_timing.get("coverage_fraction"))
        visible_end = _number(measured_timing.get("visible_end_seconds"))
        if coverage is not None and coverage <= 0.35 and visible_end is not None:
            post_action_handle = max(0.5, duration * 0.05)
            evidence_horizon = min(evidence_horizon, visible_end + post_action_handle)
    if duration > 0 and (
        len(evidence_times) < required_evidence
        or max(evidence_times, default=0.0) < evidence_horizon
    ):
        issues.append("insufficient_temporal_evidence")
    if duration > 0 and any(timestamp > duration * 1.05 + 0.1 for timestamp in evidence_times):
        issues.append("evidence_timestamp_out_of_range")

    secondary = set(_tokens(result.get("secondary_families")))
    cleanup_text = " ".join(_texts(_path(result, "usability", "cleanup_notes"))).lower()
    for family_name in secondary:
        if f"no {family_name}" in cleanup_text:
            issues.append(f"semantic_conflict:secondary_{family_name}_but_cleanup_says_none")
    viewpoint = str(_path(result, "composition", "viewpoint") or "")
    summary_text = str(result.get("summary") or "").lower()
    if viewpoint == "top_down" and "sky" in summary_text and any(
        term in summary_text for term in ("pole", "street light", "lamp post")
    ):
        issues.append("semantic_conflict:top_down_view_but_looking_up_at_sky_and_pole")
    loopability = str(_path(result, "motion", "loopability") or "unknown")
    if (
        family == "explosion"
        and arc in {"instantaneous", "build_peak_decay", "burst_then_dissipate"}
        and loopability in {"clean_loop", "possible_loop"}
    ):
        issues.append("semantic_conflict:one_shot_explosion_marked_loopable")
    declared_uncertainty = " ".join(_texts(result.get("uncertainty"))).lower()
    if any(
        phrase in declared_uncertainty
        for phrase in ("ambiguous", "unclear", "cannot determine", "uncertain")
    ):
        issues.append("model_declared_material_uncertainty")
    lighting_value = str(_path(result, "appearance", "lighting") or "unknown")
    verified_key_plate = (
        family == "plate"
        and measured_key_screen in {"green_screen", "blue_screen"}
        and measured_backing == measured_key_screen
    )
    if not verified_key_plate and lighting_value == "unknown" and any(
        term in declared_uncertainty for term in ("light", "lighting", "illumination")
    ):
        issues.append("model_declared_field_uncertainty:lighting")

    backing = str(_path(result, "appearance", "backing") or "")
    has_border_measurement = "bright_border_contact" in activity
    measured_edges = set(_tokens(activity.get("bright_border_contact")))
    claimed_edges = set(_tokens(_path(result, "composition", "edge_contact")))
    claimed_edges.discard("none")
    claimed_edges.discard("unknown")
    if _edge_measurement_applies(backing, context) and has_border_measurement:
        for edge_name in sorted(claimed_edges.difference(measured_edges)):
            issues.append(f"measured_edge_conflict:claimed_{edge_name}_without_border_pixels")
        for edge_name in sorted(measured_edges.difference(claimed_edges)):
            issues.append(f"measured_edge_conflict:missed_{edge_name}_border_contact")

    filename_hint = str((context or {}).get("source_filename_hint") or "").lower()
    filename_terms = set(_token(filename_hint).split("_"))
    semantic_text = json.dumps(result, sort_keys=True).lower().replace("_", " ")
    semantic_terms = set(re.findall(r"[a-z0-9]+", semantic_text))
    family_set = {family, *_tokens(result.get("secondary_families"))}
    if filename_terms.intersection({"hit", "impact", "strike", "burst"}) and not semantic_terms.intersection(
        {"impact", "impacting", "hit", "strike", "burst", "explode", "explodes", "exploding", "explosion"}
    ):
        issues.append("filename_visual_disagreement:impact_cue_missing_from_analysis")
    filename_family_hints = {
        "bird": {"bird", "animal", "plate"},
        "birds": {"bird", "animal", "plate"},
        "cloud": {"cloud", "fog", "atmosphere", "smoke"},
        "debris": {"debris", "dust", "particle"},
        "lightning": {"electricity", "light", "energy"},
        "steam": {"steam", "smoke", "atmosphere"},
    }
    for hint, expected_families in filename_family_hints.items():
        if hint in filename_terms and not family_set.intersection(expected_families):
            issues.append(f"filename_visual_disagreement:{hint}_family_cue")

    critical_prefixes = (
        "parse_failure",
        "low_confidence",
        "low_field_confidence",
        "nonstandard_primary_family",
        "unverified_core_motion",
        "unverified_core_framing",
        "low_completeness",
        "activity_conflict",
        "insufficient_temporal_evidence",
        "evidence_timestamp_out_of_range",
        "evidence_timestamp_invalid",
        "semantic_conflict",
        "filename_visual_disagreement",
        "model_declared_material_uncertainty",
        "model_declared_field_uncertainty",
        "measured_edge_conflict",
        "key_screen_conflict",
        "screen_subtype_conflict",
        "missing_or_unknown:primary_family",
        "missing_or_unknown:motion.temporal_arc",
    )
    critical = any(issue.startswith(critical_prefixes) for issue in issues)
    issue_penalty = min(0.30, 0.035 * len(issues))
    score = max(0.0, min(1.0, 0.50 * confidence + 0.50 * completeness - issue_penalty))
    return AnalysisAssessment(not critical, round(score, 4), confidence, round(completeness, 4), issues)


def apply_deterministic_refinements(
    result: dict[str, Any],
    activity: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge measurements that are more reliable than free-form visual inference."""

    refined = deepcopy(result)
    activity = activity or {}
    context = context or {}
    refinements: list[dict[str, Any]] = []
    backing = str(_path(refined, "appearance", "backing") or "")

    if context.get("semantic_input_representation") == "grayscale_alpha_matte":
        appearance = refined.setdefault("appearance", {})
        measured_values = {
            "backing": "transparent",
            "colour": ["unknown"],
            "brightness": "unknown",
            "lighting": "unknown",
        }
        for field, value in measured_values.items():
            previous = deepcopy(appearance.get(field))
            if previous == value:
                continue
            appearance[field] = deepcopy(value)
            refinements.append(
                {
                    "field": f"appearance.{field}",
                    "method": "alpha_matte_input_guard",
                    "previous": previous,
                    "value": deepcopy(value),
                }
            )
        for field in ("summary", "search_text"):
            previous_text = str(refined.get(field) or "")
            safe_text = _alpha_matte_safe_text(previous_text)
            if safe_text == previous_text:
                continue
            refined[field] = safe_text
            refinements.append(
                {
                    "field": field,
                    "method": "alpha_matte_input_guard",
                    "previous": previous_text,
                    "value": safe_text,
                }
            )
        backing = "transparent"

    if "bright_border_contact" in activity and _edge_measurement_applies(backing, context):
        measured = sorted(set(_tokens(activity.get("bright_border_contact"))))
        measured_value = measured or ["none"]
        composition = refined.setdefault("composition", {})
        claimed = _tokens(composition.get("edge_contact"))
        if claimed != measured_value:
            composition["edge_contact"] = measured_value
            refinements.append(
                {
                    "field": "composition.edge_contact",
                    "method": "measured_bright_border_contact",
                    "previous": claimed,
                    "value": measured_value,
                }
            )

    family = str(refined.get("primary_family") or "")
    motion = refined.setdefault("motion", {})
    temporal_arc = str(motion.get("temporal_arc") or "unknown")
    loopability = str(motion.get("loopability") or "unknown")
    if family == "plate" and loopability == "unknown":
        motion["loopability"] = "not_loopable"
        loopability = "not_loopable"
        refinements.append(
            {
                "field": "motion.loopability",
                "method": "production_plate_canonicalization",
                "previous": "unknown",
                "value": "not_loopable",
            }
        )
    if (
        family == "explosion"
        and temporal_arc in {"instantaneous", "build_peak_decay", "burst_then_dissipate"}
        and loopability in {"clean_loop", "possible_loop"}
        and str(motion.get("event_count") or "unknown") in {"single", "unknown"}
    ):
        motion["loopability"] = "one_shot"
        refinements.append(
            {
                "field": "motion.loopability",
                "method": "transient_explosion_canonicalization",
                "previous": loopability,
                "value": "one_shot",
            }
        )
    vertical_bias = _number(activity.get("vertical_luminance_bias"))
    non_emissive_volumes = {"cloud", "smoke", "steam", "fog", "dust", "debris", "atmosphere"}
    if (
        family in non_emissive_volumes
        and vertical_bias is not None
        and vertical_bias >= 0.015
        and _edge_measurement_applies(backing, context)
    ):
        appearance = refined.setdefault("appearance", {})
        previous_lighting = str(appearance.get("lighting") or "unknown")
        if previous_lighting != "top_lit":
            appearance["lighting"] = "top_lit"
            refinements.append(
                {
                    "field": "appearance.lighting",
                    "method": "foreground_vertical_luminance_bias",
                    "previous": previous_lighting,
                    "value": "top_lit",
                    "measured_bias": round(vertical_bias, 6),
                }
            )

    semantic_text = json.dumps(refined, sort_keys=True).lower()
    if family == "electricity" and "branch" in semantic_text:
        search_text = str(refined.get("search_text") or "")
        if "multiple branch" not in search_text.lower():
            refined["search_text"] = f"{search_text}, multiple branches, forked lightning".strip(", ")
            refinements.append(
                {
                    "field": "search_text",
                    "method": "branching_electricity_search_expansion",
                    "value": "multiple branches, forked lightning",
                }
            )

    measured_timing = activity.get("action_timing")
    if isinstance(measured_timing, dict) and measured_timing.get("available"):
        timing = _action_timing(measured_timing)
        duration = _number(activity.get("duration_seconds")) or 0.0
        if family == "plate" and duration > 0:
            timing.update(optimal_start_seconds=0.0, optimal_end_seconds=round(duration, 4),
                          optimal_duration_seconds=round(duration, 4),
                          verification="full_clip_until_artist_review", review_required=True)
        temporal_arc = str(_path(refined, "motion", "temporal_arc") or "")
        visible_start = _number(timing.get("visible_start_seconds")) or 0.0
        visible_end = _number(timing.get("visible_end_seconds")) or 0.0
        visible_coverage = max(0.0, visible_end - visible_start) / max(duration, 1e-6)
        if (
            family != "plate"
            and temporal_arc in {"continuous", "looped", "static"}
            and timing.get("pattern") == "burst"
            and duration > 0
            and visible_coverage >= 0.60
        ):
            timing = {
                **timing,
                "main_action_start_seconds": round(visible_start, 4),
                "peak_seconds": round((visible_start + visible_end) / 2.0, 4),
                "main_action_end_seconds": round(visible_end, 4),
                "optimal_start_seconds": round(visible_start, 4),
                "optimal_end_seconds": round(visible_end, 4),
                "optimal_duration_seconds": round(visible_end - visible_start, 4),
                "coverage_fraction": round(visible_coverage, 4),
                "pattern": "continuous",
                "event_segments": [
                    {
                        "start_seconds": round(visible_start, 4),
                        "peak_seconds": round((visible_start + visible_end) / 2.0, 4),
                        "end_seconds": round(visible_end, 4),
                    }
                ],
                "signal_source": "semantic_continuous_visible_range",
                "confidence": max(0.8, float(timing.get("confidence") or 0.0)),
            }
        if backing == "environment" and temporal_arc in {"continuous", "looped", "static"} and duration > 0:
            timing = {
                **timing,
                "visible_start_seconds": 0.0,
                "main_action_start_seconds": 0.0,
                "peak_seconds": round(duration / 2.0, 4),
                "main_action_end_seconds": round(duration, 4),
                "visible_end_seconds": round(duration, 4),
                "optimal_start_seconds": 0.0,
                "optimal_end_seconds": round(duration, 4),
                "optimal_duration_seconds": round(duration, 4),
                "coverage_fraction": 1.0,
                "pattern": "continuous",
                "event_segments": [
                    {
                        "start_seconds": 0.0,
                        "peak_seconds": round(duration / 2.0, 4),
                        "end_seconds": round(duration, 4),
                    }
                ],
                "signal_source": "semantic_continuous_override",
                "confidence": max(0.8, float(timing.get("confidence") or 0.0)),
            }
        timing = add_frame_ranges(
            timing,
            _number(activity.get("source_fps")),
            int(_number(activity.get("source_frame_count")) or 0),
        )
        previous_timing = _action_timing(refined.get("action_timing"))
        if timing != previous_timing:
            refined["action_timing"] = timing
            refinements.append(
                {
                    "field": "action_timing",
                    "method": timing.get("version") or "measured_action_timing",
                    "value": timing,
                }
            )

    return refined, refinements


def _alpha_matte_safe_text(value: str) -> str:
    text = re.sub(
        r"\b(?:(?:against|over|on)\s+(?:a\s+)?(?:solid\s+|pure\s+)?)?"
        r"(?:black|white|grey|gray)\s+(?:background|backing|void)\b",
        "on a transparent background",
        value,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:white|grey|gray|black)[,\s-]+"
        r"(?=(?:bird|birds|silhouette|silhouettes|shape|shapes|particle|particles|object|objects)\b)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s{2,}", " ", text).strip()


def _edge_measurement_applies(backing: str, context: dict[str, Any] | None) -> bool:
    backing_token = _token(backing)
    context = context or {}
    return (
        "black" in backing_token
        or backing_token in {"transparent", "alpha", "isolated"}
        or bool(context.get("has_alpha_channel"))
        or bool(context.get("over_black_likely"))
    )


def _temporal_detail(value: Any) -> dict[str, Any]:
    detail = _mapping(value)
    if not detail:
        return {}
    confidence = _number(detail.get("confidence")) or 0.0
    return {
        "timeline_summary": _text(detail.get("timeline_summary")),
        "observed_materials": _tokens(detail.get("observed_materials")),
        "initial_screen_motion": _text(detail.get("initial_screen_motion")),
        "depth_motion": _token(detail.get("depth_motion")) or "unknown",
        "later_motion": _text(detail.get("later_motion")),
        "motion_phases": _texts(detail.get("motion_phases")),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "uncertainty": _texts(detail.get("uncertainty")),
        "source": _token(detail.get("source")) or "unknown",
    }


def _action_timing(value: Any) -> dict[str, Any]:
    timing = _mapping(value)
    if not timing or not timing.get("available", True):
        return {}
    number_fields = (
        "visible_start_seconds",
        "main_action_start_seconds",
        "peak_seconds",
        "main_action_end_seconds",
        "visible_end_seconds",
        "optimal_start_seconds",
        "optimal_end_seconds",
        "optimal_duration_seconds",
        "coverage_fraction",
        "confidence",
        "source_fps",
    )
    integer_fields = (
        "source_frame_count",
        "visible_start_frame",
        "main_action_start_frame",
        "peak_frame",
        "main_action_end_frame",
        "visible_end_frame",
        "optimal_start_frame",
        "optimal_end_frame",
        "optimal_duration_frames",
    )
    result: dict[str, Any] = {
        "version": _token(timing.get("version")) or "unknown",
        "available": True,
        "verification": _token(timing.get("verification")) or "unverified",
        "review_required": bool(timing.get("review_required", True)),
        "pattern": _token(timing.get("pattern")) or "unknown",
        "signal_source": _token(timing.get("signal_source")) or "unknown",
        "sample_count": int(_number(timing.get("sample_count")) or 0),
        "frame_numbering": _token(timing.get("frame_numbering")) or "unknown",
        "event_segments": [],
    }
    for field in number_fields:
        number = _number(timing.get(field))
        result[field] = round(number, 4) if number is not None else None
    for field in integer_fields:
        number = _number(timing.get(field))
        result[field] = int(number) if number is not None else None
    for segment in timing.get("event_segments") or []:
        if not isinstance(segment, dict):
            continue
        normalized_segment = {
            "start_seconds": round(_number(segment.get("start_seconds")) or 0.0, 4),
            "peak_seconds": round(_number(segment.get("peak_seconds")) or 0.0, 4),
            "end_seconds": round(_number(segment.get("end_seconds")) or 0.0, 4),
        }
        for field in ("start_frame", "peak_frame", "end_frame"):
            number = _number(segment.get(field))
            if number is not None:
                normalized_segment[field] = int(number)
        result["event_segments"].append(normalized_segment)
    return result


def analyses_agree(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_primary = first.get("primary_family")
    second_primary = second.get("primary_family")
    if first_primary != second_primary:
        first_set = {first_primary, *_tokens(first.get("secondary_families"))}
        second_set = {second_primary, *_tokens(second.get("secondary_families"))}
        if not first_set.intersection(second_set):
            return False
    first_type = first.get("effect_type")
    second_type = second.get("effect_type")
    if first_type not in {"unknown", second_type} and second_type != "unknown":
        return False
    first_arc = _path(first, "motion", "temporal_arc")
    second_arc = _path(second, "motion", "temporal_arc")
    if first_arc not in {"unknown", second_arc} and second_arc != "unknown":
        return False
    for section, field in (("motion", "direction"), ("motion", "depth_motion"),
                           ("motion", "event_count"), ("composition", "instance_count"),
                           ("composition", "edge_contact"), ("appearance", "backing")):
        left, right = _path(first, section, field), _path(second, section, field)
        if not _known(left) or not _known(right):
            continue
        if _agreement_value(left) != _agreement_value(right):
            return False
    return True


def _agreement_value(value: Any):
    aliases = {"upwards": "up", "upward": "up", "rising": "up",
               "downwards": "down", "downward": "down", "falling": "down",
               "leftwards": "left", "rightwards": "right"}
    if isinstance(value, list):
        return frozenset(_agreement_value(item) for item in value)
    token = _token(value)
    return aliases.get(token, token)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _token(value: Any) -> str:
    return "_".join(_text(value).lower().replace("/", " ").replace("-", " ").split())


def _texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple, set)):
        return [_text(value)] if _text(value) else []
    return [_text(item) for item in value if _text(item)]


def _tokens(value: Any) -> list[str]:
    result: list[str] = []
    for item in _texts(value):
        token = _token(item)
        if token and token not in result:
            result.append(token)
    return result


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


def _confidence_map(value: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in _mapping(value).items():
        number = _number(item)
        result[_token(key)] = round(max(0.0, min(1.0, number)), 4) if number is not None else 0.0
    return result


def _evidence(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return result
    for item in value:
        if not isinstance(item, dict):
            continue
        observation = _text(item.get("observation"))
        timestamp = _number(item.get("timestamp_seconds"))
        if observation:
            result.append({"timestamp_seconds": timestamp, "observation": observation})
    return result


def _notable_objects(value: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if not isinstance(value, list):
        return result
    for item in value:
        if not isinstance(item, dict):
            continue
        object_name = _text(item.get("object"))
        if not object_name:
            continue
        result.append(
            {
                "object": object_name,
                "location": _token(item.get("location")) or "unknown",
                "role": _token(item.get("role")) or "unknown",
            }
        )
    return result


def _path(value: dict[str, Any], *parts: str) -> Any:
    current: Any = value
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _known(value: Any) -> bool:
    if value is None or value == "" or value == "unknown":
        return False
    if isinstance(value, list):
        return bool(value) and any(
            item is not None and item != "" and item != "unknown" for item in value
        )
    return True

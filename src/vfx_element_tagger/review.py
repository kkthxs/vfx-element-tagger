from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .models import Element
from .review_vocabulary import EFFECT_TYPES_BY_FAMILY, vocabulary_for
from .stages import VECTOR_DIMS, embed_element


FIELD_LABELS = {
    "primary_family": "Top-level family",
    "secondary_families": "Related families",
    "effect_type": "Element subtype",
    "composition.viewpoint": "Viewpoint",
    "composition.shot_scale": "Shot scale",
    "composition.origin": "Origin in frame",
    "composition.direction": "Compositional direction",
    "composition.edge_contact": "Edges touched",
    "composition.frame_coverage": "Frame coverage",
    "composition.spatial_distribution": "Spatial distribution",
    "motion.temporal_arc": "Action pattern",
    "motion.onset": "Onset",
    "motion.speed": "Speed",
    "motion.direction": "Motion direction",
    "motion.character": "Motion character",
    "motion.depth_motion": "Depth motion",
    "motion.event_count": "Event count",
    "motion.expansion": "Expansion",
    "motion.loopability": "Loopability",
    "appearance.backing": "Backing",
    "appearance.colour": "Colours",
    "appearance.density": "Density",
    "appearance.texture": "Texture",
    "appearance.lighting": "Lighting",
}

LIST_FIELDS = {
    "secondary_families",
    "composition.edge_contact",
    "motion.character",
    "appearance.colour",
    "appearance.texture",
}

FIELD_ORDER = list(FIELD_LABELS)
REVIEWABLE_FIELDS = set(FIELD_ORDER)


def build_review_context(element: Element) -> dict[str, Any]:
    analysis = element.semantic_analysis or {}
    escalation = element.analysis_escalation or {}
    final_issues = list((escalation.get("final_assessment") or {}).get("issues") or [])
    uncertainty = [str(value) for value in analysis.get("uncertainty") or [] if value]
    reasons = [humanize_issue(value) for value in final_issues]
    if escalation.get("human_review_required"):
        reasons.append("Artist confirmation is required for new AI results; model confidence is not calibrated accuracy.")
    if (escalation.get("ingest_refresh") or {}).get("source_changed"):
        reasons.append("Source footage changed. Retained descriptions and previous artist decisions must be checked against the new pixels.")
    if escalation.get("candidate_disagreement"):
        reasons.append("The primary and challenger models disagree on one or more tags.")
    reasons.extend(uncertainty)
    reasons = _dedupe(reasons)
    if not reasons:
        reasons = ["The analysis gate could not verify this result strongly enough to accept it automatically."]

    paths = {"primary_family", "effect_type"}
    path_reasons: dict[str, list[str]] = {}
    for issue in final_issues:
        message = humanize_issue(issue)
        for path in _paths_for_text(issue):
            paths.add(path)
            path_reasons.setdefault(path, []).append(message)
    for value in uncertainty:
        for path in _paths_for_text(value):
            paths.add(path)
            path_reasons.setdefault(path, []).append(value)

    candidates = [
        candidate
        for candidate in element.analysis_candidates
        if isinstance(candidate, dict) and isinstance(candidate.get("result"), dict)
    ]
    if escalation.get("candidate_disagreement"):
        for path in FIELD_ORDER:
            values = [_analysis_value(analysis, path)]
            values.extend(_analysis_value(candidate["result"], path) for candidate in candidates)
            if len({_value_key(value) for value in values if _present(value)}) > 1:
                paths.add(path)
                path_reasons.setdefault(path, []).append(
                    "The model passes proposed different values for this field."
                )

    ordered_paths = [path for path in FIELD_ORDER if path in paths][:10]
    fields = [
        _review_field(path, analysis, candidates, path_reasons.get(path) or [])
        for path in ordered_paths
    ]
    candidate_summary = []
    for candidate in candidates:
        result = candidate["result"]
        candidate_summary.append(
            {
                "role": candidate.get("role") or "candidate",
                "model": candidate.get("model") or "unknown model",
                "confidence": candidate.get("confidence"),
                "accepted": bool(candidate.get("accepted")),
                "primary_family": result.get("primary_family"),
                "effect_type": result.get("effect_type"),
            }
        )
    return {
        "review_reasons": reasons,
        "review_fields": fields,
        "candidate_summary": candidate_summary,
        "selected_role": escalation.get("selected_role"),
        "selected_model": escalation.get("selected_model"),
    }


def review_element(element: Element, selections: dict[str, Any], note: str = "") -> dict[str, Any]:
    if element.analysis_status != "needs_review":
        raise ValueError("Element is not currently awaiting review")
    if not isinstance(selections, dict):
        raise ValueError("fields must be a JSON object")
    clean_fields: dict[str, Any] = {}
    for path, value in selections.items():
        if path not in REVIEWABLE_FIELDS:
            raise ValueError(f"Field is not reviewable: {path}")
        clean_fields[path] = normalize_review_value(path, value)
    if not clean_fields:
        raise ValueError("Confirm at least one review field")

    previous_status = element.analysis_status
    before = {path: _analysis_value(element.semantic_analysis, path) for path in clean_fields}
    apply_override_fields(element.semantic_analysis, clean_fields)
    classification_changed = any(
        path in {"primary_family", "effect_type"} and before.get(path) != value
        for path, value in clean_fields.items()
    )
    if classification_changed:
        _rebuild_reviewed_text(element.semantic_analysis)
    _sync_compatibility_fields(element)
    _refresh_review_embedding(element)
    reviewed_at = datetime.now(tz=timezone.utc).isoformat()
    existing_fields = dict((element.human_overrides or {}).get("fields") or {})
    element.human_overrides = {
        **(element.human_overrides or {}),
        "fields": {**existing_fields, **deepcopy(clean_fields)},
        "reviewed_at": reviewed_at,
        "review_note": str(note or "").strip()[:2000],
        "previous_status": previous_status,
    }
    review_record = {
        "reviewed_at": reviewed_at,
        "note": element.human_overrides["review_note"],
        "fields": deepcopy(clean_fields),
        "previous_values": before,
    }
    history = list((element.analysis_escalation or {}).get("human_review_history") or [])
    element.analysis_escalation = {
        **(element.analysis_escalation or {}),
        "human_review": review_record,
        "human_review_history": history + [review_record],
    }
    provenance = dict(element.provenance.get("semantic_analysis") or {})
    element.provenance["semantic_analysis"] = {
        **provenance,
        "status": "accepted",
        "human_reviewed": True,
        "human_reviewed_at": reviewed_at,
    }
    element.analysis_status = "accepted"
    return review_record


def apply_saved_human_overrides(element: Element) -> None:
    fields = (element.human_overrides or {}).get("fields") or {}
    if not isinstance(fields, dict) or not fields:
        return
    before = {path: _analysis_value(element.semantic_analysis, path) for path in fields}
    apply_override_fields(element.semantic_analysis, fields)
    if any(
        path in {"primary_family", "effect_type"} and before.get(path) != value
        for path, value in fields.items()
    ):
        _rebuild_reviewed_text(element.semantic_analysis)
    _sync_compatibility_fields(element)
    _refresh_review_embedding(element)


def apply_override_fields(analysis: dict[str, Any], fields: dict[str, Any]) -> None:
    for path, value in fields.items():
        if path not in REVIEWABLE_FIELDS:
            continue
        _set_analysis_value(analysis, path, deepcopy(value))


def normalize_review_value(path: str, value: Any) -> Any:
    if path in LIST_FIELDS:
        values = value if isinstance(value, list) else str(value or "").split(",")
        normalized = [_token(item) for item in values if str(item).strip()]
        return _dedupe(normalized) or ["none"]
    if isinstance(value, (dict, list)):
        raise ValueError(f"Expected one value for {path}")
    normalized = _token(value)
    if not normalized:
        raise ValueError(f"Value cannot be empty for {path}")
    return normalized


def humanize_issue(issue: str) -> str:
    value = str(issue or "")
    if "model_declared_material_uncertainty" in value:
        return "The model is unsure which material or effect family is actually visible."
    if "measured_multiple_windows_but_event_count_single" in value:
        return "Measured activity contains multiple windows, but the model described a single event."
    if "insufficient_temporal_evidence" in value:
        return "The model did not provide enough observations across the clip to verify the full action."
    if "candidate_disagreement" in value:
        return "The primary and challenger models disagree on one or more tags."
    if "low_confidence" in value:
        score = value.rsplit(":", 1)[-1]
        return f"Overall model confidence is low ({score})."
    if "parse_failure" in value:
        return "One model pass did not return a valid structured result."
    if "measured_edge_conflict" in value:
        return "The model's edge-contact description conflicted with measured border pixels."
    if "screen_subtype_conflict" in value:
        return "The proposed screen-plate subtype does not match the backing visible in the clip."
    if "activity_conflict" in value:
        return "The described action pattern conflicted with measured temporal activity."
    return value.replace("_", " ").replace(":", ": ").strip().capitalize()


def _review_field(
    path: str,
    analysis: dict[str, Any],
    candidates: list[dict[str, Any]],
    reasons: list[str],
) -> dict[str, Any]:
    current = _analysis_value(analysis, path)
    options: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_option(value: Any, source: str) -> None:
        if not _present(value):
            return
        key = _value_key(value)
        if key in seen:
            for option in options:
                if _value_key(option["value"]) == key and source not in option["sources"]:
                    option["sources"].append(source)
            return
        seen.add(key)
        options.append({"value": deepcopy(value), "sources": [source]})

    add_option(current, "selected result")
    for candidate in candidates:
        role = str(candidate.get("role") or "candidate").replace("_", " ")
        model = str(candidate.get("model") or "model").rsplit("/", 1)[-1]
        add_option(_analysis_value(candidate["result"], path), f"{role} · {model}")
    if not options:
        add_option(["none"] if path in LIST_FIELDS else "unknown", "selected result")
    return {
        "path": path,
        "label": FIELD_LABELS[path],
        "current": deepcopy(current),
        "options": options,
        "vocabulary": vocabulary_for(path, str(analysis.get("primary_family") or "unknown")),
        "vocabulary_by_family": EFFECT_TYPES_BY_FAMILY if path == "effect_type" else {},
        "multiple": path in LIST_FIELDS,
        "reason": " ".join(_dedupe(reasons)) or _default_field_reason(path),
    }


def _default_field_reason(path: str) -> str:
    if path == "primary_family":
        return "Confirm the proposed top-level organisational family."
    if path == "effect_type":
        return "Confirm the more specific subtype artists should browse and search."
    return "Confirm the selected model value or replace it with your own."


def _paths_for_text(value: str) -> list[str]:
    text = str(value or "").lower()
    paths: list[str] = []
    if any(term in text for term in ("material", "rain vs", "mist", "family")):
        paths.extend(("primary_family", "effect_type"))
    if "subtype" in text or "effect type" in text:
        paths.append("effect_type")
    if any(term in text for term in ("loop", "nontransient")):
        paths.extend(("motion.loopability", "motion.temporal_arc"))
    if "event_count" in text or "event count" in text or "multiple window" in text:
        paths.append("motion.event_count")
    if "temporal" in text or "full action" in text:
        paths.extend(("motion.temporal_arc", "motion.event_count", "motion.loopability"))
    if "edge" in text or "border" in text or "crop" in text:
        paths.append("composition.edge_contact")
    if "origin" in text:
        paths.append("composition.origin")
    if "scale" in text or "distance" in text:
        paths.append("composition.shot_scale")
    if "viewpoint" in text or "angle" in text:
        paths.append("composition.viewpoint")
    if "depth" in text:
        paths.append("motion.depth_motion")
    if "direction" in text:
        paths.append("motion.direction")
    if "low_confidence" in text or "low confidence" in text:
        paths.extend(("composition.viewpoint", "composition.shot_scale"))
    return [path for path in _dedupe(paths) if path in REVIEWABLE_FIELDS]


def _analysis_value(analysis: dict[str, Any], path: str) -> Any:
    current: Any = analysis
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return deepcopy(current)


def _set_analysis_value(analysis: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = analysis
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _sync_compatibility_fields(element: Element) -> None:
    analysis = element.semantic_analysis
    element.primary_family = str(analysis.get("primary_family") or element.primary_family)
    effect_tokens = set(_token(analysis.get("effect_type")).split("_"))
    inferred_secondary = [
        family
        for family in EFFECT_TYPES_BY_FAMILY
        if family in effect_tokens and family != element.primary_family
    ]
    secondary = _dedupe(
        [
            _token(value)
            for value in [*inferred_secondary, *(analysis.get("secondary_families") or [])]
            if _token(value) not in {"", "unknown", "none", element.primary_family}
        ]
    )
    analysis["secondary_families"] = secondary
    element.secondary_families = secondary
    element.category = element.primary_family
    element.category_confidence = element.analysis_confidence
    element.caption = str(analysis.get("summary") or element.caption)
    element.qwen_tags = {
        **element.qwen_tags,
        "effect_type": analysis.get("effect_type"),
        "secondary_families": element.secondary_families,
        "search_text": analysis.get("search_text"),
    }


def _rebuild_reviewed_text(analysis: dict[str, Any]) -> None:
    effect = str(analysis.get("effect_type") or "element").replace("_", " ")
    family = str(analysis.get("primary_family") or "VFX").replace("_", " ")
    composition = analysis.get("composition") if isinstance(analysis.get("composition"), dict) else {}
    motion = analysis.get("motion") if isinstance(analysis.get("motion"), dict) else {}
    viewpoint = str(composition.get("viewpoint") or "unknown").replace("_", " ")
    shot_scale = str(composition.get("shot_scale") or "unknown").replace("_", " ")
    direction = str(motion.get("direction") or composition.get("direction") or "unknown").replace("_", " ")
    characters = [str(value).replace("_", " ") for value in motion.get("character") or []]
    action = ", ".join(characters[:3]) or "visible motion"
    framing = " ".join(value for value in (shot_scale, viewpoint) if value != "unknown")
    framing_clause = f" in a {framing} view" if framing else ""
    concise_direction = direction if len(direction.split()) <= 5 else "unknown"
    direction_clause = f", moving {concise_direction}" if concise_direction != "unknown" else ""
    summary = f"A {effect} {family} element{framing_clause}{direction_clause}, with {action} motion."
    analysis["summary"] = summary
    usability = analysis.setdefault("usability", {})
    usability["best_uses"] = _dedupe([effect, f"{family} compositing element"])
    search_bits = [effect, family, framing, direction, *characters]
    analysis["search_text"] = " ".join(value for value in search_bits if value and value != "unknown")


def _refresh_review_embedding(element: Element) -> None:
    if len(element.image_embed) not in {0, VECTOR_DIMS}:
        return
    if len(element.caption_embed) not in {0, VECTOR_DIMS}:
        return
    embed_element(element)
    element.embedding_model_version = (
        "deterministic-hash-embedding-0.1.5 + human-reviewed semantic refresh"
    )


def _token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != []


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _dedupe(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = _value_key(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result

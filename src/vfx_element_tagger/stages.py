from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

from .models import Element
from .taxonomy import CATEGORY_KEYWORDS, load_taxonomy
from .util import stable_id


VECTOR_DIMS = 64
FALLBACK_SOURCE = "filename"
FALLBACK_CONF = 0.25
UNKNOWN_CONF = 0.1
QWEN_TAG_SOURCE = "qwen_caption_structured_tags"
QWEN_TAG_FIELDS = (
    "subject",
    "backing",
    "shot_scale",
    "frame_role",
    "edge_contact",
    "prop",
    "action",
    "effect_reference",
    "camera_motion",
    "keyability",
    "motion_summary",
    "confidence",
)

_QWEN_TAG_FIELD_SET = set(QWEN_TAG_FIELDS)
_QWEN_TAG_ALIASES = {
    "main_subject": "subject",
    "visible_subject": "subject",
    "primary_subject": "subject",
    "background": "backing",
    "backdrop": "backing",
    "screen_backing": "backing",
    "plate_backing": "backing",
    "shot": "shot_scale",
    "scale": "shot_scale",
    "framing": "shot_scale",
    "framing_distance": "shot_scale",
    "role": "frame_role",
    "comp_role": "frame_role",
    "compositing_role": "frame_role",
    "placement": "frame_role",
    "crop": "edge_contact",
    "cropping": "edge_contact",
    "edges": "edge_contact",
    "edge": "edge_contact",
    "held_prop": "prop",
    "object": "prop",
    "weapon": "prop",
    "launcher": "prop",
    "performance_action": "action",
    "motion_action": "action",
    "effect_refs": "effect_reference",
    "effect_references": "effect_reference",
    "reference_effects": "effect_reference",
    "reference_elements": "effect_reference",
    "useful_for": "effect_reference",
    "vfx_reference": "effect_reference",
    "camera": "camera_motion",
    "camera_move": "camera_motion",
    "camera_movement": "camera_motion",
    "keyable": "keyability",
    "key": "keyability",
    "motion": "motion_summary",
    "temporal_motion": "motion_summary",
    "summary_motion": "motion_summary",
    "tag_confidence": "confidence",
}
_EMPTY_TAG_VALUES = {"", "unknown", "n/a", "na", "none", "null", "not sure", "unclear"}


# Filename keyword tables for the per-category content-facet fallback. Each
# inner dict maps a taxonomy facet value to substrings whose presence in the
# element text vote for that value. The first match wins per facet.
_CONTENT_KEYWORDS: dict[str, dict[str, dict[str, set[str]]]] = {
    "smoke": {
        "density": {
            "opaque": {"opaque", "solid"},
            "heavy": {"heavy", "thick", "dense"},
            "wispy": {"wispy", "thin", "light"},
        },
        "colour": {
            "black": {"black"},
            "dark": {"dark"},
            "white": {"white"},
            "grey": {"grey", "gray"},
            "coloured": {"red", "blue", "green", "purple", "orange"},
        },
        "temperature": {
            "hot": {"hot"},
            "warm": {"warm"},
            "cool": {"cool"},
        },
    },
    "fire": {
        "type": {
            "ember_shower": {"ember"},
            "torch": {"torch"},
            "candle": {"candle"},
            "gas_jet": {"gas", "jet"},
            "ignition": {"ignit"},
            "sustained_burn": {"burn"},
            "flicker": {"flicker"},
        },
        "colour": {
            "white_hot": {"white"},
            "blue": {"blue"},
            "yellow": {"yellow"},
            "coloured": {"green", "purple", "red"},
        },
        "scale": {
            "landscape": {"landscape"},
            "structure": {"structure", "building"},
            "room": {"room", "interior"},
            "body": {"body"},
            "tabletop": {"tabletop"},
        },
    },
    "debris": {
        "material": {
            "rock": {"rock", "stone"},
            "concrete": {"concrete"},
            "glass": {"glass"},
            "wood": {"wood"},
            "metal": {"metal"},
            "paper": {"paper"},
            "ash": {"ash"},
        },
        "scale": {
            "large": {"large"},
            "chunky": {"chunky", "chunk"},
            "fine": {"fine", "small"},
        },
        "quantity": {
            "swarm": {"swarm"},
            "dense": {"dense"},
            "sparse": {"sparse"},
        },
    },
    "water": {
        "type": {
            "splash": {"splash"},
            "spray": {"spray"},
            "wave": {"wave"},
            "droplet": {"droplet"},
            "sheet": {"sheet"},
            "cascade": {"cascade", "waterfall"},
            "mist": {"mist"},
        },
        "scale": {
            "hero": {"hero"},
            "large": {"large"},
            "small": {"small"},
        },
    },
    "dust": {
        "density": {
            "heavy": {"heavy", "thick", "dense"},
            "light": {"light", "thin"},
        },
        "particle_scale": {
            "coarse": {"coarse"},
            "fine": {"fine", "powder"},
        },
    },
    "atmosphere": {
        "type": {
            "heat_distortion": {"heat", "distort"},
            "low_cloud": {"cloud"},
            "fog": {"fog"},
            "mist": {"mist"},
            "haze": {"haze"},
        },
        "density": {
            "thick": {"thick", "heavy"},
            "thin": {"thin", "wispy"},
        },
    },
    "lens_effect": {
        "type": {
            "chromatic_aberration": {"chromatic"},
            "bokeh": {"bokeh"},
            "scratch": {"scratch"},
            "dirt": {"dirt"},
            "glint": {"glint"},
            "light_leak": {"leak"},
            "flare": {"flare"},
        },
        "intensity": {
            "strong": {"strong"},
            "subtle": {"subtle"},
        },
    },
    "practical_light": {
        "type": {
            "muzzle_flash": {"muzzle"},
            "lightning": {"lightning"},
            "neon": {"neon"},
            "fluorescent": {"fluorescent"},
            "headlight": {"headlight"},
            "torch": {"torch"},
            "fire_light": {"firelight"},
        },
        "colour": {
            "blue": {"blue"},
            "white": {"white"},
            "yellow": {"yellow"},
            "coloured": {"green", "red", "purple"},
        },
    },
    "plate": {
        "subject": {
            "sky": {"sky"},
            "skyline": {"skyline"},
            "water_surface": {"water"},
            "landscape": {"landscape"},
            "interior": {"interior"},
            "street": {"street"},
            "crowd": {"crowd"},
            "vehicle": {"vehicle", "car"},
        },
        "time_of_day": {
            "dawn": {"dawn"},
            "magic_hour": {"magichour", "goldenhour", "magic", "golden"},
            "dusk": {"dusk"},
            "night": {"night"},
            "day": {"day", "noon"},
        },
        "weather": {
            "rain": {"rain"},
            "snow": {"snow"},
            "fog": {"fog"},
            "overcast": {"overcast"},
            "clear": {"clear"},
        },
    },
    "impact": {
        "type": {
            "bullet_hit": {"bullet"},
            "blood_spray": {"blood"},
            "ground_impact": {"ground"},
            "water_impact": {"water"},
            "blunt_impact": {"blunt"},
        },
    },
    "sparks": {
        "source": {
            "welding": {"weld"},
            "grinding": {"grind"},
            "electrical": {"electric", "electrical"},
            "pyrotechnic": {"pyro"},
            "ricochet": {"ricochet"},
        },
    },
}


# Per-category motion-facet fallbacks driven by filename + duration. Universal
# direction/speed are emitted by _universal_motion; this table only adds the
# category-specific axes that the spec's per-category schemas demand.
_MOTION_KEYWORDS: dict[str, dict[str, dict[str, set[str]]]] = {
    "smoke": {
        "character": {
            "billowing": {"billow", "plume"},
            "surging": {"surge"},
            "dissipating": {"dissipat"},
            "rising": {"rising", "rise"},
            "rolling": {"rolling", "roll"},
            "static": {"static"},
            "drifting": {"drift"},
        },
        "density_evolution": {
            "building": {"build"},
            "dissipating": {"dissipat"},
        },
        "loopable": {
            "looped": {"loop"},
            "one_shot": {"oneshot", "shot"},
        },
    },
    "fire": {
        "character": {
            "explosive_onset": {"explos"},
            "flickering": {"flicker"},
            "surging": {"surge"},
            "dying": {"dying", "die"},
            "sustained": {"sustained"},
        },
        "wind_influence": {
            "strong": {"strongwind", "gale"},
            "gentle": {"gentlewind", "breeze"},
        },
        "intensity_arc": {
            "building": {"build"},
            "dissipating": {"dying", "dissipat"},
        },
        "containment": {
            "column": {"column"},
            "climbing": {"climb"},
            "contained": {"contained"},
        },
    },
    "explosion": {
        "character": {"explosive_onset": {"explos", "detonat", "blast"}},
        "intensity_arc": {
            "building": {"build"},
            "dissipating": {"dissipat"},
        },
    },
    "debris": {
        "trajectory": {
            "exploding_outward": {"explod", "blast"},
            "falling_gravity": {"fall", "drop", "gravity"},
            "tumbling": {"tumbl"},
            "settling": {"settl"},
            "linear": {"linear", "straight"},
        },
        "gravity_feel": {
            "heavy_impact": {"impact", "heavy"},
            "weighted": {"weight"},
            "floating": {"float"},
            "no_gravity": {"zerog", "nograv"},
        },
        "quantity_arc": {
            "building": {"build"},
            "dissipating": {"dissipat"},
        },
    },
    "water": {
        "direction": {
            "vertical_up": {"up", "upward"},
            "vertical_down": {"down", "fall"},
            "arcing": {"arc"},
            "lateral": {"left", "right", "lateral"},
            "radial": {"radial", "splash"},
        },
        "gravity_feel": {
            "atomised": {"spray", "mist"},
            "sheet": {"sheet"},
            "gravity_driven": {"fall", "down"},
        },
        "loopable": {
            "looped": {"loop"},
            "one_shot": {"oneshot"},
        },
    },
    "dust": {
        "character": {
            "kicked_up": {"kick"},
            "swirling": {"swirl"},
            "settling": {"settl"},
            "advecting": {"advect"},
            "drifting": {"drift"},
        },
    },
    "atmosphere": {
        "character": {
            "advecting": {"advect"},
            "settling": {"settl"},
            "drifting": {"drift"},
            "static": {"static"},
        },
    },
    "lens_effect": {
        "character": {
            "flickering": {"flicker"},
            "sweeping": {"sweep"},
            "pulsing": {"puls"},
            "static": {"static"},
        },
    },
    "practical_light": {
        "behaviour": {
            "strobing": {"strob"},
            "flickering": {"flicker"},
            "pulsing": {"puls"},
            "sweeping": {"sweep"},
            "static": {"static"},
        },
        "change_profile": {
            "ramping_in": {"rampin", "fadein"},
            "ramping_out": {"rampout", "fadeout"},
            "oscillating": {"oscill"},
            "steady": {"steady"},
        },
    },
    "plate": {
        "camera_motion": {
            "aerial": {"aerial", "drone"},
            "dolly": {"dolly"},
            "pan": {"pan"},
            "tilt": {"tilt"},
            "handheld": {"handheld"},
            "slow_drift": {"drift"},
            "locked_off": {"locked", "lockoff"},
        },
        "parallax": {
            "high": {"highparallax"},
            "none": {"flat"},
        },
    },
    "impact": {
        "onset": {
            "instant": {"instant"},
            "fast_buildup": {"fastbuild"},
            "slow_buildup": {"slowbuild"},
        },
        "dissipation": {
            "smoke_trail": {"trail"},
            "fast": {"fast"},
            "slow": {"slow"},
        },
        "debris_pattern": {
            "vertical_column": {"column"},
            "forward_cone": {"cone"},
            "radial": {"radial"},
        },
    },
    "sparks": {
        "trajectory": {
            "shower": {"shower"},
            "arcing": {"arc"},
            "scattering": {"scatter"},
            "falling": {"fall"},
        },
        "quantity_arc": {
            "burst": {"burst"},
            "dissipating": {"dissipat"},
        },
    },
}


def classify_element(element: Element) -> None:
    text = _element_text(element)
    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score:
            scores[category] = score
    if not scores:
        element.category = "unknown"
        element.category_confidence = 0.15
        element.provenance["category"] = {"source": FALLBACK_SOURCE, "confidence": 0.15}
        element.content_facets = _content_facets(element, text)
        return

    best_category, best_score = max(scores.items(), key=lambda item: item[1])
    element.category = best_category
    element.category_confidence = min(0.85, 0.35 + best_score * 0.15)
    element.provenance["category"] = {
        "source": FALLBACK_SOURCE,
        "confidence": element.category_confidence,
    }
    element.content_facets = _content_facets(element, text)


def derive_content_facets(element: Element) -> None:
    """Populate content facets for the element's current category.

    Model-backed classification sets `element.category` before this runs. Keeping
    this separate from `classify_element()` prevents a facet refresh from
    downgrading a model-derived category back to filename heuristics.
    """
    element.content_facets = _content_facets(element, _element_text(element))


def enrich_search_facets(element: Element) -> None:
    """Add broad VFX search facets extracted from the generated caption.

    These facets deliberately include synonyms users are likely to type, so
    retrieval is not dependent on the caption using exactly the same wording.
    """
    text = _search_text(element)
    additions = _caption_search_facets(text)
    additions.update(_qwen_tag_content_facets(element.qwen_tags))
    if not additions:
        return
    confidence = _qwen_tag_confidence(element.qwen_tags) or 0.75
    for name, value in additions.items():
        if name == "camera_motion":
            existing_motion = facet_value(element.motion_facets.get(name))
            if existing_motion in {None, "", "unknown"}:
                element.motion_facets[name] = _facet(value, QWEN_TAG_SOURCE, confidence)
            continue
        existing = facet_value(element.content_facets.get(name))
        if name == "search_terms" or existing in {None, "", "unknown"}:
            source = QWEN_TAG_SOURCE if name in element.qwen_tags else "caption_keyword_enrichment"
            element.content_facets[name] = _facet(value, source, confidence)


def store_qwen_caption_tags(element: Element, qwen_output: str) -> None:
    """Store clean caption text plus compact structured tags from Qwen output."""
    caption, tags = parse_qwen_tags(qwen_output, element)
    element.caption = caption
    element.qwen_tags = tags
    enrich_search_facets(element)


def parse_qwen_tags(qwen_output: str, element: Element | None = None) -> tuple[str, dict[str, Any]]:
    """Parse Qwen caption output into `(clean_caption, compact_tags)`.

    Qwen is prompted for a `TAGS: {...}` line, but this accepts fenced JSON,
    whole-output JSON, and simple key/value lines before falling back to
    deterministic caption and motion heuristics.
    """
    raw = (qwen_output or "").strip()
    json_tags: dict[str, Any] = {}
    line_tags: dict[str, Any] = {}
    tag_spans: list[tuple[int, int]] = []
    caption_hint: str | None = None

    for data, span in _qwen_json_candidates(raw):
        candidate, candidate_caption = _qwen_tag_mapping(data)
        candidate_tags = _normalise_qwen_tags(candidate)
        if not candidate_tags:
            continue
        json_tags.update(candidate_tags)
        tag_spans.append(span)
        if candidate_caption and not caption_hint:
            caption_hint = candidate_caption
        break

    line_tags = _qwen_line_tags(raw)
    parsed_tags = {**line_tags, **json_tags}
    deterministic_tags = _deterministic_qwen_tags(raw, element)
    had_model_tags = any(key != "confidence" for key in parsed_tags)

    tags = dict(parsed_tags)
    for key, value in deterministic_tags.items():
        tags.setdefault(key, value)
    if "confidence" not in tags:
        if had_model_tags:
            tags["confidence"] = 0.78
        elif any(key != "confidence" for key in tags):
            tags["confidence"] = 0.55
        else:
            tags["confidence"] = 0.2

    caption = _clean_qwen_caption(raw, tag_spans)
    if caption_hint:
        caption = _clean_qwen_caption(caption_hint, [])
    if not caption:
        caption = raw

    return caption, _ordered_qwen_tags(tags)


def _qwen_json_candidates(text: str) -> list[tuple[dict[str, Any], tuple[int, int]]]:
    candidates: list[tuple[dict[str, Any], tuple[int, int]]] = []
    seen_spans: set[tuple[int, int]] = set()
    fenced_pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    for match in fenced_pattern.finditer(text):
        data = _loads_mapping(match.group(1).strip())
        if data is not None:
            span = match.span()
            candidates.append((data, span))
            seen_spans.add(span)

    for start_match in re.finditer(r"\{", text):
        start = start_match.start()
        end = _balanced_json_end(text, start)
        if end is None:
            continue
        span = (start, end)
        if span in seen_spans:
            continue
        data = _loads_mapping(text[start:end])
        if data is not None:
            candidates.append((data, span))
            seen_spans.add(span)
    return candidates


def _balanced_json_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escape = False
    quote = ""
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {"'", '"'}:
            in_string = True
            quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _loads_mapping(text: str) -> dict[str, Any] | None:
    variants = [text, re.sub(r",\s*([}\]])", r"\1", text)]
    for variant in dict.fromkeys(variants):
        for loader in (json.loads, ast.literal_eval):
            try:
                data = loader(variant)
            except (ValueError, SyntaxError, TypeError):
                continue
            if isinstance(data, dict):
                return data
    return None


def _qwen_tag_mapping(data: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    caption_hint = None
    for caption_key in ("caption", "description", "summary"):
        value = data.get(caption_key)
        if isinstance(value, str) and value.strip():
            caption_hint = value
            break

    for tag_key in ("tags", "qwen_tags", "structured_tags", "structured_tag_payload"):
        value = data.get(tag_key)
        if isinstance(value, dict):
            merged = dict(value)
            for key, top_level_value in data.items():
                if _normalise_qwen_tag_key(str(key)) and key not in merged:
                    merged[key] = top_level_value
            return merged, caption_hint

    if any(_normalise_qwen_tag_key(str(key)) for key in data):
        return data, caption_hint
    return {}, caption_hint


def _qwen_line_tags(text: str) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    pattern = re.compile(
        r"(?im)^[ \t]*(?:[-*]\s*)?[\"']?([a-z][a-z0-9 _-]{1,40})[\"']?\s*[:=]\s*(.+?)\s*$"
    )
    for match in pattern.finditer(text):
        key = _normalise_qwen_tag_key(match.group(1))
        if key is None:
            continue
        value = _normalise_qwen_tag_value(key, match.group(2))
        if value is not None:
            tags[key] = value
    return tags


def _normalise_qwen_tags(data: dict[str, Any]) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    for raw_key, raw_value in data.items():
        key = _normalise_qwen_tag_key(str(raw_key))
        if key is None:
            continue
        value = _normalise_qwen_tag_value(key, raw_value)
        if value is not None:
            tags[key] = value
    return tags


def _normalise_qwen_tag_key(key: str) -> str | None:
    normalised = _snake_token(key)
    normalised = _QWEN_TAG_ALIASES.get(normalised, normalised)
    if normalised in _QWEN_TAG_FIELD_SET:
        return normalised
    return None


def _normalise_qwen_tag_value(field: str, value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if field == "confidence":
        return _parse_confidence(value)
    text = _compact_tag_text(value)
    if _is_empty_tag_value(text):
        return None
    if field == "motion_summary":
        return text[:160]
    if field == "action":
        return _normalise_action_value(text)
    return _normalise_controlled_qwen_value(field, text)


def _parse_confidence(value: Any) -> float | None:
    if isinstance(value, (float, int)):
        return _clamp01(float(value))
    text = _compact_tag_text(value)
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if percent:
        return _clamp01(float(percent.group(1)) / 100.0)
    number = re.search(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", text)
    if number:
        return _clamp01(float(number.group(0)))
    return None


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _compact_tag_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(item) for item in value)
    text = str(value).strip().strip("\"'`")
    text = re.sub(r"^[{\[]|[}\]]$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;,")


def _is_empty_tag_value(value: str) -> bool:
    return value.lower().strip() in _EMPTY_TAG_VALUES


def _normalise_controlled_qwen_value(field: str, value: str) -> str:
    text = value.lower().replace("_", " ").replace("-", " ")
    padded = f" {text} "
    if field == "subject":
        if _has_any(
            padded,
            "person",
            "actor",
            "performer",
            "operator",
            "talent",
            "human",
            "man",
            "woman",
            "character",
            "soldier",
        ):
            return "person"
        if _has_any(padded, "vehicle", "car", "truck", "motorcycle", "bike"):
            return "vehicle"
        if _has_any(padded, "crowd", "people"):
            return "crowd"
        if _has_any(padded, "skyline"):
            return "skyline"
        if _has_any(padded, "sky"):
            return "sky"
        if _has_any(padded, "water", "ocean", "sea"):
            return "water_surface"
        if _has_any(padded, "landscape", "mountain", "terrain"):
            return "landscape"
    if field == "backing":
        if _has_any(padded, "green screen", "greenscreen", "green backdrop", "chroma key"):
            return "green_screen"
        if _has_any(padded, "blue screen", "bluescreen", "blue backdrop"):
            return "blue_screen"
        if _has_any(padded, "black", "over black"):
            return "black"
        if _has_any(padded, "white"):
            return "white"
        if _has_any(padded, "transparent", "alpha"):
            return "transparent"
        if _has_any(padded, "practical", "environment", "natural", "clean plate", "none"):
            return "none"
    if field == "shot_scale":
        if _has_any(padded, "medium wide", "mediumwide"):
            return "medium_wide"
        if _has_any(padded, "wide", "full body", "full length"):
            return "wide"
        if _has_any(padded, "medium", "waist"):
            return "medium"
        if _has_any(padded, "extreme close", "macro"):
            return "extreme_close_up"
        if _has_any(padded, "close", "tight"):
            return "close_up"
    if field == "frame_role":
        if _has_any(padded, "foreground", "fg"):
            return "foreground"
        if _has_any(padded, "midground", "mg"):
            return "midground"
        if _has_any(padded, "background", "bg"):
            return "background"
        if _has_any(padded, "insert", "detail"):
            return "insert"
    if field == "edge_contact":
        if _has_any(padded, "clean margin", "no edge", "fully contained", "not cropped", "clear edges"):
            return "clean_margin"
        if _has_any(padded, "cropped", "edge contact", "touches edge", "cut off", "clipped"):
            return "cropped"
    if field == "prop":
        if _has_any(padded, "rpg", "rocket launcher", "rocket propelled grenade", "launcher"):
            return "rocket_launcher"
        if _has_any(padded, "weapon", "gun", "rifle", "firearm"):
            return "weapon"
        if _has_any(padded, "handheld device", "held device"):
            return "handheld_device"
        if _has_any(padded, "prop"):
            return "prop"
    if field == "effect_reference":
        return _normalise_effect_reference_value(text)
    if field == "camera_motion":
        if _has_any(padded, "locked off", "lock off", "lockoff", "static camera", "tripod"):
            return "locked_off"
        if _has_any(padded, "handheld", "hand held"):
            return "handheld"
        if _has_any(padded, "slow drift", "drift"):
            return "slow_drift"
        if _has_any(padded, " pan ", "panning"):
            return "pan"
        if _has_any(padded, "tilt", "tilting"):
            return "tilt"
        if _has_any(padded, "dolly", "tracking", "push in", "pull back"):
            return "dolly"
        if _has_any(padded, "aerial", "drone"):
            return "aerial"
    if field == "keyability":
        if _has_any(padded, "not keyable", "unkeyable", "cannot key"):
            return "not_keyable"
        if _has_any(padded, "difficult", "rough", "poor", "spill", "uneven"):
            return "difficult_key"
        if _has_any(padded, "keyable", "clean key", "chroma key", "clean extraction"):
            return "keyable"
    return _snake_token(value)


def _normalise_action_value(value: str) -> str:
    text = value.lower().replace("_", " ").replace("-", " ")
    actions: list[str] = []
    if _has_any(
        text,
        "rpg",
        "rocket launcher",
        "launcher",
        "weapon",
        "rifle",
        "gun",
        "shoot",
        "shooting",
        "fires",
        "firing",
        "launch",
        "launching",
    ):
        actions.append("firing_weapon")
    if _has_any(text, "turning", "turns", "rotate", "rotating", "rotation", "spinning"):
        actions.append("turning")
    if _has_any(text, "gesture", "gestures", "gesturing", "pointing", "arms"):
        actions.append("gesturing")
    if _has_any(text, "walking", "walks", "walk"):
        actions.append("walking")
    if _has_any(text, "running", "runs", "run"):
        actions.append("running")
    if _has_any(text, "jump", "jumping"):
        actions.append("jumping")
    if _has_any(text, "fall", "falling"):
        actions.append("falling")
    if _has_any(text, "static", "standing", "idle", "still"):
        actions.append("static")
    if actions:
        return " ".join(dict.fromkeys(actions))
    return _snake_token(value).replace("_", " ")


def _normalise_effect_reference_value(value: str) -> str:
    text = value.lower().replace("_", " ").replace("-", " ")
    refs: list[str] = []
    if _has_any(text, "smoke", "plume", "vapour", "vapor"):
        refs.append("smoke")
    if _has_any(text, "fire", "flame", "fiery"):
        refs.append("fire")
    if _has_any(text, "muzzle flash", "muzzle", "flash"):
        refs.append("muzzle_flash")
    if _has_any(text, "backblast", "back blast", "exhaust blast", "rocket exhaust"):
        refs.append("backblast")
    if _has_any(text, "explosion", "blast", "pyro", "pyrotechnic"):
        refs.append("explosion")
    if _has_any(text, "dust", "debris"):
        refs.append("dust_debris")
    if refs:
        return " ".join(dict.fromkeys(refs))
    return _snake_token(value).replace("_", " ")


def _snake_token(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _deterministic_qwen_tags(text: str, element: Element | None) -> dict[str, Any]:
    search_text = text
    if element is not None:
        search_text = " ".join(
            [
                search_text,
                element.category,
                _facets_text(element, include_qwen_tags=False),
                _element_text(element),
            ]
        )
    search_text = search_text.lower().replace("-", " ")
    tags = {
        key: value
        for key, value in _caption_search_facets(search_text).items()
        if key in _QWEN_TAG_FIELD_SET and key != "confidence"
    }

    camera_motion = _infer_camera_motion(search_text, element)
    if camera_motion:
        tags.setdefault("camera_motion", camera_motion)

    keyability = _infer_keyability(search_text)
    if keyability:
        tags.setdefault("keyability", keyability)

    motion_summary = _infer_motion_summary(tags, element)
    if motion_summary:
        tags.setdefault("motion_summary", motion_summary)
    return tags


def _infer_camera_motion(text: str, element: Element | None) -> str | None:
    value = _normalise_controlled_qwen_value("camera_motion", text)
    if value and value != _snake_token(text):
        return value
    if element is not None:
        motion_value = facet_value(element.motion_facets.get("camera_motion"))
        if motion_value not in {None, "", "unknown"}:
            return str(motion_value)
    return None


def _infer_keyability(text: str) -> str | None:
    if _has_any(text, "not keyable", "unkeyable", "cannot key"):
        return "not_keyable"
    if _has_any(text, "difficult key", "poor key", "rough key", "green spill", "uneven green"):
        return "difficult_key"
    if _has_any(text, "keyable", "clean key", "clean extraction", "chroma key", "green screen", "blue screen"):
        return "keyable"
    return None


def _infer_motion_summary(tags: dict[str, Any], element: Element | None) -> str | None:
    parts: list[str] = []
    action = tags.get("action")
    if action:
        parts.append(str(action))
    camera_motion = tags.get("camera_motion")
    if camera_motion:
        parts.append(f"{str(camera_motion).replace('_', ' ')} camera")
    if element is not None:
        for key in ("speed", "direction", "character", "trajectory"):
            value = facet_value(element.motion_facets.get(key))
            if value not in {None, "", "unknown", "no_dominant"}:
                parts.append(f"{key} {value}")
    if not parts:
        return None
    return ", ".join(dict.fromkeys(parts))[:160]


def _qwen_tag_content_facets(tags: dict[str, Any]) -> dict[str, str]:
    additions: dict[str, str] = {}
    for key in (
        "subject",
        "backing",
        "shot_scale",
        "frame_role",
        "edge_contact",
        "prop",
        "action",
        "effect_reference",
        "camera_motion",
        "keyability",
    ):
        value = tags.get(key)
        if value not in {None, "", "unknown"}:
            additions[key] = str(value)
    return additions


def _qwen_tag_confidence(tags: dict[str, Any]) -> float | None:
    value = tags.get("confidence")
    if isinstance(value, (float, int)):
        return _clamp01(float(value))
    return None


def _ordered_qwen_tags(tags: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for field in QWEN_TAG_FIELDS:
        value = tags.get(field)
        if value is None:
            continue
        if field != "confidence" and _is_empty_tag_value(str(value)):
            continue
        ordered[field] = value
    if set(ordered) == {"confidence"}:
        return {}
    return ordered


def _clean_qwen_caption(text: str, tag_spans: list[tuple[int, int]]) -> str:
    cleaned = text
    for start, end in sorted(tag_spans, reverse=True):
        cleaned = f"{cleaned[:start]} {cleaned[end:]}"
    lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if re.match(r"(?i)^tags?\s*:", stripped):
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            continue
        tag_line = re.match(
            r"^[ \t]*(?:[-*]\s*)?[\"']?([a-z][a-z0-9 _-]{1,40})[\"']?\s*[:=]\s*(.+?)\s*$",
            stripped,
            re.IGNORECASE,
        )
        if tag_line and _normalise_qwen_tag_key(tag_line.group(1)):
            continue
        lines.append(stripped)
    cleaned = " ".join(lines)
    cleaned = re.sub(r"(?i)^\s*(caption|description)\s*:\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip("\"'")
    return cleaned


def derive_motion(element: Element) -> None:
    text = _element_text(element)
    duration = element.duration_seconds or 0.0
    facets: dict[str, Any] = _universal_motion(text, duration)
    facets.update(_category_motion(element.category, text))
    element.motion_facets = facets
    element.measured_features = {
        "dominant_flow_angle": None,
        "median_flow_magnitude": None,
        "flow_divergence_mean": None,
        "brightness_variance": None,
        "implementation": "deterministic fallback; optical-flow analyzer not yet enabled",
    }
    element.motion_warnings = ["motion facets are low-confidence filename/duration fallbacks"]


def caption_element(element: Element) -> None:
    element.qwen_tags = {}
    primary = element.primary()
    category = element.category if element.category != "unknown" else "element"
    resolution = f"{element.width}x{element.height}" if element.width and element.height else "unknown resolution"
    duration = f"{element.duration_seconds:.2f}s" if element.duration_seconds else "unknown duration"
    alpha = "soft-alpha plate-ready" if element.alpha_is_soft else "no verified alpha"
    root = primary.filename_root or Path(primary.path).stem
    motion_bits = []
    for key, payload in element.motion_facets.items():
        if isinstance(payload, dict) and payload.get("value"):
            motion_bits.append(f"{key} {payload['value']}")
    motion = ", ".join(motion_bits) if motion_bits else "motion not yet analysed"
    element.caption = (
        f"{category.replace('_', ' ')} element inferred from '{root}', {resolution}, "
        f"{duration}, {alpha}. Fallback motion read: {motion}."
    )


def embed_element(element: Element) -> None:
    visual_text = " ".join(
        [
            element.category,
            Path(element.primary().path).stem,
            str(element.width or ""),
            str(element.height or ""),
        ]
    )
    enrich_search_facets(element)
    caption_text = caption_embedding_text(element)
    element.image_embed = embed_text(visual_text)
    element.caption_embed = embed_text(caption_text)


def embed_text(text: str) -> list[float]:
    vector = [0.0] * VECTOR_DIMS
    tokens = tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        bucket = int(stable_id("tok", token, length=8).split("_", 1)[1], 16) % VECTOR_DIMS
        vector[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def facet_value(payload: Any) -> Any:
    """Extract the scalar value from a provenance-wrapped facet, or pass through bare scalars."""
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload


def caption_embedding_text(element: Element) -> str:
    return " ".join([element.caption, _facets_text(element)])


def _element_text(element: Element) -> str:
    parts: list[str] = []
    for representation in element.source_representations:
        parts.extend(
            [
                representation.filename_root or "",
                Path(representation.path).stem,
                Path(representation.original_path).name,
            ]
        )
    return " ".join(parts).lower()


def _content_facets(element: Element, text: str) -> dict[str, Any]:
    facets: dict[str, Any] = {}
    facets["resolution_tier"] = _facet(_resolution_tier(element.width, element.height), "ffprobe", 0.95)
    facets["frame_rate_tier"] = _facet(_frame_rate_tier(element.fps), "ffprobe", 0.95)
    facets["alpha_status"] = _facet(
        "has_true_alpha" if element.alpha_is_soft else "no_alpha",
        "ffmpeg_alpha_probe" if element.alpha_is_soft else "filename",
        0.85 if element.alpha_is_soft else 0.3,
    )
    facets["bit_depth_tier"] = _facet(str(element.bit_depth or "unknown"), "ffprobe", 0.95)
    facets["color_space"] = _facet(_normalise_color_space(element.color_space), "ffprobe", 0.7)

    keyword_block = _CONTENT_KEYWORDS.get(element.category, {})
    schema = _taxonomy().content_facets_for(element.category)
    for facet_name in schema:
        match = _first_keyword_match(text, keyword_block.get(facet_name, {}))
        if match is not None:
            facets[facet_name] = _facet(match, FALLBACK_SOURCE, FALLBACK_CONF)
        else:
            facets[facet_name] = _facet("unknown", FALLBACK_SOURCE, UNKNOWN_CONF)
    return facets


def _search_text(element: Element) -> str:
    parts = [element.category, element.caption, _facets_text(element)]
    for representation in element.source_representations:
        parts.append(Path(representation.path).stem)
        parts.append(representation.filename_root or "")
    return " ".join(parts).lower().replace("-", " ")


def _caption_search_facets(text: str) -> dict[str, str]:
    facets: dict[str, str] = {}
    terms: set[str] = set()

    def add_terms(*values: str) -> None:
        for value in values:
            terms.update(tokenize(value))

    if _has_any(text, "green screen", "greenscreen", "green-screen", "green backdrop", "green studio", "chroma key"):
        facets["backing"] = "green_screen"
        facets["keyability"] = "keyable"
        add_terms("green screen", "green_screen", "greenscreen", "chroma key", "chroma_key", "keyable")
    elif _has_any(text, "blue screen", "bluescreen", "blue-screen", "blue backdrop"):
        facets["backing"] = "blue_screen"
        facets["keyability"] = "keyable"
        add_terms("blue screen", "blue_screen", "bluescreen", "chroma key", "chroma_key", "keyable")
    elif _has_any(text, "black background", "black backdrop", "over black"):
        facets["backing"] = "black"
        add_terms("black background", "black backdrop", "over black")
    elif _has_any(text, "white background", "white backdrop"):
        facets["backing"] = "white"
        add_terms("white background", "white backdrop")

    if _has_any(text, "keyable", "clean key", "clean extraction", "chroma key"):
        facets["keyability"] = "keyable"
        add_terms("keyable", "clean key", "clean extraction", "chroma key")

    if _has_any(
        text,
        "male performer",
        "male actor",
        " man ",
        "performer",
        "actor",
        "operator",
        "talent",
        "person",
        "human",
        "soldier",
    ):
        facets["subject"] = "person"
        add_terms("person", "man", "male", "performer", "actor", "operator", "human", "character", "soldier")
    elif _has_any(text, "woman", "female performer", "female actor"):
        facets["subject"] = "person"
        add_terms("person", "woman", "female", "performer", "actor", "human", "character")
    elif _has_any(text, "vehicle", "car", "truck"):
        facets["subject"] = "vehicle"
        add_terms("vehicle", "car", "truck")

    if _has_any(text, "medium wide", "medium-wide"):
        facets["shot_scale"] = "medium_wide"
        add_terms("wide", "medium wide", "medium_wide", "full body")
    elif _has_any(text, "wide shot", "wide frame", "wide framing"):
        facets["shot_scale"] = "wide"
        add_terms("wide", "wide shot", "wide frame")
    elif _has_any(text, "medium shot", "medium frame"):
        facets["shot_scale"] = "medium"
        add_terms("medium", "medium shot")
    elif _has_any(text, "close up", "close-up", "tight close"):
        facets["shot_scale"] = "close_up"
        add_terms("close up", "close_up", "close-up", "tight")

    if _has_any(text, "foreground"):
        facets["frame_role"] = "foreground"
        add_terms("foreground", "fg")
    elif _has_any(text, "midground"):
        facets["frame_role"] = "midground"
        add_terms("midground", "mg")
    elif _has_any(text, "background"):
        facets["frame_role"] = "background"
        add_terms("background", "bg")
    elif _has_any(text, "insert"):
        facets["frame_role"] = "insert"
        add_terms("insert")

    if _has_any(text, "no edge contact", "clean margin", "no cropping", "fully contained"):
        facets["edge_contact"] = "clean_margin"
        add_terms("clean margin", "no edge contact", "fully contained", "not cropped")
    elif _has_any(text, "cropped", "edge contact", "touches the edge", "touching the edge"):
        facets["edge_contact"] = "cropped"
        add_terms("cropped", "edge contact", "touches edge")

    if _has_any(text, "rpg", "rocket launcher", "rocket propelled grenade", "launcher"):
        facets["prop"] = "rocket_launcher"
        add_terms("rpg", "rocket", "rocket launcher", "rocket_launcher", "launcher", "weapon")
    elif _has_any(text, "weapon", "rifle", "gun", "firearm"):
        facets["prop"] = "weapon"
        add_terms("weapon", "rifle", "gun", "firearm")
    elif _has_any(text, "handheld device", "held device"):
        facets["prop"] = "handheld_device"
        add_terms("handheld device", "held device", "prop")

    effect_refs: list[str] = []
    if _has_any(text, "smoke", "plume", "vapour", "vapor"):
        effect_refs.append("smoke")
        add_terms("smoke", "smoke reference", "smoke_reference", "plume")
    if _has_any(text, "fire", "flame", "fiery"):
        effect_refs.append("fire")
        add_terms("fire", "fire reference", "fire_reference", "flame")
    if _has_any(text, "muzzle flash", "muzzle", "flash"):
        effect_refs.append("muzzle_flash")
        add_terms("muzzle flash", "muzzle_flash", "flash")
    if _has_any(text, "backblast", "back blast", "exhaust blast", "rocket exhaust"):
        effect_refs.append("backblast")
        add_terms("backblast", "back blast", "rocket exhaust")
    if _has_any(text, "explosion", "blast", "pyro", "pyrotechnic"):
        effect_refs.append("explosion")
        add_terms("explosion", "blast", "pyro", "pyrotechnic")
    if effect_refs:
        facets["effect_reference"] = " ".join(dict.fromkeys(effect_refs))

    actions = []
    if _has_any(
        text,
        "rpg",
        "rocket launcher",
        "launcher",
        "weapon",
        "rifle",
        "gun",
        "shoot",
        "shooting",
        "fires",
        "firing",
        "launch",
        "launching",
    ):
        actions.append("firing_weapon")
        add_terms("shoot", "shooting", "fires", "firing", "launch", "launching", "firing weapon", "firing_weapon")
    if _has_any(text, "turning", "turns", "rotate", "rotating", "rotation", "spinning"):
        actions.append("turning")
        add_terms("turning", "rotating", "rotation", "spin", "spinning")
    if _has_any(text, "gesture", "gestures", "gesturing", "pointing", "arms"):
        actions.append("gesturing")
        add_terms("gesture", "gestures", "gesturing", "pointing", "arms")
    if _has_any(text, "walking", "walks"):
        actions.append("walking")
        add_terms("walking", "walk")
    if _has_any(text, "running", "runs"):
        actions.append("running")
        add_terms("running", "run")
    if actions:
        facets["action"] = " ".join(dict.fromkeys(actions))

    if terms:
        facets["search_terms"] = " ".join(sorted(terms))
    return facets


def _has_any(text: str, *needles: str) -> bool:
    padded = f" {text} "
    return any(needle in padded for needle in needles)


def _universal_motion(text: str, duration: float) -> dict[str, Any]:
    speed = "medium"
    if duration >= 5:
        speed = "slow"
    elif duration and duration <= 1.5:
        speed = "fast"
    if "slow" in text:
        speed = "slow"
    elif "fast" in text or "quick" in text:
        speed = "fast"

    direction = "no_dominant"
    for word, label in (
        ("left", "left"),
        ("right", "right"),
        ("upward", "up"),
        ("up", "up"),
        ("downward", "down"),
        ("down", "down"),
        ("radial", "radial"),
        ("expand", "expanding"),
    ):
        if word in text:
            direction = label
            break
    return {
        "speed": _facet(speed, FALLBACK_SOURCE, FALLBACK_CONF),
        "direction": _facet(direction, FALLBACK_SOURCE, FALLBACK_CONF),
    }


def _category_motion(category: str, text: str) -> dict[str, Any]:
    facets: dict[str, Any] = {}
    keyword_block = _MOTION_KEYWORDS.get(category, {})
    schema = _taxonomy().motion_facets_for(category)
    for facet_name in schema:
        if facet_name in {"speed", "direction"}:
            continue
        match = _first_keyword_match(text, keyword_block.get(facet_name, {}))
        if match is not None:
            facets[facet_name] = _facet(match, FALLBACK_SOURCE, FALLBACK_CONF)
        else:
            facets[facet_name] = _facet("unknown", FALLBACK_SOURCE, UNKNOWN_CONF)
    return facets


def _first_keyword_match(text: str, table: dict[str, set[str]]) -> str | None:
    for value, keywords in table.items():
        if any(keyword in text for keyword in keywords):
            return value
    return None


def _facet(value: Any, source: str, confidence: float) -> dict[str, Any]:
    return {"value": value, "source": source, "confidence": confidence}


def _resolution_tier(width: int | None, height: int | None) -> str:
    if not width or not height:
        return "unknown"
    long_edge = max(width, height)
    if long_edge >= 6000:
        return "6k_plus"
    if long_edge >= 3840:
        return "4k"
    if long_edge >= 2048:
        return "2k"
    if long_edge >= 1280:
        return "hd"
    return "sd"


def _frame_rate_tier(fps: float | None) -> str:
    if not fps:
        return "variable"
    common = [24, 25, 30, 48, 50, 60, 120]
    nearest = min(common, key=lambda value: abs(value - fps))
    if abs(nearest - fps) <= 0.75:
        return "120_plus" if nearest == 120 else str(nearest)
    return "variable"


def _normalise_color_space(value: str | None) -> str:
    if not value:
        return "unknown"
    lowered = value.lower()
    if "bt709" in lowered or "rec709" in lowered:
        return "rec709"
    if "2020" in lowered:
        return "rec2020"
    if "srgb" in lowered:
        return "srgb"
    if "linear" in lowered:
        return "linear"
    return lowered


def _facets_text(element: Element, include_qwen_tags: bool = True) -> str:
    parts: list[str] = []
    if include_qwen_tags:
        parts.append(_qwen_tags_text(element))
    for value in element.content_facets.values():
        parts.append(str(facet_value(value)))
    for value in element.motion_facets.values():
        parts.append(str(facet_value(value)))
    return " ".join(parts)


def _qwen_tags_text(element: Element) -> str:
    return " ".join(str(value) for value in element.qwen_tags.values())


_TAXONOMY = None


def _taxonomy():
    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY = load_taxonomy()
    return _TAXONOMY

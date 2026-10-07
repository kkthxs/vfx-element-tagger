from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SIGLIP_LOW_CONFIDENCE = 0.55


def resolve_initial_category(
    siglip_category: str,
    siglip_confidence: float,
    poster_path: Path,
) -> tuple[str, float, dict[str, Any]]:
    """Resolve the category before captioning.

    SigLIP is useful, but it is a forced-choice classifier. Low confidence
    should not hard-steer Qwen into hallucinating category-specific details.
    """

    if siglip_confidence >= SIGLIP_LOW_CONFIDENCE:
        return siglip_category, siglip_confidence, {"source": "siglip2"}
    if looks_like_key_screen_plate(poster_path):
        return (
            "plate",
            max(siglip_confidence, 0.6),
            {
                "source": "siglip2_low_confidence_key_screen_heuristic",
                "siglip_top_category": siglip_category,
                "siglip_top_confidence": siglip_confidence,
            },
        )
    return (
        "unknown",
        siglip_confidence,
        {
            "source": "siglip2_low_confidence",
            "siglip_top_category": siglip_category,
            "siglip_top_confidence": siglip_confidence,
        },
    )


def resolve_caption_category(
    current_category: str,
    current_confidence: float,
    caption: str,
    content_facets: dict[str, Any] | None = None,
) -> tuple[str, float, dict[str, Any]] | None:
    """Return a final category correction when caption/facets contradict routing."""

    text = f" {caption.lower().replace('-', ' ')} "
    facets = content_facets or {}
    backing = _facet_value(facets.get("backing"))
    subject = _facet_value(facets.get("subject"))
    keyability = _facet_value(facets.get("keyability"))
    subject_terms = (
        " performer ",
        " actor ",
        " person ",
        " human ",
        " man ",
        " woman ",
        " soldier ",
        " operator ",
        " talent ",
        " character ",
    )
    weapon_or_prop_terms = (
        " rpg ",
        " rocket launcher ",
        " launcher ",
        " weapon ",
        " rifle ",
        " gun ",
        " firing ",
        " shooting ",
        " shoots ",
        " fires ",
        " handheld device ",
    )
    has_plate_subject = (
        subject in {"person", "vehicle", "prop", "greenscreen_subject", "bluescreen_subject"}
        or any(term in text for term in subject_terms)
        or any(term in text for term in weapon_or_prop_terms)
    )
    if (
        backing in {"green_screen", "blue_screen"}
        or " green screen " in text
        or " blue screen " in text
        or " greenscreen " in text
        or " bluescreen " in text
        or " chroma key " in text
    ) and has_plate_subject:
        confidence = max(current_confidence, 0.85)
        if current_category == "plate":
            return None
        return (
            "plate",
            confidence,
            {
                "source": "caption_category_resolver",
                "previous_category": current_category,
                "previous_confidence": current_confidence,
            },
        )
    return None


def looks_like_key_screen_plate(path: Path) -> bool:
    try:
        image = Image.open(path).convert("RGB").resize((160, 90), Image.Resampling.BILINEAR)
    except OSError:
        return False
    pixels = np.asarray(image).astype(np.float32)
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    green_screen = (green > 80) & (green > red * 1.25) & (green > blue * 1.15)
    blue_screen = (blue > 80) & (blue > red * 1.25) & (blue > green * 1.15)
    return float(np.mean(green_screen | blue_screen)) > 0.2


def _facet_value(payload: Any) -> Any:
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload

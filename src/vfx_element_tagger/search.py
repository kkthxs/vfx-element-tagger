from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .models import Element
from .stages import cosine, facet_value, tokenize


RRF_K = 60
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "asset",
    "element",
    "for",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vfx",
    "with",
}

# Canonical concepts let artist wording match the structured vocabulary without
# allowing a weak embedding rank to admit every element in the library.
PHRASE_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("away", "from", "camera"), "away_from_camera"),
    (("toward", "the", "camera"), "toward_camera"),
    (("towards", "the", "camera"), "toward_camera"),
    (("green", "screen"), "greenscreen"),
    (("blue", "screen"), "bluescreen"),
    (("chroma", "key"), "greenscreen"),
    (("top", "down"), "top_down"),
    (("top", "view"), "top_down"),
    (("overhead", "view"), "top_down"),
    (("birds", "eye"), "top_down"),
    (("bird", "eye"), "top_down"),
    (("low", "angle"), "low_angle"),
    (("right", "edge"), "right_edge"),
    (("left", "edge"), "left_edge"),
    (("top", "edge"), "top_edge"),
    (("bottom", "edge"), "bottom_edge"),
    (("full", "frame"), "full_frame"),
    (("ground", "level"), "ground"),
)

TOKEN_ALIASES = {
    "actors": "person",
    "actor": "person",
    "approaches": "walk",
    "approaching": "walk",
    "aerial": "top_down",
    "birds": "bird",
    "blasts": "explosion",
    "blast": "explosion",
    "bolts": "electricity",
    "burning": "fire",
    "burned": "fire",
    "burns": "fire",
    "burnt": "fire",
    "circles": "circling",
    "circular": "circling",
    "cropped": "edge_contact",
    "crosses": "edge_contact",
    "crossing": "edge_contact",
    "cuts": "edge_contact",
    "cutting": "edge_contact",
    "dances": "dance",
    "dancing": "dance",
    "debris": "debris",
    "descending": "fall",
    "dissipates": "dissipate",
    "dissipated": "dissipate",
    "dissipation": "dissipate",
    "dissipating": "dissipate",
    "dissolves": "dissipate",
    "dissolving": "dissipate",
    "drifting": "drift",
    "drifted": "drift",
    "drifts": "drift",
    "drops": "water",
    "downward": "down",
    "downwards": "down",
    "electric": "electricity",
    "embers": "ember",
    "expands": "expand",
    "expanded": "expand",
    "expanding": "expand",
    "explode": "explosion",
    "exploded": "explosion",
    "explodes": "explosion",
    "exploding": "explosion",
    "explosive": "explosion",
    "explosions": "explosion",
    "detonate": "explosion",
    "detonated": "explosion",
    "detonates": "explosion",
    "detonating": "explosion",
    "detonation": "explosion",
    "falling": "fall",
    "falls": "fall",
    "fell": "fall",
    "fast-moving": "fast",
    "female": "person",
    "fireball": "explosion",
    "fireballs": "explosion",
    "flames": "fire",
    "flame": "fire",
    "flashes": "flash",
    "foggy": "fog",
    "gestures": "gesture",
    "gesturing": "gesture",
    "growing": "expand",
    "growth": "expand",
    "human": "person",
    "humans": "person",
    "lightning": "electricity",
    "magic": "energy",
    "male": "person",
    "man": "person",
    "mist": "fog",
    "model": "person",
    "models": "person",
    "multiple": "multiple",
    "overhead": "top_down",
    "particles": "particle",
    "performer": "person",
    "performers": "person",
    "poses": "pose",
    "posing": "pose",
    "quick": "fast",
    "rapid": "fast",
    "rises": "rise",
    "rose": "rise",
    "rising": "rise",
    "rotates": "turn",
    "rotating": "turn",
    "rotation": "turn",
    "several": "multiple",
    "slowly": "slow",
    "smoky": "smoke",
    "snowing": "snow",
    "sparks": "spark",
    "spell": "energy",
    "spells": "energy",
    "spinning": "turn",
    "splashes": "water",
    "splash": "water",
    "turning": "turn",
    "turns": "turn",
    "turned": "turn",
    "touches": "edge_contact",
    "touching": "edge_contact",
    "vapour": "steam",
    "vapor": "steam",
    "upward": "up",
    "upwards": "up",
    "walking": "walk",
    "walks": "walk",
    "woman": "person",
}

FAMILY_CONCEPTS = {
    "bird",
    "bluescreen",
    "cloud",
    "debris",
    "dust",
    "electricity",
    "energy",
    "explosion",
    "fire",
    "fluid",
    "fog",
    "greenscreen",
    "particle",
    "plate",
    "projectile",
    "rain",
    "smoke",
    "snow",
    "spark",
    "steam",
    "water",
}

BUCKET_WEIGHTS = {
    "identity": 1.0,
    "summary": 0.92,
    "composition": 0.90,
    "motion": 0.90,
    "appearance": 0.86,
    "facets": 0.82,
    "filename": 0.74,
    "caption": 0.70,
    "usability": 0.62,
}


@dataclass
class SearchHit:
    element: Element
    score: float
    reasons: list[str]
    explain: dict[str, Any]


def search_elements(
    elements: list[Element],
    query: str,
    count: int = 20,
    filters: dict[str, str] | None = None,
) -> list[SearchHit]:
    filters = filters or {}
    query_tokens = _query_tokens(query)
    query_concepts = _concept_tokens(query)

    candidates = [element for element in elements if _passes_filters(element, filters)]
    if not candidates:
        return []

    if not query_tokens and not query_concepts:
        return [
            SearchHit(
                element=element,
                score=0.0,
                reasons=[],
                explain={"score": 0.0, "rank": rank, "query_terms": [], "query_concepts": []},
            )
            for rank, element in enumerate(candidates[:count], start=1)
        ]

    has_image_ai = any(len(element.image_embed) == 1024 for element in candidates)
    has_caption_ai = any(len(element.caption_embed) == 4096 for element in candidates)
    image_query_embed = _ai_query_embed(query, "image", candidates) if has_image_ai else None
    caption_query_embed = _ai_query_embed(query, "caption", candidates) if has_caption_ai else None

    image_scores = {
        element.element_id: _vector_score(image_query_embed or [], element.image_embed)
        for element in candidates
    }
    caption_scores = {
        element.element_id: _vector_score(caption_query_embed or [], element.caption_embed)
        for element in candidates
    }
    lexical_explains = {
        element.element_id: _lexical_explain(element, query_tokens) for element in candidates
    }
    lexical_scores = {eid: explain["score"] for eid, explain in lexical_explains.items()}

    semantic_explains = {
        element.element_id: _semantic_relevance(element, query_concepts)
        for element in candidates
    }
    retained = []
    for element in candidates:
        eid = element.element_id
        semantic = semantic_explains[eid]
        strongest_vector = max(image_scores[eid], caption_scores[eid])
        vector_only_match = (
            strongest_vector >= 0.48
            and not semantic["required_families"]
            and (has_image_ai or has_caption_ai)
        )
        if not semantic["eligible"] and not vector_only_match:
            continue
        direct_score = 0.84 * float(semantic["score"]) + 0.16 * max(0.0, strongest_vector)
        retained.append((element, direct_score))

    if not retained:
        return []

    retained.sort(key=lambda item: item[1], reverse=True)
    retained_ids = {element.element_id for element, _score in retained}
    image_rank = _rank({eid: score for eid, score in image_scores.items() if eid in retained_ids})
    caption_rank = _rank({eid: score for eid, score in caption_scores.items() if eid in retained_ids})
    lexical_rank = _rank({eid: score for eid, score in lexical_scores.items() if eid in retained_ids})
    semantic_rank = _rank(
        {eid: float(value["score"]) for eid, value in semantic_explains.items() if eid in retained_ids}
    )

    hits: list[SearchHit] = []
    for element, direct_score in retained:
        eid = element.element_id
        signals = {
            "image": _score_signal(image_scores[eid], image_rank.get(eid)),
            "caption": _score_signal(caption_scores[eid], caption_rank.get(eid)),
            "lexical": _score_signal(lexical_scores[eid], lexical_rank.get(eid)),
            "semantic": _score_signal(
                float(semantic_explains[eid]["score"]), semantic_rank.get(eid)
            ),
        }
        reasons = [
            _reason_text(name, signal)
            for name, signal in signals.items()
            if signal["score"] > 0
        ]
        explain = {
            "score": direct_score,
            "query_terms": sorted(query_tokens),
            "query_concepts": sorted(query_concepts),
            "signals": signals,
            "lexical": lexical_explains[eid],
            "semantic_relevance": semantic_explains[eid],
        }
        hits.append(
            SearchHit(element=element, score=round(direct_score, 6), reasons=reasons, explain=explain)
        )
    diversified = _diversify(sorted(hits, key=lambda hit: hit.score, reverse=True))
    for rank, hit in enumerate(diversified, start=1):
        hit.explain["rank"] = rank
    return diversified[:count]


def prewarm_ai_query_models(elements: list[Element]) -> None:
    """Load query-side AI models early when model-backed vectors are present."""
    if any(len(element.image_embed) == 1024 for element in elements):
        try:
            from .ai_runtime import siglip_text_stack

            siglip_text_stack()
        except Exception:
            pass
    if any(len(element.caption_embed) == 4096 for element in elements):
        try:
            from .ai_runtime import qwen_embedding_model

            qwen_embedding_model()
        except Exception:
            pass


def _rank(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {eid: position for position, (eid, score) in enumerate(ordered, start=1) if score > 0}


def _rrf_contribution(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (RRF_K + rank)


def _score_signal(score: float, rank: int | None) -> dict[str, float | int | None]:
    return {
        "rank": rank,
        "score": score,
        "rrf": _rrf_contribution(rank),
    }


def _reason_text(name: str, signal: dict[str, float | int | None]) -> str:
    rank = signal["rank"] if signal["rank"] is not None else "-"
    return f"{name} rank {rank} ({float(signal['score']):.2f})"


def _vector_score(query_embed: list[float], element_embed: list[float]) -> float:
    if not query_embed or not element_embed or len(query_embed) != len(element_embed):
        return 0.0
    return cosine(query_embed, element_embed)


def _passes_filters(element: Element, filters: dict[str, str]) -> bool:
    for key, expected in filters.items():
        if not expected:
            continue
        if key == "category":
            if expected not in {element.primary_family, element.category}:
                return False
            continue
        if key == "primary_family":
            if element.primary_family != expected:
                return False
            continue
        if key == "secondary_family":
            if expected not in element.secondary_families:
                return False
            continue
        if key.startswith("semantic."):
            actual = _nested_value(element.semantic_analysis, key.removeprefix("semantic."))
            if isinstance(actual, list):
                if expected not in {str(item) for item in actual}:
                    return False
            elif str(actual) != expected:
                return False
            continue
        actual: Any = facet_value(element.content_facets.get(key))
        if actual is None:
            actual = facet_value(element.motion_facets.get(key))
        if actual is None or str(actual) != expected:
            return False
    return True


def _lexical_explain(element: Element, query_tokens: set[str]) -> dict[str, Any]:
    if not query_tokens:
        return {
            "score": 0.0,
            "matched_terms": [],
            "buckets": {
                "category": [],
                "caption": [],
                "filename": [],
                "facets": [],
                "semantic": [],
            },
            "facet_matches": {},
        }
    facet_matches = _facet_matches(element, query_tokens)
    facet_terms = sorted({term for terms in facet_matches.values() for term in terms})
    buckets = {
        "category": _matched_query_terms(element.category, query_tokens),
        "caption": _matched_query_terms(element.caption, query_tokens),
        "filename": _matched_query_terms(
            _filename_search_text(element.primary().path), query_tokens
        ),
        "facets": facet_terms,
        "semantic": _matched_query_terms(_semantic_search_text(element), query_tokens),
    }
    matched_terms = sorted({term for terms in buckets.values() for term in terms})
    return {
        "score": len(matched_terms) / max(1, len(query_tokens)),
        "matched_terms": matched_terms,
        "buckets": buckets,
        "facet_matches": facet_matches,
    }


def _facet_matches(element: Element, query_tokens: set[str]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for prefix, facets in (() if element.semantic_analysis else (
        ("content", element.content_facets),
        ("motion", element.motion_facets),
    )):
        for key, payload in facets.items():
            terms = _matched_query_terms(str(facet_value(payload)), query_tokens)
            if terms:
                matches[f"{prefix}.{key}"] = terms
    for key, value in _search_values(element.semantic_analysis):
        terms = _matched_query_terms(str(value), query_tokens)
        if terms:
            matches[f"semantic.{key}"] = terms
    return matches


def _matched_query_terms(text: str, query_tokens: set[str]) -> list[str]:
    normalized = _affirmative_text(text).replace("_", " ").replace("-", " ")
    return sorted(set(tokenize(normalized)).intersection(query_tokens))


def _diversify(hits: list[SearchHit]) -> list[SearchHit]:
    seen_groups: set[str] = set()
    primary: list[SearchHit] = []
    variants: list[SearchHit] = []
    for hit in hits:
        group = hit.element.near_duplicate_group_id
        if not group:
            primary.append(hit)
            continue
        if group in seen_groups:
            variants.append(hit)
        else:
            seen_groups.add(group)
            primary.append(hit)
    return primary + variants


def _filename_search_text(value: str) -> str:
    name = value.rsplit("/", 1)[-1]
    return name.replace(".", " ").replace("_", " ").replace("-", " ")


def _query_tokens(query: str) -> set[str]:
    return {token for token in tokenize(query) if token not in QUERY_STOPWORDS}


def _concept_tokens(text: str) -> set[str]:
    """Convert artist phrasing and schema values into comparable concepts."""

    raw_tokens = tokenize(text.replace("_", " ").replace("-", " "))
    consumed: set[int] = set()
    concepts: set[str] = set()
    for phrase, canonical in PHRASE_ALIASES:
        length = len(phrase)
        for index in range(0, len(raw_tokens) - length + 1):
            positions = set(range(index, index + length))
            if positions.intersection(consumed):
                continue
            if tuple(raw_tokens[index : index + length]) == phrase:
                concepts.add(canonical)
                consumed.update(positions)
    for index, token in enumerate(raw_tokens):
        if index in consumed or token in QUERY_STOPWORDS or token.isdigit() or len(token) < 2:
            continue
        concepts.add(TOKEN_ALIASES.get(token, token))
    return concepts


def _semantic_relevance(element: Element, query_concepts: set[str]) -> dict[str, Any]:
    buckets = _search_concept_buckets(element)
    matched_by_bucket = {
        name: sorted(concepts.intersection(query_concepts))
        for name, concepts in buckets.items()
        if concepts.intersection(query_concepts)
    }
    matched = set(term for name, terms in matched_by_bucket.items()
                  if name != "filename" or not element.semantic_analysis for term in terms)
    required_families = query_concepts.intersection(FAMILY_CONCEPTS)
    element_families = _element_family_concepts(element)
    # A clear selected summary is strong enough family evidence for integrated
    # plates such as a helicopter exploding, even when the secondary-family tag
    # was omitted. Lower-confidence buckets (cleanup notes, captions) do not
    # relax this gate.
    confirmed = {key: value for key, value in (element.semantic_analysis or {}).items()
                 if key in {"summary", "primary_family", "secondary_families", "effect_type",
                            "appearance", "composition", "motion", "evidence", "temporal_detail"}}
    summary_family_evidence = _concept_tokens(_affirmative_text(_flatten_text(confirmed))).intersection(FAMILY_CONCEPTS)
    element_families.update(summary_family_evidence)
    family_match = required_families.issubset(element_families)
    required_matches = _minimum_concept_matches(len(query_concepts))
    eligible = family_match and len(matched) >= required_matches

    if not query_concepts:
        score = 0.0
        coverage = 0.0
        weighted_coverage = 0.0
    else:
        coverage = len(matched) / len(query_concepts)
        best_weights = [
            max(
                (
                    BUCKET_WEIGHTS[name]
                    for name, terms in matched_by_bucket.items()
                    if concept in terms
                ),
                default=0.0,
            )
            for concept in query_concepts
        ]
        weighted_coverage = sum(best_weights) / len(query_concepts)
        primary_concepts = _concept_tokens(
            " ".join([element.primary_family, element.category, *element.secondary_families])
        )
        primary_family_fraction = (
            len(required_families.intersection(primary_concepts)) / len(required_families)
            if required_families
            else 0.0
        )
        score = min(
            1.0,
            0.58 * coverage + 0.32 * weighted_coverage + 0.10 * primary_family_fraction,
        )
    return {
        "score": round(score, 6),
        "eligible": eligible,
        "coverage": round(coverage, 4),
        "weighted_coverage": round(weighted_coverage, 4),
        "required_matches": required_matches,
        "matched_concepts": sorted(matched),
        "missing_concepts": sorted(query_concepts.difference(matched)),
        "required_families": sorted(required_families),
        "element_families": sorted(element_families),
        "family_match": family_match,
        "matching_mode": "concept_coverage_with_vector_tiebreak",
        "filename_is_visual_evidence": False,
        "buckets": matched_by_bucket,
    }


def _minimum_concept_matches(concept_count: int) -> int:
    if concept_count <= 1:
        return concept_count
    if concept_count == 2:
        return 2
    return max(2, math.ceil(concept_count * 0.60))


def _search_concept_buckets(element: Element) -> dict[str, set[str]]:
    semantic = element.semantic_analysis or {}
    primary = element.primary()
    buckets = {
        "identity": _concept_tokens(
            " ".join(
                [
                    str(semantic.get("primary_family") or element.primary_family),
                    element.category if not semantic else "",
                    *(semantic.get("secondary_families") or element.secondary_families),
                    str(semantic.get("effect_type") or ""),
                ]
            )
        ),
        "summary": _concept_tokens(
            " ".join(
                [
                    _affirmative_text(str(semantic.get("summary") or "")),
                ]
            )
        ),
        "composition": _concept_tokens(_flatten_text(semantic.get("composition"))),
        "motion": _concept_tokens(_flatten_text(semantic.get("motion"))),
        "appearance": _concept_tokens(_flatten_text(semantic.get("appearance"))),
        "usability": _concept_tokens(_flatten_text(semantic.get("usability"))),
        "facets": set() if semantic else _concept_tokens(
            " ".join(
                str(facet_value(value))
                for value in [*element.content_facets.values(), *element.motion_facets.values()]
            )
        ),
        "caption": set() if semantic else _concept_tokens(_affirmative_text(element.caption)),
        "filename": _concept_tokens(
            _filename_search_text(primary.original_path or primary.path)
        ),
    }
    composition = semantic.get("composition") or {}
    edge_contacts = composition.get("edge_contact") or []
    if isinstance(edge_contacts, str):
        edge_contacts = [edge_contacts]
    for edge in edge_contacts:
        edge_name = str(edge).lower()
        if edge_name in {"top", "bottom", "left", "right"}:
            buckets["composition"].update({"edge_contact", f"{edge_name}_edge"})
    return buckets


def _element_family_concepts(element: Element) -> set[str]:
    semantic = element.semantic_analysis or {}
    primary = element.primary()
    text = " ".join(
        [
            str(semantic.get("primary_family") or element.primary_family),
            element.category if not semantic else "",
            *(semantic.get("secondary_families") or element.secondary_families),
            str(semantic.get("effect_type") or ""),
            _filename_search_text(primary.original_path or primary.path) if not semantic else "",
        ]
    )
    return _concept_tokens(text).intersection(FAMILY_CONCEPTS)


def _flatten_text(value: Any) -> str:
    return ". ".join(_affirmative_text(str(item)) for _key, item in _search_values(value))


def _semantic_search_text(element: Element) -> str:
    values = [_affirmative_text(str(value)) for _key, value in _search_values(element.semantic_analysis)]
    values.extend([element.primary_family, *element.secondary_families])
    return " ".join(values)


def _search_values(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"uncertainty", "search_text", "search_terms", "confidence", "field_confidence",
                       "overall_confidence", "action_timing"}:
                continue
            yield from _search_values(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for item in value:
            yield from _search_values(item, prefix)
    elif value is not None:
        yield prefix, value


def _affirmative_text(text: str) -> str:
    clauses = re.split(r"[.;,!?:]|\b(?:but|however|whereas)\b", text, flags=re.IGNORECASE)
    affirmed = []
    for clause in clauses:
        if re.search(r"\b(?:unclear|uncertain|possibly|perhaps|whether|cannot determine)\b", clause, re.I):
            continue
        words = re.findall(r"[a-z0-9_]+", clause.lower())
        for index, word in enumerate(words):
            if any(w in {"no", "not", "without", "neither", "absent"} for w in words[max(0, index-6):index]):
                continue
            if re.match(r"(?:is |are |was |were )?(?:not\b|absent\b)", " ".join(words[index+1:index+4])):
                continue
            if word not in {"no", "not", "without", "neither", "absent"}:
                affirmed.append(word)
    return " ".join(affirmed)


def _flatten_values(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_values(item, child)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                yield from _flatten_values(item, prefix)
            else:
                yield prefix, item
    elif value is not None:
        yield prefix, value


def _nested_value(value: dict[str, Any], dotted_key: str) -> Any:
    current: Any = value
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _ai_query_embed(query: str, kind: str, elements: list[Element]) -> list[float] | None:
    if not query:
        return None
    expected = 1024 if kind == "image" else 4096
    if not any(
        (len(element.image_embed) if kind == "image" else len(element.caption_embed)) == expected
        for element in elements
    ):
        return None
    try:
        if kind == "image":
            from .ai_runtime import siglip_query_embedding

            return siglip_query_embedding(query)
        from .ai_runtime import qwen_query_embedding

        return qwen_query_embedding(query)
    except Exception:
        return None

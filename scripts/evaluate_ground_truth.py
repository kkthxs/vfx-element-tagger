#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vfx_element_tagger.store import LibraryStore  # noqa: E402


DEFAULT_TRUTH = REPO_ROOT / "evaluation" / "testing_ground_truth.json"
CLAIM_INFLECTIONS = {
    "dissipates": "dissipate", "dissipated": "dissipate", "dissipating": "dissipate",
    "dissolves": "dissolve", "dissolved": "dissolve", "dissolving": "dissolve",
    "fades": "fade", "faded": "fade", "fading": "fade",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a catalog against partial artist ground truth.")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    parser.add_argument("--legacy-text", action="store_true", help="Reproduce the old all-fields keyword score (includes uncertainty).")
    parser.add_argument("--min-score", type=float, default=1.0, help="Required fraction of checks to pass; defaults to all checks.")
    args = parser.parse_args()
    if not 0 <= args.min_score <= 1:
        parser.error("--min-score must be between 0 and 1")

    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    elements = LibraryStore(args.library.resolve()).load()
    by_stem = {_element_stem(element): element for element in elements}
    results: list[dict[str, Any]] = []
    by_field: dict[str, dict[str, int]] = {}

    for stem, expectation in truth.get("assets", {}).items():
        element = by_stem.get(stem)
        if element is None:
            total = sum(_constraint_size(c) for c in expectation.get("constraints", []))
            for constraint in expectation.get("constraints", []):
                key = constraint.get("path") or constraint.get("type") or "unknown"
                metric = by_field.setdefault(key, {"passed": 0, "total": 0})
                metric["total"] += _constraint_size(constraint)
            results.append({"asset": stem, "passed": 0, "total": total, "missing": True, "failures": ["catalog element missing"]})
            continue
        checks = []
        for constraint in expectation.get("constraints", []):
            field_checks = _checks(element, constraint, legacy_text=args.legacy_text)
            checks.extend(field_checks)
            key = constraint.get("path") or constraint.get("type") or "unknown"
            metric = by_field.setdefault(key, {"passed": 0, "total": 0})
            metric["passed"] += sum(ok for ok, _ in field_checks)
            metric["total"] += len(field_checks)
        failures = [message for passed, message in checks if not passed]
        results.append(
            {
                "asset": stem,
                "element_id": element.element_id,
                "analysis_status": element.analysis_status,
                "human_reviewed": bool(element.human_overrides),
                "passed": sum(1 for passed, _message in checks if passed),
                "total": len(checks),
                "missing": False,
                "failures": failures,
            }
        )

    passed = sum(item["passed"] for item in results)
    total = sum(item["total"] for item in results)
    machine = [item for item in results if not item.get("human_reviewed")]
    machine_total = sum(item["total"] for item in machine)
    machine_passed = sum(item["passed"] for item in machine)
    payload = {
        "schema_version": truth.get("schema_version"),
        "passed": passed,
        "total": total,
        "score": round(passed / total, 4) if total else 0.0,
        "scoring_mode": "legacy_all_text" if args.legacy_text else "affirmative_selected_fields",
        "coverage": {"expected_assets": len(results), "found_assets": sum(not r["missing"] for r in results)},
        "field_metrics": {key: {**value, "score": round(value["passed"]/value["total"], 4)}
                          for key, value in by_field.items() if value["total"]},
        "accuracy_scope": "partial labeled constraints only; not general VFX accuracy",
        "machine_only": {"passed": machine_passed, "total": machine_total,
                         "score": round(machine_passed / machine_total, 4) if machine_total else None},
        "assets": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Partial ground-truth score: {passed}/{total} ({payload['score']:.1%})")
        for item in results:
            print(f"{item['asset']}: {item['passed']}/{item['total']} status={item.get('analysis_status', 'missing')}")
            for failure in item["failures"]:
                print(f"  - {failure}")
    return int(not total or any(item["missing"] for item in results) or passed / total < args.min_score)


def _checks(element, constraint: dict[str, Any], *, legacy_text: bool = False) -> list[tuple[bool, str]]:
    kind = constraint.get("type")
    values = [_normalise(str(value)) for value in constraint.get("values", [])]
    families = {_normalise(element.primary_family), *(_normalise(item) for item in element.secondary_families)}
    if kind == "family_any":
        ok = bool(families.intersection(values))
        return [(ok, f"family expected any of {constraint.get('values')}; got {sorted(families)}")]
    if kind == "family_all":
        missing = [value for value in values if value not in families]
        return [(not missing, f"family missing {missing}; got {sorted(families)}")]
    if kind == "text_all":
        texts = list(_flatten_text(element.semantic_analysis)) if legacy_text else _affirmative_text(element.semantic_analysis)
        checks: list[tuple[bool, str]] = []
        for group in constraint.get("concepts", []):
            matched = (
                any(_normalise(str(term)) in _normalise(" ".join(texts)) for term in group)
                if legacy_text
                else any(_term_present(text, str(term)) for text in texts for term in group)
            )
            checks.append((matched, f"semantic text missing concept {group}"))
        return checks
    if kind in {"path_contains", "path_excludes"}:
        actual = _nested(element.semantic_analysis, str(constraint.get("path") or ""))
        actual_values = {_normalise(str(item)) for item in (actual if isinstance(actual, list) else [actual])}
        if kind == "path_excludes":
            forbidden = [value for value in values if value in actual_values]
            return [(not forbidden, f"{constraint.get('path')} contains forbidden {forbidden}")]
        missing = [value for value in values if value not in actual_values]
        return [(not missing, f"{constraint.get('path')} missing {missing}; got {sorted(actual_values)}")]
    if kind == "range_covers":
        timing = _nested(element.semantic_analysis, str(constraint.get("path") or "action_timing")) or {}
        start, end = timing.get("optimal_start_seconds"), timing.get("optimal_end_seconds")
        ok = (isinstance(start, (int, float)) and isinstance(end, (int, float))
              and start <= float(constraint["start_seconds"]) <= float(constraint["end_seconds"]) <= end)
        return [(ok, f"action range {start}..{end} does not cover labeled event "
                     f"{constraint['start_seconds']}..{constraint['end_seconds']}")]
    return [(False, f"unknown constraint type: {kind}")]


def _constraint_size(constraint: dict[str, Any]) -> int:
    return len(constraint.get("concepts", [])) if constraint.get("type") == "text_all" else 1


def _affirmative_text(analysis: dict[str, Any]) -> list[str]:
    # Search suggestions and unresolved alternatives are useful for retrieval,
    # but are not evidence that the selected visual judgment is correct.
    fields = ("summary", "primary_family", "secondary_families", "effect_type",
              "composition", "appearance", "motion", "evidence", "temporal_detail")
    return [text for key in fields for text in _flatten_claims(analysis.get(key))]


def _flatten_claims(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in {"uncertainty", "search_terms", "search_text"}:
                yield from _flatten_claims(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_claims(item)
    elif value is not None:
        yield str(value)


def _term_present(text: str, term: str) -> bool:
    wanted = _claim_normalise(term)
    if not wanted:
        return False
    # Limit negation to a clause, so "no smoke, but fire rises" still credits fire.
    for clause in re.split(r"[.;,!?:]|\b(?:but|however|whereas)\b", text, flags=re.IGNORECASE):
        normalized = _claim_normalise(clause)
        for match in re.finditer(r"(?<!\w)" + re.escape(wanted) + r"(?!\w)", normalized):
            before = normalized[:match.start()].split()
            after = normalized[match.end():].split()
            if any(token in {"no", "not", "without", "neither", "absent"} for token in before[-6:]):
                continue
            if re.match(r"(?:is |are |was |were )?(?:not |absent|unclear|uncertain)", " ".join(after)):
                continue
            if any(token in {"unclear", "uncertain", "possibly", "perhaps", "whether"} for token in before[-8:]):
                continue
            return True
    return False


def _claim_normalise(value: str) -> str:
    return " ".join(CLAIM_INFLECTIONS.get(token, token) for token in _normalise(value).split())


def _element_stem(element) -> str:
    primary = element.primary()
    candidate = Path(primary.original_path)
    return candidate.stem if candidate.is_file() or candidate.suffix else candidate.name


def _flatten_text(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten_text(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_text(item)
    elif value is not None:
        yield str(value)


def _nested(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vfx_element_tagger.store import LibraryStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate action timing stored in an analysed catalog.")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    elements = [
        element
        for element in LibraryStore(args.library.resolve()).load()
        if element.analysis_status != "unanalysed"
    ]
    results = [_check(element) for element in elements]
    invalid = [result for result in results if result["issues"]]
    patterns = Counter(result["pattern"] for result in results if result["pattern"])
    coverage = [result["optimal_coverage"] for result in results if result["optimal_coverage"] is not None]
    payload = {
        "analysed": len(elements),
        "valid": len(elements) - len(invalid),
        "invalid": len(invalid),
        "patterns": dict(sorted(patterns.items())),
        "median_optimal_coverage": round(statistics.median(coverage), 4) if coverage else None,
        "issues": invalid,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Action timing: {payload['valid']}/{payload['analysed']} valid; "
            f"patterns={payload['patterns']}; "
            f"median optimal coverage={payload['median_optimal_coverage']}"
        )
        for result in invalid:
            print(f"  {result['element_id']}: {'; '.join(result['issues'])}")
    return 1 if invalid else 0


def _check(element) -> dict:
    timing = element.semantic_analysis.get("action_timing") or {}
    issues: list[str] = []
    duration = float(element.duration_seconds or 0.0)
    fields = [
        "visible_start_seconds",
        "optimal_start_seconds",
        "main_action_start_seconds",
        "peak_seconds",
        "main_action_end_seconds",
        "optimal_end_seconds",
        "visible_end_seconds",
    ]
    values = {field: timing.get(field) for field in fields}
    if not timing or not timing.get("available"):
        issues.append("missing action_timing")
    elif any(value is None for value in values.values()):
        issues.append("one or more range fields are missing")
    else:
        start = float(values["visible_start_seconds"])
        main_start = float(values["main_action_start_seconds"])
        peak = float(values["peak_seconds"])
        main_end = float(values["main_action_end_seconds"])
        visible_end = float(values["visible_end_seconds"])
        optimal_start = float(values["optimal_start_seconds"])
        optimal_end = float(values["optimal_end_seconds"])
        if not (0 <= start <= main_start <= peak <= main_end <= visible_end <= duration + 0.05):
            issues.append("visible/main/peak ordering is invalid")
        if not (0 <= optimal_start <= main_start and main_end <= optimal_end <= duration + 0.05):
            issues.append("optimal range does not contain main action")
        for segment in timing.get("event_segments") or []:
            if not (
                0
                <= float(segment.get("start_seconds") or 0.0)
                <= float(segment.get("peak_seconds") or 0.0)
                <= float(segment.get("end_seconds") or 0.0)
                <= duration + 0.05
            ):
                issues.append("event segment ordering is invalid")
                break
    optimal_duration = timing.get("optimal_duration_seconds")
    optimal_coverage = (
        float(optimal_duration) / duration
        if optimal_duration is not None and duration > 0
        else None
    )
    return {
        "element_id": element.element_id,
        "pattern": timing.get("pattern"),
        "optimal_coverage": round(optimal_coverage, 4) if optimal_coverage is not None else None,
        "issues": issues,
    }


if __name__ == "__main__":
    raise SystemExit(main())

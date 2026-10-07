#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vfx_element_tagger.store import LibraryStore


ANALYSED_STATUSES = {"accepted", "escalated", "needs_review"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reuse semantic analysis across catalogs using exact content fingerprints."
    )
    parser.add_argument("--source-library", type=Path, required=True)
    parser.add_argument("--target-library", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_store = LibraryStore(args.source_library.resolve())
    target_store = LibraryStore(args.target_library.resolve())
    source_elements = source_store.load()
    target_elements = target_store.load()
    source_by_fingerprint = _analysis_index(source_elements)

    reused = 0
    ambiguous = 0
    updated_targets = []
    for target in target_elements:
        matches = {
            id(source): source
            for fingerprint in _fingerprints(target)
            for source in source_by_fingerprint.get(fingerprint, [])
        }
        if len(matches) != 1:
            if len(matches) > 1:
                ambiguous += 1
            continue
        source = next(iter(matches.values()))
        if target.analysis_status in ANALYSED_STATUSES:
            continue
        _copy_analysis(source, target)
        updated_targets.append(target)
        reused += 1
        print(
            f"{target.element_id}: reused {source.element_id} "
            f"status={source.analysis_status} schema={source.analysis_schema_version}"
        )

    if not args.dry_run:
        if target_store.is_sqlite:
            for target in updated_targets:
                target_store.save_element(target)
        else:
            target_store.save(target_elements)
    print(
        f"Reused analysis for {reused} elements; ambiguous={ambiguous}; "
        f"dry_run={args.dry_run}."
    )
    return 0


def _analysis_index(elements) -> dict[str, list]:
    index: dict[str, list] = {}
    for element in elements:
        if element.analysis_status not in ANALYSED_STATUSES or not element.semantic_analysis:
            continue
        for fingerprint in _fingerprints(element):
            index.setdefault(fingerprint, []).append(element)
    return index


def _fingerprints(element) -> set[str]:
    return {
        str(representation.content_fingerprint)
        for representation in element.source_representations
        if representation.content_fingerprint
    }


def _copy_analysis(source, target) -> None:
    target.semantic_analysis = deepcopy(source.semantic_analysis)
    target.primary_family = source.primary_family
    target.secondary_families = list(source.secondary_families)
    target.analysis_candidates = deepcopy(source.analysis_candidates)
    target.analysis_confidence = source.analysis_confidence
    target.analysis_status = source.analysis_status
    target.analysis_model_version = source.analysis_model_version
    target.analysis_schema_version = source.analysis_schema_version
    target.analysis_escalation = deepcopy(source.analysis_escalation)
    target.caption = source.caption
    target.captioning_model_version = source.captioning_model_version
    target.category = source.category
    target.category_confidence = source.category_confidence
    target.qwen_tags = deepcopy(source.qwen_tags)
    target.provenance["semantic_analysis"] = {
        **deepcopy(source.provenance.get("semantic_analysis") or {}),
        "reused_from_element_id": source.element_id,
        "reuse_match": "exact_content_fingerprint",
    }


if __name__ == "__main__":
    raise SystemExit(main())

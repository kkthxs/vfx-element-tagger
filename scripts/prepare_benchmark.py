#!/usr/bin/env python3
"""Prepare a model-output-free annotation queue; never invent ground truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vfx_element_tagger.settings import library_path
from vfx_element_tagger.store import LibraryStore


def annotation_queue(elements, excluded_stems=(), *, seed=42, limit=200):
    elements = sorted(elements, key=lambda element: element.element_id)
    parents = list(range(len(elements)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    # All duplicate identifiers form connected groups, including transitive matches.
    owners = {}
    for index, element in enumerate(elements):
        identifiers = (("family", element.source_family_id),
                       ("duplicate", element.near_duplicate_group_id),
                       ("fingerprint", element.content_fingerprint),
                       ("element", element.element_id))
        for kind, value in identifiers:
            if not value:
                continue
            identifier = (kind, value)
            if identifier in owners:
                left, right = root(index), root(owners[identifier])
                parents[max(left, right)] = min(left, right)
            else:
                owners[identifier] = index

    groups = {}
    excluded_groups = set()
    for index, element in enumerate(elements):
        group = elements[root(index)].element_id
        groups.setdefault(group, []).append(element)
        if Path(element.primary().original_path).stem in excluded_stems or element.human_overrides:
            excluded_groups.add(group)
    eligible = [(group, members) for group, members in sorted(groups.items()) if group not in excluded_groups]
    random.Random(seed).shuffle(eligible)
    assets = []
    for group, members in eligible[:limit]:
        representative = sorted(members, key=lambda e: e.element_id)[0]
        assets.append({"element_id": representative.element_id,
                       "group_id": hashlib.sha256(group.encode()).hexdigest()[:16],
                       "source_path": representative.primary().original_path,
                       "preview_path": representative.preview_movie_1080_path or representative.preview_movie_480_path,
                       "content_fingerprint": representative.content_fingerprint,
                       "annotation_status": "pending", "annotator": "", "constraints": [],
                       "ambiguous_fields": [], "notes": ""})
    return {"schema_version": "vfx-blind-annotation-0.1", "seed": seed,
            "requested_groups": limit, "eligible_groups": len(eligible), "selected_groups": len(assets),
            "labeled_assets": 0, "model_output_included": False, "assets": assets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=library_path())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-truth", type=Path, default=ROOT/"evaluation/testing_ground_truth.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; refusing to replace annotations")
    if args.limit < 1:
        parser.error("--limit must be positive")
    exclusions = json.loads(args.exclude_truth.read_text()).get("assets", {}) if args.exclude_truth.is_file() else {}
    payload = annotation_queue(LibraryStore.load_readonly(args.library), exclusions, seed=args.seed, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Prepared {payload['selected_groups']} of {payload['requested_groups']} requested independent groups. "
          "All annotations are pending; this is not an accuracy result.")


if __name__ == "__main__":
    main()

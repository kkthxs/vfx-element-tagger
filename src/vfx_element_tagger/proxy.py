from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from .metadata import alpha_cluster
from .models import Element, SourceRepresentation
from .util import stable_id

if TYPE_CHECKING:  # Avoid circular import at runtime.
    from .discovery import SidecarManifest


LOSSLESS_CODEC_HINTS = {"prores", "dnx", "dnxhr", "dnxhd", "ffv1", "huffyuv", "utvideo"}
LOSSY_CODEC_HINTS = {"h264", "hevc", "h265", "mpeg4", "vp9", "av1"}
PHASH_HAMMING_THRESHOLD = 6


def build_elements_from_representations(
    representations: list[SourceRepresentation],
    sidecar_manifests: "list[SidecarManifest] | None" = None,
) -> list[Element]:
    clusters = _cluster_representations(representations, sidecar_manifests or [])
    elements = [_element_from_cluster(cluster) for cluster in clusters]
    assign_near_duplicate_groups(elements)
    return elements


def assign_near_duplicate_groups(elements: list[Element]) -> None:
    for element in elements:
        element.near_duplicate_group_id = None
    _assign_near_duplicate_groups(elements)


def _cluster_representations(
    representations: list[SourceRepresentation],
    sidecar_manifests: "list[SidecarManifest] | None" = None,
) -> list[list[SourceRepresentation]]:
    parent = {representation.representation_id: representation.representation_id for representation in representations}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    filename_buckets: dict[str, list[SourceRepresentation]] = defaultdict(list)
    fingerprint_buckets: dict[str, list[SourceRepresentation]] = defaultdict(list)
    for representation in representations:
        filename_buckets[
            representation.filename_root or Path(representation.path).stem
        ].append(representation)
        if representation.content_fingerprint:
            fingerprint_buckets[representation.content_fingerprint].append(representation)

    # A full content fingerprint is the only inferred signal strong enough to
    # link representations without corroboration.
    for bucket in fingerprint_buckets.values():
        if len(bucket) < 2:
            continue
        first = bucket[0].representation_id
        for representation in bucket[1:]:
            union(first, representation.representation_id)
        for representation in bucket:
            representation.linked_by_fingerprint = True

    # Exact logical names may link alternate encodes only when duration and
    # geometry agree. Variant numbers are preserved during discovery, so
    # SteamJet-009 and SteamJet-016 remain separate elements.
    for bucket in filename_buckets.values():
        for left_index, left in enumerate(bucket):
            for right in bucket[left_index + 1 :]:
                if not _representations_compatible(left, right):
                    continue
                union(left.representation_id, right.representation_id)
                left.linked_by_filename = True
                right.linked_by_filename = True
                if _phash_matches(left, right):
                    left.linked_by_phash = True
                    right.linked_by_phash = True

    for left, right in _near_phash_pairs(representations):
        same_root = (left.filename_root or "") == (right.filename_root or "")
        if same_root and _representations_compatible(left, right):
            union(left.representation_id, right.representation_id)
            left.linked_by_phash = True
            right.linked_by_phash = True

    # Sidecar manifests are authoritative — apply them on top of inferred unions.
    sidecar_groups: list[list[SourceRepresentation]] = []
    sidecar_separate: list[list[SourceRepresentation]] = []
    if sidecar_manifests:
        path_index = _index_representations_by_path(representations)
        for manifest in sidecar_manifests:
            for member_paths in manifest.groups:
                matched = _resolve_members(member_paths, path_index)
                if len(matched) < 2:
                    continue
                first = matched[0].representation_id
                for representation in matched[1:]:
                    union(first, representation.representation_id)
                for representation in matched:
                    representation.linked_from_sidecar = True
                sidecar_groups.append(matched)
            for separate_paths in manifest.separate:
                matched = _resolve_members(separate_paths, path_index)
                if len(matched) >= 2:
                    sidecar_separate.append(matched)

    grouped: dict[str, list[SourceRepresentation]] = defaultdict(list)
    for representation in representations:
        grouped[find(representation.representation_id)].append(representation)

    clusters = list(grouped.values())
    for cluster in clusters:
        for representation in cluster:
            representation.link_confidence = _link_confidence(representation, len(cluster))

    # Force sidecar-grouped members to confidence 1.0 (clamp to [0, 1.0]).
    for group in sidecar_groups:
        for representation in group:
            representation.link_confidence = min(1.0, 1.0)

    # Honour `separate` directives by promoting offenders into singleton clusters.
    if sidecar_separate:
        clusters = _enforce_separate(clusters, sidecar_separate)

    return clusters


def _near_phash_pairs(
    representations: list[SourceRepresentation],
) -> list[tuple[SourceRepresentation, SourceRepresentation]]:
    parsed = [
        (representation, value)
        for representation in representations
        if (value := _parse_phash64(representation.perceptual_hash)) is not None
        and _informative_phash(value)
    ]
    pairs: list[tuple[SourceRepresentation, SourceRepresentation]] = []
    for left_index, (left, left_hash) in enumerate(parsed):
        for right, right_hash in parsed[left_index + 1:]:
            if (left_hash ^ right_hash).bit_count() <= PHASH_HAMMING_THRESHOLD:
                pairs.append((left, right))
    return pairs


def _informative_phash(value: int) -> bool:
    # All-black/all-white poster frames are common for VFX with delayed onset
    # and carry no identity signal whatsoever.
    return value not in {0, (1 << 64) - 1}


def _phash_matches(left: SourceRepresentation, right: SourceRepresentation) -> bool:
    left_hash = _parse_phash64(left.perceptual_hash)
    right_hash = _parse_phash64(right.perceptual_hash)
    if left_hash is None or right_hash is None:
        return False
    if not _informative_phash(left_hash) or not _informative_phash(right_hash):
        return False
    return (left_hash ^ right_hash).bit_count() <= PHASH_HAMMING_THRESHOLD


def _representations_compatible(
    left: SourceRepresentation, right: SourceRepresentation
) -> bool:
    left_duration = left.duration_seconds or 0.0
    right_duration = right.duration_seconds or 0.0
    if left_duration > 0 and right_duration > 0:
        tolerance = max(0.25, max(left_duration, right_duration) * 0.03)
        if abs(left_duration - right_duration) > tolerance:
            return False
    if left.width and left.height and right.width and right.height:
        left_aspect = left.width / left.height
        right_aspect = right.width / right.height
        if abs(left_aspect - right_aspect) > max(0.02, left_aspect * 0.02):
            return False
    return True


def _parse_phash64(value: str | None) -> int | None:
    if not value or not value.startswith("phash_"):
        return None
    hex_value = value.removeprefix("phash_")
    if len(hex_value) != 16:
        return None
    try:
        return int(hex_value, 16)
    except ValueError:
        return None


def _index_representations_by_path(
    representations: list[SourceRepresentation],
) -> dict[str, SourceRepresentation]:
    index: dict[str, SourceRepresentation] = {}
    for representation in representations:
        try:
            resolved = str(Path(representation.path).resolve())
        except (OSError, RuntimeError):
            resolved = representation.path
        index[resolved] = representation
        index[representation.path] = representation
    return index


def _resolve_members(
    member_paths: list[Path],
    path_index: dict[str, SourceRepresentation],
) -> list[SourceRepresentation]:
    matched: list[SourceRepresentation] = []
    seen_ids: set[str] = set()
    for member in member_paths:
        candidates = [str(member), str(member.resolve()) if member.exists() else str(member)]
        rep = None
        for candidate in candidates:
            if candidate in path_index:
                rep = path_index[candidate]
                break
        if rep is None:
            continue
        if rep.representation_id in seen_ids:
            continue
        seen_ids.add(rep.representation_id)
        matched.append(rep)
    return matched


def _enforce_separate(
    clusters: list[list[SourceRepresentation]],
    sidecar_separate: list[list[SourceRepresentation]],
) -> list[list[SourceRepresentation]]:
    # Build current membership: representation_id -> cluster index.
    membership: dict[str, int] = {}
    for idx, cluster in enumerate(clusters):
        for representation in cluster:
            membership[representation.representation_id] = idx

    # Mutable working copy.
    working: list[list[SourceRepresentation]] = [list(cluster) for cluster in clusters]

    for pair in sidecar_separate:
        # Find any two representations in `pair` that share a cluster.
        for i in range(len(pair)):
            for j in range(i + 1, len(pair)):
                a, b = pair[i], pair[j]
                if membership.get(a.representation_id) == membership.get(b.representation_id):
                    target_idx = membership[b.representation_id]
                    if target_idx is None:
                        continue
                    sys.stderr.write(
                        f"[sidecar] separating {b.path} from {a.path} (cluster split)\n"
                    )
                    working[target_idx] = [
                        rep for rep in working[target_idx]
                        if rep.representation_id != b.representation_id
                    ]
                    working.append([b])
                    membership[b.representation_id] = len(working) - 1

    return [cluster for cluster in working if cluster]


def _link_confidence(representation: SourceRepresentation, cluster_size: int) -> float:
    if cluster_size <= 1:
        return 1.0
    confidence = 0.0
    if representation.linked_by_fingerprint:
        confidence += 0.45
    if representation.linked_by_filename:
        confidence += 0.3
    if representation.linked_by_phash:
        confidence += 0.25
    return min(1.0, max(0.25, confidence))


def _element_from_cluster(cluster: list[SourceRepresentation]) -> Element:
    primary = select_primary_representation(cluster)
    for representation in cluster:
        representation.is_primary = representation.representation_id == primary.representation_id
    element_id = stable_id("element", *sorted(representation.path for representation in cluster))
    alpha = alpha_cluster(primary)
    width = primary.width
    height = primary.height
    aspect_ratio = (width / height) if width and height else None
    source_family = stable_id(
        "family",
        *(sorted(representation.filename_root or representation.path for representation in cluster)),
        length=12,
    )
    return Element(
        element_id=element_id,
        source_representations=sorted(cluster, key=lambda item: item.path),
        primary_representation_id=primary.representation_id,
        duration_seconds=primary.duration_seconds,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        fps=primary.fps,
        color_space=primary.color_space,
        bit_depth=primary.bit_depth,
        is_sequence=primary.is_sequence,
        has_missing_frames=bool(primary.missing_frames),
        content_fingerprint=primary.content_fingerprint,
        perceptual_hash=primary.perceptual_hash,
        source_family_id=source_family,
        **alpha,
    )


def select_primary_representation(cluster: list[SourceRepresentation]) -> SourceRepresentation:
    return max(cluster, key=_primary_score)


def _primary_score(representation: SourceRepresentation) -> tuple[int, int, int, int]:
    format_name = representation.format.lower()
    codec = (representation.codec or "").lower()
    if representation.is_sequence and "exr" in format_name:
        format_score = 500
    elif representation.is_sequence:
        format_score = 450
    elif any(hint in codec for hint in LOSSLESS_CODEC_HINTS):
        format_score = 400
    elif any(hint in codec for hint in LOSSY_CODEC_HINTS):
        format_score = 300
    elif format_name == "single_image":
        format_score = 100
    else:
        format_score = 200
    pixels = (representation.width or 0) * (representation.height or 0)
    bit_depth = representation.bit_depth or 0
    file_size = representation.file_size_bytes or 0
    return (format_score, pixels, bit_depth, file_size)


def _assign_near_duplicate_groups(elements: list[Element]) -> None:
    buckets: dict[str, list[Element]] = defaultdict(list)
    for element in elements:
        key = element.perceptual_hash or element.content_fingerprint
        parsed = _parse_phash64(element.perceptual_hash)
        if key and (parsed is None or _informative_phash(parsed)):
            buckets[key].append(element)
    for key, bucket in buckets.items():
        if len(bucket) < 2:
            continue
        group_id = stable_id("near_duplicate", key, length=12)
        for element in bucket:
            element.near_duplicate_group_id = group_id

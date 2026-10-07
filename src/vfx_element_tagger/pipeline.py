from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path

from .artifacts import (
    ARTIFACT_VERSION,
    generate_artifacts,
    generate_representation_phash,
    perceptual_hash_from_artifact,
)
from .discovery import discover_media, discover_sidecar_manifests
from .metadata import build_source_representation
from .models import Element, ProcessingEvent
from .proxy import assign_near_duplicate_groups, build_elements_from_representations
from .stages import caption_element, classify_element, derive_motion, embed_element
from .store import LibraryStore
from .util import now_ms
from .catalog_lock import catalog_writer


@dataclass
class IngestResult:
    elements: list[Element]
    references_seen: int
    library_path: Path
    log_path: Path
    processed_elements: int = 0
    reused_elements: int = 0
    preserved_unseen_elements: int = 0
    scanned_element_ids: list[str] = field(default_factory=list)
    failed_references: int = 0


def run_ingest(
    folder: Path,
    cache_dir: Path,
    library_path: Path,
    artifacts_enabled: bool = True,
    default_fps: float = 24.0,
    force_recompute: bool = False,
    prune_missing: bool = False,
    progress=None,
    writer_locked: bool = False,
) -> IngestResult:
    kwargs = dict(artifacts_enabled=artifacts_enabled, default_fps=default_fps,
                  force_recompute=force_recompute, prune_missing=prune_missing, progress=progress)
    if writer_locked:
        return _run_ingest(folder, cache_dir, library_path, **kwargs)
    with catalog_writer(library_path):
        return _run_ingest(folder, cache_dir, library_path, **kwargs)


def _run_ingest(folder, cache_dir, library_path, artifacts_enabled, default_fps,
                force_recompute, prune_missing, progress):
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = LibraryStore(library_path)
    existing_elements = store.load()
    existing_by_id = {element.element_id: element for element in existing_elements}
    if progress:
        progress({"phase": "discovery", "completed": 0, "total": None, "current": folder.name})
    references = discover_media(folder, cache_dir=cache_dir)
    sidecar_manifests = discover_sidecar_manifests(
        folder,
        excluded=[cache_dir] if cache_dir else None,
    )
    events: list[ProcessingEvent] = []
    for manifest in sidecar_manifests:
        events.append(
            ProcessingEvent(
                element_id=str(manifest.path),
                stage="stage_0_5_sidecar",
                status="success",
                model_version="sidecar-override-0.1.6",
            )
        )

    representations = []
    for index, reference in enumerate(references):
        if progress:
            progress({"phase": "metadata", "completed": index, "total": len(references),
                      "current": str(reference.reference_id)})
        started = now_ms()
        try:
            representation = build_source_representation(reference, default_fps=default_fps)
            representation.perceptual_hash = generate_representation_phash(
                representation,
                cache_dir,
                enabled=artifacts_enabled,
            )
            representations.append(representation)
        except Exception as exc:  # Defensive: bad files should not stop the pilot.
            events.append(
                ProcessingEvent(
                    element_id=reference.reference_id,
                    stage="stage_0_metadata",
                    status="failed",
                    error_kind=exc.__class__.__name__,
                    error_message=str(exc),
                    model_version="ffprobe-fallback-0.1.5",
                    wall_clock_ms=now_ms() - started,
                )
            )

    elements = build_elements_from_representations(
        representations,
        sidecar_manifests=sidecar_manifests or None,
    )
    _guard_regrouping(existing_elements, elements)
    discovered_ids = {element.element_id for element in elements}
    output_by_id: dict[str, Element] = {}
    output_order: list[str] = []
    if not prune_missing:
        for existing in existing_elements:
            output_by_id[existing.element_id] = existing
            output_order.append(existing.element_id)

    processed_count = 0
    reused_count = 0
    for index, element in enumerate(elements):
        if progress:
            progress({"phase": "previews", "completed": index, "total": len(elements),
                      "current": Path(element.primary().original_path).name})
        existing = existing_by_id.get(element.element_id)
        if (
            not force_recompute
            and existing is not None
            and _should_reuse_existing_element(existing, element)
        ):
            final_element = existing
            reused_count += 1
            events.append(
                ProcessingEvent(
                    element_id=element.element_id,
                    stage="ingest_incremental",
                    status="skipped",
                    model_version="incremental-reuse-0.1.6",
                )
            )
        else:
            _run_element_stages(element, cache_dir, artifacts_enabled, events)
            if existing is not None:
                _preserve_analysis(existing, element)
            final_element = element
            processed_count += 1
        if final_element.element_id not in output_by_id:
            output_order.append(final_element.element_id)
        output_by_id[final_element.element_id] = final_element

    if prune_missing:
        final_elements = [output_by_id[element_id] for element_id in output_order if element_id in discovered_ids]
    else:
        final_elements = [output_by_id[element_id] for element_id in output_order]
    assign_near_duplicate_groups(final_elements)

    store.save(final_elements)
    store.append_events(events)
    failures = sum(event.status == "failed" for event in events)
    if progress:
        progress({"phase": "imported", "completed": len(elements), "total": len(elements),
                  "processed": processed_count, "reused": reused_count, "errors": failures})
        for event in events:
            if event.status == "failed":
                progress({"message": f"{event.element_id}: {event.error_message}"})
    return IngestResult(
        elements=final_elements,
        references_seen=len(references),
        library_path=library_path,
        log_path=store.log_path,
        processed_elements=processed_count,
        reused_elements=reused_count,
        preserved_unseen_elements=0 if prune_missing else len([item for item in existing_elements if item.element_id not in discovered_ids]),
        scanned_element_ids=sorted(discovered_ids),
        failed_references=failures,
    )


def _should_reuse_existing_element(existing: Element, candidate: Element) -> bool:
    return _element_source_signature(existing) == _element_source_signature(candidate)


def _guard_regrouping(existing: list[Element], candidates: list[Element]) -> None:
    owners = {rep.path: item.element_id for item in existing for rep in item.source_representations}
    for item in candidates:
        if any(owners.get(rep.path, item.element_id) != item.element_id
               for rep in item.source_representations):
            raise ValueError(
                "Master/proxy grouping changed for an existing source. Catalog was not written. "
                "Ingest this grouping into a separate catalog and reconcile artist reviews and "
                "ratings explicitly; do not prune or force an ambiguous merge."
            )


def _preserve_analysis(existing: Element, candidate: Element) -> None:
    changed = not _should_reuse_existing_element(existing, candidate)
    fields = (
        "category", "category_confidence", "content_facets", "motion_facets",
        "measured_features", "motion_warnings", "caption", "qwen_tags",
        "captioning_model_version", "embedding_model_version", "image_embed", "caption_embed",
        "primary_family", "secondary_families", "semantic_analysis", "analysis_candidates",
        "analysis_confidence", "analysis_status", "analysis_model_version", "analysis_schema_version",
        "analysis_escalation", "human_overrides", "video_embed",
    )
    for field in fields:
        setattr(candidate, field, deepcopy(getattr(existing, field)))
    candidate.provenance.update(deepcopy(existing.provenance))
    if changed:
        candidate.analysis_escalation.pop("analysis_signature", None)
        candidate.analysis_escalation["ingest_refresh"] = {
            "source_changed": True,
            "reason": "Source changed; retained analysis and artist fields require revalidation.",
        }
        candidate.image_embed = []
        candidate.caption_embed = []
        candidate.video_embed = []
        if _is_model_backed(existing) or existing.human_overrides:
            candidate.analysis_status = "needs_review"
        if existing.duration_seconds != candidate.duration_seconds:
            candidate.semantic_analysis.pop("action_timing", None)


def _element_source_signature(element: Element) -> tuple[tuple[str, str | None, int | None, str | None], ...]:
    return tuple(
        sorted(
            (
                representation.path,
                representation.content_fingerprint,
                representation.file_size_bytes,
                representation.perceptual_hash,
            )
            for representation in element.source_representations
        )
    )


def rerun_stage(element: Element, stage: str, cache_dir: Path, artifacts_enabled: bool = True) -> None:
    if stage in {"stage_1", "classification", "stage_2", "caption", "stage_3", "motion", "stage_4", "embedding"}:
        _guard_model_backed_rerun(element, stage)
    if stage in {"stage_1", "classification"}:
        classify_element(element)
    elif stage in {"stage_2", "caption"}:
        caption_element(element)
        embed_element(element)
    elif stage in {"stage_3", "motion"}:
        derive_motion(element)
        caption_element(element)
        embed_element(element)
    elif stage in {"stage_4", "embedding"}:
        embed_element(element)
    elif stage in {"stage_0_5", "artifacts"}:
        artifact_paths = generate_artifacts(
            element.element_id,
            element.primary(),
            cache_dir,
            enabled=artifacts_enabled,
        )
        element.poster_path = artifact_paths["poster_path"]
        element.preview_movie_480_path = artifact_paths["preview_movie_480_path"]
        element.preview_movie_1080_path = artifact_paths["preview_movie_1080_path"]
        element.artifact_generation_version = ARTIFACT_VERSION
        element.perceptual_hash = perceptual_hash_from_artifact(
            element.poster_path,
            fallback=element.content_fingerprint,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


def _guard_model_backed_rerun(element: Element, stage: str) -> None:
    if not _is_model_backed(element):
        return
    raise ValueError(
        f"{stage} was produced with local AI models. The QC re-run button only runs "
        "deterministic fallback stages; use scripts/analyze_gated.py to refresh "
        "model-backed analysis without downgrading this element."
    )


def _is_model_backed(element: Element) -> bool:
    caption_version = element.captioning_model_version.lower()
    embedding_version = element.embedding_model_version.lower()
    provenance_sources = {
        str(payload.get("source", "")).lower()
        for payload in element.provenance.values()
        if isinstance(payload, dict)
    }
    return (
        element.analysis_status in {"accepted", "escalated", "needs_review"}
        or bool(element.semantic_analysis)
        or "qwen" in caption_version
        or "qwen" in embedding_version
        or "siglip" in embedding_version
        or len(element.image_embed) == 1024
        or len(element.caption_embed) == 4096
        or any("qwen" in source or "siglip" in source for source in provenance_sources)
    )


def _run_element_stages(
    element: Element,
    cache_dir: Path,
    artifacts_enabled: bool,
    events: list[ProcessingEvent],
) -> None:
    _evented(events, element.element_id, "stage_0_metadata", "ffprobe-fallback-0.1.5", lambda: None)
    _evented(
        events,
        element.element_id,
        "stage_0_5_artifacts",
        ARTIFACT_VERSION,
        lambda: _stage_artifacts(element, cache_dir, artifacts_enabled),
    )
    _evented(
        events,
        element.element_id,
        "stage_1_classification",
        "filename-taxonomy-fallback-0.1.5",
        lambda: classify_element(element),
    )
    _evented(
        events,
        element.element_id,
        "stage_3_motion",
        "deterministic-motion-fallback-0.1.5",
        lambda: derive_motion(element),
    )
    _evented(
        events,
        element.element_id,
        "stage_2_caption",
        "deterministic-caption-fallback-0.1.5",
        lambda: caption_element(element),
    )
    _evented(
        events,
        element.element_id,
        "stage_4_embedding",
        "deterministic-hash-embedding-0.1.5",
        lambda: embed_element(element),
    )


def _stage_artifacts(element: Element, cache_dir: Path, artifacts_enabled: bool) -> None:
    artifact_paths = generate_artifacts(
        element.element_id,
        element.primary(),
        cache_dir,
        enabled=artifacts_enabled,
    )
    element.poster_path = artifact_paths["poster_path"]
    element.preview_movie_480_path = artifact_paths["preview_movie_480_path"]
    element.preview_movie_1080_path = artifact_paths["preview_movie_1080_path"]
    element.artifact_generation_version = ARTIFACT_VERSION
    element.perceptual_hash = perceptual_hash_from_artifact(
        element.poster_path,
        fallback=element.content_fingerprint,
    )
    element.primary().perceptual_hash = element.perceptual_hash


def _evented(
    events: list[ProcessingEvent],
    element_id: str,
    stage: str,
    model_version: str,
    callback,
) -> None:
    started = now_ms()
    try:
        callback()
    except Exception as exc:  # Keep pilot ingestion resumable and inspectable.
        events.append(
            ProcessingEvent(
                element_id=element_id,
                stage=stage,
                status="failed",
                error_kind=exc.__class__.__name__,
                error_message=str(exc),
                model_version=model_version,
                wall_clock_ms=now_ms() - started,
            )
        )
    else:
        events.append(
            ProcessingEvent(
                element_id=element_id,
                stage=stage,
                status="success",
                model_version=model_version,
                wall_clock_ms=now_ms() - started,
            )
        )

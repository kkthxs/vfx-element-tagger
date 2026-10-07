from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProcessingEvent:
    element_id: str
    stage: str
    status: str
    error_kind: str | None = None
    error_message: str | None = None
    model_version: str | None = None
    wall_clock_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceRepresentation:
    representation_id: str
    path: str
    original_path: str
    format: str
    codec: str | None = None
    container: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    bit_depth: int | None = None
    color_space: str | None = None
    is_sequence: bool = False
    sequence_first_frame: int | None = None
    sequence_last_frame: int | None = None
    sequence_frame_padding: int | None = None
    missing_frames: list[int] = field(default_factory=list)
    file_size_bytes: int | None = None
    created_at: str | None = None
    is_primary: bool = False
    linked_by_filename: bool = False
    linked_by_fingerprint: bool = False
    linked_by_phash: bool = False
    link_confidence: float = 0.0
    linked_from_sidecar: bool = False
    content_fingerprint: str | None = None
    perceptual_hash: str | None = None
    filename_root: str | None = None
    # Container for format-specific deterministic truth (EXR channels/windows,
    # deep data, timecode, codec profile, and similar fields).  Keeping this on
    # the representation prevents a caption model from guessing technical data.
    technical_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRepresentation":
        return cls(**data)


@dataclass
class Element:
    element_id: str
    source_representations: list[SourceRepresentation]
    primary_representation_id: str
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: float | None = None
    fps: float | None = None
    color_space: str | None = None
    bit_depth: int | None = None
    has_alpha_channel: bool = False
    alpha_non_empty: bool = False
    alpha_is_binary: bool = False
    alpha_is_soft: bool = False
    premult_status: str = "unknown"
    over_black_likely: bool = False
    keyable_bg_likely: bool = False
    poster_path: str | None = None
    preview_movie_480_path: str | None = None
    preview_movie_1080_path: str | None = None
    artifact_generation_version: str = "fallback-0.1.5"
    is_sequence: bool = False
    has_missing_frames: bool = False
    content_fingerprint: str | None = None
    perceptual_hash: str | None = None
    near_duplicate_group_id: str | None = None
    source_family_id: str | None = None
    category: str = "unknown"
    category_confidence: float = 0.0
    content_facets: dict[str, Any] = field(default_factory=dict)
    motion_facets: dict[str, Any] = field(default_factory=dict)
    measured_features: dict[str, Any] = field(default_factory=dict)
    motion_warnings: list[str] = field(default_factory=list)
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    alpha_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    caption: str = ""
    qwen_tags: dict[str, Any] = field(default_factory=dict)
    captioning_model_version: str = "deterministic-caption-fallback-0.1.5"
    embedding_model_version: str = "deterministic-hash-embedding-0.1.5"
    image_embed: list[float] = field(default_factory=list)
    caption_embed: list[float] = field(default_factory=list)
    # v0.2 semantic layer.  The legacy fields above remain readable while
    # catalogs migrate; new analysis writes the structured result here.
    primary_family: str = "unknown"
    secondary_families: list[str] = field(default_factory=list)
    semantic_analysis: dict[str, Any] = field(default_factory=dict)
    analysis_candidates: list[dict[str, Any]] = field(default_factory=list)
    analysis_confidence: float = 0.0
    analysis_status: str = "unanalysed"
    analysis_model_version: str = ""
    analysis_schema_version: str = ""
    analysis_escalation: dict[str, Any] = field(default_factory=dict)
    human_overrides: dict[str, Any] = field(default_factory=dict)
    video_embed: list[float] = field(default_factory=list)

    def primary(self) -> SourceRepresentation:
        for representation in self.source_representations:
            if representation.representation_id == self.primary_representation_id:
                return representation
        return self.source_representations[0]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_representations"] = [
            representation.to_dict() for representation in self.source_representations
        ]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Element":
        copied = dict(data)
        copied["source_representations"] = [
            SourceRepresentation.from_dict(item)
            for item in copied.get("source_representations", [])
        ]
        return cls(**copied)

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from .semantic import (
    SEMANTIC_SCHEMA_VERSION,
    AnalysisAssessment,
    adjudication_prompt,
    analyses_agree,
    analysis_prompt,
    apply_deterministic_refinements,
    assess_analysis,
    correction_prompt,
    parse_analysis,
)
from .temporal_specialist import (
    build_focused_storyboard,
    build_storyboard,
    detail_specialist_reasons,
    detail_storyboard_prompt,
    impact_storyboard_prompt,
    material_specialist_reasons,
    material_storyboard_prompt,
    merge_detail_specialist,
    merge_material_specialist,
    merge_temporal_specialist,
    parse_detail_specialist,
    parse_material_specialist,
    parse_temporal_specialist,
    specialist_reasons,
)


class VideoModelBackend(Protocol):
    model_id: str

    def analyze(
        self,
        video_path: Path,
        prompt: str,
        fps: float,
        max_frames: int,
        max_tokens: int,
    ) -> str: ...


@dataclass(frozen=True)
class SamplingPlan:
    fps: float
    max_frames: int


@dataclass
class AnalysisCandidate:
    model: str
    role: str
    fps: float
    max_frames: int
    result: dict[str, Any] | None
    raw_excerpt: str
    score: float
    confidence: float
    completeness: float
    accepted: bool
    issues: list[str]
    error: str | None = None
    input_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GatedAnalysisResult:
    analysis: dict[str, Any]
    status: str
    confidence: float
    model_version: str
    schema_version: str
    candidates: list[dict[str, Any]]
    escalation: dict[str, Any]


class GatedSequenceAnalyzer:
    def __init__(
        self,
        primary: VideoModelBackend,
        challengers: list[VideoModelBackend] | None = None,
        confidence_threshold: float = 0.88,
        max_challengers: int = 2,
        max_tokens: int = 1400,
    ):
        self.primary = primary
        self.challengers = challengers or []
        self.confidence_threshold = confidence_threshold
        self.max_challengers = max(0, max_challengers)
        self.max_tokens = max_tokens

    def analyze(
        self,
        video_path: Path,
        duration_seconds: float | None,
        technical_context: dict[str, Any] | None = None,
        activity: dict[str, Any] | None = None,
        detail_video_path: Path | None = None,
    ) -> GatedAnalysisResult:
        base = sampling_plan(duration_seconds)
        dense = dense_sampling_plan(duration_seconds, base)
        ultra = ultra_sampling_plan(duration_seconds, dense)
        candidates: list[AnalysisCandidate] = []

        first = self._run_candidate(
            self.primary,
            "primary",
            video_path,
            analysis_prompt(technical_context),
            base,
            activity,
            technical_context,
        )
        candidates.append(first)
        sampling_escalation_reasons = _activity_sampling_reasons(activity, base)
        if first.accepted and first.result is not None and not sampling_escalation_reasons:
            return self._finalize(
                first,
                candidates,
                "accepted",
                base,
                dense,
                ultra,
                False,
                [],
                False,
                activity,
                technical_context,
                video_path,
                detail_video_path,
            )

        dense_primary = self._run_candidate(
            self.primary,
            "primary_dense",
            video_path,
            correction_prompt(first.result, first.issues, technical_context),
            dense,
            activity,
            technical_context,
        )
        candidates.append(dense_primary)
        reference = dense_primary if dense_primary.result is not None else first
        ultra_reasons = _ultra_sampling_reasons(
            reference,
            activity,
            self.confidence_threshold,
        )
        ultra_triggered = bool(ultra_reasons)
        if ultra_triggered:
            ultra_primary = self._run_candidate(
                self.primary,
                "primary_ultra",
                video_path,
                correction_prompt(
                    reference.result,
                    list(dict.fromkeys(first.issues + dense_primary.issues + ultra_reasons)),
                    technical_context,
                ),
                ultra,
                activity,
                technical_context,
            )
            candidates.append(ultra_primary)
            if ultra_primary.result is not None:
                reference = ultra_primary

        if (
            first.accepted
            and reference.accepted
            and first.result is not None
            and reference.result is not None
            and analyses_agree(first.result, reference.result)
        ):
            selected = max((first, reference), key=lambda candidate: candidate.score)
            return self._finalize(
                selected,
                candidates,
                "escalated",
                base,
                dense,
                ultra,
                False,
                sampling_escalation_reasons + ultra_reasons,
                ultra_triggered,
                activity,
                technical_context,
                video_path,
                detail_video_path,
            )

        for challenger in self.challengers[: self.max_challengers]:
            prompt = analysis_prompt(technical_context) + (
                "\nThis is an independent observation. Judge the supplied pixels without "
                "assuming any previous model's classification, direction or event count."
            )
            candidates.append(
                self._run_candidate(
                    challenger,
                    "challenger",
                    video_path,
                    prompt,
                    ultra if ultra_triggered else dense,
                    activity,
                    technical_context,
                )
            )

        usable = [candidate for candidate in candidates if candidate.result is not None]
        if not usable:
            raise RuntimeError("No analysis model returned parseable structured output.")
        best = max(usable, key=lambda candidate: (candidate.accepted, candidate.score))

        comparisons = [
            candidate
            for candidate in usable
            if candidate is not best and _usable_for_consensus(candidate)
        ]
        disagreement = any(
            not analyses_agree(best.result or {}, candidate.result or {}) for candidate in comparisons
        )
        unstructured_evidence = [
            candidate
            for candidate in candidates
            if candidate.role == "challenger"
            and candidate.result is None
            and bool(candidate.raw_excerpt)
        ]
        needs_adjudication = disagreement or not best.accepted or bool(unstructured_evidence)
        evidence_candidates = [
            candidate for candidate in candidates if candidate.result is not None or candidate.raw_excerpt
        ]
        if needs_adjudication and len(evidence_candidates) >= 2:
            adjudication = self._run_candidate(
                self.primary,
                "adjudicator",
                video_path,
                adjudication_prompt(
                    [candidate.to_dict() for candidate in evidence_candidates], technical_context
                ),
                ultra if ultra_triggered else dense,
                activity,
                technical_context,
            )
            candidates.append(adjudication)
            if adjudication.result is not None and (
                adjudication.accepted or adjudication.score >= best.score
            ):
                best = adjudication

        adjudicated_resolution = best.role == "adjudicator" and best.accepted and any(
            candidate.role == "challenger" and _usable_for_consensus(candidate)
            and analyses_agree(best.result or {}, candidate.result or {}) for candidate in candidates
        )
        status = (
            "escalated"
            if best.accepted and (not disagreement or adjudicated_resolution)
            else "needs_review"
        )
        return self._finalize(
            best,
            candidates,
            status,
            base,
            dense,
            ultra,
            disagreement,
            sampling_escalation_reasons + ultra_reasons,
            ultra_triggered,
            activity,
            technical_context,
            video_path,
            detail_video_path,
        )

    def _run_candidate(
        self,
        backend: VideoModelBackend,
        role: str,
        video_path: Path,
        prompt: str,
        plan: SamplingPlan,
        activity: dict[str, Any] | None,
        technical_context: dict[str, Any] | None,
    ) -> AnalysisCandidate:
        try:
            if hasattr(backend, "sampling_focus"):
                backend.sampling_focus = list(((activity or {}).get("analysis_focus") or {}).get("recommended_timestamps_seconds") or [])
            raw = backend.analyze(
                video_path,
                prompt,
                fps=plan.fps,
                max_frames=plan.max_frames,
                max_tokens=self.max_tokens,
            )
            result = parse_analysis(raw)
            assessment = assess_analysis(
                result,
                activity,
                self.confidence_threshold,
                technical_context,
            )
            candidate = _candidate(backend, role, plan, result, raw, assessment)
            candidate.input_diagnostics = dict(getattr(backend, "last_input_diagnostics", {}))
            return candidate
        except Exception as exc:
            return AnalysisCandidate(
                model=backend.model_id,
                role=role,
                fps=plan.fps,
                max_frames=plan.max_frames,
                result=None,
                raw_excerpt="",
                score=0.0,
                confidence=0.0,
                completeness=0.0,
                accepted=False,
                issues=["backend_error"],
                error=f"{type(exc).__name__}: {exc}",
            )

    def _finalize(
        self,
        selected: AnalysisCandidate,
        candidates: list[AnalysisCandidate],
        status: str,
        base: SamplingPlan,
        dense: SamplingPlan,
        ultra: SamplingPlan,
        disagreement: bool,
        sampling_escalation_reasons: list[str],
        ultra_triggered: bool,
        activity: dict[str, Any] | None,
        technical_context: dict[str, Any] | None,
        video_path: Path,
        detail_video_path: Path | None,
    ) -> GatedAnalysisResult:
        assert selected.result is not None
        refined_analysis, deterministic_refinements = apply_deterministic_refinements(
            selected.result, activity, technical_context
        )
        temporal_record: dict[str, Any] = {"triggered": False}
        reasons = specialist_reasons(refined_analysis, technical_context)
        if reasons and hasattr(self.primary, "analyze_image"):
            try:
                frame_count = 16
                storyboard = build_storyboard(video_path, frame_count=frame_count)
                prompt = impact_storyboard_prompt(refined_analysis, frame_count)
                raw = self.primary.analyze_image(storyboard, prompt, max_tokens=min(900, self.max_tokens))
                specialist = parse_temporal_specialist(raw)
                if specialist is not None:
                    refined_analysis, specialist_refinements = merge_temporal_specialist(
                        refined_analysis, specialist
                    )
                else:
                    specialist_refinements = []
                temporal_record = {
                    "triggered": True,
                    "reasons": reasons,
                    "parseable": specialist is not None,
                    "result": specialist,
                    "refinements": specialist_refinements,
                    "raw_excerpt": raw[:4000],
                }
                if specialist_refinements and status == "accepted":
                    status = "escalated"
            except Exception as exc:
                temporal_record = {
                    "triggered": True,
                    "reasons": reasons,
                    "parseable": False,
                    "refinements": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
        detail_record: dict[str, Any] = {"triggered": False}
        detail_reasons = detail_specialist_reasons(refined_analysis, activity)
        if detail_reasons and hasattr(self.primary, "analyze_image"):
            detail_input = detail_video_path or video_path
            try:
                frame_count = 16
                storyboard, storyboard_metadata = build_focused_storyboard(
                    detail_input,
                    activity,
                    frame_count=frame_count,
                )
                prompt = detail_storyboard_prompt(
                    refined_analysis,
                    frame_count,
                    storyboard_metadata.get("timestamps_seconds") or [],
                )
                raw = self.primary.analyze_image(
                    storyboard,
                    prompt,
                    max_tokens=min(900, self.max_tokens),
                )
                specialist = parse_detail_specialist(raw)
                if specialist is not None:
                    refined_analysis, detail_refinements = merge_detail_specialist(
                        refined_analysis,
                        specialist,
                        allow_timing_expansion=bool((activity or {}).get("proxy_information_limited")),
                        source_fps=(activity or {}).get("source_fps"),
                        source_frame_count=(activity or {}).get("source_frame_count"),
                        duration_seconds=(activity or {}).get("duration_seconds"),
                    )
                else:
                    detail_refinements = []
                detail_record = {
                    "triggered": True,
                    "reasons": detail_reasons,
                    "input": "source_or_highest_resolution"
                    if detail_input.resolve() != video_path.resolve()
                    else "semantic_video",
                    "storyboard": storyboard_metadata,
                    "parseable": specialist is not None,
                    "result": specialist,
                    "refinements": detail_refinements,
                    "raw_excerpt": raw[:4000],
                }
                if detail_refinements and status == "accepted":
                    status = "escalated"
            except Exception as exc:
                detail_record = {
                    "triggered": True,
                    "reasons": detail_reasons,
                    "parseable": False,
                    "refinements": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
        material_record: dict[str, Any] = {"triggered": False}
        material_reasons = material_specialist_reasons(
            refined_analysis,
            activity,
            candidate_disagreement=disagreement,
        )
        if material_reasons and hasattr(self.primary, "analyze_image"):
            material_input = detail_video_path or video_path
            try:
                frame_count = 16
                storyboard, storyboard_metadata = build_focused_storyboard(
                    material_input,
                    activity,
                    frame_count=frame_count,
                )
                raw = self.primary.analyze_image(
                    storyboard,
                    material_storyboard_prompt(
                        refined_analysis,
                        frame_count,
                        storyboard_metadata.get("timestamps_seconds") or [],
                    ),
                    max_tokens=min(800, self.max_tokens),
                )
                specialist = parse_material_specialist(raw)
                if specialist is not None:
                    refined_analysis, material_refinements = merge_material_specialist(
                        refined_analysis,
                        specialist,
                    )
                else:
                    material_refinements = []
                resolved = bool(material_refinements)
                material_record = {
                    "triggered": True,
                    "reasons": material_reasons,
                    "input": "source_or_highest_resolution"
                    if material_input.resolve() != video_path.resolve()
                    else "semantic_video",
                    "storyboard": storyboard_metadata,
                    "parseable": specialist is not None,
                    "resolved": resolved,
                    "result": specialist,
                    "refinements": material_refinements,
                    "raw_excerpt": raw[:4000],
                }
                if resolved and status == "accepted":
                    status = "escalated"
                elif not resolved:
                    status = "needs_review"
            except Exception as exc:
                status = "needs_review"
                material_record = {
                    "triggered": True,
                    "reasons": material_reasons,
                    "parseable": False,
                    "resolved": False,
                    "refinements": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
        final_assessment = assess_analysis(
            refined_analysis,
            activity,
            self.confidence_threshold,
            technical_context,
        )
        if not final_assessment.accepted:
            status = "needs_review"
        if (activity or {}).get("proxy_information_limited"):
            corroborating_models = {candidate.model for candidate in candidates
                                    if candidate.model != selected.model and _usable_for_consensus(candidate)
                                    and analyses_agree(refined_analysis, candidate.result or {})}
            if not corroborating_models:
                status = "needs_review"
                final_assessment = replace(final_assessment, accepted=False,
                    issues=final_assessment.issues + ["low_information_without_independent_corroboration"])
        if final_assessment.accepted and status == "needs_review" and (
            not disagreement
        ) and not (
            material_record.get("triggered") and not material_record.get("resolved")
        ):
            status = "escalated"
        attempted_challengers = [
            candidate.model for candidate in candidates if candidate.role == "challenger"
        ]
        return GatedAnalysisResult(
            analysis=refined_analysis,
            status=status,
            confidence=selected.confidence,
            model_version=selected.model,
            schema_version=SEMANTIC_SCHEMA_VERSION,
            candidates=[candidate.to_dict() for candidate in candidates],
            escalation={
                "triggered": (
                    len(candidates) > 1
                    or bool(temporal_record.get("triggered"))
                    or bool(detail_record.get("triggered"))
                    or bool(material_record.get("triggered"))
                ),
                "initial_issues": candidates[0].issues,
                "base_sampling": asdict(base),
                "dense_sampling": asdict(dense),
                "ultra_sampling": asdict(ultra),
                "sampling_escalation_reasons": list(dict.fromkeys(sampling_escalation_reasons)),
                "ultra_triggered": ultra_triggered,
                "challengers_attempted": attempted_challengers,
                "candidate_disagreement": disagreement,
                "selected_role": selected.role,
                "selected_model": selected.model,
                "deterministic_refinements": deterministic_refinements,
                "temporal_specialist": temporal_record,
                "high_resolution_detail_specialist": detail_record,
                "material_specialist": material_record,
                "final_assessment": asdict(final_assessment),
            },
        )


def sampling_plan(duration_seconds: float | None) -> SamplingPlan:
    duration = duration_seconds or 0.0
    if 0 < duration <= 5.0:
        return SamplingPlan(8.0, 48)
    if 0 < duration <= 15.0:
        return SamplingPlan(4.0, 64)
    return SamplingPlan(2.0, 96)


def dense_sampling_plan(
    duration_seconds: float | None,
    base: SamplingPlan | None = None,
) -> SamplingPlan:
    base = base or sampling_plan(duration_seconds)
    target_fps = min(12.0, max(4.0, base.fps * 2.0))
    target_frames = min(160, max(base.max_frames + 24, base.max_frames * 2))
    return SamplingPlan(target_fps, target_frames)


def ultra_sampling_plan(
    duration_seconds: float | None,
    dense: SamplingPlan | None = None,
) -> SamplingPlan:
    dense = dense or dense_sampling_plan(duration_seconds)
    target_fps = min(24.0, max(8.0, dense.fps * 2.0))
    target_frames = min(256, max(dense.max_frames + 48, int(dense.max_frames * 1.5)))
    return SamplingPlan(target_fps, target_frames)


def _usable_for_consensus(candidate: AnalysisCandidate) -> bool:
    """Ignore structurally parsed but substantively empty challenger output."""

    return bool(
        candidate.result is not None
        and candidate.confidence >= 0.5
        and candidate.completeness >= 0.72
        and candidate.score > 0.0
    )


def _activity_sampling_reasons(
    activity: dict[str, Any] | None,
    base: SamplingPlan,
) -> list[str]:
    activity = activity or {}
    timing = activity.get("action_timing")
    if not isinstance(timing, dict) or not timing.get("available"):
        return []
    duration = float(activity.get("duration_seconds") or 0.0)
    coverage = float(timing.get("coverage_fraction") or 1.0)
    optimal_duration = float(timing.get("optimal_duration_seconds") or duration)
    reasons = []
    if (
        duration >= 12.0
        and coverage <= 0.08
        and optimal_duration <= max(1.5, duration * 0.12)
    ):
        reasons.append("activity_focus:narrow_action_window")
    events = timing.get("event_segments") or []
    if duration >= 12.0 and len(events) >= 3 and base.fps <= 2.0:
        reasons.append("activity_focus:multiple_brief_events")
    return reasons


def _ultra_sampling_reasons(
    candidate: AnalysisCandidate,
    activity: dict[str, Any] | None,
    confidence_threshold: float,
) -> list[str]:
    reasons = []
    if candidate.result is None:
        reasons.append("ultra_sampling:unparseable_dense_result")
        return reasons
    if candidate.confidence < max(0.50, confidence_threshold - 0.25):
        reasons.append("ultra_sampling:very_low_dense_confidence")
    if candidate.completeness < 0.60:
        reasons.append("ultra_sampling:very_low_dense_completeness")
    timing = (activity or {}).get("action_timing")
    duration = float((activity or {}).get("duration_seconds") or 0.0)
    if (
        "insufficient_temporal_evidence" in candidate.issues
        and isinstance(timing, dict)
        and timing.get("available")
        and duration >= 12.0
        and float(timing.get("coverage_fraction") or 1.0) <= 0.10
    ):
        reasons.append("ultra_sampling:short_action_missing_temporal_evidence")
    return reasons


def _candidate(
    backend: VideoModelBackend,
    role: str,
    plan: SamplingPlan,
    result: dict[str, Any] | None,
    raw: str,
    assessment: AnalysisAssessment,
) -> AnalysisCandidate:
    return AnalysisCandidate(
        model=backend.model_id,
        role=role,
        fps=plan.fps,
        max_frames=plan.max_frames,
        result=result,
        raw_excerpt=raw[:6000],
        score=assessment.score,
        confidence=assessment.confidence,
        completeness=assessment.completeness,
        accepted=assessment.accepted,
        issues=assessment.issues,
    )

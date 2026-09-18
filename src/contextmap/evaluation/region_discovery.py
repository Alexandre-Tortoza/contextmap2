"""Deterministic Region Discovery quality and performance evaluation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TypeAlias
from uuid import uuid4

from contextmap.visual_perception import (
    BackendDiagnostics,
    DiscoveryRunResult,
    InlineMask,
    NormalizationResult,
    PreparedImage,
    Region2D,
)

EvaluationScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class GroundTruthRegion:
    """Annotated region geometry used only for objective evaluation."""

    region_id: str
    mask: InlineMask

    def __post_init__(self) -> None:
        """Require stable annotation identity and non-empty geometry."""
        if not self.region_id:
            raise ValueError("ground-truth region_id must not be empty")
        if self.mask.area == 0:
            raise ValueError("ground-truth mask must contain foreground pixels")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible annotation record."""
        return {"region_id": self.region_id, "mask": self.mask.to_dict()}


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    """One versioned frame, source condition, constraints, and optional annotation."""

    frame_id: str
    source_condition: str
    prepared_image: PreparedImage
    annotations: tuple[GroundTruthRegion, ...] = ()

    def __post_init__(self) -> None:
        """Validate identity and annotation image space."""
        if not self.frame_id or not self.source_condition:
            raise ValueError("reference frame id and source condition must not be empty")
        if self.prepared_image.source_observation_id != self.frame_id:
            raise ValueError("reference frame id must match prepared image observation id")
        annotation_ids = [annotation.region_id for annotation in self.annotations]
        if len(set(annotation_ids)) != len(annotation_ids):
            raise ValueError("ground-truth region ids must be unique per frame")
        for annotation in self.annotations:
            if (annotation.mask.width, annotation.mask.height) != (
                self.prepared_image.width,
                self.prepared_image.height,
            ):
                raise ValueError("ground-truth masks must match prepared image dimensions")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible reference-frame manifest entry."""
        return {
            "frame_id": self.frame_id,
            "source_condition": self.source_condition,
            "prepared_image": self.prepared_image.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self.annotations],
        }


@dataclass(frozen=True, slots=True)
class RegionDiscoveryReferenceSet:
    """Exact ordered frame selection shared by every evaluated configuration."""

    version: str
    frames: tuple[ReferenceFrame, ...]

    def __post_init__(self) -> None:
        """Require versioned, non-empty, unique frame selection."""
        if not self.version:
            raise ValueError("reference-set version must not be empty")
        if not self.frames:
            raise ValueError("reference set must contain at least one frame")
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("reference frame ids must be unique")

    def to_dict(self) -> dict[str, object]:
        """Return the versioned exact frame-selection manifest."""
        return {
            "schema": "contextmap.region-discovery-reference-set/v1",
            "version": self.version,
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(frozen=True, slots=True)
class EvaluationRunDescriptor:
    """Reproducibility identity for one backend/configuration evaluation run."""

    perception_run_id: str
    perception_artifact_id: str
    backend_id: str
    backend_version: str
    checkpoint: str
    config_digest: str
    pipeline_graph_digest: str
    strategy: str
    thresholds: tuple[tuple[str, float], ...]
    variables: tuple[tuple[str, EvaluationScalar], ...]
    execution_kind: str

    def __post_init__(self) -> None:
        """Validate complete run identity and comparable variable names."""
        required = (
            self.perception_run_id,
            self.perception_artifact_id,
            self.backend_id,
            self.backend_version,
            self.checkpoint,
            self.config_digest,
            self.pipeline_graph_digest,
            self.strategy,
            self.execution_kind,
        )
        if any(not value for value in required):
            raise ValueError("evaluation run identity fields must not be empty")
        for name, values in (
            ("threshold", [key for key, _ in self.thresholds]),
            ("variable", [key for key, _ in self.variables]),
        ):
            if any(not key for key in values) or len(set(values)) != len(values):
                raise ValueError(f"evaluation {name} names must be non-empty and unique")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible reproducibility descriptor."""
        return {
            "perception_run_id": self.perception_run_id,
            "perception_artifact_id": self.perception_artifact_id,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "checkpoint": self.checkpoint,
            "config_digest": self.config_digest,
            "pipeline_graph_digest": self.pipeline_graph_digest,
            "strategy": self.strategy,
            "thresholds": dict(self.thresholds),
            "variables": dict(self.variables),
            "execution_kind": self.execution_kind,
        }


@dataclass(frozen=True, slots=True)
class EvaluatedDiscoveryFrame:
    """Canonical discovery and normalization outputs measured for one frame."""

    discovery: DiscoveryRunResult
    normalization: NormalizationResult


@dataclass(frozen=True, slots=True)
class SegmentationAccuracy:
    """Ground-truth-dependent region geometry metrics for one annotated frame."""

    mean_iou: float
    mean_dice: float
    region_recall: float
    coverage: float
    over_segmentation_rate: float
    under_segmentation_rate: float
    duplicate_region_rate: float

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible accuracy metrics."""
        return {
            "mean_iou": self.mean_iou,
            "mean_dice": self.mean_dice,
            "region_recall": self.region_recall,
            "coverage": self.coverage,
            "over_segmentation_rate": self.over_segmentation_rate,
            "under_segmentation_rate": self.under_segmentation_rate,
            "duplicate_region_rate": self.duplicate_region_rate,
        }


@dataclass(frozen=True, slots=True)
class DiscoveryDiagnosticsMetrics:
    """Ground-truth-independent candidate and geometry diagnostics."""

    raw_candidate_count: int
    accepted_region_count: int
    rejected_candidate_count: int
    duplicate_merge_ratio: float
    area_pixels: tuple[float, ...]
    invalid_geometry_count: int
    constraint_violation_count: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible diagnostic metrics."""
        return {
            "raw_candidate_count": self.raw_candidate_count,
            "accepted_region_count": self.accepted_region_count,
            "rejected_candidate_count": self.rejected_candidate_count,
            "duplicate_merge_ratio": self.duplicate_merge_ratio,
            "area_pixels": list(self.area_pixels),
            "invalid_geometry_count": self.invalid_geometry_count,
            "constraint_violation_count": self.constraint_violation_count,
        }


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Runtime and memory dimensions kept separate from discovery quality."""

    runtime_ms: float
    per_pass_runtime_ms: tuple[float, ...]
    peak_memory_mb: float | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible performance metrics."""
        return {
            "runtime_ms": self.runtime_ms,
            "per_pass_runtime_ms": list(self.per_pass_runtime_ms),
            "peak_memory_mb": self.peak_memory_mb,
        }


@dataclass(frozen=True, slots=True)
class FrameEvaluation:
    """All metrics for one exact reference frame."""

    frame_id: str
    source_condition: str
    diagnostics: DiscoveryDiagnosticsMetrics
    performance: PerformanceMetrics
    accuracy: SegmentationAccuracy | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible per-frame report."""
        return {
            "frame_id": self.frame_id,
            "source_condition": self.source_condition,
            "diagnostics": self.diagnostics.to_dict(),
            "performance": self.performance.to_dict(),
            "accuracy": None if self.accuracy is None else self.accuracy.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RegionDiscoveryEvaluationReport:
    """Versioned comparable report for one Region Discovery configuration."""

    schema: str
    metric_schema_version: str
    reference_set_version: str
    frame_selection: tuple[str, ...]
    run: EvaluationRunDescriptor
    frames: tuple[FrameEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable persisted report schema."""
        return {
            "schema": self.schema,
            "metric_schema_version": self.metric_schema_version,
            "reference_set_version": self.reference_set_version,
            "frame_selection": list(self.frame_selection),
            "run": self.run.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
            "summary": _report_summary(self),
        }


@dataclass(frozen=True, slots=True)
class RegionDiscoveryComparison:
    """Controlled ablation deltas with quality and performance separated."""

    changed_variable: str
    quality_deltas: dict[str, float]
    performance_deltas: dict[str, float]


class RegionDiscoveryEvaluator:
    """Evaluate any backend/configuration through one reference-frame protocol."""

    def __init__(self, *, match_iou_threshold: float = 0.5) -> None:
        """Configure the documented ground-truth matching threshold."""
        if not 0 <= match_iou_threshold <= 1:
            raise ValueError("match_iou_threshold must be between zero and one")
        self._match_iou_threshold = match_iou_threshold

    def evaluate(
        self,
        reference_set: RegionDiscoveryReferenceSet,
        run: EvaluationRunDescriptor,
        execute: Callable[[ReferenceFrame], EvaluatedDiscoveryFrame],
    ) -> RegionDiscoveryEvaluationReport:
        """Execute the same ordered frame selection through a supplied pipeline."""
        frames: list[FrameEvaluation] = []
        for reference_frame in reference_set.frames:
            evaluated = execute(reference_frame)
            _validate_evaluated_frame(reference_frame, evaluated)
            diagnostics = _diagnostic_metrics(evaluated)
            performance = _performance_metrics(evaluated.discovery.diagnostics)
            accuracy = None
            if reference_frame.annotations:
                accuracy = _accuracy_metrics(
                    evaluated.normalization.regions,
                    reference_frame.annotations,
                    self._match_iou_threshold,
                )
            frames.append(
                FrameEvaluation(
                    frame_id=reference_frame.frame_id,
                    source_condition=reference_frame.source_condition,
                    diagnostics=diagnostics,
                    performance=performance,
                    accuracy=accuracy,
                )
            )
        return RegionDiscoveryEvaluationReport(
            schema="contextmap.region-discovery-evaluation/v1",
            metric_schema_version="1.0.0",
            reference_set_version=reference_set.version,
            frame_selection=tuple(frame.frame_id for frame in reference_set.frames),
            run=run,
            frames=tuple(frames),
        )


def compare_region_discovery_reports(
    baseline: RegionDiscoveryEvaluationReport,
    changed: RegionDiscoveryEvaluationReport,
    *,
    changed_variable: str,
) -> RegionDiscoveryComparison:
    """Compare a controlled one-variable ablation using the same reference frames."""
    if (
        baseline.reference_set_version != changed.reference_set_version
        or baseline.frame_selection != changed.frame_selection
    ):
        raise ValueError("ablation reports must use the same reference-set version and frames")
    baseline_variables = dict(baseline.run.variables)
    changed_variables = dict(changed.run.variables)
    differing = {
        key
        for key in baseline_variables.keys() | changed_variables.keys()
        if baseline_variables.get(key) != changed_variables.get(key)
    }
    if differing != {changed_variable}:
        raise ValueError("ablation must change exactly the declared variable")
    baseline_summary = _report_summary(baseline)
    changed_summary = _report_summary(changed)
    return RegionDiscoveryComparison(
        changed_variable=changed_variable,
        quality_deltas={
            "mean_iou": _difference(changed_summary, baseline_summary, "mean_iou"),
            "mean_region_recall": _difference(
                changed_summary, baseline_summary, "mean_region_recall"
            ),
        },
        performance_deltas={
            "mean_runtime_ms": _difference(changed_summary, baseline_summary, "mean_runtime_ms"),
            "peak_memory_mb": _difference(changed_summary, baseline_summary, "peak_memory_mb"),
        },
    )


def write_region_discovery_report(path: Path, report: RegionDiscoveryEvaluationReport) -> None:
    """Atomically persist a report without overwriting an existing evaluation."""
    _write_immutable_json(path, report.to_dict(), "evaluation report")


def write_region_discovery_reference_set(
    path: Path, reference_set: RegionDiscoveryReferenceSet
) -> None:
    """Atomically persist an immutable reference-frame selection manifest."""
    _write_immutable_json(path, reference_set.to_dict(), "reference-set manifest")


def _validate_evaluated_frame(
    reference_frame: ReferenceFrame, evaluated: EvaluatedDiscoveryFrame
) -> None:
    for region in evaluated.normalization.regions:
        if region.source_observation_id != reference_frame.frame_id:
            raise ValueError("evaluated region source does not match reference frame")
        if (region.image_width, region.image_height) != (
            reference_frame.prepared_image.width,
            reference_frame.prepared_image.height,
        ):
            raise ValueError("evaluated region dimensions do not match reference frame")


def _diagnostic_metrics(evaluated: EvaluatedDiscoveryFrame) -> DiscoveryDiagnosticsMetrics:
    raw_count = sum(_raw_proposal_count(item) for item in evaluated.discovery.diagnostics)
    rejections = (*evaluated.discovery.rejected, *evaluated.normalization.rejected)
    return DiscoveryDiagnosticsMetrics(
        raw_candidate_count=raw_count,
        accepted_region_count=len(evaluated.normalization.regions),
        rejected_candidate_count=len(rejections),
        duplicate_merge_ratio=(
            len(evaluated.normalization.merge_decisions) / raw_count if raw_count else 0.0
        ),
        area_pixels=tuple(region.area_pixels for region in evaluated.normalization.regions),
        invalid_geometry_count=sum(
            rejection.reason.value == "invalid_geometry" for rejection in rejections
        ),
        constraint_violation_count=sum(
            rejection.reason.value in {"outside_valid_region", "exclusion_overlap"}
            for rejection in rejections
        ),
    )


def _performance_metrics(
    diagnostics: tuple[BackendDiagnostics, ...],
) -> PerformanceMetrics:
    runtimes = tuple(item.duration_ms for item in diagnostics)
    memory = [
        float(value)
        for item in diagnostics
        for key, value in item.metadata
        if key == "peak_memory_mb"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]
    return PerformanceMetrics(
        runtime_ms=sum(runtimes),
        per_pass_runtime_ms=runtimes,
        peak_memory_mb=max(memory) if memory else None,
    )


def _raw_proposal_count(diagnostics: BackendDiagnostics) -> int:
    for key, value in diagnostics.metadata:
        if key == "raw_proposal_count" and type(value) is int:
            return int(value)
    return diagnostics.proposal_count


def _accuracy_metrics(
    regions: tuple[Region2D, ...],
    annotations: tuple[GroundTruthRegion, ...],
    match_threshold: float,
) -> SegmentationAccuracy:
    predicted = [_region_pixels(region) for region in regions]
    expected = [_mask_pixels(annotation.mask) for annotation in annotations]
    pairwise_iou = [[_iou(prediction, target) for target in expected] for prediction in predicted]
    best_iou_by_target = [
        max((pairwise_iou[prediction][target] for prediction in range(len(predicted))), default=0.0)
        for target in range(len(expected))
    ]
    best_dice_by_target = [
        max(
            (
                _dice(predicted[prediction], expected[target])
                for prediction in range(len(predicted))
            ),
            default=0.0,
        )
        for target in range(len(expected))
    ]
    predicted_union = set().union(*predicted) if predicted else set()
    expected_union = set().union(*expected)
    assignments = [
        max(range(len(expected)), key=lambda target: pairwise_iou[index][target])
        for index in range(len(predicted))
        if expected and max(pairwise_iou[index], default=0.0) > 0
    ]
    duplicate_count = len(assignments) - len(set(assignments))
    over_segmented = sum(
        max(0, sum(_iou(prediction, target) > 0 for prediction in predicted) - 1)
        for target in expected
    )
    under_segmented = sum(
        sum(_iou(prediction, target) > 0 for target in expected) > 1 for prediction in predicted
    )
    return SegmentationAccuracy(
        mean_iou=fmean(best_iou_by_target),
        mean_dice=fmean(best_dice_by_target),
        region_recall=sum(value >= match_threshold for value in best_iou_by_target) / len(expected),
        coverage=(
            len(predicted_union & expected_union) / len(expected_union) if expected_union else 0.0
        ),
        over_segmentation_rate=over_segmented / len(predicted) if predicted else 0.0,
        under_segmentation_rate=under_segmented / len(predicted) if predicted else 0.0,
        duplicate_region_rate=duplicate_count / len(predicted) if predicted else 0.0,
    )


def _region_pixels(region: Region2D) -> set[tuple[int, int]]:
    if isinstance(region.mask, InlineMask):
        return _mask_pixels(region.mask)
    return {
        (x, y)
        for y in range(region.image_height)
        for x in range(region.image_width)
        if region.bounding_box.x_min <= x + 0.5 < region.bounding_box.x_max
        and region.bounding_box.y_min <= y + 0.5 < region.bounding_box.y_max
    }


def _mask_pixels(mask: InlineMask) -> set[tuple[int, int]]:
    return {(x, y) for y in range(mask.height) for x in range(mask.width) if mask.value_at(x, y)}


def _iou(first: set[tuple[int, int]], second: set[tuple[int, int]]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _dice(first: set[tuple[int, int]], second: set[tuple[int, int]]) -> float:
    denominator = len(first) + len(second)
    return 2 * len(first & second) / denominator if denominator else 0.0


def _report_summary(report: RegionDiscoveryEvaluationReport) -> dict[str, float | None]:
    annotated = [frame.accuracy for frame in report.frames if frame.accuracy is not None]
    runtimes = [frame.performance.runtime_ms for frame in report.frames]
    memory = [
        frame.performance.peak_memory_mb
        for frame in report.frames
        if frame.performance.peak_memory_mb is not None
    ]
    return {
        "mean_iou": fmean(item.mean_iou for item in annotated) if annotated else None,
        "mean_region_recall": (
            fmean(item.region_recall for item in annotated) if annotated else None
        ),
        "mean_runtime_ms": fmean(runtimes),
        "peak_memory_mb": max(memory) if memory else None,
    }


def _difference(
    changed: dict[str, float | None], baseline: dict[str, float | None], key: str
) -> float:
    changed_value = changed[key]
    baseline_value = baseline[key]
    if changed_value is None or baseline_value is None:
        raise ValueError(f"cannot compare unavailable metric {key}")
    return changed_value - baseline_value


def _write_immutable_json(path: Path, value: object, artifact_name: str) -> None:
    if path.exists():
        raise FileExistsError(f"{artifact_name} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"{artifact_name} already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)

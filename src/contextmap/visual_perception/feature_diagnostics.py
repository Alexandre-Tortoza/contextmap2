"""Structured Feature Extraction diagnostics and debug-artifact writer.

Required timing/status metrics are always written under ``metrics/`` when a
diagnostic exists. Human-oriented metadata and previews are written under
``debug/30-feature-extraction/`` according to an explicit debug level and are
never required to load contractual feature payloads.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.embedding_space import (
    EmbeddingSpace,
    embedding_space_fingerprint,
    encode_embedding_space,
)
from contextmap.visual_perception.models import (
    BackendProvenance,
    BoundingBox2D,
    FeatureId,
    FeatureScope,
    RegionId,
)
from contextmap.visual_perception.serialization import encode_provenance

FEATURE_METRICS_PATH = "metrics/feature-extraction.jsonl"
"""Run-relative path for required Feature Extraction metrics."""

FEATURE_DEBUG_ROOT = "debug/30-feature-extraction"
"""Run-relative root for non-contractual Feature Extraction diagnostics."""


class FeatureDebugLevel(Enum):
    """Amount of non-contractual Feature Extraction debug evidence to persist."""

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


class FeatureEventStatus(Enum):
    """Outcome of one feature extraction attempt."""

    SUCCEEDED = "succeeded"
    WARNING = "warning"
    FAILED = "failed"
    ABSTAINED = "abstained"


@dataclass(frozen=True, kw_only=True)
class DenseFeatureDiagnostic:
    """Spatial metadata for one dense feature map.

    Attributes:
        source_artifact_id: Artifact/run owning the source feature.
        grid_width: Feature-grid width.
        grid_height: Feature-grid height.
        stride_x: Horizontal sampling stride in prepared-image pixels.
        stride_y: Vertical sampling stride in prepared-image pixels.
        support_width: Horizontal support per cell in image pixels.
        support_height: Vertical support per cell in image pixels.
        coordinate_transform_id: Identity of the image-to-grid transform.
    """

    source_artifact_id: str
    grid_width: int
    grid_height: int
    stride_x: float
    stride_y: float
    support_width: float
    support_height: float
    coordinate_transform_id: str

    def __post_init__(self) -> None:
        """Validate dense spatial metadata."""
        if not self.source_artifact_id or not self.coordinate_transform_id:
            raise ValueError("dense artifact and coordinate transform identities are required")
        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError("dense grid dimensions must be positive")
        if min(self.stride_x, self.stride_y, self.support_width, self.support_height) <= 0:
            raise ValueError("dense stride and support dimensions must be positive")


@dataclass(frozen=True, kw_only=True)
class RegionFeatureDiagnostic:
    """Support/view/mask/pooling metadata for one region feature.

    Attributes:
        region_id: Frozen source region.
        bounding_box: Original frozen region box.
        support_box: Actual crop/view support, when distinct.
        source_mask_reference: Original mask payload reference.
        mask_content_hash: Hash of the decoded mask used.
        coordinate_transform_id: Identity of crop/mask/pooling transform.
        pooling_policy: Versioned pooling method, when pooling was used.
        contributing_cell_count: Dense cells contributing to pooling.
        total_weight: Pooling support weight.
        coverage_fraction: Fraction of requested support covered.
    """

    region_id: RegionId
    bounding_box: BoundingBox2D
    support_box: BoundingBox2D | None = None
    source_mask_reference: str | None = None
    mask_content_hash: str | None = None
    coordinate_transform_id: str | None = None
    pooling_policy: str | None = None
    contributing_cell_count: int | None = None
    total_weight: int | None = None
    coverage_fraction: float | None = None

    def __post_init__(self) -> None:
        """Validate optional pooling statistics as one coherent group."""
        pooling_values = (
            self.pooling_policy,
            self.contributing_cell_count,
            self.total_weight,
            self.coverage_fraction,
        )
        if any(value is not None for value in pooling_values) and not all(
            value is not None for value in pooling_values
        ):
            raise ValueError("pooling diagnostics must be all present or all absent")
        if self.contributing_cell_count is not None and self.contributing_cell_count < 0:
            raise ValueError("contributing_cell_count must be non-negative")
        if self.total_weight is not None and self.total_weight < 0:
            raise ValueError("total_weight must be non-negative")
        if self.coverage_fraction is not None and not 0.0 <= self.coverage_fraction <= 1.0:
            raise ValueError("coverage_fraction must be in [0, 1]")


@dataclass(frozen=True, kw_only=True)
class FeatureExtractionDiagnostic:
    """Common auditable record for dense, global, or region extraction.

    Attributes:
        event_id: Stable identity of this extraction event.
        source_observation_id: Physical observation being processed.
        source_prepared_image_reference: Prepared RGB payload reference.
        source_image_width: Prepared-image width.
        source_image_height: Prepared-image height.
        stage_id: Pipeline stage producing/attempting the feature.
        status: Success, warning, failure, or abstention.
        backend: Exact backend/model/configuration provenance.
        duration_seconds: Measured wall time when available.
        peak_memory_bytes: Peak measured memory when available.
        warnings: Non-fatal warning messages.
        failure_reason: Failure/abstention reason; absent for successful events.
        feature_id: Produced feature identity, absent when none was produced.
        scope: Produced feature scope.
        embedding_space: Full embedding compatibility identity.
        output_shape: Numerical payload shape.
        dtype: Numerical payload dtype.
        normalization: Applied output normalization.
        payload_reference: Contractual payload/index reference.
        preprocessing: Ordered preprocessing/view operations.
        dense: Dense spatial details for ``DENSE`` scope.
        region: Region support details for ``REGION`` scope.
    """

    event_id: str
    source_observation_id: SourceObservationId
    source_prepared_image_reference: str
    source_image_width: int
    source_image_height: int
    stage_id: str
    status: FeatureEventStatus
    backend: BackendProvenance
    duration_seconds: float | None = None
    peak_memory_bytes: int | None = None
    warnings: Sequence[str] = ()
    failure_reason: str | None = None
    feature_id: FeatureId | None = None
    scope: FeatureScope | None = None
    embedding_space: EmbeddingSpace | None = None
    output_shape: tuple[int, ...] | None = None
    dtype: str | None = None
    normalization: str | None = None
    payload_reference: str | None = None
    preprocessing: Sequence[str] = ()
    dense: DenseFeatureDiagnostic | None = None
    region: RegionFeatureDiagnostic | None = None

    def __post_init__(self) -> None:
        """Validate outcome and scope-specific metadata consistency."""
        if not self.event_id or not self.stage_id or not self.source_prepared_image_reference:
            raise ValueError("event_id, stage_id, and prepared image reference are required")
        if self.source_image_width <= 0 or self.source_image_height <= 0:
            raise ValueError("source image dimensions must be positive")
        if self.duration_seconds is not None and self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be non-negative")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be non-negative")
        if self.status in {FeatureEventStatus.FAILED, FeatureEventStatus.ABSTAINED}:
            if not self.failure_reason:
                raise ValueError("failure_reason is required for failed/abstained events")
        elif self.failure_reason is not None:
            raise ValueError("failure_reason is only valid for failed/abstained events")

        produced = self.status in {FeatureEventStatus.SUCCEEDED, FeatureEventStatus.WARNING}
        produced_fields = (
            self.feature_id,
            self.scope,
            self.embedding_space,
            self.output_shape,
            self.dtype,
            self.payload_reference,
        )
        if produced and any(value is None for value in produced_fields):
            raise ValueError("successful/warning events require complete feature metadata")
        if self.output_shape is not None and (
            not self.output_shape or any(dimension <= 0 for dimension in self.output_shape)
        ):
            raise ValueError("output_shape dimensions must be positive")
        if self.scope is FeatureScope.DENSE and (self.dense is None or self.region is not None):
            raise ValueError("DENSE diagnostics require dense metadata only")
        if self.scope is FeatureScope.REGION and (self.region is None or self.dense is not None):
            raise ValueError("REGION diagnostics require region metadata only")
        if self.scope is FeatureScope.GLOBAL and (
            self.dense is not None or self.region is not None
        ):
            raise ValueError("GLOBAL diagnostics cannot carry dense/region metadata")


@dataclass(frozen=True, kw_only=True)
class FeatureDiagnosticPreview:
    """Small human-only preview to store under the feature debug root.

    Attributes:
        relative_path: Descriptive path below ``30-feature-extraction``.
        content: Encoded preview bytes; never a contractual payload.
        minimum_level: Minimum debug level that writes this preview.
    """

    relative_path: str
    content: bytes
    minimum_level: FeatureDebugLevel

    def __post_init__(self) -> None:
        """Reject unsafe paths, disabled previews, and unexpectedly large content."""
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("relative_path must stay within the feature debug root")
        if path.parts[0] not in {"dense", "regions", "diagnostics"}:
            raise ValueError("relative_path must begin with dense/, regions/, or diagnostics/")
        if self.minimum_level is FeatureDebugLevel.NONE:
            raise ValueError("preview minimum_level cannot be NONE")
        if len(self.content) > 5 * 1024 * 1024:
            raise ValueError("debug preview must not exceed 5 MiB")


def write_feature_diagnostics(
    *,
    run_root: Path,
    diagnostics: Sequence[FeatureExtractionDiagnostic],
    previews: Sequence[FeatureDiagnosticPreview],
    debug_level: FeatureDebugLevel,
) -> Sequence[str]:
    """Write required metrics and level-controlled debug evidence.

    Returns:
        Every created path relative to ``run_root``.
    """
    if not diagnostics and not previews:
        return ()
    written: list[str] = []
    metrics_content = "".join(
        f"{json.dumps(_encode_metric(item), sort_keys=True)}\n" for item in diagnostics
    )
    _write_text(run_root, FEATURE_METRICS_PATH, metrics_content)
    written.append(FEATURE_METRICS_PATH)
    if debug_level is FeatureDebugLevel.NONE:
        return tuple(written)

    encoded = [_encode_diagnostic(item) for item in diagnostics]
    summary_path = f"{FEATURE_DEBUG_ROOT}/feature-summary.json"
    _write_json(run_root, summary_path, _summary(diagnostics))
    written.append(summary_path)
    events_path = f"{FEATURE_DEBUG_ROOT}/extraction-events.jsonl"
    _write_text(
        run_root,
        events_path,
        "".join(f"{json.dumps(item, sort_keys=True)}\n" for item in encoded),
    )
    written.append(events_path)
    spaces_path = f"{FEATURE_DEBUG_ROOT}/embedding-spaces.json"
    _write_json(run_root, spaces_path, _embedding_spaces(diagnostics))
    written.append(spaces_path)

    dense_records = [record for record in encoded if record["scope"] == FeatureScope.DENSE.value]
    if dense_records:
        dense_path = f"{FEATURE_DEBUG_ROOT}/feature-map-meta.json"
        _write_json(run_root, dense_path, dense_records)
        written.append(dense_path)

    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for diagnostic, record in zip(diagnostics, encoded, strict=True):
        if diagnostic.region is not None:
            by_region[str(diagnostic.region.region_id)].append(record)
    for region_id, records in sorted(by_region.items()):
        region_path = f"{FEATURE_DEBUG_ROOT}/regions/{region_id}/feature-meta.json"
        _write_json(run_root, region_path, {"features": records})
        written.append(region_path)
        if debug_level is FeatureDebugLevel.FULL:
            pooling = [
                _pooling_record(record)
                for record in records
                if record["region"]["pooling_policy"] is not None
            ]
            if pooling:
                pooling_path = f"{FEATURE_DEBUG_ROOT}/regions/{region_id}/pooling.json"
                _write_json(run_root, pooling_path, {"pooling": pooling})
                written.append(pooling_path)

    seen_preview_paths: set[str] = set()
    for preview in previews:
        if not _level_includes(debug_level, preview.minimum_level):
            continue
        relative = f"{FEATURE_DEBUG_ROOT}/{preview.relative_path}"
        if relative in seen_preview_paths:
            raise ValueError(f"duplicate feature preview path: {preview.relative_path}")
        seen_preview_paths.add(relative)
        target = run_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(preview.content)
        written.append(relative)
    return tuple(written)


def _encode_metric(item: FeatureExtractionDiagnostic) -> dict[str, Any]:
    """Encode required run metrics independently from debug level."""
    return {
        "event_id": item.event_id,
        "source_observation_id": str(item.source_observation_id),
        "stage_id": item.stage_id,
        "status": item.status.value,
        "backend_id": item.backend.backend_id,
        "feature_id": str(item.feature_id) if item.feature_id is not None else None,
        "scope": item.scope.value if item.scope is not None else None,
        "duration_seconds": item.duration_seconds,
        "peak_memory_bytes": item.peak_memory_bytes,
        "warnings": list(item.warnings),
        "failure_reason": item.failure_reason,
    }


def _encode_diagnostic(item: FeatureExtractionDiagnostic) -> dict[str, Any]:
    """Encode complete structured audit metadata."""
    return {
        **_encode_metric(item),
        "source_prepared_image_reference": item.source_prepared_image_reference,
        "source_image_width": item.source_image_width,
        "source_image_height": item.source_image_height,
        "backend": encode_provenance(item.backend),
        "embedding_space": (
            encode_embedding_space(item.embedding_space)
            if item.embedding_space is not None
            else None
        ),
        "embedding_space_id": (
            embedding_space_fingerprint(item.embedding_space)
            if item.embedding_space is not None
            else None
        ),
        "output_shape": list(item.output_shape) if item.output_shape is not None else None,
        "dtype": item.dtype,
        "normalization": item.normalization,
        "payload_reference": item.payload_reference,
        "preprocessing": list(item.preprocessing),
        "dense": _encode_dense(item.dense) if item.dense is not None else None,
        "region": _encode_region(item.region) if item.region is not None else None,
    }


def _encode_dense(item: DenseFeatureDiagnostic) -> dict[str, Any]:
    return {
        "source_artifact_id": item.source_artifact_id,
        "grid_width": item.grid_width,
        "grid_height": item.grid_height,
        "stride_x": item.stride_x,
        "stride_y": item.stride_y,
        "support_width": item.support_width,
        "support_height": item.support_height,
        "coordinate_transform_id": item.coordinate_transform_id,
    }


def _encode_box(box: BoundingBox2D | None) -> dict[str, int] | None:
    if box is None:
        return None
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


def _encode_region(item: RegionFeatureDiagnostic) -> dict[str, Any]:
    return {
        "region_id": str(item.region_id),
        "bounding_box": _encode_box(item.bounding_box),
        "support_box": _encode_box(item.support_box),
        "source_mask_reference": item.source_mask_reference,
        "mask_content_hash": item.mask_content_hash,
        "coordinate_transform_id": item.coordinate_transform_id,
        "pooling_policy": item.pooling_policy,
        "contributing_cell_count": item.contributing_cell_count,
        "total_weight": item.total_weight,
        "coverage_fraction": item.coverage_fraction,
    }


def _summary(items: Sequence[FeatureExtractionDiagnostic]) -> dict[str, Any]:
    produced = [item for item in items if item.feature_id is not None]
    return {
        "event_count": len(items),
        "feature_count": len(produced),
        "feature_count_by_scope": dict(
            sorted(Counter(item.scope.value for item in produced if item.scope is not None).items())
        ),
        "event_count_by_status": dict(sorted(Counter(item.status.value for item in items).items())),
        "event_count_by_backend": dict(
            sorted(Counter(item.backend.backend_id for item in items).items())
        ),
        "total_duration_seconds": sum(item.duration_seconds or 0.0 for item in items),
        "maximum_peak_memory_bytes": max(
            (item.peak_memory_bytes for item in items if item.peak_memory_bytes is not None),
            default=None,
        ),
    }


def _embedding_spaces(items: Sequence[FeatureExtractionDiagnostic]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.embedding_space is None:
            continue
        identity = embedding_space_fingerprint(item.embedding_space)
        by_id[identity] = {
            "embedding_space_id": identity,
            **encode_embedding_space(item.embedding_space),
        }
    return [by_id[key] for key in sorted(by_id)]


def _pooling_record(record: dict[str, Any]) -> dict[str, Any]:
    region = record["region"]
    return {
        "feature_id": record["feature_id"],
        "pooling_policy": region["pooling_policy"],
        "contributing_cell_count": region["contributing_cell_count"],
        "total_weight": region["total_weight"],
        "coverage_fraction": region["coverage_fraction"],
    }


def _level_includes(actual: FeatureDebugLevel, required: FeatureDebugLevel) -> bool:
    order = {
        FeatureDebugLevel.NONE: 0,
        FeatureDebugLevel.STANDARD: 1,
        FeatureDebugLevel.FULL: 2,
    }
    return order[actual] >= order[required]


def _write_text(root: Path, relative: str, content: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _write_json(root: Path, relative: str, value: Any) -> None:
    _write_text(root, relative, json.dumps(value, indent=2, sort_keys=True))

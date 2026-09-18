"""Deterministic evaluation reports for visual feature extraction.

Evaluation consumes canonical Visual Perception contracts and numerical
payloads without changing pipeline outputs. Correctness, spatial sampling,
and execution cost remain separate report sections so a faster or denser
backend is never silently treated as a higher-quality backend.

NumPy is imported only inside :func:`evaluate_feature_payload`; importing the
public :mod:`contextmap.evaluation` package therefore does not add a base
runtime dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    DenseFeatureMap,
    EmbeddingSpace,
    EmbeddingSpaceMismatchError,
    FeatureId,
    FeatureScope,
    VisualFeature,
    embedding_space_fingerprint,
    ensure_compatible_embedding_spaces,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_L2_NORMALIZATION_ATOL = 1e-5


class FeatureEvaluationError(ValueError):
    """Raised when feature evidence cannot satisfy an evaluation invariant."""


@dataclass(frozen=True, kw_only=True)
class FeatureEvaluationContext:
    """Variables that must remain fixed across a controlled comparison.

    Attributes:
        evaluation_id: Identity of this evaluation execution.
        source_observation_id: Physical observation used by every variant.
        source_image_content_hash: Hash of the exact prepared-image content.
        source_model_artifact_id: Identity of the fixed source model/checkpoint
            artifact, such as the DINO artifact used by both variants.
        region_set_id: Frozen region/mask set used downstream, when relevant.
        downstream_configuration_fingerprint: Identity of the downstream
            configuration held fixed during the comparison.
    """

    evaluation_id: str
    source_observation_id: SourceObservationId
    source_image_content_hash: str
    source_model_artifact_id: str
    region_set_id: str | None
    downstream_configuration_fingerprint: str

    def __post_init__(self) -> None:
        """Validate required comparison identities."""
        required = {
            "evaluation_id": self.evaluation_id,
            "source_observation_id": str(self.source_observation_id),
            "source_image_content_hash": self.source_image_content_hash,
            "source_model_artifact_id": self.source_model_artifact_id,
            "downstream_configuration_fingerprint": (self.downstream_configuration_fingerprint),
        }
        for name, value in required.items():
            if not value:
                raise ValueError(f"{name} must not be empty")
        if self.region_set_id == "":
            raise ValueError("region_set_id must be None or non-empty")


@dataclass(frozen=True, kw_only=True)
class FeatureNumericalReport:
    """Numerical correctness measurements for one feature payload.

    Attributes:
        element_count: Total number of scalar payload elements.
        all_finite: Whether every payload value is finite.
        minimum: Minimum scalar payload value.
        maximum: Maximum scalar payload value.
        vector_norm_min: Minimum L2 norm over vectors on the last axis.
        vector_norm_max: Maximum L2 norm over vectors on the last axis.
        max_l2_normalization_error: Maximum absolute distance from unit L2
            norm when ``normalization == "l2"``; otherwise ``None``.
    """

    element_count: int
    all_finite: bool
    minimum: float
    maximum: float
    vector_norm_min: float
    vector_norm_max: float
    max_l2_normalization_error: float | None


@dataclass(frozen=True, kw_only=True)
class FeatureSpatialReport:
    """Exact prepared-image-to-grid geometry for one dense feature.

    Attributes:
        source_artifact_id: Run/artifact owning the evaluated dense map.
        source_image_size: Prepared-image ``(width, height)`` in pixels.
        grid_size: Feature-grid ``(width, height)`` in cells.
        origin: Grid support origin ``(x, y)`` in image pixels.
        stride: Grid stride ``(x, y)`` in image pixels.
        support: Cell support ``(width, height)`` in image pixels.
        coordinate_transform_id: Identity of the spatial transform.
    """

    source_artifact_id: str
    source_image_size: tuple[int, int]
    grid_size: tuple[int, int]
    origin: tuple[float, float]
    stride: tuple[float, float]
    support: tuple[float, float]
    coordinate_transform_id: str


@dataclass(frozen=True, kw_only=True)
class FeatureCostReport:
    """Execution and storage cost, kept separate from feature correctness.

    Attributes:
        duration_seconds: Observed wall duration.
        peak_memory_bytes: Observed peak memory for the measured execution.
        payload_size_bytes: In-memory numerical payload size.
        processed_items: Number of source items represented by the duration.
        throughput_items_per_second: Derived throughput, or ``None`` when the
            measured duration is exactly zero.
    """

    duration_seconds: float
    peak_memory_bytes: int
    payload_size_bytes: int
    processed_items: int
    throughput_items_per_second: float | None

    def __post_init__(self) -> None:
        """Validate non-negative measurements and positive work count."""
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be finite and non-negative")
        if self.peak_memory_bytes < 0 or self.payload_size_bytes < 0:
            raise ValueError("memory and payload sizes must be non-negative")
        if self.processed_items <= 0:
            raise ValueError("processed_items must be positive")
        throughput = self.throughput_items_per_second
        if throughput is not None and (not math.isfinite(throughput) or throughput < 0.0):
            raise ValueError("throughput_items_per_second must be finite and non-negative")


@dataclass(frozen=True, kw_only=True)
class FeatureEvaluationReport:
    """Auditable correctness, spatial, and cost report for one feature.

    Attributes:
        context: Controlled experimental variables.
        feature_id: Local identity of the evaluated feature evidence.
        scope: Dense, global, or region feature scope.
        embedding_space: Full compatibility identity.
        embedding_space_id: Fingerprint referenced by the feature.
        backend: Exact backend/model/configuration provenance.
        payload_reference: Contractual payload reference.
        payload_hash: Deterministic hash of dtype, shape, and array values.
        shape: Exact numerical payload shape.
        dtype: Exact numerical payload dtype.
        normalization: Declared normalization policy.
        numerical: Numerical correctness measurements.
        spatial: Dense sampling geometry, absent for global/region features.
        cost: Runtime, memory, payload size, and throughput measurements.
    """

    context: FeatureEvaluationContext
    feature_id: FeatureId
    scope: FeatureScope
    embedding_space: EmbeddingSpace
    embedding_space_id: str
    backend: BackendProvenance
    payload_reference: str
    payload_hash: str
    shape: tuple[int, ...]
    dtype: str
    normalization: str | None
    numerical: FeatureNumericalReport
    spatial: FeatureSpatialReport | None
    cost: FeatureCostReport


@dataclass(frozen=True, kw_only=True)
class FeatureResolutionComparisonReport:
    """Controlled native-versus-enhanced dense feature-map comparison.

    This contract deliberately contains no aggregate feature-quality score.
    It groups two compatible reports while retaining each resolution,
    transform, numerical report, and cost report independently.

    Attributes:
        context: Variables held fixed across both variants.
        native: Baseline native-resolution dense report.
        enhanced: Compatible enhanced-resolution dense report.
    """

    context: FeatureEvaluationContext
    native: FeatureEvaluationReport
    enhanced: FeatureEvaluationReport


def evaluate_feature_payload(
    array: NDArray[Any],
    *,
    feature: VisualFeature,
    embedding_space: EmbeddingSpace,
    context: FeatureEvaluationContext,
    duration_seconds: float,
    peak_memory_bytes: int,
    processed_items: int = 1,
    dense_map: DenseFeatureMap | None = None,
) -> FeatureEvaluationReport:
    """Validate and summarize one canonical feature payload.

    Args:
        array: Numerical feature payload.
        feature: Canonical metadata for ``array``.
        embedding_space: Full space identified by
            ``feature.embedding_space_id``.
        context: Controlled input/model/downstream identities.
        duration_seconds: Separately measured wall duration.
        peak_memory_bytes: Separately measured peak memory.
        processed_items: Items represented by the duration measurement.
        dense_map: Required spatial metadata for dense features; forbidden for
            global and region features.

    Returns:
        An auditable report with separate numerical, spatial, and cost blocks.

    Raises:
        FeatureEvaluationError: If metadata, compatibility, values, declared
            normalization, or dense spatial metadata are inconsistent.
        ValueError: If cost measurements are invalid.
    """
    import numpy as np

    expected_space_id = embedding_space_fingerprint(embedding_space)
    if feature.embedding_space_id != expected_space_id:
        raise FeatureEvaluationError(
            "feature embedding space does not match the supplied EmbeddingSpace: "
            f"{feature.embedding_space_id!r} != {expected_space_id!r}"
        )
    if tuple(array.shape) != feature.shape:
        raise FeatureEvaluationError(
            f"payload shape {tuple(array.shape)} does not match feature shape {feature.shape}"
        )
    if str(array.dtype) != feature.dtype:
        raise FeatureEvaluationError(
            f"payload dtype {array.dtype!s} does not match feature dtype {feature.dtype!r}"
        )
    if not feature.shape or any(dimension <= 0 for dimension in feature.shape):
        raise FeatureEvaluationError("feature shape dimensions must be positive")
    if feature.shape[-1] != embedding_space.dimension:
        raise FeatureEvaluationError(
            f"embedding dimension {embedding_space.dimension} does not match "
            f"feature vector dimension {feature.shape[-1]}"
        )
    if not bool(np.isfinite(array).all()):
        raise FeatureEvaluationError("feature payload values must all be finite")

    vectors = array.reshape((-1, embedding_space.dimension))
    norms = np.linalg.norm(vectors, axis=1)
    normalization_error: float | None = None
    if feature.normalization == "l2":
        normalization_error = float(np.max(np.abs(norms - 1.0)))
        if normalization_error > _L2_NORMALIZATION_ATOL:
            raise FeatureEvaluationError(
                f"declared l2 normalization exceeds tolerance: max error {normalization_error:.8g}"
            )

    spatial = _spatial_report(feature=feature, dense_map=dense_map)
    duration = float(duration_seconds)
    throughput = None if duration == 0.0 else processed_items / duration
    cost = FeatureCostReport(
        duration_seconds=duration,
        peak_memory_bytes=peak_memory_bytes,
        payload_size_bytes=int(array.nbytes),
        processed_items=processed_items,
        throughput_items_per_second=throughput,
    )
    numerical = FeatureNumericalReport(
        element_count=int(array.size),
        all_finite=True,
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        vector_norm_min=float(np.min(norms)),
        vector_norm_max=float(np.max(norms)),
        max_l2_normalization_error=normalization_error,
    )
    return FeatureEvaluationReport(
        context=context,
        feature_id=feature.feature_id,
        scope=feature.scope,
        embedding_space=embedding_space,
        embedding_space_id=feature.embedding_space_id,
        backend=feature.provenance,
        payload_reference=feature.payload_reference,
        payload_hash=_payload_hash(array),
        shape=feature.shape,
        dtype=feature.dtype,
        normalization=feature.normalization,
        numerical=numerical,
        spatial=spatial,
        cost=cost,
    )


def assert_repeatable_feature_outputs(
    reference: FeatureEvaluationReport, candidate: FeatureEvaluationReport
) -> None:
    """Require equal output metadata and values under one fixed context.

    Feature and source-artifact IDs may differ between immutable executions,
    and measured costs are intentionally excluded from determinism. Backend
    configuration, embedding space, shape/dtype/normalization, spatial
    transform, and payload values must match exactly.

    Args:
        reference: Baseline evaluation report.
        candidate: Repeated evaluation report.

    Raises:
        FeatureEvaluationError: If context, output metadata, or payload differs.
    """
    if reference.context != candidate.context:
        raise FeatureEvaluationError("repeatability requires the same experimental context")

    reference_metadata = _repeatability_metadata(reference)
    candidate_metadata = _repeatability_metadata(candidate)
    if reference_metadata != candidate_metadata:
        raise FeatureEvaluationError("repeatability output metadata differs")
    if reference.payload_hash != candidate.payload_hash:
        raise FeatureEvaluationError("repeatability payload values differ")


def compare_feature_map_resolutions(
    *, native: FeatureEvaluationReport, enhanced: FeatureEvaluationReport
) -> FeatureResolutionComparisonReport:
    """Group compatible native and enhanced dense-map reports.

    Args:
        native: Baseline native-resolution report.
        enhanced: Candidate enhanced-resolution report.

    Returns:
        A comparison retaining the two independent report blocks.

    Raises:
        FeatureEvaluationError: If controlled variables differ, either report
            is not dense/spatial, or their embedding spaces are incompatible.
    """
    if native.context != enhanced.context:
        raise FeatureEvaluationError("resolution comparison requires the same experimental context")
    if (
        native.scope is not FeatureScope.DENSE
        or enhanced.scope is not FeatureScope.DENSE
        or native.spatial is None
        or enhanced.spatial is None
    ):
        raise FeatureEvaluationError("resolution comparison requires two dense spatial reports")
    try:
        ensure_compatible_embedding_spaces(native.embedding_space, enhanced.embedding_space)
    except EmbeddingSpaceMismatchError as error:
        raise FeatureEvaluationError(
            f"resolution comparison embedding space mismatch: {error}"
        ) from error
    if native.spatial.source_image_size != enhanced.spatial.source_image_size:
        raise FeatureEvaluationError("resolution comparison source image dimensions differ")
    return FeatureResolutionComparisonReport(
        context=native.context,
        native=native,
        enhanced=enhanced,
    )


def encode_feature_evaluation_report(report: FeatureEvaluationReport) -> dict[str, Any]:
    """Encode a feature evaluation report into JSON-compatible values."""
    spatial = None
    if report.spatial is not None:
        spatial = {
            "source_artifact_id": report.spatial.source_artifact_id,
            "source_image_size": list(report.spatial.source_image_size),
            "grid_size": list(report.spatial.grid_size),
            "origin": list(report.spatial.origin),
            "stride": list(report.spatial.stride),
            "support": list(report.spatial.support),
            "coordinate_transform_id": report.spatial.coordinate_transform_id,
        }
    return {
        "context": _encode_context(report.context),
        "feature_id": str(report.feature_id),
        "scope": report.scope.value,
        "embedding_space": _encode_embedding_space(report.embedding_space),
        "embedding_space_id": report.embedding_space_id,
        "backend": _encode_backend(report.backend),
        "payload_reference": report.payload_reference,
        "payload_hash": report.payload_hash,
        "shape": list(report.shape),
        "dtype": report.dtype,
        "normalization": report.normalization,
        "numerical": {
            "element_count": report.numerical.element_count,
            "all_finite": report.numerical.all_finite,
            "minimum": report.numerical.minimum,
            "maximum": report.numerical.maximum,
            "vector_norm_min": report.numerical.vector_norm_min,
            "vector_norm_max": report.numerical.vector_norm_max,
            "max_l2_normalization_error": (report.numerical.max_l2_normalization_error),
        },
        "spatial": spatial,
        "cost": {
            "duration_seconds": report.cost.duration_seconds,
            "peak_memory_bytes": report.cost.peak_memory_bytes,
            "payload_size_bytes": report.cost.payload_size_bytes,
            "processed_items": report.cost.processed_items,
            "throughput_items_per_second": report.cost.throughput_items_per_second,
        },
    }


def encode_feature_resolution_comparison_report(
    report: FeatureResolutionComparisonReport,
) -> dict[str, Any]:
    """Encode a native-versus-enhanced comparison for persistence."""
    return {
        "context": _encode_context(report.context),
        "native": encode_feature_evaluation_report(report.native),
        "enhanced": encode_feature_evaluation_report(report.enhanced),
    }


def _spatial_report(
    *, feature: VisualFeature, dense_map: DenseFeatureMap | None
) -> FeatureSpatialReport | None:
    if feature.scope is FeatureScope.DENSE:
        if dense_map is None:
            raise FeatureEvaluationError("dense feature evaluation requires a DenseFeatureMap")
        if dense_map.feature != feature:
            raise FeatureEvaluationError("DenseFeatureMap does not describe the evaluated feature")
        sampling = dense_map.sampling
        return FeatureSpatialReport(
            source_artifact_id=dense_map.source_artifact_id,
            source_image_size=(sampling.source_image_width, sampling.source_image_height),
            grid_size=(sampling.grid_width, sampling.grid_height),
            origin=(sampling.origin_x, sampling.origin_y),
            stride=(sampling.stride_x, sampling.stride_y),
            support=(sampling.support_width, sampling.support_height),
            coordinate_transform_id=sampling.coordinate_transform_id,
        )
    if dense_map is not None:
        raise FeatureEvaluationError("dense_map is only valid for DENSE feature evaluation")
    return None


def _payload_hash(array: NDArray[Any]) -> str:
    import numpy as np

    metadata = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True
    ).encode("utf-8")
    values = np.ascontiguousarray(array).tobytes(order="C")
    digest = hashlib.sha256(metadata + bytes((10,)) + values).hexdigest()
    return f"sha256:{digest}"


def _repeatability_metadata(report: FeatureEvaluationReport) -> tuple[Any, ...]:
    spatial = report.spatial
    spatial_geometry = None
    if spatial is not None:
        spatial_geometry = (
            spatial.source_image_size,
            spatial.grid_size,
            spatial.origin,
            spatial.stride,
            spatial.support,
            spatial.coordinate_transform_id,
        )
    return (
        report.scope,
        report.embedding_space,
        report.embedding_space_id,
        report.backend,
        report.shape,
        report.dtype,
        report.normalization,
        spatial_geometry,
    )


def _encode_context(context: FeatureEvaluationContext) -> dict[str, Any]:
    return {
        "evaluation_id": context.evaluation_id,
        "source_observation_id": str(context.source_observation_id),
        "source_image_content_hash": context.source_image_content_hash,
        "source_model_artifact_id": context.source_model_artifact_id,
        "region_set_id": context.region_set_id,
        "downstream_configuration_fingerprint": (context.downstream_configuration_fingerprint),
    }


def _encode_embedding_space(space: EmbeddingSpace) -> dict[str, Any]:
    return {
        "family": space.family,
        "model": space.model,
        "version": space.version,
        "checkpoint": space.checkpoint,
        "layer": space.layer,
        "dimension": space.dimension,
        "normalization": space.normalization,
    }


def _encode_backend(backend: BackendProvenance) -> dict[str, Any]:
    return {
        "backend_id": backend.backend_id,
        "capability": backend.capability,
        "provider": backend.provider,
        "model": backend.model,
        "version": backend.version,
        "configuration_fingerprint": backend.configuration_fingerprint,
    }

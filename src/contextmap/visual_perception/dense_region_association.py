"""Deterministic association of dense visual features with frozen 2D regions.

This module maps :class:`~contextmap.visual_perception.models.Region2D`
geometry from prepared-image coordinates into a backend-independent dense
feature grid. The mapping is described by :class:`DenseFeatureSampling`;
pooling therefore has no DINO-, CLIP-, or enhancement-specific branch.

The numerical array is accepted only by :func:`pool_region_feature` and is
never part of the persisted public identity. NumPy is imported lazily inside
that function so importing :mod:`contextmap.visual_perception` does not make
NumPy a base runtime dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.visual_perception.models import (
    BackendProvenance,
    BoundingBox2D,
    FeatureId,
    FeatureScope,
    Region2D,
    RegionId,
    VisualFeature,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

POOLING_POLICY = "mask_weighted_mean_preserve_l2_v2"
"""Versioned policy used by :func:`pool_region_feature`."""


class RegionAssociationError(ValueError):
    """Base class for invalid dense-to-region association inputs."""


class EmptyRegionSupportError(RegionAssociationError):
    """Raised when a region has no supported pixel in the dense feature map."""


@dataclass(frozen=True, kw_only=True)
class FeatureResolutionEnhancementProvenance:
    """Lineage and cost of one optional dense-resolution enhancement.

    Attributes:
        source_feature_id: Exact native dense feature used as input.
        source_artifact_id: Immutable artifact owning the source feature.
        source_payload_reference: Source payload reference within its artifact.
        source_embedding_space_id: Source feature-space fingerprint.
        source_dtype: Source numerical dtype.
        source_normalization: Source normalization declaration.
        source_coordinate_transform_id: Source image-to-grid transform.
        backend: Enhancement backend/model/configuration provenance.
        input_grid_size: Input feature-grid ``(width, height)``.
        output_grid_size: Output feature-grid ``(width, height)``.
        source_image_size: Prepared-image ``(width, height)``.
        output_embedding_space_id: Output feature-space fingerprint. It may
            equal the source fingerprint only when representation semantics
            remain compatible.
        device: Execution device recorded by the selected backend.
        precision: Execution precision recorded by the selected backend.
        duration_seconds: Measured enhancement wall duration.
        peak_memory_bytes: Measured peak memory/VRAM with backend-defined
            measurement semantics.
    """

    source_feature_id: FeatureId
    source_artifact_id: str
    source_payload_reference: str
    source_embedding_space_id: str
    source_dtype: str
    source_normalization: str | None
    source_coordinate_transform_id: str
    backend: BackendProvenance
    input_grid_size: tuple[int, int]
    output_grid_size: tuple[int, int]
    source_image_size: tuple[int, int]
    output_embedding_space_id: str
    device: str
    precision: str
    duration_seconds: float
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        """Validate required identities, dimensions, and cost fields."""
        required = {
            "source_artifact_id": self.source_artifact_id,
            "source_payload_reference": self.source_payload_reference,
            "source_embedding_space_id": self.source_embedding_space_id,
            "source_dtype": self.source_dtype,
            "source_coordinate_transform_id": self.source_coordinate_transform_id,
            "output_embedding_space_id": self.output_embedding_space_id,
            "device": self.device,
            "precision": self.precision,
        }
        for name, value in required.items():
            if not value:
                raise ValueError(f"{name} must not be empty")
        if self.backend.capability != "feature_resolution_enhancement":
            raise ValueError(
                "enhancement backend capability must be 'feature_resolution_enhancement'"
            )
        for name, size in {
            "input_grid_size": self.input_grid_size,
            "output_grid_size": self.output_grid_size,
            "source_image_size": self.source_image_size,
        }.items():
            if len(size) != 2 or any(dimension <= 0 for dimension in size):
                raise ValueError(f"{name} dimensions must be positive")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be finite and non-negative")
        if self.peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be non-negative")


@dataclass(frozen=True, kw_only=True)
class DenseFeatureSampling:
    """Axis-aligned sampling geometry from a prepared image to a feature grid.

    Each grid cell ``(row, column)`` represents the source-image footprint
    ``origin + index * stride`` through ``footprint + support``. Keeping
    stride and support separate represents both non-overlapping cell grids
    and overlapping receptive fields without naming a concrete backend.

    Attributes:
        grid_width: Number of feature cells along the horizontal axis.
        grid_height: Number of feature cells along the vertical axis.
        source_image_width: Prepared-image width in pixels.
        source_image_height: Prepared-image height in pixels.
        origin_x: Left edge of column zero's support, in prepared-image pixels.
        origin_y: Top edge of row zero's support, in prepared-image pixels.
        stride_x: Horizontal distance between adjacent cell supports, in pixels.
        stride_y: Vertical distance between adjacent cell supports, in pixels.
        support_width: Width represented by one cell, in prepared-image pixels.
        support_height: Height represented by one cell, in prepared-image pixels.
        coordinate_transform_id: Stable identity of the spatial transform or
            enhancement configuration that produced this sampling geometry.
    """

    grid_width: int
    grid_height: int
    source_image_width: int
    source_image_height: int
    origin_x: float
    origin_y: float
    stride_x: float
    stride_y: float
    support_width: float
    support_height: float
    coordinate_transform_id: str

    def __post_init__(self) -> None:
        """Validate dimensions and sampling values.

        Raises:
            ValueError: If dimensions, strides, supports, origins, or transform
                identity cannot describe a finite feature-grid mapping.
        """
        dimensions = {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "source_image_width": self.source_image_width,
            "source_image_height": self.source_image_height,
        }
        for name, dimension_value in dimensions.items():
            if dimension_value <= 0:
                raise ValueError(f"{name} must be positive")

        finite_values = {
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "stride_x": self.stride_x,
            "stride_y": self.stride_y,
            "support_width": self.support_width,
            "support_height": self.support_height,
        }
        for name, sampling_value in finite_values.items():
            if not math.isfinite(sampling_value):
                raise ValueError(f"{name} must be finite")
        for name in ("stride_x", "stride_y", "support_width", "support_height"):
            if finite_values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not self.coordinate_transform_id:
            raise ValueError("coordinate_transform_id must not be empty")


@dataclass(frozen=True, kw_only=True)
class DenseFeatureMap:
    """Metadata connecting one dense feature artifact to its sampling geometry.

    Attributes:
        feature: Canonical dense ``VisualFeature`` whose payload contains a
            ``(grid_height, grid_width, channels)`` array.
        sampling: Explicit mapping between feature cells and the prepared image.
        source_artifact_id: Identity of the run/artifact that owns the feature;
            required because ``FeatureId`` is only local to one result.
        enhancement: Lineage of the optional resolution-enhancement stage, or
            ``None`` for a native dense map.
    """

    feature: VisualFeature
    sampling: DenseFeatureSampling
    source_artifact_id: str
    enhancement: FeatureResolutionEnhancementProvenance | None = None

    def __post_init__(self) -> None:
        """Validate the feature scope and grid shape.

        Raises:
            ValueError: If ``feature`` is not a three-dimensional dense feature
                or its spatial shape differs from ``sampling``.
        """
        if self.feature.scope is not FeatureScope.DENSE:
            raise ValueError("DenseFeatureMap feature scope must be DENSE")
        if len(self.feature.shape) != 3:
            raise ValueError("DenseFeatureMap feature shape must be (height, width, channels)")
        expected = (self.sampling.grid_height, self.sampling.grid_width)
        if self.feature.shape[:2] != expected:
            raise ValueError(
                f"feature shape {self.feature.shape[:2]} does not match sampling grid {expected}"
            )
        if self.feature.shape[2] <= 0:
            raise ValueError("DenseFeatureMap channel dimension must be positive")
        if not self.source_artifact_id:
            raise ValueError("source_artifact_id must not be empty")


@dataclass(frozen=True, kw_only=True)
class RegionPoolingDiagnostics:
    """Support statistics for one dense-to-region pooling operation.

    Attributes:
        cell_row_range: Candidate grid-row range as ``(start, end)`` with an
            exclusive end.
        cell_col_range: Candidate grid-column range as ``(start, end)`` with an
            exclusive end.
        contributing_cell_count: Cells with at least one supported mask pixel.
        total_weight: Sum of supported pixel counts assigned to contributing
            cells. It may exceed unique coverage for overlapping supports.
        coverage_fraction: Fraction of requested mask/box pixels covered by at
            least one valid cell within the source image.
    """

    cell_row_range: tuple[int, int]
    cell_col_range: tuple[int, int]
    contributing_cell_count: int
    total_weight: int
    coverage_fraction: float


@dataclass(frozen=True, kw_only=True)
class RegionPoolingProvenance:
    """Auditable lineage for one pooled region vector.

    Attributes:
        source_feature_id: Identity of the exact dense feature being pooled.
        source_payload_reference: Artifact-relative dense payload reference.
        region_id: Identity of the frozen region providing geometry.
        source_mask_reference: Region mask reference, or ``None`` for box-only
            support.
        mask_content_hash: Hash of the decoded boolean mask used for pooling,
            or ``None`` when support is the complete box.
        source_artifact_id: Identity of the run/artifact owning the dense feature.
        pooling_policy: Versioned deterministic pooling policy.
        coordinate_transform_id: Identity of the sampling transform applied.
        configuration_fingerprint: Hash of feature, region, mask, policy, and
            complete sampling geometry.
        backend: Canonical provenance for the association operation itself.
    """

    source_feature_id: FeatureId
    source_payload_reference: str
    region_id: RegionId
    source_mask_reference: str | None
    mask_content_hash: str | None
    source_artifact_id: str
    pooling_policy: str
    coordinate_transform_id: str
    configuration_fingerprint: str
    backend: BackendProvenance


@dataclass(frozen=True, kw_only=True)
class RegionPoolingResult:
    """Numerical region representation plus compatibility and audit metadata.

    Attributes:
        vector: Pooled one-dimensional vector, with the dense payload dtype.
        embedding_space_id: Exact space inherited from the source feature.
        dtype: Exact dtype inherited from the source feature and output vector.
        normalization: Normalization preserved from the source feature. An L2
            declaration causes the pooled vector to be renormalized.
        diagnostics: Spatial support statistics for this operation.
        provenance: Source feature, region, policy, and transform lineage.
    """

    vector: NDArray[Any]
    embedding_space_id: str
    dtype: str
    normalization: str | None
    diagnostics: RegionPoolingDiagnostics
    provenance: RegionPoolingProvenance


def map_box_to_grid_cells(
    box: BoundingBox2D, *, dense_map: DenseFeatureMap
) -> tuple[int, int, int, int]:
    """Map a source-image box to candidate feature-grid cells.

    A cell is a candidate exactly when its sampling support has a non-empty
    intersection with ``box``. Returned ranges are clipped to the declared
    feature-grid bounds.

    Args:
        box: Region box in prepared-image pixel coordinates.
        dense_map: Dense feature metadata and sampling geometry.

    Returns:
        ``(row_start, row_end, column_start, column_end)`` with exclusive ends.
    """
    sampling = dense_map.sampling
    row_start, row_end = _axis_cell_range(
        box_start=float(box.y),
        box_end=float(box.y + box.height),
        origin=sampling.origin_y,
        stride=sampling.stride_y,
        support=sampling.support_height,
        cell_count=sampling.grid_height,
    )
    column_start, column_end = _axis_cell_range(
        box_start=float(box.x),
        box_end=float(box.x + box.width),
        origin=sampling.origin_x,
        stride=sampling.stride_x,
        support=sampling.support_width,
        cell_count=sampling.grid_width,
    )
    return row_start, row_end, column_start, column_end


def pool_region_feature(
    array: NDArray[Any],
    *,
    dense_map: DenseFeatureMap,
    region: Region2D,
    mask: NDArray[Any] | None = None,
) -> RegionPoolingResult:
    """Pool one dense feature map over a frozen region with pixel-mask weights.

    ``mask`` is decoded by the caller and uses box-local pixel coordinates. A
    missing mask becomes an all-true box mask, so masked, box-only, native, and
    enhanced feature maps all execute the same weighted-mean path.

    Args:
        array: Dense numerical payload declared by ``dense_map.feature``.
        dense_map: Backend-independent feature and sampling metadata.
        region: Frozen region whose geometry supplies pooling support.
        mask: Optional boolean array shaped exactly as
            ``(ceil(region.bounding_box.height),
            ceil(region.bounding_box.width))``.

    Returns:
        Pooled vector, diagnostics, compatibility metadata, and exact lineage.

    Raises:
        RegionAssociationError: If array or mask metadata is incompatible with
            the declared contracts.
        EmptyRegionSupportError: If no true mask pixel is covered by any valid
            feature-grid cell.
    """
    import numpy as np

    feature = dense_map.feature
    if tuple(array.shape) != feature.shape:
        raise RegionAssociationError(
            f"array shape {tuple(array.shape)} does not match feature shape {feature.shape}"
        )
    if str(array.dtype) != feature.dtype:
        raise RegionAssociationError(
            f"array dtype {array.dtype!s} does not match feature dtype {feature.dtype!r}"
        )

    box = region.bounding_box
    expected_mask_shape = _box_mask_shape(box)
    if mask is None:
        support_mask = np.ones(expected_mask_shape, dtype=np.bool_)
        mask_content_hash = None
    else:
        if tuple(mask.shape) != expected_mask_shape:
            raise RegionAssociationError(
                f"mask shape {tuple(mask.shape)} does not match region box {expected_mask_shape}"
            )
        if not np.issubdtype(mask.dtype, np.bool_):
            raise RegionAssociationError(f"mask dtype must be boolean, got {mask.dtype!s}")
        support_mask = mask
        mask_content_hash = f"sha256:{hashlib.sha256(mask.tobytes()).hexdigest()}"

    reference_area = int(np.count_nonzero(support_mask))
    if reference_area == 0:
        raise EmptyRegionSupportError(f"region {region.region_id!r} has no supported pixels")

    row_start, row_end, column_start, column_end = map_box_to_grid_cells(box, dense_map=dense_map)
    weighted_sum = np.zeros(feature.shape[2], dtype=np.float64)
    covered_mask = np.zeros(expected_mask_shape, dtype=np.bool_)
    total_weight = 0
    contributing_cell_count = 0
    sampling = dense_map.sampling

    for row in range(row_start, row_end):
        cell_top = sampling.origin_y + row * sampling.stride_y
        cell_bottom = cell_top + sampling.support_height
        for column in range(column_start, column_end):
            cell_left = sampling.origin_x + column * sampling.stride_x
            cell_right = cell_left + sampling.support_width
            local_bounds = _local_mask_bounds(
                box=box,
                source_width=sampling.source_image_width,
                source_height=sampling.source_image_height,
                left=cell_left,
                top=cell_top,
                right=cell_right,
                bottom=cell_bottom,
            )
            if local_bounds is None:
                continue
            local_row_start, local_row_end, local_column_start, local_column_end = local_bounds
            cell_mask = support_mask[
                local_row_start:local_row_end, local_column_start:local_column_end
            ]
            weight = int(np.count_nonzero(cell_mask))
            if weight == 0:
                continue

            weighted_sum += weight * np.asarray(array[row, column], dtype=np.float64)
            total_weight += weight
            contributing_cell_count += 1
            covered_slice = covered_mask[
                local_row_start:local_row_end, local_column_start:local_column_end
            ]
            covered_slice |= cell_mask

    if total_weight == 0:
        raise EmptyRegionSupportError(
            f"region {region.region_id!r} has no supported pixels in feature map "
            f"{feature.feature_id!r}"
        )

    pooled_vector = weighted_sum / total_weight
    if feature.normalization == "l2":
        vector_norm = float(np.linalg.norm(pooled_vector))
        if not math.isfinite(vector_norm) or vector_norm == 0.0:
            raise RegionAssociationError(
                "cannot preserve l2 normalization for a zero or non-finite pooled vector"
            )
        pooled_vector = pooled_vector / vector_norm
    vector = pooled_vector.astype(array.dtype, copy=False)
    covered_count = int(np.count_nonzero(covered_mask))
    diagnostics = RegionPoolingDiagnostics(
        cell_row_range=(row_start, row_end),
        cell_col_range=(column_start, column_end),
        contributing_cell_count=contributing_cell_count,
        total_weight=total_weight,
        coverage_fraction=covered_count / reference_area,
    )
    provenance = _pooling_provenance(
        dense_map=dense_map,
        region=region,
        mask_content_hash=mask_content_hash,
    )
    return RegionPoolingResult(
        vector=vector,
        embedding_space_id=feature.embedding_space_id,
        dtype=feature.dtype,
        normalization=feature.normalization,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _axis_cell_range(
    *,
    box_start: float,
    box_end: float,
    origin: float,
    stride: float,
    support: float,
    cell_count: int,
) -> tuple[int, int]:
    """Return clipped cells whose support overlaps one box axis."""
    start = math.floor((box_start - origin - support) / stride) + 1
    end = math.ceil((box_end - origin) / stride)
    clipped_start = min(cell_count, max(0, start))
    clipped_end = min(cell_count, max(clipped_start, end))
    return clipped_start, clipped_end


def _local_mask_bounds(
    *,
    box: BoundingBox2D,
    source_width: int,
    source_height: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[int, int, int, int] | None:
    """Convert a cell/source/box intersection to box-local integer bounds."""
    intersection_left = max(float(box.x), 0.0, left)
    intersection_top = max(float(box.y), 0.0, top)
    intersection_right = min(float(box.x + box.width), float(source_width), right)
    intersection_bottom = min(float(box.y + box.height), float(source_height), bottom)
    if intersection_left >= intersection_right or intersection_top >= intersection_bottom:
        return None

    mask_height, mask_width = _box_mask_shape(box)
    column_start = max(0, math.floor(intersection_left - box.x))
    column_end = min(mask_width, math.ceil(intersection_right - box.x))
    row_start = max(0, math.floor(intersection_top - box.y))
    row_end = min(mask_height, math.ceil(intersection_bottom - box.y))
    if column_start >= column_end or row_start >= row_end:
        return None
    return row_start, row_end, column_start, column_end


def _box_mask_shape(box: BoundingBox2D) -> tuple[int, int]:
    """Return the integer raster envelope for a possibly fractional box."""
    return math.ceil(box.height), math.ceil(box.width)


def _pooling_provenance(
    *,
    dense_map: DenseFeatureMap,
    region: Region2D,
    mask_content_hash: str | None,
) -> RegionPoolingProvenance:
    """Build deterministic, inspectable pooling lineage."""
    feature = dense_map.feature
    sampling = dense_map.sampling
    fingerprint_payload = {
        "source_feature_id": feature.feature_id,
        "source_payload_reference": feature.payload_reference,
        "source_artifact_id": dense_map.source_artifact_id,
        "embedding_space_id": feature.embedding_space_id,
        "dtype": feature.dtype,
        "normalization": feature.normalization,
        "region_id": region.region_id,
        "region_box": {
            "x": region.bounding_box.x,
            "y": region.bounding_box.y,
            "width": region.bounding_box.width,
            "height": region.bounding_box.height,
        },
        "source_mask_reference": region.mask_reference,
        "mask_content_hash": mask_content_hash,
        "pooling_policy": POOLING_POLICY,
        "sampling": {
            "grid_width": sampling.grid_width,
            "grid_height": sampling.grid_height,
            "source_image_width": sampling.source_image_width,
            "source_image_height": sampling.source_image_height,
            "origin_x": sampling.origin_x,
            "origin_y": sampling.origin_y,
            "stride_x": sampling.stride_x,
            "stride_y": sampling.stride_y,
            "support_width": sampling.support_width,
            "support_height": sampling.support_height,
            "coordinate_transform_id": sampling.coordinate_transform_id,
        },
    }
    encoded = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    backend = BackendProvenance(
        backend_id="dense_region_association",
        capability="feature_extractor",
        provider="contextmap",
        model=POOLING_POLICY,
        version="2",
        configuration_fingerprint=fingerprint,
    )
    return RegionPoolingProvenance(
        source_feature_id=feature.feature_id,
        source_payload_reference=feature.payload_reference,
        region_id=region.region_id,
        source_mask_reference=region.mask_reference,
        mask_content_hash=mask_content_hash,
        source_artifact_id=dense_map.source_artifact_id,
        pooling_policy=POOLING_POLICY,
        coordinate_transform_id=sampling.coordinate_transform_id,
        configuration_fingerprint=fingerprint,
        backend=backend,
    )

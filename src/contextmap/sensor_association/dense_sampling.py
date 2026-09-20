"""Deterministic sampling of a dense visual feature map onto visible geometry.

Dense visual evidence is anchored to persistent geometry through the public
:class:`~contextmap.visual_perception.DenseFeatureMap` contract, never through a
specific backbone. For every visible geometry point the chain is::

    GeometryReference -> prepared-image coordinate -> feature-grid cell(s) -> local sample

The mapping from prepared-image pixels to grid cells comes only from the feature's
:class:`~contextmap.visual_perception.DenseFeatureSampling` (origin, stride, support and
grid size, in prepared-image pixel *edges*). A native map and a resolution-enhanced map are
sampled by the same code, and one patch is never assumed to be one pixel.

The result is a set of **indices and weights**, not feature vectors: the association is
persisted by reference, and :meth:`DenseFeatureSamples.gather` reads the vectors from the
payload only when they are needed. Points the grid cannot serve are reported as out of
support instead of being clamped.

Fusion of samples across frames and any learned 3D encoding are outside this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import SourceObservationId
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.frame_projection import FrameProjection
from contextmap.sensor_association.models import CalibrationRef, PoseRef
from contextmap.sensor_association.serialization import encode_calibration_ref, encode_pose_ref
from contextmap.sensor_association.visibility import VisibilityResolution
from contextmap.visual_perception import (
    BackendProvenance,
    DenseFeatureMap,
    DenseFeatureSampling,
    FeatureId,
    FeatureResolutionEnhancementProvenance,
    PerceptionResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

SAMPLING_POLICY_ID = "dense-feature-sampling-v1"
"""Versioned identity of the sampling rules documented in this module."""


class InterpolationPolicy(Enum):
    """How a feature is read at a prepared-image pixel.

    Attributes:
        NEAREST: The cell whose support center is nearest to the pixel, provided that
            support contains the pixel. Valid wherever a cell covers the pixel.
        BILINEAR: The four cells around the pixel, weighted by proximity of their support
            centers. Valid only between the outer cell centers of the grid, where all four
            neighbors exist.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"


@dataclass(frozen=True, kw_only=True)
class DenseSamplingProvenance:
    """Everything a sampled feature association depended on.

    Attributes:
        policy_id: Versioned sampling rules.
        interpolation: The selected interpolation policy.
        feature_id: The dense feature that was sampled.
        source_artifact_id: The artifact that owns the feature.
        payload_reference: Where the feature payload lives within that artifact.
        embedding_space_id: The feature space the samples live in.
        dtype: Payload element type.
        normalization: Normalization the feature declares.
        extractor: The model that produced the feature.
        enhancement: The optional resolution-enhancement lineage, ``None`` for a native map.
        coordinate_transform_id: The transform that produced the sampling geometry.
        sampling_fingerprint: Hash of the complete sampling geometry.
        map_id: The geometric map the sampled points belong to.
        source_observation_id: The camera frame.
        image_transform_id: The raw-to-prepared image chain the pixels went through.
        calibration_ref: The calibration and camera used to project.
        pose_ref: The pose used to project.
        visibility_policy_id: The occlusion rule that made points eligible.
        visibility_policy_fingerprint: Hash of that rule's parameters.
    """

    policy_id: str
    interpolation: InterpolationPolicy
    feature_id: FeatureId
    source_artifact_id: str
    payload_reference: str
    embedding_space_id: str
    dtype: str
    normalization: str | None
    extractor: BackendProvenance
    enhancement: FeatureResolutionEnhancementProvenance | None
    coordinate_transform_id: str
    sampling_fingerprint: str
    map_id: MapId
    source_observation_id: SourceObservationId
    image_transform_id: str
    calibration_ref: CalibrationRef
    pose_ref: PoseRef
    visibility_policy_id: str
    visibility_policy_fingerprint: str

    def to_record(self) -> dict[str, Any]:
        """Return the provenance as JSON primitives."""
        return {
            "policy_id": self.policy_id,
            "interpolation": self.interpolation.value,
            "feature_id": str(self.feature_id),
            "source_artifact_id": self.source_artifact_id,
            "payload_reference": self.payload_reference,
            "embedding_space_id": self.embedding_space_id,
            "dtype": self.dtype,
            "normalization": self.normalization,
            "extractor": dataclasses.asdict(self.extractor),
            "enhancement": None
            if self.enhancement is None
            else dataclasses.asdict(self.enhancement),
            "coordinate_transform_id": self.coordinate_transform_id,
            "sampling_fingerprint": self.sampling_fingerprint,
            "map_id": str(self.map_id),
            "source_observation_id": str(self.source_observation_id),
            "image_transform_id": self.image_transform_id,
            "calibration_ref": encode_calibration_ref(self.calibration_ref),
            "pose_ref": encode_pose_ref(self.pose_ref),
            "visibility_policy_id": self.visibility_policy_id,
            "visibility_policy_fingerprint": self.visibility_policy_fingerprint,
        }


@dataclass(frozen=True, kw_only=True, eq=False)
class DenseFeatureSamples:
    """Where each visible point of one frame reads a dense feature map.

    The association is stored as indices and weights into the feature grid, so no
    feature vector is duplicated per geometry point.

    Attributes:
        frame: The projection the points belong to.
        provenance: The full lineage of the association.
        eligible_indices: ``(E,)`` frame positions of the visible points that were sampled
            or found out of support.
        sampled: ``(E,)`` the grid could serve the point under the interpolation policy.
        cell_rows: ``(E, T)`` grid rows read, ``T = 1`` for nearest and ``4`` for bilinear;
            ``-1`` where the point is out of support.
        cell_cols: ``(E, T)`` grid columns read, ``-1`` where out of support.
        weights: ``(E, T)`` weight of each cell, summing to ``1`` per sampled point and
            ``0`` where out of support.
    """

    frame: FrameProjection
    dense_map: DenseFeatureMap
    provenance: DenseSamplingProvenance
    eligible_indices: NDArray[Any]
    sampled: NDArray[Any]
    cell_rows: NDArray[Any]
    cell_cols: NDArray[Any]
    weights: NDArray[Any]

    @property
    def sampled_count(self) -> int:
        """Number of points the grid served."""
        return int(self.sampled.sum())

    @property
    def out_of_support_count(self) -> int:
        """Number of eligible points the grid could not serve."""
        return int((~self.sampled).sum())

    @property
    def sampled_point_indices(self) -> NDArray[Any]:
        """Frame positions of the points the grid served, in order."""
        indices: NDArray[Any] = self.eligible_indices[self.sampled]
        return indices

    def gather(self, array: NDArray[Any]) -> NDArray[Any]:
        """Read the feature vectors of the sampled points from the dense payload.

        An ``l2``-normalized feature is renormalized after interpolation, as the payload
        declares. Points out of support are not returned; see :attr:`sampled_point_indices`.

        Args:
            array: The payload declared by the dense feature, shaped
                ``(grid_height, grid_width, channels)``.

        Returns:
            ``(S, channels)`` vectors in the payload dtype, one per sampled point.

        Raises:
            ValueError: If the payload shape or dtype disagrees with the feature, or an
                ``l2``-normalized vector interpolates to zero.
        """
        import numpy as np

        feature = self.dense_map.feature
        if tuple(array.shape) != feature.shape:
            raise ValueError(
                f"payload shape {tuple(array.shape)} does not match feature shape {feature.shape}"
            )
        if str(array.dtype) != feature.dtype:
            raise ValueError(
                f"payload dtype {array.dtype!s} does not match feature dtype {feature.dtype!r}"
            )
        rows = self.cell_rows[self.sampled]
        columns = self.cell_cols[self.sampled]
        weights = self.weights[self.sampled]
        cells = np.asarray(array[rows, columns], dtype=np.float64)
        combined = (weights[..., None] * cells).sum(axis=1)
        if feature.normalization == "l2":
            norms = np.linalg.norm(combined, axis=1, keepdims=True)
            if (~np.isfinite(norms) | (norms == 0.0)).any():
                raise ValueError("cannot preserve l2 normalization for a zero interpolated vector")
            combined = combined / norms
        vectors: NDArray[Any] = combined.astype(array.dtype, copy=False)
        return vectors


def sample_dense_features(
    resolution: VisibilityResolution,
    perception_result: PerceptionResult,
    dense_map: DenseFeatureMap,
    *,
    interpolation: InterpolationPolicy,
) -> DenseFeatureSamples:
    """Map every visible point to the cell(s) of a dense feature map.

    Args:
        resolution: The visibility of the frame's points; only visible points are eligible.
        perception_result: The perception result that owns the dense feature.
        dense_map: The dense feature and its sampling geometry, native or enhanced.
        interpolation: How the feature is read at a pixel.

    Returns:
        Indices and weights per eligible point, with the lineage of the association.

    Raises:
        AssociationInputError: If the perception result is of another observation, the
            dense feature is not part of it or differs from it, the sampling geometry is
            defined over another image size than the prepared image, or the declared
            enhancement disagrees with the sampling it produced.
    """
    import numpy as np

    frame = resolution.frame
    _preflight(frame, perception_result, dense_map)
    sampling = dense_map.sampling
    eligible = np.flatnonzero(resolution.visible)
    edges = frame.prepared_pixels[eligible] + 0.5
    edge_x, edge_y = edges[:, 0], edges[:, 1]
    # Coordenada contínua da grade, em unidades de célula, com os centros em inteiros.
    grid_x = (edge_x - sampling.origin_x - sampling.support_width / 2) / sampling.stride_x
    grid_y = (edge_y - sampling.origin_y - sampling.support_height / 2) / sampling.stride_y

    if interpolation is InterpolationPolicy.NEAREST:
        rows, columns, weights, sampled = _nearest(sampling, edge_x, edge_y, grid_x, grid_y)
    else:
        rows, columns, weights, sampled = _bilinear(sampling, grid_x, grid_y)

    return DenseFeatureSamples(
        frame=frame,
        dense_map=dense_map,
        provenance=_provenance(resolution, dense_map, interpolation),
        eligible_indices=eligible,
        sampled=sampled,
        cell_rows=np.where(sampled[:, None], rows, -1),
        cell_cols=np.where(sampled[:, None], columns, -1),
        weights=np.where(sampled[:, None], weights, 0.0),
    )


def _nearest(
    sampling: DenseFeatureSampling,
    edge_x: NDArray[Any],
    edge_y: NDArray[Any],
    grid_x: NDArray[Any],
    grid_y: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    import numpy as np

    columns = np.floor(grid_x + 0.5).astype(np.int64)
    rows = np.floor(grid_y + 0.5).astype(np.int64)
    in_grid = (
        (columns >= 0)
        & (columns < sampling.grid_width)
        & (rows >= 0)
        & (rows < sampling.grid_height)
    )
    left = sampling.origin_x + columns * sampling.stride_x
    top = sampling.origin_y + rows * sampling.stride_y
    contained = (
        (edge_x >= left)
        & (edge_x < left + sampling.support_width)
        & (edge_y >= top)
        & (edge_y < top + sampling.support_height)
    )
    sampled = in_grid & contained
    return rows[:, None], columns[:, None], np.ones((len(rows), 1)), sampled


def _bilinear(
    sampling: DenseFeatureSampling, grid_x: NDArray[Any], grid_y: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    import numpy as np

    sampled = (
        (grid_x >= 0)
        & (grid_x <= sampling.grid_width - 1)
        & (grid_y >= 0)
        & (grid_y <= sampling.grid_height - 1)
    )
    column0 = np.floor(grid_x).astype(np.int64)
    row0 = np.floor(grid_y).astype(np.int64)
    fraction_x = grid_x - column0
    fraction_y = grid_y - row0
    # Sobre o último centro a fração é zero e o vizinho seguinte não existe: fica a mesma célula.
    column1 = np.minimum(column0 + 1, sampling.grid_width - 1)
    row1 = np.minimum(row0 + 1, sampling.grid_height - 1)
    rows = np.stack((row0, row0, row1, row1), axis=1)
    columns = np.stack((column0, column1, column0, column1), axis=1)
    weights = np.stack(
        (
            (1 - fraction_y) * (1 - fraction_x),
            (1 - fraction_y) * fraction_x,
            fraction_y * (1 - fraction_x),
            fraction_y * fraction_x,
        ),
        axis=1,
    )
    return rows, columns, weights, sampled


def _preflight(
    frame: FrameProjection, perception_result: PerceptionResult, dense_map: DenseFeatureMap
) -> None:
    if perception_result.source_observation_id != frame.source_observation_id:
        raise AssociationInputError(
            f"the perception result {perception_result.result_id!r} is of observation "
            f"{perception_result.source_observation_id!r}, not of {frame.source_observation_id!r}"
        )
    feature = dense_map.feature
    owned = next(
        (f for f in perception_result.features if f.feature_id == feature.feature_id), None
    )
    if owned is None:
        raise AssociationInputError(
            f"the dense feature {feature.feature_id!r} is not part of perception result "
            f"{perception_result.result_id!r}"
        )
    if owned.embedding_space_id != feature.embedding_space_id:
        raise AssociationInputError(
            f"the dense feature {feature.feature_id!r} is in embedding space "
            f"{feature.embedding_space_id!r} but the perception result declares "
            f"{owned.embedding_space_id!r}"
        )
    sampling = dense_map.sampling
    prepared_size = frame.image_transform.prepared_size
    if (sampling.source_image_width, sampling.source_image_height) != prepared_size:
        raise AssociationInputError(
            f"the dense feature is sampled over an image of "
            f"{(sampling.source_image_width, sampling.source_image_height)} but the prepared image "
            f"is {prepared_size}"
        )
    enhancement = dense_map.enhancement
    if enhancement is None:
        return
    if enhancement.output_grid_size != (sampling.grid_width, sampling.grid_height):
        raise AssociationInputError(
            f"the enhancement declares an output grid {enhancement.output_grid_size} but the "
            f"sampling grid is {(sampling.grid_width, sampling.grid_height)}"
        )
    if enhancement.source_image_size != prepared_size:
        raise AssociationInputError(
            f"the enhancement declares an image of {enhancement.source_image_size} but the "
            f"prepared image is {prepared_size}"
        )
    if enhancement.output_embedding_space_id != feature.embedding_space_id:
        raise AssociationInputError(
            f"the enhancement outputs embedding space {enhancement.output_embedding_space_id!r} "
            f"but the feature declares {feature.embedding_space_id!r}"
        )


def _provenance(
    resolution: VisibilityResolution,
    dense_map: DenseFeatureMap,
    interpolation: InterpolationPolicy,
) -> DenseSamplingProvenance:
    frame = resolution.frame
    feature = dense_map.feature
    sampling = dense_map.sampling
    return DenseSamplingProvenance(
        policy_id=SAMPLING_POLICY_ID,
        interpolation=interpolation,
        feature_id=feature.feature_id,
        source_artifact_id=dense_map.source_artifact_id,
        payload_reference=feature.payload_reference,
        embedding_space_id=feature.embedding_space_id,
        dtype=feature.dtype,
        normalization=feature.normalization,
        extractor=feature.provenance,
        enhancement=dense_map.enhancement,
        coordinate_transform_id=sampling.coordinate_transform_id,
        sampling_fingerprint=_sampling_fingerprint(sampling),
        map_id=frame.map_id,
        source_observation_id=frame.source_observation_id,
        image_transform_id=frame.image_transform.transform_id,
        calibration_ref=frame.calibration_ref,
        pose_ref=frame.pose_ref,
        visibility_policy_id=resolution.policy.policy_id,
        visibility_policy_fingerprint=resolution.policy.fingerprint(),
    )


def _sampling_fingerprint(sampling: DenseFeatureSampling) -> str:
    payload = {
        "grid": [sampling.grid_width, sampling.grid_height],
        "image": [sampling.source_image_width, sampling.source_image_height],
        "origin": [sampling.origin_x, sampling.origin_y],
        "stride": [sampling.stride_x, sampling.stride_y],
        "support": [sampling.support_width, sampling.support_height],
        "coordinate_transform_id": sampling.coordinate_transform_id,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"

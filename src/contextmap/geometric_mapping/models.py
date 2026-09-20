"""Canonical geometry contracts for Geometric Mapping.

Geometric Mapping is the authoritative 3D spatial foundation of Solution 1: it
turns sensor-local measurements into persistent geometry expressed in one
global map frame and keeps enough lineage to reconstruct how every stored
coordinate was produced. These contracts stop there. Labels, visual features,
semantic claims, entities and relations are attached later, by other
capabilities, through :class:`GeometryReference`; nothing here owns them.

Coordinate authority
    ``GeometryPoint.coordinates_m`` in the global map frame is authoritative.
    ``source_coordinates_m`` keeps the original sensor-local measurement and is
    never conflated with it. A processing segment or submap frame may only
    exist as an explicitly derived view (``P_segment = T_segment_map * P_map``),
    never as a replacement of the map-frame coordinate.

Identity
    A :class:`GeometryReference` is ``(map_id, geometry_id)`` and is local to one
    immutable map artifact. Downstream stages hold references instead of copying
    XYZ or provenance.

No ROS, PCL, Open3D or NumPy type appears here: a persisted map is readable
with the standard library alone. See
``src/contextmap/geometric_mapping/docs/contracts.md`` for the field reference.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import NewType

from contextmap.geometric_mapping.motion_correction import MotionCorrectionState
from contextmap.ingestion import (
    FrameId,
    SequenceArtifactId,
    SourceObservationId,
)
from contextmap.shared import SourceTimestamp, Vector3
from contextmap.state_estimation import (
    LookupPolicy,
    PoseEstimateId,
    StateEstimationRunId,
    TimeBounds,
    TrajectoryId,
)

MapId = NewType("MapId", str)
"""Identity of one immutable geometric-map artifact."""

GeometryId = NewType("GeometryId", str)
"""Identity of one geometry element, local to its map."""

SpatialIndexParameter = str | int | float | bool
"""A primitive value describing how a spatial index was built."""


def geometry_id_for(*, map_id: MapId, index: int) -> GeometryId:
    """Compute the deterministic identity of the ``index``-th geometry of a map.

    Args:
        map_id: The map that owns the geometry.
        index: Zero-based position of the geometry in the map.

    Returns:
        A pure function of the inputs, so references survive serialization and
        artifact round-trips without an identity registry.
    """
    return GeometryId(f"{map_id}--geom-{index:09d}")


@dataclass(frozen=True, kw_only=True)
class GeometryReference:
    """Compact reference to one geometry element of one map.

    Attributes:
        map_id: The immutable map artifact that owns the geometry.
        geometry_id: The element within that map.
    """

    map_id: MapId
    geometry_id: GeometryId


@dataclass(frozen=True, kw_only=True)
class Bounds3D:
    """Axis-aligned box that declares the frame it is expressed in.

    Boundaries are inclusive: a point on a face is contained, and two boxes that
    touch intersect. The frame is never inferred from a map name or a caller.

    Attributes:
        frame_id: Frame the extents are expressed in.
        minimum_m: ``(x, y, z)`` lower corner, in meters.
        maximum_m: ``(x, y, z)`` upper corner, in meters.
    """

    frame_id: FrameId
    minimum_m: Vector3
    maximum_m: Vector3

    def __post_init__(self) -> None:
        """Validate the frame and the extents.

        Raises:
            ValueError: If the frame is empty, a value is not finite, or the
                minimum exceeds the maximum on any axis.
        """
        if not self.frame_id:
            raise ValueError("bounds frame_id must not be empty")
        if not all(math.isfinite(value) for value in (*self.minimum_m, *self.maximum_m)):
            raise ValueError("bounds values must be finite")
        if any(low > high for low, high in zip(self.minimum_m, self.maximum_m, strict=True)):
            raise ValueError(
                f"bounds minimum must not exceed maximum on any axis, got "
                f"{self.minimum_m!r} and {self.maximum_m!r}"
            )

    @classmethod
    def enclosing(cls, coordinates_m: Iterable[Vector3], *, frame_id: FrameId) -> Bounds3D:
        """Build the tight box around a set of points.

        Args:
            coordinates_m: Points ``(x, y, z)`` expressed in ``frame_id``.
            frame_id: Frame the points are expressed in.

        Returns:
            The smallest axis-aligned box containing every point.

        Raises:
            ValueError: If there is no point.
        """
        points = list(coordinates_m)
        if not points:
            raise ValueError("enclosing bounds need at least one point")
        xs, ys, zs = zip(*points, strict=True)
        return cls(
            frame_id=frame_id,
            minimum_m=(min(xs), min(ys), min(zs)),
            maximum_m=(max(xs), max(ys), max(zs)),
        )

    def contains(self, coordinates_m: Vector3, *, frame_id: FrameId) -> bool:
        """Check whether a point lies inside the box, boundaries included.

        Args:
            coordinates_m: The point ``(x, y, z)``.
            frame_id: Frame the point is expressed in; must equal the box's.

        Returns:
            ``True`` when the point is inside or on a face.

        Raises:
            ValueError: If ``frame_id`` differs from the box's frame.
        """
        self._require_same_frame(frame_id)
        return all(
            low <= value <= high
            for low, value, high in zip(self.minimum_m, coordinates_m, self.maximum_m, strict=True)
        )

    def intersects(self, other: Bounds3D) -> bool:
        """Check whether two boxes overlap, touching faces included.

        Args:
            other: The other box; must be in the same frame.

        Returns:
            ``True`` when they share at least one point.

        Raises:
            ValueError: If the frames differ.
        """
        self._require_same_frame(other.frame_id)
        return all(
            self_low <= other_high and other_low <= self_high
            for self_low, self_high, other_low, other_high in zip(
                self.minimum_m, self.maximum_m, other.minimum_m, other.maximum_m, strict=True
            )
        )

    def _require_same_frame(self, frame_id: FrameId) -> None:
        if frame_id != self.frame_id:
            raise ValueError(
                f"bounds are expressed in frame {self.frame_id!r} but frame {frame_id!r} was "
                "given; frames are never inferred"
            )


class TransformKind(Enum):
    """Where a transform step comes from.

    Attributes:
        STATIC_CALIBRATION: A fixed extrinsic from the canonical calibration.
        DYNAMIC_POSE: A pose from a State Estimation trajectory at a timestamp.
    """

    STATIC_CALIBRATION = "static_calibration"
    DYNAMIC_POSE = "dynamic_pose"


@dataclass(frozen=True, kw_only=True)
class TransformStep:
    """One ``T_parent_child`` applied to bring a source point into the map frame.

    Attributes:
        kind: Static calibration or dynamic pose.
        parent_frame: Frame the step maps into.
        child_frame: Frame whose coordinates the step maps.
        reference: What the transform was resolved from: the calibration
            identity for a static step, or the pose identity for a dynamic one.
        source_estimate_ids: For a dynamic step, the estimated poses it came
            from (one, or two when the pose was interpolated); empty for a
            static step.
    """

    kind: TransformKind
    parent_frame: FrameId
    child_frame: FrameId
    reference: str
    source_estimate_ids: tuple[PoseEstimateId, ...] = ()

    def __post_init__(self) -> None:
        """Validate frames, reference and pose sources.

        Raises:
            ValueError: If a frame is empty or the frames are equal, the reference
                is empty, or ``source_estimate_ids`` does not match ``kind``.
        """
        if not self.parent_frame or not self.child_frame or self.parent_frame == self.child_frame:
            raise ValueError(
                "transform step parent_frame and child_frame must be non-empty and differ"
            )
        if not self.reference:
            raise ValueError("transform step reference must not be empty")
        if self.kind is TransformKind.DYNAMIC_POSE and not self.source_estimate_ids:
            raise ValueError("a dynamic step needs source_estimate_ids")
        if self.kind is TransformKind.STATIC_CALIBRATION and self.source_estimate_ids:
            raise ValueError("a static step must not carry source_estimate_ids")


@dataclass(frozen=True, kw_only=True)
class TransformLineage:
    """The chain that maps a source-frame point into the map frame.

    ``P_map = T_0 * T_1 * ... * P_source`` where ``T_i`` is ``steps[i]``: the first
    step has the map frame as its parent, and each step's child frame is the next
    step's parent. For a LiDAR point the chain is
    ``T_map_body(t) * T_body_lidar``. A record is shared by every point of one
    source observation instead of being copied per point.

    Attributes:
        steps: The transforms in application order, map side first; empty only
            for a point already expressed in the map frame.
    """

    steps: tuple[TransformStep, ...]

    def __post_init__(self) -> None:
        """Validate that consecutive steps connect.

        Raises:
            ValueError: If a step's child frame is not the next step's parent frame.
        """
        for earlier, later in zip(self.steps, self.steps[1:], strict=False):
            if earlier.child_frame != later.parent_frame:
                raise ValueError(
                    f"transform steps must be contiguous, but {earlier.child_frame!r} is "
                    f"followed by a step from {later.parent_frame!r}"
                )

    def connects(self, map_frame: FrameId, source_frame: FrameId) -> bool:
        """Check that the chain maps ``source_frame`` into ``map_frame``.

        Args:
            map_frame: Frame the chain must end in.
            source_frame: Frame the chain must start from.

        Returns:
            ``True`` when the first step's parent is ``map_frame`` and the last
            step's child is ``source_frame``; with no steps, when the frames are equal.
        """
        if not self.steps:
            return map_frame == source_frame
        return (
            self.steps[0].parent_frame == map_frame and self.steps[-1].child_frame == source_frame
        )


class PointOrigin(Enum):
    """Whether a point is one raw measurement or a merge of several.

    Attributes:
        MEASURED: One sensor measurement, transformed into the map frame.
        AGGREGATED: A point produced by an explicit deduplication or
            downsampling rule from several measurements.
    """

    MEASURED = "measured"
    AGGREGATED = "aggregated"


@dataclass(frozen=True, kw_only=True)
class GeometryPointProvenance:
    """How a stored point was produced.

    An aggregated point never pretends to be one raw measurement: it names the
    rule that merged its inputs and how many contributed.

    Attributes:
        origin: Measured or aggregated.
        aggregation_rule: Identity of the explicit rule for an aggregated point;
            ``None`` for a measured one.
        contributing_point_count: Measurements behind the point: exactly one for
            a measured point, at least two for an aggregated one.
        motion_correction: Whether the scan the point came from was corrected
            for platform motion. Explicit for every point and ``UNKNOWN`` unless a
            declaration says otherwise; never inferred.
    """

    origin: PointOrigin = PointOrigin.MEASURED
    aggregation_rule: str | None = None
    contributing_point_count: int = 1
    motion_correction: MotionCorrectionState = MotionCorrectionState.UNKNOWN

    def __post_init__(self) -> None:
        """Validate that the provenance is coherent.

        Raises:
            ValueError: If a measured point names a rule or does not come from one
                measurement, or an aggregated point has no rule or fewer than two
                contributing points.
        """
        if self.origin is PointOrigin.MEASURED:
            if self.aggregation_rule is not None:
                raise ValueError("measured provenance must not name an aggregation_rule")
            if self.contributing_point_count != 1:
                raise ValueError("a measured point has exactly one contributing_point_count")
        else:
            if not self.aggregation_rule:
                raise ValueError("aggregated provenance must name its aggregation_rule")
            if self.contributing_point_count < 2:
                raise ValueError("an aggregated point needs at least two contributing_point_count")


@dataclass(frozen=True, kw_only=True)
class GeometryPoint:
    """One persistent 3D point in the global map frame, with its origin.

    Attributes:
        geometry_id: Identity of the point within its map.
        map_id: The immutable map artifact that owns the point.
        map_frame: Global map frame ``coordinates_m`` is expressed in.
        coordinates_m: Authoritative ``(x, y, z)`` in ``map_frame``, in meters.
        source_frame: Sensor frame the point was measured in.
        source_coordinates_m: Original ``(x, y, z)`` in ``source_frame``, in meters.
        source_observation_id: Physical observation the point came from.
        source_point_index: Index of the point within that observation, when it
            is one raw measurement; ``None`` for an aggregated point.
        acquisition_timestamp: When the source observation was acquired.
        transform_lineage: The chain that produced ``coordinates_m`` from
            ``source_coordinates_m``.
        provenance: How the point was produced.
    """

    geometry_id: GeometryId
    map_id: MapId
    map_frame: FrameId
    coordinates_m: Vector3
    source_frame: FrameId
    source_coordinates_m: Vector3
    source_observation_id: SourceObservationId
    source_point_index: int | None
    acquisition_timestamp: SourceTimestamp
    transform_lineage: TransformLineage
    provenance: GeometryPointProvenance = field(default_factory=GeometryPointProvenance)

    def __post_init__(self) -> None:
        """Validate identities, frames, units, lineage and provenance.

        Raises:
            ValueError: If an identity or frame is empty, a coordinate is not
                finite, ``source_point_index`` is negative, the lineage does not
                connect the frames, or an aggregated point claims a single raw index.
        """
        if not self.geometry_id or not self.map_id:
            raise ValueError("geometry_id and map_id must not be empty")
        if not self.map_frame or not self.source_frame:
            raise ValueError("map_frame and source_frame must not be empty")
        if not all(math.isfinite(value) for value in self.coordinates_m):
            raise ValueError(f"coordinates_m must be finite, got {self.coordinates_m!r}")
        if not all(math.isfinite(value) for value in self.source_coordinates_m):
            raise ValueError(
                f"source_coordinates_m must be finite, got {self.source_coordinates_m!r}"
            )
        if self.source_point_index is not None and self.source_point_index < 0:
            raise ValueError("source_point_index must not be negative")
        if not self.transform_lineage.connects(self.map_frame, self.source_frame):
            raise ValueError(
                f"transform_lineage does not connect map frame {self.map_frame!r} to source "
                f"frame {self.source_frame!r}"
            )
        if self.provenance.origin is PointOrigin.AGGREGATED and self.source_point_index is not None:
            raise ValueError(
                "an aggregated point must not carry a source_point_index: it is not one raw "
                "measurement"
            )

    @property
    def reference(self) -> GeometryReference:
        """The compact reference downstream stages hold instead of this point."""
        return GeometryReference(map_id=self.map_id, geometry_id=self.geometry_id)


@dataclass(frozen=True, kw_only=True)
class SpatialIndexMetadata:
    """How a map's spatial index was built, when that affects query behavior.

    Attributes:
        kind: Identity of the index, e.g. ``"linear"`` or ``"voxel_hash"``.
        parameters: Primitive build parameters that affect query results.
        is_derived: ``True`` when the index can be rebuilt from the geometry; the
            authoritative data is always the persisted geometry, and a derived
            index never redefines it.
    """

    kind: str
    parameters: Mapping[str, SpatialIndexParameter]
    is_derived: bool

    def __post_init__(self) -> None:
        """Require an index identity.

        Raises:
            ValueError: If ``kind`` is empty.
        """
        if not self.kind:
            raise ValueError("spatial index kind must not be empty")


@dataclass(frozen=True, kw_only=True)
class GeometricMapProvenance:
    """Run-level traceability of a geometric map.

    Attributes:
        sequence_artifact_id: Canonical sequence the geometry was built from.
        selection_id: Deterministic identity of the sequence selection.
        trajectory_id: Trajectory whose poses placed the geometry.
        state_estimation_run_id: The persisted state-estimation run the
            trajectory was read from, when there is one.
        calibration_identity: Hash of the static calibration used.
        pose_lookup: The pose lookup policy applied at each acquisition time.
        configuration_fingerprint: Hash of the mapping, aggregation and index
            configuration; ``None`` when there is nothing configurable.
        code_version: Code revision that produced the map, when known.
    """

    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    trajectory_id: TrajectoryId
    state_estimation_run_id: StateEstimationRunId | None
    calibration_identity: str | None
    pose_lookup: LookupPolicy
    configuration_fingerprint: str | None = None
    code_version: str | None = None


@dataclass(frozen=True, kw_only=True)
class GeometricMap:
    """Identity and metadata of one persistent, immutable geometric map.

    The map object describes the geometry; it does not embed the points, which
    live in the artifact's storage and are reached through
    :class:`GeometrySource`. It owns no semantics.

    Attributes:
        map_id: Identity of the map artifact.
        frame_id: The global map frame every coordinate is expressed in.
        point_count: Number of geometry elements.
        bounds: Bounds of the geometry, in ``frame_id``.
        source_observation_ids: Physical observations that contributed geometry.
        time_bounds: Acquisition-time range of those observations.
        spatial_index: Index metadata, when an index was built.
        provenance: Run-level traceability.
    """

    map_id: MapId
    frame_id: FrameId
    point_count: int
    bounds: Bounds3D
    source_observation_ids: tuple[SourceObservationId, ...]
    time_bounds: TimeBounds
    spatial_index: SpatialIndexMetadata | None
    provenance: GeometricMapProvenance

    def __post_init__(self) -> None:
        """Validate identity, counts, frames and sources.

        Raises:
            ValueError: If an identity or frame is empty, ``point_count`` is not
                positive, the bounds are in another frame, or the source
                observations are missing or repeated.
        """
        if not self.map_id or not self.frame_id:
            raise ValueError("map_id and frame_id must not be empty")
        if self.point_count < 1:
            raise ValueError(f"point_count must be at least 1, got {self.point_count}")
        if self.bounds.frame_id != self.frame_id:
            raise ValueError(
                f"bounds frame {self.bounds.frame_id!r} differs from the map frame "
                f"{self.frame_id!r}"
            )
        if not self.source_observation_ids:
            raise ValueError("source_observation_ids must list at least one observation")
        if len(set(self.source_observation_ids)) != len(self.source_observation_ids):
            raise ValueError("source_observation_ids must not repeat an observation")

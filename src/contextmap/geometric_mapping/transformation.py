"""Deterministic transformation of scans from their source frame into the map frame.

For a LiDAR scan the chain is::

    P_map = T_map_body(t) · T_body_source · P_source

``T_map_body(t)`` is the pose resolved for the scan by
:mod:`~contextmap.geometric_mapping.inputs`; ``T_body_source`` is the static
extrinsic from the canonical calibration. Both follow the ``T_parent_child``
convention (``p_parent = R · p_child + t``, quaternion ``(x, y, z, w)``, meters),
and every step keeps its direction and its reference so any point can be traced
back to the exact pose and calibration that produced it.

The map-frame coordinate is authoritative and the original source coordinate is
kept next to it, never recomputed from the map. Points whose source coordinates
are not finite are dropped and counted, never repaired. Missing or invalid
transforms raise: nothing is inferred from a frame's name and the source and map
frames are never switched silently.

The batched arithmetic uses NumPy, imported only when a scan is transformed, in
float64 with a fixed evaluation order so results do not depend on a BLAS build.
The public result holds standard-library arrays, not NumPy objects.
See ``src/contextmap/geometric_mapping/docs/transformation.md``.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from contextmap.geometric_mapping.inputs import GeometryInput, GeometryInputPlan
from contextmap.geometric_mapping.models import (
    GeometryPoint,
    GeometryPointProvenance,
    MapId,
    TransformKind,
    TransformLineage,
    TransformStep,
    geometry_id_for,
)
from contextmap.geometric_mapping.motion_correction import MotionCorrectionState
from contextmap.ingestion import FrameId, PointFieldDataType, SourceObservationId
from contextmap.shared import (
    Quaternion,
    SourceTimestamp,
    Vector3,
    compose_rigid,
    is_unit_quaternion,
    quaternion_to_rotation_matrix,
    rotate_vector,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Tolerância absoluta, em metros, para reconstruir P_map de uma trace: a aritmética é float64
# e as coordenadas de LiDAR ficam abaixo de 1e3 m, então o erro de arredondamento é ~1e-12 m.
_DEFAULT_TRACE_TOLERANCE_M = 1e-9


class GeometryTransformError(ValueError):
    """Raised when a scan cannot be transformed without guessing."""


@dataclass(frozen=True, kw_only=True)
class TracedTransform:
    """One factor of the chain, with the numbers that were applied.

    Attributes:
        step: Direction, kind and reference (pose or calibration) of the factor.
        translation_m: Translation of ``T_parent_child`` in meters.
        rotation: Unit quaternion ``(x, y, z, w)`` of ``T_parent_child``.
    """

    step: TransformStep
    translation_m: Vector3
    rotation: Quaternion

    def apply(self, point_m: Vector3) -> Vector3:
        """Express a point of ``step.child_frame`` in ``step.parent_frame``."""
        rotated = rotate_vector(self.rotation, point_m)
        return (
            rotated[0] + self.translation_m[0],
            rotated[1] + self.translation_m[1],
            rotated[2] + self.translation_m[2],
        )


@dataclass(frozen=True, kw_only=True)
class TransformTrace:
    """Everything needed to audit how one point reached the map frame.

    Attributes:
        source_observation_id: The scan the point came from.
        source_point_index: Index of the point within that scan.
        source_frame: Frame of ``source_coordinates_m``.
        map_frame: Frame of ``map_coordinates_m``.
        acquisition_timestamp: When the scan was acquired.
        source_coordinates_m: The original measurement, in meters.
        transforms: The chain applied, outermost first
            (``P_map = T_0 · T_1 · … · P_source``).
        map_coordinates_m: The authoritative map-frame position, in meters.
    """

    source_observation_id: SourceObservationId
    source_point_index: int
    source_frame: FrameId
    map_frame: FrameId
    acquisition_timestamp: SourceTimestamp
    source_coordinates_m: Vector3
    transforms: tuple[TracedTransform, ...]
    map_coordinates_m: Vector3


def verify_transform_trace(
    trace: TransformTrace, *, tolerance_m: float = _DEFAULT_TRACE_TOLERANCE_M
) -> list[str]:
    """Rebuild the map coordinates of a trace from its own chain and compare.

    Args:
        trace: The trace to audit.
        tolerance_m: Largest accepted distance between the rebuilt and the
            recorded map coordinates.

    Returns:
        Human-readable problems; empty means the chain reproduces the map coordinates.
    """
    problems: list[str] = []
    frame = trace.map_frame
    for transform in trace.transforms:
        if transform.step.parent_frame != frame:
            problems.append(
                f"the chain is not contiguous: expected a transform from {frame!r}, "
                f"got {transform.step.parent_frame!r}"
            )
        frame = transform.step.child_frame
    if frame != trace.source_frame:
        problems.append(f"the chain ends in {frame!r} but the point is in {trace.source_frame!r}")

    rebuilt = trace.source_coordinates_m
    for transform in reversed(trace.transforms):
        rebuilt = transform.apply(rebuilt)
    distance = math.dist(rebuilt, trace.map_coordinates_m)
    if distance > tolerance_m:
        problems.append(
            f"the chain places the point at {rebuilt} but the trace records map coordinates "
            f"{trace.map_coordinates_m} ({distance:.3e} m apart)"
        )
    return problems


@dataclass(frozen=True, kw_only=True)
class TransformedScan:
    """The points of one scan in the map frame, next to their original coordinates.

    Coordinates are stored column-wise in standard-library arrays, ``(x, y, z)``
    per kept point, so a long scan is not a list of Python objects. Treat the
    arrays as read-only.

    Attributes:
        observation_id: The physical scan.
        source_frame: Frame of the original coordinates.
        map_frame: Frame of the authoritative coordinates.
        acquisition_timestamp: When the scan was acquired.
        payload_hash: ``"sha256:<hex>"`` of the scan's payload.
        motion_correction: Whether the scan was corrected for platform motion.
        transform_chain: The factors applied, outermost first.
        source_point_count: Points in the scan, kept or not.
        source_point_indices: For each kept point, its index within the scan.
        source_coordinates_m: Original ``(x, y, z)`` of each kept point, in meters.
        map_coordinates_m: Authoritative map-frame ``(x, y, z)`` of each kept point, in meters.
    """

    observation_id: SourceObservationId
    source_frame: FrameId
    map_frame: FrameId
    acquisition_timestamp: SourceTimestamp
    payload_hash: str
    motion_correction: MotionCorrectionState
    transform_chain: tuple[TracedTransform, ...]
    source_point_count: int
    source_point_indices: array[int]
    source_coordinates_m: array[float]
    map_coordinates_m: array[float]

    @property
    def point_count(self) -> int:
        """Points kept after dropping the non-finite ones."""
        return len(self.source_point_indices)

    @property
    def dropped_non_finite_count(self) -> int:
        """Points dropped because a source coordinate was not finite."""
        return self.source_point_count - self.point_count

    @property
    def transform_lineage(self) -> TransformLineage:
        """The chain's steps, shared by every point of the scan."""
        return TransformLineage(steps=tuple(transform.step for transform in self.transform_chain))

    def source_point(self, position: int) -> Vector3:
        """Return the original coordinates of the ``position``-th kept point."""
        return self._coordinates(self.source_coordinates_m, position)

    def map_point(self, position: int) -> Vector3:
        """Return the authoritative map coordinates of the ``position``-th kept point."""
        return self._coordinates(self.map_coordinates_m, position)

    def trace_point(self, position: int) -> TransformTrace:
        """Build the audit trace of the ``position``-th kept point."""
        return TransformTrace(
            source_observation_id=self.observation_id,
            source_point_index=self.source_point_indices[position],
            source_frame=self.source_frame,
            map_frame=self.map_frame,
            acquisition_timestamp=self.acquisition_timestamp,
            source_coordinates_m=self.source_point(position),
            transforms=self.transform_chain,
            map_coordinates_m=self.map_point(position),
        )

    def geometry_point(self, position: int, *, map_id: MapId, geometry_index: int) -> GeometryPoint:
        """Materialize the ``position``-th kept point as a persistent geometry point.

        Args:
            position: Position among the kept points.
            map_id: The map the point belongs to.
            geometry_index: Index the point takes within that map.

        Returns:
            A measured point with both coordinate systems and the scan's lineage.
        """
        return GeometryPoint(
            geometry_id=geometry_id_for(map_id=map_id, index=geometry_index),
            map_id=map_id,
            map_frame=self.map_frame,
            coordinates_m=self.map_point(position),
            source_frame=self.source_frame,
            source_coordinates_m=self.source_point(position),
            source_observation_id=self.observation_id,
            source_point_index=self.source_point_indices[position],
            acquisition_timestamp=self.acquisition_timestamp,
            transform_lineage=self.transform_lineage,
            provenance=GeometryPointProvenance(motion_correction=self.motion_correction),
        )

    def _coordinates(self, column: array[float], position: int) -> Vector3:
        if not 0 <= position < self.point_count:
            raise IndexError(f"position {position} is outside the {self.point_count} kept points")
        offset = 3 * position
        return (column[offset], column[offset + 1], column[offset + 2])


def transform_scans(plan: GeometryInputPlan) -> Iterator[TransformedScan]:
    """Transform the scans of a plan one at a time, in the plan's order.

    Args:
        plan: The assembled geometry inputs.

    Returns:
        A lazy iterator, so only one scan's arrays are alive at a time.

    Raises:
        GeometryTransformError: While iterating, if a scan cannot be transformed.
    """
    for item in plan.inputs:
        yield transform_scan(item, calibration_identity=plan.calibration_identity)


def transform_scan(item: GeometryInput, *, calibration_identity: str | None) -> TransformedScan:
    """Transform one scan into the map frame.

    Args:
        item: The scan with its resolved pose and static transform.
        calibration_identity: Identity of the calibration ``item``'s static
            transform came from; recorded as that step's reference.

    Returns:
        The kept points in the map frame, with their original coordinates, the
        applied chain and how many points were dropped.

    Raises:
        GeometryTransformError: If the frames of the chain do not connect, a
            static transform has no calibration identity, a rotation is not a
            unit quaternion, a translation is not finite, or a map coordinate
            is not finite.
    """
    chain = _transform_chain(item, calibration_identity)
    translation, rotation = _compose(chain)
    scan = item.observation
    source, indices, mapped = _transform_points(item, rotation, translation)
    return TransformedScan(
        observation_id=scan.observation_id,
        source_frame=item.source_frame,
        map_frame=chain[0].step.parent_frame,
        acquisition_timestamp=item.timestamp,
        payload_hash=item.payload_hash,
        motion_correction=item.motion_correction.record.state,
        transform_chain=chain,
        source_point_count=scan.point_count,
        source_point_indices=indices,
        source_coordinates_m=source,
        map_coordinates_m=mapped,
    )


def _transform_chain(
    item: GeometryInput, calibration_identity: str | None
) -> tuple[TracedTransform, ...]:
    pose = item.pose.pose
    chain = [
        TracedTransform(
            step=TransformStep(
                kind=TransformKind.DYNAMIC_POSE,
                parent_frame=pose.parent_frame,
                child_frame=pose.child_frame,
                reference=str(pose.estimate_id),
                source_estimate_ids=item.pose.source_estimate_ids,
            ),
            translation_m=pose.translation_m,
            rotation=pose.orientation,
        )
    ]
    static = item.static_transform
    if static is None:
        if pose.child_frame != item.source_frame:
            raise GeometryTransformError(
                f"scan {item.observation_id!r} is in frame {item.source_frame!r}, not the body "
                f"frame {pose.child_frame!r}, and no static transform connects them"
            )
    else:
        if static.parent_frame != pose.child_frame or static.child_frame != item.source_frame:
            raise GeometryTransformError(
                f"the frame chain of scan {item.observation_id!r} does not connect: the pose is "
                f"{pose.parent_frame!r} <- {pose.child_frame!r} and the static transform is "
                f"{static.parent_frame!r} <- {static.child_frame!r}, but the scan is in "
                f"{item.source_frame!r}"
            )
        if calibration_identity is None:
            raise GeometryTransformError(
                f"scan {item.observation_id!r} needs a static transform but no calibration "
                "identity was given to reference it"
            )
        chain.append(
            TracedTransform(
                step=TransformStep(
                    kind=TransformKind.STATIC_CALIBRATION,
                    parent_frame=static.parent_frame,
                    child_frame=static.child_frame,
                    reference=calibration_identity,
                ),
                translation_m=static.translation,
                rotation=static.rotation,
            )
        )
    for transform in chain:
        _require_valid(transform)
    return tuple(chain)


def _require_valid(transform: TracedTransform) -> None:
    step = transform.step
    if not all(math.isfinite(component) for component in transform.translation_m):
        raise GeometryTransformError(
            f"the translation {transform.translation_m} of {step.child_frame!r} -> "
            f"{step.parent_frame!r} is not finite"
        )
    if not is_unit_quaternion(transform.rotation):
        raise GeometryTransformError(
            f"the rotation {transform.rotation} of {step.child_frame!r} -> {step.parent_frame!r} "
            "is not a unit quaternion"
        )


def _compose(chain: tuple[TracedTransform, ...]) -> tuple[Vector3, Quaternion]:
    translation, rotation = chain[0].translation_m, chain[0].rotation
    for inner in chain[1:]:
        translation, rotation = compose_rigid(
            outer_translation=translation,
            outer_rotation=rotation,
            inner_translation=inner.translation_m,
            inner_rotation=inner.rotation,
        )
    return translation, rotation


def _transform_points(
    item: GeometryInput, rotation: Quaternion, translation: Vector3
) -> tuple[array[float], array[int], array[float]]:
    import numpy as np

    scan, layout = item.observation, item.layout
    code = "<f4" if layout.scalar is PointFieldDataType.FLOAT32 else "<f8"
    record = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [code, code, code],
            "offsets": [layout.x_offset_bytes, layout.y_offset_bytes, layout.z_offset_bytes],
            "itemsize": layout.point_step_bytes,
        }
    )
    points = np.frombuffer(scan.data, dtype=record, count=scan.point_count)
    columns = [points[name].astype(np.float64) for name in ("x", "y", "z")]
    finite = np.isfinite(columns[0]) & np.isfinite(columns[1]) & np.isfinite(columns[2])
    x, y, z = (column[finite] for column in columns)

    # Ordem de avaliação fixa (sem BLAS): o mesmo resultado em qualquer build do NumPy.
    (r00, r01, r02), (r10, r11, r12), (r20, r21, r22) = quaternion_to_rotation_matrix(rotation)
    map_x = r00 * x + r01 * y + r02 * z + translation[0]
    map_y = r10 * x + r11 * y + r12 * z + translation[1]
    map_z = r20 * x + r21 * y + r22 * z + translation[2]
    if not (np.isfinite(map_x).all() and np.isfinite(map_y).all() and np.isfinite(map_z).all()):
        raise GeometryTransformError(
            f"scan {item.observation_id!r} produced a map coordinate that is not finite"
        )

    source = np.column_stack((x, y, z))
    mapped = np.column_stack((map_x, map_y, map_z))
    return (
        _float_array(source),
        _index_array(np.flatnonzero(finite)),
        _float_array(mapped),
    )


def _float_array(values: NDArray[np.float64]) -> array[float]:
    result: array[float] = array("d")
    result.frombytes(values.tobytes())
    return result


def _index_array(values: NDArray[np.intp]) -> array[int]:
    result: array[int] = array("q")
    result.frombytes(values.astype("int64").tobytes())
    return result

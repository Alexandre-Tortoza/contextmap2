"""Packed, random-access storage of a map's geometry and the boundary that reads it.

The authoritative geometry of a map is one flat payload of fixed-size,
little-endian records, one per geometry element, in the order the geometry was
accumulated. The record at position ``i`` is the geometry whose identity is
``geometry_id_for(map_id=..., index=i)``, so resolving a reference is arithmetic
on the identity and needs no lookup table. A record is::

    <3d3dIqI   map x y z (m) | source x y z (m) | scan ordinal | source point index | count

``source point index`` is ``-1`` for an aggregated point and ``count`` is the
number of measurements behind the point (``1`` for a raw measurement). Everything
that is shared by the points of one scan (source observation, frame, timestamp,
the transform chain with its numbers, the motion-correction state) lives once in
the scan table, the *source index*: :class:`ScanRecord`.

Nothing here needs NumPy, ROS or a point-cloud library: a persisted map is
readable with the standard library. See
``src/contextmap/geometric_mapping/docs/accumulation.md``.
"""

from __future__ import annotations

import dataclasses
import math
import mmap
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from contextmap.geometric_mapping.models import (
    Bounds3D,
    GeometricMap,
    GeometryPoint,
    GeometryPointProvenance,
    GeometryReference,
    PointOrigin,
    TransformLineage,
    geometry_id_for,
    geometry_index_of,
)
from contextmap.geometric_mapping.motion_correction import MotionCorrectionState
from contextmap.geometric_mapping.transformation import TracedTransform
from contextmap.ingestion import FrameId, SourceObservationId
from contextmap.shared import SourceTimestamp

PACKED_POINT = struct.Struct("<3d3dIqI")
"""One geometry record: map xyz, source xyz, scan ordinal, source point index, count."""

AGGREGATED_SOURCE_INDEX = -1
"""Value of the source point index of an aggregated point, which is not one raw measurement."""

# Registros lidos por vez, para o iterador não materializar o payload inteiro.
_CHUNK_POINTS = 4096


@dataclass(frozen=True, kw_only=True)
class ScanRecord:
    """One accumulated scan: the shared origin of a contiguous run of geometry.

    Attributes:
        ordinal: Position of the scan in the source index, in accumulation order.
        observation_id: The physical observation.
        source_frame: Frame of the scan's original coordinates.
        acquisition_timestamp: When the scan was acquired.
        payload_hash: ``"sha256:<hex>"`` of the scan's payload.
        motion_correction: Whether the scan was corrected for platform motion.
        transform_chain: The factors applied, outermost first, with their numbers.
        source_point_count: Points in the scan, kept or not.
        dropped_non_finite_count: Points dropped because a source coordinate was
            not finite.
        first_geometry_index: Index of the scan's first geometry element.
        geometry_count: Geometry elements persisted from the scan (fewer than the
            kept points when they were aggregated).
        bounds: Tight map-frame bounds of those elements; ``None`` when the scan
            contributed none. Derived from the geometry, never authoritative.
    """

    ordinal: int
    observation_id: SourceObservationId
    source_frame: FrameId
    acquisition_timestamp: SourceTimestamp
    payload_hash: str
    motion_correction: MotionCorrectionState
    transform_chain: tuple[TracedTransform, ...]
    source_point_count: int
    dropped_non_finite_count: int
    first_geometry_index: int
    geometry_count: int
    bounds: Bounds3D | None

    def __post_init__(self) -> None:
        """Validate the counts and that bounds exist exactly when geometry does.

        Raises:
            ValueError: If a count or index is negative, more points were
                dropped than the scan had, or ``bounds`` disagrees with
                ``geometry_count``.
        """
        if self.ordinal < 0 or self.first_geometry_index < 0 or self.geometry_count < 0:
            raise ValueError(
                "scan ordinal, first_geometry_index and geometry_count must not be negative"
            )
        if not 0 <= self.dropped_non_finite_count <= self.source_point_count:
            raise ValueError(
                f"dropped_non_finite_count {self.dropped_non_finite_count} must be within "
                f"the {self.source_point_count} points of the scan"
            )
        if (self.bounds is None) != (self.geometry_count == 0):
            raise ValueError("a scan has bounds exactly when it contributed geometry")

    @property
    def transform_lineage(self) -> TransformLineage:
        """The chain's steps, shared by every point of the scan."""
        return TransformLineage(steps=tuple(transform.step for transform in self.transform_chain))


class PackedGeometry:
    """Read-only geometry of one map, over a packed payload and its source index.

    Implements :class:`~contextmap.geometric_mapping.GeometrySource`. The payload
    may be ``bytes`` or a memory map; nothing is copied.
    """

    def __init__(
        self,
        *,
        geometric_map: GeometricMap,
        scans: Sequence[ScanRecord],
        records: bytes | bytearray | memoryview | mmap.mmap,
    ) -> None:
        """Open a map's geometry.

        Args:
            geometric_map: The map's metadata.
            scans: The source index, in accumulation order.
            records: The packed geometry payload.

        Raises:
            ValueError: If the payload does not have exactly one record per
                point of the map, or the source index does not tile the geometry.
        """
        payload = memoryview(records)
        expected = geometric_map.point_count * PACKED_POINT.size
        if len(payload) != expected:
            raise ValueError(
                f"the geometry payload has {len(payload)} bytes but {geometric_map.point_count} "
                f"points need {expected}"
            )
        next_index = 0
        for position, scan in enumerate(scans):
            if scan.ordinal != position or scan.first_geometry_index != next_index:
                raise ValueError(
                    f"the source index does not tile the geometry: scan {position} "
                    f"({scan.observation_id!r}) starts at {scan.first_geometry_index}, "
                    f"expected {next_index}"
                )
            next_index += scan.geometry_count
        if next_index != geometric_map.point_count:
            raise ValueError(
                f"the source index covers {next_index} points but the map has "
                f"{geometric_map.point_count}"
            )
        self._map = geometric_map
        self._scans = tuple(scans)
        self._scan_by_observation = {scan.observation_id: scan for scan in self._scans}
        self._records = payload

    @property
    def geometric_map(self) -> GeometricMap:
        """The map's identity, frame, bounds and provenance."""
        return self._map

    @property
    def scans(self) -> tuple[ScanRecord, ...]:
        """The source index: one entry per accumulated scan."""
        return self._scans

    def scan_record(self, observation_id: SourceObservationId) -> ScanRecord:
        """Return the source-index entry of an accumulated observation.

        Raises:
            KeyError: If the observation was not accumulated into this map.
        """
        try:
            return self._scan_by_observation[observation_id]
        except KeyError:
            raise KeyError(
                f"observation {observation_id!r} is not in map {self._map.map_id!r}"
            ) from None

    def references_for(self, observation_id: SourceObservationId) -> Iterator[GeometryReference]:
        """Iterate the references of the geometry that came from one observation.

        Raises:
            KeyError: If the observation was not accumulated into this map.
        """
        scan = self.scan_record(observation_id)
        for index in range(
            scan.first_geometry_index, scan.first_geometry_index + scan.geometry_count
        ):
            yield GeometryReference(
                map_id=self._map.map_id,
                geometry_id=geometry_id_for(map_id=self._map.map_id, index=index),
            )

    def get(self, reference: GeometryReference) -> GeometryPoint:
        """Resolve a reference to its authoritative geometry.

        Raises:
            KeyError: If the reference does not belong to this map.
            ValueError: If the stored record is inconsistent with the source index.
        """
        if reference.map_id != self._map.map_id:
            raise KeyError(
                f"reference belongs to map {reference.map_id!r}, not {self._map.map_id!r}"
            )
        try:
            index = geometry_index_of(map_id=self._map.map_id, geometry_id=reference.geometry_id)
        except ValueError as error:
            raise KeyError(str(error)) from error
        if index >= self._map.point_count:
            raise KeyError(
                f"geometry index {index} is past the end of map {self._map.map_id!r} "
                f"({self._map.point_count} points)"
            )
        row = PACKED_POINT.unpack_from(self._records, index * PACKED_POINT.size)
        return self._point_from_row(index, row)

    def iter_geometry(self) -> Iterator[GeometryPoint]:
        """Iterate every geometry element in index order."""
        for index, row in self._iter_rows(0, self._map.point_count):
            yield self._point_from_row(index, row)

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        """Iterate the geometry inside a box, boundaries included, in index order.

        Scans whose recorded bounds miss the box are not read, and a scan wholly
        inside the box is returned without testing each point. The result is
        exactly the geometry a full filter of :meth:`iter_geometry` would return:
        the index only decides what is read, never what the geometry is.

        Raises:
            ValueError: If ``bounds`` is expressed in another frame.
        """
        if bounds.frame_id != self._map.frame_id:
            raise ValueError(
                f"bounds are expressed in frame {bounds.frame_id!r} but the map is in "
                f"{self._map.frame_id!r}; frames are never reinterpreted"
            )
        low, high = bounds.minimum_m, bounds.maximum_m
        frame = self._map.frame_id
        for scan in self._scans:
            if scan.bounds is None or not scan.bounds.intersects(bounds):
                continue
            first, stop = scan.first_geometry_index, scan.first_geometry_index + scan.geometry_count
            wholly_inside = bounds.contains(scan.bounds.minimum_m, frame_id=frame) and (
                bounds.contains(scan.bounds.maximum_m, frame_id=frame)
            )
            for index, row in self._iter_rows(first, stop):
                if wholly_inside or (
                    low[0] <= row[0] <= high[0]
                    and low[1] <= row[1] <= high[1]
                    and low[2] <= row[2] <= high[2]
                ):
                    yield self._point_from_row(index, row)

    def rebuild_scan_bounds(self) -> tuple[ScanRecord, ...]:
        """Recompute every scan's bounds from the geometry payload.

        The bounds in the source index are a derived index: this rebuilds them
        without touching a geometry identity or a coordinate.

        Returns:
            The source index with ``bounds`` recomputed from the geometry.
        """
        return tuple(
            dataclasses.replace(
                scan,
                bounds=self._bounds_of(
                    scan.first_geometry_index, scan.first_geometry_index + scan.geometry_count
                ),
            )
            for scan in self._scans
        )

    def verify_index(self) -> list[str]:
        """Check the derived bounds against the geometry they summarize.

        Returns:
            Human-readable problems; empty when every scan's bounds and the map's
            bounds are the tight envelope of the geometry. A problem here means
            the derived index is stale or corrupt; the geometry stays authoritative.
        """
        problems: list[str] = []
        rebuilt = self.rebuild_scan_bounds()
        for recorded, actual in zip(self._scans, rebuilt, strict=True):
            if recorded.bounds != actual.bounds:
                problems.append(
                    f"scan {recorded.ordinal} ({recorded.observation_id!r}) records bounds "
                    f"{recorded.bounds} but its geometry spans {actual.bounds}"
                )
        boxes = [scan.bounds for scan in rebuilt if scan.bounds is not None]
        envelope = Bounds3D(
            frame_id=self._map.frame_id,
            minimum_m=(
                min(box.minimum_m[0] for box in boxes),
                min(box.minimum_m[1] for box in boxes),
                min(box.minimum_m[2] for box in boxes),
            ),
            maximum_m=(
                max(box.maximum_m[0] for box in boxes),
                max(box.maximum_m[1] for box in boxes),
                max(box.maximum_m[2] for box in boxes),
            ),
        )
        if envelope != self._map.bounds:
            problems.append(
                f"map bounds {self._map.bounds} differ from the envelope of the geometry {envelope}"
            )
        return problems

    def _bounds_of(self, first: int, stop: int) -> Bounds3D | None:
        if stop <= first:
            return None
        low = [math.inf] * 3
        high = [-math.inf] * 3
        for _, row in self._iter_rows(first, stop):
            for axis in range(3):
                low[axis] = min(low[axis], row[axis])
                high[axis] = max(high[axis], row[axis])
        return Bounds3D(
            frame_id=self._map.frame_id,
            minimum_m=(low[0], low[1], low[2]),
            maximum_m=(high[0], high[1], high[2]),
        )

    def _iter_rows(self, first: int, stop: int) -> Iterator[tuple[int, tuple[Any, ...]]]:
        for start in range(first, stop, _CHUNK_POINTS):
            end = min(start + _CHUNK_POINTS, stop)
            view = self._records[start * PACKED_POINT.size : end * PACKED_POINT.size]
            for offset, row in enumerate(PACKED_POINT.iter_unpack(view)):
                yield start + offset, row

    def _point_from_row(self, index: int, row: tuple[Any, ...]) -> GeometryPoint:
        map_x, map_y, map_z, source_x, source_y, source_z, ordinal, source_index, count = row
        if not 0 <= ordinal < len(self._scans):
            raise ValueError(f"geometry {index} names scan ordinal {ordinal}, which does not exist")
        scan = self._scans[ordinal]
        if not scan.first_geometry_index <= index < scan.first_geometry_index + scan.geometry_count:
            raise ValueError(
                f"geometry {index} names scan ordinal {ordinal}, whose range is "
                f"[{scan.first_geometry_index}, {scan.first_geometry_index + scan.geometry_count})"
            )
        if count < 1:
            raise ValueError(f"geometry {index} has no contributing measurement")
        if count == 1:
            if source_index < 0:
                raise ValueError(f"geometry {index} is a raw measurement without a point index")
            provenance = GeometryPointProvenance(motion_correction=scan.motion_correction)
            point_index: int | None = int(source_index)
        else:
            rule = self._map.aggregation_rule
            if rule is None:
                raise ValueError(
                    f"geometry {index} is aggregated from {count} measurements but the map "
                    "declares no aggregation rule"
                )
            provenance = GeometryPointProvenance(
                origin=PointOrigin.AGGREGATED,
                aggregation_rule=rule,
                contributing_point_count=int(count),
                motion_correction=scan.motion_correction,
            )
            point_index = None
        return GeometryPoint(
            geometry_id=geometry_id_for(map_id=self._map.map_id, index=index),
            map_id=self._map.map_id,
            map_frame=self._map.frame_id,
            coordinates_m=(float(map_x), float(map_y), float(map_z)),
            source_frame=scan.source_frame,
            source_coordinates_m=(float(source_x), float(source_y), float(source_z)),
            source_observation_id=scan.observation_id,
            source_point_index=point_index,
            acquisition_timestamp=scan.acquisition_timestamp,
            transform_lineage=scan.transform_lineage,
            provenance=provenance,
        )

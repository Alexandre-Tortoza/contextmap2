"""Which map geometry one camera frame evaluates, chosen before any exact projection.

Projecting every element of a persistent map into every camera frame is sound but does
not scale: the cost and the working set grow with the whole map, while a frame can only
ever see what is near the camera. On the real corridor-02 map (26.63 M points) that cost
about 1.17 GB of retained arrays and 12 s per frame, and the geometry that actually
became evidence was never farther than 5.5 m from the camera -- indoors, occlusion
removes everything past the first surface long before range does.

This module inserts one conservative step before the exact chain::

    camera optical centre at t_rgb
        -> axis-aligned box around the candidate sphere   (Geometric Mapping query)
        -> exact test  ||P - C|| <= max_range_m           (this module)
        -> candidates, with the global index of every row
        -> exact map->camera transform, exact camera model, visibility, membership

The candidate region is a **sphere centred on the camera optical centre**, never the box
the query is expressed with: an axis-aligned box would make a point just outside a face
nearer than one inside a corner, so the box only ever prunes what is *read*.

What the sphere guarantees depends on the camera's
:class:`~contextmap.sensor_association.DepthMetric`, because that is the quantity the
occlusion rule compares:

``RAY_RANGE`` (fisheye, MEI)
    **Exact.** Depth *is* range, so every excluded element is strictly farther than every
    retained one under the compared quantity, and a point's own cell window always
    contains itself. An excluded element can therefore never have been the nearest depth
    support of a retained one, and for every retained element the projection, the pixel,
    the support, the occlusion decision, the membership and the geometry support are
    exactly what a full-map projection produces.

``OPTICAL_AXIS`` (pinhole)
    **Not exact.** The rule compares ``z``, and ``z <= range``, so an element the range
    policy excludes may still have had a smaller ``z`` than a retained one -- it only has
    to sit at a larger field angle. When the two fall in one cell window, culling removes
    a support the full map would have used. The direction is fixed: removing elements can
    only *raise* a window's minimum, so this can only make a retained element **less**
    occluded. It never drops geometry the full map associated, but it does mean the range
    policy redefines the *support* population, not only the evaluated one.

    The regime is reachable only with a coarse grid. A measured probe put the smallest
    window reach that made it fire at about 24 px, against the 12 px of the corridor-02
    run's ``cell_size_px=4, neighborhood_radius_cells=2``; that is a probe, not a proven
    bound. ``test_candidate_equivalence.py`` pins both behaviours.

The declared difference is therefore the evaluated population -- elements beyond
``max_range_m`` are not evaluated, and the per-frame counts say so -- plus, for an
optical-axis camera, the support population that follows from it.

``max_range_m = None`` selects the whole map and is exactly the full-map behaviour, kept
as the baseline an equivalence experiment compares against.

Identity is global. A :class:`CandidateGeometryCloud` row is a position in an array and carries
the global geometry index it came from; row ``i`` is never geometry ``i``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from contextmap.geometric_mapping import (
    DEFAULT_BLOCK_POINTS,
    Bounds3D,
    GeometricMap,
    GeometryBlockSource,
    GeometryReference,
    geometry_id_for,
)
from contextmap.shared import Vector3

if TYPE_CHECKING:
    from numpy.typing import NDArray

CANDIDATE_GEOMETRY_POLICY_ID = "camera-range-candidates-v1"
"""Versioned identity of the candidate rule: the geometry whose distance to the camera
optical centre is at most ``max_range_m``, or the whole map when it is ``None``."""


@dataclass(frozen=True, kw_only=True)
class CandidateGeometryPolicy:
    """Which geometry a frame evaluates, and how much of it is read at a time.

    There is no default range: how far geometry may still be evidence depends on the
    sensor, the scene and the question being asked, so it is chosen and recorded per run.

    Attributes:
        max_range_m: Largest distance, in meters, from the camera optical centre at which
            a map element is still evaluated. ``None`` evaluates the whole map, which is
            the full-map baseline.
        block_points: Rows read per block while selecting, which bounds the selection's
            own working set independently of the map's size.
    """

    policy_id: ClassVar[str] = CANDIDATE_GEOMETRY_POLICY_ID

    max_range_m: float | None
    block_points: int = DEFAULT_BLOCK_POINTS

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If ``max_range_m`` is not a positive finite distance (when it is
                given), or ``block_points`` is not positive.
        """
        if self.max_range_m is not None and (
            not math.isfinite(self.max_range_m) or self.max_range_m <= 0
        ):
            raise ValueError(
                f"max_range_m must be a positive finite distance or None, got {self.max_range_m!r}"
            )
        if self.block_points < 1:
            raise ValueError(f"block_points must be at least 1, got {self.block_points}")

    def to_record(self) -> dict[str, Any]:
        """Return the policy as JSON primitives, with its versioned identity."""
        return {
            "policy_id": self.policy_id,
            "max_range_m": self.max_range_m,
            "block_points": self.block_points,
        }

    def fingerprint(self) -> str:
        """Return ``"sha256:<hex>"`` over the policy identity and parameters."""
        payload = json.dumps(self.to_record(), sort_keys=True).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, kw_only=True, eq=False)
class CandidateGeometryCloud:
    """The geometry one frame evaluates, ready for vectorized projection.

    Attributes:
        geometric_map: The map's identity, frame and provenance.
        coordinates_m: ``(N, 3)`` authoritative map-frame coordinates, in meters.
        global_indices: ``(N,)`` global index in the map of each row, strictly
            increasing. Row ``i`` is the geometry ``global_indices[i]``, never ``i``.
    """

    geometric_map: GeometricMap
    coordinates_m: NDArray[Any]
    global_indices: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate the arrays and the identity they carry.

        Raises:
            ValueError: If the coordinates are not ``(N, 3)``, there is not one index per
                row, an index falls outside the map, or the indices are not strictly
                increasing. Strict increase is what lets a persisted geometry support be
                looked up by binary search and keeps its order that of the map.
        """
        import numpy as np

        if self.coordinates_m.ndim != 2 or self.coordinates_m.shape[1] != 3:
            raise ValueError(
                f"coordinates_m must have shape (N, 3), got {self.coordinates_m.shape}"
            )
        if self.global_indices.shape != (self.coordinates_m.shape[0],):
            raise ValueError(
                f"a candidate cloud needs one global index per row: "
                f"{self.global_indices.shape} indices for {self.coordinates_m.shape[0]} rows"
            )
        if self.global_indices.size:
            if int(self.global_indices.min()) < 0 or int(self.global_indices.max()) >= (
                self.geometric_map.point_count
            ):
                raise ValueError(
                    f"candidate indices name geometry outside the map's "
                    f"{self.geometric_map.point_count} elements"
                )
            if not bool((np.diff(self.global_indices) > 0).all()):
                raise ValueError("global_indices must be strictly increasing")

    def __len__(self) -> int:
        """Return the number of candidate elements."""
        return int(self.coordinates_m.shape[0])

    def reference(self, row: int) -> GeometryReference:
        """Return the persistent reference of one candidate row.

        Args:
            row: Position in this cloud, which is **not** a geometry index.
        """
        map_id = self.geometric_map.map_id
        index = int(self.global_indices[row])
        return GeometryReference(
            map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index)
        )

    def references(self, rows: Iterable[int]) -> tuple[GeometryReference, ...]:
        """Return the references of the given rows, in the given order."""
        return tuple(self.reference(int(row)) for row in rows)

    def rows_for(self, global_indices: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Locate map elements in this cloud by their global index.

        Args:
            global_indices: ``(K,)`` global indices to look up, in any order.

        Returns:
            ``(rows, found)``: the row of each requested element and whether the cloud
            holds it at all. A row is meaningless where ``found`` is ``False``; an
            element the candidate policy excluded is reported, never silently mapped
            onto a neighbour.
        """
        import numpy as np

        wanted = np.asarray(global_indices)
        count = self.global_indices.shape[0]
        if count == 0:
            return np.zeros(wanted.shape, dtype=np.int64), np.zeros(wanted.shape, dtype=bool)
        rows = np.clip(np.searchsorted(self.global_indices, wanted), 0, count - 1)
        return rows, self.global_indices[rows] == wanted


@dataclass(frozen=True, kw_only=True)
class CandidateGeometryReport:
    """What the candidate step of one frame decided and what it cost.

    It is the auditable half of a selection and holds no array, so a frame's record can
    travel downstream and be persisted without keeping the candidate coordinates alive.

    Attributes:
        policy: The policy applied.
        map_point_count: Elements the whole map holds.
        queried_count: Elements the spatial query returned, before the exact range test;
            it is how much the enclosing box over-selected.
        candidate_count: Elements the frame actually evaluates.
        query_seconds: Wall-clock time the selection took.
    """

    policy: CandidateGeometryPolicy
    map_point_count: int
    queried_count: int
    candidate_count: int
    query_seconds: float

    def __post_init__(self) -> None:
        """Validate the counts.

        Raises:
            ValueError: If a count is negative, or the counts do not narrow from the map
                through the query to the candidates.
        """
        if min(self.map_point_count, self.queried_count, self.candidate_count) < 0:
            raise ValueError("candidate counts must not be negative")
        if not self.candidate_count <= self.queried_count <= self.map_point_count:
            raise ValueError(
                f"candidate selection must narrow: {self.candidate_count} candidates of "
                f"{self.queried_count} queried of {self.map_point_count} map elements"
            )

    @property
    def candidate_fraction(self) -> float:
        """Share of the map this frame evaluates; ``0.0`` for an empty map."""
        return self.candidate_count / self.map_point_count if self.map_point_count else 0.0

    def to_record(self) -> dict[str, Any]:
        """Return the deterministic per-frame candidate facts as JSON primitives.

        ``query_seconds`` is deliberately absent: it is a wall-clock measurement, not a
        fact about the geometry, and a contractual record that held it would stop a rerun
        from reproducing the artifact's hashes. Timing is reported in ``metrics/``.
        """
        return {
            "policy": self.policy.to_record(),
            "policy_fingerprint": self.policy.fingerprint(),
            "map_point_count": self.map_point_count,
            "queried_count": self.queried_count,
            "candidate_count": self.candidate_count,
            "candidate_fraction": self.candidate_fraction,
        }


@dataclass(frozen=True, kw_only=True, eq=False)
class CandidateGeometrySelection:
    """The candidates of one frame, with the report of how they were chosen.

    Attributes:
        cloud: The selected geometry, with its global identities.
        report: The counts, the policy and the cost of the selection.
    """

    cloud: CandidateGeometryCloud
    report: CandidateGeometryReport

    def __post_init__(self) -> None:
        """Validate that the report describes the cloud.

        Raises:
            ValueError: If the report's candidate count is not the cloud's size.
        """
        if self.report.candidate_count != len(self.cloud):
            raise ValueError(
                f"the report counts {self.report.candidate_count} candidates but the cloud "
                f"holds {len(self.cloud)}"
            )

    @property
    def candidate_count(self) -> int:
        """Elements this frame evaluates."""
        return len(self.cloud)

    @property
    def candidate_fraction(self) -> float:
        """Share of the map this frame evaluates; ``0.0`` for an empty map."""
        return self.report.candidate_fraction

    @property
    def map_point_count(self) -> int:
        """Elements the whole map holds."""
        return self.report.map_point_count

    @property
    def queried_count(self) -> int:
        """Elements the spatial query returned, before the exact range test."""
        return self.report.queried_count

    @property
    def query_seconds(self) -> float:
        """Wall-clock time the selection took."""
        return self.report.query_seconds


def select_candidate_geometry(
    source: GeometryBlockSource, *, camera_center_m: Vector3, policy: CandidateGeometryPolicy
) -> CandidateGeometrySelection:
    """Choose the geometry one camera frame evaluates.

    Args:
        source: The block read boundary of the persistent map.
        camera_center_m: The camera optical centre, in the map frame, in meters.
        policy: The candidate rule.

    Returns:
        The candidates, their global identities and what the selection cost.
    """
    import numpy as np

    started = time.perf_counter()
    geometric_map = source.geometric_map
    centre = np.array(camera_center_m, dtype=np.float64)
    radius = policy.max_range_m
    bounds = None if radius is None else _enclosing_box(geometric_map, centre, radius)
    index_parts: list[NDArray[Any]] = []
    coordinate_parts: list[NDArray[Any]] = []
    queried = 0
    for block in source.iter_blocks(bounds=bounds, block_points=policy.block_points):
        queried += len(block)
        if radius is None:
            index_parts.append(block.indices)
            coordinate_parts.append(block.coordinates_m)
            continue
        offsets = block.coordinates_m - centre
        # Comparar distâncias ao quadrado evita a raiz e não muda quem entra na esfera.
        within = np.einsum("ij,ij->i", offsets, offsets) <= radius * radius
        if not within.any():
            continue
        index_parts.append(block.indices[within])
        coordinate_parts.append(block.coordinates_m[within])
    indices = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int64)
    coordinates = (
        np.concatenate(coordinate_parts) if coordinate_parts else np.empty((0, 3), dtype=np.float64)
    )
    cloud = CandidateGeometryCloud(
        geometric_map=geometric_map, coordinates_m=coordinates, global_indices=indices
    )
    return CandidateGeometrySelection(
        cloud=cloud,
        report=CandidateGeometryReport(
            policy=policy,
            map_point_count=geometric_map.point_count,
            queried_count=queried,
            candidate_count=len(cloud),
            query_seconds=time.perf_counter() - started,
        ),
    )


def _enclosing_box(geometric_map: GeometricMap, centre: NDArray[Any], radius: float) -> Bounds3D:
    """Return the axis-aligned box that encloses the candidate sphere.

    The box is what the spatial query can prune with; it over-selects on purpose, and
    :func:`select_candidate_geometry` narrows it to the sphere. Over-selection is safe, false
    exclusion is not.
    """
    low = centre - radius
    high = centre + radius
    return Bounds3D(
        frame_id=geometric_map.frame_id,
        minimum_m=(float(low[0]), float(low[1]), float(low[2])),
        maximum_m=(float(high[0]), float(high[1]), float(high[2])),
    )

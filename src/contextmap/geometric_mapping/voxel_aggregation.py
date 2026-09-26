"""Inter-scan voxel aggregation: a derived, provenance-preserving view of a raw map.

A raw :class:`~contextmap.geometric_mapping.GeometricMap` persists every accepted
measurement, so a platform that revisits (or stands still in) one place keeps
accumulating near-duplicate points. This module derives, from such a raw map, one
aggregate per occupied voxel of an explicit :class:`VoxelGridSpec`: the centroid of
the raw points in the voxel, the sufficient statistics to update it incrementally, and
the compact lineage that recovers exactly which raw points it summarizes.

It is **not** :class:`~contextmap.geometric_mapping.ScanVoxelPolicy`. That policy runs
inside the mapping, merges only points of the *same* scan and replaces them in the
canonical map. This one merges points of *different* scans, runs offline over a
finished raw map, never modifies it, and produces a separate derived representation.

The aggregation is geometry only. It never merges labels, claims, embeddings,
semantic hypotheses or instance identity; a derived aggregate is evidence about
*where* raw measurements were, not a belief about *what* is there.

Counts keep distinct meanings (see :class:`VoxelAggregate`): ``point_count`` counts
raw measurements, ``scan_count`` accumulated scans (source-index entries) and
``observation_count`` physical source observations. Within one raw map a scan is one
observation, so the last two coincide there; they are kept apart so the contract does
not assume that bijection.

Order and chunking of the input never change keys, counts, bounds, times or lineage.
They can change the floating-point sums by rounding only, within the bound documented
in :func:`centroid_tolerance_m`. See
``src/contextmap/geometric_mapping/docs/inter-scan-aggregation.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.geometric_mapping.bulk_geometry import (
    DEFAULT_BLOCK_POINTS,
    GeometryBlock,
    GeometryBlockSource,
)
from contextmap.geometric_mapping.geometry_storage import PackedGeometry, ScanRecord
from contextmap.geometric_mapping.models import (
    Bounds3D,
    GeometricMap,
    GeometricMapProvenance,
    GeometryReference,
    MapId,
    geometry_id_for,
)
from contextmap.ingestion import FrameId
from contextmap.shared import SourceTimestamp, Vector3

if TYPE_CHECKING:
    from numpy.typing import NDArray

VOXEL_INDEXING = "floor-half-open"
"""Voxel ``k`` covers ``[origin + k * cell_m, origin + (k + 1) * cell_m)`` on every axis."""

INTER_SCAN_VOXEL_POLICY_ID = "inter-scan-voxel-centroid"
"""Identity of the aggregation rule, independent of its parameters."""

INTER_SCAN_VOXEL_POLICY_VERSION = "0.1.0"
"""Version of what the rule computes; a change of semantics requires a new version."""

_REPRESENTATIVE = "centroid"
_STATISTICS = (
    "point_count",
    "scan_count",
    "observation_count",
    "offset_sum_m",
    "minimum_m",
    "maximum_m",
    "first_observed_ns",
    "last_observed_ns",
)
_LINEAGE = "scan-ordinal-and-point-count"

# Uma chave só é exata enquanto o quociente cabe na mantissa do float64.
_MAX_EXACT_KEY = 2**53
_EPSILON = 2.0**-52
# Linhas pendentes antes de consolidar o estado: limita a memória sem ordenar a cada bloco.
_CONSOLIDATION_ROWS = 4_000_000


class VoxelAggregationError(ValueError):
    """Raised when a raw map cannot be aggregated without breaking an invariant."""


@dataclass(frozen=True, kw_only=True)
class VoxelGridSpec:
    """The identity of a voxel grid: frame, origin, resolution and indexing convention.

    Attributes:
        frame_id: The map frame the grid is laid in; it must be the raw map's frame.
        origin_m: The corner of voxel ``(0, 0, 0)``, in meters, in ``frame_id``.
        cell_m: Voxel edge length, in meters.
    """

    frame_id: FrameId
    origin_m: Vector3
    cell_m: float

    def __post_init__(self) -> None:
        """Require a usable grid.

        Raises:
            ValueError: If the frame is empty, the origin is not finite, or ``cell_m``
                is not positive and finite.
        """
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if len(self.origin_m) != 3 or not all(math.isfinite(value) for value in self.origin_m):
            raise ValueError(f"origin_m must be three finite coordinates, got {self.origin_m}")
        if not (math.isfinite(self.cell_m) and self.cell_m > 0):
            raise ValueError(f"cell_m must be positive and finite, got {self.cell_m}")

    @property
    def indexing(self) -> str:
        """The indexing convention; part of the grid identity."""
        return VOXEL_INDEXING

    def keys_of(self, coordinates_m: NDArray[Any]) -> NDArray[Any]:
        """Return the voxel key of every coordinate.

        Args:
            coordinates_m: ``(N, 3)`` coordinates in ``frame_id``, in meters.

        Returns:
            ``(N, 3)`` ``int64`` keys, ``floor((p - origin) / cell_m)`` per axis.

        Raises:
            VoxelAggregationError: If a coordinate is not finite or so far from the
                origin, relative to ``cell_m``, that its key would not be exact.
        """
        import numpy as np

        quotients = (np.asarray(coordinates_m, dtype=np.float64) - self._origin()) / self.cell_m
        floors = np.floor(quotients)
        if floors.size and not (
            np.isfinite(floors).all() and np.abs(floors).max() < _MAX_EXACT_KEY
        ):
            raise VoxelAggregationError(
                f"a coordinate is not finite or too far from the grid origin for an exact "
                f"voxel key at cell_m={self.cell_m!r}"
            )
        keys: NDArray[Any] = floors.astype(np.int64)
        return keys

    def anchors_of(self, keys: NDArray[Any]) -> NDArray[Any]:
        """Return the lower corner of every voxel, ``origin + key * cell_m``, in meters."""
        import numpy as np

        anchors: NDArray[Any] = self._origin() + np.asarray(keys, dtype=np.float64) * self.cell_m
        return anchors

    def to_record(self) -> dict[str, Any]:
        """Describe the grid exactly, for configuration and fingerprints."""
        return {
            "frame_id": str(self.frame_id),
            "origin_m": [float(value) for value in self.origin_m],
            "cell_m": self.cell_m,
            "indexing": self.indexing,
        }

    def _origin(self) -> NDArray[Any]:
        import numpy as np

        origin: NDArray[Any] = np.array(self.origin_m, dtype=np.float64)
        return origin


@dataclass(frozen=True, kw_only=True)
class InterScanVoxelPolicy:
    """Aggregate the raw points of every scan that fall in one voxel into their centroid.

    Only the grid varies between executions of this policy version; the representative
    (the centroid), the statistics and the lineage granularity are fixed by
    :data:`INTER_SCAN_VOXEL_POLICY_VERSION`.

    Attributes:
        grid: The voxel grid.
    """

    grid: VoxelGridSpec

    @property
    def rule(self) -> str:
        """The rule a derived map declares, exact in the voxel size."""
        return f"{INTER_SCAN_VOXEL_POLICY_ID}-{self.grid.cell_m!r}m"

    def to_record(self) -> dict[str, Any]:
        """Describe everything that changes the derived representation."""
        return {
            "policy_id": INTER_SCAN_VOXEL_POLICY_ID,
            "policy_version": INTER_SCAN_VOXEL_POLICY_VERSION,
            "grid": self.grid.to_record(),
            "representative": _REPRESENTATIVE,
            "statistics": list(_STATISTICS),
            "lineage": _LINEAGE,
            "rule": self.rule,
        }

    def fingerprint(self) -> str:
        """Hash the policy: equal for equal grids, different when any grid identity changes.

        Returns:
            ``"sha256:<hex>"`` of :meth:`to_record`.
        """
        payload = json.dumps(self.to_record(), sort_keys=True)
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, kw_only=True)
class VoxelContribution:
    """The raw points one scan contributed to one aggregate.

    Which points they are is not stored: it is a pure function of the raw coordinates
    and the grid, recoverable with :meth:`VoxelAggregation.members_of`.

    Attributes:
        scan_ordinal: The scan's position in the raw map's source index.
        point_count: Raw points of that scan inside the voxel.
    """

    scan_ordinal: int
    point_count: int


@dataclass(frozen=True, kw_only=True)
class VoxelAggregate:
    """One occupied voxel of a derived aggregation.

    Attributes:
        index: Position of the aggregate in its aggregation; its identity there.
        key: The voxel's integer key in the grid.
        centroid_m: The representative: mean of the raw points, in the map frame.
        offset_sum_m: Sum of ``p - anchor`` over the raw points, where ``anchor`` is
            the voxel's lower corner. Summing offsets instead of absolute coordinates
            bounds the rounding by the voxel size instead of the map's extent.
        minimum_m: Per-axis minimum of the raw points.
        maximum_m: Per-axis maximum of the raw points.
        point_count: Raw measurements summarized.
        scan_count: Distinct scans (source-index entries) that contributed.
        observation_count: Distinct physical source observations that contributed.
        first_observed_at: Acquisition time of the earliest contributing scan.
        last_observed_at: Acquisition time of the latest contributing scan.
        contributions: One entry per contributing scan, by scan ordinal.
    """

    index: int
    key: tuple[int, int, int]
    centroid_m: Vector3
    offset_sum_m: Vector3
    minimum_m: Vector3
    maximum_m: Vector3
    point_count: int
    scan_count: int
    observation_count: int
    first_observed_at: SourceTimestamp
    last_observed_at: SourceTimestamp
    contributions: tuple[VoxelContribution, ...]


def centroid_tolerance_m(
    point_counts: NDArray[Any], cell_m: float, centroids_m: NDArray[Any]
) -> NDArray[Any]:
    """Bound, per axis, how much order or chunking can move a centroid.

    Each offset ``p - anchor`` is computed identically whatever the order, and lies
    within one voxel; only the order of the ``n`` additions changes. Two summation
    orders of ``n`` terms of magnitude ``<= cell_m`` differ by at most
    ``n * n * eps * cell_m``, so the mean differs by at most ``n * eps * cell_m``, plus
    one rounding of the final ``anchor + mean`` on each side.

    Args:
        point_counts: ``(N,)`` raw points behind each aggregate.
        cell_m: The voxel edge length.
        centroids_m: ``(N, 3)`` centroids.

    Returns:
        ``(N, 3)`` tolerances in meters.
    """
    import numpy as np

    counts = np.asarray(point_counts, dtype=np.float64)[:, None]
    tolerance: NDArray[Any] = 2.0 * counts * _EPSILON * cell_m + 2.0 * _EPSILON * np.abs(
        centroids_m
    )
    return tolerance


@dataclass(frozen=True, kw_only=True, eq=False)
class VoxelAggregation:
    """Every aggregate derived from one raw map under one policy, as arrays.

    Aggregates are ordered by key (x, then y, then z). Contributions are stored in the
    same order, ``scan_counts[i]`` consecutive entries per aggregate, by scan ordinal.

    Attributes:
        policy: The policy that produced the aggregation.
        source_map_id: The raw map the aggregates summarize.
        source_point_count: Raw points in that map; every one is in exactly one aggregate.
        clock_id: Clock domain of the observation times.
        keys: ``(V, 3)`` ``int64`` voxel keys.
        centroids_m: ``(V, 3)`` centroids.
        offset_sums_m: ``(V, 3)`` sums of offsets from each voxel's lower corner.
        minimum_m: ``(V, 3)`` per-axis minima.
        maximum_m: ``(V, 3)`` per-axis maxima.
        point_counts: ``(V,)`` raw measurements per aggregate.
        scan_counts: ``(V,)`` contributing scans per aggregate.
        observation_counts: ``(V,)`` contributing observations per aggregate.
        first_observed_ns: ``(V,)`` earliest contributing acquisition time.
        last_observed_ns: ``(V,)`` latest contributing acquisition time.
        contribution_scan_ordinals: ``(C,)`` scan ordinal of each contribution.
        contribution_point_counts: ``(C,)`` raw points of each contribution.
    """

    policy: InterScanVoxelPolicy
    source_map_id: MapId
    source_point_count: int
    clock_id: str
    keys: NDArray[Any]
    centroids_m: NDArray[Any]
    offset_sums_m: NDArray[Any]
    minimum_m: NDArray[Any]
    maximum_m: NDArray[Any]
    point_counts: NDArray[Any]
    scan_counts: NDArray[Any]
    observation_counts: NDArray[Any]
    first_observed_ns: NDArray[Any]
    last_observed_ns: NDArray[Any]
    contribution_scan_ordinals: NDArray[Any]
    contribution_point_counts: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate the internal consistency of the arrays.

        Raises:
            VoxelAggregationError: If shapes disagree, keys are not strictly increasing,
                a count is impossible, the contributions do not add up, a bound leaves
                its voxel, or a centroid is not the one its statistics give.
        """
        problems = self._consistency_problems()
        if problems:
            raise VoxelAggregationError("; ".join(problems))

    def __len__(self) -> int:
        """Return the number of aggregates."""
        return int(self.keys.shape[0])

    @property
    def contribution_offsets(self) -> NDArray[Any]:
        """``(V + 1,)`` start of each aggregate's contributions, then their total."""
        import numpy as np

        offsets: NDArray[Any] = np.concatenate(([0], np.cumsum(self.scan_counts, dtype=np.int64)))
        return offsets

    def aggregate(self, index: int) -> VoxelAggregate:
        """Return one aggregate, with its contributions.

        Raises:
            IndexError: If ``index`` is outside the aggregation.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"aggregate {index} is outside the {len(self)} aggregates")
        offsets = self.contribution_offsets
        start, stop = int(offsets[index]), int(offsets[index + 1])
        return VoxelAggregate(
            index=index,
            key=_triple_int(self.keys[index]),
            centroid_m=_triple(self.centroids_m[index]),
            offset_sum_m=_triple(self.offset_sums_m[index]),
            minimum_m=_triple(self.minimum_m[index]),
            maximum_m=_triple(self.maximum_m[index]),
            point_count=int(self.point_counts[index]),
            scan_count=int(self.scan_counts[index]),
            observation_count=int(self.observation_counts[index]),
            first_observed_at=_timestamp(int(self.first_observed_ns[index]), self.clock_id),
            last_observed_at=_timestamp(int(self.last_observed_ns[index]), self.clock_id),
            contributions=tuple(
                VoxelContribution(
                    scan_ordinal=int(self.contribution_scan_ordinals[position]),
                    point_count=int(self.contribution_point_counts[position]),
                )
                for position in range(start, stop)
            ),
        )

    def index_of(self, keys: NDArray[Any]) -> NDArray[Any]:
        """Return the aggregate index of every key, ``-1`` where no aggregate exists.

        Args:
            keys: ``(N, 3)`` integer voxel keys.
        """
        import numpy as np

        wanted = np.asarray(keys, dtype=np.int64)
        result = np.full(wanted.shape[0], -1, dtype=np.int64)
        if not len(self) or not wanted.shape[0]:
            return result
        low = self.keys.min(axis=0)
        span = self.keys.max(axis=0) - low + 1
        inside = ((wanted >= low) & (wanted < low + span)).all(axis=1)
        own = _linear(self.keys, low, span)
        codes = _linear(wanted[inside], low, span)
        positions = np.searchsorted(own, codes)
        positions = np.minimum(positions, len(own) - 1)
        found = own[positions] == codes
        selected = np.flatnonzero(inside)
        result[selected[found]] = positions[found]
        return result

    def membership(
        self, geometry: GeometryBlockSource, *, block_points: int = DEFAULT_BLOCK_POINTS
    ) -> NDArray[Any]:
        """Recover, for every raw point, the aggregate that summarizes it.

        Args:
            geometry: The raw map the aggregation was derived from.
            block_points: Rows read per block.

        Returns:
            ``(point_count,)`` aggregate index per raw geometry index; ``-1`` for a raw
            point no aggregate covers, which only happens for another map.

        Raises:
            VoxelAggregationError: If ``geometry`` is not the source map.
        """
        import numpy as np

        source = geometry.geometric_map
        if source.map_id != self.source_map_id:
            raise VoxelAggregationError(
                f"the aggregation summarizes map {self.source_map_id!r}, not {source.map_id!r}"
            )
        result = np.full(source.point_count, -1, dtype=np.int64)
        for block in geometry.iter_blocks(block_points=block_points):
            result[block.indices] = self.index_of(self.policy.grid.keys_of(block.coordinates_m))
        return result

    def members_of(self, index: int, geometry: PackedGeometry) -> tuple[GeometryReference, ...]:
        """Recover the references of the raw points one aggregate summarizes.

        Only the contributing scans are read, so the cost follows the lineage, not the map.

        Args:
            index: The aggregate.
            geometry: The raw map the aggregation was derived from.

        Returns:
            The raw references, in increasing geometry index.

        Raises:
            VoxelAggregationError: If ``geometry`` is not the source map, or the points
                found disagree with the recorded lineage.
        """
        import numpy as np

        aggregate = self.aggregate(index)
        map_id = geometry.geometric_map.map_id
        if map_id != self.source_map_id:
            raise VoxelAggregationError(
                f"the aggregation summarizes map {self.source_map_id!r}, not {map_id!r}"
            )
        key = np.array(aggregate.key, dtype=np.int64)
        found: list[int] = []
        for contribution in aggregate.contributions:
            scan = geometry.scans[contribution.scan_ordinal]
            first = scan.first_geometry_index
            coordinates = np.array(
                geometry.scan_map_coordinates(scan.observation_id), dtype=np.float64
            ).reshape(-1, 3)
            rows = np.flatnonzero((self.policy.grid.keys_of(coordinates) == key).all(axis=1))
            if len(rows) != contribution.point_count:
                raise VoxelAggregationError(
                    f"aggregate {index} records {contribution.point_count} points of scan "
                    f"{contribution.scan_ordinal} but the raw map has {len(rows)} in its voxel"
                )
            found.extend(int(first + row) for row in rows)
        return tuple(
            GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=i))
            for i in sorted(found)
        )

    def verify_lineage(
        self, geometry: PackedGeometry, *, block_points: int = DEFAULT_BLOCK_POINTS
    ) -> list[str]:
        """Re-derive the lineage from the raw map and compare it with the recorded one.

        Args:
            geometry: The raw map the aggregation claims to summarize.
            block_points: Rows read per block.

        Returns:
            Human-readable problems; empty when every raw point falls in exactly one
            aggregate and every aggregate's per-scan counts are the recorded ones.
        """
        import numpy as np

        if geometry.geometric_map.map_id != self.source_map_id:
            return [
                f"the aggregation summarizes map {self.source_map_id!r}, not "
                f"{geometry.geometric_map.map_id!r}"
            ]
        if geometry.geometric_map.point_count != self.source_point_count:
            return [
                f"the aggregation summarizes {self.source_point_count} raw points but the map "
                f"has {geometry.geometric_map.point_count}"
            ]
        membership = self.membership(geometry, block_points=block_points)
        uncovered = int((membership < 0).sum())
        if uncovered:
            return [f"{uncovered} raw points fall in no recorded aggregate"]
        ordinals = _scan_ordinals(geometry.scans, np.arange(len(membership), dtype=np.int64))
        order = np.lexsort((ordinals, membership))
        pairs = np.column_stack((membership[order], ordinals[order]))
        starts = _group_starts(pairs)
        counts = np.diff(np.concatenate((starts, [len(pairs)])))
        recorded_aggregates = np.repeat(np.arange(len(self)), self.scan_counts)
        derived = np.column_stack((pairs[starts, 0], pairs[starts, 1], counts))
        recorded = np.column_stack(
            (
                recorded_aggregates,
                self.contribution_scan_ordinals,
                self.contribution_point_counts,
            )
        )
        if derived.shape != recorded.shape or not (derived == recorded).all():
            return [
                "the per-scan contributions re-derived from the raw map differ from the lineage"
            ]
        return []

    def equivalence_problems(self, other: VoxelAggregation) -> list[str]:
        """Compare two aggregations of the same raw map under the documented tolerance.

        Keys, counts, bounds, times and lineage must be identical; offset sums and
        centroids may differ only within :func:`centroid_tolerance_m`.

        Args:
            other: Another aggregation, for example of a different chunking.

        Returns:
            Human-readable differences; empty means the two are equivalent.
        """
        import numpy as np

        if self.policy != other.policy:
            return ["the aggregations were produced by different policies"]
        if self.source_map_id != other.source_map_id:
            return ["the aggregations summarize different raw maps"]
        if len(self) != len(other) or not (self.keys == other.keys).all():
            return ["the aggregations occupy different voxels"]
        problems = [
            f"{name} differ"
            for name in (
                "point_counts",
                "scan_counts",
                "observation_counts",
                "minimum_m",
                "maximum_m",
                "first_observed_ns",
                "last_observed_ns",
                "contribution_scan_ordinals",
                "contribution_point_counts",
            )
            if not np.array_equal(getattr(self, name), getattr(other, name))
        ]
        cell = self.policy.grid.cell_m
        tolerance = centroid_tolerance_m(self.point_counts, cell, self.centroids_m)
        moved = np.abs(self.centroids_m - other.centroids_m) > tolerance
        if moved.any():
            problems.append(
                f"{int(moved.any(axis=1).sum())} centroids moved beyond the documented tolerance"
            )
        sum_tolerance = tolerance * self.point_counts[:, None]
        if (np.abs(self.offset_sums_m - other.offset_sums_m) > sum_tolerance).any():
            problems.append("offset sums differ beyond the documented tolerance")
        return problems

    def _consistency_problems(self) -> list[str]:
        import numpy as np

        count = self.keys.shape[0]
        vectors = ("keys", "centroids_m", "offset_sums_m", "minimum_m", "maximum_m")
        scalars = (
            "point_counts",
            "scan_counts",
            "observation_counts",
            "first_observed_ns",
            "last_observed_ns",
        )
        shapes = [name for name in vectors if getattr(self, name).shape != (count, 3)]
        shapes += [name for name in scalars if getattr(self, name).shape != (count,)]
        if shapes:
            return [f"arrays {shapes} do not describe {count} aggregates"]
        contributions = int(self.scan_counts.sum())
        if self.contribution_scan_ordinals.shape != (contributions,) or (
            self.contribution_point_counts.shape != (contributions,)
        ):
            return [f"the lineage does not hold the {contributions} contributions the counts give"]
        problems: list[str] = []
        if (self.point_counts < 1).any() or (self.contribution_point_counts < 1).any():
            problems.append("an aggregate or contribution has no raw point")
        if (self.scan_counts < 1).any() or (self.scan_counts > self.point_counts).any():
            problems.append("a scan count is not within [1, point_count]")
        if (self.observation_counts < 1).any() or (
            self.observation_counts > self.scan_counts
        ).any():
            problems.append("an observation count is not within [1, scan_count]")
        if problems:
            # As verificações seguintes indexam pelas contagens: só valem com contagens possíveis.
            return problems
        if count > 1:
            difference = np.diff(self.keys, axis=0)
            first_change = np.argmax(difference != 0, axis=1)
            leading = difference[np.arange(count - 1), first_change]
            if (leading <= 0).any():
                problems.append("keys are not strictly increasing")
        if int(self.point_counts.sum()) != self.source_point_count:
            problems.append(
                f"aggregates summarize {int(self.point_counts.sum())} raw points, not the "
                f"{self.source_point_count} of the source map"
            )
        if contributions:
            starts = self.contribution_offsets[:-1]
            if not np.array_equal(
                np.add.reduceat(self.contribution_point_counts, starts), self.point_counts
            ):
                problems.append("the contributions of an aggregate do not add up to its points")
            ordinals = self.contribution_scan_ordinals
            same_aggregate = np.repeat(np.arange(count), self.scan_counts)
            repeated = (np.diff(ordinals) <= 0) & (np.diff(same_aggregate) == 0)
            if repeated.any():
                problems.append("an aggregate lists a scan twice or out of order")
        if (self.first_observed_ns > self.last_observed_ns).any():
            problems.append("an aggregate was first observed after it was last observed")
        grid = self.policy.grid
        if count and not (
            np.array_equal(grid.keys_of(self.minimum_m), self.keys)
            and np.array_equal(grid.keys_of(self.maximum_m), self.keys)
        ):
            problems.append("a bound lies outside its aggregate's voxel")
        if count and not np.array_equal(
            self.centroids_m, _centroids(grid, self.keys, self.offset_sums_m, self.point_counts)
        ):
            problems.append("a centroid is not the one its offset sum and point count give")
        return problems


class InterScanVoxelAggregator:
    """Incrementally aggregates the blocks of one raw map into voxel aggregates.

    Blocks may arrive in any order and any size; each raw point must arrive exactly
    once before :meth:`finish`. The state is a table of ``(voxel, scan)`` partial
    statistics, consolidated as it grows, so memory follows the number of occupied
    ``(voxel, scan)`` pairs rather than the number of points.
    """

    def __init__(
        self,
        *,
        policy: InterScanVoxelPolicy,
        source_map: GeometricMap,
        scans: Sequence[ScanRecord],
    ) -> None:
        """Start an aggregation of one raw map.

        Args:
            policy: The aggregation policy.
            source_map: The raw map being aggregated.
            scans: Its source index, in accumulation order.

        Raises:
            VoxelAggregationError: If the map is already aggregated, is in another
                frame than the grid, or its scans do not tile it.
        """
        import numpy as np

        if source_map.aggregation_rule is not None:
            raise VoxelAggregationError(
                f"map {source_map.map_id!r} declares aggregation {source_map.aggregation_rule!r}; "
                "inter-scan aggregation is only derived from a raw map, one point per measurement"
            )
        if source_map.frame_id != policy.grid.frame_id:
            raise VoxelAggregationError(
                f"the grid is laid in frame {policy.grid.frame_id!r} but map "
                f"{source_map.map_id!r} is in {source_map.frame_id!r}"
            )
        if sum(scan.geometry_count for scan in scans) != source_map.point_count:
            raise VoxelAggregationError("the source index does not tile the raw map")
        clocks = {scan.acquisition_timestamp.clock_id for scan in scans}
        if len(clocks) > 1:
            raise VoxelAggregationError(f"the scans mix clock domains {sorted(clocks)}")
        self._policy = policy
        self._map = source_map
        self._scans = tuple(scans)
        self._clock_id = clocks.pop() if clocks else ""
        self._scan_times_ns = np.array(
            [scan.acquisition_timestamp.total_nanoseconds() for scan in self._scans],
            dtype=np.int64,
        )
        _, observation_codes = np.unique(
            np.array([str(scan.observation_id) for scan in self._scans], dtype=str),
            return_inverse=True,
        )
        self._observation_codes = observation_codes.astype(np.int64)
        self._seen = np.zeros(source_map.point_count, dtype=bool)
        self._state: _PairTable | None = None
        self._pending: list[_PairTable] = []
        self._pending_rows = 0
        self._finished = False

    def add_block(self, block: GeometryBlock) -> None:
        """Add raw points of the map, in any order and any grouping.

        Args:
            block: Coordinates of some raw points with their global indices.

        Raises:
            VoxelAggregationError: If the aggregator is finished, the block belongs to
                another map or frame, or a point was already added.
        """
        import numpy as np

        if self._finished:
            raise VoxelAggregationError("the aggregator is finished and accepts no more blocks")
        if block.map_id != self._map.map_id or block.frame_id != self._map.frame_id:
            raise VoxelAggregationError(
                f"block of map {block.map_id!r} in frame {block.frame_id!r} does not belong to "
                f"map {self._map.map_id!r} in {self._map.frame_id!r}"
            )
        if not len(block):
            return
        indices = block.indices.astype(np.int64)
        if indices.min() < 0 or indices.max() >= self._map.point_count:
            raise VoxelAggregationError("a block index is outside the raw map")
        if self._seen[indices].any() or len(np.unique(indices)) != len(indices):
            raise VoxelAggregationError("a raw point was added twice")
        self._seen[indices] = True

        grid = self._policy.grid
        coordinates = np.asarray(block.coordinates_m, dtype=np.float64)
        keys = grid.keys_of(coordinates)
        partial = _PairTable.reduce(
            keys=keys,
            ordinals=_scan_ordinals(self._scans, indices),
            counts=np.ones(len(indices), dtype=np.int64),
            offset_sums=coordinates - grid.anchors_of(keys),
            minimum=coordinates,
            maximum=coordinates,
        )
        self._pending.append(partial)
        self._pending_rows += len(partial)
        state_rows = 0 if self._state is None else len(self._state)
        if self._pending_rows >= max(state_rows, _CONSOLIDATION_ROWS):
            self._consolidate()

    def finish(self) -> VoxelAggregation:
        """Close the aggregation and return it.

        Raises:
            VoxelAggregationError: If already finished or a raw point was never added.
        """
        import numpy as np

        if self._finished:
            raise VoxelAggregationError("the aggregator is already finished")
        missing = int((~self._seen).sum())
        if missing:
            raise VoxelAggregationError(
                f"{missing} raw points were never added; an aggregation covers every raw point"
            )
        self._finished = True
        self._consolidate()
        pairs = self._state
        if pairs is None:
            raise VoxelAggregationError("the raw map has no point to aggregate")

        starts = _group_starts(pairs.keys)
        keys = pairs.keys[starts]
        point_counts = np.add.reduceat(pairs.counts, starts)
        offset_sums = np.add.reduceat(pairs.offset_sums, starts, axis=0)
        scan_counts = np.diff(np.concatenate((starts, [len(pairs)])))
        times = self._scan_times_ns[pairs.ordinals]
        voxel_of_pair = np.repeat(np.arange(len(starts)), scan_counts)
        codes = self._observation_codes[pairs.ordinals]
        by_observation = np.lexsort((codes, voxel_of_pair))
        distinct = _group_starts(
            np.column_stack((voxel_of_pair[by_observation], codes[by_observation]))
        )
        grid = self._policy.grid
        return VoxelAggregation(
            policy=self._policy,
            source_map_id=self._map.map_id,
            source_point_count=self._map.point_count,
            clock_id=self._clock_id,
            keys=keys,
            centroids_m=_centroids(grid, keys, offset_sums, point_counts),
            offset_sums_m=offset_sums,
            minimum_m=np.minimum.reduceat(pairs.minimum, starts, axis=0),
            maximum_m=np.maximum.reduceat(pairs.maximum, starts, axis=0),
            point_counts=point_counts,
            scan_counts=scan_counts,
            observation_counts=np.bincount(
                voxel_of_pair[by_observation][distinct], minlength=len(starts)
            ),
            first_observed_ns=np.minimum.reduceat(times, starts),
            last_observed_ns=np.maximum.reduceat(times, starts),
            contribution_scan_ordinals=pairs.ordinals,
            contribution_point_counts=pairs.counts,
        )

    def _consolidate(self) -> None:
        tables = ([] if self._state is None else [self._state]) + self._pending
        if tables:
            self._state = _PairTable.concatenate(tables)
        self._pending = []
        self._pending_rows = 0


def aggregate_geometry(
    geometry: PackedGeometry,
    policy: InterScanVoxelPolicy,
    *,
    block_points: int = DEFAULT_BLOCK_POINTS,
) -> VoxelAggregation:
    """Aggregate a whole raw map, reading it in blocks.

    Args:
        geometry: The raw map; it is only read.
        policy: The aggregation policy.
        block_points: Rows read per block; changes the result only within tolerance.

    Returns:
        The aggregation.

    Raises:
        VoxelAggregationError: If the map cannot be aggregated.
    """
    aggregator = InterScanVoxelAggregator(
        policy=policy, source_map=geometry.geometric_map, scans=geometry.scans
    )
    for block in geometry.iter_blocks(block_points=block_points):
        aggregator.add_block(block)
    return aggregator.finish()


class AggregatedGeometry:
    """The centroids of an aggregation, read through the block boundary of a map.

    Implements :class:`~contextmap.geometric_mapping.GeometryBlockSource`, so a consumer
    of map blocks (Sensor Association) can run unchanged over the derived representation.
    Row ``i`` of a block is aggregate ``indices[i]`` of the derived map, never a raw point:
    a reference into the derived map is resolved through the aggregation's lineage.
    """

    def __init__(self, *, aggregation: VoxelAggregation, geometric_map: GeometricMap) -> None:
        """Serve an aggregation under the derived map that describes it.

        Args:
            aggregation: The aggregates to serve.
            geometric_map: The derived map, as :meth:`derive` builds it or as it was
                persisted.

        Raises:
            VoxelAggregationError: If the map does not describe the aggregation: another
                element count, frame or rule, or the identity of the raw map.
        """
        if geometric_map.map_id == aggregation.source_map_id:
            raise VoxelAggregationError("a derived map never reuses the raw map's identity")
        if geometric_map.point_count != len(aggregation):
            raise VoxelAggregationError(
                f"the derived map declares {geometric_map.point_count} elements but the "
                f"aggregation has {len(aggregation)}"
            )
        if geometric_map.frame_id != aggregation.policy.grid.frame_id:
            raise VoxelAggregationError(
                f"the derived map is in {geometric_map.frame_id!r} but the grid is in "
                f"{aggregation.policy.grid.frame_id!r}"
            )
        if geometric_map.aggregation_rule != aggregation.policy.rule:
            raise VoxelAggregationError(
                f"the derived map declares rule {geometric_map.aggregation_rule!r}, not "
                f"{aggregation.policy.rule!r}"
            )
        self._aggregation = aggregation
        self._map = geometric_map

    @classmethod
    def derive(
        cls,
        *,
        aggregation: VoxelAggregation,
        source_map: GeometricMap,
        map_id: MapId,
        code_version: str | None,
    ) -> AggregatedGeometry:
        """Describe an aggregation as a derived map that keeps the raw map's lineage.

        Args:
            aggregation: The aggregates to serve.
            source_map: The raw map they were derived from.
            map_id: Identity of the derived map; never the raw map's.
            code_version: Code revision that derived the aggregation.

        Raises:
            VoxelAggregationError: If ``source_map`` is not the aggregation's source, the
                aggregation is empty, or ``map_id`` reuses the raw map's identity.
        """
        if source_map.map_id != aggregation.source_map_id:
            raise VoxelAggregationError(
                f"the aggregation summarizes {aggregation.source_map_id!r}, not "
                f"{source_map.map_id!r}"
            )
        if not len(aggregation):
            raise VoxelAggregationError("an empty aggregation is not a map")
        return cls(
            aggregation=aggregation,
            geometric_map=_derived_map(aggregation, source_map, map_id, code_version),
        )

    @property
    def aggregation(self) -> VoxelAggregation:
        """The aggregates behind the blocks."""
        return self._aggregation

    @property
    def geometric_map(self) -> GeometricMap:
        """The derived map: identity, frame, envelope of the centroids and lineage."""
        return self._map

    def iter_blocks(
        self, *, bounds: Bounds3D | None = None, block_points: int = DEFAULT_BLOCK_POINTS
    ) -> Iterator[GeometryBlock]:
        """Iterate the centroids as blocks, in aggregate order.

        Args:
            bounds: Keeps only centroids inside the box, boundaries included.
            block_points: Maximum rows per block.

        Raises:
            ValueError: If ``block_points`` is not positive or ``bounds`` is in another
                frame than the map.
        """
        import numpy as np

        if block_points < 1:
            raise ValueError(f"block_points must be at least 1, got {block_points}")
        if bounds is not None and bounds.frame_id != self._map.frame_id:
            raise ValueError(
                f"bounds are expressed in frame {bounds.frame_id!r} but the map is in "
                f"{self._map.frame_id!r}; frames are never reinterpreted"
            )
        centroids = self._aggregation.centroids_m
        if bounds is None:
            selected = np.arange(len(centroids), dtype=np.int64)
        else:
            low = np.array(bounds.minimum_m, dtype=np.float64)
            high = np.array(bounds.maximum_m, dtype=np.float64)
            selected = np.flatnonzero(((centroids >= low) & (centroids <= high)).all(axis=1))
        for start in range(0, len(selected), block_points):
            rows = selected[start : start + block_points].astype(np.int64)
            yield GeometryBlock(
                map_id=self._map.map_id,
                frame_id=self._map.frame_id,
                indices=rows,
                coordinates_m=np.array(centroids[rows], dtype=np.float64),
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class _PairTable:
    """Partial statistics per ``(voxel, scan)`` pair, sorted by key then scan ordinal."""

    keys: NDArray[Any]
    ordinals: NDArray[Any]
    counts: NDArray[Any]
    offset_sums: NDArray[Any]
    minimum: NDArray[Any]
    maximum: NDArray[Any]

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    @classmethod
    def reduce(
        cls,
        *,
        keys: NDArray[Any],
        ordinals: NDArray[Any],
        counts: NDArray[Any],
        offset_sums: NDArray[Any],
        minimum: NDArray[Any],
        maximum: NDArray[Any],
    ) -> _PairTable:
        """Group rows by ``(voxel, scan)``, adding counts and sums and keeping extremes.

        The sort is stable, so rows of one pair are added in the order they arrived:
        the same input and chunking always reproduce the same bits.
        """
        import numpy as np

        order = np.lexsort((ordinals, keys[:, 2], keys[:, 1], keys[:, 0]))
        sorted_keys = keys[order]
        sorted_ordinals = ordinals[order]
        starts = _group_starts(np.column_stack((sorted_keys, sorted_ordinals)))
        return cls(
            keys=sorted_keys[starts],
            ordinals=sorted_ordinals[starts],
            counts=np.add.reduceat(counts[order], starts),
            offset_sums=np.add.reduceat(offset_sums[order], starts, axis=0),
            minimum=np.minimum.reduceat(minimum[order], starts, axis=0),
            maximum=np.maximum.reduceat(maximum[order], starts, axis=0),
        )

    @classmethod
    def concatenate(cls, tables: Sequence[_PairTable]) -> _PairTable:
        """Merge partial tables into one, pairs of equal identity combined."""
        import numpy as np

        return cls.reduce(
            keys=np.concatenate([table.keys for table in tables]),
            ordinals=np.concatenate([table.ordinals for table in tables]),
            counts=np.concatenate([table.counts for table in tables]),
            offset_sums=np.concatenate([table.offset_sums for table in tables]),
            minimum=np.concatenate([table.minimum for table in tables]),
            maximum=np.concatenate([table.maximum for table in tables]),
        )


def _group_starts(rows: NDArray[Any]) -> NDArray[Any]:
    """Return where each run of equal rows starts in rows already sorted."""
    import numpy as np

    if not len(rows):
        return np.zeros(0, dtype=np.int64)
    changes = np.empty(len(rows), dtype=bool)
    changes[0] = True
    different = rows[1:] != rows[:-1]
    changes[1:] = different.any(axis=1) if different.ndim > 1 else different
    starts: NDArray[Any] = np.flatnonzero(changes)
    return starts


def _scan_ordinals(scans: Sequence[ScanRecord], indices: NDArray[Any]) -> NDArray[Any]:
    """Map global geometry indices to the ordinal of the scan whose range holds them."""
    import numpy as np

    firsts = np.array([scan.first_geometry_index for scan in scans], dtype=np.int64)
    counts = np.array([scan.geometry_count for scan in scans], dtype=np.int64)
    # Um scan sem geometria tem o mesmo início do seguinte: só os que têm pontos podem conter um.
    holding = np.flatnonzero(counts > 0)
    positions = np.searchsorted(firsts[holding], indices, side="right") - 1
    ordinals: NDArray[Any] = holding[positions].astype(np.int64)
    return ordinals


def _linear(keys: NDArray[Any], low: NDArray[Any], span: NDArray[Any]) -> NDArray[Any]:
    """Encode keys inside ``[low, low + span)`` as one ``int64`` preserving key order.

    Raises:
        VoxelAggregationError: If the grid extent does not fit in one ``int64``.
    """
    import numpy as np

    if float(span[0]) * float(span[1]) * float(span[2]) >= 2.0**63:
        raise VoxelAggregationError("the occupied grid extent does not fit a linear key")
    shifted = keys.astype(np.int64) - low
    linear: NDArray[Any] = (shifted[:, 0] * span[1] + shifted[:, 1]) * span[2] + shifted[:, 2]
    return linear


def _centroids(
    grid: VoxelGridSpec, keys: NDArray[Any], offset_sums: NDArray[Any], counts: NDArray[Any]
) -> NDArray[Any]:
    centroids: NDArray[Any] = grid.anchors_of(keys) + offset_sums / counts[:, None]
    return centroids


def _derived_map(
    aggregation: VoxelAggregation,
    source_map: GeometricMap,
    map_id: MapId,
    code_version: str | None,
) -> GeometricMap:
    """Describe the aggregates as a map that keeps the raw map's upstream lineage."""
    centroids = aggregation.centroids_m
    low, high = centroids.min(axis=0), centroids.max(axis=0)
    provenance = source_map.provenance
    return GeometricMap(
        map_id=map_id,
        frame_id=source_map.frame_id,
        point_count=len(aggregation),
        bounds=Bounds3D(
            frame_id=source_map.frame_id,
            minimum_m=_triple(low),
            maximum_m=_triple(high),
        ),
        source_observation_ids=source_map.source_observation_ids,
        time_bounds=source_map.time_bounds,
        spatial_index=None,
        provenance=GeometricMapProvenance(
            sequence_artifact_id=provenance.sequence_artifact_id,
            selection_id=provenance.selection_id,
            trajectory_id=provenance.trajectory_id,
            state_estimation_run_id=provenance.state_estimation_run_id,
            calibration_identity=provenance.calibration_identity,
            pose_lookup=provenance.pose_lookup,
            configuration_fingerprint=aggregation.policy.fingerprint(),
            code_version=code_version,
        ),
        aggregation_rule=aggregation.policy.rule,
    )


def _triple(values: Any) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _triple_int(values: Any) -> tuple[int, int, int]:
    return (int(values[0]), int(values[1]), int(values[2]))


def _timestamp(total_nanoseconds: int, clock_id: str) -> SourceTimestamp:
    seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)

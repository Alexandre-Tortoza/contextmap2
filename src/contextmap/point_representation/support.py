"""Deterministic local 3D support extraction over a ``GeometrySource``.

A representation describes the geometry *around* an element, so before any
encoder runs the neighborhood has to be selected and its coordinates prepared
explicitly and reproducibly. :class:`SupportExtractor` does only that: it reads
persistent geometry through the public :class:`GeometrySource` boundary, never
through a concrete index, and knows no model, label or visual evidence.

Order is part of the contract
    Members are ordered center first, then by ascending Euclidean distance to
    the center, ties broken by ascending geometry id. The order does not depend
    on the order in which the source yields its points, so the same map and
    policy always give the same support and the same prepared coordinates.

Coordinates are the authoritative map-frame ``coordinates_m`` in meters; the
sensor-local measurement of a point is never used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometryId,
    GeometryPoint,
    GeometryReference,
    GeometrySource,
)
from contextmap.point_representation.models import (
    CenteringMode,
    NeighborhoodMethod,
    PointSupport,
    PreparedSupport,
    ScaleNormalization,
    SupportPolicy,
    SupportStatistics,
    SupportType,
)
from contextmap.shared import Vector3

REFERENCE_LOOKUP_QUERY_METHOD = "reference-lookup"
"""Query identity of a point support: the center is resolved directly by reference."""

RADIUS_QUERY_METHOD = "bounds-query+euclidean-radius"
"""Query identity of a radius support: a box prefilter, then an exact distance filter."""

K_NEAREST_QUERY_METHOD = "bounds-query-doubling+euclidean-k-nearest"
"""Query identity of a k-nearest support: box queries of doubling radius until ``k`` qualify."""

_BOX_MARGIN_RATIO = 1e-9
"""Relative slack of the prefilter box.

Subtracting and adding a radius rounds, so a point whose exact distance is within
the radius could fall a ulp outside a tight box. The exact distance filter that
follows is the authority; the margin only keeps the prefilter conservative.
"""

_FALLBACK_SEARCH_RADIUS_M = 1.0
"""Starting search radius of a k-nearest query on a map whose points all coincide."""


class SupportExtractor:
    """Selects and prepares the local support of geometry elements under one policy."""

    def __init__(self, source: GeometrySource, policy: SupportPolicy) -> None:
        """Bind the extractor to one map and one policy.

        Args:
            source: The persistent geometry to read; only the public
                :class:`GeometrySource` boundary is used.
            policy: The support policy applied to every extraction.
        """
        self._source = source
        self._policy = policy
        self._map = source.geometric_map

    @property
    def policy(self) -> SupportPolicy:
        """The support policy applied to every extraction."""
        return self._policy

    def extract(self, center: GeometryReference) -> PreparedSupport:
        """Extract the support of one element and prepare its coordinates.

        Args:
            center: The geometry element the support is anchored to.

        Returns:
            The support, with its members in the deterministic order described
            in the module documentation, and one prepared coordinate per member.

        Raises:
            KeyError: If ``center`` does not belong to the source map.
            ValueError: If the source yields a point twice, a point of another
                map, or a point expressed in another frame than the map frame.
        """
        center_point = self._resolve(center)
        members, candidate_count = self._select_members(center_point)
        distances = [distance for distance, _ in members]
        support = PointSupport(
            policy=self._policy,
            center=center,
            geometry_refs=tuple(point.reference for _, point in members),
            map_frame=str(self._map.frame_id),
            statistics=SupportStatistics(
                count=len(members),
                min_distance_m=min(distances),
                max_distance_m=max(distances),
                mean_distance_m=math.fsum(distances) / len(distances),
                near_map_bounds=self._reaches_past_map_bounds(center_point, max(distances)),
                candidate_count=candidate_count,
            ),
            query_method=self._query_method(),
        )
        return PreparedSupport(
            support=support,
            local_coordinates_m=_prepare_coordinates(
                [point for _, point in members], center_point, self._policy
            ),
        )

    def _select_members(
        self, center: GeometryPoint
    ) -> tuple[list[tuple[float, GeometryPoint]], int | None]:
        """Order the qualifying elements and cut them to the policy limit.

        Returns:
            The ``(distance_m, point)`` members, center first, and the number of
            elements that qualified before a cap; ``None`` when not tracked.
        """
        policy = self._policy
        if policy.support_type is SupportType.POINT:
            return [(0.0, center)], None
        if policy.method is NeighborhoodMethod.RADIUS:
            assert policy.radius_m is not None
            ordered = _in_support_order(self._within(center, policy.radius_m), center)
            return ordered[: policy.max_neighbors], len(ordered)
        assert policy.k is not None
        ordered = _in_support_order(self._k_nearest_candidates(center, policy.k), center)
        return ordered[: policy.k], None

    def _k_nearest_candidates(
        self, center: GeometryPoint, k: int
    ) -> list[tuple[float, GeometryPoint]]:
        """Collect every element within a radius that is known to contain the ``k`` nearest.

        The radius doubles until ``k`` elements lie within it: then no element
        outside can be nearer than the ``k``-th, so the selection is exact and
        independent of the starting radius. The search also stops once the box
        covers the whole map, so a map with fewer than ``k`` points yields all of them.
        """
        radius_m = self._initial_search_radius_m(k)
        reach_m = self._farthest_bounds_corner_m(center)
        while True:
            found = self._within(center, radius_m)
            if len(found) >= k or radius_m >= reach_m:
                return found
            radius_m *= 2

    def _initial_search_radius_m(self, k: int) -> float:
        """Estimate, from the map's extent and density, a radius that may hold ``k`` points.

        Only a starting point: a poor estimate costs extra queries, never a wrong result.
        """
        bounds = self._map.bounds
        diagonal_m = math.dist(bounds.minimum_m, bounds.maximum_m)
        if diagonal_m == 0.0:
            return _FALLBACK_SEARCH_RADIUS_M
        return diagonal_m * math.cbrt(k / self._map.point_count)

    def _farthest_bounds_corner_m(self, center: GeometryPoint) -> float:
        """Distance from the center to the farthest corner of the map bounds."""
        bounds = self._map.bounds
        return math.hypot(
            *(
                max(abs(value - low), abs(value - high))
                for value, low, high in zip(
                    center.coordinates_m, bounds.minimum_m, bounds.maximum_m, strict=True
                )
            )
        )

    def _within(self, center: GeometryPoint, radius_m: float) -> list[tuple[float, GeometryPoint]]:
        """Every element within ``radius_m`` of the center, the center included.

        The source answers a box query and the Euclidean distance decides
        membership, so the result does not depend on the source's index.
        """
        margin_m = radius_m * _BOX_MARGIN_RATIO
        low = (
            center.coordinates_m[0] - radius_m - margin_m,
            center.coordinates_m[1] - radius_m - margin_m,
            center.coordinates_m[2] - radius_m - margin_m,
        )
        high = (
            center.coordinates_m[0] + radius_m + margin_m,
            center.coordinates_m[1] + radius_m + margin_m,
            center.coordinates_m[2] + radius_m + margin_m,
        )
        box = Bounds3D(frame_id=self._map.frame_id, minimum_m=low, maximum_m=high)
        candidates: dict[GeometryId, GeometryPoint] = {}
        for point in self._source.query_bounds(box):
            self._require_owned(point)
            if point.geometry_id in candidates:
                raise ValueError(
                    f"source yielded geometry {point.geometry_id!r} twice in one query"
                )
            candidates[point.geometry_id] = point
        # O centro pertence ao próprio suporte mesmo que a fonte não o devolva.
        candidates.setdefault(center.geometry_id, center)
        distances = (
            (math.dist(center.coordinates_m, point.coordinates_m), point)
            for point in candidates.values()
        )
        return [(distance, point) for distance, point in distances if distance <= radius_m]

    def _resolve(self, reference: GeometryReference) -> GeometryPoint:
        point = self._source.get(reference)
        self._require_owned(point)
        return point

    def _require_owned(self, point: GeometryPoint) -> None:
        if point.map_id != self._map.map_id:
            raise ValueError(
                f"geometry {point.geometry_id!r} belongs to map {point.map_id!r}, not to "
                f"{self._map.map_id!r}"
            )
        if point.map_frame != self._map.frame_id:
            raise ValueError(
                f"geometry {point.geometry_id!r} is expressed in frame {point.map_frame!r} but "
                f"the map frame is {self._map.frame_id!r}; frames are never inferred"
            )

    def _reaches_past_map_bounds(self, center: GeometryPoint, reach_m: float) -> bool:
        """Whether the ball the support may cover extends past the map bounds.

        The radius is the policy radius for a radius support, otherwise the
        farthest member: a support that is cut by the edge of the map is not
        comparable with one in the middle of it.
        """
        radius_m = self._policy.radius_m if self._policy.radius_m is not None else reach_m
        bounds = self._map.bounds
        return any(
            value - radius_m < low or value + radius_m > high
            for value, low, high in zip(
                center.coordinates_m, bounds.minimum_m, bounds.maximum_m, strict=True
            )
        )

    def _query_method(self) -> str:
        if self._policy.support_type is SupportType.POINT:
            return REFERENCE_LOOKUP_QUERY_METHOD
        if self._policy.method is NeighborhoodMethod.RADIUS:
            return RADIUS_QUERY_METHOD
        return K_NEAREST_QUERY_METHOD


def _in_support_order(
    found: Sequence[tuple[float, GeometryPoint]], center: GeometryPoint
) -> list[tuple[float, GeometryPoint]]:
    """Order members center first, then by distance, ties broken by geometry id."""
    return sorted(
        found,
        key=lambda item: (item[1].geometry_id != center.geometry_id, item[0], item[1].geometry_id),
    )


def _prepare_coordinates(
    members: Sequence[GeometryPoint], center: GeometryPoint, policy: SupportPolicy
) -> tuple[Vector3, ...]:
    """Apply the policy's centering and scale normalization to map-frame coordinates."""
    preparation = policy.preparation
    coordinates = [point.coordinates_m for point in members]
    if preparation.centering is CenteringMode.CENTER:
        origin = center.coordinates_m
    elif preparation.centering is CenteringMode.CENTROID:
        origin = (
            math.fsum(xyz[0] for xyz in coordinates) / len(coordinates),
            math.fsum(xyz[1] for xyz in coordinates) / len(coordinates),
            math.fsum(xyz[2] for xyz in coordinates) / len(coordinates),
        )
    else:
        origin = (0.0, 0.0, 0.0)
    if preparation.scale_normalization is ScaleNormalization.SUPPORT_RADIUS:
        assert policy.radius_m is not None
        scale = policy.radius_m
        return tuple(
            ((x - origin[0]) / scale, (y - origin[1]) / scale, (z - origin[2]) / scale)
            for x, y, z in coordinates
        )
    return tuple((x - origin[0], y - origin[1], z - origin[2]) for x, y, z in coordinates)

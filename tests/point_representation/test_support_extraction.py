import dataclasses
import math

import pytest
from pointrep_builders import knn_policy, radius_policy
from pointrep_geometry import (
    MAP_ID,
    ForeignFrameSource,
    GridIndexedSource,
    LinearScanSource,
    RepeatingSource,
    line_of_points,
    points_from,
    random_cloud,
)

from contextmap.geometric_mapping import GeometryPoint, GeometryReference, MapId, geometry_id_for
from contextmap.point_representation import (
    CenteringMode,
    CoordinatePreparation,
    NeighborhoodMethod,
    ScaleNormalization,
    SupportExtractor,
    SupportPolicy,
    SupportType,
)

SOURCE_FACTORIES = pytest.mark.parametrize(
    "source_factory", [LinearScanSource, GridIndexedSource], ids=["linear", "grid"]
)


def ref(index: int, *, map_id: MapId = MAP_ID) -> GeometryReference:
    return GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))


def point_policy(*, centering: CenteringMode = CenteringMode.CENTER) -> SupportPolicy:
    return SupportPolicy(
        support_type=SupportType.POINT,
        method=None,
        radius_m=None,
        k=None,
        max_neighbors=None,
        preparation=CoordinatePreparation(centering=centering),
    )


def reference_members(
    points: list[GeometryPoint],
    center_index: int,
    *,
    radius_m: float | None = None,
    limit: int | None = None,
) -> list[GeometryReference]:
    """Brute-force definition: the center first, then by distance and geometry id."""
    center = points[center_index]
    others = sorted(
        (point for point in points if point is not center),
        key=lambda point: (math.dist(center.coordinates_m, point.coordinates_m), point.geometry_id),
    )
    if radius_m is not None:
        others = [
            point
            for point in others
            if math.dist(center.coordinates_m, point.coordinates_m) <= radius_m
        ]
    members = [center, *others]
    return [point.reference for point in members[:limit]]


# --- Radius support ----------------------------------------------------------


@SOURCE_FACTORIES
def test_a_radius_support_selects_the_members_within_the_radius_in_a_stable_order(
    source_factory: type[LinearScanSource],
) -> None:
    source = source_factory(line_of_points(11))

    prepared = SupportExtractor(source, radius_policy(0.6)).extract(ref(5))

    assert prepared.support.center == ref(5)
    assert prepared.support.geometry_refs == (ref(5), ref(4), ref(6), ref(3), ref(7))
    assert prepared.support.statistics.count == 5
    assert prepared.support.statistics.min_distance_m == 0.0
    assert prepared.support.statistics.max_distance_m == 0.5
    assert prepared.support.statistics.mean_distance_m == pytest.approx(0.3)
    assert prepared.support.statistics.candidate_count == 5
    assert prepared.support.map_frame == "map"


@pytest.mark.parametrize(
    ("radius_m", "expected"),
    [(0.5, {3, 4, 5, 6, 7}), (0.49, {4, 5, 6}), (0.25, {4, 5, 6}), (0.24, {5})],
)
def test_the_radius_boundary_is_inclusive(radius_m: float, expected: set[int]) -> None:
    prepared = SupportExtractor(
        LinearScanSource(line_of_points(11)), radius_policy(radius_m)
    ).extract(ref(5))

    assert set(prepared.support.geometry_refs) == {ref(i) for i in expected}


@SOURCE_FACTORIES
@pytest.mark.parametrize("radius_m", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("center_index", [0, 57, 199])
def test_a_radius_support_matches_the_brute_force_definition(
    source_factory: type[LinearScanSource], radius_m: float, center_index: int
) -> None:
    points = random_cloud(200)

    prepared = SupportExtractor(source_factory(points), radius_policy(radius_m)).extract(
        ref(center_index)
    )

    assert list(prepared.support.geometry_refs) == reference_members(
        points, center_index, radius_m=radius_m
    )


def test_an_isolated_center_supports_itself() -> None:
    source = LinearScanSource(points_from([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]))

    prepared = SupportExtractor(source, radius_policy(0.5)).extract(ref(0))

    assert prepared.support.geometry_refs == (ref(0),)
    assert prepared.support.statistics.count == 1
    assert prepared.support.statistics.max_distance_m == 0.0
    assert prepared.local_coordinates_m == ((0.0, 0.0, 0.0),)


def test_coincident_elements_are_all_kept_and_ordered_by_geometry_id() -> None:
    coordinates = [(0.0, 0.0, 0.0)] * 5 + [(0.25, 0.0, 0.0)]

    prepared = SupportExtractor(
        LinearScanSource(points_from(coordinates)), radius_policy(0.5)
    ).extract(ref(4))

    assert prepared.support.geometry_refs == (ref(4), ref(0), ref(1), ref(2), ref(3), ref(5))


def test_a_neighbor_cap_keeps_the_nearest_and_never_drops_the_center() -> None:
    coordinates = [(0.0, 0.0, 0.0)] * 5 + [(0.25, 0.0, 0.0)]

    prepared = SupportExtractor(
        LinearScanSource(points_from(coordinates)), radius_policy(0.5, max_neighbors=2)
    ).extract(ref(4))

    assert prepared.support.geometry_refs == (ref(4), ref(0))
    assert prepared.support.statistics.count == 2
    assert prepared.support.statistics.candidate_count == 6


def test_the_radius_query_box_covers_the_whole_ball_in_the_map_frame() -> None:
    source = LinearScanSource(line_of_points(11))

    SupportExtractor(source, radius_policy(0.6)).extract(ref(5))

    (bounds,) = source.queries
    assert bounds.frame_id == "map"
    assert all(
        low <= center - 0.6 and center + 0.6 <= high
        for low, center, high in zip(
            bounds.minimum_m, (1.25, 0.0, 0.0), bounds.maximum_m, strict=True
        )
    )


# --- k-nearest support -------------------------------------------------------


@SOURCE_FACTORIES
@pytest.mark.parametrize("k", [1, 5, 17, 200, 500])
@pytest.mark.parametrize("center_index", [0, 57, 199])
def test_a_k_nearest_support_matches_the_brute_force_definition(
    source_factory: type[LinearScanSource], k: int, center_index: int
) -> None:
    points = random_cloud(200)

    prepared = SupportExtractor(source_factory(points), knn_policy(k)).extract(ref(center_index))

    assert list(prepared.support.geometry_refs) == reference_members(points, center_index, limit=k)
    assert prepared.support.statistics.count == min(k, 200)
    assert prepared.support.statistics.candidate_count is None


def test_k_nearest_finds_neighbors_far_from_a_sparse_center() -> None:
    cluster = [(index * 0.25, 0.0, 0.0) for index in range(10)]
    points = points_from([*cluster, (100.0, 0.0, 0.0)])

    prepared = SupportExtractor(LinearScanSource(points), knn_policy(3)).extract(ref(10))

    assert prepared.support.geometry_refs == (ref(10), ref(9), ref(8))


def test_a_sparse_map_yields_fewer_than_k_members_explicitly() -> None:
    source = LinearScanSource(points_from([(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 0.0, 0.0)]))

    prepared = SupportExtractor(source, knn_policy(8)).extract(ref(0))

    assert prepared.support.geometry_refs == (ref(0), ref(1), ref(2))
    assert prepared.support.policy.k == 8
    assert prepared.support.statistics.count == 3


def test_k_nearest_ties_are_broken_by_geometry_id() -> None:
    ring = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, -0.5, 0.0)]

    prepared = SupportExtractor(LinearScanSource(points_from(ring)), knn_policy(3)).extract(ref(0))

    assert prepared.support.geometry_refs == (ref(0), ref(1), ref(2))


# --- Determinism, index independence and the source boundary ------------------


def test_the_same_map_and_policy_produce_the_same_support_regardless_of_source_order() -> None:
    points = random_cloud(120)
    shuffled = list(reversed(points[::2])) + points[1::2]

    extractor = SupportExtractor(LinearScanSource(points), radius_policy(1.0))
    reordered = SupportExtractor(LinearScanSource(shuffled), radius_policy(1.0))

    assert extractor.extract(ref(17)) == extractor.extract(ref(17))
    assert extractor.extract(ref(17)) == reordered.extract(ref(17))


@pytest.mark.parametrize("policy", [radius_policy(1.0), knn_policy(9), point_policy()])
def test_supports_do_not_depend_on_how_the_map_is_indexed(policy: SupportPolicy) -> None:
    points = random_cloud(150)
    linear = SupportExtractor(LinearScanSource(points), policy)
    grid = SupportExtractor(GridIndexedSource(points, cell_size_m=0.4), policy)

    for center_index in (0, 42, 149):
        assert linear.extract(ref(center_index)) == grid.extract(ref(center_index))


def test_the_support_is_computed_from_map_frame_coordinates_only() -> None:
    points = line_of_points(11)
    tampered = [
        dataclasses.replace(point, source_coordinates_m=(99.0, 99.0, 99.0)) for point in points
    ]

    original = SupportExtractor(LinearScanSource(points), radius_policy(0.6)).extract(ref(5))
    other = SupportExtractor(LinearScanSource(tampered), radius_policy(0.6)).extract(ref(5))

    assert original == other


def test_a_center_outside_the_map_is_rejected() -> None:
    extractor = SupportExtractor(LinearScanSource(line_of_points(3)), radius_policy(0.5))

    with pytest.raises(KeyError):
        extractor.extract(ref(9))
    with pytest.raises(KeyError):
        extractor.extract(ref(0, map_id=MapId("another-map")))


def test_results_claiming_another_map_frame_are_rejected() -> None:
    extractor = SupportExtractor(ForeignFrameSource(line_of_points(3)), radius_policy(0.5))

    with pytest.raises(ValueError, match="frame"):
        extractor.extract(ref(1))


def test_a_source_yielding_the_same_element_twice_is_rejected() -> None:
    extractor = SupportExtractor(RepeatingSource(line_of_points(3)), radius_policy(0.5))

    with pytest.raises(ValueError, match="twice"):
        extractor.extract(ref(1))


def test_the_extractor_exposes_the_policy_it_was_built_with() -> None:
    policy = radius_policy(0.5)

    assert SupportExtractor(LinearScanSource(line_of_points(3)), policy).policy is policy


def test_the_query_method_is_recorded_per_neighborhood_method() -> None:
    source = LinearScanSource(line_of_points(11))

    radius = SupportExtractor(source, radius_policy(0.6)).extract(ref(5)).support
    nearest = SupportExtractor(source, knn_policy(3)).extract(ref(5)).support

    assert radius.policy.method is NeighborhoodMethod.RADIUS
    assert radius.query_method != nearest.query_method
    assert radius.query_method and nearest.query_method


# --- Behavior near the map bounds -------------------------------------------


def _cube_lattice() -> list[GeometryPoint]:
    return points_from(
        [(x * 0.25, y * 0.25, z * 0.25) for x in range(5) for y in range(5) for z in range(5)]
    )


@pytest.mark.parametrize(
    ("policy", "center_index", "near_bounds"),
    [
        (radius_policy(0.3), 0, True),
        (radius_policy(0.3), 62, False),
        (radius_policy(0.6), 62, True),
        (knn_policy(4), 0, True),
        (knn_policy(7), 62, False),
    ],
)
def test_supports_near_the_map_edge_are_flagged(
    policy: SupportPolicy, center_index: int, near_bounds: bool
) -> None:
    prepared = SupportExtractor(LinearScanSource(_cube_lattice()), policy).extract(
        ref(center_index)
    )

    assert prepared.support.statistics.near_map_bounds is near_bounds


# --- Coordinate preparation --------------------------------------------------


def test_uncentered_coordinates_are_the_map_frame_coordinates_in_support_order() -> None:
    points = line_of_points(11)
    policy = radius_policy(0.6, centering=CenteringMode.NONE)

    prepared = SupportExtractor(LinearScanSource(points), policy).extract(ref(5))

    coordinates = {point.reference: point.coordinates_m for point in points}
    assert prepared.local_coordinates_m == tuple(
        coordinates[reference] for reference in prepared.support.geometry_refs
    )


def test_center_relative_coordinates_put_the_center_at_the_origin() -> None:
    prepared = SupportExtractor(LinearScanSource(line_of_points(11)), radius_policy(0.6)).extract(
        ref(5)
    )

    assert prepared.local_coordinates_m == (
        (0.0, 0.0, 0.0),
        (-0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0),
    )


def test_centroid_relative_coordinates_have_a_zero_mean() -> None:
    policy = radius_policy(1.0, centering=CenteringMode.CENTROID)
    points = points_from([(0.0, 0.0, 0.0), (0.5, 0.25, 0.0), (0.25, 0.5, 0.25), (0.5, 0.5, 0.5)])

    prepared = SupportExtractor(LinearScanSource(points), policy).extract(ref(0))

    for axis in range(3):
        mean = sum(point[axis] for point in prepared.local_coordinates_m) / len(
            prepared.local_coordinates_m
        )
        assert mean == pytest.approx(0.0, abs=1e-12)


def test_scale_normalization_divides_by_the_recorded_radius() -> None:
    policy = radius_policy(
        0.5, centering=CenteringMode.CENTER, scale=ScaleNormalization.SUPPORT_RADIUS
    )

    prepared = SupportExtractor(LinearScanSource(line_of_points(11)), policy).extract(ref(5))

    assert prepared.support.applied_scale_m == 0.5
    assert prepared.local_coordinates_m == (
        (0.0, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    assert all(math.hypot(*point) <= 1.0 for point in prepared.local_coordinates_m)


def test_the_preparation_is_recorded_on_the_support_it_produced() -> None:
    policy = radius_policy(
        0.5, centering=CenteringMode.CENTROID, scale=ScaleNormalization.SUPPORT_RADIUS
    )

    support = SupportExtractor(LinearScanSource(line_of_points(11)), policy).extract(ref(5)).support

    assert support.policy.preparation == CoordinatePreparation(
        centering=CenteringMode.CENTROID, scale_normalization=ScaleNormalization.SUPPORT_RADIUS
    )


def test_a_point_support_is_the_center_alone() -> None:
    source = LinearScanSource(line_of_points(11))

    prepared = SupportExtractor(source, point_policy()).extract(ref(5))

    assert prepared.support.geometry_refs == (ref(5),)
    assert prepared.local_coordinates_m == ((0.0, 0.0, 0.0),)
    assert prepared.support.statistics.near_map_bounds is False
    assert source.queries == []

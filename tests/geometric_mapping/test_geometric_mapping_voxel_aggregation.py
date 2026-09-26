import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import pytest
from map_builders import MAP_ID, accumulate, make_scans
from voxel_builders import revisit_map, unit_grid, unit_policy

from contextmap.geometric_mapping import (
    AggregatedGeometry,
    Bounds3D,
    GeometryBlock,
    GeometryReference,
    InterScanVoxelAggregator,
    InterScanVoxelPolicy,
    MapId,
    PackedGeometry,
    ScanVoxelPolicy,
    VoxelAggregate,
    VoxelAggregation,
    VoxelAggregationError,
    VoxelContribution,
    VoxelGridSpec,
    aggregate_geometry,
    geometry_id_for,
)
from contextmap.ingestion import FrameId
from contextmap.shared import Vector3

TOLERANCE_M = 1e-12

# Três scans revisitam o voxel (0, 0, 0); o scan 0 também ocupa o voxel vizinho (1, 0, 0).
REVISIT: tuple[tuple[Vector3, ...], ...] = (
    ((0.25, 0.25, 0.25), (0.75, 0.50, 0.50), (0.50, 0.75, 0.25), (1.50, 0.50, 0.50)),
    ((0.40, 0.40, 0.40),),
    ((0.60, 0.20, 0.80),),
)


def _raw_coordinates(geometry: PackedGeometry) -> np.ndarray:
    return np.concatenate([block.coordinates_m for block in geometry.iter_blocks()])


def _aggregate_at(aggregation: VoxelAggregation, key: tuple[int, int, int]) -> VoxelAggregate:
    for index in range(len(aggregation)):
        aggregate = aggregation.aggregate(index)
        if aggregate.key == key:
            return aggregate
    raise AssertionError(f"no aggregate at {key}")


def _aggregate_blocks(
    geometry: PackedGeometry, policy: InterScanVoxelPolicy, blocks: Sequence[GeometryBlock]
) -> VoxelAggregation:
    aggregator = InterScanVoxelAggregator(
        policy=policy, source_map=geometry.geometric_map, scans=geometry.scans
    )
    for block in blocks:
        aggregator.add_block(block)
    return aggregator.finish()


# --- Grid identity ------------------------------------------------------------------


def test_the_grid_spec_declares_frame_origin_resolution_and_indexing() -> None:
    grid = unit_grid(cell_m=0.5, origin_m=(1.0, -2.0, 0.25))

    assert grid.to_record() == {
        "frame_id": "map",
        "origin_m": [1.0, -2.0, 0.25],
        "cell_m": 0.5,
        "indexing": "floor-half-open",
    }


@pytest.mark.parametrize(
    "changed",
    [
        unit_policy(frame="odom"),
        unit_policy(origin_m=(0.5, 0.0, 0.0)),
        unit_policy(cell_m=0.5),
    ],
    ids=["frame", "origin", "resolution"],
)
def test_changing_the_frame_the_origin_or_the_resolution_changes_the_fingerprint(
    changed: InterScanVoxelPolicy,
) -> None:
    assert unit_policy().fingerprint() == unit_policy().fingerprint()
    assert changed.fingerprint() != unit_policy().fingerprint()


def test_the_inter_scan_rule_is_distinct_from_the_intra_scan_rule() -> None:
    inter = unit_policy(cell_m=0.5)

    assert inter.rule == "inter-scan-voxel-centroid-0.5m"
    assert inter.rule != ScanVoxelPolicy(cell_m=0.5).rule


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cell_m": 0.0},
        {"cell_m": -1.0},
        {"cell_m": math.inf},
        {"origin_m": (math.nan, 0.0, 0.0)},
        {"frame": ""},
    ],
)
def test_an_unusable_grid_is_refused(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"cell_m|origin_m|frame_id"):
        unit_grid(**kwargs)  # type: ignore[arg-type]


def test_a_voxel_is_half_open_and_aligned_to_the_grid_origin() -> None:
    grid = unit_grid(cell_m=0.5, origin_m=(0.25, 0.0, 0.0))
    points = np.array([[0.25, 0.0, 0.0], [0.7499, 0.0, 0.0], [0.75, 0.0, 0.0], [0.2, -0.1, 0.0]])

    assert grid.keys_of(points).tolist() == [[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, -1, 0]]


def test_coordinates_too_far_for_an_exact_key_are_refused() -> None:
    grid = unit_grid(cell_m=1e-9)

    with pytest.raises(VoxelAggregationError, match="exact"):
        grid.keys_of(np.array([[1e9, 0.0, 0.0]]))


# --- What an aggregate says ---------------------------------------------------------


def test_points_of_different_scans_in_one_voxel_merge_into_their_centroid() -> None:
    geometry = revisit_map(REVISIT)
    raw = _raw_coordinates(geometry)
    members = raw[[0, 1, 2, 4, 5]]

    aggregate = _aggregate_at(aggregate_geometry(geometry, unit_policy()), (0, 0, 0))

    expected = tuple(math.fsum(members[:, axis]) / len(members) for axis in range(3))
    assert aggregate.centroid_m == pytest.approx(expected, abs=TOLERANCE_M)
    assert aggregate.minimum_m == tuple(members.min(axis=0).tolist())
    assert aggregate.maximum_m == tuple(members.max(axis=0).tolist())


def test_point_scan_and_observation_counts_keep_distinct_meanings() -> None:
    aggregation = aggregate_geometry(revisit_map(REVISIT), unit_policy())

    revisited = _aggregate_at(aggregation, (0, 0, 0))
    single_scan = _aggregate_at(aggregation, (1, 0, 0))

    assert (revisited.point_count, revisited.scan_count, revisited.observation_count) == (5, 3, 3)
    assert (single_scan.point_count, single_scan.scan_count, single_scan.observation_count) == (
        1,
        1,
        1,
    )


def test_two_observations_of_the_same_surface_stay_two_physical_contributions() -> None:
    geometry = revisit_map((((0.5, 0.5, 0.5),), ((0.5, 0.5, 0.5),)))

    aggregation = aggregate_geometry(geometry, unit_policy())

    assert len(aggregation) == 1
    aggregate = aggregation.aggregate(0)
    assert aggregate.point_count == 2
    assert aggregate.contributions == (
        VoxelContribution(scan_ordinal=0, point_count=1),
        VoxelContribution(scan_ordinal=1, point_count=1),
    )


def test_the_lineage_records_each_scan_with_its_own_point_count() -> None:
    aggregate = _aggregate_at(aggregate_geometry(revisit_map(REVISIT), unit_policy()), (0, 0, 0))

    assert aggregate.contributions == (
        VoxelContribution(scan_ordinal=0, point_count=3),
        VoxelContribution(scan_ordinal=1, point_count=1),
        VoxelContribution(scan_ordinal=2, point_count=1),
    )


def test_first_and_last_observed_come_from_the_contributing_scans() -> None:
    geometry = revisit_map(REVISIT)
    aggregation = aggregate_geometry(geometry, unit_policy())

    revisited = _aggregate_at(aggregation, (0, 0, 0))
    single_scan = _aggregate_at(aggregation, (1, 0, 0))

    assert revisited.first_observed_at == geometry.scans[0].acquisition_timestamp
    assert revisited.last_observed_at == geometry.scans[2].acquisition_timestamp
    assert single_scan.first_observed_at == single_scan.last_observed_at


def test_points_in_distinct_voxels_are_never_merged() -> None:
    geometry = revisit_map((((0.999, 0.5, 0.5), (1.001, 0.5, 0.5)),))

    aggregation = aggregate_geometry(geometry, unit_policy())

    assert [aggregation.aggregate(i).key for i in range(len(aggregation))] == [
        (0, 0, 0),
        (1, 0, 0),
    ]
    assert [aggregation.aggregate(i).point_count for i in range(len(aggregation))] == [1, 1]


def test_the_grid_origin_decides_what_shares_a_voxel() -> None:
    geometry = revisit_map((((0.9, 0.5, 0.5), (1.1, 0.5, 0.5)),))

    assert len(aggregate_geometry(geometry, unit_policy())) == 2
    assert len(aggregate_geometry(geometry, unit_policy(origin_m=(0.5, 0.0, 0.0)))) == 1


def test_the_intra_scan_policy_never_merges_scans_but_the_inter_scan_one_does() -> None:
    scans = make_scans(2, points=((0.1, 0.0, 0.0),))
    # O scan 1 vem de uma pose 1 m à frente, então o mesmo ponto de origem cai 1 m adiante no
    # mapa: com voxels de 2 m os dois pontos dividem um voxel, mas pertencem a scans distintos.
    accumulated, _ = accumulate(scans, aggregation=ScanVoxelPolicy(cell_m=2.0))
    assert accumulated.geometric_map.point_count == 2

    raw, packed = accumulate(scans)
    geometry = PackedGeometry(geometric_map=raw.geometric_map, scans=raw.scans, records=packed)
    aggregation = aggregate_geometry(geometry, unit_policy(cell_m=2.0))
    assert len(aggregation) == 1
    assert aggregation.aggregate(0).scan_count == 2


def test_contribution_counts_sum_to_the_eligible_input_points() -> None:
    geometry = revisit_map(REVISIT)

    aggregation = aggregate_geometry(geometry, unit_policy())

    assert aggregation.source_point_count == geometry.geometric_map.point_count == 6
    total = sum(
        contribution.point_count
        for index in range(len(aggregation))
        for contribution in aggregation.aggregate(index).contributions
    )
    assert total == 6


def test_an_aggregate_carries_no_semantic_state() -> None:
    forbidden = {
        "label",
        "labels",
        "claim",
        "claims",
        "entity",
        "entity_id",
        "instance_id",
        "embedding",
        "embeddings",
        "feature",
        "features",
        "hypotheses",
    }

    for contract in (VoxelAggregate, VoxelAggregation, VoxelContribution):
        assert not forbidden & {field.name for field in dataclasses.fields(contract)}


# --- Lineage recovery ----------------------------------------------------------------


def test_the_lineage_recovers_exactly_the_contributing_raw_references() -> None:
    geometry = revisit_map(REVISIT)
    aggregation = aggregate_geometry(geometry, unit_policy())

    revisited = _aggregate_at(aggregation, (0, 0, 0))
    members = aggregation.members_of(revisited.index, geometry)

    assert members == tuple(
        GeometryReference(map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=index))
        for index in (0, 1, 2, 4, 5)
    )


def test_every_raw_point_belongs_to_exactly_one_aggregate_of_its_own_voxel() -> None:
    geometry = revisit_map(REVISIT)
    aggregation = aggregate_geometry(geometry, unit_policy())

    membership = aggregation.membership(geometry)

    keys = unit_grid().keys_of(_raw_coordinates(geometry))
    assert membership.shape == (6,)
    assert (aggregation.keys[membership] == keys).all()
    assert aggregation.verify_lineage(geometry) == []


def test_lineage_verification_detects_a_different_source_map() -> None:
    aggregation = aggregate_geometry(revisit_map(REVISIT), unit_policy())
    other = revisit_map((((5.5, 0.5, 0.5),), ((0.5, 0.5, 0.5),), ((0.5, 0.5, 0.5),)))

    assert aggregation.verify_lineage(other)


# --- Determinism, order and batching ------------------------------------------------


def _shuffled_rows(geometry: PackedGeometry, seed: int) -> list[GeometryBlock]:
    rows = np.random.default_rng(seed).permutation(geometry.geometric_map.point_count)
    coordinates = _raw_coordinates(geometry)
    return [
        GeometryBlock(
            map_id=geometry.geometric_map.map_id,
            frame_id=geometry.geometric_map.frame_id,
            indices=np.array([row], dtype=np.int64),
            coordinates_m=coordinates[[row]],
        )
        for row in rows
    ]


def test_the_aggregation_does_not_depend_on_the_input_order() -> None:
    geometry = revisit_map(REVISIT)
    in_order = aggregate_geometry(geometry, unit_policy())

    for seed in range(5):
        shuffled = _aggregate_blocks(geometry, unit_policy(), _shuffled_rows(geometry, seed))
        assert shuffled.equivalence_problems(in_order) == []
        assert (shuffled.keys == in_order.keys).all()
        assert (shuffled.point_counts == in_order.point_counts).all()


@pytest.mark.parametrize("block_points", [1, 2, 3, 1_000_000])
def test_the_aggregation_is_invariant_to_how_the_input_is_chunked(block_points: int) -> None:
    geometry = revisit_map(REVISIT)
    reference = aggregate_geometry(geometry, unit_policy(), block_points=1_000_000)

    chunked = aggregate_geometry(geometry, unit_policy(), block_points=block_points)

    assert chunked.equivalence_problems(reference) == []


def test_equal_inputs_and_chunking_reproduce_identical_values() -> None:
    first = aggregate_geometry(revisit_map(REVISIT), unit_policy(), block_points=2)
    second = aggregate_geometry(revisit_map(REVISIT), unit_policy(), block_points=2)

    assert (first.offset_sums_m == second.offset_sums_m).all()
    assert (first.centroids_m == second.centroids_m).all()


def test_equivalence_reports_a_changed_count_and_a_moved_centroid() -> None:
    reference = aggregate_geometry(revisit_map(REVISIT), unit_policy())
    moved = revisit_map(
        (
            ((0.25, 0.25, 0.25), (0.75, 0.50, 0.50), (0.50, 0.75, 0.25), (1.50, 0.50, 0.50)),
            ((0.40, 0.40, 0.40),),
            ((0.60, 0.20, 0.90),),
        )
    )

    problems = aggregate_geometry(moved, unit_policy()).equivalence_problems(reference)

    assert any("centroid" in problem for problem in problems)


# --- What the aggregator refuses ------------------------------------------------------


def test_a_map_already_aggregated_inside_its_scans_is_refused() -> None:
    accumulated, packed = accumulate(make_scans(2), aggregation=ScanVoxelPolicy(cell_m=0.5))
    geometry = PackedGeometry(
        geometric_map=accumulated.geometric_map, scans=accumulated.scans, records=packed
    )

    with pytest.raises(VoxelAggregationError, match="raw"):
        aggregate_geometry(geometry, unit_policy())


def test_a_grid_in_another_frame_than_the_map_is_refused() -> None:
    with pytest.raises(VoxelAggregationError, match="frame"):
        aggregate_geometry(revisit_map(REVISIT), unit_policy(frame="odom"))


def test_a_block_of_another_map_is_refused() -> None:
    geometry = revisit_map(REVISIT)
    aggregator = InterScanVoxelAggregator(
        policy=unit_policy(), source_map=geometry.geometric_map, scans=geometry.scans
    )
    block = next(geometry.iter_blocks())

    with pytest.raises(VoxelAggregationError, match="map"):
        aggregator.add_block(dataclasses.replace(block, map_id=MapId("another-map")))


def test_a_point_contributed_twice_is_refused() -> None:
    geometry = revisit_map(REVISIT)
    aggregator = InterScanVoxelAggregator(
        policy=unit_policy(), source_map=geometry.geometric_map, scans=geometry.scans
    )
    block = next(geometry.iter_blocks())
    aggregator.add_block(block)

    with pytest.raises(VoxelAggregationError, match="twice"):
        aggregator.add_block(block)


def test_an_aggregation_that_misses_raw_points_is_refused() -> None:
    geometry = revisit_map(REVISIT)
    aggregator = InterScanVoxelAggregator(
        policy=unit_policy(), source_map=geometry.geometric_map, scans=geometry.scans
    )
    aggregator.add_block(next(geometry.iter_blocks()))

    with pytest.raises(VoxelAggregationError, match="every raw point"):
        aggregator.finish()


def test_a_finished_aggregator_accepts_no_more_blocks() -> None:
    geometry = revisit_map(REVISIT)
    aggregator = InterScanVoxelAggregator(
        policy=unit_policy(), source_map=geometry.geometric_map, scans=geometry.scans
    )
    for block in geometry.iter_blocks():
        aggregator.add_block(block)
    aggregator.finish()

    with pytest.raises(VoxelAggregationError, match="finished"):
        aggregator.add_block(next(geometry.iter_blocks()))


# --- The aggregates as a block read boundary ------------------------------------------


def _aggregated_geometry(geometry: PackedGeometry) -> AggregatedGeometry:
    return AggregatedGeometry.derive(
        aggregation=aggregate_geometry(geometry, unit_policy()),
        source_map=geometry.geometric_map,
        map_id=MapId("map-0001--voxel-1.0m"),
        code_version="test",
    )


def test_the_aggregates_are_served_as_centroid_blocks_with_their_own_indices() -> None:
    geometry = revisit_map(REVISIT)
    aggregated = _aggregated_geometry(geometry)

    blocks = list(aggregated.iter_blocks(block_points=1))

    assert [block.indices.tolist() for block in blocks] == [[0], [1]]
    assert all(block.map_id == MapId("map-0001--voxel-1.0m") for block in blocks)
    assert np.concatenate([b.coordinates_m for b in blocks]).tolist() == (
        aggregated.aggregation.centroids_m.tolist()
    )


def test_the_derived_map_declares_its_rule_and_keeps_the_source_lineage() -> None:
    geometry = revisit_map(REVISIT)

    derived = _aggregated_geometry(geometry).geometric_map

    source = geometry.geometric_map
    assert derived.map_id != source.map_id
    assert derived.aggregation_rule == "inter-scan-voxel-centroid-1.0m"
    assert derived.point_count == 2
    assert derived.frame_id == source.frame_id
    assert derived.time_bounds == source.time_bounds
    assert derived.source_observation_ids == source.source_observation_ids
    assert derived.provenance.trajectory_id == source.provenance.trajectory_id
    assert derived.provenance.sequence_artifact_id == source.provenance.sequence_artifact_id
    assert derived.provenance.calibration_identity == source.provenance.calibration_identity
    assert derived.provenance.configuration_fingerprint == unit_policy().fingerprint()


def test_a_bounds_query_on_the_aggregates_selects_centroids_inclusively() -> None:
    aggregated = _aggregated_geometry(revisit_map(REVISIT))
    centroid = aggregated.aggregation.centroids_m[1]
    box = Bounds3D(
        frame_id=FrameId("map"), minimum_m=tuple(centroid.tolist()), maximum_m=(9.0, 9.0, 9.0)
    )

    blocks = list(aggregated.iter_blocks(bounds=box))

    assert [block.indices.tolist() for block in blocks] == [[1]]


def test_a_bounds_query_in_another_frame_is_refused() -> None:
    aggregated = _aggregated_geometry(revisit_map(REVISIT))
    box = Bounds3D(frame_id=FrameId("odom"), minimum_m=(0.0, 0.0, 0.0), maximum_m=(1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="frame"):
        list(aggregated.iter_blocks(bounds=box))


def test_the_grid_spec_is_a_frozen_value() -> None:
    grid = unit_grid()

    with pytest.raises(dataclasses.FrozenInstanceError):
        grid.cell_m = 2.0  # type: ignore[misc]
    assert isinstance(grid, VoxelGridSpec)


def test_a_scan_without_geometry_never_receives_a_contribution() -> None:
    nan = float("nan")
    geometry = revisit_map((((0.5, 0.5, 0.5),), ((nan, nan, nan),), ((0.5, 0.5, 0.5),)))
    assert geometry.scans[1].geometry_count == 0

    aggregate = aggregate_geometry(geometry, unit_policy()).aggregate(0)

    assert [c.scan_ordinal for c in aggregate.contributions] == [0, 2]


@pytest.mark.parametrize(
    "change",
    [
        {"map_id": MAP_ID},
        {"point_count": 3},
        {"aggregation_rule": "scan-voxel-centroid-1.0m"},
    ],
    ids=["raw-identity", "element-count", "rule"],
)
def test_a_derived_map_that_does_not_describe_the_aggregation_is_refused(
    change: dict[str, object],
) -> None:
    aggregated = _aggregated_geometry(revisit_map(REVISIT))
    wrong = dataclasses.replace(aggregated.geometric_map, **change)  # type: ignore[arg-type]

    with pytest.raises(VoxelAggregationError):
        AggregatedGeometry(aggregation=aggregated.aggregation, geometric_map=wrong)

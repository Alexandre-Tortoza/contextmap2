import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from geometric_mapping_builders import SCAN_COUNT, WORLD, make_plan, write_run

from contextmap.evaluation import (
    PairedRegion,
    compare_paired_associations,
    encode_aggregation_fidelity_report,
    encode_association_stability_report,
    evaluate_aggregation_fidelity,
)
from contextmap.geometric_mapping import (
    AggregatedGeometry,
    GeometryReference,
    InterScanVoxelPolicy,
    MapId,
    PackedGeometry,
    VoxelAggregation,
    VoxelGridSpec,
    aggregate_geometry,
    geometry_id_for,
)
from contextmap.ingestion import FrameId

# A grade do mundo simulado tem 0,5 m de passo em múltiplos exatos de 0,5 m: uma origem
# deslocada de 1/8 de voxel mantém cada ponto do mundo longe de uma fronteira de voxel.
OFFSET_ORIGIN = (0.125, 0.125, 0.125)
RANGE_EDGES_M = (0.0, 2.0, 5.0, 10.0)


def _policy(
    cell_m: float, origin_m: tuple[float, float, float] = OFFSET_ORIGIN
) -> InterScanVoxelPolicy:
    return InterScanVoxelPolicy(
        grid=VoxelGridSpec(frame_id=FrameId("map"), origin_m=origin_m, cell_m=cell_m)
    )


@pytest.fixture
def corridor(tmp_path: Path) -> Iterator[PackedGeometry]:
    reader = write_run(tmp_path, make_plan())
    try:
        yield reader.geometry()
    finally:
        reader.close()


# --- Geometric fidelity -----------------------------------------------------------------


def test_revisits_of_one_world_point_collapse_with_negligible_error(
    corridor: PackedGeometry,
) -> None:
    aggregation = aggregate_geometry(corridor, _policy(0.25))

    report = evaluate_aggregation_fidelity(corridor, aggregation, range_edges_m=RANGE_EDGES_M)

    # O piso e as paredes compartilham os pontos de z = 0: cada ponto distinto é um voxel.
    distinct = len(set(WORLD))
    assert report.source_point_count == SCAN_COUNT * len(WORLD)
    assert report.aggregate_count == distinct
    assert report.reduction_ratio == pytest.approx(distinct / (SCAN_COUNT * len(WORLD)))
    assert (aggregation.scan_counts == SCAN_COUNT).all()
    assert report.distance_m.maximum < 1e-9


def test_the_distance_to_the_representative_never_exceeds_the_voxel_diagonal(
    corridor: PackedGeometry,
) -> None:
    aggregation = aggregate_geometry(corridor, _policy(2.0))

    report = evaluate_aggregation_fidelity(corridor, aggregation, range_edges_m=RANGE_EDGES_M)

    assert 0.1 < report.distance_m.maximum <= 2.0 * math.sqrt(3.0)
    assert report.distance_m.count == SCAN_COUNT * len(WORLD)
    assert report.distance_m.minimum <= report.distance_m.median <= report.distance_m.p95
    assert report.mean_distance_m == pytest.approx(
        float(np.mean(_distances(corridor, aggregation)))
    )


def _distances(geometry: PackedGeometry, aggregation: VoxelAggregation) -> np.ndarray:
    membership = aggregation.membership(geometry)
    coordinates = np.concatenate([block.coordinates_m for block in geometry.iter_blocks()])
    distances: np.ndarray = np.linalg.norm(
        coordinates - aggregation.centroids_m[membership], axis=1
    )
    return distances


def test_the_error_is_reported_per_range_to_the_sensor_that_measured_the_point(
    corridor: PackedGeometry,
) -> None:
    aggregation = aggregate_geometry(corridor, _policy(2.0))

    report = evaluate_aggregation_fidelity(corridor, aggregation, range_edges_m=RANGE_EDGES_M)

    ranges = [math.hypot(*point.source_coordinates_m) for point in corridor.iter_geometry()]
    expected = [
        sum(1 for value in ranges if low <= value < high)
        for low, high in zip(RANGE_EDGES_M, (*RANGE_EDGES_M[1:], math.inf), strict=True)
    ]
    assert [band.point_count for band in report.by_range] == expected
    assert [band.minimum_range_m for band in report.by_range] == list(RANGE_EDGES_M)
    assert report.by_range[-1].maximum_range_m is None


def test_the_statistics_keep_the_raw_bounds_and_the_centroids_stay_inside_them(
    corridor: PackedGeometry,
) -> None:
    aggregation = aggregate_geometry(corridor, _policy(2.0))

    report = evaluate_aggregation_fidelity(corridor, aggregation, range_edges_m=RANGE_EDGES_M)

    assert report.statistics_bounds_preserved
    raw, centroids = report.raw_bounds, report.centroid_bounds
    assert all(raw.minimum_m[axis] <= centroids.minimum_m[axis] for axis in range(3))
    assert all(centroids.maximum_m[axis] <= raw.maximum_m[axis] for axis in range(3))
    assert 0.0 < report.max_bounds_shrinkage_m <= 2.0


@pytest.mark.parametrize("edges", [(), (1.0, 2.0), (0.0, 2.0, 2.0), (0.0, math.inf)])
def test_range_bands_must_start_at_zero_and_increase(
    corridor: PackedGeometry, edges: tuple[float, ...]
) -> None:
    aggregation = aggregate_geometry(corridor, _policy(2.0))

    with pytest.raises(ValueError, match="range_edges_m"):
        evaluate_aggregation_fidelity(corridor, aggregation, range_edges_m=edges)


def test_the_fidelity_report_is_plain_json(corridor: PackedGeometry) -> None:
    aggregation = aggregate_geometry(corridor, _policy(2.0))
    report = evaluate_aggregation_fidelity(corridor, aggregation, range_edges_m=RANGE_EDGES_M)

    record = json.loads(json.dumps(encode_aggregation_fidelity_report(report)))

    assert record["policy_fingerprint"] == _policy(2.0).fingerprint()
    assert record["distance_m"]["maximum"] == report.distance_m.maximum
    assert len(record["by_range"]) == len(RANGE_EDGES_M)


# --- 2D↔3D association stability ---------------------------------------------------------


def _raw_refs(geometry: PackedGeometry, *indices: int) -> tuple[GeometryReference, ...]:
    map_id = geometry.geometric_map.map_id
    return tuple(
        GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=i))
        for i in sorted(indices)
    )


def _aggregate_refs(derived: AggregatedGeometry, *indices: int) -> tuple[GeometryReference, ...]:
    map_id = derived.geometric_map.map_id
    return tuple(
        GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=i))
        for i in sorted(indices)
    )


def _derived(geometry: PackedGeometry, cell_m: float = 0.25) -> AggregatedGeometry:
    return AggregatedGeometry.derive(
        aggregation=aggregate_geometry(geometry, _policy(cell_m)),
        source_map=geometry.geometric_map,
        map_id=MapId("derived-map"),
        code_version="test",
    )


REGION_A = PairedRegion(source_observation_id="frame-0", perception_result_id="r-0", region_id="a")
REGION_B = PairedRegion(source_observation_id="frame-0", perception_result_id="r-0", region_id="b")
REGION_C = PairedRegion(source_observation_id="frame-1", perception_result_id="r-1", region_id="c")


def test_an_aggregate_support_that_covers_the_raw_voxels_is_fully_retained(
    corridor: PackedGeometry,
) -> None:
    derived = _derived(corridor)
    membership = derived.aggregation.membership(corridor)
    # Dois pontos do mundo, cada um visto por dois scans: dois voxels de mesmo peso.
    revisited = (0, 1, len(WORLD), len(WORLD) + 1)
    raw = {REGION_A: _raw_refs(corridor, *revisited)}
    same_voxels = sorted({int(membership[i]) for i in revisited})

    report = compare_paired_associations(
        raw_supports=raw,
        aggregate_supports={REGION_A: _aggregate_refs(derived, *same_voxels)},
        raw_geometry=corridor,
        aggregated=derived,
        membership=membership,
        thin_support_max_points=1,
    )

    region = report.regions[0]
    assert (region.raw_support_count, region.mapped_raw_support_count) == (4, 2)
    assert region.support_retention == 1.0
    assert region.disagreement == 0.0
    assert report.region_retention_rate == 1.0
    assert region.support_centroid_shift_m is not None
    assert region.support_centroid_shift_m < 1e-9


def test_partial_overlap_measures_retention_and_disagreement_on_voxels(
    corridor: PackedGeometry,
) -> None:
    derived = _derived(corridor)
    membership = derived.aggregation.membership(corridor)
    raw_voxels = [int(membership[i]) for i in (0, 1)]
    other = next(v for v in range(len(derived.aggregation)) if v not in raw_voxels)

    report = compare_paired_associations(
        raw_supports={REGION_A: _raw_refs(corridor, 0, 1)},
        aggregate_supports={REGION_A: _aggregate_refs(derived, raw_voxels[1], other)},
        raw_geometry=corridor,
        aggregated=derived,
        membership=membership,
        thin_support_max_points=1,
    )

    region = report.regions[0]
    assert region.shared_aggregate_count == 1
    assert region.support_retention == pytest.approx(0.5)
    assert region.disagreement == pytest.approx(1 - 1 / 3)


def test_lost_and_gained_regions_are_counted_apart(corridor: PackedGeometry) -> None:
    derived = _derived(corridor)
    membership = derived.aggregation.membership(corridor)

    report = compare_paired_associations(
        raw_supports={
            REGION_A: _raw_refs(corridor, 0),
            REGION_B: (),
            REGION_C: _raw_refs(corridor, 5),
        },
        aggregate_supports={
            REGION_A: (),
            REGION_B: _aggregate_refs(derived, 3),
            REGION_C: _aggregate_refs(derived, int(membership[5])),
        },
        raw_geometry=corridor,
        aggregated=derived,
        membership=membership,
        thin_support_max_points=1,
    )

    assert report.paired_region_count == 3
    assert report.raw_supported_region_count == 2
    assert (report.retained_region_count, report.lost_region_count) == (1, 1)
    assert report.gained_region_count == 1
    assert report.region_retention_rate == pytest.approx(0.5)
    assert (report.thin_region_count, report.thin_retained_region_count) == (2, 1)


def test_regions_present_in_only_one_run_are_reported_not_compared(
    corridor: PackedGeometry,
) -> None:
    derived = _derived(corridor)

    report = compare_paired_associations(
        raw_supports={REGION_A: _raw_refs(corridor, 0), REGION_B: _raw_refs(corridor, 1)},
        aggregate_supports={REGION_A: _aggregate_refs(derived, 0), REGION_C: ()},
        raw_geometry=corridor,
        aggregated=derived,
        membership=derived.aggregation.membership(corridor),
        thin_support_max_points=1,
    )

    assert report.paired_region_count == 1
    assert (report.raw_only_region_count, report.aggregate_only_region_count) == (1, 1)


def test_a_single_point_support_has_no_bounds_overlap_to_measure(
    corridor: PackedGeometry,
) -> None:
    derived = _derived(corridor)
    membership = derived.aggregation.membership(corridor)

    report = compare_paired_associations(
        raw_supports={REGION_A: _raw_refs(corridor, 0)},
        aggregate_supports={REGION_A: _aggregate_refs(derived, int(membership[0]))},
        raw_geometry=corridor,
        aggregated=derived,
        membership=membership,
        thin_support_max_points=1,
    )

    assert report.regions[0].support_bounds_iou is None
    assert report.support_bounds_iou is None


def test_the_support_bounds_overlap_compares_raw_points_with_their_centroids(
    corridor: PackedGeometry,
) -> None:
    derived = _derived(corridor, cell_m=2.0)
    membership = derived.aggregation.membership(corridor)
    raw = _raw_refs(corridor, *range(0, 60))

    report = compare_paired_associations(
        raw_supports={REGION_A: raw},
        aggregate_supports={
            REGION_A: _aggregate_refs(derived, *sorted({int(membership[i]) for i in range(60)}))
        },
        raw_geometry=corridor,
        aggregated=derived,
        membership=membership,
        thin_support_max_points=1,
    )

    iou = report.regions[0].support_bounds_iou
    assert iou is not None
    assert 0.0 < iou <= 1.0


def test_references_of_another_map_are_refused(corridor: PackedGeometry) -> None:
    derived = _derived(corridor)

    with pytest.raises(ValueError, match="map"):
        compare_paired_associations(
            raw_supports={REGION_A: _aggregate_refs(derived, 0)},
            aggregate_supports={REGION_A: _aggregate_refs(derived, 0)},
            raw_geometry=corridor,
            aggregated=derived,
            membership=derived.aggregation.membership(corridor),
            thin_support_max_points=1,
        )


def test_a_membership_of_another_map_is_refused(corridor: PackedGeometry) -> None:
    derived = _derived(corridor)

    with pytest.raises(ValueError, match="membership"):
        compare_paired_associations(
            raw_supports={REGION_A: _raw_refs(corridor, 0)},
            aggregate_supports={REGION_A: _aggregate_refs(derived, 0)},
            raw_geometry=corridor,
            aggregated=derived,
            membership=np.zeros(3, dtype=np.int64),
            thin_support_max_points=1,
        )


def test_the_stability_report_is_plain_json(corridor: PackedGeometry) -> None:
    derived = _derived(corridor)
    membership = derived.aggregation.membership(corridor)
    report = compare_paired_associations(
        raw_supports={REGION_A: _raw_refs(corridor, 0, 1)},
        aggregate_supports={REGION_A: _aggregate_refs(derived, int(membership[0]))},
        raw_geometry=corridor,
        aggregated=derived,
        membership=membership,
        thin_support_max_points=1,
    )

    record = json.loads(json.dumps(encode_association_stability_report(report)))

    assert record["paired_region_count"] == 1
    assert record["regions"][0]["region_id"] == "a"
    assert record["support_retention"]["maximum"] == pytest.approx(0.5)

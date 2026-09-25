import numpy as np
import pytest
from projection_builders import MAP_ID, ArrayGeometrySource, make_calibration

from contextmap.geometric_mapping import GeometryReference, geometry_id_for
from contextmap.sensor_association.candidate_geometry import (
    CandidateGeometryCloud,
    CandidateGeometryPolicy,
    select_candidate_geometry,
)

# Uma linha de pontos ao longo de +x, um por metro, para o alcance ser o próprio x.
LINE = [(float(x), 0.0, 0.0) for x in range(0, 40)]


def _source(points: list[tuple[float, float, float]] | None = None) -> ArrayGeometrySource:
    return ArrayGeometrySource(
        points if points is not None else LINE, calibration=make_calibration()
    )


def _reference(index: int) -> GeometryReference:
    return GeometryReference(map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=index))


# --- The policy ----------------------------------------------------------------------------


def test_the_policy_is_versioned_and_records_its_parameters() -> None:
    policy = CandidateGeometryPolicy(max_range_m=20.0, block_points=4096)

    record = policy.to_record()

    assert record["policy_id"] == CandidateGeometryPolicy.policy_id
    assert record["max_range_m"] == 20.0
    assert record["block_points"] == 4096
    assert policy.fingerprint().startswith("sha256:")


def test_two_policies_that_differ_have_different_fingerprints() -> None:
    assert (
        CandidateGeometryPolicy(max_range_m=20.0).fingerprint()
        != CandidateGeometryPolicy(max_range_m=10.0).fingerprint()
    )


def test_an_unbounded_policy_is_explicit_and_distinguishable() -> None:
    unbounded = CandidateGeometryPolicy(max_range_m=None)

    assert unbounded.to_record()["max_range_m"] is None
    assert unbounded.fingerprint() != CandidateGeometryPolicy(max_range_m=20.0).fingerprint()


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_range_that_is_not_a_positive_finite_distance_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="max_range_m"):
        CandidateGeometryPolicy(max_range_m=bad)


def test_a_block_size_that_is_not_positive_is_rejected() -> None:
    with pytest.raises(ValueError, match="block_points"):
        CandidateGeometryPolicy(max_range_m=20.0, block_points=0)


# --- Selection -----------------------------------------------------------------------------


def test_an_unbounded_policy_selects_every_map_element_in_index_order() -> None:
    selection = select_candidate_geometry(
        _source(), camera_center_m=(0.0, 0.0, 0.0), policy=CandidateGeometryPolicy(max_range_m=None)
    )

    assert selection.candidate_count == len(LINE)
    assert selection.map_point_count == len(LINE)
    assert selection.candidate_fraction == 1.0
    np.testing.assert_array_equal(selection.cloud.global_indices, np.arange(len(LINE)))
    np.testing.assert_array_equal(selection.cloud.coordinates_m, np.array(LINE))


def test_a_bounded_policy_selects_exactly_the_geometry_within_the_range() -> None:
    selection = select_candidate_geometry(
        _source(), camera_center_m=(10.0, 0.0, 0.0), policy=CandidateGeometryPolicy(max_range_m=4.0)
    )

    # Pontos de x=6 a x=14 estão a 4 m ou menos do centro em x=10; as bordas entram.
    np.testing.assert_array_equal(selection.cloud.global_indices, np.arange(6, 15))
    assert selection.candidate_count == 9
    assert selection.map_point_count == len(LINE)


def test_the_range_is_a_sphere_around_the_camera_not_an_axis_aligned_box() -> None:
    # (3, 4, 0) está a 5 m da origem: dentro da caixa de 4 m, fora da esfera de 4 m.
    points = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (0.0, 0.0, 3.9)]

    selection = select_candidate_geometry(
        _source(points),
        camera_center_m=(0.0, 0.0, 0.0),
        policy=CandidateGeometryPolicy(max_range_m=4.0),
    )

    np.testing.assert_array_equal(selection.cloud.global_indices, np.array([0, 2]))


def test_a_point_exactly_at_the_range_boundary_is_kept() -> None:
    selection = select_candidate_geometry(
        _source([(5.0, 0.0, 0.0)]),
        camera_center_m=(0.0, 0.0, 0.0),
        policy=CandidateGeometryPolicy(max_range_m=5.0),
    )

    assert selection.candidate_count == 1


def test_selecting_nothing_is_a_valid_empty_candidate_set() -> None:
    selection = select_candidate_geometry(
        _source(),
        camera_center_m=(500.0, 0.0, 0.0),
        policy=CandidateGeometryPolicy(max_range_m=1.0),
    )

    assert selection.candidate_count == 0
    assert selection.candidate_fraction == 0.0
    assert selection.cloud.coordinates_m.shape == (0, 3)
    assert selection.cloud.global_indices.shape == (0,)


def test_the_selection_reports_what_it_cost_and_how_much_it_read() -> None:
    selection = select_candidate_geometry(
        _source(), camera_center_m=(10.0, 0.0, 0.0), policy=CandidateGeometryPolicy(max_range_m=4.0)
    )

    assert selection.query_seconds >= 0.0
    assert selection.candidate_count <= selection.queried_count <= selection.map_point_count


# --- Identity is global, never the row ------------------------------------------------------


def test_a_candidate_row_keeps_the_persistent_identity_of_the_map_element() -> None:
    selection = select_candidate_geometry(
        _source(), camera_center_m=(10.0, 0.0, 0.0), policy=CandidateGeometryPolicy(max_range_m=4.0)
    )
    cloud = selection.cloud

    # A linha 0 é o elemento global 6: identidade local nunca é identidade global.
    assert cloud.reference(0) == _reference(6)
    assert cloud.reference(0) != _reference(0)
    assert cloud.references(np.array([2, 0])) == (_reference(8), _reference(6))


def test_the_same_geometry_keeps_the_same_reference_whatever_the_candidate_set() -> None:
    wide = select_candidate_geometry(
        _source(), camera_center_m=(10.0, 0.0, 0.0), policy=CandidateGeometryPolicy(max_range_m=8.0)
    ).cloud
    narrow = select_candidate_geometry(
        _source(), camera_center_m=(10.0, 0.0, 0.0), policy=CandidateGeometryPolicy(max_range_m=2.0)
    ).cloud

    shared = sorted(set(wide.global_indices.tolist()) & set(narrow.global_indices.tolist()))
    assert shared
    rows_in_wide, _ = wide.rows_for(np.array(shared))
    rows_in_narrow, _ = narrow.rows_for(np.array(shared))

    # As linhas diferem entre os dois conjuntos, as identidades não.
    assert rows_in_wide.tolist() != rows_in_narrow.tolist()
    for index, wide_row, narrow_row in zip(shared, rows_in_wide, rows_in_narrow, strict=True):
        assert (
            wide.reference(int(wide_row)) == narrow.reference(int(narrow_row)) == _reference(index)
        )


def test_rows_are_resolved_from_global_indices_and_missing_geometry_is_reported() -> None:
    cloud = select_candidate_geometry(
        _source(), camera_center_m=(10.0, 0.0, 0.0), policy=CandidateGeometryPolicy(max_range_m=2.0)
    ).cloud

    rows, found = cloud.rows_for(np.array([9, 11, 30, 8]))

    np.testing.assert_array_equal(found, np.array([True, True, False, True]))
    np.testing.assert_array_equal(cloud.global_indices[rows[found]], np.array([9, 11, 8]))


# --- The cloud's own contract ---------------------------------------------------------------


def test_a_cloud_whose_arrays_disagree_is_rejected() -> None:
    source = _source()

    with pytest.raises(ValueError, match="one global index per row"):
        CandidateGeometryCloud(
            geometric_map=source.geometric_map,
            coordinates_m=np.zeros((3, 3)),
            global_indices=np.array([0, 1], dtype=np.int64),
        )


def test_a_cloud_whose_indices_are_not_strictly_increasing_is_rejected() -> None:
    source = _source()

    with pytest.raises(ValueError, match="strictly increasing"):
        CandidateGeometryCloud(
            geometric_map=source.geometric_map,
            coordinates_m=np.zeros((3, 3)),
            global_indices=np.array([2, 1, 5], dtype=np.int64),
        )


def test_a_cloud_naming_geometry_outside_the_map_is_rejected() -> None:
    source = _source()

    with pytest.raises(ValueError, match="outside the map"):
        CandidateGeometryCloud(
            geometric_map=source.geometric_map,
            coordinates_m=np.zeros((1, 3)),
            global_indices=np.array([len(LINE)], dtype=np.int64),
        )

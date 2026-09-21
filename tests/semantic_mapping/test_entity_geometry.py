import dataclasses
import json
import math
import random
from collections.abc import Sequence
from typing import Any

import pytest
from mapping_builders import (
    MAP_ID,
    SEMANTIC_MAP_ID,
    geometry_refs,
    make_entity,
    make_geometry,
)
from mapping_geometry_fake import InMemoryGeometrySource

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometryPoint,
    GeometryReference,
    MapId,
)
from contextmap.ingestion import FrameId
from contextmap.semantic_mapping import (
    GEOMETRY_SUMMARY_ALGORITHM_ID,
    EmptyGeometrySupportError,
    EntityGeometry,
    EntityOrientation,
    GeometryDiagnostic,
    GeometryDiagnosticKind,
    GeometryResolutionError,
    GeometrySummaryPolicy,
    OrientationPolicy,
    SupportStatistics,
    geometry_set_digest,
    resolve_geometry,
    summarize_geometry,
    verify_geometry_summary,
)
from contextmap.semantic_mapping.serialization import decode_entity, encode_entity
from contextmap.shared import Vector3

POLICY = GeometrySummaryPolicy(sparse_point_threshold=4, connectivity_radius_m=1.5)
ORIENTED = GeometrySummaryPolicy(
    sparse_point_threshold=4,
    connectivity_radius_m=1.5,
    orientation=OrientationPolicy(min_points=8, min_variance_ratio=2.0),
)

CUBE = [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]


def _grid(nx: int, ny: int, nz: int, *, spacing: float = 1.0) -> list[Vector3]:
    return [
        (x * spacing, y * spacing, z * spacing)
        for x in range(nx)
        for y in range(ny)
        for z in range(nz)
    ]


def _source(points: Sequence[Vector3], *, frame: str = "map") -> InMemoryGeometrySource:
    return InMemoryGeometrySource(MAP_ID, dict(enumerate(points)), frame=frame)


def _summarize(points: Sequence[Vector3], policy: GeometrySummaryPolicy = POLICY) -> EntityGeometry:
    return summarize_geometry(
        geometry_refs(range(len(points))), source=_source(points), policy=policy
    )


def _close(actual: Vector3, expected: Vector3) -> bool:
    return all(math.isclose(a, e, abs_tol=1e-9) for a, e in zip(actual, expected, strict=True))


class TestDerivedSummaries:
    def test_the_cube_has_the_expected_centroid_bounds_extent_and_density(self) -> None:
        geometry = _summarize(CUBE)

        assert _close(geometry.centroid_m, (0.5, 0.5, 0.5))
        assert geometry.bounds == Bounds3D(
            frame_id=FrameId("map"), minimum_m=(0.0, 0.0, 0.0), maximum_m=(1.0, 1.0, 1.0)
        )
        assert geometry.extent_m == (1.0, 1.0, 1.0)
        assert geometry.statistics.point_count == 8
        assert geometry.statistics.volume_m3 == 1.0
        assert geometry.statistics.density_per_m3 == 8.0

    def test_the_persistent_references_stay_the_authoritative_support(self) -> None:
        geometry = _summarize(CUBE)

        assert geometry.geometry_refs == geometry_refs(range(8))
        assert geometry.geometric_map_id == MAP_ID

    def test_the_result_does_not_depend_on_the_order_of_the_references(self) -> None:
        shuffled = list(geometry_refs(range(8)))
        random.Random(7).shuffle(shuffled)

        assert summarize_geometry(shuffled, source=_source(CUBE), policy=POLICY) == _summarize(CUBE)

    def test_summaries_stay_in_the_frame_of_the_source_map(self) -> None:
        source = _source(CUBE, frame="odom")

        geometry = summarize_geometry(geometry_refs(range(8)), source=source, policy=POLICY)

        assert geometry.map_frame == "odom"
        assert geometry.bounds.frame_id == "odom"
        assert geometry.summary.map_frame == "odom"

    def test_a_negative_offset_map_is_summarized_like_any_other(self) -> None:
        shifted = [(x - 10.0, y - 20.0, z - 30.0) for x, y, z in CUBE]

        geometry = _summarize(shifted)

        assert _close(geometry.centroid_m, (-9.5, -19.5, -29.5))
        assert geometry.extent_m == (1.0, 1.0, 1.0)

    def test_a_single_point_is_a_valid_flat_support(self) -> None:
        geometry = _summarize([(1.0, 2.0, 3.0)])

        assert geometry.centroid_m == (1.0, 2.0, 3.0)
        assert geometry.statistics.density_per_m3 is None
        assert geometry.diagnostic_kinds() == {
            GeometryDiagnosticKind.SPARSE_SUPPORT,
            GeometryDiagnosticKind.DEGENERATE_EXTENT,
        }

    def test_the_centroid_never_falls_outside_the_bounds_because_of_rounding(self) -> None:
        # 0.1 + 0.1 + 0.1 não é 0.3 em ponto flutuante: a média não pode escapar da caixa.
        geometry = _summarize([(0.1, 0.1, 0.1)] * 3 + [(0.1, 0.1, 0.1)])

        assert geometry.bounds.contains(geometry.centroid_m, frame_id=geometry.bounds.frame_id)


class TestSummaryProvenance:
    def test_it_identifies_the_algorithm_input_frame_conventions_and_filtering(self) -> None:
        summary = _summarize(CUBE).summary

        assert summary.algorithm_id == GEOMETRY_SUMMARY_ALGORITHM_ID
        assert summary.input_geometry_count == 8
        assert summary.input_geometry_digest == geometry_set_digest(geometry_refs(range(8)))
        assert summary.map_frame == "map"
        assert "float64" in summary.numerical_conventions
        assert summary.filtering.startswith("none")
        assert summary.configuration_fingerprint == POLICY.fingerprint()

    def test_a_different_input_set_changes_the_digest(self) -> None:
        assert geometry_set_digest(geometry_refs(range(8))) != geometry_set_digest(
            geometry_refs(range(7))
        )

    def test_a_different_policy_changes_the_fingerprint(self) -> None:
        assert POLICY.fingerprint() != ORIENTED.fingerprint()

    def test_recomputing_is_reproducible_and_never_changes_the_entity_identity(self) -> None:
        entity = make_entity()
        recomputed = _summarize(CUBE, ORIENTED)

        replaced = dataclasses.replace(entity, geometry=make_geometry())

        assert _summarize(CUBE) == _summarize(CUBE)
        assert recomputed != _summarize(CUBE)
        assert replaced.entity_id == entity.entity_id
        assert replaced.reference == entity.reference


class TestResolution:
    def test_references_resolve_to_the_persisted_geometry_in_order(self) -> None:
        source = _source(CUBE)

        points = resolve_geometry(geometry_refs((3, 1)), source=source)

        assert [point.coordinates_m for point in points] == [CUBE[3], CUBE[1]]

    def test_a_reference_of_another_map_is_refused(self) -> None:
        foreign = geometry_refs((0,), map_id=MapId("map-0002"))

        with pytest.raises(GeometryResolutionError, match="map-0002"):
            resolve_geometry(foreign, source=_source(CUBE))

    def test_a_missing_reference_is_an_explicit_error(self) -> None:
        with pytest.raises(GeometryResolutionError, match="do not exist"):
            resolve_geometry(geometry_refs((0, 99)), source=_source(CUBE))

    def test_geometry_in_another_frame_than_the_map_is_refused(self) -> None:
        class Misframed(InMemoryGeometrySource):
            def __init__(self) -> None:
                super().__init__(MAP_ID, dict(enumerate(CUBE)))
                self._odom = InMemoryGeometrySource(MAP_ID, dict(enumerate(CUBE)), frame="odom")

            def get(self, reference: GeometryReference) -> GeometryPoint:
                return self._odom.get(reference)

        with pytest.raises(GeometryResolutionError, match="not the map frame"):
            resolve_geometry(geometry_refs(range(8)), source=Misframed())

    def test_an_empty_support_is_explicit_and_cannot_become_an_entity_geometry(self) -> None:
        with pytest.raises(EmptyGeometrySupportError):
            summarize_geometry([], source=_source(CUBE), policy=POLICY)

    def test_a_repeated_reference_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sorted by geometry_id"):
            summarize_geometry(geometry_refs((1, 1)), source=_source(CUBE), policy=POLICY)


class TestSupportStatisticsAndDiagnostics:
    def test_a_flat_support_has_no_density_and_says_so(self) -> None:
        geometry = _summarize(_grid(4, 4, 1))

        assert geometry.statistics.volume_m3 == 0.0
        assert geometry.statistics.density_per_m3 is None
        assert GeometryDiagnosticKind.DEGENERATE_EXTENT in geometry.diagnostic_kinds()

    def test_a_sparse_support_is_flagged(self) -> None:
        geometry = _summarize(CUBE[:3])

        assert GeometryDiagnosticKind.SPARSE_SUPPORT in geometry.diagnostic_kinds()

    def test_a_dense_support_is_not_flagged_as_sparse(self) -> None:
        assert GeometryDiagnosticKind.SPARSE_SUPPORT not in _summarize(CUBE).diagnostic_kinds()

    def test_two_far_apart_clusters_are_a_disconnected_support(self) -> None:
        far = [(x + 10.0, y, z) for x, y, z in CUBE]

        geometry = _summarize(CUBE + far)

        assert geometry.statistics.component_count == 2
        assert geometry.statistics.largest_component_fraction == 0.5
        assert GeometryDiagnosticKind.DISCONNECTED_SUPPORT in geometry.diagnostic_kinds()

    def test_a_chain_of_close_points_is_one_component(self) -> None:
        chain = [(float(index), 0.0, 0.0) for index in range(30)]

        geometry = _summarize(chain)

        assert geometry.statistics.component_count == 1
        assert GeometryDiagnosticKind.DISCONNECTED_SUPPORT not in geometry.diagnostic_kinds()

    def test_points_exactly_at_the_radius_are_linked(self) -> None:
        policy = GeometrySummaryPolicy(sparse_point_threshold=1, connectivity_radius_m=1.0)

        geometry = _summarize([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], policy)

        assert geometry.statistics.component_count == 1

    def test_points_just_beyond_the_radius_are_not_linked(self) -> None:
        policy = GeometrySummaryPolicy(sparse_point_threshold=1, connectivity_radius_m=1.0)

        geometry = _summarize([(0.0, 0.0, 0.0), (1.0001, 0.0, 0.0)], policy)

        assert geometry.statistics.component_count == 2

    def test_connectivity_crosses_negative_cell_boundaries(self) -> None:
        policy = GeometrySummaryPolicy(sparse_point_threshold=1, connectivity_radius_m=1.0)

        geometry = _summarize([(-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)], policy)

        assert geometry.statistics.component_count == 1

    def test_a_large_support_is_summarized(self) -> None:
        geometry = _summarize(_grid(20, 20, 10))

        assert geometry.statistics.point_count == 4000
        assert geometry.statistics.component_count == 1


class TestOrientation:
    def test_it_is_absent_unless_the_policy_asks_for_it(self) -> None:
        geometry = _summarize(_grid(8, 4, 2))

        assert geometry.orientation is None
        assert GeometryDiagnosticKind.ORIENTATION_NOT_JUSTIFIED not in geometry.diagnostic_kinds()

    def test_an_elongated_box_gets_axis_aligned_principal_axes(self) -> None:
        geometry = _summarize(_grid(8, 4, 2), ORIENTED)

        orientation = geometry.orientation
        assert orientation is not None
        assert all(
            _close(axis, expected)
            for axis, expected in zip(
                orientation.axes, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), strict=True
            )
        )
        assert (
            orientation.variances_m2[0] > orientation.variances_m2[1] > orientation.variances_m2[2]
        )
        assert orientation.method == "pca-covariance-jacobi-v1"

    def test_a_rotated_box_has_its_principal_axis_along_the_rotation(self) -> None:
        angle = math.radians(30)
        rotated = [
            (
                x * math.cos(angle) - y * math.sin(angle),
                x * math.sin(angle) + y * math.cos(angle),
                z,
            )
            for x, y, z in _grid(8, 4, 2)
        ]

        orientation = _summarize(rotated, ORIENTED).orientation

        assert orientation is not None
        assert _close(orientation.axes[0], (math.cos(angle), math.sin(angle), 0.0))

    def test_the_sign_of_an_axis_is_canonical(self) -> None:
        flipped = [(-x, -y, -z) for x, y, z in _grid(8, 4, 2)]

        orientation = _summarize(flipped, ORIENTED).orientation

        assert orientation is not None
        assert all(max(axis, key=abs) > 0.0 for axis in orientation.axes[:2])

    def test_an_isotropic_support_has_no_justified_orientation_and_says_why(self) -> None:
        geometry = _summarize(CUBE, ORIENTED)

        assert geometry.orientation is None
        assert GeometryDiagnosticKind.ORIENTATION_NOT_JUSTIFIED in geometry.diagnostic_kinds()

    def test_too_few_points_have_no_justified_orientation(self) -> None:
        geometry = _summarize(_grid(2, 2, 1), ORIENTED)

        detail = next(
            item.detail
            for item in geometry.diagnostics
            if item.kind is GeometryDiagnosticKind.ORIENTATION_NOT_JUSTIFIED
        )
        assert geometry.orientation is None
        assert "fewer than" in detail

    def test_a_line_has_no_justified_orientation(self) -> None:
        geometry = _summarize([(float(index), 0.0, 0.0) for index in range(12)], ORIENTED)

        assert geometry.orientation is None
        assert GeometryDiagnosticKind.ORIENTATION_NOT_JUSTIFIED in geometry.diagnostic_kinds()

    def test_a_plane_has_axes_in_the_plane_and_its_normal(self) -> None:
        geometry = _summarize(_grid(8, 4, 1), ORIENTED)

        orientation = geometry.orientation
        assert orientation is not None
        assert _close(orientation.axes[2], (0.0, 0.0, 1.0))
        assert orientation.variances_m2[2] == 0.0

    def test_the_contract_refuses_axes_that_are_not_a_right_handed_orthonormal_frame(self) -> None:
        good = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        with pytest.raises(ValueError, match="unit vectors"):
            EntityOrientation(
                axes=((2.0, 0.0, 0.0), good[1], good[2]), variances_m2=(3.0, 2.0, 1.0), method="m"
            )
        with pytest.raises(ValueError, match="orthogonal"):
            EntityOrientation(
                axes=(good[0], (0.6, 0.8, 0.0), good[2]), variances_m2=(3.0, 2.0, 1.0), method="m"
            )
        with pytest.raises(ValueError, match="right-handed"):
            EntityOrientation(
                axes=(good[0], good[1], (0.0, 0.0, -1.0)), variances_m2=(3.0, 2.0, 1.0), method="m"
            )
        with pytest.raises(ValueError, match="descending"):
            EntityOrientation(axes=good, variances_m2=(1.0, 2.0, 3.0), method="m")


class TestVerification:
    def test_a_fresh_summary_matches_the_persistent_geometry(self) -> None:
        geometry = _summarize(CUBE, ORIENTED)

        assert verify_geometry_summary(geometry, source=_source(CUBE), policy=ORIENTED) == []

    def test_a_tampered_centroid_is_detected(self) -> None:
        geometry = dataclasses.replace(_summarize(CUBE), centroid_m=(0.4, 0.5, 0.5))

        problems = verify_geometry_summary(geometry, source=_source(CUBE), policy=POLICY)

        assert any("centroid_m" in problem for problem in problems)

    def test_geometry_that_moved_upstream_is_detected(self) -> None:
        geometry = _summarize(CUBE)
        moved = [(x + 5.0, y, z) for x, y, z in CUBE]

        problems = verify_geometry_summary(geometry, source=_source(moved), policy=POLICY)

        assert any("bounds" in problem for problem in problems)

    def test_an_unresolvable_support_is_reported_not_raised(self) -> None:
        geometry = _summarize(CUBE)

        problems = verify_geometry_summary(geometry, source=_source(CUBE[:4]), policy=POLICY)

        assert problems and "cannot be resolved" in problems[0]

    def test_a_summary_computed_under_another_policy_is_detected(self) -> None:
        geometry = _summarize(CUBE, ORIENTED)

        problems = verify_geometry_summary(geometry, source=_source(CUBE), policy=POLICY)

        assert any("different policy" in problem for problem in problems)


class TestContractInvariants:
    def _replace(self, **changes: Any) -> EntityGeometry:
        return dataclasses.replace(_summarize(CUBE), **changes)

    def test_bounds_must_be_in_the_map_frame(self) -> None:
        odom = Bounds3D(frame_id=FrameId("odom"), minimum_m=(0.0,) * 3, maximum_m=(1.0,) * 3)

        with pytest.raises(ValueError, match="not the map frame"):
            self._replace(bounds=odom)

    def test_the_summary_must_be_derived_in_the_map_frame(self) -> None:
        summary = dataclasses.replace(_summarize(CUBE).summary, map_frame=FrameId("odom"))

        with pytest.raises(ValueError, match="not the map frame"):
            self._replace(summary=summary)

    def test_the_centroid_must_lie_inside_the_bounds(self) -> None:
        with pytest.raises(ValueError, match="inside the bounds"):
            self._replace(centroid_m=(2.0, 0.5, 0.5))

    def test_the_centroid_must_be_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            self._replace(centroid_m=(math.nan, 0.5, 0.5))

    def test_the_extent_must_be_the_size_of_the_bounds(self) -> None:
        with pytest.raises(ValueError, match="size of the bounds"):
            self._replace(extent_m=(2.0, 1.0, 1.0))

    def test_the_summary_must_be_derived_from_exactly_these_references(self) -> None:
        summary = dataclasses.replace(
            _summarize(CUBE).summary,
            input_geometry_digest=geometry_set_digest(geometry_refs(range(7))),
        )

        with pytest.raises(ValueError, match="exactly these geometry references"):
            self._replace(summary=summary)

    def test_the_statistics_must_count_the_references(self) -> None:
        statistics = dataclasses.replace(
            _summarize(CUBE).statistics, point_count=9, density_per_m3=9.0
        )

        with pytest.raises(ValueError, match="must describe the 8 geometry references"):
            self._replace(statistics=statistics)

    def test_a_missing_disconnected_diagnostic_is_refused(self) -> None:
        far = [(x + 10.0, y, z) for x, y, z in CUBE]
        disconnected = _summarize(CUBE + far)

        with pytest.raises(ValueError, match="disconnected_support"):
            dataclasses.replace(disconnected, diagnostics=())

    def test_a_degenerate_diagnostic_on_a_solid_support_is_refused(self) -> None:
        item = GeometryDiagnostic(kind=GeometryDiagnosticKind.DEGENERATE_EXTENT, detail="flat")

        with pytest.raises(ValueError, match="degenerate_extent"):
            self._replace(diagnostics=(item,))

    def test_diagnostics_are_canonical(self) -> None:
        flat = _summarize(_grid(2, 1, 1))
        in_order = sorted(flat.diagnostics, key=lambda item: item.kind.value)

        with pytest.raises(ValueError, match="sorted by kind"):
            dataclasses.replace(flat, diagnostics=tuple(reversed(in_order)))

    def test_an_orientation_cannot_coexist_with_the_not_justified_diagnostic(self) -> None:
        oriented = _summarize(_grid(8, 4, 2), ORIENTED)
        item = GeometryDiagnostic(
            kind=GeometryDiagnosticKind.ORIENTATION_NOT_JUSTIFIED, detail="contradiction"
        )

        with pytest.raises(ValueError, match="cannot coexist"):
            dataclasses.replace(oriented, diagnostics=(item,))

    def test_statistics_density_must_match_the_volume(self) -> None:
        with pytest.raises(ValueError, match="density_per_m3 must equal"):
            SupportStatistics(
                point_count=8,
                volume_m3=1.0,
                density_per_m3=3.0,
                component_count=1,
                largest_component_fraction=1.0,
            )
        with pytest.raises(ValueError, match="None when the volume is zero"):
            SupportStatistics(
                point_count=8,
                volume_m3=0.0,
                density_per_m3=1.0,
                component_count=1,
                largest_component_fraction=1.0,
            )

    def test_policies_reject_nonsensical_thresholds(self) -> None:
        with pytest.raises(ValueError, match="sparse_point_threshold"):
            GeometrySummaryPolicy(sparse_point_threshold=0, connectivity_radius_m=1.0)
        with pytest.raises(ValueError, match="connectivity_radius_m"):
            GeometrySummaryPolicy(sparse_point_threshold=1, connectivity_radius_m=0.0)
        with pytest.raises(ValueError, match="min_points"):
            OrientationPolicy(min_points=2, min_variance_ratio=2.0)
        with pytest.raises(ValueError, match="min_variance_ratio"):
            OrientationPolicy(min_points=8, min_variance_ratio=1.0)


class TestSerialization:
    def test_the_geometry_round_trips_with_orientation_and_diagnostics(self) -> None:
        geometry = _summarize(_grid(8, 4, 1), ORIENTED)
        entity = make_entity(geometry=geometry)

        record = json.loads(json.dumps(encode_entity(entity), allow_nan=False))

        assert decode_entity(record).geometry == geometry
        assert record["semantic_map_id"] == SEMANTIC_MAP_ID

    def test_an_absent_orientation_and_density_survive_as_absent(self) -> None:
        geometry = _summarize(_grid(4, 4, 1))

        decoded = decode_entity(encode_entity(make_entity(geometry=geometry))).geometry

        assert decoded.orientation is None
        assert decoded.statistics.density_per_m3 is None

    def test_no_coordinates_of_the_support_are_stored(self) -> None:
        record = encode_entity(make_entity())["geometry"]

        assert set(record) == {
            "map_frame",
            "map_id",
            "deltas",
            "centroid_m",
            "bounds",
            "extent_m",
            "statistics",
            "summary",
            "orientation",
            "diagnostics",
        }

    def test_a_record_whose_centroid_left_the_bounds_is_refused(self) -> None:
        record = encode_entity(make_entity())
        record["geometry"]["centroid_m"] = [99.0, 0.0, 0.0]

        with pytest.raises(ValueError, match="inside the bounds"):
            decode_entity(record)

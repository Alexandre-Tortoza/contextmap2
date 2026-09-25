"""What candidate culling may and may not change, against the full-map baseline.

The claim the candidate step rests on is that for every element it *keeps*, nothing
downstream changes: the same pixel, the same support, the same occlusion decision, the
same membership and the same persistent reference as a full-map projection. The only
declared difference is which elements are evaluated at all. These tests compare the two
arms directly on frozen fixtures, by persistent geometry identity, never by row.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from projection_builders import (
    UNBOUNDED_CANDIDATES,
    make_calibration,
    map_point_for_camera_point,
    map_point_for_pixel,
    project_frame,
)

from contextmap.geometric_mapping import GeometryReference
from contextmap.ingestion import (
    CalibrationSet,
    CameraModel,
    FisheyeCameraModel,
    MeiCameraModel,
)
from contextmap.sensor_association import (
    CandidateGeometryPolicy,
    OcclusionPolicy,
    VisibilityState,
)
from contextmap.sensor_association.frame_projection import FrameProjection
from contextmap.sensor_association.visibility import resolve_visibility

OCCLUSION = OcclusionPolicy(
    cell_size_px=4, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02
)

# Uma cena com geometria perto (2 a 6 m) e longe (40 a 60 m) da câmera na origem.
NEAR = [map_point_for_pixel(u, v, z) for u, v, z in ((100.0, 100.0, 2.0), (300.0, 200.0, 6.0))]
FAR = [map_point_for_pixel(u, v, z) for u, v, z in ((150.0, 150.0, 40.0), (250.0, 250.0, 60.0))]
SCENE = [*NEAR, *FAR]


def _state_by_geometry(frame: FrameProjection) -> dict[GeometryReference, tuple[object, ...]]:
    """Every evaluated element's outcome, keyed by persistent identity, never by row."""
    resolution = resolve_visibility(frame, OCCLUSION)
    outcome: dict[GeometryReference, tuple[object, ...]] = {}
    for row in range(frame.candidate_count):
        correspondence = resolution.correspondence(
            row, visible_state=VisibilityState.VISIBLE_UNASSIGNED
        )
        outcome[frame.map_reference(row)] = (
            correspondence.visibility,
            correspondence.raw_pixel,
            correspondence.prepared_pixel,
            correspondence.camera_depth_m,
            correspondence.support_depth_m,
        )
    return outcome


def _fisheye() -> CameraModel:
    return FisheyeCameraModel(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0),
    )


def _mei() -> CameraModel:
    return MeiCameraModel(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        xi=1.0,
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0),
    )


# --- Equivalence when nothing is culled -----------------------------------------------------


@pytest.mark.parametrize(
    "calibration",
    [
        pytest.param(make_calibration(), id="pinhole"),
        pytest.param(make_calibration(model=_fisheye()), id="fisheye"),
        pytest.param(make_calibration(model=_mei()), id="mei"),
    ],
)
def test_a_range_that_reaches_every_element_reproduces_the_full_map_projection(
    calibration: CalibrationSet,
) -> None:
    baseline = project_frame(SCENE, calibration=calibration)
    culled = project_frame(
        SCENE, calibration=calibration, candidate_policy=CandidateGeometryPolicy(max_range_m=100.0)
    )

    assert culled.candidate_count == baseline.candidate_count == len(SCENE)
    assert _state_by_geometry(culled) == _state_by_geometry(baseline)


def test_culling_never_renumbers_the_geometry_a_row_refers_to() -> None:
    baseline = project_frame(SCENE)
    culled = project_frame(SCENE, candidate_policy=CandidateGeometryPolicy(max_range_m=10.0))

    # O mesmo elemento ocupa linhas diferentes nos dois braços, e a identidade não muda.
    kept = [int(index) for index in culled.global_indices]
    assert kept == [0, 1]
    assert culled.map_reference(0) == baseline.map_reference(0)
    assert culled.map_reference(1) == baseline.map_reference(1)
    assert culled.map_point_count == baseline.map_point_count == len(SCENE)


# --- What culling does change, and only that ------------------------------------------------


def test_only_the_elements_beyond_the_range_stop_being_evaluated() -> None:
    full = project_frame(SCENE)
    baseline = _state_by_geometry(full)
    culled = _state_by_geometry(
        project_frame(SCENE, candidate_policy=CandidateGeometryPolicy(max_range_m=10.0))
    )

    # SCENE são dois pontos perto (índices 0 e 1) e dois longe (2 e 3).
    assert set(culled) == {full.map_reference(0), full.map_reference(1)}
    assert set(baseline) - set(culled) == {full.map_reference(2), full.map_reference(3)}
    # Para todo elemento retido, o resultado é exatamente o do mapa inteiro.
    assert all(culled[reference] == baseline[reference] for reference in culled)


def test_the_candidate_counts_say_what_population_the_frame_evaluated() -> None:
    baseline = project_frame(SCENE)
    culled = project_frame(SCENE, candidate_policy=CandidateGeometryPolicy(max_range_m=10.0))

    report = culled.candidates
    assert (report.map_point_count, report.candidate_count) == (len(SCENE), 2)
    assert report.candidate_fraction == pytest.approx(0.5)
    # As contagens de estágio descrevem a população avaliada, não o mapa: é por elas que a
    # diferença entre os dois braços é auditável.
    assert sum(culled.stage_counts().values()) == 2
    assert sum(baseline.stage_counts().values()) == len(SCENE)


# --- The occlusion invariant the sphere buys -------------------------------------------------


def test_an_occluder_and_what_it_hides_are_kept_or_dropped_together() -> None:
    # Dois pontos no mesmo raio de visão: o oclusor a 3 m, o ocluído a 9 m.
    occluder = map_point_for_camera_point((0.0, 0.0, 3.0))
    hidden = map_point_for_camera_point((0.0, 0.0, 9.0))
    scene = [occluder, hidden]

    baseline = _state_by_geometry(project_frame(scene))
    wide = _state_by_geometry(
        project_frame(scene, candidate_policy=CandidateGeometryPolicy(max_range_m=20.0))
    )

    assert wide == baseline
    states = [outcome[0] for outcome in baseline.values()]
    assert states == [VisibilityState.VISIBLE_UNASSIGNED, VisibilityState.OCCLUDED]


def test_a_range_that_cuts_between_them_drops_the_hidden_one_and_never_frees_it() -> None:
    occluder = map_point_for_camera_point((0.0, 0.0, 3.0))
    hidden = map_point_for_camera_point((0.0, 0.0, 9.0))

    culled = project_frame(
        [occluder, hidden], candidate_policy=CandidateGeometryPolicy(max_range_m=5.0)
    )
    outcome = _state_by_geometry(culled)

    # O ocluído sai da população avaliada; ele nunca reaparece como visível.
    assert [int(i) for i in culled.global_indices] == [0]
    assert [state[0] for state in outcome.values()] == [VisibilityState.VISIBLE_UNASSIGNED]


def test_every_dropped_element_is_farther_than_every_kept_one() -> None:
    """The sphere's defining property, the one that makes occlusion decisions safe."""
    scene = [
        map_point_for_camera_point(point)
        for point in (
            (0.0, 0.0, 2.0),
            (1.0, 1.0, 2.0),
            (3.0, 0.0, 4.0),
            (0.0, 4.0, 8.0),
            (6.0, 6.0, 6.0),
        )
    ]
    radius = 6.0

    culled = project_frame(scene, candidate_policy=CandidateGeometryPolicy(max_range_m=radius))

    kept = set(int(index) for index in culled.global_indices)
    ranges = [math.dist((0.0, 0.0, 0.0), point) for point in scene]
    assert kept and len(kept) < len(scene)
    assert max(ranges[i] for i in kept) <= radius
    assert min(ranges[i] for i in range(len(scene)) if i not in kept) > radius


# --- Boundaries are never culled by accident -------------------------------------------------


@pytest.mark.parametrize(
    ("u", "v"),
    [(0.0, 0.0), (639.0, 0.0), (0.0, 479.0), (639.0, 479.0), (320.0, 0.0), (0.0, 240.0)],
)
def test_an_element_at_an_image_corner_or_edge_within_range_is_still_evaluated(
    u: float, v: float
) -> None:
    point = map_point_for_pixel(u, v, 3.0)

    culled = project_frame([point], candidate_policy=CandidateGeometryPolicy(max_range_m=20.0))

    assert culled.candidate_count == 1
    assert bool(culled.in_prepared_image[0])
    np.testing.assert_allclose(culled.raw_pixels[0], (u, v), atol=1e-9)


def test_an_element_exactly_at_the_range_boundary_is_evaluated() -> None:
    point = map_point_for_camera_point((0.0, 0.0, 5.0))

    culled = project_frame([point], candidate_policy=CandidateGeometryPolicy(max_range_m=5.0))

    assert culled.candidate_count == 1


# --- The unbounded policy is the baseline ----------------------------------------------------


def test_the_unbounded_policy_evaluates_the_whole_map() -> None:
    frame = project_frame(SCENE, candidate_policy=UNBOUNDED_CANDIDATES)

    assert frame.candidate_count == frame.map_point_count == len(SCENE)
    assert [int(index) for index in frame.global_indices] == list(range(len(SCENE)))

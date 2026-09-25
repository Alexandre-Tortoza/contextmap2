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
    CAMERA_CALIBRATION_ID,
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
    camera_projection_for,
)
from contextmap.sensor_association.frame_projection import FrameProjection
from contextmap.sensor_association.visibility import resolve_visibility
from contextmap.shared import Vector3

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


def _map_point_at_pixel(calibration: CalibrationSet, u: float, v: float, range_m: float) -> Vector3:
    """The map-frame point whose ray the camera's **own** model sends to pixel ``(u, v)``.

    Inverting the pinhole by hand would only ever test pinhole. Going through
    :meth:`CameraProjection.unproject` makes the same boundary case real for fisheye and MEI,
    whose in-image direction sets are shaped quite differently.
    """
    camera = camera_projection_for(calibration.entries[CAMERA_CALIBRATION_ID])
    ray = camera.unproject(np.array([[u, v]]))[0]
    return map_point_for_camera_point(
        (float(ray[0]) * range_m, float(ray[1]) * range_m, float(ray[2]) * range_m)
    )


@pytest.mark.parametrize(
    "calibration",
    [
        pytest.param(make_calibration(), id="pinhole"),
        pytest.param(make_calibration(model=_fisheye()), id="fisheye"),
        pytest.param(make_calibration(model=_mei()), id="mei"),
    ],
)
@pytest.mark.parametrize(
    ("u", "v"),
    [(0.0, 0.0), (639.0, 0.0), (0.0, 479.0), (639.0, 479.0), (320.0, 0.0), (0.0, 240.0)],
)
def test_an_element_at_an_image_corner_or_edge_within_range_is_still_evaluated(
    calibration: CalibrationSet, u: float, v: float
) -> None:
    """A point at the very edge of the field of view is never culled by accident.

    Conservative over-selection is acceptable; false exclusion is not, and the corners are where
    a bound derived from the optical axis would fail first.
    """
    point = _map_point_at_pixel(calibration, u, v, 3.0)

    culled = project_frame(
        [point], calibration=calibration, candidate_policy=CandidateGeometryPolicy(max_range_m=20.0)
    )

    assert culled.candidate_count == 1
    assert bool(culled.projectable[0])
    assert bool(culled.in_prepared_image[0])
    np.testing.assert_allclose(culled.raw_pixels[0], (u, v), atol=1e-6)


@pytest.mark.parametrize(
    "calibration",
    [
        pytest.param(make_calibration(), id="pinhole"),
        pytest.param(make_calibration(model=_fisheye()), id="fisheye"),
        pytest.param(make_calibration(model=_mei()), id="mei"),
    ],
)
def test_a_boundary_element_survives_culling_exactly_as_the_full_map_would_place_it(
    calibration: CalibrationSet,
) -> None:
    """And it lands on the same pixel in both arms, for every camera model."""
    corners = [
        _map_point_at_pixel(calibration, u, v, 4.0)
        for u, v in ((0.0, 0.0), (639.0, 479.0), (639.0, 0.0), (0.0, 479.0))
    ]

    baseline = project_frame(corners, calibration=calibration)
    culled = project_frame(
        corners, calibration=calibration, candidate_policy=CandidateGeometryPolicy(max_range_m=20.0)
    )

    assert culled.candidate_count == baseline.candidate_count == len(corners)
    np.testing.assert_array_equal(culled.raw_pixels, baseline.raw_pixels)
    np.testing.assert_array_equal(culled.in_prepared_image, baseline.in_prepared_image)
    assert _state_by_geometry(culled) == _state_by_geometry(baseline)


def test_an_element_exactly_at_the_range_boundary_is_evaluated() -> None:
    point = map_point_for_camera_point((0.0, 0.0, 5.0))

    culled = project_frame([point], candidate_policy=CandidateGeometryPolicy(max_range_m=5.0))

    assert culled.candidate_count == 1


# --- The unbounded policy is the baseline ----------------------------------------------------


def test_the_unbounded_policy_evaluates_the_whole_map() -> None:
    frame = project_frame(SCENE, candidate_policy=UNBOUNDED_CANDIDATES)

    assert frame.candidate_count == frame.map_point_count == len(SCENE)
    assert [int(index) for index in frame.global_indices] == list(range(len(SCENE)))


# --- Where the sphere's guarantee stops, and why -------------------------------------------

# Um par astride do limite de alcance, construído para inverter a ordem de z contra a ordem de
# alcance: o excluído está mais LONGE (alcance 20,2 > 20) e ao mesmo tempo mais PERTO em z
# (17,22 < 19,11), porque está num ângulo de campo maior. Só `OPTICAL_AXIS` compara z.
_ADVERSARIAL_RANGE_M = 20.0
_RETAINED = (
    _ADVERSARIAL_RANGE_M * math.sin(0.30),
    0.0,
    _ADVERSARIAL_RANGE_M * math.cos(0.30),
)
_EXCLUDED = (20.2 * math.sin(0.55), 0.0, 20.2 * math.cos(0.55))
# Grade grossa: a `OcclusionPolicy` não tem default, e uma janela larga é uma configuração
# válida. Com a grade do run real (cell 4, raio 2) o par não cai na mesma janela.
_COARSE = OcclusionPolicy(
    cell_size_px=64, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02
)


def _visibility_of_retained(calibration: CalibrationSet, *, max_range_m: float | None) -> str:
    scene = [map_point_for_camera_point(_RETAINED), map_point_for_camera_point(_EXCLUDED)]
    frame = project_frame(
        scene,
        calibration=calibration,
        candidate_policy=CandidateGeometryPolicy(max_range_m=max_range_m),
    )
    resolution = resolve_visibility(frame, _COARSE)
    rows, found = frame.rows_for(np.array([0]))
    assert bool(found[0]), "o ponto retido deve estar sempre na população avaliada"
    correspondence = resolution.correspondence(
        int(rows[0]), visible_state=VisibilityState.VISIBLE_UNASSIGNED
    )
    return correspondence.visibility.value


def test_the_adversarial_pair_really_inverts_depth_against_range() -> None:
    """The fixture is only meaningful if range and ``z`` disagree; assert that it does."""
    assert math.dist((0.0, 0.0, 0.0), _EXCLUDED) > _ADVERSARIAL_RANGE_M
    assert math.dist((0.0, 0.0, 0.0), _RETAINED) <= _ADVERSARIAL_RANGE_M
    assert _EXCLUDED[2] < _RETAINED[2]


def test_a_ray_range_camera_preserves_the_support_of_every_retained_element() -> None:
    """For fisheye and MEI the sphere is exact: depth *is* range, so the proof holds.

    Every excluded element is strictly farther than every retained one under the very metric
    the occlusion rule compares, and a point's own window always contains itself, so an
    excluded element can never have been the nearest support of a retained one.
    """
    for calibration in (make_calibration(model=_fisheye()), make_calibration(model=_mei())):
        baseline = _visibility_of_retained(calibration, max_range_m=None)
        culled = _visibility_of_retained(calibration, max_range_m=_ADVERSARIAL_RANGE_M)

        assert culled == baseline


def test_a_pinhole_camera_can_lose_the_support_that_came_from_a_neighbouring_cell() -> None:
    """For pinhole the sphere is **not** exact, and this pins the real behaviour.

    ``OPTICAL_AXIS`` compares ``z``, and ``z <= range``, so an element the range policy excludes
    may still have had a smaller ``z`` than a retained one. Here the excluded element was the
    retained one's nearest support, so culling makes the retained element *visible* where the
    full map called it occluded.

    The direction matters: removing elements can only raise a window's minimum, so this
    mechanism can only make a retained element **less** occluded. Among the retained candidates
    it never drops geometry the baseline associated -- what #562 had to guarantee -- but it does
    mean the range policy redefines the support population, and not only the evaluated one, for a
    camera whose depth metric is the optical axis.
    """
    pinhole = make_calibration()

    baseline = _visibility_of_retained(pinhole, max_range_m=None)
    culled = _visibility_of_retained(pinhole, max_range_m=_ADVERSARIAL_RANGE_M)

    assert baseline == VisibilityState.OCCLUDED.value
    assert culled == VisibilityState.VISIBLE_UNASSIGNED.value


def test_the_occlusion_grid_of_the_real_run_is_too_fine_for_that_pair_to_interact() -> None:
    """With the policy the real corridor-02 run used, the two cells are not neighbours.

    Measured probe: the smallest pixel separation that made this mechanism fire was ~24 px of
    window reach. The run's ``cell_size_px=4, neighborhood_radius_cells=2`` reaches 12 px. This
    is a probe, not a proven bound: it says the real configuration is not close to the regime,
    not that no configuration below 24 px can ever diverge.
    """
    fine = OcclusionPolicy(
        cell_size_px=4, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02
    )
    scene = [map_point_for_camera_point(_RETAINED), map_point_for_camera_point(_EXCLUDED)]
    frame = project_frame(scene)

    separation = abs(float(frame.prepared_pixels[1, 0] - frame.prepared_pixels[0, 0]))
    reach_px = (fine.neighborhood_radius_cells + 1) * fine.cell_size_px

    assert separation > 150.0
    assert reach_px == 12
    assert separation > reach_px
    # E com essa grade os dois estados coincidem, como no run real.
    resolution = resolve_visibility(frame, fine)
    assert (
        resolution.correspondence(0, visible_state=VisibilityState.VISIBLE_UNASSIGNED).visibility
        is VisibilityState.VISIBLE_UNASSIGNED
    )

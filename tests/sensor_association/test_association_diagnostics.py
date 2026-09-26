import dataclasses
import json
import math
from typing import Any

import numpy as np
import pytest
from dense_builders import make_dense_map, make_dense_result, make_enhancement, make_sampling
from numpy.typing import NDArray
from perception_builders import make_region, make_result, rect_mask
from projection_builders import (
    BODY_TO_CAMERA_ROTATION,
    IDENTITY,
    make_calibration,
    make_camera_observation,
    make_prepared_image,
    make_projector,
    map_point_for_camera_point,
    map_point_for_pixel,
    project_frame,
    scene_frame,
)

from contextmap.sensor_association import (
    CandidateGeometryPolicy,
    DepthMetric,
    ReprojectionStatistics,
    VisibilityState,
)
from contextmap.sensor_association.dense_sampling import (
    InterpolationPolicy,
    sample_dense_features,
)
from contextmap.sensor_association.diagnostics import (
    DIAGNOSTICS_DEFINITIONS_VERSION,
    DiagnosticTolerances,
    FindingCode,
    FindingSeverity,
    ReprojectionAttempt,
    ReprojectionOutcome,
    TrustedCorrespondences,
    _reprojection_findings,
    diagnose_frame,
    evaluated_correspondences,
    reprojection_statistics,
    time_offset_sweep,
)
from contextmap.sensor_association.frame_projection import FrameProjection
from contextmap.sensor_association.membership import associate_regions
from contextmap.sensor_association.visibility import OcclusionPolicy, resolve_visibility
from contextmap.shared import quaternion_multiply
from contextmap.state_estimation import LookupPolicy, LookupRejection

POLICY = OcclusionPolicy(
    cell_size_px=4,
    neighborhood_radius_cells=2,
    depth_margin_m=0.1,
    depth_margin_ratio=0.02,
)
NO_TOLERANCE = DiagnosticTolerances(
    max_pose_time_delta_ns=None,
    max_map_window_offset_ns=None,
    max_reprojection_p95_px=None,
    max_reprojection_invalid_rate=None,
)
TOLERANCES = DiagnosticTolerances(
    max_pose_time_delta_ns=20_000_000,
    max_map_window_offset_ns=10_000_000,
    max_reprojection_p95_px=5.0,
    max_reprojection_invalid_rate=0.25,
)
PIXELS = [(100, 100, 3.0), (200, 150, 3.0), (300, 200, 3.0), (400, 250, 3.0)]


def _correspondences(
    frame_pixels: np.ndarray, shifts: list[float], *, reference_id: str = "trusted-0001"
) -> TrustedCorrespondences:
    observed = np.array(frame_pixels, dtype=float)
    observed[:, 0] += shifts
    return TrustedCorrespondences(
        reference_id=reference_id,
        geometry_indices=np.arange(len(shifts)),
        observed_pixels=observed,
    )


def _diagnose(frame, tolerances=NO_TOLERANCE, **options):  # type: ignore[no-untyped-def]
    return diagnose_frame(resolve_visibility(frame, POLICY), tolerances=tolerances, **options)


# --- What is always recorded ------------------------------------------------


def test_every_diagnostic_names_the_sources_the_frame_depended_on() -> None:
    frame = scene_frame((100, 100, 3.0), (300, 200, 4.0))

    diagnostics = _diagnose(frame)

    assert diagnostics.definitions_version == DIAGNOSTICS_DEFINITIONS_VERSION
    assert diagnostics.source_observation_id == frame.source_observation_id
    assert diagnostics.image_timestamp == make_camera_observation(0).timestamp
    assert diagnostics.map_id == frame.map_id
    assert diagnostics.calibration_ref == frame.calibration_ref
    assert diagnostics.camera == frame.camera
    assert diagnostics.pose_ref == frame.pose_ref
    assert diagnostics.extrinsic == frame.extrinsic
    assert diagnostics.image_transform_id == frame.image_transform.transform_id
    assert diagnostics.prepared_image_size == (640, 480)
    assert diagnostics.visibility_policy_id == POLICY.policy_id
    assert diagnostics.visibility_policy_fingerprint == POLICY.fingerprint()
    assert diagnostics.depth_metric is DepthMetric.OPTICAL_AXIS
    assert diagnostics.map_time_bounds == frame.map_time_bounds


def test_the_geometry_timing_and_range_are_reported() -> None:
    frame = scene_frame((100, 100, 3.0), (300, 200, 4.0))

    diagnostics = _diagnose(frame)

    assert diagnostics.point_count == 2
    assert diagnostics.map_window_offset_ns == 0
    depth = diagnostics.visible_depth_m
    assert depth is not None
    assert (depth.count, depth.minimum, depth.maximum) == pytest.approx((2, 3.0, 4.0))


def test_the_state_counts_and_the_membership_summary_partition_the_points() -> None:
    frame = project_frame(
        [
            map_point_for_pixel(100, 100, 3.0),
            map_point_for_pixel(100, 100, 6.0),
            (-3.0, 0.0, 0.0),
            (0.5, -10.0, 0.0),
        ]
    )
    resolution = resolve_visibility(frame, POLICY)
    membership = associate_regions(
        resolution, make_result([make_region("region-A", rect_mask(640, 480, 90, 90, 110, 110))])
    )

    diagnostics = diagnose_frame(resolution, tolerances=NO_TOLERANCE, membership=membership)

    assert diagnostics.state_counts == {
        VisibilityState.BEHIND_CAMERA: 1,
        VisibilityState.OUTSIDE_IMAGE: 1,
        VisibilityState.OUTSIDE_VALID_SUPPORT: 0,
        VisibilityState.OCCLUDED: 1,
    }
    assert diagnostics.visible_count == 1
    assert diagnostics.membership is not None
    assert diagnostics.membership.associated_count == 1


# --- Reprojection, only against trusted references --------------------------


def test_the_reprojection_statistics_are_measured_against_the_trusted_reference() -> None:
    frame = scene_frame(*PIXELS)
    correspondences = _correspondences(frame.raw_pixels[:4], [0.0, 1.0, 2.0, 3.0])

    statistics = reprojection_statistics(frame, correspondences)

    assert statistics is not None
    assert statistics.reference_id == "trusted-0001"
    assert (statistics.correspondence_count, statistics.invalid_count) == (4, 0)
    assert statistics.mean_px == pytest.approx(1.5)
    assert statistics.median_px == pytest.approx(1.5)
    assert statistics.p95_px == pytest.approx(2.85)
    assert statistics.max_px == pytest.approx(3.0)


def test_the_residual_is_the_distance_in_both_image_axes() -> None:
    frame = scene_frame(*PIXELS)
    observed = frame.raw_pixels[:4] + np.array([3.0, 4.0])
    correspondences = TrustedCorrespondences(
        reference_id="trusted-0001", geometry_indices=np.arange(4), observed_pixels=observed
    )

    statistics = reprojection_statistics(frame, correspondences)

    assert statistics is not None
    assert (statistics.mean_px, statistics.max_px) == pytest.approx((5.0, 5.0))


def test_a_correspondence_the_camera_cannot_project_is_counted_as_invalid() -> None:
    frame = project_frame([*[map_point_for_pixel(u, v, z) for u, v, z in PIXELS], (-3.0, 0.0, 0.0)])
    observed = np.vstack([frame.raw_pixels[:4], [[10.0, 10.0]]])
    correspondences = TrustedCorrespondences(
        reference_id="trusted-0001",
        geometry_indices=np.arange(5),
        observed_pixels=observed,
    )

    statistics = reprojection_statistics(frame, correspondences)

    assert statistics is not None
    assert (statistics.correspondence_count, statistics.invalid_count) == (5, 1)
    assert statistics.max_px == pytest.approx(0.0, abs=1e-9)


def test_a_correspondence_projected_outside_the_image_is_valid_and_keeps_its_residual() -> None:
    """Invalid means the camera model cannot project the geometry, not "outside the image".

    A reference names geometry that was really observed, so a projection that lands far outside
    the image is a large residual -- the very thing the statistics exist to expose -- and not a
    correspondence to set aside. Only the behind-the-camera one is invalid here.
    """
    frame = project_frame(
        [
            map_point_for_pixel(100, 100, 3.0),
            map_point_for_pixel(-2500, 240, 3.0),
            (-3.0, 0.0, 0.0),
        ]
    )
    assert bool(frame.projectable[1]) and not bool(frame.in_prepared_image[1])
    observed = np.array([[100.0, 100.0], [-280.0, 240.0], [10.0, 10.0]])
    correspondences = TrustedCorrespondences(
        reference_id="trusted-0001", geometry_indices=np.arange(3), observed_pixels=observed
    )

    statistics = reprojection_statistics(frame, correspondences)

    assert statistics is not None
    assert (statistics.correspondence_count, statistics.invalid_count) == (3, 1)
    assert statistics.evaluated_count == 3
    assert (statistics.mean_px, statistics.max_px) == pytest.approx((1110.0, 2220.0))


def test_the_diagnostics_never_fabricate_a_reprojection_without_a_trusted_reference() -> None:
    diagnostics = _diagnose(scene_frame(*PIXELS))

    assert diagnostics.reprojection is None
    assert diagnostics.reprojection_attempt.outcome is ReprojectionOutcome.NO_REFERENCE
    assert diagnostics.reprojection_attempt.reference_id is None
    assert diagnostics.reprojection_attempt.correspondence_count == 0
    assert diagnostics.findings == ()


def test_when_no_reference_correspondence_projects_the_diagnostics_fail() -> None:
    frame = project_frame([(-3.0, 0.0, 0.0), (-2.0, 1.0, 0.0)])
    correspondences = TrustedCorrespondences(
        reference_id="trusted-0001",
        geometry_indices=np.arange(2),
        observed_pixels=np.array([[10.0, 10.0], [20.0, 20.0]]),
    )

    diagnostics = _diagnose(frame, correspondences=correspondences)

    assert diagnostics.reprojection is None
    assert diagnostics.failed
    assert FindingCode.NO_REFERENCE_CORRESPONDENCE_PROJECTS in {
        f.code for f in diagnostics.findings
    }


def test_the_reference_must_be_well_formed() -> None:
    with pytest.raises(ValueError, match="reference_id"):
        TrustedCorrespondences(
            reference_id="", geometry_indices=np.arange(1), observed_pixels=np.zeros((1, 2))
        )
    with pytest.raises(ValueError, match="shape"):
        TrustedCorrespondences(
            reference_id="r", geometry_indices=np.arange(2), observed_pixels=np.zeros((1, 2))
        )
    with pytest.raises(ValueError, match="finite"):
        TrustedCorrespondences(
            reference_id="r",
            geometry_indices=np.arange(1),
            observed_pixels=np.array([[float("nan"), 1.0]]),
        )
    with pytest.raises(ValueError, match="at least one"):
        TrustedCorrespondences(
            reference_id="r", geometry_indices=np.arange(0), observed_pixels=np.zeros((0, 2))
        )


@pytest.mark.parametrize(
    "indices",
    [
        pytest.param(np.array([0.5]), id="fractional"),
        pytest.param(np.array([float("nan")]), id="nan"),
        pytest.param(np.array([0.0, 1.0]), id="integral-valued-float"),
        pytest.param(np.array([0, float("nan")]), id="one-nan-among-integers"),
    ],
)
def test_a_reference_index_that_is_not_an_integer_is_rejected(indices: NDArray[Any]) -> None:
    """A malformed index must not be mistaken for geometry the candidate policy skipped.

    The lookup matches by equality, so ``0.5`` and ``NaN`` equal no candidate index and used
    to be counted as unevaluated. ``NaN`` also slips past the bounds guard, because every
    comparison with it is false. Both turned a malformed trusted reference into an apparent
    effect of the candidate policy, in the very metric that measures that policy. An
    integral-valued float is rejected too: it only matches by accident, and the contract is
    the integer identity of :func:`~contextmap.geometric_mapping.geometry_id_for`.
    """
    with pytest.raises(ValueError, match="integer"):
        TrustedCorrespondences(
            reference_id="r",
            geometry_indices=indices,
            observed_pixels=np.zeros((indices.shape[0], 2)),
        )


def test_a_reference_to_geometry_outside_the_map_is_rejected() -> None:
    frame = scene_frame((100, 100, 3.0))
    correspondences = TrustedCorrespondences(
        reference_id="r", geometry_indices=np.array([7]), observed_pixels=np.zeros((1, 2))
    )

    with pytest.raises(ValueError, match="geometry_indices"):
        reprojection_statistics(frame, correspondences)


# --- Misalignment produces explicit findings --------------------------------


def test_a_misaligned_calibration_fixture_is_flagged_by_its_reprojection_residual() -> None:
    points = [
        map_point_for_pixel(u, v, z)
        for u, v, z in [(100, 100, 3.0), (300, 200, 4.0), (500, 300, 5.0)]
    ]
    true_frame = project_frame(points)
    yaw_error = (0.0, 0.0, math.sin(math.radians(1.0)), math.cos(math.radians(1.0)))
    misaligned = make_calibration(
        extrinsic_rotation=quaternion_multiply(yaw_error, BODY_TO_CAMERA_ROTATION)
    )
    bad_frame = project_frame(points, calibration=misaligned)
    correspondences = TrustedCorrespondences(
        reference_id="trusted-0001",
        geometry_indices=np.arange(3),
        observed_pixels=true_frame.raw_pixels,
    )

    good = _diagnose(true_frame, TOLERANCES, correspondences=correspondences)
    bad = _diagnose(bad_frame, TOLERANCES, correspondences=correspondences)

    assert good.findings == ()
    codes = {finding.code: finding for finding in bad.findings}
    assert FindingCode.REPROJECTION_RESIDUAL_EXCEEDS_TOLERANCE in codes
    finding = codes[FindingCode.REPROJECTION_RESIDUAL_EXCEEDS_TOLERANCE]
    assert finding.severity is FindingSeverity.WARNING
    assert finding.tolerance == 5.0
    assert finding.observed is not None and finding.observed > 5.0


def test_an_invalid_rate_above_the_tolerance_is_a_warning() -> None:
    frame = project_frame([map_point_for_pixel(100, 100, 3.0), (-3.0, 0.0, 0.0), (-2.0, 0.0, 0.0)])
    correspondences = TrustedCorrespondences(
        reference_id="r",
        geometry_indices=np.arange(3),
        observed_pixels=np.vstack([frame.raw_pixels[:1], [[1.0, 1.0], [2.0, 2.0]]]),
    )

    diagnostics = _diagnose(frame, TOLERANCES, correspondences=correspondences)

    assert FindingCode.REPROJECTION_INVALID_RATE_EXCEEDS_TOLERANCE in {
        f.code for f in diagnostics.findings
    }


def test_a_pose_lookup_delta_above_the_tolerance_is_a_warning() -> None:
    poses = [(0, (0.0, 0.0, 0.0), IDENTITY), (100_000_000, (0.0, 0.0, 0.0), IDENTITY)]
    frame = project_frame(
        [map_point_for_pixel(100, 100, 3.0)],
        time_ns=130_000_000,
        poses=poses,
        pose_policy=LookupPolicy.nearest(max_time_delta_ns=50_000_000),
    )

    diagnostics = _diagnose(frame, TOLERANCES)

    codes = {finding.code: finding for finding in diagnostics.findings}
    finding = codes[FindingCode.POSE_TIME_DELTA_EXCEEDS_TOLERANCE]
    assert finding.severity is FindingSeverity.WARNING
    assert (finding.observed, finding.tolerance) == (30_000_000, 20_000_000)
    # O frame também está 30 ms depois da janela de aquisição do mapa (0 a 100 ms).
    assert diagnostics.map_window_offset_ns == 30_000_000
    assert FindingCode.FRAME_OUTSIDE_MAP_TIME_WINDOW in codes


def test_a_frame_before_the_geometry_acquisition_window_is_flagged_too() -> None:
    poses = [(-100_000_000, (0.0, 0.0, 0.0), IDENTITY), (100_000_000, (0.0, 0.0, 0.0), IDENTITY)]
    frame = project_frame(
        [map_point_for_pixel(100, 100, 3.0)],
        time_ns=-30_000_000,
        poses=poses,
        pose_policy=LookupPolicy.interpolated(),
    )

    diagnostics = _diagnose(frame, TOLERANCES)

    assert diagnostics.map_window_offset_ns == -30_000_000
    finding = {f.code: f for f in diagnostics.findings}[FindingCode.FRAME_OUTSIDE_MAP_TIME_WINDOW]
    assert (finding.observed, finding.tolerance) == (30_000_000, 10_000_000)


def test_a_frame_inside_the_tolerances_has_no_findings() -> None:
    diagnostics = _diagnose(scene_frame(*PIXELS), TOLERANCES)

    assert diagnostics.findings == ()
    assert not diagnostics.failed


def test_nothing_visible_in_the_image_is_a_failure() -> None:
    diagnostics = _diagnose(project_frame([(-3.0, 0.0, 0.0), (-1.0, 2.0, 0.0)]))

    assert diagnostics.failed
    assert FindingCode.NOTHING_VISIBLE_IN_IMAGE in {f.code for f in diagnostics.findings}


def test_the_tolerances_are_validated_and_none_disables_a_check() -> None:
    for field_name, value in (
        ("max_pose_time_delta_ns", -1),
        ("max_map_window_offset_ns", -1),
        ("max_reprojection_p95_px", float("nan")),
        ("max_reprojection_invalid_rate", 1.5),
    ):
        with pytest.raises(ValueError, match=field_name):
            dataclasses.replace(TOLERANCES, **{field_name: value})  # type: ignore[arg-type]
    assert _diagnose(scene_frame(*PIXELS), NO_TOLERANCE).findings == ()


# --- Time-offset sweep, for diagnosis only ----------------------------------

_MOVING = [(0, (0.0, 0.0, 0.0), IDENTITY), (100_000_000, (1.0, 0.0, 0.0), IDENTITY)]
_SWEEP_POINTS = [(5.0, -1.0, 0.5), (6.0, 2.0, -0.3), (7.0, -2.0, 1.0), (8.0, 0.5, 0.2)]


def _sweep_setup():  # type: ignore[no-untyped-def]
    projector = make_projector(
        _SWEEP_POINTS, poses=_MOVING, pose_policy=LookupPolicy.interpolated()
    )
    observation = make_camera_observation(50_000_000)
    prepared = make_prepared_image()
    truth = projector.project(observation, prepared)
    assert isinstance(truth, FrameProjection)
    correspondences = TrustedCorrespondences(
        reference_id="trusted-0001",
        geometry_indices=np.arange(len(_SWEEP_POINTS)),
        observed_pixels=truth.raw_pixels,
    )
    return projector, observation, prepared, correspondences


def test_the_time_offset_sweep_finds_the_offset_that_aligns_the_reference() -> None:
    projector, observation, prepared, correspondences = _sweep_setup()

    outcomes = time_offset_sweep(
        projector, observation, prepared, correspondences, [-20_000_000, 0, 20_000_000]
    )

    assert [outcome.offset_ns for outcome in outcomes] == [-20_000_000, 0, 20_000_000]
    early, exact, late = (outcome.statistics for outcome in outcomes)
    assert exact is not None and early is not None and late is not None
    assert exact.max_px == pytest.approx(0.0, abs=1e-6)
    assert early.max_px > 1.0 and late.max_px > 1.0


def test_an_offset_the_pose_lookup_rejects_is_reported_not_hidden() -> None:
    projector, observation, prepared, correspondences = _sweep_setup()

    (outcome,) = time_offset_sweep(
        projector, observation, prepared, correspondences, [1_000_000_000]
    )

    assert outcome.statistics is None
    assert outcome.rejection is LookupRejection.OUT_OF_RANGE


def test_the_sweep_never_modifies_the_calibration_or_the_timestamps() -> None:
    projector, observation, prepared, correspondences = _sweep_setup()
    before = projector.project(observation, prepared)

    time_offset_sweep(projector, observation, prepared, correspondences, [-10_000_000, 10_000_000])
    after = projector.project(observation, prepared)

    assert observation.timestamp == make_camera_observation(50_000_000).timestamp
    assert np.array_equal(before.raw_pixels, after.raw_pixels)  # type: ignore[union-attr]


# --- Native and enhanced feature paths --------------------------------------


def test_native_and_enhanced_paths_share_identical_geometry_and_calibration_diagnostics() -> None:
    frame = scene_frame((100, 100, 3.0), (300, 200, 4.0))
    resolution = resolve_visibility(frame, POLICY)
    native = make_dense_map()
    fine = make_sampling(
        (80, 60), stride=(8.0, 8.0), support=(8.0, 8.0), transform_id="enhanced-v1"
    )
    enhanced = make_dense_map(
        fine, feature_id="dense-enhanced", enhancement=make_enhancement(native, fine)
    )

    def diagnose(dense_map):  # type: ignore[no-untyped-def]
        samples = sample_dense_features(
            resolution,
            make_dense_result(dense_map),
            dense_map,
            interpolation=InterpolationPolicy.NEAREST,
        )
        return diagnose_frame(resolution, tolerances=TOLERANCES, dense_samples=(samples,))

    native_record = diagnose(native).to_record()
    enhanced_record = diagnose(enhanced).to_record()

    geometry_native = {k: v for k, v in native_record.items() if k != "dense_sampling"}
    geometry_enhanced = {k: v for k, v in enhanced_record.items() if k != "dense_sampling"}
    assert geometry_native == geometry_enhanced
    (native_summary,) = native_record["dense_sampling"]
    (enhanced_summary,) = enhanced_record["dense_sampling"]
    assert native_summary["enhanced"] is False
    assert enhanced_summary["enhanced"] is True
    assert enhanced_summary["enhancement_source_feature_id"] == "dense-native"
    assert (native_summary["eligible_count"], native_summary["sampled_count"]) == (2, 2)
    assert native_summary["feature_id"] != enhanced_summary["feature_id"]
    assert native_summary["sampling_fingerprint"] != enhanced_summary["sampling_fingerprint"]


# --- Machine-readable report ------------------------------------------------


def test_the_report_is_json_and_keeps_the_raw_components() -> None:
    frame = scene_frame(*PIXELS)
    correspondences = _correspondences(frame.raw_pixels[:4], [0.0, 1.0, 2.0, 3.0])

    record = json.loads(
        json.dumps(_diagnose(frame, TOLERANCES, correspondences=correspondences).to_record())
    )

    assert record["definitions_version"] == DIAGNOSTICS_DEFINITIONS_VERSION
    assert record["calibration_ref"]["camera_model_kind"] == "pinhole"
    assert record["image_timestamp"]["clock_id"] == "fixture:header"
    assert record["reprojection"]["median_px"] == pytest.approx(1.5)
    assert record["visibility"]["visible"] == 4
    assert record["findings"] == []
    assert record["visibility_policy"]["fingerprint"] == POLICY.fingerprint()


# --- "not evaluated" is not "not projectable" (review of PR #565) ----------------------------


def _reference_beyond(indices: tuple[int, ...], pixels: NDArray[Any]) -> TrustedCorrespondences:
    return TrustedCorrespondences(
        reference_id="trusted-range",
        geometry_indices=np.array(indices),
        observed_pixels=pixels,
    )


def test_a_reference_the_candidate_policy_excluded_is_a_warning_not_a_failure() -> None:
    """A range policy that excludes the reference geometry says nothing about the camera.

    Before this, `reprojection_statistics()` returned `None` for both "evaluated and
    unprojectable" and "never evaluated", and the frame was failed with
    `NO_REFERENCE_CORRESPONDENCE_PROJECTS` either way -- turning a declared candidate range into
    a calibration failure.
    """
    near = map_point_for_pixel(100.0, 100.0, 3.0)
    far = map_point_for_camera_point((0.0, 0.0, 60.0))
    frame = project_frame([near, far], candidate_policy=CandidateGeometryPolicy(max_range_m=20.0))
    resolution = resolve_visibility(frame, POLICY)
    # A referência nomeia só o elemento global 1, que o recorte de 20 m não avaliou.
    reference = _reference_beyond((1,), np.array([[320.0, 240.0]]))

    report = diagnose_frame(resolution, tolerances=TOLERANCES, correspondences=reference)

    assert report.reprojection is None
    codes = {finding.code: finding.severity for finding in report.findings}
    assert codes[FindingCode.REFERENCE_NOT_EVALUATED] is FindingSeverity.WARNING
    assert FindingCode.NO_REFERENCE_CORRESPONDENCE_PROJECTS not in codes
    attempt = report.reprojection_attempt
    assert attempt.outcome is ReprojectionOutcome.NOT_EVALUATED
    assert (attempt.correspondence_count, attempt.evaluated_count) == (1, 0)
    assert attempt.unevaluated_count == 1
    assert attempt.invalid_rate is None


def test_a_reference_the_frame_evaluated_but_cannot_project_is_still_a_failure() -> None:
    """The original meaning survives: evaluated geometry that will not project is a failure."""
    behind = map_point_for_camera_point((0.0, 0.0, -4.0))
    frame = project_frame([behind], candidate_policy=CandidateGeometryPolicy(max_range_m=20.0))
    resolution = resolve_visibility(frame, POLICY)
    reference = _reference_beyond((0,), np.array([[320.0, 240.0]]))

    report = diagnose_frame(resolution, tolerances=TOLERANCES, correspondences=reference)

    assert report.reprojection is None
    codes = {finding.code: finding.severity for finding in report.findings}
    assert codes[FindingCode.NO_REFERENCE_CORRESPONDENCE_PROJECTS] is FindingSeverity.FAILURE
    assert FindingCode.REFERENCE_NOT_EVALUATED not in codes


def test_the_invalid_rate_is_measured_over_the_evaluated_population_only() -> None:
    """With 90 of 100 references unevaluated, 1 invalid of 10 evaluated is 10%, not 1%.

    The diluted denominator would have hidden a real rate under any sane tolerance.
    """
    statistics = ReprojectionStatistics(
        reference_id="trusted-range",
        correspondence_count=100,
        invalid_count=1,
        unevaluated_count=90,
        mean_px=1.0,
        median_px=1.0,
        p95_px=1.0,
        max_px=1.0,
    )

    assert statistics.evaluated_count == 10
    assert statistics.invalid_rate == pytest.approx(0.1)

    strict = dataclasses.replace(TOLERANCES, max_reprojection_invalid_rate=0.05)
    findings = _reprojection_findings(
        statistics, strict, evaluated_count=10, correspondence_count=100
    )

    codes = [finding.code for finding in findings]
    assert FindingCode.REPROJECTION_INVALID_RATE_EXCEEDS_TOLERANCE in codes
    (rate_finding,) = [f for f in findings if f.code is codes[0]]
    assert rate_finding.observed == pytest.approx(0.1)


def test_statistics_need_at_least_one_evaluated_correspondence_that_projects() -> None:
    with pytest.raises(ValueError, match="at least one evaluated correspondence"):
        ReprojectionStatistics(
            reference_id="trusted-range",
            correspondence_count=10,
            invalid_count=4,
            unevaluated_count=6,
            mean_px=1.0,
            median_px=1.0,
            p95_px=1.0,
            max_px=1.0,
        )


def test_the_evaluated_count_is_defined_once_and_shared() -> None:
    near = map_point_for_pixel(100.0, 100.0, 3.0)
    far = map_point_for_camera_point((0.0, 0.0, 60.0))
    frame = project_frame([near, far], candidate_policy=CandidateGeometryPolicy(max_range_m=20.0))
    reference = _reference_beyond((0, 1), np.array([[100.0, 100.0], [320.0, 240.0]]))

    assert evaluated_correspondences(frame, reference) == 1
    statistics = reprojection_statistics(frame, reference)
    assert statistics is not None
    assert statistics.evaluated_count == 1
    assert statistics.unevaluated_count == 1


# --- The attempt contract closes its own declared states ------------------------------------


def _attempt(**overrides: object) -> ReprojectionAttempt:
    fields: dict[str, object] = {
        "outcome": ReprojectionOutcome.MEASURED,
        "reference_id": "trusted-0001",
        "correspondence_count": 4,
        "evaluated_count": 4,
        "invalid_count": 1,
    }
    fields.update(overrides)
    return ReprojectionAttempt(**fields)  # type: ignore[arg-type]


def test_a_frame_with_no_reference_cannot_count_correspondences() -> None:
    with pytest.raises(ValueError, match="no reference existed"):
        _attempt(
            outcome=ReprojectionOutcome.NO_REFERENCE,
            reference_id=None,
            correspondence_count=4,
            evaluated_count=0,
            invalid_count=0,
        )


@pytest.mark.parametrize(
    "outcome",
    [
        ReprojectionOutcome.NOT_EVALUATED,
        ReprojectionOutcome.NONE_PROJECTABLE,
        ReprojectionOutcome.MEASURED,
    ],
)
def test_a_supplied_reference_always_declares_at_least_one_correspondence(
    outcome: ReprojectionOutcome,
) -> None:
    with pytest.raises(ValueError, match="at least one correspondence"):
        _attempt(outcome=outcome, correspondence_count=0, evaluated_count=0, invalid_count=0)


def test_the_empty_attempt_of_a_frame_without_a_reference_is_valid() -> None:
    attempt = _attempt(
        outcome=ReprojectionOutcome.NO_REFERENCE,
        reference_id=None,
        correspondence_count=0,
        evaluated_count=0,
        invalid_count=0,
    )

    assert attempt.unevaluated_count == 0
    assert attempt.invalid_rate is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"invalid_count": 5}, "must narrow"),
        ({"evaluated_count": 5}, "must narrow"),
        ({"correspondence_count": -1}, "not be negative"),
        ({"outcome": ReprojectionOutcome.NOT_EVALUATED}, "nothing was evaluated"),
        ({"outcome": ReprojectionOutcome.NONE_PROJECTABLE}, "every evaluated"),
        ({"invalid_count": 4, "outcome": ReprojectionOutcome.MEASURED}, "that projects"),
        ({"reference_id": None}, "exactly when"),
    ],
)
def test_an_attempt_that_contradicts_its_own_counts_is_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _attempt(**overrides)

"""ExternalPose and FAST-LIO evaluated under one report format, through the real backends.

The FAST-LIO process is a stand-in that publishes the reference motion in the local frame
FAST-LIO anchors at its first pose. The reference here is the very pose stream ExternalPose
publishes, so ExternalPose's own error is zero by construction (a positive control); the point
is that a local-frame estimator becomes comparable only through an explicit, reported
alignment, and that both backends end up in one comparison.
"""

import math
from collections.abc import Mapping

import pytest
from calibration_builders import calibration, rigid, sensor_run
from pose_builders import make_external_pose, make_request

from contextmap.evaluation import (
    AlignmentMethod,
    ReferenceComparisonConfig,
    ReferenceRole,
    ReferenceTrajectory,
    StateEstimationEvaluationReport,
    compare_state_estimation_reports,
    evaluate_state_estimation,
)
from contextmap.ingestion import FrameId
from contextmap.shared import Quaternion, Vector3, compose_rigid, invert_rigid
from contextmap.state_estimation import (
    StateEstimationRequest,
    TrajectoryId,
    execute_state_estimation,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
)
from contextmap.state_estimation.backends.fast_lio import (
    FastLioConfig,
    FastLioEstimator,
    FastLioJob,
    FastLioRawPose,
    FastLioRunOutput,
)

MS = 1_000_000
SCANS = 8
SCAN_PERIOD_NS = 100 * MS
SEQUENCE_ARTIFACT_ID = make_request([]).sequence_artifact_id


def _reference_pose(index: int) -> tuple[Vector3, Quaternion]:
    """A gentle arc that starts away from the origin and turned, moving at every scan."""
    yaw = 0.4 + 0.15 * index
    translation = (10.0 + index, -4.0 + 0.05 * index * index, 0.5)
    return translation, (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


class _LocalFrameFastLio:
    """Publishes the reference motion in the frame anchored at the first pose."""

    def describe(self) -> Mapping[str, object]:
        return {"runner": "local-frame stand-in"}

    def run(self, job: FastLioJob) -> FastLioRunOutput:
        first_translation, first_rotation = _reference_pose(0)
        inverse_translation, inverse_rotation = invert_rigid(
            translation=first_translation, rotation=first_rotation
        )
        poses = []
        for index, scan in enumerate(job.lidar):
            translation, rotation = _reference_pose(index)
            local_translation, local_rotation = compose_rigid(
                outer_translation=inverse_translation,
                outer_rotation=inverse_rotation,
                inner_translation=translation,
                inner_rotation=rotation,
            )
            poses.append(
                FastLioRawPose(
                    timestamp_ns=scan.timestamp.total_nanoseconds() + 50 * MS,
                    translation=local_translation,
                    orientation=local_rotation,
                    covariance=None,
                )
            )
        return FastLioRunOutput(poses=tuple(poses), reported_ref="v1.0.0", warnings=())


def _evaluate_both(alignment: AlignmentMethod) -> list[StateEstimationEvaluationReport]:
    measurements = [
        make_external_pose(
            index,
            time_ns=index * SCAN_PERIOD_NS + 50 * MS,
            parent_frame="map",
            child_frame="imu",
            translation=_reference_pose(index)[0],
            orientation=_reference_pose(index)[1],
        )
        for index in range(SCANS)
    ]
    sensors = sensor_run(scans=SCANS, scan_period_ns=SCAN_PERIOD_NS)
    request = StateEstimationRequest(
        trajectory_id=TrajectoryId("run-0001--trajectory"),
        sequence_artifact_id=SEQUENCE_ARTIFACT_ID,
        selection_id="full-sequence",
        observations=(*sensors, *measurements),
        calibration=calibration(rigid("imu", "velodyne", (0.1, 0.0, 0.2))),
    )
    external = execute_state_estimation(
        ExternalPoseEstimator(
            ExternalPoseConfig(reference_frame=FrameId("map"), body_frame=FrameId("imu"))
        ),
        request,
    )
    fast_lio = execute_state_estimation(
        FastLioEstimator(
            FastLioConfig(
                reference_frame=FrameId("fast_lio_init"),
                body_frame=FrameId("imu"),
                fast_lio_ref="v1.0.0",
                scan_period_ns=SCAN_PERIOD_NS,
            ),
            _LocalFrameFastLio(),
        ),
        request,
    )
    reference = ReferenceTrajectory(
        trajectory=external.result.trajectory,
        reference_id="reference-profile:fixture@1",
        role=ReferenceRole.EVALUATION_REFERENCE,
    )
    config = ReferenceComparisonConfig(alignment=alignment, max_time_difference_ns=10 * MS)
    return [
        evaluate_state_estimation(
            trajectory=outcome.result.trajectory,
            result=outcome.result,
            reference=reference,
            reference_config=config,
        )
        for outcome in (external, fast_lio)
    ]


def test_both_backends_are_evaluated_and_compared_under_one_report_format() -> None:
    external, fast_lio = _evaluate_both(AlignmentMethod.SE3)

    comparison = compare_state_estimation_reports([external, fast_lio])

    assert [entry.estimator.backend_id for entry in comparison.entries] == [
        "external_pose",
        "fast_lio",
    ]
    # ExternalPose não consome calibração; o FAST-LIO consome: a comparação segue e cada
    # entrada guarda a sua identidade.
    assert comparison.entries[0].calibration_identity is None
    assert comparison.entries[1].calibration_identity is not None
    assert comparison.calibration_identity == comparison.entries[1].calibration_identity
    assert comparison.entries[0].ate_rmse_m == pytest.approx(0.0, abs=1e-9)
    # Com o alinhamento SE(3) explícito, o frame local do FAST-LIO passa a ser comparável.
    assert comparison.entries[1].ate_rmse_m == pytest.approx(0.0, abs=1e-9)
    assert fast_lio.accuracy is not None
    assert fast_lio.accuracy.alignment.method is AlignmentMethod.SE3
    assert fast_lio.accuracy.alignment.translation_m is not None
    assert fast_lio.accuracy.association.matched_count == SCANS


def test_without_an_alignment_the_local_frame_of_fast_lio_is_far_from_the_reference() -> None:
    external, fast_lio = _evaluate_both(AlignmentMethod.NONE)

    assert external.accuracy is not None and fast_lio.accuracy is not None
    assert external.accuracy.ate_translation_m.rmse == pytest.approx(0.0, abs=1e-9)
    assert fast_lio.accuracy.ate_translation_m.rmse > 1.0
    assert fast_lio.accuracy.alignment.translation_m is None

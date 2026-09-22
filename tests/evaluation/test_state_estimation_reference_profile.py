import hashlib
import json
import math
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from state_estimation_builders import MS, straight_line, trajectory_from

from contextmap.evaluation import (
    AlignmentMethod,
    ReferenceRole,
    StateEstimationEvaluationError,
    decode_reference_profile,
    encode_reference_profile,
    evaluate_state_estimation,
)
from contextmap.state_estimation import Trajectory

_REPOSITORY = Path(__file__).resolve().parents[2]
_PROFILE_FILE = (
    _REPOSITORY
    / "src"
    / "contextmap"
    / "evaluation"
    / "profiles"
    / "corridor-02-state-estimation-v1.json"
)


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": "contextmap.state_estimation_reference_profile/1",
        "profile_id": "reference-profile:fixture@1",
        "reference_role": "evaluation_reference",
        "reference_source": "datasets/fixture/poses.txt",
        "reference_source_sha256": "sha256:" + "0" * 64,
        "thresholds": {"max_linear_speed_mps": 5.0, "max_angular_speed_radps": 2.0},
        "comparison": {
            "alignment": "se3",
            "max_time_difference_ns": 10 * MS,
            "relative_interval_ns": 1000 * MS,
            "relative_interval_tolerance_ns": 50 * MS,
        },
    }
    record.update(overrides)
    return record


def _write_source(tmp_path: Path, content: bytes = b"0 0 0 0 0 0 0 1\n") -> tuple[Path, str]:
    path = tmp_path / "poses.txt"
    path.write_bytes(content)
    return path, "sha256:" + hashlib.sha256(content).hexdigest()


def _tum_line(trajectory: Trajectory, index: int) -> str:
    pose = trajectory.poses[index]
    seconds, nanoseconds = pose.timestamp.seconds, pose.timestamp.nanoseconds
    values = (*pose.translation_m, *pose.orientation)
    return f"{seconds}.{nanoseconds:09d} " + " ".join(repr(value) for value in values)


def _write_reference_file(tmp_path: Path, trajectory: Trajectory) -> tuple[Path, str]:
    """Write the poses of ``trajectory`` as the TUM file a reference would be read from."""
    lines = [_tum_line(trajectory, index) for index in range(len(trajectory.poses))]
    return _write_source(tmp_path, ("\n".join(lines) + "\n").encode())


def _window(trajectory: Trajectory, start: int, stop: int) -> Trajectory:
    return trajectory_from(
        [
            (pose.timestamp.total_nanoseconds(), pose.translation_m, pose.orientation)
            for pose in trajectory.poses[start:stop]
        ]
    )


def test_a_profile_declares_the_thresholds_protocol_and_role_of_its_reference() -> None:
    profile = decode_reference_profile(_record())

    assert profile.profile_id == "reference-profile:fixture@1"
    assert profile.reference_role is ReferenceRole.EVALUATION_REFERENCE
    assert profile.thresholds.max_linear_speed_mps == 5.0
    assert profile.thresholds.max_angular_speed_radps == 2.0
    # Um limite que o perfil não declara nunca é aplicado.
    assert profile.thresholds.max_translation_delta_m is None
    assert profile.comparison.alignment is AlignmentMethod.SE3
    assert profile.comparison.max_time_difference_ns == 10 * MS
    assert profile.comparison.relative_interval_ns == 1000 * MS
    assert profile.comparison.relative_interval_tolerance_ns == 50 * MS


def test_a_profile_round_trips_through_its_encoding() -> None:
    record = _record()

    assert encode_reference_profile(decode_reference_profile(record)) == record


def test_a_profile_declares_a_reference_only_for_the_file_whose_hash_it_names(
    tmp_path: Path,
) -> None:
    trajectory = straight_line(4)
    source, digest = _write_reference_file(tmp_path, trajectory)
    profile = decode_reference_profile(_record(reference_source_sha256=digest))

    reference = profile.declare_reference(trajectory, source_file=source)

    assert reference.role is ReferenceRole.EVALUATION_REFERENCE
    assert reference.reference_id == "reference-profile:fixture@1"
    assert reference.source == "datasets/fixture/poses.txt"
    assert reference.source_sha256 == digest
    assert reference.trajectory is trajectory


def test_a_file_with_the_expected_name_but_other_content_is_not_the_reference(
    tmp_path: Path,
) -> None:
    source, _ = _write_source(tmp_path, b"someone edited this file\n")
    profile = decode_reference_profile(_record())

    with pytest.raises(StateEstimationEvaluationError, match="sha256"):
        profile.declare_reference(straight_line(4), source_file=source)


def test_a_trajectory_the_verified_file_does_not_hold_is_not_declared_its_reference(
    tmp_path: Path,
) -> None:
    # Regressão da revisão da PR #419: o hash do arquivo era validado, mas qualquer trajetória
    # era aceita, então o relatório afirmava o hash de uma fonte e calculava ATE/RPE contra outra.
    source, digest = _write_reference_file(tmp_path, straight_line(4))
    profile = decode_reference_profile(_record(reference_source_sha256=digest))
    moved = straight_line(4, offset=(0.0, 0.01, 0.0))

    with pytest.raises(StateEstimationEvaluationError, match="differs from the file"):
        profile.declare_reference(moved, source_file=source)


def test_a_pose_at_a_time_the_verified_file_has_no_sample_for_is_refused(tmp_path: Path) -> None:
    source, digest = _write_reference_file(tmp_path, straight_line(4))
    profile = decode_reference_profile(_record(reference_source_sha256=digest))
    shifted = straight_line(4, time_offset_ns=MS)

    with pytest.raises(StateEstimationEvaluationError, match="no sample at that time"):
        profile.declare_reference(shifted, source_file=source)


def test_a_trajectory_that_is_only_a_window_of_the_file_is_still_its_reference(
    tmp_path: Path,
) -> None:
    full = straight_line(8)
    source, digest = _write_reference_file(tmp_path, full)
    profile = decode_reference_profile(_record(reference_source_sha256=digest))

    reference = profile.declare_reference(_window(full, 2, 6), source_file=source)

    assert len(reference.trajectory.poses) == 4


def test_an_orientation_the_source_renormalized_still_matches_the_file(tmp_path: Path) -> None:
    # O backend renormaliza um quaternion dentro da tolerância; a rotação é a mesma.
    source, digest = _write_source(tmp_path, b"1.000000000 0 0 0 0 0 0 1.0004\n")
    profile = decode_reference_profile(_record(reference_source_sha256=digest))
    trajectory = trajectory_from([(1_000_000_000, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))])

    profile.declare_reference(trajectory, source_file=source)


def test_comments_and_blank_lines_of_the_reference_file_are_skipped(tmp_path: Path) -> None:
    source, digest = _write_source(
        tmp_path, b"# t x y z qx qy qz qw\n\n1.000000000 0 0 0 0 0 0 1\n"
    )
    profile = decode_reference_profile(_record(reference_source_sha256=digest))
    trajectory = trajectory_from([(1_000_000_000, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))])

    profile.declare_reference(trajectory, source_file=source)


@pytest.mark.parametrize(
    "bad_line",
    [
        "2.0 0 0 0 0 0 0",
        "2.0 0 0 0 0 0 0 1 9",
        "two 0 0 0 0 0 0 1",
        "2.0 0 0 x 0 0 0 1",
        "2.0 nan 0 0 0 0 0 1",
        "inf 0 0 0 0 0 0 1",
        "2.0 0 0 0 0 0 0 0",
        "2.0000000001 0 0 0 0 0 0 1",
    ],
)
def test_a_reference_file_line_that_is_not_a_pose_is_refused_with_its_number(
    tmp_path: Path, bad_line: str
) -> None:
    content = f"1.000000000 0 0 0 0 0 0 1\n{bad_line}\n".encode()
    source, digest = _write_source(tmp_path, content)
    profile = decode_reference_profile(_record(reference_source_sha256=digest))
    trajectory = trajectory_from([(1_000_000_000, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))])

    with pytest.raises(StateEstimationEvaluationError, match="line 2"):
        profile.declare_reference(trajectory, source_file=source)


def test_a_reference_file_that_is_not_text_is_refused(tmp_path: Path) -> None:
    source, digest = _write_source(tmp_path, b"\xff\xfe\x00")
    profile = decode_reference_profile(_record(reference_source_sha256=digest))

    with pytest.raises(StateEstimationEvaluationError, match="not UTF-8"):
        profile.declare_reference(straight_line(2), source_file=source)


def test_a_declared_reference_is_what_the_evaluation_compares_against(tmp_path: Path) -> None:
    trajectory = straight_line(12)
    source, digest = _write_reference_file(tmp_path, trajectory)
    profile = decode_reference_profile(_record(reference_source_sha256=digest))
    reference = profile.declare_reference(trajectory, source_file=source)

    report = evaluate_state_estimation(
        trajectory=straight_line(12, offset=(0.0, 0.05, 0.0)),
        thresholds=profile.thresholds,
        reference=reference,
        reference_config=profile.comparison,
    )

    accuracy = report.accuracy
    assert accuracy is not None
    assert accuracy.reference_id == profile.profile_id
    assert accuracy.reference_source_sha256 == digest
    assert accuracy.alignment.method is AlignmentMethod.SE3
    assert accuracy.rpe is not None and accuracy.rpe.interval_ns == 1000 * MS
    assert report.motion.thresholds == profile.thresholds


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "contextmap.state_estimation_reference_profile/2"},
        {"reference_role": "ground_truth"},
        {"reference_source_sha256": "abc"},
        {"profile_id": ""},
        {"thresholds": {"max_linear_speed_mps": -1.0}},
        {"thresholds": {"max_speed": 5.0}},
    ],
)
def test_an_invalid_profile_is_refused_before_it_can_declare_anything(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises((ValueError, StateEstimationEvaluationError)):
        decode_reference_profile(_record(**overrides))


def test_a_profile_missing_a_field_names_it() -> None:
    record = _record()
    del record["comparison"]

    with pytest.raises(StateEstimationEvaluationError, match="comparison"):
        decode_reference_profile(record)


def test_the_committed_corridor_02_profile_is_valid_and_declares_its_reference() -> None:
    profile = decode_reference_profile(json.loads(_PROFILE_FILE.read_text(encoding="utf-8")))

    assert profile.profile_id == "reference-profile:corridor-02@1"
    assert profile.reference_role is ReferenceRole.EVALUATION_REFERENCE
    assert profile.reference_source == "datasets/corridor-02/corridor-02-gt.txt"
    # O quadro do FAST-LIO é local: comparar com esta referência exige alinhamento explícito.
    assert profile.comparison.alignment is AlignmentMethod.SE3
    assert profile.thresholds.max_linear_speed_mps is not None
    assert profile.thresholds.max_angular_speed_radps is not None


# --- Real data (optional) ----------------------------------------------------

_DATASET = Path(
    os.environ.get("CONTEXTMAP_CORRIDOR02_DIR", _REPOSITORY / "datasets" / "corridor-02")
)


@pytest.mark.skipif(
    not (_DATASET / "corridor-02-gt.txt").is_file(),
    reason="corridor-02 dataset is not available (it is not versioned)",
)
def test_the_real_corridor_02_reference_matches_its_profile_and_shows_no_motion_anomaly() -> None:
    profile = decode_reference_profile(json.loads(_PROFILE_FILE.read_text(encoding="utf-8")))
    source = _DATASET / "corridor-02-gt.txt"
    samples = []
    for line in source.read_text().splitlines():
        t, x, y, z, qx, qy, qz, qw = line.split()
        norm = math.sqrt(sum(float(v) ** 2 for v in (qx, qy, qz, qw)))
        samples.append(
            (
                int(Decimal(t) * 1_000_000_000),
                (float(x), float(y), float(z)),
                tuple(float(v) / norm for v in (qx, qy, qz, qw)),
            )
        )
    trajectory = trajectory_from(samples)  # type: ignore[arg-type]

    reference = profile.declare_reference(trajectory, source_file=source)
    report = evaluate_state_estimation(trajectory=trajectory, thresholds=profile.thresholds)

    assert reference.source_sha256 == profile.reference_source_sha256
    assert report.motion.anomalies == ()

    # Uma única pose deslocada em 1 mm deixa de ser uma amostra do arquivo verificado.
    moved_time, position, orientation = samples[100]
    samples[100] = (moved_time, (position[0] + 0.001, position[1], position[2]), orientation)
    with pytest.raises(StateEstimationEvaluationError, match="differs from the file"):
        profile.declare_reference(trajectory_from(samples), source_file=source)  # type: ignore[arg-type]

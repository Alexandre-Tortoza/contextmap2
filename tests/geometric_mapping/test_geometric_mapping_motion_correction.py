import dataclasses
import hashlib
import json

import pytest
from geometry_builders import make_lineage, make_point
from lidar_builders import make_scan, timestamp_ns

from contextmap.geometric_mapping import (
    GeometryPointProvenance,
    MotionCorrectionEvidence,
    MotionCorrectionPolicy,
    MotionCorrectionRecord,
    MotionCorrectionState,
    ScanDisposition,
    apply_motion_correction_policy,
    declared_raw_motion_correction,
    unknown_motion_correction,
    verify_motion_correction,
)
from contextmap.geometric_mapping.serialization import (
    decode_geometry_point,
    decode_motion_correction_policy,
    decode_motion_correction_record,
    encode_geometry_point,
    encode_motion_correction_policy,
    encode_motion_correction_record,
)
from contextmap.ingestion import LidarObservation, SourceObservationId
from contextmap.state_estimation import TrajectoryId

MS = 1_000_000
OBSERVATION_ID = SourceObservationId("scan-0001")


def _evidence(**overrides: object) -> MotionCorrectionEvidence:
    values: dict[str, object] = {
        "producer": "deskew-backend@1",
        "trajectory_id": TrajectoryId("run-0001--trajectory"),
        "payload_hash": "sha256:" + "ab" * 32,
        "configuration_fingerprint": "sha256:cfg",
    }
    values.update(overrides)
    return MotionCorrectionEvidence(**values)  # type: ignore[arg-type]


def _corrected(**overrides: object) -> MotionCorrectionRecord:
    values: dict[str, object] = {
        "observation_id": OBSERVATION_ID,
        "state": MotionCorrectionState.CORRECTED,
        "acquisition_start": timestamp_ns(0),
        "acquisition_end": timestamp_ns(100 * MS),
        "evidence": _evidence(),
    }
    values.update(overrides)
    return MotionCorrectionRecord(**values)  # type: ignore[arg-type]


def _describing(scan: LidarObservation) -> MotionCorrectionRecord:
    """A corrected record whose evidence names the scan's actual payload."""
    return _corrected(evidence=_evidence(payload_hash=_payload_hash(scan)))


def _payload_hash(scan: LidarObservation) -> str:
    return f"sha256:{hashlib.sha256(scan.data).hexdigest()}"


# --- The three states are explicit, and unknown is the default ---------------


def test_an_observation_with_no_declaration_is_unknown_never_assumed() -> None:
    scan = make_scan()

    record = unknown_motion_correction(scan)

    assert record.state is MotionCorrectionState.UNKNOWN
    assert record.observation_id == scan.observation_id
    assert record.evidence is None


def test_the_state_is_never_inferred_from_where_the_scan_came_from() -> None:
    scan = make_scan()
    fast_lio_scan = dataclasses.replace(
        scan,
        provenance=dataclasses.replace(
            scan.provenance, source_type="fast_lio", raw_metadata={"deskewed": True}
        ),
    )

    assert unknown_motion_correction(fast_lio_scan).state is MotionCorrectionState.UNKNOWN


def test_a_scan_declared_raw_is_raw_and_carries_no_correction_evidence() -> None:
    record = declared_raw_motion_correction(make_scan())

    assert record.state is MotionCorrectionState.RAW
    assert record.evidence is None


def test_a_point_carries_an_explicit_correction_state_defaulting_to_unknown() -> None:
    assert make_point().provenance.motion_correction is MotionCorrectionState.UNKNOWN

    raw = make_point(
        provenance=GeometryPointProvenance(motion_correction=MotionCorrectionState.RAW)
    )
    corrected = make_point(
        provenance=GeometryPointProvenance(motion_correction=MotionCorrectionState.CORRECTED)
    )

    assert raw.provenance.motion_correction is MotionCorrectionState.RAW
    assert corrected.provenance.motion_correction is MotionCorrectionState.CORRECTED


# --- Corrected claims need evidence ------------------------------------------


def test_a_corrected_scan_traces_to_its_producer_trajectory_configuration_and_payload() -> None:
    record = _corrected()

    assert record.evidence is not None
    assert record.evidence.producer == "deskew-backend@1"
    assert record.evidence.trajectory_id == "run-0001--trajectory"
    assert record.evidence.configuration_fingerprint == "sha256:cfg"
    assert record.evidence.payload_hash.startswith("sha256:")
    assert record.observation_id == OBSERVATION_ID


def test_marking_a_scan_corrected_without_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="evidence"):
        _corrected(evidence=None)


@pytest.mark.parametrize("state", [MotionCorrectionState.RAW, MotionCorrectionState.UNKNOWN])
def test_a_scan_that_is_not_corrected_cannot_carry_correction_evidence(
    state: MotionCorrectionState,
) -> None:
    with pytest.raises(ValueError, match="evidence"):
        MotionCorrectionRecord(observation_id=OBSERVATION_ID, state=state, evidence=_evidence())


@pytest.mark.parametrize(
    "overrides",
    [
        {"producer": ""},
        {"payload_hash": "not-a-hash"},
        {"payload_hash": "sha256:short"},
    ],
)
def test_the_evidence_must_be_complete_and_well_formed(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"producer|payload_hash"):
        _evidence(**overrides)


def test_a_correction_needs_the_timing_it_used_and_never_fabricates_it() -> None:
    with pytest.raises(ValueError, match="timing"):
        _corrected(acquisition_start=None, acquisition_end=None)

    per_point = _corrected(
        acquisition_start=None, acquisition_end=None, per_point_timing_available=True
    )

    assert per_point.acquisition_start is None
    assert per_point.per_point_timing_available


# --- Acquisition interval consistency ----------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "match"),
    [
        (timestamp_ns(100 * MS), timestamp_ns(0), "interval"),
        (timestamp_ns(0, clock_id="a"), timestamp_ns(1, clock_id="b"), "clock"),
        (timestamp_ns(0), None, "both"),
        (None, timestamp_ns(0), "both"),
    ],
)
def test_the_acquisition_interval_must_be_ordered_in_one_clock_and_complete(
    start: object, end: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        MotionCorrectionRecord(
            observation_id=OBSERVATION_ID,
            state=MotionCorrectionState.RAW,
            acquisition_start=start,  # type: ignore[arg-type]
            acquisition_end=end,  # type: ignore[arg-type]
        )


def test_a_record_must_agree_with_the_observation_it_describes() -> None:
    scan = make_scan(time_ns=50 * MS)

    assert verify_motion_correction(_describing(scan), scan) == []

    problems = verify_motion_correction(
        dataclasses.replace(_describing(scan), observation_id=SourceObservationId("other")), scan
    )
    assert any("observation" in problem for problem in problems)
    late = make_scan(time_ns=500 * MS)
    assert any(
        "outside" in problem for problem in verify_motion_correction(_describing(late), late)
    )
    other_clock = make_scan(time_ns=50 * MS, clock_id="another:clock")
    assert any(
        "clock" in problem
        for problem in verify_motion_correction(_describing(other_clock), other_clock)
    )


def test_a_corrected_record_must_name_the_payload_the_scan_delivers() -> None:
    # #595: a evidência amarra a correção a um payload; outro payload não é o corrigido.
    scan = make_scan(time_ns=50 * MS)
    declared = "sha256:" + "ab" * 32

    problems = verify_motion_correction(_corrected(evidence=_evidence(payload_hash=declared)), scan)

    assert len(problems) == 1
    assert declared in problems[0]
    assert _payload_hash(scan) in problems[0]


# --- Policy for uncorrected scans ---------------------------------------------


@pytest.mark.parametrize(
    ("policy", "state", "expected"),
    [
        (
            MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN),
            MotionCorrectionState.RAW,
            ScanDisposition.ACCEPT,
        ),
        (
            MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN),
            MotionCorrectionState.UNKNOWN,
            ScanDisposition.WARN,
        ),
        (
            MotionCorrectionPolicy(raw=ScanDisposition.REJECT, unknown=ScanDisposition.REJECT),
            MotionCorrectionState.RAW,
            ScanDisposition.REJECT,
        ),
        (
            MotionCorrectionPolicy(raw=ScanDisposition.REJECT, unknown=ScanDisposition.REJECT),
            MotionCorrectionState.CORRECTED,
            ScanDisposition.ACCEPT,
        ),
    ],
)
def test_the_policy_decides_per_state_and_always_accepts_corrected_scans(
    policy: MotionCorrectionPolicy, state: MotionCorrectionState, expected: ScanDisposition
) -> None:
    record = (
        _corrected()
        if state is MotionCorrectionState.CORRECTED
        else MotionCorrectionRecord(observation_id=OBSERVATION_ID, state=state)
    )

    verdict = apply_motion_correction_policy(record, policy)

    assert verdict.disposition is expected
    assert verdict.record is record
    assert (verdict.message is None) == (expected is ScanDisposition.ACCEPT)


def test_a_verdict_names_the_scan_and_the_reason_when_it_warns_or_rejects() -> None:
    policy = MotionCorrectionPolicy(raw=ScanDisposition.REJECT, unknown=ScanDisposition.WARN)

    rejected = apply_motion_correction_policy(declared_raw_motion_correction(make_scan()), policy)
    warned = apply_motion_correction_policy(unknown_motion_correction(make_scan()), policy)

    assert rejected.message is not None and "scan-0001" in rejected.message
    assert "raw" in rejected.message
    assert warned.message is not None and "unknown" in warned.message


# --- Serialization for the manifest and the artifact --------------------------


def test_records_and_the_policy_round_trip_through_json() -> None:
    policy = MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.REJECT)

    for record in (_corrected(), unknown_motion_correction(make_scan())):
        assert (
            decode_motion_correction_record(
                json.loads(json.dumps(encode_motion_correction_record(record)))
            )
            == record
        )
    assert (
        decode_motion_correction_policy(
            json.loads(json.dumps(encode_motion_correction_policy(policy)))
        )
        == policy
    )
    assert encode_motion_correction_policy(policy) == {"raw": "accept", "unknown": "reject"}


def test_the_point_state_round_trips_and_stays_distinguishable() -> None:
    point = make_point(
        provenance=GeometryPointProvenance(motion_correction=MotionCorrectionState.RAW),
        lineage=make_lineage(),
    )

    decoded = decode_geometry_point(json.loads(json.dumps(encode_geometry_point(point))))

    assert decoded == point
    assert decoded.provenance.motion_correction is MotionCorrectionState.RAW

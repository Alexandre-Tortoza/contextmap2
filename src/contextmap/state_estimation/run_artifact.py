"""Persisted, immutable, self-describing State Estimation run artifacts.

A ``StateEstimationRunArtifact`` is a directory holding the trajectory one
backend produced for one sequence selection, with the lineage, metrics and
preflight evidence needed to trust it. It opens without ROS, an estimator or
NumPy, and a single pose can be read by identity or time without loading the
others or any debug data. See ``src/contextmap/state_estimation/docs/artifact.md``
for the layout and the trade-offs of this first schema.

Contractual data lives in ``outputs/`` and ``metrics/`` and is inventoried with
size and SHA-256 in the manifest; ``debug/`` is human evidence that is never
inventoried, so removing it cannot invalidate the run and no downstream stage
may depend on it. Writing follows :class:`~contextmap.shared.AtomicRunDirectory`:
an interrupted write never looks like a finished run and a finished run is
never modified.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, NewType

from contextmap.ingestion import FrameId, SequenceArtifactId
from contextmap.shared import (
    AtomicRunDirectory,
    FileEntry,
    RunDirectoryError,
    SourceTimestamp,
    check_file_inventory,
)
from contextmap.state_estimation.lookup import ClockDomainMismatchError
from contextmap.state_estimation.models import (
    EstimatorProvenance,
    PoseEstimate,
    PoseEstimateId,
    Trajectory,
    TrajectoryId,
)
from contextmap.state_estimation.motion import (
    DistributionSummary,
    MotionSummary,
    motion_deltas,
    summarize_motion,
)
from contextmap.state_estimation.ports import EstimationDiagnostic
from contextmap.state_estimation.preflight import GeometryPreflightReport, PreflightStatus
from contextmap.state_estimation.serialization import (
    decode_pose_estimate,
    decode_trajectory,
    encode_pose_estimate,
    encode_trajectory_metadata,
)
from contextmap.state_estimation.service import StateEstimationOutcome

SCHEMA_VERSION = "0.2.0"
"""State Estimation run artifact schema version written and understood by this module.

Bumped from ``0.1.0`` for issue #555's ``auxiliary_sequence_artifact_id``/
``auxiliary_selection_id`` manifest fields.
"""

StateEstimationRunId = NewType("StateEstimationRunId", str)
"""Identity of one State Estimation run, local to its capability and sequence."""

_MANIFEST = "manifest.json"
_TRAJECTORY = "outputs/trajectory.json"
_POSES = "outputs/poses.jsonl"
_POSE_INDEX = "outputs/pose-index.jsonl"
_FRAME_SUMMARY = "outputs/frame-summary.json"
_QUALITY = "outputs/quality.json"
_PREFLIGHT = "metrics/preflight.json"
_MOTION = "metrics/motion.json"
_RUNTIME = "metrics/runtime.json"
_DIAGNOSTICS = "metrics/diagnostics.jsonl"
_NANOSECONDS_PER_SECOND = 1_000_000_000

# Semântica de interpolação usada por TrajectoryLookup, registrada para a leitura futura
# reproduzir a mesma pose derivada sem depender da versão do código.
_INTERPOLATION = {"translation": "linear", "orientation": "slerp_shortest_arc"}


class RunArtifactError(Exception):
    """Base class for State Estimation run artifact read/write failures."""


class IncompleteRunArtifactError(RunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class StateEstimationDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and required metrics only.
        STANDARD: Adds pose deltas, timestamp gaps and trajectory projections.
        FULL: Adds every preflight check and the backend diagnostic events.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, kw_only=True)
class StateEstimationRunManifest:
    """Authoritative metadata of a persisted State Estimation run.

    Attributes:
        run_id: Identity of the run, supplied by the caller.
        run_index: Ordinal of the run among the caller's runs of this sequence, supplied by the
            caller.
        sequence_name: Name of the processed sequence.
        sequence_artifact_id: Canonical sequence artifact consumed.
        selection_id: Deterministic identity of the sequence selection.
        auxiliary_sequence_artifact_id: The auxiliary pose sequence actually merged into this
            trajectory (issue #555), or ``None`` when none contributed. See
            :attr:`~contextmap.state_estimation.TrajectoryProvenance.
            auxiliary_sequence_artifact_id`.
        auxiliary_selection_id: Deterministic identity of the auxiliary sequence's selection,
            alongside ``auxiliary_sequence_artifact_id``. ``None`` under the same condition.
        trajectory_id: Identity of the persisted trajectory.
        estimator: Backend and configuration identity.
        calibration_identity: Hash of the calibration the estimator used.
        code_version: Code revision that produced the run, when known.
        reference_frame: Reference frame of the dynamic transform.
        body_frame: Body frame of the dynamic transform.
        clock_id: Clock domain of every timestamp in the run.
        start_time_ns: Time of the first pose, in nanoseconds of ``clock_id``.
        end_time_ns: Time of the last pose.
        pose_count: Poses in the trajectory.
        consumed_observation_count: Observations the backend considered.
        rejected_observation_count: Observations the backend rejected.
        gap_count: Recorded trajectory gaps.
        diagnostic_counts: Backend diagnostic events by code.
        preflight_status: Status of the geometry preflight, always ``"ready"``
            for a persisted run.
        debug_level: Debug evidence level that was requested.
        interpolation: Interpolation semantics lookups over this run use.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes
            the manifest, the README and ``debug/``.
    """

    run_id: StateEstimationRunId
    run_index: int
    sequence_name: str
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    auxiliary_sequence_artifact_id: SequenceArtifactId | None
    auxiliary_selection_id: str | None
    trajectory_id: TrajectoryId
    estimator: EstimatorProvenance
    calibration_identity: str | None
    code_version: str | None
    reference_frame: FrameId
    body_frame: FrameId
    clock_id: str
    start_time_ns: int
    end_time_ns: int
    pose_count: int
    consumed_observation_count: int
    rejected_observation_count: int
    gap_count: int
    diagnostic_counts: Mapping[str, int]
    preflight_status: str
    debug_level: str
    interpolation: Mapping[str, str]
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


class StateEstimationRunWriter:
    """Builds an immutable State Estimation run artifact on the local filesystem."""

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        run_id: StateEstimationRunId,
        run_index: int,
        debug_level: StateEstimationDebugLevel = StateEstimationDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            output_dir: The final directory of the artifact. The caller chooses it (in the
                runtime, ``<workspace>/<dataset>/<run>/state_estimation``); the writer
                computes no path, creates the directory atomically on finalization and
                refuses to replace one that exists.
            sequence_name: Name of the sequence the run processed.
            run_id: Identity of the run, supplied by the caller and never allocated here.
            run_index: Ordinal of this run among the caller's runs of the same sequence,
                supplied by the caller and recorded as given.
            debug_level: Amount of non-contractual debug evidence to persist.
        """
        self._sequence_name = sequence_name
        self._run_id = run_id
        self._run_index = run_index
        self._debug_level = debug_level
        self._final_dir = output_dir
        self._finalized = False

    def finalize(
        self, outcome: StateEstimationOutcome, *, runtime_s: float | None = None
    ) -> StateEstimationRunManifest:
        """Persist a completed run atomically.

        Args:
            outcome: The preflight report and the backend's result.
            runtime_s: Wall-clock time the backend took, when measured. It is
                recorded as a metric apart from every quality measure.

        Returns:
            The manifest of the finalized run.

        Raises:
            RunArtifactError: If already finalized, if the preflight was
                blocked (a blocked run is never persisted), or if a run already
                exists at the target path.
        """
        if self._finalized:
            raise RunArtifactError("writer already finalized")
        if outcome.preflight.status is PreflightStatus.BLOCKED:
            raise RunArtifactError(
                "cannot persist a run whose geometry preflight was blocked: "
                f"{[finding.code for finding in outcome.preflight.blockers]}"
            )

        try:
            with AtomicRunDirectory(self._final_dir) as run:
                trajectory = outcome.result.trajectory
                self._write_outputs(run, outcome)
                self._write_metrics(run, outcome, runtime_s)
                self._write_debug(run, outcome)
                run.publish(
                    manifest=self._manifest_record(outcome),
                    readme=_render_readme(self._run_id, self._run_index, trajectory),
                )
        except RunDirectoryError as error:
            raise RunArtifactError(str(error)) from error

        self._finalized = True
        return _load_manifest(self._final_dir)

    def _write_outputs(self, run: AtomicRunDirectory, outcome: StateEstimationOutcome) -> None:
        trajectory = outcome.result.trajectory
        run.write_text(_TRAJECTORY, _json(encode_trajectory_metadata(trajectory)))

        offset = 0
        pose_lines: list[bytes] = []
        index_lines: list[str] = []
        for pose in trajectory.poses:
            line = json.dumps(encode_pose_estimate(pose), sort_keys=True).encode("utf-8")
            pose_lines.append(line)
            index_lines.append(
                json.dumps(
                    {
                        "estimate_id": str(pose.estimate_id),
                        "timestamp_ns": pose.timestamp.total_nanoseconds(),
                        "byte_offset": offset,
                        "byte_length": len(line),
                    },
                    sort_keys=True,
                )
            )
            offset += len(line) + 1
        run.write_bytes(_POSES, b"\n".join(pose_lines) + b"\n")
        run.write_text(_POSE_INDEX, "\n".join(index_lines) + "\n")

        run.write_text(_FRAME_SUMMARY, _json(_encode_frame_summary(outcome.preflight)))
        run.write_text(_QUALITY, _json(_encode_quality(outcome)))

    def _write_metrics(
        self, run: AtomicRunDirectory, outcome: StateEstimationOutcome, runtime_s: float | None
    ) -> None:
        run.write_text(_PREFLIGHT, _json(_encode_preflight_report(outcome.preflight)))
        run.write_text(_MOTION, _json(_encode_motion(summarize_motion(outcome.result.trajectory))))
        if runtime_s is not None:
            run.write_text(_RUNTIME, _json({"runtime_s": runtime_s}))
        if outcome.result.diagnostics:
            run.write_text(
                _DIAGNOSTICS,
                _lines(_encode_diagnostic(item) for item in outcome.result.diagnostics),
            )

    def _write_debug(self, run: AtomicRunDirectory, outcome: StateEstimationOutcome) -> None:
        if self._debug_level is StateEstimationDebugLevel.NONE:
            return
        trajectory = outcome.result.trajectory
        deltas = motion_deltas(trajectory)
        run.write_text(
            "debug/pose-deltas.jsonl",
            _lines(
                {
                    "previous_estimate_id": str(d.previous_estimate_id),
                    "next_estimate_id": str(d.next_estimate_id),
                    "interval_ns": d.interval_ns,
                    "translation_delta_m": d.translation_delta_m,
                    "orientation_delta_rad": d.orientation_delta_rad,
                    "linear_speed_mps": d.linear_speed_mps,
                    "angular_speed_radps": d.angular_speed_radps,
                }
                for d in deltas
            ),
            contractual=False,
        )
        if trajectory.gaps:
            run.write_text(
                "debug/timestamp-gaps.jsonl",
                _lines(_encode_gap(gap) for gap in trajectory.gaps),
                contractual=False,
            )
        run.write_text(
            "debug/trajectory-xy.csv",
            _projection_csv(trajectory, second_axis=1),
            contractual=False,
        )
        run.write_text(
            "debug/trajectory-xz.csv",
            _projection_csv(trajectory, second_axis=2),
            contractual=False,
        )
        if self._debug_level is not StateEstimationDebugLevel.FULL:
            return
        checks = [
            {"kind": "transform", **_encode_check(check)}
            for check in outcome.preflight.transform_checks
        ] + [
            {"kind": "clock", "subject": c.subject, "passed": c.passed, "detail": c.detail}
            for c in outcome.preflight.clock_checks
        ]
        if checks:
            run.write_text("debug/preflight-checks.jsonl", _lines(checks), contractual=False)
        if outcome.result.diagnostics:
            run.write_text(
                "debug/backend-diagnostics/events.jsonl",
                _lines(_encode_diagnostic(item) for item in outcome.result.diagnostics),
                contractual=False,
            )

    def _manifest_record(self, outcome: StateEstimationOutcome) -> dict[str, Any]:
        trajectory = outcome.result.trajectory
        provenance = trajectory.provenance
        counts: dict[str, int] = {}
        for item in outcome.result.diagnostics:
            counts[item.code] = counts.get(item.code, 0) + 1
        return {
            "run_id": str(self._run_id),
            "run_index": self._run_index,
            "sequence_name": self._sequence_name,
            "sequence_artifact_id": str(provenance.sequence_artifact_id),
            "selection_id": provenance.selection_id,
            "auxiliary_sequence_artifact_id": (
                None
                if provenance.auxiliary_sequence_artifact_id is None
                else str(provenance.auxiliary_sequence_artifact_id)
            ),
            "auxiliary_selection_id": provenance.auxiliary_selection_id,
            "trajectory_id": str(trajectory.trajectory_id),
            "estimator": {
                "backend_id": provenance.estimator.backend_id,
                "backend_version": provenance.estimator.backend_version,
                "configuration_fingerprint": provenance.estimator.configuration_fingerprint,
            },
            "calibration_identity": provenance.calibration_identity,
            "code_version": provenance.code_version,
            "reference_frame": str(trajectory.reference_frame),
            "body_frame": str(trajectory.body_frame),
            "clock_id": trajectory.poses[0].timestamp.clock_id,
            "start_time_ns": trajectory.time_bounds.start.total_nanoseconds(),
            "end_time_ns": trajectory.time_bounds.end.total_nanoseconds(),
            "pose_count": len(trajectory.poses),
            "consumed_observation_count": outcome.result.consumed_observation_count,
            "rejected_observation_count": outcome.result.rejected_observation_count,
            "gap_count": len(trajectory.gaps),
            "diagnostic_counts": counts,
            "preflight_status": outcome.preflight.status.value,
            "debug_level": self._debug_level.value,
            "interpolation": dict(_INTERPOLATION),
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }


class StateEstimationRunReader:
    """Reads a finalized State Estimation run from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file
                outside it is needed.

        Raises:
            IncompleteRunArtifactError: If ``manifest.json`` is missing.
            RunArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = _load_manifest(run_dir)
        self._index: dict[PoseEstimateId, tuple[int, int, int]] | None = None

    @property
    def manifest(self) -> StateEstimationRunManifest:
        """The run's manifest."""
        return self._manifest

    def trajectory(self) -> Trajectory:
        """Load the whole trajectory.

        Returns:
            The trajectory, revalidated against the trajectory contract.
        """
        metadata = json.loads((self._root / _TRAJECTORY).read_text(encoding="utf-8"))
        poses = [
            decode_pose_estimate(json.loads(line))
            for line in (self._root / _POSES).read_text(encoding="utf-8").splitlines()
            if line
        ]
        return decode_trajectory(metadata, poses)

    def pose(self, estimate_id: PoseEstimateId) -> PoseEstimate:
        """Read one pose by identity without loading the others.

        Args:
            estimate_id: Identity of the pose.

        Returns:
            The pose.

        Raises:
            RunArtifactError: If the run has no such pose.
        """
        entry = self._load_index().get(estimate_id)
        if entry is None:
            raise RunArtifactError(f"unknown estimate_id in this run: {estimate_id!r}")
        _, offset, length = entry
        with (self._root / _POSES).open("rb") as handle:
            handle.seek(offset)
            return decode_pose_estimate(json.loads(handle.read(length)))

    def pose_at(self, timestamp: SourceTimestamp) -> PoseEstimate | None:
        """Read the pose at exactly this timestamp, if the run has one.

        Args:
            timestamp: The requested time, in the run's clock domain.

        Returns:
            The pose, or ``None`` when no pose has that timestamp. Use
            :class:`~contextmap.state_estimation.TrajectoryLookup` for
            nearest or interpolated lookups.

        Raises:
            ClockDomainMismatchError: If ``timestamp`` is in another clock domain.
        """
        if timestamp.clock_id != self._manifest.clock_id:
            raise ClockDomainMismatchError(
                f"timestamp is in clock domain {timestamp.clock_id!r} but the run uses "
                f"{self._manifest.clock_id!r}"
            )
        wanted = timestamp.total_nanoseconds()
        for estimate_id, (timestamp_ns, _, _) in self._load_index().items():
            if timestamp_ns == wanted:
                return self.pose(estimate_id)
        return None

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``outputs/`` or ``metrics/``.

        Args:
            relative_path: For example ``"outputs/quality.json"``.

        Returns:
            The parsed record.

        Raises:
            RunArtifactError: If the path is not a JSON record of a contractual
                directory; ``debug/`` is never a valid source.
        """
        if not relative_path.endswith(".json") or not relative_path.startswith(
            ("outputs/", "metrics/")
        ):
            raise RunArtifactError(f"not a contractual JSON record: {relative_path!r}")
        record: dict[str, Any] = json.loads(
            (self._root / relative_path).read_text(encoding="utf-8")
        )
        return record

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        return check_file_inventory(self._root, self._manifest.file_inventory)

    def _load_index(self) -> dict[PoseEstimateId, tuple[int, int, int]]:
        if self._index is None:
            index: dict[PoseEstimateId, tuple[int, int, int]] = {}
            text = (self._root / _POSE_INDEX).read_text(encoding="utf-8")
            for line in text.splitlines():
                record = json.loads(line)
                index[PoseEstimateId(record["estimate_id"])] = (
                    record["timestamp_ns"],
                    record["byte_offset"],
                    record["byte_length"],
                )
            self._index = index
        return self._index


def _load_manifest(run_dir: Path) -> StateEstimationRunManifest:
    manifest_path = run_dir / _MANIFEST
    if not manifest_path.is_file():
        raise IncompleteRunArtifactError(f"missing {_MANIFEST} in {run_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RunArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}"
        )
    estimator = raw["estimator"]
    return StateEstimationRunManifest(
        run_id=StateEstimationRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        sequence_artifact_id=SequenceArtifactId(raw["sequence_artifact_id"]),
        selection_id=raw["selection_id"],
        auxiliary_sequence_artifact_id=(
            None
            if raw.get("auxiliary_sequence_artifact_id") is None
            else SequenceArtifactId(raw["auxiliary_sequence_artifact_id"])
        ),
        auxiliary_selection_id=raw.get("auxiliary_selection_id"),
        trajectory_id=TrajectoryId(raw["trajectory_id"]),
        estimator=EstimatorProvenance(
            backend_id=estimator["backend_id"],
            backend_version=estimator["backend_version"],
            configuration_fingerprint=estimator["configuration_fingerprint"],
        ),
        calibration_identity=raw["calibration_identity"],
        code_version=raw["code_version"],
        reference_frame=FrameId(raw["reference_frame"]),
        body_frame=FrameId(raw["body_frame"]),
        clock_id=raw["clock_id"],
        start_time_ns=raw["start_time_ns"],
        end_time_ns=raw["end_time_ns"],
        pose_count=raw["pose_count"],
        consumed_observation_count=raw["consumed_observation_count"],
        rejected_observation_count=raw["rejected_observation_count"],
        gap_count=raw["gap_count"],
        diagnostic_counts=dict(raw["diagnostic_counts"]),
        preflight_status=raw["preflight_status"],
        debug_level=raw["debug_level"],
        interpolation=dict(raw["interpolation"]),
        schema_version=raw["schema_version"],
        created_at=raw["created_at"],
        file_inventory=tuple(
            FileEntry(
                path=entry["path"],
                size_bytes=entry["size_bytes"],
                content_hash=entry["content_hash"],
            )
            for entry in raw["file_inventory"]
        ),
    )


def _json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def _lines(records: Any) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def _encode_gap(gap: Any) -> dict[str, Any]:
    return {
        "previous_estimate_id": str(gap.previous_estimate_id),
        "next_estimate_id": str(gap.next_estimate_id),
        "duration_ns": gap.duration_ns,
    }


def _encode_diagnostic(diagnostic: EstimationDiagnostic) -> dict[str, Any]:
    return {
        "severity": diagnostic.severity.value,
        "code": diagnostic.code,
        "message": diagnostic.message,
        "observation_id": None
        if diagnostic.observation_id is None
        else str(diagnostic.observation_id),
    }


def _encode_check(check: Any) -> dict[str, Any]:
    return {
        "subject": check.subject,
        "check": check.check,
        "passed": check.passed,
        "detail": check.detail,
    }


def _encode_frame_summary(report: GeometryPreflightReport) -> dict[str, Any]:
    graph = report.frame_graph
    return {
        "dynamic": {
            "reference_frame": str(graph.reference_frame),
            "body_frame": str(graph.body_frame),
        },
        "static_frames": [str(frame) for frame in graph.static_frames],
        "static_edges": [[str(parent), str(child)] for parent, child in graph.static_edges],
        "component_count": graph.component_count,
        "calibration_identity": report.calibration_identity,
        "conventions": dict(report.conventions),
    }


def _encode_quality(outcome: StateEstimationOutcome) -> dict[str, Any]:
    trajectory = outcome.result.trajectory
    summary = trajectory.quality_summary()
    duration_s = summary.duration_ns / _NANOSECONDS_PER_SECOND
    return {
        "pose_count": summary.pose_count,
        "duration_ns": summary.duration_ns,
        "sample_rate_hz": (summary.pose_count - 1) / duration_s if duration_s > 0 else None,
        "min_interval_ns": summary.min_interval_ns,
        "median_interval_ns": summary.median_interval_ns,
        "max_interval_ns": summary.max_interval_ns,
        "gap_count": summary.gap_count,
        "gaps": [_encode_gap(gap) for gap in trajectory.gaps],
        "degraded_pose_count": summary.degraded_pose_count,
        "pose_with_covariance_count": summary.pose_with_covariance_count,
        "consumed_observation_count": outcome.result.consumed_observation_count,
        "rejected_observation_count": outcome.result.rejected_observation_count,
    }


def _encode_distribution(summary: DistributionSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "count": summary.count,
        "minimum": summary.minimum,
        "median": summary.median,
        "p95": summary.p95,
        "maximum": summary.maximum,
    }


def _encode_motion(motion: MotionSummary) -> dict[str, Any]:
    return {
        "translation_delta_m": _encode_distribution(motion.translation_delta_m),
        "orientation_delta_rad": _encode_distribution(motion.orientation_delta_rad),
        "linear_speed_mps": _encode_distribution(motion.linear_speed_mps),
        "angular_speed_radps": _encode_distribution(motion.angular_speed_radps),
    }


def _encode_preflight_report(report: GeometryPreflightReport) -> dict[str, Any]:
    return {
        "status": report.status.value,
        "required_inputs": sorted(report.required_inputs),
        "available_inputs": sorted(report.available_inputs),
        "calibration_identity": report.calibration_identity,
        "frame_graph": _encode_frame_summary(report),
        "transform_checks": [_encode_check(check) for check in report.transform_checks],
        "clock_checks": [
            {"subject": c.subject, "passed": c.passed, "detail": c.detail}
            for c in report.clock_checks
        ],
        "blockers": [{"code": f.code, "message": f.message} for f in report.blockers],
        "warnings": [{"code": f.code, "message": f.message} for f in report.warnings],
        "downstream": [
            {
                "capability": item.capability,
                "ready": item.ready,
                "missing": [{"code": f.code, "message": f.message} for f in item.missing],
            }
            for item in report.downstream
        ],
        "conventions": dict(report.conventions),
    }


def _projection_csv(trajectory: Trajectory, *, second_axis: int) -> str:
    header = "timestamp_ns,x_m," + ("y_m" if second_axis == 1 else "z_m")
    rows = [
        f"{pose.timestamp.total_nanoseconds()},{pose.translation_m[0]!r},"
        f"{pose.translation_m[second_axis]!r}"
        for pose in trajectory.poses
    ]
    return "\n".join([header, *rows]) + "\n"


def _render_readme(run_id: StateEstimationRunId, run_index: int, trajectory: Trajectory) -> str:
    provenance = trajectory.provenance
    auxiliary_line = (
        f"- Auxiliary sequence artifact: `{provenance.auxiliary_sequence_artifact_id}` "
        f"(selection `{provenance.auxiliary_selection_id}`)\n"
        if provenance.auxiliary_sequence_artifact_id is not None
        else ""
    )
    return (
        f"# State estimation run {run_index:04d}\n"
        "\n"
        f"- Run ID: `{run_id}`\n"
        f"- Sequence artifact: `{provenance.sequence_artifact_id}`\n"
        f"- Selection: `{provenance.selection_id}`\n"
        f"{auxiliary_line}"
        f"- Backend: `{provenance.estimator.backend_id}` "
        f"(version `{provenance.estimator.backend_version}`)\n"
        f"- Frames: `{trajectory.reference_frame}` <- `{trajectory.body_frame}` (T_parent_child)\n"
        f"- Poses: {len(trajectory.poses)}\n"
        "\n"
        "Contractual data is in `outputs/` and `metrics/`; `debug/` is human evidence and no "
        "downstream stage may depend on it.\n"
    )

"""Canonical sequence artifact: local, immutable, self-describing persistence.

A ``SequenceArtifact`` is the reusable, reopenable result of ingesting a
source (ROS 1 bag, ROS 2 bag, dataset) once. It is not the final
``ContextMapArtifact``; it is the normalized sensor sequence that later
pipeline stages read instead of reopening the original source. See
``src/contextmap/ingestion/docs/artifact.md`` for the on-disk layout,
schema version history, and the design trade-offs made for v0 (notably a
JSON Lines index instead of a columnar format).

Writing is atomic and streaming: :class:`SequenceArtifactWriter` writes each
observation's payload and index line to a temporary sibling directory as it is
added, so memory use does not grow with payload size, and only makes the
artifact visible under its final path after an internal integrity check
succeeds. An interrupted write can never be mistaken for a complete artifact.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any, NewType
from uuid import uuid4

from contextmap.ingestion.calibration import (
    CalibrationSet,
    decode_calibration_set,
    encode_calibration_set,
    ensure_valid_calibration_set,
)
from contextmap.ingestion.diagnostics import (
    SequenceDiagnostics,
    build_frame_graph_diagnostics,
    decode_diagnostics_summary,
    decode_dropped_event,
    decode_frame_graph,
    decode_synchronization_decision,
    encode_diagnostics,
    encode_dropped_event,
    encode_frame_graph,
    encode_synchronization_decision,
    summarize_observations,
)
from contextmap.ingestion.models import (
    MODALITY_NAMES,
    CalibrationReferenceId,
    ExternalPoseMeasurement,
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    SensorId,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
    observation_modality,
)
from contextmap.ingestion.sequence_provenance import (
    SequenceProvenance,
    decode_provenance,
    encode_provenance,
)
from contextmap.ingestion.synchronization import DroppedEvent, SynchronizationDiagnostics
from contextmap.shared import SourceTimestamp

SCHEMA_VERSION = "0.2.0"
"""Sequence artifact schema version written and understood by this module."""

SequenceArtifactId = NewType("SequenceArtifactId", str)
"""Identity of a persisted canonical sequence artifact."""

_MANIFEST_FILENAME = "manifest.json"
_INDEX_FILENAME = "index.jsonl"
_CALIBRATION_FILENAME = "calibration/calibration.json"
_PROVENANCE_FILENAME = "provenance/provenance.json"
_DIAGNOSTICS_SUMMARY_FILENAME = "diagnostics/summary.json"
_DIAGNOSTICS_WARNINGS_FILENAME = "diagnostics/warnings.jsonl"
_DIAGNOSTICS_SYNCHRONIZATION_FILENAME = "diagnostics/synchronization.jsonl"
_DIAGNOSTICS_DROPPED_EVENTS_FILENAME = "diagnostics/dropped-events.jsonl"
_DIAGNOSTICS_FRAME_GRAPH_FILENAME = "diagnostics/frame-graph.json"
_MODALITY_COUNTS_TEMPLATE: Mapping[str, int] = dict.fromkeys(MODALITY_NAMES, 0)


class SequenceArtifactError(Exception):
    """Base class for sequence artifact read/write failures."""


class IncompleteSequenceArtifactError(SequenceArtifactError):
    """Raised when a directory does not contain a complete, valid artifact."""


@dataclass(frozen=True)
class SequenceArtifactFileEntry:
    """One entry in a sequence artifact's file inventory.

    Attributes:
        path: Path relative to the artifact root, using ``/`` separators.
        size_bytes: Size of the file in bytes.
        content_hash: Content hash as ``"sha256:<hex digest>"``.
    """

    path: str
    size_bytes: int
    content_hash: str


@dataclass(frozen=True)
class SequenceArtifactManifest:
    """Authoritative metadata for a canonical sequence artifact.

    Full provenance/content-identity semantics (detecting equivalent
    re-ingestion, source content hashing, configuration hashing) are added
    by the sequence provenance and integrity issue in this milestone; this
    manifest only carries the structural identity of the artifact itself.

    Attributes:
        artifact_id: Identity of this artifact.
        sequence_name: Name of the sequence this artifact belongs to.
        schema_version: Sequence artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        observation_counts: Number of observations per modality
            ("image", "lidar", "imu", "external_pose").
        file_inventory: Every payload/index file this artifact contains,
            excluding ``manifest.json`` itself.
    """

    artifact_id: SequenceArtifactId
    sequence_name: str
    schema_version: str
    created_at: str
    observation_counts: Mapping[str, int]
    file_inventory: Sequence[SequenceArtifactFileEntry]


class SequenceArtifactWriter:
    """Builds an immutable canonical sequence artifact on the local filesystem.

    Payloads and index lines are streamed to a temporary directory by
    :meth:`add_observation`; only observation metadata (payload bytes
    excluded) is kept in memory, for the diagnostics summary written by
    :meth:`finalize`. A writer that is not finalized leaves its temporary
    directory behind until :meth:`abort` is called, so prefer the context
    manager form, which aborts on any exit path that did not finalize.

    Example:
        with SequenceArtifactWriter(
            output_dir=Path("workspace/corridor-02/run-0001/ingestion"),
            sequence_name="corridor-02",
            artifact_id=SequenceArtifactId("sequence-0001"),
        ) as writer:
            writer.add_observation(image_observation)
            writer.add_observation(lidar_observation)
            manifest = writer.finalize()
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        artifact_id: SequenceArtifactId,
    ) -> None:
        """Create a writer for a new sequence artifact.

        No filesystem access happens until the first observation is added
        or :meth:`finalize` is called.

        Args:
            output_dir: The final directory of the artifact. The caller chooses
                it (in the runtime, ``<workspace>/<dataset>/<run>/ingestion``);
                the writer computes no path, builds the artifact in a temporary
                sibling of ``output_dir`` and refuses to replace a directory
                that already exists.
            sequence_name: Name of the sequence this artifact belongs to.
            artifact_id: Identity of the artifact, supplied by the caller and
                recorded as given; the writer never allocates one.
        """
        self._sequence_name = sequence_name
        self._artifact_id = artifact_id
        self._final_dir = output_dir
        self._tmp_dir = output_dir.parent / f".tmp-{output_dir.name}-{uuid4().hex[:8]}"
        self._index_handle: IO[bytes] | None = None
        self._index_hash = hashlib.sha256()
        self._index_size = 0
        self._payload_entries: list[SequenceArtifactFileEntry] = []
        self._counts = dict(_MODALITY_COUNTS_TEMPLATE)
        self._metadata: list[SourceObservation] = []
        self._seen_observation_ids: set[str] = set()
        self._calibration: CalibrationSet | None = None
        self._provenance: SequenceProvenance | None = None
        self._diagnostic_warnings: tuple[str, ...] | None = None
        self._synchronization_diagnostics: SynchronizationDiagnostics | None = None
        self._closed_by: str | None = None
        self._tmp_dir_removed = False

    def __enter__(self) -> SequenceArtifactWriter:
        """Return the writer for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Abort the writer unless it was already finalized.

        When the block is already failing, a cleanup error is attached to that
        exception as a note instead of replacing it.
        """
        if exc_value is None:
            self.abort()
        else:
            self._abort_after_failure(exc_value)

    def set_diagnostics(
        self,
        *,
        warnings: Sequence[str] = (),
        synchronization: SynchronizationDiagnostics | None = None,
    ) -> None:
        """Enable writing diagnostics for this sequence.

        The summary itself (per-modality counts/time-range, image
        resolutions, point-cloud field layouts) is computed automatically
        from the observations added to this writer; only warning text is
        supplied by the caller.

        Args:
            warnings: Adapter/validation/synchronization warnings to
                record alongside the summary, as human-readable text.
            synchronization: Structured synchronization decisions and dropped
                events to persist alongside the summary.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize` or :meth:`abort`.
        """
        self._require_open("set diagnostics")
        self._diagnostic_warnings = tuple(warnings)
        self._synchronization_diagnostics = synchronization

    def set_provenance(self, provenance: SequenceProvenance) -> None:
        """Attach the sequence's provenance/content-identity metadata.

        Args:
            provenance: Provenance to persist alongside the artifact.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize` or :meth:`abort`.
        """
        self._require_open("set provenance")
        self._provenance = provenance

    def set_calibration(self, calibration_set: CalibrationSet) -> None:
        """Attach the sequence's calibration set, written by :meth:`finalize`.

        Args:
            calibration_set: Calibration and coordinate-frame inventory for
                this sequence.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize` or :meth:`abort`.
            CalibrationError: If ``calibration_set`` is invalid; see
                :func:`~contextmap.ingestion.calibration.validate_calibration_set`.
        """
        self._require_open("set calibration")
        ensure_valid_calibration_set(calibration_set)
        self._calibration = calibration_set

    def add_observation(self, observation: SourceObservation) -> None:
        """Write an observation's payload and index line to the temporary artifact.

        The payload is persisted immediately and not retained, so the
        caller may release ``observation`` as soon as this returns. An I/O
        failure while writing aborts the writer, since the partial
        temporary artifact can no longer be trusted.

        Args:
            observation: Any canonical source observation.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize` or
                :meth:`abort`, or if ``observation.observation_id`` was
                already added.
        """
        self._require_open("add observations")
        observation_id = str(observation.observation_id)
        if observation_id in self._seen_observation_ids:
            raise SequenceArtifactError(f"duplicate observation_id: {observation_id!r}")
        record, payload = _encode_observation(observation)
        modality = record["modality"]
        assert isinstance(modality, str)

        try:
            index_handle = self._open_temporary_artifact()
            if payload is not None:
                relative_path, data = payload
                absolute_path = self._tmp_dir / relative_path
                absolute_path.parent.mkdir(parents=True, exist_ok=True)
                absolute_path.write_bytes(data)
                self._payload_entries.append(_file_entry(relative_path, data))
            line = f"{json.dumps(record, sort_keys=True)}\n".encode()
            index_handle.write(line)
        except BaseException as error:
            self._abort_after_failure(error)
            raise

        self._index_hash.update(line)
        self._index_size += len(line)
        self._counts[modality] += 1
        self._seen_observation_ids.add(observation_id)
        self._metadata.append(_without_payload(observation))

    def abort(self) -> None:
        """Discard the temporary artifact and close the writer.

        The writer is closed immediately, even if cleanup then fails. Errors
        from flushing the index are ignored, since that data is being
        discarded. A failure to remove the temporary directory is raised so
        it is never silent; calling :meth:`abort` again retries the removal.
        It has no effect on a finalized artifact and is a no-op once the
        temporary directory is gone.

        Raises:
            OSError: If the temporary directory could not be removed.
        """
        if self._closed_by == "finalize()":
            return
        self._closed_by = "abort()"
        handle, self._index_handle = self._index_handle, None
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()
        if not self._tmp_dir_removed:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(self._tmp_dir)
            self._tmp_dir_removed = True

    def finalize(self) -> SequenceArtifactManifest:
        """Complete the temporary artifact and publish it atomically.

        The artifact is built under a temporary sibling directory and only
        moved to its final path after an internal integrity check succeeds,
        so a process interrupted mid-write never leaves a directory that
        looks like a complete artifact at the final path. Any failure
        aborts the writer and removes the temporary directory.

        Returns:
            The manifest of the finalized artifact.

        Raises:
            SequenceArtifactError: If already finalized or aborted, if an
                artifact already exists at the target path, or if writing
                fails.
        """
        if self._closed_by == "finalize()":
            raise SequenceArtifactError("writer already finalized")
        if self._closed_by is not None:
            raise SequenceArtifactError(f"writer closed by {self._closed_by}")

        try:
            if self._final_dir.exists():
                raise SequenceArtifactError(f"sequence artifact already exists: {self._final_dir}")
            self._open_temporary_artifact()
            manifest = self._write_contents()
            problems = _check_file_inventory(
                self._tmp_dir, manifest
            ) + _check_index_cross_references(self._tmp_dir, manifest)
            if problems:
                raise SequenceArtifactError(
                    f"internal consistency check failed before finalize: {problems}"
                )
            self._tmp_dir.rename(self._final_dir)
        except BaseException as error:
            self._abort_after_failure(error)
            raise

        self._closed_by = "finalize()"
        return manifest

    def _abort_after_failure(self, error: BaseException) -> None:
        """Abort without letting a cleanup failure replace ``error``, the real cause."""
        try:
            self.abort()
        except OSError as cleanup_error:
            error.add_note(
                f"could not remove temporary artifact {self._tmp_dir}: {cleanup_error}; "
                "call abort() to retry"
            )

    def _require_open(self, action: str) -> None:
        if self._closed_by is not None:
            raise SequenceArtifactError(f"cannot {action} after {self._closed_by}")

    def _open_temporary_artifact(self) -> IO[bytes]:
        if self._index_handle is None:
            self._tmp_dir.mkdir(parents=True, exist_ok=False)
            self._index_handle = (self._tmp_dir / _INDEX_FILENAME).open("wb")
        return self._index_handle

    def _write_contents(self) -> SequenceArtifactManifest:
        assert self._index_handle is not None
        self._index_handle.close()
        self._index_handle = None
        file_entries = [
            *self._payload_entries,
            SequenceArtifactFileEntry(
                path=_INDEX_FILENAME,
                size_bytes=self._index_size,
                content_hash=f"sha256:{self._index_hash.hexdigest()}",
            ),
        ]

        if self._calibration is not None:
            calibration_content = json.dumps(
                encode_calibration_set(self._calibration), indent=2, sort_keys=True
            )
            calibration_path = self._tmp_dir / _CALIBRATION_FILENAME
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            calibration_path.write_text(calibration_content, encoding="utf-8")
            file_entries.append(
                _file_entry(_CALIBRATION_FILENAME, calibration_content.encode("utf-8"))
            )

        if self._provenance is not None:
            provenance_content = json.dumps(
                encode_provenance(self._provenance), indent=2, sort_keys=True
            )
            provenance_path = self._tmp_dir / _PROVENANCE_FILENAME
            provenance_path.parent.mkdir(parents=True, exist_ok=True)
            provenance_path.write_text(provenance_content, encoding="utf-8")
            file_entries.append(
                _file_entry(_PROVENANCE_FILENAME, provenance_content.encode("utf-8"))
            )

        if self._diagnostic_warnings is not None:
            summary = summarize_observations(
                self._metadata,
                warning_count=len(self._diagnostic_warnings),
                synchronization=self._synchronization_diagnostics,
                calibration=self._calibration,
            )
            summary_content = json.dumps(
                encode_diagnostics(SequenceDiagnostics(summary=summary, warnings=())),
                indent=2,
                sort_keys=True,
            )
            summary_path = self._tmp_dir / _DIAGNOSTICS_SUMMARY_FILENAME
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary_content, encoding="utf-8")
            file_entries.append(
                _file_entry(_DIAGNOSTICS_SUMMARY_FILENAME, summary_content.encode("utf-8"))
            )

            warnings_content = "".join(
                f"{json.dumps({'warning': warning})}\n" for warning in self._diagnostic_warnings
            )
            warnings_path = self._tmp_dir / _DIAGNOSTICS_WARNINGS_FILENAME
            warnings_path.parent.mkdir(parents=True, exist_ok=True)
            warnings_path.write_text(warnings_content, encoding="utf-8")
            file_entries.append(
                _file_entry(_DIAGNOSTICS_WARNINGS_FILENAME, warnings_content.encode("utf-8"))
            )

            if self._synchronization_diagnostics is not None:
                synchronization_content = "".join(
                    f"{json.dumps(encode_synchronization_decision(decision), sort_keys=True)}\n"
                    for decision in self._synchronization_diagnostics.decisions
                )
                synchronization_path = self._tmp_dir / _DIAGNOSTICS_SYNCHRONIZATION_FILENAME
                synchronization_path.write_text(synchronization_content, encoding="utf-8")
                file_entries.append(
                    _file_entry(
                        _DIAGNOSTICS_SYNCHRONIZATION_FILENAME,
                        synchronization_content.encode("utf-8"),
                    )
                )

                dropped_content = "".join(
                    f"{json.dumps(encode_dropped_event(event), sort_keys=True)}\n"
                    for event in self._synchronization_diagnostics.dropped_events
                )
                dropped_path = self._tmp_dir / _DIAGNOSTICS_DROPPED_EVENTS_FILENAME
                dropped_path.write_text(dropped_content, encoding="utf-8")
                file_entries.append(
                    _file_entry(
                        _DIAGNOSTICS_DROPPED_EVENTS_FILENAME,
                        dropped_content.encode("utf-8"),
                    )
                )

            if self._calibration is not None:
                frame_graph_content = json.dumps(
                    encode_frame_graph(build_frame_graph_diagnostics(self._calibration)),
                    indent=2,
                    sort_keys=True,
                )
                frame_graph_path = self._tmp_dir / _DIAGNOSTICS_FRAME_GRAPH_FILENAME
                frame_graph_path.write_text(frame_graph_content, encoding="utf-8")
                file_entries.append(
                    _file_entry(
                        _DIAGNOSTICS_FRAME_GRAPH_FILENAME,
                        frame_graph_content.encode("utf-8"),
                    )
                )

        manifest = SequenceArtifactManifest(
            artifact_id=self._artifact_id,
            sequence_name=self._sequence_name,
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            observation_counts=dict(self._counts),
            file_inventory=tuple(sorted(file_entries, key=lambda entry: entry.path)),
        )
        manifest_path = self._tmp_dir / _MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest


class SequenceArtifactReader:
    """Reads a finalized canonical sequence artifact from the local filesystem."""

    def __init__(self, artifact_dir: Path) -> None:
        """Open a sequence artifact for reading.

        Args:
            artifact_dir: Path to the artifact's directory.

        Raises:
            IncompleteSequenceArtifactError: If ``manifest.json`` is missing.
            SequenceArtifactError: If the manifest's schema version is not
                understood by this module.
        """
        self._root = artifact_dir
        self._manifest = _load_manifest(artifact_dir)

    @property
    def manifest(self) -> SequenceArtifactManifest:
        """The artifact's manifest."""
        return self._manifest

    def verify_integrity(self) -> list[str]:
        """Check the artifact's file inventory and internal cross-references.

        Detects: missing files referenced by the manifest, size/content
        hash mismatches, and index.jsonl records whose ``payload_path``
        does not resolve to a file in the manifest's file inventory
        (an invalid index-to-payload cross-reference). An incompatible
        schema version is instead raised by :meth:`__init__`/:func:`_load_manifest`,
        since a reader that does not understand the schema cannot safely
        interpret anything else about the artifact.

        Returns:
            A list of human-readable problems; empty means no problem found.
        """
        return _check_file_inventory(self._root, self._manifest) + _check_index_cross_references(
            self._root, self._manifest
        )

    def read_provenance(self) -> SequenceProvenance | None:
        """Read this sequence's provenance/content-identity metadata, when written.

        Returns:
            The provenance, or ``None`` if the artifact was finalized
            without one (e.g. by a writer predating this contract).
        """
        provenance_path = self._root / _PROVENANCE_FILENAME
        if not provenance_path.is_file():
            return None
        return decode_provenance(json.loads(provenance_path.read_text(encoding="utf-8")))

    def read_diagnostics(self) -> SequenceDiagnostics | None:
        """Read this sequence's diagnostics (summary + warnings), when written.

        Returns:
            The diagnostics, or ``None`` if the artifact was finalized
            without calling
            :meth:`SequenceArtifactWriter.set_diagnostics`.
        """
        summary_path = self._root / _DIAGNOSTICS_SUMMARY_FILENAME
        if not summary_path.is_file():
            return None
        summary = decode_diagnostics_summary(json.loads(summary_path.read_text(encoding="utf-8")))

        warnings_path = self._root / _DIAGNOSTICS_WARNINGS_FILENAME
        warnings: list[str] = []
        if warnings_path.is_file():
            for line in warnings_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped:
                    warnings.append(json.loads(stripped)["warning"])

        synchronization_path = self._root / _DIAGNOSTICS_SYNCHRONIZATION_FILENAME
        decisions = (
            tuple(
                decode_synchronization_decision(json.loads(line))
                for line in synchronization_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if synchronization_path.is_file()
            else ()
        )

        dropped_path = self._root / _DIAGNOSTICS_DROPPED_EVENTS_FILENAME
        dropped_events: tuple[DroppedEvent, ...] = ()
        if dropped_path.is_file():
            dropped_records = [
                json.loads(line)
                for line in dropped_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if dropped_records:
                observations_by_id = {
                    str(observation.observation_id): observation
                    for observation in self.list_observations()
                }
                dropped_events = tuple(
                    decode_dropped_event(record, observations_by_id) for record in dropped_records
                )
        synchronization = (
            SynchronizationDiagnostics(
                dropped_events=dropped_events,
                decisions=decisions,
            )
            if synchronization_path.is_file() or dropped_path.is_file()
            else None
        )

        frame_graph_path = self._root / _DIAGNOSTICS_FRAME_GRAPH_FILENAME
        frame_graph = (
            decode_frame_graph(json.loads(frame_graph_path.read_text(encoding="utf-8")))
            if frame_graph_path.is_file()
            else None
        )

        return SequenceDiagnostics(
            summary=summary,
            warnings=tuple(warnings),
            synchronization=synchronization,
            frame_graph=frame_graph,
        )

    def read_calibration(self) -> CalibrationSet | None:
        """Read this sequence's calibration set, when one was written.

        Returns:
            The calibration set, or ``None`` if the artifact was finalized
            without one (e.g. by a writer predating the calibration
            contract, or a source with no calibration available).
        """
        calibration_path = self._root / _CALIBRATION_FILENAME
        if not calibration_path.is_file():
            return None
        return decode_calibration_set(json.loads(calibration_path.read_text(encoding="utf-8")))

    def list_observations(self) -> list[SourceObservation]:
        """Return every observation in the artifact, in index order.

        Returns:
            All observations, decoded from the index and their payload
            files.
        """
        return list(self._iter_observations())

    def get_observation(self, observation_id: SourceObservationId) -> SourceObservation:
        """Return a single observation by identity.

        Args:
            observation_id: Identity of the observation to look up.

        Returns:
            The matching observation.

        Raises:
            SequenceArtifactError: If no observation has this identity.
        """
        for observation in self._iter_observations():
            if observation.observation_id == observation_id:
                return observation
        raise SequenceArtifactError(f"observation not found: {observation_id!r}")

    def _iter_observations(self) -> Iterator[SourceObservation]:
        index_path = self._root / _INDEX_FILENAME
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                yield _decode_observation(json.loads(stripped), self._root)


def _encode_observation(
    observation: SourceObservation,
) -> tuple[dict[str, Any], tuple[str, bytes] | None]:
    record: dict[str, Any] = {
        "observation_id": str(observation.observation_id),
        "sensor_id": str(observation.sensor_id),
        "frame_id": str(observation.frame_id),
        "timestamp": {
            "seconds": observation.timestamp.seconds,
            "nanoseconds": observation.timestamp.nanoseconds,
            "clock_id": observation.timestamp.clock_id,
        },
        "provenance": {
            "source_type": observation.provenance.source_type,
            "source_path": observation.provenance.source_path,
            "source_topic": observation.provenance.source_topic,
            "source_message_index": observation.provenance.source_message_index,
            "raw_metadata": dict(observation.provenance.raw_metadata),
        },
        "calibration_id": (
            str(observation.calibration_id) if observation.calibration_id is not None else None
        ),
        "modality": observation_modality(observation),
    }

    if isinstance(observation, ImageObservation):
        record["width"] = observation.width
        record["height"] = observation.height
        record["encoding"] = observation.encoding.value
        payload_path = f"rgb/{observation.observation_id}.bin"
        record["payload_path"] = payload_path
        return record, (payload_path, observation.data)

    if isinstance(observation, LidarObservation):
        record["point_count"] = observation.point_count
        record["point_step_bytes"] = observation.point_step_bytes
        record["is_dense"] = observation.is_dense
        record["fields"] = [
            {
                "name": item.name,
                "offset_bytes": item.offset_bytes,
                "data_type": item.data_type.value,
                "count": item.count,
            }
            for item in observation.fields
        ]
        payload_path = f"pointcloud/{observation.observation_id}.bin"
        record["payload_path"] = payload_path
        return record, (payload_path, observation.data)

    if isinstance(observation, ImuObservation):
        record["linear_acceleration"] = (
            list(observation.linear_acceleration)
            if observation.linear_acceleration is not None
            else None
        )
        record["angular_velocity"] = (
            list(observation.angular_velocity) if observation.angular_velocity is not None else None
        )
        record["orientation"] = (
            list(observation.orientation) if observation.orientation is not None else None
        )
        record["linear_acceleration_covariance"] = _encode_optional_tuple(
            observation.linear_acceleration_covariance
        )
        record["angular_velocity_covariance"] = _encode_optional_tuple(
            observation.angular_velocity_covariance
        )
        record["orientation_covariance"] = _encode_optional_tuple(
            observation.orientation_covariance
        )
        return record, None

    if isinstance(observation, ExternalPoseMeasurement):
        record["parent_frame"] = str(observation.parent_frame)
        record["translation"] = list(observation.translation)
        record["orientation"] = list(observation.orientation)
        record["pose_covariance"] = _encode_optional_tuple(observation.pose_covariance)
        record["linear_velocity"] = _encode_optional_tuple(observation.linear_velocity)
        record["angular_velocity"] = _encode_optional_tuple(observation.angular_velocity)
        record["twist_covariance"] = _encode_optional_tuple(observation.twist_covariance)
        return record, None

    raise SequenceArtifactError(f"unsupported observation type: {type(observation)!r}")


def _decode_observation(record: dict[str, Any], root: Path) -> SourceObservation:
    common: dict[str, Any] = {
        "observation_id": SourceObservationId(record["observation_id"]),
        "sensor_id": SensorId(record["sensor_id"]),
        "frame_id": FrameId(record["frame_id"]),
        "timestamp": SourceTimestamp(**record["timestamp"]),
        "provenance": SourceProvenance(**record["provenance"]),
        "calibration_id": (
            CalibrationReferenceId(record["calibration_id"])
            if record["calibration_id"] is not None
            else None
        ),
    }

    modality = record["modality"]

    if modality == "image":
        data = (root / record["payload_path"]).read_bytes()
        return ImageObservation(
            **common,
            width=record["width"],
            height=record["height"],
            encoding=ImageEncoding(record["encoding"]),
            data=data,
        )

    if modality == "lidar":
        data = (root / record["payload_path"]).read_bytes()
        fields = tuple(
            PointFieldDescriptor(
                name=item["name"],
                offset_bytes=item["offset_bytes"],
                data_type=PointFieldDataType(item["data_type"]),
                count=item["count"],
            )
            for item in record["fields"]
        )
        return LidarObservation(
            **common,
            point_count=record["point_count"],
            point_step_bytes=record["point_step_bytes"],
            fields=fields,
            is_dense=record["is_dense"],
            data=data,
        )

    if modality == "imu":
        return ImuObservation(
            **common,
            linear_acceleration=(
                tuple(record["linear_acceleration"])
                if record["linear_acceleration"] is not None
                else None
            ),
            angular_velocity=(
                tuple(record["angular_velocity"])
                if record["angular_velocity"] is not None
                else None
            ),
            orientation=(
                tuple(record["orientation"]) if record["orientation"] is not None else None
            ),
            linear_acceleration_covariance=_decode_optional_tuple(
                record["linear_acceleration_covariance"]
            ),
            angular_velocity_covariance=_decode_optional_tuple(
                record["angular_velocity_covariance"]
            ),
            orientation_covariance=_decode_optional_tuple(record["orientation_covariance"]),
        )

    if modality == "external_pose":
        return ExternalPoseMeasurement(
            **common,
            parent_frame=FrameId(record["parent_frame"]),
            translation=tuple(record["translation"]),
            orientation=tuple(record["orientation"]),
            pose_covariance=_decode_optional_tuple(record["pose_covariance"]),
            linear_velocity=_decode_optional_vector3(record["linear_velocity"]),
            angular_velocity=_decode_optional_vector3(record["angular_velocity"]),
            twist_covariance=_decode_optional_tuple(record["twist_covariance"]),
        )

    raise SequenceArtifactError(f"unknown modality in index record: {modality!r}")


def _encode_optional_tuple(values: tuple[float, ...] | None) -> list[float] | None:
    """Encode an optional numeric tuple for JSON persistence."""
    return list(values) if values is not None else None


def _decode_optional_tuple(values: list[float] | None) -> tuple[float, ...] | None:
    """Decode an optional numeric tuple from JSON persistence."""
    return tuple(values) if values is not None else None


def _decode_optional_vector3(
    values: list[float] | None,
) -> tuple[float, float, float] | None:
    """Decode an optional three-element vector from JSON persistence."""
    if values is None:
        return None
    if len(values) != 3:
        raise SequenceArtifactError(f"expected a three-element vector, found {len(values)}")
    return values[0], values[1], values[2]


def _without_payload(observation: SourceObservation) -> SourceObservation:
    """Return ``observation`` minus its binary payload, keeping every metadata field."""
    if isinstance(observation, ImageObservation | LidarObservation):
        return replace(observation, data=b"")
    return observation


def _file_entry(relative_path: str, data: bytes) -> SequenceArtifactFileEntry:
    digest = hashlib.sha256(data).hexdigest()
    return SequenceArtifactFileEntry(
        path=relative_path, size_bytes=len(data), content_hash=f"sha256:{digest}"
    )


def _manifest_to_dict(manifest: SequenceArtifactManifest) -> dict[str, Any]:
    return {
        "artifact_id": str(manifest.artifact_id),
        "sequence_name": manifest.sequence_name,
        "schema_version": manifest.schema_version,
        "created_at": manifest.created_at,
        "observation_counts": dict(manifest.observation_counts),
        "file_inventory": [
            {"path": entry.path, "size_bytes": entry.size_bytes, "content_hash": entry.content_hash}
            for entry in manifest.file_inventory
        ],
    }


def _load_manifest(artifact_dir: Path) -> SequenceArtifactManifest:
    manifest_path = artifact_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise IncompleteSequenceArtifactError(f"missing {_MANIFEST_FILENAME} in {artifact_dir}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise SequenceArtifactError(
            f"unsupported sequence artifact schema_version: {schema_version!r}"
        )

    return SequenceArtifactManifest(
        artifact_id=SequenceArtifactId(raw["artifact_id"]),
        sequence_name=raw["sequence_name"],
        schema_version=schema_version,
        created_at=raw["created_at"],
        observation_counts=dict(raw["observation_counts"]),
        file_inventory=tuple(
            SequenceArtifactFileEntry(
                path=entry["path"],
                size_bytes=entry["size_bytes"],
                content_hash=entry["content_hash"],
            )
            for entry in raw["file_inventory"]
        ),
    )


def _check_file_inventory(root: Path, manifest: SequenceArtifactManifest) -> list[str]:
    problems: list[str] = []
    for entry in manifest.file_inventory:
        file_path = root / entry.path
        if not file_path.is_file():
            problems.append(f"missing file referenced by manifest: {entry.path}")
            continue
        data = file_path.read_bytes()
        if len(data) != entry.size_bytes:
            problems.append(
                f"size mismatch for {entry.path}: expected {entry.size_bytes}, found {len(data)}"
            )
            continue
        digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
        if digest != entry.content_hash:
            problems.append(f"content hash mismatch for {entry.path}")
    return problems


def _check_index_cross_references(root: Path, manifest: SequenceArtifactManifest) -> list[str]:
    problems: list[str] = []
    known_paths = {entry.path for entry in manifest.file_inventory}
    index_path = root / _INDEX_FILENAME
    if not index_path.is_file():
        return problems  # already reported by _check_file_inventory

    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            payload_path = record.get("payload_path")
            if payload_path is not None and payload_path not in known_paths:
                problems.append(
                    f"{_INDEX_FILENAME} line {line_number} references payload_path "
                    f"{payload_path!r} not present in manifest file_inventory"
                )
    return problems

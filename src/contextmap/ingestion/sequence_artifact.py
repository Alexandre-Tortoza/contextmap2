"""Canonical sequence artifact: local, immutable, self-describing persistence.

A ``SequenceArtifact`` is the reusable, reopenable result of ingesting a
source (ROS 1 bag, ROS 2 bag, dataset) once. It is not the final
``ContextMapArtifact``; it is the normalized sensor sequence that later
pipeline stages read instead of reopening the original source. See
``src/contextmap/ingestion/docs/artifact.md`` for the on-disk layout,
schema version history, and the design trade-offs made for v0 (notably:
a JSON Lines index instead of a columnar format, and which candidate
subdirectories from the issue are deferred to later ingestion issues).

Writing is atomic: :class:`SequenceArtifactWriter` builds the artifact in a
temporary sibling directory and only makes it visible under its final path
after an internal integrity check succeeds, so an interrupted write can
never be mistaken for a complete artifact.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NewType
from uuid import uuid4

from contextmap.ingestion.calibration import (
    CalibrationSet,
    decode_calibration_set,
    encode_calibration_set,
    ensure_valid_calibration_set,
)
from contextmap.ingestion.diagnostics import (
    SequenceDiagnostics,
    decode_diagnostics_summary,
    encode_diagnostics,
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
from contextmap.shared import SourceTimestamp

SCHEMA_VERSION = "0.1.0"
"""Sequence artifact schema version written and understood by this module."""

SequenceArtifactId = NewType("SequenceArtifactId", str)
"""Identity of a persisted canonical sequence artifact."""

_MANIFEST_FILENAME = "manifest.json"
_INDEX_FILENAME = "index.jsonl"
_CALIBRATION_FILENAME = "calibration/calibration.json"
_PROVENANCE_FILENAME = "provenance/provenance.json"
_DIAGNOSTICS_SUMMARY_FILENAME = "diagnostics/summary.json"
_DIAGNOSTICS_WARNINGS_FILENAME = "diagnostics/warnings.jsonl"
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

    Example:
        writer = SequenceArtifactWriter(
            workspace_root=Path("workspace"), sequence_name="corridor-02"
        )
        writer.add_observation(image_observation)
        writer.add_observation(lidar_observation)
        manifest = writer.finalize()
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        sequence_name: str,
        artifact_id: SequenceArtifactId | None = None,
    ) -> None:
        """Create a writer for a new sequence artifact.

        Args:
            workspace_root: Root of the local workspace (contains
                ``sequences/``, ``runs/``, etc., per docs/architecture.md).
            sequence_name: Name of the sequence this artifact belongs to.
            artifact_id: Explicit artifact identity. A random one is
                generated when omitted.
        """
        self._sequence_name = sequence_name
        self._artifact_id = artifact_id or SequenceArtifactId(uuid4().hex)
        sequence_dir = workspace_root / "sequences" / sequence_name
        self._final_dir = sequence_dir / self._artifact_id
        self._tmp_dir = sequence_dir / f".tmp-{self._artifact_id}-{uuid4().hex[:8]}"
        self._observations: list[SourceObservation] = []
        self._seen_observation_ids: set[str] = set()
        self._calibration: CalibrationSet | None = None
        self._provenance: SequenceProvenance | None = None
        self._diagnostic_warnings: tuple[str, ...] | None = None
        self._finalized = False

    def set_diagnostics(self, *, warnings: Sequence[str] = ()) -> None:
        """Enable writing diagnostics for this sequence.

        The summary itself (per-modality counts/time-range, image
        resolutions, point-cloud field layouts) is computed automatically
        from the observations added to this writer; only warning text is
        supplied by the caller.

        Args:
            warnings: Adapter/validation/synchronization warnings to
                record alongside the summary, as human-readable text.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize`.
        """
        if self._finalized:
            raise SequenceArtifactError("cannot set diagnostics after finalize()")
        self._diagnostic_warnings = tuple(warnings)

    def set_provenance(self, provenance: SequenceProvenance) -> None:
        """Attach the sequence's provenance/content-identity metadata.

        Args:
            provenance: Provenance to persist alongside the artifact.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize`.
        """
        if self._finalized:
            raise SequenceArtifactError("cannot set provenance after finalize()")
        self._provenance = provenance

    def set_calibration(self, calibration_set: CalibrationSet) -> None:
        """Attach the sequence's calibration set, written by :meth:`finalize`.

        Args:
            calibration_set: Calibration and coordinate-frame inventory for
                this sequence.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize`.
            CalibrationError: If ``calibration_set`` is invalid; see
                :func:`~contextmap.ingestion.calibration.validate_calibration_set`.
        """
        if self._finalized:
            raise SequenceArtifactError("cannot set calibration after finalize()")
        ensure_valid_calibration_set(calibration_set)
        self._calibration = calibration_set

    def add_observation(self, observation: SourceObservation) -> None:
        """Queue an observation to be written by :meth:`finalize`.

        Args:
            observation: Any canonical source observation.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize`, or if
                ``observation.observation_id`` was already added.
        """
        if self._finalized:
            raise SequenceArtifactError("cannot add observations after finalize()")
        observation_id = str(observation.observation_id)
        if observation_id in self._seen_observation_ids:
            raise SequenceArtifactError(f"duplicate observation_id: {observation_id!r}")
        self._seen_observation_ids.add(observation_id)
        self._observations.append(observation)

    def finalize(self) -> SequenceArtifactManifest:
        """Write every queued observation and finalize the artifact atomically.

        The artifact is built under a temporary sibling directory and only
        moved to its final path after an internal integrity check succeeds,
        so a process interrupted mid-write never leaves a directory that
        looks like a complete artifact at the final path.

        Returns:
            The manifest of the finalized artifact.

        Raises:
            SequenceArtifactError: If already finalized, if an artifact
                already exists at the target path, or if writing fails.
        """
        if self._finalized:
            raise SequenceArtifactError("writer already finalized")
        if self._final_dir.exists():
            raise SequenceArtifactError(f"sequence artifact already exists: {self._final_dir}")

        self._tmp_dir.mkdir(parents=True, exist_ok=False)
        try:
            manifest = self._write_contents()
            problems = _check_file_inventory(
                self._tmp_dir, manifest
            ) + _check_index_cross_references(self._tmp_dir, manifest)
            if problems:
                raise SequenceArtifactError(
                    f"internal consistency check failed before finalize: {problems}"
                )
            self._tmp_dir.rename(self._final_dir)
        except BaseException:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            raise

        self._finalized = True
        return manifest

    def _write_contents(self) -> SequenceArtifactManifest:
        counts = dict(_MODALITY_COUNTS_TEMPLATE)
        file_entries: list[SequenceArtifactFileEntry] = []
        index_lines: list[str] = []

        for observation in self._observations:
            record, payload = _encode_observation(observation)
            modality = record["modality"]
            assert isinstance(modality, str)
            counts[modality] += 1
            if payload is not None:
                relative_path, data = payload
                absolute_path = self._tmp_dir / relative_path
                absolute_path.parent.mkdir(parents=True, exist_ok=True)
                absolute_path.write_bytes(data)
                file_entries.append(_file_entry(relative_path, data))
            index_lines.append(json.dumps(record, sort_keys=True))

        index_content = "".join(f"{line}\n" for line in index_lines)
        (self._tmp_dir / _INDEX_FILENAME).write_text(index_content, encoding="utf-8")
        file_entries.append(_file_entry(_INDEX_FILENAME, index_content.encode("utf-8")))

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
                self._observations, warning_count=len(self._diagnostic_warnings)
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

        manifest = SequenceArtifactManifest(
            artifact_id=self._artifact_id,
            sequence_name=self._sequence_name,
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            observation_counts=counts,
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

        return SequenceDiagnostics(summary=summary, warnings=tuple(warnings))

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
        return record, None

    if isinstance(observation, ExternalPoseMeasurement):
        record["parent_frame"] = str(observation.parent_frame)
        record["translation"] = list(observation.translation)
        record["orientation"] = list(observation.orientation)
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
        )

    if modality == "external_pose":
        return ExternalPoseMeasurement(
            **common,
            parent_frame=FrameId(record["parent_frame"]),
            translation=tuple(record["translation"]),
            orientation=tuple(record["orientation"]),
        )

    raise SequenceArtifactError(f"unknown modality in index record: {modality!r}")


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

"""Persisted, immutable, self-describing Sensor Association run artifacts.

A ``SensorAssociationRunArtifact`` is a directory holding the spatial observations one run
produced, with the indexes, lineage, quality and diagnostics needed to trust and to compare
them. It opens without ROS, a model SDK or NumPy, and one observation can be read by identity
without loading the others or any debug data. See
``src/contextmap/sensor_association/docs/artifact.md`` for the layout and its trade-offs.

Contractual data lives in ``outputs/`` and ``metrics/`` and is inventoried with size and
SHA-256 in the manifest; ``debug/`` is human evidence that is never inventoried, so removing it
cannot invalidate the run and no downstream stage may depend on it. Writing follows
:class:`~contextmap.shared.AtomicRunDirectory`: an interrupted write never looks like a
finished run and a finished run is never modified.

Associations are stored as compact tables rather than repeated coordinates: the geometry a
region supports is a run of little-endian ``uint32`` positions (geometry identity is positional
within its map), and a dense feature association is indices and weights, never vectors.
"""

from __future__ import annotations

import json
import sys
from array import array
from bisect import bisect_left
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, NewType

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import SequenceArtifactId
from contextmap.sensor_association.debug_evidence import write_full_debug, write_standard_debug
from contextmap.sensor_association.dense_sampling import SAMPLING_POLICY_ID
from contextmap.sensor_association.diagnostics import DIAGNOSTICS_DEFINITIONS_VERSION
from contextmap.sensor_association.membership import (
    COVERAGE_DEFINITIONS_VERSION,
    MEMBERSHIP_POLICY_ID,
)
from contextmap.sensor_association.models import SpatialObservation
from contextmap.sensor_association.quality import QUALITY_DEFINITIONS_VERSION, ObservationQuality
from contextmap.sensor_association.serialization import (
    decode_observation_quality,
    decode_spatial_observation,
    encode_calibration_ref,
    encode_observation_quality,
    encode_pose_ref,
    encode_spatial_observation,
)
from contextmap.sensor_association.service import FrameAssociation, SensorAssociationOutcome
from contextmap.shared import (
    AtomicRunDirectory,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
)
from contextmap.visual_perception import RegionId

if TYPE_CHECKING:
    from numpy.typing import NDArray

SCHEMA_VERSION = "0.2.0"
"""Sensor Association run artifact schema version written and understood by this module.

``0.2.0`` is the candidate-selection schema (#562): the manifest gained the required
``candidate_policy``, and every projection record gained ``candidates`` and ``stage_counts``.
The single ``point_count`` is **gone**: it meant the map's size in ``0.1.0`` and would mean the
evaluated population here, which a reader could not tell apart, so both concepts are named
explicitly as ``candidates.map_point_count`` and ``candidates.candidate_count``.
``geometry-support.u32`` and the dense ``eligible_indices`` now hold global
geometry indices explicitly rather than candidate rows that happened to coincide with them.
A ``0.1.0`` artifact is refused with a clear error instead of being migrated: during ``v0.x``
a run is re-executed, never rewritten.
"""

SensorAssociationRunId = NewType("SensorAssociationRunId", str)
"""Identity of one Sensor Association run, local to its capability and sequence."""

_MANIFEST = "manifest.json"
_OBSERVATIONS = "outputs/spatial-observations.jsonl"
_OBSERVATION_INDEX = "outputs/observation-index.jsonl"
_SUPPORT = "outputs/geometry-support.u32"
_QUALITY = "outputs/observation-quality.jsonl"
_PROJECTION = "outputs/projection-records.jsonl"
_VISIBILITY = "outputs/visibility-records.jsonl"
_DENSE_INDEX = "outputs/dense-feature-associations.jsonl"
_DENSE_CELLS = "outputs/dense-feature-cells.bin"
_DIAGNOSTICS = "metrics/frame-diagnostics.jsonl"
_SUMMARY = "metrics/summary.json"
_RUNTIME = "metrics/runtime.json"
_GEOMETRY_INFIX = "--geom-"
_U32_INDEX_LIMIT = 2**32
"""Global geometry indices must stay below this to fit the ``uint32`` support and dense tables."""


class RunArtifactError(Exception):
    """Base class for Sensor Association run artifact read/write failures."""


class IncompleteRunArtifactError(RunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class SensorAssociationDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and metrics only.
        STANDARD: Adds per-point samples and the distributions of the associated support.
        FULL: Adds the state-colored projection overlays, the dense sampling coordinates and
            the exact feature sources.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, kw_only=True)
class SensorAssociationRunManifest:
    """Authoritative metadata of a persisted Sensor Association run.

    Attributes:
        run_id: Identity of the run, supplied by the caller.
        run_index: Ordinal of the run among the caller's runs of this sequence, supplied by the
            caller.
        sequence_name: Name of the processed sequence.
        sequence_artifact_id: Canonical sequence artifact consumed.
        selection_id: Deterministic identity of the sequence selection.
        geometric_map_id: The geometric map the observations reference.
        trajectory_id: The trajectory that placed the camera.
        state_estimation_run_id: The state-estimation run it came from, when there is one.
        perception_run_ids: The perception runs the evidence came from.
        calibration_identity: Hash of the calibration used.
        candidate_policy: The candidate policy record, with its fingerprint: which map
            geometry each frame evaluated before projection.
        visibility_policy: The occlusion policy record, with its fingerprint.
        membership_policy_id: The mask-membership rule.
        definitions: Versions of the coverage, quality, diagnostics and sampling definitions.
        pose_policy: The pose lookup policy applied.
        tolerances: The diagnostic tolerances applied.
        dense_channels: Each declared dense channel with the exact feature sources it consumed.
        configuration_fingerprint: Hash of the effective configuration.
        code_version: Code revision that produced the run, when known.
        frame_count: Frames that were associated.
        rejected_frame_count: Frames whose pose the lookup policy rejected.
        failed_frame_count: Associated frames whose diagnostics hold a failure.
        observation_count: Spatial observations persisted.
        finding_counts: Diagnostic findings by code.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes the manifest, the
            README and ``debug/``.
    """

    run_id: SensorAssociationRunId
    run_index: int
    sequence_name: str
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    geometric_map_id: MapId
    trajectory_id: str
    state_estimation_run_id: str | None
    perception_run_ids: tuple[str, ...]
    calibration_identity: str
    candidate_policy: Mapping[str, Any]
    visibility_policy: Mapping[str, Any]
    membership_policy_id: str
    definitions: Mapping[str, str]
    pose_policy: Mapping[str, Any]
    tolerances: Mapping[str, Any]
    dense_channels: tuple[Mapping[str, Any], ...]
    configuration_fingerprint: str
    code_version: str | None
    frame_count: int
    rejected_frame_count: int
    failed_frame_count: int
    observation_count: int
    finding_counts: Mapping[str, int]
    debug_level: str
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


@dataclass(frozen=True, kw_only=True)
class DenseAssociationRecord:
    """The persisted association of one frame's points to one dense feature channel.

    Attributes:
        source_observation_id: The camera frame.
        channel_id: The evidence channel.
        provenance: The full lineage of the association, as JSON primitives.
        terms: Cells read per point: ``1`` for nearest and ``4`` for bilinear.
        eligible_indices: Global geometry indices of the eligible points, the identity
            :func:`~contextmap.geometric_mapping.geometry_id_for` uses.
        sampled: Whether the grid served each eligible point.
        cell_rows: Row of each cell read, ``terms`` per point, ``-1`` where out of support.
        cell_cols: Column of each cell read.
        weights: Weight of each cell read.
    """

    source_observation_id: str
    channel_id: str
    provenance: Mapping[str, Any]
    terms: int
    eligible_indices: array[int]
    sampled: tuple[bool, ...]
    cell_rows: array[int]
    cell_cols: array[int]
    weights: array[float]


class SensorAssociationRunWriter:
    """Builds an immutable Sensor Association run artifact on the local filesystem.

    The writer holds the run's identity and opens a :meth:`transaction`; the transaction is
    what touches the filesystem, and it is a
    :class:`~contextmap.sensor_association.service.FrameSink`, so a run is persisted frame by
    frame instead of from a fully materialized result::

        with writer.transaction() as run:
            outcome = SensorAssociationService().run(request, sink=run)
            manifest = run.finalize(outcome)
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        run_id: SensorAssociationRunId,
        run_index: int,
        debug_level: SensorAssociationDebugLevel = SensorAssociationDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            output_dir: The final directory of the artifact. The caller chooses it (in the
                runtime, ``<workspace>/<dataset>/<run>/sensor_association``); the writer
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
        self._opened = False

    @property
    def run_id(self) -> SensorAssociationRunId:
        """Identity of the run this writer persists."""
        return self._run_id

    @property
    def run_index(self) -> int:
        """Ordinal of the run among the caller's runs of the same sequence."""
        return self._run_index

    @property
    def sequence_name(self) -> str:
        """Name of the sequence the run processed."""
        return self._sequence_name

    @property
    def debug_level(self) -> SensorAssociationDebugLevel:
        """How much non-contractual debug evidence the run persists."""
        return self._debug_level

    @property
    def final_dir(self) -> Path:
        """Where the finished artifact will live."""
        return self._final_dir

    @contextmanager
    def transaction(self) -> Iterator[SensorAssociationRunTransaction]:
        """Open the run, yielding the sink that persists each frame as it arrives.

        Leaving the block without a successful :meth:`SensorAssociationRunTransaction.finalize`
        -- normally or through an exception -- discards everything written: an interrupted run
        never leaves a publishable artifact.

        Raises:
            RunArtifactError: If this writer already opened a transaction, or a run already
                exists at the target path.
        """
        if self._opened:
            raise RunArtifactError("writer already finalized")
        self._opened = True
        try:
            with AtomicRunDirectory(self._final_dir) as run:
                transaction = SensorAssociationRunTransaction(run=run, writer=self)
                try:
                    yield transaction
                finally:
                    transaction.close()
        except RunDirectoryError as error:
            raise RunArtifactError(str(error)) from error


class SensorAssociationRunTransaction:
    """One run being written: a frame sink that persists and then forgets each frame.

    Per-frame payloads are appended to open streams and nothing about a frame is retained:
    no array, no observation, no projection. Nothing is visible until :meth:`finalize`
    publishes the run atomically.

    What the transaction does keep is bounded and scalar. Counts and byte offsets are
    constant, and one timing record per frame accumulates because #562 asks for per-frame
    candidate-query and projection times and ``metrics/runtime.json`` is a single document
    written only when the caller measured. That record is four primitives, about 567 B, so
    3096 frames cost under 1.7 MB -- a linear term worth naming rather than hiding, and four
    orders of magnitude below the 4.3 GB peak of the real 350-frame run. The rule it must keep
    obeying is that the linear term holds **scalars**, never a frame or an array.
    """

    def __init__(self, *, run: AtomicRunDirectory, writer: SensorAssociationRunWriter) -> None:
        """Open the run's contractual streams.

        Args:
            run: The atomic directory the run is built in.
            writer: The writer that owns the run's identity and debug level.
        """
        self._run = run
        self._writer = writer
        self._streams = ExitStack()
        self._closed = False
        self._finalized = False
        self._observations = self._open(_OBSERVATIONS)
        self._index = self._open(_OBSERVATION_INDEX)
        self._support = self._open(_SUPPORT)
        self._quality = self._open(_QUALITY)
        self._projection = self._open(_PROJECTION)
        self._visibility = self._open(_VISIBILITY)
        self._diagnostics = self._open(_DIAGNOSTICS)
        self._dense_index: BinaryIO | None = None
        self._dense_cells: BinaryIO | None = None
        # Agregados de run: contagens e bytes, nunca arrays de frame.
        self._support_offset = 0
        self._observation_offset = 0
        self._quality_offset = 0
        self._dense_offset = 0
        self._frame_count = 0
        self._observation_count = 0
        self._observations_without_support = 0
        self._failed_frame_count = 0
        self._finding_counts: dict[str, int] = {}
        self._state_counts: dict[str, int] = {}
        self._dense_sources: dict[str, dict[str, dict[str, Any]]] = {}
        self._debug_feature_sources: list[dict[str, Any]] = []
        self._timings: list[dict[str, Any]] = []

    def _open(self, relative_path: str) -> BinaryIO:
        return self._streams.enter_context(self._run.open_binary(relative_path))

    def accept(self, frame: FrameAssociation) -> None:
        """Persist one completed frame and update the run's aggregates.

        Implements :class:`~contextmap.sensor_association.service.FrameSink`. Nothing about the
        frame is retained: only counts, byte offsets and the frame's own timings survive the
        call.

        Raises:
            RunArtifactError: If the run is already finalized or closed, an observation's
                geometry support disagrees with its region membership, or a persisted global
                geometry index does not fit the 32-bit support or dense tables.
        """
        if self._finalized or self._closed:
            raise RunArtifactError("the run is no longer open for frames")
        self._write_frame_observations(frame)
        self._projection.write(_line(_projection_record(frame)))
        self._visibility.write(_line(_visibility_record(frame)))
        self._diagnostics.write(_line(frame.diagnostics.to_record()))
        if frame.dense_samples:
            self._write_frame_dense(frame)
        self._accumulate(frame)
        self._write_frame_debug(frame)

    def finalize(
        self, outcome: SensorAssociationOutcome, *, runtime_s: float | None = None
    ) -> SensorAssociationRunManifest:
        """Close the streams, write the run-level files and publish the artifact atomically.

        Args:
            outcome: The run's identity, policies and aggregates, as the service returned it.
            runtime_s: Wall-clock time the run took, when measured. It is a metric apart from
                every quality measure, and it never enters ``outputs/``.

        Returns:
            The manifest of the finalized run.

        Raises:
            RunArtifactError: If already finalized, the outcome counts other frames than this
                transaction persisted, or the run cannot be published.
        """
        if self._finalized:
            raise RunArtifactError("writer already finalized")
        if outcome.frame_count != self._frame_count:
            raise RunArtifactError(
                f"the outcome counts {outcome.frame_count} frames but {self._frame_count} "
                "reached the run"
            )
        self.close()
        try:
            self._run.write_text(_SUMMARY, _json(self._summary(outcome)))
            if runtime_s is not None:
                # Tempo de parede nunca entra em outputs/: um rerun não o reproduz. Aqui ele
                # só existe quando o chamador realmente mediu, o que mantém o artefato
                # reprodutível byte a byte.
                self._run.write_text(
                    _RUNTIME,
                    _json(
                        {
                            "runtime_s": runtime_s,
                            "candidate_query_seconds": sum(
                                float(t["candidate_query_seconds"]) for t in self._timings
                            ),
                            "projection_seconds": sum(
                                float(t["projection_seconds"]) for t in self._timings
                            ),
                            "frames": self._timings,
                        }
                    ),
                )
            if self._debug_feature_sources:
                self._run.write_text(
                    "debug/feature-sources.json",
                    json.dumps(self._debug_feature_sources, indent=2, sort_keys=True) + "\n",
                    contractual=False,
                )
            self._run.publish(
                manifest=self._manifest_record(outcome),
                readme=_render_readme(self._writer.run_id, self._writer.run_index, outcome),
            )
        except RunDirectoryError as error:
            raise RunArtifactError(str(error)) from error
        self._finalized = True
        return _load_manifest(self._writer.final_dir)

    def close(self) -> None:
        """Close every open stream; idempotent, and required before publishing."""
        if not self._closed:
            self._closed = True
            self._streams.close()

    # --- per frame -------------------------------------------------------------------

    def _write_frame_observations(self, frame: FrameAssociation) -> None:
        import numpy as np

        projection = frame.resolution.frame
        for region, observation, quality in zip(
            frame.membership.regions, frame.observations, frame.qualities, strict=True
        ):
            # As posições são linhas de candidatos; o que se persiste é a identidade global
            # de cada uma, que é o que o leitor reconstrói (#562).
            rows = np.asarray(region.associated_indices)
            positions = _as_u32_indices(projection.global_indices[rows], table=_SUPPORT)
            if projection.map_references(rows) != observation.geometry_support:
                raise RunArtifactError(
                    f"the geometry support of {observation.spatial_observation_id!r} "
                    "does not match its region membership"
                )
            record = encode_spatial_observation(observation)
            del record["geometry_support"]
            record["support"] = {
                "offset": self._support_offset,
                "count": int(positions.shape[0]),
            }
            observation_line = json.dumps(record, sort_keys=True).encode("utf-8")
            quality_line = json.dumps(encode_observation_quality(quality), sort_keys=True).encode(
                "utf-8"
            )
            self._index.write(
                _line(
                    {
                        "spatial_observation_id": str(observation.spatial_observation_id),
                        "source_observation_id": str(observation.source_observation_id),
                        "region_id": str(observation.region_id),
                        "observation": {
                            "byte_offset": self._observation_offset,
                            "byte_length": len(observation_line),
                        },
                        "quality": {
                            "byte_offset": self._quality_offset,
                            "byte_length": len(quality_line),
                        },
                    }
                )
            )
            self._support.write(positions.tobytes())
            self._observations.write(observation_line + b"\n")
            self._quality.write(quality_line + b"\n")
            self._support_offset += int(positions.shape[0])
            self._observation_offset += len(observation_line) + 1
            self._quality_offset += len(quality_line) + 1

    def _write_frame_dense(self, frame: FrameAssociation) -> None:
        import numpy as np

        if self._dense_index is None or self._dense_cells is None:
            self._dense_index = self._open(_DENSE_INDEX)
            self._dense_cells = self._open(_DENSE_CELLS)
        global_indices = frame.resolution.frame.global_indices
        for channel_id in sorted(frame.dense_samples):
            samples = frame.dense_samples[channel_id]
            sections = (
                _as_u32_indices(
                    global_indices[samples.eligible_indices], table=_DENSE_CELLS
                ).tobytes(),
                np.asarray(samples.sampled, dtype="u1").tobytes(),
                np.asarray(samples.cell_rows, dtype="<i4").tobytes(),
                np.asarray(samples.cell_cols, dtype="<i4").tobytes(),
                np.asarray(samples.weights, dtype="<f4").tobytes(),
            )
            self._dense_index.write(
                _line(
                    {
                        "source_observation_id": str(frame.source_observation_id),
                        "channel_id": channel_id,
                        "provenance": samples.provenance.to_record(),
                        "terms": int(samples.cell_rows.shape[1]),
                        "eligible_count": int(samples.eligible_indices.shape[0]),
                        "byte_offset": self._dense_offset,
                    }
                )
            )
            for section in sections:
                self._dense_cells.write(section)
                self._dense_offset += len(section)

    def _write_frame_debug(self, frame: FrameAssociation) -> None:
        level = self._writer.debug_level
        if level is SensorAssociationDebugLevel.NONE:
            return
        write_standard_debug(self._run, frame)
        if level is SensorAssociationDebugLevel.FULL:
            self._debug_feature_sources.extend(write_full_debug(self._run, frame))

    def _accumulate(self, frame: FrameAssociation) -> None:
        self._frame_count += 1
        self._observation_count += len(frame.observations)
        self._observations_without_support += sum(
            1 for observation in frame.observations if not observation.geometry_support
        )
        self._failed_frame_count += int(frame.diagnostics.failed)
        for finding in frame.diagnostics.findings:
            code = finding.code.value
            self._finding_counts[code] = self._finding_counts.get(code, 0) + 1
        resolution = frame.resolution
        for state, count in resolution.state_counts().items():
            self._state_counts[state.value] = self._state_counts.get(state.value, 0) + count
        self._state_counts["visible"] = (
            self._state_counts.get("visible", 0) + resolution.visible_count
        )
        projection = resolution.frame
        self._timings.append(
            {
                "source_observation_id": str(frame.source_observation_id),
                "candidate_query_seconds": projection.candidates.query_seconds,
                "projection_seconds": projection.projection_seconds,
                "candidate_count": projection.candidate_count,
            }
        )
        for channel_id, samples in frame.dense_samples.items():
            source = _dense_feature_source(samples.provenance)
            channel = self._dense_sources.setdefault(channel_id, {})
            key = json.dumps(source, sort_keys=True)
            entry = channel.setdefault(key, {**source, "frame_count": 0})
            entry["frame_count"] += 1

    # --- run level -------------------------------------------------------------------

    def _summary(self, outcome: SensorAssociationOutcome) -> dict[str, Any]:
        return {
            "frame_count": self._frame_count,
            "observation_count": self._observation_count,
            "observations_without_support": self._observations_without_support,
            "state_counts": dict(self._state_counts),
            "rejected_frames": [
                {
                    "source_observation_id": str(rejected.source_observation_id),
                    "rejection": rejected.rejection.value,
                }
                for rejected in outcome.rejected
            ],
        }

    def _manifest_record(self, outcome: SensorAssociationOutcome) -> dict[str, Any]:
        pose = outcome.pose_policy
        tolerances = outcome.tolerances
        return {
            "run_id": str(self._writer.run_id),
            "run_index": self._writer.run_index,
            "sequence_name": self._writer.sequence_name,
            "sequence_artifact_id": str(outcome.sequence_artifact_id),
            "selection_id": outcome.selection_id,
            "geometric_map_id": str(outcome.geometric_map.map_id),
            "trajectory_id": str(outcome.trajectory_id),
            "state_estimation_run_id": None
            if outcome.state_estimation_run_id is None
            else str(outcome.state_estimation_run_id),
            "perception_run_ids": [str(run_id) for run_id in outcome.perception_run_ids],
            "calibration_identity": outcome.calibration_identity,
            "candidate_policy": {
                **outcome.candidate_policy.to_record(),
                "fingerprint": outcome.candidate_policy.fingerprint(),
            },
            "visibility_policy": {
                **outcome.occlusion_policy.to_record(),
                "fingerprint": outcome.occlusion_policy.fingerprint(),
            },
            "membership_policy_id": MEMBERSHIP_POLICY_ID,
            "definitions": {
                "coverage": COVERAGE_DEFINITIONS_VERSION,
                "quality": QUALITY_DEFINITIONS_VERSION,
                "diagnostics": DIAGNOSTICS_DEFINITIONS_VERSION,
                "dense_sampling": SAMPLING_POLICY_ID,
            },
            "pose_policy": {
                "mode": pose.mode.value,
                "max_time_delta_ns": pose.max_time_delta_ns,
                "max_interpolation_gap_ns": pose.max_interpolation_gap_ns,
            },
            "tolerances": {
                "max_pose_time_delta_ns": tolerances.max_pose_time_delta_ns,
                "max_map_window_offset_ns": tolerances.max_map_window_offset_ns,
                "max_reprojection_p95_px": tolerances.max_reprojection_p95_px,
                "max_reprojection_invalid_rate": tolerances.max_reprojection_invalid_rate,
            },
            "dense_channels": [
                {
                    "channel_id": channel.channel_id,
                    "interpolation": channel.interpolation.value,
                    "feature_sources": [
                        self._dense_sources[channel.channel_id][key]
                        for key in sorted(self._dense_sources.get(channel.channel_id, {}))
                    ],
                }
                for channel in outcome.dense_channels
            ],
            "configuration_fingerprint": outcome.configuration_fingerprint,
            "code_version": outcome.code_version,
            "frame_count": self._frame_count,
            "rejected_frame_count": len(outcome.rejected),
            "failed_frame_count": self._failed_frame_count,
            "observation_count": self._observation_count,
            "finding_counts": dict(self._finding_counts),
            "debug_level": self._writer.debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }


class SensorAssociationRunReader:
    """Reads a finalized Sensor Association run from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file outside it is
                needed.

        Raises:
            IncompleteRunArtifactError: If ``manifest.json`` is missing.
            RunArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = _load_manifest(run_dir)
        self._index: dict[str, dict[str, Any]] | None = None
        self._by_frame: dict[str, list[str]] | None = None
        self._dense_index: dict[tuple[str, str], dict[str, Any]] | None = None

    @property
    def manifest(self) -> SensorAssociationRunManifest:
        """The run's manifest."""
        return self._manifest

    def observations(self) -> Iterator[SpatialObservation]:
        """Iterate every spatial observation, in run order, reading the files sequentially."""
        for identity in self._load_index():
            yield self.observation(identity)

    def observation(self, spatial_observation_id: str) -> SpatialObservation:
        """Read one observation by identity without loading the others.

        Args:
            spatial_observation_id: Identity of the observation.

        Returns:
            The observation, revalidated against the contract, with its geometry support
            resolved from the columnar table.

        Raises:
            RunArtifactError: If the run has no such observation.
        """
        entry = self._entry(spatial_observation_id)
        record = json.loads(
            _read_slice(self._root / _OBSERVATIONS, **entry["observation"]).decode("utf-8")
        )
        support = record.pop("support")
        map_id = MapId(record["provenance"]["geometric_map_id"])
        record["geometry_support"] = [
            {"map_id": str(map_id), "geometry_id": str(geometry_id_for(map_id=map_id, index=i))}
            for i in self._read_support(support["offset"], support["count"])
        ]
        return decode_spatial_observation(record)

    def observations_of_frame(self, source_observation_id: str) -> tuple[SpatialObservation, ...]:
        """Read every observation of one camera frame, in region order."""
        by_frame = self._load_by_frame()
        return tuple(self.observation(i) for i in by_frame.get(str(source_observation_id), []))

    def quality(self, spatial_observation_id: str) -> ObservationQuality:
        """Read the quality of one observation.

        Raises:
            RunArtifactError: If the run has no such observation.
        """
        entry = self._entry(spatial_observation_id)
        return decode_observation_quality(
            json.loads(_read_slice(self._root / _QUALITY, **entry["quality"]).decode("utf-8"))
        )

    def geometry_support(self, spatial_observation_id: str) -> tuple[GeometryReference, ...]:
        """Read the geometry references one observation's region supports."""
        entry = self._entry(spatial_observation_id)
        record = json.loads(
            _read_slice(self._root / _OBSERVATIONS, **entry["observation"]).decode("utf-8")
        )
        support = record["support"]
        map_id = self._manifest.geometric_map_id
        return tuple(
            GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=i))
            for i in self._read_support(support["offset"], support["count"])
        )

    def regions_of(
        self, source_observation_id: str, reference: GeometryReference
    ) -> tuple[RegionId, ...]:
        """Tell which regions of a frame support a geometry element, in region order.

        An empty tuple means no region of that frame contains it; overlaps keep every region.
        """
        if reference.map_id != self._manifest.geometric_map_id:
            return ()
        position = int(str(reference.geometry_id).rsplit(_GEOMETRY_INFIX, 1)[1])
        regions: list[RegionId] = []
        for identity in self._load_by_frame().get(str(source_observation_id), []):
            entry = self._entry(identity)
            record = json.loads(
                _read_slice(self._root / _OBSERVATIONS, **entry["observation"]).decode("utf-8")
            )
            support = self._read_support(record["support"]["offset"], record["support"]["count"])
            at = bisect_left(support, position)
            if at < len(support) and support[at] == position:
                regions.append(RegionId(entry["region_id"]))
        return tuple(regions)

    def dense_association(
        self, source_observation_id: str, channel_id: str
    ) -> DenseAssociationRecord:
        """Read the dense feature association of one frame and channel.

        Raises:
            RunArtifactError: If the run has no such association.
        """
        record = self._load_dense_index().get((str(source_observation_id), channel_id))
        if record is None:
            raise RunArtifactError(
                f"no dense association for frame {source_observation_id!r} in channel "
                f"{channel_id!r}"
            )
        eligible, terms = record["eligible_count"], record["terms"]
        # Ordem das seções: pontos elegíveis, amostrados, linhas, colunas e pesos das células.
        sizes = (
            eligible * 4,
            eligible,
            eligible * terms * 4,
            eligible * terms * 4,
            eligible * terms * 4,
        )
        with (self._root / _DENSE_CELLS).open("rb") as handle:
            handle.seek(record["byte_offset"])
            sections = [handle.read(size) for size in sizes]
        return DenseAssociationRecord(
            source_observation_id=record["source_observation_id"],
            channel_id=channel_id,
            provenance=record["provenance"],
            terms=terms,
            eligible_indices=_array("I", sections[0]),
            sampled=tuple(bool(b) for b in sections[1]),
            cell_rows=_array("i", sections[2]),
            cell_cols=_array("i", sections[3]),
            weights=_array("f", sections[4]),
        )

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``outputs/`` or ``metrics/``.

        Raises:
            RunArtifactError: If the path is not a JSON record of a contractual directory;
                ``debug/`` is never a valid source.
        """
        _require_contractual(relative_path, ".json")
        record: dict[str, Any] = json.loads(
            (self._root / relative_path).read_text(encoding="utf-8")
        )
        return record

    def read_records(self, relative_path: str) -> list[dict[str, Any]]:
        """Read the lines of a JSONL file from ``outputs/`` or ``metrics/``.

        Raises:
            RunArtifactError: If the path is not a JSONL file of a contractual directory.
        """
        _require_contractual(relative_path, ".jsonl")
        text = (self._root / relative_path).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line]

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        return check_file_inventory(self._root, self._manifest.file_inventory)

    def _entry(self, spatial_observation_id: str) -> dict[str, Any]:
        entry = self._load_index().get(str(spatial_observation_id))
        if entry is None:
            raise RunArtifactError(
                f"unknown spatial observation in this run: {spatial_observation_id!r}"
            )
        return entry

    def _read_support(self, offset: int, count: int) -> array[int]:
        with (self._root / _SUPPORT).open("rb") as handle:
            handle.seek(offset * 4)
            return _array("I", handle.read(count * 4))

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self._index is None:
            index: dict[str, dict[str, Any]] = {}
            for record in self.read_records(_OBSERVATION_INDEX):
                index[record["spatial_observation_id"]] = record
            self._index = index
        return self._index

    def _load_by_frame(self) -> dict[str, list[str]]:
        if self._by_frame is None:
            by_frame: dict[str, list[str]] = {}
            for identity, record in self._load_index().items():
                by_frame.setdefault(record["source_observation_id"], []).append(identity)
            self._by_frame = by_frame
        return self._by_frame

    def _load_dense_index(self) -> dict[tuple[str, str], dict[str, Any]]:
        if self._dense_index is None:
            index: dict[tuple[str, str], dict[str, Any]] = {}
            if (self._root / _DENSE_INDEX).is_file():
                for record in self.read_records(_DENSE_INDEX):
                    index[(record["source_observation_id"], record["channel_id"])] = record
            self._dense_index = index
        return self._dense_index


def _load_manifest(run_dir: Path) -> SensorAssociationRunManifest:
    manifest_path = run_dir / _MANIFEST
    if not manifest_path.is_file():
        raise IncompleteRunArtifactError(f"missing {_MANIFEST} in {run_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RunArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}"
        )
    return SensorAssociationRunManifest(
        run_id=SensorAssociationRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        sequence_artifact_id=SequenceArtifactId(raw["sequence_artifact_id"]),
        selection_id=raw["selection_id"],
        geometric_map_id=MapId(raw["geometric_map_id"]),
        trajectory_id=raw["trajectory_id"],
        state_estimation_run_id=raw["state_estimation_run_id"],
        perception_run_ids=tuple(raw["perception_run_ids"]),
        calibration_identity=raw["calibration_identity"],
        candidate_policy=dict(raw["candidate_policy"]),
        visibility_policy=dict(raw["visibility_policy"]),
        membership_policy_id=raw["membership_policy_id"],
        definitions=dict(raw["definitions"]),
        pose_policy=dict(raw["pose_policy"]),
        tolerances=dict(raw["tolerances"]),
        dense_channels=tuple(raw["dense_channels"]),
        configuration_fingerprint=raw["configuration_fingerprint"],
        code_version=raw["code_version"],
        frame_count=raw["frame_count"],
        rejected_frame_count=raw["rejected_frame_count"],
        failed_frame_count=raw["failed_frame_count"],
        observation_count=raw["observation_count"],
        finding_counts=dict(raw["finding_counts"]),
        debug_level=raw["debug_level"],
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


def _require_contractual(relative_path: str, suffix: str) -> None:
    if not relative_path.endswith(suffix) or not relative_path.startswith(("outputs/", "metrics/")):
        raise RunArtifactError(f"not a contractual {suffix} record: {relative_path!r}")


def _read_slice(path: Path, *, byte_offset: int, byte_length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(byte_offset)
        return handle.read(byte_length)


def _as_u32_indices(global_indices: NDArray[Any], *, table: str) -> NDArray[Any]:
    """Return global geometry indices as the little-endian ``uint32`` a table stores.

    Raises:
        RunArtifactError: If an index does not fit in 32 bits. The cast would wrap it onto
            another element's identity, so the persisted support would silently name the wrong
            geometry; the schema has no wider index, so such a map is refused instead.
    """
    import numpy as np

    if global_indices.size and int(global_indices.max()) >= _U32_INDEX_LIMIT:
        raise RunArtifactError(
            f"global geometry index {int(global_indices.max())} does not fit the 32-bit {table} "
            f"table of schema {SCHEMA_VERSION}: casting it would wrap it onto another element"
        )
    return np.asarray(global_indices, dtype="<u4")


def _array(typecode: str, data: bytes) -> array[Any]:
    """Decode little-endian 4-byte items with the standard library alone."""
    values = array(typecode)
    if values.itemsize != 4:
        raise RunArtifactError(f"the platform's {typecode!r} items are not 4 bytes wide")
    values.frombytes(data)
    if sys.byteorder == "big":
        values.byteswap()
    return values


def _line(record: Mapping[str, Any]) -> bytes:
    """Encode one JSONL record, with its newline, exactly as a batch write would."""
    return json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"


def _dense_feature_source(provenance: Any) -> dict[str, Any]:
    """The feature source one dense channel sampled in a frame, as JSON primitives."""
    enhancement = provenance.enhancement
    return {
        "source_artifact_id": provenance.source_artifact_id,
        "embedding_space_id": provenance.embedding_space_id,
        "coordinate_transform_id": provenance.coordinate_transform_id,
        "sampling_fingerprint": provenance.sampling_fingerprint,
        "extractor": {
            "backend_id": provenance.extractor.backend_id,
            "provider": provenance.extractor.provider,
            "model": provenance.extractor.model,
            "version": provenance.extractor.version,
            "configuration_fingerprint": provenance.extractor.configuration_fingerprint,
        },
        "enhancement": None
        if enhancement is None
        else {
            "backend_id": enhancement.backend.backend_id,
            "model": enhancement.backend.model,
            "version": enhancement.backend.version,
            "configuration_fingerprint": enhancement.backend.configuration_fingerprint,
            "source_embedding_space_id": enhancement.source_embedding_space_id,
            "output_embedding_space_id": enhancement.output_embedding_space_id,
            "input_grid_size": list(enhancement.input_grid_size),
            "output_grid_size": list(enhancement.output_grid_size),
        },
    }


def _json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def _projection_record(frame: FrameAssociation) -> dict[str, Any]:
    projection = frame.resolution.frame
    transform = projection.image_transform
    return {
        "source_observation_id": str(projection.source_observation_id),
        "image_timestamp": projection.image_timestamp.to_record(),
        "map_id": str(projection.map_id),
        # Sem um `point_count` só: ele significaria o tamanho do mapa no schema 0.1.0 e a
        # população avaliada no 0.2.0, e um leitor não teria como saber qual. Os dois conceitos
        # ficam nomeados dentro de `candidates` (#564).
        "candidates": projection.candidates.to_record(),
        "stage_counts": {stage.value: count for stage, count in projection.stage_counts().items()},
        "calibration_ref": encode_calibration_ref(projection.calibration_ref),
        "camera": {
            "calibration_id": str(projection.camera.calibration_id),
            "content_hash": projection.camera.content_hash,
            "camera_frame": str(projection.camera.camera_frame),
            "camera_model_kind": projection.camera.camera_model_kind,
            "image_size": list(projection.camera.image_size),
        },
        "pose_ref": encode_pose_ref(projection.pose_ref),
        "extrinsic": {
            "calibration_identity": projection.extrinsic.calibration_identity,
            "parent_frame": str(projection.extrinsic.parent_frame),
            "child_frame": str(projection.extrinsic.child_frame),
        },
        "image_transform": {
            "transform_id": transform.transform_id,
            "raw_size": list(transform.raw_size),
            "prepared_size": list(transform.prepared_size),
            "steps": [
                {
                    "operation": step.operation,
                    "input_size": list(step.input_size),
                    "output_size": list(step.output_size),
                    "offset_px": list(step.offset_px),
                    "scale": list(step.scale),
                }
                for step in transform.steps
            ],
        },
    }


def _visibility_record(frame: FrameAssociation) -> dict[str, Any]:
    resolution = frame.resolution
    membership = frame.membership
    statistics = membership.statistics()
    return {
        "source_observation_id": str(frame.source_observation_id),
        "policy_id": resolution.policy.policy_id,
        "policy_fingerprint": resolution.policy.fingerprint(),
        "depth_metric": resolution.depth_metric.value,
        "state_counts": {
            **{state.value: count for state, count in resolution.state_counts().items()},
        },
        "visible": resolution.visible_count,
        "neighborhood_only_occlusions": resolution.neighborhood_only_occlusions,
        "membership": {
            "definitions_version": statistics.definitions_version,
            "associated_count": statistics.associated_count,
            "visible_unassigned_count": statistics.visible_unassigned_count,
            "overlap_point_count": statistics.overlap_point_count,
            "membership_pair_count": statistics.membership_pair_count,
            "points_by_region_count": {
                str(k): v for k, v in statistics.points_by_region_count.items()
            },
        },
        "regions": [
            {
                "region_id": str(region.region_id),
                "mask_area_px": region.mask_area_px,
                "associated_count": region.associated_count,
                "occluded_count": region.occluded_count,
                "outside_valid_support_count": region.outside_valid_support_count,
                "covered_pixel_count": region.covered_pixel_count,
            }
            for region in membership.regions
        ],
        "skipped_regions": [
            {"region_id": str(s.region_id), "reason": s.reason.value} for s in membership.skipped
        ],
    }


def _render_readme(
    run_id: SensorAssociationRunId, run_index: int, outcome: SensorAssociationOutcome
) -> str:
    channels = ", ".join(f"`{c.channel_id}`" for c in outcome.dense_channels) or "none"
    return (
        f"# Sensor association run {run_index:04d}\n"
        "\n"
        f"- Run ID: `{run_id}`\n"
        f"- Sequence artifact: `{outcome.sequence_artifact_id}`\n"
        f"- Selection: `{outcome.selection_id}`\n"
        f"- Geometric map: `{outcome.geometric_map.map_id}`\n"
        f"- Trajectory: `{outcome.trajectory_id}`\n"
        f"- Dense feature channels: {channels}\n"
        f"- Frames: {outcome.frame_count} associated, {len(outcome.rejected)} rejected\n"
        "\n"
        "Contractual data is in `outputs/` and `metrics/`; `debug/` is human evidence and no "
        "downstream stage may depend on it.\n"
    )

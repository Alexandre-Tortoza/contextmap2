"""Persisted, immutable, self-describing Geometric Mapping run artifacts.

A ``GeometricMapArtifact`` is a directory holding the map one run built from a
selected segment of a canonical sequence and a State Estimation trajectory, with
the lineage, configuration, metrics and debug evidence needed to trust it. It
opens without ROS, an estimator, a model library or NumPy, and a
:class:`~contextmap.geometric_mapping.GeometryReference` resolves after the run is
closed and reopened to the same authoritative XYZ and provenance. See
``src/contextmap/geometric_mapping/docs/artifact.md`` for the layout.

Contractual data (``outputs/``, ``metrics/``, ``lineage.json``, ``config.json`` and
``environment.json``) is inventoried with size and SHA-256 in the manifest;
``debug/`` is human evidence that is never inventoried, so removing it cannot
invalidate the run and no downstream stage may depend on it. Writing follows
:class:`~contextmap.shared.AtomicRunDirectory`: an interrupted write never looks
like a finished run and a finished run is never modified.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import platform
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, NewType

from contextmap.geometric_mapping.accumulation import (
    SCAN_BOUNDS_INDEX,
    AccumulatedMap,
    ScanVoxelPolicy,
    accumulate_plan,
)
from contextmap.geometric_mapping.geometry_storage import PACKED_POINT, PackedGeometry, ScanRecord
from contextmap.geometric_mapping.inputs import GeometryInputPlan
from contextmap.geometric_mapping.models import (
    GeometricMap,
    GeometryReference,
    MapId,
    geometry_id_for,
)
from contextmap.geometric_mapping.serialization import (
    decode_geometric_map,
    decode_scan_record,
    encode_bounds,
    encode_geometric_map,
    encode_lookup_policy,
    encode_motion_correction_policy,
    encode_scan_record,
    encode_transform_trace,
)
from contextmap.geometric_mapping.transformation import (
    TransformTrace,
    transform_trace_residual_m,
    verify_transform_trace,
)
from contextmap.ingestion import (
    FrameId,
    SequenceArtifactId,
    encode_selection,
)
from contextmap.shared import (
    AtomicRunDirectory,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
)
from contextmap.state_estimation import (
    StateEstimationRunId,
    TrajectoryId,
    summarize_lookups,
)

SCHEMA_VERSION = "0.1.0"
"""Geometric Mapping run artifact schema version written and understood by this module."""

GeometricMapRunId = NewType("GeometricMapRunId", str)
"""Identity of one Geometric Mapping run, local to its capability and sequence."""

_MANIFEST = "manifest.json"
_LINEAGE = "lineage.json"
_CONFIG = "config.json"
_ENVIRONMENT = "environment.json"
_GEOMETRY = "outputs/geometry.bin"
_SOURCE_INDEX = "outputs/source-index.jsonl"
_MAP_METADATA = "outputs/map-metadata.json"
_INPUT_PLAN = "metrics/input-plan.json"
_MAPPING = "metrics/mapping.json"
_RUNTIME = "metrics/runtime.json"

_CONTRACTUAL_RECORDS = frozenset({_LINEAGE, _CONFIG, _ENVIRONMENT})
_CONTRACTUAL_DIRECTORIES = ("outputs/", "metrics/")

# Amostras de traces de transformação: quantos scans e quantos pontos por scan.
_STANDARD_TRACE_SCANS, _STANDARD_TRACE_POINTS = 5, 2
_FULL_TRACE_SCANS, _FULL_TRACE_POINTS = 50, 3


class MapArtifactError(Exception):
    """Base class for Geometric Mapping run artifact read/write failures."""


class IncompleteMapArtifactError(MapArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class MapDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and required metrics only.
        STANDARD: Adds scan and bounds summaries, the trajectory over the map and
            a small sample of transform traces.
        FULL: Adds a denser sample of traces and their reconstruction residuals.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, kw_only=True)
class GeometricMapArtifactManifest:
    """Authoritative metadata of a persisted Geometric Mapping run.

    Attributes:
        run_id: Identity of the run, supplied by the caller.
        run_index: Ordinal of the run among the caller's runs of this sequence, supplied by the
            caller.
        sequence_name: Name of the mapped sequence.
        map_id: Identity of the map; part of every geometry reference.
        map_frame: The global frame every coordinate is expressed in.
        sequence_artifact_id: Canonical sequence artifact consumed.
        selection_id: Deterministic identity of the sequence selection.
        trajectory_id: Trajectory whose poses placed the geometry.
        state_estimation_run_id: The State Estimation run behind the trajectory, if known.
        calibration_identity: Hash of the calibration the extrinsics came from.
        configuration_fingerprint: Hash of the effective mapping configuration.
        code_version: Code revision that produced the run, when known.
        point_count: Geometry elements persisted.
        source_point_count: Finite points measured before any aggregation.
        scan_count: Scans accumulated into the map.
        rejected_scan_count: Selected scans left out, with reasons in the metrics.
        aggregation_rule: The explicit aggregation rule; ``None`` when every point
            is a raw measurement.
        spatial_index_kind: Identity of the derived spatial index.
        clock_id: Clock domain of the time range.
        start_time_ns: Acquisition time of the earliest contributing scan.
        end_time_ns: Acquisition time of the latest contributing scan.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes the
            manifest, the README and ``debug/``.
    """

    run_id: GeometricMapRunId
    run_index: int
    sequence_name: str
    map_id: MapId
    map_frame: FrameId
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    trajectory_id: TrajectoryId
    state_estimation_run_id: StateEstimationRunId | None
    calibration_identity: str | None
    configuration_fingerprint: str
    code_version: str | None
    point_count: int
    source_point_count: int
    scan_count: int
    rejected_scan_count: int
    aggregation_rule: str | None
    spatial_index_kind: str
    clock_id: str
    start_time_ns: int
    end_time_ns: int
    debug_level: str
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


def encode_mapping_configuration(
    plan: GeometryInputPlan, aggregation: ScanVoxelPolicy | None
) -> dict[str, Any]:
    """Describe the effective mapping configuration of a run.

    Everything that changes the persisted geometry appears here; the debug level,
    which does not, does not.

    Args:
        plan: The assembled geometry inputs.
        aggregation: The explicit aggregation rule, or ``None`` for raw points.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "map_frame": str(plan.map_frame),
        "body_frame": str(plan.body_frame),
        "pose_lookup": encode_lookup_policy(plan.pose_lookup),
        "motion_correction_policy": encode_motion_correction_policy(plan.motion_correction_policy),
        "aggregation": None
        if aggregation is None
        else {
            "kind": "scan_voxel_centroid",
            "cell_m": aggregation.cell_m,
            "rule": aggregation.rule,
        },
        "packed_point": {"format": PACKED_POINT.format, "size_bytes": PACKED_POINT.size},
        "spatial_index": {
            "kind": SCAN_BOUNDS_INDEX.kind,
            "is_derived": SCAN_BOUNDS_INDEX.is_derived,
        },
    }


def mapping_configuration_fingerprint(
    plan: GeometryInputPlan, aggregation: ScanVoxelPolicy | None
) -> str:
    """Hash the effective mapping configuration.

    Args:
        plan: The assembled geometry inputs.
        aggregation: The explicit aggregation rule, or ``None`` for raw points.

    Returns:
        ``"sha256:<hex>"``: equal for equal configuration, different when the pose
        lookup, the correction policy, the aggregation or the storage format changes.
    """
    payload = json.dumps(encode_mapping_configuration(plan, aggregation), sort_keys=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class GeometricMapArtifactWriter:
    """Builds an immutable Geometric Mapping run artifact on the local filesystem."""

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        run_id: GeometricMapRunId,
        run_index: int,
        debug_level: MapDebugLevel = MapDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            output_dir: The final directory of the artifact. The caller chooses it (in the
                runtime, ``<workspace>/<dataset>/<run>/geometric_mapping``); the writer
                computes no path, creates the directory atomically on finalization and
                refuses to replace one that exists.
            sequence_name: Name of the sequence the run maps.
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
        self,
        *,
        plan: GeometryInputPlan,
        aggregation: ScanVoxelPolicy | None,
        code_version: str | None,
        runtime_s: float | None = None,
        peak_memory_bytes: int | None = None,
    ) -> GeometricMapArtifactManifest:
        """Transform, accumulate and persist a run atomically.

        The geometry is streamed to disk one scan at a time, so a map larger
        than memory can be written.

        Args:
            plan: The assembled geometry inputs.
            aggregation: The explicit aggregation rule, or ``None`` to persist
                every point as a raw measurement.
            code_version: Code revision that builds the map, when known.
            runtime_s: Wall-clock time of the run, when measured. Recorded as a
                metric apart from every quality measure.
            peak_memory_bytes: Peak memory of the run, when measured.

        Returns:
            The manifest of the finalized run.

        Raises:
            MapArtifactError: If already finalized or a run already exists at the
                target path.
            GeometryTransformError: If a scan cannot be transformed.
            AccumulationError: If the scans cannot form one map (for example,
                none produced geometry). Nothing is left on disk.
        """
        if self._finalized:
            raise MapArtifactError("writer already finalized")
        map_id = MapId(f"{self._sequence_name}--{self._run_id}")
        fingerprint = mapping_configuration_fingerprint(plan, aggregation)

        try:
            with AtomicRunDirectory(self._final_dir) as run:
                with run.open_binary(_GEOMETRY) as sink:
                    accumulated = accumulate_plan(
                        plan,
                        map_id=map_id,
                        sink=sink,
                        aggregation=aggregation,
                        configuration_fingerprint=fingerprint,
                        code_version=code_version,
                    )
                self._write_outputs(run, accumulated)
                self._write_lineage(run, plan, code_version)
                self._write_configuration(run, plan, aggregation, fingerprint)
                self._write_environment(run, code_version)
                self._write_metrics(run, plan, accumulated, runtime_s, peak_memory_bytes)
                self._write_debug(run, plan, accumulated)
                run.publish(
                    manifest=self._manifest_record(plan, accumulated, fingerprint, code_version),
                    readme=_render_readme(self._run_id, self._run_index, accumulated),
                )
        except RunDirectoryError as error:
            raise MapArtifactError(str(error)) from error

        self._finalized = True
        return _load_manifest(self._final_dir)

    def _write_outputs(self, run: AtomicRunDirectory, accumulated: AccumulatedMap) -> None:
        run.write_text(
            _SOURCE_INDEX, _lines(encode_scan_record(scan) for scan in accumulated.scans)
        )
        run.write_text(_MAP_METADATA, _json(encode_geometric_map(accumulated.geometric_map)))

    def _write_lineage(
        self, run: AtomicRunDirectory, plan: GeometryInputPlan, code_version: str | None
    ) -> None:
        run.write_text(
            _LINEAGE,
            _json(
                {
                    "sequence_artifact_id": str(plan.sequence_artifact_id),
                    "selection": encode_selection(plan.selection),
                    "selection_id": plan.selection_id,
                    "state_estimation_run_id": None
                    if plan.state_estimation_run_id is None
                    else str(plan.state_estimation_run_id),
                    "trajectory_id": str(plan.trajectory_id),
                    "calibration_identity": plan.calibration_identity,
                    "map_frame": str(plan.map_frame),
                    "body_frame": str(plan.body_frame),
                    "code_version": code_version,
                    "source_observation_ids": [str(item.observation_id) for item in plan.inputs],
                    "rejected_observation_ids": [
                        str(item.observation_id) for item in plan.rejections
                    ],
                }
            ),
        )

    def _write_configuration(
        self,
        run: AtomicRunDirectory,
        plan: GeometryInputPlan,
        aggregation: ScanVoxelPolicy | None,
        fingerprint: str,
    ) -> None:
        run.write_text(
            _CONFIG,
            _json(
                {
                    **encode_mapping_configuration(plan, aggregation),
                    "debug_level": self._debug_level.value,
                    "configuration_fingerprint": fingerprint,
                }
            ),
        )

    def _write_environment(self, run: AtomicRunDirectory, code_version: str | None) -> None:
        run.write_text(
            _ENVIRONMENT,
            _json(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": _installed_version("numpy"),
                    "code_version": code_version,
                }
            ),
        )

    def _write_metrics(
        self,
        run: AtomicRunDirectory,
        plan: GeometryInputPlan,
        accumulated: AccumulatedMap,
        runtime_s: float | None,
        peak_memory_bytes: int | None,
    ) -> None:
        run.write_text(_INPUT_PLAN, _json(_encode_input_plan_metrics(plan)))
        run.write_text(_MAPPING, _json(_encode_mapping_metrics(accumulated)))
        if runtime_s is not None or peak_memory_bytes is not None:
            measured: dict[str, Any] = {}
            if runtime_s is not None:
                measured["runtime_s"] = runtime_s
            if peak_memory_bytes is not None:
                measured["peak_memory_bytes"] = peak_memory_bytes
            run.write_text(_RUNTIME, _json(measured))

    def _write_debug(
        self, run: AtomicRunDirectory, plan: GeometryInputPlan, accumulated: AccumulatedMap
    ) -> None:
        if self._debug_level is MapDebugLevel.NONE:
            return
        scans = accumulated.scans
        run.write_text(
            "debug/scans-by-time.jsonl",
            _lines(
                {
                    "ordinal": scan.ordinal,
                    "observation_id": str(scan.observation_id),
                    "timestamp_ns": scan.acquisition_timestamp.total_nanoseconds(),
                    "geometry_count": scan.geometry_count,
                    "dropped_non_finite_count": scan.dropped_non_finite_count,
                    "bounds": None if scan.bounds is None else encode_bounds(scan.bounds),
                }
                for scan in scans
            ),
            contractual=False,
        )
        run.write_text(
            "debug/trajectory-over-map.csv",
            "ordinal,timestamp_ns,x_m,y_m,z_m\n"
            + "".join(
                f"{ordinal},{item.timestamp.total_nanoseconds()},"
                f"{item.pose.pose.translation_m[0]!r},{item.pose.pose.translation_m[1]!r},"
                f"{item.pose.pose.translation_m[2]!r}\n"
                for ordinal, item in enumerate(plan.inputs)
            ),
            contractual=False,
        )
        run.write_text(
            "debug/bounds-summary.json",
            _json(_encode_bounds_summary(accumulated)),
            contractual=False,
        )
        run.write_text("debug/warnings.jsonl", _lines(_warnings(plan)), contractual=False)

        full = self._debug_level is MapDebugLevel.FULL
        max_scans, per_scan = (
            (_FULL_TRACE_SCANS, _FULL_TRACE_POINTS)
            if full
            else (_STANDARD_TRACE_SCANS, _STANDARD_TRACE_POINTS)
        )
        traces = self._sample_traces(run, accumulated, max_scans=max_scans, per_scan=per_scan)
        run.write_text(
            "debug/selected-transform-traces.jsonl",
            _lines(encode_transform_trace(trace) for trace in traces),
            contractual=False,
        )
        if full:
            run.write_text(
                "debug/trace-residuals.jsonl",
                _lines(
                    {
                        "source_observation_id": str(trace.source_observation_id),
                        "source_point_index": trace.source_point_index,
                        "residual_m": transform_trace_residual_m(trace),
                        "problems": verify_transform_trace(trace),
                    }
                    for trace in traces
                ),
                contractual=False,
            )

    def _sample_traces(
        self,
        run: AtomicRunDirectory,
        accumulated: AccumulatedMap,
        *,
        max_scans: int,
        per_scan: int,
    ) -> list[TransformTrace]:
        """Trace a deterministic sample of persisted points, read back from the payload."""
        indices = _sample_geometry_indices(
            accumulated.scans, max_scans=max_scans, per_scan=per_scan
        )
        with run.written_path(_GEOMETRY).open("rb") as handle:
            payload = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            geometry = PackedGeometry(
                geometric_map=accumulated.geometric_map, scans=accumulated.scans, records=payload
            )
            try:
                map_id = accumulated.geometric_map.map_id
                return [
                    geometry.trace(
                        GeometryReference(
                            map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index)
                        )
                    )
                    for index in indices
                ]
            finally:
                geometry.close()
                payload.close()

    def _manifest_record(
        self,
        plan: GeometryInputPlan,
        accumulated: AccumulatedMap,
        fingerprint: str,
        code_version: str | None,
    ) -> dict[str, Any]:
        geometric_map = accumulated.geometric_map
        return {
            "run_id": str(self._run_id),
            "run_index": self._run_index,
            "sequence_name": self._sequence_name,
            "map_id": str(geometric_map.map_id),
            "map_frame": str(geometric_map.frame_id),
            "sequence_artifact_id": str(plan.sequence_artifact_id),
            "selection_id": plan.selection_id,
            "trajectory_id": str(plan.trajectory_id),
            "state_estimation_run_id": None
            if plan.state_estimation_run_id is None
            else str(plan.state_estimation_run_id),
            "calibration_identity": plan.calibration_identity,
            "configuration_fingerprint": fingerprint,
            "code_version": code_version,
            "point_count": geometric_map.point_count,
            "source_point_count": accumulated.source_point_count,
            "scan_count": len(accumulated.scans),
            "rejected_scan_count": len(plan.rejections),
            "aggregation_rule": geometric_map.aggregation_rule,
            "spatial_index_kind": SCAN_BOUNDS_INDEX.kind,
            "clock_id": geometric_map.time_bounds.start.clock_id,
            "start_time_ns": geometric_map.time_bounds.start.total_nanoseconds(),
            "end_time_ns": geometric_map.time_bounds.end.total_nanoseconds(),
            "debug_level": self._debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }


class GeometricMapArtifactReader:
    """Reads a finalized Geometric Mapping run from its own directory.

    Use it as a context manager (or call :meth:`close`) so the memory map of the
    geometry payload is released.
    """

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file
                outside it is needed.

        Raises:
            IncompleteMapArtifactError: If ``manifest.json`` is missing.
            MapArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = _load_manifest(run_dir)
        self._geometry: PackedGeometry | None = None
        self._payload: BinaryIO | None = None
        self._map: mmap.mmap | None = None

    def __enter__(self) -> GeometricMapArtifactReader:
        """Return the reader for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the geometry payload."""
        self.close()

    @property
    def manifest(self) -> GeometricMapArtifactManifest:
        """The run's manifest."""
        return self._manifest

    def geometry(self) -> PackedGeometry:
        """Open the map's geometry as a :class:`~contextmap.geometric_mapping.GeometrySource`.

        The payload is memory-mapped, not read.

        Returns:
            The geometry, backed by the persisted payload and source index.

        Raises:
            MapArtifactError: If a contractual file is missing or the payload does
                not match the map's metadata (for example, it is truncated).
        """
        if self._geometry is not None:
            return self._geometry
        for relative in (_GEOMETRY, _SOURCE_INDEX, _MAP_METADATA):
            if not (self._root / relative).is_file():
                raise MapArtifactError(f"missing {relative} in {self._root}")
        geometric_map = decode_geometric_map(_read_json(self._root / _MAP_METADATA))
        scans = [
            decode_scan_record(json.loads(line))
            for line in (self._root / _SOURCE_INDEX).read_text(encoding="utf-8").splitlines()
            if line
        ]
        handle = (self._root / _GEOMETRY).open("rb")
        try:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as error:
            handle.close()
            raise MapArtifactError(f"{_GEOMETRY} cannot be mapped: {error}") from error
        try:
            geometry = PackedGeometry(geometric_map=geometric_map, scans=scans, records=mapped)
        except ValueError as error:
            mapped.close()
            handle.close()
            raise MapArtifactError(str(error)) from error
        self._payload, self._map, self._geometry = handle, mapped, geometry
        return geometry

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a contractual JSON record.

        Args:
            relative_path: For example ``"metrics/mapping.json"`` or ``"lineage.json"``.

        Returns:
            The parsed record.

        Raises:
            MapArtifactError: If the path is not a contractual JSON record;
                ``debug/`` and the geometry payload are never valid sources.
        """
        contractual = relative_path in _CONTRACTUAL_RECORDS or (
            relative_path.endswith(".json") and relative_path.startswith(_CONTRACTUAL_DIRECTORIES)
        )
        if not contractual or ".." in relative_path:
            raise MapArtifactError(f"not a contractual JSON record: {relative_path!r}")
        return _read_json(self._root / relative_path)

    def verify_integrity(self, *, check_index: bool = True) -> list[str]:
        """Check the file inventory and the derived index against what is on disk.

        Args:
            check_index: Also recompute the derived bounds from the geometry and
                compare them with the recorded ones; this reads the whole payload.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        problems = check_file_inventory(self._root, self._manifest.file_inventory)
        if check_index:
            try:
                problems.extend(self.geometry().verify_index())
            except MapArtifactError as error:
                problems.append(str(error))
        return problems

    def close(self) -> None:
        """Release the geometry payload; the geometry must not be read afterwards."""
        if self._geometry is not None:
            self._geometry.close()
        if self._map is not None:
            self._map.close()
        if self._payload is not None:
            self._payload.close()
        self._geometry = self._map = self._payload = None


def _load_manifest(run_dir: Path) -> GeometricMapArtifactManifest:
    manifest_path = run_dir / _MANIFEST
    if not manifest_path.is_file():
        raise IncompleteMapArtifactError(f"missing {_MANIFEST} in {run_dir}")
    raw = _read_json(manifest_path)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise MapArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}"
        )
    run_id = raw["state_estimation_run_id"]
    return GeometricMapArtifactManifest(
        run_id=GeometricMapRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        map_id=MapId(raw["map_id"]),
        map_frame=FrameId(raw["map_frame"]),
        sequence_artifact_id=SequenceArtifactId(raw["sequence_artifact_id"]),
        selection_id=raw["selection_id"],
        trajectory_id=TrajectoryId(raw["trajectory_id"]),
        state_estimation_run_id=None if run_id is None else StateEstimationRunId(run_id),
        calibration_identity=raw["calibration_identity"],
        configuration_fingerprint=raw["configuration_fingerprint"],
        code_version=raw["code_version"],
        point_count=raw["point_count"],
        source_point_count=raw["source_point_count"],
        scan_count=raw["scan_count"],
        rejected_scan_count=raw["rejected_scan_count"],
        aggregation_rule=raw["aggregation_rule"],
        spatial_index_kind=raw["spatial_index_kind"],
        clock_id=raw["clock_id"],
        start_time_ns=raw["start_time_ns"],
        end_time_ns=raw["end_time_ns"],
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


def _read_json(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


def _json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def _lines(records: Iterable[Any]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def _installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _encode_input_plan_metrics(plan: GeometryInputPlan) -> dict[str, Any]:
    lookups = summarize_lookups(item.pose for item in plan.inputs)
    states = Counter(item.motion_correction.record.state.value for item in plan.inputs)
    dispositions = Counter(item.motion_correction.disposition.value for item in plan.inputs)
    pose_rejections = sum(
        1 for item in plan.rejections if item.reason.value == "pose_lookup_rejected"
    )
    return {
        "selected_scan_count": len(plan.inputs) + len(plan.rejections),
        "accepted_scan_count": len(plan.inputs),
        "rejected_scan_count": len(plan.rejections),
        "rejections": [
            {
                "observation_id": str(item.observation_id),
                "reason": item.reason.value,
                "detail": item.detail,
            }
            for item in plan.rejections
        ],
        "ignored_observation_counts": dict(plan.ignored_observation_counts),
        "pose_lookup": {
            "lookup_count": lookups.lookup_count,
            "exact_count": lookups.exact_count,
            "nearest_count": lookups.nearest_count,
            "interpolated_count": lookups.interpolated_count,
            "min_time_delta_ns": lookups.min_time_delta_ns,
            "median_time_delta_ns": lookups.median_time_delta_ns,
            "max_time_delta_ns": lookups.max_time_delta_ns,
        },
        "pose_lookup_rejected_count": pose_rejections,
        "motion_correction": {
            "policy": encode_motion_correction_policy(plan.motion_correction_policy),
            "state_counts": dict(states),
            "disposition_counts": dict(dispositions),
            "warnings": [
                item.motion_correction.message
                for item in plan.inputs
                if item.motion_correction.message is not None
            ],
        },
    }


def _encode_mapping_metrics(accumulated: AccumulatedMap) -> dict[str, Any]:
    geometric_map = accumulated.geometric_map
    return {
        "map_id": str(geometric_map.map_id),
        "scan_count": len(accumulated.scans),
        "scans_without_geometry": sum(1 for scan in accumulated.scans if scan.geometry_count == 0),
        "source_point_count": accumulated.source_point_count,
        "dropped_non_finite_count": sum(
            scan.dropped_non_finite_count for scan in accumulated.scans
        ),
        "point_count": geometric_map.point_count,
        "aggregation_rule": geometric_map.aggregation_rule,
        "reduction_ratio": geometric_map.point_count / accumulated.source_point_count,
        "bounds": encode_bounds(geometric_map.bounds),
        "time_bounds": {
            "start": geometric_map.time_bounds.start.to_record(),
            "end": geometric_map.time_bounds.end.to_record(),
        },
        "geometry_size_bytes": geometric_map.point_count * PACKED_POINT.size,
    }


def _encode_bounds_summary(accumulated: AccumulatedMap) -> dict[str, Any]:
    geometric_map = accumulated.geometric_map
    low, high = geometric_map.bounds.minimum_m, geometric_map.bounds.maximum_m
    counts = sorted(scan.geometry_count for scan in accumulated.scans)
    return {
        "map_bounds": encode_bounds(geometric_map.bounds),
        "extent_m": [high[axis] - low[axis] for axis in range(3)],
        "point_count": geometric_map.point_count,
        "points_per_scan": {
            "minimum": counts[0],
            "median": counts[len(counts) // 2],
            "maximum": counts[-1],
        },
    }


def _warnings(plan: GeometryInputPlan) -> Iterator[dict[str, Any]]:
    for rejection in plan.rejections:
        yield {
            "kind": "rejected_scan",
            "observation_id": str(rejection.observation_id),
            "reason": rejection.reason.value,
            "detail": rejection.detail,
        }
    for item in plan.inputs:
        if item.motion_correction.message is not None:
            yield {
                "kind": "motion_correction",
                "observation_id": str(item.observation_id),
                "message": item.motion_correction.message,
            }


def _sample_geometry_indices(
    scans: Sequence[ScanRecord], *, max_scans: int, per_scan: int
) -> list[int]:
    """Pick geometry indices evenly across the scans that contributed geometry."""
    contributing = [scan for scan in scans if scan.geometry_count > 0]
    count = min(len(contributing), max_scans)
    if count == 0:
        return []
    positions = sorted(
        {round(i * (len(contributing) - 1) / max(count - 1, 1)) for i in range(count)}
    )
    indices: list[int] = []
    for position in positions:
        scan = contributing[position]
        last = scan.geometry_count - 1
        offsets = {0, last // 2, last} if per_scan >= 3 else {0, last} if per_scan == 2 else {0}
        indices.extend(scan.first_geometry_index + offset for offset in sorted(offsets))
    return indices


def _render_readme(run_id: GeometricMapRunId, run_index: int, accumulated: AccumulatedMap) -> str:
    geometric_map: GeometricMap = accumulated.geometric_map
    provenance = geometric_map.provenance
    return (
        f"# Geometric mapping run {run_index:04d}\n"
        "\n"
        f"- Run ID: `{run_id}`\n"
        f"- Map: `{geometric_map.map_id}` in frame `{geometric_map.frame_id}`\n"
        f"- Sequence artifact: `{provenance.sequence_artifact_id}`\n"
        f"- Selection: `{provenance.selection_id}`\n"
        f"- Trajectory: `{provenance.trajectory_id}`\n"
        f"- Points: {geometric_map.point_count} from {len(accumulated.scans)} scans\n"
        "\n"
        "Contractual data is in `outputs/`, `metrics/`, `lineage.json`, `config.json` and "
        "`environment.json`; `debug/` is human evidence and no downstream stage may depend on it.\n"
    )

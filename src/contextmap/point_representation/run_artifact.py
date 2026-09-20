"""Persisted, immutable, self-describing Point Representation run artifacts.

A ``PointRepresentationRunArtifact`` is a directory holding the representations
one encoder produced over one geometric map, with the lineage, metrics and
failed supports needed to trust them. It opens without NumPy, a model library or
the geometric map itself, and a single representation or vector can be read by
identity or by the geometry it is anchored to without loading the others. See
``src/contextmap/point_representation/docs/artifact.md`` for the layout and the
trade-offs of this first schema.

Contractual data lives in ``outputs/`` and ``metrics/`` and is inventoried with
size and SHA-256 in the manifest; ``debug/`` is human evidence that is never
inventoried, so removing it cannot invalidate the run and no downstream stage
may depend on it. Writing follows :class:`~contextmap.shared.AtomicRunDirectory`:
an interrupted write never looks like a finished run and a finished run is never
modified.

The artifact does not copy the geometric map (only the references and the
lineage that identify it), nor any visual feature or semantic claim, and a
representation is persisted as evidence, not as semantic truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.geometric_mapping import GeometricMap, GeometryId, GeometryReference, MapId
from contextmap.point_representation.compatibility import representation_space_fingerprint
from contextmap.point_representation.models import (
    EncoderIdentity,
    FailedSupport,
    FailureReason,
    PointRepresentation,
    PointRepresentationId,
    PointRepresentationRunId,
    RepresentationSpace,
    SupportPolicy,
)
from contextmap.point_representation.serialization import (
    decode_failed_support,
    decode_point_representation,
    decode_representation_space,
    decode_support_policy,
    encode_failed_support,
    encode_point_representation,
    encode_representation_space,
    encode_support_policy,
)
from contextmap.point_representation.service import EncodedRepresentation, RepresentationMetrics
from contextmap.shared import (
    AtomicRunDirectory,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
    next_run_index,
    write_run_registry,
)

SCHEMA_VERSION = "0.1.0"
"""Point Representation run artifact schema version written and understood by this module."""

_MANIFEST = "manifest.json"
_REPRESENTATIONS = "outputs/representations.jsonl"
_REPRESENTATION_INDEX = "outputs/representation-index.jsonl"
_SPACES = "outputs/representation-spaces.json"
_GEOMETRY_INDEX = "outputs/geometry-representation-index.jsonl"
_FAILED_SUPPORTS = "outputs/failed-supports.jsonl"
_COUNTS = "metrics/counts.json"
_SUPPORT_SIZE = "metrics/support-size.json"
_NORMS = "metrics/norms.json"
_RUNTIME = "metrics/runtime.json"

_DTYPES = {"float32": ("f", 4, "f32"), "float64": ("d", 8, "f64")}
"""``struct`` code, item size in bytes and file suffix per payload dtype."""

_TRACE_LIMIT = 20
"""Support traces kept at the ``standard`` and ``full`` debug levels."""


class RunArtifactError(Exception):
    """Base class for Point Representation run artifact read/write failures."""


class IncompleteRunArtifactError(RunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class PointRepresentationDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and required metrics only.
        STANDARD: Adds representative support traces and a norm per representation.
        FULL: Adds per-component statistics of the stored vectors.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, kw_only=True)
class PointRepresentationRunManifest:
    """Authoritative metadata of a persisted Point Representation run.

    Attributes:
        run_id: Identity of the run.
        run_index: Monotonic index within this sequence's point-representation runs.
        sequence_name: Name of the processed sequence.
        geometric_map_id: The immutable geometric map the geometry came from.
        geometric_map_frame: Frame of that map.
        geometric_map_point_count: Elements in that map.
        sequence_artifact_id: Canonical sequence the map was built from.
        geometry_selection_id: Selection identity the map was built for.
        trajectory_id: Trajectory whose poses placed the map's geometry.
        association_context_id: Association run used as selection/context input
            when explicitly given, else ``None``; never a hidden feature source.
        center_selection_id: Identity of the set of requested centers,
            independent of request order.
        support_policy: The support policy of the run, with its coordinate
            preparation.
        representation_space_id: Fingerprint of the run's representation space.
        dimension: Vector dimensionality.
        dtype: Payload element type.
        payload_path: Run-relative path of the vector payload.
        encoder: Encoder identity, including the checkpoint hash when learned.
        code_version: Code revision that produced the run.
        representation_count: Representations persisted, partial ones included.
        partial_count: Representations with undefined components.
        failed_count: Supports that produced no representation.
        failed_by_reason: Failed supports per reason.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes
            the manifest, the README and ``debug/``.
    """

    run_id: PointRepresentationRunId
    run_index: int
    sequence_name: str
    geometric_map_id: MapId
    geometric_map_frame: str
    geometric_map_point_count: int
    sequence_artifact_id: str
    geometry_selection_id: str
    trajectory_id: str
    association_context_id: str | None
    center_selection_id: str
    support_policy: SupportPolicy
    representation_space_id: str
    dimension: int
    dtype: str
    payload_path: str
    encoder: EncoderIdentity
    code_version: str
    representation_count: int
    partial_count: int
    failed_count: int
    failed_by_reason: Mapping[str, int]
    debug_level: str
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


def _sequence_dir(workspace_root: Path, sequence_name: str) -> Path:
    return workspace_root / "runs" / "point-representation" / sequence_name


class PointRepresentationRunWriter:
    """Builds an immutable Point Representation run artifact on the local filesystem.

    Outcomes are added one at a time, as the execution service streams them, and
    the run is written atomically by :meth:`finalize`. The writer holds the
    encoded records in memory until then, so a run's size is bounded by memory:
    ``shared.AtomicRunDirectory`` writes whole files.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        sequence_name: str,
        run_id: PointRepresentationRunId,
        run_index: int,
        selection_label: str,
        backend_label: str,
        geometric_map: GeometricMap,
        space: RepresentationSpace,
        encoder_identity: EncoderIdentity,
        code_version: str,
        association_context_id: str | None = None,
        debug_level: PointRepresentationDebugLevel = PointRepresentationDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            workspace_root: Root of the local workspace.
            sequence_name: Name of the sequence the run processed.
            run_id: Identity of the run.
            run_index: Monotonic index for this sequence's point-representation
                runs (see :func:`allocate_run_index`).
            selection_label: Short readable selection description for the
                directory name, e.g. ``"voxel-5cm"``.
            backend_label: Short readable backend description for the directory
                name, e.g. ``"geometric-descriptor"``.
            geometric_map: The map the geometry came from; only its identity
                and lineage are recorded, never its points.
            space: The run's representation space.
            encoder_identity: The encoder that produced the run.
            code_version: Code revision that produced the run.
            association_context_id: Identity of an association run explicitly
                used as selection/context input, if any.
            debug_level: Amount of non-contractual debug evidence to persist.

        Raises:
            RunArtifactError: If ``association_context_id`` is given but empty.
        """
        if association_context_id is not None and not association_context_id:
            raise RunArtifactError("association_context_id must be None or a non-empty identity")
        self._workspace_root = workspace_root
        self._sequence_name = sequence_name
        self._run_id = run_id
        self._run_index = run_index
        self._map = geometric_map
        self._space = space
        self._space_id = representation_space_fingerprint(space)
        self._encoder = encoder_identity
        self._code_version = code_version
        self._association_context_id = association_context_id
        self._debug_level = debug_level
        self._final_dir = _sequence_dir(workspace_root, sequence_name) / (
            f"run-{run_index:04d}__{selection_label}__{backend_label}"
        )
        code, self._item_size, suffix = _DTYPES[space.dtype]
        self._row = struct.Struct(f"<{space.dimension}{code}")
        self._payload_path = f"outputs/payloads/vectors.{suffix}"
        self._payload = bytearray()
        self._records: list[bytes] = []
        self._index_lines: list[str] = []
        self._geometry_lines: list[str] = []
        self._failed_lines: list[str] = []
        self._offset = 0
        self._centers: set[GeometryId] = set()
        self._representation_ids: set[PointRepresentationId] = set()
        self._partial = 0
        self._failed_by_reason: dict[FailureReason, int] = {}
        self._support_sizes: list[int] = []
        self._norms: list[tuple[PointRepresentationId, float]] = []
        self._traces: list[dict[str, Any]] = []
        self._component_stats = _ComponentStatistics(space.dimension)
        self._finalized = False

    def add(self, outcome: EncodedRepresentation | FailedSupport) -> None:
        """Add one outcome of the execution service.

        Args:
            outcome: A representation with its vector, or a failed support.

        Raises:
            RunArtifactError: If the writer is finalized, the outcome belongs to
                another map, space or encoder, its center already has an outcome,
                or its vector breaks the space (dimension, non-finite values, a
                value that does not fit the payload dtype).
        """
        self._require_open()
        if isinstance(outcome, FailedSupport):
            self._add_failed(outcome)
        else:
            self._add_represented(outcome)

    def finalize(
        self,
        *,
        metrics: RepresentationMetrics,
        backend_diagnostics: Mapping[str, str | int | float | None] | None = None,
    ) -> PointRepresentationRunManifest:
        """Persist a completed run atomically.

        Args:
            metrics: The execution service's metrics for the run; they must
                match the outcomes added, so a stream that stopped early is
                never persisted as a complete run.
            backend_diagnostics: Backend-specific runtime figures, for example
                peak device memory, recorded apart from every quality measure.

        Returns:
            The manifest of the finalized run.

        Raises:
            RunArtifactError: If already finalized, if the outcomes do not match
                the metrics, or if a run already exists at the target path.
        """
        self._require_open()
        self._require_complete(metrics)
        try:
            with AtomicRunDirectory(self._final_dir) as run:
                self._write_outputs(run)
                self._write_metrics(run, metrics, backend_diagnostics)
                self._write_debug(run)
                run.publish(manifest=self._manifest_record(), readme=self._render_readme())
        except RunDirectoryError as error:
            raise RunArtifactError(str(error)) from error
        self._finalized = True
        rebuild_run_registry(workspace_root=self._workspace_root, sequence_name=self._sequence_name)
        return _load_manifest(self._final_dir)

    def _require_open(self) -> None:
        if self._finalized:
            raise RunArtifactError("writer already finalized")

    def _require_complete(self, metrics: RepresentationMetrics) -> None:
        failed = sum(self._failed_by_reason.values())
        represented = len(self._representation_ids)
        observed = (represented + failed, represented, self._partial, self._failed_by_reason)
        reported = (
            metrics.requested,
            metrics.represented,
            metrics.partial,
            dict(metrics.failed_by_reason),
        )
        if observed != reported:
            raise RunArtifactError(
                "incomplete run: the outcomes added (requested/represented/partial/failed by "
                f"reason: {observed}) do not match the run metrics ({reported})"
            )

    def _require_unclaimed_center(self, center: GeometryReference) -> None:
        if center.map_id != self._map.map_id:
            raise RunArtifactError(
                f"center {center.geometry_id!r} belongs to map {center.map_id!r}, but this run "
                f"is over map {self._map.map_id!r}"
            )
        if center.geometry_id in self._centers:
            raise RunArtifactError(f"center {center.geometry_id!r} already has an outcome")

    def _add_failed(self, failed: FailedSupport) -> None:
        self._require_unclaimed_center(failed.support.center)
        self._centers.add(failed.support.center.geometry_id)
        self._failed_lines.append(json.dumps(encode_failed_support(failed), sort_keys=True))
        self._failed_by_reason[failed.reason] = self._failed_by_reason.get(failed.reason, 0) + 1
        self._support_sizes.append(failed.support.statistics.count)
        if self._debug_level is not PointRepresentationDebugLevel.NONE:
            self._traces.append(
                {
                    "outcome": "failed",
                    "reason": failed.reason.value,
                    "center": str(failed.support.center.geometry_id),
                    "member_count": failed.support.statistics.count,
                }
            )

    def _add_represented(self, outcome: EncodedRepresentation) -> None:
        representation = outcome.representation
        self._require_matches_run(representation)
        values = self._require_storable_vector(outcome.values)
        self._require_unclaimed_center(representation.geometry_reference)
        if representation.representation_id in self._representation_ids:
            raise RunArtifactError(
                f"representation {representation.representation_id!r} already has an outcome"
            )
        try:
            packed = self._row.pack(*values)
        except (struct.error, OverflowError) as error:
            raise RunArtifactError(
                f"a vector value does not fit {self._space.dtype} and would be stored as "
                f"infinity: {error}"
            ) from error
        # Tudo foi validado: só agora o estado do writer muda, então um add recusado é repetível.
        self._centers.add(representation.geometry_reference.geometry_id)
        row = len(self._representation_ids)
        stored = replace(representation, payload_reference=f"{self._payload_path}#{row}")
        line = json.dumps(encode_point_representation(stored), sort_keys=True).encode("utf-8")
        self._payload += packed
        self._records.append(line)
        self._index_lines.append(
            json.dumps(
                {
                    "representation_id": str(stored.representation_id),
                    "byte_offset": self._offset,
                    "byte_length": len(line),
                },
                sort_keys=True,
            )
        )
        self._offset += len(line) + 1
        self._geometry_lines.append(
            json.dumps(
                {
                    "map_id": str(stored.geometry_reference.map_id),
                    "geometry_id": str(stored.geometry_reference.geometry_id),
                    "representation_id": str(stored.representation_id),
                },
                sort_keys=True,
            )
        )
        self._representation_ids.add(stored.representation_id)
        self._observe(stored, values)

    def _observe(self, representation: PointRepresentation, values: tuple[float, ...]) -> None:
        undefined = set(representation.undefined_components)
        defined = [value for index, value in enumerate(values) if index not in undefined]
        self._partial += representation.is_partial
        self._support_sizes.append(representation.support.statistics.count)
        if defined:
            self._norms.append((representation.representation_id, math.hypot(*defined)))
        if self._debug_level is PointRepresentationDebugLevel.FULL:
            self._component_stats.observe(values, undefined)
        if self._debug_level is not PointRepresentationDebugLevel.NONE:
            support_stats = representation.support.statistics
            self._traces.append(
                {
                    "outcome": "represented",
                    "representation_id": str(representation.representation_id),
                    "center": str(representation.geometry_reference.geometry_id),
                    "member_count": support_stats.count,
                    "max_distance_m": support_stats.max_distance_m,
                    "near_map_bounds": support_stats.near_map_bounds,
                    "undefined_components": list(representation.undefined_components),
                }
            )

    def _require_matches_run(self, representation: PointRepresentation) -> None:
        if representation.representation_space_id != self._space_id:
            raise RunArtifactError(
                "the representation space of this representation is not the run's: "
                f"{representation.representation_space_id} != {self._space_id}"
            )
        if representation.encoder_identity != self._encoder:
            raise RunArtifactError("this representation was produced by another encoder")
        if (
            representation.shape != (self._space.dimension,)
            or representation.dtype != self._space.dtype
            or representation.normalization != self._space.normalization
        ):
            raise RunArtifactError(
                "this representation's shape, dtype or normalization does not match its space"
            )
        if representation.payload_reference is not None:
            raise RunArtifactError(
                "the payload_reference is assigned by the writer and must not be set beforehand"
            )

    def _require_storable_vector(self, values: tuple[float, ...]) -> tuple[float, ...]:
        if len(values) != self._space.dimension:
            raise RunArtifactError(
                f"the vector has {len(values)} values but the space is "
                f"{self._space.dimension}-dimensional"
            )
        if not all(math.isfinite(value) for value in values):
            raise RunArtifactError("the vector holds a non-finite value, which is never stored")
        return values

    def _write_outputs(self, run: AtomicRunDirectory) -> None:
        run.write_bytes(
            _REPRESENTATIONS, b"\n".join(self._records) + (b"\n" if self._records else b"")
        )
        run.write_text(_REPRESENTATION_INDEX, _lines_text(self._index_lines))
        run.write_text(
            _SPACES,
            _json(
                {
                    "spaces": [
                        {
                            "representation_space_id": self._space_id,
                            **encode_representation_space(self._space),
                        }
                    ]
                }
            ),
        )
        run.write_text(_GEOMETRY_INDEX, _lines_text(self._geometry_lines))
        run.write_text(_FAILED_SUPPORTS, _lines_text(self._failed_lines))
        run.write_bytes(self._payload_path, bytes(self._payload))

    def _write_metrics(
        self,
        run: AtomicRunDirectory,
        metrics: RepresentationMetrics,
        backend_diagnostics: Mapping[str, str | int | float | None] | None,
    ) -> None:
        run.write_text(
            _COUNTS,
            _json(
                {
                    "requested": metrics.requested,
                    "represented": metrics.represented,
                    "partial": metrics.partial,
                    "failed": metrics.failed,
                    "failed_by_reason": _reasons(self._failed_by_reason),
                }
            ),
        )
        run.write_text(_SUPPORT_SIZE, _json(_distribution(self._support_sizes)))
        run.write_text(
            _NORMS,
            _json({**_distribution([norm for _, norm in self._norms]), "non_finite_values": 0}),
        )
        run.write_text(
            _RUNTIME,
            _json(
                {
                    "support_extraction_seconds": metrics.support_extraction_seconds,
                    "encoding_seconds": metrics.encoding_seconds,
                    "backend": dict(backend_diagnostics or {}),
                }
            ),
        )

    def _write_debug(self, run: AtomicRunDirectory) -> None:
        if self._debug_level is PointRepresentationDebugLevel.NONE:
            return
        step = max(1, len(self._traces) // _TRACE_LIMIT)
        run.write_text(
            "debug/support-traces.jsonl",
            _lines_text(
                json.dumps(trace, sort_keys=True) for trace in self._traces[::step][:_TRACE_LIMIT]
            ),
            contractual=False,
        )
        run.write_text(
            "debug/representation-norms.csv",
            "".join(
                ["representation_id,norm\n"]
                + [f"{identity},{norm!r}\n" for identity, norm in self._norms]
            ),
            contractual=False,
        )
        if self._debug_level is PointRepresentationDebugLevel.FULL:
            run.write_text(
                "debug/component-statistics.json",
                _json({"components": self._component_stats.summary(self._space.feature_names)}),
                contractual=False,
            )

    def _manifest_record(self) -> dict[str, Any]:
        provenance = self._map.provenance
        return {
            "run_id": str(self._run_id),
            "run_index": self._run_index,
            "sequence_name": self._sequence_name,
            "geometric_map": {
                "map_id": str(self._map.map_id),
                "frame_id": str(self._map.frame_id),
                "point_count": self._map.point_count,
                "sequence_artifact_id": str(provenance.sequence_artifact_id),
                "selection_id": provenance.selection_id,
                "trajectory_id": str(provenance.trajectory_id),
            },
            "association_context_id": self._association_context_id,
            "center_selection_id": center_selection_id(
                map_id=self._map.map_id,
                centers=(
                    GeometryReference(map_id=self._map.map_id, geometry_id=geometry_id)
                    for geometry_id in self._centers
                ),
            ),
            "support_policy": encode_support_policy(self._space.support_semantics),
            "representation_space_id": self._space_id,
            "dimension": self._space.dimension,
            "dtype": self._space.dtype,
            "payload_path": self._payload_path,
            "encoder": {
                "backend_id": self._encoder.backend_id,
                "backend_version": self._encoder.backend_version,
                "configuration_fingerprint": self._encoder.configuration_fingerprint,
                "checkpoint_hash": self._encoder.checkpoint_hash,
            },
            "code_version": self._code_version,
            "representation_count": len(self._representation_ids),
            "partial_count": self._partial,
            "failed_count": sum(self._failed_by_reason.values()),
            "failed_by_reason": _reasons(self._failed_by_reason),
            "debug_level": self._debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _render_readme(self) -> str:
        return (
            f"# Point representation run {self._run_index:04d}\n"
            "\n"
            f"- Run ID: `{self._run_id}`\n"
            f"- Geometric map: `{self._map.map_id}`\n"
            f"- Encoder: `{self._encoder.backend_id}` (version `{self._encoder.backend_version}`)\n"
            f"- Representation space: `{self._space_id}`\n"
            f"- Representations: {len(self._representation_ids)} "
            f"({self._partial} partial), failed supports: {sum(self._failed_by_reason.values())}\n"
            "\n"
            "Contractual data is in `outputs/` and `metrics/`; `debug/` is human evidence and no "
            "downstream stage may depend on it. A representation is 3D evidence, not semantic "
            "truth.\n"
        )


class PointRepresentationRunReader:
    """Reads a finalized Point Representation run from its own directory."""

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
        self._offsets: dict[PointRepresentationId, tuple[int, int]] | None = None
        self._by_geometry: dict[GeometryReference, PointRepresentationId] | None = None

    @property
    def manifest(self) -> PointRepresentationRunManifest:
        """The run's manifest."""
        return self._manifest

    def space(self) -> RepresentationSpace:
        """Load the run's representation space, checked against its fingerprint.

        Raises:
            RunArtifactError: If the persisted space does not match the manifest.
        """
        record = json.loads((self._root / _SPACES).read_text(encoding="utf-8"))
        space = decode_representation_space(record["spaces"][0])
        if representation_space_fingerprint(space) != self._manifest.representation_space_id:
            raise RunArtifactError("the persisted representation space does not match the manifest")
        return space

    def representation(self, representation_id: PointRepresentationId) -> PointRepresentation:
        """Read one representation by identity without loading the others.

        Raises:
            RunArtifactError: If the run has no such representation.
        """
        entry = self._load_offsets().get(representation_id)
        if entry is None:
            raise RunArtifactError(f"unknown representation_id in this run: {representation_id!r}")
        offset, length = entry
        with (self._root / _REPRESENTATIONS).open("rb") as handle:
            handle.seek(offset)
            return self._decode(handle.read(length))

    def representation_for(self, geometry: GeometryReference) -> PointRepresentation | None:
        """Read the representation anchored to a geometry element, if the run has one."""
        representation_id = self._load_geometry_index().get(geometry)
        return None if representation_id is None else self.representation(representation_id)

    def iter_representations(self) -> Iterator[PointRepresentation]:
        """Iterate every representation in the order they were persisted."""
        with (self._root / _REPRESENTATIONS).open("rb") as handle:
            for line in handle:
                if line.strip():
                    yield self._decode(line)

    def vector(self, representation_id: PointRepresentationId) -> tuple[float, ...]:
        """Read one representation's vector without reading the rest of the payload.

        Components the representation lists as undefined hold zero placeholders
        and must not be interpreted.

        Raises:
            RunArtifactError: If the run has no such representation, its
                ``payload_reference`` does not point into this run's payload, or
                the payload is truncated.
        """
        representation = self.representation(representation_id)
        path, _, row_text = (representation.payload_reference or "").partition("#")
        if path != self._manifest.payload_path or not row_text.isdigit():
            raise RunArtifactError(
                f"payload_reference {representation.payload_reference!r} does not point into "
                f"this run's payload {self._manifest.payload_path!r}"
            )
        code, item_size, _ = _DTYPES[self._manifest.dtype]
        size = item_size * self._manifest.dimension
        with (self._root / self._manifest.payload_path).open("rb") as handle:
            handle.seek(int(row_text) * size)
            data = handle.read(size)
        if len(data) != size:
            raise RunArtifactError(
                f"payload truncated: row {row_text} of {self._manifest.payload_path} needs "
                f"{size} bytes, found {len(data)}"
            )
        return tuple(
            float(value) for value in struct.unpack(f"<{self._manifest.dimension}{code}", data)
        )

    def failed_supports(self) -> list[FailedSupport]:
        """Read every support that produced no representation, with its reason."""
        text = (self._root / _FAILED_SUPPORTS).read_text(encoding="utf-8")
        return [decode_failed_support(json.loads(line)) for line in text.splitlines() if line]

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``outputs/`` or ``metrics/``.

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

    def _decode(self, line: bytes) -> PointRepresentation:
        representation = decode_point_representation(json.loads(line))
        if representation.representation_space_id != self._manifest.representation_space_id:
            raise RunArtifactError(
                f"representation {representation.representation_id!r} is not in the run's space"
            )
        return representation

    def _load_offsets(self) -> dict[PointRepresentationId, tuple[int, int]]:
        if self._offsets is None:
            offsets: dict[PointRepresentationId, tuple[int, int]] = {}
            text = (self._root / _REPRESENTATION_INDEX).read_text(encoding="utf-8")
            for line in text.splitlines():
                record = json.loads(line)
                offsets[PointRepresentationId(record["representation_id"])] = (
                    record["byte_offset"],
                    record["byte_length"],
                )
            self._offsets = offsets
        return self._offsets

    def _load_geometry_index(self) -> dict[GeometryReference, PointRepresentationId]:
        if self._by_geometry is None:
            index: dict[GeometryReference, PointRepresentationId] = {}
            text = (self._root / _GEOMETRY_INDEX).read_text(encoding="utf-8")
            for line in text.splitlines():
                record = json.loads(line)
                reference = GeometryReference(
                    map_id=MapId(record["map_id"]), geometry_id=GeometryId(record["geometry_id"])
                )
                index[reference] = PointRepresentationId(record["representation_id"])
            self._by_geometry = index
        return self._by_geometry


def allocate_run_index(*, workspace_root: Path, sequence_name: str) -> int:
    """Compute the next monotonic run index for a sequence's point-representation runs.

    Scans the run directories, never the registry, so an interrupted or
    corrupted run is not counted.

    Args:
        workspace_root: Root of the local workspace.
        sequence_name: Name of the sequence.

    Returns:
        The next index, starting at ``1``.
    """
    return next_run_index(_sequence_dir(workspace_root, sequence_name), index_of=_valid_run_index)


def center_selection_id(*, map_id: MapId, centers: Iterable[GeometryReference]) -> str:
    """Identity of a set of requested centers of one map, independent of request order.

    The manifest records it, so an evaluation that fixes the same centers across
    encoders can prove it compared like with like.

    Args:
        map_id: The geometric map the centers belong to.
        centers: The requested centers, each a reference into ``map_id``.

    Returns:
        ``"sha256:<hex digest>"`` of the map identity and the sorted center ids.
    """
    text = "\n".join([str(map_id), *sorted(str(center.geometry_id) for center in centers)])
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def rebuild_run_registry(*, workspace_root: Path, sequence_name: str) -> None:
    """Rebuild a sequence's ``runs.json`` convenience registry from its valid runs.

    Args:
        workspace_root: Root of the local workspace.
        sequence_name: Name of the sequence.
    """
    write_run_registry(_sequence_dir(workspace_root, sequence_name), describe=_registry_record)


def _valid_run_index(run_dir: Path) -> int | None:
    # Um diretório ilegível ou com manifest malformado simplesmente não é um run válido.
    try:
        reader = PointRepresentationRunReader(run_dir)
    except (RunArtifactError, ValueError, KeyError, OSError):
        return None
    return None if reader.verify_integrity() else reader.manifest.run_index


def _registry_record(run_dir: Path) -> dict[str, Any] | None:
    index = _valid_run_index(run_dir)
    if index is None:
        return None
    manifest = PointRepresentationRunReader(run_dir).manifest
    return {"run_index": index, "run_id": str(manifest.run_id), "directory": run_dir.name}


def _load_manifest(run_dir: Path) -> PointRepresentationRunManifest:
    manifest_path = run_dir / _MANIFEST
    if not manifest_path.is_file():
        raise IncompleteRunArtifactError(f"missing {_MANIFEST} in {run_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RunArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}"
        )
    geometric_map = raw["geometric_map"]
    encoder = raw["encoder"]
    return PointRepresentationRunManifest(
        run_id=PointRepresentationRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        geometric_map_id=MapId(geometric_map["map_id"]),
        geometric_map_frame=geometric_map["frame_id"],
        geometric_map_point_count=geometric_map["point_count"],
        sequence_artifact_id=geometric_map["sequence_artifact_id"],
        geometry_selection_id=geometric_map["selection_id"],
        trajectory_id=geometric_map["trajectory_id"],
        association_context_id=raw["association_context_id"],
        center_selection_id=raw["center_selection_id"],
        support_policy=decode_support_policy(raw["support_policy"]),
        representation_space_id=raw["representation_space_id"],
        dimension=raw["dimension"],
        dtype=raw["dtype"],
        payload_path=raw["payload_path"],
        encoder=EncoderIdentity(
            backend_id=encoder["backend_id"],
            backend_version=encoder["backend_version"],
            configuration_fingerprint=encoder["configuration_fingerprint"],
            checkpoint_hash=encoder["checkpoint_hash"],
        ),
        code_version=raw["code_version"],
        representation_count=raw["representation_count"],
        partial_count=raw["partial_count"],
        failed_count=raw["failed_count"],
        failed_by_reason=dict(raw["failed_by_reason"]),
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


class _ComponentStatistics:
    """Running per-component minimum, maximum and mean over the defined values."""

    def __init__(self, dimension: int) -> None:
        self._count = [0] * dimension
        self._sum = [0.0] * dimension
        self._minimum = [math.inf] * dimension
        self._maximum = [-math.inf] * dimension

    def observe(self, values: tuple[float, ...], undefined: set[int]) -> None:
        for index, value in enumerate(values):
            if index in undefined:
                continue
            self._count[index] += 1
            self._sum[index] += value
            self._minimum[index] = min(self._minimum[index], value)
            self._maximum[index] = max(self._maximum[index], value)

    def summary(self, feature_names: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "name": feature_names[index] if feature_names else None,
                "defined_count": count,
                "minimum": self._minimum[index] if count else None,
                "maximum": self._maximum[index] if count else None,
                "mean": self._sum[index] / count if count else None,
            }
            for index, count in enumerate(self._count)
        ]


def _distribution(values: list[int] | list[float]) -> dict[str, Any]:
    """Count, minimum, median, 95th percentile (nearest rank) and maximum."""
    if not values:
        return {"count": 0, "minimum": None, "median": None, "p95": None, "maximum": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "maximum": ordered[-1],
    }


def _reasons(counts: Mapping[FailureReason, int]) -> dict[str, int]:
    return {
        reason.value: count for reason, count in sorted(counts.items(), key=lambda i: i[0].value)
    }


def _json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def _lines_text(lines: Iterable[str]) -> str:
    return "".join(f"{line}\n" for line in lines)

"""Persisted, immutable, self-describing Semantic Mapping run artifacts.

A ``SemanticMappingRunArtifact`` is a directory holding the semantic entities of one run with the
indexes, lineage, metrics and rejected candidates needed to trust and reuse them. It opens without
NumPy, a perception runtime, a fusion runtime or a model library, and one entity can be read by
reference without loading the others. See ``src/contextmap/semantic_mapping/docs/artifact.md`` for
the layout and the trade-offs of this first schema.

Contractual data lives in ``outputs/`` and ``metrics/`` and is inventoried with size and SHA-256 in
the manifest; ``debug/`` is human evidence that is never inventoried, so removing it cannot
invalidate the run and Entity Resolution and Spatial Relations may not depend on it. Writing
follows :class:`~contextmap.shared.AtomicRunDirectory`: an interrupted write never looks like a
finished run and a finished run is never modified.

Nothing upstream is duplicated: geometry is stored as positional deltas, and fused evidence,
observations, features and 3D representations are only referenced. Every hypothesis, conflict,
abstention and unscored signal of an entity is kept; the artifact never stores just a label. There
is no merge, split or resolution state: that belongs to Entity Resolution.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, NewType, TypeVar

from contextmap.geometric_mapping import GeometrySource, MapId
from contextmap.ingestion import SourceObservationId
from contextmap.point_representation import PointRepresentationRunId
from contextmap.semantic_fusion import SemanticFusionRunId, SemanticFusionRunManifest
from contextmap.semantic_mapping.evidence import fusion_artifact_digest
from contextmap.semantic_mapping.evidence_integrity import (
    EvidenceIntegrityIssue,
    FusedEvidenceSource,
    validate_evidence_of_entities,
)
from contextmap.semantic_mapping.materialization import CandidateRejection, RejectionReason
from contextmap.semantic_mapping.models import (
    Entity,
    EntityId,
    EntityReference,
    EntitySet,
    ForeignEntityReferenceError,
    SemanticMapId,
    UnknownEntityError,
)
from contextmap.semantic_mapping.semantic_state import AmbiguityState
from contextmap.semantic_mapping.serialization import (
    ENTITY_SCHEMA_VERSION,
    decode_entity,
    encode_entity,
)
from contextmap.shared import (
    AtomicRunDirectory,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
    next_run_index,
    write_run_registry,
)
from contextmap.visual_perception import PerceptionRunId

SCHEMA_VERSION = "0.1.0"
"""Semantic Mapping run artifact schema version written and understood by this module."""

SemanticMappingRunId = NewType("SemanticMappingRunId", str)
"""Identity of one Semantic Mapping run."""

_MANIFEST = "manifest.json"
_ENTITIES = "outputs/entities.jsonl"
_ENTITY_INDEX = "outputs/entity-index.jsonl"
_GEOMETRY_INDEX = "outputs/entity-geometry-index.jsonl"
_EVIDENCE_INDEX = "outputs/entity-evidence-index.jsonl"
_OBSERVATION_INDEX = "outputs/entity-observation-index.jsonl"
_SEMANTIC_STATE = "outputs/entity-semantic-state.jsonl"
_TEMPORAL_STATE = "outputs/entity-temporal-state.jsonl"
_REJECTED = "outputs/rejected-candidates.jsonl"
_COUNTS = "metrics/counts.json"
_DISTRIBUTIONS = "metrics/distributions.json"
_PAYLOAD = "metrics/payload.json"
_RUNTIME = "metrics/runtime.json"

_DEBUG_ENTITY_LIMIT = 20
"""Entities that get debug evidence at the ``standard`` level; ``full`` covers all of them."""

_T = TypeVar("_T")


class MappingRunArtifactError(Exception):
    """Base class for Semantic Mapping run artifact read/write failures."""


class IncompleteMappingRunArtifactError(MappingRunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class MappingDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and required metrics only.
        STANDARD: Adds a summary, the geometry, the semantic state, the evidence links and the
            temporal history of a sample of entities.
        FULL: The same for every entity.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, kw_only=True)
class MappingRunLineage:
    """The upstream artifacts a semantic mapping run consumed.

    Attributes:
        sequence_artifact_id: The canonical sequence everything was built from.
        geometric_map_id: The immutable geometric map the geometry belongs to.
        fusion_run_id: The selected Semantic Fusion run the entities were materialized from.
        fusion_schema_version: The schema version of that run.
        fusion_artifact_digest: Digest of that run's identity and inventory.
        association_run_ids: The Sensor Association runs behind the fusion run, sorted and
            unique, through its lineage.
        perception_run_ids: The perception runs behind it, sorted and unique.
        point_representation_run_ids: The Point Representation runs behind it, sorted and
            unique; empty when none was used.
    """

    sequence_artifact_id: str
    geometric_map_id: MapId
    fusion_run_id: SemanticFusionRunId
    fusion_schema_version: str
    fusion_artifact_digest: str
    association_run_ids: tuple[str, ...]
    perception_run_ids: tuple[PerceptionRunId, ...]
    point_representation_run_ids: tuple[PointRepresentationRunId, ...] = ()

    def __post_init__(self) -> None:
        """Validate that every identity is present and every selection sorted and unique.

        Raises:
            ValueError: If an identity is empty, or a selection is not sorted and unique.
        """
        for name in (
            "sequence_artifact_id",
            "geometric_map_id",
            "fusion_run_id",
            "fusion_schema_version",
            "fusion_artifact_digest",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        for name in ("association_run_ids", "perception_run_ids", "point_representation_run_ids"):
            items: tuple[str, ...] = getattr(self, name)
            if list(items) != sorted(set(items)):
                raise ValueError(f"{name} must be sorted and unique")


def lineage_from_fusion_manifest(manifest: SemanticFusionRunManifest) -> MappingRunLineage:
    """Derive the lineage of a mapping run from the fusion run it consumes.

    Args:
        manifest: The manifest of the selected Semantic Fusion run.

    Returns:
        The lineage: that run's identity, version and digest, and the map, sequence, association,
        perception and point representation runs behind it.
    """
    lineage = manifest.lineage
    return MappingRunLineage(
        sequence_artifact_id=lineage.sequence_artifact_id,
        geometric_map_id=lineage.geometric_map_id,
        fusion_run_id=manifest.run_id,
        fusion_schema_version=manifest.schema_version,
        fusion_artifact_digest=fusion_artifact_digest(manifest),
        association_run_ids=lineage.association_run_ids,
        perception_run_ids=lineage.perception_run_ids,
        point_representation_run_ids=lineage.point_representation_run_ids,
    )


@dataclass(frozen=True, kw_only=True)
class SemanticMappingRunManifest:
    """Authoritative metadata of a persisted Semantic Mapping run.

    Attributes:
        run_id: Identity of the run.
        run_index: Monotonic index within this sequence's semantic-mapping runs.
        sequence_name: Name of the processed sequence.
        semantic_map_id: The semantic map the run holds; the scope of every entity id in it.
        lineage: The upstream artifacts the run consumed.
        materialization_policy_id: The materialization policy, ``None`` for an empty run.
        identity_policy_id: The entity id allocation policy, ``None`` for an empty run.
        configuration_fingerprint: Hash of the materialization configuration.
        code_version: Code revision that produced the run.
        code_digest: Digest of the code that produced the run, exactly as the caller supplied it;
            the capability never computes one from a source tree.
        entity_count: Entities persisted.
        rejected_count: Candidates that could not become entities, persisted explicitly.
        warnings: Human-readable warnings of the run.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        entity_schema_version: Version of the canonical entity record the run stores
            (``ENTITY_SCHEMA_VERSION``), independent of the artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes the manifest, the
            README and ``debug/``.
    """

    run_id: SemanticMappingRunId
    run_index: int
    sequence_name: str
    semantic_map_id: SemanticMapId
    lineage: MappingRunLineage
    materialization_policy_id: str | None
    identity_policy_id: str | None
    configuration_fingerprint: str | None
    code_version: str
    code_digest: str
    entity_count: int
    rejected_count: int
    warnings: tuple[str, ...]
    debug_level: str
    schema_version: str
    entity_schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


def _sequence_dir(workspace_root: Path, sequence_name: str) -> Path:
    return workspace_root / "runs" / "semantic-mapping" / sequence_name


class SemanticMappingRunWriter:
    """Builds an immutable Semantic Mapping run artifact on the local filesystem.

    Entities are streamed to disk as they are produced, so the run is not bounded by memory, and
    the run is published atomically when the stream ends.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        sequence_name: str,
        run_id: SemanticMappingRunId,
        run_index: int,
        selection_label: str,
        policy_label: str,
        semantic_map_id: SemanticMapId,
        lineage: MappingRunLineage,
        code_version: str,
        code_digest: str,
        debug_level: MappingDebugLevel = MappingDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            workspace_root: Root of the local workspace.
            sequence_name: Name of the sequence the run processed.
            run_id: Identity of the run.
            run_index: Monotonic index for this sequence's runs (see
                :func:`allocate_mapping_run_index`).
            selection_label: Short readable description of the selected fusion run, for the
                directory name.
            policy_label: Short readable description of the materialization policy, for the
                directory name.
            semantic_map_id: The semantic map the run holds.
            lineage: The explicit upstream selection.
            code_version: Code revision that produced the run.
            code_digest: Digest of the code that produced the run, recorded exactly as given;
                the caller owns how it is computed (for example a commit or a build hash).
            debug_level: Amount of non-contractual debug evidence to persist.

        Raises:
            ValueError: If the code digest is empty.
        """
        if not code_digest.strip():
            raise ValueError("code_digest must not be empty")
        self._workspace_root = workspace_root
        self._sequence_name = sequence_name
        self._run_id = run_id
        self._run_index = run_index
        self._semantic_map_id = semantic_map_id
        self._lineage = lineage
        self._code_version = code_version
        self._code_digest = code_digest
        self._debug_level = debug_level
        self._final_dir = _sequence_dir(workspace_root, sequence_name) / (
            f"run-{run_index:04d}__{selection_label}__{policy_label}"
        )

    def write(
        self,
        entities: Iterable[Entity],
        *,
        rejections: Iterable[CandidateRejection] = (),
        warnings: Sequence[str] = (),
        runtime: Mapping[str, float | int | None] | None = None,
    ) -> SemanticMappingRunManifest:
        """Persist a run atomically, consuming the entities as a stream.

        Args:
            entities: The entities, sorted by identity.
            rejections: The candidates that could not become entities, sorted by support, kept as
                explicit evidence.
            warnings: Human-readable warnings to record.
            runtime: Figures such as seconds and peak memory, recorded apart from every quality
                measure and only when the caller measured them.

        Returns:
            The manifest of the finalized run.

        Raises:
            MappingRunArtifactError: If a run already exists at the target path, the entities are
                not strictly sorted by identity, an entity belongs to another semantic map or
                breaks the lineage or the run's single policy, a candidate is both an entity and a
                rejection, or the write fails.
        """
        tally = _Tally(self._semantic_map_id, self._lineage)
        try:
            with AtomicRunDirectory(self._final_dir) as run:
                self._stream(run, entities, tally)
                rejected = [_encode_rejection(item) for item in rejections]
                tally.check_rejections([row["fusion_support_id"] for row in rejected])
                self._write_tables(run, tally, rejected)
                self._write_metrics(run, tally, rejected, warnings, runtime)
                run.publish(
                    manifest=self._manifest_record(tally, rejected, warnings),
                    readme=self._render_readme(tally, rejected),
                )
        except RunDirectoryError as error:
            raise MappingRunArtifactError(str(error)) from error
        rebuild_mapping_run_registry(
            workspace_root=self._workspace_root, sequence_name=self._sequence_name
        )
        return _load_manifest(self._final_dir)

    def _stream(self, run: AtomicRunDirectory, entities: Iterable[Entity], tally: _Tally) -> None:
        with run.open_binary(_ENTITIES) as out:
            offset = 0
            for entity in entities:
                tally.check(entity)
                line = _line(encode_entity(entity)) + b"\n"
                # O stream não expõe posição: os deslocamentos são contados aqui.
                tally.add(entity, offset=offset, length=len(line) - 1)
                out.write(line)
                offset += len(line)
                self._write_debug(run, entity, tally.entity_count)
        tally.output_sizes[_ENTITIES] = offset

    def _write_tables(
        self, run: AtomicRunDirectory, tally: _Tally, rejected: list[dict[str, Any]]
    ) -> None:
        tables = {
            _ENTITY_INDEX: tally.entity_rows,
            _GEOMETRY_INDEX: tally.geometry_rows,
            _EVIDENCE_INDEX: tally.evidence_rows,
            _OBSERVATION_INDEX: tally.observation_rows,
            _SEMANTIC_STATE: tally.semantic_rows,
            _TEMPORAL_STATE: tally.temporal_rows,
            _REJECTED: rejected,
        }
        for path, rows in tables.items():
            text = "".join(_line(row).decode() + "\n" for row in rows)
            run.write_text(path, text)
            tally.output_sizes[path] = len(text.encode())

    def _write_metrics(
        self,
        run: AtomicRunDirectory,
        tally: _Tally,
        rejected: list[dict[str, Any]],
        warnings: Sequence[str],
        runtime: Mapping[str, float | int | None] | None,
    ) -> None:
        run.write_text(_COUNTS, _json(tally.counts(rejected, len(warnings))))
        run.write_text(_DISTRIBUTIONS, _json(tally.distributions()))
        sizes = dict(sorted(tally.output_sizes.items()))
        run.write_text(_PAYLOAD, _json({"files": sizes, "total_bytes": sum(sizes.values())}))
        if runtime is not None:
            run.write_text(_RUNTIME, _json(dict(runtime)))

    def _write_debug(self, run: AtomicRunDirectory, entity: Entity, number: int) -> None:
        if self._debug_level is MappingDebugLevel.NONE:
            return
        if self._debug_level is MappingDebugLevel.STANDARD and number > _DEBUG_ENTITY_LIMIT:
            return
        base = f"debug/entities/{entity.entity_id}"
        record = encode_entity(entity)
        files = {
            "summary.json": _debug_summary(entity),
            "geometry-summary.json": record["geometry"],
            "semantic-state.json": record["semantic_state"],
            "evidence-trace.json": record["evidence"],
            "temporal-history.json": record["temporal_state"],
        }
        for name, content in files.items():
            run.write_text(f"{base}/{name}", _json(content), contractual=False)

    def _manifest_record(
        self, tally: _Tally, rejected: list[dict[str, Any]], warnings: Sequence[str]
    ) -> dict[str, Any]:
        lineage = self._lineage
        return {
            "run_id": str(self._run_id),
            "run_index": self._run_index,
            "sequence_name": self._sequence_name,
            "semantic_map_id": str(self._semantic_map_id),
            "lineage": {
                "sequence_artifact_id": lineage.sequence_artifact_id,
                "geometric_map_id": str(lineage.geometric_map_id),
                "fusion_run_id": str(lineage.fusion_run_id),
                "fusion_schema_version": lineage.fusion_schema_version,
                "fusion_artifact_digest": lineage.fusion_artifact_digest,
                "association_run_ids": list(lineage.association_run_ids),
                "perception_run_ids": [str(item) for item in lineage.perception_run_ids],
                "point_representation_run_ids": [
                    str(item) for item in lineage.point_representation_run_ids
                ],
            },
            "policies": {
                "materialization_policy_id": tally.materialization_policy_id,
                "identity_policy_id": tally.identity_policy_id,
                "configuration_fingerprint": tally.configuration_fingerprint,
            },
            "code_version": self._code_version,
            "code_digest": self._code_digest,
            "counts": {"entities": tally.entity_count, "rejected_candidates": len(rejected)},
            "warnings": list(warnings),
            "debug_level": self._debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "entity_schema_version": ENTITY_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _render_readme(self, tally: _Tally, rejected: list[dict[str, Any]]) -> str:
        return (
            f"# Semantic mapping run {self._run_index:04d}\n"
            "\n"
            f"- Run ID: `{self._run_id}`\n"
            f"- Semantic map: `{self._semantic_map_id}`\n"
            f"- Fusion run: `{self._lineage.fusion_run_id}`\n"
            f"- Materialization policy: `{tally.materialization_policy_id}`\n"
            f"- Entities: {tally.entity_count}, rejected candidates: {len(rejected)}\n"
            "\n"
            "Contractual data is in `outputs/` and `metrics/`; `debug/` is human evidence and no "
            "downstream stage may depend on it. An entity id is local to this semantic map. "
            "Entities keep every hypothesis, conflict, abstention and unscored signal, and no "
            "same-object decision has been made yet.\n"
        )


class SemanticMappingRunReader:
    """Reads a finalized Semantic Mapping run from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file outside it is needed,
                and no perception, fusion or model runtime.

        Raises:
            IncompleteMappingRunArtifactError: If ``manifest.json`` is missing.
            MappingRunArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = _load_manifest(run_dir)
        self._rows: dict[EntityId, dict[str, Any]] | None = None
        self._observation_entities: dict[SourceObservationId, list[EntityId]] | None = None

    @property
    def manifest(self) -> SemanticMappingRunManifest:
        """The run's manifest."""
        return self._manifest

    @property
    def semantic_map_id(self) -> SemanticMapId:
        """The semantic map the run holds."""
        return self._manifest.semantic_map_id

    def entity_ids(self) -> list[EntityId]:
        """The identities of every entity, in persisted order."""
        return list(self._entity_rows())

    def entity(self, reference: EntityReference) -> Entity:
        """Resolve a reference to its entity without loading the others.

        Args:
            reference: A reference to an entity of this run's semantic map.

        Returns:
            The same canonical entity that was written.

        Raises:
            ForeignEntityReferenceError: If the reference names another semantic map.
            UnknownEntityError: If the run has no such entity.
            MappingRunArtifactError: If the entity's record is malformed or truncated.
        """
        if reference.semantic_map_id != self._manifest.semantic_map_id:
            raise ForeignEntityReferenceError(
                f"reference names semantic map {reference.semantic_map_id!r}, but this run holds "
                f"{self._manifest.semantic_map_id!r}"
            )
        row = self._entity_rows().get(reference.entity_id)
        if row is None:
            raise UnknownEntityError(reference.entity_id)
        raw = self._read_at(_ENTITIES, row["offset"], row["length"])
        return _decode(raw, decode_entity, f"entity {reference.entity_id!r}")

    def iter_entities(self) -> Iterator[Entity]:
        """Iterate every entity, in persisted order."""
        with (self._root / _ENTITIES).open("rb") as handle:
            for line in handle:
                yield _decode(line, decode_entity, "entity")

    def entities(self) -> EntitySet:
        """Load every entity of the run, addressable by reference."""
        return EntitySet(
            semantic_map_id=self._manifest.semantic_map_id, entities=tuple(self.iter_entities())
        )

    def entities_of_observation(self, observation_id: SourceObservationId) -> list[EntityId]:
        """The entities a physical observation contributed to, sorted.

        Args:
            observation_id: A physical observation.

        Returns:
            Its entities; empty when it contributed to none.
        """
        if self._observation_entities is None:
            index: dict[SourceObservationId, list[EntityId]] = {}
            for row in self._read_rows(_OBSERVATION_INDEX):
                index.setdefault(SourceObservationId(row["physical_observation_id"]), []).append(
                    EntityId(row["entity_id"])
                )
            self._observation_entities = index
        return sorted(self._observation_entities.get(observation_id, []))

    def rejected_candidates(self) -> list[CandidateRejection]:
        """The candidates that could not become entities, with why."""
        return [
            CandidateRejection(
                fusion_support_id=row["fusion_support_id"],
                fused_evidence_id=row["fused_evidence_id"],
                reason=RejectionReason(row["reason"]),
                detail=row["detail"],
            )
            for row in self._read_rows(_REJECTED)
        ]

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``metrics/``.

        Raises:
            MappingRunArtifactError: If the path is not a JSON record of ``metrics/``; ``debug/``
                is never a valid source.
        """
        if not relative_path.endswith(".json") or not relative_path.startswith("metrics/"):
            raise MappingRunArtifactError(f"not a contractual JSON record: {relative_path!r}")
        record: dict[str, Any] = json.loads(
            (self._root / relative_path).read_text(encoding="utf-8")
        )
        return record

    def read_table(self, relative_path: str) -> list[dict[str, Any]]:
        """Read a contractual JSON Lines table from ``outputs/``.

        Raises:
            MappingRunArtifactError: If the path is not a table of ``outputs/`` or is malformed.
        """
        if not relative_path.endswith(".jsonl") or not relative_path.startswith("outputs/"):
            raise MappingRunArtifactError(f"not a contractual table: {relative_path!r}")
        return self._read_rows(relative_path)

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        return check_file_inventory(self._root, self._manifest.file_inventory)

    def validate_references(
        self,
        *,
        fusion_runs: Mapping[SemanticFusionRunId, FusedEvidenceSource],
        geometry: GeometrySource | None = None,
    ) -> tuple[EvidenceIntegrityIssue, ...]:
        """Check that the evidence every entity references is still there and unchanged.

        Args:
            fusion_runs: The persisted fusion runs to check against, by run identity.
            geometry: The read boundary of the geometric map, to also check the geometry.

        Returns:
            The issues found; empty means every reference resolves. Missing or corrupt upstream
            payloads are reported explicitly.
        """
        return validate_evidence_of_entities(
            self.iter_entities(), fusion_runs=fusion_runs, geometry=geometry
        )

    def _entity_rows(self) -> dict[EntityId, dict[str, Any]]:
        if self._rows is None:
            self._rows = {EntityId(row["entity_id"]): row for row in self._read_rows(_ENTITY_INDEX)}
        return self._rows

    def _read_rows(self, relative_path: str) -> list[dict[str, Any]]:
        text = (self._root / relative_path).read_text(encoding="utf-8")
        try:
            return [json.loads(line) for line in text.splitlines() if line]
        except ValueError as error:
            raise MappingRunArtifactError(f"malformed table {relative_path}: {error}") from error

    def _read_at(self, relative_path: str, offset: int, length: int) -> bytes:
        with (self._root / relative_path).open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise MappingRunArtifactError(
                f"{relative_path} is truncated: {length} bytes expected at {offset}, "
                f"found {len(data)}"
            )
        return data


def allocate_mapping_run_index(*, workspace_root: Path, sequence_name: str) -> int:
    """Compute the next monotonic run index for a sequence's semantic-mapping runs.

    Scans the run directories, never the registry, so an interrupted or corrupted run is not
    counted.

    Args:
        workspace_root: Root of the local workspace.
        sequence_name: Name of the sequence.

    Returns:
        The next index, starting at ``1``.
    """
    return next_run_index(_sequence_dir(workspace_root, sequence_name), index_of=_valid_run_index)


def rebuild_mapping_run_registry(*, workspace_root: Path, sequence_name: str) -> None:
    """Rebuild a sequence's ``runs.json`` convenience registry from its valid runs.

    Args:
        workspace_root: Root of the local workspace.
        sequence_name: Name of the sequence.
    """
    write_run_registry(_sequence_dir(workspace_root, sequence_name), describe=_registry_record)


class _Tally:
    """Checks the entities as they stream and collects what the tables and metrics need."""

    def __init__(self, semantic_map_id: SemanticMapId, lineage: MappingRunLineage) -> None:
        self._semantic_map_id = semantic_map_id
        self._lineage = lineage
        self.entity_count = 0
        self.output_sizes: dict[str, int] = {}
        self.entity_rows: list[dict[str, Any]] = []
        self.geometry_rows: list[dict[str, Any]] = []
        self.evidence_rows: list[dict[str, Any]] = []
        self.observation_rows: list[dict[str, Any]] = []
        self.semantic_rows: list[dict[str, Any]] = []
        self.temporal_rows: list[dict[str, Any]] = []
        self.materialization_policy_id: str | None = None
        self.identity_policy_id: str | None = None
        self.configuration_fingerprint: str | None = None
        self._previous: EntityId | None = None
        self._supports: set[str] = set()
        self._physical: set[str] = set()
        self._inference = 0
        self._points = 0
        self._states: Counter[str] = Counter()
        self._with_primary = 0
        self._with_conflicts = 0
        self._without_features = 0
        self._without_representations = 0
        self._per_entity: dict[str, list[int]] = {
            "geometry_points_per_entity": [],
            "physical_observations_per_entity": [],
            "inference_results_per_entity": [],
            "hypotheses_per_entity": [],
        }

    def check(self, entity: Entity) -> None:
        if self._previous is not None and entity.entity_id <= self._previous:
            raise MappingRunArtifactError(
                f"entities must be strictly sorted by identity: {entity.entity_id!r} follows "
                f"{self._previous!r}"
            )
        self._previous = entity.entity_id
        if entity.semantic_map_id != self._semantic_map_id:
            raise MappingRunArtifactError(
                f"entity {entity.entity_id!r} belongs to semantic map "
                f"{entity.semantic_map_id!r}, but the run holds {self._semantic_map_id!r}"
            )
        if entity.geometry.geometric_map_id != self._lineage.geometric_map_id:
            raise MappingRunArtifactError(
                f"entity {entity.entity_id!r} is over map {entity.geometry.geometric_map_id!r}, "
                f"but the run's lineage names {self._lineage.geometric_map_id!r}"
            )
        for ref in entity.evidence.fused_evidence:
            if (
                ref.fusion_run_id != self._lineage.fusion_run_id
                or ref.fusion_schema_version != self._lineage.fusion_schema_version
                or ref.fusion_artifact_digest != self._lineage.fusion_artifact_digest
            ):
                raise MappingRunArtifactError(
                    f"entity {entity.entity_id!r} references fusion run {ref.fusion_run_id!r}, "
                    f"which is not the run the lineage names, {self._lineage.fusion_run_id!r}"
                )
            if ref.sequence_artifact_id != self._lineage.sequence_artifact_id:
                raise MappingRunArtifactError(
                    f"entity {entity.entity_id!r} was materialized over sequence "
                    f"{ref.sequence_artifact_id!r}, but the run's lineage names "
                    f"{self._lineage.sequence_artifact_id!r}"
                )
        for feature in entity.evidence.visual_feature_refs:
            if feature.perception_run_id not in self._lineage.perception_run_ids:
                raise MappingRunArtifactError(
                    f"entity {entity.entity_id!r} references perception run "
                    f"{feature.perception_run_id!r}, which the run's lineage does not list"
                )
        for representation in entity.evidence.point_representation_refs:
            if representation.run_id not in self._lineage.point_representation_run_ids:
                raise MappingRunArtifactError(
                    f"entity {entity.entity_id!r} references point representation run "
                    f"{representation.run_id!r}, which the run's lineage does not list"
                )
        self._single_policy(entity)

    def check_rejections(self, supports: list[str]) -> None:
        if supports != sorted(set(supports)):
            raise MappingRunArtifactError(
                "rejected candidates must be sorted by support and unique"
            )
        clash = self._supports.intersection(supports)
        if clash:
            raise MappingRunArtifactError(
                f"a candidate cannot be both an entity and a rejection: {sorted(clash)!r}"
            )

    def _single_policy(self, entity: Entity) -> None:
        provenance = entity.provenance
        current = (
            provenance.materialization_policy_id,
            provenance.identity_policy_id,
            provenance.configuration_fingerprint,
        )
        first = (
            self.materialization_policy_id,
            self.identity_policy_id,
            self.configuration_fingerprint,
        )
        if self.entity_count == 0:
            (
                self.materialization_policy_id,
                self.identity_policy_id,
                self.configuration_fingerprint,
            ) = current
        elif current != first:
            raise MappingRunArtifactError(
                f"entity {entity.entity_id!r} was materialized under a different policy or "
                f"configuration than the rest of the run: one run keeps one"
            )

    def add(self, entity: Entity, *, offset: int, length: int) -> None:
        self.entity_count += 1
        entity_id = str(entity.entity_id)
        geometry, state = entity.geometry, entity.semantic_state
        temporal, links = entity.temporal_state, entity.evidence
        for ref in links.fused_evidence:
            self._supports.add(str(ref.fusion_support_id))
        self._physical.update(str(item) for item in links.physical_observation_ids)
        self._inference += temporal.inference_result_count
        self._points += geometry.statistics.point_count
        self._states[state.ambiguity_state.value] += 1
        self._with_primary += state.primary_hypothesis is not None
        self._with_conflicts += bool(state.conflicts)
        self._without_features += not links.visual_feature_refs
        self._without_representations += not links.point_representation_refs
        per = self._per_entity
        per["geometry_points_per_entity"].append(geometry.statistics.point_count)
        per["physical_observations_per_entity"].append(temporal.physical_observation_count)
        per["inference_results_per_entity"].append(temporal.inference_result_count)
        per["hypotheses_per_entity"].append(len(state.hypotheses))

        self.entity_rows.append(
            {
                "entity_id": entity_id,
                "semantic_map_id": str(entity.semantic_map_id),
                "offset": offset,
                "length": length,
            }
        )
        self.geometry_rows.append(
            {
                "entity_id": entity_id,
                "geometric_map_id": str(geometry.geometric_map_id),
                "map_frame": str(geometry.map_frame),
                "point_count": geometry.statistics.point_count,
                "centroid_m": list(geometry.centroid_m),
                "bounds": {
                    "minimum_m": list(geometry.bounds.minimum_m),
                    "maximum_m": list(geometry.bounds.maximum_m),
                },
                "extent_m": list(geometry.extent_m),
                "component_count": geometry.statistics.component_count,
                "diagnostics": sorted(item.kind.value for item in geometry.diagnostics),
            }
        )
        self.evidence_rows.append(
            {
                "entity_id": entity_id,
                "fused_evidence": [
                    {
                        "fusion_run_id": str(ref.fusion_run_id),
                        "fused_evidence_id": str(ref.fused_evidence_id),
                        "fusion_support_id": str(ref.fusion_support_id),
                    }
                    for ref in links.fused_evidence
                ],
                "spatial_observation_count": len(links.spatial_observation_ids),
                "visual_feature_count": len(links.visual_feature_refs),
                "point_representation_count": len(links.point_representation_refs),
            }
        )
        for item in temporal.observation_refs:
            self.observation_rows.append(
                {
                    "entity_id": entity_id,
                    "physical_observation_id": str(item.physical_observation_id),
                    "acquisition_timestamp": item.acquisition_timestamp.to_record(),
                    "inference_result_count": item.inference_result_count,
                }
            )
        primary = state.primary
        self.semantic_rows.append(
            {
                "entity_id": entity_id,
                "ambiguity_state": state.ambiguity_state.value,
                "primary_label": None if primary is None else primary.label,
                "hypothesis_labels": [item.label for item in state.hypotheses],
                "attributes": [
                    {"name": item.name, "value": item.value, "origin": item.origin.value}
                    for item in state.attributes
                ],
                "uncertainty_kinds": sorted({item.record.kind.value for item in state.uncertainty}),
                "conflict_count": len(state.conflicts),
            }
        )
        self.temporal_rows.append(
            {
                "entity_id": entity_id,
                "first_seen": temporal.first_seen.to_record(),
                "last_seen": temporal.last_seen.to_record(),
                "physical_observation_count": temporal.physical_observation_count,
                "inference_result_count": temporal.inference_result_count,
                "lifecycle": None if temporal.lifecycle is None else temporal.lifecycle.value,
            }
        )

    def counts(self, rejected: list[dict[str, Any]], warnings: int) -> dict[str, Any]:
        reasons = Counter(row["reason"] for row in rejected)
        return {
            "entities": self.entity_count,
            "rejected_candidates": {
                "total": len(rejected),
                "by_reason": {
                    reason.value: reasons.get(reason.value, 0) for reason in RejectionReason
                },
            },
            "semantic_state": {
                state.value: self._states.get(state.value, 0) for state in AmbiguityState
            },
            "entities_with_primary_hypothesis": self._with_primary,
            "entities_with_conflicts": self._with_conflicts,
            "entities_without_visual_features": self._without_features,
            "entities_without_point_representations": self._without_representations,
            "physical_observations": len(self._physical),
            "inference_results": self._inference,
            "geometry_points": self._points,
            "warnings": warnings,
        }

    def distributions(self) -> dict[str, Any]:
        return {name: _distribution(values) for name, values in self._per_entity.items()}


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _encode_rejection(item: CandidateRejection) -> dict[str, Any]:
    return {
        "fusion_support_id": str(item.fusion_support_id),
        "fused_evidence_id": str(item.fused_evidence_id),
        "reason": item.reason.value,
        "detail": item.detail,
    }


def _debug_summary(entity: Entity) -> dict[str, Any]:
    state = entity.semantic_state
    primary = state.primary
    return {
        "entity_id": str(entity.entity_id),
        "semantic_map_id": str(entity.semantic_map_id),
        "ambiguity_state": state.ambiguity_state.value,
        "primary_label": None if primary is None else primary.label,
        "hypothesis_labels": [item.label for item in state.hypotheses],
        "geometry_points": entity.geometry.statistics.point_count,
        "physical_observation_count": entity.temporal_state.physical_observation_count,
        "inference_result_count": entity.temporal_state.inference_result_count,
        "fused_evidence": [str(ref.fused_evidence_id) for ref in entity.evidence.fused_evidence],
    }


def _decode(raw: bytes, decoder: Callable[[Mapping[str, Any]], _T], what: str) -> _T:
    try:
        return decoder(json.loads(raw))
    except (ValueError, KeyError, TypeError) as error:
        raise MappingRunArtifactError(f"malformed {what}: {error}") from error


def _valid_run_index(run_dir: Path) -> int | None:
    # Um diretório ilegível ou com manifest malformado simplesmente não é um run válido.
    try:
        reader = SemanticMappingRunReader(run_dir)
    except (MappingRunArtifactError, ValueError, KeyError, OSError):
        return None
    return None if reader.verify_integrity() else reader.manifest.run_index


def _registry_record(run_dir: Path) -> dict[str, Any] | None:
    index = _valid_run_index(run_dir)
    if index is None:
        return None
    manifest = SemanticMappingRunReader(run_dir).manifest
    return {"run_index": index, "run_id": str(manifest.run_id), "directory": run_dir.name}


def _load_manifest(run_dir: Path) -> SemanticMappingRunManifest:
    manifest_path = run_dir / _MANIFEST
    if not manifest_path.is_file():
        raise IncompleteMappingRunArtifactError(f"missing {_MANIFEST} in {run_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise MappingRunArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}"
        )
    if raw.get("entity_schema_version") != ENTITY_SCHEMA_VERSION:
        raise MappingRunArtifactError(
            f"unsupported entity_schema_version: {raw.get('entity_schema_version')!r}"
        )
    lineage = raw["lineage"]
    policies = raw["policies"]
    counts = raw["counts"]
    return SemanticMappingRunManifest(
        run_id=SemanticMappingRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        semantic_map_id=SemanticMapId(raw["semantic_map_id"]),
        lineage=MappingRunLineage(
            sequence_artifact_id=lineage["sequence_artifact_id"],
            geometric_map_id=MapId(lineage["geometric_map_id"]),
            fusion_run_id=SemanticFusionRunId(lineage["fusion_run_id"]),
            fusion_schema_version=lineage["fusion_schema_version"],
            fusion_artifact_digest=lineage["fusion_artifact_digest"],
            association_run_ids=tuple(lineage["association_run_ids"]),
            perception_run_ids=tuple(PerceptionRunId(i) for i in lineage["perception_run_ids"]),
            point_representation_run_ids=tuple(
                PointRepresentationRunId(i) for i in lineage["point_representation_run_ids"]
            ),
        ),
        materialization_policy_id=policies["materialization_policy_id"],
        identity_policy_id=policies["identity_policy_id"],
        configuration_fingerprint=policies["configuration_fingerprint"],
        code_version=raw["code_version"],
        code_digest=raw["code_digest"],
        entity_count=counts["entities"],
        rejected_count=counts["rejected_candidates"],
        warnings=tuple(raw["warnings"]),
        debug_level=raw["debug_level"],
        schema_version=raw["schema_version"],
        entity_schema_version=raw["entity_schema_version"],
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


def _line(record: Mapping[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"

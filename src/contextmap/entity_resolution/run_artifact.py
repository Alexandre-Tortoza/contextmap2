"""Persisted, immutable, self-describing Entity Resolution run artifacts.

An ``EntityResolutionRunArtifact`` is a directory holding everything one resolution run decided and
why: the candidate sets, the typed match evidence, every ``MATCH``/``DISTINCT``/``UNRESOLVED``
decision, the resolved entities with their merge lineage, the transitivity contradictions, the
optional split candidates, and the metrics. It opens without NumPy, a perception runtime, a point
representation runtime or a model library, and one resolved entity can be read by reference without
loading the others. See ``src/contextmap/entity_resolution/docs/artifact.md`` for the layout.

Contractual data lives in ``outputs/`` and ``metrics/`` and is inventoried with size and SHA-256 in
the manifest; ``debug/`` is human evidence that is never inventoried, so removing it cannot
invalidate the run and Spatial Relations may not depend on it. The writer takes the **final
directory of the artifact**: it does not allocate a run index, keep a registry or compute a
workspace path, and the run identity is supplied by the caller. Writing follows
:class:`~contextmap.shared.AtomicRunDirectory`: an interrupted write never looks like a finished run
and a finished run is never modified.

Nothing upstream is duplicated: the source entities, the geometry, the features and the
representations are only referenced by identity. Every decision keeps the rules that fired, and
every resolved entity can be traced to its exact source entities and the decisions that grouped
them. The policies of the run are not passed in: they are *derived from the records* (each role must
have one policy), so the manifest cannot disagree with the data.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from contextmap.entity_resolution._codec import from_record, to_record
from contextmap.entity_resolution.channels import EvidenceStatus, MatchChannel
from contextmap.entity_resolution.decision import ResolutionDecision, ResolutionOutcome
from contextmap.entity_resolution.evidence import EntityMatchEvidence
from contextmap.entity_resolution.models import (
    EntityResolutionRunId,
    PolicyRef,
    ResolvedEntityId,
    ResolvedEntityReference,
    reference_order,
)
from contextmap.entity_resolution.pair_resolution import PairResolution
from contextmap.entity_resolution.resolved_entity import (
    ContradictionId,
    ForeignResolvedEntityReferenceError,
    ResolvedEntity,
    ResolvedEntityMaterialization,
    ResolvedEntitySet,
    TransitivityContradiction,
    UnknownResolvedEntityError,
)
from contextmap.entity_resolution.retrieval import EntityCandidateSet, candidate_pairs
from contextmap.entity_resolution.split_detection import SplitCandidate
from contextmap.geometric_mapping import MapId
from contextmap.semantic_mapping import (
    EntityReference,
    SemanticMapId,
    SemanticMappingRunManifest,
)
from contextmap.shared import (
    AtomicRunDirectory,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
)

SCHEMA_VERSION = "0.1.0"
"""Entity Resolution run artifact schema version written and understood by this module."""

_MANIFEST = "manifest.json"
_CANDIDATE_SETS = "outputs/candidate-sets.jsonl"
_MATCH_EVIDENCE = "outputs/match-evidence.jsonl"
_DECISIONS = "outputs/resolution-decisions.jsonl"
_RESOLVED = "outputs/resolved-entities.jsonl"
_RESOLVED_INDEX = "outputs/resolved-entity-index.jsonl"
_UNRESOLVED = "outputs/unresolved-entities.jsonl"
_SOURCE_INDEX = "outputs/source-to-resolved-index.jsonl"
_MERGE_LINEAGE = "outputs/merge-lineage.jsonl"
_CONTRADICTIONS = "outputs/transitivity-contradictions.jsonl"
_SPLITS = "outputs/split-candidates.jsonl"
_COUNTS = "metrics/counts.json"
_DISTRIBUTIONS = "metrics/distributions.json"
_PAYLOAD = "metrics/payload.json"
_RUNTIME = "metrics/runtime.json"

_DEBUG_DECISION_LIMIT = 20
"""Decisions that get debug evidence at the ``standard`` level; ``full`` covers all of them."""

_T = TypeVar("_T")


class RunArtifactError(Exception):
    """Base class for Entity Resolution run artifact read/write failures."""


class IncompleteRunArtifactError(RunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class ResolutionDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and metrics only.
        STANDARD: Adds the decision trace of a sample of decisions.
        FULL: The same for every decision.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, kw_only=True)
class ResolutionRunLineage:
    """The upstream artifacts a resolution run consumed.

    Attributes:
        sequence_artifact_id: The canonical sequence everything was built from.
        geometric_map_id: The immutable geometric map the entities' geometry belongs to.
        semantic_mapping_run_id: The Semantic Mapping run the source entities come from.
        semantic_mapping_schema_version: The schema version of that run.
        semantic_mapping_artifact_digest: Digest of that run's identity and inventory.
        semantic_map_ids: The semantic maps whose entities were resolved, sorted and unique.
        point_representation_run_ids: The Point Representation runs behind the entities, sorted and
            unique; empty when none was used.
        perception_run_ids: The perception runs behind the entities, sorted and unique.
    """

    sequence_artifact_id: str
    geometric_map_id: MapId
    semantic_mapping_run_id: str
    semantic_mapping_schema_version: str
    semantic_mapping_artifact_digest: str
    semantic_map_ids: tuple[SemanticMapId, ...]
    point_representation_run_ids: tuple[str, ...]
    perception_run_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that every identity is present and every selection sorted and unique.

        Raises:
            ValueError: If an identity is empty, no semantic map is named, or a selection is not
                sorted and unique.
        """
        for name in (
            "sequence_artifact_id",
            "geometric_map_id",
            "semantic_mapping_run_id",
            "semantic_mapping_schema_version",
            "semantic_mapping_artifact_digest",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if not self.semantic_map_ids:
            raise ValueError("semantic_map_ids must not be empty")
        for name in ("semantic_map_ids", "point_representation_run_ids", "perception_run_ids"):
            items: tuple[str, ...] = getattr(self, name)
            if list(items) != sorted(set(items)):
                raise ValueError(f"{name} must be sorted and unique")


@dataclass(frozen=True, kw_only=True)
class PolicyRecord:
    """The policy a role of the run was produced under.

    Attributes:
        role: What the policy governs, for example ``resolution`` or ``candidate_retrieval``.
        policy: The policy identity and configuration fingerprint.
    """

    role: str
    policy: PolicyRef


@dataclass(frozen=True, kw_only=True)
class Count:
    """One named count of the run.

    Attributes:
        name: What is counted.
        value: The count.
    """

    name: str
    value: int


@dataclass(frozen=True, kw_only=True)
class EntityResolutionRunManifest:
    """Authoritative metadata of a persisted Entity Resolution run.

    Attributes:
        run_id: Identity of the run, supplied by the caller; the scope of every resolved id.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        code_version: Code revision that produced the run.
        lineage: The upstream artifacts the run consumed.
        policies: The policy of each role, derived from the records, sorted by role.
        counts: The headline counts of the run, sorted by name.
        warnings: Human-readable warnings of the run.
        debug_level: Debug evidence level that was requested.
        file_inventory: Every contractual file, with size and hash; excludes the manifest, the
            README and ``debug/``.
    """

    run_id: EntityResolutionRunId
    schema_version: str
    created_at: str
    code_version: str
    lineage: ResolutionRunLineage
    policies: tuple[PolicyRecord, ...]
    counts: tuple[Count, ...]
    warnings: tuple[str, ...]
    debug_level: str
    file_inventory: tuple[FileEntry, ...]

    def count(self, name: str) -> int:
        """The value of one named count.

        Raises:
            KeyError: If the run recorded no such count.
        """
        for item in self.counts:
            if item.name == name:
                return item.value
        raise KeyError(name)

    def policy(self, role: str) -> PolicyRef | None:
        """The policy of one role, or ``None`` when the run had none for it."""
        for item in self.policies:
            if item.role == role:
                return item.policy
        return None


@dataclass(frozen=True, kw_only=True)
class UnresolvedEntity:
    """A source entity a decision left open, indexed so it can be found without loading the rest.

    Attributes:
        entity_ref: The source entity.
        resolved_entity_id: The resolved entity it belongs to.
        unresolved_neighbor_refs: Entities a decision left ``UNRESOLVED`` against its resolved
            entity, sorted and unique.
        contradiction_ids: The transitivity contradictions withholding it, sorted and unique.
    """

    entity_ref: EntityReference
    resolved_entity_id: ResolvedEntityId
    unresolved_neighbor_refs: tuple[EntityReference, ...]
    contradiction_ids: tuple[ContradictionId, ...]


def mapping_artifact_digest(manifest: SemanticMappingRunManifest) -> str:
    """Digest the identity and the inventory of a persisted Semantic Mapping run.

    The digest covers the run identity, its schema version and the size and hash of every
    contractual file, so it is independent of the run's internal layout and changes if anything the
    resolution was built from changes.

    Args:
        manifest: The manifest of the run.

    Returns:
        ``sha256:`` followed by the digest.
    """
    return _run_digest(str(manifest.run_id), manifest.schema_version, manifest.file_inventory)


def resolution_artifact_digest(manifest: EntityResolutionRunManifest) -> str:
    """Digest the identity and the inventory of a persisted Entity Resolution run.

    This is the value a downstream artifact records to pin the exact resolution it was built
    from. It is computed here, by the owner of the artifact, with the rule
    :func:`mapping_artifact_digest` uses for Semantic Mapping: the run identity, its schema version
    and the hash of every contractual file. It ignores ``debug/`` and the moment of writing, so the
    same contractual content has the same digest.

    Args:
        manifest: The manifest of the run.

    Returns:
        ``sha256:`` followed by the digest.
    """
    return _run_digest(str(manifest.run_id), manifest.schema_version, manifest.file_inventory)


def _run_digest(run_id: str, schema_version: str, inventory: tuple[FileEntry, ...]) -> str:
    """The digest of a run's identity, schema version and file hashes, independent of layout."""
    canonical = json.dumps(
        {
            "run_id": run_id,
            "schema_version": schema_version,
            "files": sorted([entry.path, entry.content_hash] for entry in inventory),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def lineage_from_mapping_manifest(manifest: SemanticMappingRunManifest) -> ResolutionRunLineage:
    """Derive the lineage of a resolution run from the Semantic Mapping run it consumes.

    Args:
        manifest: The manifest of the selected Semantic Mapping run.

    Returns:
        The lineage: that run's identity, version and digest, and the sequence, geometric map,
        point representation and perception runs behind it.
    """
    lineage = manifest.lineage
    return ResolutionRunLineage(
        sequence_artifact_id=lineage.sequence_artifact_id,
        geometric_map_id=lineage.geometric_map_id,
        semantic_mapping_run_id=str(manifest.run_id),
        semantic_mapping_schema_version=manifest.schema_version,
        semantic_mapping_artifact_digest=mapping_artifact_digest(manifest),
        semantic_map_ids=(manifest.semantic_map_id,),
        point_representation_run_ids=tuple(
            str(item) for item in lineage.point_representation_run_ids
        ),
        perception_run_ids=tuple(str(item) for item in lineage.perception_run_ids),
    )


def _line(record: object) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def _json(record: object) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


class EntityResolutionRunWriter:
    """Builds an immutable Entity Resolution run artifact on the local filesystem."""

    def __init__(
        self,
        *,
        output_dir: Path,
        run_id: EntityResolutionRunId,
        lineage: ResolutionRunLineage,
        code_version: str,
        debug_level: ResolutionDebugLevel = ResolutionDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            output_dir: The final directory of the artifact. It must not exist: a finished run is
                immutable and re-execution creates another one. Nothing is computed from it:
                there is no run index and no registry.
            run_id: Identity of the run, the scope of every resolved id in it.
            lineage: The explicit upstream selection.
            code_version: Code revision that produced the run.
            debug_level: Amount of non-contractual debug evidence to persist.
        """
        self._output_dir = output_dir
        self._run_id = run_id
        self._lineage = lineage
        self._code_version = code_version
        self._debug_level = debug_level

    def write(
        self,
        *,
        candidate_sets: Sequence[EntityCandidateSet],
        resolutions: Sequence[PairResolution],
        materialization: ResolvedEntityMaterialization,
        split_candidates: Sequence[SplitCandidate] = (),
        warnings: Sequence[str] = (),
        runtime: Mapping[str, float | int | None] | None = None,
    ) -> EntityResolutionRunManifest:
        """Persist a run atomically.

        Args:
            candidate_sets: The candidate sets of the run, one per source entity.
            resolutions: The evidence and decision of every compared pair, which must be candidate
                pairs.
            materialization: The resolved entities and contradictions the decisions justify.
            split_candidates: The optional split diagnostics, if the run ran them.
            warnings: Human-readable warnings to record.
            runtime: Figures such as seconds and peak memory, recorded apart from every quality
                measure and only when the caller measured them.

        Returns:
            The manifest of the finalized run.

        Raises:
            RunArtifactError: If a run already exists at the target path, the records are
                inconsistent (a compared pair that is not a candidate, a decision without its
                evidence, more than one policy for a role, an entity outside the lineage, resolved
                entities of another run) or the write fails.
        """
        records = _Records.check(
            self._run_id,
            self._lineage,
            candidate_sets,
            resolutions,
            materialization,
            split_candidates,
        )
        try:
            with AtomicRunDirectory(self._output_dir) as run:
                sizes = self._write_outputs(run, records)
                self._write_metrics(run, records, sizes, warnings, runtime)
                self._write_debug(run, records)
                run.publish(
                    manifest=self._manifest_record(records, warnings),
                    readme=self._render_readme(records),
                )
        except RunDirectoryError as error:
            raise RunArtifactError(str(error)) from error
        return _load_manifest(self._output_dir)

    def _write_outputs(self, run: AtomicRunDirectory, records: _Records) -> dict[str, int]:
        sizes: dict[str, int] = {}
        offset = 0
        index_rows = []
        with run.open_binary(_RESOLVED) as out:
            for entity in records.resolved.entities:
                line = _line(to_record(entity)) + b"\n"
                index_rows.append(
                    {
                        "resolved_entity_id": str(entity.resolved_entity_id),
                        "offset": offset,
                        "length": len(line) - 1,
                    }
                )
                out.write(line)
                offset += len(line)
        sizes[_RESOLVED] = offset
        # Cada tabela é gravada linha a linha: a evidência de todos os pares nunca vira um texto só.
        tables: dict[str, Iterable[Mapping[str, Any]]] = {
            _CANDIDATE_SETS: (to_record(item) for item in records.candidate_sets),
            _MATCH_EVIDENCE: (to_record(item.evidence) for item in records.resolutions),
            _DECISIONS: (to_record(item.decision) for item in records.resolutions),
            _RESOLVED_INDEX: index_rows,
            _UNRESOLVED: records.unresolved_rows(),
            _SOURCE_INDEX: records.source_rows(),
            _MERGE_LINEAGE: records.lineage_rows(),
            _CONTRADICTIONS: (to_record(item) for item in records.contradictions),
            _SPLITS: (to_record(item) for item in records.split_candidates),
        }
        for path, rows in tables.items():
            size = 0
            with run.open_binary(path) as out:
                for row in rows:
                    line = _line(row) + b"\n"
                    out.write(line)
                    size += len(line)
            sizes[path] = size
        return sizes

    def _write_metrics(
        self,
        run: AtomicRunDirectory,
        records: _Records,
        sizes: dict[str, int],
        warnings: Sequence[str],
        runtime: Mapping[str, float | int | None] | None,
    ) -> None:
        run.write_text(
            _COUNTS, _json({"counts": dict(records.counts()), "warnings": len(warnings)})
        )
        run.write_text(_DISTRIBUTIONS, _json(records.distributions()))
        ordered = dict(sorted(sizes.items()))
        run.write_text(_PAYLOAD, _json({"files": ordered, "total_bytes": sum(ordered.values())}))
        if runtime is not None:
            run.write_text(_RUNTIME, _json(dict(runtime)))

    def _write_debug(self, run: AtomicRunDirectory, records: _Records) -> None:
        if self._debug_level is ResolutionDebugLevel.NONE:
            return
        pairs = records.candidate_lookup()
        limit = (
            _DEBUG_DECISION_LIMIT
            if self._debug_level is ResolutionDebugLevel.STANDARD
            else len(records.resolutions)
        )
        for item in records.resolutions[:limit]:
            base = f"debug/decisions/{item.decision.decision_id}"
            files: dict[str, object] = {
                "decision.json": to_record(item.decision),
                "entity-a.json": to_record(item.evidence.entity_a_ref),
                "entity-b.json": to_record(item.evidence.entity_b_ref),
                "candidate-retrieval.json": pairs.get(
                    (item.evidence.entity_a_ref, item.evidence.entity_b_ref), {}
                ),
                "merge-lineage.json": records.lineage_of(item.evidence),
            }
            for channel, evidence in item.evidence.channels():
                files[f"{channel.value.replace('_', '-')}-evidence.json"] = to_record(evidence)
            for name, content in files.items():
                run.write_text(f"{base}/{name}", _json(content), contractual=False)

    def _manifest_record(self, records: _Records, warnings: Sequence[str]) -> dict[str, Any]:
        manifest = EntityResolutionRunManifest(
            run_id=self._run_id,
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            code_version=self._code_version,
            lineage=self._lineage,
            policies=records.policy_records(),
            counts=tuple(Count(name=name, value=value) for name, value in records.counts()),
            warnings=tuple(warnings),
            debug_level=self._debug_level.value,
            file_inventory=(),
        )
        record: dict[str, Any] = to_record(manifest)
        del record["file_inventory"]
        return record

    def _render_readme(self, records: _Records) -> str:
        counts = dict(records.counts())
        return (
            f"# Entity resolution run\n"
            "\n"
            f"- Run ID: `{self._run_id}`\n"
            f"- Semantic mapping run: `{self._lineage.semantic_mapping_run_id}`\n"
            f"- Source entities: {counts['source_entities']}, candidate pairs compared: "
            f"{counts['compared_pairs']}\n"
            f"- MATCH: {counts['decision_match']}, DISTINCT: {counts['decision_distinct']}, "
            f"UNRESOLVED: {counts['decision_unresolved']}\n"
            f"- Resolved entities: {counts['resolved_entities']}, contradictions: "
            f"{counts['transitivity_contradictions']}\n"
            "\n"
            "Contractual data is in `outputs/` and `metrics/`; `debug/` is human evidence and no "
            "downstream stage may depend on it. A resolved entity id is local to this run. Source "
            "entities are referenced, never copied or modified, and every source entity belongs to "
            "exactly one resolved entity.\n"
        )


@dataclass(frozen=True)
class _Records:
    """The records of one run, checked against each other before anything is written."""

    run_id: EntityResolutionRunId
    lineage: ResolutionRunLineage
    candidate_sets: tuple[EntityCandidateSet, ...]
    resolutions: tuple[PairResolution, ...]
    resolved: ResolvedEntitySet
    contradictions: tuple[TransitivityContradiction, ...]
    split_candidates: tuple[SplitCandidate, ...]
    policies: tuple[PolicyRecord, ...]

    @classmethod
    def check(
        cls,
        run_id: EntityResolutionRunId,
        lineage: ResolutionRunLineage,
        candidate_sets: Sequence[EntityCandidateSet],
        resolutions: Sequence[PairResolution],
        materialization: ResolvedEntityMaterialization,
        split_candidates: Sequence[SplitCandidate],
    ) -> _Records:
        if materialization.resolved.resolution_run_id != run_id:
            raise RunArtifactError(
                "the resolved entities belong to run "
                f"{materialization.resolved.resolution_run_id!r}, "
                f"but the writer is for run {run_id!r}"
            )
        ordered_sets = tuple(
            sorted(candidate_sets, key=lambda item: reference_order(item.source_entity_ref))
        )
        _require_candidate_set_integrity(ordered_sets, materialization)
        ordered = tuple(
            sorted(
                resolutions,
                key=lambda item: (
                    reference_order(item.evidence.entity_a_ref),
                    reference_order(item.evidence.entity_b_ref),
                ),
            )
        )
        _require_unique(
            [(item.evidence.entity_a_ref, item.evidence.entity_b_ref) for item in ordered],
            "a pair was resolved more than once",
        )
        allowed = set(candidate_pairs(ordered_sets))
        for item in ordered:
            pair = (item.evidence.entity_a_ref, item.evidence.entity_b_ref)
            if pair not in allowed:
                raise RunArtifactError(
                    f"pair {pair[0].entity_id!r} / {pair[1].entity_id!r} was compared but is not a "
                    f"candidate pair: only candidates are compared"
                )
        _require_lineage(lineage, materialization)
        known = {item.decision.decision_id for item in ordered}
        for entity in materialization.resolved.entities:
            unknown = [ref for ref in entity.resolution_decision_refs if ref not in known]
            if unknown:
                raise RunArtifactError(
                    f"resolved entity {entity.resolved_entity_id!r} cites decisions that were not "
                    f"written: {unknown[:3]!r}"
                )
        for contradiction in materialization.contradictions:
            cited = (contradiction.distinct_decision_id, *contradiction.match_path)
            unknown = [ref for ref in cited if ref not in known]
            if unknown:
                raise RunArtifactError(
                    f"contradiction {contradiction.contradiction_id!r} cites decisions that were "
                    f"not written: {unknown[:3]!r}"
                )
        return cls(
            run_id=run_id,
            lineage=lineage,
            candidate_sets=ordered_sets,
            resolutions=ordered,
            resolved=materialization.resolved,
            contradictions=materialization.contradictions,
            split_candidates=tuple(
                sorted(split_candidates, key=lambda item: reference_order(item.entity_ref))
            ),
            policies=_derive_policies(ordered_sets, ordered, materialization, split_candidates),
        )

    def policy_records(self) -> tuple[PolicyRecord, ...]:
        return self.policies

    def counts(self) -> list[tuple[str, int]]:
        outcomes = Counter(item.decision.decision for item in self.resolutions)
        pairs = candidate_pairs(self.candidate_sets)
        counts: dict[str, int] = {
            "source_entities": sum(len(item.members) for item in self.resolved.entities),
            "candidate_sets": len(self.candidate_sets),
            "candidate_pairs": len(pairs),
            "compared_pairs": len(self.resolutions),
            "blocked_comparisons": sum(1 for item in self.resolutions if item.evidence.blocked),
            "decision_match": outcomes[ResolutionOutcome.MATCH],
            "decision_distinct": outcomes[ResolutionOutcome.DISTINCT],
            "decision_unresolved": outcomes[ResolutionOutcome.UNRESOLVED],
            "resolved_entities": len(self.resolved.entities),
            "merged_entities": sum(1 for item in self.resolved.entities if len(item.members) > 1),
            "transitivity_contradictions": len(self.contradictions),
            "split_candidates": len(self.split_candidates),
        }
        for channel in MatchChannel:
            measured = unavailable = not_evaluated = 0
            for item in self.resolutions:
                evidence = dict(item.evidence.channels()).get(channel)
                if evidence is None:
                    not_evaluated += 1
                elif evidence.status is EvidenceStatus.UNAVAILABLE:
                    unavailable += 1
                else:
                    measured += 1
            counts[f"channel_{channel.value}_measured"] = measured
            counts[f"channel_{channel.value}_unavailable"] = unavailable
            counts[f"channel_{channel.value}_not_evaluated"] = not_evaluated
        return sorted(counts.items())

    def distributions(self) -> dict[str, Any]:
        sizes = [len(item.members) for item in self.resolved.entities]
        histogram = Counter(sizes)
        return {
            "merge_group_size": {
                "histogram": [
                    {"size": size, "count": histogram[size]} for size in sorted(histogram)
                ],
                "max": max(sizes, default=0),
                "mean": statistics.fmean(sizes) if sizes else None,
            },
            "candidates_per_entity": _summary(len(item.candidates) for item in self.candidate_sets),
            "examined_per_entity": _summary(
                item.diagnostics.examined_entity_count for item in self.candidate_sets
            ),
        }

    def unresolved_rows(self) -> list[dict[str, Any]]:
        rows = []
        for entity in self.resolved.entities:
            if not (entity.unresolved_neighbor_refs or entity.contradiction_ids):
                continue
            for reference in entity.member_entity_refs:
                rows.append(
                    to_record(
                        UnresolvedEntity(
                            entity_ref=reference,
                            resolved_entity_id=entity.resolved_entity_id,
                            unresolved_neighbor_refs=entity.unresolved_neighbor_refs,
                            contradiction_ids=entity.contradiction_ids,
                        )
                    )
                )
        return sorted(
            rows,
            key=lambda row: (row["entity_ref"]["semantic_map_id"], row["entity_ref"]["entity_id"]),
        )

    def source_rows(self) -> list[dict[str, Any]]:
        rows = [
            {**to_record(reference), "resolved_entity_id": str(entity.resolved_entity_id)}
            for entity in self.resolved.entities
            for reference in entity.member_entity_refs
        ]
        return sorted(rows, key=lambda row: (row["semantic_map_id"], row["entity_id"]))

    def lineage_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "resolved_entity_id": str(entity.resolved_entity_id),
                "members": [to_record(member) for member in entity.members],
                "resolution_decision_refs": list(entity.resolution_decision_refs),
                "contradiction_ids": list(entity.contradiction_ids),
                "policy": to_record(entity.provenance.policy),
            }
            for entity in self.resolved.entities
        ]

    def lineage_of(self, evidence: EntityMatchEvidence) -> dict[str, Any]:
        return {
            "entity_a": str(self.resolved.resolved_of(evidence.entity_a_ref).resolved_entity_id),
            "entity_b": str(self.resolved.resolved_of(evidence.entity_b_ref).resolved_entity_id),
        }

    def candidate_lookup(self) -> dict[tuple[EntityReference, EntityReference], Any]:
        lookup: dict[tuple[EntityReference, EntityReference], Any] = {}
        for item in self.candidate_sets:
            for candidate in item.candidates:
                pair = tuple(
                    sorted((item.source_entity_ref, candidate.entity_ref), key=reference_order)
                )
                lookup[(pair[0], pair[1])] = {
                    "policy": to_record(item.policy),
                    "candidate": to_record(candidate),
                }
        return lookup


def _summary(values: Iterable[int]) -> dict[str, Any]:
    items = sorted(values)
    if not items:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {"count": len(items), "min": items[0], "max": items[-1], "mean": statistics.fmean(items)}


def _require_unique(items: Sequence[object], message: str) -> None:
    if len(set(items)) != len(items):
        raise RunArtifactError(message)


def _require_candidate_set_integrity(
    candidate_sets: Sequence[EntityCandidateSet], materialization: ResolvedEntityMaterialization
) -> None:
    """Require exactly one candidate set per source entity, and no reference outside that universe.

    The universe of source entities is the one the materialization actually has (every member of
    every resolved entity): a candidate set for an entity the run does not have, two candidate sets
    for the same entity, a source entity with none, or a candidate that names an entity outside
    that universe would silently corrupt what retrieval reproduces, so all four are refused here
    instead of reaching disk.

    Raises:
        RunArtifactError: If a source entity has no candidate set, more than one, a set names a
            source outside the materialized universe, or a candidate is outside it.
    """
    source_refs = {
        reference
        for entity in materialization.resolved.entities
        for reference in entity.member_entity_refs
    }
    _require_unique(
        [item.source_entity_ref for item in candidate_sets],
        "a source entity has more than one candidate set",
    )
    named = {item.source_entity_ref for item in candidate_sets}
    missing = sorted(source_refs - named, key=reference_order)
    if missing:
        raise RunArtifactError(
            f"{len(missing)} source entities have no candidate set: "
            f"{[ref.entity_id for ref in missing[:3]]!r}"
        )
    foreign_sources = sorted(named - source_refs, key=reference_order)
    if foreign_sources:
        raise RunArtifactError(
            "candidate sets name entities the materialization does not have: "
            f"{[ref.entity_id for ref in foreign_sources[:3]]!r}"
        )
    for item in candidate_sets:
        foreign_candidates = [ref for ref in item.candidate_entity_refs if ref not in source_refs]
        if foreign_candidates:
            names = [ref.entity_id for ref in foreign_candidates[:3]]
            raise RunArtifactError(
                f"the candidate set of {item.source_entity_ref.entity_id!r} names candidates the "
                f"materialization does not have: {names!r}"
            )


def _require_lineage(
    lineage: ResolutionRunLineage, materialization: ResolvedEntityMaterialization
) -> None:
    maps = set(lineage.semantic_map_ids)
    for entity in materialization.resolved.entities:
        for reference in entity.member_entity_refs:
            if reference.semantic_map_id not in maps:
                raise RunArtifactError(
                    f"entity {reference.entity_id!r} belongs to semantic map "
                    f"{reference.semantic_map_id!r}, which the lineage does not name"
                )
        if entity.geometry.geometric_map_id != lineage.geometric_map_id:
            raise RunArtifactError(
                f"resolved entity {entity.resolved_entity_id!r} is over map "
                f"{entity.geometry.geometric_map_id!r}, but the lineage names "
                f"{lineage.geometric_map_id!r}"
            )


def _derive_policies(
    candidate_sets: Sequence[EntityCandidateSet],
    resolutions: Sequence[PairResolution],
    materialization: ResolvedEntityMaterialization,
    split_candidates: Sequence[SplitCandidate],
) -> tuple[PolicyRecord, ...]:
    """The single policy of each role, read from the records; two for one role are refused."""
    found: dict[str, set[PolicyRef]] = {
        "candidate_retrieval": {item.policy for item in candidate_sets},
        "resolution": {item.decision.policy for item in resolutions},
        "materialization": {materialization.policy},
        "split_detection": {item.policy for item in split_candidates},
    }
    for item in resolutions:
        for channel, evidence in item.evidence.channels():
            found.setdefault(f"channel_{channel.value}", set()).add(evidence.policy)
    records = []
    for role, policies in sorted(found.items()):
        if len(policies) > 1:
            raise RunArtifactError(
                f"the {role} policy is not unique in this run: "
                f"{sorted(item.policy_id for item in policies)!r}"
            )
        if policies:
            records.append(PolicyRecord(role=role, policy=next(iter(policies))))
    return tuple(records)


def _load_manifest(run_dir: Path) -> EntityResolutionRunManifest:
    path = run_dir / _MANIFEST
    if not path.is_file():
        raise IncompleteRunArtifactError(f"missing {_MANIFEST} in {run_dir}")
    record = json.loads(path.read_text(encoding="utf-8"))
    version = record.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RunArtifactError(f"unsupported schema_version: {version!r}")
    try:
        return from_record(EntityResolutionRunManifest, record)
    except ValueError as error:
        raise IncompleteRunArtifactError(f"malformed {_MANIFEST}: {error}") from error


class EntityResolutionRunReader:
    """Reads a finalized Entity Resolution run from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file outside it is needed,
                and no perception, feature, point representation or model runtime.

        Raises:
            IncompleteRunArtifactError: If ``manifest.json`` is missing or malformed.
            RunArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = _load_manifest(run_dir)
        self._offsets: dict[ResolvedEntityId, tuple[int, int]] | None = None
        self._sources: dict[EntityReference, ResolvedEntityId] | None = None
        self._spatial_observations: dict[str, tuple[ResolvedEntityReference, ...]] | None = None

    @property
    def manifest(self) -> EntityResolutionRunManifest:
        """The run's manifest."""
        return self._manifest

    @property
    def run_id(self) -> EntityResolutionRunId:
        """The run identity, the scope of every resolved id in it."""
        return self._manifest.run_id

    def candidate_sets(self) -> tuple[EntityCandidateSet, ...]:
        """Every candidate set, in canonical order."""
        return self._read(_CANDIDATE_SETS, EntityCandidateSet)

    def match_evidence(self) -> tuple[EntityMatchEvidence, ...]:
        """The evidence of every compared pair, in canonical order."""
        return self._read(_MATCH_EVIDENCE, EntityMatchEvidence)

    def decisions(self) -> tuple[ResolutionDecision, ...]:
        """Every decision in canonical pair order: ``MATCH``, ``DISTINCT`` and ``UNRESOLVED``."""
        return self._read(_DECISIONS, ResolutionDecision)

    def contradictions(self) -> tuple[TransitivityContradiction, ...]:
        """The transitivity contradictions, surfaced and never hidden."""
        return self._read(_CONTRADICTIONS, TransitivityContradiction)

    def split_candidates(self) -> tuple[SplitCandidate, ...]:
        """The optional split diagnostics; empty when the run did not run detection."""
        return self._read(_SPLITS, SplitCandidate)

    def unresolved_entities(self) -> tuple[UnresolvedEntity, ...]:
        """The source entities a decision left open, without loading the resolved entities."""
        return self._read(_UNRESOLVED, UnresolvedEntity)

    def resolved_entities(self) -> ResolvedEntitySet:
        """Load every resolved entity of the run, addressable by reference."""
        return ResolvedEntitySet(
            resolution_run_id=self.run_id, entities=self._read(_RESOLVED, ResolvedEntity)
        )

    def materialization(self) -> ResolvedEntityMaterialization:
        """The resolved entities, the contradictions and the materialization policy of the run."""
        policy = self._manifest.policy("materialization")
        if policy is None:
            raise IncompleteRunArtifactError("the run records no materialization policy")
        return ResolvedEntityMaterialization(
            resolved=self.resolved_entities(), contradictions=self.contradictions(), policy=policy
        )

    def resolved_entity(self, reference: ResolvedEntityReference) -> ResolvedEntity:
        """Resolve a reference to its resolved entity without loading the others.

        Args:
            reference: A reference to a resolved entity of this run.

        Returns:
            The same canonical resolved entity that was written.

        Raises:
            ForeignResolvedEntityReferenceError: If the reference names another run.
            UnknownResolvedEntityError: If the run has no such resolved entity.
            RunArtifactError: If the entity's record is malformed or truncated.
        """
        if reference.resolution_run_id != self.run_id:
            raise ForeignResolvedEntityReferenceError(
                f"reference names resolution run {reference.resolution_run_id!r}, but this run "
                f"is {self.run_id!r}"
            )
        entry = self._resolved_offsets().get(reference.resolved_entity_id)
        if entry is None:
            raise UnknownResolvedEntityError(reference.resolved_entity_id)
        offset, length = entry
        with (self._root / _RESOLVED).open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise RunArtifactError(
                f"{_RESOLVED} is truncated: {length} bytes expected at {offset}, found {len(data)}"
            )
        try:
            record = json.loads(data)
        except ValueError as error:
            raise RunArtifactError(f"malformed record in {_RESOLVED}: {error}") from error
        return _decode(record, ResolvedEntity, f"resolved entity {reference.resolved_entity_id!r}")

    def resolved_of(self, entity_ref: EntityReference) -> ResolvedEntityReference:
        """The resolved entity a source entity belongs to, from the source index.

        Raises:
            KeyError: If the run has no such source entity.
        """
        if self._sources is None:
            self._sources = {
                EntityReference(
                    semantic_map_id=SemanticMapId(row["semantic_map_id"]),
                    entity_id=row["entity_id"],
                ): ResolvedEntityId(row["resolved_entity_id"])
                for row in self._rows(_SOURCE_INDEX)
            }
        return ResolvedEntityReference(
            resolution_run_id=self.run_id, resolved_entity_id=self._sources[entity_ref]
        )

    def resolved_of_spatial_observation(
        self, spatial_observation_id: str
    ) -> tuple[ResolvedEntityReference, ...]:
        """The resolved entities a spatial observation supports: the link from an upstream claim.

        A statement about a region reaches a resolved entity only through the spatial observation
        that tied the region to 3D. Zero entities means the observation supports none (it was
        never materialized into an entity); more than one means the observation supports entities
        the resolution kept apart. Both are returned as they are, never collapsed to a guess, so
        the consumer refuses or decides explicitly.

        Args:
            spatial_observation_id: The spatial observation named by the upstream claim.

        Returns:
            The references of the resolved entities whose evidence includes it, sorted by id.
        """
        if self._spatial_observations is None:
            found: dict[str, list[ResolvedEntityReference]] = {}
            for entity in self.resolved_entities().entities:
                for observation_id in entity.evidence.spatial_observation_ids:
                    found.setdefault(str(observation_id), []).append(entity.reference)
            self._spatial_observations = {key: tuple(value) for key, value in found.items()}
        return self._spatial_observations.get(str(spatial_observation_id), ())

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``metrics/``.

        Raises:
            RunArtifactError: If the path is not a JSON record of ``metrics/``; ``debug/`` is never
                a valid source.
        """
        if not relative_path.endswith(".json") or not relative_path.startswith("metrics/"):
            raise RunArtifactError(f"not a contractual JSON record: {relative_path!r}")
        record: dict[str, Any] = json.loads(
            (self._root / relative_path).read_text(encoding="utf-8")
        )
        return record

    def read_table(self, relative_path: str) -> list[dict[str, Any]]:
        """Read a contractual JSON Lines table from ``outputs/``.

        Raises:
            RunArtifactError: If the path is not a table of ``outputs/`` or is malformed.
        """
        if not relative_path.endswith(".jsonl") or not relative_path.startswith("outputs/"):
            raise RunArtifactError(f"not a contractual table: {relative_path!r}")
        return self._rows(relative_path)

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        return check_file_inventory(self._root, self._manifest.file_inventory)

    def _resolved_offsets(self) -> dict[ResolvedEntityId, tuple[int, int]]:
        if self._offsets is None:
            self._offsets = {
                ResolvedEntityId(row["resolved_entity_id"]): (row["offset"], row["length"])
                for row in self._rows(_RESOLVED_INDEX)
            }
        return self._offsets

    def _rows(self, relative_path: str) -> list[dict[str, Any]]:
        text = (self._root / relative_path).read_text(encoding="utf-8")
        try:
            return [json.loads(line) for line in text.splitlines() if line]
        except ValueError as error:
            raise RunArtifactError(f"malformed table {relative_path}: {error}") from error

    def _read(self, relative_path: str, cls: type[_T]) -> tuple[_T, ...]:
        return tuple(_decode(row, cls, relative_path) for row in self._rows(relative_path))


def _decode(record: Any, cls: type[_T], where: str) -> _T:
    try:
        return from_record(cls, record)
    except ValueError as error:
        raise RunArtifactError(f"malformed record in {where}: {error}") from error

"""Persisted, immutable, self-describing Spatial Relations run artifacts.

A ``SpatialRelationsRunArtifact`` is a directory holding the relations of one run with the
evidence that supports or contradicts each, the decision behind each, the candidates that were and
were not evaluated, and an index of the relations of every resolved entity, so that Context Map
assembly can reuse them without recomputing any entity-pair evaluation. It opens without NumPy, a
perception runtime or a model library. See ``src/contextmap/spatial_relations/docs/artifact.md``
for the layout.

Contractual data lives in ``outputs/`` and ``metrics/`` and is inventoried with size and SHA-256 in
the manifest; ``debug/`` is human evidence that is never inventoried, so removing it cannot
invalidate the run and no consumer may depend on it. Writing follows
:class:`~contextmap.shared.AtomicRunDirectory`: an interrupted write never looks like a finished
run and a finished run is never modified.

The artifact is written to a directory the caller chooses. There is no run counter, no registry and
no path computed inside: the identity of the run is supplied by the caller. Nothing upstream is
duplicated: entities are referenced by ``ResolvedEntityReference`` and geometry is only identified
by digest, never copied. Supported, rejected and unresolved relations are all persisted and stay
distinguishable, and evidence, decisions and candidates keep their own tables so that every
relation traces to the evidence that decided it.

Deviation from the issue text, following the sibling artifacts: the lineage, the policies and the
effective configuration live inside ``manifest.json`` instead of separate ``config.yaml``,
``lineage.json``, ``environment.json`` and ``events.jsonl`` files.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, NewType, TypeVar

from contextmap.entity_resolution import (
    EntityResolutionRunId,
    EntityResolutionRunManifest,
    EntityResolutionRunReader,
    ForeignResolvedEntityReferenceError,
    ResolvedEntityReference,
    UnknownResolvedEntityError,
    decode_resolved_entity_reference,
    encode_resolved_entity_reference,
    resolution_artifact_digest,
)
from contextmap.geometric_mapping import MapId
from contextmap.shared import AtomicRunDirectory, FileEntry, RunDirectoryError, check_file_inventory
from contextmap.spatial_relations._identity import directed_key, reference_key
from contextmap.spatial_relations.candidates import (
    CANDIDATE_POLICY_ID,
    CandidatePolicy,
    RelationCandidateSet,
)
from contextmap.spatial_relations.contact_predicates import (
    CONTACT_POLICY_ID,
    ContactPredicatePolicy,
)
from contextmap.spatial_relations.decision import (
    CONSERVATIVE_DECISION_POLICY_ID,
    RelationDecision,
    RelationDecisionResult,
    decision_policy_fingerprint,
)
from contextmap.spatial_relations.evidence import (
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceId,
)
from contextmap.spatial_relations.frame_conventions import (
    FRAME_CONVENTIONS_POLICY_ID,
    FrameConventions,
)
from contextmap.spatial_relations.geometric_predicates import (
    GEOMETRIC_POLICY_ID,
    GeometricPredicatePolicy,
)
from contextmap.spatial_relations.models import Relation, RelationId
from contextmap.spatial_relations.observation_evidence import (
    OBSERVATION_RULE_ID,
    observation_evidence_fingerprint,
)
from contextmap.spatial_relations.serialization import (
    decode_candidate_set,
    decode_relation,
    decode_relation_decision,
    decode_relation_evidence,
    encode_candidate_set,
    encode_relation,
    encode_relation_decision,
    encode_relation_evidence,
)
from contextmap.spatial_relations.taxonomy import TAXONOMY_VERSION

SCHEMA_VERSION = "0.1.0"
"""Spatial Relations run artifact schema version written and understood by this module."""

SpatialRelationsRunId = NewType("SpatialRelationsRunId", str)
"""Identity of one Spatial Relations run, supplied by the caller."""

_MANIFEST = "manifest.json"
_RELATIONS = "outputs/relations.jsonl"
_EVIDENCE = "outputs/relation-evidence.jsonl"
_CANDIDATES = "outputs/relation-candidates.jsonl"
_DECISIONS = "outputs/relation-decisions.jsonl"
_ENTITY_INDEX = "outputs/entity-relation-index.jsonl"
_COUNTS = "metrics/counts.json"
_RUNTIME = "metrics/runtime.json"

_DEBUG_RELATION_LIMIT = 20
"""Relations that get debug evidence at the ``standard`` level; ``full`` covers all of them."""

_T = TypeVar("_T")


class RelationsRunArtifactError(Exception):
    """Base class for Spatial Relations run artifact read/write failures."""


class IncompleteRelationsRunArtifactError(RelationsRunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class RelationsRunDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and required metrics only.
        STANDARD: Adds, for a sample of relations, the relation, its decision and its evidence
            with every measurement.
        FULL: The same for every relation.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, kw_only=True)
class RelationsRunLineage:
    """The upstream artifacts a Spatial Relations run consumed.

    Attributes:
        entity_resolution_run_id: The selected Entity Resolution run: the scope of every resolved
            entity reference in the run.
        entity_resolution_schema_version: The schema version of that run.
        entity_resolution_artifact_digest: Digest of that run's identity and inventory, so that
            a later change of the upstream artifact is detectable; it is Entity Resolution's own
            ``resolution_artifact_digest``.
        geometric_map_id: The immutable geometric map the geometric evidence was measured on.
    """

    entity_resolution_run_id: EntityResolutionRunId
    entity_resolution_schema_version: str
    entity_resolution_artifact_digest: str
    geometric_map_id: MapId

    def __post_init__(self) -> None:
        """Require every identity.

        Raises:
            ValueError: If an identity is empty.
        """
        for name in (
            "entity_resolution_run_id",
            "entity_resolution_schema_version",
            "entity_resolution_artifact_digest",
            "geometric_map_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")


def lineage_from_resolution_manifest(manifest: EntityResolutionRunManifest) -> RelationsRunLineage:
    """Derive the lineage of a relations run from the resolution run it consumes.

    Args:
        manifest: The manifest of the selected Entity Resolution run.

    Returns:
        The lineage: that run's identity, schema version and digest (computed by Entity
        Resolution, the owner of the artifact) and the geometric map its entities were resolved on.
    """
    return RelationsRunLineage(
        entity_resolution_run_id=manifest.run_id,
        entity_resolution_schema_version=manifest.schema_version,
        entity_resolution_artifact_digest=resolution_artifact_digest(manifest),
        geometric_map_id=manifest.lineage.geometric_map_id,
    )


@dataclass(frozen=True, kw_only=True)
class RelationsRunPolicies:
    """The effective policies a run was produced under.

    Attributes:
        frame_conventions: The axes the run declared for the map frame.
        candidate: The candidate generation policy.
        geometric: The geometric predicate policy, when geometric evidence was produced.
        contact: The contact predicate policy, when contact evidence was produced.
    """

    frame_conventions: FrameConventions
    candidate: CandidatePolicy
    geometric: GeometricPredicatePolicy | None = None
    contact: ContactPredicatePolicy | None = None


@dataclass(frozen=True, kw_only=True)
class SpatialRelationsRunManifest:
    """Authoritative metadata of a persisted Spatial Relations run.

    Attributes:
        run_id: Identity of the run.
        lineage: The upstream artifacts the run consumed.
        taxonomy_version: The relation vocabulary version.
        policies: The effective policies, each with its identity, fingerprint and parameters.
        code_version: Code revision that produced the run.
        counts: A summary of what the run holds.
        warnings: Human-readable warnings of the run.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes the manifest, the
            README and ``debug/``.
    """

    run_id: SpatialRelationsRunId
    lineage: RelationsRunLineage
    taxonomy_version: str
    policies: dict[str, Any]
    code_version: str
    counts: dict[str, Any]
    warnings: tuple[str, ...]
    debug_level: str
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


class SpatialRelationsRunWriter:
    """Builds an immutable Spatial Relations run artifact in a directory the caller chooses."""

    def __init__(
        self,
        *,
        output_dir: Path,
        run_id: SpatialRelationsRunId,
        lineage: RelationsRunLineage,
        policies: RelationsRunPolicies,
        code_version: str,
        debug_level: RelationsRunDebugLevel = RelationsRunDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            output_dir: The final directory of the artifact. It must not exist yet: a finished run
                is immutable and re-execution creates a new one.
            run_id: Identity of the run, supplied by the caller.
            lineage: The explicit upstream selection.
            policies: The effective policies the relations were produced under.
            code_version: Code revision that produced the run.
            debug_level: Amount of non-contractual debug evidence to persist.
        """
        self._output_dir = output_dir
        self._run_id = run_id
        self._lineage = lineage
        self._policies = policies
        self._code_version = code_version
        self._debug_level = debug_level

    def write(
        self,
        *,
        candidates: RelationCandidateSet,
        evidence: Sequence[RelationEvidence],
        decisions: RelationDecisionResult,
        warnings: Sequence[str] = (),
        runtime: Mapping[str, float | int | None] | None = None,
    ) -> SpatialRelationsRunManifest:
        """Persist a run atomically.

        Args:
            candidates: The candidates that were and were not evaluated, with the reasons.
            evidence: Every evidence record the decisions read, from every channel.
            decisions: The relations and the decision behind each.
            warnings: Human-readable warnings to record.
            runtime: Figures such as seconds and peak memory, recorded apart from every quality
                measure and only when the caller measured them.

        Returns:
            The manifest of the finalized run.

        Raises:
            RelationsRunArtifactError: If the directory already exists, a relation is about an
                entity of another resolution run, the candidates or the evidence come from another
                geometric map, frame conventions or policy than declared, a relation cites
                evidence that was not given, evidence is not about a candidate, or the write fails.
        """
        ordered_evidence = sorted(evidence, key=lambda item: item.evidence_id)
        self._validate(candidates, ordered_evidence, decisions)
        index = _entity_index(decisions.relations)
        counts = _counts(candidates, ordered_evidence, decisions, index, len(warnings))
        try:
            with AtomicRunDirectory(self._output_dir) as run:
                run.write_text(
                    _RELATIONS, _lines(encode_relation(item) for item in decisions.relations)
                )
                run.write_text(
                    _EVIDENCE, _lines(encode_relation_evidence(item) for item in ordered_evidence)
                )
                run.write_text(_CANDIDATES, _lines(encode_candidate_set(candidates)))
                run.write_text(
                    _DECISIONS,
                    _lines(encode_relation_decision(item) for item in decisions.decisions),
                )
                run.write_text(_ENTITY_INDEX, _lines(index))
                run.write_text(_COUNTS, _json(counts))
                if runtime is not None:
                    run.write_text(_RUNTIME, _json(dict(runtime)))
                self._write_debug(run, ordered_evidence, decisions)
                run.publish(
                    manifest=self._manifest_record(counts, warnings),
                    readme=self._render_readme(counts),
                )
        except RunDirectoryError as error:
            raise RelationsRunArtifactError(str(error)) from error
        return _load_manifest(self._output_dir)

    def _validate(
        self,
        candidates: RelationCandidateSet,
        evidence: Sequence[RelationEvidence],
        decisions: RelationDecisionResult,
    ) -> None:
        lineage = self._lineage
        for relation in decisions.relations:
            for reference in (relation.subject_entity_ref, relation.object_entity_ref):
                if reference.resolution_run_id != lineage.entity_resolution_run_id:
                    raise RelationsRunArtifactError(
                        f"relation {relation.relation_id!r} is about an entity of resolution run "
                        f"{reference.resolution_run_id!r}, but the run selected "
                        f"{lineage.entity_resolution_run_id!r}"
                    )
        self._require_scope(candidates.provenance.geometric_map_id, "the candidates")
        provenance = candidates.provenance
        if provenance.configuration_fingerprint != self._policies.candidate.fingerprint():
            raise RelationsRunArtifactError(
                "the candidates were generated under a policy whose fingerprint is not the "
                "declared candidate policy's"
            )
        if (
            provenance.frame_conventions_fingerprint
            != self._policies.frame_conventions.fingerprint()
        ):
            raise RelationsRunArtifactError(
                "the candidates were generated under other frame conventions than the declared ones"
            )
        fingerprints = self._channel_fingerprints()
        known = {
            directed_key(item.subject_entity_ref, item.predicate, item.object_entity_ref)
            for item in candidates.candidates
        }
        seen: set[RelationEvidenceId] = set()
        for record in evidence:
            if record.evidence_id in seen:
                raise RelationsRunArtifactError(f"evidence {record.evidence_id!r} was given twice")
            seen.add(record.evidence_id)
            expected = fingerprints.get(record.channel)
            if expected is None:
                raise RelationsRunArtifactError(
                    f"evidence {record.evidence_id!r} is of the {record.channel.value} channel, "
                    f"but no {record.channel.value} policy was declared"
                )
            if record.provenance.configuration_fingerprint != expected:
                raise RelationsRunArtifactError(
                    f"evidence {record.evidence_id!r} was produced under a configuration whose "
                    f"fingerprint is not the declared {record.channel.value} policy's"
                )
            self._require_scope(
                record.provenance.geometric_map_id, f"evidence {record.evidence_id!r}"
            )
            key = directed_key(
                record.subject_entity_ref, record.predicate, record.object_entity_ref
            )
            if key not in known:
                raise RelationsRunArtifactError(
                    f"evidence {record.evidence_id!r} is not about a candidate of this run"
                )
        self._validate_decisions(candidates, seen, decisions, known)

    def _validate_decisions(
        self,
        candidates: RelationCandidateSet,
        evidence_ids: set[RelationEvidenceId],
        decisions: RelationDecisionResult,
        known: set[tuple[str, str, str, str, str]],
    ) -> None:
        by_id = {item.relation_id: item for item in decisions.decisions}
        evaluated: set[tuple[str, str, str, str, str]] = set()
        for relation in decisions.relations:
            provenance = relation.provenance
            if (
                provenance.decision_policy_id != CONSERVATIVE_DECISION_POLICY_ID
                or provenance.configuration_fingerprint != decision_policy_fingerprint()
            ):
                raise RelationsRunArtifactError(
                    f"relation {relation.relation_id!r} was decided by a policy other than the "
                    f"one the run declares"
                )
            missing = [ref for ref in relation.relation_evidence_refs if ref not in evidence_ids]
            decision = by_id[relation.relation_id]
            missing += [
                ref
                for ref in (
                    *decision.deciding_evidence_refs,
                    *(u.evidence_id for u in decision.ignored),
                )
                if ref not in evidence_ids
            ]
            if missing:
                raise RelationsRunArtifactError(
                    f"relation {relation.relation_id!r} cites evidence that was not given: "
                    f"{sorted(set(missing))!r}"
                )
            if relation.derived_from is None:
                evaluated.add(
                    directed_key(
                        relation.subject_entity_ref, relation.predicate, relation.object_entity_ref
                    )
                )
        if evaluated != known:
            raise RelationsRunArtifactError(
                "the evaluated relations and the candidates of the run do not correspond one to one"
            )

    def _require_scope(self, map_id: MapId | None, what: str) -> None:
        if map_id is not None and map_id != self._lineage.geometric_map_id:
            raise RelationsRunArtifactError(
                f"{what} come from geometric map {map_id!r}, but the lineage says "
                f"{self._lineage.geometric_map_id!r}"
            )

    def _channel_fingerprints(self) -> dict[RelationEvidenceChannel, str]:
        fingerprints = {RelationEvidenceChannel.OBSERVATION: observation_evidence_fingerprint()}
        if self._policies.geometric is not None:
            fingerprints[RelationEvidenceChannel.GEOMETRY] = self._policies.geometric.fingerprint()
        if self._policies.contact is not None:
            fingerprints[RelationEvidenceChannel.CONTACT] = self._policies.contact.fingerprint()
        return fingerprints

    def _write_debug(
        self,
        run: AtomicRunDirectory,
        evidence: Sequence[RelationEvidence],
        decisions: RelationDecisionResult,
    ) -> None:
        if self._debug_level is RelationsRunDebugLevel.NONE:
            return
        records = {item.evidence_id: item for item in evidence}
        limit = (
            len(decisions.relations)
            if self._debug_level is RelationsRunDebugLevel.FULL
            else _DEBUG_RELATION_LIMIT
        )
        for relation, decision in list(zip(decisions.relations, decisions.decisions, strict=True))[
            :limit
        ]:
            summary = {
                "relation": encode_relation(relation),
                "decision": encode_relation_decision(decision),
                "evidence": [
                    encode_relation_evidence(records[ref])
                    for ref in relation.relation_evidence_refs
                ],
            }
            run.write_text(
                f"debug/relations/{relation.relation_id}.json", _json(summary), contractual=False
            )

    def _manifest_record(self, counts: dict[str, Any], warnings: Sequence[str]) -> dict[str, Any]:
        lineage = self._lineage
        return {
            "run_id": str(self._run_id),
            "lineage": {
                "entity_resolution_run_id": str(lineage.entity_resolution_run_id),
                "entity_resolution_schema_version": lineage.entity_resolution_schema_version,
                "entity_resolution_artifact_digest": lineage.entity_resolution_artifact_digest,
                "geometric_map_id": str(lineage.geometric_map_id),
            },
            "taxonomy_version": TAXONOMY_VERSION,
            "policies": _policies_record(self._policies),
            "code_version": self._code_version,
            "counts": {
                key: counts[key] for key in ("relations", "evidence", "candidates", "by_state")
            },
            "warnings": list(warnings),
            "debug_level": self._debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _render_readme(self, counts: dict[str, Any]) -> str:
        states = ", ".join(f"{name}: {number}" for name, number in counts["by_state"].items())
        return (
            f"# Spatial relations run `{self._run_id}`\n"
            "\n"
            f"- Entity resolution run: `{self._lineage.entity_resolution_run_id}`\n"
            f"- Geometric map: `{self._lineage.geometric_map_id}`\n"
            f"- Taxonomy: `{TAXONOMY_VERSION}`\n"
            f"- Relations: {counts['relations']} ({states})\n"
            "\n"
            "Contractual data is in `outputs/` and `metrics/`; `debug/` is human evidence and no "
            "downstream stage may depend on it. Supported, rejected and unresolved relations are "
            "all kept and stay distinguishable; every relation lists the evidence it rests on.\n"
        )


class SpatialRelationsRunReader:
    """Reads a finalized Spatial Relations run from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file outside it is needed,
                and no perception or model runtime.

        Raises:
            IncompleteRelationsRunArtifactError: If ``manifest.json`` is missing.
            RelationsRunArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = _load_manifest(run_dir)
        self._relations: dict[RelationId, Relation] | None = None
        self._evidence: dict[RelationEvidenceId, RelationEvidence] | None = None
        self._decisions: dict[RelationId, RelationDecision] | None = None
        self._index: list[dict[str, Any]] | None = None

    @property
    def manifest(self) -> SpatialRelationsRunManifest:
        """The run's manifest."""
        return self._manifest

    def iter_relations(self) -> Iterator[Relation]:
        """Iterate every relation, in canonical order."""
        yield from self._load_relations().values()

    def relation(self, relation_id: RelationId) -> Relation:
        """Look a relation up by identity.

        Raises:
            KeyError: If the run has no such relation.
        """
        return self._load_relations()[relation_id]

    def iter_evidence(self) -> Iterator[RelationEvidence]:
        """Iterate every evidence record, sorted by identity."""
        yield from self._load_evidence().values()

    def evidence(self, evidence_id: RelationEvidenceId) -> RelationEvidence:
        """Look an evidence record up by identity.

        Raises:
            KeyError: If the run has no such record.
        """
        return self._load_evidence()[evidence_id]

    def evidence_of(self, relation: Relation) -> tuple[RelationEvidence, ...]:
        """The evidence a relation rests on, in identity order."""
        records = self._load_evidence()
        return tuple(records[ref] for ref in relation.relation_evidence_refs)

    def iter_decisions(self) -> Iterator[RelationDecision]:
        """Iterate every decision, in the order of the relations."""
        yield from self._load_decisions().values()

    def decision(self, relation_id: RelationId) -> RelationDecision:
        """The decision behind a relation.

        Raises:
            KeyError: If the run has no such relation.
        """
        return self._load_decisions()[relation_id]

    def relations_of(
        self,
        entity: ResolvedEntityReference,
        *,
        as_subject: bool = True,
        as_object: bool = True,
    ) -> tuple[Relation, ...]:
        """The relations a resolved entity takes part in, using the entity index.

        Args:
            entity: A resolved entity reference.
            as_subject: Include the relations where it is the subject.
            as_object: Include the relations where it is the object.

        Returns:
            The relations in canonical order; empty when the entity takes part in none.
        """
        wanted = encode_resolved_entity_reference(entity)
        ids: set[str] = set()
        for row in self._load_index():
            if row["entity_ref"] == wanted:
                if as_subject:
                    ids.update(row["as_subject"])
                if as_object:
                    ids.update(row["as_object"])
        return tuple(
            item for item in self._load_relations().values() if str(item.relation_id) in ids
        )

    def candidate_set(self) -> RelationCandidateSet:
        """Rebuild the candidates, the exclusions with their reasons and the skipped predicates."""
        return _decode(self._read_rows(_CANDIDATES), decode_candidate_set, "candidates")

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``metrics/``.

        Raises:
            RelationsRunArtifactError: If the path is not a JSON record of ``metrics/``;
                ``debug/`` is never a valid source.
        """
        if not relative_path.endswith(".json") or not relative_path.startswith("metrics/"):
            raise RelationsRunArtifactError(f"not a contractual JSON record: {relative_path!r}")
        record: dict[str, Any] = json.loads(
            (self._root / relative_path).read_text(encoding="utf-8")
        )
        return record

    def read_table(self, relative_path: str) -> list[dict[str, Any]]:
        """Read a contractual JSON Lines table from ``outputs/``.

        Raises:
            RelationsRunArtifactError: If the path is not a table of ``outputs/`` or is malformed.
        """
        if not relative_path.endswith(".jsonl") or not relative_path.startswith("outputs/"):
            raise RelationsRunArtifactError(f"not a contractual table: {relative_path!r}")
        return self._read_rows(relative_path)

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        return check_file_inventory(self._root, self._manifest.file_inventory)

    def validate_resolution(self, resolution: EntityResolutionRunReader) -> tuple[str, ...]:
        """Check the run against the Entity Resolution run it says it was built on.

        Args:
            resolution: A reader of the Entity Resolution run the lineage names.

        Returns:
            Human-readable problems; empty means the lineage matches that run (its identity, schema
            version and digest) and every resolved entity the relations name exists in it. A run
            that is not the one named is reported alone, since nothing else can be compared.
        """
        lineage = self._manifest.lineage
        if lineage.entity_resolution_run_id != resolution.run_id:
            return (
                f"the run names resolution run {lineage.entity_resolution_run_id!r}, but the "
                f"reader opened {resolution.run_id!r}",
            )
        issues: list[str] = []
        manifest = resolution.manifest
        if lineage.entity_resolution_schema_version != manifest.schema_version:
            issues.append(
                f"the run records resolution schema version "
                f"{lineage.entity_resolution_schema_version!r}, but that run has "
                f"{manifest.schema_version!r}"
            )
        if lineage.entity_resolution_artifact_digest != resolution_artifact_digest(manifest):
            issues.append(
                "the resolution artifact digest recorded by the run is not the digest of that "
                "run: the upstream artifact changed"
            )
        for row in self._load_index():
            reference = decode_resolved_entity_reference(row["entity_ref"])
            try:
                resolution.resolved_entity(reference)
            except (UnknownResolvedEntityError, ForeignResolvedEntityReferenceError):
                issues.append(
                    f"the run references resolved entity {reference.resolved_entity_id!r}, which "
                    f"the resolution run does not have"
                )
        return tuple(issues)

    def _load_relations(self) -> dict[RelationId, Relation]:
        if self._relations is None:
            rows = self._read_rows(_RELATIONS)
            self._relations = {
                item.relation_id: item
                for item in (_decode(r, decode_relation, "relation") for r in rows)
            }
        return self._relations

    def _load_evidence(self) -> dict[RelationEvidenceId, RelationEvidence]:
        if self._evidence is None:
            rows = self._read_rows(_EVIDENCE)
            self._evidence = {
                item.evidence_id: item
                for item in (_decode(r, decode_relation_evidence, "evidence") for r in rows)
            }
        return self._evidence

    def _load_decisions(self) -> dict[RelationId, RelationDecision]:
        if self._decisions is None:
            rows = self._read_rows(_DECISIONS)
            self._decisions = {
                item.relation_id: item
                for item in (_decode(r, decode_relation_decision, "decision") for r in rows)
            }
        return self._decisions

    def _load_index(self) -> list[dict[str, Any]]:
        if self._index is None:
            self._index = self._read_rows(_ENTITY_INDEX)
        return self._index

    def _read_rows(self, relative_path: str) -> list[dict[str, Any]]:
        text = (self._root / relative_path).read_text(encoding="utf-8")
        try:
            return [json.loads(line) for line in text.splitlines() if line]
        except ValueError as error:
            raise RelationsRunArtifactError(f"malformed table {relative_path}: {error}") from error


def _entity_index(relations: Sequence[Relation]) -> list[dict[str, Any]]:
    """Index the relations of every resolved entity, by relation identity."""
    as_subject: dict[ResolvedEntityReference, list[str]] = {}
    as_object: dict[ResolvedEntityReference, list[str]] = {}
    for relation in relations:
        as_subject.setdefault(relation.subject_entity_ref, []).append(str(relation.relation_id))
        as_object.setdefault(relation.object_entity_ref, []).append(str(relation.relation_id))
    references = sorted(set(as_subject) | set(as_object), key=reference_key)
    return [
        {
            "entity_ref": encode_resolved_entity_reference(reference),
            "as_subject": sorted(as_subject.get(reference, [])),
            "as_object": sorted(as_object.get(reference, [])),
        }
        for reference in references
    ]


def _counts(
    candidates: RelationCandidateSet,
    evidence: Sequence[RelationEvidence],
    decisions: RelationDecisionResult,
    index: Sequence[Mapping[str, Any]],
    warning_count: int,
) -> dict[str, Any]:
    by_predicate_state: dict[str, Counter[str]] = {}
    for relation in decisions.relations:
        by_predicate_state.setdefault(relation.predicate.value, Counter())[
            relation.state.value
        ] += 1
    by_channel_status: dict[str, Counter[str]] = {}
    for record in evidence:
        by_channel_status.setdefault(record.channel.value, Counter())[record.status.value] += 1
    return {
        "relations": len(decisions.relations),
        "evaluated_relations": sum(1 for r in decisions.relations if r.derived_from is None),
        "derived_relations": sum(1 for r in decisions.relations if r.derived_from is not None),
        "by_state": dict(sorted(Counter(r.state.value for r in decisions.relations).items())),
        "by_predicate_state": {
            predicate: dict(sorted(states.items()))
            for predicate, states in sorted(by_predicate_state.items())
        },
        "unresolved_by_uncertainty": dict(
            sorted(
                Counter(
                    item.kind.value
                    for relation in decisions.relations
                    for item in relation.uncertainty
                ).items()
            )
        ),
        "decisions_by_rule": dict(
            sorted(Counter(item.rule.value for item in decisions.decisions).items())
        ),
        "evidence": len(evidence),
        "evidence_by_channel_status": {
            channel: dict(sorted(statuses.items()))
            for channel, statuses in sorted(by_channel_status.items())
        },
        "candidates": len(candidates.candidates),
        "exclusions": len(candidates.exclusions),
        "skipped_predicates": len(candidates.skipped_predicates),
        "pairs_not_enumerated": candidates.pairs_not_enumerated,
        "entities_with_relations": len(index),
        "warnings": warning_count,
    }


def _policies_record(policies: RelationsRunPolicies) -> dict[str, Any]:
    conventions = policies.frame_conventions
    record: dict[str, Any] = {
        "decision": {
            "policy_id": CONSERVATIVE_DECISION_POLICY_ID,
            "fingerprint": decision_policy_fingerprint(),
        },
        "candidate": {
            "policy_id": CANDIDATE_POLICY_ID,
            "fingerprint": policies.candidate.fingerprint(),
            "parameters": {
                "predicates": sorted(item.value for item in policies.candidate.predicates),
                "proximity_radius_m": policies.candidate.proximity_radius_m,
                "directional_radius_m": policies.candidate.directional_radius_m,
            },
        },
        "frame_conventions": {
            "policy_id": FRAME_CONVENTIONS_POLICY_ID,
            "fingerprint": conventions.fingerprint(),
            "map_frame": conventions.map_frame,
            "up_axis": None if conventions.up_axis is None else conventions.up_axis.value,
            "forward_axis": (
                None if conventions.forward_axis is None else conventions.forward_axis.value
            ),
        },
        "observation": {
            "rule_id": OBSERVATION_RULE_ID,
            "fingerprint": observation_evidence_fingerprint(),
        },
    }
    if policies.geometric is not None:
        record["geometric"] = {
            "policy_id": GEOMETRIC_POLICY_ID,
            "fingerprint": policies.geometric.fingerprint(),
            "parameters": dataclasses.asdict(policies.geometric),
        }
    if policies.contact is not None:
        record["contact"] = {
            "policy_id": CONTACT_POLICY_ID,
            "fingerprint": policies.contact.fingerprint(),
            "parameters": dataclasses.asdict(policies.contact),
        }
    return record


def _load_manifest(run_dir: Path) -> SpatialRelationsRunManifest:
    manifest_path = run_dir / _MANIFEST
    if not manifest_path.is_file():
        raise IncompleteRelationsRunArtifactError(f"missing {_MANIFEST} in {run_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RelationsRunArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}"
        )
    lineage = raw["lineage"]
    return SpatialRelationsRunManifest(
        run_id=SpatialRelationsRunId(raw["run_id"]),
        lineage=RelationsRunLineage(
            entity_resolution_run_id=EntityResolutionRunId(lineage["entity_resolution_run_id"]),
            entity_resolution_schema_version=lineage["entity_resolution_schema_version"],
            entity_resolution_artifact_digest=lineage["entity_resolution_artifact_digest"],
            geometric_map_id=MapId(lineage["geometric_map_id"]),
        ),
        taxonomy_version=raw["taxonomy_version"],
        policies=raw["policies"],
        code_version=raw["code_version"],
        counts=raw["counts"],
        warnings=tuple(raw["warnings"]),
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


def _decode(source: Any, decoder: Callable[[Any], _T], what: str) -> _T:
    try:
        return decoder(source)
    except (ValueError, KeyError, TypeError) as error:
        raise RelationsRunArtifactError(f"malformed {what}: {error}") from error


def _lines(records: Any) -> str:
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )


def _json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"

"""Deterministic validation of Semantic Mapping runs: entities, evidence and reproducibility.

Semantic Mapping is validated as a materialization stage, and its failures must stay visible: this
module reads a persisted mapping run through the public reader, together with the fusion run and
the geometric map it was built from, and reports six layers of checks, each with the number of
items it examined and the exact failures it found:

* **contract** -- unique artifact-local ids, references that resolve, an explicit semantic map
  identity, valid geometry that resolves to the authoritative map, a semantic state that obeys its
  invariants and complete provenance;
* **semantic preservation** -- every hypothesis, alternative, conflict, abstention and unscored
  signal of the fused evidence survives, and no primary label hides an alternative;
* **evidence lineage** -- the references resolve and the provenance can be traversed from the entity
  to its views and physical frames, and from its geometry to the source observations;
* **temporal** -- ``first_seen``/``last_seen`` and the counts are reproducible from the exact
  evidence, and repeated inference never inflates the physical observations;
* **materialization boundary** -- the baseline stays a function of each support alone: one support,
  one entity, identity from the support, no cross-support state, and re-materializing the same
  selection reproduces the persisted entities exactly;
* **artifact round trip** -- the run is intact, every entity reopens unchanged and the derived
  indexes agree with the entities.

No layer is merged into a score: a report either passes a check or lists why it does not. This
does not evaluate same-object or entity-resolution accuracy, does not use relation extraction to
validate entities and never repairs anything: there is no automatic merge or split. ``debug/`` is
never read.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from contextmap.geometric_mapping import GeometrySource
from contextmap.semantic_fusion import (
    EvidenceStance,
    FusedEvidence,
    FusedHypothesis,
    FusionRunArtifactError,
    SemanticFusionRunManifest,
    SemanticFusionRunReader,
)
from contextmap.semantic_mapping import (
    ENTITY_MATERIALIZATION_POLICY_ID,
    AmbiguityState,
    Entity,
    EntityHypothesis,
    EntityMaterializationPolicy,
    EntityReference,
    EvidenceTraceError,
    ForeignEntityReferenceError,
    GeometryResolutionError,
    SemanticMapId,
    SemanticMappingRunManifest,
    SemanticMappingRunReader,
    TemporalEvidenceError,
    UnknownEntityError,
    decode_entity,
    derive_ambiguity_state,
    encode_entity,
    entity_id_for,
    fusion_artifact_digest,
    materialize_entities,
    semantic_state_from_fused_evidence,
    summarize_temporal_state,
    trace_entity_evidence,
    trace_geometry_sources,
    validate_entity_evidence,
    verify_geometry_summary,
)

EVALUATOR_VERSION = "1"
"""Bumped whenever a check's definition changes, so reports stay comparable."""

_FOREIGN_MAP = SemanticMapId("semantic-map--evaluation-foreign-probe")


class SemanticMappingEvaluationError(ValueError):
    """Raised when an evaluation cannot be performed at all, so a report would mislead."""


class SemanticMappingValidationLayer(Enum):
    """The layers of validation, reported separately.

    Attributes:
        CONTRACT: Identity, references, geometry, semantic state and provenance invariants.
        SEMANTIC_PRESERVATION: What the fused evidence said survives materialization.
        EVIDENCE_LINEAGE: References resolve and provenance can be traversed.
        TEMPORAL: Temporal summaries are reproducible and counts are not inflated.
        MATERIALIZATION_BOUNDARY: The baseline stays a function of each support alone.
        ARTIFACT_ROUND_TRIP: The persisted run is intact and reopens unchanged.
    """

    CONTRACT = "contract"
    SEMANTIC_PRESERVATION = "semantic_preservation"
    EVIDENCE_LINEAGE = "evidence_lineage"
    TEMPORAL = "temporal"
    MATERIALIZATION_BOUNDARY = "materialization_boundary"
    ARTIFACT_ROUND_TRIP = "artifact_round_trip"


@dataclass(frozen=True, kw_only=True)
class SemanticMappingValidationCheck:
    """One check, with what it examined and exactly what failed.

    Attributes:
        layer: The layer the check belongs to.
        check_id: A stable identity of the check.
        examined: How many items the check looked at.
        failures: A deterministic, human-readable description of each failure, in order; empty
            means the check passed.
    """

    layer: SemanticMappingValidationLayer
    check_id: str
    examined: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether the check found nothing wrong."""
        return not self.failures


@dataclass(frozen=True, kw_only=True)
class SemanticMappingEvaluationLineage:
    """Everything a validation report needs to be reproduced and compared.

    Attributes:
        mapping_run_id: The Semantic Mapping run that was validated.
        semantic_map_id: The semantic map it holds.
        entity_schema_version: The schema version of the run artifact.
        fusion_run_id: The Semantic Fusion run the entities were materialized from.
        fusion_schema_version: The schema version of that run.
        fusion_artifact_digest: Digest of that run's identity and inventory.
        geometric_map_id: The geometric map the geometry belongs to.
        materialization_policy_id: The materialization policy and its version, ``None`` for an
            empty run.
        identity_policy_id: The entity id allocation policy, ``None`` for an empty run.
        configuration_fingerprint: Digest of the materialization configuration.
        code_version: The code revision that produced the entities.
        evaluator_version: The version of these checks.
    """

    mapping_run_id: str
    semantic_map_id: SemanticMapId
    entity_schema_version: str
    fusion_run_id: str
    fusion_schema_version: str
    fusion_artifact_digest: str
    geometric_map_id: str
    materialization_policy_id: str | None
    identity_policy_id: str | None
    configuration_fingerprint: str | None
    code_version: str
    evaluator_version: str


@dataclass(frozen=True, kw_only=True)
class SemanticMappingEvaluationReport:
    """The validation of one Semantic Mapping run.

    Attributes:
        lineage: The identities that make the report reproducible.
        entity_count: Entities validated.
        rejected_count: Candidates that were rejected and are kept as evidence.
        checks: Every check, in a fixed order, grouped by layer.
    """

    lineage: SemanticMappingEvaluationLineage
    entity_count: int
    rejected_count: int
    checks: tuple[SemanticMappingValidationCheck, ...]

    @property
    def passed(self) -> bool:
        """Whether every check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[SemanticMappingValidationCheck, ...]:
        """The checks that found something wrong."""
        return tuple(check for check in self.checks if not check.passed)

    def layer(
        self, layer: SemanticMappingValidationLayer
    ) -> tuple[SemanticMappingValidationCheck, ...]:
        """The checks of one layer."""
        return tuple(check for check in self.checks if check.layer is layer)

    def check(self, check_id: str) -> SemanticMappingValidationCheck:
        """One check by identity.

        Raises:
            KeyError: If the report has no such check.
        """
        for item in self.checks:
            if item.check_id == check_id:
                return item
        raise KeyError(check_id)


def evaluate_semantic_mapping(
    mapping: SemanticMappingRunReader,
    *,
    fusion: SemanticFusionRunReader,
    geometry: GeometrySource,
    policy: EntityMaterializationPolicy,
) -> SemanticMappingEvaluationReport:
    """Validate a persisted Semantic Mapping run against the evidence it was built from.

    The mapping run, the fusion run and the geometric map are only read. Nothing is repaired: a
    check that fails lists why, and no candidate is merged, split or re-resolved.

    Args:
        mapping: The persisted mapping run.
        fusion: The persisted Semantic Fusion run the entities were materialized from.
        geometry: The read boundary of the geometric map.
        policy: The materialization configuration the run is expected to have used.

    Returns:
        The report, with every check of every layer.

    Raises:
        SemanticMappingEvaluationError: If the offered fusion run or geometric map is not the one
            the mapping run's lineage names, so the checks would compare unrelated things.
    """
    manifest = mapping.manifest
    _require_same_upstream(manifest, fusion.manifest, geometry)
    entities = tuple(mapping.iter_entities())
    facts = _Facts(mapping=mapping, fusion=fusion, geometry=geometry, policy=policy)
    facts.load(entities)
    checks = (
        *_contract_checks(facts, entities),
        *_preservation_checks(facts, entities),
        *_lineage_checks(facts, entities),
        *_temporal_checks(facts, entities),
        *_boundary_checks(facts, entities),
        *_round_trip_checks(facts, entities),
    )
    lineage = manifest.lineage
    return SemanticMappingEvaluationReport(
        lineage=SemanticMappingEvaluationLineage(
            mapping_run_id=str(manifest.run_id),
            semantic_map_id=manifest.semantic_map_id,
            entity_schema_version=manifest.schema_version,
            fusion_run_id=str(lineage.fusion_run_id),
            fusion_schema_version=lineage.fusion_schema_version,
            fusion_artifact_digest=lineage.fusion_artifact_digest,
            geometric_map_id=str(lineage.geometric_map_id),
            materialization_policy_id=manifest.materialization_policy_id,
            identity_policy_id=manifest.identity_policy_id,
            configuration_fingerprint=manifest.configuration_fingerprint,
            code_version=manifest.code_version,
            evaluator_version=EVALUATOR_VERSION,
        ),
        entity_count=len(entities),
        rejected_count=manifest.rejected_count,
        checks=tuple(checks),
    )


def encode_semantic_mapping_report(report: SemanticMappingEvaluationReport) -> dict[str, Any]:
    """Encode a report as JSON-compatible data, with every identity and every failure.

    Args:
        report: The report.

    Returns:
        A dictionary of JSON primitives; there is no composite score.
    """
    lineage = report.lineage
    return {
        "evaluator_version": lineage.evaluator_version,
        "lineage": {
            "mapping_run_id": lineage.mapping_run_id,
            "semantic_map_id": str(lineage.semantic_map_id),
            "entity_schema_version": lineage.entity_schema_version,
            "fusion_run_id": lineage.fusion_run_id,
            "fusion_schema_version": lineage.fusion_schema_version,
            "fusion_artifact_digest": lineage.fusion_artifact_digest,
            "geometric_map_id": lineage.geometric_map_id,
            "materialization_policy_id": lineage.materialization_policy_id,
            "identity_policy_id": lineage.identity_policy_id,
            "configuration_fingerprint": lineage.configuration_fingerprint,
            "code_version": lineage.code_version,
        },
        "entity_count": report.entity_count,
        "rejected_count": report.rejected_count,
        "passed": report.passed,
        "checks": [
            {
                "layer": check.layer.value,
                "check_id": check.check_id,
                "examined": check.examined,
                "passed": check.passed,
                "failures": list(check.failures),
            }
            for check in report.checks
        ],
    }


def _require_same_upstream(
    manifest: SemanticMappingRunManifest,
    fusion: SemanticFusionRunManifest,
    geometry: GeometrySource,
) -> None:
    lineage = manifest.lineage
    if (
        fusion.run_id != lineage.fusion_run_id
        or fusion.schema_version != lineage.fusion_schema_version
        or fusion_artifact_digest(fusion) != lineage.fusion_artifact_digest
    ):
        raise SemanticMappingEvaluationError(
            f"the offered fusion run is not the one the mapping run was materialized from, "
            f"{lineage.fusion_run_id!r}"
        )
    if geometry.geometric_map.map_id != lineage.geometric_map_id:
        raise SemanticMappingEvaluationError(
            f"the offered geometry serves map {geometry.geometric_map.map_id!r}, but the mapping "
            f"run names {lineage.geometric_map_id!r}"
        )


class _Facts:
    """The upstream evidence of every entity, read once and shared by every check."""

    def __init__(
        self,
        *,
        mapping: SemanticMappingRunReader,
        fusion: SemanticFusionRunReader,
        geometry: GeometrySource,
        policy: EntityMaterializationPolicy,
    ) -> None:
        self.mapping = mapping
        self.fusion = fusion
        self.geometry = geometry
        self.policy = policy
        self.fused: dict[str, FusedEvidence] = {}
        self.unreadable: dict[str, str] = {}

    def load(self, entities: Sequence[Entity]) -> None:
        for entity in entities:
            # A baseline entity links to exactly one fused evidence; a boundary check flags more.
            ref = entity.evidence.fused_evidence[0]
            try:
                self.fused[str(entity.entity_id)] = self.fusion.fused_evidence(
                    ref.fusion_support_id
                )
            except (KeyError, FusionRunArtifactError) as error:
                self.unreadable[str(entity.entity_id)] = f"cannot read its fused evidence: {error}"

    def failures(
        self,
        entities: Sequence[Entity],
        judge: Callable[[Entity, FusedEvidence], Sequence[str]],
    ) -> tuple[str, ...]:
        """Run ``judge(entity, fused)`` over the entities, reporting unreadable evidence."""
        found: list[str] = []
        for entity in entities:
            key = str(entity.entity_id)
            fused = self.fused.get(key)
            if fused is None:
                found.append(f"{key}: {self.unreadable.get(key, 'no fused evidence')}")
                continue
            found.extend(f"{key}: {problem}" for problem in judge(entity, fused))
        return tuple(found)


def _check(
    layer: SemanticMappingValidationLayer,
    check_id: str,
    examined: int,
    failures: Sequence[str],
) -> SemanticMappingValidationCheck:
    return SemanticMappingValidationCheck(
        layer=layer, check_id=check_id, examined=examined, failures=tuple(failures)
    )


def _contract_checks(
    facts: _Facts, entities: Sequence[Entity]
) -> list[SemanticMappingValidationCheck]:
    layer = SemanticMappingValidationLayer.CONTRACT
    manifest = facts.mapping.manifest
    ids = [str(entity.entity_id) for entity in entities]

    identity: list[str] = []
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        identity.append(f"entity ids are repeated: {duplicates!r}")
    if len(entities) != manifest.entity_count:
        identity.append(
            f"the manifest counts {manifest.entity_count} entities, the run holds {len(entities)}"
        )
    if ids != [str(item) for item in facts.mapping.entity_ids()]:
        identity.append("the entity index does not list the entities in the persisted order")

    scope = [
        f"{entity.entity_id}: belongs to semantic map {entity.semantic_map_id!r}, "
        f"not {manifest.semantic_map_id!r}"
        for entity in entities
        if entity.semantic_map_id != manifest.semantic_map_id
    ]

    resolution: list[str] = []
    for entity in entities:
        try:
            if facts.mapping.entity(entity.reference) != entity:
                resolution.append(f"{entity.entity_id}: its reference resolves to another record")
        except (UnknownEntityError, ForeignEntityReferenceError) as error:
            resolution.append(f"{entity.entity_id}: its reference does not resolve: {error}")
    if entities:
        foreign = EntityReference(semantic_map_id=_FOREIGN_MAP, entity_id=entities[0].entity_id)
        try:
            facts.mapping.entity(foreign)
            resolution.append("a reference of another semantic map was not refused")
        except ForeignEntityReferenceError:
            pass

    geometry: list[str] = []
    frame = facts.geometry.geometric_map.frame_id
    for entity in entities:
        if entity.geometry.map_frame != frame:
            geometry.append(
                f"{entity.entity_id}: geometry is in {entity.geometry.map_frame!r}, not {frame!r}"
            )
        for problem in verify_geometry_summary(
            entity.geometry, source=facts.geometry, policy=facts.policy.geometry
        ):
            geometry.append(f"{entity.entity_id}: {problem}")

    states: list[str] = []
    for entity in entities:
        state = entity.semantic_state
        if state.ambiguity_state is not derive_ambiguity_state(state.hypotheses, state.uncertainty):
            states.append(f"{entity.entity_id}: the ambiguity state does not follow its records")
        if (
            state.primary_hypothesis is not None
            and state.ambiguity_state is not AmbiguityState.UNAMBIGUOUS
        ):
            states.append(f"{entity.entity_id}: a primary hypothesis hides alternatives")
        for attribute in state.attributes:
            if not attribute.evidence and attribute.origin.value != "external_knowledge":
                states.append(f"{entity.entity_id}: attribute {attribute.name!r} cites no evidence")

    lineage = manifest.lineage
    provenance: list[str] = []
    for entity in entities:
        item = entity.provenance
        if item.configuration_fingerprint != manifest.configuration_fingerprint:
            provenance.append(f"{entity.entity_id}: its configuration differs from the run's")
        if item.configuration_fingerprint != facts.policy.fingerprint():
            provenance.append(
                f"{entity.entity_id}: it was not materialized under the offered policy"
            )
        if not item.code_version:
            provenance.append(f"{entity.entity_id}: no code version is recorded")
        for ref in entity.evidence.fused_evidence:
            if (
                ref.fusion_run_id != lineage.fusion_run_id
                or ref.fusion_schema_version != lineage.fusion_schema_version
                or ref.fusion_artifact_digest != lineage.fusion_artifact_digest
                or ref.sequence_artifact_id != lineage.sequence_artifact_id
            ):
                provenance.append(
                    f"{entity.entity_id}: its fused evidence is not from the lineage's run"
                )

    return [
        _check(layer, "entity_ids_unique_and_counted", len(entities), identity),
        _check(layer, "semantic_map_identity_explicit", len(entities), scope),
        _check(layer, "entity_references_resolve", len(entities) + 1, resolution),
        _check(layer, "geometry_valid_and_authoritative", len(entities), geometry),
        _check(layer, "semantic_state_invariants", len(entities), states),
        _check(layer, "provenance_complete", len(entities), provenance),
    ]


def _preservation_checks(
    facts: _Facts, entities: Sequence[Entity]
) -> list[SemanticMappingValidationCheck]:
    layer = SemanticMappingValidationLayer.SEMANTIC_PRESERVATION

    def hypotheses(entity: Entity, fused: FusedEvidence) -> list[str]:
        kept = [
            (item.hypothesis_id, item.label, item.evidence)
            for item in entity.semantic_state.hypotheses
        ]
        said = [(item.hypothesis_id, item.label, item.evidence) for item in fused.hypotheses]
        return [] if kept == said else ["the hypotheses differ from the fused evidence"]

    def uncertainty(entity: Entity, fused: FusedEvidence) -> list[str]:
        kept = [item.record for item in entity.semantic_state.uncertainty]
        return (
            []
            if kept == list(fused.uncertainty)
            else ["the conflicts and ambiguity differ from the fused evidence"]
        )

    def abstention(entity: Entity, fused: FusedEvidence) -> list[str]:
        kept = _abstentions(entity.semantic_state.hypotheses)
        said = _abstentions(fused.hypotheses)
        return [] if kept == said else ["abstentions differ from the fused evidence"]

    def unscored(entity: Entity, fused: FusedEvidence) -> list[str]:
        kept = _unscored(entity.semantic_state.hypotheses)
        said = _unscored(fused.hypotheses)
        return [] if kept == said else ["unscored signals differ from the fused evidence"]

    def no_forced_label(entity: Entity, fused: FusedEvidence) -> list[str]:
        state = entity.semantic_state
        if state.primary_hypothesis is None:
            return []
        if len(fused.hypotheses) != 1 or fused.uncertainty:
            return ["a primary hypothesis was exposed although the fused evidence competes"]
        return []

    def faithful(entity: Entity, fused: FusedEvidence) -> list[str]:
        expected = semantic_state_from_fused_evidence(fused)
        return (
            []
            if entity.semantic_state == expected
            else ["the semantic state is not the mapping of its fused evidence"]
        )

    return [
        _check(layer, "hypotheses_preserved", len(entities), facts.failures(entities, hypotheses)),
        _check(
            layer, "uncertainty_preserved", len(entities), facts.failures(entities, uncertainty)
        ),
        _check(layer, "abstention_preserved", len(entities), facts.failures(entities, abstention)),
        _check(
            layer, "unscored_signals_preserved", len(entities), facts.failures(entities, unscored)
        ),
        _check(
            layer,
            "no_forced_single_label",
            len(entities),
            facts.failures(entities, no_forced_label),
        ),
        _check(
            layer,
            "semantic_state_is_the_mapping_of_the_evidence",
            len(entities),
            facts.failures(entities, faithful),
        ),
    ]


def _abstentions(hypotheses: Iterable[EntityHypothesis | FusedHypothesis]) -> int:
    return sum(
        1
        for hypothesis in hypotheses
        for item in hypothesis.evidence
        if item.stance is EvidenceStance.ABSTAINING
    )


def _unscored(hypotheses: Iterable[EntityHypothesis | FusedHypothesis]) -> Counter[str]:
    return Counter(
        signal.kind.value
        for hypothesis in hypotheses
        for item in hypothesis.evidence
        for signal in item.signals
        if signal.value is None
    )


def _lineage_checks(
    facts: _Facts, entities: Sequence[Entity]
) -> list[SemanticMappingValidationCheck]:
    layer = SemanticMappingValidationLayer.EVIDENCE_LINEAGE
    fusion_runs = {facts.fusion.manifest.run_id: facts.fusion}

    references: list[str] = []
    provenance: list[str] = []
    geometry_trace: list[str] = []
    for entity in entities:
        for issue in validate_entity_evidence(
            entity, fusion_runs=fusion_runs, geometry=facts.geometry
        ):
            references.append(f"{entity.entity_id}: [{issue.kind.value}] {issue.detail}")
        try:
            trace = trace_entity_evidence(entity, fusion_runs=fusion_runs)
        except EvidenceTraceError as error:
            provenance.append(f"{entity.entity_id}: the provenance cannot be traversed: {error}")
        else:
            if trace.physical_observation_ids != entity.evidence.physical_observation_ids:
                provenance.append(f"{entity.entity_id}: the trace ends at other physical frames")
            if {item.spatial_observation_id for item in trace.contributions} != set(
                entity.evidence.spatial_observation_ids
            ):
                provenance.append(
                    f"{entity.entity_id}: the trace reaches other spatial observations"
                )
        try:
            sources = trace_geometry_sources(entity.geometry, source=facts.geometry)
        except GeometryResolutionError as error:
            geometry_trace.append(f"{entity.entity_id}: the geometry cannot be traversed: {error}")
        else:
            if sum(item.point_count for item in sources) != len(entity.geometry.geometry_refs):
                geometry_trace.append(
                    f"{entity.entity_id}: the source observations miss some points"
                )

    return [
        _check(layer, "references_resolve", len(entities), references),
        _check(layer, "entity_to_source_observation_traversal", len(entities), provenance),
        _check(layer, "entity_geometry_to_source_traversal", len(entities), geometry_trace),
    ]


def _temporal_checks(
    facts: _Facts, entities: Sequence[Entity]
) -> list[SemanticMappingValidationCheck]:
    layer = SemanticMappingValidationLayer.TEMPORAL

    def reproducible(entity: Entity, fused: FusedEvidence) -> list[str]:
        try:
            derived = summarize_temporal_state(fused.physical_observation_groups)
        except TemporalEvidenceError as error:
            return [f"the temporal evidence cannot be derived: {error}"]
        problems: list[str] = []
        if derived.first_seen != entity.temporal_state.first_seen or derived.last_seen != (
            entity.temporal_state.last_seen
        ):
            problems.append("first_seen or last_seen is not reproducible from the exact evidence")
        if derived.time_bounds != fused.temporal_summary:
            problems.append("the interval disagrees with the fused evidence's temporal summary")
        if derived.observation_refs != entity.temporal_state.observation_refs:
            problems.append("the observation history differs from the physical observations")
        return problems

    def counts(entity: Entity, fused: FusedEvidence) -> list[str]:
        state = entity.temporal_state
        problems: list[str] = []
        if state.physical_observation_count != fused.physical_observation_count:
            problems.append("the physical observation count differs from the fused evidence")
        if state.inference_result_count != fused.inference_result_count:
            problems.append("the inference result count differs from the fused evidence")
        if state.physical_observation_count > state.inference_result_count:
            problems.append("there are more physical observations than inference results")
        return problems

    return [
        _check(
            layer,
            "first_and_last_seen_reproducible",
            len(entities),
            facts.failures(entities, reproducible),
        ),
        _check(
            layer,
            "physical_observations_not_inflated",
            len(entities),
            facts.failures(entities, counts),
        ),
    ]


def _boundary_checks(
    facts: _Facts, entities: Sequence[Entity]
) -> list[SemanticMappingValidationCheck]:
    layer = SemanticMappingValidationLayer.MATERIALIZATION_BOUNDARY
    manifest = facts.mapping.manifest

    one_to_one: list[str] = []
    supports: Counter[str] = Counter()
    for entity in entities:
        refs = entity.evidence.fused_evidence
        if len(refs) != 1:
            one_to_one.append(
                f"{entity.entity_id}: it links to {len(refs)} fused evidence, not one"
            )
        for ref in refs:
            supports[str(ref.fusion_support_id)] += 1
    one_to_one.extend(
        f"support {support!r} became {count} entities"
        for support, count in sorted(supports.items())
        if count > 1
    )

    identity = [
        f"{entity.entity_id}: the id is not the one its support derives"
        for entity in entities
        if len(entity.evidence.fused_evidence) == 1
        and entity.entity_id
        != entity_id_for(fusion_support_id=entity.evidence.fused_evidence[0].fusion_support_id)
    ]

    policy_check = (
        []
        if manifest.materialization_policy_id in (None, ENTITY_MATERIALIZATION_POLICY_ID)
        else [
            f"the run used {manifest.materialization_policy_id!r}, whose boundary this "
            f"evaluator cannot assert"
        ]
    )

    return [
        _check(layer, "one_support_one_entity", len(entities), one_to_one),
        _check(layer, "identity_is_a_function_of_the_support", len(entities), identity),
        _check(layer, "known_materialization_policy", len(entities), policy_check),
        _check(
            layer,
            "rematerialization_reproduces_the_entities",
            len(entities),
            _rematerialize(facts, entities),
        ),
    ]


def _rematerialize(facts: _Facts, entities: Sequence[Entity]) -> list[str]:
    manifest = facts.mapping.manifest
    selected = {
        str(ref.fusion_support_id) for entity in entities for ref in entity.evidence.fused_evidence
    }
    selected.update(str(item.fusion_support_id) for item in facts.mapping.rejected_candidates())
    try:
        outcomes = [
            outcome
            for outcome in facts.fusion.iter_outcomes()
            if str(outcome.support.fusion_support_id) in selected
        ]
    except (FusionRunArtifactError, ValueError) as error:
        # O leitor de fusão levanta ValueError quando os arquivos paralelos têm tamanhos diferentes.
        return [f"the fusion run cannot be read to re-materialize the selection: {error}"]
    again = materialize_entities(
        outcomes,
        fusion_manifest=facts.fusion.manifest,
        geometry=facts.geometry,
        semantic_map_id=manifest.semantic_map_id,
        policy=facts.policy,
        code_version=manifest.code_version,
    )
    problems: list[str] = []
    if again.entities != tuple(entities):
        problems.append(
            "re-materializing the same selection does not reproduce the persisted entities"
        )
    if list(again.rejections) != facts.mapping.rejected_candidates():
        problems.append(
            "re-materializing the same selection does not reproduce the rejected candidates"
        )
    return problems


def _round_trip_checks(
    facts: _Facts, entities: Sequence[Entity]
) -> list[SemanticMappingValidationCheck]:
    layer = SemanticMappingValidationLayer.ARTIFACT_ROUND_TRIP
    integrity = facts.mapping.verify_integrity()

    reopened: list[str] = []
    for entity in entities:
        if decode_entity(json.loads(json.dumps(encode_entity(entity)))) != entity:
            reopened.append(f"{entity.entity_id}: encoding and decoding it changes the record")
        if facts.mapping.entity(entity.reference) != entity:
            reopened.append(f"{entity.entity_id}: it does not reopen unchanged by reference")

    return [
        _check(layer, "run_integrity", len(facts.mapping.manifest.file_inventory), integrity),
        _check(layer, "entities_reopen_unchanged", len(entities), reopened),
        _check(
            layer,
            "derived_indexes_agree_with_the_entities",
            len(entities),
            _index_failures(facts, entities),
        ),
    ]


def _index_failures(facts: _Facts, entities: Sequence[Entity]) -> list[str]:
    mapping = facts.mapping
    by_id = {str(entity.entity_id): entity for entity in entities}
    problems: list[str] = []

    def rows(name: str) -> list[dict[str, Any]]:
        return mapping.read_table(f"outputs/{name}.jsonl")

    geometry = rows("entity-geometry-index")
    semantic = rows("entity-semantic-state")
    temporal = rows("entity-temporal-state")
    for name, table in (
        ("geometry", geometry),
        ("semantic-state", semantic),
        ("temporal-state", temporal),
    ):
        if [row["entity_id"] for row in table] != list(by_id):
            problems.append(f"the {name} index does not list exactly the entities")
    for row in geometry:
        entity = by_id.get(row["entity_id"])
        if entity is not None and row["point_count"] != len(entity.geometry.geometry_refs):
            problems.append(f"{row['entity_id']}: the geometry index counts other points")
    for row in semantic:
        entity = by_id.get(row["entity_id"])
        if entity is not None and (
            row["ambiguity_state"] != entity.semantic_state.ambiguity_state.value
            or row["hypothesis_labels"] != [item.label for item in entity.semantic_state.hypotheses]
        ):
            problems.append(f"{row['entity_id']}: the semantic-state index differs from the entity")
    for row in temporal:
        entity = by_id.get(row["entity_id"])
        if entity is not None and (
            row["physical_observation_count"] != entity.temporal_state.physical_observation_count
            or row["inference_result_count"] != entity.temporal_state.inference_result_count
        ):
            problems.append(f"{row['entity_id']}: the temporal-state index differs from the entity")
    observed = Counter(row["entity_id"] for row in rows("entity-observation-index"))
    for entity_id, entity in by_id.items():
        if observed.get(entity_id, 0) != len(entity.temporal_state.observation_refs):
            problems.append(f"{entity_id}: the observation index lists other frames")
    counts = mapping.read_record("metrics/counts.json")
    if counts["entities"] != len(entities):
        problems.append("the metrics count other entities than the run holds")
    return problems

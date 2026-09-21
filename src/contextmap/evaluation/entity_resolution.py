"""Deterministic validation and identity evaluation of Entity Resolution runs.

Entity Resolution decides which source entities are one physical object, so a false merge combines
distinct objects and a missed merge leaves one object duplicated. This module reads a persisted
resolution run through its public reader and reports, without collapsing anything into a score:

* **contract and provenance** -- decisions point to evidence that exists, only candidate pairs were
  compared, every source entity belongs to exactly one resolved entity, materializing the same
  decisions again reproduces the resolved entities and their ids, retrieving candidates again
  reproduces the candidate sets, and no measured channel mixes embedding or representation spaces;
* **identity** -- against an explicit :class:`~contextmap.evaluation.IdentityAnnotationSet`, never
  inferred from equal labels or file names: the registry metrics ``entity.false_merge.rate`` and
  ``entity.duplicate.rate`` exactly as defined there, plus every failure class kept as its own
  count (false merge, missed merge by cause, false distinct decision, false match, abstention);
* **retrieval** -- whether the true match was retrieved at all, judged *before* the policy, so a
  miss caused by retrieval is distinguishable from a policy error;
* **split diagnostics** -- evaluated apart from merge quality, on entities labeled single or
  multi-object;
* **channel ablations** -- one arm per set of enabled channels over the same source entities,
  candidates and reference, so only the channels change.

Identity references are keyed by *observation occurrences* (sample, observation, region), not by
entities, so the link between an occurrence and the source entity that came from it is an explicit
input (:class:`OccurrenceLink`), never guessed. Only occurrences in observations with a ``COMPLETE``
identity scope are evaluated; an entity without an annotated occurrence is excluded, not counted.
``entity.semantic_accuracy.rate`` is a semantic measure and is deliberately not reported here:
label correctness is never reported as identity correctness.

This is a **harness**. It never repairs a run, tunes a threshold, or reads ``debug/``. The
thresholds of a policy must not be tuned on the same reference set the final report uses without
saying so; the report records which reference and which policies it was computed with. All tests use
synthetic entities: nothing here is evidence of quality on real data.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from itertools import combinations
from typing import Any

from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    EntityCandidateSet,
    EntityMatchEvidence,
    EntityResolutionRunId,
    EntityResolutionRunReader,
    MatchChannel,
    MatchEvidenceBuilder,
    ResolutionDecision,
    ResolutionOutcome,
    ResolvedEntity,
    ResolvedEntityReference,
    ResolvedEntitySet,
    SplitCandidate,
    SplitStatus,
    TransitivityContradiction,
    candidate_pairs,
    materialize_resolved_entities,
    resolution_artifact_digest,
    resolve_candidate_pairs,
    retrieve_candidate_sets,
)
from contextmap.entity_resolution import ConservativeResolutionPolicy as ResolutionPolicy
from contextmap.evaluation.annotations import Coverage, IdentityAnnotationSet
from contextmap.evaluation.reference_set import ReferenceSampleId
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_mapping import Entity, EntityReference

ENTITY_RESOLUTION_EVALUATOR_ID = "entity-resolution-evaluator"
"""Identity of this evaluator, as the metric registry names it."""

ENTITY_RESOLUTION_EVALUATOR_VERSION = "0.1.0"
"""Version of the checks, definitions and report schema in this module."""

_OccurrenceKey = tuple[str, str, str | None]
_Pair = tuple[EntityReference, EntityReference]


class EntityResolutionEvaluationError(ValueError):
    """Raised when the inputs of an evaluation are inconsistent and cannot be evaluated."""


class ResolutionValidationLayer(Enum):
    """The layer of contract and provenance checks a check belongs to.

    Attributes:
        CONTRACT: Records that must be consistent with each other.
        REPRODUCIBILITY: Results that identical inputs must reproduce.
        COMPATIBILITY: Embedding and representation spaces that must never be mixed.
    """

    CONTRACT = "contract"
    REPRODUCIBILITY = "reproducibility"
    COMPATIBILITY = "compatibility"


@dataclass(frozen=True, kw_only=True)
class ResolutionValidationCheck:
    """One check of a resolution run: what it examined and exactly what failed.

    Attributes:
        name: What the check verifies.
        layer: The layer it belongs to.
        examined: How many items it examined; ``0`` means the check had nothing to look at.
        failures: The exact failures found; empty means the check passed.
    """

    name: str
    layer: ResolutionValidationLayer
    examined: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether the check found no failure."""
        return not self.failures


@dataclass(frozen=True, kw_only=True)
class OccurrenceLink:
    """The explicit link between an annotated occurrence and the source entity it produced.

    Attributes:
        sample_id: The reference sample the occurrence is in.
        observation_id: The physical observation the occurrence is in.
        region_id: The region of the occurrence, or ``None`` when the annotation names none.
        entity_ref: The source entity that came from that occurrence.
    """

    sample_id: ReferenceSampleId
    observation_id: SourceObservationId
    region_id: str | None
    entity_ref: EntityReference


@dataclass(frozen=True, kw_only=True)
class SplitReference:
    """The truth about one entity for split evaluation.

    Attributes:
        entity_ref: The source entity.
        contains_multiple_objects: Whether it really holds more than one physical object.
    """

    entity_ref: EntityReference
    contains_multiple_objects: bool


@dataclass(frozen=True, kw_only=True)
class IdentityEvaluation:
    """Identity quality of one resolution, with every failure class visible.

    Rates are ``None`` when their denominator is zero (the data is excluded, never a zero score).

    Attributes:
        resolved_entities: Resolved entities with at least one annotated member: the population
            of the false-merge rate.
        false_merge_entities: Resolved entities that merge annotated-distinct identities.
        false_merge_rate: ``false_merge_entities / resolved_entities``, the registry metric
            ``entity.false_merge.rate``.
        annotated_identities: Identities with at least one linked, evaluable entity.
        duplicated_identities: Identities represented by more than one resolved entity.
        duplicate_rate: ``duplicated_identities / annotated_identities``, the registry metric
            ``entity.duplicate.rate``.
        same_pairs: Pairs of entities that share an annotated identity.
        distinct_pairs: Pairs of entities of identities declared distinct.
        same_pair_outcomes: How each same pair ended, by mutually exclusive cause, sorted by cause.
        distinct_pair_outcomes: How each declared-distinct pair ended, sorted by cause.
        true_merges: Same pairs that ended in one resolved entity.
        false_merges: Declared-distinct pairs that ended in one resolved entity.
        pairwise_precision: ``true_merges / (true_merges + false_merges)``.
        pairwise_recall: ``true_merges / same_pairs``.
        source_entities_spanning_identities: Source entities linked to more than one identity:
            over-merged before resolution, so not a resolution error.
        identity_of_resolved_entity: The annotated identity of every resolved entity whose
            evaluable members all belong to one identity, sorted by reference. This is the mapping
            a consumer of the resolved entities needs to evaluate its own output by identity
            (``dict(...)`` of it). Several resolved entities may map to the same identity (that is
            a duplicate, and stays visible in ``duplicated_identities``).
        resolved_entities_spanning_identities: Resolved entities whose evaluable members belong to
            more than one identity, so they have no single identity: a false merge, or a source
            entity that was already over-merged. They are left out of the mapping, never guessed.
        transitivity_contradictions: Contradictions the materialization surfaced.
        ignored_links: Links whose occurrence is not in a ``COMPLETE`` scope or not annotated.
    """

    resolved_entities: int
    false_merge_entities: int
    false_merge_rate: float | None
    annotated_identities: int
    duplicated_identities: int
    duplicate_rate: float | None
    same_pairs: int
    distinct_pairs: int
    same_pair_outcomes: tuple[tuple[str, int], ...]
    distinct_pair_outcomes: tuple[tuple[str, int], ...]
    true_merges: int
    false_merges: int
    pairwise_precision: float | None
    pairwise_recall: float | None
    source_entities_spanning_identities: int
    identity_of_resolved_entity: tuple[tuple[ResolvedEntityReference, str], ...]
    resolved_entities_spanning_identities: tuple[ResolvedEntityReference, ...]
    transitivity_contradictions: int
    ignored_links: int


@dataclass(frozen=True, kw_only=True)
class RetrievalEvaluation:
    """Whether candidate retrieval found the pairs the policy would need, before any decision.

    Attributes:
        same_pairs: Pairs of entities that share an annotated identity.
        retrieved_same_pairs: Those that are candidate pairs.
        retrieval_recall: ``retrieved_same_pairs / same_pairs``.
        candidate_pairs: Distinct candidate pairs of the run.
        candidates_per_entity_mean: Mean candidates per source entity.
        candidates_per_entity_max: Most candidates of one entity.
        compared_pairs: Candidate pairs that were compared.
    """

    same_pairs: int
    retrieved_same_pairs: int
    retrieval_recall: float | None
    candidate_pairs: int
    candidates_per_entity_mean: float | None
    candidates_per_entity_max: int
    compared_pairs: int


@dataclass(frozen=True, kw_only=True)
class SplitEvaluation:
    """Split diagnostics judged apart from merge quality.

    Attributes:
        multi_object_entities: Labeled entities that really hold several objects.
        multi_object_outcomes: How the detector ended for them, by status or ``not_flagged``.
        single_object_entities: Labeled entities that hold one object.
        single_object_outcomes: How the detector ended for them.
        suggested_recall: Share of multi-object entities with a ``suggested`` candidate.
        false_suggestion_rate: Share of single-object entities with a ``suggested`` candidate.
    """

    multi_object_entities: int
    multi_object_outcomes: tuple[tuple[str, int], ...]
    single_object_entities: int
    single_object_outcomes: tuple[tuple[str, int], ...]
    suggested_recall: float | None
    false_suggestion_rate: float | None


@dataclass(frozen=True, kw_only=True)
class ArmEvaluation:
    """The result of one channel-ablation arm.

    Attributes:
        name: The arm, for example ``geometry`` or ``geometry+semantic``.
        channels: The channels the arm's policy uses.
        min_supporting_channels: How many channels must support a match in this arm.
        decision_counts: Decisions by outcome, sorted by outcome.
        identity: The identity evaluation of the arm's resolved entities.
        resolve_seconds: Wall time to build the evidence and decide every pair, measured here.
        materialize_seconds: Wall time to materialize the resolved entities.
    """

    name: str
    channels: tuple[str, ...]
    min_supporting_channels: int
    decision_counts: tuple[tuple[str, int], ...]
    identity: IdentityEvaluation
    resolve_seconds: float
    materialize_seconds: float


@dataclass(frozen=True, kw_only=True)
class ResolutionArm:
    """One arm of a channel ablation.

    Attributes:
        name: The arm.
        builder: Collects the evidence of a pair with the arm's channels.
        policy: Decides each pair, using the channels the arm enables.
    """

    name: str
    builder: MatchEvidenceBuilder
    policy: ResolutionPolicy


@dataclass(frozen=True, kw_only=True)
class EvaluationReproducibility:
    """What a report was computed from, so it can be reproduced and audited.

    Attributes:
        evaluator_id: The evaluator that computed it.
        evaluator_version: The version of the evaluator.
        run_id: The resolution run evaluated.
        resolution_schema_version: The schema version of that run, from its manifest.
        resolution_artifact_digest: Digest of that run's identity and inventory, computed by
            Entity Resolution: the exact artifact a downstream artifact should pin.
        semantic_mapping_run_id: The source semantic-entity artifact.
        semantic_mapping_artifact_digest: Digest of that artifact's identity and inventory.
        reference_set_id: The version of the annotated reference the run was evaluated against.
        annotation_digest: Digest of the identity annotations used.
        policies: ``role:policy_id:fingerprint`` of every policy of the run, sorted.
        code_version: Code revision that produced the run.
    """

    evaluator_id: str
    evaluator_version: str
    run_id: str
    resolution_schema_version: str
    resolution_artifact_digest: str
    semantic_mapping_run_id: str
    semantic_mapping_artifact_digest: str
    reference_set_id: str
    annotation_digest: str
    policies: tuple[str, ...]
    code_version: str


@dataclass(frozen=True, kw_only=True)
class EntityResolutionEvaluationReport:
    """Every layer of the evaluation of one resolution run; none is merged into a score.

    Attributes:
        reproducibility: What the report was computed from.
        checks: The contract, reproducibility and compatibility checks.
        identity: The identity evaluation of the run.
        retrieval: The retrieval evaluation, judged before the policy.
        split: The split evaluation, or ``None`` when no split reference was given.
        artifact_bytes: Total size of the contractual files of the run.
    """

    reproducibility: EvaluationReproducibility
    checks: tuple[ResolutionValidationCheck, ...]
    identity: IdentityEvaluation
    retrieval: RetrievalEvaluation
    split: SplitEvaluation | None
    artifact_bytes: int

    @property
    def passed(self) -> bool:
        """Whether every contract, reproducibility and compatibility check passed."""
        return all(check.passed for check in self.checks)


def evaluate_entity_resolution(
    reader: EntityResolutionRunReader,
    *,
    entities: Iterable[Entity],
    reference: IdentityAnnotationSet,
    links: Sequence[OccurrenceLink],
    retrieval_policy: CandidateRetrievalPolicy,
    reference_set_id: str,
    split_reference: Sequence[SplitReference] = (),
) -> EntityResolutionEvaluationReport:
    """Evaluate a persisted resolution run against an explicit identity reference.

    Args:
        reader: The opened run; only contractual files are read, never ``debug/``.
        entities: The source entities the run was built from, to reproduce retrieval and
            materialization.
        reference: The annotated identities; never inferred from labels.
        links: The explicit link between annotated occurrences and source entities.
        retrieval_policy: The retrieval policy of the run, to check that retrieval reproduces.
        reference_set_id: The version of the annotated reference set, recorded on the report.
        split_reference: Entities labeled single or multi-object, to evaluate split diagnostics.

    Returns:
        The report: checks, identity, retrieval, optional split, and reproducibility.

    Raises:
        EntityResolutionEvaluationError: If a link names an entity the run does not have.
    """
    source_entities = tuple(entities)
    candidate_sets = reader.candidate_sets()
    decisions = reader.decisions()
    evidence = reader.match_evidence()
    resolved = reader.resolved_entities()
    contradictions = reader.contradictions()
    manifest = reader.manifest
    checks = _checks(
        reader,
        source_entities,
        candidate_sets,
        decisions,
        evidence,
        resolved,
        contradictions,
        retrieval_policy,
    )
    identity = evaluate_identity(
        reference=reference,
        links=links,
        resolved=resolved,
        candidate_sets=candidate_sets,
        decisions=decisions,
        contradictions=contradictions,
    )
    retrieval = _retrieval(reference, links, resolved, candidate_sets, decisions)
    policies = tuple(
        sorted(
            f"{item.role}:{item.policy.policy_id}:{item.policy.configuration_fingerprint}"
            for item in manifest.policies
        )
    )
    return EntityResolutionEvaluationReport(
        reproducibility=EvaluationReproducibility(
            evaluator_id=ENTITY_RESOLUTION_EVALUATOR_ID,
            evaluator_version=ENTITY_RESOLUTION_EVALUATOR_VERSION,
            run_id=str(manifest.run_id),
            resolution_schema_version=manifest.schema_version,
            resolution_artifact_digest=resolution_artifact_digest(manifest),
            semantic_mapping_run_id=manifest.lineage.semantic_mapping_run_id,
            semantic_mapping_artifact_digest=manifest.lineage.semantic_mapping_artifact_digest,
            reference_set_id=reference_set_id,
            annotation_digest=_annotation_digest(reference),
            policies=policies,
            code_version=manifest.code_version,
        ),
        checks=checks,
        identity=identity,
        retrieval=retrieval,
        split=evaluate_splits(reader.split_candidates(), split_reference)
        if split_reference
        else None,
        artifact_bytes=sum(entry.size_bytes for entry in manifest.file_inventory),
    )


def evaluate_identity(
    *,
    reference: IdentityAnnotationSet,
    links: Sequence[OccurrenceLink],
    resolved: ResolvedEntitySet,
    candidate_sets: Sequence[EntityCandidateSet],
    decisions: Sequence[ResolutionDecision],
    contradictions: Sequence[TransitivityContradiction] = (),
) -> IdentityEvaluation:
    """Evaluate resolved entities against explicit identities, keeping every failure class.

    Args:
        reference: The annotated identities and the pairs declared distinct.
        links: The explicit link between annotated occurrences and source entities.
        resolved: The resolved entities.
        candidate_sets: The candidate sets, to tell a retrieval miss from a policy outcome.
        decisions: The decisions, to tell how each annotated pair ended.
        contradictions: The transitivity contradictions surfaced by the materialization.

    Returns:
        The identity evaluation. Rates are ``None`` when their denominator is zero.

    Raises:
        EntityResolutionEvaluationError: If a link names an entity the resolved set does not have.
    """
    identities_of, ignored = _identities_of_entities(reference, links, resolved)
    declared = {frozenset((pair.first, pair.second)) for pair in reference.distinct_pairs}
    resolved_of = {
        member: entity for entity in resolved.entities for member in entity.member_entity_refs
    }
    same, distinct = _annotated_pairs(identities_of, declared)
    candidates = set(candidate_pairs(candidate_sets))
    decided = {(item.entity_a_ref, item.entity_b_ref): item for item in decisions}
    same_outcomes = Counter(
        _outcome(pair, "same", resolved_of, candidates, decided) for pair in same
    )
    distinct_outcomes = Counter(
        _outcome(pair, "distinct", resolved_of, candidates, decided) for pair in distinct
    )
    true_merges = same_outcomes["merged"]
    false_merges = distinct_outcomes["false_merge"]
    population = [
        entity
        for entity in resolved.entities
        if any(member in identities_of for member in entity.member_entity_refs)
    ]
    false_entities = len(
        {
            resolved_of[first].resolved_entity_id
            for first, second in distinct
            if _same_entity(first, second, resolved_of)
        }
    )
    entities_of_identity: dict[str, set[str]] = {}
    for entity_ref, identity_ids in identities_of.items():
        for identity_id in identity_ids:
            entities_of_identity.setdefault(identity_id, set()).add(
                str(resolved_of[entity_ref].resolved_entity_id)
            )
    duplicated = sum(1 for members in entities_of_identity.values() if len(members) > 1)
    identities_of_resolved: dict[ResolvedEntityReference, set[str]] = {}
    for entity_ref, identity_ids in identities_of.items():
        identities_of_resolved.setdefault(resolved_of[entity_ref].reference, set()).update(
            identity_ids
        )
    ordered = sorted(
        identities_of_resolved, key=lambda ref: (ref.resolution_run_id, ref.resolved_entity_id)
    )
    return IdentityEvaluation(
        resolved_entities=len(population),
        false_merge_entities=false_entities,
        false_merge_rate=_rate(false_entities, len(population)),
        annotated_identities=len(entities_of_identity),
        duplicated_identities=duplicated,
        duplicate_rate=_rate(duplicated, len(entities_of_identity)),
        same_pairs=len(same),
        distinct_pairs=len(distinct),
        same_pair_outcomes=tuple(sorted(same_outcomes.items())),
        distinct_pair_outcomes=tuple(sorted(distinct_outcomes.items())),
        true_merges=true_merges,
        false_merges=false_merges,
        pairwise_precision=_rate(true_merges, true_merges + false_merges),
        pairwise_recall=_rate(true_merges, len(same)),
        source_entities_spanning_identities=sum(
            1 for ids in identities_of.values() if len(ids) > 1
        ),
        identity_of_resolved_entity=tuple(
            (ref, next(iter(identities_of_resolved[ref])))
            for ref in ordered
            if len(identities_of_resolved[ref]) == 1
        ),
        resolved_entities_spanning_identities=tuple(
            ref for ref in ordered if len(identities_of_resolved[ref]) > 1
        ),
        transitivity_contradictions=len(contradictions),
        ignored_links=ignored,
    )


def evaluate_splits(
    candidates: Sequence[SplitCandidate], reference: Sequence[SplitReference]
) -> SplitEvaluation:
    """Evaluate split diagnostics on entities labeled single or multi-object.

    Kept apart from merge quality on purpose: it says whether the detector flags over-merged
    entities and leaves single objects alone, and nothing about pairwise resolution.

    Args:
        candidates: The split candidates the detector produced.
        reference: The entities labeled single or multi-object.

    Returns:
        The split evaluation.
    """
    status = {item.entity_ref: item.status.value for item in candidates}
    multi = Counter(
        status.get(item.entity_ref, "not_flagged")
        for item in reference
        if item.contains_multiple_objects
    )
    single = Counter(
        status.get(item.entity_ref, "not_flagged")
        for item in reference
        if not item.contains_multiple_objects
    )
    return SplitEvaluation(
        multi_object_entities=sum(multi.values()),
        multi_object_outcomes=tuple(sorted(multi.items())),
        single_object_entities=sum(single.values()),
        single_object_outcomes=tuple(sorted(single.items())),
        suggested_recall=_rate(multi[SplitStatus.SUGGESTED.value], sum(multi.values())),
        false_suggestion_rate=_rate(single[SplitStatus.SUGGESTED.value], sum(single.values())),
    )


def evaluate_channel_ablation(
    *,
    arms: Sequence[ResolutionArm],
    entities: Sequence[Entity],
    candidate_sets: Sequence[EntityCandidateSet],
    reference: IdentityAnnotationSet,
    links: Sequence[OccurrenceLink],
    resolution_run_id: EntityResolutionRunId,
) -> tuple[ArmEvaluation, ...]:
    """Run each arm over the same entities and candidates and evaluate the resolved entities.

    The source entities, the candidate sets, the reference and the links are fixed across arms, so
    only the enabled channels change. Latency is measured here and reported apart from quality.

    Args:
        arms: One arm per set of enabled channels, for example geometry only, geometry plus
            semantic, and so on.
        entities: The source entities.
        candidate_sets: The candidate sets, shared by every arm.
        reference: The annotated identities.
        links: The explicit link between annotated occurrences and source entities.
        resolution_run_id: The identity given to the resolved entities of each arm.

    Returns:
        One evaluation per arm, in the order given.

    Raises:
        EntityResolutionEvaluationError: If two arms share a name.
    """
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise EntityResolutionEvaluationError("arm names must be unique")
    results = []
    for arm in arms:
        started = time.perf_counter()
        resolutions = resolve_candidate_pairs(entities, candidate_sets, arm.builder, arm.policy)
        resolved_at = time.perf_counter()
        materialization = materialize_resolved_entities(
            entities,
            [item.decision for item in resolutions],
            resolution_run_id=resolution_run_id,
        )
        finished = time.perf_counter()
        decisions = [item.decision for item in resolutions]
        counts = Counter(item.decision.value for item in decisions)
        results.append(
            ArmEvaluation(
                name=arm.name,
                channels=tuple(channel.value for channel in arm.policy.use_channels),
                min_supporting_channels=arm.policy.min_supporting_channels,
                decision_counts=tuple(sorted(counts.items())),
                identity=evaluate_identity(
                    reference=reference,
                    links=links,
                    resolved=materialization.resolved,
                    candidate_sets=candidate_sets,
                    decisions=decisions,
                    contradictions=materialization.contradictions,
                ),
                resolve_seconds=resolved_at - started,
                materialize_seconds=finished - resolved_at,
            )
        )
    return tuple(results)


def encode_entity_resolution_report(report: EntityResolutionEvaluationReport) -> dict[str, Any]:
    """Encode a report as JSON-compatible data, with every count and layer kept apart."""
    return _encode(report)  # type: ignore[no-any-return]


def _encode(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple | list):
        return [_encode(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        record: dict[str, Any] = {
            item.name: _encode(getattr(value, item.name)) for item in fields(value)
        }
        if isinstance(value, ResolutionValidationCheck):
            record["passed"] = value.passed
        return record
    return value


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _annotation_digest(reference: IdentityAnnotationSet) -> str:
    canonical = json.dumps(reference.to_record(), sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def _identities_of_entities(
    reference: IdentityAnnotationSet, links: Sequence[OccurrenceLink], resolved: ResolvedEntitySet
) -> tuple[dict[EntityReference, set[str]], int]:
    """The annotated identities of each linked entity, from occurrences in COMPLETE scopes only."""
    owner: dict[_OccurrenceKey, str] = {}
    for identity in reference.identities:
        for occurrence in identity.occurrences:
            owner[(occurrence.sample_id, occurrence.observation_id, occurrence.region_id)] = (
                identity.identity_id
            )
    known = {member for entity in resolved.entities for member in entity.member_entity_refs}
    identities_of: dict[EntityReference, set[str]] = {}
    ignored = 0
    for link in links:
        if link.entity_ref not in known:
            raise EntityResolutionEvaluationError(
                f"link names entity {link.entity_ref.entity_id!r}, which the run does not have"
            )
        key = (link.sample_id, link.observation_id, link.region_id)
        coverage = reference.coverage_of(link.sample_id, link.observation_id)
        if key not in owner or coverage is not Coverage.COMPLETE:
            ignored += 1
            continue
        identities_of.setdefault(link.entity_ref, set()).add(owner[key])
    return identities_of, ignored


def _annotated_pairs(
    identities_of: Mapping[EntityReference, set[str]], declared: set[frozenset[str]]
) -> tuple[list[_Pair], list[_Pair]]:
    same: list[_Pair] = []
    distinct: list[_Pair] = []
    ordered = sorted(identities_of, key=lambda ref: (ref.semantic_map_id, ref.entity_id))
    for first, second in combinations(ordered, 2):
        first_ids, second_ids = identities_of[first], identities_of[second]
        if first_ids & second_ids:
            same.append((first, second))
        elif any(frozenset((a, b)) in declared for a in first_ids for b in second_ids):
            distinct.append((first, second))
    return same, distinct


def _same_entity(
    first: EntityReference,
    second: EntityReference,
    resolved_of: Mapping[EntityReference, ResolvedEntity],
) -> bool:
    return resolved_of[first].resolved_entity_id == resolved_of[second].resolved_entity_id


def _outcome(
    pair: _Pair,
    kind: str,
    resolved_of: Mapping[EntityReference, ResolvedEntity],
    candidates: set[_Pair],
    decided: Mapping[_Pair, ResolutionDecision],
) -> str:
    """How an annotated pair ended, by mutually exclusive cause, in order of precedence."""
    first, second = pair
    if _same_entity(first, second, resolved_of):
        return "merged" if kind == "same" else "false_merge"
    if pair not in candidates:
        return "retrieval_miss" if kind == "same" else "not_candidate"
    decision = decided.get(pair)
    if decision is None:
        return "not_compared"
    if decision.decision is ResolutionOutcome.MATCH:
        withheld = bool(resolved_of[first].contradiction_ids)
        return "match_withheld_by_contradiction" if withheld else "match_not_merged"
    if decision.decision is ResolutionOutcome.DISTINCT:
        return "false_distinct" if kind == "same" else "decided_distinct"
    return "unresolved"


def _retrieval(
    reference: IdentityAnnotationSet,
    links: Sequence[OccurrenceLink],
    resolved: ResolvedEntitySet,
    candidate_sets: Sequence[EntityCandidateSet],
    decisions: Sequence[ResolutionDecision],
) -> RetrievalEvaluation:
    identities_of, _ = _identities_of_entities(reference, links, resolved)
    same, _ = _annotated_pairs(identities_of, set())
    candidates = set(candidate_pairs(candidate_sets))
    retrieved = sum(1 for pair in same if pair in candidates)
    counts = [len(item.candidates) for item in candidate_sets]
    return RetrievalEvaluation(
        same_pairs=len(same),
        retrieved_same_pairs=retrieved,
        retrieval_recall=_rate(retrieved, len(same)),
        candidate_pairs=len(candidates),
        candidates_per_entity_mean=sum(counts) / len(counts) if counts else None,
        candidates_per_entity_max=max(counts, default=0),
        compared_pairs=len(decisions),
    )


def _checks(
    reader: EntityResolutionRunReader,
    entities: Sequence[Entity],
    candidate_sets: Sequence[EntityCandidateSet],
    decisions: Sequence[ResolutionDecision],
    evidence: Sequence[EntityMatchEvidence],
    resolved: ResolvedEntitySet,
    contradictions: Sequence[TransitivityContradiction],
    retrieval_policy: CandidateRetrievalPolicy,
) -> tuple[ResolutionValidationCheck, ...]:
    layer = ResolutionValidationLayer
    by_comparison = {item.comparison_id: item for item in evidence}
    pairs = set(candidate_pairs(candidate_sets))
    source_refs = {entity.reference for entity in entities}
    members = [ref for item in resolved.entities for ref in item.member_entity_refs]
    materialization = materialize_resolved_entities(
        entities, decisions, resolution_run_id=reader.run_id
    )
    return (
        ResolutionValidationCheck(
            name="every decision points to evidence that exists for its pair",
            layer=layer.CONTRACT,
            examined=len(decisions),
            failures=tuple(
                f"decision {item.decision_id!r}: no evidence {item.evidence_ref!r}"
                for item in decisions
                if item.evidence_ref not in by_comparison
                or (
                    by_comparison[item.evidence_ref].entity_a_ref,
                    by_comparison[item.evidence_ref].entity_b_ref,
                )
                != (item.entity_a_ref, item.entity_b_ref)
            ),
        ),
        ResolutionValidationCheck(
            name="only candidate pairs were compared",
            layer=layer.CONTRACT,
            examined=len(decisions),
            failures=tuple(
                f"pair {item.entity_a_ref.entity_id!r}/{item.entity_b_ref.entity_id!r} "
                f"is not a candidate"
                for item in decisions
                if (item.entity_a_ref, item.entity_b_ref) not in pairs
            ),
        ),
        ResolutionValidationCheck(
            name="every source entity belongs to exactly one resolved entity",
            layer=layer.CONTRACT,
            examined=len(source_refs),
            failures=tuple(
                [
                    f"source entity {ref.entity_id!r} is in no resolved entity"
                    for ref in sorted(source_refs - set(members), key=lambda r: r.entity_id)
                ]
                + [
                    f"resolved member {ref.entity_id!r} is not a source entity"
                    for ref in sorted(set(members) - source_refs, key=lambda r: r.entity_id)
                ]
            ),
        ),
        ResolutionValidationCheck(
            name="the artifact is intact and its integrity holds",
            layer=layer.CONTRACT,
            examined=len(reader.manifest.file_inventory),
            failures=tuple(reader.verify_integrity()),
        ),
        ResolutionValidationCheck(
            name="materializing the same decisions reproduces the resolved entities and their ids",
            layer=layer.REPRODUCIBILITY,
            examined=len(resolved.entities),
            failures=()
            if materialization.resolved == resolved
            else ("the resolved entities differ from a fresh materialization",),
        ),
        ResolutionValidationCheck(
            name="contradictions are surfaced exactly as a fresh materialization finds them",
            layer=layer.REPRODUCIBILITY,
            examined=len(contradictions),
            failures=()
            if materialization.contradictions == tuple(contradictions)
            else ("the contradictions differ from a fresh materialization",),
        ),
        ResolutionValidationCheck(
            name="retrieving candidates again reproduces the candidate sets",
            layer=layer.REPRODUCIBILITY,
            examined=len(candidate_sets),
            failures=()
            if tuple(retrieve_candidate_sets(entities, retrieval_policy)) == tuple(candidate_sets)
            else ("the candidate sets differ from a fresh retrieval",),
        ),
        ResolutionValidationCheck(
            name="no measured channel mixes embedding or representation spaces",
            layer=layer.COMPATIBILITY,
            examined=len(evidence),
            failures=_mixed_spaces(evidence),
        ),
    )


def _mixed_spaces(evidence: Sequence[EntityMatchEvidence]) -> tuple[str, ...]:
    spaces: dict[MatchChannel, set[str]] = {}
    for item in evidence:
        appearance, representation = item.appearance, item.point_representation
        if appearance is not None and appearance.measurement is not None:
            spaces.setdefault(MatchChannel.APPEARANCE, set()).add(
                appearance.measurement.embedding_space_id
            )
        if representation is not None and representation.measurement is not None:
            spaces.setdefault(MatchChannel.POINT_REPRESENTATION, set()).add(
                representation.measurement.representation_space_id
            )
    return tuple(
        f"the {channel.value} channel was measured in {len(found)} different spaces"
        for channel, found in sorted(spaces.items(), key=lambda pair: pair[0].value)
        if len(found) > 1
    )

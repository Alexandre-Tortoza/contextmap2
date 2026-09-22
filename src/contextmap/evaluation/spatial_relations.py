"""Evaluation of persisted spatial relations against an annotated reference.

Incorrect relations can make an otherwise correct contextual map misleading, and they can be wrong
for four different reasons: the entities were resolved wrongly, the candidate stage never looked at
the pair, a geometric predicate measured it wrongly, or the reconciliation left it undecided. This
evaluator keeps those apart instead of folding them into one number, and it never repairs a wrong
entity to save a relation.

It reads a persisted ``SpatialRelationsRunArtifact`` and a ``RelationAnnotationSet``. The two only
meet through the ``IdentityEvaluation`` of Entity Resolution's own evaluation, which says which
annotated identity every resolved entity has and which resolved entities span several identities
(false merges): a reference relation whose identity has no matched entity is counted as an
unmatched reference, attributed to entity resolution, and is not scored as a relation error. That
identity evaluation is trusted only once its own reproducibility metadata names this exact
resolution run and artifact digest; nothing about which run it is about is ever inferred from the
identity mapping itself, since an empty mapping is a valid outcome and proves nothing.

Everything is reported **per canonical predicate**; there is no overall score. For each predicate
the counts are kept separate:

* ``true_positives`` -- annotated to hold and predicted ``SUPPORTED``;
* ``false_negatives`` (missed relations), split into ``missed_unresolved``, ``missed_rejected`` and
  ``missed_not_retrieved`` (the candidate stage never produced the pair: a retrieval miss, kept with
  its reason);
* ``false_positives`` -- annotated *not* to hold and predicted ``SUPPORTED`` (negative violations);
* ``true_negatives`` and ``unresolved_on_negatives`` for annotated negatives;
* ``unannotated_supported`` -- supported relations nobody annotated. The annotation is open-world,
  so an unannotated prediction is reported, never scored as a false positive.

Reference relations with an ambiguous or unknown status are counted and never scored.

The reference's predicate wording is mapped to a canonical predicate only through the annotation
set's own versioned normalization and the exact canonical names; anything else is reported as
unmapped. The taxonomy expands the reference by inverse and symmetry, and a reference vocabulary
that declares a predicate's symmetry or inverse differently from the taxonomy is refused rather
than silently reinterpreted. The internal consistency of the persisted relations (inverse,
symmetric and mutual support of directed predicates) is checked apart from any annotation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.evaluation.annotations import (
    AnnotationFamily,
    RelationAnnotationSet,
    RelationStatus,
)
from contextmap.evaluation.entity_resolution import EvaluationReproducibility, IdentityEvaluation
from contextmap.evaluation.metrics import EvaluationStage, MetricRegistry
from contextmap.evaluation.reference_set import ReferenceSetIdentity
from contextmap.evaluation.report_schema import (
    ArtifactIdentity,
    EvaluationReport,
    EvaluatorIdentity,
    MetricResult,
    MetricStatus,
    ReproducibilityMetadata,
    assemble_evaluation_report,
)
from contextmap.spatial_relations import (
    PREDICATE_SPECS,
    Relation,
    RelationCandidateSet,
    RelationPredicate,
    RelationState,
    SpatialRelationsRunReader,
    canonical_predicate,
    predicate_spec,
)

SPATIAL_RELATIONS_EVALUATOR_ID = "spatial-relations-evaluator"
"""Identity of the evaluator, as the metric registry names it."""

SPATIAL_RELATIONS_EVALUATOR_VERSION = "2"
"""Version of the evaluation rules described in this module.

Version 2 requires the identity evaluation's own reproducibility metadata
(``identity_reproducibility``) and validates its run id and artifact digest against this run's
lineage before scoring; version 1 inferred the run id from the resolved-entity references present
in the identity mapping, which an empty mapping (a valid outcome) left unchecked.
"""

_Key = tuple[ResolvedEntityReference, RelationPredicate, ResolvedEntityReference]


class SpatialRelationsEvaluationError(ValueError):
    """Raised when the reference cannot be compared with the run under the taxonomy."""


@dataclass(frozen=True, kw_only=True)
class RelationPredicateEvaluation:
    """The outcomes of one canonical predicate; every figure is a count until a rate is asked.

    Attributes:
        predicate: The canonical predicate.
        annotated_holds: Reference relations annotated to hold, over matched entities.
        annotated_does_not_hold: Reference relations annotated not to hold, over matched entities.
        true_positives: Annotated to hold and predicted ``SUPPORTED``.
        missed_unresolved: Annotated to hold and predicted ``UNRESOLVED``.
        missed_rejected: Annotated to hold and predicted ``REJECTED``.
        missed_not_retrieved: Annotated to hold and never a candidate: a retrieval miss.
        false_positives: Annotated not to hold and predicted ``SUPPORTED``: a negative violation.
        true_negatives: Annotated not to hold and predicted ``REJECTED`` or never a candidate.
        unresolved_on_negatives: Annotated not to hold and predicted ``UNRESOLVED``.
        unannotated_supported: Predicted ``SUPPORTED`` between matched entities with no reference.
    """

    predicate: RelationPredicate
    annotated_holds: int = 0
    annotated_does_not_hold: int = 0
    true_positives: int = 0
    missed_unresolved: int = 0
    missed_rejected: int = 0
    missed_not_retrieved: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    unresolved_on_negatives: int = 0
    unannotated_supported: int = 0

    @property
    def false_negatives(self) -> int:
        """Annotated to hold and not predicted ``SUPPORTED``, for any of the three reasons."""
        return self.missed_unresolved + self.missed_rejected + self.missed_not_retrieved

    @property
    def precision(self) -> float | None:
        """``TP / (TP + FP)`` over annotated relations; ``None`` without a decided prediction."""
        return _ratio(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float | None:
        """``TP / annotated holds``; ``None`` without an annotated relation that holds."""
        return _ratio(self.true_positives, self.annotated_holds)

    @property
    def f1(self) -> float | None:
        """Harmonic mean of precision and recall; ``None`` unless both are defined."""
        precision, recall = self.precision, self.recall
        if precision is None or recall is None:
            return None
        return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)

    @property
    def false_relation_rate(self) -> float | None:
        """``FP / annotated negatives``: the registry's ``relations.negative_violation.rate``."""
        return _ratio(self.false_positives, self.annotated_does_not_hold)

    @property
    def missed_relation_rate(self) -> float | None:
        """``FN / annotated holds``."""
        return _ratio(self.false_negatives, self.annotated_holds)

    @property
    def unresolved_rate(self) -> float | None:
        """Annotated decided relations that ended ``UNRESOLVED``, over all annotated decided."""
        unresolved = self.missed_unresolved + self.unresolved_on_negatives
        return _ratio(unresolved, self.annotated_holds + self.annotated_does_not_hold)

    @property
    def candidate_retrieval_recall(self) -> float | None:
        """Annotated relations that hold and reached the candidate stage, over all that hold."""
        return _ratio(self.annotated_holds - self.missed_not_retrieved, self.annotated_holds)

    @property
    def decided_population(self) -> int:
        """Annotated relations that hold or do not hold, over matched entities."""
        return self.annotated_holds + self.annotated_does_not_hold


@dataclass(frozen=True, kw_only=True)
class RelationConsistencyViolation:
    """A structural contradiction in the persisted relations, found without any annotation.

    Attributes:
        kind: ``inverse``, ``symmetric`` or ``mutual_support``.
        relation_ids: The relations involved.
        detail: A deterministic explanation.
    """

    kind: str
    relation_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True, kw_only=True)
class RelationUnmatchedReport:
    """What could not be compared, and whose failure it is.

    Attributes:
        reference_without_entity: Reference relations (after expansion) with an identity that no
            resolved entity was matched to: attributed to entity resolution, never to relations.
        identities_with_several_entities: Identities matched to more than one entity, sorted;
            their relations are skipped, since a duplicated entity makes the comparison unfair.
        supported_on_unmatched_entities: Supported relations of entities with no identity.
        entities_spanning_identities: Resolved entities, by id and sorted, whose members belong to
            several identities (a false merge, or an over-merged source entity). Entity Resolution
            leaves them out of the identity mapping, so their relations are never scored.
        unmapped_predicates: Reference predicate wordings that are not canonical, sorted.
        ambiguous_reference: Reference relations annotated as ambiguous.
        unknown_reference: Reference relations annotated as unknown.
        conflicting_reference: Expanded reference keys annotated with different statuses; none of
            them is scored.
    """

    reference_without_entity: int
    identities_with_several_entities: tuple[str, ...]
    supported_on_unmatched_entities: int
    entities_spanning_identities: tuple[str, ...]
    unmapped_predicates: tuple[str, ...]
    ambiguous_reference: int
    unknown_reference: int
    conflicting_reference: int


@dataclass(frozen=True, kw_only=True)
class SpatialRelationsEvaluationReport:
    """The evaluation of one run against one reference, with everything needed to reproduce it.

    Attributes:
        evaluator_id: The evaluator's identity.
        evaluator_version: The evaluation rules' version.
        spatial_relations_run_id: The evaluated run.
        artifact_schema_version: The run artifact schema version.
        entity_resolution_run_id: The resolved-entity artifact the relations are about.
        entity_resolution_artifact_digest: Its digest, from the run's lineage.
        geometric_map_id: The geometric map the evidence was measured on.
        taxonomy_version: The relation vocabulary version.
        policies: The effective policies of the run, each with its id and fingerprint.
        reference_normalization: The annotation set's label normalization policy.
        reference_relation_count: Relations the annotation set holds, before expansion.
        predicates: One evaluation per canonical predicate, in a fixed order.
        retrieval_misses: Reasons the candidate stage never produced a pair that holds, counted.
        consistency_violations: Structural contradictions in the persisted relations.
        unmatched: What could not be compared and whose failure it is.
        code_version: Code revision of the evaluator run, when known.
    """

    evaluator_id: str
    evaluator_version: str
    spatial_relations_run_id: str
    artifact_schema_version: str
    entity_resolution_run_id: str
    entity_resolution_artifact_digest: str
    geometric_map_id: str
    taxonomy_version: str
    policies: dict[str, dict[str, str]]
    reference_normalization: str
    reference_relation_count: int
    predicates: tuple[RelationPredicateEvaluation, ...]
    retrieval_misses: tuple[tuple[str, int], ...]
    consistency_violations: tuple[RelationConsistencyViolation, ...]
    unmatched: RelationUnmatchedReport
    code_version: str | None

    def predicate(self, predicate: RelationPredicate) -> RelationPredicateEvaluation:
        """The evaluation of one canonical predicate."""
        return next(item for item in self.predicates if item.predicate is predicate)

    def configuration_digest(self) -> str:
        """Digest of the effective policies of the run, for the reproducibility metadata."""
        canonical = json.dumps(self.policies, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        """The report as JSON-compatible data, with every count and every derived rate."""
        return {
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "spatial_relations_run_id": self.spatial_relations_run_id,
            "artifact_schema_version": self.artifact_schema_version,
            "entity_resolution_run_id": self.entity_resolution_run_id,
            "entity_resolution_artifact_digest": self.entity_resolution_artifact_digest,
            "geometric_map_id": self.geometric_map_id,
            "taxonomy_version": self.taxonomy_version,
            "policies": self.policies,
            "reference_normalization": self.reference_normalization,
            "reference_relation_count": self.reference_relation_count,
            "predicates": [_encode_predicate(item) for item in self.predicates],
            "retrieval_misses": [
                {"reason": reason, "count": count} for reason, count in self.retrieval_misses
            ],
            "consistency_violations": [
                {"kind": item.kind, "relation_ids": list(item.relation_ids), "detail": item.detail}
                for item in self.consistency_violations
            ],
            "unmatched": {
                "reference_without_entity": self.unmatched.reference_without_entity,
                "identities_with_several_entities": list(
                    self.unmatched.identities_with_several_entities
                ),
                "supported_on_unmatched_entities": self.unmatched.supported_on_unmatched_entities,
                "entities_spanning_identities": list(self.unmatched.entities_spanning_identities),
                "unmapped_predicates": list(self.unmatched.unmapped_predicates),
                "ambiguous_reference": self.unmatched.ambiguous_reference,
                "unknown_reference": self.unmatched.unknown_reference,
                "conflicting_reference": self.unmatched.conflicting_reference,
            },
            "code_version": self.code_version,
        }


def evaluate_spatial_relations(
    reader: SpatialRelationsRunReader,
    *,
    reference: RelationAnnotationSet,
    identity: IdentityEvaluation,
    identity_reproducibility: EvaluationReproducibility,
    code_version: str | None = None,
) -> SpatialRelationsEvaluationReport:
    """Evaluate the relations of a persisted run against an annotated reference.

    Args:
        reader: The persisted run.
        reference: The annotated relations, with their predicate vocabulary.
        identity: The identity evaluation of Entity Resolution's own evaluator, for the same
            resolution run the relations are about. Its ``identity_of_resolved_entity`` says which
            annotated identity every resolved entity has; an entity that is absent from it has no
            identity, and a relation is never evaluated by repairing that.
        identity_reproducibility: The reproducibility metadata of that same identity evaluation
            (``EntityResolutionEvaluationReport.reproducibility``, or built the same way). Its
            ``run_id`` and ``resolution_artifact_digest`` are the only proof of which resolution
            run and which exact artifact the identity evaluation is about; an empty
            ``identity_of_resolved_entity`` is a valid outcome (no annotated identity) and proves
            nothing by itself, so provenance is never inferred from it.
        code_version: Code revision of the evaluator run, when known.

    Returns:
        The report, per canonical predicate, with no aggregate score. Its
        ``entity_resolution_run_id`` and ``entity_resolution_artifact_digest`` come from
        ``identity_reproducibility``, the identity evaluation actually supplied, once it is
        confirmed to match the run these relations are built on.

    Raises:
        SpatialRelationsEvaluationError: If the identity evaluation's reproducibility names a
            different resolution run or a different artifact digest than the one the relations
            are built on, or the reference declares a predicate's symmetry or inverse differently
            from the taxonomy.
    """
    relations = {
        (item.subject_entity_ref, item.predicate, item.object_entity_ref): item
        for item in reader.iter_relations()
    }
    _require_same_resolution(reader, identity_reproducibility)
    identity_of_entity = dict(identity.identity_of_resolved_entity)
    entity_of_identity, shared = _invert(identity_of_entity)
    truth, unmapped, ambiguous, unknown, conflicting = _expand_reference(reference)
    exclusions = _exclusion_reasons(reader.candidate_set())
    counters = {predicate: Counter[str]() for predicate in PREDICATE_SPECS}
    misses: Counter[str] = Counter()
    without_entity = 0
    scored: set[_Key] = set()
    for (subject_id, predicate, object_id), status in sorted(
        truth.items(), key=lambda item: (item[0][1].value, item[0][0], item[0][2])
    ):
        if subject_id in shared or object_id in shared:
            continue
        subject, obj = entity_of_identity.get(subject_id), entity_of_identity.get(object_id)
        if subject is None or obj is None:
            without_entity += 1
            continue
        key = (subject, predicate, obj)
        scored.add(key)
        _score(counters[predicate], status, relations.get(key), key, exclusions, misses)
    supported_unmatched = 0
    for key, relation in relations.items():
        subject, predicate, obj = key
        if relation.state is not RelationState.SUPPORTED:
            continue
        if subject not in identity_of_entity or obj not in identity_of_entity:
            supported_unmatched += 1
        elif key not in scored:
            counters[predicate]["unannotated_supported"] += 1
    manifest = reader.manifest
    return SpatialRelationsEvaluationReport(
        evaluator_id=SPATIAL_RELATIONS_EVALUATOR_ID,
        evaluator_version=SPATIAL_RELATIONS_EVALUATOR_VERSION,
        spatial_relations_run_id=str(manifest.run_id),
        artifact_schema_version=manifest.schema_version,
        entity_resolution_run_id=identity_reproducibility.run_id,
        entity_resolution_artifact_digest=identity_reproducibility.resolution_artifact_digest,
        geometric_map_id=str(manifest.lineage.geometric_map_id),
        taxonomy_version=manifest.taxonomy_version,
        policies=_policy_identities(manifest.policies),
        reference_normalization=reference.normalization.policy_id,
        reference_relation_count=len(reference.relations),
        predicates=tuple(
            _predicate_evaluation(predicate, counters[predicate])
            for predicate in sorted(PREDICATE_SPECS, key=lambda item: item.value)
        ),
        retrieval_misses=tuple(sorted(misses.items())),
        consistency_violations=_consistency(relations),
        unmatched=RelationUnmatchedReport(
            reference_without_entity=without_entity,
            identities_with_several_entities=tuple(sorted(shared)),
            supported_on_unmatched_entities=supported_unmatched,
            entities_spanning_identities=tuple(
                sorted(
                    str(item.resolved_entity_id)
                    for item in identity.resolved_entities_spanning_identities
                )
            ),
            unmapped_predicates=tuple(sorted(unmapped)),
            ambiguous_reference=ambiguous,
            unknown_reference=unknown,
            conflicting_reference=conflicting,
        ),
        code_version=code_version,
    )


def spatial_relations_evaluation_report(
    report: SpatialRelationsEvaluationReport,
    *,
    registry: MetricRegistry,
    reference_set: ReferenceSetIdentity,
) -> EvaluationReport:
    """Lift a spatial relations report into the common evaluation envelope.

    The headline quality metrics are reported **per canonical predicate** (as strata), never as one
    aggregate: ``relations.f1`` and ``relations.negative_violation.rate``. A predicate with no
    annotated population is not applicable, never zero.

    Args:
        report: The report of :func:`evaluate_spatial_relations`.
        registry: The metric registry the results are validated against.
        reference_set: The identity of the reference set the annotations belong to.

    Returns:
        The report envelope, carrying the stage report untouched.
    """
    quality: list[MetricResult] = []
    populated = [item for item in report.predicates if item.decided_population > 0]
    rows: list[tuple[tuple[tuple[str, str], ...], int, float | None, float | None]] = [
        (
            (("predicate", item.predicate.value),),
            item.decided_population,
            item.f1,
            item.false_relation_rate,
        )
        for item in populated
    ] or [((), 0, None, None)]
    for strata, count, f1, violation_rate in rows:
        for name, value in (
            ("relations.f1", f1),
            ("relations.negative_violation.rate", violation_rate),
        ):
            quality.append(
                MetricResult(
                    metric=name,
                    metric_version="1",
                    status=MetricStatus.NOT_APPLICABLE if value is None else MetricStatus.VALUE,
                    value=value,
                    sample_count=count,
                    strata=strata,
                )
            )
    return assemble_evaluation_report(
        registry,
        stage=EvaluationStage.SPATIAL_RELATIONS,
        reproducibility=ReproducibilityMetadata(
            evaluator=EvaluatorIdentity(
                evaluator_id=report.evaluator_id, evaluator_version=report.evaluator_version
            ),
            reference_set=reference_set,
            annotation_schemas=(
                AnnotationFamily.RELATIONS.schema,
                AnnotationFamily.IDENTITY.schema,
            ),
            input_artifacts=(
                ArtifactIdentity(
                    kind="spatial_relations_run",
                    artifact_id=report.spatial_relations_run_id,
                    digest=None,
                ),
                ArtifactIdentity(
                    kind="entity_resolution_run",
                    artifact_id=report.entity_resolution_run_id,
                    digest=report.entity_resolution_artifact_digest,
                ),
            ),
            configuration_digest=report.configuration_digest(),
            code_version=report.code_version,
            metric_registry=registry.identity(),
        ),
        quality_metrics=tuple(quality),
        performance_metrics=(),
        stage_report=report.to_dict(),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def _require_same_resolution(
    reader: SpatialRelationsRunReader,
    identity_reproducibility: EvaluationReproducibility,
) -> None:
    """Refuse an identity evaluation whose own provenance is not this run's resolution artifact.

    The run id and digest come from ``identity_reproducibility`` itself, never inferred from which
    resolved-entity references happen to appear in the identity mapping: an empty mapping (no
    annotated identity) is a valid outcome and would otherwise let any identity evaluation through
    unchecked.
    """
    lineage = reader.manifest.lineage
    expected_run_id = str(lineage.entity_resolution_run_id)
    if identity_reproducibility.run_id != expected_run_id:
        raise SpatialRelationsEvaluationError(
            f"the identity evaluation is about resolution run {identity_reproducibility.run_id!r}, "
            f"but the relations are built on resolution run {expected_run_id!r}"
        )
    expected_digest = lineage.entity_resolution_artifact_digest
    if identity_reproducibility.resolution_artifact_digest != expected_digest:
        raise SpatialRelationsEvaluationError(
            f"the identity evaluation is about resolution run {expected_run_id!r} at artifact "
            f"digest {identity_reproducibility.resolution_artifact_digest!r}, but the relations "
            f"are built on artifact digest {expected_digest!r} of that same run"
        )


def _invert(
    identity_of_entity: Mapping[ResolvedEntityReference, str],
) -> tuple[dict[str, ResolvedEntityReference], set[str]]:
    """Map identities to their entity, and set aside identities matched to several entities."""
    entity_of: dict[str, ResolvedEntityReference] = {}
    shared: set[str] = set()
    for entity, identity in identity_of_entity.items():
        if identity in entity_of:
            shared.add(identity)
        entity_of[identity] = entity
    return entity_of, shared


def _expand_reference(
    reference: RelationAnnotationSet,
) -> tuple[dict[tuple[str, RelationPredicate, str], RelationStatus], set[str], int, int, int]:
    """Map the reference to canonical predicates and expand it by the taxonomy.

    Returns:
        The decided truth by ``(subject identity, predicate, object identity)``, the unmapped
        predicate wordings, and the counts of ambiguous, unknown and conflicting entries.
    """
    key = reference.normalization.key
    canonical: dict[str, RelationPredicate] = {}
    for rule in reference.predicate_rules:
        predicate = canonical_predicate(key(rule.predicate))
        if predicate is not None:
            canonical[key(rule.predicate)] = predicate
            _require_rule_matches_taxonomy(
                rule.predicate, rule.symmetric, rule.inverse, predicate, key
            )
    entries: dict[tuple[str, RelationPredicate, str], set[RelationStatus]] = {}
    unmapped: set[str] = set()
    for relation in reference.relations:
        predicate = canonical.get(key(relation.predicate))
        if predicate is None:
            unmapped.add(relation.predicate)
            continue
        for triple in _implied(
            relation.subject_identity_id, predicate, relation.object_identity_id
        ):
            entries.setdefault(triple, set()).add(relation.status)
    truth: dict[tuple[str, RelationPredicate, str], RelationStatus] = {}
    ambiguous = unknown = conflicting = 0
    for triple, statuses in entries.items():
        if len(statuses) > 1:
            conflicting += 1
            continue
        (status,) = statuses
        if status is RelationStatus.AMBIGUOUS:
            ambiguous += 1
        elif status is RelationStatus.UNKNOWN:
            unknown += 1
        else:
            truth[triple] = status
    return truth, unmapped, ambiguous, unknown, conflicting


def _implied(
    subject: str, predicate: RelationPredicate, obj: str
) -> list[tuple[str, RelationPredicate, str]]:
    """The relation and what the taxonomy says it implies: its inverse, or its symmetric twin."""
    triples = [(subject, predicate, obj)]
    spec = predicate_spec(predicate)
    if spec.inverse is not None:
        triples.append((obj, spec.inverse, subject))
    return triples


def _require_rule_matches_taxonomy(
    wording: str,
    symmetric: bool,
    inverse: str | None,
    predicate: RelationPredicate,
    key: Callable[[str], str],
) -> None:
    """Refuse a reference vocabulary that contradicts the taxonomy for a canonical predicate.

    Omitting a property is fine, since the taxonomy supplies it. Declaring one the taxonomy
    contradicts is not: a symmetric ``above`` or an inverse that is another canonical predicate.
    An inverse that is not canonical (the reference's own ``supports``) is left unmapped.
    """
    spec = predicate_spec(predicate)
    if symmetric and not spec.symmetric:
        raise SpatialRelationsEvaluationError(
            f"the reference declares {wording!r} symmetric, but the taxonomy says {predicate.name} "
            f"is directed"
        )
    if inverse is None:
        return
    declared = canonical_predicate(key(inverse))
    if declared is not None and declared is not spec.inverse:
        raise SpatialRelationsEvaluationError(
            f"the reference declares the inverse of {wording!r} as {inverse!r}, but the taxonomy "
            f"says the inverse of {predicate.name} is "
            f"{None if spec.inverse is None else spec.inverse.name}"
        )


def _score(
    counter: Counter[str],
    status: RelationStatus,
    relation: Relation | None,
    key: _Key,
    exclusions: _Exclusions,
    misses: Counter[str],
) -> None:
    """Count one decided reference relation against what the run predicted for it."""
    state = None if relation is None else relation.state
    if status is RelationStatus.HOLDS:
        counter["annotated_holds"] += 1
        if state is RelationState.SUPPORTED:
            counter["true_positives"] += 1
        elif state is RelationState.UNRESOLVED:
            counter["missed_unresolved"] += 1
        elif state is RelationState.REJECTED:
            counter["missed_rejected"] += 1
        else:
            counter["missed_not_retrieved"] += 1
            misses[exclusions.reason(key)] += 1
    else:
        counter["annotated_does_not_hold"] += 1
        if state is RelationState.SUPPORTED:
            counter["false_positives"] += 1
        elif state is RelationState.UNRESOLVED:
            counter["unresolved_on_negatives"] += 1
        else:
            counter["true_negatives"] += 1


@dataclass(frozen=True)
class _Exclusions:
    """Why the candidate stage dropped a pair, indexed for lookup."""

    by_pair: dict[_Key, str]
    skipped: frozenset[RelationPredicate]

    def reason(self, key: _Key) -> str:
        """Why a pair that holds was never a candidate, from what the run recorded."""
        subject, predicate, obj = key
        spec = predicate_spec(predicate)
        if spec.is_derived and spec.inverse is not None:
            subject, predicate, obj = obj, spec.inverse, subject
        for pair in ((subject, predicate, obj), (obj, predicate, subject)):
            if pair in self.by_pair:
                return f"excluded:{self.by_pair[pair]}"
        if predicate in self.skipped:
            return "skipped_predicate:frame_conventions"
        return "pair_not_enumerated_or_predicate_not_selected"


def _exclusion_reasons(candidates: RelationCandidateSet) -> _Exclusions:
    return _Exclusions(
        by_pair={
            (item.subject_entity_ref, item.predicate, item.object_entity_ref): item.reason.value
            for item in candidates.exclusions
        },
        skipped=frozenset(item.predicate for item in candidates.skipped_predicates),
    )


def _predicate_evaluation(
    predicate: RelationPredicate, counts: Counter[str]
) -> RelationPredicateEvaluation:
    return RelationPredicateEvaluation(
        predicate=predicate, **{name: counts[name] for name in _COUNT_FIELDS}
    )


_COUNT_FIELDS = (
    "annotated_holds",
    "annotated_does_not_hold",
    "true_positives",
    "missed_unresolved",
    "missed_rejected",
    "missed_not_retrieved",
    "false_positives",
    "true_negatives",
    "unresolved_on_negatives",
    "unannotated_supported",
)


def _consistency(relations: Mapping[_Key, Relation]) -> tuple[RelationConsistencyViolation, ...]:
    """Check the persisted relations for inverse, symmetric and mutual-support contradictions.

    Each contradiction is reported once, from the relation with the smaller identity.
    """
    violations: list[RelationConsistencyViolation] = []
    for (subject, predicate, obj), relation in sorted(
        relations.items(), key=lambda item: str(item[1].relation_id)
    ):
        spec = predicate_spec(predicate)
        counterparts: list[tuple[str, _Key]] = []
        if spec.inverse is not None:
            counterparts.append(
                ("symmetric" if spec.symmetric else "inverse", (obj, spec.inverse, subject))
            )
        if not spec.symmetric:
            counterparts.append(("mutual_support", (obj, predicate, subject)))
        for kind, other_key in counterparts:
            other = relations.get(other_key)
            if other is None or str(other.relation_id) < str(relation.relation_id):
                continue
            both_supported = (
                relation.state is RelationState.SUPPORTED and other.state is RelationState.SUPPORTED
            )
            contradicts = (
                both_supported if kind == "mutual_support" else other.state is not relation.state
            )
            if contradicts:
                violations.append(_violation(kind, relation, other))
    return tuple(violations)


def _violation(kind: str, first: Relation, second: Relation) -> RelationConsistencyViolation:
    return RelationConsistencyViolation(
        kind=kind,
        relation_ids=tuple(sorted((str(first.relation_id), str(second.relation_id)))),
        detail=(
            f"{first.predicate.name} is {first.state.value} while {second.predicate.name} of the "
            f"other direction is {second.state.value}"
        ),
    )


def _policy_identities(policies: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for name, record in sorted(policies.items()):
        identity = record.get("policy_id", record.get("rule_id"))
        identities[name] = {"id": str(identity), "fingerprint": str(record["fingerprint"])}
    return identities


def _encode_predicate(item: RelationPredicateEvaluation) -> dict[str, Any]:
    return {
        "predicate": item.predicate.value,
        **{name: getattr(item, name) for name in _COUNT_FIELDS},
        "false_negatives": item.false_negatives,
        "precision": item.precision,
        "recall": item.recall,
        "f1": item.f1,
        "false_relation_rate": item.false_relation_rate,
        "missed_relation_rate": item.missed_relation_rate,
        "unresolved_rate": item.unresolved_rate,
        "candidate_retrieval_recall": item.candidate_retrieval_recall,
    }

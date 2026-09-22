"""The baseline relation decision policy: evidence in, conservative relations out.

Spatial Relations may receive geometric and contact evidence that agrees, conflicts or is
unavailable. This policy turns it into canonical relations without hiding that uncertainty, and it
is deliberately conservative: it prefers ``UNRESOLVED`` when the evidence is insufficient or
materially contradictory. It is a set of explicit rules, not a weighted sum, and it has no
threshold of its own: every number lives in the evidence it reads.

Only the *measured* channels (geometry and contact) decide. Observation-level evidence, which is
upstream statements held by reference, is recorded next to the verdict but is corroborating only: it
can never establish a relation, reject one or override a measured verdict, even when it contradicts
it. The contradiction is kept visible in the decision instead. A relation with no measured evidence
that decided stays ``UNRESOLVED`` however many statements assert it, and a policy that runs with no
observation evidence at all decides exactly the same.

For each candidate the evidence of the measured channels is read:

* some channel supports the predicate and none contradicts it: ``SUPPORTED``;
* some channel contradicts it and none supports it: ``REJECTED``;
* a channel supports it and another contradicts it: ``UNRESOLVED``, and the conflict stays visible;
* no channel decided (ambiguous, unavailable, or no evidence at all): ``UNRESOLVED``. Evidence that
  did not decide is *not a negative vote*.

Then the structure of the whole set is checked. A directed predicate cannot be supported both ways
(``a ABOVE b`` and ``b ABOVE a``), and a symmetric predicate evaluated in both directions must
agree; a violation demotes the relations involved to ``UNRESOLVED`` with an explicit structural
reason instead of silently keeping both. Only then are inverse and symmetric relations generated
from the final state of the relation that was evaluated, so ``BELOW`` never disagrees with
``ABOVE`` and a symmetric twin never disagrees with its source.

Nothing here reads a label, completes a relation from world knowledge or produces text: the result
is relations, and for each one the decision that produced it, naming the evidence that decided it
and the evidence that did not.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.spatial_relations._checks import require_canonical, require_present
from contextmap.spatial_relations._identity import directed_key
from contextmap.spatial_relations.candidates import RelationCandidateSet
from contextmap.spatial_relations.evidence import (
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceId,
    RelationEvidenceStatus,
)
from contextmap.spatial_relations.models import (
    Relation,
    RelationId,
    RelationProvenance,
    RelationState,
    RelationUncertainty,
    RelationUncertaintyKind,
    relation_id_for,
)
from contextmap.spatial_relations.taxonomy import (
    TAXONOMY_VERSION,
    RelationPredicate,
    predicate_spec,
)

CONSERVATIVE_DECISION_POLICY_ID = "conservative-relation-decision-v1"
"""Versioned identity of the decision rules described in this module."""

_DECISIVE = (RelationEvidenceStatus.SUPPORTS, RelationEvidenceStatus.CONFLICTS)
_UNDECIDED = (RelationEvidenceStatus.AMBIGUOUS, RelationEvidenceStatus.UNAVAILABLE)
_MEASURED_CHANNELS = frozenset({RelationEvidenceChannel.GEOMETRY, RelationEvidenceChannel.CONTACT})
"""The channels that can decide a relation; the others only corroborate."""


class DecisionRule(Enum):
    """The rule that fixed the state of a relation.

    Attributes:
        CHANNELS_SUPPORT: At least one channel supports the predicate and none contradicts it.
        CHANNELS_CONFLICT: At least one channel contradicts the predicate and none supports it.
        CHANNELS_DISAGREE: A channel supports the predicate and another contradicts it.
        NO_DECISIVE_EVIDENCE: No channel decided, or there is no evidence.
        STRUCTURAL_INCONSISTENCY: The relation contradicts another relation that the taxonomy
            makes incompatible with it.
        DERIVED_FROM_INVERSE: Generated from the inverse predicate that was evaluated.
        DERIVED_BY_SYMMETRY: Generated as the twin, in the other direction, of a symmetric
            relation that was evaluated.
    """

    CHANNELS_SUPPORT = "channels_support"
    CHANNELS_CONFLICT = "channels_conflict"
    CHANNELS_DISAGREE = "channels_disagree"
    NO_DECISIVE_EVIDENCE = "no_decisive_evidence"
    STRUCTURAL_INCONSISTENCY = "structural_inconsistency"
    DERIVED_FROM_INVERSE = "derived_from_inverse"
    DERIVED_BY_SYMMETRY = "derived_by_symmetry"


@dataclass(frozen=True, kw_only=True)
class EvidenceUse:
    """A piece of evidence that did not decide, and what it said.

    Attributes:
        evidence_id: The evidence record.
        channel: The channel that produced it.
        status: What the record said. It is ambiguous or unavailable when a measured channel did
            not decide, and any status for a corroborating channel, which never decides.
        contradicts_relation: Whether the record's status contradicts the state the relation ended
            up with: an observation that denies a supported relation, or asserts a rejected one.
            Such a contradiction is kept visible and never changes the state.
    """

    evidence_id: RelationEvidenceId
    channel: RelationEvidenceChannel
    status: RelationEvidenceStatus
    contradicts_relation: bool = False

    def __post_init__(self) -> None:
        """Validate the identity and that the record could not have decided.

        Raises:
            ValueError: If the identity is empty, a measured channel's decisive record is listed
                as ignored, or a record claims to contradict the relation without being decisive.
        """
        require_present(self, "evidence_id")
        if self.status in _DECISIVE and self.channel in _MEASURED_CHANNELS:
            raise ValueError(
                "decisive evidence of a measured channel decides the relation and cannot be ignored"
            )
        if self.contradicts_relation and self.status not in _DECISIVE:
            raise ValueError("only decisive evidence can contradict the relation")


@dataclass(frozen=True, kw_only=True)
class RelationDecision:
    """Why one relation has its state: the rule, the evidence that decided and what was ignored.

    Attributes:
        relation_id: The relation the decision is about.
        rule: The rule that fixed its state.
        deciding_evidence_refs: The records that supported or contradicted the predicate, sorted
            and unique. A relation demoted by structure keeps the records that decided it before.
        ignored: The records that did not decide, sorted by identity, with what they said.
        detail: A deterministic, human-readable explanation.
    """

    relation_id: RelationId
    rule: DecisionRule
    deciding_evidence_refs: tuple[RelationEvidenceId, ...]
    ignored: tuple[EvidenceUse, ...]
    detail: str

    def __post_init__(self) -> None:
        """Validate that the record is canonical and consistent.

        Raises:
            ValueError: If an identity or the detail is empty, a collection is not sorted and
                unique, or a record both decided and was ignored.
        """
        require_present(self, "relation_id", "detail")
        require_canonical("deciding_evidence_refs", self.deciding_evidence_refs, lambda i: (i,))
        require_canonical("ignored", self.ignored, lambda item: (item.evidence_id,))
        overlap = set(self.deciding_evidence_refs) & {item.evidence_id for item in self.ignored}
        if overlap:
            raise ValueError(f"evidence cannot be both deciding and ignored: {sorted(overlap)!r}")


@dataclass(frozen=True, kw_only=True)
class RelationDecisionResult:
    """The relations of one run and the decision behind each.

    Attributes:
        relations: Every relation, evaluated or generated, sorted by subject, predicate and
            object.
        decisions: One decision per relation, in the same order.
    """

    relations: tuple[Relation, ...]
    decisions: tuple[RelationDecision, ...]

    def __post_init__(self) -> None:
        """Validate that the relations are canonical and each has exactly one decision.

        Raises:
            ValueError: If the relations are not sorted and unique, or the decisions do not match
                them one to one.
        """
        require_canonical(
            "relations",
            self.relations,
            lambda item: directed_key(
                item.subject_entity_ref, item.predicate, item.object_entity_ref
            ),
        )
        if [item.relation_id for item in self.decisions] != [
            item.relation_id for item in self.relations
        ]:
            raise ValueError("there must be exactly one decision per relation, in relation order")


def decision_policy_fingerprint() -> str:
    """Hash the policy identity, its rules and the taxonomy it decides over, for provenance.

    Returns:
        ``sha256:`` followed by the digest of the canonical configuration. The baseline has no
        threshold of its own, so the fingerprint is a function of the rules and the taxonomy.
    """
    canonical = json.dumps(
        {
            "policy_id": CONSERVATIVE_DECISION_POLICY_ID,
            "taxonomy_version": TAXONOMY_VERSION,
            "rules": sorted(rule.value for rule in DecisionRule),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


_Key = tuple[str, str, str, str, str]


@dataclass
class _Evaluated:
    """A relation being decided: its evidence, its state so far and why."""

    subject: ResolvedEntityReference
    predicate: RelationPredicate
    obj: ResolvedEntityReference
    evidence: list[RelationEvidence]
    state: RelationState
    rule: DecisionRule
    uncertainty: tuple[RelationUncertainty, ...]
    detail: str


def decide_relations(
    candidates: RelationCandidateSet, evidence: Sequence[RelationEvidence]
) -> RelationDecisionResult:
    """Decide the relations of a candidate set from the evidence of every channel.

    Args:
        candidates: The candidates to decide, one relation each. A candidate with no evidence is
            ``UNRESOLVED``, not dropped and not rejected.
        evidence: The evidence records, from any channel, each about one candidate. A candidate
            may have one record per channel.

    Returns:
        Every evaluated relation, the inverse and symmetric relations generated from them, and
        the decision behind each. The same candidates and evidence always give the same result,
        whatever the order they are given in.

    Raises:
        ValueError: If a record is not about a candidate, a record is given twice, or a record's
            lineage (taxonomy version, map frame, geometric map or frame conventions) is not the
            one the candidates were generated under.
    """
    grouped = _group_evidence(candidates, evidence)
    evaluated: dict[_Key, _Evaluated] = {}
    for candidate in candidates.candidates:
        key = directed_key(
            candidate.subject_entity_ref, candidate.predicate, candidate.object_entity_ref
        )
        evaluated[key] = _decide_from_channels(
            candidate.subject_entity_ref,
            candidate.predicate,
            candidate.object_entity_ref,
            grouped.get(key, []),
        )
    _demote_inconsistent(evaluated)
    relations: list[Relation] = []
    decisions: dict[RelationId, RelationDecision] = {}
    for item in evaluated.values():
        source = _relation_of(item)
        relations.append(source)
        decisions[source.relation_id] = _decision_of(source, item)
        for derived, rule, detail in _derive(item, source, evaluated):
            relations.append(derived)
            decisions[derived.relation_id] = RelationDecision(
                relation_id=derived.relation_id,
                rule=rule,
                deciding_evidence_refs=decisions[source.relation_id].deciding_evidence_refs,
                ignored=decisions[source.relation_id].ignored,
                detail=detail,
            )
    ordered = sorted(
        relations,
        key=lambda item: directed_key(
            item.subject_entity_ref, item.predicate, item.object_entity_ref
        ),
    )
    return RelationDecisionResult(
        relations=tuple(ordered),
        decisions=tuple(decisions[item.relation_id] for item in ordered),
    )


def _group_evidence(
    candidates: RelationCandidateSet, evidence: Sequence[RelationEvidence]
) -> dict[_Key, list[RelationEvidence]]:
    known = {
        directed_key(item.subject_entity_ref, item.predicate, item.object_entity_ref)
        for item in candidates.candidates
    }
    provenance = candidates.provenance
    seen: set[RelationEvidenceId] = set()
    grouped: dict[_Key, list[RelationEvidence]] = {}
    for record in sorted(evidence, key=lambda item: item.evidence_id):
        if record.evidence_id in seen:
            raise ValueError(f"evidence {record.evidence_id!r} was given twice")
        seen.add(record.evidence_id)
        key = directed_key(record.subject_entity_ref, record.predicate, record.object_entity_ref)
        if key not in known:
            raise ValueError(
                f"evidence {record.evidence_id!r} is not about a candidate of this set: "
                f"{record.predicate.name} of {record.subject_entity_ref.resolved_entity_id!r} "
                f"and {record.object_entity_ref.resolved_entity_id!r}"
            )
        origin = record.provenance
        axes = origin.frame_conventions_fingerprint
        # Evidência de observação não tem coordenadas: só as de canais medidos carregam o escopo.
        if (
            origin.taxonomy_version != provenance.taxonomy_version
            or (origin.map_frame is not None and origin.map_frame != provenance.map_frame)
            or (
                origin.geometric_map_id is not None
                and origin.geometric_map_id != provenance.geometric_map_id
            )
            or (axes is not None and axes != provenance.frame_conventions_fingerprint)
        ):
            raise ValueError(
                f"evidence {record.evidence_id!r} has an incompatible lineage: it was produced "
                f"under a different taxonomy version, map frame, geometric map or set of frame "
                f"conventions than the candidates"
            )
        grouped.setdefault(key, []).append(record)
    return grouped


def _decide_from_channels(
    subject: ResolvedEntityReference,
    predicate: RelationPredicate,
    obj: ResolvedEntityReference,
    records: list[RelationEvidence],
) -> _Evaluated:
    measured = [item for item in records if item.channel in _MEASURED_CHANNELS]
    corroborating = [item for item in records if item.channel not in _MEASURED_CHANNELS]
    supporting = [item for item in measured if item.status is RelationEvidenceStatus.SUPPORTS]
    conflicting = [item for item in measured if item.status is RelationEvidenceStatus.CONFLICTS]
    undecided = [item for item in measured if item.status in _UNDECIDED]
    uncertainty: tuple[RelationUncertainty, ...]
    if supporting and conflicting:
        state = RelationState.UNRESOLVED
        rule = DecisionRule.CHANNELS_DISAGREE
        detail = (
            f"{_describe(supporting)} support the predicate while {_describe(conflicting)} "
            f"contradict it"
        )
        uncertainty = (
            RelationUncertainty(
                kind=RelationUncertaintyKind.CONFLICTING_EVIDENCE,
                detail=detail,
                evidence_refs=_ids(supporting + conflicting),
            ),
        )
    elif supporting:
        state, rule, uncertainty = RelationState.SUPPORTED, DecisionRule.CHANNELS_SUPPORT, ()
        detail = f"{_describe(supporting)} support the predicate and none contradicts it"
    elif conflicting:
        state, rule, uncertainty = RelationState.REJECTED, DecisionRule.CHANNELS_CONFLICT, ()
        detail = f"{_describe(conflicting)} contradict the predicate and none supports it"
    else:
        state = RelationState.UNRESOLVED
        rule = DecisionRule.NO_DECISIVE_EVIDENCE
        detail = (
            f"no measured channel decided: {_describe(undecided)}"
            if undecided
            else "no measured evidence was produced for this candidate"
        )
        if corroborating:
            detail += (
                f"; observation-level evidence ({_describe(corroborating)}) cannot decide "
                f"a relation on its own"
            )
        uncertainty = (
            RelationUncertainty(
                kind=RelationUncertaintyKind.INSUFFICIENT_EVIDENCE,
                detail=detail,
                evidence_refs=_ids(undecided + corroborating),
            ),
        )
    return _Evaluated(
        subject=subject,
        predicate=predicate,
        obj=obj,
        evidence=records,
        state=state,
        rule=rule,
        uncertainty=uncertainty,
        detail=detail,
    )


def _describe(records: list[RelationEvidence]) -> str:
    return ", ".join(sorted(f"{item.channel.value} ({item.status.value})" for item in records))


def _ids(records: list[RelationEvidence]) -> tuple[RelationEvidenceId, ...]:
    return tuple(sorted(item.evidence_id for item in records))


def _demote_inconsistent(evaluated: dict[_Key, _Evaluated]) -> None:
    """Demote the relations the taxonomy makes incompatible with each other.

    A directed predicate cannot be supported in both directions, and a symmetric predicate
    evaluated in both directions must reach the same state. Only relations that were decided are
    demoted: an unresolved relation already says it does not know.
    """
    demoted: dict[_Key, str] = {}
    for key, item in evaluated.items():
        partner_key = directed_key(item.obj, item.predicate, item.subject)
        partner = evaluated.get(partner_key)
        if partner is None or item.state is RelationState.UNRESOLVED:
            continue
        symmetric = predicate_spec(item.predicate).symmetric
        if symmetric and partner.state is not item.state:
            demoted[key] = "a symmetric predicate evaluated both ways reached different states"
        elif (
            not symmetric
            and item.state is RelationState.SUPPORTED
            and partner.state is RelationState.SUPPORTED
        ):
            demoted[key] = f"both directions of the directed predicate {item.predicate.name} hold"
    for key, reason in demoted.items():
        item = evaluated[key]
        partner_id = relation_id_for(
            subject_entity_ref=item.obj,
            predicate=item.predicate,
            object_entity_ref=item.subject,
        )
        detail = f"contradicts {partner_id}: {reason}"
        item.state = RelationState.UNRESOLVED
        item.rule = DecisionRule.STRUCTURAL_INCONSISTENCY
        item.detail = detail
        item.uncertainty = (
            RelationUncertainty(
                kind=RelationUncertaintyKind.INCONSISTENT_STRUCTURE,
                detail=detail,
                evidence_refs=_ids(_deciding(item.evidence)),
            ),
        )


def _deciding(records: list[RelationEvidence]) -> list[RelationEvidence]:
    """The records that decide a relation: decisive evidence of a measured channel."""
    return [
        item for item in records if item.status in _DECISIVE and item.channel in _MEASURED_CHANNELS
    ]


def _contradicts(state: RelationState, status: RelationEvidenceStatus) -> bool:
    return (state is RelationState.SUPPORTED and status is RelationEvidenceStatus.CONFLICTS) or (
        state is RelationState.REJECTED and status is RelationEvidenceStatus.SUPPORTS
    )


def _relation_of(item: _Evaluated) -> Relation:
    subject, obj = item.subject, item.obj
    return Relation(
        relation_id=relation_id_for(
            subject_entity_ref=subject, predicate=item.predicate, object_entity_ref=obj
        ),
        subject_entity_ref=subject,
        predicate=item.predicate,
        object_entity_ref=obj,
        state=item.state,
        relation_evidence_refs=_ids(item.evidence),
        uncertainty=item.uncertainty,
        provenance=_provenance(),
    )


def _provenance() -> RelationProvenance:
    return RelationProvenance(
        taxonomy_version=TAXONOMY_VERSION,
        decision_policy_id=CONSERVATIVE_DECISION_POLICY_ID,
        configuration_fingerprint=decision_policy_fingerprint(),
    )


def _decision_of(relation: Relation, item: _Evaluated) -> RelationDecision:
    deciding = _deciding(item.evidence)
    decided = {record.evidence_id for record in deciding}
    return RelationDecision(
        relation_id=relation.relation_id,
        rule=item.rule,
        deciding_evidence_refs=_ids(deciding),
        ignored=tuple(
            sorted(
                (
                    EvidenceUse(
                        evidence_id=record.evidence_id,
                        channel=record.channel,
                        status=record.status,
                        contradicts_relation=_contradicts(item.state, record.status),
                    )
                    for record in item.evidence
                    if record.evidence_id not in decided
                ),
                key=lambda use: use.evidence_id,
            )
        ),
        detail=item.detail,
    )


def _derive(
    item: _Evaluated, source: Relation, evaluated: dict[_Key, _Evaluated]
) -> list[tuple[Relation, DecisionRule, str]]:
    """Generate the inverse or symmetric relation of an evaluated one, from its final state."""
    spec = predicate_spec(item.predicate)
    subject, obj = item.subject, item.obj
    if spec.symmetric:
        twin_key = directed_key(obj, item.predicate, subject)
        if twin_key in evaluated:
            return []
        predicate, rule = item.predicate, DecisionRule.DERIVED_BY_SYMMETRY
    elif spec.inverse is not None:
        predicate, rule = spec.inverse, DecisionRule.DERIVED_FROM_INVERSE
    else:
        return []
    derived = Relation(
        relation_id=relation_id_for(
            subject_entity_ref=obj, predicate=predicate, object_entity_ref=subject
        ),
        subject_entity_ref=obj,
        predicate=predicate,
        object_entity_ref=subject,
        state=source.state,
        relation_evidence_refs=source.relation_evidence_refs,
        uncertainty=source.uncertainty,
        derived_from=source.relation_id,
        provenance=source.provenance,
    )
    detail = (
        f"generated from {source.relation_id}: {item.predicate.name} of the other direction "
        f"is {source.state.value}"
    )
    return [(derived, rule, detail)]

"""Canonical contracts for Spatial Relations.

A :class:`Relation` is a versioned predicate between two *resolved entities*, with an explicit
state and the evidence it was decided on. It is knowledge derived from entities and consolidated
evidence, never a replacement for that evidence: the measurements live in
:class:`~contextmap.spatial_relations.RelationEvidence` and a relation only points at them.

A relation is between resolved entities, held by
:class:`~contextmap.entity_resolution.ResolvedEntityReference`. It is never between raw pixels,
regions, points or unresolved candidates, and it never copies or modifies an entity: the resolved
entities stay exactly as Entity Resolution produced them.

Three states keep what is known apart from what is not:

* ``SUPPORTED`` -- the evidence supports the predicate and nothing contradicts it;
* ``REJECTED`` -- the evidence contradicts the predicate and nothing supports it;
* ``UNRESOLVED`` -- the evidence is missing, ambiguous or contradictory, and the record says why.

Direction, symmetry and inverse are properties of the predicate and live in
:mod:`~contextmap.spatial_relations.taxonomy`. An inverse or symmetric relation is generated from
the relation that was evaluated and names it in ``derived_from``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.spatial_relations._checks import require_canonical, require_present
from contextmap.spatial_relations._identity import candidate_digest, require_relatable_pair
from contextmap.spatial_relations.evidence import RelationEvidenceId
from contextmap.spatial_relations.taxonomy import RelationPredicate

RelationId = NewType("RelationId", str)
"""Identity of one relation, local to its spatial-relations artifact."""


class RelationState(Enum):
    """The decided state of a relation.

    Attributes:
        SUPPORTED: The evidence supports the predicate and nothing contradicts it.
        REJECTED: The evidence contradicts the predicate and nothing supports it.
        UNRESOLVED: The evidence is missing, ambiguous or contradictory.
    """

    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class RelationUncertaintyKind(Enum):
    """Why a relation could not be decided.

    Attributes:
        INSUFFICIENT_EVIDENCE: No evidence was decisive: all of it is ambiguous or unavailable,
            or there is none.
        CONFLICTING_EVIDENCE: Evidence supports the predicate and other evidence contradicts it.
        INCONSISTENT_STRUCTURE: The relation contradicts another relation that the taxonomy makes
            incompatible with it, such as both directions of a directed predicate.
    """

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INCONSISTENT_STRUCTURE = "inconsistent_structure"


@dataclass(frozen=True, kw_only=True)
class RelationUncertainty:
    """A stated reason a relation is unresolved.

    Attributes:
        kind: The kind of uncertainty.
        detail: A deterministic, human-readable explanation.
        evidence_refs: The evidence that produced the uncertainty, sorted and unique.
    """

    kind: RelationUncertaintyKind
    detail: str
    evidence_refs: tuple[RelationEvidenceId, ...] = ()

    def __post_init__(self) -> None:
        """Validate the explanation and the references.

        Raises:
            ValueError: If the detail is empty or the references are not sorted and unique.
        """
        require_present(self, "detail")
        require_canonical("evidence_refs", self.evidence_refs, lambda item: (item,))


@dataclass(frozen=True, kw_only=True)
class RelationProvenance:
    """How a relation was decided.

    Attributes:
        taxonomy_version: The vocabulary version the predicate is defined in.
        decision_policy_id: The versioned policy that turned evidence into the state.
        configuration_fingerprint: Hash of the decision policy's configuration.
        code_version: Code revision that produced the relation, when known.
    """

    taxonomy_version: str
    decision_policy_id: str
    configuration_fingerprint: str
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Require the identities.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(self, "taxonomy_version", "decision_policy_id", "configuration_fingerprint")


def relation_id_for(
    *,
    subject_entity_ref: ResolvedEntityReference,
    predicate: RelationPredicate,
    object_entity_ref: ResolvedEntityReference,
) -> RelationId:
    """Compute the deterministic identity of a directed relation.

    Args:
        subject_entity_ref: The entity the statement is about.
        predicate: The predicate.
        object_entity_ref: The entity it is related to.

    Returns:
        A pure function of the inputs: the same triple always names the same relation, and the
        opposite direction names a different one.
    """
    digest = candidate_digest(subject_entity_ref, predicate, object_entity_ref)
    return RelationId(f"relation--{predicate.value}--{digest}")


@dataclass(frozen=True, kw_only=True)
class Relation:
    """A relation between two resolved entities, with its state and its evidence trace.

    Attributes:
        relation_id: Identity of the relation, local to its artifact.
        subject_entity_ref: The resolved entity the statement is about.
        predicate: The canonical predicate.
        object_entity_ref: The resolved entity it is related to.
        state: What the evidence decided.
        relation_evidence_refs: The evidence the decision was made on, sorted and unique. A
            supported or rejected relation always has some; an unresolved one may have none.
        uncertainty: Why the relation is unresolved; empty for a supported or rejected one.
        derived_from: The relation this one was generated from as the inverse or the symmetric
            twin, or ``None`` when it was evaluated directly.
        provenance: The taxonomy and decision policy behind the state.
    """

    relation_id: RelationId
    subject_entity_ref: ResolvedEntityReference
    predicate: RelationPredicate
    object_entity_ref: ResolvedEntityReference
    state: RelationState
    relation_evidence_refs: tuple[RelationEvidenceId, ...]
    provenance: RelationProvenance
    uncertainty: tuple[RelationUncertainty, ...] = ()
    derived_from: RelationId | None = None

    def __post_init__(self) -> None:
        """Validate that the state is justified by the evidence trace and the uncertainty.

        Raises:
            ValueError: If the identity is empty, the pair is not relatable, the references are
                not canonical, a decided relation has no evidence or carries uncertainty, an
                unresolved relation does not say why, an uncertainty cites evidence the relation
                does not use, or the relation is derived from itself.
        """
        require_present(self, "relation_id")
        require_relatable_pair(self.subject_entity_ref, self.object_entity_ref)
        require_canonical(
            "relation_evidence_refs", self.relation_evidence_refs, lambda item: (item,)
        )
        require_canonical(
            "uncertainty", self.uncertainty, lambda item: (item.kind.value, item.detail)
        )
        if self.state is RelationState.UNRESOLVED:
            if not self.uncertainty:
                raise ValueError("an unresolved relation needs uncertainty that says why")
        else:
            if not self.relation_evidence_refs:
                raise ValueError(f"a {self.state.value} relation needs the evidence it rests on")
            if self.uncertainty:
                raise ValueError(f"a {self.state.value} relation carries no uncertainty")
        used = set(self.relation_evidence_refs)
        for item in self.uncertainty:
            for reference in item.evidence_refs:
                if reference not in used:
                    raise ValueError(
                        f"uncertainty cites {reference!r}, which the relation does not use"
                    )
        if self.derived_from == self.relation_id:
            raise ValueError("a relation cannot be derived from itself")

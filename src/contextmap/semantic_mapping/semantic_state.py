"""The semantic state of an entity.

An entity keeps the semantic evidence accumulated upstream instead of collapsing it into one
label and one confidence. Each hypothesis keeps every claim behind it, with the typed signals
each producer emitted, so a missing score stays distinguishable from a low one and an
abstention from a negative. Conflicts, ambiguity and lack of evidence are kept as explicit
records, and a primary hypothesis is exposed only when a documented rule justifies it, never as
a replacement for the alternatives.

Several distinctions are preserved on purpose:

* a **hypothesis** is what the entity may be; an **attribute** is a property of it, each with its
  own evidence and the rule that derived it;
* an **alternative** competes with a hypothesis; a **contradiction** is distinct physical
  observations backing incompatible ones;
* an **abstention** (``unknown``) is neither support nor evidence against, and **unscored**
  (``None``) is not a low score;
* an **observed** property comes from explicit evidence; **external knowledge** never does, and
  is only accepted when a documented reasoning stage derived it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from contextmap.semantic_fusion import (
    EvidenceReference,
    EvidenceStance,
    FusedEvidenceId,
    FusedHypothesisId,
    HypothesisEvidence,
    SupportSignal,
    UncertaintyKind,
    UncertaintyRecord,
)
from contextmap.semantic_mapping._checks import require_canonical, require_present

_COMPETITIVE_KINDS = frozenset({UncertaintyKind.AMBIGUITY, UncertaintyKind.NEAR_TIE})


class AmbiguityState(Enum):
    """How settled the semantic state of an entity is.

    The value is derived from the hypotheses and the uncertainty records by
    :func:`derive_ambiguity_state` and the contract refuses any other value, so it can be read
    without inspecting the records.

    Attributes:
        UNAMBIGUOUS: A hypothesis exists and nothing competes with it.
        AMBIGUOUS: Hypotheses compete, through alternatives or a near tie, without a
            contradiction between observations.
        CONFLICTING: Distinct physical observations back incompatible hypotheses.
        INSUFFICIENT_EVIDENCE: No hypothesis is supported.
    """

    UNAMBIGUOUS = "unambiguous"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AttributeOrigin(Enum):
    """Where the value of an attribute comes from.

    Attributes:
        OBSERVED: Stated by explicit evidence, such as a claim.
        DERIVED: Computed by a documented rule from other evidence-backed state.
        EXTERNAL_KNOWLEDGE: Inferred by a documented reasoning stage; never observed. It is
            distinct on purpose so it cannot be mistaken for evidence.
    """

    OBSERVED = "observed"
    DERIVED = "derived"
    EXTERNAL_KNOWLEDGE = "external_knowledge"


@dataclass(frozen=True, kw_only=True, order=True)
class EntityHypothesisRef:
    """Identity of one hypothesis inside an entity.

    Attributes:
        fused_evidence_id: The fused evidence the hypothesis comes from.
        hypothesis_id: The hypothesis, local to that fused evidence.
    """

    fused_evidence_id: FusedEvidenceId
    hypothesis_id: FusedHypothesisId

    def __post_init__(self) -> None:
        """Require both identities.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(self, "fused_evidence_id", "hypothesis_id")


@dataclass(frozen=True, kw_only=True)
class EntityHypothesis:
    """One semantic candidate of an entity, with every piece of evidence behind it.

    The label is the open-vocabulary text exactly as it was proposed. There is no probability
    and no combined confidence: the typed signals of each evidence item stay separate.

    Attributes:
        fused_evidence_id: The fused evidence the hypothesis comes from.
        hypothesis_id: The hypothesis, local to that fused evidence.
        label: The hypothesis text, verbatim.
        evidence: The claims that support, conflict with, abstain from or are ambiguous about
            the hypothesis, sorted by contribution and claim and unique.
    """

    fused_evidence_id: FusedEvidenceId
    hypothesis_id: FusedHypothesisId
    label: str
    evidence: tuple[HypothesisEvidence, ...]

    def __post_init__(self) -> None:
        """Validate the identities, the ordering and that something supports the hypothesis.

        Raises:
            ValueError: If an identity or the label is empty, the evidence is not sorted and
                unique, or no evidence item supports the hypothesis.
        """
        require_present(self, "fused_evidence_id", "hypothesis_id", "label")
        require_canonical(
            "evidence", self.evidence, lambda item: (item.contribution_id, item.claim_id)
        )
        if not any(item.stance is EvidenceStance.SUPPORTING for item in self.evidence):
            raise ValueError(
                f"hypothesis {self.label!r} needs at least one supporting evidence item"
            )

    @property
    def ref(self) -> EntityHypothesisRef:
        """The identity of this hypothesis inside its entity."""
        return EntityHypothesisRef(
            fused_evidence_id=self.fused_evidence_id, hypothesis_id=self.hypothesis_id
        )


@dataclass(frozen=True, kw_only=True)
class EntityAttribute:
    """A structured property of an entity, with the evidence and the rule behind its value.

    A property is never assumed: it comes from explicit evidence, from a documented derivation
    or from a documented reasoning stage that says so.

    Attributes:
        name: The property, for example ``class``, ``material`` or ``condition``.
        value: Its value, verbatim.
        origin: Whether it was observed, derived or comes from external knowledge.
        derivation_id: The versioned rule or stage that produced it.
        evidence: The contributions and claims it rests on, sorted and unique; required unless
            the origin is external knowledge.
        support: Typed signals of its own, when the producer emitted any, sorted by kind and
            producer and unique; empty means it carries no signal, not that the support is zero.
    """

    name: str
    value: str
    origin: AttributeOrigin
    derivation_id: str
    evidence: tuple[EvidenceReference, ...] = ()
    support: tuple[SupportSignal, ...] = ()

    def __post_init__(self) -> None:
        """Validate the text, the ordering and that the value is traceable.

        Raises:
            ValueError: If the name, value or derivation is empty, the evidence or support is
                not sorted and unique, or an observed or derived attribute has no evidence.
        """
        require_present(self, "name", "value", "derivation_id")
        require_canonical(
            "evidence", self.evidence, lambda ref: (ref.contribution_id, ref.claim_id or "")
        )
        require_canonical(
            "support",
            self.support,
            lambda signal: (
                signal.kind.value,
                signal.producer.backend_id,
                signal.producer.capability,
                signal.producer.provider,
                signal.producer.model,
                signal.producer.version,
                signal.producer.configuration_fingerprint or "",
            ),
        )
        if self.origin is not AttributeOrigin.EXTERNAL_KNOWLEDGE and not self.evidence:
            raise ValueError(
                f"attribute {self.name!r} is {self.origin.value} but cites no evidence: only "
                f"external knowledge may be accepted without it"
            )


@dataclass(frozen=True, kw_only=True)
class EntityUncertainty:
    """A conflict, ambiguity or lack of evidence, with the exact evidence behind it.

    Attributes:
        fused_evidence_id: The fused evidence that reported it.
        record: The record as Semantic Fusion produced it: kind, hypotheses involved, the exact
            contributions and claims, and the rule that declared it.
    """

    fused_evidence_id: FusedEvidenceId
    record: UncertaintyRecord

    def __post_init__(self) -> None:
        """Require the fused evidence identity.

        Raises:
            ValueError: If the identity is empty.
        """
        require_present(self, "fused_evidence_id")


@dataclass(frozen=True, kw_only=True)
class SemanticStateProvenance:
    """How the semantic state was derived from the fused evidence.

    Attributes:
        mapping_rule_id: Versioned rule that mapped fused evidence into the state.
        primary_policy_id: Versioned rule that decided whether to expose a primary hypothesis.
    """

    mapping_rule_id: str
    primary_policy_id: str

    def __post_init__(self) -> None:
        """Require both rules.

        Raises:
            ValueError: If a rule identity is empty.
        """
        require_present(self, "mapping_rule_id", "primary_policy_id")


def derive_ambiguity_state(
    hypotheses: tuple[EntityHypothesis, ...], uncertainty: tuple[EntityUncertainty, ...]
) -> AmbiguityState:
    """Derive how settled a semantic state is from its hypotheses and uncertainty.

    A contradiction dominates, then any competition, then a lack of evidence. Several
    hypotheses of one fused evidence with nothing that settles them are competing even when
    no record says so.

    Args:
        hypotheses: The hypotheses of the state.
        uncertainty: The uncertainty records of the state.

    Returns:
        The ambiguity state.
    """
    kinds = {item.record.kind for item in uncertainty}
    if UncertaintyKind.CONTRADICTION in kinds:
        return AmbiguityState.CONFLICTING
    per_evidence = Counter(item.fused_evidence_id for item in hypotheses)
    if kinds & _COMPETITIVE_KINDS or any(count > 1 for count in per_evidence.values()):
        return AmbiguityState.AMBIGUOUS
    if not hypotheses or UncertaintyKind.INSUFFICIENT_EVIDENCE in kinds:
        return AmbiguityState.INSUFFICIENT_EVIDENCE
    return AmbiguityState.UNAMBIGUOUS


@dataclass(frozen=True, kw_only=True)
class EntitySemanticState:
    """What an entity may be, with nothing collapsed.

    Attributes:
        hypotheses: Every candidate interpretation, sorted by fused evidence and hypothesis
            and unique; empty when no view produced a claim.
        ambiguity_state: How settled the state is; always the one
            :func:`derive_ambiguity_state` gives.
        provenance: How the state was derived from the fused evidence.
        primary_hypothesis: A hypothesis exposed for convenience, only when the state is
            unambiguous; ``None`` otherwise. The alternatives always stay available.
        attributes: Structured properties, sorted by name, value, origin and derivation and
            unique.
        uncertainty: Conflicts, ambiguity and lack of evidence, sorted and unique.
    """

    hypotheses: tuple[EntityHypothesis, ...]
    ambiguity_state: AmbiguityState
    provenance: SemanticStateProvenance
    primary_hypothesis: EntityHypothesisRef | None = None
    attributes: tuple[EntityAttribute, ...] = ()
    uncertainty: tuple[EntityUncertainty, ...] = ()

    def __post_init__(self) -> None:
        """Validate ordering, references and that the state agrees with its own records.

        Raises:
            ValueError: If a collection is not sorted and unique, one fused evidence lists the
                same label twice, an uncertainty record names an unknown hypothesis, the
                primary hypothesis is unknown or is exposed while the state is not
                unambiguous, or the ambiguity state is not the one the records imply.
        """
        require_canonical(
            "hypotheses",
            self.hypotheses,
            lambda item: (item.fused_evidence_id, item.hypothesis_id),
        )
        require_canonical(
            "attributes",
            self.attributes,
            lambda item: (item.name, item.value, item.origin.value, item.derivation_id),
        )
        require_canonical("uncertainty", self.uncertainty, _uncertainty_key)
        self._require_unique_labels()
        known = {item.ref for item in self.hypotheses}
        self._require_uncertainty_resolves(known)
        expected = derive_ambiguity_state(self.hypotheses, self.uncertainty)
        if self.ambiguity_state is not expected:
            raise ValueError(
                f"ambiguity_state {self.ambiguity_state.value!r} does not match the hypotheses "
                f"and uncertainty, which imply {expected.value!r}"
            )
        if self.primary_hypothesis is not None:
            if self.primary_hypothesis not in known:
                raise ValueError(f"primary hypothesis {self.primary_hypothesis!r} is not known")
            if self.ambiguity_state is not AmbiguityState.UNAMBIGUOUS:
                raise ValueError(
                    f"a primary hypothesis cannot be exposed while the state is "
                    f"{self.ambiguity_state.value}: the alternatives would be hidden"
                )

    @property
    def primary(self) -> EntityHypothesis | None:
        """The primary hypothesis, when one is exposed."""
        if self.primary_hypothesis is None:
            return None
        return next(item for item in self.hypotheses if item.ref == self.primary_hypothesis)

    @property
    def alternative_hypotheses(self) -> tuple[EntityHypothesis, ...]:
        """Every hypothesis other than the primary one."""
        return tuple(item for item in self.hypotheses if item.ref != self.primary_hypothesis)

    @property
    def conflicts(self) -> tuple[EntityUncertainty, ...]:
        """The uncertainty records that are contradictions between physical observations."""
        return tuple(
            item for item in self.uncertainty if item.record.kind is UncertaintyKind.CONTRADICTION
        )

    def _require_unique_labels(self) -> None:
        labels: set[tuple[str, str]] = set()
        for hypothesis in self.hypotheses:
            key = (hypothesis.fused_evidence_id, hypothesis.label)
            if key in labels:
                raise ValueError(
                    f"label {hypothesis.label!r} appears twice in fused evidence "
                    f"{hypothesis.fused_evidence_id!r}"
                )
            labels.add(key)

    def _require_uncertainty_resolves(self, known: set[EntityHypothesisRef]) -> None:
        for item in self.uncertainty:
            for hypothesis_id in item.record.hypothesis_ids:
                ref = EntityHypothesisRef(
                    fused_evidence_id=item.fused_evidence_id, hypothesis_id=hypothesis_id
                )
                if ref not in known:
                    raise ValueError(
                        f"uncertainty of fused evidence {item.fused_evidence_id!r} names "
                        f"unknown hypothesis {hypothesis_id!r}"
                    )


def _uncertainty_key(item: EntityUncertainty) -> tuple[str, ...]:
    record = item.record
    return (
        item.fused_evidence_id,
        record.kind.value,
        record.rule_id,
        "|".join(record.hypothesis_ids),
        "|".join(f"{ref.contribution_id}/{ref.claim_id or ''}" for ref in record.evidence),
    )

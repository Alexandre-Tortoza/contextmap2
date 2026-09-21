"""Observation-level relation evidence: upstream statements as a channel of their own.

Visual perception or language reasoning may say something relational about a scene. That is a
signal worth keeping, but it is not a measurement: it can be wrong in ways geometry cannot, it names
things in upstream terms, and it comes from a model. So it enters here as its own evidence channel,
separate from geometry, and never becomes a relation by itself.

Turning statements into evidence follows three rules:

* **Explicit linkage only.** A statement's two ends are already tied to resolved entities by
  :class:`~contextmap.spatial_relations.EndpointLink`, each with the evidence that tie went
  through. An end that is not in the selected resolved-entity set, or a statement whose ends come
  from different resolution artifacts, is refused with a named error; nothing is approximated.
* **No hidden vocabulary.** The predicate wording is mapped only when it is exactly a canonical
  predicate name. A statement with any other wording is *unmapped*: it is reported, never turned
  into a relation and never mapped through a synonym. Missing or unmapped statements are neutral,
  not negative.
* **Canonical direction.** A statement in a derived wording (``"below"``) is recorded for the
  evaluated direction of its inverse with the ends swapped, and a symmetric one in the canonical
  order of the two references, so it lands on the candidate the geometry evaluated.

Statements about the same candidate are merged into one record whose status follows them: all
assert is ``SUPPORTS``, all deny is ``CONFLICTS`` and a mix is ``AMBIGUOUS`` with a caveat. The
exact statements, with their upstream run, statement, observation and producer, stay in the
record.

The decision policy reads this channel as *corroborating only*: it can be recorded next to a
measured verdict, even a contradicting one, but it cannot establish, reject or override a relation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from itertools import pairwise

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.spatial_relations._identity import directed_key, reference_key
from contextmap.spatial_relations.evidence import (
    EvidenceCaveat,
    EvidenceCaveatKind,
    Quantity,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceProvenance,
    RelationEvidenceStatus,
    evidence_id_for,
)
from contextmap.spatial_relations.statements import ObservationRelationStatement, StatementPolarity
from contextmap.spatial_relations.taxonomy import (
    TAXONOMY_VERSION,
    RelationPredicate,
    canonical_predicate,
    predicate_spec,
)

OBSERVATION_RULE_ID = "upstream-statements-v1"
"""Versioned identity of the rule that turns statements into evidence."""

CONFLICTING_STATEMENTS_CAVEAT = (
    "the upstream statements about this candidate assert and deny the predicate"
)
"""The caveat recorded when statements about one candidate disagree."""


class UnlinkedEndpointError(ValueError):
    """Raised when an end of a statement is not an entity of the selected resolved-entity set."""


class IncompatibleLineageError(ValueError):
    """Raised when a statement's ends, or the selected set, come from different resolutions."""


@dataclass(frozen=True, kw_only=True)
class ObservationEvidenceResult:
    """The evidence built from a set of statements, and the statements left out.

    Attributes:
        evidence: One record per candidate the statements are about, sorted by subject, predicate
            and object.
        unmapped_statements: The statements whose wording is not a canonical predicate, sorted.
            They are neutral: no relation and no evidence follows from them.
    """

    evidence: tuple[RelationEvidence, ...]
    unmapped_statements: tuple[ObservationRelationStatement, ...]


def observation_evidence_fingerprint() -> str:
    """Hash the rule identity and the taxonomy its wording maps into, for provenance."""
    canonical = json.dumps(
        {"rule_id": OBSERVATION_RULE_ID, "taxonomy_version": TAXONOMY_VERSION},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def observation_evidence_from_statements(
    statements: Iterable[ObservationRelationStatement],
    *,
    entities: Collection[ResolvedEntityReference],
) -> ObservationEvidenceResult:
    """Turn linked upstream statements into observation-channel evidence.

    Args:
        statements: The upstream statements, each with its ends already linked to resolved
            entities.
        entities: The selected resolved-entity set: every end must be one of these.

    Returns:
        The evidence records and the statements whose wording is not canonical. The same
        statements always give the same result, whatever their order.

    Raises:
        IncompatibleLineageError: If the selected set spans several resolution artifacts, or the
            two ends of a statement come from different ones.
        UnlinkedEndpointError: If an end of a statement is not in the selected set.
        ValueError: If the same statement is given twice.
    """
    selected = frozenset(entities)
    if len({reference.resolution_run_id for reference in selected}) > 1:
        raise IncompatibleLineageError("the selected entities span several resolution artifacts")
    ordered = sorted(statements, key=lambda item: item.sort_key)
    for previous, current in pairwise(ordered):
        if previous.sort_key == current.sort_key:
            raise ValueError(
                f"statement {current.source.statement_id!r} of run "
                f"{current.source.source_run_id!r} was given twice"
            )
    grouped: dict[tuple[str, ...], list[ObservationRelationStatement]] = {}
    keys: dict[
        tuple[str, ...], tuple[ResolvedEntityReference, RelationPredicate, ResolvedEntityReference]
    ] = {}
    unmapped: list[ObservationRelationStatement] = []
    for statement in ordered:
        _require_linked(statement, selected)
        predicate = canonical_predicate(statement.predicate_text)
        if predicate is None:
            unmapped.append(statement)
            continue
        subject, evaluated, obj = _evaluated_direction(
            statement.subject.entity_ref, predicate, statement.object.entity_ref
        )
        key = directed_key(subject, evaluated, obj)
        grouped.setdefault(key, []).append(statement)
        keys[key] = (subject, evaluated, obj)
    records = [
        _record(*keys[key], members) for key, members in sorted(grouped.items(), key=lambda i: i[0])
    ]
    return ObservationEvidenceResult(evidence=tuple(records), unmapped_statements=tuple(unmapped))


def _require_linked(
    statement: ObservationRelationStatement, selected: frozenset[ResolvedEntityReference]
) -> None:
    subject, obj = statement.subject.entity_ref, statement.object.entity_ref
    if subject.resolution_run_id != obj.resolution_run_id:
        raise IncompatibleLineageError(
            f"the ends of statement {statement.source.statement_id!r} come from different "
            f"resolution artifacts, {subject.resolution_run_id!r} and {obj.resolution_run_id!r}"
        )
    for link in (statement.subject, statement.object):
        if link.entity_ref not in selected:
            raise UnlinkedEndpointError(
                f"{link.upstream_ref!r} of statement {statement.source.statement_id!r} is linked "
                f"to {link.entity_ref.resolved_entity_id!r}, which is not in the selected "
                f"resolved-entity set"
            )


def _evaluated_direction(
    subject: ResolvedEntityReference, predicate: RelationPredicate, obj: ResolvedEntityReference
) -> tuple[ResolvedEntityReference, RelationPredicate, ResolvedEntityReference]:
    """Express a statement in the direction the evaluators measure.

    A derived predicate is replaced by its inverse with the ends swapped, and a symmetric one is
    put in the canonical order of the two references, so the record lands on the candidate that
    candidate generation produced.
    """
    spec = predicate_spec(predicate)
    if spec.is_derived and spec.inverse is not None:
        return obj, spec.inverse, subject
    if spec.symmetric and reference_key(obj) < reference_key(subject):
        return obj, predicate, subject
    return subject, predicate, obj


def _record(
    subject: ResolvedEntityReference,
    predicate: RelationPredicate,
    obj: ResolvedEntityReference,
    statements: list[ObservationRelationStatement],
) -> RelationEvidence:
    asserting = sum(1 for item in statements if item.polarity is StatementPolarity.ASSERTS)
    denying = len(statements) - asserting
    caveats: tuple[EvidenceCaveat, ...] = ()
    if asserting and denying:
        status = RelationEvidenceStatus.AMBIGUOUS
        caveats = (
            EvidenceCaveat(
                kind=EvidenceCaveatKind.CONFLICTING_STATEMENTS,
                detail=CONFLICTING_STATEMENTS_CAVEAT,
            ),
        )
    elif asserting:
        status = RelationEvidenceStatus.SUPPORTS
    else:
        status = RelationEvidenceStatus.CONFLICTS
    return RelationEvidence(
        evidence_id=evidence_id_for(
            channel=RelationEvidenceChannel.OBSERVATION,
            subject_entity_ref=subject,
            predicate=predicate,
            object_entity_ref=obj,
        ),
        channel=RelationEvidenceChannel.OBSERVATION,
        subject_entity_ref=subject,
        predicate=predicate,
        object_entity_ref=obj,
        status=status,
        measurements=(
            Quantity(name="asserting_statements", value=float(asserting), unit="count"),
            Quantity(name="denying_statements", value=float(denying), unit="count"),
        ),
        thresholds=(),
        geometry=(),
        caveats=caveats,
        provenance=RelationEvidenceProvenance(
            rule_id=OBSERVATION_RULE_ID,
            configuration_fingerprint=observation_evidence_fingerprint(),
            taxonomy_version=TAXONOMY_VERSION,
        ),
        statements=tuple(statements),
    )

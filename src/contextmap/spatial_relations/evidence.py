"""Evidence about one relation candidate: what was measured, by which rule, with what verdict.

A :class:`RelationEvidence` is *not* a relation. It records that a rule looked at a directed
candidate ``subject PREDICATE object`` in one evidence channel and reached a verdict about it,
together with the exact numbers behind the verdict, the thresholds they were compared with, the
geometry they were measured over and the caveats that limit them. Whether a relation is materialized
is a later, separate decision that reads this record and never rewrites it.

Channels stay apart on purpose. Bounds-based geometry (proximity, direction, topology) and
point-level contact are different measurements with different failure modes, so each is its own
record with its own status; nothing here folds them into one score.

The four statuses keep unknown apart from negative:

* ``SUPPORTS`` -- the measurements meet the conditions of the predicate;
* ``CONFLICTS`` -- the measurements clearly fail a condition of the predicate;
* ``AMBIGUOUS`` -- the measurements fall inside a tolerance band, or the geometry is not reliable
  enough to decide;
* ``UNAVAILABLE`` -- the evidence could not be computed at all.

An ambiguous or unavailable record must say why: a caveat is stated, never implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.geometric_mapping import GeometryReference, MapId
from contextmap.spatial_relations._checks import require_canonical, require_finite, require_present
from contextmap.spatial_relations._identity import candidate_digest, require_relatable_pair
from contextmap.spatial_relations.statements import ObservationRelationStatement
from contextmap.spatial_relations.taxonomy import RelationPredicate, predicate_spec

RelationEvidenceId = NewType("RelationEvidenceId", str)
"""Identity of one evidence record, local to its spatial-relations artifact."""


class RelationEvidenceChannel(Enum):
    """The kind of measurement a piece of evidence comes from.

    Attributes:
        GEOMETRY: Bounds-based measurements of two entities: proximity, direction and topology.
        CONTACT: Point-level measurements of contact and support.
        OBSERVATION: Upstream relational statements about two entities, held by reference. It
            corroborates measured evidence and never decides a relation on its own.
    """

    GEOMETRY = "geometry"
    CONTACT = "contact"
    OBSERVATION = "observation"


class RelationEvidenceStatus(Enum):
    """What one piece of evidence says about its candidate.

    Attributes:
        SUPPORTS: The measurements meet the conditions of the predicate.
        CONFLICTS: The measurements clearly fail a condition of the predicate.
        AMBIGUOUS: The measurements are inside a tolerance band or the geometry is not reliable
            enough to decide.
        UNAVAILABLE: The evidence could not be computed.
    """

    SUPPORTS = "supports"
    CONFLICTS = "conflicts"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


class EvidenceCaveatKind(Enum):
    """Why a piece of evidence is not decisive.

    Attributes:
        WITHIN_TOLERANCE: A measurement fell inside the tolerance band of its threshold.
        UNRELIABLE_GEOMETRY: The support of an entity is flagged by the geometry summary as
            sparse or disconnected, so its bounds may not describe the object.
        DEGENERATE_GEOMETRY: A measurement needs an extent the geometry does not have, such as a
            footprint with no area.
        MISSING_INPUT: An input the rule needs was not available.
        CONFLICTING_STATEMENTS: Upstream statements about the candidate assert and deny it.
    """

    WITHIN_TOLERANCE = "within_tolerance"
    UNRELIABLE_GEOMETRY = "unreliable_geometry"
    DEGENERATE_GEOMETRY = "degenerate_geometry"
    MISSING_INPUT = "missing_input"
    CONFLICTING_STATEMENTS = "conflicting_statements"


@dataclass(frozen=True, kw_only=True)
class Quantity:
    """A named number with its unit.

    Attributes:
        name: What the number is, in a rule's own vocabulary such as ``bounds_gap``.
        value: The number; always finite.
        unit: The unit, such as ``m``, ``m2``, ``m3``, ``ratio``, ``count`` or ``deg``.
    """

    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        """Validate the name, the unit and the number.

        Raises:
            ValueError: If the name or the unit is empty, or the value is not finite.
        """
        require_present(self, "name", "unit")
        require_finite(f"quantity {self.name!r}", self.value)


@dataclass(frozen=True, kw_only=True)
class EvidenceCaveat:
    """A stated reason why evidence is not decisive.

    Attributes:
        kind: The kind of limitation.
        detail: A deterministic, human-readable explanation.
    """

    kind: EvidenceCaveatKind
    detail: str

    def __post_init__(self) -> None:
        """Require an explanation.

        Raises:
            ValueError: If the detail is empty.
        """
        require_present(self, "detail")


@dataclass(frozen=True, kw_only=True)
class MeasuredGeometry:
    """The geometry one measurement was taken over.

    The evidence never copies coordinates. A whole support is identified by the digest of its
    geometry identities; a small subset, such as the points near a contact, may also list its
    references so the exact points can be resolved from the geometric map.

    Attributes:
        role: What the geometry is in the measurement, such as ``subject`` or
            ``subject_contact_band``.
        point_count: Geometry elements of the set.
        geometry_digest: Digest of the exact set of geometry identities.
        geometry_refs: The references themselves when the set is small enough to list, sorted by
            geometry id; empty when only the digest is kept.
    """

    role: str
    point_count: int
    geometry_digest: str
    geometry_refs: tuple[GeometryReference, ...] = ()

    def __post_init__(self) -> None:
        """Validate the description of the set.

        Raises:
            ValueError: If the role or the digest is empty, the count is not positive, listed
                references are not sorted and unique, span several maps or disagree with the
                count.
        """
        require_present(self, "role", "geometry_digest")
        if self.point_count < 1:
            raise ValueError(f"point_count must be at least 1, got {self.point_count}")
        if not self.geometry_refs:
            return
        require_canonical(
            "geometry_refs",
            self.geometry_refs,
            lambda reference: (reference.geometry_id,),
            detail="by geometry_id ",
        )
        if len({reference.map_id for reference in self.geometry_refs}) > 1:
            raise ValueError("geometry_refs must belong to one map")
        if len(self.geometry_refs) != self.point_count:
            raise ValueError(
                f"point_count {self.point_count} disagrees with the {len(self.geometry_refs)} "
                f"listed geometry_refs"
            )


@dataclass(frozen=True, kw_only=True)
class RelationEvidenceProvenance:
    """Which rule produced a piece of evidence, under which configuration and in which frame.

    Attributes:
        rule_id: The versioned rule that produced the evidence.
        configuration_fingerprint: Hash of the policy and thresholds the rule ran under.
        taxonomy_version: The vocabulary version the predicate is defined in.
        map_frame: The frame every measured coordinate is expressed in.
        geometric_map_id: The geometric map the measured geometry belongs to. It is given
            together with ``map_frame``: coordinates are only comparable inside one map.
        frame_conventions_fingerprint: The declared axes the rule used, when the predicate
            needs any.
        code_version: Code revision that produced the evidence, when known.
    """

    rule_id: str
    configuration_fingerprint: str
    taxonomy_version: str
    map_frame: str | None = None
    geometric_map_id: MapId | None = None
    frame_conventions_fingerprint: str | None = None
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Validate the identities.

        Raises:
            ValueError: If an identity is empty, or only one of the frame and the map is given.
        """
        require_present(self, "rule_id", "configuration_fingerprint", "taxonomy_version")
        if (self.map_frame is None) != (self.geometric_map_id is None):
            raise ValueError("map_frame and geometric_map_id must be given together")
        if self.map_frame is not None:
            require_present(self, "map_frame", "geometric_map_id")


def evidence_id_for(
    *,
    channel: RelationEvidenceChannel,
    subject_entity_ref: ResolvedEntityReference,
    predicate: RelationPredicate,
    object_entity_ref: ResolvedEntityReference,
) -> RelationEvidenceId:
    """Compute the deterministic identity of the evidence of one candidate in one channel.

    Args:
        channel: The evidence channel.
        subject_entity_ref: The entity the statement is about.
        predicate: The predicate evaluated.
        object_entity_ref: The entity it is related to.

    Returns:
        A pure function of the inputs, so evidence can be referenced without a registry.
    """
    digest = candidate_digest(subject_entity_ref, predicate, object_entity_ref)
    return RelationEvidenceId(f"evidence--{channel.value}--{predicate.value}--{digest}")


@dataclass(frozen=True, kw_only=True)
class RelationEvidence:
    """What one channel measured about one directed relation candidate.

    Attributes:
        evidence_id: Identity of the record, local to its artifact.
        channel: The measurement channel.
        subject_entity_ref: The resolved entity the statement is about.
        predicate: The predicate evaluated. It is always the direction that is evaluated
            directly: the inverse of a derived predicate is generated from it, never measured.
        object_entity_ref: The resolved entity it is related to.
        status: What the measurements say about the candidate.
        measurements: The exact measured numbers, sorted by name and unique.
        thresholds: The thresholds and tolerances the measurements were compared with, sorted by
            name and unique.
        geometry: The geometry the measurements were taken over, sorted by role and unique;
            required for measured channels and absent for observation evidence.
        caveats: Why the record is not decisive, sorted and unique.
        provenance: The rule, configuration and frame behind the record.
        statements: The upstream statements an observation record rests on, sorted and unique;
            required for the observation channel and absent for measured ones.
    """

    evidence_id: RelationEvidenceId
    channel: RelationEvidenceChannel
    subject_entity_ref: ResolvedEntityReference
    predicate: RelationPredicate
    object_entity_ref: ResolvedEntityReference
    status: RelationEvidenceStatus
    measurements: tuple[Quantity, ...]
    thresholds: tuple[Quantity, ...]
    geometry: tuple[MeasuredGeometry, ...]
    provenance: RelationEvidenceProvenance
    caveats: tuple[EvidenceCaveat, ...] = ()
    statements: tuple[ObservationRelationStatement, ...] = ()

    def __post_init__(self) -> None:
        """Validate that the record is coherent and grounded.

        Raises:
            ValueError: If the identity is empty, the pair is not relatable, the predicate is
                derived, a collection is not canonical, an ambiguous or unavailable record has no
                caveat, a decisive record has no measurements, or a geometric record lacks the
                geometry or the map scope of its coordinates.
        """
        require_present(self, "evidence_id")
        require_relatable_pair(self.subject_entity_ref, self.object_entity_ref)
        if predicate_spec(self.predicate).is_derived:
            raise ValueError(
                f"{self.predicate.name} is derived: evidence is recorded for the evaluated "
                f"direction of its inverse, {predicate_spec(self.predicate).inverse}"
            )
        require_canonical("measurements", self.measurements, lambda item: (item.name,))
        require_canonical("thresholds", self.thresholds, lambda item: (item.name,))
        require_canonical("geometry", self.geometry, lambda item: (item.role,), detail="by role ")
        require_canonical("caveats", self.caveats, lambda item: (item.kind.value, item.detail))
        if not self.caveats and self.status in _UNDECIDED:
            raise ValueError(
                f"{self.status.value} evidence needs a caveat that says why it is not decisive"
            )
        if not self.measurements and self.status in _DECISIVE:
            raise ValueError(f"{self.status.value} evidence needs the measurements behind it")
        if self.channel is RelationEvidenceChannel.OBSERVATION:
            self._require_observation_shape()
        else:
            self._require_measured_shape()

    def _require_observation_shape(self) -> None:
        if not self.statements:
            raise ValueError("observation evidence needs the upstream statements it rests on")
        if self.geometry:
            raise ValueError("observation evidence carries no geometry")
        require_canonical("statements", self.statements, lambda item: item.sort_key)
        pair = {self.subject_entity_ref, self.object_entity_ref}
        for statement in self.statements:
            if {statement.subject.entity_ref, statement.object.entity_ref} != pair:
                raise ValueError(
                    f"statement {statement.source.statement_id!r} is not about the entities of "
                    f"this record"
                )

    def _require_measured_shape(self) -> None:
        if self.statements:
            raise ValueError("only observation evidence carries upstream statements")
        if not self.geometry:
            raise ValueError("evidence needs the geometry its measurements were taken over")
        if self.provenance.map_frame is None:
            raise ValueError(
                "evidence needs the map frame and geometric map of its coordinates in its "
                "provenance"
            )


_DECISIVE = frozenset({RelationEvidenceStatus.SUPPORTS, RelationEvidenceStatus.CONFLICTS})
_UNDECIDED = frozenset({RelationEvidenceStatus.AMBIGUOUS, RelationEvidenceStatus.UNAVAILABLE})

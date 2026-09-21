"""Candidate generation: which entity pairs are worth evaluating, and why the others are not.

Evaluating every predicate for every pair of entities is quadratic in the map size and mostly
pointless: an object on the other side of the building is not next to, above or inside anything
here. Candidate generation narrows the work with explicit, cheap geometric preconditions and
records why each pair was kept or dropped. It never decides that a relation is true: a candidate is
a *reason to measure*, not evidence, and it carries no status, score or state.

The stage reads only entity references and their
:class:`~contextmap.semantic_mapping.EntityGeometry`. It has no input through which a semantic
label could create or exclude a candidate.

How it stays sub-quadratic: entities are ordered along the axis on which they are most spread out
and a sweep only pairs boxes whose gap along that axis is within the largest reach of the policy
(sweep and prune), so a long corridor of entities costs work proportional to the pairs that are
actually near each other. Pairs the sweep proves farther apart than every reach are not enumerated
and are only counted, which keeps the exclusion record bounded and honest.

Every precondition is a *necessary* condition of the corresponding evaluator, on the assumption
that the reaches of the policy cover the distance tolerances of the evaluators. If they do not, a
true relation can be lost before it is measured; that loss is a candidate-retrieval failure and is
measured separately from predicate quality.

Preconditions are checked in a fixed order and the first that fails is the recorded reason:

1. the bounds gap is within the reach of the predicate (proximity or directional);
2. for directional and support predicates, the cross-sections perpendicular to the axis overlap
   (a subject is only above what it is over);
3. for the same predicates, the subject's center lies further along the axis than the object's
   (so that ``a ABOVE b`` and ``b ABOVE a`` are never both plausible);
4. for ``INSIDE``, the subject can fit inside the object on every axis.

A symmetric predicate is a candidate once per unordered pair, in canonical reference order. A
directed predicate is checked in both directions, each on its own. Derived predicates are never
candidates: their relations are generated from the evaluated direction of their inverse.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.geometric_mapping import Bounds3D, MapId
from contextmap.semantic_mapping import EntityGeometry
from contextmap.spatial_relations._bounds import (
    axis_overlap_m,
    bounds_gap_m,
    cross_section_axes,
    directed_interval,
)
from contextmap.spatial_relations._checks import require_canonical, require_finite, require_present
from contextmap.spatial_relations._identity import reference_key, require_relatable_pair
from contextmap.spatial_relations.frame_conventions import AxisDirection, FrameConventions
from contextmap.spatial_relations.taxonomy import (
    TAXONOMY_VERSION,
    FrameRequirement,
    RelationPredicate,
    predicate_spec,
)

CANDIDATE_POLICY_ID = "bounds-neighborhood-candidates-v1"
"""Versioned identity of the candidate rules described in this module."""


class CandidateReason(Enum):
    """A precondition that held, and so a reason the candidate was kept.

    Attributes:
        WITHIN_PROXIMITY_RADIUS: The bounds gap is within the proximity reach.
        WITHIN_DIRECTIONAL_RADIUS: The bounds gap is within the directional reach.
        FOOTPRINT_OVERLAP: The cross-sections perpendicular to the predicate's axis overlap.
        SUBJECT_ON_DIRECTED_SIDE: The subject's center lies further along the predicate's axis
            than the object's.
        CONTAINMENT_POSSIBLE: The subject can fit inside the object on every axis.
    """

    WITHIN_PROXIMITY_RADIUS = "within_proximity_radius"
    WITHIN_DIRECTIONAL_RADIUS = "within_directional_radius"
    FOOTPRINT_OVERLAP = "footprint_overlap"
    SUBJECT_ON_DIRECTED_SIDE = "subject_on_directed_side"
    CONTAINMENT_POSSIBLE = "containment_possible"


class CandidateExclusionReason(Enum):
    """The first precondition that failed, and so the reason a pair was dropped.

    Attributes:
        BEYOND_PROXIMITY_RADIUS: The bounds gap exceeds the proximity reach.
        BEYOND_DIRECTIONAL_RADIUS: The bounds gap exceeds the directional reach.
        NO_FOOTPRINT_OVERLAP: The cross-sections perpendicular to the axis do not overlap.
        SUBJECT_NOT_ON_DIRECTED_SIDE: The subject's center is not further along the axis than the
            object's.
        CONTAINMENT_IMPOSSIBLE: The subject is larger than the object on some axis.
    """

    BEYOND_PROXIMITY_RADIUS = "beyond_proximity_radius"
    BEYOND_DIRECTIONAL_RADIUS = "beyond_directional_radius"
    NO_FOOTPRINT_OVERLAP = "no_footprint_overlap"
    SUBJECT_NOT_ON_DIRECTED_SIDE = "subject_not_on_directed_side"
    CONTAINMENT_IMPOSSIBLE = "containment_impossible"


@dataclass(frozen=True, kw_only=True)
class CandidatePolicy:
    """Explicit configuration of candidate generation.

    There are no defaults: how far a relation may reach is a scientific choice that a profile
    declares. Both reaches should cover the distance tolerances of the evaluators they feed, or
    true relations are lost before they are measured.

    Attributes:
        predicates: The predicates to generate candidates for, unique, each one evaluated
            directly (never a derived predicate).
        proximity_radius_m: The largest bounds gap, in meters, for proximity, topological and
            contact predicates.
        directional_radius_m: The largest bounds gap, in meters, for directional predicates. It
            is separate because "above" and "in front of" reach much further than "next to".
    """

    predicates: tuple[RelationPredicate, ...]
    proximity_radius_m: float
    directional_radius_m: float

    def __post_init__(self) -> None:
        """Validate the selection and the reaches.

        Raises:
            ValueError: If no predicate is selected, one is repeated or derived, or a reach is not
                finite and positive.
        """
        if not self.predicates:
            raise ValueError("predicates must not be empty")
        if len(set(self.predicates)) != len(self.predicates):
            raise ValueError("predicates must be unique")
        for predicate in self.predicates:
            spec = predicate_spec(predicate)
            if spec.is_derived and spec.inverse is not None:
                raise ValueError(
                    f"{predicate.name} is derived from {spec.inverse.name}: select "
                    f"{spec.inverse.name}, and the {predicate.name} relations are generated"
                )
        for name in ("proximity_radius_m", "directional_radius_m"):
            value: float = getattr(self, name)
            require_finite(name, value)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")

    def fingerprint(self) -> str:
        """Hash the policy identity and choices, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration, independent of the
            order the predicates were listed in.
        """
        canonical = json.dumps(
            {
                "policy_id": CANDIDATE_POLICY_ID,
                "predicates": sorted(predicate.value for predicate in self.predicates),
                "proximity_radius_m": self.proximity_radius_m,
                "directional_radius_m": self.directional_radius_m,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


@dataclass(frozen=True, kw_only=True)
class RelationCandidate:
    """A directed pair worth measuring for one predicate.

    A candidate is a reason to evaluate, never evidence: it has no status and no score.

    Attributes:
        subject_entity_ref: The entity the statement would be about.
        predicate: The predicate to evaluate, one evaluated directly.
        object_entity_ref: The entity it would be related to.
        reasons: The preconditions that held, sorted and never empty.
        bounds_gap_m: The measured gap between the entities' bounds, in meters.
    """

    subject_entity_ref: ResolvedEntityReference
    predicate: RelationPredicate
    object_entity_ref: ResolvedEntityReference
    reasons: tuple[CandidateReason, ...]
    bounds_gap_m: float

    def __post_init__(self) -> None:
        """Validate that the candidate is a relatable pair with reasons.

        Raises:
            ValueError: If the pair is not relatable, the predicate is derived, the reasons are
                missing or not canonical, or the gap is negative or not finite.
        """
        _require_candidate_shape(self.subject_entity_ref, self.predicate, self.object_entity_ref)
        if not self.reasons:
            raise ValueError("a candidate needs the reasons it was kept")
        require_canonical("reasons", self.reasons, lambda item: (item.value,))
        _require_gap(self.bounds_gap_m)


@dataclass(frozen=True, kw_only=True)
class CandidateExclusion:
    """A directed pair that was looked at and dropped, and why.

    Attributes:
        subject_entity_ref: The entity the statement would have been about.
        predicate: The predicate that was ruled out.
        object_entity_ref: The entity it would have been related to.
        reason: The first precondition that failed.
        bounds_gap_m: The measured gap between the entities' bounds, in meters.
    """

    subject_entity_ref: ResolvedEntityReference
    predicate: RelationPredicate
    object_entity_ref: ResolvedEntityReference
    reason: CandidateExclusionReason
    bounds_gap_m: float

    def __post_init__(self) -> None:
        """Validate that the exclusion is about a relatable pair.

        Raises:
            ValueError: If the pair is not relatable, the predicate is derived, or the gap is
                negative or not finite.
        """
        _require_candidate_shape(self.subject_entity_ref, self.predicate, self.object_entity_ref)
        _require_gap(self.bounds_gap_m)


@dataclass(frozen=True, kw_only=True)
class SkippedPredicate:
    """A selected predicate that the declared frame conventions cannot carry.

    This is a fact about the run, not about any pair, so it is recorded once instead of once per
    pair.

    Attributes:
        predicate: The predicate that was not evaluated.
        requirement: What it needs from the map frame.
        detail: A deterministic, human-readable explanation.
    """

    predicate: RelationPredicate
    requirement: FrameRequirement
    detail: str

    def __post_init__(self) -> None:
        """Require an explanation.

        Raises:
            ValueError: If the detail is empty.
        """
        require_present(self, "detail")


@dataclass(frozen=True, kw_only=True)
class CandidateProvenance:
    """Which rules, configuration and frame produced a candidate set.

    Attributes:
        policy_id: The versioned candidate rules.
        configuration_fingerprint: Hash of the :class:`CandidatePolicy`.
        taxonomy_version: The vocabulary version the predicates are defined in.
        map_frame: The frame the geometry is expressed in.
        geometric_map_id: The geometric map the geometry belongs to; ``None`` when there was no
            entity to read it from.
        frame_conventions_fingerprint: The axes the run declared for the map frame.
    """

    policy_id: str
    configuration_fingerprint: str
    taxonomy_version: str
    map_frame: str
    geometric_map_id: MapId | None
    frame_conventions_fingerprint: str

    def __post_init__(self) -> None:
        """Require the identities.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(
            self,
            "policy_id",
            "configuration_fingerprint",
            "taxonomy_version",
            "map_frame",
            "frame_conventions_fingerprint",
        )


@dataclass(frozen=True, kw_only=True)
class RelationCandidateSet:
    """The outcome of candidate generation over one set of resolved entities.

    Attributes:
        candidates: The directed pairs worth measuring, sorted by subject, predicate and object.
        exclusions: The enumerated pairs that were dropped, sorted the same way, each with the
            first precondition that failed.
        skipped_predicates: The selected predicates the frame conventions cannot carry, sorted by
            predicate.
        entity_count: The entities considered.
        pairs_not_enumerated: The unordered pairs the sweep proved farther apart than every reach.
            They are counted, not listed.
        provenance: The rules, configuration and frame behind the set.
    """

    candidates: tuple[RelationCandidate, ...]
    exclusions: tuple[CandidateExclusion, ...]
    skipped_predicates: tuple[SkippedPredicate, ...]
    entity_count: int
    pairs_not_enumerated: int
    provenance: CandidateProvenance

    def __post_init__(self) -> None:
        """Validate that the record is canonical and consistent.

        Raises:
            ValueError: If a collection is not sorted and unique, a pair is both a candidate and
                an exclusion, or a count is negative or inconsistent.
        """
        require_canonical("candidates", self.candidates, _directed_key)
        require_canonical("exclusions", self.exclusions, _directed_key)
        require_canonical(
            "skipped_predicates", self.skipped_predicates, lambda i: (i.predicate.value,)
        )
        if {_directed_key(item) for item in self.candidates} & {
            _directed_key(item) for item in self.exclusions
        }:
            raise ValueError("a directed pair cannot be both a candidate and an exclusion")
        if self.entity_count < 0 or self.pairs_not_enumerated < 0:
            raise ValueError("counts must not be negative")
        if self.pairs_not_enumerated > self.entity_count * (self.entity_count - 1) // 2:
            raise ValueError("pairs_not_enumerated exceeds the pairs the entities can form")


def generate_relation_candidates(
    entities: Mapping[ResolvedEntityReference, EntityGeometry],
    *,
    policy: CandidatePolicy,
    conventions: FrameConventions,
) -> RelationCandidateSet:
    """Narrow the entity pairs to those worth evaluating, deterministically.

    Args:
        entities: The geometry of every resolved entity of one resolution artifact, by reference.
            Only the geometry is read; nothing about what an entity is can affect the result.
        policy: The predicates to generate candidates for and how far they reach.
        conventions: The axes the run declared for the map frame. A predicate that needs an axis
            that is not declared is skipped and reported, never guessed.

    Returns:
        The candidates, the exclusions with their reasons, the skipped predicates and the counts.
        The same entities, policy and conventions always give the same set, whatever the order of
        ``entities``.

    Raises:
        IncompatibleFrameError: If a geometry is not expressed in the frame of ``conventions`` or
            the geometries belong to different geometric maps.
        ValueError: If the entities come from more than one resolution artifact.
    """
    references = sorted(entities, key=reference_key)
    if len({reference.resolution_run_id for reference in references}) > 1:
        raise ValueError("entities come from different resolution artifacts")
    geometries = [entities[reference] for reference in references]
    conventions.require_evaluable(FrameRequirement.MAP_FRAME, *geometries)

    evaluated: list[RelationPredicate] = []
    skipped: list[SkippedPredicate] = []
    for predicate in sorted(policy.predicates, key=lambda item: item.value):
        requirement = predicate_spec(predicate).frame_requirement
        if conventions.supports(requirement):
            evaluated.append(predicate)
        else:
            skipped.append(_skipped(predicate, requirement, conventions))

    candidates: list[RelationCandidate] = []
    exclusions: list[CandidateExclusion] = []
    enumerated = 0
    reach = max(policy.proximity_radius_m, policy.directional_radius_m)
    for first, second in _neighbor_pairs(geometries, reach):
        enumerated += 1
        gap = bounds_gap_m(geometries[first].bounds, geometries[second].bounds)
        for predicate in evaluated:
            directions = [(first, second)]
            if predicate_spec(predicate).directed:
                directions.append((second, first))
            for subject, obj in directions:
                outcome = _assess(
                    predicate, geometries[subject], geometries[obj], gap, policy, conventions
                )
                if isinstance(outcome, CandidateExclusionReason):
                    exclusions.append(
                        CandidateExclusion(
                            subject_entity_ref=references[subject],
                            predicate=predicate,
                            object_entity_ref=references[obj],
                            reason=outcome,
                            bounds_gap_m=gap,
                        )
                    )
                else:
                    candidates.append(
                        RelationCandidate(
                            subject_entity_ref=references[subject],
                            predicate=predicate,
                            object_entity_ref=references[obj],
                            reasons=tuple(sorted(outcome, key=lambda item: item.value)),
                            bounds_gap_m=gap,
                        )
                    )
    count = len(references)
    return RelationCandidateSet(
        candidates=tuple(sorted(candidates, key=_directed_key)),
        exclusions=tuple(sorted(exclusions, key=_directed_key)),
        skipped_predicates=tuple(skipped),
        entity_count=count,
        pairs_not_enumerated=count * (count - 1) // 2 - enumerated,
        provenance=CandidateProvenance(
            policy_id=CANDIDATE_POLICY_ID,
            configuration_fingerprint=policy.fingerprint(),
            taxonomy_version=TAXONOMY_VERSION,
            map_frame=conventions.map_frame,
            geometric_map_id=geometries[0].geometric_map_id if geometries else None,
            frame_conventions_fingerprint=conventions.fingerprint(),
        ),
    )


def _neighbor_pairs(geometries: list[EntityGeometry], reach: float) -> Iterator[tuple[int, int]]:
    """Enumerate the index pairs whose bounds may lie within ``reach`` of each other.

    Sweep and prune: the boxes are ordered along the axis on which their centers are most spread
    out and each box is only paired with the following ones whose lower face starts within
    ``reach`` of its upper face. A pair is yielded only when its gap is within ``reach`` on every
    axis, which every pair with an Euclidean gap within ``reach`` satisfies.

    Args:
        geometries: The entities' geometry, in canonical reference order.
        reach: The largest bounds gap any predicate can accept, in meters.

    Yields:
        ``(first, second)`` with ``first < second``.
    """
    count = len(geometries)
    if count < 2:
        return
    bounds = [geometry.bounds for geometry in geometries]
    axis = max(range(3), key=lambda k: _center_spread(bounds, k))
    order = sorted(range(count), key=lambda index: (bounds[index].minimum_m[axis], index))
    for position, first in enumerate(order):
        limit = bounds[first].maximum_m[axis] + reach
        for second in order[position + 1 :]:
            if bounds[second].minimum_m[axis] > limit:
                break
            if all(
                axis_overlap_m(bounds[first], bounds[second], other) >= -reach for other in range(3)
            ):
                yield (min(first, second), max(first, second))


def _center_spread(bounds: list[Bounds3D], axis: int) -> float:
    centers = [(box.minimum_m[axis] + box.maximum_m[axis]) / 2.0 for box in bounds]
    return max(centers) - min(centers)


def _assess(
    predicate: RelationPredicate,
    subject: EntityGeometry,
    obj: EntityGeometry,
    gap: float,
    policy: CandidatePolicy,
    conventions: FrameConventions,
) -> list[CandidateReason] | CandidateExclusionReason:
    """Check the preconditions of one directed pair, in the documented order.

    Returns:
        The reasons the pair was kept, or the first precondition that failed.
    """
    rule = _RULES[predicate]
    reasons: list[CandidateReason] = []
    if rule.directional_reach:
        if gap > policy.directional_radius_m:
            return CandidateExclusionReason.BEYOND_DIRECTIONAL_RADIUS
        reasons.append(CandidateReason.WITHIN_DIRECTIONAL_RADIUS)
    else:
        if gap > policy.proximity_radius_m:
            return CandidateExclusionReason.BEYOND_PROXIMITY_RADIUS
        reasons.append(CandidateReason.WITHIN_PROXIMITY_RADIUS)
    direction = _axis_of(rule.axis_role, conventions)
    if direction is not None:
        if not all(
            axis_overlap_m(subject.bounds, obj.bounds, other) >= 0.0
            for other in cross_section_axes(direction)
        ):
            return CandidateExclusionReason.NO_FOOTPRINT_OVERLAP
        reasons.append(CandidateReason.FOOTPRINT_OVERLAP)
        if _center_along(subject, direction) <= _center_along(obj, direction):
            return CandidateExclusionReason.SUBJECT_NOT_ON_DIRECTED_SIDE
        reasons.append(CandidateReason.SUBJECT_ON_DIRECTED_SIDE)
    if rule.containment:
        if any(
            subject.extent_m[axis] > obj.extent_m[axis] + policy.proximity_radius_m
            for axis in range(3)
        ):
            return CandidateExclusionReason.CONTAINMENT_IMPOSSIBLE
        reasons.append(CandidateReason.CONTAINMENT_POSSIBLE)
    return reasons


def _center_along(geometry: EntityGeometry, direction: AxisDirection) -> float:
    low, high = directed_interval(geometry.bounds, direction)
    return (low + high) / 2.0


class _AxisRole(Enum):
    """Which declared axis a predicate's side and footprint preconditions read."""

    UP = "up"
    FORWARD = "forward"


@dataclass(frozen=True, kw_only=True)
class _Rule:
    """The preconditions of one predicate.

    Attributes:
        directional_reach: Whether the directional reach applies instead of the proximity one.
        axis_role: The declared axis the footprint and side preconditions read, or ``None`` when
            the predicate has none.
        containment: Whether the subject must be able to fit inside the object.
    """

    directional_reach: bool = False
    axis_role: _AxisRole | None = None
    containment: bool = False


# Predicados derivados (BELOW, BEHIND, CONTAINS) não aparecem: nunca são candidatos, pois suas
# relações são geradas a partir do sentido avaliado do inverso.
_RULES: dict[RelationPredicate, _Rule] = {
    RelationPredicate.NEXT_TO: _Rule(),
    RelationPredicate.INTERSECTS: _Rule(),
    RelationPredicate.TOUCHING: _Rule(),
    RelationPredicate.INSIDE: _Rule(containment=True),
    RelationPredicate.ABOVE: _Rule(directional_reach=True, axis_role=_AxisRole.UP),
    RelationPredicate.IN_FRONT_OF: _Rule(directional_reach=True, axis_role=_AxisRole.FORWARD),
    RelationPredicate.ON_TOP_OF: _Rule(axis_role=_AxisRole.UP),
    RelationPredicate.LEANING_AGAINST: _Rule(),
}


def _axis_of(role: _AxisRole | None, conventions: FrameConventions) -> AxisDirection | None:
    """Read the declared axis a rule needs; the caller has checked that it is declared."""
    if role is None:
        return None
    axis = conventions.up_axis if role is _AxisRole.UP else conventions.forward_axis
    if axis is None:
        raise ValueError(f"the {role.value} axis is not declared")
    return axis


def _directed_key(item: RelationCandidate | CandidateExclusion) -> tuple[str, str, str, str, str]:
    """Canonical order of a directed pair: subject, predicate, object."""
    return (
        *reference_key(item.subject_entity_ref),
        item.predicate.value,
        *reference_key(item.object_entity_ref),
    )


def _require_candidate_shape(
    subject: ResolvedEntityReference, predicate: RelationPredicate, obj: ResolvedEntityReference
) -> None:
    require_relatable_pair(subject, obj)
    if predicate_spec(predicate).is_derived:
        raise ValueError(f"{predicate.name} is derived and is never a candidate of its own")


def _require_gap(gap_m: float) -> None:
    require_finite("bounds_gap_m", gap_m)
    if gap_m < 0.0:
        raise ValueError(f"bounds_gap_m must not be negative, got {gap_m!r}")


def _skipped(
    predicate: RelationPredicate, requirement: FrameRequirement, conventions: FrameConventions
) -> SkippedPredicate:
    return SkippedPredicate(
        predicate=predicate,
        requirement=requirement,
        detail=(
            f"{predicate.name} needs {requirement.value}, which the conventions for frame "
            f"{conventions.map_frame!r} do not declare"
        ),
    )

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
and are only counted.

How the exclusion record stays bounded: every enumerated pair yields a candidate or an exclusion
per evaluated predicate and direction, so exclusions grow with the pairs near each other. They are
grouped by predicate and reason, and a group is listed whole only while it holds at most
``_MAX_LISTED_EXCLUSIONS`` records. A larger group lists only that many *nearest* exclusions, the
smallest bounds gaps (ties broken by subject, predicate and object), and a
:class:`CandidateExclusionSummary` keeps the count, the extreme gaps and a digest of all of them.
Each group is kept bounded while the pairs are enumerated, never after the fact.

The digest of a group is ``sha256-multiset:`` followed by the 64 hexadecimal digits of the sum,
modulo ``2**256``, of the SHA-256 of every exclusion of the group, each encoded as compact JSON with
sorted keys over exactly its ``subject_entity_ref``, ``predicate``, ``object_entity_ref``,
``reason`` and ``bounds_gap_m`` (the persisted exclusion record without its ``record`` tag). A sum
does not depend on the order the exclusions are found in, so it takes constant memory while pairs
are enumerated in sweep order, and anyone holding the full list can recompute it in any order. It
is weaker than a hash of the sorted list against collisions built on purpose; an exclusive-or
would be weaker still, since equal records would cancel out.

Every precondition is a *necessary* condition for the corresponding evaluator to *support* the
relation, on the assumption that the reaches of the policy cover the distance tolerances of the
evaluators. If they did not, a true relation could be lost before it is measured. The assumption is
verified where the policies of a run meet:
:class:`~contextmap.spatial_relations.RelationsRunPolicies` refuses a ``proximity_radius_m`` below
``next_to_max_gap_m`` or ``2 * containment_slack_m`` of the geometric evaluators, or below
``contact_distance_m + contact_tolerance_m`` of the contact evaluators, for each evaluator the run
declares. This module alone cannot check it, since it does not know the evaluators; a caller that
pairs the policies without ``RelationsRunPolicies`` carries the assumption itself. A loss that
remains is a candidate-retrieval failure and is measured separately from predicate quality.

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

import bisect
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import Any

from contextmap.entity_resolution import ResolvedEntityReference, encode_resolved_entity_reference
from contextmap.geometric_mapping import MapId
from contextmap.semantic_mapping import EntityGeometry
from contextmap.spatial_relations._bounds import (
    axis_overlap_m,
    bounds_gap_m,
    cross_section_axes,
    directed_interval,
    widest_spread_axis,
)
from contextmap.spatial_relations._checks import require_canonical, require_finite, require_present
from contextmap.spatial_relations._identity import (
    directed_key,
    reference_key,
    require_relatable_pair,
)
from contextmap.spatial_relations.frame_conventions import AxisDirection, FrameConventions
from contextmap.spatial_relations.taxonomy import (
    TAXONOMY_VERSION,
    FrameRequirement,
    RelationPredicate,
    predicate_spec,
)

CANDIDATE_POLICY_ID = "bounds-neighborhood-candidates-v1"
"""Versioned identity of the candidate rules described in this module."""

_MAX_LISTED_EXCLUSIONS = 32
"""Most exclusions listed for one ``(predicate, reason)`` group; a larger group is summarized.

A record limit, not a scientific threshold: it changes what the candidate set lists, never which
pairs are candidates, so it is not part of :class:`CandidatePolicy` or its fingerprint. It is the
smallest of 8, 16, 32 and 64 under which every retrieval miss of the annotated storeroom fixture
keeps the exclusion that explains it without a ceiling (#601); the matrix and the rule are in
``docs/candidates.md``, and ``tests/spatial_relations/test_relation_evaluation.py`` pins them.
"""

_EXCLUSION_DIGEST_PREFIX = "sha256-multiset:"
_DIGEST_MODULUS = 2**256


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
    true relations are lost before they are measured. This policy only validates itself;
    :class:`~contextmap.spatial_relations.RelationsRunPolicies` verifies the proximity reach
    against the tolerances of the evaluators a run declares (see the module docstring).

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
class CandidateExclusionSummary:
    """The exclusions of one predicate and reason that were too many to list.

    Only a group past the listing ceiling has a summary: a smaller group is listed whole in
    :attr:`RelationCandidateSet.exclusions`.

    Attributes:
        predicate: The predicate that was ruled out.
        reason: The precondition that failed.
        count: Every exclusion of the group, listed or not.
        min_gap_m: The smallest bounds gap of the group, in meters.
        max_gap_m: The largest bounds gap of the group, in meters.
        digest: ``sha256-multiset:`` digest of every exclusion of the group, independent of their
            order (see the module docstring).
        nearest: The listed exclusions of the group: those with the smallest bounds gaps, sorted
            by gap and then by subject, predicate and object; fewer than ``count``.
    """

    predicate: RelationPredicate
    reason: CandidateExclusionReason
    count: int
    min_gap_m: float
    max_gap_m: float
    digest: str
    nearest: tuple[CandidateExclusion, ...]

    def __post_init__(self) -> None:
        """Validate that the summary describes one group and lists its nearest exclusions.

        Raises:
            ValueError: If no exclusion is listed or all of them are, a listed exclusion belongs
                to another predicate or reason, the listed ones are not sorted nearest first and
                unique, the gaps are not those of the listed ones' group, or the digest is not a
                ``sha256-multiset:`` digest.
        """
        if not 0 < len(self.nearest) < self.count:
            raise ValueError(
                f"a summary lists some, not all, of its exclusions: {len(self.nearest)} listed "
                f"of {self.count}"
            )
        if any(
            (item.predicate, item.reason) != (self.predicate, self.reason) for item in self.nearest
        ):
            raise ValueError("every listed exclusion must have the summary's predicate and reason")
        if any(
            _nearest_first(left) >= _nearest_first(right) for left, right in pairwise(self.nearest)
        ):
            raise ValueError("nearest must be sorted by gap, subject, predicate, object and unique")
        require_finite("max_gap_m", self.max_gap_m)
        if self.min_gap_m != self.nearest[0].bounds_gap_m or (
            self.max_gap_m < self.nearest[-1].bounds_gap_m
        ):
            raise ValueError(
                "min_gap_m must be the gap of the nearest exclusion, and max_gap_m at least the "
                "gap of the farthest listed one"
            )
        digits = self.digest.removeprefix(_EXCLUSION_DIGEST_PREFIX)
        if digits == self.digest or len(digits) != 64 or digits.strip("0123456789abcdef"):
            raise ValueError(f"digest must be {_EXCLUSION_DIGEST_PREFIX} and 64 hex digits")


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
            first precondition that failed: every exclusion of each ``(predicate, reason)`` group
            within the listing ceiling.
        skipped_predicates: The selected predicates the frame conventions cannot carry, sorted by
            predicate.
        entity_count: The entities considered.
        pairs_not_enumerated: The unordered pairs the sweep proved farther apart than every reach.
            They are counted, not listed.
        provenance: The rules, configuration and frame behind the set.
        exclusion_summaries: The ``(predicate, reason)`` groups past the listing ceiling, sorted
            by predicate and reason, each with its count, gaps, digest and nearest exclusions.
    """

    candidates: tuple[RelationCandidate, ...]
    exclusions: tuple[CandidateExclusion, ...]
    skipped_predicates: tuple[SkippedPredicate, ...]
    entity_count: int
    pairs_not_enumerated: int
    provenance: CandidateProvenance
    exclusion_summaries: tuple[CandidateExclusionSummary, ...] = ()

    def __post_init__(self) -> None:
        """Validate that the record is canonical and consistent.

        Raises:
            ValueError: If a collection is not sorted and unique, a pair is both a candidate and
                an exclusion, a group is both listed whole and summarized, or a count is negative
                or inconsistent.
        """
        require_canonical("candidates", self.candidates, _directed_key)
        require_canonical("exclusions", self.exclusions, _directed_key)
        require_canonical(
            "skipped_predicates", self.skipped_predicates, lambda i: (i.predicate.value,)
        )
        require_canonical(
            "exclusion_summaries",
            self.exclusion_summaries,
            lambda item: (item.predicate.value, item.reason.value),
        )
        summarized = {(item.predicate, item.reason) for item in self.exclusion_summaries}
        if any((item.predicate, item.reason) in summarized for item in self.exclusions):
            raise ValueError("an exclusion group is either listed whole or summarized, not both")
        excluded = [_directed_key(item) for item in self.exclusions] + [
            _directed_key(item) for summary in self.exclusion_summaries for item in summary.nearest
        ]
        if len(set(excluded)) != len(excluded):
            raise ValueError("a directed pair cannot be excluded twice")
        if {_directed_key(item) for item in self.candidates} & set(excluded):
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
    groups: dict[tuple[RelationPredicate, CandidateExclusionReason], _ExclusionGroup] = {}
    # A codificação de cada referência entra no digest de cada exclusão que a nomeia: calculá-la
    # uma vez por entidade evita refazê-la a cada par.
    encoded = [encode_resolved_entity_reference(reference) for reference in references]
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
                    group = groups.get((predicate, outcome))
                    if group is None:
                        group = groups[(predicate, outcome)] = _ExclusionGroup(
                            predicate, outcome, _MAX_LISTED_EXCLUSIONS
                        )
                    group.add(
                        references[subject],
                        references[obj],
                        gap,
                        _exclusion_record(encoded[subject], predicate, encoded[obj], outcome, gap),
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
    ordered = [groups[key] for key in sorted(groups, key=lambda key: (key[0].value, key[1].value))]
    count = len(references)
    return RelationCandidateSet(
        candidates=tuple(sorted(candidates, key=_directed_key)),
        exclusions=tuple(
            sorted(
                (item for group in ordered if group.whole for item in group.nearest),
                key=_directed_key,
            )
        ),
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
        exclusion_summaries=tuple(group.summary() for group in ordered if not group.whole),
    )


class _ExclusionGroup:
    """The exclusions of one predicate and reason, held in bounded memory while pairs are swept.

    Only the ``limit`` nearest exclusions are kept, with the count, the extreme gaps and the
    running digest of all of them; the pairs are enumerated in sweep order, and none of these
    depends on that order.
    """

    def __init__(
        self, predicate: RelationPredicate, reason: CandidateExclusionReason, limit: int
    ) -> None:
        self._predicate = predicate
        self._reason = reason
        self._limit = limit
        self._kept: list[tuple[tuple[float, str, str, str, str, str], CandidateExclusion]] = []
        self._count = 0
        self._min_gap_m = float("inf")
        self._max_gap_m = float("-inf")
        self._digest_sum = 0

    @property
    def whole(self) -> bool:
        """Whether every exclusion of the group is kept, so it is listed rather than summarized."""
        return self._count <= self._limit

    @property
    def nearest(self) -> tuple[CandidateExclusion, ...]:
        """The kept exclusions, nearest first."""
        return tuple(exclusion for _, exclusion in self._kept)

    def add(
        self,
        subject: ResolvedEntityReference,
        obj: ResolvedEntityReference,
        gap_m: float,
        record: Mapping[str, Any],
    ) -> None:
        """Count one exclusion of the group and keep it if it is among the nearest.

        Args:
            subject: The entity the statement would have been about.
            obj: The entity it would have been related to.
            gap_m: Their bounds gap, in meters.
            record: The exclusion's canonical fields (:func:`exclusion_fields`), for the digest.
        """
        self._count += 1
        self._min_gap_m = min(self._min_gap_m, gap_m)
        self._max_gap_m = max(self._max_gap_m, gap_m)
        self._digest_sum = (self._digest_sum + _record_hash(record)) % _DIGEST_MODULUS
        key = (gap_m, *directed_key(subject, self._predicate, obj))
        if len(self._kept) == self._limit and key > self._kept[-1][0]:
            return
        # As chaves são únicas no grupo (um par dirigido por predicado), então a tupla nunca
        # chega a comparar as exclusões. Só a que fica é construída.
        exclusion = CandidateExclusion(
            subject_entity_ref=subject,
            predicate=self._predicate,
            object_entity_ref=obj,
            reason=self._reason,
            bounds_gap_m=gap_m,
        )
        bisect.insort(self._kept, (key, exclusion))
        if len(self._kept) > self._limit:
            self._kept.pop()

    def summary(self) -> CandidateExclusionSummary:
        """The summary of a group past the ceiling."""
        return CandidateExclusionSummary(
            predicate=self._predicate,
            reason=self._reason,
            count=self._count,
            min_gap_m=self._min_gap_m,
            max_gap_m=self._max_gap_m,
            digest=f"{_EXCLUSION_DIGEST_PREFIX}{self._digest_sum:064x}",
            nearest=self.nearest,
        )


def exclusion_fields(exclusion: CandidateExclusion) -> dict[str, Any]:
    """Encode an exclusion as JSON-compatible fields: the persisted record and its digest input.

    Args:
        exclusion: The exclusion.

    Returns:
        Its ``subject_entity_ref``, ``predicate``, ``object_entity_ref``, ``reason`` and
        ``bounds_gap_m``, with the references encoded by Entity Resolution.
    """
    return _exclusion_record(
        encode_resolved_entity_reference(exclusion.subject_entity_ref),
        exclusion.predicate,
        encode_resolved_entity_reference(exclusion.object_entity_ref),
        exclusion.reason,
        exclusion.bounds_gap_m,
    )


def _exclusion_record(
    subject: Mapping[str, Any],
    predicate: RelationPredicate,
    obj: Mapping[str, Any],
    reason: CandidateExclusionReason,
    gap_m: float,
) -> dict[str, Any]:
    """The fields of an exclusion record, from references already encoded."""
    return {
        "subject_entity_ref": subject,
        "predicate": predicate.value,
        "object_entity_ref": obj,
        "reason": reason.value,
        "bounds_gap_m": gap_m,
    }


def _record_hash(record: Mapping[str, Any]) -> int:
    """The SHA-256 of an exclusion's canonical encoding, as an integer to add to a digest.

    The encoding is compact JSON with sorted keys, the same the run artifact writes.
    """
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest(), "big")


def _nearest_first(exclusion: CandidateExclusion) -> tuple[float, str, str, str, str, str]:
    """Order of the listed exclusions of a group: bounds gap, then subject, predicate, object."""
    return (exclusion.bounds_gap_m, *_directed_key(exclusion))


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
    axis = widest_spread_axis(bounds)
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
    return directed_key(item.subject_entity_ref, item.predicate, item.object_entity_ref)


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

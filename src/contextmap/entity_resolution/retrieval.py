"""Candidate retrieval: a cheap, permissive first step that decides nothing.

Comparing every entity with every other one does not scale, so before any expensive evidence is
computed the resolution retrieves, for each entity, the entities that are *plausibly* the same
object using only interpretable spatial and temporal filters. Retrieval is **not** a match
decision: a returned candidate says nothing about identity, and an entity that is not returned was
excluded by an explicit, diagnosable rule of a versioned policy (:func:`explain_candidacy`).

The baseline, ``entity-candidate-retrieval-v1``, keeps a pair when *any* of these holds:

* the centroids are within ``centroid_radius_m``;
* the axis-aligned bounds overlap or touch;
* the bounds are within ``bounds_margin_m`` of each other.

The bounds rules exist because a large planar entity and an object against it have distant
centroids. Two things never exclude a candidate: **semantics** (upstream labels may be wrong, so
two entities with different or no labels are still retrieved) and **missing optional evidence**.
Time excludes only when the policy declares ``max_time_gap_ns`` and both histories share a clock
domain; timestamps of different clocks are not comparable, so they never drop a candidate.
Entities over a different geometric map or map frame are never candidates: their coordinates are
not comparable without an explicit alignment.

To avoid the all-pairs comparison the entities are bucketed in an in-memory uniform grid
(:class:`EntitySpatialIndex`). Semantic Mapping persists only ``entity-geometry-index.jsonl`` (one
bounds and centroid per entity), not a spatial structure, so this index is built here from that
same data and is not persisted. Every candidate found through the grid is confirmed by the same
exact rule as the definition, so the grid can only make retrieval faster, never different.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum

from contextmap.entity_resolution._checks import require_canonical, require_non_negative
from contextmap.entity_resolution._spatial import bounds_gap
from contextmap.entity_resolution.models import PolicyRef, reference_order
from contextmap.semantic_mapping import Entity, EntityReference, UnknownEntityError

CANDIDATE_RETRIEVAL_POLICY_ID = "entity-candidate-retrieval-v1"
"""Versioned identity of the baseline retrieval rules described in this module."""

# Uma entidade que cobriria mais células que isto (um piso ou uma parede muito maiores que o raio)
# não entra na grade: fica numa lista curta examinada em toda consulta. É só uma decisão de custo e
# nunca muda o resultado, porque todo candidato é confirmado pela regra exata.
_MAX_CELLS_PER_ENTITY = 512

# Folga da região de consulta, em metros: absorve o arredondamento de ponto flutuante nas bordas das
# células, de modo que um par com distância exatamente igual ao limiar nunca escape da grade.
_QUERY_SLACK_M = 1e-6

_Cell = tuple[int, int, int]


class RetrievalReason(Enum):
    """Why an entity was retrieved as a candidate; the definition order is the canonical order.

    Attributes:
        CENTROID_WITHIN_RADIUS: The centroids are within the policy's radius.
        BOUNDS_OVERLAP: The axis-aligned bounds overlap or touch.
        BOUNDS_WITHIN_MARGIN: The bounds do not touch but are within the policy's margin.
    """

    CENTROID_WITHIN_RADIUS = "centroid_within_radius"
    BOUNDS_OVERLAP = "bounds_overlap"
    BOUNDS_WITHIN_MARGIN = "bounds_within_margin"


class ExclusionReason(Enum):
    """Why an entity is not a candidate of another.

    Attributes:
        SAME_ENTITY: It is the entity itself.
        INCOMPATIBLE_MAP: Different geometric map or map frame; the coordinates are not comparable
            without an explicit alignment.
        OUTSIDE_SPATIAL_RANGE: Beyond the radius and the bounds margin.
        TEMPORAL_GAP_EXCEEDED: Spatially plausible, but seen further apart in time than the policy
            allows.
    """

    SAME_ENTITY = "same_entity"
    INCOMPATIBLE_MAP = "incompatible_map"
    OUTSIDE_SPATIAL_RANGE = "outside_spatial_range"
    TEMPORAL_GAP_EXCEEDED = "temporal_gap_exceeded"


@dataclass(frozen=True, kw_only=True)
class CandidateRetrievalPolicy:
    """Explicit configuration of the baseline retrieval.

    There are no defaults for the spatial thresholds: they are scientific choices, tied to the
    scale of the scene, that a profile declares.

    Attributes:
        centroid_radius_m: Centroid distance under which two entities are plausible, in meters.
        bounds_margin_m: Distance between bounds under which two entities are plausible, in meters;
            ``0`` keeps only overlapping or touching bounds.
        max_time_gap_ns: Largest gap between the observation intervals of two entities that keeps
            them candidates, in nanoseconds; ``None`` means time never excludes.
    """

    centroid_radius_m: float
    bounds_margin_m: float
    max_time_gap_ns: int | None = None

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ValueError: If the radius is not finite and positive, the margin is not finite and
                non-negative, or the time gap is negative.
        """
        if not (math.isfinite(self.centroid_radius_m) and self.centroid_radius_m > 0.0):
            raise ValueError(
                f"centroid_radius_m must be finite and positive, got {self.centroid_radius_m!r}"
            )
        require_non_negative(self, "bounds_margin_m")
        if self.max_time_gap_ns is not None and self.max_time_gap_ns < 0:
            raise ValueError(f"max_time_gap_ns must not be negative, got {self.max_time_gap_ns}")

    def fingerprint(self) -> str:
        """Hash the policy identity and thresholds, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "policy_id": CANDIDATE_RETRIEVAL_POLICY_ID,
                "centroid_radius_m": self.centroid_radius_m,
                "bounds_margin_m": self.bounds_margin_m,
                "max_time_gap_ns": self.max_time_gap_ns,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def ref(self) -> PolicyRef:
        """The policy identity and configuration, as recorded on a candidate set."""
        return PolicyRef(
            policy_id=CANDIDATE_RETRIEVAL_POLICY_ID, configuration_fingerprint=self.fingerprint()
        )


@dataclass(frozen=True, kw_only=True)
class CandidacyAssessment:
    """Whether one entity is a candidate of another, and exactly why.

    Attributes:
        exclusion: Why it is not a candidate; ``None`` when it is one.
        reasons: The spatial rules that hold, in canonical order; empty when it is outside the
            spatial range or its map is incompatible.
        centroid_distance_m: Distance between the centroids, in meters; ``None`` when the
            coordinates are not comparable.
        bounds_gap_m: Distance between the bounds, in meters; ``None`` when not comparable.
        time_gap_ns: Gap between the observation intervals, in nanoseconds, ``0`` when they
            overlap; ``None`` when the histories use different clock domains.
        detail: A deterministic, human-readable explanation, naming the values compared.
    """

    exclusion: ExclusionReason | None
    reasons: tuple[RetrievalReason, ...]
    centroid_distance_m: float | None
    bounds_gap_m: float | None
    time_gap_ns: int | None
    detail: str


def explain_candidacy(
    entity: Entity, other: Entity, policy: CandidateRetrievalPolicy
) -> CandidacyAssessment:
    """Decide, by the exact rule of the policy, whether ``other`` is a candidate of ``entity``.

    This is the single definition of candidacy: the spatial index only narrows down which pairs
    are worth asking, and every retrieved candidate is confirmed here. It is symmetric.

    Args:
        entity: The entity retrieval is asked for.
        other: The entity that may be a candidate.
        policy: The retrieval policy.

    Returns:
        The assessment, with the reasons when ``other`` is a candidate and the exclusion when it
        is not.
    """
    if entity.reference == other.reference:
        return _excluded(ExclusionReason.SAME_ENTITY, f"{entity.entity_id!r} is the entity itself")
    geometry, other_geometry = entity.geometry, other.geometry
    if (geometry.geometric_map_id, geometry.map_frame) != (
        other_geometry.geometric_map_id,
        other_geometry.map_frame,
    ):
        return _excluded(
            ExclusionReason.INCOMPATIBLE_MAP,
            f"geometric map {geometry.geometric_map_id!r} in frame {geometry.map_frame!r} and "
            f"geometric map {other_geometry.geometric_map_id!r} in frame "
            f"{other_geometry.map_frame!r} are not comparable without an alignment",
        )
    distance = math.dist(geometry.centroid_m, other_geometry.centroid_m)
    gap = bounds_gap(geometry.bounds, other_geometry.bounds)
    time_gap = _time_gap_ns(entity, other)
    reasons = _spatial_reasons(distance, gap, policy)
    measured = f"centroid distance {distance:.3f} m, bounds gap {gap:.3f} m"
    if not reasons:
        return CandidacyAssessment(
            exclusion=ExclusionReason.OUTSIDE_SPATIAL_RANGE,
            reasons=(),
            centroid_distance_m=distance,
            bounds_gap_m=gap,
            time_gap_ns=time_gap,
            detail=(
                f"{measured}: beyond the radius {policy.centroid_radius_m} m and the bounds "
                f"margin {policy.bounds_margin_m} m"
            ),
        )
    if (
        policy.max_time_gap_ns is not None
        and time_gap is not None
        and time_gap > policy.max_time_gap_ns
    ):
        return CandidacyAssessment(
            exclusion=ExclusionReason.TEMPORAL_GAP_EXCEEDED,
            reasons=reasons,
            centroid_distance_m=distance,
            bounds_gap_m=gap,
            time_gap_ns=time_gap,
            detail=(
                f"{measured}: spatially plausible, but seen {time_gap} ns apart, more than the "
                f"{policy.max_time_gap_ns} ns the policy allows"
            ),
        )
    return CandidacyAssessment(
        exclusion=None,
        reasons=reasons,
        centroid_distance_m=distance,
        bounds_gap_m=gap,
        time_gap_ns=time_gap,
        detail=measured,
    )


def _excluded(reason: ExclusionReason, detail: str) -> CandidacyAssessment:
    return CandidacyAssessment(
        exclusion=reason,
        reasons=(),
        centroid_distance_m=None,
        bounds_gap_m=None,
        time_gap_ns=None,
        detail=detail,
    )


def _spatial_reasons(
    distance_m: float, gap_m: float, policy: CandidateRetrievalPolicy
) -> tuple[RetrievalReason, ...]:
    reasons: list[RetrievalReason] = []
    if distance_m <= policy.centroid_radius_m:
        reasons.append(RetrievalReason.CENTROID_WITHIN_RADIUS)
    if gap_m == 0.0:
        reasons.append(RetrievalReason.BOUNDS_OVERLAP)
    elif gap_m <= policy.bounds_margin_m:
        reasons.append(RetrievalReason.BOUNDS_WITHIN_MARGIN)
    return tuple(reasons)


def _time_gap_ns(entity: Entity, other: Entity) -> int | None:
    state, other_state = entity.temporal_state, other.temporal_state
    if state.first_seen.clock_id != other_state.first_seen.clock_id:
        return None
    gap = max(
        other_state.first_seen.total_nanoseconds() - state.last_seen.total_nanoseconds(),
        state.first_seen.total_nanoseconds() - other_state.last_seen.total_nanoseconds(),
    )
    return max(0, gap)


@dataclass(frozen=True, kw_only=True)
class EntityCandidate:
    """A plausible comparison target of a source entity, with why it was retrieved.

    Attributes:
        entity_ref: The candidate entity.
        reasons: The spatial rules that hold, in canonical order and never empty.
        centroid_distance_m: Distance between the centroids, in meters.
        bounds_gap_m: Distance between the bounds, in meters; ``0`` when they overlap.
        time_gap_ns: Gap between the observation intervals, in nanoseconds; ``None`` when the
            histories use different clock domains.
    """

    entity_ref: EntityReference
    reasons: tuple[RetrievalReason, ...]
    centroid_distance_m: float
    bounds_gap_m: float
    time_gap_ns: int | None

    def __post_init__(self) -> None:
        """Validate that the candidate says why it was retrieved.

        Raises:
            ValueError: If there is no reason, the reasons are not in canonical order and unique,
                a distance is negative or not finite, or the time gap is negative.
        """
        if not self.reasons:
            raise ValueError("reasons must not be empty: a candidate must say why it was retrieved")
        order = list(RetrievalReason)
        indexes = [order.index(reason) for reason in self.reasons]
        if indexes != sorted(set(indexes)):
            raise ValueError("reasons must be in canonical order and unique")
        require_non_negative(self, "centroid_distance_m", "bounds_gap_m")
        if self.time_gap_ns is not None and self.time_gap_ns < 0:
            raise ValueError("time_gap_ns must not be negative")


@dataclass(frozen=True, kw_only=True)
class RetrievalDiagnostics:
    """How much work retrieval did for one source entity.

    Attributes:
        compatible_entity_count: The other entities over the same geometric map and frame: what an
            all-pairs comparison would have examined.
        examined_entity_count: The entities the exact rule was actually applied to.
    """

    compatible_entity_count: int
    examined_entity_count: int

    def __post_init__(self) -> None:
        """Validate that the counts are coherent.

        Raises:
            ValueError: If a count is negative or more entities were examined than exist.
        """
        if self.compatible_entity_count < 0 or self.examined_entity_count < 0:
            raise ValueError("counts must not be negative")
        if self.examined_entity_count > self.compatible_entity_count:
            raise ValueError("examined_entity_count cannot exceed compatible_entity_count")


@dataclass(frozen=True, kw_only=True)
class EntityCandidateSet:
    """The plausible comparison targets of one entity: retrieval, never a decision.

    A candidate being present does not imply a match, and one being absent means the policy
    excluded it, which :func:`explain_candidacy` can state.

    Attributes:
        source_entity_ref: The entity retrieval was asked for.
        candidates: The plausible targets, sorted by reference and unique, never including the
            source itself.
        policy: The retrieval policy and configuration that produced the set.
        diagnostics: How much work retrieval did.
    """

    source_entity_ref: EntityReference
    candidates: tuple[EntityCandidate, ...]
    policy: PolicyRef
    diagnostics: RetrievalDiagnostics

    def __post_init__(self) -> None:
        """Validate ordering, that the source is not its own candidate and the diagnostics agree.

        Raises:
            ValueError: If the candidates are not sorted by reference and unique, the source is
                among them, or there are more candidates than examined entities.
        """
        require_canonical(
            "candidates", self.candidates, lambda item: reference_order(item.entity_ref)
        )
        if any(item.entity_ref == self.source_entity_ref for item in self.candidates):
            raise ValueError("the source entity cannot be one of its own candidates")
        if len(self.candidates) > self.diagnostics.examined_entity_count:
            raise ValueError("there cannot be more candidates than examined entities")

    @property
    def candidate_entity_refs(self) -> tuple[EntityReference, ...]:
        """The references of the candidates, in canonical order."""
        return tuple(item.entity_ref for item in self.candidates)


class _CompatibleGroup:
    """The entities over one geometric map and frame, bucketed in a uniform grid of bounds."""

    def __init__(self, cell_size_m: float) -> None:
        self._cell_size_m = cell_size_m
        self._cells: dict[_Cell, list[Entity]] = defaultdict(list)
        self._large: list[Entity] = []
        self.size = 0

    def add(self, entity: Entity) -> None:
        self.size += 1
        low, high = self._cell_range(
            entity.geometry.bounds.minimum_m, entity.geometry.bounds.maximum_m
        )
        if _count(low, high) > _MAX_CELLS_PER_ENTITY:
            self._large.append(entity)
            return
        for cell in _cells(low, high):
            self._cells[cell].append(entity)

    def near(
        self, entity: Entity, policy: CandidateRetrievalPolicy
    ) -> dict[EntityReference, Entity]:
        """The entities that may satisfy a rule of the policy with ``entity``, itself excluded.

        The result is a superset of the candidates: an entity is returned when its bounds touch a
        cell of the query regions, which are the box of the radius around the centroid (a
        candidate by centroid has its centroid, hence its bounds, there) and the bounds of the
        entity grown by the margin.
        """
        geometry = entity.geometry
        radius = policy.centroid_radius_m + _QUERY_SLACK_M
        margin = policy.bounds_margin_m + _QUERY_SLACK_M
        regions = (
            self._cell_range(
                tuple(value - radius for value in geometry.centroid_m),
                tuple(value + radius for value in geometry.centroid_m),
            ),
            self._cell_range(
                tuple(value - margin for value in geometry.bounds.minimum_m),
                tuple(value + margin for value in geometry.bounds.maximum_m),
            ),
        )
        found: dict[EntityReference, Entity] = {item.reference: item for item in self._large}
        if sum(_count(low, high) for low, high in regions) > len(self._cells):
            # Região com mais células que as ocupadas: é mais barato varrer as ocupadas.
            for cell, members in self._cells.items():
                if any(_inside(cell, low, high) for low, high in regions):
                    found.update((item.reference, item) for item in members)
        else:
            for low, high in regions:
                for cell in _cells(low, high):
                    found.update((item.reference, item) for item in self._cells.get(cell, ()))
        found.pop(entity.reference, None)
        return found

    def _cell_range(
        self, minimum: Sequence[float], maximum: Sequence[float]
    ) -> tuple[_Cell, _Cell]:
        size = self._cell_size_m
        low = (
            math.floor(minimum[0] / size),
            math.floor(minimum[1] / size),
            math.floor(minimum[2] / size),
        )
        high = (
            math.floor(maximum[0] / size),
            math.floor(maximum[1] / size),
            math.floor(maximum[2] / size),
        )
        return low, high


def _count(low: _Cell, high: _Cell) -> int:
    return math.prod(top - bottom + 1 for bottom, top in zip(low, high, strict=True))


def _cells(low: _Cell, high: _Cell) -> Iterator[_Cell]:
    return (
        (x, y, z)
        for x in range(low[0], high[0] + 1)
        for y in range(low[1], high[1] + 1)
        for z in range(low[2], high[2] + 1)
    )


def _inside(cell: _Cell, low: _Cell, high: _Cell) -> bool:
    return all(bottom <= index <= top for index, bottom, top in zip(cell, low, high, strict=True))


class EntitySpatialIndex:
    """An in-memory grid over the entities' bounds that retrieves candidates without all-pairs.

    The index is derived from the entities it is given (their bounds and centroids, which Semantic
    Mapping also persists in ``entity-geometry-index.jsonl``), is not persisted and is rebuilt per
    run. Entities over different geometric maps or map frames live in separate groups and never
    see each other.
    """

    def __init__(self, entities: Iterable[Entity], policy: CandidateRetrievalPolicy) -> None:
        """Index the entities.

        Args:
            entities: The entities to retrieve among, in any order; may span several semantic
                maps that share geometric maps.
            policy: The retrieval policy; its scales set the cell size of the grid.

        Raises:
            ValueError: If two entities share a reference.
        """
        self._policy = policy
        cell_size_m = max(policy.centroid_radius_m, policy.bounds_margin_m)
        self._entities: dict[EntityReference, Entity] = {}
        self._groups: dict[tuple[str, str], _CompatibleGroup] = {}
        for entity in sorted(entities, key=lambda item: reference_order(item.reference)):
            if entity.reference in self._entities:
                raise ValueError(f"entity reference {entity.reference!r} is repeated")
            self._entities[entity.reference] = entity
            group = self._groups.setdefault(_group_key(entity), _CompatibleGroup(cell_size_m))
            group.add(entity)

    def references(self) -> tuple[EntityReference, ...]:
        """The references of every indexed entity, in canonical order."""
        return tuple(self._entities)

    def retrieve(self, reference: EntityReference) -> EntityCandidateSet:
        """Retrieve the candidates of one entity.

        Args:
            reference: An indexed entity.

        Returns:
            Its candidate set.

        Raises:
            UnknownEntityError: If no such entity was indexed.
        """
        try:
            entity = self._entities[reference]
        except KeyError:
            raise UnknownEntityError(reference) from None
        group = self._groups[_group_key(entity)]
        nearby = group.near(entity, self._policy)
        candidates: list[EntityCandidate] = []
        for other in nearby.values():
            assessment = explain_candidacy(entity, other, self._policy)
            distance, gap = assessment.centroid_distance_m, assessment.bounds_gap_m
            if assessment.exclusion is not None or distance is None or gap is None:
                continue
            candidates.append(
                EntityCandidate(
                    entity_ref=other.reference,
                    reasons=assessment.reasons,
                    centroid_distance_m=distance,
                    bounds_gap_m=gap,
                    time_gap_ns=assessment.time_gap_ns,
                )
            )
        candidates.sort(key=lambda item: reference_order(item.entity_ref))
        return EntityCandidateSet(
            source_entity_ref=reference,
            candidates=tuple(candidates),
            policy=self._policy.ref(),
            diagnostics=RetrievalDiagnostics(
                compatible_entity_count=group.size - 1, examined_entity_count=len(nearby)
            ),
        )


def _group_key(entity: Entity) -> tuple[str, str]:
    return (str(entity.geometry.geometric_map_id), str(entity.geometry.map_frame))


def retrieve_candidate_sets(
    entities: Iterable[Entity], policy: CandidateRetrievalPolicy
) -> tuple[EntityCandidateSet, ...]:
    """Retrieve the candidates of every entity, without an all-pairs comparison.

    Args:
        entities: The entities, in any order.
        policy: The retrieval policy.

    Returns:
        One candidate set per entity, in canonical reference order; the result does not depend on
        the order of ``entities``.

    Raises:
        ValueError: If two entities share a reference.
    """
    index = EntitySpatialIndex(entities, policy)
    return tuple(index.retrieve(reference) for reference in index.references())


def candidate_pairs(
    candidate_sets: Iterable[EntityCandidateSet],
) -> tuple[tuple[EntityReference, EntityReference], ...]:
    """The distinct pairs to compare, each once and in canonical order.

    Retrieval is symmetric, so a pair usually appears in the sets of both entities; the pair is
    listed once, with the entity whose reference sorts first first.

    Args:
        candidate_sets: The candidate sets.

    Returns:
        The pairs, sorted.
    """
    pairs = {
        _ordered_pair(item.source_entity_ref, other)
        for item in candidate_sets
        for other in item.candidate_entity_refs
    }
    return tuple(
        sorted(pairs, key=lambda pair: (reference_order(pair[0]), reference_order(pair[1])))
    )


def _ordered_pair(
    first: EntityReference, second: EntityReference
) -> tuple[EntityReference, EntityReference]:
    return (first, second) if reference_order(first) < reference_order(second) else (second, first)

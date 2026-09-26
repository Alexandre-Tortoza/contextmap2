"""Scale gates of Spatial Relations: the candidate sweep and the contact evaluation.

Contact evaluation, over a dense stack of crates on a floor (#601). Small fixtures hid the costs
this gate exists for: every contact candidate bucketed its object's points into a new grid, and the
points of every entity stayed resident until the evaluation ended. A dense stack is where both
hurt, because every crate is a candidate with each neighbor beside, above and below it, and the
floor with every crate that rests on it. The entity identities are shuffled with a fixed seed, as
resolved identities are digests that say nothing about where an entity is.

Candidate sweep, over a corridor that grows at a fixed local density (#614). Candidate generation is
only sub-quadratic because the sweep stops at the first box beyond reach along the axis on which
the scene is most spread out. Neither choice changes which pairs come out, so the differential
gate (``test_relation_candidate_pairs_differential.py``) cannot see either one regress: this gate
counts the reads of box faces the sweep makes instead. Those reads are its Python-level work; a copy
made in C, such as slicing the sweep order, is invisible to them.

CI keeps to counters and exact equivalence, against a baseline measured here: the same candidates
evaluated one at a time, which builds one grid per candidate, and the resident set the old cache
kept, which was every entity it had resolved. The wall-clock and memory figures measured on a
development machine are in ``src/contextmap/spatial_relations/docs/contact-predicates.md``.
"""

from __future__ import annotations

import random
import weakref
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest
from relation_builders import entity_ref
from relation_scene import LATTICE_POLICY, Scene

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.geometric_mapping import (
    Bounds3D,
    GeometryPoint,
    GeometryReference,
    GeometrySource,
)
from contextmap.semantic_mapping import EntityGeometry
from contextmap.shared import Vector3
from contextmap.spatial_relations import (
    CONTACT_PREDICATES,
    AxisDirection,
    CandidatePolicy,
    ContactPredicatePolicy,
    FrameConventions,
    RelationEvidence,
    RelationPredicate,
    contact_predicates,
    evaluate_contact_candidates,
    evaluate_contact_predicate,
    generate_relation_candidates,
)
from contextmap.spatial_relations.candidates import _neighbor_pairs

COLUMNS, ROWS, LEVELS = 6, 3, 2
CRATE_M = 0.4
SPACING_M = 0.1
CONVENTIONS = FrameConventions(
    map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
)
CONTACT = ContactPredicatePolicy(
    contact_distance_m=0.05,
    contact_tolerance_m=0.02,
    min_contact_points=3,
    support_height_tolerance_m=0.05,
    support_footprint_fraction=0.5,
    leaning_min_tilt_deg=10.0,
    leaning_max_tilt_deg=80.0,
    tilt_tolerance_deg=2.0,
    leaning_min_vertical_overlap_m=0.3,
)
CANDIDATES = CandidatePolicy(
    predicates=(
        RelationPredicate.TOUCHING,
        RelationPredicate.ON_TOP_OF,
        RelationPredicate.LEANING_AGAINST,
    ),
    proximity_radius_m=0.2,
    directional_radius_m=1.0,
)


def dense_stack() -> tuple[Scene, dict[ResolvedEntityReference, EntityGeometry]]:
    """A floor and ``COLUMNS x ROWS`` stacks of ``LEVELS`` crates, every crate touching its
    neighbors; the columns run along ``x``, the longest side of the scene."""
    scene = Scene()
    scene.add_lattice(
        "floor", (0.0, 0.0, -SPACING_M), (COLUMNS * CRATE_M, ROWS * CRATE_M, 0.0), SPACING_M
    )
    names = ["floor"]
    for column in range(COLUMNS):
        for row in range(ROWS):
            for level in range(LEVELS):
                name = f"crate-{column}-{row}-{level}"
                low = (column * CRATE_M, row * CRATE_M, level * CRATE_M)
                high = (low[0] + CRATE_M, low[1] + CRATE_M, low[2] + CRATE_M)
                scene.add_lattice(name, low, high, SPACING_M)
                names.append(name)
    numbers = list(range(1, len(names) + 1))
    random.Random(601).shuffle(numbers)
    entities = {
        entity_ref(number): scene.geometry(name, policy=LATTICE_POLICY)
        for number, name in zip(numbers, names, strict=True)
    }
    return scene, entities


class TrackedCloud(list[GeometryPoint]):
    """Resolved points whose lifetime the gate observes with a weak reference."""


class Probe:
    """Counts resolutions and grids, and how many clouds are resident at each measurement."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.grids: Counter[frozenset[str]] = Counter()
        self.resolutions: Counter[frozenset[str]] = Counter()
        self.resident: list[int] = []
        self._clouds: list[weakref.ref[TrackedCloud]] = []
        resolve = contact_predicates.resolve_geometry
        build = contact_predicates._contact_grid
        measure = contact_predicates._measure_contact

        def tracked_resolve(
            references: Sequence[GeometryReference], *, source: GeometrySource
        ) -> TrackedCloud:
            cloud = TrackedCloud(resolve(references, source=source))
            self.resolutions[_identity(cloud)] += 1
            self._clouds.append(weakref.ref(cloud))
            return cloud

        def counted_grid(points: Sequence[GeometryPoint], search_radius_m: float) -> Any:
            self.grids[_identity(points)] += 1
            return build(points, search_radius_m)

        def observed_measure(*args: Any) -> Any:
            self.resident.append(sum(ref() is not None for ref in self._clouds))
            return measure(*args)

        monkeypatch.setattr(contact_predicates, "resolve_geometry", tracked_resolve)
        monkeypatch.setattr(contact_predicates, "_contact_grid", counted_grid)
        monkeypatch.setattr(contact_predicates, "_measure_contact", observed_measure)


def _identity(points: Sequence[GeometryPoint]) -> frozenset[str]:
    return frozenset(point.geometry_id for point in points)


def one_at_a_time(
    scene: Scene, entities: dict[ResolvedEntityReference, EntityGeometry], candidates: Any
) -> list[RelationEvidence]:
    """The baseline: each contact candidate evaluated on its own, with nothing shared."""
    return [
        evaluate_contact_predicate(
            candidate.predicate,
            subject_entity_ref=candidate.subject_entity_ref,
            subject_geometry=entities[candidate.subject_entity_ref],
            object_entity_ref=candidate.object_entity_ref,
            object_geometry=entities[candidate.object_entity_ref],
            geometry_source=scene.source(),
            policy=CONTACT,
            conventions=CONVENTIONS,
        )
        for candidate in candidates.candidates
        if candidate.predicate in CONTACT_PREDICATES
    ]


def test_a_dense_stack_shares_each_grid_and_keeps_only_the_sweep_front_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene, entities = dense_stack()
    candidates = generate_relation_candidates(entities, policy=CANDIDATES, conventions=CONVENTIONS)
    contact = [item for item in candidates.candidates if item.predicate in CONTACT_PREDICATES]
    objects = {item.object_entity_ref for item in contact}
    named = objects | {item.subject_entity_ref for item in contact}
    baseline = Probe(monkeypatch)
    expected = one_at_a_time(scene, entities, candidates)
    monkeypatch.undo()
    probe = Probe(monkeypatch)

    evidence = evaluate_contact_candidates(
        candidates,
        entities=entities,
        geometry_source=scene.source(),
        policy=CONTACT,
        conventions=CONVENTIONS,
    )

    assert list(evidence) == expected
    assert len(named) == 1 + COLUMNS * ROWS * LEVELS
    # Uma grade por candidato no baseline; uma por entidade-objeto na avaliação do conjunto.
    assert sum(baseline.grids.values()) == len(contact) > 10 * len(objects)
    assert sum(probe.grids.values()) == len(probe.grids) == len(objects)
    assert sum(probe.resolutions.values()) == len(probe.resolutions) == len(named)
    # O cache antigo mantinha residente toda entidade já resolvida (todas, ao fim); a varredura
    # ao longo de x só mantém duas colunas de caixotes e o piso, que as sustenta todas.
    assert max(probe.resident) == 2 * ROWS * LEVELS + 1 < len(named)


# --- Candidate sweep: the work per entity follows the local density, not the map (#614) ---------


class CountingBounds:
    """The real bounds of one box, counting every read of its faces.

    A read is the unit of the sweep's work: visiting the next box along the sweep reads its lower
    face, and checking a pair on one axis reads both boxes. The count is deterministic, so the gate
    needs no clock.
    """

    def __init__(self, bounds: Bounds3D, reads: list[int]) -> None:
        self._bounds = bounds
        self._reads = reads

    @property
    def minimum_m(self) -> Vector3:
        self._reads[0] += 1
        return self._bounds.minimum_m

    @property
    def maximum_m(self) -> Vector3:
        self._reads[0] += 1
        return self._bounds.maximum_m


@dataclass(frozen=True)
class CountedGeometry:
    """An entity's geometry as the sweep reads it: only its bounds, counted."""

    bounds: CountingBounds


def corridor(count: int) -> list[EntityGeometry]:
    """Four boxes per meter along ``y``, the corridor's long axis, at a fixed local density.

    The corridor runs along ``y`` so that a sweep along a fixed axis (``x``) would find every box
    within reach of every other one: only the choice of the most spread axis keeps the front short.
    """
    rng = random.Random(f"corridor-{count}")
    scene = Scene()
    for index in range(count):
        center = (rng.uniform(0.0, 1.0), index // 4 + rng.uniform(0.0, 0.2), rng.uniform(0.0, 1.0))
        half = (rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3))
        scene.add_box(
            f"box-{index:04d}",
            (center[0] - half[0], center[1] - half[1], center[2] - half[2]),
            (center[0] + half[0], center[1] + half[1], center[2] + half[2]),
        )
    return [scene.geometry(f"box-{index:04d}") for index in range(count)]


def sweep_counting_reads(
    geometries: list[EntityGeometry], reach: float
) -> tuple[list[tuple[int, int]], int]:
    reads = [0]
    counted = [CountedGeometry(CountingBounds(item.bounds, reads)) for item in geometries]
    pairs = list(_neighbor_pairs(cast("list[EntityGeometry]", counted), reach))
    return pairs, reads[0]


CORRIDOR_REACH_M = 0.5
# Oito leituras por caixa escolhem o eixo e ordenam; depois, uma por vizinho visitado e quatro por
# eixo checado. Com ~2,5 caixas ao alcance à frente de cada uma, a medida é ~35 por caixa em
# qualquer tamanho; a força bruta faz (count - 1) / 2 visitas por caixa.
MAX_READS_PER_ENTITY = 60


@pytest.mark.parametrize("count", [50, 200, 800])
def test_growing_the_corridor_at_fixed_density_does_not_grow_the_sweep_work_per_entity(
    count: int,
) -> None:
    geometries = corridor(count)

    pairs, reads = sweep_counting_reads(geometries, CORRIDOR_REACH_M)

    # O contador é transparente: a varredura instrumentada devolve os mesmos pares.
    assert pairs == list(_neighbor_pairs(geometries, CORRIDOR_REACH_M))
    assert len(pairs) >= count
    assert reads <= MAX_READS_PER_ENTITY * count

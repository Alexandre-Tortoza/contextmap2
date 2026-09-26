"""A furnished storeroom whose annotated relations are partly lost by candidate generation.

The scene is dense enough that several ``(predicate, reason)`` groups of exclusions hold far more
than eight records, and it is annotated the way a person would: crates in a row are "next to"
each other even when they stand 0.7 m or 0.8 m apart (beyond the 0.6 m proximity reach), and a
picture hanging on the wall is "above" the sofa pushed against that wall, although their
footprints miss each other by a few centimeters. Those three relations never become candidates,
so the evaluator must explain each one with the exclusion that dropped it. The run is decided
with every channel, like :mod:`relation_run_fixture`, and uses the same policies.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable

import pytest
from relation_builders import entity_ref
from relation_run_fixture import CANDIDATES, CONTACT, CONVENTIONS, GEOMETRIC, Run
from relation_scene import LATTICE_POLICY, Scene

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.semantic_mapping import EntityGeometry
from contextmap.shared import Vector3
from contextmap.spatial_relations import (
    CandidateExclusion,
    RelationCandidateSet,
    candidates,
    decide_relations,
    evaluate_contact_candidates,
    evaluate_geometric_candidates,
    generate_relation_candidates,
)

SHELF_BOXES = 6
CRATE_X_M = (0.2, 1.0, 2.2, 3.0, 4.3)
"""Left faces of the floor crates, 0.5 m wide: gaps of 0.3, 0.7, 0.3 and 0.8 m."""


def _boxes() -> dict[str, tuple[Vector3, Vector3]]:
    boxes: dict[str, tuple[Vector3, Vector3]] = {
        "floor": ((0.0, 0.0, -0.1), (6.0, 4.0, 0.0)),
        "sofa": ((0.3, 0.1, 0.0), (1.9, 0.9, 0.8)),
        # O quadro fica na parede (y < 0,04) e o sofá encostado nela a partir de y = 0,1.
        "picture": ((0.5, 0.0, 1.2), (1.3, 0.04, 1.7)),
        "board-low": ((0.2, 3.4, 0.75), (3.4, 3.9, 0.8)),
        "board-high": ((0.2, 3.4, 1.5), (3.4, 3.9, 1.55)),
    }
    for level, bottom in (("low", 0.8), ("high", 1.55)):
        for index in range(SHELF_BOXES):
            x = 0.25 + 0.5 * index
            boxes[f"box-{level}-{index}"] = ((x, 3.5, bottom), (x + 0.35, 3.8, bottom + 0.3))
    for index, x in enumerate(CRATE_X_M):
        boxes[f"crate-{index}"] = ((x, 2.0, 0.0), (x + 0.5, 2.5, 0.5))
    return boxes


@dataclasses.dataclass
class Storeroom:
    """The decided run of the storeroom and the reference of each named object."""

    run: Run
    references: dict[str, ResolvedEntityReference]

    def candidate_set(self) -> RelationCandidateSet:
        """Generate the candidates again, under the exclusion ceiling in force now.

        The ceiling only changes which exclusions are listed, never the candidates, so the
        evidence and the decisions of :attr:`run` stay valid for the result.
        """
        return generate_relation_candidates(
            self.run.entities, policy=CANDIDATES, conventions=CONVENTIONS
        )


def multiset_digest(exclusions: Iterable[CandidateExclusion]) -> str:
    """The documented group digest, computed here on its own: the sum of the SHA-256 of each
    record's compact, key-sorted JSON, modulo 2**256."""
    total = 0
    for item in exclusions:
        record = {
            "subject_entity_ref": {
                "resolution_run_id": str(item.subject_entity_ref.resolution_run_id),
                "resolved_entity_id": str(item.subject_entity_ref.resolved_entity_id),
            },
            "predicate": item.predicate.value,
            "object_entity_ref": {
                "resolution_run_id": str(item.object_entity_ref.resolution_run_id),
                "resolved_entity_id": str(item.object_entity_ref.resolved_entity_id),
            },
            "reason": item.reason.value,
            "bounds_gap_m": item.bounds_gap_m,
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        total += int.from_bytes(hashlib.sha256(encoded).digest(), "big")
    return f"sha256-multiset:{total % 2**256:064x}"


def exclusion_ceiling(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    """Set, for one test, the most exclusions listed per ``(predicate, reason)`` group."""
    monkeypatch.setattr(candidates, "_MAX_LISTED_EXCLUSIONS", value)


def storeroom_scene() -> tuple[
    Scene, dict[str, ResolvedEntityReference], dict[ResolvedEntityReference, EntityGeometry]
]:
    """The storeroom's points, the reference of each named object and each entity's geometry."""
    scene = Scene()
    boxes = _boxes()
    for name, (low, high) in boxes.items():
        scene.add_lattice(name, low, high, 0.1)
    references = {name: entity_ref(number) for number, name in enumerate(boxes, start=1)}
    entities = {references[name]: scene.geometry(name, policy=LATTICE_POLICY) for name in boxes}
    return scene, references, entities


def build_storeroom() -> Storeroom:
    """Build, evaluate and decide the storeroom with the fixture policies."""
    scene, references, entities = storeroom_scene()
    generated = generate_relation_candidates(entities, policy=CANDIDATES, conventions=CONVENTIONS)
    evidence = [
        *evaluate_geometric_candidates(
            generated, entities=entities, policy=GEOMETRIC, conventions=CONVENTIONS
        ),
        *evaluate_contact_candidates(
            generated,
            entities=entities,
            geometry_source=scene.source(),
            policy=CONTACT,
            conventions=CONVENTIONS,
        ),
    ]
    run = Run(generated, evidence, decide_relations(generated, evidence), entities)
    return Storeroom(run=run, references=references)

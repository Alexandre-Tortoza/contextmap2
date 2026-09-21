"""Hard validity gates: what prevents two entities from being compared at all."""

from __future__ import annotations

from mapping_builders import make_entity, make_geometry

from contextmap.entity_resolution import evaluate_comparison_gates
from contextmap.geometric_mapping import MapId
from contextmap.semantic_mapping import Entity, SemanticMapId


def gates_by_id(first: Entity, second: Entity) -> dict[str, bool]:
    return {gate.gate_id: gate.passed for gate in evaluate_comparison_gates(first, second)}


def test_two_entities_of_one_map_and_one_frame_pass_every_gate() -> None:
    first = make_entity("entity--support-000001")
    second = make_entity("entity--support-000002")

    assert gates_by_id(first, second) == {
        "distinct-entities": True,
        "same-geometric-map": True,
        "same-map-frame": True,
    }


def test_an_entity_is_never_compared_with_itself() -> None:
    entity = make_entity("entity--support-000001")

    assert gates_by_id(entity, entity)["distinct-entities"] is False


def test_entities_over_different_geometric_maps_are_blocked_with_the_reason() -> None:
    first = make_entity("entity--support-000001")
    second = make_entity(
        "entity--support-000002",
        semantic_map_id=SemanticMapId("semantic-map-0002"),
        geometry=make_geometry(map_id=MapId("map-0002")),
    )

    results = {gate.gate_id: gate for gate in evaluate_comparison_gates(first, second)}

    assert results["same-geometric-map"].passed is False
    assert "map-0001" in results["same-geometric-map"].detail
    assert "map-0002" in results["same-geometric-map"].detail
    assert results["distinct-entities"].passed is True


def test_the_results_are_sorted_by_gate_so_they_encode_reproducibly() -> None:
    first = make_entity("entity--support-000001")
    second = make_entity("entity--support-000002")

    ids = [gate.gate_id for gate in evaluate_comparison_gates(first, second)]

    assert ids == sorted(ids)

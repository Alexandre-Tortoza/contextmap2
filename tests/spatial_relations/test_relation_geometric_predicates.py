"""Geometric predicates: deterministic evidence from entity bounds, never a relation by itself."""

from __future__ import annotations

import inspect
from itertools import product

import pytest
from relation_builders import entity_ref
from relation_scene import Scene

from contextmap.semantic_mapping import EntityGeometry, GeometrySummaryPolicy, geometry_set_digest
from contextmap.shared import Vector3
from contextmap.spatial_relations import (
    GEOMETRIC_PREDICATES,
    TAXONOMY_VERSION,
    AxisDirection,
    CandidatePolicy,
    EvidenceCaveatKind,
    FrameConventions,
    GeometricPredicatePolicy,
    IncompatibleFrameError,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceStatus,
    RelationPredicate,
    UndeclaredAxisError,
    evaluate_geometric_candidates,
    evaluate_geometric_predicate,
    evidence_id_for,
    generate_relation_candidates,
)

Box = tuple[Vector3, Vector3]
SUPPORTS = RelationEvidenceStatus.SUPPORTS
CONFLICTS = RelationEvidenceStatus.CONFLICTS
AMBIGUOUS = RelationEvidenceStatus.AMBIGUOUS

CONVENTIONS = FrameConventions(
    map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
)
POLICY = GeometricPredicatePolicy(
    boundary_tolerance_m=0.02,
    next_to_max_gap_m=0.5,
    adjacent_penetration_m=0.05,
    containment_slack_m=0.05,
    directional_overlap_fraction=0.5,
)
FLOOR: Box = ((0.0, 0.0, 0.0), (4.0, 4.0, 0.1))
UNIT: Box = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def _shift(box: Box, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Box:
    (x0, y0, z0), (x1, y1, z1) = box
    return ((x0 + dx, y0 + dy, z0 + dz), (x1 + dx, y1 + dy, z1 + dz))


def _geometries(subject: Box, obj: Box) -> tuple[EntityGeometry, EntityGeometry]:
    scene = Scene()
    scene.add_box("subject", *subject)
    scene.add_box("object", *obj)
    return scene.geometry("subject"), scene.geometry("object")


def _evaluate(
    predicate: RelationPredicate,
    subject: Box,
    obj: Box,
    *,
    conventions: FrameConventions = CONVENTIONS,
    policy: GeometricPredicatePolicy = POLICY,
) -> RelationEvidence:
    subject_geometry, object_geometry = _geometries(subject, obj)
    return evaluate_geometric_predicate(
        predicate,
        subject_entity_ref=entity_ref(1),
        subject_geometry=subject_geometry,
        object_entity_ref=entity_ref(2),
        object_geometry=object_geometry,
        policy=policy,
        conventions=conventions,
    )


def _measured(evidence: RelationEvidence, name: str) -> float:
    return {item.name: item.value for item in evidence.measurements}[name]


def _caveat_kinds(evidence: RelationEvidence) -> set[EvidenceCaveatKind]:
    return {item.kind for item in evidence.caveats}


# --- NEXT_TO ---


def test_next_to_is_supported_across_a_small_gap_and_records_the_exact_gap() -> None:
    evidence = _evaluate(RelationPredicate.NEXT_TO, UNIT, _shift(UNIT, dx=1.3))
    assert evidence.status is SUPPORTS
    assert _measured(evidence, "bounds_gap") == pytest.approx(0.3)
    assert _measured(evidence, "min_axis_overlap") == pytest.approx(-0.3)


def test_next_to_is_supported_for_boxes_that_touch() -> None:
    evidence = _evaluate(RelationPredicate.NEXT_TO, UNIT, _shift(UNIT, dx=1.0))
    assert evidence.status is SUPPORTS
    assert _measured(evidence, "bounds_gap") == 0.0


def test_next_to_is_contradicted_by_a_large_gap_and_by_deep_interpenetration() -> None:
    assert _evaluate(RelationPredicate.NEXT_TO, UNIT, _shift(UNIT, dx=3.0)).status is CONFLICTS
    assert _evaluate(RelationPredicate.NEXT_TO, UNIT, _shift(UNIT, dx=0.4)).status is CONFLICTS


def test_next_to_is_ambiguous_within_the_tolerance_of_its_threshold() -> None:
    evidence = _evaluate(RelationPredicate.NEXT_TO, UNIT, _shift(UNIT, dx=1.51))
    assert evidence.status is AMBIGUOUS
    assert _caveat_kinds(evidence) == {EvidenceCaveatKind.WITHIN_TOLERANCE}
    assert "bounds_gap" in evidence.caveats[0].detail


def test_next_to_states_its_thresholds() -> None:
    evidence = _evaluate(RelationPredicate.NEXT_TO, UNIT, _shift(UNIT, dx=1.3))
    thresholds = {item.name: item.value for item in evidence.thresholds}
    assert thresholds == {
        "adjacent_penetration": 0.05,
        "boundary_tolerance": 0.02,
        "next_to_max_gap": 0.5,
    }


# --- INTERSECTS ---


def test_intersects_needs_interpenetration_along_every_axis() -> None:
    overlapping = _evaluate(RelationPredicate.INTERSECTS, UNIT, _shift(UNIT, dx=0.5, dy=0.5))
    assert overlapping.status is SUPPORTS
    assert _measured(overlapping, "min_axis_overlap") == pytest.approx(0.5)
    assert [_measured(overlapping, f"axis_overlap_{axis}") for axis in "xyz"] == [0.5, 0.5, 1.0]


def test_touching_faces_do_not_intersect_but_are_next_to() -> None:
    touching = _shift(UNIT, dx=1.0)
    assert _evaluate(RelationPredicate.INTERSECTS, UNIT, touching).status is CONFLICTS
    assert _evaluate(RelationPredicate.NEXT_TO, UNIT, touching).status is SUPPORTS


def test_separated_boxes_do_not_intersect() -> None:
    assert _evaluate(RelationPredicate.INTERSECTS, UNIT, _shift(UNIT, dx=2.0)).status is CONFLICTS


def test_next_to_and_intersects_split_at_the_adjacent_penetration_and_never_both_hold() -> None:
    for dx in (1.4, 1.0, 0.98, 0.96, 0.94, 0.9, 0.6, 0.0):
        subject, obj = UNIT, _shift(UNIT, dx=dx)
        next_to = _evaluate(RelationPredicate.NEXT_TO, subject, obj).status
        intersects = _evaluate(RelationPredicate.INTERSECTS, subject, obj).status
        assert not (next_to is SUPPORTS and intersects is SUPPORTS), dx
    ambiguous = _evaluate(RelationPredicate.INTERSECTS, UNIT, _shift(UNIT, dx=0.95))
    assert ambiguous.status is AMBIGUOUS


# --- INSIDE ---


def test_inside_is_supported_when_the_subject_bounds_lie_within_the_object_bounds() -> None:
    big: Box = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    small: Box = ((4.0, 4.0, 4.0), (5.0, 5.0, 5.0))
    evidence = _evaluate(RelationPredicate.INSIDE, small, big)
    assert evidence.status is SUPPORTS
    assert _measured(evidence, "containment_margin") == 4.0
    assert _evaluate(RelationPredicate.INSIDE, big, small).status is CONFLICTS


def test_inside_absorbs_a_protrusion_within_the_slack_and_flags_the_border() -> None:
    big: Box = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    slight: Box = ((4.0, 4.0, 4.0), (10.01, 5.0, 5.0))
    border: Box = ((4.0, 4.0, 4.0), (10.06, 5.0, 5.0))
    far: Box = ((4.0, 4.0, 4.0), (10.5, 5.0, 5.0))
    assert _evaluate(RelationPredicate.INSIDE, slight, big).status is SUPPORTS
    assert _evaluate(RelationPredicate.INSIDE, border, big).status is AMBIGUOUS
    assert _evaluate(RelationPredicate.INSIDE, far, big).status is CONFLICTS


# --- ABOVE / IN_FRONT_OF ---


CRATE_ON_FLOOR: Box = ((1.0, 1.0, 0.1), (2.0, 2.0, 1.1))


def test_a_crate_resting_on_the_floor_is_above_it_and_the_floor_is_not_above_the_crate() -> None:
    above = _evaluate(RelationPredicate.ABOVE, CRATE_ON_FLOOR, FLOOR)
    assert above.status is SUPPORTS
    assert _measured(above, "vertical_clearance") == pytest.approx(0.0)
    assert _evaluate(RelationPredicate.ABOVE, FLOOR, CRATE_ON_FLOOR).status is CONFLICTS


def test_a_hovering_subject_is_above_and_a_sunken_one_is_not() -> None:
    hovering = _evaluate(RelationPredicate.ABOVE, _shift(CRATE_ON_FLOOR, dz=0.5), FLOOR)
    sunken = _evaluate(RelationPredicate.ABOVE, _shift(CRATE_ON_FLOOR, dz=-0.3), FLOOR)
    assert hovering.status is SUPPORTS
    assert _measured(hovering, "vertical_clearance") == pytest.approx(0.5)
    assert sunken.status is CONFLICTS


def test_above_needs_overlapping_footprints_with_the_overlap_recorded_per_axis() -> None:
    beside = _evaluate(RelationPredicate.ABOVE, _shift(CRATE_ON_FLOOR, dx=5.0), FLOOR)
    assert beside.status is CONFLICTS
    on_edge = _evaluate(RelationPredicate.ABOVE, _shift(CRATE_ON_FLOOR, dx=2.7), FLOOR)
    assert on_edge.status is CONFLICTS
    assert _measured(on_edge, "footprint_overlap_x") == pytest.approx(0.3)
    assert _measured(on_edge, "footprint_overlap_fraction_x") == pytest.approx(0.3)
    required = {item.name: item.value for item in on_edge.thresholds}
    assert required["required_footprint_overlap_x"] == pytest.approx(0.5)
    hanging = _evaluate(RelationPredicate.ABOVE, _shift(CRATE_ON_FLOOR, dx=2.49), FLOOR)
    assert hanging.status is AMBIGUOUS
    assert _caveat_kinds(hanging) == {EvidenceCaveatKind.WITHIN_TOLERANCE}


def test_the_up_axis_comes_from_the_conventions_and_flipping_it_flips_the_answer() -> None:
    stacked = _evaluate(RelationPredicate.ABOVE, CRATE_ON_FLOOR, FLOOR)
    flipped = FrameConventions(
        map_frame="map", up_axis=AxisDirection.NEGATIVE_Z, forward_axis=AxisDirection.POSITIVE_X
    )
    assert stacked.status is SUPPORTS
    assert _evaluate(
        RelationPredicate.ABOVE, CRATE_ON_FLOOR, FLOOR, conventions=flipped
    ).status is (CONFLICTS)
    assert _evaluate(
        RelationPredicate.ABOVE, FLOOR, CRATE_ON_FLOOR, conventions=flipped
    ).status is (SUPPORTS)


def test_a_scene_stacked_along_y_is_above_only_when_y_is_declared_up() -> None:
    lower: Box = ((0.0, 0.0, 0.0), (4.0, 0.1, 4.0))
    upper: Box = ((1.0, 0.1, 1.0), (2.0, 1.1, 2.0))
    y_up = FrameConventions(
        map_frame="map", up_axis=AxisDirection.POSITIVE_Y, forward_axis=AxisDirection.POSITIVE_X
    )
    assert _evaluate(RelationPredicate.ABOVE, upper, lower, conventions=y_up).status is SUPPORTS
    assert _evaluate(RelationPredicate.ABOVE, upper, lower).status is CONFLICTS


def test_in_front_of_reads_the_declared_forward_axis() -> None:
    subject = _shift(UNIT, dx=1.5)
    ahead = _evaluate(RelationPredicate.IN_FRONT_OF, subject, UNIT)
    assert ahead.status is SUPPORTS
    assert _measured(ahead, "forward_clearance") == pytest.approx(0.5)
    reverse = FrameConventions(
        map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.NEGATIVE_X
    )
    assert _evaluate(RelationPredicate.IN_FRONT_OF, subject, UNIT, conventions=reverse).status is (
        CONFLICTS
    )
    lateral = _evaluate(RelationPredicate.IN_FRONT_OF, _shift(UNIT, dx=1.5, dy=3.0), UNIT)
    assert lateral.status is CONFLICTS


# --- frames, axes, vocabulary ---


def test_a_predicate_whose_axis_is_not_declared_fails_loudly() -> None:
    no_axes = FrameConventions(map_frame="map")
    with pytest.raises(UndeclaredAxisError, match="up"):
        _evaluate(RelationPredicate.ABOVE, CRATE_ON_FLOOR, FLOOR, conventions=no_axes)
    only_up = FrameConventions(map_frame="map", up_axis=AxisDirection.POSITIVE_Z)
    with pytest.raises(UndeclaredAxisError, match="forward"):
        _evaluate(RelationPredicate.IN_FRONT_OF, UNIT, UNIT, conventions=only_up)
    assert _evaluate(RelationPredicate.NEXT_TO, UNIT, _shift(UNIT, dx=1.3), conventions=no_axes)


def test_geometry_in_another_frame_fails_loudly() -> None:
    with pytest.raises(IncompatibleFrameError):
        _evaluate(
            RelationPredicate.NEXT_TO,
            UNIT,
            _shift(UNIT, dx=1.3),
            conventions=FrameConventions(map_frame="odom"),
        )


def test_the_geometric_channel_covers_exactly_the_evaluated_geometric_predicates() -> None:
    expected = {
        RelationPredicate.NEXT_TO,
        RelationPredicate.ABOVE,
        RelationPredicate.IN_FRONT_OF,
        RelationPredicate.INSIDE,
        RelationPredicate.INTERSECTS,
    }
    assert set(GEOMETRIC_PREDICATES) == expected


def test_derived_and_contact_predicates_are_not_evaluated_here() -> None:
    with pytest.raises(ValueError, match="ABOVE"):
        _evaluate(RelationPredicate.BELOW, FLOOR, CRATE_ON_FLOOR)
    with pytest.raises(ValueError, match="contact"):
        _evaluate(RelationPredicate.TOUCHING, UNIT, _shift(UNIT, dx=1.0))


def test_only_geometry_can_reach_a_predicate() -> None:
    parameters = inspect.signature(evaluate_geometric_predicate).parameters
    assert set(parameters) == {
        "predicate",
        "subject_entity_ref",
        "subject_geometry",
        "object_entity_ref",
        "object_geometry",
        "policy",
        "conventions",
    }


# --- uncertainty ---


def test_sparse_or_disconnected_support_is_never_decisive() -> None:
    scene = Scene()
    scene.add_points("sparse", [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)])
    scene.add_box("solid", *_shift(UNIT, dx=1.3))
    clusters = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)]
    clusters += [(3.0, 0.0, 0.0), (3.1, 0.0, 0.0), (3.0, 0.1, 0.0)]
    scene.add_points("split", clusters)
    connected = GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.5)
    for name in ("sparse", "split"):
        evidence = evaluate_geometric_predicate(
            RelationPredicate.NEXT_TO,
            subject_entity_ref=entity_ref(1),
            subject_geometry=scene.geometry(name, policy=connected),
            object_entity_ref=entity_ref(2),
            object_geometry=scene.geometry("solid"),
            policy=POLICY,
            conventions=CONVENTIONS,
        )
        assert evidence.status is AMBIGUOUS
        assert EvidenceCaveatKind.UNRELIABLE_GEOMETRY in _caveat_kinds(evidence)
        assert evidence.measurements


def test_a_flat_support_cannot_state_an_interpenetration_depth() -> None:
    plane: Box = ((0.0, 0.0, 0.5), (1.0, 1.0, 0.5))
    for predicate in (RelationPredicate.NEXT_TO, RelationPredicate.INTERSECTS):
        evidence = _evaluate(predicate, plane, _shift(UNIT, dx=1.3))
        assert evidence.status is AMBIGUOUS
        assert EvidenceCaveatKind.DEGENERATE_GEOMETRY in _caveat_kinds(evidence)
    on_plane = _evaluate(RelationPredicate.ABOVE, _shift(UNIT, dz=0.5), plane)
    assert on_plane.status is SUPPORTS


# --- provenance ---


def test_evidence_records_the_geometry_the_rule_and_the_frame() -> None:
    subject_geometry, object_geometry = _geometries(CRATE_ON_FLOOR, FLOOR)
    evidence = _evaluate(RelationPredicate.ABOVE, CRATE_ON_FLOOR, FLOOR)
    assert evidence.channel is RelationEvidenceChannel.GEOMETRY
    assert evidence.evidence_id == evidence_id_for(
        channel=RelationEvidenceChannel.GEOMETRY,
        subject_entity_ref=entity_ref(1),
        predicate=RelationPredicate.ABOVE,
        object_entity_ref=entity_ref(2),
    )
    roles = {item.role: item for item in evidence.geometry}
    assert roles["subject"].geometry_digest == geometry_set_digest(subject_geometry.geometry_refs)
    assert roles["object"].point_count == object_geometry.statistics.point_count
    provenance = evidence.provenance
    assert provenance.rule_id == "bounds-above-v1"
    assert provenance.configuration_fingerprint == POLICY.fingerprint()
    assert provenance.taxonomy_version == TAXONOMY_VERSION
    assert provenance.map_frame == "map"
    assert str(provenance.geometric_map_id) == "map-0001"
    assert provenance.frame_conventions_fingerprint == CONVENTIONS.fingerprint()


def test_axis_free_predicates_do_not_claim_the_conventions() -> None:
    evidence = _evaluate(RelationPredicate.NEXT_TO, UNIT, _shift(UNIT, dx=1.3))
    assert evidence.provenance.rule_id == "bounds-next-to-v1"
    assert evidence.provenance.frame_conventions_fingerprint is None


def test_the_same_inputs_give_the_same_evidence() -> None:
    first = _evaluate(RelationPredicate.ABOVE, CRATE_ON_FLOOR, FLOOR)
    assert first == _evaluate(RelationPredicate.ABOVE, CRATE_ON_FLOOR, FLOOR)


# --- consistency ---


def test_symmetric_predicates_give_the_same_verdict_and_numbers_in_both_orders() -> None:
    for predicate in (RelationPredicate.NEXT_TO, RelationPredicate.INTERSECTS):
        for dx, dy in product((0.0, 0.5, 1.0, 1.3, 3.0), (0.0, 0.7)):
            other = _shift(UNIT, dx=dx, dy=dy)
            forward = _evaluate(predicate, UNIT, other)
            backward = _evaluate(predicate, other, UNIT)
            assert forward.status is backward.status, (predicate, dx, dy)
            assert forward.measurements == backward.measurements, (predicate, dx, dy)


def test_a_directed_predicate_is_never_supported_in_both_directions() -> None:
    offsets = [step * 0.25 for step in range(-8, 9)]
    for predicate in (
        RelationPredicate.ABOVE,
        RelationPredicate.IN_FRONT_OF,
        RelationPredicate.INSIDE,
    ):
        for dx, dz in product(offsets, offsets):
            if predicate is RelationPredicate.INSIDE and dx == 0.0 and dz == 0.0:
                # Caixas idênticas estão uma dentro da outra; a política de decisão (#146) reporta
                # esse par como estrutura inconsistente em vez de o avaliador decidir.
                continue
            other = _shift(UNIT, dx=dx, dz=dz)
            forward = _evaluate(predicate, UNIT, other)
            backward = _evaluate(predicate, other, UNIT)
            both = forward.status is SUPPORTS and backward.status is SUPPORTS
            assert not both, (predicate, dx, dz)


# --- policy ---


def test_the_policy_is_explicit_and_validated() -> None:
    valid = {
        "boundary_tolerance_m": 0.02,
        "next_to_max_gap_m": 0.5,
        "adjacent_penetration_m": 0.05,
        "containment_slack_m": 0.05,
        "directional_overlap_fraction": 0.5,
    }
    for name in ("boundary_tolerance_m", "next_to_max_gap_m"):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match=name):
                GeometricPredicatePolicy(**{**valid, name: bad})
    for name in ("adjacent_penetration_m", "containment_slack_m"):
        with pytest.raises(ValueError, match=name):
            GeometricPredicatePolicy(**{**valid, name: -0.1})
        GeometricPredicatePolicy(**{**valid, name: 0.0})
    for bad in (0.0, -0.5, 1.5, float("nan")):
        with pytest.raises(ValueError, match="directional_overlap_fraction"):
            GeometricPredicatePolicy(**{**valid, "directional_overlap_fraction": bad})


def test_the_policy_fingerprint_follows_every_threshold() -> None:
    changed = GeometricPredicatePolicy(
        boundary_tolerance_m=0.03,
        next_to_max_gap_m=0.5,
        adjacent_penetration_m=0.05,
        containment_slack_m=0.05,
        directional_overlap_fraction=0.5,
    )
    assert POLICY.fingerprint().startswith("sha256:")
    assert POLICY.fingerprint() != changed.fingerprint()


# --- from candidates ---


def test_candidates_are_evaluated_and_contact_predicates_are_left_to_their_channel() -> None:
    scene = Scene()
    scene.add_box("floor", *FLOOR)
    scene.add_box("crate", *CRATE_ON_FLOOR)
    entities = {entity_ref(1): scene.geometry("floor"), entity_ref(2): scene.geometry("crate")}
    candidates = generate_relation_candidates(
        entities,
        policy=CandidatePolicy(
            predicates=(
                RelationPredicate.ABOVE,
                RelationPredicate.NEXT_TO,
                RelationPredicate.TOUCHING,
            ),
            proximity_radius_m=0.6,
            directional_radius_m=2.0,
        ),
        conventions=CONVENTIONS,
    )
    evidence = evaluate_geometric_candidates(
        candidates, entities=entities, policy=POLICY, conventions=CONVENTIONS
    )
    keys = [(item.subject_entity_ref, item.predicate) for item in evidence]
    assert keys == [
        (entity_ref(1), RelationPredicate.NEXT_TO),
        (entity_ref(2), RelationPredicate.ABOVE),
    ]
    assert [item.status for item in evidence] == [SUPPORTS, SUPPORTS]


def test_candidates_generated_under_other_conventions_are_refused() -> None:
    scene = Scene()
    scene.add_box("a", *UNIT)
    scene.add_box("b", *_shift(UNIT, dx=1.3))
    entities = {entity_ref(1): scene.geometry("a"), entity_ref(2): scene.geometry("b")}
    candidates = generate_relation_candidates(
        entities,
        policy=CandidatePolicy(
            predicates=(RelationPredicate.NEXT_TO,),
            proximity_radius_m=0.6,
            directional_radius_m=2.0,
        ),
        conventions=FrameConventions(map_frame="map"),
    )
    with pytest.raises(ValueError, match="conventions"):
        evaluate_geometric_candidates(
            candidates, entities=entities, policy=POLICY, conventions=CONVENTIONS
        )

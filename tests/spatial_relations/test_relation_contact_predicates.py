"""Contact and support predicates: point-level evidence that a distance alone cannot give."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import weakref
from collections import Counter
from collections.abc import Sequence
from typing import Any

import pytest
from relation_builders import entity_ref
from relation_run_fixture import build_run
from relation_scene import CONNECTED_POLICY, LATTICE_POLICY, ORIENTED_LATTICE_POLICY, Scene

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.geometric_mapping import GeometryPoint, GeometryReference, GeometrySource, MapId
from contextmap.semantic_mapping import (
    EntityGeometry,
    GeometryResolutionError,
    GeometrySummaryPolicy,
    geometry_set_digest,
)
from contextmap.shared import Vector3
from contextmap.spatial_relations import (
    CONTACT_PREDICATES,
    TAXONOMY_VERSION,
    AxisDirection,
    CandidatePolicy,
    ContactPredicatePolicy,
    EvidenceCaveatKind,
    FrameConventions,
    GeometricPredicatePolicy,
    IncompatibleFrameError,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceStatus,
    RelationPredicate,
    UndeclaredAxisError,
    contact_predicates,
    encode_relation_evidence,
    evaluate_contact_candidates,
    evaluate_contact_predicate,
    evaluate_geometric_predicate,
    generate_relation_candidates,
)

Box = tuple[Vector3, Vector3]
SUPPORTS = RelationEvidenceStatus.SUPPORTS
CONFLICTS = RelationEvidenceStatus.CONFLICTS
AMBIGUOUS = RelationEvidenceStatus.AMBIGUOUS
UNAVAILABLE = RelationEvidenceStatus.UNAVAILABLE

CONVENTIONS = FrameConventions(
    map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
)
POLICY = ContactPredicatePolicy(
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
SPACING = 0.1
TABLE: Box = ((0.0, 0.0, 0.0), (2.0, 2.0, 0.1))
CRATE: Box = ((0.5, 0.5, 0.1), (1.0, 1.0, 0.6))
WALL: Box = ((0.0, -1.0, 0.0), (0.2, 1.0, 3.0))


def _shift(box: Box, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Box:
    (x0, y0, z0), (x1, y1, z1) = box
    return ((x0 + dx, y0 + dy, z0 + dz), (x1 + dx, y1 + dy, z1 + dz))


def _lattice_scene(subject: Box, obj: Box) -> Scene:
    scene = Scene()
    scene.add_lattice("subject", *subject, SPACING)
    scene.add_lattice("object", *obj, SPACING)
    return scene


def _evaluate_scene(
    predicate: RelationPredicate,
    scene: Scene,
    *,
    conventions: FrameConventions = CONVENTIONS,
    policy: ContactPredicatePolicy = POLICY,
    geometry_policy: GeometrySummaryPolicy = LATTICE_POLICY,
) -> RelationEvidence:
    return evaluate_contact_predicate(
        predicate,
        subject_entity_ref=entity_ref(1),
        subject_geometry=scene.geometry("subject", policy=geometry_policy),
        object_entity_ref=entity_ref(2),
        object_geometry=scene.geometry("object", policy=geometry_policy),
        geometry_source=scene.source(),
        policy=policy,
        conventions=conventions,
    )


def _evaluate(
    predicate: RelationPredicate,
    subject: Box,
    obj: Box,
    *,
    conventions: FrameConventions = CONVENTIONS,
) -> RelationEvidence:
    scene = _lattice_scene(subject, obj)
    return _evaluate_scene(predicate, scene, conventions=conventions)


def _measured(evidence: RelationEvidence, name: str) -> float:
    return {item.name: item.value for item in evidence.measurements}[name]


def _caveat_kinds(evidence: RelationEvidence) -> set[EvidenceCaveatKind]:
    return {item.kind for item in evidence.caveats}


def _plank(scene: Scene, name: str, *, tilt_deg: float, top: Vector3, length: float = 2.0) -> None:
    """A thin plank whose top end is at ``top``, tilted ``tilt_deg`` from vertical toward +x."""
    tilt = math.radians(tilt_deg)
    direction = (math.sin(tilt), 0.0, -math.cos(tilt))  # do topo para a base
    normal = (math.cos(tilt), 0.0, math.sin(tilt))
    points = []
    for step in range(int(length / 0.05) + 1):
        along = step * 0.05
        for width in (-0.1, 0.0, 0.1):
            for thickness in (-0.02, 0.02):
                points.append(
                    (
                        top[0] + direction[0] * along + normal[0] * thickness,
                        top[1] + width,
                        top[2] + direction[2] * along + normal[2] * thickness,
                    )
                )
    scene.add_points(name, points)


def _leaning_scene(tilt_deg: float, *, gap_m: float = 0.01, top_height: float = 1.8) -> Scene:
    """A wall face at ``x = 0.2`` and a plank whose top end is ``gap_m`` from it."""
    scene = Scene()
    scene.add_lattice("object", *WALL, SPACING)
    _plank(scene, "subject", tilt_deg=tilt_deg, top=(0.2 + gap_m, 0.0, top_height))
    return scene


# --- TOUCHING ---


def test_boxes_that_share_a_face_are_touching_with_the_contact_measured() -> None:
    evidence = _evaluate(RelationPredicate.TOUCHING, ((0, 0, 0), (1, 1, 1)), ((1, 0, 0), (2, 1, 1)))
    assert evidence.status is SUPPORTS
    assert evidence.channel is RelationEvidenceChannel.CONTACT
    assert _measured(evidence, "nearest_support_distance") == 0.0
    assert _measured(evidence, "contact_points_subject") == 121.0
    assert _measured(evidence, "contact_points_object") == 121.0
    assert _measured(evidence, "bounds_gap") == 0.0


def test_nearby_boxes_are_next_to_each_other_but_not_touching() -> None:
    subject: Box = ((0, 0, 0), (1, 1, 1))
    nearby: Box = ((1.2, 0, 0), (2.2, 1, 1))
    touching = _evaluate(RelationPredicate.TOUCHING, subject, nearby)
    assert touching.status is CONFLICTS
    assert _measured(touching, "pairs_within_search_radius") == 0.0
    assert "nearest_support_distance" not in {item.name for item in touching.measurements}
    assert _measured(touching, "bounds_gap") == pytest.approx(0.2)
    nearby_scene = _lattice_scene(subject, nearby)
    geometric = evaluate_geometric_predicate(
        RelationPredicate.NEXT_TO,
        subject_entity_ref=entity_ref(1),
        subject_geometry=nearby_scene.geometry("subject", policy=LATTICE_POLICY),
        object_entity_ref=entity_ref(2),
        object_geometry=nearby_scene.geometry("object", policy=LATTICE_POLICY),
        policy=GeometricPredicatePolicy(
            boundary_tolerance_m=0.02,
            next_to_max_gap_m=0.5,
            adjacent_penetration_m=0.05,
            containment_slack_m=0.05,
            directional_overlap_fraction=0.5,
        ),
        conventions=CONVENTIONS,
    )
    assert geometric.status is SUPPORTS


def test_a_gap_within_the_tolerance_of_the_contact_distance_is_ambiguous() -> None:
    evidence = _evaluate(
        RelationPredicate.TOUCHING, ((0, 0, 0), (1, 1, 1)), ((1.06, 0, 0), (2.06, 1, 1))
    )
    assert evidence.status is AMBIGUOUS
    assert _caveat_kinds(evidence) == {EvidenceCaveatKind.WITHIN_TOLERANCE}
    assert _measured(evidence, "nearest_support_distance") == pytest.approx(0.06)


def test_a_contact_of_too_few_points_is_ambiguous_because_no_area_is_measurable() -> None:
    scene = Scene()
    scene.add_points("subject", [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)])
    scene.add_points("object", [(1.01, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)])
    evidence = _evaluate_scene(RelationPredicate.TOUCHING, scene, geometry_policy=CONNECTED_POLICY)
    assert evidence.status is AMBIGUOUS
    assert _measured(evidence, "contact_points_subject") == 1.0
    assert EvidenceCaveatKind.UNRELIABLE_GEOMETRY in _caveat_kinds(evidence)
    assert any("contact area" in item.detail for item in evidence.caveats)


def test_touching_is_symmetric_in_verdict_and_swaps_the_contact_counts() -> None:
    small: Box = ((0, 0, 0), (1, 1, 1))
    tall: Box = ((1, 0, 0), (2, 1, 2))
    forward = _evaluate(RelationPredicate.TOUCHING, small, tall)
    backward = _evaluate(RelationPredicate.TOUCHING, tall, small)
    assert forward.status is backward.status is SUPPORTS
    assert _measured(forward, "contact_points_subject") == _measured(
        backward, "contact_points_object"
    )
    assert _measured(forward, "nearest_support_distance") == _measured(
        backward, "nearest_support_distance"
    )


def test_the_contact_subsets_and_thresholds_are_preserved() -> None:
    scene = _lattice_scene(((0, 0, 0), (1, 1, 1)), ((1, 0, 0), (2, 1, 1)))
    evidence = _evaluate_scene(RelationPredicate.TOUCHING, scene)
    roles = {item.role: item for item in evidence.geometry}
    assert set(roles) == {"object", "object_contact_points", "subject", "subject_contact_points"}
    contact = roles["subject_contact_points"]
    assert contact.point_count == 121
    assert len(contact.geometry_refs) == 121
    assert contact.geometry_digest == geometry_set_digest(contact.geometry_refs)
    subject_refs = set(scene.references("subject"))
    assert set(contact.geometry_refs) <= subject_refs
    thresholds = {item.name: item.value for item in evidence.thresholds}
    assert thresholds["contact_distance"] == 0.05
    assert thresholds["contact_tolerance"] == 0.02
    assert thresholds["min_contact_points"] == 3.0


def test_a_large_contact_keeps_only_the_digest_of_its_subset() -> None:
    evidence = _evaluate(
        RelationPredicate.TOUCHING, ((0, 0, 0), (4, 4, 0.1)), ((0, 0, 0.1), (4, 4, 0.2))
    )
    contact = {item.role: item for item in evidence.geometry}["subject_contact_points"]
    assert contact.point_count > 256
    assert contact.geometry_refs == ()


def test_sparse_or_disconnected_support_is_never_decisive_for_contact() -> None:
    scene = Scene()
    scene.add_points("subject", [(0.0, 0.0, 0.0), (0.02, 0.0, 0.0)])
    scene.add_lattice("object", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), SPACING)
    evidence = _evaluate_scene(RelationPredicate.TOUCHING, scene)
    assert evidence.status is AMBIGUOUS
    assert EvidenceCaveatKind.UNRELIABLE_GEOMETRY in _caveat_kinds(evidence)


# --- ON_TOP_OF ---


def test_a_crate_resting_on_a_table_is_on_top_of_it_and_the_table_is_not_on_the_crate() -> None:
    on_top = _evaluate(RelationPredicate.ON_TOP_OF, CRATE, TABLE)
    assert on_top.status is SUPPORTS
    assert _measured(on_top, "support_height_error") == pytest.approx(0.0)
    assert _measured(on_top, "support_footprint_overlap_x") == pytest.approx(0.5)
    assert _measured(on_top, "support_footprint_overlap_fraction_x") == pytest.approx(1.0)
    assert _measured(on_top, "nearest_support_distance") == pytest.approx(0.0, abs=1e-9)
    reverse = _evaluate(RelationPredicate.ON_TOP_OF, TABLE, CRATE)
    assert reverse.status is CONFLICTS


def test_a_crate_hovering_above_the_table_is_not_on_top_of_it() -> None:
    hovering = _evaluate(RelationPredicate.ON_TOP_OF, _shift(CRATE, dz=0.3), TABLE)
    assert hovering.status is CONFLICTS
    assert _measured(hovering, "pairs_within_search_radius") == 0.0


def test_a_crate_hanging_over_the_edge_is_not_supported_by_the_table() -> None:
    hanging = _evaluate(RelationPredicate.ON_TOP_OF, _shift(CRATE, dx=1.3), TABLE)
    assert hanging.status is CONFLICTS
    assert _measured(hanging, "support_footprint_overlap_fraction_x") == pytest.approx(0.4)
    half = _evaluate(RelationPredicate.ON_TOP_OF, _shift(CRATE, dx=1.25), TABLE)
    assert half.status is AMBIGUOUS
    assert _caveat_kinds(half) == {EvidenceCaveatKind.WITHIN_TOLERANCE}


def test_a_crate_sunk_into_the_table_is_not_on_top_of_it() -> None:
    sunk = _evaluate(RelationPredicate.ON_TOP_OF, _shift(CRATE, dz=-0.05), TABLE)
    assert _measured(sunk, "support_height_error") == pytest.approx(-0.05)
    deeply = _evaluate(RelationPredicate.ON_TOP_OF, _shift(CRATE, dz=-0.3), TABLE)
    assert deeply.status is CONFLICTS


def test_on_top_of_follows_the_declared_up_axis() -> None:
    flipped = FrameConventions(
        map_frame="map", up_axis=AxisDirection.NEGATIVE_Z, forward_axis=AxisDirection.POSITIVE_X
    )
    assert _evaluate(RelationPredicate.ON_TOP_OF, CRATE, TABLE, conventions=flipped).status is (
        CONFLICTS
    )
    assert _evaluate(RelationPredicate.ON_TOP_OF, TABLE, CRATE, conventions=flipped).status is (
        CONFLICTS
    )
    no_axes = FrameConventions(map_frame="map")
    with pytest.raises(UndeclaredAxisError, match="up"):
        _evaluate(RelationPredicate.ON_TOP_OF, CRATE, TABLE, conventions=no_axes)


# --- LEANING_AGAINST ---


def test_a_plank_tilted_against_a_wall_is_leaning_against_it() -> None:
    evidence = _evaluate_scene(
        RelationPredicate.LEANING_AGAINST,
        _leaning_scene(30.0),
        geometry_policy=ORIENTED_LATTICE_POLICY,
    )
    assert evidence.status is SUPPORTS
    assert _measured(evidence, "tilt_from_up") == pytest.approx(30.0, abs=1.0)
    assert _measured(evidence, "nearest_support_distance") <= 0.05
    assert _measured(evidence, "vertical_overlap") >= 1.0
    thresholds = {item.name: item.value for item in evidence.thresholds}
    assert thresholds["leaning_min_tilt"] == 10.0
    assert thresholds["leaning_max_tilt"] == 80.0


def test_an_upright_or_flat_plank_touching_a_wall_is_not_leaning() -> None:
    for tilt in (0.0, 90.0):
        evidence = _evaluate_scene(
            RelationPredicate.LEANING_AGAINST,
            _leaning_scene(tilt),
            geometry_policy=ORIENTED_LATTICE_POLICY,
        )
        assert evidence.status is CONFLICTS, tilt


def test_a_tilted_plank_that_does_not_touch_the_wall_is_not_leaning_on_it() -> None:
    far = _leaning_scene(30.0, gap_m=0.5)
    evidence = _evaluate_scene(
        RelationPredicate.LEANING_AGAINST, far, geometry_policy=ORIENTED_LATTICE_POLICY
    )
    assert evidence.status is CONFLICTS
    assert _measured(evidence, "pairs_within_search_radius") == 0.0


def test_a_tilt_within_the_tolerance_of_its_bound_is_ambiguous() -> None:
    evidence = _evaluate_scene(
        RelationPredicate.LEANING_AGAINST,
        _leaning_scene(11.0),
        geometry_policy=ORIENTED_LATTICE_POLICY,
    )
    assert evidence.status is AMBIGUOUS
    assert _caveat_kinds(evidence) == {EvidenceCaveatKind.WITHIN_TOLERANCE}


def test_without_an_orientation_a_touching_pair_cannot_be_called_leaning() -> None:
    evidence = _evaluate_scene(
        RelationPredicate.LEANING_AGAINST, _leaning_scene(30.0), geometry_policy=LATTICE_POLICY
    )
    assert evidence.status is UNAVAILABLE
    assert _caveat_kinds(evidence) == {EvidenceCaveatKind.MISSING_INPUT}
    assert "tilt_from_up" not in {item.name for item in evidence.measurements}
    assert _measured(evidence, "nearest_support_distance") <= 0.05


def test_without_contact_the_missing_orientation_does_not_matter() -> None:
    evidence = _evaluate_scene(
        RelationPredicate.LEANING_AGAINST,
        _leaning_scene(30.0, gap_m=0.5),
        geometry_policy=LATTICE_POLICY,
    )
    assert evidence.status is CONFLICTS


# --- frames, vocabulary, provenance ---


def test_contact_needs_the_points_of_the_geometric_map_the_entities_belong_to() -> None:
    scene = _lattice_scene(((0, 0, 0), (1, 1, 1)), ((1, 0, 0), (2, 1, 1)))
    foreign = Scene(map_id=MapId("map-0002"))
    foreign.add_lattice("subject", (0, 0, 0), (1, 1, 1), SPACING)
    with pytest.raises(GeometryResolutionError):
        evaluate_contact_predicate(
            RelationPredicate.TOUCHING,
            subject_entity_ref=entity_ref(1),
            subject_geometry=scene.geometry("subject", policy=LATTICE_POLICY),
            object_entity_ref=entity_ref(2),
            object_geometry=scene.geometry("object", policy=LATTICE_POLICY),
            geometry_source=foreign.source(),
            policy=POLICY,
            conventions=CONVENTIONS,
        )


def test_geometry_in_another_frame_fails_loudly() -> None:
    with pytest.raises(IncompatibleFrameError):
        _evaluate(
            RelationPredicate.TOUCHING,
            ((0, 0, 0), (1, 1, 1)),
            ((1, 0, 0), (2, 1, 1)),
            conventions=FrameConventions(map_frame="odom"),
        )


def test_touching_needs_no_declared_axis() -> None:
    evidence = _evaluate(
        RelationPredicate.TOUCHING,
        ((0, 0, 0), (1, 1, 1)),
        ((1, 0, 0), (2, 1, 1)),
        conventions=FrameConventions(map_frame="map"),
    )
    assert evidence.status is SUPPORTS
    assert evidence.provenance.frame_conventions_fingerprint is None


def test_only_contact_predicates_are_evaluated_here() -> None:
    assert set(CONTACT_PREDICATES) == {
        RelationPredicate.TOUCHING,
        RelationPredicate.ON_TOP_OF,
        RelationPredicate.LEANING_AGAINST,
    }
    scene = _lattice_scene(((0, 0, 0), (1, 1, 1)), ((1, 0, 0), (2, 1, 1)))
    with pytest.raises(ValueError, match="geometric"):
        _evaluate_scene(RelationPredicate.NEXT_TO, scene)
    with pytest.raises(ValueError, match="geometric"):
        _evaluate_scene(RelationPredicate.BELOW, scene)


def test_only_geometry_can_reach_a_contact_predicate() -> None:
    parameters = inspect.signature(evaluate_contact_predicate).parameters
    assert set(parameters) == {
        "predicate",
        "subject_entity_ref",
        "subject_geometry",
        "object_entity_ref",
        "object_geometry",
        "geometry_source",
        "policy",
        "conventions",
    }


def test_provenance_names_the_rule_the_policy_and_the_axes() -> None:
    evidence = _evaluate(RelationPredicate.ON_TOP_OF, CRATE, TABLE)
    provenance = evidence.provenance
    assert provenance.rule_id == "point-contact-on-top-of-v1"
    assert provenance.configuration_fingerprint == POLICY.fingerprint()
    assert provenance.taxonomy_version == TAXONOMY_VERSION
    assert provenance.frame_conventions_fingerprint == CONVENTIONS.fingerprint()
    assert str(provenance.geometric_map_id) == "map-0001"


def test_the_same_inputs_give_the_same_evidence() -> None:
    assert _evaluate(RelationPredicate.ON_TOP_OF, CRATE, TABLE) == _evaluate(
        RelationPredicate.ON_TOP_OF, CRATE, TABLE
    )


# --- policy ---


def _policy_kwargs(**changes: float) -> dict[str, float]:
    values: dict[str, float] = {
        "contact_distance_m": 0.05,
        "contact_tolerance_m": 0.02,
        "min_contact_points": 3,
        "support_height_tolerance_m": 0.05,
        "support_footprint_fraction": 0.5,
        "leaning_min_tilt_deg": 10.0,
        "leaning_max_tilt_deg": 80.0,
        "tilt_tolerance_deg": 2.0,
        "leaning_min_vertical_overlap_m": 0.3,
    }
    values.update(changes)
    return values


def test_the_policy_is_explicit_and_validated() -> None:
    ContactPredicatePolicy(**_policy_kwargs())  # type: ignore[arg-type]
    invalid = {
        "contact_distance_m": (0.0, -1.0, float("nan")),
        "contact_tolerance_m": (0.0, -0.1),
        "min_contact_points": (0, -1),
        "support_height_tolerance_m": (-0.1,),
        "support_footprint_fraction": (0.0, 1.5),
        "leaning_min_tilt_deg": (0.0, -5.0),
        "leaning_max_tilt_deg": (90.0, 100.0),
        "tilt_tolerance_deg": (0.0, -1.0),
        "leaning_min_vertical_overlap_m": (-0.1,),
    }
    for name, values in invalid.items():
        for value in values:
            with pytest.raises(ValueError, match=name):
                ContactPredicatePolicy(**_policy_kwargs(**{name: value}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="leaning_min_tilt_deg"):
        crossed = _policy_kwargs(leaning_min_tilt_deg=60.0, leaning_max_tilt_deg=50.0)
        ContactPredicatePolicy(**crossed)  # type: ignore[arg-type]


def test_the_policy_fingerprint_follows_every_threshold() -> None:
    changed = ContactPredicatePolicy(**_policy_kwargs(contact_distance_m=0.06))  # type: ignore[arg-type]
    assert POLICY.fingerprint().startswith("sha256:")
    assert POLICY.fingerprint() != changed.fingerprint()


# --- from candidates ---


def test_contact_candidates_are_evaluated_and_geometric_ones_are_left_alone() -> None:
    scene = _lattice_scene(CRATE, TABLE)
    entities: dict[ResolvedEntityReference, EntityGeometry] = {
        entity_ref(1): scene.geometry("subject", policy=LATTICE_POLICY),
        entity_ref(2): scene.geometry("object", policy=LATTICE_POLICY),
    }
    candidates = generate_relation_candidates(
        entities,
        policy=CandidatePolicy(
            predicates=(
                RelationPredicate.NEXT_TO,
                RelationPredicate.ON_TOP_OF,
                RelationPredicate.TOUCHING,
            ),
            proximity_radius_m=0.2,
            directional_radius_m=1.0,
        ),
        conventions=CONVENTIONS,
    )
    evidence = evaluate_contact_candidates(
        candidates,
        entities=entities,
        geometry_source=scene.source(),
        policy=POLICY,
        conventions=CONVENTIONS,
    )
    assert [(item.predicate, item.status) for item in evidence] == [
        (RelationPredicate.ON_TOP_OF, SUPPORTS),
        (RelationPredicate.TOUCHING, SUPPORTS),
    ]


# --- scale: equivalence of the candidate evaluation (#601) ---


RANDOM_CANDIDATES = CandidatePolicy(
    predicates=(
        RelationPredicate.NEXT_TO,
        RelationPredicate.TOUCHING,
        RelationPredicate.ON_TOP_OF,
        RelationPredicate.LEANING_AGAINST,
    ),
    proximity_radius_m=0.2,
    directional_radius_m=1.0,
)


def _jittered_box(
    rng: random.Random, low: Vector3, high: Vector3, *, spacing: float = 0.1
) -> list[Vector3]:
    """A lattice filling a box, each point moved up to 1 cm, so distances are not all on a grid."""
    axes = []
    for lower, upper in zip(low, high, strict=True):
        steps = max(1, round((upper - lower) / spacing))
        axes.append([lower + (upper - lower) * step / steps for step in range(steps + 1)])
    return [
        (x + rng.uniform(-0.01, 0.01), y + rng.uniform(-0.01, 0.01), z + rng.uniform(-0.01, 0.01))
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    ]


def random_contact_scene(
    seed: int,
) -> tuple[Scene, dict[ResolvedEntityReference, EntityGeometry]]:
    """A floor and seven crates dropped on it or on each other, some hovering, some sunk."""
    rng = random.Random(seed)
    scene = Scene()
    scene.add_points("e01", _jittered_box(rng, (0.0, 0.0, 0.0), (2.0, 2.0, 0.1)))
    tops = [0.1]
    for number in range(2, 9):
        sx, sy, sz = (rng.uniform(0.2, 0.5) for _ in range(3))
        x, y = rng.uniform(0.0, 2.0 - sx), rng.uniform(0.0, 2.0 - sy)
        z = rng.choice(tops) + rng.choice((-0.03, 0.0, 0.0, 0.04, 0.06))
        scene.add_points(f"e{number:02d}", _jittered_box(rng, (x, y, z), (x + sx, y + sy, z + sz)))
        tops.append(z + sz)
    entities = {
        entity_ref(number): scene.geometry(f"e{number:02d}", policy=ORIENTED_LATTICE_POLICY)
        for number in range(1, 9)
    }
    return scene, entities


def _evidence_digest(evidence: Sequence[RelationEvidence]) -> str:
    """Digest of every evidence record, in the order given."""
    records = [encode_relation_evidence(item) for item in evidence]
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def _random_contact_digest(seed: int) -> str:
    scene, entities = random_contact_scene(seed)
    candidates = generate_relation_candidates(
        entities, policy=RANDOM_CANDIDATES, conventions=CONVENTIONS
    )
    return _evidence_digest(
        evaluate_contact_candidates(
            candidates,
            entities=entities,
            geometry_source=scene.source(),
            policy=POLICY,
            conventions=CONVENTIONS,
        )
    )


def _fixture_contact_digest() -> str:
    run = build_run()
    return _evidence_digest(
        [item for item in run.evidence if item.channel is RelationEvidenceChannel.CONTACT]
    )


# #601 (SR-03): gravados antes de a grade de contato ser reaproveitada entre candidatos.
RECORDED_CONTACT_EVIDENCE = {
    "fixture": "628471baa2a29934fcab44f50ac5804b6e1ea119c5619c7c6a6cb59fe139cd24",
    "seed-0": "3b0ea1bbe4334f28249f74ba7355e1c419668d511edcca5c265d7cd2ac863f65",
    "seed-1": "9ee47de01789997659e53dae7194ea16ff780e34a67e9abe962178971fe16ba9",
    "seed-2": "b258bf0f0d640a603ecc78e2c5b8c7218d8c7abac8fa482c5761cd51f043a143",
    "seed-3": "e1298c0dd48476aac161720070b3fa05a4a217e91c35b57814ca6f5c2324b43d",
    "seed-4": "5e1720e156ee5e6a785caedc431ef8fbb9182d1fd81042dd8ffdd4900e469767",
}


@pytest.mark.parametrize("case", sorted(RECORDED_CONTACT_EVIDENCE))
def test_the_contact_evaluation_matches_the_recorded_evidence(case: str) -> None:
    digest = (
        _fixture_contact_digest()
        if case == "fixture"
        else _random_contact_digest(int(case.removeprefix("seed-")))
    )
    assert digest == RECORDED_CONTACT_EVIDENCE[case]


# --- scale: grids built once per object entity (#601) ---


GridKey = tuple[frozenset[str], float]


def _count_grids(monkeypatch: pytest.MonkeyPatch) -> Counter[GridKey]:
    """Count every contact grid built, by the points it buckets and the radius it uses."""
    built: Counter[GridKey] = Counter()
    build = contact_predicates._contact_grid

    def counting(points: Sequence[GeometryPoint], search_radius_m: float) -> object:
        built[(frozenset(point.geometry_id for point in points), search_radius_m)] += 1
        return build(points, search_radius_m)

    monkeypatch.setattr(contact_predicates, "_contact_grid", counting)
    return built


def _crates_on_a_table(count: int) -> tuple[Scene, dict[ResolvedEntityReference, EntityGeometry]]:
    """``count`` crates resting on one table, far enough apart to never touch each other."""
    scene = Scene()
    scene.add_lattice("table", *TABLE, SPACING)
    for index in range(count):
        x = 0.1 + 0.4 * index
        scene.add_lattice(f"crate{index}", (x, 0.5, 0.1), (x + 0.2, 0.7, 0.3), SPACING)
    entities = {entity_ref(1): scene.geometry("table", policy=LATTICE_POLICY)}
    for index in range(count):
        entities[entity_ref(index + 2)] = scene.geometry(f"crate{index}", policy=LATTICE_POLICY)
    return scene, entities


def test_a_shared_object_is_bucketed_once_however_many_candidates_name_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SR-03: a grade do objeto era reconstruída a cada candidato que o nomeia.
    crates = 5
    scene, entities = _crates_on_a_table(crates)
    candidates = generate_relation_candidates(
        entities,
        policy=CandidatePolicy(
            predicates=(RelationPredicate.ON_TOP_OF,),
            proximity_radius_m=0.2,
            directional_radius_m=1.0,
        ),
        conventions=CONVENTIONS,
    )
    built = _count_grids(monkeypatch)

    evidence = evaluate_contact_candidates(
        candidates,
        entities=entities,
        geometry_source=scene.source(),
        policy=POLICY,
        conventions=CONVENTIONS,
    )

    assert [(item.object_entity_ref, item.status) for item in evidence] == [
        (entity_ref(1), SUPPORTS)
    ] * crates
    table = frozenset(str(item.geometry_id) for item in scene.references("table"))
    assert built == {(table, POLICY.search_radius_m): 1}


class TrackedCloud(list[GeometryPoint]):
    """Resolved points whose lifetime a test can observe with a weak reference."""


class TrackedGrid(dict[tuple[int, int, int], Sequence[GeometryPoint]]):
    """A contact grid whose lifetime a test can observe with a weak reference."""


def _shelf(numbers: Sequence[int]) -> tuple[Scene, dict[ResolvedEntityReference, EntityGeometry]]:
    """Boxes in a row along ``x``, each touching the next, named by ``numbers`` in that order."""
    scene = Scene()
    for index in range(len(numbers)):
        scene.add_lattice(
            f"box{index}", (0.5 * index, 0.0, 0.0), (0.5 * index + 0.5, 0.5, 0.5), SPACING
        )
    entities = {
        entity_ref(number): scene.geometry(f"box{index}", policy=LATTICE_POLICY)
        for index, number in enumerate(numbers)
    }
    return scene, entities


def test_only_the_entities_around_the_sweep_front_stay_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SR-03: o cache de pontos nunca evictava, então todas as nuvens ficavam residentes. As
    # identidades embaralhadas mostram que o limite não depende de como as referências se ordenam.
    numbers = (5, 2, 8, 1, 7, 3, 6, 4)
    count = len(numbers)
    scene, entities = _shelf(numbers)
    candidates = generate_relation_candidates(
        entities,
        policy=CandidatePolicy(
            predicates=(RelationPredicate.TOUCHING,),
            proximity_radius_m=0.2,
            directional_radius_m=1.0,
        ),
        conventions=CONVENTIONS,
    )
    clouds: list[weakref.ref[TrackedCloud]] = []
    grids: list[weakref.ref[TrackedGrid]] = []
    resolve = contact_predicates.resolve_geometry
    build = contact_predicates._contact_grid
    measure = contact_predicates._measure_contact
    resident: list[tuple[int, int]] = []

    def tracked_resolve(
        references: Sequence[GeometryReference], *, source: GeometrySource
    ) -> TrackedCloud:
        cloud = TrackedCloud(resolve(references, source=source))
        clouds.append(weakref.ref(cloud))
        return cloud

    def tracked_grid(points: Sequence[GeometryPoint], search_radius_m: float) -> TrackedGrid:
        grid = TrackedGrid(build(points, search_radius_m))
        grids.append(weakref.ref(grid))
        return grid

    def observed_measure(*args: Any) -> Any:
        resident.append(
            (
                sum(ref() is not None for ref in clouds),
                sum(ref() is not None for ref in grids),
            )
        )
        return measure(*args)

    monkeypatch.setattr(contact_predicates, "resolve_geometry", tracked_resolve)
    monkeypatch.setattr(contact_predicates, "_contact_grid", tracked_grid)
    monkeypatch.setattr(contact_predicates, "_measure_contact", observed_measure)

    evidence = evaluate_contact_candidates(
        candidates,
        entities=entities,
        geometry_source=scene.source(),
        policy=POLICY,
        conventions=CONVENTIONS,
    )

    assert [item.status for item in evidence] == [SUPPORTS] * (count - 1)
    objects = {item.object_entity_ref for item in candidates.candidates}
    assert (len(clouds), len(grids)) == (count, len(objects))
    # A varredura segue a fileira: enquanto um par é medido, só as duas caixas dele estão
    # residentes (pontos e, se já foram objeto, grade), qualquer que seja a ordem das identidades.
    assert max(clouds for clouds, _ in resident) == 2
    assert max(grids for _, grids in resident) <= 2

"""Evidence collection and the policy execution service, over real entities."""

from __future__ import annotations

from typing import Any

import pytest
from resolution_entity_builders import entity_at, entity_over, lattice, scene_source
from resolution_feature_fakes import SPACE_A, InMemoryFeatureSource

from contextmap.entity_resolution import (
    AppearanceComparator,
    AppearanceComparisonPolicy,
    CandidateRetrievalPolicy,
    ComparisonChannels,
    ConservativeResolutionPolicy,
    EvidenceStatus,
    GeometryComparisonPolicy,
    MatchChannel,
    MatchEvidenceBuilder,
    PairResolution,
    RepresentationComparator,
    RepresentationComparisonPolicy,
    ResolutionOutcome,
    RunReaderRepresentationSource,
    SemanticCompatibilityPolicy,
    TemporalCompatibilityPolicy,
    UnavailableReason,
    decide,
    resolve_candidate_pairs,
    retrieve_candidate_sets,
)
from contextmap.geometric_mapping import MapId
from contextmap.semantic_mapping import Entity, SemanticMapId, UnknownEntityError

GEOMETRY = GeometryComparisonPolicy(
    min_shared_support_jaccard=0.5,
    min_bounds_iou=0.5,
    min_bounds_containment=0.9,
    min_conflict_gap_m=0.5,
    min_extent_ratio=0.3,
)
GEOMETRY_ONLY = ComparisonChannels(geometry=GEOMETRY)
POLICY = ConservativeResolutionPolicy(
    use_channels=(MatchChannel.GEOMETRY,), min_supporting_channels=1
)


def scene() -> dict[str, Entity]:
    """Four boxes on a line: a and b overlap, c is near, d is clearly apart."""
    centers = {"a": 0.0, "b": 0.1, "c": 0.9, "d": 1.5}
    points: list[tuple[float, float, float]] = []
    for center in centers.values():
        points += lattice((center, 0.0, 0.0), 0.6)
    source = scene_source(dict(enumerate(points)))
    return {
        name: entity_over(name, source, range(27 * index, 27 * (index + 1)))
        for index, name in enumerate(centers)
    }


# --- evidence collection ------------------------------------------------------------------------


def test_the_geometry_only_path_evaluates_nothing_else() -> None:
    entities = scene()

    evidence = MatchEvidenceBuilder(GEOMETRY_ONLY).build(entities["a"], entities["b"])

    assert evidence.geometry is not None and evidence.geometry.measurement is not None
    assert (evidence.semantic, evidence.appearance, evidence.temporal) == (None, None, None)
    assert evidence.point_representation is None
    assert not evidence.blocked
    assert [gate.passed for gate in evidence.gates] == [True, True, True]


def test_configured_channels_are_evaluated_and_the_pair_is_canonical() -> None:
    entities = scene()
    builder = MatchEvidenceBuilder(
        ComparisonChannels(
            geometry=GEOMETRY,
            semantic=SemanticCompatibilityPolicy(refinement_modifiers=()),
            temporal=TemporalCompatibilityPolicy(static_scene=False),
        ),
        code_version="test-version",
    )

    forward = builder.build(entities["a"], entities["b"])
    backward = builder.build(entities["b"], entities["a"])

    assert forward == backward
    assert {channel for channel, _ in forward.channels()} == {
        MatchChannel.GEOMETRY,
        MatchChannel.SEMANTIC,
        MatchChannel.TEMPORAL,
    }
    assert forward.provenance.code_version == "test-version"
    assert str(forward.entity_a_ref.entity_id) == "a"


def test_an_entity_is_never_compared_with_itself() -> None:
    entity = scene()["a"]

    with pytest.raises(ValueError, match="itself"):
        MatchEvidenceBuilder(GEOMETRY_ONLY).build(entity, entity)


def test_a_failed_gate_blocks_every_configured_channel_without_computing_any() -> None:
    entities = scene()
    elsewhere = entity_at(
        "z", (0, 0, 0), map_id=MapId("map-0002"), semantic_map_id=SemanticMapId("semantic-map-0002")
    )
    source = InMemoryFeatureSource({})
    builder = MatchEvidenceBuilder(
        ComparisonChannels(
            geometry=GEOMETRY,
            semantic=SemanticCompatibilityPolicy(refinement_modifiers=()),
            temporal=TemporalCompatibilityPolicy(static_scene=True),
            appearance=AppearanceComparator(
                source,
                AppearanceComparisonPolicy(
                    embedding_space_id=SPACE_A, min_supporting_similarity=0.8
                ),
            ),
            representation=RepresentationComparator(
                RunReaderRepresentationSource({}),
                RepresentationComparisonPolicy(
                    representation_space_id="sha256:space",
                    min_supporting_similarity=0.9,
                    max_conflicting_similarity=None,
                    min_defined_component_fraction=0.5,
                ),
            ),
        )
    )

    evidence = builder.build(entities["a"], elsewhere)

    assert evidence.blocked
    assert [gate.gate_id for gate in evidence.failed_gates()] == ["same-geometric-map"]
    assert evidence.unavailable_channels() == (
        MatchChannel.GEOMETRY,
        MatchChannel.SEMANTIC,
        MatchChannel.APPEARANCE,
        MatchChannel.TEMPORAL,
        MatchChannel.POINT_REPRESENTATION,
    )
    for _, item in evidence.channels():
        assert item.unavailable is not None
        assert item.unavailable.reason is UnavailableReason.BLOCKED_BY_GATE
        assert "map-0002" in item.unavailable.detail
    assert source.loads == []


def test_a_blocked_pair_only_records_the_channels_that_were_configured() -> None:
    elsewhere = entity_at(
        "z", (0, 0, 0), map_id=MapId("map-0002"), semantic_map_id=SemanticMapId("semantic-map-0002")
    )

    evidence = MatchEvidenceBuilder(GEOMETRY_ONLY).build(scene()["a"], elsewhere)

    assert evidence.blocked
    assert [channel for channel, _ in evidence.channels()] == [MatchChannel.GEOMETRY]
    assert decide(evidence, POLICY).decision is ResolutionOutcome.UNRESOLVED


def test_a_missing_optional_channel_is_unavailable_and_does_not_block() -> None:
    entities = scene()
    builder = MatchEvidenceBuilder(
        ComparisonChannels(
            geometry=GEOMETRY,
            appearance=AppearanceComparator(
                InMemoryFeatureSource({}),
                AppearanceComparisonPolicy(
                    embedding_space_id=SPACE_A, min_supporting_similarity=0.8
                ),
            ),
        )
    )

    evidence = builder.build(entities["a"], entities["b"])

    assert evidence.appearance is not None
    assert evidence.appearance.status is EvidenceStatus.UNAVAILABLE
    assert not evidence.blocked
    assert decide(evidence, POLICY).decision is ResolutionOutcome.MATCH


# --- the execution service ----------------------------------------------------------------------


def run(entities: dict[str, Entity]) -> tuple[PairResolution, ...]:
    sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=1.6, bounds_margin_m=0.2)
    )
    return resolve_candidate_pairs(
        entities.values(), sets, MatchEvidenceBuilder(GEOMETRY_ONLY), POLICY
    )


def outcomes(resolutions: tuple[PairResolution, ...]) -> dict[tuple[str, str], ResolutionOutcome]:
    return {
        (str(item.decision.entity_a_ref.entity_id), str(item.decision.entity_b_ref.entity_id)): (
            item.decision.decision
        )
        for item in resolutions
    }


def test_every_outcome_is_reached_from_real_geometry() -> None:
    result = outcomes(run(scene()))

    assert result[("a", "b")] is ResolutionOutcome.MATCH
    assert result[("a", "d")] is ResolutionOutcome.DISTINCT
    assert result[("b", "d")] is ResolutionOutcome.DISTINCT
    # Vizinhos próximos e sem sobreposição não são forçados a uma resposta binária.
    assert result[("a", "c")] is ResolutionOutcome.UNRESOLVED
    assert result[("b", "c")] is ResolutionOutcome.UNRESOLVED
    assert result[("c", "d")] is ResolutionOutcome.UNRESOLVED


def test_only_the_candidate_pairs_are_compared_each_once_in_canonical_order() -> None:
    entities = scene()
    sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.1)
    )

    resolutions = resolve_candidate_pairs(
        entities.values(), sets, MatchEvidenceBuilder(GEOMETRY_ONLY), POLICY
    )

    pairs = [
        (str(item.evidence.entity_a_ref.entity_id), str(item.evidence.entity_b_ref.entity_id))
        for item in resolutions
    ]
    assert pairs == sorted(set(pairs))
    assert ("a", "d") not in pairs  # 1,5 m de distância: fora do raio, não é candidato
    assert all(item.decision.evidence_ref == item.evidence.comparison_id for item in resolutions)


def test_the_run_is_deterministic_and_leaves_the_entities_untouched() -> None:
    entities = scene()
    before = {name: entity for name, entity in entities.items()}

    first, second = run(entities), run(entities)

    assert first == second
    assert entities == before


def test_a_candidate_set_that_names_an_unknown_entity_is_refused() -> None:
    entities = scene()
    sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=1.6, bounds_margin_m=0.2)
    )
    without_b = [entity for name, entity in entities.items() if name != "b"]

    with pytest.raises(UnknownEntityError):
        resolve_candidate_pairs(without_b, sets, MatchEvidenceBuilder(GEOMETRY_ONLY), POLICY)


def test_a_pair_resolution_must_hold_a_decision_of_its_own_evidence() -> None:
    entities = scene()
    one = MatchEvidenceBuilder(GEOMETRY_ONLY).build(entities["a"], entities["b"])
    other = MatchEvidenceBuilder(GEOMETRY_ONLY).build(entities["a"], entities["c"])

    with pytest.raises(ValueError, match="evidence"):
        PairResolution(evidence=one, decision=decide(other, POLICY))


def test_the_policy_can_be_swapped_without_changing_the_contracts() -> None:
    entities = scene()
    demanding: Any = ConservativeResolutionPolicy(
        use_channels=(MatchChannel.GEOMETRY, MatchChannel.SEMANTIC), min_supporting_channels=2
    )
    builder = MatchEvidenceBuilder(
        ComparisonChannels(
            geometry=GEOMETRY, semantic=SemanticCompatibilityPolicy(refinement_modifiers=())
        )
    )
    sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=1.6, bounds_margin_m=0.2)
    )

    relaxed = resolve_candidate_pairs(entities.values(), sets, builder, POLICY)
    strict = resolve_candidate_pairs(entities.values(), sets, builder, demanding)

    assert {type(item.decision) for item in relaxed + strict} == {type(relaxed[0].decision)}
    assert relaxed[0].decision.policy != strict[0].decision.policy
    # Sob o mesmo tipo de contrato, a política exigente decide diferente do mesmo par.
    assert [item.evidence for item in relaxed] == [item.evidence for item in strict]

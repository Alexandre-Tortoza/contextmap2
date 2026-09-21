import dataclasses
import json

import pytest
from mapping_builders import (
    FUSED_EVIDENCE_ID,
    FUSION_RUN_ID,
    FUSION_SUPPORT_ID,
    MAP_ID,
    SEMANTIC_MAP_ID,
    claim_signal,
    geometry_refs,
    make_entity,
    make_evidence_item,
    make_evidence_links,
    make_geometry,
    make_hypothesis,
    make_semantic_state,
    make_temporal_state,
    scorer_signal,
    timestamp,
)

from contextmap.geometric_mapping import MapId
from contextmap.semantic_fusion import EvidenceStance, FusedEvidenceId
from contextmap.semantic_mapping import (
    EmptyGeometrySupportError,
    Entity,
    EntityEvidenceLinks,
    EntityHypothesis,
    EntityId,
    EntityProvenance,
    EntityReference,
    EntitySemanticState,
    EntitySet,
    EntityTemporalState,
    ForeignEntityReferenceError,
    FusedEvidenceRef,
    SemanticMapId,
    UnknownEntityError,
)
from contextmap.semantic_mapping.serialization import (
    decode_entity,
    decode_entity_reference,
    encode_entity,
    encode_entity_reference,
)
from contextmap.visual_perception import HypothesisRole


def _rich_entity() -> Entity:
    """An entity that keeps alternatives, an abstention and both scored and unscored evidence."""
    pallet = make_hypothesis(
        "hypothesis-0001",
        "pallet",
        evidence=(
            make_evidence_item(
                claim="claim-0001", signals=(claim_signal(0.8), scorer_signal(None))
            ),
            make_evidence_item(
                contribution="contribution--support-000001--spatial-b",
                claim="claim-0002",
                stance=EvidenceStance.ABSTAINING,
                role=HypothesisRole.ALTERNATIVE,
                signals=(claim_signal(None),),
            ),
        ),
    )
    crate = make_hypothesis(
        "hypothesis-0002",
        "wooden crate",
        evidence=(
            make_evidence_item(
                contribution="contribution--support-000001--spatial-b",
                claim="claim-0003",
                role=HypothesisRole.ALTERNATIVE,
                signals=(claim_signal(0.4), scorer_signal(0.2)),
            ),
        ),
    )
    return make_entity(semantic_state=make_semantic_state((pallet, crate)))


class TestEntityContract:
    def test_it_references_geometry_semantics_evidence_and_time_without_copying_them(self) -> None:
        entity = _rich_entity()

        assert entity.geometry.geometry_refs == geometry_refs((0, 1, 2, 3))
        assert [item.label for item in entity.semantic_state.hypotheses] == [
            "pallet",
            "wooden crate",
        ]
        assert entity.evidence.fused_evidence[0].fused_evidence_id == FUSED_EVIDENCE_ID
        assert entity.temporal_state.physical_observation_count == 2
        # Nada de XYZ, embedding ou payload: só identidades.
        assert "coordinates_m" not in json.dumps(encode_entity(entity))

    def test_it_is_not_reduced_to_a_label_a_confidence_and_a_point(self) -> None:
        fields = {field.name for field in dataclasses.fields(Entity)}

        assert {"geometry", "semantic_state", "evidence", "temporal_state"} <= fields
        assert not {"label", "confidence", "xyz", "centroid"} & fields

    def test_it_requires_an_explicit_identity_and_semantic_map(self) -> None:
        with pytest.raises(ValueError, match="entity_id"):
            make_entity(" ")
        with pytest.raises(ValueError, match="semantic_map_id"):
            make_entity(semantic_map_id=SemanticMapId(""))

    def test_a_hypothesis_must_come_from_evidence_the_entity_links_to(self) -> None:
        state = make_semantic_state(
            (make_hypothesis(fused_evidence_id=FusedEvidenceId("fused--other")),)
        )

        with pytest.raises(ValueError, match="does not link to"):
            make_entity(semantic_state=state)

    def test_it_needs_no_entity_resolution_to_be_constructed(self) -> None:
        # Só existem os contratos: não há matching, merge ou re-identificação no construtor.
        entity = make_entity()

        assert entity.reference == EntityReference(
            semantic_map_id=SEMANTIC_MAP_ID, entity_id=EntityId("entity--support-000001")
        )

    def test_it_is_immutable(self) -> None:
        entity = make_entity()

        with pytest.raises(dataclasses.FrozenInstanceError):
            entity.entity_id = EntityId("other")  # type: ignore[misc]


class TestIdentityScope:
    def test_the_same_entity_id_in_two_maps_names_two_different_references(self) -> None:
        one = make_entity(semantic_map_id=SemanticMapId("semantic-map-a"))
        other = make_entity(semantic_map_id=SemanticMapId("semantic-map-b"))

        assert one.entity_id == other.entity_id
        assert one.reference != other.reference

    def test_a_reference_needs_both_identities(self) -> None:
        with pytest.raises(ValueError, match="semantic_map_id"):
            EntityReference(semantic_map_id=SemanticMapId(""), entity_id=EntityId("entity-1"))
        with pytest.raises(ValueError, match="entity_id"):
            EntityReference(semantic_map_id=SEMANTIC_MAP_ID, entity_id=EntityId(""))

    def test_a_reference_resolves_inside_its_own_semantic_map(self) -> None:
        first, second = make_entity("entity-0001"), make_entity("entity-0002")
        entities = EntitySet.of(SEMANTIC_MAP_ID, [second, first])

        assert entities.resolve(second.reference) is second
        assert [entity.entity_id for entity in entities.entities] == ["entity-0001", "entity-0002"]

    def test_a_reference_of_another_semantic_map_is_refused_even_with_a_known_id(self) -> None:
        entities = EntitySet.of(SEMANTIC_MAP_ID, [make_entity("entity-0001")])
        foreign = EntityReference(
            semantic_map_id=SemanticMapId("semantic-map-other"), entity_id=EntityId("entity-0001")
        )

        with pytest.raises(ForeignEntityReferenceError, match="semantic-map-other"):
            entities.resolve(foreign)

    def test_an_unknown_entity_is_an_explicit_error(self) -> None:
        entities = EntitySet.of(SEMANTIC_MAP_ID, [make_entity("entity-0001")])

        with pytest.raises(UnknownEntityError):
            entities.resolve(
                EntityReference(semantic_map_id=SEMANTIC_MAP_ID, entity_id=EntityId("entity-0009"))
            )

    def test_entity_ids_are_unique_inside_a_map(self) -> None:
        with pytest.raises(ValueError, match="sorted and unique"):
            EntitySet.of(SEMANTIC_MAP_ID, [make_entity("entity-0001"), make_entity("entity-0001")])

    def test_an_entity_of_another_map_cannot_join_the_set(self) -> None:
        foreign = make_entity("entity-0001", semantic_map_id=SemanticMapId("semantic-map-other"))

        with pytest.raises(ValueError, match="belongs to semantic map"):
            EntitySet.of(SEMANTIC_MAP_ID, [foreign])


class TestEntityGeometryAuthority:
    def test_empty_support_cannot_masquerade_as_a_valid_geometry(self) -> None:
        with pytest.raises(EmptyGeometrySupportError, match="must not be empty"):
            dataclasses.replace(make_geometry(), geometry_refs=())

    def test_support_must_be_sorted_and_unique(self) -> None:
        with pytest.raises(ValueError, match="sorted by geometry_id"):
            dataclasses.replace(make_geometry(), geometry_refs=geometry_refs((2, 1)))
        with pytest.raises(ValueError, match="sorted by geometry_id"):
            dataclasses.replace(make_geometry(), geometry_refs=geometry_refs((1, 1)))

    def test_support_must_come_from_one_map(self) -> None:
        mixed = geometry_refs((0,)) + geometry_refs((1,), map_id=MapId("map-0002"))

        with pytest.raises(ValueError, match="one map"):
            dataclasses.replace(make_geometry(), geometry_refs=mixed)

    def test_the_map_frame_is_explicit(self) -> None:
        with pytest.raises(ValueError, match="map_frame"):
            dataclasses.replace(make_geometry(), map_frame=" ")  # type: ignore[arg-type]

    def test_it_exposes_the_map_that_owns_the_support(self) -> None:
        assert make_geometry().geometric_map_id == MAP_ID


class TestEntitySemanticState:
    def test_a_hypothesis_needs_supporting_evidence(self) -> None:
        with pytest.raises(ValueError, match="supporting evidence"):
            make_hypothesis(evidence=(make_evidence_item(stance=EvidenceStance.AMBIGUOUS),))

    def test_a_label_is_not_repeated_inside_one_fused_evidence(self) -> None:
        first = make_hypothesis("hypothesis-0001", "pallet")
        again = make_hypothesis("hypothesis-0002", "pallet")

        with pytest.raises(ValueError, match="appears twice"):
            make_semantic_state((first, again))

    def test_hypotheses_are_canonically_ordered(self) -> None:
        first = make_hypothesis("hypothesis-0001", "pallet")
        second = make_hypothesis("hypothesis-0002", "crate")

        with pytest.raises(ValueError, match="sorted and unique"):
            make_semantic_state((second, first))

    def test_an_entity_may_have_no_hypothesis_when_no_view_proposed_one(self) -> None:
        assert make_entity(semantic_state=EntitySemanticState()).semantic_state.hypotheses == ()

    def test_evidence_items_are_canonically_ordered(self) -> None:
        later = make_evidence_item(contribution="contribution--support-000001--spatial-b")
        earlier = make_evidence_item(contribution="contribution--support-000001--spatial-a")

        with pytest.raises(ValueError, match="sorted and unique"):
            EntityHypothesis(
                fused_evidence_id=FUSED_EVIDENCE_ID,
                hypothesis_id="hypothesis-0001",  # type: ignore[arg-type]
                label="pallet",
                evidence=(later, earlier),
            )


class TestEntityEvidenceLinks:
    def test_an_entity_needs_evidence_behind_it(self) -> None:
        with pytest.raises(ValueError, match="fused_evidence must not be empty"):
            EntityEvidenceLinks(fused_evidence=())

    def test_references_are_canonical(self) -> None:
        later = FusedEvidenceRef(
            fusion_run_id=FUSION_RUN_ID,
            fused_evidence_id=FusedEvidenceId("fused--support-000002"),
            fusion_support_id=FUSION_SUPPORT_ID,
        )
        earlier = make_evidence_links().fused_evidence[0]

        with pytest.raises(ValueError, match="sorted and unique"):
            EntityEvidenceLinks(fused_evidence=(later, earlier))

    def test_a_reference_needs_every_identity(self) -> None:
        with pytest.raises(ValueError, match="fusion_run_id"):
            FusedEvidenceRef(
                fusion_run_id="",  # type: ignore[arg-type]
                fused_evidence_id=FUSED_EVIDENCE_ID,
                fusion_support_id=FUSION_SUPPORT_ID,
            )


class TestEntityTemporalState:
    def test_first_and_last_seen_are_ordered(self) -> None:
        with pytest.raises(ValueError, match="must not precede"):
            EntityTemporalState(
                first_seen=timestamp(12),
                last_seen=timestamp(10),
                physical_observation_count=1,
                inference_result_count=1,
            )

    def test_the_interval_stays_in_one_clock_domain(self) -> None:
        with pytest.raises(ValueError, match="clock domain"):
            EntityTemporalState(
                first_seen=timestamp(10),
                last_seen=timestamp(12, clock_id="other:clock"),
                physical_observation_count=1,
                inference_result_count=1,
            )

    def test_physical_observations_and_inference_results_are_counted_apart(self) -> None:
        state = make_temporal_state(physical=1, inference=3)

        assert (state.physical_observation_count, state.inference_result_count) == (1, 3)

    def test_inference_cannot_be_counted_below_the_frames_it_interpreted(self) -> None:
        with pytest.raises(ValueError, match="cannot be lower"):
            make_temporal_state(physical=3, inference=2)

    def test_an_entity_was_observed_at_least_once(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            make_temporal_state(physical=0, inference=0)


class TestProvenance:
    def test_both_policies_are_required(self) -> None:
        with pytest.raises(ValueError, match="identity_policy_id"):
            EntityProvenance(
                materialization_policy_id="one-support-one-entity-v1", identity_policy_id=""
            )


class TestSerialization:
    def test_an_entity_survives_a_json_round_trip_unchanged(self) -> None:
        entity = _rich_entity()

        record = json.loads(json.dumps(encode_entity(entity), allow_nan=False))

        assert decode_entity(record) == entity

    def test_unscored_and_low_scored_evidence_stay_distinct_and_abstentions_survive(self) -> None:
        decoded = decode_entity(json.loads(json.dumps(encode_entity(_rich_entity()))))
        pallet, crate = decoded.semantic_state.hypotheses

        scored, abstention = pallet.evidence
        assert [signal.value for signal in scored.signals] == [0.8, None]
        assert abstention.stance is EvidenceStance.ABSTAINING
        assert [signal.value for signal in abstention.signals] == [None]
        assert [signal.value for signal in crate.evidence[0].signals] == [0.4, 0.2]

    def test_geometry_is_stored_as_compact_positional_deltas(self) -> None:
        entity = make_entity(geometry=make_geometry(range(1000, 1200)))

        record = encode_entity(entity)["geometry"]

        assert record["deltas"][0] == 1000 and set(record["deltas"][1:]) == {1}
        assert decode_entity(encode_entity(entity)).geometry == entity.geometry

    def test_a_tampered_record_is_refused_rather_than_trusted(self) -> None:
        record = encode_entity(_rich_entity())
        record["temporal_state"]["last_seen"] = timestamp(1).to_record()

        with pytest.raises(ValueError, match="must not precede"):
            decode_entity(record)

    def test_a_record_that_breaks_geometry_canonical_order_is_refused(self) -> None:
        record = encode_entity(make_entity())
        record["geometry"]["deltas"] = [3, -1]

        with pytest.raises(ValueError, match="sorted by geometry_id"):
            decode_entity(record)

    def test_a_reference_round_trips(self) -> None:
        reference = make_entity().reference

        assert decode_entity_reference(encode_entity_reference(reference)) == reference

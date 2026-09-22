"""Coordinate, extent and capability semantics of the map metadata."""

from __future__ import annotations

import json
import math

import pytest
from context_map_builders import (
    anchored_map_frame,
    bounds,
    capabilities,
    context_map,
    estimator_local_anchor,
    external_anchor,
    map_frame,
    metadata,
    timestamp,
    window,
)

from contextmap.artifact import (
    AnchorKind,
    DeclaredCapabilities,
    Handedness,
    LengthUnit,
    MapAnchor,
    MapCapability,
    MapFrame,
    ObservationWindow,
    context_map_from_record,
    context_map_to_record,
)
from contextmap.spatial_relations import RelationPredicate

# --- frame -------------------------------------------------------------------------------------


def test_the_frame_states_everything_needed_to_read_an_xyz() -> None:
    frame = map_frame()

    assert frame.frame_id == "map"
    assert frame.unit is LengthUnit.METER
    assert frame.handedness is Handedness.RIGHT_HANDED
    assert frame.up_direction == (0.0, 0.0, 1.0)
    assert frame.anchor.kind is AnchorKind.ESTIMATOR_LOCAL


@pytest.mark.parametrize("blank", ["", "  "])
def test_the_frame_needs_an_identity(blank: str) -> None:
    with pytest.raises(ValueError, match="frame_id"):
        map_frame(frame_id=blank)


def test_the_up_direction_may_be_explicitly_unknown() -> None:
    # Um frame local de estimador não garante que z aponte para cima: desconhecido é explícito.
    assert map_frame(up_direction=None).up_direction is None


def test_the_up_direction_is_not_assumed_from_the_frame_name() -> None:
    frame = map_frame(frame_id="corridor-02", up_direction=None)

    assert frame.up_direction is None


@pytest.mark.parametrize(
    "direction",
    [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.6, 0.0, 0.8)],
)
def test_any_unit_direction_is_a_valid_up_direction(direction: tuple[float, float, float]) -> None:
    assert map_frame(up_direction=direction).up_direction == direction


@pytest.mark.parametrize(
    "direction",
    [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (1.0, 1.0, 0.0), (math.nan, 0.0, 1.0), (0.0, 0.0, math.inf)],
)
def test_an_up_direction_must_be_a_finite_unit_vector(
    direction: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="up_direction"):
        map_frame(up_direction=direction)


def test_handedness_is_explicit_and_never_inferred() -> None:
    assert map_frame(handedness=Handedness.LEFT_HANDED).handedness is Handedness.LEFT_HANDED


# --- anchor ------------------------------------------------------------------------------------


def test_an_estimator_local_origin_names_no_external_reference() -> None:
    anchor = estimator_local_anchor()

    assert anchor.kind is AnchorKind.ESTIMATOR_LOCAL
    assert anchor.reference_frame_id is None


def test_an_externally_anchored_origin_names_its_reference() -> None:
    anchor = external_anchor("site-a/enu")

    assert anchor.kind is AnchorKind.EXTERNALLY_ANCHORED
    assert anchor.reference_frame_id == "site-a/enu"


def test_a_local_origin_cannot_claim_an_external_reference() -> None:
    with pytest.raises(ValueError, match="reference_frame_id"):
        estimator_local_anchor(reference_frame_id="site-a/enu")


@pytest.mark.parametrize("reference", [None, "", "  "])
def test_an_external_origin_must_name_its_reference(reference: str | None) -> None:
    with pytest.raises(ValueError, match="reference_frame_id"):
        external_anchor(reference_frame_id=reference)


@pytest.mark.parametrize("blank", ["", "  "])
def test_an_origin_must_be_defined(blank: str) -> None:
    with pytest.raises(ValueError, match="origin_definition"):
        estimator_local_anchor(origin_definition=blank)


def test_two_local_frames_are_never_comparable_even_with_the_same_name() -> None:
    first = map_frame(frame_id="map")
    second = map_frame(frame_id="map")

    assert not first.is_comparable_with(second)


def test_frames_anchored_to_the_same_reference_are_comparable() -> None:
    first = anchored_map_frame("site-a/enu")
    second = anchored_map_frame("site-a/enu")

    assert first.is_comparable_with(second)
    assert second.is_comparable_with(first)


def test_frames_anchored_to_different_references_are_not_comparable() -> None:
    first = anchored_map_frame("site-a/enu")
    second = anchored_map_frame("site-b/enu")

    assert not first.is_comparable_with(second)


def test_a_local_frame_is_not_comparable_to_an_anchored_one() -> None:
    local = map_frame()
    anchored = anchored_map_frame()

    assert not local.is_comparable_with(anchored)
    assert not anchored.is_comparable_with(local)


def test_frames_with_different_handedness_are_not_comparable() -> None:
    right = anchored_map_frame()
    left = anchored_map_frame(handedness=Handedness.LEFT_HANDED)

    assert not right.is_comparable_with(left)


def test_an_externally_anchored_frame_id_must_equal_its_reference() -> None:
    # Reprodução do bug relatado na revisão: sem esta invariante, dois mapas podiam citar o
    # mesmo reference_frame_id mas ter bases locais (frame_id) diferentes, e is_comparable_with
    # ainda os declarava diretamente comparáveis, mesmo sem nenhum transform que os ligasse.
    with pytest.raises(ValueError, match="reference_frame_id"):
        map_frame(frame_id="map-a", anchor=external_anchor("site-a/enu"))


def test_two_maps_sharing_a_reference_but_not_the_same_alignment_are_refused() -> None:
    # A combinação que produzia o falso positivo (mesma reference_frame_id, frame_id
    # diferente, ou seja, bases locais potencialmente distintas) deixa de ser representável:
    # pelo menos um dos dois mapas falha na construção antes que is_comparable_with seja chamado.
    shared_reference = external_anchor("site-a/enu")

    valid = map_frame(frame_id="site-a/enu", anchor=shared_reference)
    assert valid.is_comparable_with(valid)

    with pytest.raises(ValueError, match="reference_frame_id"):
        map_frame(frame_id="warehouse-local-basis", anchor=shared_reference)


# --- extent ------------------------------------------------------------------------------------


def test_the_spatial_bounds_are_in_the_map_frame() -> None:
    result = metadata()

    assert result.bounds.frame_id == result.frame.frame_id


def test_bounds_in_another_frame_are_rejected() -> None:
    with pytest.raises(ValueError, match="frame"):
        metadata(bounds=bounds(frame_id="odom"))


def test_the_time_window_is_a_closed_interval_in_one_clock() -> None:
    assert window().start.total_nanoseconds() < window().end.total_nanoseconds()
    assert ObservationWindow(start=timestamp(5), end=timestamp(5)).start == timestamp(5)


def test_a_time_window_cannot_mix_clock_domains() -> None:
    with pytest.raises(ValueError, match="clock"):
        ObservationWindow(start=timestamp(1), end=timestamp(2, clock_id="other:clock"))


def test_a_time_window_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValueError, match="end"):
        ObservationWindow(start=timestamp(9), end=timestamp(8))


# --- capabilities ------------------------------------------------------------------------------


def test_a_map_with_only_geometry_declares_only_geometry() -> None:
    result = capabilities()

    assert result.content == (MapCapability.GEOMETRY,)
    assert result.relation_predicates == ()


def test_optional_content_is_declared_explicitly() -> None:
    declared = DeclaredCapabilities(
        content=(
            MapCapability.ENTITIES,
            MapCapability.GEOMETRY,
            MapCapability.POINT_REPRESENTATION_EVIDENCE,
            MapCapability.RELATIONS,
        ),
        relation_predicates=(RelationPredicate.INSIDE, RelationPredicate.ON_TOP_OF),
    )

    assert MapCapability.RELATIONS in declared.content
    assert declared.relation_predicates == (
        RelationPredicate.INSIDE,
        RelationPredicate.ON_TOP_OF,
    )


def test_geometry_is_always_declared() -> None:
    with pytest.raises(ValueError, match="geometry"):
        capabilities(content=(MapCapability.ENTITIES,))


def test_capabilities_are_sorted_and_unique() -> None:
    with pytest.raises(ValueError, match="sorted"):
        capabilities(content=(MapCapability.GEOMETRY, MapCapability.ENTITIES))
    with pytest.raises(ValueError, match="unique"):
        capabilities(content=(MapCapability.GEOMETRY, MapCapability.GEOMETRY))


def test_relations_cannot_be_declared_without_entities() -> None:
    with pytest.raises(ValueError, match="entities"):
        capabilities(
            content=(MapCapability.GEOMETRY, MapCapability.RELATIONS),
            relation_predicates=(RelationPredicate.ON_TOP_OF,),
        )


def test_relation_types_are_only_declared_with_the_relations_capability() -> None:
    with pytest.raises(ValueError, match="relation_predicates"):
        capabilities(relation_predicates=(RelationPredicate.ON_TOP_OF,))


def test_relation_predicates_are_sorted_and_unique() -> None:
    both = (MapCapability.ENTITIES, MapCapability.GEOMETRY, MapCapability.RELATIONS)
    with pytest.raises(ValueError, match="sorted"):
        capabilities(
            content=both,
            relation_predicates=(RelationPredicate.ON_TOP_OF, RelationPredicate.INSIDE),
        )
    with pytest.raises(ValueError, match="unique"):
        capabilities(
            content=both,
            relation_predicates=(RelationPredicate.ON_TOP_OF, RelationPredicate.ON_TOP_OF),
        )


def test_relations_may_be_declared_with_no_predicate_yet() -> None:
    # "Capacidade presente e vazia" difere de "capacidade ausente": o estágio rodou e nada achou.
    both = (MapCapability.ENTITIES, MapCapability.GEOMETRY, MapCapability.RELATIONS)

    assert capabilities(content=both, relation_predicates=()).relation_predicates == ()


# --- records -----------------------------------------------------------------------------------


def test_the_frame_and_extent_survive_a_json_round_trip() -> None:
    original = context_map(
        metadata=metadata(
            frame=anchored_map_frame(up_direction=None, handedness=Handedness.LEFT_HANDED),
            bounds=bounds(frame_id="site-a/enu"),
        )
    )

    record = json.loads(json.dumps(context_map_to_record(original), allow_nan=False))

    assert context_map_from_record(record) == original


def test_the_record_writes_enums_as_values_and_absent_optionals_as_null() -> None:
    record = context_map_to_record(
        context_map(metadata=metadata(frame=map_frame(up_direction=None)))
    )

    frame = record["metadata"]["frame"]
    assert frame["unit"] == "meter"
    assert frame["handedness"] == "right_handed"
    assert frame["up_direction"] is None
    assert frame["anchor"]["kind"] == "estimator_local"
    assert frame["anchor"]["reference_frame_id"] is None
    assert record["metadata"]["capabilities"]["content"] == ["geometry"]


def test_an_unknown_enum_value_in_a_record_is_rejected() -> None:
    record = context_map_to_record(context_map())
    record["metadata"]["frame"]["unit"] = "foot"

    with pytest.raises(ValueError, match="foot"):
        context_map_from_record(record)


def test_frame_metadata_that_became_ambiguous_is_rejected_on_decode() -> None:
    record = context_map_to_record(context_map())
    record["metadata"]["frame"]["anchor"]["reference_frame_id"] = "site-a/enu"

    with pytest.raises(ValueError, match="reference_frame_id"):
        context_map_from_record(record)


def test_the_frame_types_are_part_of_the_public_surface() -> None:
    assert MapFrame.__module__.startswith("contextmap.artifact")
    assert MapAnchor.__module__.startswith("contextmap.artifact")

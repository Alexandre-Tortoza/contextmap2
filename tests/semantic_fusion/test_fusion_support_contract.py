import dataclasses
import math
import typing

import pytest
from fusion_builders import (
    MAP_ID,
    geometry_refs,
    make_bounds,
    make_support,
    make_time_bounds,
    spatial_id,
)

from contextmap.geometric_mapping import MapId
from contextmap.semantic_fusion import FusionSupport, FusionSupportId, FusionSupportProvenance

SEMANTIC_OR_ENTITY_FIELDS = {
    "label",
    "hypothesis",
    "hypotheses",
    "category",
    "confidence",
    "entity_id",
    "object_id",
    "instance_id",
}


def test_a_support_states_where_it_is_and_which_observations_it_accumulates() -> None:
    support = make_support()

    assert support.geometric_map_id == MAP_ID
    assert support.geometry_support == geometry_refs(range(4))
    assert support.spatial_observation_ids == (
        spatial_id("run-a", "frame-0120"),
        spatial_id("run-a", "frame-0121"),
    )
    assert support.bounds == make_bounds()
    assert support.bounds.frame_id == "map"
    assert support.centroid_m == (1.0, 2.0, 0.5)
    assert support.time_bounds == make_time_bounds()
    assert support.provenance.support_policy_id == "overlap-support-v1"


def test_a_support_asserts_no_label_and_no_object_identity() -> None:
    names = {field.name for field in dataclasses.fields(FusionSupport)}

    assert not names & SEMANTIC_OR_ENTITY_FIELDS


def test_the_centroid_on_a_face_of_the_bounds_is_accepted() -> None:
    support = make_support(centroid_m=(2.0, 4.0, 1.0))

    assert support.centroid_m == (2.0, 4.0, 1.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"fusion_support_id": FusionSupportId("")}, "fusion_support_id"),
        ({"geometry_support": ()}, "geometry_support"),
        ({"spatial_observation_ids": ()}, "spatial_observation_ids"),
        ({"centroid_m": (3.0, 2.0, 0.5)}, "centroid"),
        ({"centroid_m": (math.nan, 2.0, 0.5)}, "centroid"),
    ],
)
def test_an_incomplete_or_inconsistent_support_is_rejected(
    overrides: dict[str, typing.Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_support(**overrides)


def test_geometry_from_another_map_is_rejected() -> None:
    foreign = geometry_refs([0], map_id=MapId("map-0002"))

    with pytest.raises(ValueError, match="only the support map"):
        make_support(geometry_support=geometry_refs([0, 1]) + foreign)


def test_geometry_that_is_not_sorted_and_unique_is_rejected() -> None:
    with pytest.raises(ValueError, match="sorted by geometry_id and unique"):
        make_support(geometry_support=geometry_refs([1, 0]))
    with pytest.raises(ValueError, match="sorted by geometry_id and unique"):
        make_support(geometry_support=geometry_refs([0, 0]))


def test_spatial_observations_that_are_not_sorted_and_unique_are_rejected() -> None:
    first, second = spatial_id("run-a", "frame-0120"), spatial_id("run-a", "frame-0121")

    with pytest.raises(ValueError, match="sorted and unique"):
        make_support(spatial_observation_ids=(second, first))
    with pytest.raises(ValueError, match="sorted and unique"):
        make_support(spatial_observation_ids=(first, first))


def test_the_support_policy_must_be_named() -> None:
    with pytest.raises(ValueError, match="support_policy_id"):
        FusionSupportProvenance(support_policy_id=" ")

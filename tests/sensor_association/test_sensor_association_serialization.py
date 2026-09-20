import dataclasses
import json
from typing import Any

import pytest
from association_builders import make_pose_ref, make_spatial_observation, refs

from contextmap.sensor_association import (
    PointCorrespondence,
    VisibilityState,
    VisualFeatureRef,
)
from contextmap.sensor_association.serialization import (
    decode_point_correspondence,
    decode_spatial_observation,
    encode_point_correspondence,
    encode_spatial_observation,
)
from contextmap.visual_perception import FeatureId, FeatureScope, RegionId


def _through_json(record: dict[str, Any]) -> dict[str, Any]:
    decoded: dict[str, Any] = json.loads(json.dumps(record))
    return decoded


def test_a_spatial_observation_round_trips_through_json() -> None:
    observation = make_spatial_observation(support=(3, 5, 9, 21))

    record = _through_json(encode_spatial_observation(observation))

    assert decode_spatial_observation(record) == observation
    assert record["region_id"] == "region-0001"
    assert record["geometry_support"] == [
        {"map_id": "map-0001", "geometry_id": ref.geometry_id} for ref in refs((3, 5, 9, 21))
    ]
    assert record["visibility"]["associated"] == 4


def test_a_direct_pose_reference_round_trips() -> None:
    direct = dataclasses.replace(
        make_spatial_observation(), pose_ref=make_pose_ref(interpolated=False)
    )

    assert decode_spatial_observation(_through_json(encode_spatial_observation(direct))) == direct


def test_region_scoped_feature_references_round_trip() -> None:
    scoped = VisualFeatureRef(
        feature_id=FeatureId("f-1"),
        embedding_space_id="alphaclip:vitl14",
        scope=FeatureScope.REGION,
        region_id=RegionId("region-0001"),
    )
    observation = make_spatial_observation(feature_refs=[scoped])

    decoded = decode_spatial_observation(_through_json(encode_spatial_observation(observation)))

    assert decoded.visual_feature_refs == (scoped,)


def test_decoding_revalidates_the_contract() -> None:
    record = _through_json(encode_spatial_observation(make_spatial_observation()))
    record["geometry_support"].reverse()

    with pytest.raises(ValueError, match="geometry_support"):
        decode_spatial_observation(record)


def test_a_point_correspondence_round_trips_including_absent_fields() -> None:
    associated = PointCorrespondence(
        geometry=refs((3,))[0],
        visibility=VisibilityState.ASSOCIATED,
        camera_range_m=4.2,
        raw_pixel=(320.5, 240.25),
        prepared_pixel=(160.25, 120.125),
        support_range_m=4.1,
    )
    behind = PointCorrespondence(
        geometry=refs((4,))[0],
        visibility=VisibilityState.BEHIND_CAMERA,
        camera_range_m=1.0,
        raw_pixel=None,
        prepared_pixel=None,
        support_range_m=None,
    )

    for record in (associated, behind):
        assert (
            decode_point_correspondence(_through_json(encode_point_correspondence(record)))
            == record
        )
    assert encode_point_correspondence(behind)["prepared_pixel"] is None

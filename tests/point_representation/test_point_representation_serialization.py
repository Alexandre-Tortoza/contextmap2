import json
from typing import Any

import pytest
from pointrep_builders import (
    knn_policy,
    make_representation,
    make_space,
    make_support,
    radius_policy,
)

from contextmap.point_representation import (
    CenteringMode,
    FailedSupport,
    FailureReason,
    ScaleNormalization,
    representation_space_fingerprint,
)
from contextmap.point_representation.serialization import (
    decode_failed_support,
    decode_point_representation,
    decode_point_support,
    decode_representation_space,
    encode_failed_support,
    encode_point_representation,
    encode_point_support,
    encode_representation_space,
)


def _through_json(record: dict[str, Any]) -> dict[str, Any]:
    decoded: dict[str, Any] = json.loads(json.dumps(record))
    return decoded


@pytest.mark.parametrize("policy", [radius_policy(0.75, max_neighbors=32), knn_policy(6)])
def test_a_space_round_trips_and_keeps_its_fingerprint(policy: object) -> None:
    space = make_space(policy=policy)  # type: ignore[arg-type]

    decoded = decode_representation_space(_through_json(encode_representation_space(space)))

    assert decoded == space
    assert representation_space_fingerprint(decoded) == representation_space_fingerprint(space)


def test_a_support_round_trips_with_every_reference_and_the_recorded_preparation() -> None:
    policy = radius_policy(
        0.5, centering=CenteringMode.CENTROID, scale=ScaleNormalization.SUPPORT_RADIUS
    )
    support = make_support(2, (2, 0, 5), policy=policy)

    record = _through_json(encode_point_support(support))

    assert decode_point_support(record) == support
    assert record["policy"]["preparation"] == {
        "centering": "centroid",
        "scale_normalization": "support_radius",
    }
    # O mapa é único por invariante: fica no registro uma vez, não em cada membro.
    assert record["map_id"] == str(support.center.map_id)
    assert record["center_geometry_id"] == str(support.center.geometry_id)
    assert record["geometry_ids"] == [str(ref.geometry_id) for ref in support.geometry_refs]


def test_a_representation_round_trips_including_undefined_components() -> None:
    representation = make_representation(undefined_components=(1, 3), payload_reference=None)

    record = _through_json(encode_point_representation(representation))

    assert decode_point_representation(record) == representation
    assert record["undefined_components"] == [1, 3]
    assert record["payload_reference"] is None
    assert record["encoder_identity"]["backend_id"] == "fake_encoder"


def test_decoding_revalidates_the_contracts() -> None:
    record = _through_json(encode_point_representation(make_representation()))
    record["geometry_reference"]["geometry_id"] = "map-0001--geom-000000099"

    with pytest.raises(ValueError, match="geometry_reference"):
        decode_point_representation(record)


def test_the_encoded_representation_never_embeds_the_numerical_payload() -> None:
    record = encode_point_representation(make_representation())

    assert not {"vector", "values", "payload", "data"} & set(record)
    assert record["payload_reference"] == "payloads/vectors.f32#0"


def test_a_failed_support_round_trips_with_its_reason_and_detail() -> None:
    failed = FailedSupport(
        support=make_support(2, (2, 0, 5)),
        reason=FailureReason.UNENCODABLE_SUPPORT,
        detail="fewer than 5 supporting points",
    )

    record = _through_json(encode_failed_support(failed))

    assert decode_failed_support(record) == failed
    assert record["reason"] == "unencodable_support"


def test_a_persisted_support_states_its_map_once() -> None:
    record = encode_point_support(make_support(2, (2, 0, 5, 7)))

    assert "map_id" not in record["policy"]
    assert not any(isinstance(item, dict) for item in record["geometry_ids"])

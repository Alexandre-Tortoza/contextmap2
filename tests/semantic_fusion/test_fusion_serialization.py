import copy
import dataclasses
import json
from typing import Any

import pytest
from fusion_builders import make_fused_evidence, make_support
from rich_fusion import make_rich

from contextmap.semantic_fusion import SupportSignalKind, UncertaintyKind
from contextmap.semantic_fusion.serialization import (
    decode_fused_evidence,
    decode_fusion_support,
    encode_fused_evidence,
    encode_fusion_support,
)


def _through_json(record: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(record, allow_nan=False, sort_keys=True)
    decoded: dict[str, Any] = json.loads(text)
    return decoded


def test_a_support_round_trips_through_json() -> None:
    support = make_support()

    assert decode_fusion_support(_through_json(encode_fusion_support(support))) == support


def test_a_rich_fused_evidence_round_trips_without_losing_anything() -> None:
    rich = make_rich()

    decoded = decode_fused_evidence(_through_json(encode_fused_evidence(rich.evidence)))

    assert decoded == rich.evidence
    assert decoded.weighting is not None
    assert {c.channel.value for c in decoded.channels} >= {
        "semantic_scores",
        "point_representation",
    }


def test_the_support_of_a_real_build_round_trips() -> None:
    rich = make_rich()

    assert decode_fusion_support(_through_json(encode_fusion_support(rich.support))) == rich.support


def test_unscored_and_zero_scores_stay_different_after_persistence() -> None:
    decoded = decode_fused_evidence(_through_json(encode_fused_evidence(make_rich().evidence)))

    values = {
        (item.contribution_id, item.claim_id): [
            s.value for s in item.signals if s.kind is SupportSignalKind.CLAIM_CONFIDENCE
        ]
        for hypothesis in decoded.hypotheses
        for item in hypothesis.evidence
    }
    assert [None] in values.values()
    assert [0.0] in values.values()
    assert [0.9] in values.values()


def test_ambiguity_conflict_and_abstention_survive_persistence() -> None:
    rich = make_rich()

    decoded = decode_fused_evidence(_through_json(encode_fused_evidence(rich.evidence)))

    assert {u.kind for u in decoded.uncertainty} == {
        UncertaintyKind.CONTRADICTION,
        UncertaintyKind.NEAR_TIE,
    }
    assert decoded.uncertainty == rich.evidence.uncertainty
    stances = {i.stance.value for h in decoded.hypotheses for i in h.evidence}
    assert stances == {"supporting", "conflicting", "ambiguous", "abstaining"}


def test_physical_observation_and_inference_counts_survive_persistence() -> None:
    rich = make_rich()

    decoded = decode_fused_evidence(_through_json(encode_fused_evidence(rich.evidence)))

    assert decoded.physical_observation_count == rich.evidence.physical_observation_count == 3
    assert decoded.inference_result_count == rich.evidence.inference_result_count == 4


def test_geometry_is_stored_compactly_as_positional_deltas() -> None:
    rich = make_rich(geometry=range(0, 600))

    record = encode_fused_evidence(rich.evidence)

    geometry = record["contributions"][0]["geometry"]
    assert geometry["deltas"][:3] == [0, 1, 1]
    assert len(geometry["deltas"]) == 600
    assert "geom-" not in json.dumps(geometry)


def test_the_record_is_pure_json_with_no_python_objects() -> None:
    record = encode_fused_evidence(make_rich().evidence)

    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_decoding_revalidates_the_contracts() -> None:
    record = _through_json(encode_fused_evidence(make_rich().evidence))
    tampered = copy.deepcopy(record)
    tampered["contributions"][0]["claim_ids"] = ["b-claim", "a-claim"]

    with pytest.raises(ValueError, match="claim_refs"):
        decode_fused_evidence(tampered)


def test_decoding_a_weighting_that_no_longer_matches_is_refused() -> None:
    record = _through_json(encode_fused_evidence(make_rich().evidence))
    record["weighting"]["hypotheses"][0]["weighted_support"] = 99.0

    with pytest.raises(ValueError, match="weighted_support"):
        decode_fused_evidence(record)


def test_a_baseline_evidence_has_no_weighting_and_round_trips() -> None:
    fused = make_fused_evidence()

    record = _through_json(encode_fused_evidence(fused))

    assert record["weighting"] is None
    assert decode_fused_evidence(record) == fused
    assert dataclasses.replace(fused, weighting=None) == fused

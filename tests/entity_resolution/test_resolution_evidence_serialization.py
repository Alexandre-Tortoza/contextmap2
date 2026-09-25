"""Persisting evidence and decisions needs no model or backend runtime, and revalidates on read."""

from __future__ import annotations

import json
from typing import Any

import pytest
from resolution_builders import (
    appearance_evidence,
    decision,
    geometry_evidence,
    match_evidence,
    representation_evidence,
    semantic_evidence,
    temporal_evidence,
)

from contextmap.entity_resolution import (
    EvidenceStatus,
    ResolutionOutcome,
    UnresolvedReason,
    decode_match_evidence,
    decode_resolution_decision,
    encode_match_evidence,
    encode_resolution_decision,
)


def everything() -> Any:
    return match_evidence(
        geometry=geometry_evidence(EvidenceStatus.CONFLICTING),
        semantic=semantic_evidence(),
        appearance=appearance_evidence(),
        temporal=temporal_evidence(),
        point_representation=representation_evidence(),
    )


def test_match_evidence_survives_a_json_round_trip() -> None:
    evidence = everything()

    record = json.loads(json.dumps(encode_match_evidence(evidence)))

    assert decode_match_evidence(record) == evidence


def test_the_encoding_is_deterministic() -> None:
    assert json.dumps(encode_match_evidence(everything()), sort_keys=True) == json.dumps(
        encode_match_evidence(everything()), sort_keys=True
    )


@pytest.mark.parametrize("outcome", list(ResolutionOutcome))
def test_every_outcome_survives_a_round_trip(outcome: ResolutionOutcome) -> None:
    kwargs: dict[str, Any] = {}
    if outcome is ResolutionOutcome.UNRESOLVED:
        kwargs = {
            "unresolved_reason": UnresolvedReason.INSUFFICIENT_EVIDENCE,
            "channels_used": (),
        }
    result = decision(outcome, **kwargs)

    record = json.loads(json.dumps(encode_resolution_decision(result)))

    assert decode_resolution_decision(record) == result
    assert decode_resolution_decision(record).decision is outcome


def test_a_not_evaluated_channel_stays_absent_after_a_round_trip() -> None:
    evidence = match_evidence()

    decoded = decode_match_evidence(json.loads(json.dumps(encode_match_evidence(evidence))))

    assert decoded.appearance is None and decoded.point_representation is None


def test_decoding_revalidates_the_invariants() -> None:
    record = encode_match_evidence(everything())
    record["comparison_id"] = "comparison--tampered"

    with pytest.raises(ValueError, match="comparison_id"):
        decode_match_evidence(record)


def test_decoding_refuses_an_impossible_measurement() -> None:
    record = encode_match_evidence(everything())
    record["geometry"]["measurement"]["bounds_iou"] = 4.0

    with pytest.raises(ValueError, match="bounds_iou"):
        decode_match_evidence(record)


def test_decoding_names_a_missing_field() -> None:
    record = encode_resolution_decision(decision())
    del record["policy"]

    with pytest.raises(ValueError, match="policy"):
        decode_resolution_decision(record)


def test_decoding_refuses_a_field_it_does_not_know() -> None:
    record = encode_resolution_decision(decision())
    record["score"] = 0.99

    with pytest.raises(ValueError, match="score"):
        decode_resolution_decision(record)


def test_decoding_refuses_an_unknown_outcome() -> None:
    record = encode_resolution_decision(decision())
    record["decision"] = "probably"

    with pytest.raises(ValueError, match="decision"):
        decode_resolution_decision(record)


def test_decoding_refuses_a_value_of_the_wrong_type() -> None:
    record = encode_match_evidence(everything())
    record["geometry"]["measurement"]["centroid_distance_m"] = "near"

    with pytest.raises(ValueError, match="centroid_distance_m"):
        decode_match_evidence(record)

"""JSON-friendly encoding of the Entity Resolution contracts.

Records contain only JSON primitives, so a persisted resolution is readable without ROS, NumPy or
any model runtime. Decoding rebuilds the contracts through their constructors, so every invariant
is revalidated and a tampered record is refused rather than trusted: a missing field, an unknown
field, a value of the wrong type and an unknown enum member are all errors that name the value.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contextmap.entity_resolution._codec import from_record, to_record
from contextmap.entity_resolution.decision import ResolutionDecision
from contextmap.entity_resolution.evidence import EntityMatchEvidence
from contextmap.entity_resolution.models import ResolvedEntityReference


def encode_resolved_entity_reference(reference: ResolvedEntityReference) -> dict[str, Any]:
    """Encode the stable handle of a resolved entity."""
    record: dict[str, Any] = to_record(reference)
    return record


def decode_resolved_entity_reference(record: Mapping[str, Any]) -> ResolvedEntityReference:
    """Decode a resolved-entity reference and revalidate it.

    Args:
        record: The output of :func:`encode_resolved_entity_reference`.

    Returns:
        The reference.

    Raises:
        ValueError: If a field is missing or unknown, or an identity is empty.
    """
    return from_record(ResolvedEntityReference, record)


def encode_match_evidence(evidence: EntityMatchEvidence) -> dict[str, Any]:
    """Encode the evidence of one comparison, channel by channel."""
    record: dict[str, Any] = to_record(evidence)
    return record


def decode_match_evidence(record: Mapping[str, Any]) -> EntityMatchEvidence:
    """Decode the evidence of one comparison and revalidate every invariant.

    Args:
        record: The output of :func:`encode_match_evidence`.

    Returns:
        The evidence.

    Raises:
        ValueError: If the record is malformed or violates the contract.
    """
    return from_record(EntityMatchEvidence, record)


def encode_resolution_decision(decision: ResolutionDecision) -> dict[str, Any]:
    """Encode one resolution decision with its rules and channels."""
    record: dict[str, Any] = to_record(decision)
    return record


def decode_resolution_decision(record: Mapping[str, Any]) -> ResolutionDecision:
    """Decode a resolution decision and revalidate every invariant.

    Args:
        record: The output of :func:`encode_resolution_decision`.

    Returns:
        The decision.

    Raises:
        ValueError: If the record is malformed or violates the contract.
    """
    return from_record(ResolutionDecision, record)

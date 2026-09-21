"""JSON-friendly encoding of the Entity Resolution contracts.

Records contain only JSON primitives, so a persisted resolution is readable without ROS, NumPy or
any model runtime. Decoding rebuilds the contracts through their constructors, so every invariant
is revalidated and a tampered record is refused rather than trusted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contextmap.entity_resolution.models import (
    EntityResolutionRunId,
    ResolvedEntityId,
    ResolvedEntityReference,
)


def _field(record: Mapping[str, Any], name: str) -> Any:
    """Read a required field, naming it when the record does not have it."""
    try:
        return record[name]
    except KeyError:
        raise ValueError(f"record is missing the field {name!r}") from None


def encode_resolved_entity_reference(reference: ResolvedEntityReference) -> dict[str, Any]:
    """Encode the stable handle of a resolved entity."""
    return {
        "resolution_run_id": str(reference.resolution_run_id),
        "resolved_entity_id": str(reference.resolved_entity_id),
    }


def decode_resolved_entity_reference(record: Mapping[str, Any]) -> ResolvedEntityReference:
    """Decode a resolved-entity reference and revalidate it.

    Args:
        record: The output of :func:`encode_resolved_entity_reference`.

    Returns:
        The reference.

    Raises:
        ValueError: If a field is missing or an identity is empty.
    """
    return ResolvedEntityReference(
        resolution_run_id=EntityResolutionRunId(_field(record, "resolution_run_id")),
        resolved_entity_id=ResolvedEntityId(_field(record, "resolved_entity_id")),
    )

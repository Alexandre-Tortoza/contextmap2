"""Deterministic builders for ContextMap schema tests.

Every builder returns a valid object built only from public contracts, so a test states
what it changes instead of repeating the whole map.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from contextmap.artifact import (
    ContextMap,
    ContextMapId,
    ContextMapMetadata,
    GeometricMapLink,
    MapCreation,
    PolicyRef,
    SourceSequence,
)
from contextmap.geometric_mapping import MapId

CONTEXT_MAP_ID = ContextMapId("context-map--corridor-02--0001")
GEOMETRIC_MAP_ID = MapId("corridor-02--map-run-0001")
SEQUENCE_ARTIFACT_ID = "sequence--corridor-02--e145f73f"
SELECTION_ID = "selection--90s-20frames"
POINT_COUNT = 1_000


def policy(policy_id: str = "context-map-assembly", version: str = "1") -> PolicyRef:
    return PolicyRef(policy_id=policy_id, version=version)


def creation(**overrides: Any) -> MapCreation:
    return replace(
        MapCreation(
            assembly_policy=policy(),
            code_version="a1b2c3d",
            configuration_fingerprint="sha256:cfg-0001",
        ),
        **overrides,
    )


def source_sequence(
    sequence_artifact_id: str = SEQUENCE_ARTIFACT_ID, selection_id: str = SELECTION_ID
) -> SourceSequence:
    return SourceSequence(sequence_artifact_id=sequence_artifact_id, selection_id=selection_id)


def metadata(**overrides: Any) -> ContextMapMetadata:
    return replace(
        ContextMapMetadata(creation=creation(), source_sequences=(source_sequence(),)),
        **overrides,
    )


def geometry_link(**overrides: Any) -> GeometricMapLink:
    return replace(GeometricMapLink(map_id=GEOMETRIC_MAP_ID, point_count=POINT_COUNT), **overrides)


def context_map(**overrides: Any) -> ContextMap:
    return replace(
        ContextMap(
            context_map_id=CONTEXT_MAP_ID,
            schema_version="0.1.0",
            metadata=metadata(),
            geometry_ref=geometry_link(),
        ),
        **overrides,
    )

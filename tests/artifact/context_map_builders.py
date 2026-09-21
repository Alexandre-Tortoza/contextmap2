"""Deterministic builders for ContextMap schema tests.

Every builder returns a valid object built only from public contracts, so a test states
what it changes instead of repeating the whole map.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from contextmap.artifact import (
    AnchorKind,
    ContextMap,
    ContextMapId,
    ContextMapMetadata,
    DeclaredCapabilities,
    GeometricMapLink,
    Handedness,
    LengthUnit,
    MapAnchor,
    MapCapability,
    MapCreation,
    MapFrame,
    ObservationWindow,
    PolicyRef,
    SourceSequence,
)
from contextmap.geometric_mapping import Bounds3D, MapId
from contextmap.ingestion import FrameId
from contextmap.shared import SourceTimestamp

CONTEXT_MAP_ID = ContextMapId("context-map--corridor-02--0001")
GEOMETRIC_MAP_ID = MapId("corridor-02--map-run-0001")
SEQUENCE_ARTIFACT_ID = "sequence--corridor-02--e145f73f"
SELECTION_ID = "selection--90s-20frames"
POINT_COUNT = 1_000
MAP_FRAME_ID = "map"
CLOCK_ID = "fixture:header"


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


def timestamp(seconds: int, nanoseconds: int = 0, *, clock_id: str = CLOCK_ID) -> SourceTimestamp:
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)


def window(**overrides: Any) -> ObservationWindow:
    return replace(ObservationWindow(start=timestamp(100), end=timestamp(190)), **overrides)


def bounds(frame_id: str = MAP_FRAME_ID, **overrides: Any) -> Bounds3D:
    return replace(
        Bounds3D(
            frame_id=FrameId(frame_id), minimum_m=(-10.0, -5.0, 0.0), maximum_m=(10.0, 5.0, 3.0)
        ),
        **overrides,
    )


def estimator_local_anchor(**overrides: Any) -> MapAnchor:
    return replace(
        MapAnchor(
            kind=AnchorKind.ESTIMATOR_LOCAL,
            origin_definition="pose of the first accepted scan of the estimator run",
            reference_frame_id=None,
        ),
        **overrides,
    )


def external_anchor(reference_frame_id: str | None = "site-a/enu", **overrides: Any) -> MapAnchor:
    return replace(
        MapAnchor(
            kind=AnchorKind.EXTERNALLY_ANCHORED,
            origin_definition="surveyed control point of the site",
            reference_frame_id=reference_frame_id,
        ),
        **overrides,
    )


def map_frame(**overrides: Any) -> MapFrame:
    return replace(
        MapFrame(
            frame_id=MAP_FRAME_ID,
            unit=LengthUnit.METER,
            handedness=Handedness.RIGHT_HANDED,
            up_direction=(0.0, 0.0, 1.0),
            anchor=estimator_local_anchor(),
        ),
        **overrides,
    )


def capabilities(**overrides: Any) -> DeclaredCapabilities:
    return replace(
        DeclaredCapabilities(content=(MapCapability.GEOMETRY,), relation_predicates=()),
        **overrides,
    )


def metadata(**overrides: Any) -> ContextMapMetadata:
    return replace(
        ContextMapMetadata(
            creation=creation(),
            source_sequences=(source_sequence(),),
            frame=map_frame(),
            bounds=bounds(),
            time_bounds=window(),
            capabilities=capabilities(),
        ),
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

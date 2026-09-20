"""Deterministic builders for Point Representation tests."""

from __future__ import annotations

from collections.abc import Sequence

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.point_representation import (
    CenteringMode,
    CoordinatePreparation,
    EncoderIdentity,
    NeighborhoodMethod,
    PointRepresentation,
    PointRepresentationRunId,
    PointSupport,
    RepresentationProvenance,
    RepresentationSpace,
    ScaleNormalization,
    SupportPolicy,
    SupportStatistics,
    SupportType,
    representation_id_for,
    representation_space_fingerprint,
)

MAP_ID = MapId("map-0001")
RUN_ID = PointRepresentationRunId("run-0001")


def ref(index: int, *, map_id: MapId = MAP_ID) -> GeometryReference:
    return GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))


def radius_policy(
    radius_m: float = 0.5,
    *,
    max_neighbors: int | None = None,
    centering: CenteringMode = CenteringMode.CENTER,
    scale: ScaleNormalization = ScaleNormalization.NONE,
) -> SupportPolicy:
    return SupportPolicy(
        support_type=SupportType.NEIGHBORHOOD,
        method=NeighborhoodMethod.RADIUS,
        radius_m=radius_m,
        k=None,
        max_neighbors=max_neighbors,
        preparation=CoordinatePreparation(centering=centering, scale_normalization=scale),
    )


def knn_policy(k: int = 8) -> SupportPolicy:
    return SupportPolicy(
        support_type=SupportType.NEIGHBORHOOD,
        method=NeighborhoodMethod.K_NEAREST,
        radius_m=None,
        k=k,
        max_neighbors=None,
        preparation=CoordinatePreparation(),
    )


def make_statistics(count: int = 4, *, max_distance_m: float = 0.4) -> SupportStatistics:
    return SupportStatistics(
        count=count,
        min_distance_m=0.0,
        max_distance_m=max_distance_m,
        mean_distance_m=max_distance_m / 2,
        near_map_bounds=False,
        candidate_count=None,
    )


def make_support(
    center_index: int = 0,
    member_indexes: Sequence[int] = (0, 1, 2, 3),
    *,
    policy: SupportPolicy | None = None,
    map_id: MapId = MAP_ID,
    query_method: str = "bounds-query+euclidean-filter",
) -> PointSupport:
    return PointSupport(
        policy=policy if policy is not None else radius_policy(),
        center=ref(center_index, map_id=map_id),
        geometry_refs=tuple(ref(i, map_id=map_id) for i in member_indexes),
        map_frame="map",
        statistics=make_statistics(len(member_indexes)),
        query_method=query_method,
    )


def make_space(
    *,
    family: str = "geometric_descriptor",
    model: str = "local-covariance-shape",
    version: str = "1",
    dimension: int = 4,
    checkpoint: str | None = None,
    normalization: str = "none",
    dtype: str = "float32",
    policy: SupportPolicy | None = None,
    feature_names: tuple[str, ...] = ("a", "b", "c", "d"),
) -> RepresentationSpace:
    return RepresentationSpace(
        family=family,
        model=model,
        version=version,
        checkpoint=checkpoint,
        dimension=dimension,
        dtype=dtype,
        normalization=normalization,
        input_definition="xyz-local-prepared",
        support_semantics=policy if policy is not None else radius_policy(),
        feature_names=feature_names if feature_names and len(feature_names) == dimension else (),
    )


ENCODER = EncoderIdentity(
    backend_id="fake_encoder",
    backend_version="1",
    configuration_fingerprint="sha256:cfg",
    checkpoint_hash=None,
)


def make_representation(
    index: int = 0,
    *,
    space: RepresentationSpace | None = None,
    support: PointSupport | None = None,
    payload_reference: str | None = "payloads/vectors.f32#0",
    undefined_components: tuple[int, ...] = (),
) -> PointRepresentation:
    chosen_space = space if space is not None else make_space()
    chosen_support = support if support is not None else make_support(index)
    return PointRepresentation(
        representation_id=representation_id_for(run_id=RUN_ID, index=index),
        geometry_reference=chosen_support.center,
        support=chosen_support,
        representation_space_id=representation_space_fingerprint(chosen_space),
        shape=(chosen_space.dimension,),
        dtype=chosen_space.dtype,
        normalization=chosen_space.normalization,
        payload_reference=payload_reference,
        encoder_identity=ENCODER,
        provenance=RepresentationProvenance(code_version="test"),
        undefined_components=undefined_components,
    )

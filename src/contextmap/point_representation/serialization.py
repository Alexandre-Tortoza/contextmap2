"""JSON-friendly encoding of the Point Representation contracts.

Records contain only JSON primitives, so a persisted representation is readable
without NumPy or a model library. A representation record never embeds its
vector: the numerical payload lives in the artifact's storage and is reached
through ``payload_reference``. Decoding rebuilds the contracts, so their
invariants are revalidated on every read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contextmap.geometric_mapping import GeometryId, GeometryReference, MapId
from contextmap.point_representation.models import (
    CenteringMode,
    CoordinatePreparation,
    EncoderIdentity,
    FailedSupport,
    FailureReason,
    NeighborhoodMethod,
    PointRepresentation,
    PointRepresentationId,
    PointSupport,
    RepresentationProvenance,
    RepresentationSpace,
    ScaleNormalization,
    SupportPolicy,
    SupportStatistics,
    SupportType,
)


def encode_representation_space(space: RepresentationSpace) -> dict[str, Any]:
    """Encode a space, including the support policy that is part of its identity."""
    return {
        "family": space.family,
        "model": space.model,
        "version": space.version,
        "checkpoint": space.checkpoint,
        "dimension": space.dimension,
        "dtype": space.dtype,
        "normalization": space.normalization,
        "input_definition": space.input_definition,
        "support_semantics": encode_support_policy(space.support_semantics),
        "feature_names": list(space.feature_names),
    }


def decode_representation_space(record: Mapping[str, Any]) -> RepresentationSpace:
    """Decode a space and revalidate it.

    Raises:
        ValueError: If the record is malformed or violates the space contract.
    """
    return RepresentationSpace(
        family=record["family"],
        model=record["model"],
        version=record["version"],
        checkpoint=record["checkpoint"],
        dimension=record["dimension"],
        dtype=record["dtype"],
        normalization=record["normalization"],
        input_definition=record["input_definition"],
        support_semantics=decode_support_policy(record["support_semantics"]),
        feature_names=tuple(record["feature_names"]),
    )


def encode_point_support(support: PointSupport) -> dict[str, Any]:
    """Encode a support with every geometry element that formed it.

    Every member belongs to the center's map (a support invariant), so the map is
    stated once and the members are listed by geometry id: a persisted run holds
    one support per representation, and repeating the map per member would
    dominate its size.
    """
    statistics = support.statistics
    return {
        "policy": encode_support_policy(support.policy),
        "map_id": str(support.center.map_id),
        "center_geometry_id": str(support.center.geometry_id),
        "geometry_ids": [str(reference.geometry_id) for reference in support.geometry_refs],
        "map_frame": support.map_frame,
        "statistics": {
            "count": statistics.count,
            "min_distance_m": statistics.min_distance_m,
            "max_distance_m": statistics.max_distance_m,
            "mean_distance_m": statistics.mean_distance_m,
            "near_map_bounds": statistics.near_map_bounds,
            "candidate_count": statistics.candidate_count,
        },
        "query_method": support.query_method,
    }


def decode_point_support(record: Mapping[str, Any]) -> PointSupport:
    """Decode a support and revalidate that it is reconstructable.

    Raises:
        ValueError: If the record is malformed or violates the support contract.
    """
    statistics = record["statistics"]
    map_id = MapId(record["map_id"])
    return PointSupport(
        policy=decode_support_policy(record["policy"]),
        center=GeometryReference(
            map_id=map_id, geometry_id=GeometryId(record["center_geometry_id"])
        ),
        geometry_refs=tuple(
            GeometryReference(map_id=map_id, geometry_id=GeometryId(geometry_id))
            for geometry_id in record["geometry_ids"]
        ),
        map_frame=record["map_frame"],
        statistics=SupportStatistics(
            count=statistics["count"],
            min_distance_m=statistics["min_distance_m"],
            max_distance_m=statistics["max_distance_m"],
            mean_distance_m=statistics["mean_distance_m"],
            near_map_bounds=statistics["near_map_bounds"],
            candidate_count=statistics["candidate_count"],
        ),
        query_method=record["query_method"],
    )


def encode_point_representation(representation: PointRepresentation) -> dict[str, Any]:
    """Encode a representation's metadata; the vector is never embedded."""
    encoder = representation.encoder_identity
    return {
        "representation_id": str(representation.representation_id),
        "geometry_reference": _encode_reference(representation.geometry_reference),
        "support": encode_point_support(representation.support),
        "representation_space_id": representation.representation_space_id,
        "shape": list(representation.shape),
        "dtype": representation.dtype,
        "normalization": representation.normalization,
        "payload_reference": representation.payload_reference,
        "encoder_identity": {
            "backend_id": encoder.backend_id,
            "backend_version": encoder.backend_version,
            "configuration_fingerprint": encoder.configuration_fingerprint,
            "checkpoint_hash": encoder.checkpoint_hash,
        },
        "provenance": {"code_version": representation.provenance.code_version},
        "undefined_components": list(representation.undefined_components),
    }


def decode_point_representation(record: Mapping[str, Any]) -> PointRepresentation:
    """Decode a representation and revalidate its contract.

    Raises:
        ValueError: If the record is malformed or violates the representation contract.
    """
    encoder = record["encoder_identity"]
    return PointRepresentation(
        representation_id=PointRepresentationId(record["representation_id"]),
        geometry_reference=_decode_reference(record["geometry_reference"]),
        support=decode_point_support(record["support"]),
        representation_space_id=record["representation_space_id"],
        shape=tuple(record["shape"]),
        dtype=record["dtype"],
        normalization=record["normalization"],
        payload_reference=record["payload_reference"],
        encoder_identity=EncoderIdentity(
            backend_id=encoder["backend_id"],
            backend_version=encoder["backend_version"],
            configuration_fingerprint=encoder["configuration_fingerprint"],
            checkpoint_hash=encoder["checkpoint_hash"],
        ),
        provenance=RepresentationProvenance(code_version=record["provenance"]["code_version"]),
        undefined_components=tuple(record["undefined_components"]),
    )


def encode_failed_support(failed: FailedSupport) -> dict[str, Any]:
    """Encode a failed support: which geometry was involved, why, and the encoder's detail."""
    return {
        "support": encode_point_support(failed.support),
        "reason": failed.reason.value,
        "detail": failed.detail,
    }


def decode_failed_support(record: Mapping[str, Any]) -> FailedSupport:
    """Decode a failed support and revalidate it.

    Raises:
        ValueError: If the record is malformed or violates the contract.
    """
    return FailedSupport(
        support=decode_point_support(record["support"]),
        reason=FailureReason(record["reason"]),
        detail=record["detail"],
    )


def _encode_reference(reference: GeometryReference) -> dict[str, Any]:
    return {"map_id": str(reference.map_id), "geometry_id": str(reference.geometry_id)}


def _decode_reference(record: Mapping[str, Any]) -> GeometryReference:
    return GeometryReference(
        map_id=MapId(record["map_id"]), geometry_id=GeometryId(record["geometry_id"])
    )


def encode_support_policy(policy: SupportPolicy) -> dict[str, Any]:
    """Encode a support policy, including how its coordinates are prepared."""
    return {
        "support_type": policy.support_type.value,
        "method": None if policy.method is None else policy.method.value,
        "radius_m": policy.radius_m,
        "k": policy.k,
        "max_neighbors": policy.max_neighbors,
        "preparation": {
            "centering": policy.preparation.centering.value,
            "scale_normalization": policy.preparation.scale_normalization.value,
        },
    }


def decode_support_policy(record: Mapping[str, Any]) -> SupportPolicy:
    """Decode a support policy and revalidate it."""
    method = record["method"]
    preparation = record["preparation"]
    return SupportPolicy(
        support_type=SupportType(record["support_type"]),
        method=None if method is None else NeighborhoodMethod(method),
        radius_m=record["radius_m"],
        k=record["k"],
        max_neighbors=record["max_neighbors"],
        preparation=CoordinatePreparation(
            centering=CenteringMode(preparation["centering"]),
            scale_normalization=ScaleNormalization(preparation["scale_normalization"]),
        ),
    )

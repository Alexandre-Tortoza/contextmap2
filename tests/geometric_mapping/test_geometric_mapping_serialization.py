import dataclasses
import json
from typing import Any

import pytest
from geometry_builders import make_bounds, make_lineage, make_map, make_point

from contextmap.geometric_mapping import (
    GeometryPointProvenance,
    PointOrigin,
    SpatialIndexMetadata,
)
from contextmap.geometric_mapping.serialization import (
    decode_bounds,
    decode_geometric_map,
    decode_geometry_point,
    decode_geometry_reference,
    encode_bounds,
    encode_geometric_map,
    encode_geometry_point,
    encode_geometry_reference,
)


def _through_json(record: dict[str, Any]) -> dict[str, Any]:
    decoded: dict[str, Any] = json.loads(json.dumps(record))
    return decoded


def test_bounds_and_references_round_trip_through_json() -> None:
    bounds = make_bounds((-1.5, 0.0, 2.0), (3.0, 4.5, 6.0))
    reference = make_point(5).reference

    assert decode_bounds(_through_json(encode_bounds(bounds))) == bounds
    assert (
        decode_geometry_reference(_through_json(encode_geometry_reference(reference))) == reference
    )


def test_a_measured_point_round_trips_with_its_full_lineage() -> None:
    point = make_point(2, lineage=make_lineage(pose_ids=("traj--pose-000007", "traj--pose-000008")))

    record = _through_json(encode_geometry_point(point))

    assert decode_geometry_point(record) == point
    assert record["coordinates_m"] == [18.41, 3.82, 1.24]
    assert record["source_coordinates_m"] == [4.21, -0.71, 0.32]
    assert record["map_frame"] == "map" and record["source_frame"] == "lidar"


def test_an_aggregated_point_round_trips_and_stays_distinguishable() -> None:
    point = make_point(
        source_point_index=None,
        provenance=GeometryPointProvenance(
            origin=PointOrigin.AGGREGATED,
            aggregation_rule="voxel-centroid-0.05m",
            contributing_point_count=12,
        ),
    )

    decoded = decode_geometry_point(_through_json(encode_geometry_point(point)))

    assert decoded == point
    assert decoded.provenance.origin is PointOrigin.AGGREGATED


def test_a_map_round_trips_including_its_index_metadata_and_lookup_policy() -> None:
    geometric_map = make_map(
        spatial_index=SpatialIndexMetadata(
            kind="voxel_hash", parameters={"cell_m": 0.5}, is_derived=True
        )
    )

    record = _through_json(encode_geometric_map(geometric_map))

    assert decode_geometric_map(record) == geometric_map
    assert record["provenance"]["pose_lookup"]["mode"] == "interpolated"
    assert record["spatial_index"]["is_derived"] is True


def test_a_map_records_its_aggregation_rule_or_the_absence_of_one() -> None:
    aggregated = make_map(aggregation_rule="scan-voxel-centroid-0.05m")

    record = _through_json(encode_geometric_map(aggregated))

    assert record["aggregation_rule"] == "scan-voxel-centroid-0.05m"
    assert decode_geometric_map(record) == aggregated
    assert _through_json(encode_geometric_map(make_map()))["aggregation_rule"] is None


def test_decoding_revalidates_the_contracts() -> None:
    record = _through_json(encode_geometry_point(make_point()))
    record["map_frame"] = "odom"

    with pytest.raises(ValueError, match="lineage"):
        decode_geometry_point(record)


def test_the_encoded_map_metadata_never_embeds_the_points() -> None:
    record = encode_geometric_map(make_map(point_count=5_000_000))

    assert record["point_count"] == 5_000_000
    assert not {"points", "geometry", "coordinates_m"} & set(record)
    assert dataclasses.is_dataclass(make_map())

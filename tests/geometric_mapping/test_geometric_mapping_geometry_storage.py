import dataclasses
import json
import struct
from typing import Any

import pytest
from map_builders import MAP_ID, accumulate, make_scans, open_geometry

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometryId,
    GeometryReference,
    GeometrySource,
    MapId,
    PackedGeometry,
    ScanVoxelPolicy,
    geometry_id_for,
    geometry_index_of,
    transform_trace_residual_m,
    verify_transform_trace,
)
from contextmap.geometric_mapping.geometry_storage import PACKED_POINT
from contextmap.geometric_mapping.serialization import (
    decode_scan_record,
    decode_transform_trace,
    encode_scan_record,
    encode_transform_trace,
)
from contextmap.ingestion import FrameId, SourceObservationId


def _reference(index: int, *, map_id: MapId = MAP_ID) -> GeometryReference:
    return GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))


# --- Reference identity ---------------------------------------------------------


def test_a_geometry_id_maps_back_to_its_index() -> None:
    for index in (0, 7, 999_999_999, 1_234_567_890):
        geometry_id = geometry_id_for(map_id=MAP_ID, index=index)

        assert geometry_index_of(map_id=MAP_ID, geometry_id=geometry_id) == index


@pytest.mark.parametrize(
    "geometry_id",
    ["map-0001--geom-1", "map-0001--geom--3", "other--geom-000000001", "map-0001--geom-abc", ""],
)
def test_only_the_canonical_spelling_of_a_geometry_id_is_accepted(geometry_id: str) -> None:
    with pytest.raises(ValueError, match="geometry id"):
        geometry_index_of(map_id=MAP_ID, geometry_id=GeometryId(geometry_id))


# --- Resolution --------------------------------------------------------------------


def test_every_reference_resolves_to_the_authoritative_geometry() -> None:
    scans = make_scans(3)
    accumulated, packed = accumulate(scans)
    geometry = open_geometry(accumulated, packed)

    assert isinstance(geometry, GeometrySource)
    assert geometry.geometric_map == accumulated.geometric_map
    assert [p.reference for p in geometry.iter_geometry()] == [_reference(i) for i in range(6)]
    for index in range(6):
        assert geometry.get(_reference(index)).reference == _reference(index)


def test_resolving_twice_returns_the_same_geometry() -> None:
    geometry = open_geometry(*accumulate(make_scans(2)))

    assert geometry.get(_reference(3)) == geometry.get(_reference(3))


@pytest.mark.parametrize(
    "reference",
    [
        _reference(6),
        _reference(0, map_id=MapId("another-map")),
        GeometryReference(map_id=MAP_ID, geometry_id=GeometryId("map-0001--geom-1")),
    ],
    ids=["past-the-end", "another-map", "non-canonical"],
)
def test_a_reference_that_does_not_belong_to_the_map_is_a_key_error(
    reference: GeometryReference,
) -> None:
    geometry = open_geometry(*accumulate(make_scans(3)))

    with pytest.raises(KeyError):
        geometry.get(reference)


# --- Source index ----------------------------------------------------------------------


def test_the_source_index_lists_the_references_of_each_observation() -> None:
    geometry = open_geometry(*accumulate(make_scans(3)))

    references = list(geometry.references_for(SourceObservationId("scan-0001")))

    assert references == [_reference(2), _reference(3)]
    assert {str(geometry.get(r).source_observation_id) for r in references} == {"scan-0001"}
    assert geometry.scan_record(SourceObservationId("scan-0001")).first_geometry_index == 2


def test_an_observation_that_contributed_nothing_is_a_key_error() -> None:
    geometry = open_geometry(*accumulate(make_scans(2)))

    with pytest.raises(KeyError):
        geometry.scan_record(SourceObservationId("scan-9999"))


# --- Bounds query, linear baseline ------------------------------------------------------


def _box(minimum: tuple[float, float, float], maximum: tuple[float, float, float]) -> Bounds3D:
    return Bounds3D(frame_id=FrameId("map"), minimum_m=minimum, maximum_m=maximum)


def test_a_bounds_query_returns_only_the_geometry_inside_in_index_order() -> None:
    geometry = open_geometry(*accumulate(make_scans(3)))

    # Pontos: scan k -> (1.5 + k, 0, 0.25) e (0.5 + k, 2, 0.75).
    found = list(geometry.query_bounds(_box((1.0, -1.0, 0.0), (2.6, 1.0, 1.0))))

    assert [p.reference for p in found] == [_reference(0), _reference(2)]
    assert [p.coordinates_m for p in found] == [(1.5, 0.0, 0.25), (2.5, 0.0, 0.25)]


def test_faces_are_inclusive_and_a_point_on_a_face_is_returned() -> None:
    geometry = open_geometry(*accumulate(make_scans(1)))

    on_face = list(geometry.query_bounds(_box((1.5, 0.0, 0.25), (1.5, 0.0, 0.25))))

    assert [p.reference for p in on_face] == [_reference(0)]


def test_a_box_that_touches_nothing_returns_nothing() -> None:
    geometry = open_geometry(*accumulate(make_scans(2)))

    assert list(geometry.query_bounds(_box((50.0, 50.0, 50.0), (60.0, 60.0, 60.0)))) == []


def test_a_query_in_another_frame_is_refused_instead_of_reinterpreted() -> None:
    geometry = open_geometry(*accumulate(make_scans(1)))
    elsewhere = Bounds3D(
        frame_id=FrameId("odom"), minimum_m=(0.0, 0.0, 0.0), maximum_m=(9.0, 9.0, 9.0)
    )

    with pytest.raises(ValueError, match="frame"):
        list(geometry.query_bounds(elsewhere))


# --- Integrity at open ---------------------------------------------------------------------


def test_a_payload_of_the_wrong_size_is_refused() -> None:
    accumulated, packed = accumulate(make_scans(2))

    with pytest.raises(ValueError, match="bytes"):
        open_geometry(accumulated, packed[:-1])


def test_a_source_index_that_does_not_tile_the_geometry_is_refused() -> None:
    accumulated, packed = accumulate(make_scans(3))
    first, second, third = accumulated.scans
    gapped = (first, dataclasses.replace(second, first_geometry_index=3), third)

    with pytest.raises(ValueError, match="source index"):
        PackedGeometry(geometric_map=accumulated.geometric_map, scans=gapped, records=packed)


def test_a_point_that_points_at_another_scans_range_is_detected_on_access() -> None:
    accumulated, packed = accumulate(make_scans(2))
    corrupted = bytearray(packed)
    # O ponto 0 passa a declarar que veio do scan 1, cujo intervalo não o contém.
    struct.pack_into("<I", corrupted, 48, 1)
    geometry = open_geometry(accumulated, bytes(corrupted))

    with pytest.raises(ValueError, match="scan"):
        geometry.get(_reference(0))


def test_an_aggregated_point_in_a_map_that_declares_no_aggregation_is_detected() -> None:
    accumulated, packed = accumulate(make_scans(1))
    corrupted = bytearray(packed)
    struct.pack_into("<I", corrupted, 60, 3)
    geometry = open_geometry(accumulated, bytes(corrupted))

    with pytest.raises(ValueError, match="aggregat"):
        geometry.get(_reference(0))


def test_an_aggregated_map_reopens_with_its_rule() -> None:
    accumulated, packed = accumulate(
        make_scans(1, points=((1.1, 0.0, 0.0), (1.2, 0.0, 0.0))),
        aggregation=ScanVoxelPolicy(cell_m=0.5),
    )

    (point,) = open_geometry(accumulated, packed).iter_geometry()

    assert point.provenance.aggregation_rule == "scan-voxel-centroid-0.5m"
    assert PACKED_POINT.size == 64


# --- Traces of persisted points -----------------------------------------------------------------


def test_a_persisted_point_is_traced_through_its_chain_without_the_scan() -> None:
    scans = make_scans(2)
    accumulated, packed = accumulate(scans)
    geometry = open_geometry(accumulated, packed)

    trace = geometry.trace(_reference(3))

    assert verify_transform_trace(trace) == []
    assert str(trace.source_observation_id) == "scan-0001"
    assert trace.source_point_index == 1
    assert trace.map_coordinates_m == geometry.get(_reference(3)).coordinates_m
    assert trace.transforms == scans[1].transform_chain


def test_an_aggregated_point_is_traced_without_a_point_index() -> None:
    accumulated, packed = accumulate(
        make_scans(1, points=((1.1, 0.0, 0.0), (1.2, 0.0, 0.0))),
        aggregation=ScanVoxelPolicy(cell_m=0.5),
    )
    trace = open_geometry(accumulated, packed).trace(_reference(0))

    assert trace.source_point_index is None
    assert verify_transform_trace(trace, tolerance_m=1e-6) == []
    assert decode_transform_trace(_through_json(encode_transform_trace(trace))) == trace


def test_a_reference_that_is_not_in_the_map_cannot_be_traced() -> None:
    geometry = open_geometry(*accumulate(make_scans(1)))

    with pytest.raises(KeyError):
        geometry.trace(_reference(9))


def test_the_residual_of_a_sound_trace_is_zero_and_of_a_tampered_one_is_its_distance() -> None:
    geometry = open_geometry(*accumulate(make_scans(1)))
    trace = geometry.trace(_reference(0))

    assert transform_trace_residual_m(trace) == pytest.approx(0.0, abs=1e-9)
    moved = dataclasses.replace(trace, map_coordinates_m=(1.5 + 3.0, 0.0 + 4.0, 0.25))
    assert transform_trace_residual_m(moved) == pytest.approx(5.0, abs=1e-9)


def test_closing_the_geometry_releases_the_payload() -> None:
    accumulated, packed = accumulate(make_scans(1))
    payload = bytearray(packed)
    geometry = PackedGeometry(
        geometric_map=accumulated.geometric_map, scans=accumulated.scans, records=payload
    )

    with pytest.raises(BufferError):
        payload.append(0)  # enquanto a geometria está aberta, o payload exportado não redimensiona
    geometry.close()
    payload.append(0)

    assert len(payload) == len(packed) + 1


# --- Serialization -----------------------------------------------------------------------------


def _through_json(record: dict[str, Any]) -> dict[str, Any]:
    decoded: dict[str, Any] = json.loads(json.dumps(record))
    return decoded


def test_a_source_index_entry_round_trips_with_its_chain_and_bounds() -> None:
    accumulated, _ = accumulate(make_scans(3))
    record = accumulated.scans[1]

    decoded = decode_scan_record(_through_json(encode_scan_record(record)))

    assert decoded == record
    assert decoded.transform_chain == record.transform_chain
    assert decoded.bounds == record.bounds


def test_an_entry_without_geometry_round_trips_with_no_bounds() -> None:
    accumulated, _ = accumulate(
        [make_scans(1, points=((float("nan"),) * 3,))[0], make_scans(1, first_scan=1)[0]]
    )
    record = accumulated.scans[0]

    assert decode_scan_record(_through_json(encode_scan_record(record))) == record
    assert record.bounds is None

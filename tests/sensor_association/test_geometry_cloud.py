import numpy as np
import pytest
from projection_builders import MAP_ID, ArrayGeometrySource, make_calibration

from contextmap.geometric_mapping import GeometryReference, geometry_id_for
from contextmap.sensor_association.geometry_cloud import GeometryCloud

POINTS = [(1.0, 2.0, 3.0), (-4.5, 0.25, 8.0), (10.0, -10.0, 0.5)]


def test_the_cloud_holds_the_authoritative_map_frame_coordinates_in_iteration_order() -> None:
    source = ArrayGeometrySource(POINTS, calibration=make_calibration())

    cloud = GeometryCloud.from_source(source)

    np.testing.assert_array_equal(cloud.coordinates_m, np.array(POINTS))
    assert cloud.geometric_map == source.geometric_map
    assert len(cloud) == 3


def test_a_reference_is_the_stable_identity_of_the_point_at_that_position() -> None:
    cloud = GeometryCloud.from_source(ArrayGeometrySource(POINTS, calibration=make_calibration()))

    assert cloud.reference(1) == GeometryReference(
        map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=1)
    )
    assert cloud.references(np.array([2, 0])) == (cloud.reference(2), cloud.reference(0))


def test_a_source_whose_ids_are_not_positional_is_rejected() -> None:
    source = ArrayGeometrySource(POINTS, calibration=make_calibration(), positional_ids=False)

    with pytest.raises(ValueError, match="positional"):
        GeometryCloud.from_source(source)


def test_a_source_that_disagrees_with_its_declared_point_count_is_rejected() -> None:
    class ShortSource(ArrayGeometrySource):
        def iter_geometry(self):  # type: ignore[no-untyped-def]
            return iter(list(super().iter_geometry())[:-1])

    source = ShortSource(POINTS, calibration=make_calibration())

    with pytest.raises(ValueError, match="point_count"):
        GeometryCloud.from_source(source)

"""Frame conventions: nothing upstream declares an up axis, so the policy declares it, or fails."""

from __future__ import annotations

import pytest
from relation_scene import MAP_ID, box_geometry

from contextmap.geometric_mapping import MapId
from contextmap.spatial_relations import (
    FRAME_CONVENTIONS_POLICY_ID,
    AxisDirection,
    FrameConventions,
    FrameRequirement,
    IncompatibleFrameError,
    UndeclaredAxisError,
)

UNIT_BOX = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def test_axis_directions_are_the_signed_principal_axes_of_the_map_frame() -> None:
    assert AxisDirection.POSITIVE_Z.vector == (0.0, 0.0, 1.0)
    assert AxisDirection.NEGATIVE_Y.vector == (0.0, -1.0, 0.0)
    assert AxisDirection.POSITIVE_X.axis_index == 0
    assert AxisDirection.NEGATIVE_Z.axis_index == 2
    assert AxisDirection.NEGATIVE_X.sign == -1
    assert AxisDirection.POSITIVE_Y.sign == 1
    assert AxisDirection.POSITIVE_Z.opposite is AxisDirection.NEGATIVE_Z
    assert AxisDirection.NEGATIVE_X.opposite is AxisDirection.POSITIVE_X


def test_policy_is_versioned() -> None:
    assert FRAME_CONVENTIONS_POLICY_ID == "map-frame-conventions-v1"


def test_no_axis_is_assumed() -> None:
    conventions = FrameConventions(map_frame="map")
    assert conventions.up_axis is None
    assert conventions.forward_axis is None
    assert conventions.supports(FrameRequirement.MAP_FRAME)
    assert not conventions.supports(FrameRequirement.UP_AXIS)
    assert not conventions.supports(FrameRequirement.UP_AND_FORWARD_AXES)


def test_declared_axes_satisfy_the_requirements() -> None:
    up_only = FrameConventions(map_frame="map", up_axis=AxisDirection.POSITIVE_Z)
    assert up_only.supports(FrameRequirement.UP_AXIS)
    assert not up_only.supports(FrameRequirement.UP_AND_FORWARD_AXES)
    both = FrameConventions(
        map_frame="map",
        up_axis=AxisDirection.POSITIVE_Z,
        forward_axis=AxisDirection.POSITIVE_X,
    )
    assert both.supports(FrameRequirement.UP_AND_FORWARD_AXES)
    assert both.supports(FrameRequirement.UP_AXIS)
    assert both.supports(FrameRequirement.MAP_FRAME)


def test_the_map_frame_must_be_named() -> None:
    with pytest.raises(ValueError, match="map_frame"):
        FrameConventions(map_frame=" ")


def test_a_forward_axis_needs_an_up_axis() -> None:
    with pytest.raises(ValueError, match="forward"):
        FrameConventions(map_frame="map", forward_axis=AxisDirection.POSITIVE_X)


def test_the_forward_axis_must_be_horizontal() -> None:
    with pytest.raises(ValueError, match="perpendicular"):
        FrameConventions(
            map_frame="map",
            up_axis=AxisDirection.POSITIVE_Z,
            forward_axis=AxisDirection.NEGATIVE_Z,
        )


def test_compatible_geometry_is_accepted() -> None:
    conventions = FrameConventions(map_frame="map", up_axis=AxisDirection.POSITIVE_Z)
    first = box_geometry(*UNIT_BOX)
    second = box_geometry((2.0, 0.0, 0.0), (3.0, 1.0, 1.0))
    conventions.require_evaluable(FrameRequirement.UP_AXIS, first, second)


def test_geometry_in_another_frame_is_refused_loudly() -> None:
    conventions = FrameConventions(map_frame="odom", up_axis=AxisDirection.POSITIVE_Z)
    with pytest.raises(IncompatibleFrameError, match="odom"):
        conventions.require_evaluable(
            FrameRequirement.MAP_FRAME, box_geometry(*UNIT_BOX), box_geometry(*UNIT_BOX)
        )


def test_geometry_of_different_maps_is_refused_even_with_the_same_frame_name() -> None:
    conventions = FrameConventions(map_frame="map")
    first = box_geometry(*UNIT_BOX, map_id=MAP_ID)
    second = box_geometry(*UNIT_BOX, map_id=MapId("map-0002"))
    with pytest.raises(IncompatibleFrameError, match="map"):
        conventions.require_evaluable(FrameRequirement.MAP_FRAME, first, second)


def test_a_predicate_that_needs_an_undeclared_axis_is_refused_loudly() -> None:
    conventions = FrameConventions(map_frame="map")
    geometry = box_geometry(*UNIT_BOX)
    with pytest.raises(UndeclaredAxisError, match="up"):
        conventions.require_evaluable(FrameRequirement.UP_AXIS, geometry, geometry)
    with_up = FrameConventions(map_frame="map", up_axis=AxisDirection.POSITIVE_Z)
    with pytest.raises(UndeclaredAxisError, match="forward"):
        with_up.require_evaluable(FrameRequirement.UP_AND_FORWARD_AXES, geometry, geometry)


def test_both_refusals_share_one_error_family() -> None:
    assert issubclass(IncompatibleFrameError, ValueError)
    assert issubclass(UndeclaredAxisError, ValueError)


def test_fingerprint_is_stable_and_follows_every_declared_value() -> None:
    base = FrameConventions(map_frame="map", up_axis=AxisDirection.POSITIVE_Z)
    same = FrameConventions(map_frame="map", up_axis=AxisDirection.POSITIVE_Z)
    assert base.fingerprint() == same.fingerprint()
    assert base.fingerprint().startswith("sha256:")
    other_up = FrameConventions(map_frame="map", up_axis=AxisDirection.NEGATIVE_Y)
    other_frame = FrameConventions(map_frame="odom", up_axis=AxisDirection.POSITIVE_Z)
    with_forward = FrameConventions(
        map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
    )
    assert len({base.fingerprint(), other_up.fingerprint(), other_frame.fingerprint()}) == 3
    assert with_forward.fingerprint() != base.fingerprint()

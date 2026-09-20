import math

import pytest

from contextmap.shared import (
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_norm,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def test_identity_quaternion_has_unit_norm_with_w_as_last_component() -> None:
    assert quaternion_norm(IDENTITY) == pytest.approx(1.0)
    assert is_unit_quaternion(IDENTITY)


def test_is_unit_quaternion_respects_the_requested_tolerance() -> None:
    slightly_long = (0.0, 0.0, 0.0, 1.001)

    assert not is_unit_quaternion(slightly_long)
    assert is_unit_quaternion(slightly_long, tolerance=1e-2)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_is_unit_quaternion_rejects_non_finite_components(bad_value: float) -> None:
    assert not is_unit_quaternion((0.0, 0.0, bad_value, 1.0), tolerance=1.0)


def test_normalize_quaternion_returns_a_unit_quaternion() -> None:
    assert normalize_quaternion((0.0, 0.0, 0.0, 2.0)) == (0.0, 0.0, 0.0, 1.0)

    normalized = normalize_quaternion((1.0, 1.0, 1.0, 1.0))

    assert normalized == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert is_unit_quaternion(normalized)


@pytest.mark.parametrize(
    "invalid",
    [
        (0.0, 0.0, 0.0, 0.0),
        (math.nan, 0.0, 0.0, 1.0),
        (0.0, 0.0, math.inf, 1.0),
    ],
)
def test_normalize_quaternion_rejects_zero_and_non_finite_input(
    invalid: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValueError, match="quaternion"):
        normalize_quaternion(invalid)

import math

import pytest

from contextmap.shared import (
    Quaternion,
    Vector3,
    compose_rigid,
    invert_rigid,
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_angle_between,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_norm,
    quaternion_to_rotation_matrix,
    rotate_vector,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)
QUARTER_TURN_Z = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))


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


# --- Rotation algebra -------------------------------------------------------


def test_multiplying_by_the_identity_leaves_a_rotation_unchanged() -> None:
    assert quaternion_multiply(IDENTITY, QUARTER_TURN_Z) == pytest.approx(QUARTER_TURN_Z)
    assert quaternion_multiply(QUARTER_TURN_Z, IDENTITY) == pytest.approx(QUARTER_TURN_Z)


def test_two_quarter_turns_about_z_make_a_half_turn() -> None:
    half_turn = quaternion_multiply(QUARTER_TURN_Z, QUARTER_TURN_Z)

    assert half_turn == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-12)


def test_conjugate_negates_the_vector_part_only() -> None:
    assert quaternion_conjugate((0.1, 0.2, 0.3, 0.4)) == (-0.1, -0.2, -0.3, 0.4)


def test_a_quarter_turn_about_z_rotates_x_into_y() -> None:
    assert rotate_vector(QUARTER_TURN_Z, (1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-12
    )


def test_rotation_matrix_of_a_quarter_turn_about_z() -> None:
    matrix = quaternion_to_rotation_matrix(QUARTER_TURN_Z)

    expected = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    for row, expected_row in zip(matrix, expected, strict=True):
        assert row == pytest.approx(expected_row, abs=1e-12)


def test_rotation_matrix_of_the_identity_is_the_identity() -> None:
    assert quaternion_to_rotation_matrix(IDENTITY) == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def test_angle_between_rotations_ignores_the_quaternion_sign() -> None:
    x, y, z, w = QUARTER_TURN_Z
    negated = (-x, -y, -z, -w)

    assert quaternion_angle_between(QUARTER_TURN_Z, negated) == pytest.approx(0.0, abs=1e-12)
    assert quaternion_angle_between(IDENTITY, QUARTER_TURN_Z) == pytest.approx(math.pi / 2)


def test_angle_between_tiny_rotations_keeps_its_precision() -> None:
    tiny = normalize_quaternion((0.0, 0.0, 1e-9, 1.0))

    assert quaternion_angle_between(IDENTITY, tiny) == pytest.approx(2e-9, rel=1e-6)


# --- Rigid transforms -------------------------------------------------------


def test_composing_rigid_transforms_follows_t_a_c_equals_t_a_b_times_t_b_c() -> None:
    translation, rotation = compose_rigid(
        outer_translation=(1.0, 0.0, 0.0),
        outer_rotation=QUARTER_TURN_Z,
        inner_translation=(1.0, 0.0, 0.0),
        inner_rotation=IDENTITY,
    )

    assert translation == pytest.approx((1.0, 1.0, 0.0), abs=1e-12)
    assert rotation == pytest.approx(QUARTER_TURN_Z)


def test_composition_matches_applying_the_transforms_one_after_the_other() -> None:
    outer = ((0.5, -1.0, 2.0), QUARTER_TURN_Z)
    inner = ((0.0, 3.0, 1.0), normalize_quaternion((0.1, 0.2, 0.3, 0.9)))
    point = (0.7, -0.4, 1.3)

    translation, rotation = compose_rigid(
        outer_translation=outer[0],
        outer_rotation=outer[1],
        inner_translation=inner[0],
        inner_rotation=inner[1],
    )

    def transformed(rotation_: Quaternion, translation_: Vector3, vector: Vector3) -> Vector3:
        x, y, z = rotate_vector(rotation_, vector)
        return (x + translation_[0], y + translation_[1], z + translation_[2])

    expected = transformed(outer[1], outer[0], transformed(inner[1], inner[0], point))
    combined = transformed(rotation, translation, point)
    assert combined == pytest.approx(expected)


def test_a_transform_composed_with_its_inverse_is_the_identity() -> None:
    rotation = normalize_quaternion((0.1, 0.2, 0.3, 0.9))
    translation = (1.5, -2.0, 0.25)

    inverse_translation, inverse_rotation = invert_rigid(translation=translation, rotation=rotation)
    round_trip_translation, round_trip_rotation = compose_rigid(
        outer_translation=translation,
        outer_rotation=rotation,
        inner_translation=inverse_translation,
        inner_rotation=inverse_rotation,
    )

    assert round_trip_translation == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert quaternion_angle_between(round_trip_rotation, IDENTITY) == pytest.approx(0.0, abs=1e-12)

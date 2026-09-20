"""Library-neutral 3D geometry primitives shared across capabilities.

Rotations are unit quaternions ordered ``(x, y, z, w)``; ``(0, 0, 0, 1)`` is
the identity rotation. This is the same order used by the calibration and
external-pose contracts owned by Ingestion, so values cross capability
boundaries without reordering.

Nothing here requires NumPy or a robotics library: the contracts built on
top of these aliases (poses, transforms, geometry) stay readable with the
standard library alone.
"""

from __future__ import annotations

import math

Vector3 = tuple[float, float, float]
"""``(x, y, z)``; meters unless the owning contract states another unit."""

Quaternion = tuple[float, float, float, float]
"""Rotation quaternion ``(x, y, z, w)``; canonical values have unit norm."""

RotationMatrix = tuple[Vector3, Vector3, Vector3]
"""Row-major 3x3 rotation matrix ``R`` such that ``v_parent = R @ v_child``."""

UNIT_QUATERNION_TOLERANCE = 1e-6
"""Default absolute tolerance on ``|norm - 1|`` for a canonical quaternion."""


def quaternion_norm(quaternion: Quaternion) -> float:
    """Return the Euclidean norm of a quaternion.

    Args:
        quaternion: Quaternion ``(x, y, z, w)``.

    Returns:
        The norm; not guaranteed finite when a component is not finite.
    """
    return math.hypot(*quaternion)


def is_unit_quaternion(
    quaternion: Quaternion, *, tolerance: float = UNIT_QUATERNION_TOLERANCE
) -> bool:
    """Check that a quaternion is finite and has unit norm within a tolerance.

    Args:
        quaternion: Quaternion ``(x, y, z, w)``.
        tolerance: Maximum accepted ``|norm - 1|``.

    Returns:
        ``True`` when every component is finite and the norm is within
        ``tolerance`` of one.
    """
    if not all(math.isfinite(component) for component in quaternion):
        return False
    return abs(quaternion_norm(quaternion) - 1.0) <= tolerance


def normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    """Scale a quaternion to unit norm.

    Args:
        quaternion: Quaternion ``(x, y, z, w)``.

    Returns:
        The quaternion divided by its norm.

    Raises:
        ValueError: If a component is not finite or the norm is zero, since
            neither describes a rotation.
    """
    if not all(math.isfinite(component) for component in quaternion):
        raise ValueError(f"quaternion components must be finite, got {quaternion!r}")
    norm = quaternion_norm(quaternion)
    if norm == 0.0:
        raise ValueError("cannot normalize a zero-norm quaternion")
    x, y, z, w = quaternion
    return (x / norm, y / norm, z / norm, w / norm)


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    """Return the Hamilton product ``left * right``.

    As rotations, ``right`` is applied first and ``left`` second, so
    ``T_a_c`` uses ``quaternion_multiply(q_a_b, q_b_c)``.

    Args:
        left: Quaternion ``(x, y, z, w)``.
        right: Quaternion ``(x, y, z, w)``.

    Returns:
        The product, in the same ``(x, y, z, w)`` order.
    """
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def quaternion_conjugate(quaternion: Quaternion) -> Quaternion:
    """Return the conjugate, which is the inverse rotation for a unit quaternion.

    Args:
        quaternion: Quaternion ``(x, y, z, w)``.

    Returns:
        ``(-x, -y, -z, w)``.
    """
    x, y, z, w = quaternion
    return (-x, -y, -z, w)


def rotate_vector(rotation: Quaternion, vector: Vector3) -> Vector3:
    """Rotate a vector by a unit quaternion.

    Args:
        rotation: Unit quaternion ``(x, y, z, w)``.
        vector: Vector ``(x, y, z)``.

    Returns:
        The rotated vector; ``rotate_vector(q_a_b, v_b)`` is ``v`` expressed in ``a``.
    """
    qx, qy, qz, qw = rotation
    vx, vy, vz = vector
    # v' = v + w * t + q_vec x t, com t = 2 * (q_vec x v): rotação sem montar a matriz.
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def quaternion_to_rotation_matrix(quaternion: Quaternion) -> RotationMatrix:
    """Return the rotation matrix of a unit quaternion.

    The result is only orthonormal when ``quaternion`` has unit norm; a
    preflight can use that to detect a bad rotation.

    Args:
        quaternion: Quaternion ``(x, y, z, w)``.

    Returns:
        Row-major matrix ``R`` such that ``v_parent = R @ v_child``.
    """
    x, y, z, w = quaternion
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def quaternion_angle_between(first: Quaternion, second: Quaternion) -> float:
    """Return the smallest rotation angle in radians between two unit quaternions.

    ``q`` and ``-q`` are the same rotation, so the angle is ``0`` between them.

    Args:
        first: Unit quaternion ``(x, y, z, w)``.
        second: Unit quaternion ``(x, y, z, w)``.

    Returns:
        The angle in ``[0, pi]``, computed from the relative rotation with
        ``atan2`` so it stays accurate for very small angles.
    """
    x, y, z, w = quaternion_multiply(quaternion_conjugate(first), second)
    return 2.0 * math.atan2(math.hypot(x, y, z), abs(w))


def compose_rigid(
    *,
    outer_translation: Vector3,
    outer_rotation: Quaternion,
    inner_translation: Vector3,
    inner_rotation: Quaternion,
) -> tuple[Vector3, Quaternion]:
    """Compose two rigid transforms, ``T_a_c = T_a_b * T_b_c``.

    Args:
        outer_translation: Translation of ``T_a_b``, in meters.
        outer_rotation: Rotation of ``T_a_b``.
        inner_translation: Translation of ``T_b_c``, in meters.
        inner_rotation: Rotation of ``T_b_c``.

    Returns:
        ``(translation, rotation)`` of ``T_a_c``, with the rotation
        renormalized to limit drift over long chains.
    """
    rotated = rotate_vector(outer_rotation, inner_translation)
    translation = (
        outer_translation[0] + rotated[0],
        outer_translation[1] + rotated[1],
        outer_translation[2] + rotated[2],
    )
    rotation = normalize_quaternion(quaternion_multiply(outer_rotation, inner_rotation))
    return translation, rotation


def invert_rigid(*, translation: Vector3, rotation: Quaternion) -> tuple[Vector3, Quaternion]:
    """Invert a rigid transform, ``T_b_a = T_a_b^-1``.

    Args:
        translation: Translation of ``T_a_b``, in meters.
        rotation: Unit rotation of ``T_a_b``.

    Returns:
        ``(translation, rotation)`` of ``T_b_a``.
    """
    inverse_rotation = quaternion_conjugate(rotation)
    negated = (-translation[0], -translation[1], -translation[2])
    return rotate_vector(inverse_rotation, negated), inverse_rotation

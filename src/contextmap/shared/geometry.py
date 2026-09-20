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

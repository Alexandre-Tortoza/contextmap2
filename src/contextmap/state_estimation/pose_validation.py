"""Validation of pose values shared by every backend that publishes poses.

Backends read poses from different sources (an external measurement, an
estimator's output file) but publish through one contract, so they apply the
same rules to the numbers: finite translation, a finite unit orientation
(renormalized only within a declared tolerance, and recorded when it is) and a
finite covariance. Frame, clock and ordering rules depend on the source and
stay with each backend. Nothing here repairs a value silently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from contextmap.shared import (
    Quaternion,
    Vector3,
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_norm,
)
from contextmap.state_estimation.models import PoseEstimate, TrajectoryGap


@dataclass(frozen=True, kw_only=True)
class PoseSampleViolation:
    """One rule a pose sample breaks.

    Attributes:
        rule: ``"non_finite_translation"``, ``"invalid_orientation"`` or
            ``"invalid_covariance"``; backends prefix it with their identity.
        detail: What is wrong, with the numbers involved.
    """

    rule: str
    detail: str


def check_pose_values(
    *,
    translation: Vector3,
    orientation: Quaternion,
    covariance: Sequence[float] | None,
    orientation_norm_tolerance: float,
) -> PoseSampleViolation | None:
    """Check the numbers of a pose sample.

    Args:
        translation: ``(x, y, z)`` in meters.
        orientation: Quaternion ``(x, y, z, w)`` as the source reported it.
        covariance: Row-major 6x6 covariance, or ``None`` when absent.
        orientation_norm_tolerance: Largest ``|norm - 1|`` a source quaternion
            may have and still be renormalized instead of rejected.

    Returns:
        The first violated rule, or ``None`` when the values are publishable.
    """
    if not all(math.isfinite(value) for value in translation):
        return PoseSampleViolation(
            rule="non_finite_translation", detail=f"translation is not finite: {translation!r}"
        )
    norm = quaternion_norm(orientation)
    if not all(math.isfinite(value) for value in orientation) or norm == 0.0:
        return PoseSampleViolation(
            rule="invalid_orientation",
            detail=f"orientation is not a finite non-zero quaternion: {orientation!r}",
        )
    if abs(norm - 1.0) > orientation_norm_tolerance:
        return PoseSampleViolation(
            rule="invalid_orientation",
            detail=(
                f"orientation norm {norm:.6f} differs from 1 by more than "
                f"{orientation_norm_tolerance}"
            ),
        )
    if covariance is not None and not all(math.isfinite(value) for value in covariance):
        return PoseSampleViolation(
            rule="invalid_covariance", detail="pose covariance contains a non-finite value"
        )
    return None


def canonical_orientation(orientation: Quaternion) -> tuple[Quaternion, tuple[str, ...]]:
    """Return the canonical unit orientation and the conversion it needed.

    Args:
        orientation: A finite, non-zero quaternion already accepted by
            :func:`check_pose_values`.

    Returns:
        The orientation unchanged when it is already unit within the contract's
        tolerance, otherwise the renormalized one with a note for
        ``PoseProvenance.conversions_applied``.
    """
    if is_unit_quaternion(orientation):
        return orientation, ()
    note = f"renormalized orientation (norm={quaternion_norm(orientation):.9f})"
    return normalize_quaternion(orientation), (note,)


def gap_between(
    previous: PoseEstimate, current: PoseEstimate, *, max_gap_ns: int | None
) -> TrajectoryGap | None:
    """Return a gap record when two consecutive poses are too far apart in time.

    Args:
        previous: The earlier pose.
        current: The later pose.
        max_gap_ns: Interval beyond which interpolation is not trusted;
            ``None`` records no gaps.

    Returns:
        A :class:`TrajectoryGap`, or ``None`` when the interval is acceptable.
    """
    if max_gap_ns is None:
        return None
    interval_ns = current.timestamp.total_nanoseconds() - previous.timestamp.total_nanoseconds()
    if interval_ns <= max_gap_ns:
        return None
    return TrajectoryGap(
        previous_estimate_id=previous.estimate_id,
        next_estimate_id=current.estimate_id,
        duration_ns=interval_ns,
    )

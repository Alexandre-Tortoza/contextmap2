"""Time alignment: the pose of the body at an observation's timestamp.

Downstream geometry needs ``T_map_body(t)`` for the timestamp of each LiDAR or
RGB observation. A good static calibration is not enough when pose and sensor
streams are misaligned in time, so the lookup is explicit and auditable
instead of an implicit "nearest pose" rule:

* the :class:`LookupPolicy` states which lookups are acceptable and with what
  tolerance;
* every result records the source poses, the time delta and the tolerance, so
  a temporal misalignment can be told apart from a spatial calibration error;
* a pose derived by interpolation stays distinguishable from an estimated one
  (``PoseProvenance.derived_from``);
* a request in another clock domain raises instead of being silently compared.

This module never estimates a camera-to-LiDAR time offset and never projects
points; both belong elsewhere. See
``src/contextmap/state_estimation/docs/lookup.md``.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from statistics import median_low

from contextmap.ingestion import SourceObservation, SourceObservationId
from contextmap.shared import Quaternion, SourceTimestamp, Vector3, normalize_quaternion
from contextmap.state_estimation.models import (
    PoseEstimate,
    PoseEstimateId,
    PoseProvenance,
    PoseValidity,
    Trajectory,
)

# Acima deste produto escalar as duas orientações são praticamente iguais e
# sin(theta) tende a zero; a interpolação linear normalizada é então estável
# e indistinguível da esférica dentro da tolerância do contrato.
_NLERP_DOT_THRESHOLD = 0.9995


class ClockDomainMismatchError(ValueError):
    """Raised when a lookup is requested in another clock domain than the trajectory's."""


class LookupMode(Enum):
    """How a lookup resolves a timestamp that falls between samples.

    Attributes:
        EXACT: Only a pose with the identical timestamp is acceptable.
        NEAREST: The closest pose within a tolerance is acceptable.
        INTERPOLATED: A pose is derived between the two neighboring poses.
    """

    EXACT = "exact"
    NEAREST = "nearest"
    INTERPOLATED = "interpolated"


@dataclass(frozen=True, kw_only=True)
class LookupPolicy:
    """Explicit acceptance rule for one pose lookup.

    Prefer the constructors :meth:`exact`, :meth:`nearest` and
    :meth:`interpolated`.

    Attributes:
        mode: How the timestamp is resolved.
        max_time_delta_ns: For ``NEAREST``, the largest accepted distance
            between the query and the pose, in nanoseconds.
        max_interpolation_gap_ns: For ``INTERPOLATED``, the largest interval
            between the two neighboring poses across which interpolation is
            accepted; ``None`` leaves only recorded gaps as a limit.
    """

    mode: LookupMode
    max_time_delta_ns: int | None = None
    max_interpolation_gap_ns: int | None = None

    def __post_init__(self) -> None:
        """Reject tolerances that do not belong to the selected mode.

        Raises:
            ValueError: If a tolerance is missing, negative, or supplied for
                a mode that does not use it.
        """
        if self.mode is LookupMode.EXACT:
            if self.max_time_delta_ns is not None or self.max_interpolation_gap_ns is not None:
                raise ValueError("lookup policy 'exact' accepts no tolerance")
        elif self.mode is LookupMode.NEAREST:
            if self.max_time_delta_ns is None:
                raise ValueError("lookup policy 'nearest' requires max_time_delta_ns")
            if self.max_time_delta_ns < 0:
                raise ValueError("lookup policy max_time_delta_ns must not be negative")
            if self.max_interpolation_gap_ns is not None:
                raise ValueError("lookup policy 'nearest' does not use max_interpolation_gap_ns")
        else:
            if self.max_time_delta_ns is not None:
                raise ValueError("lookup policy 'interpolated' does not use max_time_delta_ns")
            if self.max_interpolation_gap_ns is not None and self.max_interpolation_gap_ns <= 0:
                raise ValueError("lookup policy max_interpolation_gap_ns must be positive")

    @classmethod
    def exact(cls) -> LookupPolicy:
        """Accept only a pose with the identical timestamp."""
        return cls(mode=LookupMode.EXACT)

    @classmethod
    def nearest(cls, *, max_time_delta_ns: int) -> LookupPolicy:
        """Accept the closest pose within ``max_time_delta_ns`` nanoseconds."""
        return cls(mode=LookupMode.NEAREST, max_time_delta_ns=max_time_delta_ns)

    @classmethod
    def interpolated(cls, *, max_interpolation_gap_ns: int | None = None) -> LookupPolicy:
        """Interpolate between neighboring poses unless the interval is too large."""
        return cls(mode=LookupMode.INTERPOLATED, max_interpolation_gap_ns=max_interpolation_gap_ns)


class LookupOutcome(Enum):
    """How an accepted lookup was resolved."""

    EXACT = "exact"
    NEAREST = "nearest"
    INTERPOLATED = "interpolated"


class LookupRejection(Enum):
    """Why a lookup was rejected instead of resolved.

    Attributes:
        OUT_OF_RANGE: The timestamp lies outside the trajectory (there is no
            extrapolation) and no pose is close enough to accept.
        NO_EXACT_MATCH: ``EXACT`` was requested and no pose has that timestamp.
        TOLERANCE_EXCEEDED: The closest pose is further than the tolerance.
        INTERPOLATION_GAP: The neighboring poses are separated by a recorded
            gap or by more than the configured interpolation tolerance.
    """

    OUT_OF_RANGE = "out_of_range"
    NO_EXACT_MATCH = "no_exact_match"
    TOLERANCE_EXCEEDED = "tolerance_exceeded"
    INTERPOLATION_GAP = "interpolation_gap"


@dataclass(frozen=True, kw_only=True)
class ResolvedPose:
    """An accepted lookup with the evidence needed to audit it.

    Attributes:
        pose: The pose at the query timestamp; a derived pose when
            ``outcome`` is ``INTERPOLATED``.
        outcome: How the pose was resolved.
        query_timestamp: The requested timestamp.
        policy: The policy the lookup was resolved under.
        query_observation_id: Observation the pose was requested for, when the
            lookup was made through :meth:`TrajectoryLookup.pose_for_observation`.
        source_estimate_ids: Estimated poses that contributed (one, or the two
            neighbors of an interpolation).
        time_delta_ns: Distance in nanoseconds from the query to the nearest
            contributing pose.
        interpolation_fraction: Position between the two neighbors in
            ``[0, 1]`` for an interpolated pose; ``None`` otherwise.
    """

    pose: PoseEstimate
    outcome: LookupOutcome
    query_timestamp: SourceTimestamp
    policy: LookupPolicy
    query_observation_id: SourceObservationId | None
    source_estimate_ids: tuple[PoseEstimateId, ...]
    time_delta_ns: int
    interpolation_fraction: float | None


@dataclass(frozen=True, kw_only=True)
class RejectedLookup:
    """A rejected lookup, kept as data so it can be counted and audited.

    Attributes:
        rejection: Why the lookup was rejected.
        detail: Human-readable numbers behind the decision.
        query_timestamp: The requested timestamp.
        policy: The policy the lookup was evaluated under.
        query_observation_id: Observation the pose was requested for, if any.
    """

    rejection: LookupRejection
    detail: str
    query_timestamp: SourceTimestamp
    policy: LookupPolicy
    query_observation_id: SourceObservationId | None


PoseLookupResult = ResolvedPose | RejectedLookup
"""Either an accepted or a rejected lookup; consumers narrow with ``isinstance``."""


class TrajectoryLookup:
    """Resolves the pose of a trajectory's body at arbitrary timestamps.

    The trajectory's timestamps are indexed once, so repeated lookups over a
    long sequence stay cheap. Results depend only on the trajectory, the query
    and the policy, so identical inputs always reproduce identical results.
    """

    def __init__(self, trajectory: Trajectory) -> None:
        """Index a trajectory for lookup.

        Args:
            trajectory: The trajectory to resolve poses from.
        """
        self._trajectory = trajectory
        self._times_ns = tuple(pose.timestamp.total_nanoseconds() for pose in trajectory.poses)
        index_by_id = {pose.estimate_id: index for index, pose in enumerate(trajectory.poses)}
        self._gap_starts = frozenset(
            index_by_id[gap.previous_estimate_id] for gap in trajectory.gaps
        )

    @property
    def trajectory(self) -> Trajectory:
        """The trajectory poses are resolved from."""
        return self._trajectory

    def pose_at(self, timestamp: SourceTimestamp, *, policy: LookupPolicy) -> PoseLookupResult:
        """Resolve the pose at a timestamp.

        Args:
            timestamp: The requested time, in the trajectory's clock domain.
            policy: Which lookups are acceptable and with what tolerance.

        Returns:
            A :class:`ResolvedPose`, or a :class:`RejectedLookup` when the
            policy does not accept any pose for the timestamp.

        Raises:
            ClockDomainMismatchError: If ``timestamp`` is in another clock domain.
        """
        return self._resolve(timestamp, policy, query_observation_id=None)

    def pose_for_observation(
        self, observation: SourceObservation, *, policy: LookupPolicy
    ) -> PoseLookupResult:
        """Resolve the pose at a canonical observation's timestamp.

        Args:
            observation: Any canonical observation; its ``timestamp`` is used and
                its ``observation_id`` is recorded in the result.
            policy: Which lookups are acceptable and with what tolerance.

        Returns:
            A :class:`ResolvedPose` or a :class:`RejectedLookup`.

        Raises:
            ClockDomainMismatchError: If the observation is in another clock domain.
        """
        return self._resolve(
            observation.timestamp, policy, query_observation_id=observation.observation_id
        )

    def _resolve(
        self,
        timestamp: SourceTimestamp,
        policy: LookupPolicy,
        *,
        query_observation_id: SourceObservationId | None,
    ) -> PoseLookupResult:
        clock_id = self._trajectory.poses[0].timestamp.clock_id
        if timestamp.clock_id != clock_id:
            raise ClockDomainMismatchError(
                f"lookup timestamp is in clock domain {timestamp.clock_id!r} but the trajectory "
                f"uses {clock_id!r}; clock domains are never compared implicitly"
            )

        query_ns = timestamp.total_nanoseconds()
        times = self._times_ns
        poses = self._trajectory.poses
        index = bisect_left(times, query_ns)

        def resolved(
            pose: PoseEstimate,
            outcome: LookupOutcome,
            sources: tuple[PoseEstimateId, ...],
            delta_ns: int,
            fraction: float | None = None,
        ) -> ResolvedPose:
            return ResolvedPose(
                pose=pose,
                outcome=outcome,
                query_timestamp=timestamp,
                policy=policy,
                query_observation_id=query_observation_id,
                source_estimate_ids=sources,
                time_delta_ns=delta_ns,
                interpolation_fraction=fraction,
            )

        def rejected(reason: LookupRejection, detail: str) -> RejectedLookup:
            return RejectedLookup(
                rejection=reason,
                detail=detail,
                query_timestamp=timestamp,
                policy=policy,
                query_observation_id=query_observation_id,
            )

        if index < len(times) and times[index] == query_ns:
            pose = poses[index]
            return resolved(pose, LookupOutcome.EXACT, (pose.estimate_id,), 0)

        inside = times[0] < query_ns < times[-1]

        if policy.mode is LookupMode.EXACT:
            if inside:
                return rejected(
                    LookupRejection.NO_EXACT_MATCH, "no pose has the requested timestamp"
                )
            return rejected(LookupRejection.OUT_OF_RANGE, "timestamp is outside the trajectory")

        if policy.mode is LookupMode.NEAREST:
            candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
            # Empate resolvido a favor da pose mais antiga, para o resultado ser determinístico.
            nearest = min(candidates, key=lambda i: (abs(times[i] - query_ns), i))
            delta_ns = abs(times[nearest] - query_ns)
            assert policy.max_time_delta_ns is not None
            if delta_ns <= policy.max_time_delta_ns:
                pose = poses[nearest]
                return resolved(pose, LookupOutcome.NEAREST, (pose.estimate_id,), delta_ns)
            reason = LookupRejection.TOLERANCE_EXCEEDED if inside else LookupRejection.OUT_OF_RANGE
            return rejected(
                reason,
                f"closest pose is {delta_ns} ns away, tolerance is {policy.max_time_delta_ns} ns",
            )

        if not inside:
            return rejected(LookupRejection.OUT_OF_RANGE, "timestamp is outside the trajectory")
        lower, upper = index - 1, index
        interval_ns = times[upper] - times[lower]
        if lower in self._gap_starts:
            return rejected(
                LookupRejection.INTERPOLATION_GAP,
                f"neighboring poses are separated by a recorded gap of {interval_ns} ns",
            )
        if (
            policy.max_interpolation_gap_ns is not None
            and interval_ns > policy.max_interpolation_gap_ns
        ):
            return rejected(
                LookupRejection.INTERPOLATION_GAP,
                f"neighboring poses are {interval_ns} ns apart, tolerance is "
                f"{policy.max_interpolation_gap_ns} ns",
            )

        fraction = (query_ns - times[lower]) / interval_ns
        derived = self._interpolate(lower, upper, timestamp, fraction)
        delta_ns = min(query_ns - times[lower], times[upper] - query_ns)
        return resolved(
            derived,
            LookupOutcome.INTERPOLATED,
            (poses[lower].estimate_id, poses[upper].estimate_id),
            delta_ns,
            fraction,
        )

    def _interpolate(
        self, lower_index: int, upper_index: int, timestamp: SourceTimestamp, fraction: float
    ) -> PoseEstimate:
        lower = self._trajectory.poses[lower_index]
        upper = self._trajectory.poses[upper_index]
        translation: Vector3 = (
            (1.0 - fraction) * lower.translation_m[0] + fraction * upper.translation_m[0],
            (1.0 - fraction) * lower.translation_m[1] + fraction * upper.translation_m[1],
            (1.0 - fraction) * lower.translation_m[2] + fraction * upper.translation_m[2],
        )
        degraded = PoseValidity.DEGRADED in (lower.validity, upper.validity)
        sources = tuple(
            dict.fromkeys(
                lower.provenance.source_observation_ids + upper.provenance.source_observation_ids
            )
        )
        return PoseEstimate(
            estimate_id=PoseEstimateId(
                f"{self._trajectory.trajectory_id}--derived-{timestamp.total_nanoseconds()}"
            ),
            timestamp=timestamp,
            parent_frame=self._trajectory.reference_frame,
            child_frame=self._trajectory.body_frame,
            translation_m=translation,
            orientation=_slerp(lower.orientation, upper.orientation, fraction),
            validity=PoseValidity.DEGRADED if degraded else PoseValidity.VALID,
            provenance=PoseProvenance(
                source_observation_ids=sources,
                conversions_applied=(
                    f"interpolated between {lower.estimate_id} and {upper.estimate_id} "
                    f"(fraction={fraction:.9f})",
                ),
                derived_from=(lower.estimate_id, upper.estimate_id),
            ),
            # A incerteza de uma pose interpolada não é definida: fica ausente, não inventada.
            covariance=None,
        )


def _slerp(start: Quaternion, end: Quaternion, fraction: float) -> Quaternion:
    """Interpolate two unit quaternions along the shortest arc."""
    dot = sum(a * b for a, b in zip(start, end, strict=True))
    if dot < 0.0:
        # q e -q representam a mesma rotação; inverter um deles seleciona o arco curto.
        end = (-end[0], -end[1], -end[2], -end[3])
        dot = -dot
    if dot > _NLERP_DOT_THRESHOLD:
        blended = (
            (1.0 - fraction) * start[0] + fraction * end[0],
            (1.0 - fraction) * start[1] + fraction * end[1],
            (1.0 - fraction) * start[2] + fraction * end[2],
            (1.0 - fraction) * start[3] + fraction * end[3],
        )
        return normalize_quaternion(blended)
    theta = math.acos(min(dot, 1.0))
    sin_theta = math.sin(theta)
    weight_start = math.sin((1.0 - fraction) * theta) / sin_theta
    weight_end = math.sin(fraction * theta) / sin_theta
    return normalize_quaternion(
        (
            weight_start * start[0] + weight_end * end[0],
            weight_start * start[1] + weight_end * end[1],
            weight_start * start[2] + weight_end * end[2],
            weight_start * start[3] + weight_end * end[3],
        )
    )


@dataclass(frozen=True, kw_only=True)
class TemporalAlignmentSummary:
    """Run-level temporal-alignment metrics over a set of lookups.

    The time deltas cover accepted lookups only and are the distance from
    each query to its nearest contributing pose. They can later be correlated
    with camera-LiDAR reprojection quality to tell a temporal problem from a
    calibration problem.

    Attributes:
        lookup_count: Total lookups summarized.
        exact_count: Lookups resolved on an identical timestamp.
        nearest_count: Lookups resolved to the nearest pose.
        interpolated_count: Lookups resolved by interpolation.
        rejected_count: Lookups the policy rejected.
        rejections_by_reason: Rejected lookups per :class:`LookupRejection`.
        min_time_delta_ns: Smallest accepted time delta, or ``None`` if none.
        median_time_delta_ns: Lower median accepted time delta, or ``None``.
        max_time_delta_ns: Largest accepted time delta, or ``None``.
    """

    lookup_count: int
    exact_count: int
    nearest_count: int
    interpolated_count: int
    rejected_count: int
    rejections_by_reason: Mapping[LookupRejection, int]
    min_time_delta_ns: int | None
    median_time_delta_ns: int | None
    max_time_delta_ns: int | None


def summarize_lookups(results: Iterable[PoseLookupResult]) -> TemporalAlignmentSummary:
    """Summarize outcomes, rejections and time deltas of a set of lookups.

    Args:
        results: Lookup results, typically every pose request of one run.

    Returns:
        The temporal-alignment summary.
    """
    outcomes: Counter[LookupOutcome] = Counter()
    rejections: Counter[LookupRejection] = Counter()
    deltas: list[int] = []
    total = 0
    for result in results:
        total += 1
        if isinstance(result, ResolvedPose):
            outcomes[result.outcome] += 1
            deltas.append(result.time_delta_ns)
        else:
            rejections[result.rejection] += 1
    return TemporalAlignmentSummary(
        lookup_count=total,
        exact_count=outcomes[LookupOutcome.EXACT],
        nearest_count=outcomes[LookupOutcome.NEAREST],
        interpolated_count=outcomes[LookupOutcome.INTERPOLATED],
        rejected_count=sum(rejections.values()),
        rejections_by_reason=dict(rejections),
        min_time_delta_ns=min(deltas) if deltas else None,
        median_time_delta_ns=median_low(deltas) if deltas else None,
        max_time_delta_ns=max(deltas) if deltas else None,
    )

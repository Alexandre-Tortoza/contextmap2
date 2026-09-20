import math

import pytest
from pose_builders import (
    make_imu_observation,
    make_pose,
    make_trajectory,
    timestamp_ns,
)

from contextmap.state_estimation import (
    ClockDomainMismatchError,
    LookupMode,
    LookupOutcome,
    LookupPolicy,
    LookupRejection,
    PoseValidity,
    RejectedLookup,
    ResolvedPose,
    TrajectoryGap,
    TrajectoryLookup,
    summarize_lookups,
)

MS = 1_000_000


def _lookup(*, gap_between: tuple[int, int] | None = None) -> TrajectoryLookup:
    """Poses at 0, 100, 200, 300 ms with x = 0, 1, 2, 3 (identity orientation)."""
    poses = [make_pose(index) for index in range(4)]
    gaps = []
    if gap_between is not None:
        previous, following = gap_between
        gaps.append(
            TrajectoryGap(
                previous_estimate_id=poses[previous].estimate_id,
                next_estimate_id=poses[following].estimate_id,
                duration_ns=100 * MS,
            )
        )
    return TrajectoryLookup(make_trajectory(poses, gaps=gaps))


def _resolved(result: object) -> ResolvedPose:
    assert isinstance(result, ResolvedPose), result
    return result


def _rejected(result: object) -> RejectedLookup:
    assert isinstance(result, RejectedLookup), result
    return result


# --- Policy contract --------------------------------------------------------


def test_policy_constructors_declare_their_tolerances() -> None:
    assert LookupPolicy.exact().mode is LookupMode.EXACT
    nearest = LookupPolicy.nearest(max_time_delta_ns=5 * MS)
    assert nearest.mode is LookupMode.NEAREST
    assert nearest.max_time_delta_ns == 5 * MS
    interpolated = LookupPolicy.interpolated(max_interpolation_gap_ns=250 * MS)
    assert interpolated.mode is LookupMode.INTERPOLATED
    assert interpolated.max_interpolation_gap_ns == 250 * MS


@pytest.mark.parametrize(
    "build",
    [
        lambda: LookupPolicy(mode=LookupMode.EXACT, max_time_delta_ns=1),
        lambda: LookupPolicy(mode=LookupMode.NEAREST),
        lambda: LookupPolicy(mode=LookupMode.NEAREST, max_time_delta_ns=-1),
        lambda: LookupPolicy(mode=LookupMode.INTERPOLATED, max_interpolation_gap_ns=0),
        lambda: LookupPolicy(
            mode=LookupMode.INTERPOLATED, max_interpolation_gap_ns=1, max_time_delta_ns=1
        ),
    ],
)
def test_policy_rejects_tolerances_that_do_not_belong_to_its_mode(build: object) -> None:
    assert callable(build)
    with pytest.raises(ValueError, match="policy"):
        build()


# --- Exact ------------------------------------------------------------------


def test_exact_policy_returns_the_pose_at_the_identical_timestamp() -> None:
    lookup = _lookup()

    result = _resolved(lookup.pose_at(timestamp_ns(200 * MS), policy=LookupPolicy.exact()))

    assert result.outcome is LookupOutcome.EXACT
    assert result.pose == lookup.trajectory.poses[2]
    assert result.source_estimate_ids == (lookup.trajectory.poses[2].estimate_id,)
    assert result.time_delta_ns == 0
    assert result.interpolation_fraction is None


def test_exact_policy_rejects_a_timestamp_between_samples() -> None:
    result = _rejected(_lookup().pose_at(timestamp_ns(150 * MS), policy=LookupPolicy.exact()))

    assert result.rejection is LookupRejection.NO_EXACT_MATCH


def test_exact_policy_reports_out_of_range_requests() -> None:
    lookup = _lookup()

    before = _rejected(lookup.pose_at(timestamp_ns(-1), policy=LookupPolicy.exact()))
    after = _rejected(lookup.pose_at(timestamp_ns(300 * MS + 1), policy=LookupPolicy.exact()))

    assert before.rejection is LookupRejection.OUT_OF_RANGE
    assert after.rejection is LookupRejection.OUT_OF_RANGE


# --- Nearest ----------------------------------------------------------------


def test_nearest_policy_returns_the_closest_pose_within_tolerance() -> None:
    lookup = _lookup()

    result = _resolved(
        lookup.pose_at(
            timestamp_ns(130 * MS), policy=LookupPolicy.nearest(max_time_delta_ns=40 * MS)
        )
    )

    assert result.outcome is LookupOutcome.NEAREST
    assert result.pose == lookup.trajectory.poses[1]
    assert result.time_delta_ns == 30 * MS


def test_nearest_policy_breaks_ties_toward_the_earlier_pose() -> None:
    lookup = _lookup()

    result = _resolved(
        lookup.pose_at(
            timestamp_ns(150 * MS), policy=LookupPolicy.nearest(max_time_delta_ns=50 * MS)
        )
    )

    assert result.pose == lookup.trajectory.poses[1]


def test_nearest_policy_reports_an_exact_hit_as_exact() -> None:
    result = _resolved(
        _lookup().pose_at(timestamp_ns(100 * MS), policy=LookupPolicy.nearest(max_time_delta_ns=MS))
    )

    assert result.outcome is LookupOutcome.EXACT


def test_nearest_policy_rejects_when_the_closest_pose_exceeds_the_tolerance() -> None:
    result = _rejected(
        _lookup().pose_at(
            timestamp_ns(150 * MS), policy=LookupPolicy.nearest(max_time_delta_ns=40 * MS)
        )
    )

    assert result.rejection is LookupRejection.TOLERANCE_EXCEEDED


def test_nearest_policy_accepts_a_boundary_pose_just_outside_the_range() -> None:
    lookup = _lookup()
    policy = LookupPolicy.nearest(max_time_delta_ns=20 * MS)

    accepted = _resolved(lookup.pose_at(timestamp_ns(-10 * MS), policy=policy))
    rejected = _rejected(lookup.pose_at(timestamp_ns(400 * MS), policy=policy))

    assert accepted.pose == lookup.trajectory.poses[0]
    assert rejected.rejection is LookupRejection.OUT_OF_RANGE


# --- Interpolated -----------------------------------------------------------


def test_interpolated_policy_blends_translation_and_records_its_sources() -> None:
    lookup = _lookup()

    result = _resolved(lookup.pose_at(timestamp_ns(150 * MS), policy=LookupPolicy.interpolated()))

    poses = lookup.trajectory.poses
    assert result.outcome is LookupOutcome.INTERPOLATED
    assert result.pose.translation_m == pytest.approx((1.5, 0.0, 0.0))
    assert result.pose.timestamp == timestamp_ns(150 * MS)
    assert result.source_estimate_ids == (poses[1].estimate_id, poses[2].estimate_id)
    assert result.interpolation_fraction == pytest.approx(0.5)
    assert result.time_delta_ns == 50 * MS


def test_interpolated_policy_returns_the_original_pose_on_an_exact_hit() -> None:
    lookup = _lookup()

    result = _resolved(lookup.pose_at(timestamp_ns(200 * MS), policy=LookupPolicy.interpolated()))

    assert result.outcome is LookupOutcome.EXACT
    assert result.pose == lookup.trajectory.poses[2]


def test_interpolated_orientation_follows_the_shortest_arc() -> None:
    quarter_turn = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    lookup = TrajectoryLookup(
        make_trajectory([make_pose(0), make_pose(1, orientation=quarter_turn)])
    )

    result = _resolved(lookup.pose_at(timestamp_ns(50 * MS), policy=LookupPolicy.interpolated()))

    eighth_turn = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    assert result.pose.orientation == pytest.approx(eighth_turn)


def test_interpolated_orientation_treats_q_and_minus_q_as_the_same_rotation() -> None:
    lookup = TrajectoryLookup(
        make_trajectory([make_pose(0), make_pose(1, orientation=(0.0, 0.0, 0.0, -1.0))])
    )

    result = _resolved(lookup.pose_at(timestamp_ns(50 * MS), policy=LookupPolicy.interpolated()))

    assert abs(result.pose.orientation[3]) == pytest.approx(1.0)


def test_interpolated_orientation_of_identical_rotations_is_stable() -> None:
    result = _resolved(
        _lookup().pose_at(timestamp_ns(120 * MS), policy=LookupPolicy.interpolated())
    )

    assert result.pose.orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_interpolated_pose_is_distinguishable_from_estimated_poses() -> None:
    lookup = _lookup()
    poses = lookup.trajectory.poses

    derived = _resolved(
        lookup.pose_at(timestamp_ns(150 * MS), policy=LookupPolicy.interpolated())
    ).pose

    assert derived.provenance.derived_from == (poses[1].estimate_id, poses[2].estimate_id)
    assert poses[1].provenance.derived_from == ()
    assert derived.estimate_id not in {pose.estimate_id for pose in poses}
    assert derived.covariance is None
    assert derived.provenance.source_observation_ids == (
        poses[1].provenance.source_observation_ids + poses[2].provenance.source_observation_ids
    )
    assert any("interpolat" in note for note in derived.provenance.conversions_applied)
    assert (derived.parent_frame, derived.child_frame) == (
        poses[1].parent_frame,
        poses[1].child_frame,
    )


def test_interpolated_pose_is_degraded_when_a_source_pose_is_degraded() -> None:
    lookup = TrajectoryLookup(
        make_trajectory([make_pose(0), make_pose(1, validity=PoseValidity.DEGRADED)])
    )

    result = _resolved(lookup.pose_at(timestamp_ns(50 * MS), policy=LookupPolicy.interpolated()))

    assert result.pose.validity is PoseValidity.DEGRADED


def test_interpolation_across_a_recorded_gap_is_rejected() -> None:
    result = _rejected(
        _lookup(gap_between=(1, 2)).pose_at(
            timestamp_ns(150 * MS), policy=LookupPolicy.interpolated()
        )
    )

    assert result.rejection is LookupRejection.INTERPOLATION_GAP


def test_interpolation_is_rejected_when_the_bracket_exceeds_the_tolerance() -> None:
    lookup = TrajectoryLookup(make_trajectory([make_pose(0), make_pose(1, time_ns=1_000 * MS)]))
    query = timestamp_ns(500 * MS)

    rejected = _rejected(
        lookup.pose_at(query, policy=LookupPolicy.interpolated(max_interpolation_gap_ns=250 * MS))
    )
    accepted = _resolved(
        lookup.pose_at(query, policy=LookupPolicy.interpolated(max_interpolation_gap_ns=2_000 * MS))
    )

    assert rejected.rejection is LookupRejection.INTERPOLATION_GAP
    assert accepted.pose.translation_m == pytest.approx((0.5, 0.0, 0.0))


def test_interpolation_never_extrapolates() -> None:
    lookup = _lookup()

    result = _rejected(lookup.pose_at(timestamp_ns(301 * MS), policy=LookupPolicy.interpolated()))

    assert result.rejection is LookupRejection.OUT_OF_RANGE


# --- Clock domains, observations, determinism -------------------------------


def test_lookup_refuses_to_cross_clock_domains() -> None:
    with pytest.raises(ClockDomainMismatchError, match="clock"):
        _lookup().pose_at(
            timestamp_ns(100 * MS, clock_id="another:clock"), policy=LookupPolicy.exact()
        )


def test_pose_for_observation_uses_and_records_the_observation_identity() -> None:
    lookup = _lookup()
    observation = make_imu_observation("imu-0007", time_ns=150 * MS)

    result = _resolved(lookup.pose_for_observation(observation, policy=LookupPolicy.interpolated()))

    assert result.query_observation_id == observation.observation_id
    assert result.query_timestamp == observation.timestamp
    assert result.pose.translation_m == pytest.approx((1.5, 0.0, 0.0))


def test_identical_inputs_reproduce_identical_results() -> None:
    lookup = _lookup()
    query = timestamp_ns(170 * MS)
    policy = LookupPolicy.interpolated()

    assert lookup.pose_at(query, policy=policy) == lookup.pose_at(query, policy=policy)
    assert lookup.pose_at(query, policy=policy) == _lookup().pose_at(query, policy=policy)


# --- Run-level temporal alignment metrics -----------------------------------


def test_temporal_alignment_summary_separates_outcomes_and_rejections() -> None:
    lookup = _lookup(gap_between=(2, 3))
    interpolated = LookupPolicy.interpolated()
    results = [
        lookup.pose_at(timestamp_ns(100 * MS), policy=interpolated),  # exact
        lookup.pose_at(timestamp_ns(130 * MS), policy=interpolated),  # interpolated, delta 30
        lookup.pose_at(timestamp_ns(180 * MS), policy=interpolated),  # interpolated, delta 20
        lookup.pose_at(timestamp_ns(250 * MS), policy=interpolated),  # gap
        lookup.pose_at(timestamp_ns(900 * MS), policy=interpolated),  # out of range
        lookup.pose_at(
            timestamp_ns(140 * MS), policy=LookupPolicy.nearest(max_time_delta_ns=50 * MS)
        ),  # nearest, delta 40
    ]

    summary = summarize_lookups(results)

    assert summary.lookup_count == 6
    assert summary.exact_count == 1
    assert summary.interpolated_count == 2
    assert summary.nearest_count == 1
    assert summary.rejected_count == 2
    assert summary.rejections_by_reason == {
        LookupRejection.INTERPOLATION_GAP: 1,
        LookupRejection.OUT_OF_RANGE: 1,
    }
    assert summary.min_time_delta_ns == 0
    assert summary.median_time_delta_ns == 20 * MS
    assert summary.max_time_delta_ns == 40 * MS


def test_temporal_alignment_summary_of_no_lookups_is_empty() -> None:
    summary = summarize_lookups([])

    assert summary.lookup_count == 0
    assert summary.min_time_delta_ns is None
    assert summary.rejections_by_reason == {}

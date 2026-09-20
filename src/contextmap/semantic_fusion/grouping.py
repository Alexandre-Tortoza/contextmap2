"""Group spatial observations by the physical observation they were inferred from.

Repeated inference over one physical frame -- several runs, backends or prompts --
is *correlated* evidence about one observation, not several independent views.
:func:`group_by_physical_observation` keeps the two levels apart: it groups the
spatial observations of an explicit selection of perception runs by physical
:class:`~contextmap.ingestion.SourceObservationId` and reports the counts a fusion
policy needs to avoid turning repeated inference into artificial confidence.

It never picks a winner, never discards a repeated inference and assumes no
statistical independence between backends. Runs are never merged implicitly: the
caller names exactly the runs to group, and an observation from any other run is
an error.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion.models import PhysicalObservationGroup, _require_canonical
from contextmap.sensor_association import SpatialObservation, SpatialObservationId
from contextmap.shared import SourceTimestamp
from contextmap.visual_perception import PerceptionResultId, PerceptionRun, PerceptionRunId

PHYSICAL_OBSERVATION_GROUPING_POLICY_ID = "physical-observation-grouping-v1"
"""Versioned rule: one group per physical observation of one sequence, from selected runs."""


@dataclass(frozen=True, kw_only=True)
class PhysicalObservationGrouping:
    """Spatial observations grouped by physical observation, with the counts kept apart.

    Attributes:
        grouping_policy_id: The versioned grouping rule.
        selected_run_ids: The perception runs the caller selected, sorted and unique.
        groups: One group per physical observation, sorted by observation.
        inference_variant_count: Number of distinct backend identities among the runs that
            produced evidence; runs of one backend configuration are repeated inference,
            not new variants.
    """

    grouping_policy_id: str
    selected_run_ids: tuple[PerceptionRunId, ...]
    groups: tuple[PhysicalObservationGroup, ...]
    inference_variant_count: int

    def __post_init__(self) -> None:
        """Validate the ordering, that groups use only selected runs and the variant count.

        Raises:
            ValueError: If the policy is empty, a collection is not sorted and unique, a group
                uses a run outside ``selected_run_ids``, or the variant count is inconsistent
                with the runs that produced evidence.
        """
        if not self.grouping_policy_id.strip():
            raise ValueError("grouping_policy_id must not be empty")
        _require_canonical("selected_run_ids", self.selected_run_ids, lambda item: (item,))
        _require_canonical("groups", self.groups, lambda group: (group.physical_observation_id,))
        outside = set(self._used_run_ids()) - set(self.selected_run_ids)
        if outside:
            raise ValueError(
                f"groups use perception runs {sorted(outside)!r} that are not in "
                f"selected_run_ids {list(self.selected_run_ids)!r}"
            )
        runs = self.perception_run_count
        if not (0 if runs == 0 else 1) <= self.inference_variant_count <= runs:
            raise ValueError(
                f"inference_variant_count must be between 1 and the {runs} runs that produced "
                f"evidence (0 when there is none), got {self.inference_variant_count}"
            )

    @property
    def physical_observation_count(self) -> int:
        """Number of distinct physical observations."""
        return len(self.groups)

    @property
    def inference_result_count(self) -> int:
        """Number of inference results, correlated within each physical observation."""
        return sum(group.inference_result_count for group in self.groups)

    @property
    def perception_run_count(self) -> int:
        """Number of perception runs that produced evidence."""
        return len(self._used_run_ids())

    def _used_run_ids(self) -> set[PerceptionRunId]:
        return {run_id for group in self.groups for run_id in group.perception_run_ids}


def group_by_physical_observation(
    observations: Iterable[SpatialObservation],
    *,
    selected_runs: Sequence[PerceptionRun],
    acquisition_timestamps: Mapping[SourceObservationId, SourceTimestamp],
) -> PhysicalObservationGrouping:
    """Group spatial observations by physical observation under an explicit run selection.

    Args:
        observations: The spatial observations to group, in any order.
        selected_runs: The perception runs whose evidence may be grouped. Nothing outside
            this selection is ever merged in.
        acquisition_timestamps: When each physical observation was acquired, in one clock
            domain chosen by the caller (normally taken from the canonical sequence).

    Returns:
        The grouping; it does not depend on the order of ``observations`` or of
        ``selected_runs``.

    Raises:
        ValueError: If no run is selected, a run is selected twice or the runs process
            different sequences; an observation comes from an unselected run, from another
            sequence than its run, or is repeated; a run has two results for one physical
            observation, or one result describes two; or a physical observation has no
            acquisition timestamp.
    """
    runs = _index_selected_runs(selected_runs)
    seen: set[SpatialObservationId] = set()
    frame_of_result: dict[PerceptionResultId, SourceObservationId] = {}
    result_of_run_and_frame: dict[tuple[PerceptionRunId, SourceObservationId], PerceptionResultId]
    result_of_run_and_frame = {}
    by_frame: dict[SourceObservationId, list[SpatialObservation]] = {}

    for observation in observations:
        run_id = observation.provenance.perception_run_id
        run = runs.get(run_id)
        if run is None:
            raise ValueError(
                f"spatial observation {observation.spatial_observation_id!r} comes from "
                f"perception run {run_id!r}, which is not among the selected runs "
                f"{sorted(runs)!r}"
            )
        if observation.provenance.sequence_artifact_id != run.sequence_artifact_id:
            raise ValueError(
                f"spatial observation {observation.spatial_observation_id!r} belongs to "
                f"sequence artifact {observation.provenance.sequence_artifact_id!r}, but its "
                f"run {run_id!r} processed {run.sequence_artifact_id!r}"
            )
        if observation.spatial_observation_id in seen:
            raise ValueError(
                f"duplicate spatial observation {observation.spatial_observation_id!r}"
            )
        seen.add(observation.spatial_observation_id)

        frame = observation.source_observation_id
        result = observation.perception_result_id
        known_frame = frame_of_result.setdefault(result, frame)
        if known_frame != frame:
            raise ValueError(
                f"perception result {result!r} describes two physical observations, "
                f"{known_frame!r} and {frame!r}"
            )
        known_result = result_of_run_and_frame.setdefault((run_id, frame), result)
        if known_result != result:
            raise ValueError(
                f"perception run {run_id!r} has more than one perception result for physical "
                f"observation {frame!r}: {known_result!r} and {result!r}"
            )
        by_frame.setdefault(frame, []).append(observation)

    groups = tuple(
        _group_of(frame, by_frame[frame], acquisition_timestamps) for frame in sorted(by_frame)
    )
    used_runs = {run_id for group in groups for run_id in group.perception_run_ids}
    return PhysicalObservationGrouping(
        grouping_policy_id=PHYSICAL_OBSERVATION_GROUPING_POLICY_ID,
        selected_run_ids=tuple(sorted(runs)),
        groups=groups,
        inference_variant_count=len({_variant_of(runs[run_id]) for run_id in used_runs}),
    )


def _index_selected_runs(
    selected_runs: Sequence[PerceptionRun],
) -> dict[PerceptionRunId, PerceptionRun]:
    if not selected_runs:
        raise ValueError("a grouping needs at least one perception run selected explicitly")
    runs: dict[PerceptionRunId, PerceptionRun] = {}
    for run in selected_runs:
        if run.run_id in runs:
            raise ValueError(f"perception run {run.run_id!r} is selected more than once")
        runs[run.run_id] = run
    sequences = {run.sequence_artifact_id for run in runs.values()}
    if len(sequences) > 1:
        raise ValueError(
            f"selected perception runs must process one sequence artifact, "
            f"got {sorted(sequences)!r}"
        )
    return runs


def _group_of(
    frame: SourceObservationId,
    members: list[SpatialObservation],
    acquisition_timestamps: Mapping[SourceObservationId, SourceTimestamp],
) -> PhysicalObservationGroup:
    timestamp = acquisition_timestamps.get(frame)
    if timestamp is None:
        raise ValueError(f"no acquisition timestamp for physical observation {frame!r}")
    return PhysicalObservationGroup(
        physical_observation_id=frame,
        acquisition_timestamp=timestamp,
        spatial_observation_ids=tuple(sorted(item.spatial_observation_id for item in members)),
        perception_result_ids=tuple(sorted({item.perception_result_id for item in members})),
        perception_run_ids=tuple(sorted({item.provenance.perception_run_id for item in members})),
    )


def _variant_of(run: PerceptionRun) -> tuple[tuple[str, ...], ...]:
    """Identify a run by the backends it ran, not by its own identity."""
    return tuple(
        sorted(
            (
                capability,
                backend.backend_id,
                backend.provider,
                backend.model,
                backend.version,
                backend.configuration_fingerprint or "",
            )
            for capability, backend in run.backend_provenance.items()
        )
    )

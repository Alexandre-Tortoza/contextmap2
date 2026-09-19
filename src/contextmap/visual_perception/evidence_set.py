"""Read-only, multi-run evidence view over explicitly selected perception runs.

A researcher may process an entire sequence once, then rerun a
problematic frame range with a different backend/prompt, and want to
compare or later feed both into Semantic Fusion. :class:`PerceptionEvidenceSet`
groups the results of several *explicitly selected*
:class:`~contextmap.visual_perception.run_artifact.PerceptionRunReader`
runs by the physical
:class:`~contextmap.ingestion.SourceObservation` they were produced for,
without ever fusing, comparing, or otherwise interpreting that evidence —
see ``src/contextmap/visual_perception/docs/evidence_set.md`` for what
this view deliberately does not do (that belongs to Semantic Fusion).

Selection is always explicit: this module never infers "use every run
found" or "prefer the newest run". Excluding a run is simply not passing
it to :meth:`PerceptionEvidenceSet.open`; its artifact is never read,
modified, or deleted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.models import PerceptionResult, PerceptionRunId
from contextmap.visual_perception.run_artifact import PerceptionRunReader


class EvidenceSetError(Exception):
    """Raised when selected runs cannot be viewed together."""


@dataclass(frozen=True, kw_only=True)
class ObservationEvidence:
    """One physical observation's results across the selected runs.

    Attributes:
        source_observation_id: The physical observation.
        results_by_run: One :class:`~contextmap.visual_perception.models.PerceptionResult`
            per selected run that produced a result for this observation,
            keyed by ``run_id``. A run absent here simply produced no
            result for this observation (e.g. it was outside that run's
            selection) — this is never treated as a failure.
    """

    source_observation_id: SourceObservationId
    results_by_run: Mapping[PerceptionRunId, PerceptionResult]


class PerceptionEvidenceSet:
    """A read-only, structural view over explicitly selected perception runs.

    Grouping by ``SourceObservation`` never means the grouped results are
    independent physical observations, and this view never chooses a
    winning label, combines confidences, or associates regions across
    runs as the same object.
    """

    def __init__(self, runs: Sequence[PerceptionRunReader]) -> None:
        """Build a view over already-opened run readers.

        Args:
            runs: The runs to include, each already opened. Prefer
                :meth:`open` when starting from paths.

        Raises:
            EvidenceSetError: If ``runs`` is empty, or the selected runs
                reference more than one canonical sequence artifact,
                declare duplicate ``run_id`` values, or contain a result
                whose ``run_id`` disagrees with its owning manifest.
        """
        if not runs:
            raise EvidenceSetError("at least one run must be selected")

        sequence_ids = {run.manifest.sequence_artifact_id for run in runs}
        if len(sequence_ids) > 1:
            raise EvidenceSetError(
                f"selected runs reference incompatible sequence artifacts: {sorted(sequence_ids)}"
            )

        run_ids = [run.manifest.run_id for run in runs]
        if len(set(run_ids)) != len(run_ids):
            raise EvidenceSetError("selected runs contain a duplicate run_id")

        self._runs = tuple(runs)
        self._by_observation: dict[
            SourceObservationId, dict[PerceptionRunId, PerceptionResult]
        ] = {}
        for run in self._runs:
            for result in run.list_results():
                if result.run_id != run.manifest.run_id:
                    raise EvidenceSetError(
                        f"result run_id {result.run_id!r} does not match owning manifest "
                        f"run_id {run.manifest.run_id!r}"
                    )
                self._by_observation.setdefault(result.source_observation_id, {})[
                    run.manifest.run_id
                ] = result

    @classmethod
    def open(cls, run_dirs: Sequence[Path]) -> PerceptionEvidenceSet:
        """Open a view over the runs at the given directories.

        Args:
            run_dirs: Directories of the runs to select, each opened
                independently (no ``runs.json`` is read or required).

        Returns:
            The resulting view.

        Raises:
            EvidenceSetError: If ``run_dirs`` is empty, or the selected
                runs are not mutually compatible; see :meth:`__init__`.
        """
        return cls([PerceptionRunReader(run_dir) for run_dir in run_dirs])

    def run_ids(self) -> list[PerceptionRunId]:
        """Return the identities of every selected run.

        Returns:
            Run identities, in the order they were selected.
        """
        return [run.manifest.run_id for run in self._runs]

    def observation_ids(self) -> list[SourceObservationId]:
        """Return every physical observation with evidence in this view.

        Returns:
            Observation identities, sorted for deterministic iteration.
        """
        return sorted(self._by_observation.keys())

    def evidence_for(self, source_observation_id: SourceObservationId) -> ObservationEvidence:
        """Return the selected runs' evidence for one physical observation.

        Args:
            source_observation_id: The observation to look up.

        Returns:
            The grouped evidence. ``results_by_run`` is empty when no
            selected run produced a result for this observation — this
            is not an error.
        """
        results = self._by_observation.get(source_observation_id, {})
        return ObservationEvidence(
            source_observation_id=source_observation_id, results_by_run=dict(results)
        )

"""Execution of an experiment manifest: one run manifest per arm and one comparison.

The runner executes every arm of a validated :class:`ExperimentManifest`
through an injected :data:`ArmExecutor` (the runtime that resolves and runs a
topology is not this module's concern) and emits, immutably:

* one **run manifest** per arm with its resolved topology, backend, model,
  configuration, artifact and reference-set identities, and its outcome;
* one **evaluation report** per completed arm, in the common envelope;
* one **comparison manifest** that lists the arms' metrics side by side, proves
  which upstream artifacts they share, and states whether the comparison is
  complete.

A failed or unavailable arm stays explicit: it is recorded with the reason, has
no metrics, and makes the comparison incomplete. A result that does not match
the manifest (another reference set, another configuration, a recomputed pinned
artifact, a missing declared metric) is recorded as a failed arm, never as a
comparable one. There is no fallback and there is no winner: metrics are never
combined into one score. See ``src/contextmap/evaluation/docs/experiments.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

from contextmap.evaluation._persistence import canonical_digest, write_immutable_json
from contextmap.evaluation.experiments import (
    AblationMode,
    ExperimentArm,
    ExperimentError,
    ExperimentManifest,
    ExperimentVariable,
    FixedControl,
    ResolvedTopology,
    experiment_artifact_identity,
    validate_experiment_manifest,
    write_experiment,
)
from contextmap.evaluation.metrics import (
    EvaluationStage,
    MetricKind,
    MetricRegistry,
    MetricRegistryIdentity,
)
from contextmap.evaluation.reference_integrity import ValidatedReferenceSet
from contextmap.evaluation.reference_set import ReferenceSetIdentity
from contextmap.evaluation.report_schema import (
    ArtifactIdentity,
    EvaluationReport,
    MetricResult,
    MetricStatus,
    encode_evaluation_report,
    validate_evaluation_report,
    write_evaluation_report,
)

ARM_RUN_SCHEMA = "contextmap.experiment-arm-run/v1"
COMPARISON_SCHEMA = "contextmap.experiment-comparison/v1"
COMPARISON_FILENAME = "comparison.json"
ARM_RUN_FILENAME = "run.json"
ARM_REPORT_FILENAME = "report.json"
ARMS_DIRECTORY = "arms"


class ArmUnavailableError(Exception):
    """Raised by an executor when the backend, model or API an arm needs is not available."""


class IncompleteComparisonError(ExperimentError):
    """Raised when a comparison is used as complete while an arm failed or was unavailable."""


class ArmStatus(Enum):
    """How an arm ended."""

    COMPLETED = "completed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class FailureKind(Enum):
    """Why an arm did not complete."""

    UNAVAILABLE = "unavailable"
    """A required backend, model or external API was not available."""

    ERROR = "error"
    """The executor raised."""

    INVALID_RESULT = "invalid_result"
    """The executor returned a result that does not match the manifest."""


@dataclass(frozen=True, kw_only=True)
class StageArtifact:
    """The artifact a stage produced or reused in one arm."""

    stage_id: str
    artifact: ArtifactIdentity

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"stage_id": self.stage_id, "artifact": self.artifact.to_record()}


@dataclass(frozen=True, kw_only=True)
class ArmExecution:
    """What an executor returns for a completed arm.

    Attributes:
        report: The arm's report in the common envelope.
        stage_artifacts: One artifact per stage of the arm's topology: the
            pinned artifact for a reused stage, the produced one otherwise.
    """

    report: EvaluationReport
    stage_artifacts: tuple[StageArtifact, ...]


ArmExecutor: TypeAlias = Callable[[ExperimentManifest, ExperimentArm], ArmExecution]
"""Runs one arm and returns its report and artifacts, or raises.

Raise :class:`ArmUnavailableError` when a required backend or API is not
available; anything else that is raised is recorded as an error. Never fall
back to another backend: an unavailable arm stays unavailable.
"""


@dataclass(frozen=True, kw_only=True)
class ArmFailure:
    """Why an arm did not complete."""

    kind: FailureKind
    error_type: str | None
    message: str

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"kind": self.kind.value, "error_type": self.error_type, "message": self.message}


@dataclass(frozen=True, kw_only=True)
class ArmRunManifest:
    """The immutable record of one arm's run.

    Attributes:
        experiment: The exact experiment manifest the run belongs to.
        arm_id: The arm.
        assignments: The value of every variable in this arm.
        topology: The resolved topology the arm ran.
        status: Completed, failed or unavailable.
        failure: Why, when the arm did not complete.
        stage_artifacts: The artifact of every stage; empty when not completed.
        report_digest: Digest of the arm's report; ``None`` when not completed.
        reference_set: The reference set the run used.
        registry: The metric registry the run used.
    """

    experiment: ArtifactIdentity
    arm_id: str
    assignments: tuple[tuple[str, str], ...]
    topology: ResolvedTopology
    status: ArmStatus
    failure: ArmFailure | None
    stage_artifacts: tuple[StageArtifact, ...]
    report_digest: str | None
    reference_set: ReferenceSetIdentity
    registry: MetricRegistryIdentity

    @property
    def topology_digest(self) -> str:
        """Return the digest of the resolved topology."""
        return self.topology.digest()

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible content the digest is computed over."""
        return {
            "schema": ARM_RUN_SCHEMA,
            "experiment": self.experiment.to_record(),
            "arm_id": self.arm_id,
            "assignments": [list(item) for item in self.assignments],
            "topology": self.topology.to_record(),
            "topology_digest": self.topology_digest,
            "status": self.status.value,
            "failure": None if self.failure is None else self.failure.to_record(),
            "stage_artifacts": [item.to_record() for item in self.stage_artifacts],
            "report_digest": self.report_digest,
            "reference_set": self.reference_set.to_record(),
            "registry": self.registry.to_record(),
        }

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the run manifest."""
        return canonical_digest(self.to_record())


# ------------------------------------------------------------------------- comparison


@dataclass(frozen=True, kw_only=True)
class ComparisonArm:
    """One arm in the comparison, whether or not it completed."""

    arm_id: str
    assignments: tuple[tuple[str, str], ...]
    status: ArmStatus
    topology_digest: str
    failure: ArmFailure | None

    @property
    def comparable(self) -> bool:
        """Return whether the arm contributes metrics; only a completed arm does."""
        return self.status is ArmStatus.COMPLETED

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "arm_id": self.arm_id,
            "assignments": [list(item) for item in self.assignments],
            "status": self.status.value,
            "topology_digest": self.topology_digest,
            "comparable": self.comparable,
            "failure": None if self.failure is None else self.failure.to_record(),
        }


@dataclass(frozen=True, kw_only=True)
class SharedArtifact:
    """An artifact every completed arm reports for the same stage, identically."""

    stage_id: str
    artifact: ArtifactIdentity
    arm_ids: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "stage_id": self.stage_id,
            "artifact": self.artifact.to_record(),
            "arm_ids": list(self.arm_ids),
        }


@dataclass(frozen=True, kw_only=True)
class MetricEntry:
    """One arm's result for one metric and stratum, and its difference from the baseline."""

    arm_id: str
    status: MetricStatus
    value: float | None
    sample_count: int | None
    delta_from_baseline: float | None

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "arm_id": self.arm_id,
            "status": self.status.value,
            "value": self.value,
            "sample_count": self.sample_count,
            "delta_from_baseline": self.delta_from_baseline,
        }


@dataclass(frozen=True, kw_only=True)
class MetricComparison:
    """One declared metric (and stratum) across the completed arms.

    The metric stays a metric: entries are listed, differences are measured
    against the baseline for the same metric only, and nothing ranks arms.

    Attributes:
        metric: Metric name.
        version: Metric version.
        kind: Quality or performance.
        strata: The stratum, empty for the whole population.
        entries: One entry per completed arm that reported it.
        same_population: Whether every value was computed over the same number of samples.
    """

    metric: str
    version: str
    kind: MetricKind
    strata: tuple[tuple[str, str], ...]
    entries: tuple[MetricEntry, ...]
    same_population: bool

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "metric": self.metric,
            "version": self.version,
            "kind": self.kind.value,
            "strata": [list(item) for item in self.strata],
            "entries": [item.to_record() for item in self.entries],
            "same_population": self.same_population,
        }


@dataclass(frozen=True, kw_only=True)
class ComparisonManifest:
    """The comparison of an experiment's arms, with every identity it rests on.

    Attributes:
        experiment: The exact experiment manifest.
        reference_set: The reference set the arms were evaluated on.
        registry: The metric registry of the metrics.
        evaluated_stage: The stage every arm's quality measures.
        mode: The ablation mode.
        baseline_arm_id: The arm differences are measured against.
        physical_sample_count: Distinct physical samples; repetitions never add to it.
        repetitions_per_sample: Repeated inferences per sample (correlated evidence).
        variables: The declared variables.
        fixed_controls: Values held constant across arms.
        arms: Every arm, completed or not.
        shared_artifacts: Artifacts every completed arm reports identically.
        metrics: The declared metrics, side by side.
    """

    experiment: ArtifactIdentity
    reference_set: ReferenceSetIdentity
    registry: MetricRegistryIdentity
    evaluated_stage: EvaluationStage
    mode: AblationMode
    baseline_arm_id: str
    physical_sample_count: int
    repetitions_per_sample: int
    variables: tuple[ExperimentVariable, ...]
    fixed_controls: tuple[FixedControl, ...]
    arms: tuple[ComparisonArm, ...]
    shared_artifacts: tuple[SharedArtifact, ...]
    metrics: tuple[MetricComparison, ...]

    @property
    def complete(self) -> bool:
        """Return whether every arm completed."""
        return all(item.comparable for item in self.arms)

    @property
    def incomplete_arm_ids(self) -> tuple[str, ...]:
        """Return the arms that did not complete."""
        return tuple(item.arm_id for item in self.arms if not item.comparable)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible content the digest is computed over."""
        return {
            "schema": COMPARISON_SCHEMA,
            "experiment": self.experiment.to_record(),
            "reference_set": self.reference_set.to_record(),
            "registry": self.registry.to_record(),
            "evaluated_stage": self.evaluated_stage.value,
            "mode": self.mode.value,
            "baseline_arm_id": self.baseline_arm_id,
            "physical_sample_count": self.physical_sample_count,
            "repetitions_per_sample": self.repetitions_per_sample,
            "variables": [item.to_record() for item in self.variables],
            "fixed_controls": [item.to_record() for item in self.fixed_controls],
            "complete": self.complete,
            "arms": [item.to_record() for item in self.arms],
            "shared_artifacts": [item.to_record() for item in self.shared_artifacts],
            "metrics": [item.to_record() for item in self.metrics],
        }

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the comparison."""
        return canonical_digest(self.to_record())


def require_complete_comparison(comparison: ComparisonManifest) -> None:
    """Refuse a comparison in which any arm failed or was unavailable.

    Raises:
        IncompleteComparisonError: Naming the arms that did not complete.
    """
    if not comparison.complete:
        failed = {item.arm_id: item for item in comparison.arms if not item.comparable}
        reasons = "; ".join(
            f"{arm_id} ({item.status.value}: "
            f"{'' if item.failure is None else item.failure.message})"
            for arm_id, item in failed.items()
        )
        raise IncompleteComparisonError(f"the comparison is incomplete: {reasons}")


def _strata_key(result: MetricResult) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(result.strata))


def _metric_comparisons(
    manifest: ExperimentManifest,
    completed: list[str],
    reports: Mapping[str, EvaluationReport],
) -> tuple[MetricComparison, ...]:
    comparisons: list[MetricComparison] = []
    declared = [(item, MetricKind.QUALITY) for item in manifest.quality_metrics] + [
        (item, MetricKind.PERFORMANCE) for item in manifest.resource_capture
    ]
    for reference, kind in declared:
        per_arm: dict[str, dict[tuple[tuple[str, str], ...], MetricResult]] = {}
        for arm_id in completed:
            report = reports[arm_id]
            results = (
                report.quality_metrics if kind is MetricKind.QUALITY else report.performance_metrics
            )
            per_arm[arm_id] = {
                _strata_key(item): item
                for item in results
                if item.metric == reference.name and item.metric_version == reference.version
            }
        strata = sorted({key for found in per_arm.values() for key in found})
        for key in strata:
            baseline = per_arm.get(manifest.baseline_arm_id, {}).get(key)
            entries: list[MetricEntry] = []
            for arm_id in completed:
                result = per_arm[arm_id].get(key)
                if result is None:
                    continue
                delta: float | None = None
                if (
                    arm_id != manifest.baseline_arm_id
                    and result.status is MetricStatus.VALUE
                    and baseline is not None
                    and baseline.status is MetricStatus.VALUE
                    and result.value is not None
                    and baseline.value is not None
                ):
                    delta = result.value - baseline.value
                entries.append(
                    MetricEntry(
                        arm_id=arm_id,
                        status=result.status,
                        value=result.value,
                        sample_count=result.sample_count,
                        delta_from_baseline=delta,
                    )
                )
            counts = {item.sample_count for item in entries if item.status is MetricStatus.VALUE}
            comparisons.append(
                MetricComparison(
                    metric=reference.name,
                    version=reference.version,
                    kind=kind,
                    strata=key,
                    entries=tuple(entries),
                    same_population=len(counts) <= 1,
                )
            )
    return tuple(comparisons)


def _shared_artifacts(
    manifest: ExperimentManifest, arm_runs: list[ArmRunManifest]
) -> tuple[SharedArtifact, ...]:
    completed = [item for item in arm_runs if item.status is ArmStatus.COMPLETED]
    if len(completed) < 2:
        return ()
    stage_ids = list(
        dict.fromkeys(item.stage_id for arm in manifest.arms for item in arm.topology.stages)
    )
    shared: list[SharedArtifact] = []
    for stage_id in stage_ids:
        identities = []
        for item in completed:
            found = {entry.stage_id: entry.artifact for entry in item.stage_artifacts}
            if stage_id not in found:
                break
            identities.append(found[stage_id])
        else:
            if all(identity == identities[0] for identity in identities):
                shared.append(
                    SharedArtifact(
                        stage_id=stage_id,
                        artifact=identities[0],
                        arm_ids=tuple(item.arm_id for item in completed),
                    )
                )
    return tuple(shared)


def build_comparison(
    manifest: ExperimentManifest,
    arm_runs: list[ArmRunManifest],
    reports: Mapping[str, EvaluationReport],
) -> ComparisonManifest:
    """Build the comparison of an experiment's arms.

    Only completed arms contribute metrics and shared artifacts; the others are
    listed with their reason and make the comparison incomplete.
    """
    completed = [item.arm_id for item in arm_runs if item.status is ArmStatus.COMPLETED]
    return ComparisonManifest(
        experiment=experiment_artifact_identity(manifest),
        reference_set=manifest.selection.reference_set,
        registry=manifest.registry,
        evaluated_stage=manifest.evaluated_stage,
        mode=manifest.mode,
        baseline_arm_id=manifest.baseline_arm_id,
        physical_sample_count=manifest.physical_sample_count,
        repetitions_per_sample=manifest.repetitions_per_sample,
        variables=manifest.variables,
        fixed_controls=manifest.fixed_controls,
        arms=tuple(
            ComparisonArm(
                arm_id=item.arm_id,
                assignments=item.assignments,
                status=item.status,
                topology_digest=item.topology_digest,
                failure=item.failure,
            )
            for item in arm_runs
        ),
        shared_artifacts=_shared_artifacts(manifest, arm_runs),
        metrics=_metric_comparisons(manifest, completed, reports),
    )


# ---------------------------------------------------------------------------- running


def _check_execution(
    manifest: ExperimentManifest,
    arm: ExperimentArm,
    execution: ArmExecution,
    registry: MetricRegistry,
) -> str | None:
    """Return the first way a result contradicts the manifest, or ``None``."""
    report = execution.report
    metadata = report.reproducibility
    if report.stage is not manifest.evaluated_stage:
        return (
            f"the report evaluates stage {report.stage.value!r}, not the experiment's "
            f"{manifest.evaluated_stage.value!r}"
        )
    if metadata.reference_set != manifest.selection.reference_set:
        return "the report cites another reference set than the experiment's selection"
    if metadata.metric_registry != manifest.registry:
        return "the report cites another metric registry than the experiment's"
    if metadata.configuration_digest != arm.topology.digest():
        return "the report's configuration digest is not the digest of the arm's resolved topology"
    if experiment_artifact_identity(manifest) not in metadata.input_artifacts:
        return "the report does not cite this exact experiment manifest among its inputs"
    try:
        validate_evaluation_report(report, registry)
    except ValueError as error:
        return f"the report is inconsistent with the metric registry: {error}"
    for reference in manifest.quality_metrics:
        if not any(
            item.metric == reference.name and item.metric_version == reference.version
            for item in report.quality_metrics
        ):
            return f"the report lacks the declared quality metric {reference.key}"
    for reference in manifest.resource_capture:
        if not any(
            item.metric == reference.name and item.metric_version == reference.version
            for item in report.performance_metrics
        ):
            return f"the report lacks the declared resource metric {reference.key}"
    reported: dict[str, ArtifactIdentity] = {}
    for item in execution.stage_artifacts:
        if item.stage_id in reported:
            return f"stage artifacts name stage {item.stage_id!r} twice"
        if item.artifact.digest is None:
            return f"the artifact of stage {item.stage_id!r} has no digest"
        reported[item.stage_id] = item.artifact
    for stage in arm.topology.stages:
        if stage.stage_id not in reported:
            return f"stage artifacts do not cover stage {stage.stage_id!r}"
        if stage.artifact is not None and reported[stage.stage_id] != stage.artifact:
            return (
                f"stage {stage.stage_id!r} is pinned to artifact {stage.artifact.artifact_id!r} "
                f"but the executor reported {reported[stage.stage_id].artifact_id!r}: the pinned "
                "artifact was not reused"
            )
    unknown = sorted(set(reported) - {item.stage_id for item in arm.topology.stages})
    if unknown:
        return f"stage artifacts name stages outside the arm's topology: {unknown}"
    return None


def _run_arm(
    manifest: ExperimentManifest,
    arm: ExperimentArm,
    executor: ArmExecutor,
    registry: MetricRegistry,
) -> tuple[ArmRunManifest, EvaluationReport | None]:
    def record(
        status: ArmStatus,
        failure: ArmFailure | None,
        execution: ArmExecution | None = None,
    ) -> ArmRunManifest:
        return ArmRunManifest(
            experiment=experiment_artifact_identity(manifest),
            arm_id=arm.arm_id,
            assignments=arm.assignments,
            topology=arm.topology,
            status=status,
            failure=failure,
            stage_artifacts=() if execution is None else execution.stage_artifacts,
            report_digest=(
                None
                if execution is None
                else canonical_digest(encode_evaluation_report(execution.report))
            ),
            reference_set=manifest.selection.reference_set,
            registry=manifest.registry,
        )

    try:
        execution = executor(manifest, arm)
    except ArmUnavailableError as error:
        failure = ArmFailure(kind=FailureKind.UNAVAILABLE, error_type=None, message=str(error))
        return record(ArmStatus.UNAVAILABLE, failure), None
    except Exception as error:
        failure = ArmFailure(
            kind=FailureKind.ERROR, error_type=type(error).__name__, message=str(error)
        )
        return record(ArmStatus.FAILED, failure), None
    problem = _check_execution(manifest, arm, execution, registry)
    if problem is not None:
        failure = ArmFailure(kind=FailureKind.INVALID_RESULT, error_type=None, message=problem)
        return record(ArmStatus.FAILED, failure), None
    return record(ArmStatus.COMPLETED, None, execution), execution.report


@dataclass(frozen=True, kw_only=True)
class ExperimentRun:
    """The outcome of running an experiment.

    Attributes:
        manifest: The experiment that ran.
        arm_runs: One run manifest per arm, in the manifest's order.
        reports: The report of every completed arm, by arm id.
        comparison: The comparison of the arms.
    """

    manifest: ExperimentManifest
    arm_runs: tuple[ArmRunManifest, ...]
    reports: Mapping[str, EvaluationReport]
    comparison: ComparisonManifest


def run_experiment(
    manifest: ExperimentManifest,
    *,
    executor: ArmExecutor,
    registry: MetricRegistry,
    reference_set: ValidatedReferenceSet,
    root: Path,
) -> ExperimentRun:
    """Run every arm of an experiment and emit the run manifests and the comparison.

    The reference set must have passed integrity validation (that is what a
    :class:`ValidatedReferenceSet` proves), the manifest must match it and the
    registry, and ``root`` must be a new directory: re-execution creates a new
    identity, it never overwrites a previous run. The baseline arm runs first;
    a failing arm never stops the others.

    Raises:
        ExperimentError: If the manifest does not match the reference set or
            the registry, or is a tuning experiment on the held-out split.
        MetricRegistryError: If a declared metric is unknown.
        MetricCompatibilityError: If the reference set lacks a metric's annotations.
        FileExistsError: If ``root`` already holds a run.
    """
    validate_experiment_manifest(manifest, reference_set=reference_set.manifest, registry=registry)
    write_experiment(root, manifest)
    order = [manifest.baseline_arm] + [
        item for item in manifest.arms if item.arm_id != manifest.baseline_arm_id
    ]
    outcomes = {item.arm_id: _run_arm(manifest, item, executor, registry) for item in order}
    arm_runs = [outcomes[item.arm_id][0] for item in manifest.arms]
    reports = {arm_id: report for arm_id, (_, report) in outcomes.items() if report is not None}
    for run in arm_runs:
        directory = root / ARMS_DIRECTORY / run.arm_id
        write_immutable_json(directory / ARM_RUN_FILENAME, encode_arm_run(run), "arm run manifest")
        if run.arm_id in reports:
            write_evaluation_report(directory / ARM_REPORT_FILENAME, reports[run.arm_id])
    comparison = build_comparison(manifest, arm_runs, reports)
    write_immutable_json(
        root / COMPARISON_FILENAME, encode_comparison(comparison), "comparison manifest"
    )
    return ExperimentRun(
        manifest=manifest, arm_runs=tuple(arm_runs), reports=reports, comparison=comparison
    )


# ------------------------------------------------------------------------ persistence


def encode_arm_run(run: ArmRunManifest) -> dict[str, Any]:
    """Return the run manifest document, including its own digest."""
    return {**run.to_record(), "digest": run.digest()}


def encode_comparison(comparison: ComparisonManifest) -> dict[str, Any]:
    """Return the comparison document, including its own digest."""
    return {**comparison.to_record(), "digest": comparison.digest()}


def read_verified_document(path: Path) -> dict[str, Any]:
    """Read a run manifest or comparison and verify its digest.

    Raises:
        ExperimentError: If the file is not valid JSON, has no digest or was modified.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ExperimentError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(document, dict) or "digest" not in document:
        raise ExperimentError(f"{path.name} is not a digested experiment document")
    content = {key: value for key, value in document.items() if key != "digest"}
    if canonical_digest(content) != document["digest"]:
        raise ExperimentError(
            f"{path.name} digest mismatch: the document was modified after it was written"
        )
    return document

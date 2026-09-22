"""Versioned experiment manifests and ablation matrices for controlled comparisons.

Comparing models, policies, optional evidence channels or optional DAG stages is
only meaningful when everything except the declared variable is identical. An
:class:`ExperimentManifest` therefore binds, in one hashed document, the
reference-set version and the exact ordered sample selection, the base
configuration and the *resolved* topology of every arm, the exact immutable
upstream artifacts to reuse, the variables under test with their variants, the
fixed controls, the metrics to report and the resource capture policy.

Two rules make a manifest a controlled comparison, and are enforced when it is
built, so an invalid experiment cannot exist:

* **only declared variables vary**: every difference between an arm and the
  baseline (a stage inserted or removed, a dependency rewired, another backend,
  another configuration, another pinned artifact) must be covered by a variable
  the arm assigns a non-baseline value to, and of a kind that allows it;
* **the varied part consumes exact immutable inputs**: every upstream
  dependency of a stage a variable touches is a pinned artifact, so all arms
  provably share it.

Resolving a configuration into a DAG is the runtime's job; this module only
verifies the resolved topologies it is given. See
``src/contextmap/evaluation/docs/experiments.md``.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.evaluation._persistence import canonical_digest, write_immutable_json
from contextmap.evaluation._validation import require_sha256, require_text, require_unique
from contextmap.evaluation.metrics import (
    EvaluationStage,
    MetricKind,
    MetricRegistry,
    MetricRegistryIdentity,
    require_annotation_compatibility,
)
from contextmap.evaluation.reference_set import (
    ReferenceSampleId,
    ReferenceSetError,
    ReferenceSetIdentity,
    ReferenceSetManifest,
    SplitRole,
)
from contextmap.evaluation.report_schema import ArtifactIdentity

EXPERIMENT_SCHEMA = "contextmap.experiment/v1"
"""Schema identifier of the experiment manifest document."""

EXPERIMENT_FILENAME = "experiment.json"
"""File name of the manifest inside an experiment run directory."""

_ARM_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")


class ExperimentError(ValueError):
    """Raised when an experiment cannot be trusted as a controlled comparison."""


class VariationKind(Enum):
    """What a variable is allowed to change in the stages it touches."""

    BACKEND = "backend"
    """The backend or model of a stage (and the configuration that comes with it)."""

    POLICY = "policy"
    """A policy of a stage, expressed through its configuration."""

    CONFIGURATION = "configuration"
    """Any other configuration of a stage."""

    EVIDENCE_CHANNELS = "evidence_channels"
    """The subset of evidence channels a stage consumes, expressed through its configuration."""

    TOPOLOGY = "topology"
    """Inserting or removing a stage, or rewiring its dependencies."""


_CONFIGURATION_KINDS = frozenset(
    {
        VariationKind.BACKEND,
        VariationKind.POLICY,
        VariationKind.CONFIGURATION,
        VariationKind.EVIDENCE_CHANNELS,
    }
)


class AblationMode(Enum):
    """How the variables' values are combined into arms."""

    ONE_AT_A_TIME = "one_at_a_time"
    """The baseline plus, for each variable and non-baseline value, one arm varying only it."""

    FULL_FACTORIAL = "full_factorial"
    """Every combination of every variable's values."""


class ExperimentPurpose(Enum):
    """Whether the experiment tunes something or only measures it."""

    TUNING = "tuning"
    EVALUATION = "evaluation"


# ----------------------------------------------------------------------------- topology


@dataclass(frozen=True, kw_only=True)
class StageImplementation:
    """The exact backend, model and configuration behind one stage.

    Attributes:
        backend_id: Backend or adapter identity.
        backend_version: Backend version.
        model: Model or checkpoint identity, when the backend has one.
        configuration_digest: ``sha256:<hex>`` of the stage's effective configuration.
    """

    backend_id: str
    backend_version: str
    model: str | None
    configuration_digest: str

    def __post_init__(self) -> None:
        """Require identity and a valid configuration digest."""
        require_text("backend_id", self.backend_id)
        require_text("backend_version", self.backend_version)
        require_sha256("configuration_digest", self.configuration_digest)

    @property
    def backend(self) -> tuple[str, str, str | None]:
        """Return backend, version and model: what changes when the backend changes."""
        return (self.backend_id, self.backend_version, self.model)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "model": self.model,
            "configuration_digest": self.configuration_digest,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> StageImplementation:
        """Rebuild an implementation from :meth:`to_record` output."""
        return cls(
            backend_id=record["backend_id"],
            backend_version=record["backend_version"],
            model=record["model"],
            configuration_digest=record["configuration_digest"],
        )


@dataclass(frozen=True, kw_only=True)
class TopologyStage:
    """One stage of a resolved DAG.

    Attributes:
        stage_id: Identity of the stage inside the topology.
        capability: The capability that owns the stage; never changes across arms.
        implementation: Backend, model and configuration of the stage.
        depends_on: The stages whose outputs it consumes.
        artifact: The exact immutable artifact reused instead of running the
            stage. A pinned artifact needs a digest.
    """

    stage_id: str
    capability: str
    implementation: StageImplementation
    depends_on: tuple[str, ...] = ()
    artifact: ArtifactIdentity | None = None

    def __post_init__(self) -> None:
        """Require identity, unique dependencies and a digest on a pinned artifact."""
        require_text("stage_id", self.stage_id)
        require_text("capability", self.capability)
        require_unique(f"dependency of stage {self.stage_id!r}", self.depends_on)
        if self.artifact is not None and self.artifact.digest is None:
            raise ValueError(
                f"the pinned artifact of stage {self.stage_id!r} needs a digest: "
                "an artifact without one cannot prove it is the same immutable artifact"
            )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "stage_id": self.stage_id,
            "capability": self.capability,
            "implementation": self.implementation.to_record(),
            "depends_on": list(self.depends_on),
            "artifact": None if self.artifact is None else self.artifact.to_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> TopologyStage:
        """Rebuild a stage from :meth:`to_record` output."""
        artifact = record["artifact"]
        return cls(
            stage_id=record["stage_id"],
            capability=record["capability"],
            implementation=StageImplementation.from_record(record["implementation"]),
            depends_on=tuple(record["depends_on"]),
            artifact=None if artifact is None else ArtifactIdentity.from_record(artifact),
        )


@dataclass(frozen=True, kw_only=True)
class ResolvedTopology:
    """A resolved stage DAG: what actually runs, with what, on what inputs."""

    stages: tuple[TopologyStage, ...]

    def __post_init__(self) -> None:
        """Require unique stages and dependencies that exist and form no cycle."""
        require_unique("stage id", (item.stage_id for item in self.stages))
        known = {item.stage_id for item in self.stages}
        for item in self.stages:
            for dependency in item.depends_on:
                if dependency == item.stage_id:
                    raise ValueError(f"stage {item.stage_id!r} depends on itself")
                if dependency not in known:
                    raise ValueError(
                        f"stage {item.stage_id!r} depends on unknown stage {dependency!r}"
                    )
        remaining = {item.stage_id: set(item.depends_on) for item in self.stages}
        while remaining:
            ready = [name for name, pending in remaining.items() if not pending]
            if not ready:
                raise ValueError(f"the topology has a dependency cycle among {sorted(remaining)}")
            for name in ready:
                del remaining[name]
            for pending in remaining.values():
                pending.difference_update(ready)

    def stage(self, stage_id: str) -> TopologyStage:
        """Return one stage by identity."""
        for item in self.stages:
            if item.stage_id == stage_id:
                return item
        raise ExperimentError(f"the topology has no stage {stage_id!r}")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record the digest is computed over."""
        return {"stages": [item.to_record() for item in self.stages]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ResolvedTopology:
        """Rebuild a topology from :meth:`to_record` output."""
        return cls(stages=tuple(TopologyStage.from_record(item) for item in record["stages"]))

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the whole resolved topology."""
        return canonical_digest(self.to_record())


# ------------------------------------------------------------------ variables and arms


@dataclass(frozen=True, kw_only=True)
class ExperimentVariable:
    """A variable under test, with the stages it may change and its variants.

    Attributes:
        name: Identity of the variable.
        kind: What kind of change the variable stands for.
        touches: Stages the variable may insert, remove, rewire or reconfigure.
        values: The variants; each arm assigns exactly one.
        baseline_value: The variant the baseline arm holds.
        description: What is being compared and why.
    """

    name: str
    kind: VariationKind
    touches: tuple[str, ...]
    values: tuple[str, ...]
    baseline_value: str
    description: str = ""

    def __post_init__(self) -> None:
        """Require at least two unique values, one of which is the baseline."""
        require_text("variable name", self.name)
        if not self.touches:
            raise ValueError(f"variable {self.name!r} must touch at least one stage")
        require_unique(f"stage touched by variable {self.name!r}", self.touches)
        if len(self.values) < 2:
            raise ValueError(f"variable {self.name!r} needs at least two values to vary")
        for value in self.values:
            require_text("variable value", value)
        require_unique(f"value of variable {self.name!r}", self.values)
        if self.baseline_value not in self.values:
            raise ValueError(
                f"the baseline value {self.baseline_value!r} of variable {self.name!r} "
                "is not one of its values"
            )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "touches": list(self.touches),
            "values": list(self.values),
            "baseline_value": self.baseline_value,
            "description": self.description,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ExperimentVariable:
        """Rebuild a variable from :meth:`to_record` output."""
        return cls(
            name=record["name"],
            kind=VariationKind(record["kind"]),
            touches=tuple(record["touches"]),
            values=tuple(record["values"]),
            baseline_value=record["baseline_value"],
            description=record["description"],
        )


@dataclass(frozen=True, kw_only=True)
class ExperimentArm:
    """One cell of the ablation matrix, with its resolved topology.

    Attributes:
        arm_id: Identity of the arm; also its directory name.
        assignments: ``(variable, value)`` for every variable.
        topology: The resolved DAG the arm runs.
        description: Free text.
    """

    arm_id: str
    assignments: tuple[tuple[str, str], ...]
    topology: ResolvedTopology
    description: str = ""

    def __post_init__(self) -> None:
        """Require a path-safe identity and one value per variable."""
        if not _ARM_ID.fullmatch(self.arm_id):
            raise ValueError(
                f"arm id {self.arm_id!r} must be lowercase letters, digits, '.', '_' or '-'"
            )
        require_unique(
            f"variable assigned by arm {self.arm_id!r}", (n for n, _ in self.assignments)
        )

    @property
    def assignment(self) -> dict[str, str]:
        """Return the assignments as a mapping."""
        return dict(self.assignments)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "arm_id": self.arm_id,
            "assignments": [list(item) for item in self.assignments],
            "topology": self.topology.to_record(),
            "description": self.description,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ExperimentArm:
        """Rebuild an arm from :meth:`to_record` output."""
        return cls(
            arm_id=record["arm_id"],
            assignments=tuple((name, value) for name, value in record["assignments"]),
            topology=ResolvedTopology.from_record(record["topology"]),
            description=record["description"],
        )


def ablation_cells(
    variables: tuple[ExperimentVariable, ...], mode: AblationMode
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Return the assignments of every arm of an ablation matrix, baseline first.

    The order is deterministic: variables and values in declaration order.
    """
    baseline = tuple((item.name, item.baseline_value) for item in variables)
    if mode is AblationMode.ONE_AT_A_TIME:
        cells = [baseline]
        for index, variable in enumerate(variables):
            for value in variable.values:
                if value == variable.baseline_value:
                    continue
                varied = list(baseline)
                varied[index] = (variable.name, value)
                cells.append(tuple(varied))
        return tuple(cells)
    combinations = [
        tuple(
            (variable.name, value) for variable, value in zip(variables, combination, strict=True)
        )
        for combination in itertools.product(*(item.values for item in variables))
    ]
    combinations.remove(baseline)
    return (baseline, *combinations)


# ----------------------------------------------------------------------- selection etc.


@dataclass(frozen=True, kw_only=True)
class SelectionBinding:
    """The exact ordered samples an experiment runs on, bound to a reference set.

    Attributes:
        reference_set: Identity (id, version, digest) of the reference set.
        scheme_id: The split scheme the selection comes from.
        split: The split of that scheme.
        role: The role of the split; a tuning experiment never uses ``TEST``.
        sample_ids: The ordered samples of the split.
    """

    reference_set: ReferenceSetIdentity
    scheme_id: str
    split: str
    role: SplitRole
    sample_ids: tuple[ReferenceSampleId, ...]

    def __post_init__(self) -> None:
        """Require a scheme, a split and at least one unique sample."""
        require_text("scheme_id", self.scheme_id)
        require_text("split", self.split)
        if not self.sample_ids:
            raise ValueError("an experiment selection needs at least one sample")
        require_unique("selected sample", self.sample_ids)

    @classmethod
    def for_split(
        cls, reference_set: ReferenceSetManifest, scheme_id: str, split: str
    ) -> SelectionBinding:
        """Bind an experiment to one split of a reference set.

        Raises:
            ReferenceSetError: If the scheme or the split does not exist.
        """
        samples = reference_set.selection(scheme_id, split)
        scheme = next(item for item in reference_set.split_schemes if item.scheme_id == scheme_id)
        role = next(item.role for item in scheme.splits if item.name == split)
        return cls(
            reference_set=reference_set.identity(),
            scheme_id=scheme_id,
            split=split,
            role=role,
            sample_ids=tuple(item.sample_id for item in samples),
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "reference_set": self.reference_set.to_record(),
            "scheme_id": self.scheme_id,
            "split": self.split,
            "role": self.role.value,
            "sample_ids": list(self.sample_ids),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SelectionBinding:
        """Rebuild a selection from :meth:`to_record` output."""
        return cls(
            reference_set=ReferenceSetIdentity.from_record(record["reference_set"]),
            scheme_id=record["scheme_id"],
            split=record["split"],
            role=SplitRole(record["role"]),
            sample_ids=tuple(ReferenceSampleId(item) for item in record["sample_ids"]),
        )


@dataclass(frozen=True, kw_only=True)
class FixedControl:
    """A value held constant across every arm (prompt version, seed, evidence variant, ...)."""

    name: str
    value: str

    def __post_init__(self) -> None:
        """Require a name and a value."""
        require_text("control name", self.name)
        require_text("control value", self.value)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FixedControl:
        """Rebuild a control from :meth:`to_record` output."""
        return cls(name=record["name"], value=record["value"])


@dataclass(frozen=True, kw_only=True)
class MetricRef:
    """A metric of the registry the experiment reports, by name and version."""

    name: str
    version: str

    def __post_init__(self) -> None:
        """Require both parts."""
        require_text("metric name", self.name)
        require_text("metric version", self.version)

    @property
    def key(self) -> str:
        """Return ``name/version``."""
        return f"{self.name}/{self.version}"

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> MetricRef:
        """Rebuild a reference from :meth:`to_record` output."""
        return cls(name=record["name"], version=record["version"])


# ---------------------------------------------------------------------------- manifest


@dataclass(frozen=True, kw_only=True)
class ExperimentManifest:
    """A hashed, versioned definition of one controlled comparison.

    Attributes:
        experiment_id: Stable name of the experiment.
        version: Version; a changed manifest is a new version.
        description: What the experiment asks.
        purpose: Tuning or evaluation; tuning never runs on the held-out split.
        evaluated_stage: The stage whose quality every arm's report measures.
        selection: The reference-set version and the exact ordered samples.
        repetitions_per_sample: Repeated inferences per sample. They are
            correlated evidence about one physical observation, never new samples.
        base_configuration_digest: Digest of the base/default effective configuration.
        variables: The variables under test.
        mode: How the variables combine into arms.
        baseline_arm_id: The arm holding every baseline value.
        arms: One arm per cell of the ablation matrix, with its resolved topology.
        fixed_controls: Values held constant across arms.
        quality_metrics: Quality metrics every arm reports.
        resource_capture: Performance metrics every arm reports (the runtime,
            memory, storage and throughput capture policy).
        registry: Identity of the metric registry the metrics come from.
    """

    experiment_id: str
    version: str
    description: str
    purpose: ExperimentPurpose
    evaluated_stage: EvaluationStage
    selection: SelectionBinding
    repetitions_per_sample: int
    base_configuration_digest: str
    variables: tuple[ExperimentVariable, ...]
    mode: AblationMode
    baseline_arm_id: str
    arms: tuple[ExperimentArm, ...]
    fixed_controls: tuple[FixedControl, ...]
    quality_metrics: tuple[MetricRef, ...]
    resource_capture: tuple[MetricRef, ...]
    registry: MetricRegistryIdentity

    def __post_init__(self) -> None:
        """Require a controlled comparison: only declared variables vary."""
        require_text("experiment_id", self.experiment_id)
        require_text("experiment version", self.version)
        require_sha256("base_configuration_digest", self.base_configuration_digest)
        if self.repetitions_per_sample < 1:
            raise ValueError("repetitions_per_sample must be at least 1")
        if not self.quality_metrics:
            raise ExperimentError("an experiment needs at least one quality metric")
        require_unique("quality metric", (item.key for item in self.quality_metrics))
        require_unique("resource metric", (item.key for item in self.resource_capture))
        require_unique("variable", (item.name for item in self.variables))
        require_unique("arm id", (item.arm_id for item in self.arms))
        self._check_controls()
        self._check_matrix()
        self._check_touched_stages_exist()
        baseline = self.baseline_arm
        for arm in self.arms:
            if arm.arm_id != baseline.arm_id:
                self._check_only_declared_variables_vary(baseline, arm)
        self._check_pinned_upstream()

    # ------------------------------------------------------------------- accessors

    @property
    def baseline_arm(self) -> ExperimentArm:
        """Return the baseline arm."""
        for arm in self.arms:
            if arm.arm_id == self.baseline_arm_id:
                return arm
        raise ExperimentError(f"the baseline arm {self.baseline_arm_id!r} is not among the arms")

    @property
    def physical_sample_count(self) -> int:
        """Return the number of distinct physical samples; repetitions never add to it."""
        return len(self.selection.sample_ids)

    def arm(self, arm_id: str) -> ExperimentArm:
        """Return an arm by identity."""
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise ExperimentError(f"unknown arm {arm_id!r}")

    # ------------------------------------------------------------------ invariants

    def _check_controls(self) -> None:
        require_unique("fixed control name", (item.name for item in self.fixed_controls))
        names = {item.name for item in self.variables}
        for control in self.fixed_controls:
            if control.name in names:
                raise ExperimentError(
                    f"fixed control {control.name!r} is also a variable: a control cannot vary"
                )

    def _check_matrix(self) -> None:
        if not self.variables:
            raise ExperimentError("an experiment needs at least one variable under test")
        by_name = {item.name: item for item in self.variables}
        for arm in self.arms:
            assigned = arm.assignment
            for name in by_name:
                if name not in assigned:
                    raise ExperimentError(f"arm {arm.arm_id!r} does not assign variable {name!r}")
            for name, value in assigned.items():
                variable = by_name.get(name)
                if variable is None:
                    raise ExperimentError(f"arm {arm.arm_id!r} assigns unknown variable {name!r}")
                if value not in variable.values:
                    raise ExperimentError(
                        f"arm {arm.arm_id!r} assigns {value!r} to variable {name!r}, which "
                        f"allows only {list(variable.values)}"
                    )
        declared = {tuple(sorted(cell)) for cell in ablation_cells(self.variables, self.mode)}
        actual = {tuple(sorted(arm.assignments)) for arm in self.arms}
        if actual != declared or len(self.arms) != len(declared):
            raise ExperimentError(
                f"the arms are not the cells of the {self.mode.value} ablation matrix: "
                f"expected {len(declared)} arms with these assignments, found {len(self.arms)}"
            )
        baseline = self.baseline_arm
        if tuple(sorted(baseline.assignments)) != tuple(
            sorted((item.name, item.baseline_value) for item in self.variables)
        ):
            raise ExperimentError(
                f"the baseline arm {baseline.arm_id!r} must hold every baseline value"
            )

    def _active_variables(self, arm: ExperimentArm) -> list[ExperimentVariable]:
        assigned = arm.assignment
        return [item for item in self.variables if assigned[item.name] != item.baseline_value]

    def _check_only_declared_variables_vary(
        self, baseline: ExperimentArm, arm: ExperimentArm
    ) -> None:
        active = self._active_variables(arm)
        before = {item.stage_id: item for item in baseline.topology.stages}
        after = {item.stage_id: item for item in arm.topology.stages}

        def covered(stage_id: str, kinds: Iterable[VariationKind]) -> bool:
            allowed = set(kinds)
            return any(stage_id in item.touches and item.kind in allowed for item in active)

        def undeclared(what: str, stage_id: str) -> ExperimentError:
            return ExperimentError(
                f"undeclared change in arm {arm.arm_id!r}: {what} of stage {stage_id!r} is not "
                "covered by any assigned variable of a kind that allows it"
            )

        for stage_id in sorted(set(before) | set(after)):
            old, new = before.get(stage_id), after.get(stage_id)
            if old is None or new is None:
                if not covered(stage_id, {VariationKind.TOPOLOGY}):
                    raise undeclared("the presence", stage_id)
                continue
            if old.capability != new.capability:
                raise undeclared("the capability", stage_id)
            if old.depends_on != new.depends_on and not covered(stage_id, {VariationKind.TOPOLOGY}):
                raise undeclared("the dependencies", stage_id)
            if old.implementation.backend != new.implementation.backend and not covered(
                stage_id, {VariationKind.BACKEND}
            ):
                raise undeclared("the backend or model", stage_id)
            if (
                old.implementation.configuration_digest != new.implementation.configuration_digest
                and not covered(stage_id, _CONFIGURATION_KINDS)
            ):
                raise undeclared("the configuration", stage_id)
            if old.artifact != new.artifact and not covered(stage_id, VariationKind):
                raise undeclared("the pinned artifact", stage_id)

        for variable in active:
            if not any(before.get(name) != after.get(name) for name in variable.touches):
                raise ExperimentError(
                    f"variable {variable.name!r} is assigned {arm.assignment[variable.name]!r} in "
                    f"arm {arm.arm_id!r} but changes none of the stages it touches"
                )

    def _touched_stage_ids(self) -> set[str]:
        return {name for variable in self.variables for name in variable.touches}

    def _check_touched_stages_exist(self) -> None:
        known = {item.stage_id for arm in self.arms for item in arm.topology.stages}
        for variable in self.variables:
            for name in variable.touches:
                if name not in known:
                    raise ExperimentError(
                        f"variable {variable.name!r} touches unknown stage {name!r}: "
                        "it is in no arm"
                    )

    def _check_pinned_upstream(self) -> None:
        touched = self._touched_stage_ids()
        for arm in self.arms:
            for item in arm.topology.stages:
                if item.stage_id not in touched:
                    continue
                for dependency in item.depends_on:
                    if dependency in touched:
                        continue
                    if arm.topology.stage(dependency).artifact is None:
                        raise ExperimentError(
                            f"stage {dependency!r}, upstream of touched stage {item.stage_id!r} "
                            f"in arm {arm.arm_id!r}, must be a pinned immutable artifact so that "
                            "every arm provably shares it"
                        )

    # ----------------------------------------------------------------- serialization

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible content the digest is computed over."""
        return {
            "schema": EXPERIMENT_SCHEMA,
            "experiment_id": self.experiment_id,
            "version": self.version,
            "description": self.description,
            "purpose": self.purpose.value,
            "evaluated_stage": self.evaluated_stage.value,
            "selection": self.selection.to_record(),
            "repetitions_per_sample": self.repetitions_per_sample,
            "base_configuration_digest": self.base_configuration_digest,
            "variables": [item.to_record() for item in self.variables],
            "mode": self.mode.value,
            "baseline_arm_id": self.baseline_arm_id,
            "arms": [item.to_record() for item in self.arms],
            "fixed_controls": [item.to_record() for item in self.fixed_controls],
            "quality_metrics": [item.to_record() for item in self.quality_metrics],
            "resource_capture": [item.to_record() for item in self.resource_capture],
            "registry": self.registry.to_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ExperimentManifest:
        """Rebuild a manifest from :meth:`to_record` output."""
        return cls(
            experiment_id=record["experiment_id"],
            version=record["version"],
            description=record["description"],
            purpose=ExperimentPurpose(record["purpose"]),
            evaluated_stage=EvaluationStage(record["evaluated_stage"]),
            selection=SelectionBinding.from_record(record["selection"]),
            repetitions_per_sample=record["repetitions_per_sample"],
            base_configuration_digest=record["base_configuration_digest"],
            variables=tuple(ExperimentVariable.from_record(item) for item in record["variables"]),
            mode=AblationMode(record["mode"]),
            baseline_arm_id=record["baseline_arm_id"],
            arms=tuple(ExperimentArm.from_record(item) for item in record["arms"]),
            fixed_controls=tuple(
                FixedControl.from_record(item) for item in record["fixed_controls"]
            ),
            quality_metrics=tuple(
                MetricRef.from_record(item) for item in record["quality_metrics"]
            ),
            resource_capture=tuple(
                MetricRef.from_record(item) for item in record["resource_capture"]
            ),
            registry=MetricRegistryIdentity.from_record(record["registry"]),
        )

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the whole manifest."""
        return canonical_digest(self.to_record())


def experiment_artifact_identity(manifest: ExperimentManifest) -> ArtifactIdentity:
    """Return the identity an arm's report cites to bind itself to this exact manifest."""
    return ArtifactIdentity(
        kind="experiment_manifest",
        artifact_id=f"{manifest.experiment_id}/{manifest.version}",
        digest=manifest.digest(),
    )


# --------------------------------------------------------------------------- validation


def validate_experiment_manifest(
    manifest: ExperimentManifest,
    *,
    reference_set: ReferenceSetManifest,
    registry: MetricRegistry,
) -> None:
    """Check a manifest against the reference set and the metric registry it cites.

    Raises:
        ExperimentError: If the selection does not reproduce the reference
            set's split, a tuning experiment targets the held-out split, the
            registry differs, or a metric is in the wrong place.
        MetricRegistryError: If a metric is unknown.
        MetricCompatibilityError: If the reference set lacks the annotation
            schemas a metric needs.
    """
    selection = manifest.selection
    if selection.reference_set != reference_set.identity():
        raise ExperimentError(
            f"the selection cites reference set {selection.reference_set.to_record()} but the "
            f"reference set given is {reference_set.identity().to_record()}"
        )
    try:
        expected = reference_set.selection(selection.scheme_id, selection.split)
    except ReferenceSetError as error:
        raise ExperimentError(
            f"the selection names a scheme or split the reference set does not have: {error}"
        ) from error
    if tuple(item.sample_id for item in expected) != selection.sample_ids:
        raise ExperimentError(
            f"the selection does not reproduce the ordered samples of split "
            f"{selection.split!r} of scheme {selection.scheme_id!r}"
        )
    scheme = next(
        item for item in reference_set.split_schemes if item.scheme_id == selection.scheme_id
    )
    role = next(item.role for item in scheme.splits if item.name == selection.split)
    if role is not selection.role:
        raise ExperimentError(
            f"the selection declares role {selection.role.value!r} but the split is {role.value!r}"
        )
    if manifest.purpose is ExperimentPurpose.TUNING and role is SplitRole.TEST:
        raise ExperimentError(
            f"a tuning experiment cannot use the held-out split {selection.split!r}: "
            "hyperparameters are never tuned on held-out evaluation data"
        )
    if manifest.registry != registry.identity():
        raise ExperimentError(
            "the manifest cites a different metric registry than the one it is validated against"
        )
    available = {entry.schema for entry in reference_set.annotations}
    for reference in manifest.quality_metrics:
        definition = registry.get(reference.name, reference.version)
        if definition.kind is not MetricKind.QUALITY:
            raise ExperimentError(
                f"metric {definition.key} is a {definition.kind.value} metric, not a quality metric"
            )
        if definition.stage is not manifest.evaluated_stage:
            raise ExperimentError(
                f"quality metric {definition.key} evaluates stage "
                f"{None if definition.stage is None else definition.stage.value!r}, not "
                f"{manifest.evaluated_stage.value!r}"
            )
        require_annotation_compatibility(definition, available)
    for reference in manifest.resource_capture:
        definition = registry.get(reference.name, reference.version)
        if definition.kind is not MetricKind.PERFORMANCE:
            raise ExperimentError(
                f"resource metric {definition.key} is a {definition.kind.value} metric, "
                "not a performance metric"
            )


# ------------------------------------------------------------------------ persistence


def encode_experiment(manifest: ExperimentManifest) -> dict[str, Any]:
    """Return the manifest document, including its own digest."""
    return {**manifest.to_record(), "digest": manifest.digest()}


def decode_experiment(document: Mapping[str, Any]) -> ExperimentManifest:
    """Rebuild a manifest and verify its declared digest.

    Raises:
        ExperimentError: If the schema is unknown, a field is missing or
            invalid, or the declared digest does not match the content.
    """
    try:
        if document["schema"] != EXPERIMENT_SCHEMA:
            raise ExperimentError(
                f"unsupported experiment schema {document['schema']!r}; "
                f"expected {EXPERIMENT_SCHEMA!r}"
            )
        declared = document["digest"]
        manifest = ExperimentManifest.from_record(document)
    except ExperimentError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ExperimentError(f"invalid experiment manifest: {error}") from error
    if manifest.digest() != declared:
        raise ExperimentError(
            f"experiment digest mismatch: declared {declared!r}, computed {manifest.digest()!r}"
        )
    return manifest


def write_experiment(root: Path, manifest: ExperimentManifest) -> None:
    """Atomically publish an immutable manifest under ``root``.

    Raises:
        FileExistsError: If a manifest already exists there; a changed
            experiment is a new version in a new directory.
    """
    write_immutable_json(
        root / EXPERIMENT_FILENAME, encode_experiment(manifest), "experiment manifest"
    )


def read_experiment(root: Path) -> ExperimentManifest:
    """Read the manifest under ``root`` and verify its digest.

    Raises:
        ExperimentError: If the document is not valid JSON or fails :func:`decode_experiment`.
    """
    text = (root / EXPERIMENT_FILENAME).read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ExperimentError(f"experiment manifest is not valid JSON: {error}") from error
    return decode_experiment(document)

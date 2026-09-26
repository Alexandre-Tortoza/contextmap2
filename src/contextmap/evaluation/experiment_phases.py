"""Factorized experiment phases expanded into resolved ContextMap2 configurations (#527).

The perception experiment of milestone #22 is run in named phases (backend grid, native task,
grounding/refinement topology, feature backend, semantic interpreter, prompt, view, scene
context, visual budget, selected compositions, downstream confirmation). A blind Cartesian
product of every factor would be wasteful and scientifically hard to read, so a
:class:`PhaseSpec` declares one phase: the base runtime configuration, the factors it varies
(each level is a set of runtime configuration overrides, plus the capabilities and the
producer -> consumer wiring it uses) and, for a composition phase, the exact cells to run.

:func:`expand_phase` turns a phase into ordinary runtime configurations without importing or
loading any model:

* every arm resolves through the runtime's own configuration and topology resolution
  (``resolve_effective_config``, ``check_selection``, ``resolve_plan``); an arm that does
  not resolve fails the phase;
* its selections are mapped to capabilities of the capability matrix (#522) and its wiring is
  checked there: an incompatible or undeclared edge fails the phase, while a planned or blocked
  capability or edge keeps the arm, recorded as ``blocked`` with the exact reason;
* arms with the same effective configuration and wiring are deduplicated deterministically;
* the runnable arms become an :class:`~contextmap.evaluation.experiments.ExperimentManifest`,
  whose construction enforces the matched-arm invariants of #545.

The :class:`PhaseManifest` records every arm, runnable or not, with its effective
configuration digest, topology digest and state; :func:`record_phase_outcomes` adds what the
execution did (executed, failed, skipped), never dropping an arm. Runtime cannot import
evaluation, so this layer lives here and only calls the runtime's public resolution functions.
See ``src/contextmap/evaluation/docs/experiment-phases.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.evaluation._persistence import (
    canonical_digest,
    canonical_json,
    write_immutable_json,
)
from contextmap.evaluation._validation import require_text, require_unique
from contextmap.evaluation.capability_matrix import (
    CAPABILITY_MATRIX,
    Capability,
    CapabilityMatrix,
    CapabilityMatrixError,
    CompositionStatus,
    ImplementationStatus,
)
from contextmap.evaluation.experiment_runner import ArmStatus, ExperimentRun
from contextmap.evaluation.experiments import (
    AblationMode,
    ExperimentArm,
    ExperimentError,
    ExperimentManifest,
    ExperimentPurpose,
    ExperimentVariable,
    FixedControl,
    MetricRef,
    ResolvedTopology,
    SelectionBinding,
    StageImplementation,
    TopologyStage,
    VariationKind,
    ablation_cells,
    experiment_artifact_identity,
)
from contextmap.evaluation.metrics import EvaluationStage, MetricRegistryIdentity
from contextmap.evaluation.report_schema import ArtifactIdentity
from contextmap.runtime import (
    CONFIG_SCHEMA_VERSION,
    BackendConfigurationError,
    ConfigurationError,
    EffectiveConfig,
    PlannedStage,
    check_selection,
    resolve_effective_config,
    resolve_plan,
    resolve_semantic_request_policy,
)

PHASE_SCHEMA = "contextmap.experiment-phase/v1"
"""Schema identifier of the phase manifest document."""

PHASE_FILENAME = "phase.json"
"""File name of the phase manifest inside its directory."""

BASELINE_ARM_ID = "baseline"
"""Identity of the arm that holds every factor's baseline level."""

_BACKEND_VERSION = f"runtime-config/{CONFIG_SCHEMA_VERSION}"
_STAGE_BACKEND = "stage"
_SEMANTIC_COMPONENT = "visual_perception.semantic_interpretation"
_CONFIGURATION_KINDS = frozenset(
    {
        VariationKind.BACKEND,
        VariationKind.POLICY,
        VariationKind.CONFIGURATION,
        VariationKind.EVIDENCE_CHANNELS,
    }
)


class PhaseExpansionError(ExperimentError):
    """Raised when a phase cannot be expanded into valid, resolvable, admissible arms."""


class PhaseKind(Enum):
    """The named phases of the perception experiment (#521)."""

    BACKEND_GRID = "backend_grid"
    NATIVE_TASK = "native_task"
    GROUNDING_REFINEMENT = "grounding_refinement"
    FEATURE_BACKEND = "feature_backend"
    SEMANTIC_INTERPRETER = "semantic_interpreter"
    SEMANTIC_PROMPT = "semantic_prompt"
    VISUAL_VIEW = "visual_view"
    SCENE_CONTEXT = "scene_context"
    VISUAL_BUDGET = "visual_budget"
    PIPELINE_COMPOSITION = "pipeline_composition"
    DOWNSTREAM_CONFIRMATION = "downstream_confirmation"


@dataclass(frozen=True)
class _PhaseRule:
    """What a phase may vary and how its cells are chosen."""

    kinds: frozenset[VariationKind]
    single_factor: bool
    selected: bool


_ALL_KINDS = frozenset(VariationKind)
_RULES: Mapping[PhaseKind, _PhaseRule] = {
    PhaseKind.BACKEND_GRID: _PhaseRule(frozenset({VariationKind.BACKEND}), False, False),
    PhaseKind.NATIVE_TASK: _PhaseRule(frozenset({VariationKind.CONFIGURATION}), True, False),
    PhaseKind.GROUNDING_REFINEMENT: _PhaseRule(frozenset({VariationKind.TOPOLOGY}), True, False),
    PhaseKind.FEATURE_BACKEND: _PhaseRule(
        frozenset({VariationKind.BACKEND, VariationKind.EVIDENCE_CHANNELS}), True, False
    ),
    PhaseKind.SEMANTIC_INTERPRETER: _PhaseRule(frozenset({VariationKind.BACKEND}), True, False),
    PhaseKind.SEMANTIC_PROMPT: _PhaseRule(frozenset({VariationKind.POLICY}), True, False),
    PhaseKind.VISUAL_VIEW: _PhaseRule(frozenset({VariationKind.EVIDENCE_CHANNELS}), True, False),
    PhaseKind.SCENE_CONTEXT: _PhaseRule(frozenset({VariationKind.EVIDENCE_CHANNELS}), True, False),
    PhaseKind.VISUAL_BUDGET: _PhaseRule(frozenset({VariationKind.CONFIGURATION}), True, False),
    PhaseKind.PIPELINE_COMPOSITION: _PhaseRule(_ALL_KINDS, False, True),
    PhaseKind.DOWNSTREAM_CONFIRMATION: _PhaseRule(_ALL_KINDS, False, True),
}
"""Phases 2-9 vary exactly one factor of one kind; topology changes are their own phase, and
the composition phases run only the cells they select."""


# ------------------------------------------------------------------------------- the spec


@dataclass(frozen=True, kw_only=True)
class FactorLevel:
    """One value of a factor: what it changes in the runtime configuration and uses.

    Attributes:
        value: The level's name.
        overrides: ``(dotted runtime configuration path, JSON value)`` applied on top of the
            base configuration. Only ``components.*`` and ``pipeline.stages.*`` may vary.
        capabilities: Matrix capabilities the level uses that no runtime component selects
            (a semantic scorer, a blocked 3D proposal source).
        edges: ``(producer, consumer)`` capability wiring the level adds.
    """

    value: str
    overrides: tuple[tuple[str, Any], ...] = ()
    capabilities: tuple[str, ...] = ()
    edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Require a name, unique overrides of variable paths and well-formed edges."""
        require_text("level value", self.value)
        require_unique(f"override path of level {self.value!r}", (p for p, _ in self.overrides))
        for path, _ in self.overrides:
            parts = path.split(".")
            if not (
                (parts[0] == "components" and len(parts) >= 4)
                or (parts[:2] == ["pipeline", "stages"] and len(parts) == 3)
            ):
                raise ValueError(
                    f"level {self.value!r} overrides {path!r}: a factor varies only "
                    "components.<capability>.<slot>.* or pipeline.stages.<stage>"
                )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "value": self.value,
            "overrides": [[path, value] for path, value in self.overrides],
            "capabilities": list(self.capabilities),
            "edges": [list(edge) for edge in self.edges],
        }


@dataclass(frozen=True, kw_only=True)
class Factor:
    """A factor a phase varies, with its levels and baseline level."""

    name: str
    kind: VariationKind
    levels: tuple[FactorLevel, ...]
    baseline: str
    description: str = ""

    def __post_init__(self) -> None:
        """Require at least two unique levels, one of which is the baseline."""
        require_text("factor name", self.name)
        if len(self.levels) < 2:
            raise ValueError(f"factor {self.name!r} needs at least two levels")
        require_unique(f"level of factor {self.name!r}", (item.value for item in self.levels))
        if self.baseline not in {item.value for item in self.levels}:
            raise ValueError(f"the baseline of factor {self.name!r} is not one of its levels")

    def level(self, value: str) -> FactorLevel:
        """Return one level by value."""
        for item in self.levels:
            if item.value == value:
                return item
        raise PhaseExpansionError(f"factor {self.name!r} has no level {value!r}")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "baseline": self.baseline,
            "description": self.description,
            "levels": [item.to_record() for item in self.levels],
        }


@dataclass(frozen=True, kw_only=True)
class MatchedParameters:
    """Parameters every runnable arm must give the backend selected for one component.

    A backend substitution compares only if the backend-neutral treatment is the same: the
    prompt and view policy of a Qwen arm and of an Eagle 2.5 arm, for example. The parameters
    live in each backend's own block, so the field-level check of #545 cannot see them.
    """

    component_id: str
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require a component and at least one unique parameter name."""
        require_text("component_id", self.component_id)
        if not self.names:
            raise ValueError("matched parameters need at least one name")
        require_unique("matched parameter", self.names)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"component_id": self.component_id, "names": list(self.names)}


@dataclass(frozen=True, kw_only=True)
class PhaseSpec:
    """One named phase: what is fixed, what varies and which cells run.

    Attributes:
        experiment_id: The experiment the phase belongs to.
        phase_id: Identity of the phase inside the experiment.
        kind: Which of the named phases it is; it limits the factors it may vary.
        version: Version of the phase; a changed phase is a new version.
        description: What the phase asks.
        purpose: Tuning or evaluation.
        evaluated_stage: The stage whose quality every arm's report measures.
        selection: The frozen reference set and the exact ordered samples.
        base_overrides: The base runtime configuration, as overrides of the canonical profile.
        base_capabilities: Capabilities every arm uses that no runtime component selects.
        base_edges: The producer -> consumer wiring every arm uses.
        factors: The factors the phase varies.
        targets: The stages the arms produce; their upstream stages are included.
        pinned: ``(stage, artifact)`` reused as immutable inputs in every arm.
        compositions: The non-baseline cells of a composition phase; empty otherwise.
        matched: Backend parameters every runnable arm must share.
        quality_metrics: Quality metrics every arm reports.
        resource_capture: Performance metrics every arm reports.
        registry: The metric registry.
        fixed_controls: Values held constant across arms.
        repetitions_per_sample: Repeated inferences per sample (correlated evidence).
    """

    experiment_id: str
    phase_id: str
    kind: PhaseKind
    version: str
    description: str
    purpose: ExperimentPurpose
    evaluated_stage: EvaluationStage
    selection: SelectionBinding
    base_overrides: tuple[tuple[str, Any], ...]
    factors: tuple[Factor, ...]
    targets: tuple[str, ...]
    pinned: tuple[tuple[str, ArtifactIdentity], ...]
    quality_metrics: tuple[MetricRef, ...]
    resource_capture: tuple[MetricRef, ...]
    registry: MetricRegistryIdentity
    base_capabilities: tuple[str, ...] = ()
    base_edges: tuple[tuple[str, str], ...] = ()
    compositions: tuple[tuple[tuple[str, str], ...], ...] = ()
    matched: tuple[MatchedParameters, ...] = ()
    fixed_controls: tuple[FixedControl, ...] = ()
    repetitions_per_sample: int = 1

    def __post_init__(self) -> None:
        """Require a phase that varies only what its kind allows, over declared cells."""
        require_text("experiment_id", self.experiment_id)
        require_text("phase_id", self.phase_id)
        require_text("phase version", self.version)
        if not self.targets:
            raise PhaseExpansionError("a phase needs at least one target stage")
        require_unique("pinned stage", (stage for stage, _ in self.pinned))
        if not self.factors:
            raise PhaseExpansionError("a phase needs at least one factor")
        require_unique("factor", (item.name for item in self.factors))
        rule = _RULES[self.kind]
        if rule.single_factor and len(self.factors) != 1:
            raise PhaseExpansionError(
                f"a {self.kind.value} phase varies exactly one factor, not {len(self.factors)}"
            )
        for item in self.factors:
            if item.kind not in rule.kinds:
                raise PhaseExpansionError(
                    f"a {self.kind.value} phase cannot vary {item.name!r}, a {item.kind.value} "
                    f"factor; it varies only {sorted(kind.value for kind in rule.kinds)}"
                )
        if rule.selected and not self.compositions:
            raise PhaseExpansionError(
                f"a {self.kind.value} phase runs only its selected compositions; none is given"
            )
        if not rule.selected and self.compositions:
            raise PhaseExpansionError(
                f"a {self.kind.value} phase expands one factor level at a time; selected "
                "compositions belong to a composition phase"
            )
        self._check_compositions()

    def _check_compositions(self) -> None:
        names = [item.name for item in self.factors]
        seen: set[tuple[tuple[str, str], ...]] = set()
        for cell in self.compositions:
            assigned = dict(cell)
            if len(assigned) != len(cell) or sorted(assigned) != sorted(names):
                raise PhaseExpansionError(
                    f"a selected composition must assign every factor {names} exactly once, "
                    f"got {[name for name, _ in cell]}"
                )
            for name, value in cell:
                self.factor(name).level(value)
            key = tuple(sorted(cell))
            if key == tuple(sorted(self.baseline_cell)):
                raise PhaseExpansionError("the baseline is always an arm; do not select it")
            if key in seen:
                raise PhaseExpansionError(f"composition {list(cell)} is selected twice")
            seen.add(key)

    def factor(self, name: str) -> Factor:
        """Return one factor by name."""
        for item in self.factors:
            if item.name == name:
                return item
        raise PhaseExpansionError(f"unknown factor {name!r}")

    @property
    def baseline_cell(self) -> tuple[tuple[str, str], ...]:
        """Return the assignment of every factor's baseline level."""
        return tuple((item.name, item.baseline) for item in self.factors)

    def cells(self) -> tuple[tuple[tuple[str, str], ...], ...]:
        """Return every arm's assignment, baseline first, in a deterministic order."""
        if self.compositions:
            order = [item.name for item in self.factors]
            return (
                self.baseline_cell,
                *(
                    tuple(sorted(cell, key=lambda item: order.index(item[0])))
                    for cell in self.compositions
                ),
            )
        variables = tuple(
            ExperimentVariable(
                name=item.name,
                kind=item.kind,
                touches=("phase",),
                values=tuple(level.value for level in item.levels),
                baseline_value=item.baseline,
            )
            for item in self.factors
        )
        return ablation_cells(variables, AblationMode.ONE_AT_A_TIME)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record of the declaration."""
        return {
            "experiment_id": self.experiment_id,
            "phase_id": self.phase_id,
            "kind": self.kind.value,
            "version": self.version,
            "description": self.description,
            "purpose": self.purpose.value,
            "evaluated_stage": self.evaluated_stage.value,
            "selection": self.selection.to_record(),
            "base_overrides": [[path, value] for path, value in self.base_overrides],
            "base_capabilities": list(self.base_capabilities),
            "base_edges": [list(edge) for edge in self.base_edges],
            "factors": [item.to_record() for item in self.factors],
            "cells": "selected" if self.compositions else "one_at_a_time",
            "compositions": [[list(item) for item in cell] for cell in self.compositions],
            "targets": list(self.targets),
            "pinned": [[stage, artifact.to_record()] for stage, artifact in self.pinned],
            "matched": [item.to_record() for item in self.matched],
            "quality_metrics": [item.to_record() for item in self.quality_metrics],
            "resource_capture": [item.to_record() for item in self.resource_capture],
            "registry": self.registry.to_record(),
            "fixed_controls": [item.to_record() for item in self.fixed_controls],
            "repetitions_per_sample": self.repetitions_per_sample,
        }


# ------------------------------------------------------------------------------ the arms


class ArmState(Enum):
    """Where an arm stands: never dropped, whatever happened to it."""

    PLANNED = "planned"
    EXECUTED = "executed"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Not run: a duplicate of another arm, or the backend was unavailable."""
    BLOCKED = "blocked"
    """Uses a capability or wiring that is planned or waits for upstream assets."""


@dataclass(frozen=True, kw_only=True)
class PhaseArm:
    """One arm of a phase and its state.

    Attributes:
        arm_id: Identity of the arm.
        assignments: ``(factor, level)`` for every factor.
        state: Planned, executed, failed, skipped or blocked.
        reason: Why it is blocked, skipped or failed.
        issues: The issues that unblock a blocked arm.
        effective_config_digest: Digest of the arm's resolved runtime configuration.
        topology_digest: Digest of the arm's resolved topology.
        capabilities: The matrix capabilities the arm uses.
        edges: The producer -> consumer wiring the arm uses.
        request_policy_fingerprint: Identity of the semantic request policy (prompt, ordered
            views, scene context) the arm resolves to, when it interprets semantics.
        duplicate_of: The arm this one duplicates, when skipped as a duplicate.
    """

    arm_id: str
    assignments: tuple[tuple[str, str], ...]
    state: ArmState
    reason: str | None
    issues: tuple[int, ...]
    effective_config_digest: str
    topology_digest: str
    capabilities: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    request_policy_fingerprint: str | None
    duplicate_of: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "arm_id": self.arm_id,
            "assignments": [list(item) for item in self.assignments],
            "state": self.state.value,
            "reason": self.reason,
            "issues": list(self.issues),
            "effective_config_digest": self.effective_config_digest,
            "topology_digest": self.topology_digest,
            "capabilities": list(self.capabilities),
            "edges": [list(edge) for edge in self.edges],
            "request_policy_fingerprint": self.request_policy_fingerprint,
            "duplicate_of": self.duplicate_of,
        }


@dataclass(frozen=True, kw_only=True)
class PhaseManifest:
    """A phase expanded into arms, with the controlled comparison of the runnable ones.

    Attributes:
        spec: The phase declaration.
        base_configuration_digest: Digest of the resolved base configuration.
        matrix_version: Version of the capability matrix the wiring was checked against.
        arms: Every arm, in cell order, whatever its state.
        experiment: The controlled comparison of the planned arms; ``None`` when fewer than
            two arms can run.
    """

    spec: PhaseSpec
    base_configuration_digest: str
    matrix_version: str
    arms: tuple[PhaseArm, ...]
    experiment: ExperimentManifest | None

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible content the digest is computed over."""
        return {
            "schema": PHASE_SCHEMA,
            **self.spec.to_record(),
            "base_configuration_digest": self.base_configuration_digest,
            "matrix_version": self.matrix_version,
            "arms": [item.to_record() for item in self.arms],
            "experiment": (
                None
                if self.experiment is None
                else experiment_artifact_identity(self.experiment).to_record()
            ),
        }

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the phase manifest."""
        return canonical_digest(self.to_record())


@dataclass(frozen=True)
class _Resolved:
    """One arm while it is being expanded."""

    arm: PhaseArm
    effective: EffectiveConfig
    topology: ResolvedTopology
    stage_of: Mapping[str, str]
    """The runtime stage of every topology node."""


# ----------------------------------------------------------------------------- expansion


def expand_phase(spec: PhaseSpec, *, matrix: CapabilityMatrix = CAPABILITY_MATRIX) -> PhaseManifest:
    """Expand a phase into resolved arms and the controlled comparison of the runnable ones.

    Nothing is imported or loaded besides the runtime's configuration and topology resolution.

    Raises:
        PhaseExpansionError: If an arm does not resolve, leaves a variation point of its
            topology unselected, selects what the matrix does not know, wires an incompatible
            or undeclared edge, breaks a matched parameter or a backend substitution's task,
            has a blocked baseline, or differs from the baseline in anything undeclared.
    """
    base = _resolve(spec.base_overrides, "the base configuration")
    resolved = [
        _expand_arm(spec, matrix, BASELINE_ARM_ID if index == 0 else f"arm-{index:02d}", cell)
        for index, cell in enumerate(spec.cells())
    ]
    if resolved[0].arm.state is ArmState.BLOCKED:
        raise PhaseExpansionError(
            f"the baseline arm is blocked, so the phase compares nothing: {resolved[0].arm.reason}"
        )
    resolved = _deduplicate(resolved)
    runnable = [item for item in resolved if item.arm.state is ArmState.PLANNED]
    _check_matched(spec, runnable)
    _check_backend_substitutions(spec, matrix, runnable)
    return PhaseManifest(
        spec=spec,
        base_configuration_digest=base.digest,
        matrix_version=matrix.version,
        arms=tuple(item.arm for item in resolved),
        experiment=_experiment(spec, base, runnable) if len(runnable) >= 2 else None,
    )


def _resolve(overrides: Iterable[tuple[str, Any]], what: str) -> EffectiveConfig:
    """Resolve overrides of the canonical profile through the runtime, as a user's would be."""
    try:
        return resolve_effective_config(
            overrides=[f"{path}={json.dumps(value)}" for path, value in overrides]
        )
    except ConfigurationError as error:
        raise PhaseExpansionError(f"{what} does not resolve: {error}") from error


def _expand_arm(
    spec: PhaseSpec, matrix: CapabilityMatrix, arm_id: str, cell: tuple[tuple[str, str], ...]
) -> _Resolved:
    levels = [spec.factor(name).level(value) for name, value in cell]
    effective = _resolve(
        (*spec.base_overrides, *(item for level in levels for item in level.overrides)),
        f"arm {arm_id!r}",
    )
    plan = resolve_plan(effective)
    if plan.problems:
        problems = "; ".join(str(item) for item in plan.problems)
        raise PhaseExpansionError(f"arm {arm_id!r} has an invalid topology: {problems}")
    stages = _included_stages(spec, plan.stages, arm_id)
    owned = {component for stage in stages for component in stage.components}
    unselected = [
        str(item)
        for item in check_selection(effective.config)
        if item.path.removeprefix("components.") in owned
    ]
    if unselected:
        raise PhaseExpansionError(f"arm {arm_id!r} is incomplete: {'; '.join(unselected)}")
    components = effective.config.to_document()["components"]
    declared = (*spec.base_capabilities, *(item for level in levels for item in level.capabilities))
    capabilities, reasons, issues = _capabilities(matrix, arm_id, stages, components, declared)
    edges = tuple(
        dict.fromkeys((*spec.base_edges, *(item for level in levels for item in level.edges)))
    )
    edge_reasons, edge_issues = _check_edges(matrix, arm_id, edges, capabilities)
    topology, stage_of = _topology(spec, stages, components)
    reasons += edge_reasons
    issues += edge_issues
    arm = PhaseArm(
        arm_id=arm_id,
        assignments=cell,
        state=ArmState.BLOCKED if reasons else ArmState.PLANNED,
        reason="; ".join(reasons) if reasons else None,
        issues=tuple(dict.fromkeys(issues)),
        effective_config_digest=effective.digest,
        topology_digest=topology.digest(),
        capabilities=capabilities,
        edges=edges,
        request_policy_fingerprint=_request_policy_fingerprint(effective, owned, arm_id),
    )
    return _Resolved(arm=arm, effective=effective, topology=topology, stage_of=stage_of)


def _request_policy_fingerprint(
    effective: EffectiveConfig, owned: set[str], arm_id: str
) -> str | None:
    """Return the identity of the arm's semantic request policy, resolved without a backend."""
    component = effective.config.components.get(_SEMANTIC_COMPONENT)
    if _SEMANTIC_COMPONENT not in owned or component is None or component.backend is None:
        return None
    try:
        return resolve_semantic_request_policy(effective).fingerprint()
    except (ConfigurationError, BackendConfigurationError) as error:
        raise PhaseExpansionError(
            f"the semantic request policy of arm {arm_id!r} does not resolve: {error}"
        ) from error


def _included_stages(
    spec: PhaseSpec, planned: Sequence[PlannedStage], arm_id: str
) -> tuple[PlannedStage, ...]:
    """Return the targets and every stage upstream of them, in plan order."""
    by_id = {stage.stage_id: stage for stage in planned}
    missing = [stage for stage in spec.targets if stage not in by_id]
    if missing:
        raise PhaseExpansionError(f"arm {arm_id!r} does not enable target stages {missing}")
    included: set[str] = set()
    pending = list(spec.targets)
    while pending:
        stage_id = pending.pop()
        if stage_id in included:
            continue
        included.add(stage_id)
        pending.extend(item.source for item in by_id[stage_id].inputs if item.source in by_id)
    outside = [stage for stage, _ in spec.pinned if stage not in included]
    if outside:
        raise PhaseExpansionError(f"pinned stages {outside} are not part of arm {arm_id!r}")
    return tuple(stage for stage in planned if stage.stage_id in included)


def _values_at(value: Any, parts: Sequence[str]) -> list[Any]:
    """Return the values at a dotted path, fanning out over every list along it."""
    if isinstance(value, list | tuple):
        return [found for item in value for found in _values_at(item, parts)]
    if not parts:
        return [value]
    if isinstance(value, Mapping) and parts[0] in value:
        return _values_at(value[parts[0]], parts[1:])
    return []


def _capabilities(
    matrix: CapabilityMatrix,
    arm_id: str,
    stages: Sequence[PlannedStage],
    components: Mapping[str, Any],
    declared: Iterable[str],
) -> tuple[tuple[str, ...], list[str], list[int]]:
    """Map an arm's selections to matrix capabilities, and say which ones block it."""
    found: list[str] = []
    stage_ids = {stage.stage_id for stage in stages}
    # Só os pontos de variação que a matriz inventaria são mapeados; os demais (fonte de
    # ingestão, políticas de associação...) não são capacidades do experimento de percepção.
    inventoried = {
        item.runtime.component_id for item in matrix.capabilities if item.runtime is not None
    }
    for stage in stages:
        for component_id, backend in stage.components.items():
            if backend is None or component_id not in inventoried:
                continue
            capability_name, slot = component_id.split(".", 1)
            parameters = components[capability_name][slot].get(backend, {})
            matches = [
                item.capability_id
                for item in matrix.capabilities
                if item.runtime is not None
                and (item.runtime.component_id, item.runtime.backend_id) == (component_id, backend)
                and all(
                    any(value in accepted for value in _values_at(parameters, name.split(".")))
                    for name, accepted in item.runtime.settings
                )
            ]
            if not matches:
                raise PhaseExpansionError(
                    f"arm {arm_id!r} selects {backend!r} for {component_id} with parameters that "
                    "match no capability of the matrix"
                )
            found.extend(matches)
    found.extend(
        item.capability_id
        for item in matrix.capabilities
        if item.runtime is not None
        and item.runtime.component_id is None
        and item.runtime.stage_id in stage_ids
    )
    for capability_id in declared:
        capability = _capability(matrix, capability_id)
        if capability.runtime is not None:
            raise PhaseExpansionError(
                f"{capability_id} is selected through the runtime configuration; a level declares "
                "only capabilities no runtime component selects"
            )
        found.append(capability_id)
    reasons: list[str] = []
    issues: list[int] = []
    for capability_id in dict.fromkeys(found):
        capability = _capability(matrix, capability_id)
        if capability.status is ImplementationStatus.OUT_OF_SCOPE:
            raise PhaseExpansionError(
                f"arm {arm_id!r} uses {capability_id}, which is out of scope: "
                f"{capability.status_reason}"
            )
        if capability.status is not ImplementationStatus.SUPPORTED:
            reasons.append(
                f"{capability_id} is {capability.status.value} (#{capability.issue}): "
                f"{capability.status_reason}"
            )
            issues.append(int(str(capability.issue)))
    return tuple(dict.fromkeys(found)), reasons, issues


def _capability(matrix: CapabilityMatrix, capability_id: str) -> Capability:
    try:
        return matrix.capability(capability_id)
    except CapabilityMatrixError as error:
        raise PhaseExpansionError(str(error)) from error


def _check_edges(
    matrix: CapabilityMatrix,
    arm_id: str,
    edges: Iterable[tuple[str, str]],
    capabilities: Sequence[str],
) -> tuple[list[str], list[int]]:
    """Refuse inadmissible wiring; return why admissible but unavailable wiring blocks the arm."""
    reasons: list[str] = []
    issues: list[int] = []
    for producer, consumer in edges:
        missing = [item for item in (producer, consumer) if item not in capabilities]
        if missing:
            raise PhaseExpansionError(
                f"arm {arm_id!r} wires {producer} -> {consumer}, but {missing} is not in the arm"
            )
        try:
            composition = matrix.composition(producer, consumer)
        except CapabilityMatrixError as error:
            raise PhaseExpansionError(f"arm {arm_id!r}: {error}") from error
        status = composition.status
        if status in {CompositionStatus.INCOMPATIBLE, CompositionStatus.NOT_COMPARABLE}:
            raise PhaseExpansionError(
                f"arm {arm_id!r} wires {composition.composition_id}, which is {status.value}: "
                f"{composition.note}"
            )
        if status is not CompositionStatus.SUPPORTED:
            reasons.append(
                f"composition {composition.composition_id} is {status.value} "
                f"(#{composition.issue}): {composition.note}"
            )
            issues.append(int(str(composition.issue)))
    return reasons, issues


def _topology(
    spec: PhaseSpec, stages: Sequence[PlannedStage], components: Mapping[str, Any]
) -> tuple[ResolvedTopology, dict[str, str]]:
    """Build one topology node per selected variation point (or per stage without one).

    A node's configuration is the component's resolved block (backend and its parameters), so
    #545 compares arms field by field; a node depends on every node of its input stages.
    """
    pinned = dict(spec.pinned)
    nodes_of: dict[str, list[str]] = {}
    built: list[TopologyStage] = []
    for stage in stages:
        entries: list[tuple[str, str, Mapping[str, Any]]] = []
        for component_id, backend in stage.components.items():
            if backend is not None:
                capability_name, slot = component_id.split(".", 1)
                entries.append((component_id, backend, components[capability_name][slot]))
        if not entries:
            entries.append((stage.stage_id, _STAGE_BACKEND, {}))
        depends_on = tuple(
            node
            for item in stage.inputs
            if item.source in nodes_of
            for node in nodes_of[item.source]
        )
        nodes_of[stage.stage_id] = [node_id for node_id, _, _ in entries]
        for node_id, backend, configuration in entries:
            built.append(
                TopologyStage(
                    stage_id=node_id,
                    capability=stage.capability,
                    implementation=StageImplementation.from_configuration(
                        backend_id=backend,
                        backend_version=_BACKEND_VERSION,
                        model=None,
                        configuration=configuration,
                    ),
                    depends_on=depends_on,
                    artifact=pinned.get(stage.stage_id),
                )
            )
    stage_of = {node: stage for stage, nodes in nodes_of.items() for node in nodes}
    return ResolvedTopology(stages=tuple(built)), stage_of


def _deduplicate(resolved: list[_Resolved]) -> list[_Resolved]:
    """Keep the first of arms with the same configuration and wiring; skip the others."""
    seen: dict[tuple[Any, ...], str] = {}
    result: list[_Resolved] = []
    for item in resolved:
        arm = item.arm
        key = (
            arm.effective_config_digest,
            arm.topology_digest,
            tuple(sorted(arm.capabilities)),
            tuple(sorted(arm.edges)),
        )
        if arm.state is ArmState.PLANNED and key in seen:
            first = seen[key]
            arm = replace(
                arm,
                state=ArmState.SKIPPED,
                reason=f"same effective configuration and wiring as arm {first!r}",
                duplicate_of=first,
            )
        elif arm.state is ArmState.PLANNED:
            seen[key] = arm.arm_id
        result.append(replace(item, arm=arm))
    return result


def _selected_parameters(item: _Resolved, component_id: str) -> Mapping[str, Any]:
    capability_name, slot = component_id.split(".", 1)
    block = item.effective.config.to_document()["components"][capability_name][slot]
    backend = block["backend"]
    if backend is None:
        raise PhaseExpansionError(
            f"arm {item.arm.arm_id!r} selects no backend for matched component {component_id}"
        )
    parameters: Mapping[str, Any] = block.get(backend, {})
    return parameters


def _check_matched(spec: PhaseSpec, runnable: Sequence[_Resolved]) -> None:
    """Require every runnable arm to give its backend the same matched parameters."""
    baseline = runnable[0]
    for matched in spec.matched:
        reference = _selected_parameters(baseline, matched.component_id)
        for item in runnable[1:]:
            parameters = _selected_parameters(item, matched.component_id)
            for name in matched.names:
                expected = canonical_json(reference.get(name))
                found = canonical_json(parameters.get(name))
                if expected != found:
                    raise PhaseExpansionError(
                        f"matched parameter {name!r} of {matched.component_id} differs: arm "
                        f"{item.arm.arm_id!r} has {found}, the baseline has {expected}"
                    )


def _check_backend_substitutions(
    spec: PhaseSpec, matrix: CapabilityMatrix, runnable: Sequence[_Resolved]
) -> None:
    """Require a pure backend substitution to keep the task: same comparison groups."""
    baseline = runnable[0].arm
    for item in runnable[1:]:
        active = [
            spec.factor(name)
            for name, value in item.arm.assignments
            if value != spec.factor(name).baseline
        ]
        if not all(factor.kind is VariationKind.BACKEND for factor in active):
            continue
        added = set(item.arm.capabilities) - set(baseline.capabilities)
        removed = set(baseline.capabilities) - set(item.arm.capabilities)
        for mine, theirs in ((added, removed), (removed, added)):
            for capability in sorted(mine):
                if not any(matrix.comparable(capability, other) for other in theirs):
                    raise PhaseExpansionError(
                        f"arm {item.arm.arm_id!r} is a backend substitution, but {capability} has "
                        f"no comparable counterpart among {sorted(theirs)}: it changes the task"
                    )


def _experiment(
    spec: PhaseSpec, base: EffectiveConfig, runnable: Sequence[_Resolved]
) -> ExperimentManifest:
    """Build the controlled comparison of the runnable arms (#545 checks run here)."""
    varied = [
        factor
        for factor in spec.factors
        if any(dict(item.arm.assignments)[factor.name] != factor.baseline for item in runnable)
    ]
    try:
        variables = tuple(_variable(factor, runnable) for factor in varied)
        names = {item.name for item in variables}
        return ExperimentManifest(
            experiment_id=f"{spec.experiment_id}.{spec.phase_id}",
            version=spec.version,
            description=spec.description,
            purpose=spec.purpose,
            evaluated_stage=spec.evaluated_stage,
            selection=spec.selection,
            repetitions_per_sample=spec.repetitions_per_sample,
            base_configuration_digest=base.digest,
            variables=variables,
            mode=AblationMode.SELECTED,
            baseline_arm_id=BASELINE_ARM_ID,
            arms=tuple(
                ExperimentArm(
                    arm_id=item.arm.arm_id,
                    assignments=tuple(
                        (name, value) for name, value in item.arm.assignments if name in names
                    ),
                    topology=item.topology,
                )
                for item in runnable
            ),
            fixed_controls=spec.fixed_controls,
            quality_metrics=spec.quality_metrics,
            resource_capture=spec.resource_capture,
            registry=spec.registry,
        )
    except (ExperimentError, ValueError) as error:
        raise PhaseExpansionError(f"phase {spec.phase_id!r}: {error}") from error


def _variable(factor: Factor, runnable: Sequence[_Resolved]) -> ExperimentVariable:
    """Declare a factor as a variable: the nodes its overrides address and their fields."""
    present = {node.stage_id for item in runnable for node in item.topology.stages}
    addressed: set[str] = set()
    fields: set[str] = set()
    for level in factor.levels:
        for path, _ in level.overrides:
            parts = path.split(".")
            if parts[0] == "pipeline":
                addressed.update(
                    node
                    for item in runnable
                    for node, stage in item.stage_of.items()
                    if stage == parts[2]
                )
                continue
            node_id = f"{parts[1]}.{parts[2]}"
            addressed.add(node_id)
            remainder = ".".join(parts[3:])
            fields.add(remainder)
            if remainder == "backend":
                fields.update(
                    node.implementation.backend_id
                    for item in runnable
                    for node in item.topology.stages
                    if node.stage_id == node_id
                )
    touches = addressed & present
    if factor.kind is VariationKind.TOPOLOGY:
        touches |= {
            node.stage_id
            for item in runnable
            for node in item.topology.stages
            if touches & set(node.depends_on)
        }
    return ExperimentVariable(
        name=factor.name,
        kind=factor.kind,
        touches=tuple(sorted(touches)),
        values=tuple(level.value for level in factor.levels),
        baseline_value=factor.baseline,
        description=factor.description,
        configuration_fields=(tuple(sorted(fields)) if factor.kind in _CONFIGURATION_KINDS else ()),
    )


# ------------------------------------------------------------------ outcomes, persistence


def record_phase_outcomes(manifest: PhaseManifest, run: ExperimentRun) -> PhaseManifest:
    """Return the phase with what the execution did to each runnable arm.

    Executed, failed (error or OOM) and unavailable arms are recorded as such; blocked and
    duplicate arms keep their state. No arm is dropped.

    Raises:
        PhaseExpansionError: If the run is not of this phase's experiment manifest.
    """
    if manifest.experiment is None or run.manifest.digest() != manifest.experiment.digest():
        raise PhaseExpansionError("the run belongs to another experiment than this phase's")
    runs = {item.arm_id: item for item in run.arm_runs}
    arms: list[PhaseArm] = []
    for arm in manifest.arms:
        outcome = runs.get(arm.arm_id)
        if outcome is None:
            arms.append(arm)
        elif outcome.status is ArmStatus.COMPLETED:
            arms.append(replace(arm, state=ArmState.EXECUTED, reason=None))
        else:
            failure = outcome.failure
            message = "" if failure is None else failure.message
            if outcome.status is ArmStatus.UNAVAILABLE:
                arms.append(replace(arm, state=ArmState.SKIPPED, reason=f"unavailable: {message}"))
            else:
                kind = "" if failure is None else f"{failure.kind.value}: "
                error = (
                    ""
                    if failure is None or failure.error_type is None
                    else f"{failure.error_type}: "
                )
                arms.append(replace(arm, state=ArmState.FAILED, reason=f"{kind}{error}{message}"))
    return replace(manifest, arms=tuple(arms))


def write_phase_manifest(root: Path, manifest: PhaseManifest) -> None:
    """Atomically publish an immutable phase manifest under ``root``.

    Raises:
        FileExistsError: If one already exists there; a changed phase is a new directory.
    """
    write_immutable_json(
        root / PHASE_FILENAME,
        {**manifest.to_record(), "digest": manifest.digest()},
        "phase manifest",
    )

"""Evidence for the keep/defer/default-change decision on an optional technique.

A global average hides where a technique helps and where it hurts, so the
evidence of a :class:`~contextmap.evaluation.technique_protocols.TechniqueProtocol`
is built per metric **and per stratum**: for every metric the whole population
and every stratum get their own effect against the baseline (improved,
regressed, unchanged, changed, or not comparable), the regressions are listed
explicitly, and quality and cost are separate sections. Effects use the
direction each metric declares in the registry and an explicit policy of
tolerances; nothing is combined into a score and nothing ranks the arms.

The evidence never touches a configuration. A human records the decision with
:func:`record_technique_decision`; even ``CHANGE_DEFAULT`` only produces a record
that requires re-validation through the canonical end-to-end acceptance flow.
See ``src/contextmap/evaluation/docs/optional-techniques.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.evaluation._persistence import canonical_digest, write_immutable_json
from contextmap.evaluation._validation import require_text, require_unique
from contextmap.evaluation.experiment_runner import (
    ArmStatus,
    ComparisonManifest,
    MetricComparison,
    MetricEntry,
    SharedArtifact,
)
from contextmap.evaluation.experiments import experiment_artifact_identity
from contextmap.evaluation.metrics import (
    ALLOWED_UNITS,
    EvaluationStage,
    MetricDefinition,
    MetricDirection,
    MetricKind,
    MetricRegistry,
    MetricRegistryIdentity,
)
from contextmap.evaluation.reference_set import ReferenceSetIdentity, ReferenceSetManifest
from contextmap.evaluation.report_schema import ArtifactIdentity, MetricStatus
from contextmap.evaluation.technique_protocols import (
    BASELINE_ARM_ID,
    VARIANT_ARM_ID,
    TechniqueProtocol,
)

EVIDENCE_SCHEMA = "contextmap.technique-evidence/v1"
DECISION_SCHEMA = "contextmap.technique-decision/v1"
EVIDENCE_FILENAME = "evidence.json"
DECISION_FILENAME = "decision.json"

COST_STRATA: Mapping[str, frozenset[str]] = {
    "device": frozenset({"cpu", "gpu"}),
    "artifact_role": frozenset({"intermediate", "final"}),
}
"""Strata a cost metric may carry: peak memory per device, storage per artifact role."""

Strata = tuple[tuple[str, str], ...]


class TechniqueEvidenceError(ValueError):
    """Raised when comparisons cannot be turned into trustworthy technique evidence."""


class Effect(Enum):
    """What the technique did to one metric in one stratum, against the baseline."""

    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    """The metric has no better direction, and it moved beyond the tolerance."""

    NOT_COMPARABLE = "not_comparable"
    """No effect can be stated: not applicable, unsupported, missing or too few samples."""


class StageStatus(Enum):
    """Whether a stage's comparison could be turned into effects."""

    EVALUATED = "evaluated"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, kw_only=True)
class EffectPolicy:
    """The explicit, versioned tolerances effects are judged by; there are no defaults.

    Attributes:
        policy_id: Version of the policy, cited by the evidence.
        tolerance_by_unit: ``(unit, tolerance)``: a difference within it is ``unchanged``.
        min_samples_per_stratum: Fewer samples than this make an effect not comparable,
            so a noisy stratum is not read as a gain or a regression.
    """

    policy_id: str
    tolerance_by_unit: tuple[tuple[str, float], ...]
    min_samples_per_stratum: int

    def __post_init__(self) -> None:
        """Require an identity, known units, sane tolerances and a positive sample floor."""
        require_text("policy_id", self.policy_id)
        require_unique("tolerance unit", (unit for unit, _ in self.tolerance_by_unit))
        for unit, tolerance in self.tolerance_by_unit:
            if unit not in ALLOWED_UNITS:
                raise ValueError(f"tolerance declared for unknown unit {unit!r}")
            if not (math.isfinite(tolerance) and tolerance >= 0):
                raise ValueError(f"the tolerance of unit {unit!r} must be a finite number >= 0")
        if self.min_samples_per_stratum < 1:
            raise ValueError("min_samples_per_stratum must be at least 1")

    def tolerance(self, unit: str) -> float:
        """Return the tolerance of a unit.

        Raises:
            TechniqueEvidenceError: If the policy declares none for it.
        """
        for name, value in self.tolerance_by_unit:
            if name == unit:
                return value
        raise TechniqueEvidenceError(f"the effect policy declares no tolerance for unit {unit!r}")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "policy_id": self.policy_id,
            "tolerance_by_unit": [list(item) for item in self.tolerance_by_unit],
            "min_samples_per_stratum": self.min_samples_per_stratum,
        }


@dataclass(frozen=True, kw_only=True)
class StratumEffect:
    """The effect on one metric in the whole population or in one stratum."""

    strata: Strata
    baseline_status: MetricStatus | None
    baseline_value: float | None
    baseline_sample_count: int | None
    variant_status: MetricStatus | None
    variant_value: float | None
    variant_sample_count: int | None
    delta: float | None
    effect: Effect
    reason: str

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "strata": [list(item) for item in self.strata],
            "baseline": {
                "status": None if self.baseline_status is None else self.baseline_status.value,
                "value": self.baseline_value,
                "sample_count": self.baseline_sample_count,
            },
            "variant": {
                "status": None if self.variant_status is None else self.variant_status.value,
                "value": self.variant_value,
                "sample_count": self.variant_sample_count,
            },
            "delta": self.delta,
            "effect": self.effect.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, kw_only=True)
class MetricEvidence:
    """One metric's effect in the whole population and in every stratum."""

    metric: str
    version: str
    kind: MetricKind
    unit: str
    direction: MetricDirection
    whole_population: StratumEffect | None
    strata: tuple[StratumEffect, ...]

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "metric": self.metric,
            "version": self.version,
            "kind": self.kind.value,
            "unit": self.unit,
            "direction": self.direction.value,
            "whole_population": (
                None if self.whole_population is None else self.whole_population.to_record()
            ),
            "strata": [item.to_record() for item in self.strata],
        }


@dataclass(frozen=True, kw_only=True)
class IncompleteArm:
    """An arm that did not complete, with the reason it stayed explicit."""

    arm_id: str
    status: ArmStatus
    reason: str

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"arm_id": self.arm_id, "status": self.status.value, "reason": self.reason}


@dataclass(frozen=True, kw_only=True)
class StageEvidence:
    """The evidence of one evaluated stage.

    Attributes:
        stage: The stage whose quality was measured.
        experiment: The exact experiment manifest the comparison belongs to.
        comparison_digest: Digest of the comparison the effects come from.
        status: ``EVALUATED``, or ``INCOMPLETE`` when an arm did not complete.
        incomplete_arms: The arms that did not complete; no effect is stated then.
        shared_artifacts: Artifacts both arms report identically (the upstream they share).
        quality: Quality effects, per metric and stratum.
        costs: Cost effects (time, memory, storage, throughput, failures), apart from quality.
    """

    stage: EvaluationStage
    experiment: ArtifactIdentity
    comparison_digest: str
    status: StageStatus
    incomplete_arms: tuple[IncompleteArm, ...]
    shared_artifacts: tuple[SharedArtifact, ...]
    quality: tuple[MetricEvidence, ...]
    costs: tuple[MetricEvidence, ...]

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "stage": self.stage.value,
            "experiment": self.experiment.to_record(),
            "comparison_digest": self.comparison_digest,
            "status": self.status.value,
            "incomplete_arms": [item.to_record() for item in self.incomplete_arms],
            "shared_artifacts": [item.to_record() for item in self.shared_artifacts],
            "quality": [item.to_record() for item in self.quality],
            "costs": [item.to_record() for item in self.costs],
        }


@dataclass(frozen=True, kw_only=True)
class UnevaluatedStage:
    """A stage of the protocol with no comparison yet: its evidence is missing, not neutral."""

    stage: EvaluationStage
    reason: str

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"stage": self.stage.value, "reason": self.reason}


@dataclass(frozen=True, kw_only=True)
class FactorStatus:
    """Whether a stratification factor is available: declared and actually reported."""

    factor: str
    declared_in_reference_set: bool
    reported: bool

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "factor": self.factor,
            "declared_in_reference_set": self.declared_in_reference_set,
            "reported": self.reported,
        }


@dataclass(frozen=True, kw_only=True)
class Regression:
    """A quality or cost effect that got worse, located by stage, metric and stratum."""

    stage: EvaluationStage
    metric: str
    version: str
    strata: Strata
    delta: float

    @property
    def key(self) -> str:
        """Return ``stage:metric/version@factor=value,...``, the identity a decision cites."""
        located = ",".join(f"{name}={value}" for name, value in self.strata) or "whole"
        return f"{self.stage.value}:{self.metric}/{self.version}@{located}"

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"key": self.key, "delta": self.delta}


@dataclass(frozen=True, kw_only=True)
class TechniqueEvidence:
    """Where an optional technique gains and regresses, with what it costs.

    Attributes:
        technique_id: The technique evaluated.
        protocol_version: Version of the protocol.
        protocol_digest: Digest of the protocol, binding the experiments.
        reference_set: The reference set the evidence was measured against.
        registry: The metric registry.
        policy: The tolerances effects were judged by.
        baseline_arm_id: The technique off.
        variant_arm_id: The technique on.
        stages: Evidence per stage that has a comparison.
        unevaluated_stages: Protocol stages with no comparison, explicitly missing.
        factors: Availability of every stratification factor of the protocol.
    """

    technique_id: str
    protocol_version: str
    protocol_digest: str
    reference_set: ReferenceSetIdentity
    registry: MetricRegistryIdentity
    policy: EffectPolicy
    baseline_arm_id: str
    variant_arm_id: str
    stages: tuple[StageEvidence, ...]
    unevaluated_stages: tuple[UnevaluatedStage, ...]
    factors: tuple[FactorStatus, ...]

    @property
    def complete(self) -> bool:
        """Return whether every stage of the protocol was evaluated with both arms."""
        return not self.unevaluated_stages and all(
            item.status is StageStatus.EVALUATED for item in self.stages
        )

    def _regressions(self, *, costs: bool) -> tuple[Regression, ...]:
        found: list[Regression] = []
        for stage in self.stages:
            for metric in stage.costs if costs else stage.quality:
                effects = [metric.whole_population, *metric.strata]
                for effect in effects:
                    if effect is not None and effect.effect is Effect.REGRESSED:
                        assert effect.delta is not None
                        found.append(
                            Regression(
                                stage=stage.stage,
                                metric=metric.metric,
                                version=metric.version,
                                strata=effect.strata,
                                delta=effect.delta,
                            )
                        )
        return tuple(found)

    @property
    def quality_regressions(self) -> tuple[Regression, ...]:
        """Return every quality effect that got worse, whole population and strata."""
        return self._regressions(costs=False)

    @property
    def cost_regressions(self) -> tuple[Regression, ...]:
        """Return every cost that got worse (more expensive), separately from quality."""
        return self._regressions(costs=True)

    @property
    def has_quality_improvement(self) -> bool:
        """Return whether any quality effect improved."""
        return any(
            effect is not None and effect.effect is Effect.IMPROVED
            for stage in self.stages
            for metric in stage.quality
            for effect in (metric.whole_population, *metric.strata)
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible content the digest is computed over."""
        return {
            "schema": EVIDENCE_SCHEMA,
            "technique_id": self.technique_id,
            "protocol_version": self.protocol_version,
            "protocol_digest": self.protocol_digest,
            "reference_set": self.reference_set.to_record(),
            "registry": self.registry.to_record(),
            "policy": self.policy.to_record(),
            "baseline_arm_id": self.baseline_arm_id,
            "variant_arm_id": self.variant_arm_id,
            "complete": self.complete,
            "stages": [item.to_record() for item in self.stages],
            "unevaluated_stages": [item.to_record() for item in self.unevaluated_stages],
            "factors": [item.to_record() for item in self.factors],
            "quality_regressions": [item.to_record() for item in self.quality_regressions],
            "cost_regressions": [item.to_record() for item in self.cost_regressions],
        }

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the evidence."""
        return canonical_digest(self.to_record())


# -------------------------------------------------------------------------- building


def _effect(
    policy: EffectPolicy,
    definition: MetricDefinition,
    strata: Strata,
    baseline: MetricEntry | None,
    variant: MetricEntry | None,
) -> StratumEffect:
    def make(effect: Effect, reason: str, delta: float | None = None) -> StratumEffect:
        return StratumEffect(
            strata=strata,
            baseline_status=None if baseline is None else baseline.status,
            baseline_value=None if baseline is None else baseline.value,
            baseline_sample_count=None if baseline is None else baseline.sample_count,
            variant_status=None if variant is None else variant.status,
            variant_value=None if variant is None else variant.value,
            variant_sample_count=None if variant is None else variant.sample_count,
            delta=delta,
            effect=effect,
            reason=reason,
        )

    missing = [
        name for name, entry in (("baseline", baseline), ("variant", variant)) if entry is None
    ]
    if missing:
        return make(Effect.NOT_COMPARABLE, f"not reported by the {' and '.join(missing)} arm")
    assert baseline is not None and variant is not None
    unavailable = [
        f"{name}: {entry.status.value}"
        for name, entry in (("baseline", baseline), ("variant", variant))
        if entry.status is not MetricStatus.VALUE
    ]
    if unavailable:
        return make(Effect.NOT_COMPARABLE, "; ".join(unavailable))
    assert baseline.value is not None and variant.value is not None
    if baseline.sample_count != variant.sample_count:
        return make(
            Effect.NOT_COMPARABLE,
            f"the arms were computed over {baseline.sample_count} and {variant.sample_count} "
            "samples",
        )
    if baseline.sample_count is None or baseline.sample_count < policy.min_samples_per_stratum:
        return make(
            Effect.NOT_COMPARABLE,
            f"only {baseline.sample_count} sample(s); at least "
            f"{policy.min_samples_per_stratum} are needed",
        )
    delta = variant.value - baseline.value
    if abs(delta) <= policy.tolerance(definition.unit):
        return make(Effect.UNCHANGED, "within the tolerance", delta)
    if definition.direction is MetricDirection.NOT_ORDERED:
        return make(Effect.CHANGED, "the metric has no better direction", delta)
    better = delta > 0 if definition.direction is MetricDirection.HIGHER_IS_BETTER else delta < 0
    return make(Effect.IMPROVED if better else Effect.REGRESSED, "beyond the tolerance", delta)


def _check_strata(
    definition: MetricDefinition,
    strata: Strata,
    declared: Mapping[str, frozenset[str]],
) -> None:
    is_cost = definition.kind is MetricKind.PERFORMANCE
    allowed = COST_STRATA if is_cost else declared
    for factor, value in strata:
        if factor not in allowed:
            if is_cost:
                raise TechniqueEvidenceError(
                    f"cost metric {definition.key} is reported by stratum {factor!r}; a cost "
                    f"can only be split by {sorted(COST_STRATA)}"
                )
            raise TechniqueEvidenceError(
                f"metric {definition.key} is reported by stratum {factor!r}, which the "
                "reference set does not declare"
            )
        if value not in allowed[factor]:
            raise TechniqueEvidenceError(
                f"metric {definition.key} uses the value {value!r} of stratum {factor!r}, "
                f"which allows only {sorted(allowed[factor])}"
            )


def _metric_evidence(
    definition: MetricDefinition,
    policy: EffectPolicy,
    declared: Mapping[str, frozenset[str]],
    grouped: Mapping[tuple[str, str], list[MetricComparison]],
) -> MetricEvidence:
    whole: StratumEffect | None = None
    per_stratum: list[StratumEffect] = []
    for item in grouped.get((definition.name, definition.version), []):
        _check_strata(definition, item.strata, declared)
        entries = {entry.arm_id: entry for entry in item.entries}
        effect = _effect(
            policy,
            definition,
            item.strata,
            entries.get(BASELINE_ARM_ID),
            entries.get(VARIANT_ARM_ID),
        )
        if item.strata:
            per_stratum.append(effect)
        else:
            whole = effect
    return MetricEvidence(
        metric=definition.name,
        version=definition.version,
        kind=definition.kind,
        unit=definition.unit,
        direction=definition.direction,
        whole_population=whole,
        strata=tuple(per_stratum),
    )


def build_technique_evidence(
    protocol: TechniqueProtocol,
    comparisons: Mapping[EvaluationStage, ComparisonManifest],
    *,
    registry: MetricRegistry,
    reference_set: ReferenceSetManifest,
    policy: EffectPolicy,
) -> TechniqueEvidence:
    """Turn the comparisons of a protocol's experiments into per-stratum evidence.

    A stage without a comparison is listed as unevaluated, and a stage whose arm did
    not complete is listed as incomplete with the reason: neither is read as neutral.
    Quality strata must be declared by the reference set; cost strata may only be the
    device or the artifact role.

    Raises:
        TechniqueEvidenceError: If a comparison does not belong to the protocol, the
            reference set or the registry, is for a stage the protocol does not
            evaluate, uses an undeclared stratum, or the policy lacks a tolerance.
    """
    if protocol.experiments[0].selection.reference_set != reference_set.identity():
        raise TechniqueEvidenceError(
            "the protocol was built for another reference set than the one given"
        )
    unknown = sorted(stage.value for stage in comparisons if stage not in protocol.stages)
    if unknown:
        raise TechniqueEvidenceError(f"comparisons for stages outside the protocol: {unknown}")
    declared = {item.name: frozenset(item.values) for item in reference_set.stratum_definitions}

    stages: list[StageEvidence] = []
    unevaluated: list[UnevaluatedStage] = []
    reported: set[str] = set()
    for experiment in protocol.experiments:
        stage = experiment.evaluated_stage
        comparison = comparisons.get(stage)
        if comparison is None:
            unevaluated.append(
                UnevaluatedStage(
                    stage=stage,
                    reason="no comparison was provided: this stage is not evaluated yet",
                )
            )
            continue
        identity = experiment_artifact_identity(experiment)
        if comparison.experiment != identity:
            raise TechniqueEvidenceError(
                f"the comparison for stage {stage.value!r} belongs to another experiment than "
                "the protocol's"
            )
        if comparison.reference_set != reference_set.identity():
            raise TechniqueEvidenceError(
                f"the comparison for stage {stage.value!r} used another reference set"
            )
        if comparison.registry != registry.identity():
            raise TechniqueEvidenceError(
                f"the comparison for stage {stage.value!r} used another metric registry"
            )
        incomplete = tuple(
            IncompleteArm(
                arm_id=arm.arm_id,
                status=arm.status,
                reason="" if arm.failure is None else arm.failure.message,
            )
            for arm in comparison.arms
            if not arm.comparable
        )
        if incomplete:
            stages.append(
                StageEvidence(
                    stage=stage,
                    experiment=identity,
                    comparison_digest=comparison.digest(),
                    status=StageStatus.INCOMPLETE,
                    incomplete_arms=incomplete,
                    shared_artifacts=(),
                    quality=(),
                    costs=(),
                )
            )
            continue
        grouped: dict[tuple[str, str], list[MetricComparison]] = {}
        for item in comparison.metrics:
            grouped.setdefault((item.metric, item.version), []).append(item)
            if item.kind is MetricKind.QUALITY:
                reported.update(factor for factor, _ in item.strata)
        quality = tuple(
            _metric_evidence(registry.get(ref.name, ref.version), policy, declared, grouped)
            for ref in experiment.quality_metrics
        )
        costs = tuple(
            _metric_evidence(registry.get(ref.name, ref.version), policy, declared, grouped)
            for ref in experiment.resource_capture
        )
        stages.append(
            StageEvidence(
                stage=stage,
                experiment=identity,
                comparison_digest=comparison.digest(),
                status=StageStatus.EVALUATED,
                incomplete_arms=(),
                shared_artifacts=comparison.shared_artifacts,
                quality=quality,
                costs=costs,
            )
        )
    return TechniqueEvidence(
        technique_id=protocol.technique_id,
        protocol_version=protocol.protocol_version,
        protocol_digest=protocol.digest(),
        reference_set=reference_set.identity(),
        registry=registry.identity(),
        policy=policy,
        baseline_arm_id=BASELINE_ARM_ID,
        variant_arm_id=VARIANT_ARM_ID,
        stages=tuple(stages),
        unevaluated_stages=tuple(unevaluated),
        factors=tuple(
            FactorStatus(
                factor=name,
                declared_in_reference_set=name in declared,
                reported=name in reported,
            )
            for name in protocol.stratification_factors
        ),
    )


def write_technique_evidence(root: Path, evidence: TechniqueEvidence) -> None:
    """Atomically publish immutable evidence under ``root``.

    Raises:
        FileExistsError: If evidence already exists there.
    """
    write_immutable_json(
        root / EVIDENCE_FILENAME,
        {**evidence.to_record(), "digest": evidence.digest()},
        "technique evidence",
    )


# --------------------------------------------------------------------------- decisions


class TechniqueDecisionKind(Enum):
    """The decision a person records about an optional technique."""

    KEEP_OPTIONAL = "keep_optional"
    """Keep the technique available, off by default."""

    DEFER = "defer"
    """Not enough evidence yet."""

    CHANGE_DEFAULT = "change_default"
    """Propose making the technique part of the default profile."""


@dataclass(frozen=True, kw_only=True)
class TechniqueDecision:
    """A documented human decision about an optional technique, bound to its evidence.

    A record never changes any configuration: ``modifies_default_profile`` is always
    ``False``, and ``CHANGE_DEFAULT`` only *proposes* a change that must be
    re-validated through the canonical end-to-end acceptance flow.

    Attributes:
        technique_id: The technique the decision is about.
        decision: Keep optional, defer or change the default.
        evidence_digest: Digest of the exact evidence the decision rests on.
        decided_by: Who decided.
        rationale: Why, including what the technique costs.
        acknowledged_regressions: Quality regressions (by key) the decider accepts.
    """

    technique_id: str
    decision: TechniqueDecisionKind
    evidence_digest: str
    decided_by: str
    rationale: str
    acknowledged_regressions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require who decided and why."""
        require_text("technique_id", self.technique_id)
        require_text("decided_by", self.decided_by)
        require_text("rationale", self.rationale)

    @property
    def requires_e2e_revalidation(self) -> bool:
        """Return whether the decision has to pass the canonical end-to-end acceptance flow."""
        return self.decision is TechniqueDecisionKind.CHANGE_DEFAULT

    @property
    def modifies_default_profile(self) -> bool:
        """Return ``False``: a decision record never changes a configuration."""
        return False

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible content the digest is computed over."""
        return {
            "schema": DECISION_SCHEMA,
            "technique_id": self.technique_id,
            "decision": self.decision.value,
            "evidence_digest": self.evidence_digest,
            "decided_by": self.decided_by,
            "rationale": self.rationale,
            "acknowledged_regressions": list(self.acknowledged_regressions),
            "requires_e2e_revalidation": self.requires_e2e_revalidation,
            "modifies_default_profile": self.modifies_default_profile,
        }

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the decision."""
        return canonical_digest(self.to_record())


def record_technique_decision(
    evidence: TechniqueEvidence,
    *,
    decision: TechniqueDecisionKind,
    decided_by: str,
    rationale: str,
    acknowledged_regressions: tuple[str, ...] = (),
) -> TechniqueDecision:
    """Record a decision on the evidence, refusing one the evidence cannot support.

    - ``DEFER`` is always possible.
    - ``KEEP_OPTIONAL`` needs at least one evaluated stage.
    - ``CHANGE_DEFAULT`` needs complete evidence, at least one quality improvement, and every
      quality regression explicitly acknowledged by key; it stays a proposal that requires
      end-to-end re-validation.

    Raises:
        TechniqueEvidenceError: If the evidence does not support the decision, or an
            acknowledged key is not a regression of this evidence.
    """
    regressions = {item.key for item in evidence.quality_regressions}
    acknowledged = tuple(acknowledged_regressions)
    require_unique("acknowledged regression", acknowledged)
    for key in acknowledged:
        if key not in regressions:
            raise TechniqueEvidenceError(
                f"{key!r} is not a quality regression of this evidence; "
                f"known: {sorted(regressions)}"
            )
    if decision is TechniqueDecisionKind.KEEP_OPTIONAL and not any(
        item.status is StageStatus.EVALUATED for item in evidence.stages
    ):
        raise TechniqueEvidenceError(
            "keeping the technique optional needs at least one evaluated stage"
        )
    if decision is TechniqueDecisionKind.CHANGE_DEFAULT:
        if not evidence.complete:
            unevaluated = [item.stage.value for item in evidence.unevaluated_stages]
            incomplete = [
                item.stage.value
                for item in evidence.stages
                if item.status is StageStatus.INCOMPLETE
            ]
            raise TechniqueEvidenceError(
                "the default cannot change on incomplete evidence: "
                f"{unevaluated} unevaluated, {incomplete} incomplete"
            )
        if not evidence.has_quality_improvement:
            raise TechniqueEvidenceError("no quality effect improved: nothing supports a change")
        unacknowledged = sorted(regressions - set(acknowledged))
        if unacknowledged:
            raise TechniqueEvidenceError(
                f"regressions must be acknowledged before the default changes: {unacknowledged}"
            )
    return TechniqueDecision(
        technique_id=evidence.technique_id,
        decision=decision,
        evidence_digest=evidence.digest(),
        decided_by=decided_by,
        rationale=rationale,
        acknowledged_regressions=acknowledged,
    )


def write_technique_decision(root: Path, decision: TechniqueDecision) -> None:
    """Atomically publish an immutable decision record under ``root``.

    Raises:
        FileExistsError: If a decision already exists there.
    """
    write_immutable_json(
        root / DECISION_FILENAME,
        {**decision.to_record(), "digest": decision.digest()},
        "technique decision",
    )

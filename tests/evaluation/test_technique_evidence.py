"""Optional-technique evidence tests: where gains and regressions occur, and the decision."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from experiment_builders import REGISTRY, pinned
from technique_builders import (
    STRATA,
    Costs,
    Values,
    feature_resolution_protocol,
    quality_fusion_protocol,
    run_protocol,
    technique_executor,
    technique_reference,
)

from contextmap.evaluation.experiment_runner import (
    ArmStatus,
    ArmUnavailableError,
    ComparisonManifest,
    read_verified_document,
)
from contextmap.evaluation.metrics import EvaluationStage, MetricKind
from contextmap.evaluation.reference_integrity import ValidatedReferenceSet
from contextmap.evaluation.technique_evidence import (
    Effect,
    EffectPolicy,
    StageStatus,
    TechniqueDecisionKind,
    TechniqueEvidence,
    TechniqueEvidenceError,
    build_technique_evidence,
    record_technique_decision,
    write_technique_decision,
    write_technique_evidence,
)
from contextmap.evaluation.technique_protocols import (
    FEATURE_EXTRACTION_STAGE,
    SENSOR_ASSOCIATION_STAGE,
    TechniqueProtocol,
)

POLICY = EffectPolicy(
    policy_id="effects/1",
    tolerance_by_unit=(
        ("ratio", 0.01),
        ("pixels", 0.25),
        ("seconds", 0.1),
        ("bytes", 1000.0),
        ("per_second", 0.5),
        ("count", 0.0),
        ("score", 0.0),
        ("meters", 0.01),
        ("degrees", 0.1),
    ),
    min_samples_per_stratum=2,
)
WHOLE = ()
NEAR = (("range_band", "near"),)
FAR = (("range_band", "far"),)
REPROJECTION = "association.reprojection_error.median"
RECOVERY = "fusion.reference_recovery.rate"

VALUES: Values = {
    "baseline": {
        REPROJECTION: {WHOLE: (2.0, 2), NEAR: (1.0, 2), FAR: (3.0, 2)},
        RECOVERY: {WHOLE: (0.60, 2), NEAR: (0.70, 2), FAR: (0.50, 2)},
    },
    "variant": {
        REPROJECTION: {WHOLE: (1.5, 2), NEAR: (1.0, 2), FAR: (2.0, 2)},
        RECOVERY: {WHOLE: (0.62, 2), NEAR: (0.75, 2), FAR: (0.45, 2)},
    },
}
COSTS: Costs = {
    "baseline": {
        "runtime.wall_time": {WHOLE: 10.0},
        "runtime.peak_memory": {(("device", "gpu"),): 1000.0, (("device", "cpu"),): 500.0},
        "runtime.storage_size": {
            (("artifact_role", "final"),): 100.0,
            (("artifact_role", "intermediate"),): 200.0,
        },
    },
    "variant": {
        "runtime.wall_time": {WHOLE: 14.0},
        "runtime.peak_memory": {(("device", "gpu"),): 3000.0, (("device", "cpu"),): 500.0},
        "runtime.storage_size": {
            (("artifact_role", "final"),): 100.0,
            (("artifact_role", "intermediate"),): 5000.0,
        },
    },
}


@pytest.fixture
def validated(tmp_path: Path) -> ValidatedReferenceSet:
    return technique_reference(tmp_path / "reference")


def _evidence(
    validated: ValidatedReferenceSet,
    tmp_path: Path,
    *,
    values: Values = VALUES,
    costs: Costs = COSTS,
    only: tuple[EvaluationStage, ...] | None = None,
    protocol: TechniqueProtocol | None = None,
    failures: dict[str, BaseException] | None = None,
    policy: EffectPolicy = POLICY,
) -> TechniqueEvidence:
    protocol = protocol or feature_resolution_protocol(validated.manifest)
    comparisons = run_protocol(
        protocol,
        validated,
        tmp_path / "runs",
        technique_executor(values, costs=costs, failures=failures),
        only=only,
    )
    return build_technique_evidence(
        protocol,
        comparisons,
        registry=REGISTRY,
        reference_set=validated.manifest,
        policy=policy,
    )


def _quality(evidence: TechniqueEvidence, stage: EvaluationStage, metric: str):  # type: ignore[no-untyped-def]
    stage_evidence = next(item for item in evidence.stages if item.stage is stage)
    return next(item for item in stage_evidence.quality if item.metric == metric)


def _cost(evidence: TechniqueEvidence, stage: EvaluationStage, metric: str):  # type: ignore[no-untyped-def]
    stage_evidence = next(item for item in evidence.stages if item.stage is stage)
    return next(item for item in stage_evidence.costs if item.metric == metric)


def _by_strata(metric):  # type: ignore[no-untyped-def]
    return {item.strata: item for item in metric.strata}


# ------------------------------------------------------------ gains and regressions


def test_evidence_shows_where_a_technique_gains_and_where_it_regresses(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path)

    reprojection = _quality(evidence, EvaluationStage.SENSOR_ASSOCIATION, REPROJECTION)
    recovery = _quality(evidence, EvaluationStage.SEMANTIC_FUSION, RECOVERY)

    assert evidence.complete
    assert reprojection.whole_population is not None
    assert reprojection.whole_population.effect is Effect.IMPROVED
    assert reprojection.whole_population.delta == pytest.approx(-0.5)
    by_strata = _by_strata(reprojection)
    assert by_strata[NEAR].effect is Effect.UNCHANGED
    assert by_strata[FAR].effect is Effect.IMPROVED
    assert recovery.whole_population is not None
    assert recovery.whole_population.effect is Effect.IMPROVED
    assert _by_strata(recovery)[NEAR].effect is Effect.IMPROVED
    assert _by_strata(recovery)[FAR].effect is Effect.REGRESSED
    assert {item.key for item in evidence.quality_regressions} == {
        "semantic_fusion:fusion.reference_recovery.rate/1@range_band=far"
    }
    assert evidence.has_quality_improvement


def test_a_global_improvement_cannot_hide_a_regression_in_a_stratum(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path)

    recovery = _quality(evidence, EvaluationStage.SEMANTIC_FUSION, RECOVERY)
    far = _by_strata(recovery)[FAR]

    assert recovery.whole_population is not None
    assert recovery.whole_population.effect is Effect.IMPROVED
    assert far.effect is Effect.REGRESSED
    assert far.delta == pytest.approx(-0.05)
    assert (far.baseline_value, far.variant_value) == (0.5, 0.45)


def test_metrics_without_a_difference_are_unchanged_not_missing(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path)

    entity = _quality(evidence, EvaluationStage.ENTITY_RESOLUTION, "entity.duplicate.rate")

    assert entity.whole_population is not None
    assert entity.whole_population.effect is Effect.UNCHANGED
    assert entity.direction.value == "lower_is_better"


def test_quality_and_cost_are_separate_sections(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path)

    stage = next(
        item for item in evidence.stages if item.stage is EvaluationStage.SENSOR_ASSOCIATION
    )

    assert all(item.kind is MetricKind.QUALITY for item in stage.quality)
    assert all(item.kind is MetricKind.PERFORMANCE for item in stage.costs)
    assert {item.metric for item in stage.costs} == {
        "runtime.wall_time",
        "runtime.peak_memory",
        "runtime.storage_size",
        "runtime.throughput",
        "runtime.failure_rate",
    }
    wall = _cost(evidence, EvaluationStage.SENSOR_ASSOCIATION, "runtime.wall_time")
    assert wall.whole_population is not None
    assert wall.whole_population.effect is Effect.REGRESSED
    assert wall.whole_population.delta == pytest.approx(4.0)
    assert not {item.key for item in evidence.quality_regressions} & {
        item.key for item in evidence.cost_regressions
    }


def test_memory_and_storage_are_reported_per_device_and_artifact_role(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path)

    memory = _by_strata(_cost(evidence, EvaluationStage.SENSOR_ASSOCIATION, "runtime.peak_memory"))
    storage = _by_strata(
        _cost(evidence, EvaluationStage.SENSOR_ASSOCIATION, "runtime.storage_size")
    )

    assert memory[(("device", "gpu"),)].effect is Effect.REGRESSED
    assert memory[(("device", "cpu"),)].effect is Effect.UNCHANGED
    assert storage[(("artifact_role", "intermediate"),)].effect is Effect.REGRESSED
    assert storage[(("artifact_role", "final"),)].effect is Effect.UNCHANGED
    keys = {item.key for item in evidence.cost_regressions}
    assert "sensor_association:runtime.peak_memory/1@device=gpu" in keys
    assert "sensor_association:runtime.wall_time/1@whole" in keys


def test_the_evidence_never_combines_metrics_or_ranks_anything(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    record = _evidence(validated, tmp_path).to_record()

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    assert not keys(record) & {"score", "winner", "rank", "ranking", "best", "overall", "verdict"}
    json.dumps(record)


# ------------------------------------------------------ not comparable, never zero


def test_a_stratum_with_too_few_samples_is_not_comparable(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    values: Values = {
        "baseline": {RECOVERY: {WHOLE: (0.6, 2), FAR: (0.5, 1)}},
        "variant": {RECOVERY: {WHOLE: (0.62, 2), FAR: (0.1, 1)}},
    }

    evidence = _evidence(validated, tmp_path, values=values)

    far = _by_strata(_quality(evidence, EvaluationStage.SEMANTIC_FUSION, RECOVERY))[FAR]
    assert far.effect is Effect.NOT_COMPARABLE
    assert "only 1 sample" in far.reason
    assert far.delta is None
    assert evidence.quality_regressions == ()


def test_not_applicable_stays_not_applicable_and_is_never_a_zero(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    values: Values = {
        "baseline": {RECOVERY: {WHOLE: (0.6, 2), NEAR: (0.7, 2)}},
        "variant": {RECOVERY: {WHOLE: (0.6, 2), NEAR: (None, 2)}},
    }

    evidence = _evidence(validated, tmp_path, values=values)

    near = _by_strata(_quality(evidence, EvaluationStage.SEMANTIC_FUSION, RECOVERY))[NEAR]
    assert near.effect is Effect.NOT_COMPARABLE
    assert near.reason == "variant: not_applicable"
    assert near.variant_value is None and near.delta is None


def test_arms_computed_over_different_populations_are_not_compared(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    values: Values = {
        "baseline": {RECOVERY: {WHOLE: (0.6, 2)}},
        "variant": {RECOVERY: {WHOLE: (0.9, 3)}},
    }

    evidence = _evidence(validated, tmp_path, values=values)

    whole = _quality(evidence, EvaluationStage.SEMANTIC_FUSION, RECOVERY).whole_population
    assert whole is not None
    assert whole.effect is Effect.NOT_COMPARABLE
    assert "2 and 3 samples" in whole.reason


def test_a_stratum_reported_by_one_arm_only_is_not_comparable(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    values: Values = {
        "baseline": {RECOVERY: {WHOLE: (0.6, 2), NEAR: (0.7, 2)}},
        "variant": {RECOVERY: {WHOLE: (0.6, 2)}},
    }

    evidence = _evidence(validated, tmp_path, values=values)

    near = _by_strata(_quality(evidence, EvaluationStage.SEMANTIC_FUSION, RECOVERY))[NEAR]
    assert near.effect is Effect.NOT_COMPARABLE
    assert "not reported by the variant" in near.reason


# ------------------------------------------------------------- strata and the reference set


def test_strata_must_be_declared_and_valid_in_the_reference_set(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    undeclared: Values = {
        "baseline": {RECOVERY: {WHOLE: (0.6, 2), (("weather", "rain"),): (0.6, 2)}},
        "variant": {RECOVERY: {WHOLE: (0.6, 2), (("weather", "rain"),): (0.6, 2)}},
    }
    bad_value: Values = {
        "baseline": {RECOVERY: {WHOLE: (0.6, 2), (("range_band", "medium"),): (0.6, 2)}},
        "variant": {RECOVERY: {WHOLE: (0.6, 2), (("range_band", "medium"),): (0.6, 2)}},
    }

    with pytest.raises(TechniqueEvidenceError, match="does not declare"):
        _evidence(validated, tmp_path / "a", values=undeclared)
    with pytest.raises(TechniqueEvidenceError, match="allows only"):
        _evidence(validated, tmp_path / "b", values=bad_value)


def test_costs_can_only_be_split_by_device_or_artifact_role(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    costs: Costs = {
        "baseline": {"runtime.wall_time": {NEAR: 1.0}},
        "variant": {"runtime.wall_time": {NEAR: 1.0}},
    }

    with pytest.raises(TechniqueEvidenceError, match="a cost can only be split by"):
        _evidence(validated, tmp_path, costs=costs)


def test_factor_availability_is_explicit(validated: ValidatedReferenceSet, tmp_path: Path) -> None:
    evidence = _evidence(validated, tmp_path)

    factors = {item.factor: item for item in evidence.factors}

    assert all(item.declared_in_reference_set for item in factors.values())
    assert factors["range_band"].reported
    assert not factors["visibility"].reported


def test_factors_the_reference_set_lacks_are_reported_as_undeclared(tmp_path: Path) -> None:
    sparse = technique_reference(tmp_path / "sparse", strata=STRATA[:2])

    evidence = _evidence(sparse, tmp_path / "out")

    factors = {item.factor: item for item in evidence.factors}
    assert factors["range_band"].declared_in_reference_set
    assert not factors["support_density"].declared_in_reference_set
    assert not factors["support_density"].reported


# ---------------------------------------------------------- unavailable, missing, shared


def test_a_stage_without_a_comparison_is_missing_evidence_not_neutral(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path, only=(EvaluationStage.SENSOR_ASSOCIATION,))

    assert not evidence.complete
    assert {item.stage for item in evidence.unevaluated_stages} == {
        EvaluationStage.SEMANTIC_FUSION,
        EvaluationStage.ENTITY_RESOLUTION,
        EvaluationStage.SPATIAL_RELATIONS,
    }
    assert "not evaluated yet" in evidence.unevaluated_stages[0].reason


def test_an_unavailable_enhancement_backend_stays_explicit(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(
        validated,
        tmp_path,
        failures={"variant": ArmUnavailableError("the resolution enhancer is not installed")},
    )

    assert not evidence.complete
    for stage in evidence.stages:
        assert stage.status is StageStatus.INCOMPLETE
        assert stage.quality == () and stage.costs == ()
        assert stage.incomplete_arms[0].arm_id == "variant"
        assert stage.incomplete_arms[0].status is ArmStatus.UNAVAILABLE
        assert "not installed" in stage.incomplete_arms[0].reason
    assert evidence.quality_regressions == ()


def test_the_shared_upstream_artifacts_are_part_of_the_evidence(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path)

    stage = next(
        item for item in evidence.stages if item.stage is EvaluationStage.SENSOR_ASSOCIATION
    )
    shared = {item.stage_id: item for item in stage.shared_artifacts}

    assert shared[FEATURE_EXTRACTION_STAGE].artifact == pinned("perception_run", "dino-run-X")
    assert shared[FEATURE_EXTRACTION_STAGE].arm_ids == ("baseline", "variant")
    assert SENSOR_ASSOCIATION_STAGE not in shared


def test_quality_aware_fusion_evidence_shares_the_association_artifact(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    high = (("ambiguity_level", "high"),)
    values: Values = {
        "baseline": {RECOVERY: {WHOLE: (0.6, 2), high: (0.4, 2)}},
        "variant": {RECOVERY: {WHOLE: (0.62, 2), high: (0.55, 2)}},
    }
    protocol = quality_fusion_protocol(validated.manifest)

    evidence = _evidence(validated, tmp_path, values=values, protocol=protocol)

    fusion = next(item for item in evidence.stages if item.stage is EvaluationStage.SEMANTIC_FUSION)
    recovery = next(item for item in fusion.quality if item.metric == RECOVERY)
    assert _by_strata(recovery)[high].effect is Effect.IMPROVED
    assert {item.stage_id for item in fusion.shared_artifacts} == {SENSOR_ASSOCIATION_STAGE}
    factors = {item.factor: item for item in evidence.factors}
    assert (
        factors["ambiguity_level"].declared_in_reference_set and factors["ambiguity_level"].reported
    )
    assert not factors["quality.support_depth"].declared_in_reference_set


# ------------------------------------------------------------------- binding and policy


def _comparisons(
    validated: ValidatedReferenceSet, tmp_path: Path, protocol: TechniqueProtocol
) -> dict[EvaluationStage, ComparisonManifest]:
    return run_protocol(protocol, validated, tmp_path, technique_executor(VALUES, costs=COSTS))


def test_a_comparison_of_another_experiment_is_refused(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    protocol = feature_resolution_protocol(validated.manifest)
    comparisons = _comparisons(validated, tmp_path, protocol)
    swapped = {
        EvaluationStage.SENSOR_ASSOCIATION: comparisons[EvaluationStage.SEMANTIC_FUSION],
        EvaluationStage.SEMANTIC_FUSION: comparisons[EvaluationStage.SENSOR_ASSOCIATION],
    }

    with pytest.raises(TechniqueEvidenceError, match="another experiment"):
        build_technique_evidence(
            protocol, swapped, registry=REGISTRY, reference_set=validated.manifest, policy=POLICY
        )


def test_comparisons_for_stages_outside_the_protocol_are_refused(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    fusion_protocol = quality_fusion_protocol(validated.manifest)
    resolution = feature_resolution_protocol(validated.manifest)
    comparisons = _comparisons(validated, tmp_path, resolution)

    with pytest.raises(TechniqueEvidenceError, match="outside the protocol"):
        build_technique_evidence(
            fusion_protocol,
            comparisons,
            registry=REGISTRY,
            reference_set=validated.manifest,
            policy=POLICY,
        )


def test_a_protocol_for_another_reference_set_is_refused(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    protocol = feature_resolution_protocol(validated.manifest)
    other = replace(validated.manifest, version="2.0.0")

    with pytest.raises(TechniqueEvidenceError, match="another reference set than the one given"):
        build_technique_evidence(
            protocol, {}, registry=REGISTRY, reference_set=other, policy=POLICY
        )


def test_a_comparison_from_another_registry_is_refused(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    protocol = feature_resolution_protocol(validated.manifest)
    comparisons = _comparisons(validated, tmp_path, protocol)
    other = replace(REGISTRY, registry_version="9")

    with pytest.raises(TechniqueEvidenceError, match="another metric registry"):
        build_technique_evidence(
            protocol, comparisons, registry=other, reference_set=validated.manifest, policy=POLICY
        )


def test_the_policy_must_declare_every_unit_it_judges_and_only_known_ones(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    missing = replace(
        POLICY,
        tolerance_by_unit=tuple(item for item in POLICY.tolerance_by_unit if item[0] != "pixels"),
    )

    with pytest.raises(TechniqueEvidenceError, match="no tolerance for unit 'pixels'"):
        _evidence(validated, tmp_path, policy=missing)
    with pytest.raises(ValueError, match="unknown unit"):
        replace(POLICY, tolerance_by_unit=(("furlongs", 1.0),))
    with pytest.raises(ValueError, match=">= 0"):
        replace(POLICY, tolerance_by_unit=(("ratio", -0.1),))
    with pytest.raises(ValueError, match="at least 1"):
        replace(POLICY, min_samples_per_stratum=0)


def test_evidence_is_reproducible_and_bound_to_its_protocol(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    first = _evidence(validated, tmp_path / "one")
    second = _evidence(validated, tmp_path / "two")

    assert first.digest() == second.digest()
    assert first.protocol_digest == feature_resolution_protocol(validated.manifest).digest()
    assert first.reference_set == validated.manifest.identity()
    assert first.registry == REGISTRY.identity()


def test_evidence_is_written_immutably_and_verified_on_read(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path / "runs")
    root = tmp_path / "evidence"

    write_technique_evidence(root, evidence)

    document = read_verified_document(root / "evidence.json")
    assert document["digest"] == evidence.digest()
    assert document["complete"] is True
    with pytest.raises(FileExistsError):
        write_technique_evidence(root, evidence)


# ------------------------------------------------------------------------- decisions


def test_deferring_is_always_possible_even_on_incomplete_evidence(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path, only=(EvaluationStage.SENSOR_ASSOCIATION,))

    decision = record_technique_decision(
        evidence,
        decision=TechniqueDecisionKind.DEFER,
        decided_by="reviewer",
        rationale="entity and relation quality are not evaluated yet",
    )

    assert decision.decision is TechniqueDecisionKind.DEFER
    assert decision.evidence_digest == evidence.digest()
    assert not decision.requires_e2e_revalidation


def test_keeping_a_technique_optional_needs_an_evaluated_stage(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    protocol = feature_resolution_protocol(validated.manifest)
    empty = build_technique_evidence(
        protocol, {}, registry=REGISTRY, reference_set=validated.manifest, policy=POLICY
    )
    evaluated = _evidence(validated, tmp_path)

    with pytest.raises(TechniqueEvidenceError, match="at least one evaluated stage"):
        record_technique_decision(
            empty,
            decision=TechniqueDecisionKind.KEEP_OPTIONAL,
            decided_by="reviewer",
            rationale="nothing was measured",
        )
    kept = record_technique_decision(
        evaluated,
        decision=TechniqueDecisionKind.KEEP_OPTIONAL,
        decided_by="reviewer",
        rationale="helps far range, regresses fusion recovery there, costs GPU memory",
    )
    assert kept.decision is TechniqueDecisionKind.KEEP_OPTIONAL


def test_the_default_cannot_change_on_incomplete_evidence(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    missing = _evidence(validated, tmp_path / "a", only=(EvaluationStage.SENSOR_ASSOCIATION,))
    unavailable = _evidence(
        validated, tmp_path / "b", failures={"variant": ArmUnavailableError("no backend")}
    )

    for evidence in (missing, unavailable):
        with pytest.raises(TechniqueEvidenceError, match="incomplete evidence"):
            record_technique_decision(
                evidence,
                decision=TechniqueDecisionKind.CHANGE_DEFAULT,
                decided_by="reviewer",
                rationale="looks good",
            )


def test_every_quality_regression_must_be_acknowledged_before_the_default_changes(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path)
    key = "semantic_fusion:fusion.reference_recovery.rate/1@range_band=far"

    with pytest.raises(TechniqueEvidenceError, match="must be acknowledged"):
        record_technique_decision(
            evidence,
            decision=TechniqueDecisionKind.CHANGE_DEFAULT,
            decided_by="reviewer",
            rationale="improves association",
        )
    decision = record_technique_decision(
        evidence,
        decision=TechniqueDecisionKind.CHANGE_DEFAULT,
        decided_by="reviewer",
        rationale="improves association at far range; the far-range fusion regression is accepted",
        acknowledged_regressions=(key,),
    )

    assert decision.acknowledged_regressions == (key,)
    assert decision.requires_e2e_revalidation
    assert not decision.modifies_default_profile
    assert decision.to_record()["modifies_default_profile"] is False


def test_an_acknowledgement_must_name_a_real_regression(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path)

    with pytest.raises(TechniqueEvidenceError, match="is not a quality regression"):
        record_technique_decision(
            evidence,
            decision=TechniqueDecisionKind.KEEP_OPTIONAL,
            decided_by="reviewer",
            rationale="kept",
            acknowledged_regressions=("semantic_fusion:made.up/1@whole",),
        )


def test_nothing_supports_a_change_when_no_quality_effect_improved(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    flat = _evidence(validated, tmp_path, values={}, costs={})

    with pytest.raises(TechniqueEvidenceError, match="nothing supports a change"):
        record_technique_decision(
            flat,
            decision=TechniqueDecisionKind.CHANGE_DEFAULT,
            decided_by="reviewer",
            rationale="cheaper to maintain",
        )


def test_a_decision_needs_a_person_and_a_rationale(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path)

    with pytest.raises(ValueError, match="decided_by"):
        record_technique_decision(
            evidence, decision=TechniqueDecisionKind.DEFER, decided_by=" ", rationale="later"
        )
    with pytest.raises(ValueError, match="rationale"):
        record_technique_decision(
            evidence, decision=TechniqueDecisionKind.DEFER, decided_by="reviewer", rationale=""
        )


def test_a_decision_is_written_immutably_and_never_touches_a_profile(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    evidence = _evidence(validated, tmp_path / "runs")
    decision = record_technique_decision(
        evidence,
        decision=TechniqueDecisionKind.DEFER,
        decided_by="reviewer",
        rationale="wait for entity and relation evaluators",
    )
    root = tmp_path / "decision"

    write_technique_decision(root, decision)

    document = read_verified_document(root / "decision.json")
    assert document["evidence_digest"] == evidence.digest()
    assert document["modifies_default_profile"] is False
    assert document["decision"] == "defer"
    with pytest.raises(FileExistsError):
        write_technique_decision(root, decision)

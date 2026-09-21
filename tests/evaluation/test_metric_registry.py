"""Stage-level metric registry tests."""

import json
from dataclasses import replace

import pytest

from contextmap.evaluation.annotations import AnnotationFamily
from contextmap.evaluation.metrics import (
    EvaluationStage,
    MetricCompatibilityError,
    MetricDefinition,
    MetricDirection,
    MetricKind,
    MetricRegistry,
    MetricRegistryError,
    MissingDataBehavior,
    default_metric_registry,
    require_annotation_compatibility,
)

REGIONS = AnnotationFamily.REGIONS.schema


def _definition(**overrides: object) -> MetricDefinition:
    fields: dict[str, object] = {
        "name": "region.iou.mean",
        "version": "1",
        "stage": EvaluationStage.REGION_DISCOVERY,
        "kind": MetricKind.QUALITY,
        "description": "mean IoU of matched regions",
        "population": "annotated regions of frames with a regions annotation",
        "unit": "ratio",
        "minimum": 0.0,
        "maximum": 1.0,
        "direction": MetricDirection.HIGHER_IS_BETTER,
        "aggregation": "mean over annotated frames of the per-frame mean IoU",
        "required_annotations": (REGIONS,),
        "missing_data": MissingDataBehavior.EXCLUDE_UNANNOTATED,
        "evaluator_id": "region-discovery-evaluator",
        "evaluator_version": "1",
    }
    fields.update(overrides)
    return MetricDefinition(**fields)  # type: ignore[arg-type]


def test_a_definition_identifies_everything_the_issue_requires() -> None:
    definition = _definition()

    record = definition.to_record()

    assert definition.key == "region.iou.mean/1"
    for field in (
        "name",
        "version",
        "population",
        "unit",
        "minimum",
        "maximum",
        "direction",
        "aggregation",
        "required_annotations",
        "missing_data",
        "evaluator_id",
        "evaluator_version",
    ):
        assert field in record
    assert MetricDefinition.from_record(json.loads(json.dumps(record))) == definition


def test_definitions_reject_meaningless_or_misleading_fields() -> None:
    with pytest.raises(ValueError, match="unit"):
        _definition(unit="probability")
    with pytest.raises(ValueError, match="range"):
        _definition(minimum=1.0, maximum=0.0)
    with pytest.raises(ValueError, match="annotation schema"):
        _definition(required_annotations=("contextmap.reference.regions/v9",))
    with pytest.raises(ValueError, match="stage"):
        _definition(stage=None)
    with pytest.raises(ValueError, match="annotation"):
        _definition(kind=MetricKind.PERFORMANCE, stage=None, unit="seconds")
    with pytest.raises(ValueError, match="population"):
        _definition(population=" ")


def test_a_performance_metric_may_span_every_stage() -> None:
    wall_time = _definition(
        name="runtime.wall_time",
        kind=MetricKind.PERFORMANCE,
        stage=None,
        unit="seconds",
        minimum=0.0,
        maximum=None,
        direction=MetricDirection.LOWER_IS_BETTER,
        required_annotations=(),
        missing_data=MissingDataBehavior.NOT_APPLICABLE,
    )

    assert wall_time.stage is None
    assert wall_time.kind is MetricKind.PERFORMANCE


def test_registry_keeps_metric_versions_apart_and_rejects_duplicates() -> None:
    first = _definition()
    second = _definition(version="2")

    registry = MetricRegistry(
        registry_id="test-metrics", registry_version="1", definitions=(first, second)
    )

    assert registry.get("region.iou.mean", "1") == first
    assert registry.get("region.iou.mean", "2") == second
    with pytest.raises(MetricRegistryError, match="unknown"):
        registry.get("region.iou.mean", "3")
    with pytest.raises(MetricRegistryError, match="unknown"):
        registry.get("region.dice.mean", "1")
    with pytest.raises(ValueError, match="unique"):
        MetricRegistry(registry_id="r", registry_version="1", definitions=(first, first))


def test_the_default_registry_covers_every_stage_of_solution_1() -> None:
    registry = default_metric_registry()

    stages_with_quality = {
        item.stage for item in registry.definitions if item.kind is MetricKind.QUALITY
    }

    assert stages_with_quality == set(EvaluationStage) - {EvaluationStage.RUNTIME}
    assert {
        "ingestion_integrity",
        "state_estimation",
        "region_discovery",
        "feature_extraction",
        "semantic_interpretation",
        "sensor_association",
        "semantic_fusion",
        "entity_resolution",
        "spatial_relations",
        "artifact_integrity",
        "runtime",
    } <= {stage.value for stage in EvaluationStage}
    assert any(item.kind is MetricKind.PERFORMANCE for item in registry.definitions)
    assert registry.definitions_for(EvaluationStage.SEMANTIC_FUSION)


def test_default_definitions_are_machine_readable_and_consistent() -> None:
    registry = default_metric_registry()

    rebuilt = MetricRegistry.from_record(json.loads(json.dumps(registry.to_record())))

    assert rebuilt == registry
    assert rebuilt.digest() == registry.digest()
    for definition in registry.definitions:
        assert definition.unit != "probability"
        if definition.kind is MetricKind.PERFORMANCE:
            assert not definition.required_annotations
    quality = [item for item in registry.definitions if item.kind is MetricKind.QUALITY]
    assert all(item.stage is not None for item in quality)
    assert "overall" not in {item.name.split(".")[0] for item in registry.definitions}


def test_the_registry_identity_changes_with_its_content() -> None:
    registry = default_metric_registry()
    other = replace(registry, registry_version="9")

    assert registry.identity().digest != other.identity().digest
    assert registry.identity().registry_id == registry.registry_id


def test_evaluators_reject_incompatible_annotation_versions() -> None:
    definition = _definition()

    require_annotation_compatibility(definition, {REGIONS, AnnotationFamily.SEMANTICS.schema})
    with pytest.raises(MetricCompatibilityError, match="regions"):
        require_annotation_compatibility(definition, {"contextmap.reference.regions/v2"})
    with pytest.raises(MetricCompatibilityError, match="regions"):
        require_annotation_compatibility(definition, set())
    no_annotations = _definition(
        name="runtime.wall_time",
        kind=MetricKind.PERFORMANCE,
        stage=None,
        unit="seconds",
        required_annotations=(),
        minimum=0.0,
        maximum=None,
    )
    require_annotation_compatibility(no_annotations, set())


def test_the_registry_carries_the_metrics_the_optional_technique_protocols_need() -> None:
    registry = default_metric_registry()

    assert registry.registry_version == "2"
    expected = {
        "association.feature_anchoring.rate": EvaluationStage.SENSOR_ASSOCIATION,
        "fusion.view_consistency.rate": EvaluationStage.SEMANTIC_FUSION,
        "entity.semantic_accuracy.rate": EvaluationStage.ENTITY_RESOLUTION,
    }
    for name, stage in expected.items():
        definition = registry.get(name, "1")
        assert definition.stage is stage
        assert definition.kind is MetricKind.QUALITY
        assert (definition.minimum, definition.maximum) == (0.0, 1.0)
    accuracy = registry.get("entity.semantic_accuracy.rate", "1")
    assert set(accuracy.required_annotations) == {
        AnnotationFamily.SEMANTICS.schema,
        AnnotationFamily.IDENTITY.schema,
    }
    failure = registry.get("runtime.failure_rate", "1")
    assert failure.kind is MetricKind.PERFORMANCE
    assert failure.maximum == 1.0
    assert failure.direction is MetricDirection.LOWER_IS_BETTER
    assert not failure.required_annotations

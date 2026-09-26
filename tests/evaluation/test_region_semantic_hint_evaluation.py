"""Scoring Florence-2 task-native region text against semantic references."""

from __future__ import annotations

import json
from hashlib import sha256

import pytest

from contextmap.evaluation import (
    MATCHING_POLICY,
    EvaluatedDiscoveryFrame,
    EvaluationRunDescriptor,
    GroundTruthRegion,
    ReferenceFrame,
    RegionDiscoveryEvaluator,
    RegionDiscoveryReferenceSet,
    RegionSemanticHintInput,
    SemanticAnnotation,
    SemanticEvaluationContext,
    SemanticEvaluationError,
    encode_region_semantic_hint_report,
    evaluate_region_semantic_hints,
)
from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    HintContribution,
    InlineMask,
    PreparedImage,
    RegionSemanticHint,
    derive_region_semantic_hints,
    normalize_regions,
    run_discovery_passes,
)
from contextmap.visual_perception.backends.florence2 import (
    Florence2Config,
    Florence2NativeOutput,
    Florence2NativeRegion,
    Florence2RegionDiscovery,
)
from contextmap.visual_perception.discovery import DiscoveryInput

_BOXES = ((0.0, 0.0, 2.0, 2.0), (3.0, 1.0, 5.0, 3.0), (5.0, 0.0, 6.0, 1.0))


class CountingFlorence2Runtime:
    """Deterministic Florence-2 runtime that records how many native inferences ran."""

    def __init__(self, labels: tuple[str | None, ...]) -> None:
        self.labels = labels
        self.calls = 0

    def predict(
        self, discovery_input: DiscoveryInput, config: Florence2Config
    ) -> Florence2NativeOutput:
        self.calls += 1
        return Florence2NativeOutput(
            regions=tuple(
                Florence2NativeRegion(
                    proposal_id=f"florence2-box-{index:06d}",
                    box=box,
                    score=None,
                    parsed_text=label,
                )
                for index, (box, label) in enumerate(zip(_BOXES, self.labels, strict=True))
            )
        )


def _prepared() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-florence"),
        payload_reference="reference/frame-florence.png",
        payload_artifact=ArtifactReference(
            uri="reference/frame-florence.png",
            sha256=sha256(b"frame-florence").hexdigest(),
            media_type="image/png",
        ),
        width=6,
        height=4,
    )


def _discover(
    backend: Florence2RegionDiscovery, prepared: PreparedImage
) -> EvaluatedDiscoveryFrame:
    discovery = run_discovery_passes(
        prepared_image=prepared,
        backend=backend,
        perception_run_id="run-florence",
        perception_result_id="result-florence",
    )
    normalization = normalize_regions(discovery.candidates, prepared, backend.backend_provenance())
    return EvaluatedDiscoveryFrame(discovery=discovery, normalization=normalization)


def _backend(task: str, labels: tuple[str | None, ...]) -> Florence2RegionDiscovery:
    return Florence2RegionDiscovery(
        config=Florence2Config(checkpoint="florence-community/Florence-2-large", task=task),
        runtime=CountingFlorence2Runtime(labels),
    )


def _hints(task: str, labels: tuple[str | None, ...]) -> tuple[RegionSemanticHint, ...]:
    frame = _discover(_backend(task, labels), _prepared())
    return derive_region_semantic_hints(frame.discovery.candidates, frame.normalization)


def _context() -> SemanticEvaluationContext:
    return SemanticEvaluationContext(
        evaluation_id="eval-florence-od-hints",
        reference_set_version="reference-v1",
        selection_id="selection-1",
        perception_run_id="run-florence",
        artifact_id="stage-florence-od",
        pipeline_configuration_digest="sha256:pipeline",
        evaluator_version="1",
    )


def test_native_text_is_scored_against_references_with_partial_annotations() -> None:
    chair, table, lamp = _hints("<OD>", ("Chair ", "table", "lamp"))

    report = evaluate_region_semantic_hints(
        context=_context(),
        inputs=(
            RegionSemanticHintInput(
                hint=chair,
                annotation=SemanticAnnotation(acceptable_hypotheses=("chair", "armchair")),
            ),
            RegionSemanticHintInput(
                hint=table,
                annotation=SemanticAnnotation(
                    acceptable_hypotheses=("desk",), rejected_hypotheses=("table",)
                ),
            ),
            RegionSemanticHintInput(hint=lamp),
        ),
    )

    assert report.matching_policy == MATCHING_POLICY
    assert report.task == "<OD>"
    assert report.prompt is None
    assert report.backend_id == "florence2_region_discovery"
    assert report.checkpoint == "florence-community/Florence-2-large"
    assert report.config_digest == chair.provenance.config_digest
    assert [(sample.text, sample.acceptable, sample.rejected) for sample in report.samples] == [
        ("Chair ", True, None),
        ("table", False, True),
        ("lamp", None, None),
    ]
    assert report.samples[0].native_proposal_id == "florence2-box-000000"
    assert report.samples[0].region_id == chair.region_id
    assert report.samples[0].contribution == HintContribution.REPRESENTATIVE.value
    assert report.hint_count == 3
    assert report.annotated_hint_count == 2
    assert report.assessed_hint_count == 2
    assert report.acceptable_hint_rate == 0.5
    assert report.unsupported_hint_rate == 0.5
    assert report.rejected_hint_rate == 1.0
    encoded = encode_region_semantic_hint_report(report)
    assert json.loads(json.dumps(encoded)) == encoded
    assert encoded["samples"][2]["acceptable"] is None


def test_unannotated_hints_leave_quality_rates_undefined() -> None:
    hints = _hints("<DENSE_REGION_CAPTION>", ("a wooden chair", "a low table", "a lamp"))

    report = evaluate_region_semantic_hints(
        context=_context(),
        inputs=tuple(RegionSemanticHintInput(hint=hint) for hint in hints),
    )

    assert report.assessed_hint_count == 0
    assert report.acceptable_hint_rate is None
    assert report.unsupported_hint_rate is None
    assert report.rejected_hint_rate is None


def test_hint_evaluation_refuses_an_uncontrolled_or_empty_comparison() -> None:
    od = _hints("<OD>", ("chair", "table", "lamp"))
    captions = _hints("<DENSE_REGION_CAPTION>", ("a wooden chair", "a low table", "a lamp"))

    with pytest.raises(SemanticEvaluationError, match="native configuration"):
        evaluate_region_semantic_hints(
            context=_context(),
            inputs=(
                RegionSemanticHintInput(hint=od[0]),
                RegionSemanticHintInput(hint=captions[1]),
            ),
        )
    with pytest.raises(SemanticEvaluationError, match="at least one"):
        evaluate_region_semantic_hints(context=_context(), inputs=())
    with pytest.raises(SemanticEvaluationError, match="duplicate"):
        evaluate_region_semantic_hints(
            context=_context(),
            inputs=(RegionSemanticHintInput(hint=od[0]), RegionSemanticHintInput(hint=od[0])),
        )


def test_one_native_inference_is_evaluated_for_both_geometry_and_text() -> None:
    backend_runtime = CountingFlorence2Runtime(("chair", "table", None))
    backend = Florence2RegionDiscovery(
        config=Florence2Config(checkpoint="florence-community/Florence-2-large", task="<OD>"),
        runtime=backend_runtime,
    )
    prepared = _prepared()
    chair_mask = InlineMask(
        width=6,
        height=4,
        data=tuple(x < 2 and y < 2 for y in range(4) for x in range(6)),
    )
    reference_frame = ReferenceFrame(
        frame_id="frame-florence",
        source_condition="indoor",
        prepared_image=prepared,
        annotations=(GroundTruthRegion(region_id="gt-chair", mask=chair_mask),),
    )
    evaluated: list[EvaluatedDiscoveryFrame] = []

    def execute(frame: ReferenceFrame) -> EvaluatedDiscoveryFrame:
        evaluated.append(_discover(backend, frame.prepared_image))
        return evaluated[-1]

    provenance = backend.backend_provenance()
    geometry = RegionDiscoveryEvaluator().evaluate(
        RegionDiscoveryReferenceSet(version="reference-v1", frames=(reference_frame,)),
        EvaluationRunDescriptor(
            perception_run_id="run-florence",
            perception_artifact_id="stage-florence-od",
            backend_id=provenance.backend_id,
            backend_version=provenance.version,
            checkpoint=provenance.model,
            config_digest=provenance.configuration_fingerprint or "",
            pipeline_graph_digest="sha256:graph",
            strategy="<OD>",
            thresholds=(),
            variables=(("task", "<OD>"),),
            execution_kind="ci_contract",
        ),
        execute,
    )
    (frame,) = evaluated
    hints = derive_region_semantic_hints(frame.discovery.candidates, frame.normalization)
    semantics = evaluate_region_semantic_hints(
        context=_context(),
        inputs=tuple(
            RegionSemanticHintInput(
                hint=hint,
                annotation=SemanticAnnotation(acceptable_hypotheses=("chair", "table")),
            )
            for hint in hints
        ),
    )

    assert backend_runtime.calls == 1
    accuracy = geometry.frames[0].accuracy
    assert accuracy is not None
    assert accuracy.region_recall == 1.0
    assert [sample.text for sample in semantics.samples] == ["chair", "table"]
    assert semantics.acceptable_hint_rate == 1.0
    assert {sample.region_id for sample in semantics.samples} <= {
        str(region.region_id) for region in frame.normalization.regions
    }

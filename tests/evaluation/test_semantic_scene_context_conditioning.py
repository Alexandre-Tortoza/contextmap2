"""#529: the evaluator's with/without scene-context hook over real Qwen/Gemini requests.

The adapters are the real ones; only their model runtime/provider client is a deterministic
fake, so the requests, rendered prompts and parsed evidence are exactly what a run produces.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from contextmap.evaluation.semantic_interpretation import (
    SemanticEvaluationContext,
    SemanticEvaluationInput,
    compare_evidence_variants,
    evaluate_semantic_interpretation,
)
from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    SEMANTIC_PROMPT_TEMPLATES,
    BackendProvenance,
    PerceptionResultId,
    RegionId,
    SceneContext,
    SemanticEvidenceReference,
    SemanticInferenceProvenance,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
)
from contextmap.visual_perception.backends.gemini import (
    GeminiProviderResponse,
    GeminiSemanticConfig,
    GeminiSemanticInterpreter,
)
from contextmap.visual_perception.backends.qwen import (
    QwenGenerationResponse,
    QwenSemanticConfig,
    QwenSemanticInterpreter,
)

FRAME = SourceObservationId("frame-0001")
RESULT = PerceptionResultId("run-0001--frame-0001")
_ANSWER = json.dumps(
    {
        "abstained": False,
        "claims": [
            {
                "hypothesis": "pallet jack",
                "role": "primary",
                "category": None,
                "region_kind": "thing",
                "attributes": {},
            }
        ],
    }
)


class _QwenRuntime:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, *, visual_views: Any, prompt: str, config: Any) -> QwenGenerationResponse:
        self.prompts.append(prompt)
        return QwenGenerationResponse(text=_ANSWER)


class _GeminiClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, *, visual_views: Any, prompt: str, config: Any) -> GeminiProviderResponse:
        self.prompts.append(prompt)
        return GeminiProviderResponse(text=_ANSWER)


def _qwen() -> tuple[QwenSemanticInterpreter, _QwenRuntime]:
    runtime = _QwenRuntime()
    config = QwenSemanticConfig(
        model="Qwen/Qwen-x", device="cpu", precision="float32", max_new_tokens=32, temperature=0
    )
    return QwenSemanticInterpreter(config=config, runtime=runtime), runtime


def _gemini() -> tuple[GeminiSemanticInterpreter, _GeminiClient]:
    client = _GeminiClient()
    config = GeminiSemanticConfig(model="gemini-x", timeout_s=5, max_retries=0, temperature=0)
    return GeminiSemanticInterpreter(config=config, client=client), client


def _scene_context() -> SceneContext:
    return SceneContext(
        source_observation_id=FRAME,
        perception_result_id=RESULT,
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id="qwen_semantic",
                capability="semantic_interpreter",
                provider="qwen",
                model="Qwen/Qwen-x",
                version="1",
            ),
            task_identity="qwen-scene-interpretation",
            prompt_template_id="scene/v1",
            output_schema_version="semantic-response/1",
        ),
        scene_type="warehouse",
        environment="indoor",
    )


def _request(interpreter: Any, *, conditioned: bool) -> SemanticInterpretationRequest:
    region = RegionId("run-0001--frame-0001--region-0000")
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId(f"region-0000-{'context' if conditioned else 'plain'}"),
        source_observation_id=FRAME,
        perception_result_id=RESULT,
        mode=SemanticInterpretationMode.REGION,
        region_id=region,
        visual_views=(
            SemanticVisualView(
                view_id="v-tight",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference="outputs/semantic-views/tight.png",
                source_observation_id=FRAME,
                region_id=region,
                sha256="0" * 64,
            ),
        ),
        prompt_template_id="region-scene-context/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=interpreter.configuration_fingerprint,
        scene_context_reference=(
            SemanticEvidenceReference(evidence_type="scene_context", evidence_id=str(RESULT))
            if conditioned
            else None
        ),
        scene_context=_scene_context() if conditioned else None,
    )


@pytest.mark.parametrize("build", [_qwen, _gemini], ids=["qwen", "gemini"])
def test_the_with_and_without_scene_context_hook_runs_on_real_adapter_executions(
    build: Any,
) -> None:
    interpreter, fake = build()
    assert interpreter.capabilities().accepts_scene_context

    plain = interpreter.interpret(_request(interpreter, conditioned=False))
    conditioned = interpreter.interpret(_request(interpreter, conditioned=True))

    # O mesmo template nos dois braços: só o insumo de contexto de cena varia.
    assert plain.rendered_prompt.template_id == conditioned.rendered_prompt.template_id
    assert '"scene_type":"warehouse"' in fake.prompts[1]
    assert '"scene_type":"warehouse"' not in fake.prompts[0]
    assert conditioned.parsed.claims[0].evidence_references[-1] == SemanticEvidenceReference(
        evidence_type="scene_context", evidence_id=str(RESULT)
    )
    report = evaluate_semantic_interpretation(
        context=SemanticEvaluationContext(
            evaluation_id="scene-context-ablation",
            reference_set_version="fixture/1",
            selection_id="selection-0001",
            perception_run_id="run-0001",
            artifact_id="artifact-0001",
            pipeline_configuration_digest="sha256:pipeline",
            evaluator_version="semantic-evaluator/2",
        ),
        inputs=(
            SemanticEvaluationInput(execution=plain, evidence_variant_id="without_scene_context"),
            SemanticEvaluationInput(
                execution=conditioned, evidence_variant_id="with_scene_context"
            ),
        ),
        failures=(),
    )

    comparison = compare_evidence_variants(report, baseline_variant_id="without_scene_context")

    baseline, with_context = comparison.entries
    assert comparison.paired_region_count == 1
    assert baseline.evidence_channels == ("view:tight_crop",)
    assert with_context.evidence_channels == ("scene_context", "view:tight_crop")
    assert baseline.prompt_template_ids == with_context.prompt_template_ids
    assert with_context.primary_agreement_count == 1


@pytest.mark.parametrize("build", [_qwen, _gemini], ids=["qwen", "gemini"])
def test_a_scene_context_is_never_dropped_by_a_template_that_cannot_render_it(
    build: Any,
) -> None:
    interpreter, fake = build()
    request = replace(_request(interpreter, conditioned=True), prompt_template_id="region/v1")

    with pytest.raises(ValueError, match="does not render scene context"):
        interpreter.interpret(request)

    assert fake.prompts == []
    assert SEMANTIC_PROMPT_TEMPLATES["region/v1"].scene_context_instructions is None

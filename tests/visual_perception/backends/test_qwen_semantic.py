"""Contract tests for the Qwen semantic interpreter adapter."""

import json

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    PerceptionResultId,
    PipelinePreset,
    RegionId,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreter,
    SemanticRequestId,
    SemanticVisualView,
    StageSpec,
    VisualViewKind,
    execute_stage_graph,
    resolve_pipeline,
)
from contextmap.visual_perception.backends.qwen import (
    QwenGenerationResponse,
    QwenSemanticConfig,
    QwenSemanticInterpreter,
)


class _FakeQwenRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str, QwenSemanticConfig]] = []

    def generate(
        self,
        *,
        visual_payload_references: tuple[str, ...],
        prompt: str,
        config: QwenSemanticConfig,
    ) -> QwenGenerationResponse:
        self.calls.append((visual_payload_references, prompt, config))
        return QwenGenerationResponse(
            text=json.dumps(
                {
                    "abstained": False,
                    "claims": [
                        {
                            "hypothesis": "wooden pallet",
                            "role": "primary",
                            "category": None,
                            "region_kind": "thing",
                            "attributes": {},
                            "confidence": None,
                        }
                    ],
                    "scene_context": None,
                }
            ),
            input_tokens=120,
            output_tokens=24,
            peak_memory_bytes=1024,
            warnings=("deterministic fake",),
        )


def _request(adapter: QwenSemanticInterpreter) -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("request-0001"),
        source_observation_id=SourceObservationId("frame-0124"),
        perception_result_id=PerceptionResultId("run-0001--frame-0124"),
        mode=SemanticInterpretationMode.REGION,
        region_id=RegionId("region-0007"),
        visual_views=(
            SemanticVisualView(
                view_id="tight-crop",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference="outputs/views/region-0007.jpg",
                source_observation_id=SourceObservationId("frame-0124"),
                region_id=RegionId("region-0007"),
            ),
        ),
        prompt_template_id="region/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )


def test_qwen_maps_request_and_returns_canonical_unscored_claim() -> None:
    runtime = _FakeQwenRuntime()
    config = QwenSemanticConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        device="cuda:0",
        precision="bfloat16",
        quantization="4bit",
        max_new_tokens=128,
        temperature=0.0,
    )
    adapter = QwenSemanticInterpreter(config=config, runtime=runtime)

    execution = adapter.interpret(_request(adapter))

    assert isinstance(adapter, SemanticInterpreter)
    assert execution.parsed.claims[0].hypothesis == "wooden pallet"
    assert execution.parsed.claims[0].confidence is None
    assert execution.diagnostics.input_tokens == 120
    assert execution.effective_configuration["quantization"] == "4bit"
    assert runtime.calls[0][0] == ("outputs/views/region-0007.jpg",)
    assert "region/v1" in runtime.calls[0][1]
    assert (
        adapter.backend_provenance().configuration_fingerprint == adapter.configuration_fingerprint
    )


def test_qwen_configuration_rejects_ambiguous_generation_settings() -> None:
    with pytest.raises(ValueError, match="temperature"):
        QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cuda",
            precision="bfloat16",
            max_new_tokens=128,
            temperature=-0.1,
        )

    with pytest.raises(ValueError, match="quantization"):
        QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cuda",
            precision="bfloat16",
            quantization="unknown",
            max_new_tokens=128,
            temperature=0.0,
        )


def test_qwen_rejects_request_for_another_effective_configuration() -> None:
    adapter = QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        ),
        runtime=_FakeQwenRuntime(),
    )
    request = _request(adapter)
    mismatched = SemanticInterpretationRequest(
        **{**request.__dict__, "configuration_fingerprint": "sha256:other"}
    )

    with pytest.raises(ValueError, match="configuration fingerprint"):
        adapter.interpret(mismatched)


def test_qwen_is_selected_through_pipeline_configuration() -> None:
    adapter = QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        ),
        runtime=_FakeQwenRuntime(),
    )
    preset = PipelinePreset(
        preset_id="qwen-semantic/1",
        stages=(
            StageSpec(stage_id="request", capability="semantic_request"),
            StageSpec(
                stage_id="interpret",
                capability="semantic_interpreter",
                inputs={"request": "request"},
                backend_id="qwen",
            ),
        ),
    )

    resolved = resolve_pipeline(preset, backend_factories={"interpret": lambda parameters: adapter})
    outcomes = execute_stage_graph(resolved.build_stage_graph({"request": _request(adapter)}))

    execution = outcomes[-1].output
    assert isinstance(execution, SemanticInterpretationExecution)
    assert execution.parsed.claims[0].hypothesis == "wooden pallet"

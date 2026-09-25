"""Contract tests for the Qwen semantic interpreter adapter."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    SEMANTIC_PROMPT_TEMPLATES,
    BackendProvenance,
    BoundingBox2D,
    PerceptionResultId,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    PipelinePreset,
    Region2D,
    RegionId,
    SemanticConfidencePolicy,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreter,
    SemanticRequestId,
    SemanticVisualView,
    StageSpec,
    VisualViewKind,
    assemble_perception_result,
    execute_stage_graph,
    render_semantic_prompt,
    resolve_pipeline,
)
from contextmap.visual_perception.backends.qwen import (
    QwenGenerationResponse,
    QwenSemanticConfig,
    QwenSemanticInterpreter,
)
from contextmap.visual_perception.semantic_backend import (
    SemanticVisualInputMeasurement,
    decode_semantic_execution,
    encode_semantic_execution,
)


class _FakeQwenRuntime:
    def __init__(
        self,
        confidence: float | None = None,
        visual_inputs: tuple[SemanticVisualInputMeasurement, ...] | None = None,
    ) -> None:
        self.calls: list[tuple[tuple[SemanticVisualView, ...], str, QwenSemanticConfig]] = []
        self.confidence = confidence
        self.visual_inputs = visual_inputs

    def generate(
        self,
        *,
        visual_views: tuple[SemanticVisualView, ...],
        prompt: str,
        config: QwenSemanticConfig,
    ) -> QwenGenerationResponse:
        self.calls.append((visual_views, prompt, config))
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
                            "confidence": self.confidence,
                        }
                    ],
                    "scene_context": None,
                }
            ),
            input_tokens=120,
            output_tokens=24,
            peak_memory_bytes=1024,
            warnings=("deterministic fake",),
            visual_inputs=self.visual_inputs,
        )


_VIEW_PAYLOAD = b"qwen semantic view pixels"


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
                payload_reference="outputs/semantic-views/region-0007.jpg",
                source_observation_id=SourceObservationId("frame-0124"),
                region_id=RegionId("region-0007"),
                sha256=hashlib.sha256(_VIEW_PAYLOAD).hexdigest(),
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
    request = _request(adapter)

    execution = adapter.interpret(request)

    assert isinstance(adapter, SemanticInterpreter)
    assert execution.parsed.claims[0].hypothesis == "wooden pallet"
    assert execution.parsed.claims[0].confidence is None
    assert execution.diagnostics.input_tokens == 120
    assert execution.effective_configuration["quantization"] == "4bit"
    # O runtime recebe a identidade completa da view (incluindo o sha256), não só o caminho.
    assert runtime.calls[0][0] == request.visual_views
    assert runtime.calls[0][0][0].sha256 == hashlib.sha256(_VIEW_PAYLOAD).hexdigest()
    assert "region/v1" in runtime.calls[0][1]
    assert (
        adapter.backend_provenance().configuration_fingerprint == adapter.configuration_fingerprint
    )


def _cpu_adapter(runtime: _FakeQwenRuntime) -> QwenSemanticInterpreter:
    return QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        ),
        runtime=runtime,
    )


def test_qwen_consumes_exactly_the_prompt_policy_the_request_selects() -> None:
    """#542: a non-canonical policy reaches the model; nothing falls back to ``region/v1``."""
    runtime = _FakeQwenRuntime()
    adapter = _cpu_adapter(runtime)
    request = replace(_request(adapter), prompt_template_id="region-abstention/v1")

    execution = adapter.interpret(request)

    expected = render_semantic_prompt(
        request,
        SEMANTIC_PROMPT_TEMPLATES["region-abstention/v1"],
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )
    assert runtime.calls[0][1] == expected.text
    assert execution.rendered_prompt == expected
    assert execution.parsed.claims[0].provenance.prompt_template_id == "region-abstention/v1"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prompt_template_id": "region/v9"}, "unknown semantic prompt template 'region/v9'"),
        ({"prompt_template_id": "scene/v1"}, "mode must match"),
        ({"requested_output_schema": "semantic-response/2"}, "schema must match"),
    ],
)
def test_qwen_refuses_a_prompt_policy_it_cannot_render_before_inference(
    changes: dict[str, str], message: str
) -> None:
    runtime = _FakeQwenRuntime()
    adapter = _cpu_adapter(runtime)

    with pytest.raises(ValueError, match=message):
        adapter.interpret(replace(_request(adapter), **changes))  # type: ignore[arg-type]

    assert runtime.calls == []


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


def test_qwen_rejects_model_reported_confidence() -> None:
    adapter = QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        ),
        runtime=_FakeQwenRuntime(confidence=0.93),
    )

    from contextmap.visual_perception import SemanticInterpretationFailedError

    # Still a rejection, but the drifting response is now preserved instead of discarded:
    # a model reporting its own confidence is exactly the drift you need the raw text for.
    with pytest.raises(SemanticInterpretationFailedError, match="confidence must be null"):
        adapter.interpret(_request(adapter))


def test_qwen_stage_materializes_and_persists_canonical_result(tmp_path: Path) -> None:
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
            StageSpec(stage_id="regions", capability="region_source"),
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
    region = Region2D(
        region_id=RegionId("region-0007"),
        bounding_box=BoundingBox2D(x=0, y=0, width=10, height=10),
        provenance=BackendProvenance(
            backend_id="fixture",
            capability="region_discovery",
            provider="test",
            model="fixture",
            version="1",
        ),
    )
    outcomes = execute_stage_graph(
        resolved.build_stage_graph({"request": _request(adapter), "regions": (region,)})
    )

    execution = outcomes[-1].output
    assert isinstance(execution, SemanticInterpretationExecution)
    assert execution.parsed.claims[0].hypothesis == "wooden pallet"
    result = assemble_perception_result(
        result_id=PerceptionResultId("run-0001--frame-0124"),
        source_observation_id=SourceObservationId("frame-0124"),
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="sequence-artifact-0001",
        created_at="2026-01-01T00:00:00+00:00",
        outcomes=outcomes,
        region_stage_id="regions",
        semantic_execution_stage_ids=("interpret",),
    )
    writer = PerceptionRunWriter(
        output_dir=tmp_path / "visual_perception",
        sequence_name="sequence",
        run_id=PerceptionRunId("run-0001"),
        run_index=1,
        sequence_artifact_id="sequence-artifact-0001",
        selection_id="sha256:selection",
        enabled_capabilities=frozenset({"semantic_interpreter"}),
        pipeline_preset=preset,
        configuration_digest=resolved.configuration_digest(),
    )
    writer.add_result(result)
    writer.add_semantic_view_payload(execution.request.visual_views[0], _VIEW_PAYLOAD)
    writer.add_stage_outcomes(outcomes)
    writer.finalize()

    reader = PerceptionRunReader(tmp_path / "visual_perception")
    assert reader.list_results()[0].claims[0].hypothesis == "wooden pallet"
    assert reader.list_semantic_executions()[0].request == _request(adapter)
    assert reader.verify_integrity() == []


def _cpu_config(**changes: Any) -> QwenSemanticConfig:
    return replace(
        QwenSemanticConfig(
            model="Qwen/Qwen3-VL-4B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        ),
        **changes,
    )


_BUDGET = {"min_pixels": 256 * 32 * 32, "max_pixels": 1280 * 32 * 32}
"""A visual input budget of 256 to 1280 merged 32 px patches per image."""


def test_an_unset_visual_budget_keeps_the_configuration_identity_it_had_before_526() -> None:
    """Existing configurations keep their fingerprint; no budget key is invented for them."""
    before_526 = {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "device": "cpu",
        "precision": "float32",
        "max_new_tokens": 32,
        "temperature": 0.0,
        "quantization": None,
        "revision": None,
    }
    encoded = json.dumps(before_526, sort_keys=True, separators=(",", ":")).encode("utf-8")

    adapter = QwenSemanticInterpreter(config=_cpu_config(), runtime=_FakeQwenRuntime())

    assert _cpu_config().to_dict() == before_526
    assert adapter.configuration_fingerprint == "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_the_visual_input_budget_is_part_of_the_configuration_identity() -> None:
    """#526: the budget is a scientific input, so it is fingerprinted and recorded."""
    unset = _cpu_config()
    budget = _cpu_config(**_BUDGET)
    tighter = _cpu_config(**{**_BUDGET, "max_pixels": 640 * 32 * 32})

    fingerprints = {
        QwenSemanticInterpreter(config=config, runtime=_FakeQwenRuntime()).configuration_fingerprint
        for config in (unset, budget, tighter)
    }
    adapter = QwenSemanticInterpreter(config=budget, runtime=_FakeQwenRuntime())
    execution = adapter.interpret(_request(adapter))

    assert len(fingerprints) == 3
    assert execution.effective_configuration["min_pixels"] == 262_144
    assert execution.effective_configuration["max_pixels"] == 1_310_720


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        ({"max_pixels": 1_310_720}, "set together"),
        ({"min_pixels": 262_144}, "set together"),
        ({"min_pixels": 0, "max_pixels": 1_310_720}, "min_pixels must be positive"),
        ({"min_pixels": 262_144, "max_pixels": -1}, "max_pixels must be positive"),
        ({"min_pixels": 1_310_720, "max_pixels": 262_144}, "must not exceed max_pixels"),
    ],
)
def test_an_incomplete_or_impossible_visual_budget_is_rejected(
    budget: dict[str, int], message: str
) -> None:
    """A partial budget would leave the other bound at a checkpoint default nobody recorded."""
    with pytest.raises(ValueError, match=message):
        _cpu_config(**budget)


def test_measured_visual_inputs_are_execution_diagnostics_not_identity() -> None:
    measured = (
        SemanticVisualInputMeasurement(
            view_id="tight-crop", height_px=448, width_px=320, visual_tokens=140
        ),
    )
    adapter = QwenSemanticInterpreter(
        config=_cpu_config(**_BUDGET), runtime=_FakeQwenRuntime(visual_inputs=measured)
    )
    fingerprint = adapter.configuration_fingerprint

    execution = adapter.interpret(_request(adapter))
    decoded = decode_semantic_execution(
        encode_semantic_execution(execution, raw_response_reference="raw.txt")
    )

    assert execution.diagnostics.visual_inputs == measured
    assert decoded.diagnostics.visual_inputs == measured
    assert "visual_inputs" not in execution.effective_configuration
    assert adapter.configuration_fingerprint == fingerprint


class _NonScalarAttributeQwenRuntime:
    """Reproduces a real failure family that survives the confidence fix.

    The model returns a list where `attributes` values must be scalars: 163 of the 457
    rejected responses in the real 360-frame corridor-02 run failed exactly this way. (The
    once-dominant "missing confidence" family is gone: omitting the key is now legal under
    UNSCORED_ONLY, since null was its only permitted value.)
    """

    def generate(
        self,
        *,
        visual_views: tuple[SemanticVisualView, ...],
        prompt: str,
        config: QwenSemanticConfig,
    ) -> QwenGenerationResponse:
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
                            "attributes": {"features": ["slatted", "wooden"]},
                            "confidence": None,
                        }
                    ],
                    "scene_context": None,
                }
            ),
            input_tokens=120,
            output_tokens=24,
            peak_memory_bytes=1024,
            warnings=(),
            visual_inputs=(
                SemanticVisualInputMeasurement(
                    view_id="tight-crop", height_px=448, width_px=320, visual_tokens=140
                ),
            ),
        )


def test_a_rejected_qwen_response_is_preserved_as_evidence_not_reduced_to_a_string() -> None:
    """PR #438 review: the real raw_response must survive a parse failure."""
    from contextmap.visual_perception import SemanticInterpretationFailedError

    config = QwenSemanticConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        device="cuda:0",
        precision="bfloat16",
        quantization="4bit",
        max_new_tokens=128,
        temperature=0.0,
    )
    adapter = QwenSemanticInterpreter(config=config, runtime=_NonScalarAttributeQwenRuntime())
    request = _request(adapter)

    with pytest.raises(SemanticInterpretationFailedError) as raised:
        adapter.interpret(request)

    failed = raised.value.failure
    assert "wooden pallet" in failed.raw_response, "the observed response must not be lost"
    assert failed.raw_response_sha256
    assert failed.failure.kind == "SemanticResponseParseError"
    assert "scalar value" in failed.failure.message
    assert failed.request.request_id == request.request_id, "shares the attempt identity"
    assert failed.provenance.backend.model == config.model
    assert failed.diagnostics.input_tokens == 120
    # #526: a rejected response still records what the model actually consumed.
    assert failed.diagnostics.visual_inputs is not None
    assert failed.diagnostics.visual_inputs[0].visual_tokens == 140
    assert failed.rendered_prompt.text, "the exact prompt sent must be preserved"

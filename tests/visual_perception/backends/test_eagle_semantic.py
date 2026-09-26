"""Contract tests for the Eagle 2.5 semantic interpreter adapter, with a fake runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    SEMANTIC_PROMPT_TEMPLATES,
    BackendProvenance,
    FeatureId,
    FeatureScope,
    PerceptionResultId,
    RegionId,
    SceneContext,
    SemanticConfidencePolicy,
    SemanticEvidenceReference,
    SemanticFeatureReference,
    SemanticInferenceProvenance,
    SemanticInterpretationFailedError,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreter,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
    decode_failed_semantic_interpretation,
    encode_failed_semantic_interpretation,
    render_semantic_prompt,
)
from contextmap.visual_perception.backends.eagle2_5 import (
    EagleGenerationResponse,
    EagleSemanticConfig,
    EagleSemanticInterpreter,
)
from contextmap.visual_perception.semantic_backend import (
    decode_semantic_execution,
    encode_semantic_execution,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"

_REGION_RESPONSE = {
    "abstained": False,
    "claims": [
        {
            "hypothesis": "wooden pallet",
            "role": "primary",
            "category": None,
            "region_kind": "thing",
            "attributes": {},
            "confidence": None,
        },
        {
            "hypothesis": "crate",
            "role": "alternative",
            "category": None,
            "region_kind": "thing",
            "attributes": {},
            "confidence": None,
        },
    ],
    "scene_context": None,
}


class _FakeEagleRuntime:
    """Deterministic stand-in for the transformers runtime; records every call."""

    def __init__(self, text: str | None = None) -> None:
        self.calls: list[tuple[tuple[SemanticVisualView, ...], str, EagleSemanticConfig]] = []
        self.text = json.dumps(_REGION_RESPONSE) if text is None else text

    def generate(
        self,
        *,
        visual_views: tuple[SemanticVisualView, ...],
        prompt: str,
        config: EagleSemanticConfig,
    ) -> EagleGenerationResponse:
        self.calls.append((visual_views, prompt, config))
        return EagleGenerationResponse(
            text=self.text,
            input_tokens=1_300,
            output_tokens=48,
            peak_memory_bytes=17_000_000_000,
            warnings=("deterministic fake",),
        )


def _config(**overrides: Any) -> EagleSemanticConfig:
    values: dict[str, Any] = {
        "model": "nvidia/Eagle2.5-8B",
        "revision": REVISION,
        "device": "cuda",
        "precision": "bfloat16",
        "max_new_tokens": 256,
        "temperature": 0.0,
        "max_dynamic_tiles": 4,
    }
    values.update(overrides)
    return EagleSemanticConfig(**values)


def _view(view_id: str, kind: VisualViewKind, payload: bytes) -> SemanticVisualView:
    return SemanticVisualView(
        view_id=view_id,
        kind=kind,
        payload_reference=f"outputs/semantic-views/{view_id}.png",
        source_observation_id=SourceObservationId("frame-0124"),
        region_id=None if kind is VisualViewKind.FULL_FRAME else RegionId("region-0007"),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _region_request(
    adapter: EagleSemanticInterpreter,
    *,
    views: tuple[SemanticVisualView, ...] | None = None,
) -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("request-0001"),
        source_observation_id=SourceObservationId("frame-0124"),
        perception_result_id=PerceptionResultId("run-0001--frame-0124"),
        mode=SemanticInterpretationMode.REGION,
        region_id=RegionId("region-0007"),
        visual_views=views or (_view("tight-crop", VisualViewKind.TIGHT_CROP, b"crop"),),
        prompt_template_id="region/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )


def _scene_request(adapter: EagleSemanticInterpreter) -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("request-0002"),
        source_observation_id=SourceObservationId("frame-0124"),
        perception_result_id=PerceptionResultId("run-0001--frame-0124"),
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(_view("full-frame", VisualViewKind.FULL_FRAME, b"frame"),),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )


class TestCapabilities:
    def test_declares_scene_and_region_over_every_canonical_view_kind(self) -> None:
        adapter = EagleSemanticInterpreter(config=_config(), runtime=_FakeEagleRuntime())

        capabilities = adapter.capabilities()

        assert isinstance(adapter, SemanticInterpreter)
        assert capabilities.supported_modes == frozenset(
            {SemanticInterpretationMode.SCENE, SemanticInterpretationMode.REGION}
        )
        assert capabilities.supported_view_kinds == frozenset(VisualViewKind)
        assert capabilities.required_view_kinds == frozenset()
        assert capabilities.accepts_visual_features is False
        assert capabilities.accepts_scene_context is False

    def test_provenance_names_the_backend_the_model_and_the_effective_configuration(
        self,
    ) -> None:
        adapter = EagleSemanticInterpreter(config=_config(), runtime=_FakeEagleRuntime())

        provenance = adapter.backend_provenance()

        assert provenance.backend_id == "eagle2_5_semantic"
        assert provenance.capability == "semantic_interpreter"
        assert provenance.provider == "nvidia"
        assert provenance.model == "nvidia/Eagle2.5-8B"
        assert provenance.configuration_fingerprint == adapter.configuration_fingerprint


class TestInterpretation:
    def test_region_request_becomes_unscored_canonical_claims_with_alternatives(self) -> None:
        runtime = _FakeEagleRuntime()
        adapter = EagleSemanticInterpreter(config=_config(), runtime=runtime)
        request = _region_request(adapter)

        execution = adapter.interpret(request)

        primary, alternative = execution.parsed.claims
        assert (primary.hypothesis, primary.role.value) == ("wooden pallet", "primary")
        assert (alternative.hypothesis, alternative.role.value) == ("crate", "alternative")
        assert all(claim.confidence is None for claim in execution.parsed.claims)
        assert primary.provenance.backend == adapter.backend_provenance()
        assert primary.provenance.task_identity == "eagle2_5-region-interpretation"
        assert primary.region_id == RegionId("region-0007")
        assert execution.raw_response == runtime.text
        assert execution.request == request

    def test_scene_request_becomes_scene_context_evidence(self) -> None:
        response = {
            "abstained": False,
            "claims": [],
            "scene_context": {"scene_type": "warehouse aisle", "lighting": "artificial"},
        }
        adapter = EagleSemanticInterpreter(
            config=_config(), runtime=_FakeEagleRuntime(json.dumps(response))
        )

        execution = adapter.interpret(_scene_request(adapter))

        assert execution.parsed.claims == ()
        assert execution.parsed.scene_context is not None
        assert execution.parsed.scene_context.scene_type == "warehouse aisle"
        assert execution.parsed.scene_context.lighting == "artificial"

    def test_an_explicit_abstention_is_preserved_as_abstention(self) -> None:
        response = {"abstained": True, "claims": [], "scene_context": None}
        adapter = EagleSemanticInterpreter(
            config=_config(), runtime=_FakeEagleRuntime(json.dumps(response))
        )

        execution = adapter.interpret(_region_request(adapter))

        assert execution.parsed.abstained is True
        assert execution.parsed.claims == ()

    def test_the_exact_view_order_of_the_request_reaches_the_runtime(self) -> None:
        runtime = _FakeEagleRuntime()
        adapter = EagleSemanticInterpreter(config=_config(), runtime=runtime)
        views = (
            _view("contextual", VisualViewKind.CONTEXTUAL_CROP, b"context"),
            _view("masked", VisualViewKind.MASKED_SUBJECT, b"masked"),
            _view("full-frame", VisualViewKind.FULL_FRAME, b"frame"),
            _view("tight", VisualViewKind.TIGHT_CROP, b"tight"),
        )

        adapter.interpret(_region_request(adapter, views=views))

        (received, _prompt, _config_received), *_ = runtime.calls
        assert received == views

    def test_diagnostics_and_effective_configuration_are_recorded(self) -> None:
        config = _config(max_dynamic_tiles=6, min_dynamic_tiles=2, use_thumbnail=False)
        adapter = EagleSemanticInterpreter(config=config, runtime=_FakeEagleRuntime())

        execution = adapter.interpret(_region_request(adapter))

        assert execution.diagnostics.input_tokens == 1_300
        assert execution.diagnostics.output_tokens == 48
        assert execution.diagnostics.peak_memory_bytes == 17_000_000_000
        assert execution.diagnostics.warnings == ("deterministic fake",)
        assert execution.diagnostics.latency_ms >= 0
        assert dict(execution.effective_configuration) == config.to_dict()
        assert execution.effective_configuration["max_dynamic_tiles"] == 6
        assert execution.effective_configuration["min_dynamic_tiles"] == 2
        assert execution.effective_configuration["use_thumbnail"] is False
        assert execution.effective_configuration["revision"] == REVISION


class TestPromptPolicy:
    def test_consumes_exactly_the_prompt_policy_the_request_selects(self) -> None:
        """#542: a non-canonical policy reaches the model; nothing falls back to ``region/v1``."""
        runtime = _FakeEagleRuntime()
        adapter = EagleSemanticInterpreter(config=_config(), runtime=runtime)
        request = replace(_region_request(adapter), prompt_template_id="region-abstention/v1")

        execution = adapter.interpret(request)

        expected = render_semantic_prompt(
            request,
            SEMANTIC_PROMPT_TEMPLATES["region-abstention/v1"],
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        )
        assert runtime.calls[0][1] == expected.text
        assert execution.rendered_prompt == expected
        assert execution.parsed.claims[0].provenance.prompt_template_id == "region-abstention/v1"

    def test_the_rendered_prompt_is_identical_to_every_instruction_following_backend(
        self,
    ) -> None:
        """Matched arms: the same request renders the same bytes as Qwen and Gemini render."""
        runtime = _FakeEagleRuntime()
        adapter = EagleSemanticInterpreter(config=_config(), runtime=runtime)
        request = _region_request(adapter)

        execution = adapter.interpret(request)

        assert execution.rendered_prompt == render_semantic_prompt(
            request,
            SEMANTIC_PROMPT_TEMPLATES["region/v1"],
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        )


def _prior_scene_context() -> SceneContext:
    """The scene context a conditioned region request would carry (#529)."""
    return SceneContext(
        source_observation_id=SourceObservationId("frame-0124"),
        perception_result_id=PerceptionResultId("run-0001--frame-0124"),
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
    )


class TestFailsBeforeInference:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"prompt_template_id": "region/v9"}, "unknown semantic prompt template 'region/v9'"),
            ({"prompt_template_id": "scene/v1"}, "mode must match"),
            ({"requested_output_schema": "semantic-response/2"}, "schema must match"),
            ({"configuration_fingerprint": "sha256:other"}, "configuration fingerprint"),
            (
                {
                    "visual_features": (
                        SemanticFeatureReference(
                            feature_id=FeatureId("feature-0001"),
                            embedding_space_id="dinov3:space",
                            scope=FeatureScope.REGION,
                            region_id=RegionId("region-0007"),
                        ),
                    )
                },
                "does not accept visual features",
            ),
            (
                {
                    "scene_context_reference": SemanticEvidenceReference(
                        evidence_type="scene_context", evidence_id="run-0001--frame-0124"
                    ),
                    "scene_context": _prior_scene_context(),
                },
                "does not accept scene context",
            ),
        ],
    )
    def test_an_unsupported_request_is_refused_without_calling_the_runtime(
        self, changes: dict[str, Any], message: str
    ) -> None:
        runtime = _FakeEagleRuntime()
        adapter = EagleSemanticInterpreter(config=_config(), runtime=runtime)

        with pytest.raises(ValueError, match=message):
            adapter.interpret(replace(_region_request(adapter), **changes))

        assert runtime.calls == []


class TestObservedResponseIsEvidence:
    def test_self_reported_confidence_is_never_taken_as_calibrated(self) -> None:
        response = json.loads(json.dumps(_REGION_RESPONSE))
        response["claims"][0]["confidence"] = 0.93
        adapter = EagleSemanticInterpreter(
            config=_config(), runtime=_FakeEagleRuntime(json.dumps(response))
        )

        with pytest.raises(SemanticInterpretationFailedError, match="confidence must be null"):
            adapter.interpret(_region_request(adapter))

    def test_a_rejected_response_is_preserved_verbatim_with_its_audit_context(self) -> None:
        raw = '{"abstained": false, "claims": [{"hypothesis": "wooden pallet"'  # truncado
        config = _config()
        adapter = EagleSemanticInterpreter(config=config, runtime=_FakeEagleRuntime(raw))
        request = _region_request(adapter)

        with pytest.raises(SemanticInterpretationFailedError) as raised:
            adapter.interpret(request)

        failed = raised.value.failure
        assert failed.raw_response == raw, "the observed response must not be lost"
        assert failed.raw_response_sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert failed.failure.kind == "SemanticResponseParseError"
        assert failed.request == request, "shares the attempt identity"
        assert failed.rendered_prompt.template_id == "region/v1"
        assert failed.provenance.backend.model == config.model
        assert failed.provenance.raw_response_reference == (
            "debug/40-semantic-interpretation/request-0001/raw-response.txt"
        )
        assert failed.diagnostics.input_tokens == 1_300
        assert dict(failed.effective_configuration) == config.to_dict()

    def test_executions_and_failures_survive_their_persisted_records_intact(self) -> None:
        adapter = EagleSemanticInterpreter(config=_config(), runtime=_FakeEagleRuntime())
        execution = adapter.interpret(_region_request(adapter))
        failing = EagleSemanticInterpreter(config=_config(), runtime=_FakeEagleRuntime("{"))
        with pytest.raises(SemanticInterpretationFailedError) as raised:
            failing.interpret(_region_request(failing))

        execution_record = encode_semantic_execution(
            execution, raw_response_reference="debug/raw-response.txt"
        )
        failure_record = encode_failed_semantic_interpretation(raised.value.failure)

        # O registro persistido passa por JSON: orçamento visual e revisão não são redigidos.
        assert decode_semantic_execution(json.loads(json.dumps(execution_record))) == execution
        assert (
            decode_failed_semantic_interpretation(json.loads(json.dumps(failure_record)))
            == raised.value.failure
        )


class TestConfigurationIdentity:
    def test_the_fingerprint_is_the_digest_of_the_sorted_effective_configuration(self) -> None:
        config = _config()
        encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))

        first = EagleSemanticInterpreter(config=config, runtime=_FakeEagleRuntime())
        second = EagleSemanticInterpreter(config=_config(), runtime=_FakeEagleRuntime())

        expected = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        assert first.configuration_fingerprint == expected
        assert second.configuration_fingerprint == expected

    def test_the_effective_configuration_names_every_field_that_changes_the_output(self) -> None:
        assert set(_config().to_dict()) == {
            "model",
            "revision",
            "device",
            "precision",
            "max_new_tokens",
            "temperature",
            "max_dynamic_tiles",
            "min_dynamic_tiles",
            "use_thumbnail",
        }

    @pytest.mark.parametrize(
        "change",
        [
            {"revision": "f" * 40},
            {"model": "nvidia/Eagle2-2B"},
            {"max_dynamic_tiles": 12},
            {"min_dynamic_tiles": 2},
            {"use_thumbnail": False},
            {"max_new_tokens": 128},
            {"temperature": 0.2},
            {"precision": "float16"},
        ],
    )
    def test_every_model_budget_or_generation_change_is_a_new_identity(
        self, change: dict[str, Any]
    ) -> None:
        baseline = EagleSemanticInterpreter(config=_config(), runtime=_FakeEagleRuntime())
        changed = EagleSemanticInterpreter(config=_config(**change), runtime=_FakeEagleRuntime())

        assert baseline.configuration_fingerprint != changed.configuration_fingerprint

    def test_the_visual_budget_bounds_the_tiles_of_each_view(self) -> None:
        # O thumbnail só é acrescentado quando a imagem vira mais de uma tile.
        assert _config(max_dynamic_tiles=4).max_tiles_per_view == 5
        assert _config(max_dynamic_tiles=4, use_thumbnail=False).max_tiles_per_view == 4
        assert _config(max_dynamic_tiles=1).max_tiles_per_view == 1

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"model": " "}, "model"),
            ({"device": ""}, "device"),
            ({"precision": ""}, "precision"),
            ({"max_new_tokens": 0}, "max_new_tokens"),
            ({"temperature": -0.1}, "temperature"),
            ({"max_dynamic_tiles": 0}, "max_dynamic_tiles"),
            ({"min_dynamic_tiles": 0}, "min_dynamic_tiles"),
            ({"min_dynamic_tiles": 5, "max_dynamic_tiles": 4}, "min_dynamic_tiles"),
            ({"revision": "main"}, "revision"),
        ],
    )
    def test_an_ambiguous_or_impossible_configuration_is_rejected(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _config(**change)

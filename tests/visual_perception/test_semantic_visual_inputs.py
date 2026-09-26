"""Contract tests for the measured visual inputs of a semantic backend call (#526).

What a model actually consumed for each view (the size after the backend's own preprocessing
and the visual tokens it became) is an execution diagnostic: it is recorded next to the
request that names the views, never folded into the configuration identity.
"""

from __future__ import annotations

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
    FailedSemanticInterpretation,
    PerceptionResultId,
    RegionId,
    SemanticBackendDiagnostics,
    SemanticConfidencePolicy,
    SemanticDebugLevel,
    SemanticInferenceProvenance,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticParseFailure,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
    decode_failed_semantic_interpretation,
    encode_failed_semantic_interpretation,
    render_semantic_prompt,
    write_semantic_audit,
)
from contextmap.visual_perception.semantic_backend import SemanticVisualInputMeasurement

SOURCE_ID = SourceObservationId("frame-0124")


def _view(view_id: str, kind: VisualViewKind) -> SemanticVisualView:
    return SemanticVisualView(
        view_id=view_id,
        kind=kind,
        payload_reference=f"outputs/semantic-views/{view_id}.png",
        source_observation_id=SOURCE_ID,
        region_id=RegionId("region-0007"),
        sha256=hashlib.sha256(view_id.encode("utf-8")).hexdigest(),
    )


def _request() -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("region-request-0007"),
        source_observation_id=SOURCE_ID,
        perception_result_id=PerceptionResultId("run-0001--frame-0124"),
        mode=SemanticInterpretationMode.REGION,
        region_id=RegionId("region-0007"),
        visual_views=(
            _view("masked", VisualViewKind.MASKED_SUBJECT),
            _view("tight", VisualViewKind.TIGHT_CROP),
        ),
        prompt_template_id="region/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint="sha256:config",
    )


def _measured(view_id: str, *, height: int = 448, width: int = 320) -> Any:
    return SemanticVisualInputMeasurement(
        view_id=view_id,
        height_px=height,
        width_px=width,
        visual_tokens=height * width // (32 * 32),
    )


def _failed(
    visual_inputs: tuple[SemanticVisualInputMeasurement, ...] | None,
) -> FailedSemanticInterpretation:
    request = _request()
    raw = json.dumps({"abstained": False, "claims": [{"hypothesis": "a pallet"}]})
    return FailedSemanticInterpretation(
        request=request,
        rendered_prompt=render_semantic_prompt(
            request,
            SEMANTIC_PROMPT_TEMPLATES[request.prompt_template_id],
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        ),
        raw_response=raw,
        raw_response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id="qwen_semantic",
                capability="semantic_interpreter",
                provider="qwen",
                model="Qwen/Qwen3-VL-4B-Instruct",
                version="1",
                configuration_fingerprint="sha256:config",
            ),
            task_identity="qwen-region-interpretation",
            prompt_template_id="region/v1",
            output_schema_version="semantic-response/1",
            raw_response_reference="debug/40-semantic-interpretation/region-request-0007/raw.txt",
        ),
        diagnostics=SemanticBackendDiagnostics(
            latency_ms=1.0, input_tokens=300, visual_inputs=visual_inputs
        ),
        failure=SemanticParseFailure(kind="SemanticResponseParseError", message="bad claim"),
        effective_configuration={"min_pixels": 65_536, "max_pixels": 262_144},
        occurred_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"view_id": " "}, "view_id must not be empty"),
        ({"height_px": 0}, "height_px must be positive"),
        ({"width_px": -32}, "width_px must be positive"),
        ({"visual_tokens": -1}, "visual_tokens must be non-negative"),
    ],
)
def test_a_measurement_describes_a_real_model_input(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_measured("tight"), **changes)


def test_measurements_must_describe_exactly_the_request_views_in_order() -> None:
    """The number of images and what each became are recorded together, never guessed."""
    for mismatched in (
        (_measured("tight"),),
        (_measured("tight"), _measured("masked")),
        (_measured("masked"), _measured("tight"), _measured("context")),
    ):
        with pytest.raises(ValueError, match="visual input measurements"):
            _failed(mismatched)


def test_an_unmeasured_call_is_recorded_as_unmeasured() -> None:
    assert _failed(None).diagnostics.visual_inputs is None


def test_measured_visual_inputs_round_trip_through_the_persisted_record() -> None:
    failed = _failed((_measured("masked"), _measured("tight", height=224, width=224)))

    encoded = encode_failed_semantic_interpretation(failed)

    assert encoded["diagnostics"]["visual_inputs"] == [
        {"view_id": "masked", "height_px": 448, "width_px": 320, "visual_tokens": 140},
        {"view_id": "tight", "height_px": 224, "width_px": 224, "visual_tokens": 49},
    ]
    assert decode_failed_semantic_interpretation(json.loads(json.dumps(encoded))) == failed


def test_a_record_written_before_visual_inputs_were_measured_stays_readable() -> None:
    """Records frozen under run artifact schema 0.5.0 simply did not measure them."""
    encoded = encode_failed_semantic_interpretation(_failed(None))
    del encoded["diagnostics"]["visual_inputs"]

    assert decode_failed_semantic_interpretation(encoded).diagnostics.visual_inputs is None


def test_the_human_audit_shows_what_each_view_became(tmp_path: Path) -> None:
    failed = _failed((_measured("masked"), _measured("tight")))

    write_semantic_audit(run_root=tmp_path, execution=failed, debug_level=SemanticDebugLevel.FULL)

    recorded = json.loads(
        (
            tmp_path / "debug/40-semantic-interpretation/region-request-0007/diagnostics.json"
        ).read_text(encoding="utf-8")
    )
    assert [item["view_id"] for item in recorded["visual_inputs"]] == ["masked", "tight"]
    assert recorded["effective_configuration"]["max_pixels"] == 262_144

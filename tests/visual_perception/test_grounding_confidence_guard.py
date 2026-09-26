"""Guards of #573: native grounding/refinement diagnostics stay raw, never confidence.

No grounding or refinement contract has a slot for a confidence or a calibrated value, so
an unvalidated decoder probability cannot be labelled ``confidence``; a calibrated score
would need a separate, versioned contract carrying its calibrator identity, which does not
exist until a calibration is accepted. Raw values survive persistence unchanged, and a
missing value stays missing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    ArtifactReference,
    BackendScore,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingQuery,
    GroundingTask,
    PerceptionResult,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    PreparedImage,
    RefinementOutcome,
    RefinementPrompt,
    Region2D,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionRefinementExecution,
    encode_region_grounding_execution,
    perception_result_id_for,
    with_grounded_regions,
)
from contextmap.visual_perception.backends.locateanything import (
    PHRASE_GROUNDING_POLICY,
    LocateAnythingConfig,
    LocateAnythingGeneration,
    LocateAnythingGenerationMode,
    LocateAnythingRegionGrounding,
)

FORBIDDEN = ("confidence", "probability", "calibrated")
RESULT_ID = perception_result_id_for(
    run_id=PerceptionRunId("run-0001"), source_observation_id=SourceObservationId("frame-0124")
)


@pytest.mark.parametrize(
    "contract",
    [
        GroundingOutput,
        GroundingDiagnostics,
        RegionGroundingExecution,
        RefinementPrompt,
        RefinementOutcome,
        RegionRefinementExecution,
        Region2D,
        BackendScore,
    ],
)
def test_no_grounding_or_refinement_contract_can_hold_a_confidence(contract: type) -> None:
    names = [item.name for item in dataclasses.fields(contract)]

    assert not [name for name in names if any(word in name for word in FORBIDDEN)]


def _backend(statistics: str | None) -> LocateAnythingRegionGrounding:
    class Runtime:
        def generate(self, **_: Any) -> LocateAnythingGeneration:
            return LocateAnythingGeneration(
                text="<ref>chair</ref><box><0><0><500><500></box><|im_end|>",
                statistics=statistics,
            )

    config = LocateAnythingConfig(
        model="nvidia/LocateAnything-3B",
        revision="0123456789abcdef0123456789abcdef01234567",
        device="cuda",
        dtype="bfloat16",
        generation_mode=LocateAnythingGenerationMode.HYBRID,
        max_new_tokens=64,
        temperature=0.0,
        text_attention="sdpa",
        vision_attention="sdpa",
    )
    return LocateAnythingRegionGrounding(config=config, runtime=Runtime())


def _execution(statistics: str | None) -> RegionGroundingExecution:
    backend = _backend(statistics)
    return backend.ground(
        RegionGroundingRequest(
            perception_result_id=RESULT_ID,
            image=PreparedImage(
                source_observation_id=SourceObservationId("frame-0124"),
                payload_reference="frame-0124.png",
                payload_artifact=ArtifactReference(
                    uri="frame-0124.png", sha256="f" * 64, media_type="image/png"
                ),
                width=640,
                height=480,
            ),
            query=GroundingQuery(
                task=GroundingTask.PHRASE_GROUNDING,
                policy_id=PHRASE_GROUNDING_POLICY,
                geometry=GroundingGeometry.BOX,
                text="the chair",
            ),
            configuration_fingerprint=str(backend.backend_provenance().configuration_fingerprint),
        )
    )


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def test_a_persisted_grounding_record_has_no_confidence_anywhere(tmp_path: Path) -> None:
    record = encode_region_grounding_execution(
        _execution("Statistic Info, switch_to_ar=3"), raw_response_reference="r.txt"
    )

    assert not [key for key in _keys(record) if any(word in key for word in FORBIDDEN)]
    names = [item["name"] for item in record["diagnostics"]["native"]]
    assert not [name for name in names if any(word in name for word in FORBIDDEN)]


def test_raw_decoder_diagnostics_survive_the_run_artifact_unchanged(tmp_path: Path) -> None:
    statistics = (
        "\nStatistic Info, num_tokens=18; generate_time(s)=0.30000000000000004; "
        "tps=59.99999999999999; prefill_time=nan; switch_to_ar=2\n"
    )
    execution = _execution(statistics)
    writer = PerceptionRunWriter(
        output_dir=tmp_path / "run",
        sequence_name="corridor-02",
        run_id=PerceptionRunId("run-0001"),
        run_index=1,
        sequence_artifact_id="sequence-0001",
        selection_id="sha256:aaaa",
        enabled_capabilities=frozenset({"region_grounding"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:test",
    )
    result = PerceptionResult(
        result_id=RESULT_ID,
        source_observation_id=SourceObservationId("frame-0124"),
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="sequence-0001",
        created_at="2026-01-01T00:00:00+00:00",
    )
    writer.add_result(with_grounded_regions(result, (execution,)))
    writer.add_region_grounding(execution)
    writer.finalize()

    (reopened,) = PerceptionRunReader(tmp_path / "run").list_region_groundings()

    native = dict(reopened.diagnostics.native)
    assert reopened.diagnostics.native == execution.diagnostics.native
    assert native["stats.raw"] == statistics
    assert native["stats.generate_time(s)"] == 0.30000000000000004
    assert native["stats.tps"] == 59.99999999999999
    assert native["stats.switch_to_ar"] == 2
    # Um valor não finito continua o texto nativo, nunca vira 0 nem 1.
    assert native["stats.prefill_time"] == "nan"


def test_a_missing_native_diagnostic_stays_missing(tmp_path: Path) -> None:
    without_stats = _execution(None)
    partial = _execution("Statistic Info, num_tokens=18")

    assert without_stats.diagnostics.native == ()
    assert "stats.switch_to_ar" not in dict(partial.diagnostics.native)
    assert without_stats.outputs[0].native_diagnostics == ()


def test_a_native_score_needs_its_own_named_semantics() -> None:
    with pytest.raises(ValueError, match="semantics"):
        BackendScore(name="predicted_iou", value=0.9, semantics="")
    with pytest.raises(ValueError, match="finite"):
        BackendScore(name="predicted_iou", value=float("nan"), semantics="SAM2-native IoU")

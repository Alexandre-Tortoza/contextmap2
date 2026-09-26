"""Refinement executions persisted with, and linked to, the perception run artifact (#568)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    ArtifactReference,
    BackendProvenance,
    BackendScore,
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingPoint,
    GroundingQuery,
    GroundingTask,
    InlineMask,
    PerceptionResult,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    PreparedImage,
    RefinementDiagnostics,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionRefinementExecution,
    RegionRefinementRequest,
    RunArtifactError,
    perception_result_id_for,
    refinement_outcome,
    refinement_prompts_from,
    with_grounded_regions,
    with_refined_regions,
)

RUN_ID = PerceptionRunId("run-0001")
SOURCE_ID = SourceObservationId("frame-0124")
RESULT_ID = perception_result_id_for(run_id=RUN_ID, source_observation_id=SOURCE_ID)
WIDTH, HEIGHT = 20, 10
REFINER = BackendProvenance(
    backend_id="sam2",
    capability="region_refinement",
    provider="facebook",
    model="sam2.1_hiera_large",
    version="2.1",
    configuration_fingerprint="sha256:refiner",
)


def _image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SOURCE_ID,
        payload_reference="frame-0124.png",
        payload_artifact=ArtifactReference(
            uri="frame-0124.png", sha256="b" * 64, media_type="image/png"
        ),
        width=WIDTH,
        height=HEIGHT,
    )


def _grounding(text: str = "the chair") -> RegionGroundingExecution:
    return RegionGroundingExecution(
        request=RegionGroundingRequest(
            perception_result_id=RESULT_ID,
            image=_image(),
            query=GroundingQuery(
                task=GroundingTask.PHRASE_GROUNDING,
                policy_id="fake.phrase/1",
                geometry=GroundingGeometry.BOX,
                text=text,
            ),
            configuration_fingerprint="sha256:grounding",
        ),
        provenance=BackendProvenance(
            backend_id="fake_grounding",
            capability="region_grounding",
            provider="fake",
            model="fake",
            version="1",
            configuration_fingerprint="sha256:grounding",
        ),
        rendered_prompt=f"find {text}",
        raw_response="<box><100><200><600><800></box><box><900><900></box>",
        outputs=(
            GroundingOutput(
                output_index=0,
                native_text="<box><100><200><600><800></box>",
                box=BoundingBox2D(x=2.0, y=2.0, width=10.0, height=6.0),
            ),
            GroundingOutput(
                output_index=1,
                native_text="<box><900><900></box>",
                point=GroundingPoint(x=18.0, y=9.0),
            ),
        ),
        diagnostics=GroundingDiagnostics(latency_ms=1.0),
        effective_configuration=MappingProxyType({}),
    )


def _mask(x0: int, y0: int, x1: int, y1: int) -> InlineMask:
    return InlineMask(
        width=WIDTH,
        height=HEIGHT,
        data=tuple(x0 <= x < x1 and y0 <= y < y1 for y in range(HEIGHT) for x in range(WIDTH)),
    )


def _refinement(grounding: RegionGroundingExecution) -> RegionRefinementExecution:
    request = RegionRefinementRequest(
        perception_result_id=RESULT_ID,
        image=_image(),
        prompts=refinement_prompts_from(grounding),
        configuration_fingerprint="sha256:refiner",
    )
    box, point = request.prompts
    score = BackendScore(name="predicted_iou", value=0.91, semantics="SAM2-native predicted IoU")
    return RegionRefinementExecution(
        request=request,
        provenance=REFINER,
        outcomes=(
            refinement_outcome(
                request=request,
                prompt=box,
                provenance=REFINER,
                mask=_mask(4, 3, 9, 7),
                native_scores=(score,),
            ),
            refinement_outcome(
                request=request, prompt=point, provenance=REFINER, mask=_mask(0, 0, 0, 0)
            ),
        ),
        diagnostics=RefinementDiagnostics(latency_ms=2.0),
        effective_configuration=MappingProxyType({"checkpoint": "sam2.1_hiera_large"}),
    )


def _result(
    groundings: tuple[RegionGroundingExecution, ...],
    refinements: tuple[RegionRefinementExecution, ...],
) -> PerceptionResult:
    result = PerceptionResult(
        result_id=RESULT_ID,
        source_observation_id=SOURCE_ID,
        run_id=RUN_ID,
        sequence_artifact_id="sequence-0001",
        created_at="2026-01-01T00:00:00+00:00",
    )
    return with_refined_regions(with_grounded_regions(result, groundings), refinements)


def _writer(tmp_path: Path) -> PerceptionRunWriter:
    return PerceptionRunWriter(
        output_dir=tmp_path / "run",
        sequence_name="corridor-02",
        run_id=RUN_ID,
        run_index=1,
        sequence_artifact_id="sequence-0001",
        selection_id="sha256:aaaa",
        enabled_capabilities=frozenset({"region_grounding", "region_refinement"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:test",
    )


def _write(tmp_path: Path) -> tuple[RegionGroundingExecution, RegionRefinementExecution]:
    grounding = _grounding()
    refinement = _refinement(grounding)
    writer = _writer(tmp_path)
    writer.add_result(_result((grounding,), (refinement,)))
    writer.add_region_grounding(grounding)
    writer.add_region_refinement(refinement)
    writer.finalize()
    return grounding, refinement


def test_refined_masks_and_their_lineage_survive_the_run_artifact(tmp_path: Path) -> None:
    grounding, refinement = _write(tmp_path)

    reader = PerceptionRunReader(tmp_path / "run")
    (reopened,) = reader.list_region_refinements()
    result = reader.result(SOURCE_ID)
    refined_id = refinement.regions[0].region_id

    assert reader.verify_integrity() == []
    assert reopened.request == refinement.request
    assert reopened.outcomes[1] == refinement.outcomes[1]
    refined = next(region for region in result.regions if region.region_id == refined_id)
    assert reopened.outcomes[0].region == refined
    assert refined.mask is None and refined.mask_reference is not None
    assert refined.contributor_candidate_ids == (refinement.request.prompts[0].proposal_id,)
    assert reader.mask_store().load(SOURCE_ID, refined_id) == _mask(4, 3, 9, 7)
    assert reopened.outcomes[0].native_scores == refinement.outcomes[0].native_scores
    assert result.regions[0] == grounding.regions[0]
    assert (tmp_path / "run" / "outputs/region-refinement.jsonl").is_file()


def test_the_refinement_stream_never_inlines_mask_pixels(tmp_path: Path) -> None:
    _write(tmp_path)

    record = json.loads((tmp_path / "run" / "outputs/region-refinement.jsonl").read_text())

    assert "mask" not in record["outcomes"][0]["region"]
    assert record["outcomes"][0]["region"]["mask_reference"].startswith("outputs/masks/")


def test_a_run_without_refinement_has_no_refinement_stream(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.add_result(_result((), ()))
    manifest = writer.finalize()

    assert PerceptionRunReader(tmp_path / "run").list_region_refinements() == []
    assert all("region-refinement" not in entry.path for entry in manifest.file_inventory)


def test_finalize_rejects_refined_regions_missing_from_the_result(tmp_path: Path) -> None:
    grounding = _grounding()
    writer = _writer(tmp_path)
    writer.add_result(_result((grounding,), ()))
    writer.add_region_grounding(grounding)
    writer.add_region_refinement(_refinement(grounding))

    with pytest.raises(RunArtifactError, match="materialized"):
        writer.finalize()


def test_finalize_rejects_a_prompt_whose_grounding_is_not_in_the_run(tmp_path: Path) -> None:
    grounding = _grounding()
    refinement = _refinement(grounding)
    writer = _writer(tmp_path)
    writer.add_result(_result((grounding,), (refinement,)))
    writer.add_region_refinement(refinement)

    with pytest.raises(RunArtifactError, match="grounding"):
        writer.finalize()


def test_finalize_rejects_a_prompt_that_is_not_the_grounding_output_it_names(
    tmp_path: Path,
) -> None:
    grounding = _grounding()
    refinement = _refinement(grounding)
    point_prompt = refinement.request.prompts[1]
    moved = replace(point_prompt, point=GroundingPoint(x=1.0, y=1.0))
    tampered_request = replace(refinement.request, prompts=(refinement.request.prompts[0], moved))
    tampered = replace(
        refinement,
        request=tampered_request,
        outcomes=(
            refinement.outcomes[0],
            replace(refinement.outcomes[1], prompt=moved),
        ),
    )
    writer = _writer(tmp_path)
    writer.add_result(_result((grounding,), (tampered,)))
    writer.add_region_grounding(grounding)
    writer.add_region_refinement(tampered)

    with pytest.raises(RunArtifactError, match="grounding output"):
        writer.finalize()


def test_writer_rejects_the_same_refinement_request_twice(tmp_path: Path) -> None:
    refinement = _refinement(_grounding())
    writer = _writer(tmp_path)
    writer.add_region_refinement(refinement)

    with pytest.raises(RunArtifactError, match="duplicate refinement request"):
        writer.add_region_refinement(refinement)


def test_writer_refuses_refinement_after_finalize(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.add_result(_result((), ()))
    writer.finalize()

    with pytest.raises(RunArtifactError, match="finalize"):
        writer.add_region_refinement(_refinement(_grounding()))

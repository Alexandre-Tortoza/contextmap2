"""Refined grounding masks feed the existing 2D→3D membership path (#568).

A grounded box is box-only evidence and is skipped by mask membership; the SAM2-refined
mask of the same proposal is associated, and only through its mask: geometry inside the
proposal box but outside the mask is never attributed to it.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from perception_builders import OBSERVATION_ID, RESULT_ID, RUN_ID, SEQUENCE_ID, rect_mask
from projection_builders import scene_frame

from contextmap.sensor_association.membership import SkipReason, associate_regions
from contextmap.sensor_association.visibility import OcclusionPolicy, resolve_visibility
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    ArtifactReference,
    BackendProvenance,
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingQuery,
    GroundingTask,
    PerceptionResult,
    PerceptionRunReader,
    PerceptionRunWriter,
    PreparedImage,
    RefinementDiagnostics,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionRefinementExecution,
    RegionRefinementRequest,
    refinement_outcome,
    refinement_prompts_from,
    with_grounded_regions,
    with_refined_regions,
)

POLICY = OcclusionPolicy(
    cell_size_px=4, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02
)
REFINER = BackendProvenance(
    backend_id="sam2",
    capability="region_refinement",
    provider="facebook",
    model="sam2",
    version="2.1",
    configuration_fingerprint="sha256:refiner",
)


def _image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=OBSERVATION_ID,
        payload_reference="frame-0001.png",
        payload_artifact=ArtifactReference(
            uri="frame-0001.png", sha256="c" * 64, media_type="image/png"
        ),
        width=640,
        height=480,
    )


def _grounding() -> RegionGroundingExecution:
    return RegionGroundingExecution(
        request=RegionGroundingRequest(
            perception_result_id=RESULT_ID,
            image=_image(),
            query=GroundingQuery(
                task=GroundingTask.PHRASE_GROUNDING,
                policy_id="fake.phrase/1",
                geometry=GroundingGeometry.BOX,
                text="the chair",
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
        rendered_prompt="find the chair",
        raw_response="<box>...</box>",
        outputs=(
            GroundingOutput(
                output_index=0,
                native_text="<box>...</box>",
                box=BoundingBox2D(x=80.0, y=80.0, width=80.0, height=40.0),
            ),
        ),
        diagnostics=GroundingDiagnostics(latency_ms=1.0),
        effective_configuration=MappingProxyType({}),
    )


def test_a_refined_mask_is_associated_and_its_box_proposal_is_not(tmp_path: Path) -> None:
    grounding = _grounding()
    request = RegionRefinementRequest(
        perception_result_id=RESULT_ID,
        image=_image(),
        prompts=refinement_prompts_from(grounding),
        configuration_fingerprint="sha256:refiner",
    )
    refinement = RegionRefinementExecution(
        request=request,
        provenance=REFINER,
        outcomes=(
            refinement_outcome(
                request=request,
                prompt=request.prompts[0],
                provenance=REFINER,
                mask=rect_mask(640, 480, 90, 90, 110, 110),
            ),
        ),
        diagnostics=RefinementDiagnostics(latency_ms=1.0),
        effective_configuration=MappingProxyType({}),
    )
    result = PerceptionResult(
        result_id=RESULT_ID,
        source_observation_id=OBSERVATION_ID,
        run_id=RUN_ID,
        sequence_artifact_id=SEQUENCE_ID,
        created_at="2026-01-01T00:00:00Z",
    )
    writer = PerceptionRunWriter(
        output_dir=tmp_path / "run-0001",
        sequence_name="corridor-02",
        run_id=RUN_ID,
        run_index=1,
        sequence_artifact_id=SEQUENCE_ID,
        selection_id="sha256:aaaa",
        enabled_capabilities=frozenset({"region_grounding", "region_refinement"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:test",
    )
    writer.add_result(
        with_refined_regions(with_grounded_regions(result, (grounding,)), (refinement,))
    )
    writer.add_region_grounding(grounding)
    writer.add_region_refinement(refinement)
    writer.finalize()
    reader = PerceptionRunReader(tmp_path / "run-0001")
    # Ponto 0 cai na máscara; ponto 1 cai na caixa da proposta, mas fora da máscara.
    frame = scene_frame((100, 100, 3.0), (140, 100, 3.0))

    membership = associate_regions(
        resolve_visibility(frame, POLICY),
        reader.result(OBSERVATION_ID),
        mask_loader=reader.mask_store(),
    )

    refined_id = refinement.regions[0].region_id
    assert [region.region_id for region in membership.regions] == [refined_id]
    assert membership.points_of(refined_id).tolist() == [0]
    assert [(item.region_id, item.reason) for item in membership.skipped] == [
        (grounding.regions[0].region_id, SkipReason.NO_INLINE_MASK)
    ]

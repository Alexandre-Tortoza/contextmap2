"""Contract tests for grounding-to-mask refinement (#568).

A grounding proposal (a grounded box region, or a grounded point) is used only as a
segmentation prompt. The refined mask becomes a separately identified ``Region2D``;
the grounding evidence is never mutated, and an empty or off-prompt mask is an explicit
rejection that never replaces the proposal.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BackendProvenance,
    BackendScore,
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingPoint,
    GroundingQuery,
    GroundingRejectionReason,
    GroundingTask,
    InlineMask,
    PerceptionResult,
    PerceptionRunId,
    PreparedImage,
    RefinementDiagnostics,
    RefinementOutcome,
    RefinementPrompt,
    RefinementRejectionReason,
    RefinementRequestError,
    RegionDiscovery,
    RegionGrounding,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionRefinement,
    RegionRefinementCapabilities,
    RegionRefinementExecution,
    RegionRefinementRequest,
    RejectedGroundingOutput,
    decode_region_refinement_execution,
    encode_region_refinement_execution,
    grounding_point_id_for,
    grounding_region_id_for,
    perception_result_id_for,
    refined_region_id_for,
    refinement_outcome,
    refinement_prompts_from,
    validate_refinement_request,
    with_grounded_regions,
    with_refined_regions,
)

SOURCE_ID = SourceObservationId("frame-0124")
RUN_ID = PerceptionRunId("run-0001")
RESULT_ID = perception_result_id_for(run_id=RUN_ID, source_observation_id=SOURCE_ID)
WIDTH, HEIGHT = 20, 10
GROUNDING_FINGERPRINT = "sha256:grounding"
REFINER_FINGERPRINT = "sha256:refiner"


def _image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SOURCE_ID,
        payload_reference="frame-0124.png",
        payload_artifact=ArtifactReference(
            uri="frame-0124.png", sha256="a" * 64, media_type="image/png"
        ),
        width=WIDTH,
        height=HEIGHT,
    )


def _grounding_provenance() -> BackendProvenance:
    return BackendProvenance(
        backend_id="locateanything",
        capability="region_grounding",
        provider="nvidia",
        model="nvidia/LocateAnything-3B",
        version="0" * 40,
        configuration_fingerprint=GROUNDING_FINGERPRINT,
    )


def _refiner_provenance(fingerprint: str = REFINER_FINGERPRINT) -> BackendProvenance:
    return BackendProvenance(
        backend_id="sam2",
        capability="region_refinement",
        provider="facebook",
        model="sam2.1_hiera_large",
        version="2.1",
        configuration_fingerprint=fingerprint,
    )


def _grounding() -> RegionGroundingExecution:
    """A grounding answer with a box (index 0), a rejected span (1) and a point (2)."""
    return RegionGroundingExecution(
        request=RegionGroundingRequest(
            perception_result_id=RESULT_ID,
            image=_image(),
            query=GroundingQuery(
                task=GroundingTask.PHRASE_GROUNDING,
                policy_id="locateanything.phrase-grounding/1",
                geometry=GroundingGeometry.BOX,
                text="the chair",
            ),
            configuration_fingerprint=GROUNDING_FINGERPRINT,
        ),
        provenance=_grounding_provenance(),
        rendered_prompt="Locate all the instances that match the following description: chair.",
        raw_response="<ref>the chair</ref><box><100><200><600><800></box><box><9></box>"
        "<box><900><900></box>",
        outputs=(
            GroundingOutput(
                output_index=0,
                native_text="<box><100><200><600><800></box>",
                label="the chair",
                box=BoundingBox2D(x=2.0, y=2.0, width=10.0, height=6.0),
            ),
            GroundingOutput(
                output_index=2,
                native_text="<box><900><900></box>",
                label="the chair",
                point=GroundingPoint(x=18.0, y=9.0),
            ),
        ),
        rejected_outputs=(
            RejectedGroundingOutput(
                output_index=1,
                native_text="<box><9></box>",
                reason=GroundingRejectionReason.MALFORMED_GEOMETRY,
                detail="one coordinate",
                label="the chair",
            ),
        ),
        diagnostics=GroundingDiagnostics(latency_ms=3.0),
        effective_configuration=MappingProxyType({"model": "nvidia/LocateAnything-3B"}),
    )


def _mask(*cells: tuple[int, int]) -> InlineMask:
    marked = set(cells)
    return InlineMask(
        width=WIDTH,
        height=HEIGHT,
        data=tuple((x, y) in marked for y in range(HEIGHT) for x in range(WIDTH)),
    )


def _rect(x0: int, y0: int, x1: int, y1: int) -> InlineMask:
    return _mask(*((x, y) for y in range(y0, y1) for x in range(x0, x1)))


def _request(
    prompts: tuple[RefinementPrompt, ...] | None = None, *, fingerprint: str = REFINER_FINGERPRINT
) -> RegionRefinementRequest:
    return RegionRefinementRequest(
        perception_result_id=RESULT_ID,
        image=_image(),
        prompts=prompts if prompts is not None else refinement_prompts_from(_grounding()),
        configuration_fingerprint=fingerprint,
    )


def _outcome(
    prompt: RefinementPrompt, mask: InlineMask, *, fingerprint: str = REFINER_FINGERPRINT
) -> RefinementOutcome:
    return refinement_outcome(
        request=_request(fingerprint=fingerprint),
        prompt=prompt,
        provenance=_refiner_provenance(fingerprint),
        mask=mask,
    )


def _box_prompt() -> RefinementPrompt:
    return refinement_prompts_from(_grounding())[0]


def _point_prompt() -> RefinementPrompt:
    return refinement_prompts_from(_grounding())[1]


# --- prompts --------------------------------------------------------------------------------


def test_every_accepted_grounding_output_becomes_a_prompt_in_answer_order() -> None:
    grounding = _grounding()

    box, point = refinement_prompts_from(grounding)

    assert box.proposal_id == grounding_region_id_for(
        result_id=RESULT_ID, request_id=str(grounding.request_id), index=0
    )
    assert box.proposal_id == grounding.regions[0].region_id
    assert box.box == grounding.outputs[0].box and box.point is None
    assert point.proposal_id == grounding_point_id_for(
        result_id=RESULT_ID, request_id=str(grounding.request_id), index=2
    )
    assert point.point == grounding.outputs[1].point and point.box is None
    assert {prompt.grounding_request_id for prompt in (box, point)} == {grounding.request_id}
    assert [prompt.output_index for prompt in (box, point)] == [0, 2]
    assert box.grounding_provenance == grounding.provenance


def test_a_prompt_carries_exactly_one_geometry_from_a_grounding_backend() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        replace(_box_prompt(), point=GroundingPoint(x=1.0, y=1.0))
    with pytest.raises(ValueError, match="region_grounding"):
        replace(_box_prompt(), grounding_provenance=_refiner_provenance())


# --- request ----------------------------------------------------------------------------------


def test_the_request_identity_follows_its_prompts_image_and_configuration() -> None:
    assert _request().request_id == _request().request_id
    assert str(_request().request_id).startswith("refinement-")
    assert _request(fingerprint="sha256:other").request_id != _request().request_id
    assert _request((_box_prompt(),)).request_id != _request().request_id


def test_a_request_needs_unique_prompts_inside_the_image() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _request(())
    with pytest.raises(ValueError, match="unique"):
        _request((_box_prompt(), _box_prompt()))
    outside = replace(_point_prompt(), point=GroundingPoint(x=21.0, y=1.0))
    with pytest.raises(ValueError, match="image"):
        _request((outside,))


def test_an_unsupported_prompt_geometry_is_refused_before_inference() -> None:
    box_only = RegionRefinementCapabilities(prompt_geometries=frozenset({GroundingGeometry.BOX}))

    validate_refinement_request(_request((_box_prompt(),)), box_only)
    with pytest.raises(RefinementRequestError, match="point"):
        validate_refinement_request(_request(), box_only)


# --- outcomes ---------------------------------------------------------------------------------


def test_a_box_prompt_produces_a_separately_identified_mask_backed_region() -> None:
    grounding = _grounding()
    prompt = refinement_prompts_from(grounding)[0]
    mask = _rect(4, 3, 9, 7)

    outcome = _outcome(prompt, mask)

    region = outcome.region
    assert region is not None
    assert region.region_id == refined_region_id_for(
        result_id=RESULT_ID,
        proposal_id=prompt.proposal_id,
        configuration_fingerprint=REFINER_FINGERPRINT,
    )
    assert region.region_id != prompt.proposal_id
    assert region.contributor_candidate_ids == (prompt.proposal_id,)
    assert region.mask == mask
    assert (region.bounding_box.x_min, region.bounding_box.x_max) == (4.0, 9.0)
    assert (region.bounding_box.y_min, region.bounding_box.y_max) == (3.0, 7.0)
    assert region.area_pixels == 20
    assert region.provenance.capability == "region_refinement"
    assert region.region_kind is None
    assert outcome.rejection_reason is None
    assert dict(outcome.diagnostics)["mask_area_px"] == 20
    assert dict(outcome.diagnostics)["mask_px_outside_prompt_box"] == 0
    assert grounding == _grounding()
    assert grounding.regions[0].mask is None


def test_mask_pixels_outside_the_prompt_box_are_kept_and_counted_not_clipped() -> None:
    outcome = _outcome(_box_prompt(), _rect(10, 3, 14, 5))

    assert outcome.region is not None
    assert outcome.region.bounding_box.x_max == 14.0
    assert dict(outcome.diagnostics)["mask_px_inside_prompt_box"] == 4
    assert dict(outcome.diagnostics)["mask_px_outside_prompt_box"] == 4


def test_an_empty_mask_is_an_explicit_rejection_that_never_replaces_the_proposal() -> None:
    outcome = _outcome(_box_prompt(), _mask())

    assert outcome.region is None
    assert outcome.rejection_reason is RefinementRejectionReason.EMPTY_MASK
    assert outcome.rejection_detail


def test_a_mask_that_misses_its_prompt_is_rejected() -> None:
    off_box = _outcome(_box_prompt(), _rect(15, 0, 18, 2))
    off_point = _outcome(_point_prompt(), _rect(0, 0, 3, 3))

    assert off_box.rejection_reason is RefinementRejectionReason.PROMPT_NOT_COVERED
    assert off_point.rejection_reason is RefinementRejectionReason.PROMPT_NOT_COVERED
    assert dict(off_point.diagnostics)["prompt_point_covered"] is False


def test_image_edge_prompts_are_handled_deterministically() -> None:
    edge_box = replace(_box_prompt(), box=BoundingBox2D(x=15.0, y=5.0, width=5.0, height=5.0))
    edge_point = replace(_point_prompt(), point=GroundingPoint(x=20.0, y=10.0))
    corner = _rect(17, 7, 20, 10)

    box_outcome = _outcome(edge_box, corner)
    point_outcome = _outcome(edge_point, _mask((19, 9)))

    assert box_outcome.region is not None
    assert box_outcome.region.bounding_box.x_max == WIDTH
    assert box_outcome.region.bounding_box.y_max == HEIGHT
    assert point_outcome.region is not None
    assert dict(point_outcome.diagnostics)["prompt_point_covered"] is True
    assert _outcome(edge_box, corner) == box_outcome


def test_a_mask_in_another_image_space_is_an_error_not_a_rejection() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        _outcome(_box_prompt(), InlineMask(width=2, height=2, data=(True,) * 4))


def test_another_refiner_configuration_is_another_refinement_identity() -> None:
    mask = _rect(4, 3, 9, 7)

    first = _outcome(_box_prompt(), mask).region
    again = _outcome(_box_prompt(), mask).region
    other = _outcome(_box_prompt(), mask, fingerprint="sha256:other-refiner").region

    assert first is not None and again is not None and other is not None
    assert first.region_id == again.region_id
    assert first.region_id != other.region_id


def test_an_outcome_is_either_a_region_or_a_rejection() -> None:
    accepted = _outcome(_box_prompt(), _rect(4, 3, 9, 7))

    with pytest.raises(ValueError, match="either"):
        replace(
            accepted, rejection_reason=RefinementRejectionReason.EMPTY_MASK, rejection_detail="x"
        )
    with pytest.raises(ValueError, match="contributor"):
        assert accepted.region is not None
        replace(accepted, region=replace(accepted.region, contributor_candidate_ids=()))


# --- execution --------------------------------------------------------------------------------


def _execution(
    outcomes: tuple[RefinementOutcome, ...] | None = None,
) -> RegionRefinementExecution:
    box, point = refinement_prompts_from(_grounding())
    return RegionRefinementExecution(
        request=_request(),
        provenance=_refiner_provenance(),
        outcomes=outcomes
        if outcomes is not None
        else (_outcome(box, _rect(4, 3, 9, 7)), _outcome(point, _mask())),
        diagnostics=RefinementDiagnostics(latency_ms=5.0, peak_memory_bytes=1024),
        effective_configuration=MappingProxyType({"checkpoint": "sam2.1_hiera_large"}),
        runtime_identity=MappingProxyType({"sam2": "1.1.0"}),
    )


def test_an_execution_answers_every_prompt_in_order() -> None:
    execution = _execution()

    assert [item.prompt for item in execution.outcomes] == list(execution.request.prompts)
    assert len(execution.regions) == 1
    assert [item.rejection_reason for item in execution.rejected] == [
        RefinementRejectionReason.EMPTY_MASK
    ]
    with pytest.raises(ValueError, match="prompt"):
        _execution(outcomes=_execution().outcomes[:1])


def test_the_execution_provenance_is_the_refiner_that_was_asked() -> None:
    with pytest.raises(ValueError, match="region_refinement"):
        replace(
            _execution(), provenance=replace(_refiner_provenance(), capability="region_discovery")
        )
    with pytest.raises(ValueError, match="fingerprint"):
        replace(_execution(), provenance=_refiner_provenance("sha256:other"))


def test_refinement_evidence_and_lineage_survive_serialization() -> None:
    execution = _execution()
    region = execution.regions[0]
    persisted_region = replace(
        region, mask=None, mask_reference=f"outputs/masks/{SOURCE_ID}/{region.region_id}.npy"
    )
    persisted = replace(
        execution,
        outcomes=(replace(execution.outcomes[0], region=persisted_region), execution.outcomes[1]),
    )

    decoded = decode_region_refinement_execution(encode_region_refinement_execution(persisted))

    assert decoded == persisted
    assert decoded.outcomes[0].region is not None
    assert decoded.outcomes[0].region.contributor_candidate_ids == (
        execution.request.prompts[0].proposal_id,
    )
    assert decoded.request.prompts[0].grounding_provenance == _grounding_provenance()


def test_a_native_refiner_score_survives_serialization_and_a_missing_one_stays_missing() -> None:
    score = BackendScore(
        name="predicted_iou",
        value=0.8765432109876,
        semantics="SAM2-native predicted mask IoU; not a calibrated probability",
    )
    box, point = refinement_prompts_from(_grounding())
    scored = replace(_outcome(box, _rect(4, 3, 9, 7)), native_scores=(score,))
    assert scored.region is not None
    persisted_region = replace(scored.region, mask=None, mask_reference="outputs/masks/m.npy")
    execution = _execution(
        outcomes=(replace(scored, region=persisted_region), _outcome(point, _mask()))
    )

    decoded = decode_region_refinement_execution(encode_region_refinement_execution(execution))

    assert decoded.outcomes[0].native_scores == (score,)
    assert decoded.outcomes[1].native_scores == ()


def test_refined_regions_join_the_result_next_to_the_untouched_grounded_boxes() -> None:
    grounding = _grounding()
    result = with_grounded_regions(
        PerceptionResult(
            result_id=RESULT_ID,
            source_observation_id=SOURCE_ID,
            run_id=RUN_ID,
            sequence_artifact_id="sequence-0001",
            created_at="2026-01-01T00:00:00+00:00",
        ),
        (grounding,),
    )
    execution = _execution()

    refined = with_refined_regions(result, (execution,))

    assert refined.regions == (*grounding.regions, *execution.regions)
    assert refined.regions[0] == grounding.regions[0]
    assert refined.regions[0].mask is None


def test_refined_regions_never_join_another_result() -> None:
    other = PerceptionResult(
        result_id=perception_result_id_for(
            run_id=PerceptionRunId("run-0002"), source_observation_id=SOURCE_ID
        ),
        source_observation_id=SOURCE_ID,
        run_id=PerceptionRunId("run-0002"),
        sequence_artifact_id="sequence-0001",
        created_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="perception_result_id"):
        with_refined_regions(other, (_execution(),))


def test_refinement_is_a_port_of_its_own() -> None:
    class _Refiner:
        def backend_provenance(self) -> BackendProvenance:
            return _refiner_provenance()

        def capabilities(self) -> RegionRefinementCapabilities:
            return RegionRefinementCapabilities(
                prompt_geometries=frozenset({GroundingGeometry.BOX, GroundingGeometry.POINT})
            )

        def refine(self, request: RegionRefinementRequest) -> RegionRefinementExecution:
            return _execution()

    assert isinstance(_Refiner(), RegionRefinement)
    assert not isinstance(_Refiner(), RegionGrounding)
    assert not isinstance(_Refiner(), RegionDiscovery)

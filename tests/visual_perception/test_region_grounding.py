"""Contract tests for prompt-conditioned region grounding (#566).

Grounding answers an explicit language query about one prepared image. These tests pin
the request identity, the capability validation that happens before any inference, the
serialization of the query, and the rule that only box outputs become canonical
``Region2D`` evidence. Every backend here is a deterministic fake.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from types import MappingProxyType

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BackendProvenance,
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingPoint,
    GroundingQuery,
    GroundingQueryPolicy,
    GroundingRejectionReason,
    GroundingRequestError,
    GroundingTask,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PreparedImage,
    Region2D,
    RegionDiscovery,
    RegionGrounding,
    RegionGroundingCapabilities,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionId,
    RejectedGroundingOutput,
    TransformationRecord,
    ValidRegion,
    decode_region_grounding_execution,
    decode_region_grounding_request,
    encode_region_grounding_execution,
    encode_region_grounding_request,
    grounding_region_id_for,
    perception_result_id_for,
    validate_grounding_query,
    with_grounded_regions,
)
from contextmap.visual_perception.region_models import InlineMask

SOURCE_ID = SourceObservationId("frame-0124")
RUN_ID = PerceptionRunId("run-0001")
RESULT_ID = perception_result_id_for(run_id=RUN_ID, source_observation_id=SOURCE_ID)
FINGERPRINT = "sha256:grounding-config"
IMAGE_SHA = "a" * 64
BOX_POLICY = "fake.category-detection/1"
PHRASE_POLICY = "fake.phrase-grounding/1"
POINT_POLICY = "fake.pointing/1"


def _image(*, sha256: str = IMAGE_SHA, width: int = 640, height: int = 480) -> PreparedImage:
    return PreparedImage(
        source_observation_id=SOURCE_ID,
        payload_reference="frame-0124.png",
        payload_artifact=ArtifactReference(
            uri="frame-0124.png", sha256=sha256, media_type="image/png"
        ),
        width=width,
        height=height,
    )


def _categories(*categories: str) -> GroundingQuery:
    return GroundingQuery(
        task=GroundingTask.CATEGORY_DETECTION,
        policy_id=BOX_POLICY,
        geometry=GroundingGeometry.BOX,
        categories=categories or ("chair", "table"),
    )


def _phrase(
    text: str = "the red chair", *, geometry: GroundingGeometry = GroundingGeometry.BOX
) -> GroundingQuery:
    return GroundingQuery(
        task=GroundingTask.PHRASE_GROUNDING,
        policy_id=POINT_POLICY if geometry is GroundingGeometry.POINT else PHRASE_POLICY,
        geometry=geometry,
        text=text,
    )


def _request(
    query: GroundingQuery | None = None,
    *,
    result_id: PerceptionResultId = RESULT_ID,
    image: PreparedImage | None = None,
    fingerprint: str = FINGERPRINT,
) -> RegionGroundingRequest:
    return RegionGroundingRequest(
        perception_result_id=result_id,
        image=image or _image(),
        query=query or _categories(),
        configuration_fingerprint=fingerprint,
    )


def _provenance(fingerprint: str = FINGERPRINT) -> BackendProvenance:
    return BackendProvenance(
        backend_id="fake_region_grounding",
        capability="region_grounding",
        provider="fake",
        model="fake-grounder",
        version="1",
        configuration_fingerprint=fingerprint,
    )


def _box_output(index: int = 0, *, label: str | None = "chair") -> GroundingOutput:
    return GroundingOutput(
        output_index=index,
        native_text="<box><100><200><300><400></box>",
        label=label,
        box=BoundingBox2D(x=64.0, y=96.0, width=128.0, height=96.0),
    )


def _point_output(index: int = 1) -> GroundingOutput:
    return GroundingOutput(
        output_index=index,
        native_text="<box><500><500></box>",
        label="the red chair",
        point=GroundingPoint(x=320.0, y=240.0),
    )


def _execution(
    request: RegionGroundingRequest | None = None,
    *,
    outputs: tuple[GroundingOutput, ...] | None = None,
    rejected: tuple[RejectedGroundingOutput, ...] = (),
    no_match_labels: tuple[str | None, ...] = (),
    native: tuple[tuple[str, str | int | float | bool | None], ...] = (),
    raw_response: str = "<ref>chair</ref><box><100><200><300><400></box><|im_end|>",
) -> RegionGroundingExecution:
    return RegionGroundingExecution(
        request=request or _request(),
        provenance=_provenance(),
        rendered_prompt="Locate all the instances: chair</c>table.",
        raw_response=raw_response,
        outputs=(_box_output(),) if outputs is None else outputs,
        rejected_outputs=rejected,
        no_match_labels=no_match_labels,
        diagnostics=GroundingDiagnostics(latency_ms=12.5, native=native),
        effective_configuration=MappingProxyType({"model": "fake-grounder", "temperature": 0.0}),
        runtime_identity=MappingProxyType({"torch": "2.8.0"}),
    )


def _capabilities() -> RegionGroundingCapabilities:
    return RegionGroundingCapabilities(
        query_policies=(
            GroundingQueryPolicy(
                policy_id=BOX_POLICY,
                task=GroundingTask.CATEGORY_DETECTION,
                geometry=GroundingGeometry.BOX,
            ),
            GroundingQueryPolicy(
                policy_id=PHRASE_POLICY,
                task=GroundingTask.PHRASE_GROUNDING,
                geometry=GroundingGeometry.BOX,
            ),
        )
    )


# --- request identity ---------------------------------------------------------------


def test_same_image_query_and_configuration_produce_identical_request_identity() -> None:
    assert _request().request_id == _request().request_id
    assert str(_request().request_id).startswith("grounding-")


def test_request_identity_is_a_content_key_independent_of_the_owning_run() -> None:
    other_result = perception_result_id_for(
        run_id=PerceptionRunId("run-0002"), source_observation_id=SOURCE_ID
    )

    assert _request(result_id=other_result).request_id == _request().request_id


def test_changing_only_the_query_text_changes_request_and_dependent_region_identity() -> None:
    first = _execution(_request(_phrase("the red chair")))
    second = _execution(_request(_phrase("the blue chair")))

    assert first.request_id != second.request_id
    assert first.regions[0].region_id != second.regions[0].region_id


@pytest.mark.parametrize(
    "changed",
    [
        _request(_categories("table", "chair")),
        _request(image=_image(sha256="b" * 64)),
        _request(fingerprint="sha256:another-config"),
        _request(_phrase(geometry=GroundingGeometry.POINT)),
        _request(replace(_categories(), policy_id="fake.category-detection/2")),
    ],
    ids=["category-order", "image-content", "configuration", "geometry", "policy-version"],
)
def test_every_inference_input_takes_part_in_request_identity(
    changed: RegionGroundingRequest,
) -> None:
    assert changed.request_id != _request().request_id


# --- query and request invariants ---------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        {"task": GroundingTask.CATEGORY_DETECTION, "categories": ()},
        {"task": GroundingTask.CATEGORY_DETECTION, "categories": ("chair",), "text": "x"},
        {"task": GroundingTask.CATEGORY_DETECTION, "categories": ("chair", " ")},
        {"task": GroundingTask.CATEGORY_DETECTION, "categories": ("chair", "chair")},
        {"task": GroundingTask.PHRASE_GROUNDING, "text": None},
        {"task": GroundingTask.PHRASE_GROUNDING, "text": "   "},
        {"task": GroundingTask.PHRASE_GROUNDING, "text": "chair", "categories": ("chair",)},
    ],
    ids=[
        "no-categories",
        "categories-with-text",
        "blank-category",
        "duplicate-category",
        "no-text",
        "blank-text",
        "text-with-categories",
    ],
)
def test_query_shape_must_match_its_task(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GroundingQuery(policy_id=BOX_POLICY, geometry=GroundingGeometry.BOX, **values)  # type: ignore[arg-type]


def test_query_requires_a_versioned_policy_identity() -> None:
    with pytest.raises(ValueError, match="policy_id"):
        replace(_categories(), policy_id=" ")


def test_request_requires_a_content_addressed_prepared_image() -> None:
    unaddressed = PreparedImage(
        source_observation_id=SOURCE_ID, payload_reference="frame-0124.png", width=640, height=480
    )

    with pytest.raises(ValueError, match="payload_artifact"):
        _request(image=unaddressed)


def test_request_refuses_spatial_constraints_it_would_not_apply() -> None:
    mask = InlineMask(width=2, height=2, data=(True, True, True, False))
    constrained = replace(
        _image(width=2, height=2),
        valid_region=ValidRegion(mask=mask, reason="lens vignetting", source="calibration"),
    )

    with pytest.raises(ValueError, match="constraint"):
        _request(image=constrained)


def test_request_requires_a_configuration_fingerprint() -> None:
    with pytest.raises(ValueError, match="configuration_fingerprint"):
        _request(fingerprint="")


# --- capability validation before inference -----------------------------------------


def test_a_supported_query_validates() -> None:
    validate_grounding_query(_categories(), _capabilities())
    validate_grounding_query(_phrase(), _capabilities())


def test_unsupported_requested_geometry_fails_before_inference() -> None:
    with pytest.raises(GroundingRequestError, match="point"):
        validate_grounding_query(_phrase(geometry=GroundingGeometry.POINT), _capabilities())


def test_a_policy_declared_for_another_geometry_is_refused() -> None:
    query = replace(_phrase(), geometry=GroundingGeometry.POINT)

    with pytest.raises(GroundingRequestError, match="geometry"):
        validate_grounding_query(query, _capabilities())


def test_a_policy_declared_for_another_task_is_refused() -> None:
    query = GroundingQuery(
        task=GroundingTask.PHRASE_GROUNDING,
        policy_id=BOX_POLICY,
        geometry=GroundingGeometry.BOX,
        text="chair",
    )

    with pytest.raises(GroundingRequestError, match="task"):
        validate_grounding_query(query, _capabilities())


def test_an_unknown_policy_is_refused() -> None:
    with pytest.raises(GroundingRequestError, match="policy"):
        validate_grounding_query(replace(_categories(), policy_id="other/1"), _capabilities())


def test_capabilities_declare_unique_policies() -> None:
    policy = _capabilities().query_policies[0]

    with pytest.raises(ValueError, match="unique"):
        RegionGroundingCapabilities(query_policies=(policy, policy))
    with pytest.raises(ValueError):
        RegionGroundingCapabilities(query_policies=())


# --- serialization --------------------------------------------------------------------


def test_query_text_survives_request_serialization_exactly() -> None:
    request = _request(_phrase("  a  chair\tnext to the café — left  "))
    transformed = replace(
        request,
        image=replace(
            request.image,
            transformations=(
                TransformationRecord(
                    operation="resize",
                    provenance_source="image_preparation",
                    parameters=(("width", 640), ("interpolation", "bilinear")),
                    input_dimensions=(1280, 960),
                    output_dimensions=(640, 480),
                    output_image=ArtifactReference(
                        uri="frame-0124.png", sha256=IMAGE_SHA, media_type="image/png"
                    ),
                ),
            ),
        ),
    )

    decoded = decode_region_grounding_request(encode_region_grounding_request(transformed))

    assert decoded == transformed
    assert decoded.query.text == "  a  chair\tnext to the café — left  "
    assert decoded.request_id == transformed.request_id


def test_ordered_categories_survive_request_serialization() -> None:
    request = _request(_categories("table", "chair", "door"))

    decoded = decode_region_grounding_request(encode_region_grounding_request(request))

    assert decoded.query.categories == ("table", "chair", "door")


def test_a_tampered_request_identity_is_refused_on_decode() -> None:
    record = encode_region_grounding_request(_request())
    record["request_id"] = "grounding-" + "0" * 64

    with pytest.raises(ValueError, match="request_id"):
        decode_region_grounding_request(record)


def test_execution_round_trips_with_the_raw_response_held_separately() -> None:
    execution = _execution(
        outputs=(_box_output(0), _point_output(2)),
        rejected=(
            RejectedGroundingOutput(
                output_index=1,
                native_text="<box><900><10><100><20></box>",
                reason=GroundingRejectionReason.INVERTED_GEOMETRY,
                detail="x2 must be greater than x1",
                label="chair",
            ),
        ),
        no_match_labels=("table",),
    )

    record = encode_region_grounding_execution(
        execution, raw_response_reference="outputs/region-grounding-raw/x.txt"
    )

    assert "raw_response" not in record
    assert (
        record["raw_response_sha256"]
        == hashlib.sha256(execution.raw_response.encode("utf-8")).hexdigest()
    )
    decoded = decode_region_grounding_execution(record, raw_response=execution.raw_response)
    assert decoded == execution


def test_a_raw_response_that_does_not_match_its_hash_is_refused() -> None:
    record = encode_region_grounding_execution(
        _execution(), raw_response_reference="outputs/region-grounding-raw/x.txt"
    )

    with pytest.raises(ValueError, match="raw response"):
        decode_region_grounding_execution(record, raw_response="<box>none</box>")


def test_native_diagnostics_survive_serialization_unchanged_and_are_never_confidence() -> None:
    native = (
        ("stats.switch_to_ar", 2),
        ("stats.tps", 12.345678901234),
        ("stats.raw", "num_tokens=18; switch_to_ar=2"),
        ("stats.prefill_time", None),
    )
    output = replace(_box_output(), native_diagnostics=(("decode_mode", "ar"),))
    execution = _execution(outputs=(output,), native=native)

    decoded = decode_region_grounding_execution(
        encode_region_grounding_execution(execution, raw_response_reference="r.txt"),
        raw_response=execution.raw_response,
    )

    assert decoded.diagnostics.native == native
    assert decoded.outputs[0].native_diagnostics == (("decode_mode", "ar"),)
    assert not hasattr(decoded.outputs[0], "confidence")
    assert not hasattr(decoded.regions[0], "confidence")


# --- canonical evidence -------------------------------------------------------------


def test_only_box_outputs_become_region_evidence() -> None:
    execution = _execution(outputs=(_box_output(0), _point_output(1)))

    assert [region.bounding_box for region in execution.regions] == [_box_output().box]
    assert execution.unsupported_outputs == (_point_output(1),)


def test_point_only_output_is_not_promoted_to_a_fabricated_region() -> None:
    execution = _execution(
        _request(_phrase(geometry=GroundingGeometry.POINT)), outputs=(_point_output(0),)
    )

    assert execution.regions == ()
    assert execution.outputs[0].geometry is GroundingGeometry.POINT
    assert execution.outputs[0].box is None


def test_grounded_region_carries_image_space_and_grounding_provenance() -> None:
    region = _execution().regions[0]

    assert region.region_id == grounding_region_id_for(
        result_id=RESULT_ID, request_id=_request().request_id, index=0
    )
    assert region.provenance.capability == "region_grounding"
    assert region.source_observation_id == SOURCE_ID
    assert (region.image_width, region.image_height) == (640, 480)
    assert region.area_pixels == pytest.approx(128.0 * 96.0)
    assert region.region_kind is None
    assert region.is_accepted


def test_same_observation_in_another_run_gets_distinct_local_evidence_identity() -> None:
    other_result = perception_result_id_for(
        run_id=PerceptionRunId("run-0002"), source_observation_id=SOURCE_ID
    )

    first = _execution(_request()).regions[0]
    second = _execution(_request(result_id=other_result)).regions[0]

    assert first.region_id != second.region_id
    assert first.source_observation_id == second.source_observation_id


def test_no_match_is_explicit() -> None:
    empty = _execution(outputs=(), no_match_labels=("chair", "table"))
    rejected_only = _execution(
        outputs=(),
        rejected=(
            RejectedGroundingOutput(
                output_index=0,
                native_text="garbage",
                reason=GroundingRejectionReason.UNRECOGNIZED_TEXT,
                detail="text outside the grounding grammar",
            ),
        ),
    )

    assert empty.no_match
    assert empty.no_match_labels == ("chair", "table")
    assert not rejected_only.no_match
    assert not _execution().no_match


def test_output_indices_are_unique_across_accepted_and_rejected_outputs() -> None:
    rejected = RejectedGroundingOutput(
        output_index=0,
        native_text="<box><1></box>",
        reason=GroundingRejectionReason.MALFORMED_GEOMETRY,
        detail="one coordinate",
    )

    with pytest.raises(ValueError, match="output_index"):
        _execution(rejected=(rejected,))
    with pytest.raises(ValueError, match="output_index"):
        _execution(outputs=(_box_output(0), _box_output(0)))


def test_output_carries_exactly_one_geometry_inside_the_image() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        GroundingOutput(output_index=0, native_text="x")
    with pytest.raises(ValueError, match="exactly one"):
        replace(_box_output(), point=GroundingPoint(x=1.0, y=1.0))
    outside = replace(_box_output(), box=BoundingBox2D(x=600.0, y=0.0, width=100.0, height=10.0))
    with pytest.raises(ValueError, match="image"):
        _execution(outputs=(outside,))
    beyond = replace(_point_output(0), point=GroundingPoint(x=641.0, y=10.0))
    with pytest.raises(ValueError, match="image"):
        _execution(outputs=(beyond,))


def test_execution_provenance_must_be_the_grounding_backend_that_was_asked() -> None:
    with pytest.raises(ValueError, match="region_grounding"):
        replace(_execution(), provenance=replace(_provenance(), capability="region_discovery"))
    with pytest.raises(ValueError, match="fingerprint"):
        replace(_execution(), provenance=_provenance("sha256:other"))


def test_raw_response_is_part_of_the_execution_record() -> None:
    execution = _execution()

    assert (
        execution.raw_response_sha256
        == hashlib.sha256(execution.raw_response.encode("utf-8")).hexdigest()
    )


def test_grounded_regions_join_the_owning_result_next_to_discovered_regions() -> None:
    discovered = Region2D(
        region_id=RegionId(f"{RESULT_ID}--region-0000"),
        bounding_box=BoundingBox2D(x=0, y=0, width=10, height=10),
        provenance=replace(_provenance(), capability="region_discovery"),
    )
    result = PerceptionResult(
        result_id=RESULT_ID,
        source_observation_id=SOURCE_ID,
        run_id=RUN_ID,
        sequence_artifact_id="sequence-0001",
        created_at="2026-01-01T00:00:00+00:00",
        regions=(discovered,),
    )
    execution = _execution()

    merged = with_grounded_regions(result, (execution,))

    assert merged.regions == (discovered, *execution.regions)
    assert result.regions == (discovered,)


def test_grounded_regions_never_join_a_result_that_did_not_own_the_request() -> None:
    other = perception_result_id_for(
        run_id=PerceptionRunId("run-0002"), source_observation_id=SOURCE_ID
    )
    result = PerceptionResult(
        result_id=other,
        source_observation_id=SOURCE_ID,
        run_id=PerceptionRunId("run-0002"),
        sequence_artifact_id="sequence-0001",
        created_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="perception_result_id"):
        with_grounded_regions(result, (_execution(),))


# --- ports ------------------------------------------------------------------------------


class _FakeGrounding:
    def backend_provenance(self) -> BackendProvenance:
        return _provenance()

    def capabilities(self) -> RegionGroundingCapabilities:
        return _capabilities()

    def ground(self, request: RegionGroundingRequest) -> RegionGroundingExecution:
        validate_grounding_query(request.query, self.capabilities())
        return _execution(request)


class _FakeDiscovery:
    def backend_provenance(self) -> BackendProvenance:
        return replace(_provenance(), capability="region_discovery")

    def discover(self, image: PreparedImage) -> Sequence[Region2D]:
        return ()


def test_automatic_discovery_and_prompt_conditioned_grounding_are_distinct_ports() -> None:
    assert isinstance(_FakeDiscovery(), RegionDiscovery)
    assert not isinstance(_FakeDiscovery(), RegionGrounding)
    assert isinstance(_FakeGrounding(), RegionGrounding)
    assert not isinstance(_FakeGrounding(), RegionDiscovery)


def test_a_grounding_backend_refuses_an_unsupported_request_before_inference() -> None:
    with pytest.raises(GroundingRequestError):
        _FakeGrounding().ground(_request(_phrase(geometry=GroundingGeometry.POINT)))

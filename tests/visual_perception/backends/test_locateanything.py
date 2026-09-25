"""Contract tests for the LocateAnything region-grounding adapter (#567).

The runtime is a deterministic fake that returns scripted upstream answers, so these tests
pin the output grammar, the pixel conversion, the explicit rejections and the evidence
kept around each output without a model, a GPU or a network.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    ArtifactReference,
    GroundingGeometry,
    GroundingQuery,
    GroundingRejectionReason,
    GroundingRequestError,
    GroundingTask,
    PerceptionResult,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    PreparedImage,
    RegionGrounding,
    RegionGroundingRequest,
    perception_result_id_for,
    with_grounded_regions,
)
from contextmap.visual_perception.backends.locateanything import (
    CATEGORY_DETECTION_POLICY,
    PHRASE_GROUNDING_POLICY,
    POINTING_POLICY,
    LocateAnythingConfig,
    LocateAnythingGeneration,
    LocateAnythingGenerationMode,
    LocateAnythingRegionGrounding,
    parse_locateanything_response,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"
SOURCE_ID = SourceObservationId("frame-0124")
RUN_ID = PerceptionRunId("run-0001")
RESULT_ID = perception_result_id_for(run_id=RUN_ID, source_observation_id=SOURCE_ID)
STATS = (
    "\nStatistic Info, num_tokens=18; generate_time(s)=0.5000; tps=36.0000; "
    "forward_step=5; num_boxes=2; bps=4.0000; prefill_time=0.1000; switch_to_ar=1\n"
)


def _config(**overrides: Any) -> LocateAnythingConfig:
    values: dict[str, Any] = {
        "model": "nvidia/LocateAnything-3B",
        "revision": REVISION,
        "device": "cuda",
        "dtype": "bfloat16",
        "generation_mode": LocateAnythingGenerationMode.HYBRID,
        "max_new_tokens": 2048,
        "temperature": 0.0,
    }
    values.update(overrides)
    return LocateAnythingConfig(**values)


class FakeRuntime:
    """Returns one scripted generation and records every call it receives."""

    def __init__(self, generation: LocateAnythingGeneration | str) -> None:
        self.generation = (
            generation
            if isinstance(generation, LocateAnythingGeneration)
            else LocateAnythingGeneration(text=generation)
        )
        self.calls: list[tuple[PreparedImage, str, LocateAnythingConfig]] = []

    def generate(
        self, *, image: PreparedImage, prompt: str, config: LocateAnythingConfig
    ) -> LocateAnythingGeneration:
        self.calls.append((image, prompt, config))
        return self.generation


def _backend(
    generation: LocateAnythingGeneration | str = "<|im_end|>", **overrides: Any
) -> tuple[LocateAnythingRegionGrounding, FakeRuntime]:
    runtime = FakeRuntime(generation)
    return LocateAnythingRegionGrounding(config=_config(**overrides), runtime=runtime), runtime


def _image(width: int = 640, height: int = 480) -> PreparedImage:
    return PreparedImage(
        source_observation_id=SOURCE_ID,
        payload_reference="frame-0124.png",
        payload_artifact=ArtifactReference(
            uri="frame-0124.png", sha256="d" * 64, media_type="image/png"
        ),
        width=width,
        height=height,
    )


def _categories(*categories: str) -> GroundingQuery:
    return GroundingQuery(
        task=GroundingTask.CATEGORY_DETECTION,
        policy_id=CATEGORY_DETECTION_POLICY,
        geometry=GroundingGeometry.BOX,
        categories=categories or ("chair", "table"),
    )


def _phrase(text: str = "the red chair") -> GroundingQuery:
    return GroundingQuery(
        task=GroundingTask.PHRASE_GROUNDING,
        policy_id=PHRASE_GROUNDING_POLICY,
        geometry=GroundingGeometry.BOX,
        text=text,
    )


def _pointing(text: str = "the door handle") -> GroundingQuery:
    return GroundingQuery(
        task=GroundingTask.PHRASE_GROUNDING,
        policy_id=POINTING_POLICY,
        geometry=GroundingGeometry.POINT,
        text=text,
    )


def _request(
    backend: LocateAnythingRegionGrounding,
    query: GroundingQuery | None = None,
    *,
    result_id: Any = RESULT_ID,
    image: PreparedImage | None = None,
) -> RegionGroundingRequest:
    return RegionGroundingRequest(
        perception_result_id=result_id,
        image=image or _image(),
        query=query or _categories(),
        configuration_fingerprint=str(backend.backend_provenance().configuration_fingerprint),
    )


# --- response grammar -------------------------------------------------------------------


def test_one_box_is_converted_from_normalized_coordinates_to_prepared_image_pixels() -> None:
    parsed = parse_locateanything_response(
        "<ref>chair</ref><box><100><200><300><400></box><|im_end|>",
        image_width=640,
        image_height=480,
    )

    (output,) = parsed.outputs
    assert output.box is not None
    assert (output.box.x, output.box.y) == (64.0, 96.0)
    assert (output.box.width, output.box.height) == (128.0, 96.0)
    assert output.label == "chair"
    assert output.native_text == "<box><100><200><300><400></box>"
    assert parsed.terminated
    assert parsed.rejected_outputs == ()


def test_multiple_boxes_keep_response_order_and_their_category_association() -> None:
    parsed = parse_locateanything_response(
        "<ref>table</ref><box><0><0><10><10></box><box><20><20><30><30></box>"
        "<ref>chair</ref><box><40><40><50><50></box><|im_end|>",
        image_width=1000,
        image_height=1000,
    )

    assert [output.output_index for output in parsed.outputs] == [0, 1, 2]
    assert [output.label for output in parsed.outputs] == ["table", "table", "chair"]
    assert [output.box.x for output in parsed.outputs if output.box] == [0.0, 20.0, 40.0]


def test_an_explicit_no_object_answer_is_a_no_match_not_a_box() -> None:
    parsed = parse_locateanything_response(
        "<ref>chair</ref><box>none</box><ref>table</ref><box>none</box><|im_end|>",
        image_width=640,
        image_height=480,
    )

    assert parsed.outputs == ()
    assert parsed.rejected_outputs == ()
    assert parsed.no_match_labels == ("chair", "table")


@pytest.mark.parametrize("answer", ["", "<|im_end|>", "  \n<|im_end|>"])
def test_an_answer_without_geometry_has_no_output(answer: str) -> None:
    parsed = parse_locateanything_response(answer, image_width=640, image_height=480)

    assert parsed.outputs == ()
    assert parsed.rejected_outputs == ()


def test_coordinates_at_the_image_boundaries_map_to_the_image_edges() -> None:
    parsed = parse_locateanything_response(
        "<box><0><0><1000><1000></box>", image_width=641, image_height=479
    )

    (output,) = parsed.outputs
    assert output.box is not None
    assert (output.box.x_min, output.box.y_min) == (0.0, 0.0)
    assert (output.box.x_max, output.box.y_max) == (641.0, 479.0)


def test_a_box_ending_on_the_edge_after_rounding_stays_inside_the_image() -> None:
    backend, _ = _backend("<box><333><777><1000><1000></box><|im_end|>")

    execution = backend.ground(_request(backend, image=_image(width=641, height=479)))

    (region,) = execution.regions
    assert region.bounding_box.x_max == pytest.approx(641.0)
    assert region.bounding_box.y_max == pytest.approx(479.0)


def test_a_two_coordinate_answer_is_a_point_never_a_box() -> None:
    parsed = parse_locateanything_response(
        "<ref>the door handle</ref><box><500><250></box><|im_end|>",
        image_width=640,
        image_height=480,
    )

    (output,) = parsed.outputs
    assert output.box is None
    assert output.point is not None
    assert (output.point.x, output.point.y) == (320.0, 120.0)


@pytest.mark.parametrize(
    ("span", "reason"),
    [
        ("<box><1><2><3></box>", GroundingRejectionReason.MALFORMED_GEOMETRY),
        ("<box><1><2><3><4><5></box>", GroundingRejectionReason.MALFORMED_GEOMETRY),
        ("<box><nan><1><2><3></box>", GroundingRejectionReason.MALFORMED_GEOMETRY),
        ("<box><inf><1><2><3></box>", GroundingRejectionReason.MALFORMED_GEOMETRY),
        ("<box><-1><1><2><3></box>", GroundingRejectionReason.MALFORMED_GEOMETRY),
        ("<box><1.5><1><2><3></box>", GroundingRejectionReason.MALFORMED_GEOMETRY),
        ("<box>somewhere</box>", GroundingRejectionReason.MALFORMED_GEOMETRY),
        ("<box><1001><0><5><5></box>", GroundingRejectionReason.COORDINATE_OUT_OF_RANGE),
        ("<box><0><0><5><2000></box>", GroundingRejectionReason.COORDINATE_OUT_OF_RANGE),
        ("<box><500><10><100><20></box>", GroundingRejectionReason.INVERTED_GEOMETRY),
        ("<box><10><90><20><10></box>", GroundingRejectionReason.INVERTED_GEOMETRY),
        ("<box><10><10><10><20></box>", GroundingRejectionReason.INVERTED_GEOMETRY),
        ("<box><10><20>", GroundingRejectionReason.MALFORMED_GEOMETRY),
        ("a chair is here", GroundingRejectionReason.UNRECOGNIZED_TEXT),
    ],
)
def test_malformed_out_of_range_and_inverted_geometry_is_rejected_explicitly(
    span: str, reason: GroundingRejectionReason
) -> None:
    parsed = parse_locateanything_response(
        f"<ref>chair</ref><box><0><0><10><10></box>{span}<box><20><20><30><30></box>",
        image_width=640,
        image_height=480,
    )

    (rejected,) = parsed.rejected_outputs
    assert rejected.reason is reason
    assert rejected.native_text == span
    assert rejected.output_index == 1
    assert [output.output_index for output in parsed.outputs] == [0, 2]


def test_text_after_the_end_of_response_token_is_rejected() -> None:
    parsed = parse_locateanything_response(
        "<box><0><0><10><10></box><|im_end|>trailing", image_width=640, image_height=480
    )

    (rejected,) = parsed.rejected_outputs
    assert rejected.reason is GroundingRejectionReason.UNRECOGNIZED_TEXT
    assert rejected.native_text == "trailing"


def test_a_response_without_an_end_token_is_reported_as_possibly_truncated() -> None:
    backend, _ = _backend("<ref>chair</ref><box><0><0><10><10></box>", max_new_tokens=64)

    execution = backend.ground(_request(backend))

    assert any("max_new_tokens=64" in warning for warning in execution.diagnostics.warnings)


# --- adapter ----------------------------------------------------------------------------


def test_the_adapter_satisfies_the_grounding_port_and_declares_its_policies() -> None:
    backend, _ = _backend()

    assert isinstance(backend, RegionGrounding)
    policies = {
        (policy.policy_id, policy.task, policy.geometry)
        for policy in backend.capabilities().query_policies
    }
    assert policies == {
        (CATEGORY_DETECTION_POLICY, GroundingTask.CATEGORY_DETECTION, GroundingGeometry.BOX),
        (PHRASE_GROUNDING_POLICY, GroundingTask.PHRASE_GROUNDING, GroundingGeometry.BOX),
        (POINTING_POLICY, GroundingTask.PHRASE_GROUNDING, GroundingGeometry.POINT),
    }


@pytest.mark.parametrize(
    ("query", "prompt"),
    [
        (
            _categories("table", "chair", "door"),
            "Locate all the instances that matches the following description: "
            "table</c>chair</c>door.",
        ),
        (
            _phrase("people wearing red shirts"),
            "Locate all the instances that match the following description: "
            "people wearing red shirts.",
        ),
        (_pointing("the traffic light"), "Point to: the traffic light."),
    ],
)
def test_queries_are_rendered_with_the_upstream_templates_in_query_order(
    query: GroundingQuery, prompt: str
) -> None:
    backend, runtime = _backend()

    execution = backend.ground(_request(backend, query))

    assert runtime.calls[0][1] == prompt
    assert execution.rendered_prompt == prompt


def test_unsupported_requested_geometry_fails_before_inference() -> None:
    backend, runtime = _backend()
    query = replace(_categories(), geometry=GroundingGeometry.POINT)

    with pytest.raises(GroundingRequestError):
        backend.ground(_request(backend, query))
    assert runtime.calls == []


def test_a_category_containing_the_upstream_separator_fails_before_inference() -> None:
    backend, runtime = _backend()

    with pytest.raises(GroundingRequestError, match="</c>"):
        backend.ground(_request(backend, _categories("chair</c>table")))
    assert runtime.calls == []


def test_a_request_fingerprinted_for_another_configuration_fails_before_inference() -> None:
    backend, runtime = _backend()
    request = replace(_request(backend), configuration_fingerprint="sha256:other")

    with pytest.raises(GroundingRequestError, match="fingerprint"):
        backend.ground(request)
    assert runtime.calls == []


def test_valid_boxes_become_region_evidence_with_exact_provenance() -> None:
    backend, _ = _backend("<ref>chair</ref><box><100><200><300><400></box><|im_end|>")

    execution = backend.ground(_request(backend))

    (region,) = execution.regions
    assert region.provenance == backend.backend_provenance()
    assert region.provenance.model == "nvidia/LocateAnything-3B"
    assert region.provenance.version == REVISION
    assert region.source_observation_id == SOURCE_ID
    assert execution.raw_response == "<ref>chair</ref><box><100><200><300><400></box><|im_end|>"


def test_point_only_output_is_not_misrepresented_as_a_region() -> None:
    backend, _ = _backend("<ref>the door handle</ref><box><500><250></box><|im_end|>")

    execution = backend.ground(_request(backend, _pointing()))

    assert execution.regions == ()
    assert len(execution.unsupported_outputs) == 1
    assert execution.outputs[0].point is not None


def test_a_point_answering_a_box_request_is_kept_as_a_point_and_reported() -> None:
    backend, _ = _backend("<ref>chair</ref><box><500><250></box><|im_end|>")

    execution = backend.ground(_request(backend))

    assert execution.regions == ()
    assert len(execution.unsupported_outputs) == 1
    assert any("point" in warning for warning in execution.diagnostics.warnings)


def test_query_and_category_order_are_preserved_and_labels_are_not_claims() -> None:
    backend, _ = _backend(
        "<ref>door</ref><box><0><0><10><10></box><ref>table</ref><box>none</box>"
        "<ref>chair</ref><box><20><20><30><30></box><|im_end|>"
    )

    execution = backend.ground(_request(backend, _categories("table", "chair", "door")))

    assert execution.request.query.categories == ("table", "chair", "door")
    assert [output.label for output in execution.outputs] == ["door", "chair"]
    assert execution.no_match_labels == ("table",)
    assert all(region.region_kind is None for region in execution.regions)


def test_no_object_is_an_explicit_no_match_result() -> None:
    backend, _ = _backend("<ref>chair</ref><box>none</box><|im_end|>")

    execution = backend.ground(_request(backend, _categories("chair")))

    assert execution.no_match
    assert execution.regions == ()
    assert execution.no_match_labels == ("chair",)


def test_decoder_statistics_are_kept_as_raw_native_diagnostics_never_as_confidence() -> None:
    backend, _ = _backend(
        LocateAnythingGeneration(
            text="<ref>chair</ref><box><0><0><10><10></box><|im_end|>",
            statistics=STATS,
            peak_memory_bytes=1234,
        )
    )

    execution = backend.ground(_request(backend))

    native = dict(execution.diagnostics.native)
    assert native["stats.raw"] == STATS
    assert native["stats.switch_to_ar"] == 1
    assert native["stats.num_tokens"] == 18
    assert native["stats.generate_time(s)"] == 0.5
    assert execution.diagnostics.peak_memory_bytes == 1234
    assert not any("confidence" in name for name in native)


def test_missing_decoder_statistics_stay_missing() -> None:
    backend, _ = _backend("<box><0><0><10><10></box><|im_end|>")

    execution = backend.ground(_request(backend))

    assert execution.diagnostics.native == ()
    assert execution.outputs[0].native_diagnostics == ()


def test_each_output_records_which_decoder_produced_it() -> None:
    history = (
        ("mtp", "<ref>chair</ref>"),
        ("mtp", "<box><0><0><10><10></box>"),
        ("mtp", "<box><20><20>"),
        ("ar", "<30>"),
        ("ar", "<30>"),
        ("ar", "</box>"),
        ("mtp", "<|im_end|>"),
    )
    backend, _ = _backend(
        LocateAnythingGeneration(text="".join(text for _, text in history), decode_steps=history)
    )

    execution = backend.ground(_request(backend))

    assert [dict(output.native_diagnostics)["decoder"] for output in execution.outputs] == [
        "mtp",
        "mtp+ar",
    ]


def test_a_decode_history_that_does_not_rebuild_the_answer_is_reported_not_used() -> None:
    backend, _ = _backend(
        LocateAnythingGeneration(
            text="<box><0><0><10><10></box><|im_end|>", decode_steps=(("mtp", "<box>"),)
        )
    )

    execution = backend.ground(_request(backend))

    assert execution.outputs[0].native_diagnostics == ()
    assert any("decode history" in warning for warning in execution.diagnostics.warnings)


def test_runtime_identity_and_effective_configuration_are_recorded() -> None:
    backend, _ = _backend(
        LocateAnythingGeneration(
            text="<|im_end|>", runtime_identity=(("transformers", "4.57.1"), ("torch", "2.8.0"))
        )
    )

    execution = backend.ground(_request(backend))

    assert dict(execution.runtime_identity) == {"transformers": "4.57.1", "torch": "2.8.0"}
    assert execution.effective_configuration["revision"] == REVISION
    assert execution.effective_configuration["generation_mode"] == "hybrid"


def test_the_same_request_in_another_run_gets_distinct_local_evidence() -> None:
    answer = "<box><0><0><10><10></box><|im_end|>"
    backend, _ = _backend(answer)
    other = perception_result_id_for(
        run_id=PerceptionRunId("run-0002"), source_observation_id=SOURCE_ID
    )

    first = backend.ground(_request(backend))
    second = backend.ground(_request(backend, result_id=other))

    assert first.request_id == second.request_id
    assert first.regions[0].region_id != second.regions[0].region_id


def test_raw_output_and_parsed_evidence_survive_the_run_artifact(tmp_path: Path) -> None:
    backend, _ = _backend(
        LocateAnythingGeneration(
            text="<ref>chair</ref><box><100><200><300><400></box><box><500><250></box>"
            "<box><9></box><|im_end|>",
            statistics=STATS,
        )
    )
    execution = backend.ground(_request(backend))
    writer = PerceptionRunWriter(
        output_dir=tmp_path / "run",
        sequence_name="corridor-02",
        run_id=RUN_ID,
        run_index=1,
        sequence_artifact_id="sequence-0001",
        selection_id="sha256:aaaa",
        enabled_capabilities=frozenset({"region_grounding"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:test",
    )
    result = PerceptionResult(
        result_id=RESULT_ID,
        source_observation_id=SOURCE_ID,
        run_id=RUN_ID,
        sequence_artifact_id="sequence-0001",
        created_at="2026-01-01T00:00:00+00:00",
    )
    writer.add_result(with_grounded_regions(result, (execution,)))
    writer.add_region_grounding(execution)
    writer.finalize()

    reader = PerceptionRunReader(tmp_path / "run")
    (reopened,) = reader.list_region_groundings()

    assert reopened == execution
    assert reader.result(SOURCE_ID).regions == execution.regions
    assert len(reopened.rejected_outputs) == 1
    assert len(reopened.unsupported_outputs) == 1


# --- configuration identity ---------------------------------------------------------------


def test_the_revision_must_be_an_immutable_commit() -> None:
    with pytest.raises(ValueError, match="revision"):
        _config(revision="main")


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_new_tokens": 0},
        {"temperature": -0.1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"top_k": -1},
        {"repetition_penalty": 0.0},
        {"model": " "},
    ],
)
def test_generation_settings_are_validated_before_any_runtime(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "nvidia/LocateAnything-3B-ft"},
        {"revision": "f" * 40},
        {"device": "cuda:1"},
        {"dtype": "float16"},
        {"generation_mode": LocateAnythingGenerationMode.SLOW},
        {"max_new_tokens": 8192},
        {"temperature": 0.7},
        {"top_p": 0.8},
        {"top_k": 5},
        {"repetition_penalty": 1.0},
    ],
)
def test_any_inference_setting_changes_the_configuration_fingerprint(
    overrides: dict[str, Any],
) -> None:
    assert _config(**overrides).fingerprint != _config().fingerprint

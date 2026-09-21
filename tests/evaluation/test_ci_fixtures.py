"""CI fixture subset: reproducibility, integrity and cross-module regression tests.

The subset is generated from formulas, committed for review, and regenerated
here to prove the committed files are exactly what the generator produces. The
regression tests run the implemented capabilities on it with no GPU, network or
model.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contextmap.evaluation.annotations import (
    RegionAnnotationSet,
    SemanticAnnotationSet,
    ground_truth_regions,
    read_annotation_set,
)
from contextmap.evaluation.ci_fixtures import (
    CATALOGUE_FILENAME,
    CI_FIXTURE_ID,
    CI_FIXTURE_VERSION,
    CoverageStatus,
    FixtureCatalogueError,
    build_synthetic_sequence,
    decode_catalogue,
    encode_catalogue,
    generate_ci_fixture_subset,
    read_catalogue,
)
from contextmap.evaluation.reference_integrity import open_validated_reference_set
from contextmap.evaluation.reference_set import ReferenceSampleId
from contextmap.evaluation.semantic_interpretation import (
    SemanticEvaluationContext,
    SemanticEvaluationInput,
    evaluate_semantic_interpretation,
)
from contextmap.ingestion import (
    FrameId,
    ImageObservation,
    LidarObservation,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SourceObservationId,
)
from contextmap.sensor_association import camera_projection_for
from contextmap.shared import compose_rigid, invert_rigid, rotate_vector
from contextmap.state_estimation import StateEstimationRequest, TrajectoryId
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
)
from contextmap.visual_perception import (
    BackendProvenance,
    PerceptionResultId,
    RegionId,
    SemanticBackendDiagnostics,
    SemanticConfidencePolicy,
    SemanticInferenceProvenance,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticPromptTemplate,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
    parse_semantic_response,
    render_semantic_prompt,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ci_subset" / CI_FIXTURE_VERSION


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_the_committed_subset_is_exactly_what_the_generator_produces(tmp_path: Path) -> None:
    regenerated = tmp_path / CI_FIXTURE_VERSION

    generate_ci_fixture_subset(regenerated)

    assert _files(regenerated) == _files(FIXTURE_ROOT)


def test_generation_is_immutable(tmp_path: Path) -> None:
    generate_ci_fixture_subset(tmp_path)

    with pytest.raises(FileExistsError):
        generate_ci_fixture_subset(tmp_path)


def test_the_subset_is_versioned_by_its_directory_and_its_documents() -> None:
    validated = open_validated_reference_set(FIXTURE_ROOT)
    catalogue = read_catalogue(FIXTURE_ROOT)

    assert FIXTURE_ROOT.name == CI_FIXTURE_VERSION
    assert validated.manifest.reference_set_id == CI_FIXTURE_ID
    assert validated.manifest.version == catalogue.version == CI_FIXTURE_VERSION
    assert catalogue.reference_set == validated.manifest.identity()


def test_the_subset_reference_set_passes_integrity_validation_with_no_finding() -> None:
    validated = open_validated_reference_set(FIXTURE_ROOT)

    assert validated.report.is_valid
    assert validated.report.files_checked
    assert validated.report.findings == ()
    sources = validated.manifest.sources
    assert all(source.redistributable and source.license for source in sources)
    assert all(
        entry.trust.value == "trusted_ground_truth" for entry in validated.manifest.annotations
    )


def test_every_case_has_identity_provenance_license_hash_and_edge_cases() -> None:
    catalogue = read_catalogue(FIXTURE_ROOT)

    assert len({case.fixture_id for case in catalogue.cases}) == len(catalogue.cases)
    for case in catalogue.cases:
        assert case.content_hash == case.compute_hash()
        assert case.generator and case.license and case.redistributable
        assert case.edge_cases, case.fixture_id
        assert case.expected, case.fixture_id
    assert catalogue.digest().startswith("sha256:")


def test_coverage_is_explicit_about_what_the_subset_does_not_cover() -> None:
    catalogue = read_catalogue(FIXTURE_ROOT)
    by_requirement = {entry.requirement: entry for entry in catalogue.coverage}

    assert set(by_requirement) == {
        "canonical RGB/LiDAR/pose/calibration ingestion",
        "known 3D transforms/projections",
        "2D masks/regions and overlapping regions",
        "semantic alternatives/abstention/unscored claims",
        "repeated inference over one physical observation",
        "multi-view fusion",
        "entity duplicate/distinct cases",
        "spatial relation fixtures",
        "final ContextMapArtifact round-trip",
    }
    for entry in catalogue.coverage:
        if entry.status is CoverageStatus.COVERED:
            assert entry.fixture_ids
        else:
            assert entry.note, entry.requirement
    assert by_requirement["multi-view fusion"].status is CoverageStatus.NOT_YET_AVAILABLE
    assert (
        by_requirement["final ContextMapArtifact round-trip"].status
        is CoverageStatus.NOT_YET_AVAILABLE
    )
    assert by_requirement["entity duplicate/distinct cases"].status is (
        CoverageStatus.ANNOTATION_LEVEL
    )


def test_a_modified_case_or_catalogue_is_detected() -> None:
    document = json.loads((FIXTURE_ROOT / CATALOGUE_FILENAME).read_text(encoding="utf-8"))
    assert decode_catalogue(document) == read_catalogue(FIXTURE_ROOT)

    tampered = json.loads(json.dumps(document))
    tampered["cases"][0]["expected"]["observation_counts"]["image"] = 99
    with pytest.raises(FixtureCatalogueError, match="content hash"):
        decode_catalogue(tampered)

    renamed = json.loads(json.dumps(document))
    renamed["version"] = "9.9.9"
    with pytest.raises(FixtureCatalogueError, match="digest"):
        decode_catalogue(renamed)

    unknown = json.loads(json.dumps(document))
    unknown["coverage"][0]["fixture_ids"] = ["no-such-fixture"]
    with pytest.raises(FixtureCatalogueError, match="unknown fixture"):
        decode_catalogue(unknown)

    assert encode_catalogue(read_catalogue(FIXTURE_ROOT))["digest"] == document["digest"]


def _case(fixture_id: str) -> Any:
    return read_catalogue(FIXTURE_ROOT).case(fixture_id)


def test_ingestion_round_trips_the_synthetic_sequence(tmp_path: Path) -> None:
    case = _case("ingestion-canonical-sequence")
    sequence = build_synthetic_sequence()

    with SequenceArtifactWriter(
        workspace_root=tmp_path,
        sequence_name=sequence.name,
        artifact_id=SequenceArtifactId("ci-fixture"),
    ) as writer:
        writer.set_calibration(sequence.calibration)
        for observation in sequence.observations:
            writer.add_observation(observation)
        writer.finalize()
    reader = SequenceArtifactReader(tmp_path / "sequences" / sequence.name / "ci-fixture")
    restored = reader.list_observations()

    assert reader.verify_integrity() == []
    for modality, count in case.expected["observation_counts"].items():
        assert reader.manifest.observation_counts[modality] == count
    assert [str(item.observation_id) for item in restored] == case.expected["observation_ids"]
    assert restored == list(sequence.observations)
    calibration = reader.read_calibration()
    assert calibration is not None
    assert calibration.static_transforms == sequence.calibration.static_transforms
    assert calibration.entries.keys() == sequence.calibration.entries.keys()
    for key, entry in sequence.calibration.entries.items():
        restored_entry = calibration.entries[key]
        assert (restored_entry.sensor_id, restored_entry.frame_id) == (
            entry.sensor_id,
            entry.frame_id,
        )
        assert restored_entry.camera_model == entry.camera_model
        assert restored_entry.content_hash == entry.content_hash
    assert sum(isinstance(item, ImageObservation) for item in restored) == 3
    assert sum(isinstance(item, LidarObservation) for item in restored) == 3


def _camera_frame_points(index: int) -> np.ndarray[Any, Any]:
    """Compose the calibration extrinsics through the shared rigid-transform primitives."""
    optical = next(
        item
        for item in build_synthetic_sequence().calibration.static_transforms
        if item.child_frame == FrameId("front_camera_optical")
    )
    pose = tuple(_case("trajectory-external-pose").expected["translations_m"][index])
    world_from_camera = compose_rigid(
        outer_translation=pose,
        outer_rotation=(0.0, 0.0, 0.0, 1.0),
        inner_translation=optical.translation,
        inner_rotation=optical.rotation,
    )
    camera_from_world = invert_rigid(
        translation=world_from_camera[0], rotation=world_from_camera[1]
    )
    landmarks = _case("projection-known-landmarks").inputs["landmarks"].values()
    return np.array(
        [
            np.array(rotate_vector(camera_from_world[1], tuple(point))) + camera_from_world[0]
            for point in landmarks
        ],
        dtype=float,
    )


def test_calibrated_projection_matches_the_analytic_expectation() -> None:
    case = _case("projection-known-landmarks")
    tolerance = case.tolerances["pixel_abs"]
    entry = next(
        item
        for item in build_synthetic_sequence().calibration.entries.values()
        if item.camera_model is not None
    )
    projection = camera_projection_for(entry)
    names = list(case.inputs["landmarks"])

    for index, frame_second in enumerate(case.inputs["frame_times_s"]):
        projected = projection.project(_camera_frame_points(index))
        for row, name in enumerate(names):
            expected = case.expected["projections"][f"{name}@{index}"]
            if expected["status"] == "behind_camera":
                assert projected.depth_m[row] <= 0.0, (name, frame_second)
                continue
            assert projected.projectable[row], (name, frame_second)
            np.testing.assert_allclose(projected.pixels[row], expected["pixel"], atol=tolerance)
            inside = bool(projection.in_image(projected.pixels[row : row + 1])[0])
            assert inside is (expected["status"] == "visible"), (name, frame_second)


def test_external_poses_become_the_recorded_trajectory() -> None:
    case = _case("trajectory-external-pose")
    poses = [
        item
        for item in build_synthetic_sequence().observations
        if str(item.observation_id).startswith("odom-")
    ]
    request = StateEstimationRequest(
        trajectory_id=TrajectoryId("ci-fixture--trajectory"),
        sequence_artifact_id=SequenceArtifactId("ci-fixture"),
        selection_id="full-sequence",
        observations=tuple(poses),
        calibration=None,
    )

    trajectory = (
        ExternalPoseEstimator(
            ExternalPoseConfig(reference_frame=FrameId("odom"), body_frame=FrameId("base_link"))
        )
        .estimate(request)
        .trajectory
    )

    assert [list(pose.translation_m) for pose in trajectory.poses] == case.expected[
        "translations_m"
    ]
    assert [
        str(pose.provenance.source_observation_ids[0]) for pose in trajectory.poses
    ] == case.expected["pose_observation_ids"]


def test_overlapping_masks_in_the_annotation_file_match_the_catalogue() -> None:
    case = _case("regions-overlapping-masks")
    annotations = read_annotation_set(FIXTURE_ROOT / "annotations" / "regions.json")
    assert isinstance(annotations, RegionAnnotationSet)

    first, second = (region.mask for region in ground_truth_regions(annotations.frames[0]))
    intersection = sum(a and b for a, b in zip(first.data, second.data, strict=True))
    union = sum(a or b for a, b in zip(first.data, second.data, strict=True))

    assert (first.area, second.area) == (case.expected["area_a"], case.expected["area_b"])
    assert (intersection, union) == (case.expected["intersection"], case.expected["union"])
    assert intersection / union == pytest.approx(
        case.expected["iou"], abs=case.tolerances["iou_abs"]
    )
    assert annotations.frames[0].exclusion_areas and annotations.frames[0].valid_areas


def _request(observation_id: str) -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("region-request-0001"),
        source_observation_id=SourceObservationId(observation_id),
        perception_result_id=PerceptionResultId("result-0001"),
        mode=SemanticInterpretationMode.REGION,
        visual_views=(
            SemanticVisualView(
                view_id="crop",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference="outputs/semantic-views/crop.jpg",
                source_observation_id=SourceObservationId(observation_id),
                region_id=RegionId("region-a"),
                sha256="0" * 64,
            ),
        ),
        region_id=RegionId("region-a"),
        prompt_template_id="region/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint="sha256:config",
    )


def _provenance() -> SemanticInferenceProvenance:
    return SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="canned_semantic",
            capability="semantic_interpreter",
            provider="fixture",
            model="none",
            version="1",
            configuration_fingerprint="sha256:config",
        ),
        task_identity="canned-region",
        prompt_template_id="region/v1",
        output_schema_version="semantic-response/1",
    )


def _execute(raw: dict[str, Any], observation_id: str, policy: SemanticConfidencePolicy) -> Any:
    request = _request(observation_id)
    rendered = render_semantic_prompt(
        request, SemanticPromptTemplate.default_for(request.mode), confidence_policy=policy
    )
    text = json.dumps(raw)
    return SemanticInterpretationExecution(
        request=request,
        rendered_prompt=rendered,
        raw_response=text,
        parsed=parse_semantic_response(text, request, _provenance(), confidence_policy=policy),
        diagnostics=SemanticBackendDiagnostics(latency_ms=1, input_tokens=1, output_tokens=1),
        effective_configuration={"model": "none"},
    )


def test_canned_model_responses_parse_into_the_documented_claims() -> None:
    case = _case("semantic-claims-variants")
    observation = case.inputs["observation_id"]
    policies = {
        "primary_with_alternative": SemanticConfidencePolicy.UNSCORED_ONLY,
        "abstention": SemanticConfidencePolicy.UNSCORED_ONLY,
        "measured": SemanticConfidencePolicy.MEASURED,
        "repeated_inference": SemanticConfidencePolicy.UNSCORED_ONLY,
    }
    executions = {
        name: _execute(raw, observation, policies[name])
        for name, raw in case.inputs["responses"].items()
    }

    for name, execution in executions.items():
        expected = case.expected[name]
        assert execution.parsed.abstained is expected["abstained"], name
        assert [claim.role.value for claim in execution.parsed.claims] == expected["roles"], name
        confidences = [claim.confidence for claim in execution.parsed.claims]
        assert confidences == pytest.approx(
            expected["confidences"], abs=case.tolerances["confidence_abs"]
        ), name
    inferences = [executions["primary_with_alternative"], executions["repeated_inference"]]
    assert len(inferences) == case.expected["inference_count"]
    assert (
        len({str(item.request.source_observation_id) for item in inferences})
        == case.expected["physical_observation_count"]
    )


def test_the_semantic_annotations_grade_the_canned_responses() -> None:
    annotations = read_annotation_set(FIXTURE_ROOT / "annotations" / "semantics.json")
    assert isinstance(annotations, SemanticAnnotationSet)
    case = _case("semantic-claims-variants")
    labeled = annotations.find(
        ReferenceSampleId("sample-0000"), SourceObservationId("frame-0000"), "region-a"
    )
    assert labeled is not None
    execution = _execute(
        case.inputs["responses"]["primary_with_alternative"],
        case.inputs["observation_id"],
        SemanticConfidencePolicy.UNSCORED_ONLY,
    )

    report = evaluate_semantic_interpretation(
        context=SemanticEvaluationContext(
            evaluation_id="ci-fixture-eval",
            reference_set_version=CI_FIXTURE_VERSION,
            selection_id="ci",
            perception_run_id="canned-run",
            artifact_id="canned-artifact",
            pipeline_configuration_digest="sha256:pipeline",
            evaluator_version="semantic-evaluator/1",
        ),
        inputs=(
            SemanticEvaluationInput(
                execution=execution,
                evidence_variant_id="tight-crop",
                annotation=annotations.to_semantic_annotation(labeled),
            ),
        ),
    )

    sample = report.samples[0]
    assert sample.acceptable_claim_count == 1
    assert sample.alternative_claim_count == 1
    assert sample.unsupported_claim_count == 1
    assert not sample.abstained


def test_a_replaced_case_would_change_the_catalogue_digest() -> None:
    catalogue = read_catalogue(FIXTURE_ROOT)
    other = replace(catalogue, version="1.0.1")

    assert other.digest() != catalogue.digest()

import json
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from hashlib import sha256
from pathlib import Path

import pytest

from contextmap.evaluation import (
    EvaluatedDiscoveryFrame,
    EvaluationRunDescriptor,
    GroundTruthRegion,
    ReferenceFrame,
    RegionDiscoveryEvaluator,
    RegionDiscoveryReferenceSet,
    compare_region_discovery_reports,
    write_region_discovery_reference_set,
    write_region_discovery_report,
)
from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BackendDiagnostics,
    BackendProvenance,
    BoundingBox,
    DiscoveryPass,
    DiscoveryRunResult,
    InlineMask,
    PassKind,
    PreparedImage,
    RegionCandidate,
    RegionProvenance,
    normalize_regions,
)


def _mask(width: int, height: int, box: BoundingBox) -> InlineMask:
    return InlineMask(
        width=width,
        height=height,
        data=tuple(
            box.x_min <= x < box.x_max and box.y_min <= y < box.y_max
            for y in range(height)
            for x in range(width)
        ),
    )


def _frame(frame_id: str, condition: str, *, annotated: bool) -> ReferenceFrame:
    width, height = 6, 4
    box = BoundingBox(1, 1, 4, 3)
    prepared = PreparedImage(
        source_observation_id=SourceObservationId(frame_id),
        payload_reference=f"reference/{frame_id}.png",
        payload_artifact=ArtifactReference(
            uri=f"reference/{frame_id}.png",
            sha256=sha256(frame_id.encode()).hexdigest(),
            media_type="image/png",
        ),
        width=width,
        height=height,
        transformations=(),
    )
    annotations = (
        (GroundTruthRegion(region_id="gt-1", mask=_mask(width, height, box)),) if annotated else ()
    )
    return ReferenceFrame(
        frame_id=frame_id,
        source_condition=condition,
        prepared_image=prepared,
        annotations=annotations,
    )


def _evaluated(
    frame: ReferenceFrame, backend_id: str, duration_ms: float
) -> EvaluatedDiscoveryFrame:
    box = BoundingBox(1, 1, 4, 3)
    candidate = RegionCandidate(
        candidate_id=f"{backend_id}-proposal",
        source_observation_id=frame.prepared_image.source_observation_id,
        perception_run_id=f"run-{backend_id}",
        perception_result_id=f"result-{frame.frame_id}",
        image_width=frame.prepared_image.width,
        image_height=frame.prepared_image.height,
        bounding_box=box,
        mask=_mask(frame.prepared_image.width, frame.prepared_image.height, box),
        provenance=RegionProvenance(
            backend_id=backend_id,
            backend_version="1",
            checkpoint=f"{backend_id}-checkpoint",
            config_digest=f"sha256:{backend_id}",
            discovery_pass_id="full-frame",
            native_proposal_id="proposal-1",
        ),
    )
    discovery = DiscoveryRunResult(
        candidates=(candidate,),
        rejected=(),
        passes=(
            DiscoveryPass(
                pass_id="full-frame",
                kind=PassKind.FULL_FRAME,
                window=BoundingBox(0, 0, frame.prepared_image.width, frame.prepared_image.height),
            ),
        ),
        diagnostics=(
            BackendDiagnostics(
                duration_ms=duration_ms,
                proposal_count=1,
                warnings=(),
                metadata=(("peak_memory_mb", 256.0),),
            ),
        ),
    )
    return EvaluatedDiscoveryFrame(
        discovery=discovery,
        normalization=normalize_regions(
            discovery.candidates,
            frame.prepared_image,
            BackendProvenance(
                backend_id=backend_id,
                capability="region_discovery",
                provider="test-provider",
                model=f"{backend_id}-checkpoint",
                version="1",
                configuration_fingerprint=f"sha256:{backend_id}",
            ),
        ),
    )


def _descriptor(
    backend_id: str, *, tiling: bool = False, duration_tag: str = "baseline"
) -> EvaluationRunDescriptor:
    return EvaluationRunDescriptor(
        perception_run_id=f"run-{backend_id}-{duration_tag}",
        perception_artifact_id=f"artifact-{backend_id}-{duration_tag}",
        backend_id=backend_id,
        backend_version="1",
        checkpoint=f"{backend_id}-checkpoint",
        config_digest=f"sha256:{backend_id}-controlled-config",
        pipeline_graph_digest="sha256:pipeline-v1",
        strategy="automatic",
        thresholds=(("score", 0.5),),
        variables=(("tiling", tiling), ("query_strategy", "automatic")),
        execution_kind="ci_contract",
    )


def _change_checkpoint(descriptor: EvaluationRunDescriptor) -> EvaluationRunDescriptor:
    return replace(descriptor, checkpoint="sam3-other-checkpoint")


def _change_strategy(descriptor: EvaluationRunDescriptor) -> EvaluationRunDescriptor:
    return replace(descriptor, strategy="text_prompt")


def _change_threshold(descriptor: EvaluationRunDescriptor) -> EvaluationRunDescriptor:
    return replace(descriptor, thresholds=(("score", 0.9),))


def _change_pipeline_graph(descriptor: EvaluationRunDescriptor) -> EvaluationRunDescriptor:
    return replace(descriptor, pipeline_graph_digest="sha256:pipeline-v2")


def _change_config_digest(descriptor: EvaluationRunDescriptor) -> EvaluationRunDescriptor:
    return replace(descriptor, config_digest="sha256:hidden-config-change")


def test_same_reference_selection_uses_one_report_schema_for_all_backends() -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1",
        frames=(
            _frame("frame-pinhole", "indoor_pinhole", annotated=True),
            _frame("frame-wide", "outdoor_wide_angle", annotated=False),
        ),
    )
    evaluator = RegionDiscoveryEvaluator(match_iou_threshold=0.5)

    reports = {
        backend: evaluator.evaluate(
            reference_set,
            _descriptor(backend),
            partial(_evaluated, backend_id=backend, duration_ms=5.0),
        )
        for backend in ("sam2", "sam3", "florence2_region_discovery")
    }

    assert {report.schema for report in reports.values()} == {
        "contextmap.region-discovery-evaluation/v1"
    }
    assert {report.frame_selection for report in reports.values()} == {
        ("frame-pinhole", "frame-wide")
    }
    sam3 = reports["sam3"]
    assert sam3.frames[0].accuracy is not None
    assert sam3.frames[0].accuracy.mean_iou == 1.0
    assert sam3.frames[0].accuracy.mean_dice == 1.0
    assert sam3.frames[0].accuracy.region_recall == 1.0
    assert sam3.frames[1].accuracy is None
    assert sam3.frames[1].diagnostics.raw_candidate_count == 1
    assert sam3.frames[1].performance.peak_memory_mb == 256.0


def test_report_serialization_is_deterministic_and_immutable(tmp_path: Path) -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    report = RegionDiscoveryEvaluator().evaluate(
        reference_set,
        _descriptor("sam3"),
        lambda frame: _evaluated(frame, "sam3", 4.0),
    )
    output = tmp_path / "sam3-report.json"
    manifest = tmp_path / "reference-set.json"

    write_region_discovery_reference_set(manifest, reference_set)
    write_region_discovery_report(output, report)

    serialized = output.read_text()
    assert json.loads(serialized)["metric_schema_version"] == "1.0.0"
    assert json.loads(manifest.read_text())["frames"][0]["source_condition"] == "synthetic"
    with pytest.raises(FileExistsError, match="already exists"):
        write_region_discovery_report(output, report)
    assert output.read_text() == serialized


def test_ablation_comparison_separates_quality_and_performance() -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    evaluator = RegionDiscoveryEvaluator()
    baseline = evaluator.evaluate(
        reference_set,
        _descriptor("sam3", tiling=False, duration_tag="baseline"),
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )
    changed = evaluator.evaluate(
        reference_set,
        _descriptor("sam3", tiling=True, duration_tag="tiling"),
        lambda frame: _evaluated(frame, "sam3", 10.0),
    )

    comparison = compare_region_discovery_reports(baseline, changed, changed_variable="tiling")

    assert comparison.quality_deltas["mean_iou"] == 0.0
    assert comparison.performance_deltas["mean_runtime_ms"] == 2.0
    assert comparison.changed_variable == "tiling"


def test_ablation_rejects_uncontrolled_variable_changes() -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    evaluator = RegionDiscoveryEvaluator()
    baseline = evaluator.evaluate(
        reference_set,
        _descriptor("sam3", tiling=False, duration_tag="baseline"),
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )
    changed_descriptor = replace(
        _descriptor("sam3", tiling=True, duration_tag="changed"),
        variables=(("tiling", True), ("query_strategy", "text_prompt")),
    )
    changed = evaluator.evaluate(
        reference_set,
        changed_descriptor,
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )

    with pytest.raises(ValueError, match="exactly the declared variable"):
        compare_region_discovery_reports(baseline, changed, changed_variable="tiling")


@pytest.mark.parametrize(
    "mutate",
    [
        _change_checkpoint,
        _change_strategy,
        _change_threshold,
        _change_pipeline_graph,
        _change_config_digest,
    ],
)
def test_ablation_rejects_unreported_run_descriptor_changes(
    mutate: Callable[[EvaluationRunDescriptor], EvaluationRunDescriptor],
) -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    evaluator = RegionDiscoveryEvaluator()
    baseline = evaluator.evaluate(
        reference_set,
        _descriptor("sam3", tiling=False, duration_tag="baseline"),
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )
    changed_descriptor = mutate(_descriptor("sam3", tiling=True, duration_tag="changed"))
    changed = evaluator.evaluate(
        reference_set,
        changed_descriptor,
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )

    with pytest.raises(ValueError, match="uncontrolled run fields"):
        compare_region_discovery_reports(baseline, changed, changed_variable="tiling")


def test_versioned_ci_reference_and_sam3_contract_baseline_are_persisted() -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "region_discovery"
    reference = json.loads((fixture_root / "reference-set-v1.json").read_text())
    baseline = json.loads((fixture_root / "sam3-ci-contract-baseline-v1.json").read_text())

    assert reference["schema"] == "contextmap.region-discovery-reference-set/v1"
    assert len({frame["source_condition"] for frame in reference["frames"]}) == 2
    assert baseline["schema"] == "contextmap.region-discovery-evaluation/v1"
    assert baseline["frame_selection"] == [frame["frame_id"] for frame in reference["frames"]]
    assert baseline["run"]["backend_id"] == "sam3"
    assert baseline["run"]["execution_kind"] == "ci_contract"
    assert baseline["run"]["checkpoint"].startswith("test://")

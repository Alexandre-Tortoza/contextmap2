"""End-to-end core contract test: fixture -> service -> run artifact -> evidence view.

canonical observation
    -> PreparedImage
    -> VisualPerceptionService (fake region discovery, dense + region feature
       extraction, scene + region semantic interpretation)
    -> PerceptionResult
    -> PerceptionRunArtifact
    -> isolated PerceptionRunReader
    -> multi-run PerceptionEvidenceSet

No GPU, model download, network API, or ROS dependency is used anywhere
in this file.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from fakes import (
    FakeDenseFeatureExtractor,
    FakeFailingRegionDiscovery,
    FakeRegionDiscovery,
    FakeRegionFeatureExtractor,
    FakeSemanticInterpreter,
)

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    PerceptionEvidenceSet,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    PreparedImage,
    Region2D,
    RegionDiscovery,
    StageDefinition,
    StageOutcome,
    StageStatus,
    assemble_perception_result,
    execute_stage_graph,
)


def _prepared_image(observation_id: str) -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId(observation_id),
        payload_reference=f"debug/{observation_id}/prepared.jpg",
        width=640,
        height=480,
    )


def _build_canonical_stage_graph(
    image: PreparedImage,
    *,
    result_id: PerceptionResultId,
    region_discovery: RegionDiscovery | None = None,
) -> list[StageDefinition]:
    discovery: RegionDiscovery = region_discovery or FakeRegionDiscovery()
    dense_extractor = FakeDenseFeatureExtractor()
    region_extractor = FakeRegionFeatureExtractor()
    interpreter = FakeSemanticInterpreter(result_id)

    def run_region_discovery(ctx: Mapping[str, object]) -> Sequence[Region2D]:
        return discovery.discover(image)

    def run_dense_features(ctx: Mapping[str, object]) -> Sequence[object]:
        return dense_extractor.extract(image)

    def run_region_features(ctx: Mapping[str, object]) -> Sequence[object]:
        regions = ctx["region_discovery"]
        assert isinstance(regions, Sequence)
        return region_extractor.extract(image, regions=regions)  # type: ignore[arg-type]

    def run_scene_interpretation(ctx: Mapping[str, object]) -> object:
        return interpreter.interpret_scene(image)

    def run_region_semantic_interpretation(ctx: Mapping[str, object]) -> Sequence[object]:
        regions = ctx["region_discovery"]
        assert isinstance(regions, Sequence)
        return interpreter.interpret_regions(image, regions)  # type: ignore[arg-type]

    return [
        StageDefinition(
            stage_id="image_preparation", capability="image_preparation", run=lambda ctx: image
        ),
        StageDefinition(
            stage_id="region_discovery",
            capability="region_discovery",
            depends_on=frozenset({"image_preparation"}),
            run=run_region_discovery,
        ),
        StageDefinition(
            stage_id="dense_feature_extraction",
            capability="feature_extractor",
            depends_on=frozenset({"image_preparation"}),
            run=run_dense_features,
        ),
        StageDefinition(
            stage_id="region_feature_extraction",
            capability="feature_extractor",
            depends_on=frozenset({"region_discovery"}),
            run=run_region_features,
        ),
        StageDefinition(
            stage_id="scene_interpretation",
            capability="semantic_interpreter",
            depends_on=frozenset({"image_preparation"}),
            run=run_scene_interpretation,
        ),
        StageDefinition(
            stage_id="region_semantic_interpretation",
            capability="semantic_interpreter",
            depends_on=frozenset({"region_discovery"}),
            run=run_region_semantic_interpretation,
        ),
    ]


def _process_observation(
    observation_id: str, run_id: str, *, region_discovery: RegionDiscovery | None = None
) -> tuple[Sequence[StageOutcome], PerceptionResult]:
    image = _prepared_image(observation_id)
    result_id = PerceptionResultId(f"{run_id}--{observation_id}")
    outcomes = execute_stage_graph(
        _build_canonical_stage_graph(image, result_id=result_id, region_discovery=region_discovery)
    )
    result = assemble_perception_result(
        result_id=result_id,
        source_observation_id=SourceObservationId(observation_id),
        run_id=PerceptionRunId(run_id),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
        outcomes=outcomes,
        region_stage_id="region_discovery",
        feature_stage_ids=("dense_feature_extraction", "region_feature_extraction"),
        claim_stage_ids=("region_semantic_interpretation",),
        scene_context_stage_id="scene_interpretation",
    )
    return outcomes, result


def test_full_pipeline_produces_regions_features_claims_and_scene_context() -> None:
    _outcomes, result = _process_observation("frame-0124", "run-0001")

    assert len(result.regions) == 1
    assert len(result.features) == 2  # one dense + one region feature
    assert len(result.claims) == 1
    assert result.scene_context is not None
    assert result.scene_context.claims[0].hypothesis == "an indoor corridor"


def test_repeated_processing_of_same_frame_yields_distinct_results_same_observation() -> None:
    _outcomes_a, result_a = _process_observation("frame-0124", "run-0001")
    _outcomes_b, result_b = _process_observation("frame-0124", "run-0002")

    assert result_a.result_id != result_b.result_id
    assert result_a.source_observation_id == result_b.source_observation_id == "frame-0124"


def test_failing_region_discovery_still_preserves_independent_dense_feature_evidence() -> None:
    outcomes, result = _process_observation(
        "frame-0124", "run-0001", region_discovery=FakeFailingRegionDiscovery()
    )

    by_id = {outcome.stage_id: outcome for outcome in outcomes}
    assert by_id["region_discovery"].status is StageStatus.FAILED
    assert by_id["region_feature_extraction"].status is StageStatus.SKIPPED
    assert by_id["dense_feature_extraction"].status is StageStatus.SUCCEEDED

    # Partial evidence: no regions/region-features/claims, but the dense
    # feature (independent of region discovery) is still present.
    assert result.regions == ()
    assert result.claims == ()
    assert len(result.features) == 1
    assert result.features[0].scope.value == "dense"


def test_end_to_end_persists_reopens_and_groups_via_multi_run_evidence_set(tmp_path: Path) -> None:
    def _persist_run(run_index: int, run_id: str, observation_ids: list[str]) -> Path:
        writer = PerceptionRunWriter(
            workspace_root=tmp_path,
            sequence_name="corridor-02",
            run_id=PerceptionRunId(run_id),
            run_index=run_index,
            sequence_artifact_id="corridor-02-a1b2c3",
            selection_id="sha256:aaaa",
            enabled_capabilities=frozenset(
                {"region_discovery", "feature_extractor", "semantic_interpreter"}
            ),
            pipeline_preset=CANONICAL_PRESET_V1,
            configuration_digest="sha256:test",
            selection_label="frames",
            profile_label="fake",
        )
        for observation_id in observation_ids:
            outcomes, result = _process_observation(observation_id, run_id)
            writer.add_result(result)
            writer.add_stage_outcomes(outcomes)
        writer.finalize()
        return (
            tmp_path
            / "runs"
            / "visual-perception"
            / "corridor-02"
            / f"run-{run_index:04d}__frames__fake"
        )

    run_0001_dir = _persist_run(1, "run-0001", ["frame-0100", "frame-0124"])
    run_0002_dir = _persist_run(2, "run-0002", ["frame-0124"])

    # Isolated reopening, no registry required.
    reader = PerceptionRunReader(run_0001_dir)
    assert reader.verify_integrity() == []
    reopened_result = reader.result(SourceObservationId("frame-0124"))
    assert reopened_result.regions[0].bounding_box.width == 10

    # Multi-run evidence view groups both runs' results for the shared frame.
    evidence_set = PerceptionEvidenceSet.open([run_0001_dir, run_0002_dir])
    shared = evidence_set.evidence_for(SourceObservationId("frame-0124"))
    assert set(shared.results_by_run.keys()) == {
        PerceptionRunId("run-0001"),
        PerceptionRunId("run-0002"),
    }

    only_run_0001 = evidence_set.evidence_for(SourceObservationId("frame-0100"))
    assert set(only_run_0001.results_by_run.keys()) == {PerceptionRunId("run-0001")}

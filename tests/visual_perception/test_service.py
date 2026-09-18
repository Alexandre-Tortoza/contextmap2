from collections.abc import Mapping

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    PerceptionResultId,
    PerceptionRunId,
    Region2D,
    RegionId,
    StageDefinition,
    StageGraphError,
    StageStatus,
    assemble_perception_result,
    execute_stage_graph,
)
from contextmap.visual_perception.models import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    VisualFeature,
)
from contextmap.visual_perception.models import FeatureScope as _FeatureScope

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="fake", provider="fake", model="fake", version="0.1"
)


def _region() -> Region2D:
    return Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=0, y=0, width=10, height=10),
        provenance=_PROVENANCE,
    )


def test_independent_stages_run_in_deterministic_alphabetical_order() -> None:
    order: list[str] = []

    stages = [
        StageDefinition(stage_id="b", capability="fake", run=lambda ctx: order.append("b")),
        StageDefinition(stage_id="a", capability="fake", run=lambda ctx: order.append("a")),
    ]

    execute_stage_graph(stages)

    assert order == ["a", "b"]


def test_dependent_stage_runs_after_its_dependency() -> None:
    order: list[str] = []

    stages = [
        StageDefinition(
            stage_id="downstream",
            capability="fake",
            depends_on=frozenset({"upstream"}),
            run=lambda ctx: order.append("downstream"),
        ),
        StageDefinition(
            stage_id="upstream", capability="fake", run=lambda ctx: order.append("upstream")
        ),
    ]

    execute_stage_graph(stages)

    assert order == ["upstream", "downstream"]


def test_downstream_stage_receives_upstream_output() -> None:
    def upstream(context: Mapping[str, object]) -> int:
        return 42

    def downstream(context: Mapping[str, object]) -> int:
        return context["upstream"] * 2  # type: ignore[operator]

    stages = [
        StageDefinition(stage_id="upstream", capability="fake", run=upstream),
        StageDefinition(
            stage_id="downstream",
            capability="fake",
            depends_on=frozenset({"upstream"}),
            run=downstream,
        ),
    ]

    outcomes = execute_stage_graph(stages)

    by_id = {o.stage_id: o for o in outcomes}
    assert by_id["downstream"].output == 84


def test_failing_stage_does_not_crash_the_run() -> None:
    def failing(context: Mapping[str, object]) -> None:
        raise RuntimeError("backend exploded")

    outcomes = execute_stage_graph(
        [StageDefinition(stage_id="region_discovery", capability="region_discovery", run=failing)]
    )

    assert outcomes[0].status is StageStatus.FAILED
    assert "backend exploded" in (outcomes[0].error or "")


def test_dependent_of_failing_stage_is_skipped_but_independent_branch_still_runs() -> None:
    def failing(context: Mapping[str, object]) -> None:
        raise RuntimeError("region discovery exploded")

    def dependent(context: Mapping[str, object]) -> str:
        return "region features"

    def independent(context: Mapping[str, object]) -> str:
        return "dense features"

    stages = [
        StageDefinition(stage_id="region_discovery", capability="region_discovery", run=failing),
        StageDefinition(
            stage_id="region_feature_extraction",
            capability="feature_extractor",
            depends_on=frozenset({"region_discovery"}),
            run=dependent,
        ),
        StageDefinition(
            stage_id="dense_feature_extraction", capability="feature_extractor", run=independent
        ),
    ]

    outcomes = execute_stage_graph(stages)
    by_id = {o.stage_id: o for o in outcomes}

    assert by_id["region_discovery"].status is StageStatus.FAILED
    assert by_id["region_feature_extraction"].status is StageStatus.SKIPPED
    assert by_id["dense_feature_extraction"].status is StageStatus.SUCCEEDED
    assert by_id["dense_feature_extraction"].output == "dense features"


def test_duplicate_stage_id_is_rejected() -> None:
    stages = [
        StageDefinition(stage_id="a", capability="fake", run=lambda ctx: None),
        StageDefinition(stage_id="a", capability="fake", run=lambda ctx: None),
    ]

    with pytest.raises(StageGraphError, match="duplicate"):
        execute_stage_graph(stages)


def test_dependency_on_unknown_stage_is_rejected() -> None:
    stages = [
        StageDefinition(
            stage_id="a",
            capability="fake",
            depends_on=frozenset({"does-not-exist"}),
            run=lambda ctx: None,
        ),
    ]

    with pytest.raises(StageGraphError, match="unknown"):
        execute_stage_graph(stages)


def test_cycle_is_rejected() -> None:
    stages = [
        StageDefinition(
            stage_id="a", capability="fake", depends_on=frozenset({"b"}), run=lambda ctx: None
        ),
        StageDefinition(
            stage_id="b", capability="fake", depends_on=frozenset({"a"}), run=lambda ctx: None
        ),
    ]

    with pytest.raises(StageGraphError, match="cycle"):
        execute_stage_graph(stages)


def test_assemble_perception_result_uses_only_succeeded_stages() -> None:
    region = _region()
    feature = VisualFeature(
        feature_id="feat-0001",  # type: ignore[arg-type]
        scope=_FeatureScope.DENSE,
        embedding_space_id="dinov3-vitl",
        shape=(384,),
        dtype="float32",
        payload_reference="features/feat-0001.bin",
        provenance=_PROVENANCE,
    )
    claim = SemanticClaim(
        claim_id=ClaimId("claim-0001"),
        text="a doorway",
        role=HypothesisRole.PRIMARY,
        provenance=_PROVENANCE,
        region_id=region.region_id,
    )

    def region_discovery(ctx: Mapping[str, object]) -> tuple[Region2D, ...]:
        return (region,)

    def dense_features(ctx: Mapping[str, object]) -> tuple[VisualFeature, ...]:
        return (feature,)

    def failing_semantics(ctx: Mapping[str, object]) -> None:
        raise RuntimeError("semantic backend unavailable")

    outcomes = execute_stage_graph(
        [
            StageDefinition(
                stage_id="region_discovery", capability="region_discovery", run=region_discovery
            ),
            StageDefinition(
                stage_id="dense_feature_extraction",
                capability="feature_extractor",
                run=dense_features,
            ),
            StageDefinition(
                stage_id="semantic_interpretation",
                capability="semantic_interpreter",
                run=failing_semantics,
            ),
        ]
    )

    result = assemble_perception_result(
        result_id=PerceptionResultId("result-0001"),
        source_observation_id=SourceObservationId("frame-0124"),
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
        outcomes=outcomes,
        region_stage_id="region_discovery",
        feature_stage_ids=("dense_feature_extraction",),
        claim_stage_ids=("semantic_interpretation",),
    )

    assert result.regions == (region,)
    assert result.features == (feature,)
    assert result.claims == ()  # semantic_interpretation failed; no claim contributed
    assert claim.text == "a doorway"  # sanity: claim object itself was never touched

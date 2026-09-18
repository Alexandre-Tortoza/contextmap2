import dataclasses

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    PerceptionResult,
    PerceptionRunId,
    Region2D,
    claim_id_for,
    feature_id_for,
    perception_result_id_for,
    region_id_for,
)

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="region_discovery", provider="fake", model="fake", version="0.1"
)


def test_perception_result_id_is_deterministic() -> None:
    run_id = PerceptionRunId("run-0001")
    frame_id = SourceObservationId("frame-0124")

    assert perception_result_id_for(run_id=run_id, source_observation_id=frame_id) == (
        perception_result_id_for(run_id=run_id, source_observation_id=frame_id)
    )


def test_repeated_runs_over_the_same_frame_produce_distinct_result_ids() -> None:
    frame_id = SourceObservationId("frame-0124")

    result_id_a = perception_result_id_for(
        run_id=PerceptionRunId("run-0001"), source_observation_id=frame_id
    )
    result_id_b = perception_result_id_for(
        run_id=PerceptionRunId("run-0002"), source_observation_id=frame_id
    )

    assert result_id_a != result_id_b


def test_region_feature_claim_ids_are_scoped_to_their_result() -> None:
    result_id_a = perception_result_id_for(
        run_id=PerceptionRunId("run-0001"), source_observation_id=SourceObservationId("frame-0124")
    )
    result_id_b = perception_result_id_for(
        run_id=PerceptionRunId("run-0002"), source_observation_id=SourceObservationId("frame-0124")
    )

    region_a = region_id_for(result_id=result_id_a, index=0)
    region_b = region_id_for(result_id=result_id_b, index=0)

    assert region_a != region_b
    assert feature_id_for(result_id=result_id_a, index=0) != feature_id_for(
        result_id=result_id_b, index=0
    )
    assert claim_id_for(result_id=result_id_a, index=0) != claim_id_for(
        result_id=result_id_b, index=0
    )


def test_overlapping_selections_produce_independent_results_per_run() -> None:
    """run-0001 (frames 0..1000) and run-0005 (frames 430..480) never merge their evidence."""
    shared_frame = SourceObservationId("frame-0450")

    run_0001_result_id = perception_result_id_for(
        run_id=PerceptionRunId("run-0001"), source_observation_id=shared_frame
    )
    run_0005_result_id = perception_result_id_for(
        run_id=PerceptionRunId("run-0005"), source_observation_id=shared_frame
    )

    assert run_0001_result_id != run_0005_result_id
    # Each run's region-0000 for this shared frame remains a distinct identity.
    assert region_id_for(result_id=run_0001_result_id, index=0) != region_id_for(
        result_id=run_0005_result_id, index=0
    )


def test_identity_survives_a_plain_serialization_round_trip() -> None:
    result_id = perception_result_id_for(
        run_id=PerceptionRunId("run-0001"), source_observation_id=SourceObservationId("frame-0124")
    )
    region = Region2D(
        region_id=region_id_for(result_id=result_id, index=0),
        bounding_box=BoundingBox2D(x=0, y=0, width=10, height=10),
        provenance=_PROVENANCE,
    )
    result = PerceptionResult(
        result_id=result_id,
        source_observation_id=SourceObservationId("frame-0124"),
        run_id=PerceptionRunId("run-0001"),
        sequence_artifact_id="corridor-02-a1b2c3",
        created_at="2026-01-01T00:00:00+00:00",
        regions=(region,),
    )

    # A plain dict round-trip (the shape any JSON encoder would produce)
    # must preserve every identity exactly as a string.
    as_dict = dataclasses.asdict(result)
    rebuilt = PerceptionResult(
        result_id=as_dict["result_id"],
        source_observation_id=as_dict["source_observation_id"],
        run_id=as_dict["run_id"],
        sequence_artifact_id=as_dict["sequence_artifact_id"],
        created_at=as_dict["created_at"],
        regions=(
            Region2D(
                region_id=as_dict["regions"][0]["region_id"],
                bounding_box=BoundingBox2D(**as_dict["regions"][0]["bounding_box"]),
                provenance=BackendProvenance(**as_dict["regions"][0]["provenance"]),
            ),
        ),
    )

    assert rebuilt.result_id == result.result_id
    assert rebuilt.regions[0].region_id == result.regions[0].region_id

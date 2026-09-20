import dataclasses
import typing

import pytest
from fusion_builders import (
    make_contribution,
    make_region_feature,
    make_scorer,
    spatial_id,
)

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import (
    EvidenceContribution,
    EvidenceContributionId,
    ObservationQualityRef,
    ScoreReference,
)
from contextmap.sensor_association import QualityComponent
from contextmap.visual_perception import ClaimId


def test_a_contribution_names_the_physical_observation_and_the_inference_apart() -> None:
    contribution = make_contribution(run="run-a", frame="frame-0120")

    assert contribution.physical_observation_id == "frame-0120"
    assert contribution.perception_result_id == "run-a--frame-0120"
    assert contribution.perception_run_id == "run-a"
    assert contribution.spatial_observation_id == spatial_id("run-a", "frame-0120")


def test_two_inference_runs_over_one_frame_share_the_physical_observation() -> None:
    first = make_contribution(run="run-a", frame="frame-0120")
    second = make_contribution(run="run-b", frame="frame-0120")

    assert first.physical_observation_id == second.physical_observation_id
    assert first.perception_result_id != second.perception_result_id
    assert first.contribution_id != second.contribution_id


def test_one_claim_over_many_geometry_points_is_one_claim_with_spatial_support() -> None:
    contribution = make_contribution(geometry_indexes=range(500))

    assert len(contribution.geometry_support) == 500
    assert [ref.claim_id for ref in contribution.claim_refs] == ["claim-0001"]


def test_a_view_without_claims_is_kept_as_evidence_without_semantic_content() -> None:
    contribution = make_contribution(claim_ids=())

    assert contribution.claim_refs == ()
    assert contribution.geometry_support


def test_scorer_output_is_referenced_by_claim_and_scorer_identity() -> None:
    reference = ScoreReference(claim_id=ClaimId("claim-0001"), scorer=make_scorer())
    contribution = make_contribution(score_refs=(reference,))

    assert contribution.score_refs == (reference,)


def test_observation_quality_is_a_reference_and_never_a_score() -> None:
    contribution = make_contribution()
    hints = typing.get_type_hints(EvidenceContribution)

    assert isinstance(contribution.observation_quality, ObservationQualityRef)
    assert contribution.observation_quality.definitions_version == "observation-quality-v1"
    assert hints["observation_quality"] == ObservationQualityRef | None
    assert not {"confidence", "quality", "weight", "score"} & {
        field.name for field in dataclasses.fields(EvidenceContribution)
    }


def test_missing_observation_quality_is_absent_and_not_a_zero() -> None:
    assert make_contribution(with_quality=False).observation_quality is None


def test_the_quality_reference_carries_no_semantic_or_component_values() -> None:
    names = {field.name for field in dataclasses.fields(ObservationQualityRef)}
    component_names = {component.value for component in QualityComponent}

    assert names == {"spatial_observation_id", "definitions_version"}
    assert not names & component_names


def test_static_structural_evidence_is_not_attached_to_a_view() -> None:
    names = {field.name for field in dataclasses.fields(EvidenceContribution)}

    assert not {name for name in names if "point_representation" in name}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"geometry_indexes": ()}, "geometry_support"),
        ({"geometry_indexes": (2, 1)}, "sorted by geometry_id and unique"),
        ({"geometry_indexes": (1, 1)}, "sorted by geometry_id and unique"),
        ({"claim_ids": ("claim-0002", "claim-0001")}, "claim_refs must be sorted and unique"),
        ({"claim_ids": ("claim-0001", "claim-0001")}, "claim_refs must be sorted and unique"),
    ],
)
def test_a_malformed_contribution_is_rejected(
    overrides: dict[str, typing.Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_contribution(**overrides)


def test_a_score_for_a_claim_the_view_does_not_carry_is_rejected() -> None:
    orphan = ScoreReference(claim_id=ClaimId("claim-0099"), scorer=make_scorer())

    with pytest.raises(ValueError, match="claim-0099"):
        make_contribution(score_refs=(orphan,))


def test_duplicate_or_unsorted_score_references_are_rejected() -> None:
    first = ScoreReference(claim_id=ClaimId("claim-0001"), scorer=make_scorer("clip_a"))
    second = ScoreReference(claim_id=ClaimId("claim-0001"), scorer=make_scorer("clip_b"))

    with pytest.raises(ValueError, match="score_refs must be sorted and unique"):
        make_contribution(score_refs=(first, first))
    with pytest.raises(ValueError, match="score_refs must be sorted and unique"):
        make_contribution(score_refs=(second, first))
    assert make_contribution(score_refs=(first, second)).score_refs == (first, second)


def test_a_region_feature_of_another_region_is_rejected() -> None:
    foreign = make_region_feature(region="region-0009")

    with pytest.raises(ValueError, match="region-0009"):
        make_contribution(feature_refs=(foreign,))


def test_features_must_be_sorted_and_unique() -> None:
    first = make_region_feature(feature="feature-0001")
    second = make_region_feature(feature="feature-0002")

    with pytest.raises(ValueError, match="visual_feature_refs must be sorted and unique"):
        make_contribution(feature_refs=(second, first))
    assert make_contribution(feature_refs=(first, second)).visual_feature_refs == (first, second)


def test_a_quality_reference_to_another_observation_is_rejected() -> None:
    other = ObservationQualityRef(
        spatial_observation_id=spatial_id("run-b", "frame-0120"),
        definitions_version="observation-quality-v1",
    )

    with pytest.raises(ValueError, match="observation_quality"):
        make_contribution(quality=other)


def test_an_unversioned_quality_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="definitions_version"):
        ObservationQualityRef(
            spatial_observation_id=spatial_id("run-a", "frame-0120"), definitions_version=""
        )


def test_every_identity_must_be_present() -> None:
    valid = make_contribution()

    with pytest.raises(ValueError, match="contribution_id"):
        dataclasses.replace(valid, contribution_id=EvidenceContributionId(""))
    with pytest.raises(ValueError, match="physical_observation_id"):
        dataclasses.replace(valid, physical_observation_id=SourceObservationId(""))


def test_the_geometry_of_a_view_must_belong_to_one_map() -> None:
    mixed = (
        GeometryReference(
            map_id=MapId("map-0001"), geometry_id=geometry_id_for(map_id=MapId("map-0001"), index=0)
        ),
        GeometryReference(
            map_id=MapId("map-0002"), geometry_id=geometry_id_for(map_id=MapId("map-0002"), index=1)
        ),
    )

    with pytest.raises(ValueError, match="one map"):
        dataclasses.replace(make_contribution(), geometry_support=mixed)

import copy
import dataclasses
import json

import numpy as np
import pytest
from perception_builders import (
    RESULT_ID,
    make_claim,
    make_feature,
    make_region,
    make_result,
    rect_mask,
)
from projection_builders import (
    artifact,
    make_prepared_image,
    map_point_for_pixel,
    mask_with,
    project_frame,
    scene_frame,
)

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import SequenceArtifactId, SourceObservationId
from contextmap.sensor_association import (
    SpatialObservation,
    VisibilityDiagnostics,
    VisibilityState,
    spatial_observation_id_for,
)
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.frame_projection import FrameProjection
from contextmap.sensor_association.membership import (
    COVERAGE_DEFINITIONS_VERSION,
    MEMBERSHIP_POLICY_ID,
    FrameMembership,
    SkippedRegion,
    SkipReason,
    associate_regions,
    build_spatial_observations,
)
from contextmap.sensor_association.serialization import (
    decode_spatial_observation,
    encode_spatial_observation,
)
from contextmap.sensor_association.visibility import OcclusionPolicy, resolve_visibility
from contextmap.visual_perception import (
    BoundingBox,
    CropOperation,
    ExclusionRegion,
    FeatureScope,
    RegionId,
    ResizeOperation,
)

POLICY = OcclusionPolicy(
    cell_size_px=4,
    neighborhood_radius_cells=2,
    depth_margin_m=0.1,
    depth_margin_ratio=0.02,
)
A, B, C = RegionId("region-A"), RegionId("region-B"), RegionId("region-C")


def _associate(
    frame: FrameProjection, *regions: object, **result_options: object
) -> FrameMembership:
    result = make_result(list(regions), **result_options)  # type: ignore[arg-type]
    return associate_regions(resolve_visibility(frame, POLICY), result)


def _rect(x0: int, y0: int, x1: int, y1: int):  # type: ignore[no-untyped-def]
    return rect_mask(640, 480, x0, y0, x1, y1)


# --- Membership -------------------------------------------------------------


def test_visible_geometry_inside_a_frozen_mask_is_associated_reproducibly() -> None:
    frame = scene_frame((100, 100, 3.0), (200, 100, 3.0), (300, 100, 3.0))
    region = make_region("region-A", _rect(90, 90, 210, 110))

    first = _associate(frame, region)
    second = _associate(frame, region)

    assert first.points_of(A).tolist() == [0, 1]
    assert first.visible_unassigned.tolist() == [False, False, True]
    assert np.array_equal(first.point_indices, second.point_indices)
    assert np.array_equal(first.point_region_slots, second.point_region_slots)


def test_a_region_that_sees_no_visible_geometry_is_still_evaluated() -> None:
    frame = scene_frame((100, 100, 3.0))
    empty = make_region("region-A", _rect(400, 300, 500, 400))

    membership = _associate(frame, empty)

    assert membership.points_of(A).tolist() == []
    assert [region.region_id for region in membership.regions] == [A]
    assert membership.regions_of(0) == ()


def test_overlapping_masks_all_keep_the_same_geometry_and_none_is_a_winner() -> None:
    frame = scene_frame((100, 100, 3.0), (150, 100, 3.0), (240, 100, 3.0))
    regions = (
        make_region("region-A", _rect(90, 90, 210, 110)),
        make_region("region-B", _rect(140, 90, 260, 110)),
        make_region("region-C", _rect(95, 95, 160, 105)),
    )

    membership = _associate(frame, *regions)

    assert membership.regions_of(0) == (A, C)
    assert membership.regions_of(1) == (A, B, C)
    assert membership.regions_of(2) == (B,)
    assert set(membership.points_of(A).tolist()) & set(membership.points_of(B).tolist()) == {1}
    assert membership.statistics().overlap_point_count == 2


def test_a_region_is_the_same_whatever_the_order_of_the_points() -> None:
    pixels = [(100, 100, 3.0), (150, 100, 3.0), (240, 100, 3.0), (400, 300, 3.0)]
    regions = (
        make_region("region-A", _rect(90, 90, 210, 110)),
        make_region("region-B", _rect(140, 90, 260, 110)),
    )
    order = np.random.default_rng(3).permutation(len(pixels))

    forward = _associate(scene_frame(*pixels), *regions)
    shuffled = _associate(scene_frame(*[pixels[i] for i in order]), *regions)

    for shuffled_index, original_index in enumerate(order):
        assert shuffled.regions_of(shuffled_index) == forward.regions_of(int(original_index))


def test_a_pixel_is_in_a_mask_by_the_pixel_whose_center_is_nearest() -> None:
    # (446, 166) cai em (172.75, 57.75) na imagem preparada: o pixel de máscara é (173, 58).
    prepared = _crop_and_resize()
    point = map_point_for_pixel(446, 166, 2.0)
    frame = project_frame([point], prepared=prepared)

    nearest = _associate(frame, make_region("region-A", mask_with(200, 150, (173, 58))))
    truncated = _associate(frame, make_region("region-A", mask_with(200, 150, (172, 57))))

    assert nearest.points_of(A).tolist() == [0]
    assert truncated.points_of(A).tolist() == []


# --- Occlusion and support around the mask ----------------------------------


def test_the_mask_footprint_also_counts_what_was_hidden_or_unsupported() -> None:
    prepared = make_prepared_image(
        exclusion_regions=(
            ExclusionRegion(
                name="own_body", mask=mask_with(640, 480, (170, 150)), reason="fixture", source="t"
            ),
        )
    )
    frame = scene_frame(
        (150, 150, 2.0), (150, 150, 6.0), (170, 150, 3.0), (600, 400, 3.0), prepared=prepared
    )

    membership = _associate(frame, make_region("region-A", _rect(140, 140, 180, 160)))
    region = membership.regions[0]

    assert membership.points_of(A).tolist() == [0]
    assert region.occluded_count == 1
    assert region.outside_valid_support_count == 1
    assert region.footprint_point_count == 3
    assert region.visible_share == pytest.approx(1 / 3)


def test_the_coverage_is_the_share_of_mask_pixels_with_supported_geometry() -> None:
    # Dois pontos no mesmo pixel e um em outro: 3 pontos associados cobrem 2 pixels.
    frame = scene_frame((150, 150, 2.0), (150, 150, 2.02), (155, 150, 2.0))
    membership = _associate(frame, make_region("region-A", _rect(140, 140, 170, 160)))
    region = membership.regions[0]

    assert region.mask_area_px == 30 * 20
    assert region.associated_count == 3
    assert region.covered_pixel_count == 2
    assert region.support_pixel_coverage == pytest.approx(2 / 600)
    assert COVERAGE_DEFINITIONS_VERSION == "region-support-metrics-v1"


def test_a_footprint_with_nothing_in_it_has_no_visible_share() -> None:
    membership = _associate(
        scene_frame((100, 100, 3.0)), make_region("region-A", _rect(400, 300, 420, 320))
    )

    assert membership.regions[0].visible_share is None
    assert membership.regions[0].support_pixel_coverage == 0.0


def test_the_frame_statistics_account_for_every_point() -> None:
    frame = project_frame(
        [
            map_point_for_pixel(100, 100, 3.0),
            map_point_for_pixel(150, 100, 3.0),
            map_point_for_pixel(150, 100, 6.0),
            map_point_for_pixel(240, 100, 3.0),
            map_point_for_pixel(400, 300, 3.0),
            (-3.0, 0.0, 0.0),
            (0.5, -10.0, 0.0),
        ]
    )
    regions = (
        make_region("region-A", _rect(90, 90, 210, 110)),
        make_region("region-B", _rect(140, 90, 260, 110)),
    )

    statistics = _associate(frame, *regions).statistics()

    assert statistics.definitions_version == COVERAGE_DEFINITIONS_VERSION
    assert statistics.point_count == 7
    assert statistics.behind_camera_count == 1
    assert statistics.outside_image_count == 1
    assert statistics.outside_valid_support_count == 0
    assert statistics.occluded_count == 1
    assert statistics.visible_count == 4
    assert statistics.associated_count == 3
    assert statistics.visible_unassigned_count == 1
    assert statistics.overlap_point_count == 1
    assert statistics.membership_pair_count == 4
    assert statistics.points_by_region_count == {0: 1, 1: 2, 2: 1}
    assert (
        statistics.behind_camera_count
        + statistics.outside_image_count
        + statistics.outside_valid_support_count
        + statistics.occluded_count
        + statistics.visible_count
        == statistics.point_count
    )


# --- Coordinate spaces ------------------------------------------------------


def _crop_and_resize():  # type: ignore[no-untyped-def]
    return make_prepared_image(
        (
            CropOperation(
                box=BoundingBox(x_min=100, y_min=50, x_max=500, y_max=350),
                output_image=artifact("crop"),
                provenance_source="cfg",
            ),
            ResizeOperation(
                width=200, height=150, output_image=artifact("resize"), provenance_source="cfg"
            ),
        )
    )


def test_crop_and_resize_do_not_shift_mask_membership() -> None:
    # O ponto base cai no pixel preparado (172, 57); só a máscara nesse pixel o captura.
    frame = project_frame([map_point_for_pixel(445, 165, 2.0)], prepared=_crop_and_resize())

    exact = _associate(frame, make_region("region-A", mask_with(200, 150, (172, 57))))
    shifted = _associate(frame, make_region("region-A", mask_with(200, 150, (171, 57))))

    assert exact.points_of(A).tolist() == [0]
    assert shifted.points_of(A).tolist() == []


def test_a_mask_in_the_raw_image_space_is_rejected_for_a_prepared_image() -> None:
    frame = project_frame([map_point_for_pixel(445, 165, 2.0)], prepared=_crop_and_resize())

    with pytest.raises(AssociationInputError, match="prepared image"):
        _associate(frame, make_region("region-A", _rect(400, 150, 500, 200)))


def test_a_result_of_another_observation_is_rejected() -> None:
    frame = scene_frame((100, 100, 3.0))

    with pytest.raises(AssociationInputError, match="observation"):
        _associate(
            frame,
            make_region("region-A", _rect(90, 90, 110, 110)),
            observation_id=SourceObservationId("frame-9999"),
        )


# --- Regions that are not evaluated -----------------------------------------


def test_rejected_and_box_only_regions_are_reported_and_never_evaluated() -> None:
    frame = scene_frame((100, 100, 3.0))
    regions = (
        make_region("region-A", _rect(90, 90, 110, 110)),
        make_region("region-B", _rect(90, 90, 110, 110), accepted=False),
        make_region("region-C", None),
    )

    membership = _associate(frame, *regions)

    assert [region.region_id for region in membership.regions] == [A]
    assert membership.skipped == (
        SkippedRegion(region_id=B, reason=SkipReason.REJECTED),
        SkippedRegion(region_id=C, reason=SkipReason.NO_INLINE_MASK),
    )
    assert membership.points_of(A).tolist() == [0]
    assert membership.regions_of(0) == (A,)


# --- Spatial observations ---------------------------------------------------


def _observed_scene() -> tuple[FrameMembership, tuple[SpatialObservation, ...]]:
    frame = scene_frame(
        (100, 100, 3.0), (150, 100, 3.0), (150, 100, 6.0), (240, 100, 3.0), (400, 300, 3.0)
    )
    result = make_result(
        [
            make_region("region-A", _rect(90, 90, 210, 110)),
            make_region("region-B", _rect(140, 90, 260, 110)),
        ],
        features=[
            make_feature("f-a", FeatureScope.REGION, region_id="region-A"),
            make_feature("f-b", FeatureScope.REGION, region_id="region-B"),
            make_feature("f-dense", FeatureScope.DENSE),
            make_feature("f-global", FeatureScope.GLOBAL),
        ],
        claims=[
            make_claim("c-a", region_id="region-A"),
            make_claim("c-b", region_id="region-B"),
            make_claim("c-scene"),
        ],
    )
    membership = associate_regions(resolve_visibility(frame, POLICY), result)
    observations = build_spatial_observations(
        membership, configuration_fingerprint="sha256:run-config", code_version="test"
    )
    return membership, observations


def test_each_evaluated_region_yields_one_spatial_observation() -> None:
    membership, observations = _observed_scene()

    assert [o.region_id for o in observations] == [A, B]
    assert [o.spatial_observation_id for o in observations] == [
        spatial_observation_id_for(perception_result_id=RESULT_ID, region_id=A),
        spatial_observation_id_for(perception_result_id=RESULT_ID, region_id=B),
    ]
    assert all(o.perception_result_id == RESULT_ID for o in observations)
    assert all(o.source_observation_id == "frame-0001" for o in observations)
    assert membership.frame.source_observation_id == "frame-0001"


def test_the_support_is_the_references_of_the_associated_geometry_and_may_overlap() -> None:
    membership, (obs_a, obs_b) = _observed_scene()
    frame = membership.frame

    assert obs_a.geometry_support == tuple(frame.map_reference(i) for i in (0, 1))
    assert obs_b.geometry_support == tuple(frame.map_reference(i) for i in (1, 3))
    assert set(obs_a.geometry_support) & set(obs_b.geometry_support) == {frame.map_reference(1)}


def test_the_visibility_counts_explain_what_the_region_did_not_get() -> None:
    _, (obs_a, _) = _observed_scene()

    assert obs_a.visibility == VisibilityDiagnostics(
        counts={VisibilityState.ASSOCIATED: 2, VisibilityState.OCCLUDED: 1}
    )
    summary = obs_a.projection_summary
    assert (summary.camera_model_kind, summary.considered_count, summary.visible_count) == (
        "pinhole",
        5,
        4,
    )
    assert summary.prepared_image_size == (640, 480)
    assert summary.image_transform_id.startswith("sha256:")


def test_evidence_is_referenced_by_region_and_never_copied() -> None:
    _, (obs_a, obs_b) = _observed_scene()

    assert [ref.feature_id for ref in obs_a.visual_feature_refs] == ["f-a", "f-dense"]
    assert [ref.feature_id for ref in obs_b.visual_feature_refs] == ["f-b", "f-dense"]
    assert [ref.claim_id for ref in obs_a.semantic_claim_refs] == ["c-a"]
    assert [ref.claim_id for ref in obs_b.semantic_claim_refs] == ["c-b"]
    assert {ref.embedding_space_id for ref in obs_a.visual_feature_refs} == {"dinov2:b14"}


def test_the_provenance_keeps_the_exact_perception_and_map_identities() -> None:
    membership, (obs_a, _) = _observed_scene()

    provenance = obs_a.provenance
    assert provenance.geometric_map_id == MapId("map-0001")
    assert provenance.perception_run_id == "run-0001"
    assert provenance.sequence_artifact_id == SequenceArtifactId("sequence-0001")
    assert provenance.membership_policy_id == MEMBERSHIP_POLICY_ID == "mask-membership-v1"
    assert provenance.visibility_policy_id == POLICY.policy_id
    assert provenance.configuration_fingerprint == "sha256:run-config"
    assert obs_a.calibration_ref == membership.frame.calibration_ref
    assert obs_a.pose_ref == membership.frame.pose_ref


def test_a_region_without_visible_geometry_is_still_a_valid_observation() -> None:
    frame = scene_frame((100, 100, 3.0))
    membership = _associate(frame, make_region("region-A", _rect(400, 300, 500, 400)))

    (observation,) = build_spatial_observations(
        membership, configuration_fingerprint=None, code_version=None
    )

    assert observation.geometry_support == ()
    assert observation.visibility == VisibilityDiagnostics(counts={})


def test_an_observation_survives_a_json_round_trip() -> None:
    _, (obs_a, obs_b) = _observed_scene()

    for observation in (obs_a, obs_b):
        record = json.loads(json.dumps(encode_spatial_observation(observation)))
        assert decode_spatial_observation(record) == observation


def test_the_membership_does_not_carry_labels_or_masks() -> None:
    membership, _ = _observed_scene()

    fields = {f.name for f in dataclasses.fields(membership.regions[0])}

    assert not fields & {"mask", "label", "labels", "entity_id", "class_id", "claims"}


def test_associating_never_mutates_the_frozen_regions_or_their_masks() -> None:
    frame = scene_frame((100, 100, 3.0), (150, 100, 3.0), (150, 100, 8.0))
    result = make_result(
        [
            make_region("region-A", rect_mask(640, 480, 90, 90, 210, 110)),
            make_region("region-B", rect_mask(640, 480, 140, 90, 260, 110)),
        ]
    )
    before = [(r, r.region_id, r.mask, r.mask.data if r.mask else None) for r in result.regions]
    copies = [copy.deepcopy(r) for r in result.regions]

    membership = associate_regions(resolve_visibility(frame, POLICY), result)
    build_spatial_observations(membership, configuration_fingerprint=None, code_version=None)

    assert list(result.regions) == copies
    for region, region_id, mask, data in before:
        assert (region.region_id, region.mask, region.mask.data if region.mask else None) == (
            region_id,
            mask,
            data,
        )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.regions[0].region_id = RegionId("changed")  # type: ignore[misc]

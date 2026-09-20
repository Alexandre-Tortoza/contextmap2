import dataclasses
import subprocess
import sys
from typing import Any

import pytest
from association_builders import (
    MAP_ID,
    RESULT_ID,
    make_calibration_ref,
    make_pose_ref,
    make_provenance,
    make_spatial_observation,
    make_summary,
    refs,
)

import contextmap.sensor_association as sensor_association
from contextmap.geometric_mapping import GeometryReference, MapId
from contextmap.ingestion import CalibrationReferenceId, FrameId
from contextmap.sensor_association import (
    CalibrationRef,
    DepthMetric,
    PixelCoordinate,
    PointCorrespondence,
    ProjectionSummary,
    SemanticClaimRef,
    SpatialObservation,
    VisibilityDiagnostics,
    VisibilityState,
    VisualFeatureRef,
    spatial_observation_id_for,
)
from contextmap.state_estimation import LookupOutcome, PoseEstimateId
from contextmap.visual_perception import ClaimId, FeatureId, FeatureScope, RegionId


def test_public_api_exports_resolve() -> None:
    for name in sensor_association.__all__:
        assert hasattr(sensor_association, name), name


def test_the_contracts_are_readable_without_robotics_or_model_libraries() -> None:
    code = (
        "import sys, contextmap.sensor_association;"
        "bad = [m for m in ('numpy', 'rosbags', 'torch', 'open3d') if m in sys.modules];"
        "assert not bad, bad"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


# --- SpatialObservation -----------------------------------------------------


def test_one_region_is_anchored_to_many_stable_geometry_references() -> None:
    observation = make_spatial_observation(support=(3, 5, 9, 11))

    assert observation.geometry_support == refs((3, 5, 9, 11))
    assert all(isinstance(ref, GeometryReference) for ref in observation.geometry_support)
    assert observation.region_id == RegionId("region-0001")
    assert observation.perception_result_id == RESULT_ID


def test_overlapping_regions_can_share_the_same_geometry_without_a_winner() -> None:
    pallet = make_spatial_observation(region="region-0001", support=(3, 5, 9))
    surface = make_spatial_observation(region="region-0004", support=(5, 9, 12))

    shared = set(pallet.geometry_support) & set(surface.geometry_support)

    assert shared == set(refs((5, 9)))
    assert pallet.spatial_observation_id != surface.spatial_observation_id


def test_the_support_is_sorted_unique_and_belongs_to_one_map() -> None:
    with pytest.raises(ValueError, match="geometry_support"):
        make_spatial_observation(support=(5, 3))
    with pytest.raises(ValueError, match="geometry_support"):
        make_spatial_observation(support=(3, 3))
    other = MapId("other-map")
    mixed = (*refs((3,)), *refs((5,), map_id=other))
    with pytest.raises(ValueError, match="map"):
        dataclasses.replace(make_spatial_observation(), geometry_support=mixed)


def test_the_support_must_belong_to_the_map_the_provenance_names() -> None:
    with pytest.raises(ValueError, match="map"):
        dataclasses.replace(
            make_spatial_observation(),
            provenance=make_provenance(map_id=MapId("other-map")),
        )


def test_a_region_with_no_visible_support_is_representable() -> None:
    observation = dataclasses.replace(
        make_spatial_observation(),
        geometry_support=(),
        visibility=VisibilityDiagnostics(counts={VisibilityState.OCCLUDED: 7}),
    )

    assert observation.geometry_support == ()


def test_evidence_is_referenced_never_copied_into_the_observation() -> None:
    observation = make_spatial_observation()

    (feature,) = observation.visual_feature_refs
    (claim,) = observation.semantic_claim_refs
    assert feature.feature_id == FeatureId("run-0001--frame-0124--feature-0000")
    assert feature.embedding_space_id == "dinov2:vitb14"
    assert claim.claim_id == ClaimId("run-0001--frame-0124--claim-0000")
    forbidden = {"label", "labels", "embedding", "payload", "entity", "entity_id", "point_label"}
    assert not forbidden & {field.name for field in dataclasses.fields(SpatialObservation)}


def test_it_is_not_a_fused_belief_or_an_entity() -> None:
    names = {field.name for field in dataclasses.fields(SpatialObservation)}

    assert not {"fused_evidence", "entity_id", "identity", "final_label"} & names


def test_the_observation_is_fully_traceable_to_geometry_calibration_pose_and_visual_evidence() -> (
    None
):
    observation = make_spatial_observation()

    assert observation.source_observation_id == "frame-0124"
    assert observation.calibration_ref.calibration_identity == "sha256:calibration"
    assert observation.calibration_ref.camera_model_kind == "mei"
    assert observation.pose_ref.trajectory_id == "traj"
    assert observation.pose_ref.source_estimate_ids
    assert observation.provenance.geometric_map_id == MAP_ID
    assert observation.provenance.perception_run_id == "run-0001"
    assert observation.projection_summary.image_transform_id == "sha256:raw-to-prepared"
    assert observation.provenance.visibility_policy_id


def test_the_observation_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_spatial_observation().region_id = RegionId("x")  # type: ignore[misc]


def test_the_identity_is_deterministic_per_result_and_region() -> None:
    first = spatial_observation_id_for(perception_result_id=RESULT_ID, region_id=RegionId("r1"))

    assert first == spatial_observation_id_for(
        perception_result_id=RESULT_ID, region_id=RegionId("r1")
    )
    assert first != spatial_observation_id_for(
        perception_result_id=RESULT_ID, region_id=RegionId("r2")
    )


@pytest.mark.parametrize(
    "field_name",
    ["spatial_observation_id", "source_observation_id", "perception_result_id", "region_id"],
)
def test_required_identities_must_not_be_empty(field_name: str) -> None:
    empty: dict[str, Any] = {field_name: ""}
    with pytest.raises(ValueError, match=field_name):
        dataclasses.replace(make_spatial_observation(), **empty)


# --- Visibility -------------------------------------------------------------


def test_the_visibility_states_cover_every_reason_a_point_is_not_evidence() -> None:
    assert {state.value for state in VisibilityState} >= {
        "behind_camera",
        "outside_image",
        "outside_valid_support",
        "occluded",
        "visible_unassigned",
        "associated",
    }


def test_visibility_counts_are_non_negative_and_the_associated_count_matches_the_support() -> None:
    with pytest.raises(ValueError, match="count"):
        VisibilityDiagnostics(counts={VisibilityState.OCCLUDED: -1})
    with pytest.raises(ValueError, match="associated"):
        dataclasses.replace(
            make_spatial_observation(support=(3, 5, 9)),
            visibility=VisibilityDiagnostics(counts={VisibilityState.ASSOCIATED: 2}),
        )
    assert VisibilityDiagnostics(counts={VisibilityState.OCCLUDED: 4}).total == 4


def test_the_projection_summary_is_consistent() -> None:
    assert make_summary(considered=100, visible=60).visible_count == 60
    with pytest.raises(ValueError, match="visible_count"):
        make_summary(considered=10, visible=11)
    with pytest.raises(ValueError, match="image_transform_id"):
        ProjectionSummary(
            camera_model_kind="pinhole",
            depth_metric=DepthMetric.OPTICAL_AXIS,
            image_transform_id="",
            prepared_image_size=(640, 480),
            considered_count=1,
            visible_count=1,
        )
    with pytest.raises(ValueError, match="prepared_image_size"):
        ProjectionSummary(
            camera_model_kind="pinhole",
            depth_metric=DepthMetric.OPTICAL_AXIS,
            image_transform_id="x",
            prepared_image_size=(0, 480),
            considered_count=1,
            visible_count=1,
        )


def test_the_projection_summary_states_how_depth_is_measured() -> None:
    summary = make_spatial_observation().projection_summary

    assert summary.depth_metric is DepthMetric.RAY_RANGE
    assert {metric.value for metric in DepthMetric} == {"optical_axis", "ray_range"}


def test_the_region_diagnostics_cannot_exceed_what_the_projection_considered_or_kept_visible() -> (
    None
):
    with pytest.raises(ValueError, match="considered"):
        dataclasses.replace(
            make_spatial_observation(support=(3, 5, 9)),
            projection_summary=make_summary(considered=2, visible=2),
        )
    with pytest.raises(ValueError, match="visible_count"):
        dataclasses.replace(
            make_spatial_observation(support=(3, 5, 9)),
            projection_summary=make_summary(considered=100, visible=2),
        )


def test_a_region_scoped_feature_reference_must_belong_to_the_observed_region() -> None:
    other = VisualFeatureRef(
        feature_id=FeatureId("f-other"),
        embedding_space_id="alphaclip:vitl14",
        scope=FeatureScope.REGION,
        region_id=RegionId("region-0099"),
    )

    with pytest.raises(ValueError, match="region"):
        make_spatial_observation(region="region-0001", feature_refs=[other])


def test_the_provenance_names_every_policy_and_upstream_artifact() -> None:
    complete = make_provenance()

    for field_name in (
        "perception_run_id",
        "sequence_artifact_id",
        "visibility_policy_id",
        "membership_policy_id",
    ):
        empty: dict[str, Any] = {field_name: ""}
        with pytest.raises(ValueError, match=field_name):
            dataclasses.replace(complete, **empty)
    with pytest.raises(ValueError, match="geometric_map_id"):
        dataclasses.replace(complete, geometric_map_id=MapId(""))


# --- References, calibration and pose ---------------------------------------


def test_a_region_scoped_feature_reference_names_its_region() -> None:
    with pytest.raises(ValueError, match="region_id"):
        VisualFeatureRef(
            feature_id=FeatureId("f"),
            embedding_space_id="clip",
            scope=FeatureScope.REGION,
            region_id=None,
        )
    with pytest.raises(ValueError, match="region_id"):
        VisualFeatureRef(
            feature_id=FeatureId("f"),
            embedding_space_id="clip",
            scope=FeatureScope.GLOBAL,
            region_id=RegionId("r"),
        )
    with pytest.raises(ValueError, match="embedding_space_id"):
        VisualFeatureRef(
            feature_id=FeatureId("f"),
            embedding_space_id="",
            scope=FeatureScope.DENSE,
            region_id=None,
        )


def test_a_claim_reference_needs_its_identity() -> None:
    with pytest.raises(ValueError, match="claim_id"):
        SemanticClaimRef(claim_id=ClaimId(""))


def test_calibration_reference_identifies_the_calibration_and_camera() -> None:
    reference = make_calibration_ref()

    assert reference.camera_frame == FrameId("rgb_camera")
    assert reference.camera_calibration_id == CalibrationReferenceId("camera0")
    with pytest.raises(ValueError, match="calibration_identity"):
        dataclasses.replace(reference, calibration_identity="")
    with pytest.raises(ValueError, match="camera_model_kind"):
        CalibrationRef(
            calibration_identity="x",
            camera_calibration_id=CalibrationReferenceId("c"),
            camera_model_kind="",
            camera_frame=FrameId("f"),
        )


def test_an_interpolated_pose_reference_names_both_source_estimates() -> None:
    pose = make_pose_ref(interpolated=True)

    assert pose.lookup_outcome is LookupOutcome.INTERPOLATED
    assert len(pose.source_estimate_ids) == 2
    assert pose.interpolation_fraction == 0.25
    with pytest.raises(ValueError, match="interpolat"):
        dataclasses.replace(pose, source_estimate_ids=(PoseEstimateId("a"),))
    with pytest.raises(ValueError, match="interpolat"):
        dataclasses.replace(pose, interpolation_fraction=None)


def test_a_direct_pose_reference_has_one_source_and_no_fraction() -> None:
    pose = make_pose_ref(interpolated=False)

    assert (pose.lookup_outcome, len(pose.source_estimate_ids), pose.interpolation_fraction) == (
        LookupOutcome.EXACT,
        1,
        None,
    )
    with pytest.raises(ValueError, match="interpolat"):
        dataclasses.replace(pose, interpolation_fraction=0.5)
    with pytest.raises(ValueError, match="time_delta_ns"):
        dataclasses.replace(pose, time_delta_ns=-1)
    with pytest.raises(ValueError, match="source_estimate_ids"):
        dataclasses.replace(pose, source_estimate_ids=())


# --- Lower-level point records ----------------------------------------------


def _correspondence(state: VisibilityState, **overrides: object) -> PointCorrespondence:
    values: dict[str, object] = {
        "geometry": refs((3,))[0],
        "visibility": state,
        "camera_depth_m": 4.2,
        "raw_pixel": (320.5, 240.25),
        "prepared_pixel": (160.25, 120.125),
        "support_depth_m": 4.1,
    }
    values.update(overrides)
    return PointCorrespondence(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "state",
    [
        VisibilityState.OUTSIDE_VALID_SUPPORT,
        VisibilityState.OCCLUDED,
        VisibilityState.VISIBLE_UNASSIGNED,
        VisibilityState.ASSOCIATED,
    ],
)
def test_a_point_that_reaches_the_prepared_image_carries_its_pixel(state: VisibilityState) -> None:
    record = _correspondence(state, support_depth_m=3.0)

    assert record.prepared_pixel == (160.25, 120.125)
    with pytest.raises(ValueError, match="prepared_pixel"):
        _correspondence(state, prepared_pixel=None)


def test_a_depth_is_negative_only_for_a_point_behind_the_camera() -> None:
    with pytest.raises(ValueError, match="camera_depth_m"):
        _correspondence(VisibilityState.ASSOCIATED, camera_depth_m=-1.0)
    with pytest.raises(ValueError, match="camera_depth_m"):
        _correspondence(VisibilityState.OUTSIDE_IMAGE, camera_depth_m=-1.0, support_depth_m=None)
    with pytest.raises(ValueError, match="support_depth_m"):
        _correspondence(VisibilityState.ASSOCIATED, support_depth_m=-0.1)
    behind = _correspondence(
        VisibilityState.BEHIND_CAMERA,
        camera_depth_m=-2.0,
        raw_pixel=None,
        prepared_pixel=None,
        support_depth_m=None,
    )
    assert behind.camera_depth_m == -2.0


def test_a_point_the_camera_model_cannot_project_has_no_pixel_and_no_support() -> None:
    behind = _correspondence(
        VisibilityState.BEHIND_CAMERA,
        camera_depth_m=2.0,
        raw_pixel=None,
        prepared_pixel=None,
        support_depth_m=None,
    )

    assert behind.prepared_pixel is None
    with pytest.raises(ValueError, match="prepared_pixel"):
        _correspondence(VisibilityState.BEHIND_CAMERA, camera_depth_m=2.0)


def test_a_point_outside_the_image_may_keep_the_pixel_the_model_produced() -> None:
    kept = _correspondence(
        VisibilityState.OUTSIDE_IMAGE,
        raw_pixel=(700.0, 10.0),
        prepared_pixel=(350.0, 5.0),
        support_depth_m=None,
    )
    dropped = _correspondence(
        VisibilityState.OUTSIDE_IMAGE, raw_pixel=None, prepared_pixel=None, support_depth_m=None
    )

    assert kept.prepared_pixel == (350.0, 5.0)
    assert dropped.prepared_pixel is None
    with pytest.raises(ValueError, match="support_depth_m"):
        _correspondence(VisibilityState.OUTSIDE_IMAGE, prepared_pixel=None, raw_pixel=None)


def test_an_occluded_point_names_the_nearer_surface_that_hides_it() -> None:
    occluded = _correspondence(VisibilityState.OCCLUDED, camera_depth_m=6.0, support_depth_m=3.5)

    assert occluded.support_depth_m == 3.5
    with pytest.raises(ValueError, match="support_depth_m"):
        _correspondence(VisibilityState.OCCLUDED, support_depth_m=None)
    with pytest.raises(ValueError, match="support_depth_m"):
        _correspondence(VisibilityState.OCCLUDED, camera_depth_m=4.0, support_depth_m=4.0)


def test_pixels_and_depths_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        _correspondence(VisibilityState.ASSOCIATED, camera_depth_m=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _correspondence(VisibilityState.ASSOCIATED, prepared_pixel=(float("inf"), 1.0))
    with pytest.raises(ValueError, match="finite"):
        _correspondence(VisibilityState.ASSOCIATED, raw_pixel=(1.0, float("nan")))


def test_a_pixel_coordinate_is_a_pair_of_floats() -> None:
    pixel: PixelCoordinate = (1.5, 2.5)

    assert len(pixel) == 2

import dataclasses
import json
import math

import pytest
from perception_builders import make_region, make_result, rect_mask
from projection_builders import (
    IDENTITY,
    map_point_for_pixel,
    project_frame,
    scene_frame,
)

import contextmap.sensor_association as sensor_association
from contextmap.sensor_association import (
    DepthMetric,
    ObservationQuality,
    QualityComponent,
    ReprojectionStatistics,
    SpatialObservation,
    ValueSummary,
)
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.frame_projection import FrameProjection
from contextmap.sensor_association.membership import (
    FrameMembership,
    associate_regions,
    build_spatial_observations,
)
from contextmap.sensor_association.quality import QUALITY_DEFINITIONS_VERSION
from contextmap.sensor_association.quality_derivation import derive_observation_quality
from contextmap.sensor_association.serialization import (
    decode_observation_quality,
    encode_observation_quality,
)
from contextmap.sensor_association.visibility import OcclusionPolicy, resolve_visibility
from contextmap.state_estimation import LookupOutcome, LookupPolicy

POLICY = OcclusionPolicy(
    cell_size_px=4,
    neighborhood_radius_cells=2,
    depth_margin_m=0.1,
    depth_margin_ratio=0.02,
)
REFERENCE = ReprojectionStatistics(
    reference_id="trusted-correspondences-0001",
    correspondence_count=50,
    invalid_count=2,
    mean_px=1.2,
    median_px=0.9,
    p95_px=3.4,
    max_px=7.7,
)

# Região A: três pontos associados (z = 2, 3 e 4), um ocluído atrás do segundo e um visível fora.
POINTS = [(100, 100, 2.0), (150, 60, 3.0), (200, 100, 4.0), (150, 60, 8.0), (400, 300, 3.0)]


def _scene() -> tuple[FrameMembership, tuple[SpatialObservation, ...]]:
    return _associate(scene_frame(*POINTS))


def _associate(
    frame: FrameProjection,
) -> tuple[FrameMembership, tuple[SpatialObservation, ...]]:
    result = make_result(
        [
            make_region("region-A", rect_mask(640, 480, 90, 40, 210, 110)),
            make_region("region-B", rect_mask(640, 480, 500, 400, 520, 420)),
        ]
    )
    membership = associate_regions(resolve_visibility(frame, POLICY), result)
    observations = build_spatial_observations(
        membership, configuration_fingerprint="sha256:cfg", code_version="test"
    )
    return membership, observations


def _quality(
    reprojection: ReprojectionStatistics | None = None,
) -> tuple[SpatialObservation, ObservationQuality, ObservationQuality]:
    membership, observations = _scene()
    quality_a, quality_b = derive_observation_quality(
        membership, observations, reprojection=reprojection
    )
    return observations[0], quality_a, quality_b


# --- Components, reproducible from the association inputs -------------------


def test_the_range_components_summarize_the_associated_support() -> None:
    _, quality, _ = _quality()

    assert quality.depth_metric is DepthMetric.OPTICAL_AXIS
    depth = quality.support_depth_m
    assert depth is not None
    assert (depth.count, depth.minimum, depth.median, depth.maximum) == pytest.approx(
        (3, 2.0, 3.0, 4.0)
    )
    angles = sorted(
        math.atan2(math.hypot(u - 320.0, v - 240.0), 500.0)
        for u, v, _ in (POINTS[0], POINTS[1], POINTS[2])
    )
    summary = quality.support_off_axis_angle_rad
    assert summary is not None
    assert (summary.count, summary.minimum, summary.median, summary.maximum) == pytest.approx(
        (3, angles[0], angles[1], angles[2])
    )


def test_the_image_region_component_is_the_distance_to_the_prepared_image_border() -> None:
    _, quality, _ = _quality()

    # Bordas: (100.5, 100.5) -> 100.5; (150.5, 60.5) -> 60.5; (200.5, 100.5) -> 100.5.
    border = quality.border_distance_px
    assert border is not None
    assert (border.count, border.minimum, border.median, border.maximum) == pytest.approx(
        (3, 60.5, 100.5, 100.5)
    )


def test_the_visibility_and_support_components_come_from_the_region_footprint() -> None:
    _, quality, _ = _quality()

    assert quality.associated_count == 3
    assert quality.footprint_count == 4
    assert quality.visible_share == pytest.approx(3 / 4)
    assert quality.occluded_fraction == pytest.approx(1 / 4)
    assert quality.outside_valid_support_fraction == 0.0
    assert quality.mask_area_px == 120 * 70
    assert quality.support_density_per_mask_pixel == pytest.approx(3 / (120 * 70))
    assert quality.support_pixel_coverage == pytest.approx(3 / (120 * 70))


def test_the_timing_component_is_the_pose_lookup_delta_and_interpolation() -> None:
    frame = project_frame(
        [map_point_for_pixel(320, 240, 4.0)],
        time_ns=25_000_000,
        poses=[(0, (0.0, 0.0, 0.0), IDENTITY), (100_000_000, (1.0, 0.0, 0.0), IDENTITY)],
        pose_policy=LookupPolicy.interpolated(),
    )
    result = make_result([make_region("region-A", rect_mask(640, 480, 300, 220, 340, 260))])
    membership = associate_regions(resolve_visibility(frame, POLICY), result)
    observations = build_spatial_observations(
        membership, configuration_fingerprint=None, code_version=None
    )

    (quality,) = derive_observation_quality(membership, observations)

    assert quality.temporal_offset_ns == 25_000_000
    assert quality.pose_ref.lookup_outcome is LookupOutcome.INTERPOLATED
    assert quality.pose_ref.interpolation_fraction == pytest.approx(0.25)


def test_the_same_inputs_give_the_same_quality() -> None:
    first = _quality()[1]
    second = _quality()[1]

    assert first == second


# --- Missing measurements stay explicit -------------------------------------


def test_a_region_with_no_associated_geometry_marks_the_unavailable_components() -> None:
    _, _, quality = _quality()

    assert quality.associated_count == 0
    assert quality.footprint_count == 0
    assert quality.support_density_per_mask_pixel == 0.0
    assert quality.support_depth_m is None
    assert quality.support_off_axis_angle_rad is None
    assert quality.border_distance_px is None
    assert quality.visible_share is None
    assert quality.occluded_fraction is None
    assert quality.outside_valid_support_fraction is None
    assert set(quality.unavailable) == {
        QualityComponent.SUPPORT_DEPTH,
        QualityComponent.SUPPORT_OFF_AXIS_ANGLE,
        QualityComponent.BORDER_DISTANCE,
        QualityComponent.VISIBLE_SHARE,
        QualityComponent.OCCLUDED_FRACTION,
        QualityComponent.OUTSIDE_VALID_SUPPORT_FRACTION,
        QualityComponent.REPROJECTION,
    }
    assert all(reason for reason in quality.unavailable.values())


def test_the_reprojection_residual_is_never_fabricated() -> None:
    _, without, _ = _quality()
    _, with_reference, _ = _quality(REFERENCE)

    assert without.reprojection is None
    assert "trusted" in without.unavailable[QualityComponent.REPROJECTION]
    assert with_reference.reprojection == REFERENCE
    assert QualityComponent.REPROJECTION not in with_reference.unavailable


def test_the_availability_of_every_component_is_consistent() -> None:
    _, quality, _ = _quality()

    with pytest.raises(ValueError, match="support_depth"):
        dataclasses.replace(quality, support_depth_m=None)
    with pytest.raises(ValueError, match="visible_share"):
        dataclasses.replace(
            quality, unavailable={**quality.unavailable, QualityComponent.VISIBLE_SHARE: "why"}
        )
    with pytest.raises(ValueError, match="reason"):
        dataclasses.replace(
            quality, reprojection=None, unavailable={QualityComponent.REPROJECTION: ""}
        )


# --- Not a semantic confidence ----------------------------------------------


def test_the_quality_cannot_be_confused_with_a_semantic_confidence_or_a_weight() -> None:
    names = {field.name for field in dataclasses.fields(ObservationQuality)}

    assert not names & {"confidence", "score", "probability", "weight", "similarity", "label"}
    assert not any(
        hasattr(ObservationQuality, name)
        for name in ("combined_score", "overall", "weight", "as_confidence", "to_weight")
    )
    claim_fields = {f.name for f in dataclasses.fields(sensor_association.SemanticClaimRef)}
    assert claim_fields == {"claim_id"}


def test_the_components_keep_their_own_units_and_are_not_collapsed() -> None:
    _, quality, _ = _quality()
    scalar_fields = [
        f.name
        for f in dataclasses.fields(quality)
        if isinstance(getattr(quality, f.name), float)
        and not isinstance(getattr(quality, f.name), bool)
    ]

    # Só razões e densidades escalares; nenhuma nota única resume o conjunto.
    assert set(scalar_fields) <= {
        "support_density_per_mask_pixel",
        "support_pixel_coverage",
        "visible_share",
        "occluded_fraction",
        "outside_valid_support_fraction",
    }


# --- Provenance -------------------------------------------------------------


def test_every_quality_is_tied_to_its_observation_and_the_exact_inputs() -> None:
    observation, quality, _ = _quality()

    assert quality.definitions_version == QUALITY_DEFINITIONS_VERSION == "observation-quality-v1"
    assert quality.spatial_observation_id == observation.spatial_observation_id
    assert quality.source_observation_id == observation.source_observation_id
    assert quality.geometric_map_id == observation.provenance.geometric_map_id
    assert quality.calibration_ref == observation.calibration_ref
    assert quality.pose_ref == observation.pose_ref
    assert quality.image_transform_id == observation.projection_summary.image_transform_id
    assert quality.visibility_policy_id == observation.provenance.visibility_policy_id


def test_regions_of_one_physical_frame_share_its_identity_so_they_are_not_new_views() -> None:
    _, quality_a, quality_b = _quality()

    assert quality_a.source_observation_id == quality_b.source_observation_id
    assert quality_a.spatial_observation_id != quality_b.spatial_observation_id


def test_the_observations_must_belong_to_the_membership() -> None:
    membership, observations = _scene()

    with pytest.raises(AssociationInputError, match="observations"):
        derive_observation_quality(membership, observations[:1])
    with pytest.raises(AssociationInputError, match="observations"):
        derive_observation_quality(membership, tuple(reversed(observations)))


# --- Contract validation ----------------------------------------------------


def test_the_value_summaries_are_ordered_and_finite() -> None:
    with pytest.raises(ValueError, match="count"):
        ValueSummary(count=0, minimum=1.0, median=1.0, maximum=1.0)
    with pytest.raises(ValueError, match="order"):
        ValueSummary(count=2, minimum=3.0, median=2.0, maximum=4.0)
    with pytest.raises(ValueError, match="finite"):
        ValueSummary(count=2, minimum=1.0, median=float("nan"), maximum=4.0)


def test_the_reprojection_statistics_are_consistent() -> None:
    with pytest.raises(ValueError, match="invalid_count"):
        dataclasses.replace(REFERENCE, invalid_count=51)
    with pytest.raises(ValueError, match="order"):
        dataclasses.replace(REFERENCE, median_px=5.0)
    with pytest.raises(ValueError, match="reference_id"):
        dataclasses.replace(REFERENCE, reference_id="")


def test_the_fractions_and_counts_are_bounded() -> None:
    _, quality, _ = _quality()

    with pytest.raises(ValueError, match="visible_share"):
        dataclasses.replace(quality, visible_share=1.5)
    with pytest.raises(ValueError, match="associated_count"):
        dataclasses.replace(quality, associated_count=9)


# --- Serialization ----------------------------------------------------------


def test_a_quality_survives_a_json_round_trip_including_what_is_unavailable() -> None:
    _, with_data, without_data = _quality(REFERENCE)

    for quality in (with_data, without_data):
        record = json.loads(json.dumps(encode_observation_quality(quality)))
        assert decode_observation_quality(record) == quality
    assert (
        encode_observation_quality(without_data)["unavailable"]["support_depth"]
        == without_data.unavailable[QualityComponent.SUPPORT_DEPTH]
    )
    assert encode_observation_quality(without_data)["support_depth_m"] is None


def test_decoding_revalidates_the_quality() -> None:
    _, quality, _ = _quality()
    record = json.loads(json.dumps(encode_observation_quality(quality)))
    record["visible_share"] = 2.0

    with pytest.raises(ValueError, match="visible_share"):
        decode_observation_quality(record)


def test_semantic_fusion_reads_the_quality_from_the_public_api_alone() -> None:
    public = set(sensor_association.__all__)

    assert {
        "ObservationQuality",
        "QualityComponent",
        "ReprojectionStatistics",
        "ValueSummary",
    } <= public

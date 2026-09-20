"""A small, deterministic association run for the service and artifact tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from dense_builders import make_dense_map, make_enhancement, make_sampling
from perception_builders import make_claim, make_feature, make_region, make_result, rect_mask
from projection_builders import (
    IDENTITY,
    SEQUENCE_ID,
    ArrayGeometrySource,
    make_calibration,
    make_camera_observation,
    make_lookup,
    make_prepared_image,
    make_trajectory,
    map_point_for_pixel,
)

from contextmap.ingestion import SourceObservationId
from contextmap.sensor_association.dense_sampling import InterpolationPolicy
from contextmap.sensor_association.diagnostics import DiagnosticTolerances, TrustedCorrespondences
from contextmap.sensor_association.service import (
    AssociationFrameInput,
    DenseChannel,
    SensorAssociationRequest,
)
from contextmap.sensor_association.visibility import OcclusionPolicy
from contextmap.state_estimation import LookupPolicy
from contextmap.visual_perception import (
    DenseFeatureMap,
    FeatureScope,
    PerceptionResultId,
)

OCCLUSION = OcclusionPolicy(
    cell_size_px=4,
    neighborhood_radius_cells=2,
    depth_margin_m=0.1,
    depth_margin_ratio=0.02,
)
TOLERANCES = DiagnosticTolerances(
    max_pose_time_delta_ns=20_000_000,
    max_map_window_offset_ns=10_000_000,
    max_reprojection_p95_px=5.0,
    max_reprojection_invalid_rate=0.25,
)
# Duas regiões; o ponto de fundo (150, 100, z = 8) fica atrás do (150, 100, z = 3).
SCENE_PIXELS = [(100, 100, 3.0), (150, 100, 3.0), (150, 100, 8.0), (240, 100, 3.0), (400, 300, 3.0)]
FRAME_TIMES_NS = (0, 100_000_000)
NATIVE = DenseChannel(channel_id="dino-native", interpolation=InterpolationPolicy.NEAREST)
ENHANCED = DenseChannel(channel_id="dino-enhanced", interpolation=InterpolationPolicy.BILINEAR)


def frame_id(index: int) -> str:
    return f"frame-{index:04d}"


def dense_maps(index: int, channels: Sequence[DenseChannel]) -> dict[str, DenseFeatureMap]:
    native = make_dense_map(feature_id=f"dense-native-{index}")
    maps: dict[str, DenseFeatureMap] = {}
    for channel in channels:
        if channel.channel_id == NATIVE.channel_id:
            maps[channel.channel_id] = native
        else:
            fine = make_sampling(
                (80, 60), stride=(8.0, 8.0), support=(8.0, 8.0), transform_id="enhanced-grid-v1"
            )
            maps[channel.channel_id] = make_dense_map(
                fine,
                feature_id=f"dense-enhanced-{index}",
                enhancement=make_enhancement(native, fine),
            )
    return maps


def frame_input(
    index: int,
    *,
    time_ns: int | None = None,
    channels: Sequence[DenseChannel] = (),
    with_reference: bool = False,
) -> AssociationFrameInput:
    observation_id = frame_id(index)
    result_id = PerceptionResultId(f"run-0001--{observation_id}")
    maps = dense_maps(index, channels)
    features = [dense_map.feature for dense_map in maps.values()]
    claims = []
    if index == 0:
        features.append(make_feature("f-a", FeatureScope.REGION, region_id="region-A"))
        claims.append(
            make_claim(
                "c-a",
                region_id="region-A",
                observation_id=SourceObservationId(observation_id),
                result_id=result_id,
            )
        )
    result = make_result(
        [
            make_region("region-A", rect_mask(640, 480, 90, 90, 210, 110)),
            make_region("region-B", rect_mask(640, 480, 140, 90, 260, 110)),
        ],
        features=features,
        claims=claims,
        observation_id=SourceObservationId(observation_id),
        result_id=result_id,
    )
    correspondences = None
    if with_reference:
        # Referência confiável: a projeção exata dos quatro primeiros pontos, deslocada de 1 px.
        expected = np.array([[100.0, 100.0], [150.0, 100.0], [150.0, 100.0], [240.0, 100.0]])
        correspondences = TrustedCorrespondences(
            reference_id="trusted-0001",
            geometry_indices=np.arange(4),
            observed_pixels=expected + np.array([1.0, 0.0]),
        )
    return AssociationFrameInput(
        observation=make_camera_observation(
            FRAME_TIMES_NS[index] if time_ns is None else time_ns, observation_id=observation_id
        ),
        prepared_image=make_prepared_image(source_observation_id=observation_id),
        perception_result=result,
        dense_maps=maps,
        correspondences=correspondences,
    )


def make_request(
    *,
    frames: Sequence[AssociationFrameInput] | None = None,
    channels: Sequence[DenseChannel] = (),
    occlusion: OcclusionPolicy = OCCLUSION,
    pose_policy: LookupPolicy | None = None,
) -> SensorAssociationRequest:
    calibration = make_calibration()
    source = ArrayGeometrySource(
        [map_point_for_pixel(u, v, z) for u, v, z in SCENE_PIXELS], calibration=calibration
    )
    trajectory = make_trajectory(
        calibration,
        [(0, (0.0, 0.0, 0.0), IDENTITY), (100_000_000, (0.0, 0.0, 0.0), IDENTITY)],
    )
    return SensorAssociationRequest(
        sequence_artifact_id=SEQUENCE_ID,
        selection_id="full-sequence",
        geometry=source,
        trajectory=make_lookup(trajectory),
        pose_policy=pose_policy if pose_policy is not None else LookupPolicy.exact(),
        calibration=calibration,
        occlusion_policy=occlusion,
        tolerances=TOLERANCES,
        dense_channels=tuple(channels),
        frames=tuple(
            frames
            if frames is not None
            else [frame_input(0, channels=channels), frame_input(1, channels=channels)]
        ),
        code_version="test",
    )

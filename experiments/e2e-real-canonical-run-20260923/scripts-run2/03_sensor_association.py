"""Real sensor_association over run-0002's trajectory/geometry + the reused real perception run.

The PerceptionRunArtifact is not regenerated: it was produced through the direct-script path
that already preserved SemanticInterpretationExecution/view payloads (unaffected by the #555
schema bump), and finding #2's fix already has integration coverage via TDD (issue #438 review).
"""

from __future__ import annotations

from pathlib import Path

from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import SensorAssociationExecutor
from contextmap.sensor_association import (
    DiagnosticTolerances,
    OcclusionPolicy,
    SensorAssociationRunReader,
)
from contextmap.state_estimation import LookupPolicy

WORKSPACE = Path(__file__).resolve().parents[3] / "outputs"
BAG_ARTIFACT_ID = "720a486de8d44c16a9d3d2ff9fa7b1a4"
STATE_ESTIMATION_ARTIFACT_ID = "ee9cf3f2702f9ff1b18a74f7dcadc5fb"
GEOMETRIC_MAP_ARTIFACT_ID = "66ad39c22da8ee74384220942337e535"
PERCEPTION_LOCATION = "e2e-real/visual_perception/workspace/corridor-02/run-0001/visual_perception"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0002/sensor_association"


def main() -> None:
    request = StageRequest(
        stage_id="sensor_association",
        inputs={
            "sequence": (
                ArtifactRef(
                    stage_id="ingestion",
                    contract="SequenceArtifact",
                    artifact_id=BAG_ARTIFACT_ID,
                    content_hash="sha256:bag-content",
                    location=f"ingest-real/sequences/corridor-02/{BAG_ARTIFACT_ID}",
                ),
            ),
            "trajectory": (
                ArtifactRef(
                    stage_id="state_estimation",
                    contract="StateEstimationRunArtifact",
                    artifact_id=STATE_ESTIMATION_ARTIFACT_ID,
                    content_hash="sha256:trajectory-content",
                    location="e2e-real/run-0002/state_estimation",
                ),
            ),
            "geometry": (
                ArtifactRef(
                    stage_id="geometric_mapping",
                    contract="GeometricMapArtifact",
                    artifact_id=GEOMETRIC_MAP_ARTIFACT_ID,
                    content_hash="sha256:geometry-content",
                    location="e2e-real/run-0002/geometric_mapping",
                ),
            ),
            "perception": (
                ArtifactRef(
                    stage_id="visual_perception",
                    contract="PerceptionRunArtifact",
                    artifact_id="vp-sam2-canonical-1-0-4",
                    content_hash="sha256:perception-content",
                    location=PERCEPTION_LOCATION,
                ),
            ),
        },
        components={},
        config_digest="e2e-real-canonical-run-0002",
        output_dir=OUTPUT_DIR,
        workspace=WORKSPACE,
    )

    executor = SensorAssociationExecutor(
        occlusion=OcclusionPolicy(
            cell_size_px=4, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02
        ),
        tolerances=DiagnosticTolerances(
            max_pose_time_delta_ns=250_000_000,
            max_map_window_offset_ns=1_000_000_000,
            max_reprojection_p95_px=None,
            max_reprojection_invalid_rate=None,
        ),
        pose_policy=LookupPolicy.interpolated(max_interpolation_gap_ns=450_000_000),
    )

    ref = executor.execute(request)
    print("SensorAssociationRunArtifact published:", ref)

    reader = SensorAssociationRunReader(OUTPUT_DIR)
    problems = reader.verify_integrity()
    print("verify_integrity problems:", problems)


if __name__ == "__main__":
    main()

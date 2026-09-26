"""Real state_estimation over the real corridor-02 bag + pose artifacts (issue #177).

Merges the main bag SequenceArtifact with the auxiliary pose SequenceArtifact through
StateEstimationExecutor (issue #555's bridge), already validated once as a throwaway script;
this is the permanent version, writing into the coherent e2e-real run tree.
"""

from __future__ import annotations

from pathlib import Path

from contextmap.ingestion import FrameId
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import StateEstimationExecutor
from contextmap.state_estimation import StateEstimationRunReader
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
    InvalidSamplePolicy,
)

WORKSPACE = Path(__file__).resolve().parents[3] / "outputs"
BAG_ARTIFACT_ID = "720a486de8d44c16a9d3d2ff9fa7b1a4"
POSE_ARTIFACT_ID = "e2d832c152b1493999082d4f67210b5b"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0001-profiled/state_estimation"


def main() -> None:
    request = StageRequest(
        stage_id="state_estimation",
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
            "pose_sequence": (
                ArtifactRef(
                    stage_id="pose_ingestion",
                    contract="SequenceArtifact",
                    artifact_id=POSE_ARTIFACT_ID,
                    content_hash="sha256:pose-content",
                    location=f"ingest-real/sequences/corridor-02-pose/{POSE_ARTIFACT_ID}",
                ),
            ),
        },
        components={},
        config_digest="e2e-real-canonical-run-0001",
        output_dir=OUTPUT_DIR,
        workspace=WORKSPACE,
    )

    estimator = ExternalPoseEstimator(
        ExternalPoseConfig(
            reference_frame=FrameId("map"),
            body_frame=FrameId("epson"),
            invalid_sample_policy=InvalidSamplePolicy.SKIP,
        )
    )
    executor = StateEstimationExecutor(estimator, allow_ground_truth_trajectory=True)

    ref = executor.execute(request)
    print("StateEstimationRunArtifact published:", ref)

    run = StateEstimationRunReader(OUTPUT_DIR)
    trajectory = run.trajectory()
    print(f"trajectory poses: {len(trajectory.poses)}")
    print(f"trajectory gaps: {len(trajectory.gaps)}")
    print(f"reference_frame={trajectory.reference_frame} body_frame={trajectory.body_frame}")
    print(f"first pose ts: {trajectory.poses[0].timestamp}")
    print(f"last pose ts: {trajectory.poses[-1].timestamp}")


if __name__ == "__main__":
    main()

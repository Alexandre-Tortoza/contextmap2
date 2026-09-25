"""Real state_estimation, regenerated under the fixed #555 lineage contract (PR #438 review).

Supersedes run-0001's StateEstimationRunArtifact, which was produced under schema_version
"0.1.0" and never recorded which artifact its merged poses actually came from
(TrajectoryProvenance.auxiliary_sequence_artifact_id did not exist yet). Same real inputs
(the real corridor-02 bag + the real corridor-02-gt.txt pose sequence), same estimator/policy --
only the code changed. Asserts explicitly that the new manifest names the pose artifact, so this
evidence is objective, not just implied by the run finishing.
"""

from __future__ import annotations

from pathlib import Path

from contextmap.ingestion import FrameId, SequenceArtifactId
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import StateEstimationExecutor
from contextmap.state_estimation import StateEstimationRunReader
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
    InvalidSamplePolicy,
)

WORKSPACE = Path("/home/alexmrtr/Projects/contextmap2/outputs")
BAG_ARTIFACT_ID = "720a486de8d44c16a9d3d2ff9fa7b1a4"
POSE_ARTIFACT_ID = "e2d832c152b1493999082d4f67210b5b"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0003/state_estimation"


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
        config_digest="e2e-real-canonical-run-0003",
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
    manifest = run.manifest
    trajectory = run.trajectory()
    print(f"trajectory poses: {len(trajectory.poses)}")
    print(f"trajectory gaps: {len(trajectory.gaps)}")
    print(f"reference_frame={trajectory.reference_frame} body_frame={trajectory.body_frame}")
    print(f"first pose ts: {trajectory.poses[0].timestamp}")
    print(f"last pose ts: {trajectory.poses[-1].timestamp}")
    print(f"schema_version: {manifest.schema_version}")
    print(f"auxiliary_sequence_artifact_id: {manifest.auxiliary_sequence_artifact_id}")
    print(f"auxiliary_selection_id: {manifest.auxiliary_selection_id}")

    # Objective evidence for issue #555, not just an implication of the run finishing: this
    # scenario's trajectory must be named as coming from corridor-02-gt.txt's real
    # SequenceArtifact, not just "some auxiliary".
    assert manifest.auxiliary_sequence_artifact_id == SequenceArtifactId(POSE_ARTIFACT_ID), (
        f"expected the auxiliary pose sequence to be {POSE_ARTIFACT_ID!r}, "
        f"found {manifest.auxiliary_sequence_artifact_id!r}"
    )
    assert manifest.auxiliary_selection_id is not None
    print("ASSERTION PASSED: auxiliary_sequence_artifact_id matches corridor-02-gt.txt's artifact")


if __name__ == "__main__":
    main()

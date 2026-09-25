"""Real geometric_mapping over the regenerated run-0002 trajectory (PR #438 review rerun)."""

from __future__ import annotations

from pathlib import Path

from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    MotionCorrectionPolicy,
    ScanDisposition,
)
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import GeometricMappingExecutor
from contextmap.state_estimation import LookupPolicy

WORKSPACE = Path(__file__).resolve().parents[3] / "outputs"
BAG_ARTIFACT_ID = "720a486de8d44c16a9d3d2ff9fa7b1a4"
STATE_ESTIMATION_ARTIFACT_ID = "ee9cf3f2702f9ff1b18a74f7dcadc5fb"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0002/geometric_mapping"


def main() -> None:
    request = StageRequest(
        stage_id="geometric_mapping",
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
        },
        components={},
        config_digest="e2e-real-canonical-run-0002",
        output_dir=OUTPUT_DIR,
        workspace=WORKSPACE,
    )

    executor = GeometricMappingExecutor(
        pose_lookup=LookupPolicy.interpolated(max_interpolation_gap_ns=400_000_000),
        motion_correction=MotionCorrectionPolicy(
            raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN
        ),
    )

    ref = executor.execute(request)
    print("GeometricMapArtifact published:", ref)

    reader = GeometricMapArtifactReader(OUTPUT_DIR)
    problems = reader.verify_integrity()
    print("verify_integrity problems:", problems)
    manifest = reader.manifest
    print("map frame:", manifest.map_frame if hasattr(manifest, "map_frame") else "n/a")


if __name__ == "__main__":
    main()

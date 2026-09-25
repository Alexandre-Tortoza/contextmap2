"""Real geometric_mapping over the real corridor-02 sequence + trajectory (issue #177).

Policy values match the real precedent already used and validated in
outputs/validation/2026-09-21/geometric_mapping/scripts/03_run_geometric_mapping.py and
experiments/semantic-fusion-corridor-02-20260921/scripts/s02_geometric_mapping.py.
"""

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
STATE_ESTIMATION_ARTIFACT_ID = "b13c6e588aeef1e5142f890399e190b9"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0001-profiled/geometric_mapping"


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
                    location="e2e-real/run-0001/state_estimation",
                ),
            ),
        },
        components={},
        config_digest="e2e-real-canonical-run-0001",
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

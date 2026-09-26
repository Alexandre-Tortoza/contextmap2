"""Real semantic_fusion over run-0003's association + the reused real perception + geometry."""

from __future__ import annotations

from pathlib import Path

from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import SemanticFusionExecutor
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    GeometryOverlapSupportPolicy,
    SemanticFusionRunReader,
)

WORKSPACE = Path(__file__).resolve().parents[3] / "outputs"
BAG_ARTIFACT_ID = "720a486de8d44c16a9d3d2ff9fa7b1a4"
GEOMETRIC_MAP_ARTIFACT_ID = "e9f0a30db6be13ec8021caa243688d46"
SENSOR_ASSOCIATION_ARTIFACT_ID = "3ca14b10885481f5ae3abd1ad26f2ab8"
PERCEPTION_LOCATION = "e2e-real/visual_perception/workspace/corridor-02/run-0001/visual_perception"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0003/semantic_fusion"


def main() -> None:
    request = StageRequest(
        stage_id="semantic_fusion",
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
            "association": (
                ArtifactRef(
                    stage_id="sensor_association",
                    contract="SensorAssociationRunArtifact",
                    artifact_id=SENSOR_ASSOCIATION_ARTIFACT_ID,
                    content_hash="sha256:association-content",
                    location="e2e-real/run-0003/sensor_association",
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
            "geometry": (
                ArtifactRef(
                    stage_id="geometric_mapping",
                    contract="GeometricMapArtifact",
                    artifact_id=GEOMETRIC_MAP_ARTIFACT_ID,
                    content_hash="sha256:geometry-content",
                    location="e2e-real/run-0003/geometric_mapping",
                ),
            ),
        },
        components={},
        config_digest="e2e-real-canonical-run-0003",
        output_dir=OUTPUT_DIR,
        workspace=WORKSPACE,
    )

    executor = SemanticFusionExecutor(
        support_policy=GeometryOverlapSupportPolicy(min_geometry_count=5, min_overlap=0.3),
        accumulation_policy=BaselineAccumulationPolicy(),
    )

    ref = executor.execute(request)
    print("SemanticFusionRunArtifact published:", ref)

    reader = SemanticFusionRunReader(OUTPUT_DIR)
    problems = reader.verify_integrity()
    print("verify_integrity problems:", problems)


if __name__ == "__main__":
    main()

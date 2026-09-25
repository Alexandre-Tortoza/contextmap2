"""Real semantic_mapping over run-0003's semantic_fusion run."""

from __future__ import annotations

from pathlib import Path

from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import SemanticMappingExecutor
from contextmap.semantic_mapping import (
    EntityMaterializationPolicy,
    GeometrySummaryPolicy,
    SemanticMapId,
    SemanticMappingRunReader,
)

WORKSPACE = Path(__file__).resolve().parents[3] / "outputs"
GEOMETRIC_MAP_ARTIFACT_ID = "e9f0a30db6be13ec8021caa243688d46"
SEMANTIC_FUSION_ARTIFACT_ID = "1d12efdeb2727ee1fa1fbb27772542c0"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0003/semantic_mapping"


def main() -> None:
    request = StageRequest(
        stage_id="semantic_mapping",
        inputs={
            "fusion": (
                ArtifactRef(
                    stage_id="semantic_fusion",
                    contract="SemanticFusionRunArtifact",
                    artifact_id=SEMANTIC_FUSION_ARTIFACT_ID,
                    content_hash="sha256:fusion-content",
                    location="e2e-real/run-0003/semantic_fusion",
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

    executor = SemanticMappingExecutor(
        policy=EntityMaterializationPolicy(
            geometry=GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.5)
        ),
        semantic_map_id=SemanticMapId("semantic-map-corridor-02-run0003"),
        code_digest="sha256:" + "e2" * 32,
        code_version="e2e-real-canonical-run-20260925",
    )

    ref = executor.execute(request)
    print("SemanticMappingRunArtifact published:", ref)

    reader = SemanticMappingRunReader(OUTPUT_DIR)
    problems = reader.verify_integrity()
    print("verify_integrity problems:", problems)


if __name__ == "__main__":
    main()

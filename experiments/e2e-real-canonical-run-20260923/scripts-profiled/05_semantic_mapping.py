"""Real semantic_mapping over the real semantic_fusion run (issue #177).

No real precedent exists anywhere for this stage on this dataset -- built directly from the
runtime's real composed executor, using tests/end_to_end/test_runtime_chain.py's synthetic-chain
construction as the only currently-endorsed real parameterization of EntityMaterializationPolicy/
GeometrySummaryPolicy in this codebase (geometry thresholds are reused as-is from that test, not
independently tuned for corridor-02's real scale -- flagged for follow-up).
"""

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

WORKSPACE = Path("/home/alexmrtr/Projects/contextmap2/outputs")
GEOMETRIC_MAP_ARTIFACT_ID = "d6ed712aa1a77b91dc93fd951273d938"
SEMANTIC_FUSION_ARTIFACT_ID = "7253ce0b68b142d58f52a528ace9af13"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0001-profiled/semantic_mapping"


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
                    location="e2e-real/run-0001/semantic_fusion",
                ),
            ),
            "geometry": (
                ArtifactRef(
                    stage_id="geometric_mapping",
                    contract="GeometricMapArtifact",
                    artifact_id=GEOMETRIC_MAP_ARTIFACT_ID,
                    content_hash="sha256:geometry-content",
                    location="e2e-real/run-0001/geometric_mapping",
                ),
            ),
        },
        components={},
        config_digest="e2e-real-canonical-run-0001",
        output_dir=OUTPUT_DIR,
        workspace=WORKSPACE,
    )

    executor = SemanticMappingExecutor(
        policy=EntityMaterializationPolicy(
            geometry=GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.5)
        ),
        semantic_map_id=SemanticMapId("semantic-map-corridor-02"),
        code_digest="sha256:" + "e2" * 32,
        code_version="e2e-real-canonical-run-20260923",
    )

    ref = executor.execute(request)
    print("SemanticMappingRunArtifact published:", ref)

    reader = SemanticMappingRunReader(OUTPUT_DIR)
    problems = reader.verify_integrity()
    print("verify_integrity problems:", problems)


if __name__ == "__main__":
    main()

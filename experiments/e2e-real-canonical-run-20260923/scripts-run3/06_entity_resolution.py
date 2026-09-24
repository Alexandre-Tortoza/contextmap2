"""Real entity_resolution over run-0003's semantic_mapping run."""

from __future__ import annotations

from pathlib import Path

from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    ComparisonChannels,
    ConservativeResolutionPolicy,
    EntityResolutionRunReader,
    GeometryComparisonPolicy,
    MatchChannel,
    MatchEvidenceBuilder,
)
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import EntityResolutionExecutor

WORKSPACE = Path("/home/alexmrtr/Projects/contextmap2/outputs")
SEMANTIC_MAPPING_ARTIFACT_ID = "a543b1dd08a8891fe0bae1a8b1f9a9eb"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0003/entity_resolution"


def main() -> None:
    request = StageRequest(
        stage_id="entity_resolution",
        inputs={
            "entities": (
                ArtifactRef(
                    stage_id="semantic_mapping",
                    contract="SemanticEntityArtifact",
                    artifact_id=SEMANTIC_MAPPING_ARTIFACT_ID,
                    content_hash="sha256:entities-content",
                    location="e2e-real/run-0003/semantic_mapping",
                ),
            ),
        },
        components={},
        config_digest="e2e-real-canonical-run-0003",
        output_dir=OUTPUT_DIR,
        workspace=WORKSPACE,
    )

    executor = EntityResolutionExecutor(
        retrieval=CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1),
        builder=MatchEvidenceBuilder(
            ComparisonChannels(
                geometry=GeometryComparisonPolicy(
                    min_shared_support_jaccard=0.5,
                    min_bounds_iou=0.5,
                    min_bounds_containment=0.9,
                    min_conflict_gap_m=0.5,
                    min_extent_ratio=0.3,
                )
            )
        ),
        resolution=ConservativeResolutionPolicy(
            use_channels=(MatchChannel.GEOMETRY,), min_supporting_channels=1
        ),
        code_version="e2e-real-canonical-run-20260925",
    )

    ref = executor.execute(request)
    print("EntityResolutionRunArtifact published:", ref)

    reader = EntityResolutionRunReader(OUTPUT_DIR)
    problems = reader.verify_integrity()
    print("verify_integrity problems:", problems)


if __name__ == "__main__":
    main()

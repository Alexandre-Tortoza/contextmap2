"""Real entity_resolution over the real semantic_mapping run (issue #177).

No real precedent exists anywhere for this stage on this dataset -- policy values reused as-is
from tests/end_to_end/test_runtime_chain.py's synthetic-chain construction (not independently
tuned for corridor-02's real scale -- flagged for follow-up).
"""

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
SEMANTIC_MAPPING_ARTIFACT_ID = "c9137c48ca2baa50425101497eee6947"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0001/entity_resolution"


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
                    location="e2e-real/run-0001/semantic_mapping",
                ),
            ),
        },
        components={},
        config_digest="e2e-real-canonical-run-0001",
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
        code_version="e2e-real-canonical-run-20260923",
    )

    ref = executor.execute(request)
    print("EntityResolutionRunArtifact published:", ref)

    reader = EntityResolutionRunReader(OUTPUT_DIR)
    problems = reader.verify_integrity()
    print("verify_integrity problems:", problems)


if __name__ == "__main__":
    main()

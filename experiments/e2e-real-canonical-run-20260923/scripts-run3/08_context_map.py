"""Real context_map assembly: the regenerated ContextMapArtifact for corridor-02 (run-0003).

Also exercises the #438 review's fix #3 (ContextMapExecutor now opens and verifies the
``sequence`` input against the geometry it was built over) on real data, implicitly: this
succeeds because the real bag artifact really is the one the real geometry was built from.
"""

from __future__ import annotations

from pathlib import Path

from contextmap.artifact import ContextMapArtifactReader
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import ContextMapExecutor
from contextmap.spatial_relations import AxisDirection

WORKSPACE = Path("/home/alexmrtr/Projects/contextmap2/outputs")
BAG_ARTIFACT_ID = "720a486de8d44c16a9d3d2ff9fa7b1a4"
GEOMETRIC_MAP_ARTIFACT_ID = "e9f0a30db6be13ec8021caa243688d46"
ENTITY_RESOLUTION_ARTIFACT_ID = "0318ef0a8f0893984f17b1ad9bec86fa"
SPATIAL_RELATIONS_ARTIFACT_ID = "24edca76f62def8f8c808e8c0a4ceb7c"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0003/context_map"


def main() -> None:
    request = StageRequest(
        stage_id="context_map",
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
            "geometry": (
                ArtifactRef(
                    stage_id="geometric_mapping",
                    contract="GeometricMapArtifact",
                    artifact_id=GEOMETRIC_MAP_ARTIFACT_ID,
                    content_hash="sha256:geometry-content",
                    location="e2e-real/run-0003/geometric_mapping",
                ),
            ),
            "entities": (
                ArtifactRef(
                    stage_id="entity_resolution",
                    contract="EntityResolutionRunArtifact",
                    artifact_id=ENTITY_RESOLUTION_ARTIFACT_ID,
                    content_hash="sha256:resolution-content",
                    location="e2e-real/run-0003/entity_resolution",
                ),
            ),
            "relations": (
                ArtifactRef(
                    stage_id="spatial_relations",
                    contract="SpatialRelationsRunArtifact",
                    artifact_id=SPATIAL_RELATIONS_ARTIFACT_ID,
                    content_hash="sha256:relations-content",
                    location="e2e-real/run-0003/spatial_relations",
                ),
            ),
        },
        components={},
        config_digest="e2e-real-canonical-run-0003",
        output_dir=OUTPUT_DIR,
        workspace=WORKSPACE,
    )

    executor = ContextMapExecutor(
        up_axis=AxisDirection.POSITIVE_Z, code_version="e2e-real-canonical-run-20260925"
    )

    ref = executor.execute(request)
    print("ContextMapArtifact published:", ref)

    reader = ContextMapArtifactReader.open(OUTPUT_DIR, verify_hashes=True)
    manifest = reader.manifest
    print("context_map_id:", manifest.context_map_id)
    print("entity_count:", manifest.entity_count)
    print("relation_count:", manifest.relation_count)
    print("content_identity:", manifest.content_identity)


if __name__ == "__main__":
    main()

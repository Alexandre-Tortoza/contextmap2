"""Real spatial_relations over the real entity_resolution + geometry (issue #177).

No real precedent exists anywhere for this stage on this dataset -- policy values reused as-is
from tests/end_to_end/test_runtime_chain.py's synthetic-chain construction (not independently
tuned for corridor-02's real scale -- flagged for follow-up). map_frame="map" and
up_axis=POSITIVE_Z/forward_axis=POSITIVE_X match corridor-02's real geometric map frame and the
standard ROS body-frame convention (Z up, X forward) -- ContextMapExecutor reuses this exact
up_axis next, never a value derived again.
"""

from __future__ import annotations

from pathlib import Path

from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import SpatialRelationsExecutor
from contextmap.semantic_mapping import GeometrySummaryPolicy
from contextmap.spatial_relations import (
    AxisDirection,
    CandidatePolicy,
    ContactPredicatePolicy,
    FrameConventions,
    GeometricPredicatePolicy,
    RelationPredicate,
    RelationsRunPolicies,
    SpatialRelationsRunReader,
)

WORKSPACE = Path("/home/alexmrtr/Projects/contextmap2/outputs")
GEOMETRIC_MAP_ARTIFACT_ID = "d6ed712aa1a77b91dc93fd951273d938"
ENTITY_RESOLUTION_ARTIFACT_ID = "07616fecdb889fb0cf492f230e812706"
OUTPUT_DIR = WORKSPACE / "e2e-real/run-0001/spatial_relations"

UP_AXIS = AxisDirection.POSITIVE_Z


def main() -> None:
    request = StageRequest(
        stage_id="spatial_relations",
        inputs={
            "entities": (
                ArtifactRef(
                    stage_id="entity_resolution",
                    contract="EntityResolutionRunArtifact",
                    artifact_id=ENTITY_RESOLUTION_ARTIFACT_ID,
                    content_hash="sha256:resolution-content",
                    location="e2e-real/run-0001/entity_resolution",
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

    executor = SpatialRelationsExecutor(
        code_version="e2e-real-canonical-run-20260923",
        policies=RelationsRunPolicies(
            frame_conventions=FrameConventions(
                map_frame="map",
                up_axis=UP_AXIS,
                forward_axis=AxisDirection.POSITIVE_X,
            ),
            candidate=CandidatePolicy(
                predicates=(RelationPredicate.NEXT_TO, RelationPredicate.TOUCHING),
                proximity_radius_m=0.6,
                directional_radius_m=2.0,
            ),
            geometric=GeometricPredicatePolicy(
                boundary_tolerance_m=0.02,
                next_to_max_gap_m=0.5,
                adjacent_penetration_m=0.05,
                containment_slack_m=0.05,
                directional_overlap_fraction=0.5,
            ),
            contact=ContactPredicatePolicy(
                contact_distance_m=0.05,
                contact_tolerance_m=0.02,
                min_contact_points=3,
                support_height_tolerance_m=0.05,
                support_footprint_fraction=0.5,
                leaning_min_tilt_deg=10.0,
                leaning_max_tilt_deg=80.0,
                tilt_tolerance_deg=2.0,
                leaning_min_vertical_overlap_m=0.3,
            ),
            geometry_summary=GeometrySummaryPolicy(
                sparse_point_threshold=3, connectivity_radius_m=0.5
            ),
        ),
    )

    ref = executor.execute(request)
    print("SpatialRelationsRunArtifact published:", ref)

    reader = SpatialRelationsRunReader(OUTPUT_DIR)
    problems = reader.verify_integrity()
    print("verify_integrity problems:", problems)


if __name__ == "__main__":
    main()

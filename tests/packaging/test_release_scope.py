"""The v0.1.0 scope freeze: every public schema version the release promises is pinned here.

``docs/release-v0.1.0.md`` freezes what v0.1.0 supports. A frozen scope that only lives in prose
drifts, so the schema versions it publishes are asserted against the code. A schema change after
the freeze fails this test, which is the intended behavior: it forces an explicit scope and
version revision instead of letting a release ship a contract nobody re-validated.

To change one: revise the scope document, decide whether the release version still applies, and
update the expectation here in the same change.
"""

from __future__ import annotations

from contextmap.artifact.versioning import CONTEXT_MAP_SCHEMA_VERSION
from contextmap.entity_resolution.run_artifact import SCHEMA_VERSION as ENTITY_RESOLUTION_RUN
from contextmap.evaluation import SCENARIO_ID, SCENARIO_VERSION
from contextmap.geometric_mapping.run_artifact import SCHEMA_VERSION as GEOMETRIC_MAPPING_RUN
from contextmap.ingestion.calibration import SCHEMA_VERSION as CALIBRATION
from contextmap.ingestion.diagnostics import SCHEMA_VERSION as INGESTION_DIAGNOSTICS
from contextmap.ingestion.sequence_artifact import SCHEMA_VERSION as SEQUENCE_ARTIFACT
from contextmap.ingestion.sequence_provenance import SCHEMA_VERSION as SEQUENCE_PROVENANCE
from contextmap.point_representation.run_artifact import SCHEMA_VERSION as POINT_REPRESENTATION_RUN
from contextmap.runtime.catalog import CANONICAL_PROFILE_ID
from contextmap.runtime.config import CONFIG_SCHEMA_VERSION
from contextmap.runtime.ingestion_service import INGESTION_REQUEST_SCHEMA_VERSION
from contextmap.runtime.lifecycle import RUN_SCHEMA_VERSION
from contextmap.runtime.pipeline import PLAN_SCHEMA_VERSION
from contextmap.runtime.reuse import REUSE_SCHEMA_VERSION
from contextmap.runtime.selection import CATALOG_SCHEMA_VERSION
from contextmap.semantic_fusion.run_artifact import SCHEMA_VERSION as SEMANTIC_FUSION_RUN
from contextmap.semantic_mapping.run_artifact import SCHEMA_VERSION as SEMANTIC_MAPPING_RUN
from contextmap.semantic_mapping.serialization import ENTITY_SCHEMA_VERSION
from contextmap.sensor_association.run_artifact import SCHEMA_VERSION as SENSOR_ASSOCIATION_RUN
from contextmap.spatial_relations.run_artifact import SCHEMA_VERSION as SPATIAL_RELATIONS_RUN
from contextmap.state_estimation.run_artifact import SCHEMA_VERSION as STATE_ESTIMATION_RUN
from contextmap.visual_perception.embedding_space import SCHEMA_VERSION as EMBEDDING_SPACE
from contextmap.visual_perception.feature_store import FEATURE_INDEX_SCHEMA_VERSION
from contextmap.visual_perception.mask_store import MASK_INDEX_SCHEMA_VERSION
from contextmap.visual_perception.pipeline import PIPELINE_SCHEMA_VERSION
from contextmap.visual_perception.run_artifact import SCHEMA_VERSION as PERCEPTION_RUN


def test_every_frozen_artifact_schema_version_matches_the_code() -> None:
    """One entry per public artifact of the canonical pipeline, in topology order."""
    frozen = {
        "SequenceArtifact": SEQUENCE_ARTIFACT,
        "CalibrationSet": CALIBRATION,
        "SequenceProvenance": SEQUENCE_PROVENANCE,
        "ingestion diagnostics": INGESTION_DIAGNOSTICS,
        "PerceptionRunArtifact": PERCEPTION_RUN,
        "perception pipeline": PIPELINE_SCHEMA_VERSION,
        "embedding space": EMBEDDING_SPACE,
        "feature index": FEATURE_INDEX_SCHEMA_VERSION,
        "mask index": MASK_INDEX_SCHEMA_VERSION,
        "StateEstimationRunArtifact": STATE_ESTIMATION_RUN,
        "GeometricMapArtifact": GEOMETRIC_MAPPING_RUN,
        "SensorAssociationRunArtifact": SENSOR_ASSOCIATION_RUN,
        "PointRepresentationRunArtifact": POINT_REPRESENTATION_RUN,
        "SemanticFusionRunArtifact": SEMANTIC_FUSION_RUN,
        "SemanticEntityArtifact": SEMANTIC_MAPPING_RUN,
        "entity serialization": ENTITY_SCHEMA_VERSION,
        "EntityResolutionRunArtifact": ENTITY_RESOLUTION_RUN,
        "SpatialRelationsRunArtifact": SPATIAL_RELATIONS_RUN,
    }

    assert frozen == {
        "SequenceArtifact": "0.2.0",
        "CalibrationSet": "0.1.0",
        "SequenceProvenance": "0.1.0",
        "ingestion diagnostics": "0.2.0",
        "PerceptionRunArtifact": "0.5.0",
        "perception pipeline": "0.2.0",
        "embedding space": "0.1.0",
        "feature index": "0.1.0",
        "mask index": "0.1.0",
        "StateEstimationRunArtifact": "0.2.0",
        "GeometricMapArtifact": "0.1.0",
        "SensorAssociationRunArtifact": "0.2.0",
        "PointRepresentationRunArtifact": "0.1.0",
        "SemanticFusionRunArtifact": "0.2.0",
        "SemanticEntityArtifact": "0.1.0",
        "entity serialization": "0.1.0",
        "EntityResolutionRunArtifact": "0.1.0",
        "SpatialRelationsRunArtifact": "0.1.0",
    }, (
        "a public artifact schema changed after the v0.1.0 scope freeze: revise "
        "docs/release-v0.1.0.md and decide whether the release version still applies"
    )


def test_the_context_map_artifact_schema_version_is_frozen() -> None:
    """The release's public product; its version is what a consumer reads first."""
    assert CONTEXT_MAP_SCHEMA_VERSION == "0.1.0"


def test_every_frozen_runtime_schema_version_matches_the_code() -> None:
    frozen = {
        "config": CONFIG_SCHEMA_VERSION,
        "plan": PLAN_SCHEMA_VERSION,
        "run": RUN_SCHEMA_VERSION,
        "catalog": CATALOG_SCHEMA_VERSION,
        "reuse": REUSE_SCHEMA_VERSION,
        "ingestion_request": INGESTION_REQUEST_SCHEMA_VERSION,
    }

    assert frozen == {
        "config": "0.1.0",
        "plan": "0.1.0",
        "run": "0.1.0",
        "catalog": "0.1.0",
        "reuse": "0.1.0",
        "ingestion_request": "0.1.0",
    }


def test_the_frozen_canonical_profile_and_release_scenario_are_the_ones_documented() -> None:
    """The scope names one topology and one acceptance contract; both are pinned."""
    assert CANONICAL_PROFILE_ID == "canonical/1"
    assert (SCENARIO_ID, SCENARIO_VERSION) == ("solution-1-canonical", "1.0.5")

"""The v0.1.0 scope freeze and how the code may move past it.

``docs/release-v0.1.0.md`` freezes what v0.1.0 shipped, and ``V0_1_0_*`` below is that record: it
never changes. A frozen scope that only lives in prose drifts, so every public schema version the
code writes is checked against it. A version may leave the record only with its transition
(``<name> <old> → <new>``) under ``[Não lançado]`` in ``CHANGELOG.md``: a schema change is never
silent, and the released artifacts stay described as they were. The frozen demo in
``examples/v0.1.0`` stays readable (``test_release_examples.py``).

The canonical profile and the acceptance scenario remain pinned: changing them is a revision of
the release contract itself.
"""

from __future__ import annotations

from pathlib import Path

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

CHANGELOG = Path(__file__).resolve().parents[2] / "CHANGELOG.md"

# O que a v0.1.0 entregou (docs/release-v0.1.0.md). É registro: nunca muda depois do release.
V0_1_0_ARTIFACTS = {
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
    "ContextMapArtifact": "0.1.0",
}
V0_1_0_RUNTIME = {
    "config": "0.1.0",
    "plan": "0.1.0",
    "run": "0.1.0",
    "catalog": "0.1.0",
    "reuse": "0.1.0",
    "ingestion_request": "0.1.0",
}


def _code_artifact_versions() -> dict[str, str]:
    """One entry per public artifact of the canonical pipeline, in topology order."""
    return {
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
        "ContextMapArtifact": CONTEXT_MAP_SCHEMA_VERSION,
    }


def _code_runtime_versions() -> dict[str, str]:
    return {
        "config": CONFIG_SCHEMA_VERSION,
        "plan": PLAN_SCHEMA_VERSION,
        "run": RUN_SCHEMA_VERSION,
        "catalog": CATALOG_SCHEMA_VERSION,
        "reuse": REUSE_SCHEMA_VERSION,
        "ingestion_request": INGESTION_REQUEST_SCHEMA_VERSION,
    }


def _unrecorded_changes(code: dict[str, str], frozen: dict[str, str], changelog: str) -> list[str]:
    """Every version that left the v0.1.0 record without its transition under ``[Não lançado]``.

    A transition is written ``<name> <v0.1.0 version> → <code version>``; backticks around the
    name are allowed. Only the unreleased section counts: a transition belongs to the code that
    has not shipped yet.
    """
    unreleased = changelog.split("## [Não lançado]", 1)[1].split("\n## [", 1)[0]
    unreleased = unreleased.replace("`", "")
    transitions = (
        f"{name} {frozen[name]} → {version}"
        for name, version in code.items()
        if version != frozen[name]
    )
    return [item for item in transitions if item not in unreleased]


def test_every_artifact_schema_version_is_v0_1_0_or_a_recorded_transition() -> None:
    code = _code_artifact_versions()

    assert code.keys() == V0_1_0_ARTIFACTS.keys()
    assert not _unrecorded_changes(code, V0_1_0_ARTIFACTS, CHANGELOG.read_text("utf-8")), (
        "a public artifact schema changed after v0.1.0 without its transition in "
        "CHANGELOG.md [Não lançado] (write '<name> <old> → <new>'); the v0.1.0 record in "
        "docs/release-v0.1.0.md never changes"
    )


def test_every_runtime_schema_version_is_v0_1_0_or_a_recorded_transition() -> None:
    code = _code_runtime_versions()

    assert code.keys() == V0_1_0_RUNTIME.keys()
    assert not _unrecorded_changes(code, V0_1_0_RUNTIME, CHANGELOG.read_text("utf-8"))


def test_the_frozen_canonical_profile_and_release_scenario_are_the_ones_documented() -> None:
    """The scope names one topology and one acceptance contract; both are pinned."""
    assert CANONICAL_PROFILE_ID == "canonical/1"
    assert (SCENARIO_ID, SCENARIO_VERSION) == ("solution-1-canonical", "1.0.5")


def test_a_schema_change_since_v0_1_0_must_name_its_transition_in_the_changelog() -> None:
    frozen = {"SpatialRelationsRunArtifact": "0.1.0", "GeometricMapArtifact": "0.1.0"}
    code = {"SpatialRelationsRunArtifact": "0.2.0", "GeometricMapArtifact": "0.1.0"}

    assert _unrecorded_changes(code, frozen, "## [Não lançado]\n\n## [0.1.0] - 2026-09-25\n") == [
        "SpatialRelationsRunArtifact 0.1.0 → 0.2.0"
    ]
    recorded = (
        "## [Não lançado]\n\n### Alterado\n\n"
        "- Schema do `SpatialRelationsRunArtifact` 0.1.0 → 0.2.0: exclusões limitadas.\n\n"
        "## [0.1.0] - 2026-09-25\n"
    )
    assert _unrecorded_changes(code, frozen, recorded) == []
    # Uma transição registrada numa release passada não vale para o código ainda não lançado.
    released_only = (
        "## [Não lançado]\n\n## [0.1.0] - 2026-09-25\n\n"
        "- `SpatialRelationsRunArtifact` 0.1.0 → 0.2.0\n"
    )
    assert _unrecorded_changes(code, frozen, released_only) == [
        "SpatialRelationsRunArtifact 0.1.0 → 0.2.0"
    ]

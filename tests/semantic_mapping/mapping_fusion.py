"""Real fused evidence for Semantic Mapping tests, built through Semantic Fusion itself."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evidence_builders import ClaimSpec, View, build_scenario, build_scenario_parts
from fusion_builders import MAP_ID
from fusion_run_fixtures import LINEAGE, make_run_fixture
from mapping_builders import CODE_DIGEST, SEMANTIC_MAP_ID, SUMMARY_POLICY
from mapping_geometry_fake import InMemoryGeometrySource

from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    ExcludedObservation,
    FusedEvidence,
    FusionOutcome,
    FusionSupport,
    SemanticFusionRunId,
    SemanticFusionRunManifest,
    SemanticFusionRunReader,
    SemanticFusionRunWriter,
    accumulate_baseline_evidence,
)
from contextmap.semantic_mapping import (
    CandidateRejection,
    Entity,
    EntityMaterializationPolicy,
    SemanticMapId,
    SemanticMappingRunId,
    SemanticMappingRunWriter,
    lineage_from_fusion_manifest,
    materialize_entities,
)

__all__ = [
    "ClaimSpec",
    "FusionRun",
    "View",
    "entity_from_outcome",
    "fuse",
    "outcomes_for",
    "write_fusion_run",
    "write_mapping_run",
    "write_run",
]


def fuse(
    views: Sequence[View], *, policy: BaselineAccumulationPolicy | None = None
) -> tuple[FusionSupport, FusedEvidence]:
    """Fuse the evidence of views that see the same geometry into one support."""
    scenario = build_scenario(views)
    evidence = accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        policy=policy,
        code_version="test",
    )
    return scenario.support, evidence


@dataclass(frozen=True)
class FusionRun:
    """A real, persisted Semantic Fusion run and the geometry its supports were built over."""

    run_dir: Path
    reader: SemanticFusionRunReader
    manifest: SemanticFusionRunManifest
    outcomes: tuple[FusionOutcome, ...]
    geometry: InMemoryGeometrySource


def write_run(
    workspace: Path,
    outcomes: Sequence[FusionOutcome],
    *,
    excluded: Sequence[ExcludedObservation] = (),
) -> FusionRun:
    """Persist fusion outcomes as a real run artifact and reopen it from disk."""
    run_dir = workspace / "semantic_fusion"
    manifest = SemanticFusionRunWriter(
        output_dir=run_dir,
        sequence_name="sequence-0001",
        run_id=SemanticFusionRunId("fusion-run-0001"),
        run_index=1,
        lineage=LINEAGE,
        code_version="test",
    ).write(outcomes, excluded=excluded)
    return FusionRun(
        run_dir=run_dir,
        reader=SemanticFusionRunReader(run_dir),
        manifest=manifest,
        outcomes=tuple(outcomes),
        geometry=InMemoryGeometrySource(
            MAP_ID, {index: (index * 0.1, 0.0, 0.0) for index in range(1_000)}
        ),
    )


def write_fusion_run(workspace: Path) -> FusionRun:
    """Write the three-support fusion run of the fusion fixtures and reopen it from disk."""
    fixture = make_run_fixture()
    return write_run(workspace, fixture.outcomes, excluded=fixture.excluded)


def outcomes_for(
    views: Sequence[View], *, policy: BaselineAccumulationPolicy | None = None
) -> tuple[FusionOutcome, ...]:
    """Fuse views into as many supports as their geometry produces, one outcome per support."""
    parts = build_scenario_parts(views, geometry_points=1_000)
    return tuple(
        FusionOutcome(
            support=support,
            evidence=accumulate_baseline_evidence(
                support,
                observations=parts.observations,
                grouping=parts.grouping,
                perception_results=parts.results,
                policy=policy,
                code_version="test",
            ),
        )
        for support in parts.supports
    )


def entity_from_outcome(
    outcome: FusionOutcome,
    run: FusionRun,
    *,
    semantic_map_id: SemanticMapId = SEMANTIC_MAP_ID,
) -> Entity:
    """Materialize the entity of one fusion outcome through the real service."""
    result = materialize_entities(
        [outcome],
        fusion_manifest=run.manifest,
        geometry=run.geometry,
        semantic_map_id=semantic_map_id,
        policy=EntityMaterializationPolicy(geometry=SUMMARY_POLICY),
        code_version="test",
    )
    (entity,) = result.entities
    return entity


def write_mapping_run(
    workspace: Path,
    run: FusionRun,
    entities: Sequence[Entity],
    *,
    rejections: Sequence[CandidateRejection] = (),
) -> Path:
    """Persist entities as a real Semantic Mapping run and return the run directory."""
    run_dir = workspace / "semantic_mapping"
    SemanticMappingRunWriter(
        output_dir=run_dir,
        sequence_name="sequence-0001",
        run_id=SemanticMappingRunId("mapping-run-0001"),
        run_index=1,
        semantic_map_id=SEMANTIC_MAP_ID,
        lineage=lineage_from_fusion_manifest(run.manifest),
        code_version="test",
        code_digest=CODE_DIGEST,
    ).write(entities, rejections=rejections)
    return run_dir

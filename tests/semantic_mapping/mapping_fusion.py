"""Real fused evidence for Semantic Mapping tests, built through Semantic Fusion itself."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evidence_builders import ClaimSpec, View, build_scenario
from fusion_builders import MAP_ID
from fusion_run_fixtures import LINEAGE, make_run_fixture
from mapping_builders import SEMANTIC_MAP_ID, SUMMARY_POLICY, make_provenance
from mapping_geometry_fake import InMemoryGeometrySource

from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
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
    Entity,
    EntityId,
    SemanticMapId,
    evidence_links_from_fused_evidence,
    semantic_state_from_fused_evidence,
    summarize_geometry,
    summarize_temporal_state,
)

__all__ = ["ClaimSpec", "FusionRun", "View", "entity_from_outcome", "fuse", "write_fusion_run"]


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


def write_fusion_run(workspace: Path) -> FusionRun:
    """Write the three-support fusion run of the fusion fixtures and reopen it from disk."""
    fixture = make_run_fixture()
    manifest = SemanticFusionRunWriter(
        workspace_root=workspace,
        sequence_name="sequence-0001",
        run_id=SemanticFusionRunId("fusion-run-0001"),
        run_index=1,
        selection_label="all-frames",
        policy_label="quality-aware",
        lineage=LINEAGE,
        code_version="test",
    ).write(fixture.outcomes, excluded=fixture.excluded)
    run_dir = (
        workspace
        / "runs"
        / "semantic-fusion"
        / "sequence-0001"
        / f"run-{manifest.run_index:04d}__all-frames__quality-aware"
    )
    return FusionRun(
        run_dir=run_dir,
        reader=SemanticFusionRunReader(run_dir),
        manifest=manifest,
        outcomes=fixture.outcomes,
        geometry=InMemoryGeometrySource(
            MAP_ID, {index: (index * 0.1, 0.0, 0.0) for index in range(1_000)}
        ),
    )


def entity_from_outcome(
    outcome: FusionOutcome,
    run: FusionRun,
    *,
    semantic_map_id: SemanticMapId = SEMANTIC_MAP_ID,
) -> Entity:
    """Assemble the entity of one fusion outcome from the parts each issue defines."""
    evidence = outcome.evidence
    return Entity(
        entity_id=EntityId(f"entity--{outcome.support.fusion_support_id}"),
        semantic_map_id=semantic_map_id,
        geometry=summarize_geometry(
            outcome.support.geometry_support, source=run.geometry, policy=SUMMARY_POLICY
        ),
        semantic_state=semantic_state_from_fused_evidence(evidence),
        evidence=evidence_links_from_fused_evidence(evidence, manifest=run.manifest),
        temporal_state=summarize_temporal_state(evidence.physical_observation_groups),
        provenance=make_provenance(),
    )

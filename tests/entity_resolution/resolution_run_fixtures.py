"""A complete resolution run over real entities, with every outcome and a contradiction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from resolution_builders import channel_policy, decision_between, match_evidence
from resolution_entity_builders import entity_at

from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    EntityCandidateSet,
    EntityResolutionRunId,
    EntityResolutionRunWriter,
    PairResolution,
    ResolutionOutcome,
    ResolutionRunLineage,
    ResolvedEntityMaterialization,
    SemanticEvidence,
    Unavailability,
    UnavailableReason,
    comparison_id_for,
    materialize_resolved_entities,
    retrieve_candidate_sets,
)
from contextmap.entity_resolution.decision import ResolutionDecision
from contextmap.geometric_mapping import MapId
from contextmap.semantic_mapping import Entity, SemanticMapId

RUN = EntityResolutionRunId("resolution-run-0001")
MATCH, DISTINCT, UNRESOLVED = (
    ResolutionOutcome.MATCH,
    ResolutionOutcome.DISTINCT,
    ResolutionOutcome.UNRESOLVED,
)

LINEAGE = ResolutionRunLineage(
    sequence_artifact_id="sequence-0001",
    geometric_map_id=MapId("map-0001"),
    semantic_mapping_run_id="mapping-run-0001",
    semantic_mapping_schema_version="0.1.0",
    semantic_mapping_artifact_digest="sha256:mapping",
    semantic_map_ids=(SemanticMapId("semantic-map-0001"),),
    point_representation_run_ids=(),
    perception_run_ids=("perception-run-0001",),
)


@dataclass(frozen=True)
class RunInputs:
    """Everything a writer needs, and the entities it was built from."""

    entities: dict[str, Entity]
    candidate_sets: tuple[EntityCandidateSet, ...]
    resolutions: tuple[PairResolution, ...]
    materialization: ResolvedEntityMaterialization


def resolution_of(
    first: Entity,
    second: Entity,
    outcome: ResolutionOutcome,
    *,
    semantic_unavailable: bool = False,
) -> PairResolution:
    """The evidence and the decision of one pair, consistent with each other."""
    a, b = sorted(
        (first.reference, second.reference), key=lambda ref: (ref.semantic_map_id, ref.entity_id)
    )
    channels: dict[str, object] = {}
    if semantic_unavailable:
        channels["semantic"] = SemanticEvidence(
            policy=channel_policy("entity-semantic-compatibility-v1"),
            unavailable=Unavailability(
                reason=UnavailableReason.MISSING_EVIDENCE, detail="no hypothesis"
            ),
        )
    evidence = match_evidence(
        comparison_id=comparison_id_for(a, b), entity_a_ref=a, entity_b_ref=b, **channels
    )
    decision: ResolutionDecision = decision_between(a, b, outcome)
    return PairResolution(evidence=evidence, decision=decision)


def build_inputs() -> RunInputs:
    """Six entities; a~b~c is contradicted by a!=c; e~f merge; a-d unresolved; b!=d distinct."""
    entities = {
        name: entity_at(
            name, (index * 1.0, 0.0, 0.0), support_number=index + 1, spatial=(f"spatial--{name}",)
        )
        for index, name in enumerate("abcdef")
    }
    sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1)
    )
    resolutions = tuple(
        resolution_of(entities[x], entities[y], outcome, semantic_unavailable=(x, y) == ("a", "d"))
        for x, y, outcome in (
            ("a", "b", MATCH),
            ("b", "c", MATCH),
            ("a", "c", DISTINCT),
            ("a", "d", UNRESOLVED),
            ("b", "d", DISTINCT),
            ("e", "f", MATCH),
        )
    )
    materialization = materialize_resolved_entities(
        entities.values(), [item.decision for item in resolutions], resolution_run_id=RUN
    )
    return RunInputs(entities, sets, resolutions, materialization)


def write_run(
    output_dir: Path, inputs: RunInputs | None = None, **writer_kwargs: object
) -> tuple[RunInputs, object]:
    """Write a run and return the inputs and the manifest."""
    used = inputs or build_inputs()
    writer = EntityResolutionRunWriter(
        output_dir=output_dir,
        run_id=RUN,
        lineage=LINEAGE,
        code_version="test-version",
        **writer_kwargs,  # type: ignore[arg-type]
    )
    manifest = writer.write(
        candidate_sets=used.candidate_sets,
        resolutions=used.resolutions,
        materialization=used.materialization,
    )
    return used, manifest

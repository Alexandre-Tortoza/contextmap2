"""A small decided scene and its persisted run, shared by the artifact and evaluation tests.

The scene is real: lattices of points summarized by Semantic Mapping, candidates generated, every
channel evaluated and the relations decided by the baseline policy.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from relation_builders import entity_ref
from relation_scene import LATTICE_POLICY, Scene

from contextmap.entity_resolution import EntityResolutionRunId, ResolvedEntityReference
from contextmap.geometric_mapping import MapId
from contextmap.semantic_mapping import EntityGeometry
from contextmap.spatial_relations import (
    AxisDirection,
    CandidatePolicy,
    ContactPredicatePolicy,
    EndpointLink,
    FrameConventions,
    GeometricPredicatePolicy,
    ObservationRelationStatement,
    RelationCandidateSet,
    RelationDecisionResult,
    RelationEvidence,
    RelationPredicate,
    RelationsRunDebugLevel,
    RelationsRunLineage,
    RelationsRunPolicies,
    SpatialRelationsRunId,
    SpatialRelationsRunManifest,
    SpatialRelationsRunWriter,
    StatementPolarity,
    UpstreamStatementRef,
    decide_relations,
    evaluate_contact_candidates,
    evaluate_geometric_candidates,
    generate_relation_candidates,
    observation_evidence_from_statements,
)

P = RelationPredicate
CONVENTIONS = FrameConventions(
    map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
)
GEOMETRIC = GeometricPredicatePolicy(
    boundary_tolerance_m=0.02,
    next_to_max_gap_m=0.5,
    adjacent_penetration_m=0.05,
    containment_slack_m=0.05,
    directional_overlap_fraction=0.5,
)
CONTACT = ContactPredicatePolicy(
    contact_distance_m=0.05,
    contact_tolerance_m=0.02,
    min_contact_points=3,
    support_height_tolerance_m=0.05,
    support_footprint_fraction=0.5,
    leaning_min_tilt_deg=10.0,
    leaning_max_tilt_deg=80.0,
    tilt_tolerance_deg=2.0,
    leaning_min_vertical_overlap_m=0.3,
)
CANDIDATES = CandidatePolicy(
    predicates=(P.NEXT_TO, P.ABOVE, P.ON_TOP_OF, P.TOUCHING, P.LEANING_AGAINST),
    proximity_radius_m=0.6,
    directional_radius_m=2.0,
)
RUN = EntityResolutionRunId("resolution-run-0001")
LINEAGE = RelationsRunLineage(
    entity_resolution_run_id=RUN,
    entity_resolution_schema_version="0.1.0",
    entity_resolution_artifact_digest="sha256:" + "e5" * 32,
    geometric_map_id=MapId("map-0001"),
)
POLICIES = RelationsRunPolicies(
    frame_conventions=CONVENTIONS,
    candidate=CANDIDATES,
    geometric=GEOMETRIC,
    contact=CONTACT,
)


@dataclasses.dataclass
class Run:
    candidates: RelationCandidateSet
    evidence: list[RelationEvidence]
    decisions: RelationDecisionResult
    entities: dict[ResolvedEntityReference, EntityGeometry]


def build_run() -> Run:
    """A floor with a crate on it, a hovering box and a far pallet, decided with every channel.

    The crate is supported by the floor, leaning against it is rejected (their heights do not
    overlap), and the box hovers 0.06 m above the floor, within the contact tolerance, so touching
    and resting on it stay unresolved.
    """
    scene = Scene()
    scene.add_lattice("floor", (0.0, 0.0, 0.0), (2.0, 2.0, 0.1), 0.1)
    scene.add_lattice("crate", (0.5, 0.5, 0.1), (1.0, 1.0, 0.6), 0.1)
    scene.add_lattice("pallet", (9.0, 9.0, 0.0), (10.0, 10.0, 0.2), 0.1)
    scene.add_lattice("hover", (1.2, 1.2, 0.16), (1.7, 1.7, 0.66), 0.1)
    entities = {
        entity_ref(1): scene.geometry("floor", policy=LATTICE_POLICY),
        entity_ref(2): scene.geometry("crate", policy=LATTICE_POLICY),
        entity_ref(3): scene.geometry("pallet", policy=LATTICE_POLICY),
        entity_ref(4): scene.geometry("hover", policy=LATTICE_POLICY),
    }
    candidates = generate_relation_candidates(entities, policy=CANDIDATES, conventions=CONVENTIONS)
    evidence = [
        *evaluate_geometric_candidates(
            candidates, entities=entities, policy=GEOMETRIC, conventions=CONVENTIONS
        ),
        *evaluate_contact_candidates(
            candidates,
            entities=entities,
            geometry_source=scene.source(),
            policy=CONTACT,
            conventions=CONVENTIONS,
        ),
    ]
    statement = ObservationRelationStatement(
        source=UpstreamStatementRef(
            source_run_id="perception-run-0001",
            statement_id="statement-0001",
            physical_observation_id="frame-0120",
            producer="qwen-vl/prompt-v1",
        ),
        subject=EndpointLink(
            upstream_ref="region-0002", entity_ref=entity_ref(2), linked_through="fused--a"
        ),
        predicate_text="on top of",
        object=EndpointLink(
            upstream_ref="region-0001", entity_ref=entity_ref(1), linked_through="fused--b"
        ),
        polarity=StatementPolarity.ASSERTS,
    )
    evidence.extend(
        observation_evidence_from_statements([statement], entities=set(entities)).evidence
    )
    return Run(candidates, evidence, decide_relations(candidates, evidence), entities)


def write_run(
    run: Run,
    directory: Path,
    *,
    debug: RelationsRunDebugLevel = RelationsRunDebugLevel.NONE,
    lineage: RelationsRunLineage = LINEAGE,
    policies: RelationsRunPolicies = POLICIES,
    warnings: tuple[str, ...] = (),
) -> SpatialRelationsRunManifest:
    """Persist a run in ``directory``."""
    writer = SpatialRelationsRunWriter(
        output_dir=directory,
        run_id=SpatialRelationsRunId("relations-run-0001"),
        lineage=lineage,
        policies=policies,
        code_version="test",
        debug_level=debug,
    )
    return writer.write(
        candidates=run.candidates,
        evidence=run.evidence,
        decisions=run.decisions,
        warnings=warnings,
    )

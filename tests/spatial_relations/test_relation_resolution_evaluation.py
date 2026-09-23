"""Relations evaluated by the identities of Entity Resolution's own evaluation, end to end."""

from __future__ import annotations

from pathlib import Path

from relation_resolution_fixture import ER_RUN, build_inputs, er_scene_source, write_er_run
from relation_scene import CONNECTED_POLICY

from contextmap.entity_resolution import CandidateRetrievalPolicy, EntityResolutionRunReader
from contextmap.evaluation import (
    Coverage,
    EntityResolutionEvaluationReport,
    IdentityAnnotationSet,
    IdentityOccurrence,
    IdentityScope,
    LabelNormalization,
    ObservationRef,
    OccurrenceLink,
    PhysicalIdentity,
    PredicateRule,
    ReferenceSampleId,
    RelationAnnotation,
    RelationAnnotationSet,
    RelationStatus,
    evaluate_entity_resolution,
    evaluate_spatial_relations,
)
from contextmap.ingestion import SourceObservationId
from contextmap.spatial_relations import (
    CandidatePolicy,
    FrameConventions,
    GeometricPredicatePolicy,
    RelationPredicate,
    RelationsRunPolicies,
    SpatialRelationsRunId,
    SpatialRelationsRunReader,
    SpatialRelationsRunWriter,
    decide_relations,
    evaluate_geometric_candidates,
    generate_relation_candidates,
    lineage_from_resolution_manifest,
    resolved_entity_geometries,
)

CONVENTIONS = FrameConventions(map_frame="map")
GEOMETRIC = GeometricPredicatePolicy(
    boundary_tolerance_m=0.02,
    next_to_max_gap_m=0.6,
    adjacent_penetration_m=0.05,
    containment_slack_m=0.05,
    directional_overlap_fraction=0.5,
)
CANDIDATES = CandidatePolicy(
    predicates=(RelationPredicate.NEXT_TO,), proximity_radius_m=0.6, directional_radius_m=1.0
)
ANCHOR = (
    ObservationRef(
        sample_id=ReferenceSampleId("sample-a"), observation_id=SourceObservationId("frame-a")
    ),
)
GROUPS = {"I1": ["a", "b"], "I2": ["c"], "I3": ["d"], "I4": ["e", "f"]}


def _identity_reference() -> IdentityAnnotationSet:
    names = sorted({name for members in GROUPS.values() for name in members})
    return IdentityAnnotationSet(
        scope=tuple(
            IdentityScope(
                sample_id=ReferenceSampleId(f"sample-{name}"),
                observation_id=SourceObservationId(f"frame-{name}"),
                coverage=Coverage.COMPLETE,
            )
            for name in names
        ),
        identities=tuple(
            PhysicalIdentity(
                identity_id=identity,
                occurrences=tuple(
                    IdentityOccurrence(
                        sample_id=ReferenceSampleId(f"sample-{name}"),
                        observation_id=SourceObservationId(f"frame-{name}"),
                        region_id=f"region-{name}",
                    )
                    for name in members
                ),
            )
            for identity, members in sorted(GROUPS.items())
        ),
        distinct_pairs=(),
    )


def _relation(number: int, subject: str, obj: str, status: RelationStatus) -> RelationAnnotation:
    return RelationAnnotation(
        relation_id=f"rel-{number}",
        subject_identity_id=subject,
        predicate="next to",
        object_identity_id=obj,
        status=status,
        anchors=ANCHOR,
    )


def _relations_reference() -> RelationAnnotationSet:
    return RelationAnnotationSet(
        normalization=LabelNormalization(policy_id="casefold-exact/1"),
        predicate_rules=(PredicateRule(predicate="next to", symmetric=True),),
        relations=(
            _relation(1, "I2", "I3", RelationStatus.HOLDS),
            _relation(2, "I3", "I4", RelationStatus.HOLDS),
            _relation(3, "I1", "I2", RelationStatus.HOLDS),
        ),
    )


RETRIEVAL = CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1)
"""The retrieval policy ``build_inputs()`` itself used, so the reproducibility check reproduces."""


def _identity_report(resolution_dir: Path) -> EntityResolutionEvaluationReport:
    """Entity Resolution's own evaluation of the persisted run, identity and provenance together.

    This is the report a real caller gets from ``evaluate_entity_resolution``: its ``identity``
    and its ``reproducibility`` are never built apart, because Spatial Relations only trusts an
    identity evaluation once its own reproducibility names this exact resolution run and artifact.
    """
    inputs = build_inputs()
    links = [
        OccurrenceLink(
            sample_id=ReferenceSampleId(f"sample-{name}"),
            observation_id=SourceObservationId(f"frame-{name}"),
            region_id=f"region-{name}",
            entity_ref=inputs.entities[name].reference,
        )
        for name in "abcdef"
    ]
    return evaluate_entity_resolution(
        EntityResolutionRunReader(resolution_dir),
        entities=inputs.entities.values(),
        reference=_identity_reference(),
        links=links,
        retrieval_policy=RETRIEVAL,
        reference_set_id="test",
    )


def _relations_run(directory: Path, resolution_dir: Path) -> SpatialRelationsRunReader:
    resolution = EntityResolutionRunReader(resolution_dir)
    geometries = resolved_entity_geometries(
        resolution.resolved_entities(), source=er_scene_source(), policy=CONNECTED_POLICY
    )
    candidates = generate_relation_candidates(
        geometries, policy=CANDIDATES, conventions=CONVENTIONS
    )
    evidence = evaluate_geometric_candidates(
        candidates, entities=geometries, policy=GEOMETRIC, conventions=CONVENTIONS
    )
    SpatialRelationsRunWriter(
        output_dir=directory,
        run_id=SpatialRelationsRunId("relations-run-0001"),
        lineage=lineage_from_resolution_manifest(resolution.manifest),
        policies=RelationsRunPolicies(
            frame_conventions=CONVENTIONS,
            candidate=CANDIDATES,
            geometry_summary=CONNECTED_POLICY,
            geometric=GEOMETRIC,
        ),
        code_version="test",
    ).write(
        candidates=candidates, evidence=evidence, decisions=decide_relations(candidates, evidence)
    )
    return SpatialRelationsRunReader(directory)


def test_relations_are_evaluated_by_the_identity_of_entity_resolutions_own_evaluation(
    tmp_path: Path,
) -> None:
    write_er_run(tmp_path / "resolution")
    run = _relations_run(tmp_path / "relations", tmp_path / "resolution")
    identity_report = _identity_report(tmp_path / "resolution")
    identity = identity_report.identity
    assert identity.duplicated_identities == 1
    assert {
        reference.resolution_run_id for reference, _ in identity.identity_of_resolved_entity
    } == {ER_RUN}
    assert identity_report.reproducibility.run_id == ER_RUN

    report = evaluate_spatial_relations(
        run,
        reference=_relations_reference(),
        identity=identity,
        identity_reproducibility=identity_report.reproducibility,
    )

    next_to = report.predicate(RelationPredicate.NEXT_TO)
    assert (next_to.annotated_holds, next_to.true_positives) == (4, 4)
    assert (next_to.precision, next_to.recall, next_to.f1) == (1.0, 1.0, 1.0)
    # I1 é representada por duas entidades resolvidas (uma duplicata de Entity Resolution): as
    # relações dela são puladas e listadas, nunca reparadas dentro da avaliação de relações.
    assert report.unmatched.identities_with_several_entities == ("I1",)
    assert report.unmatched.reference_without_entity == 0
    assert report.entity_resolution_run_id == ER_RUN
    assert (
        report.entity_resolution_artifact_digest
        == identity_report.reproducibility.resolution_artifact_digest
    )

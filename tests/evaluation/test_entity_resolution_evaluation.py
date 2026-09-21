"""Entity Resolution evaluation: explicit identities, separate failure classes, fixed-input arms."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from mapping_builders import make_hypothesis
from resolution_builders import (
    appearance_evidence,
    appearance_measurement,
    channel_policy,
    feature_ref,
    representation_evidence,
)
from resolution_entity_builders import entity_at, entity_over, fused_id, lattice, scene_source
from resolution_run_fixtures import (
    LINEAGE,
    RUN,
    RunInputs,
    build_inputs,
    resolution_of,
    write_run,
)

from contextmap.entity_resolution import (
    AppearanceEvidence,
    CandidateRetrievalPolicy,
    ComparisonChannels,
    ConservativeResolutionPolicy,
    EntityResolutionRunReader,
    EntityResolutionRunWriter,
    FeatureContribution,
    GeometryComparisonPolicy,
    MatchChannel,
    MatchEvidenceBuilder,
    ResolutionOutcome,
    ResolvedEntityReference,
    SemanticCompatibilityPolicy,
    SplitDetectionPolicy,
    detect_split_candidates,
    materialize_resolved_entities,
    resolution_artifact_digest,
    resolve_candidate_pairs,
    retrieve_candidate_sets,
)
from contextmap.evaluation import (
    ENTITY_RESOLUTION_EVALUATOR_ID,
    ArmEvaluation,
    Coverage,
    DistinctIdentityPair,
    EntityResolutionEvaluationError,
    IdentityAnnotationSet,
    IdentityEvaluation,
    IdentityOccurrence,
    IdentityScope,
    OccurrenceLink,
    PhysicalIdentity,
    ResolutionArm,
    ResolutionValidationLayer,
    SplitReference,
    default_metric_registry,
    encode_entity_resolution_report,
    evaluate_channel_ablation,
    evaluate_entity_resolution,
    evaluate_identity,
    evaluate_splits,
)
from contextmap.evaluation.reference_set import ReferenceSampleId
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_mapping import Entity

MATCH, DISTINCT, UNRESOLVED = (
    ResolutionOutcome.MATCH,
    ResolutionOutcome.DISTINCT,
    ResolutionOutcome.UNRESOLVED,
)
RETRIEVAL = CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1)


def reference_of(
    groups: dict[str, list[str]],
    distinct: list[tuple[str, str]] = (),  # type: ignore[assignment]
    *,
    partial: tuple[str, ...] = (),
) -> IdentityAnnotationSet:
    """Identities over entity names; each entity name has its own annotated occurrence."""
    names = sorted({name for members in groups.values() for name in members})
    return IdentityAnnotationSet(
        scope=tuple(
            IdentityScope(
                sample_id=f"sample-{name}",  # type: ignore[arg-type]
                observation_id=f"frame-{name}",  # type: ignore[arg-type]
                coverage=Coverage.PARTIAL if name in partial else Coverage.COMPLETE,
            )
            for name in names
        ),
        identities=tuple(
            PhysicalIdentity(
                identity_id=identity,
                occurrences=tuple(
                    IdentityOccurrence(
                        sample_id=f"sample-{name}",  # type: ignore[arg-type]
                        observation_id=f"frame-{name}",  # type: ignore[arg-type]
                        region_id=f"region-{name}",
                    )
                    for name in members
                ),
            )
            for identity, members in sorted(groups.items())
        ),
        distinct_pairs=tuple(DistinctIdentityPair(first=a, second=b) for a, b in distinct),
    )


def occurrence(sample: str, observation: str, region: str | None, entity: Entity) -> OccurrenceLink:
    return OccurrenceLink(
        sample_id=ReferenceSampleId(sample),
        observation_id=SourceObservationId(observation),
        region_id=region,
        entity_ref=entity.reference,
    )


def links_of(entities: dict[str, Entity], *names: str) -> list[OccurrenceLink]:
    return [
        occurrence(f"sample-{name}", f"frame-{name}", f"region-{name}", entities[name])
        for name in names
    ]


def evaluate(
    inputs: RunInputs, reference: IdentityAnnotationSet, links: list[OccurrenceLink]
) -> IdentityEvaluation:
    return evaluate_identity(
        reference=reference,
        links=links,
        resolved=inputs.materialization.resolved,
        candidate_sets=inputs.candidate_sets,
        decisions=[item.decision for item in inputs.resolutions],
        contradictions=inputs.materialization.contradictions,
    )


def outcomes(pairs: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return dict(pairs)


# --- identity: registry metrics and every failure class -----------------------------------------


def test_the_registry_metrics_are_computed_exactly_and_every_failure_class_stays_visible() -> None:
    inputs = build_inputs()
    reference = reference_of(
        {"I1": ["a", "b"], "I2": ["c"], "I3": ["d"], "I4": ["e", "f"]},
        [("I1", "I2"), ("I1", "I3")],
    )

    result = evaluate(inputs, reference, links_of(inputs.entities, *"abcdef"))

    # a~b~c está contradito por a!=c: nada é fundido, e isso aparece como duplicata e como causa.
    assert result.resolved_entities == 5
    assert (result.false_merge_entities, result.false_merge_rate) == (0, 0.0)
    assert (result.annotated_identities, result.duplicated_identities) == (4, 1)
    assert result.duplicate_rate == 0.25
    assert (result.same_pairs, result.distinct_pairs) == (2, 4)
    assert outcomes(result.same_pair_outcomes) == {
        "match_withheld_by_contradiction": 1,
        "merged": 1,
    }
    assert outcomes(result.distinct_pair_outcomes) == {
        "decided_distinct": 2,
        "match_withheld_by_contradiction": 1,
        "unresolved": 1,
    }
    assert (result.true_merges, result.false_merges) == (1, 0)
    assert (result.pairwise_precision, result.pairwise_recall) == (1.0, 0.5)
    assert result.transitivity_contradictions == 1
    assert result.source_entities_spanning_identities == 0


def test_the_registry_defines_the_metrics_this_evaluator_computes() -> None:
    registry = default_metric_registry()
    names = {item.name: item for item in registry.definitions}

    for metric in ("entity.false_merge.rate", "entity.duplicate.rate"):
        assert names[metric].evaluator_id == ENTITY_RESOLUTION_EVALUATOR_ID
    # A acurácia semântica não é acurácia de identidade: este avaliador não a reporta.
    assert not hasattr(IdentityEvaluation, "semantic_accuracy_rate")


def test_a_false_merge_is_counted_as_such_and_never_hidden() -> None:
    inputs = build_inputs()
    reference = reference_of({"I1": ["a"], "I2": ["b"], "I5": ["e"], "I6": ["f"]}, [("I5", "I6")])

    result = evaluate(inputs, reference, links_of(inputs.entities, "a", "b", "e", "f"))

    assert (result.false_merge_entities, result.false_merges) == (1, 1)
    # A população são as entidades resolvidas com algum membro anotado: {a}, {b} e {e, f}.
    assert result.false_merge_rate == pytest.approx(1 / 3)
    assert outcomes(result.distinct_pair_outcomes)["false_merge"] == 1
    assert result.pairwise_precision == 0.0  # 0 fusões corretas e 1 falsa: contagens visíveis


def test_a_missed_merge_is_a_duplicate_and_keeps_its_cause() -> None:
    inputs = build_inputs()
    # a e d são o mesmo objeto, mas a decisão foi UNRESOLVED; b e d, mas foi DISTINCT.
    reference = reference_of({"I1": ["a", "b", "d"]})

    result = evaluate(inputs, reference, links_of(inputs.entities, "a", "b", "d"))

    assert outcomes(result.same_pair_outcomes) == {
        "match_withheld_by_contradiction": 1,
        "unresolved": 1,
        "false_distinct": 1,
    }
    assert result.duplicated_identities == 1
    assert result.duplicate_rate == 1.0


def test_a_retrieval_miss_is_distinguishable_from_a_policy_error() -> None:
    inputs = build_inputs()
    narrow = retrieve_candidate_sets(
        inputs.entities.values(),
        CandidateRetrievalPolicy(centroid_radius_m=1.5, bounds_margin_m=0.1),
    )
    reference = reference_of({"I1": ["a", "c"]})

    result = evaluate_identity(
        reference=reference,
        links=links_of(inputs.entities, "a", "c"),
        resolved=inputs.materialization.resolved,
        candidate_sets=narrow,
        decisions=[item.decision for item in inputs.resolutions],
    )

    assert outcomes(result.same_pair_outcomes) == {"retrieval_miss": 1}


def test_candidate_pairs_that_were_never_compared_are_reported_apart() -> None:
    inputs = build_inputs()
    reference = reference_of({"I1": ["a"], "I2": ["e"]}, [("I1", "I2")])

    result = evaluate(inputs, reference, links_of(inputs.entities, "a", "e"))

    assert outcomes(result.distinct_pair_outcomes) == {"not_compared": 1}


def test_partial_scope_unannotated_and_unknown_links_are_handled_explicitly() -> None:
    inputs = build_inputs()
    reference = reference_of({"I1": ["a", "b"], "I2": ["c"]}, [("I1", "I2")], partial=("c",))
    unannotated = OccurrenceLink(
        sample_id="sample-x",  # type: ignore[arg-type]
        observation_id="frame-x",  # type: ignore[arg-type]
        region_id=None,
        entity_ref=inputs.entities["d"].reference,
    )
    links = [*links_of(inputs.entities, "a", "b", "c"), unannotated]

    result = evaluate(inputs, reference, links)

    assert result.ignored_links == 2  # c está em escopo parcial e o d não está anotado
    assert result.annotated_identities == 1
    assert result.distinct_pairs == 0
    stranger = OccurrenceLink(
        sample_id="sample-a",  # type: ignore[arg-type]
        observation_id="frame-a",  # type: ignore[arg-type]
        region_id="region-a",
        entity_ref=entity_at("z", (99, 0, 0), support_number=9).reference,
    )
    with pytest.raises(EntityResolutionEvaluationError, match="does not have"):
        evaluate(inputs, reference, [stranger])


def test_rates_are_none_when_there_is_nothing_to_measure() -> None:
    inputs = build_inputs()

    result = evaluate(inputs, reference_of({"I1": ["a"]}), links_of(inputs.entities, "a"))

    assert (result.same_pairs, result.distinct_pairs) == (0, 0)
    assert result.pairwise_precision is None and result.pairwise_recall is None
    empty = evaluate(inputs, reference_of({"I1": ["a"]}), [])
    assert (empty.false_merge_rate, empty.duplicate_rate) == (None, None)


def test_a_source_entity_spanning_two_identities_is_not_a_resolution_error() -> None:
    inputs = build_inputs()
    reference = reference_of({"I1": ["a"], "I2": ["b"]}, [("I1", "I2")])
    links = links_of(inputs.entities, "a", "b")
    # A entidade b também nasceu da ocorrência de a: mistura duas identidades antes da resolução.
    links.append(occurrence("sample-a", "frame-a", "region-a", inputs.entities["b"]))

    result = evaluate(inputs, reference, links)

    assert result.source_entities_spanning_identities == 1


def resolved_reference(inputs: RunInputs, name: str) -> ResolvedEntityReference:
    """The resolved entity a source entity ended up in."""
    for resolved in inputs.materialization.resolved.entities:
        if inputs.entities[name].reference in resolved.member_entity_refs:
            return resolved.reference
    raise AssertionError(f"entity {name!r} is in no resolved entity")


def test_each_resolved_entity_with_one_identity_is_mapped_and_the_others_are_named() -> None:
    inputs = build_inputs()
    # d não tem ocorrência anotada; e e f foram fundidos mas a referência os declara distintos.
    reference = reference_of(
        {"I1": ["a", "b"], "I2": ["c"], "I4": ["e"], "I5": ["f"]}, [("I4", "I5")]
    )

    result = evaluate(inputs, reference, links_of(inputs.entities, *"abcef"))

    def ref(name: str) -> ResolvedEntityReference:
        return resolved_reference(inputs, name)

    # a~b~c está contradito, então a e b seguem separadas: duas entidades e uma só identidade.
    assert dict(result.identity_of_resolved_entity) == {
        ref("a"): "I1",
        ref("b"): "I1",
        ref("c"): "I2",
    }
    assert result.resolved_entities_spanning_identities == (ref("e"),)
    assert ref("e") == ref("f")
    mapped = [item for item, _ in result.identity_of_resolved_entity]
    assert mapped == sorted(
        mapped, key=lambda item: (item.resolution_run_id, item.resolved_entity_id)
    )
    # A entidade d não tem identidade anotada: fica fora do mapa, e não é acusada de nada.
    assert ref("d") not in mapped and ref("d") not in result.resolved_entities_spanning_identities


# --- real annotation format ---------------------------------------------------------------------


def test_the_ci_identity_annotations_are_evaluated_in_their_real_format() -> None:
    record = json.loads(
        Path("tests/fixtures/ci_subset/1.0.1/annotations/identity.json").read_text()
    )
    reference = IdentityAnnotationSet.from_record(record)
    entities = {
        name: entity_at(name, (index * 3.0, 0.0, 0.0), support_number=index + 1)
        for index, name in enumerate(("a0", "a1", "b0", "b1"))
    }
    occurrences = {
        "a0": ("sample-0000", "frame-0000", "region-a"),
        "a1": ("sample-0001", "frame-0001", "region-a1"),
        "b0": ("sample-0000", "frame-0000", "region-b"),
        "b1": ("sample-0001", "frame-0001", "region-b1"),
    }
    links = [
        occurrence(sample, observation, region, entities[name])
        for name, (sample, observation, region) in occurrences.items()
    ]
    # frame-0002 tem escopo parcial: a ocorrência sem entidade ligada não conta.
    links.append(
        OccurrenceLink(
            sample_id="sample-0002",  # type: ignore[arg-type]
            observation_id="frame-0002",  # type: ignore[arg-type]
            region_id=None,
            entity_ref=entities["a1"].reference,
        )
    )
    resolutions = tuple(
        resolution_of(entities[x], entities[y], outcome)
        for x, y, outcome in (
            ("a0", "a1", MATCH),
            ("b0", "b1", MATCH),
            ("a0", "b0", DISTINCT),
            ("a1", "b1", DISTINCT),
        )
    )
    sets = retrieve_candidate_sets(entities.values(), RETRIEVAL)
    materialization = materialize_resolved_entities(
        entities.values(), [item.decision for item in resolutions], resolution_run_id=RUN
    )

    result = evaluate_identity(
        reference=reference,
        links=links,
        resolved=materialization.resolved,
        candidate_sets=sets,
        decisions=[item.decision for item in resolutions],
    )

    assert (result.false_merge_rate, result.duplicate_rate) == (0.0, 0.0)
    assert (result.pairwise_precision, result.pairwise_recall) == (1.0, 1.0)
    assert result.ignored_links == 1


# --- contract, provenance and reproducibility ---------------------------------------------------


def evaluated(tmp_path: Path, inputs: RunInputs | None = None, **kwargs: Any) -> Any:
    directory = tmp_path / "artifact"
    used, _ = write_run(directory, inputs)
    reference = reference_of({"I1": ["a", "b"], "I4": ["e", "f"]}, [("I1", "I4")])
    return evaluate_entity_resolution(
        EntityResolutionRunReader(directory),
        entities=used.entities.values(),
        reference=reference,
        links=links_of(used.entities, "a", "b", "e", "f"),
        retrieval_policy=kwargs.pop("retrieval_policy", RETRIEVAL),
        reference_set_id="ci-subset-1.0.1",
        **kwargs,
    )


def test_a_consistent_run_passes_every_check_and_the_report_is_reproducible(tmp_path: Path) -> None:
    report = evaluated(tmp_path)

    assert report.passed
    assert {check.layer for check in report.checks} == set(ResolutionValidationLayer)
    assert all(check.examined >= 0 for check in report.checks)
    reproducibility = report.reproducibility
    assert reproducibility.reference_set_id == "ci-subset-1.0.1"
    assert reproducibility.run_id == "resolution-run-0001"
    manifest = EntityResolutionRunReader(tmp_path / "artifact").manifest
    assert reproducibility.resolution_schema_version == manifest.schema_version
    assert reproducibility.resolution_artifact_digest == resolution_artifact_digest(manifest)
    assert reproducibility.semantic_mapping_run_id == LINEAGE.semantic_mapping_run_id
    assert reproducibility.annotation_digest.startswith("sha256:")
    assert any(item.startswith("resolution:") for item in reproducibility.policies)
    assert reproducibility.evaluator_id == ENTITY_RESOLUTION_EVALUATOR_ID
    assert report.artifact_bytes > 0
    assert report.retrieval.retrieval_recall == 1.0
    assert report.retrieval.candidate_pairs == 15
    assert report.split is None
    again = evaluated(tmp_path / "again")
    assert again.identity == report.identity and again.reproducibility == report.reproducibility


def test_the_report_encodes_to_json_with_every_layer_apart(tmp_path: Path) -> None:
    record = encode_entity_resolution_report(evaluated(tmp_path))

    text = json.dumps(record, sort_keys=True)

    assert set(record) == {
        "reproducibility",
        "checks",
        "identity",
        "retrieval",
        "split",
        "artifact_bytes",
    }
    assert all(
        set(check) >= {"name", "layer", "examined", "failures", "passed"}
        for check in record["checks"]
    )
    assert "score" not in text


def test_a_tampered_artifact_fails_the_contract_layer(tmp_path: Path) -> None:
    directory = tmp_path / "artifact"
    used, _ = write_run(directory)
    with (directory / "metrics/counts.json").open("a") as handle:
        handle.write("\n")

    report = evaluate_entity_resolution(
        EntityResolutionRunReader(directory),
        entities=used.entities.values(),
        reference=reference_of({"I1": ["a"]}),
        links=[],
        retrieval_policy=RETRIEVAL,
        reference_set_id="x",
    )

    failing = {check.name: check for check in report.checks if not check.passed}
    assert list(failing) == ["the artifact is intact and its integrity holds"]
    assert not report.passed


def test_resolved_entities_that_do_not_follow_from_the_decisions_are_detected(
    tmp_path: Path,
) -> None:
    inputs = build_inputs()
    without_distinct = [
        item.decision for item in inputs.resolutions if item.decision.decision is not DISTINCT
    ]
    merged_anyway = materialize_resolved_entities(
        inputs.entities.values(), without_distinct, resolution_run_id=RUN
    )

    report = evaluated(tmp_path, dataclasses.replace(inputs, materialization=merged_anyway))

    failing = {check.name for check in report.checks if not check.passed}
    assert (
        "materializing the same decisions reproduces the resolved entities and their ids" in failing
    )
    assert not report.passed


def test_a_different_retrieval_is_detected_not_trusted(tmp_path: Path) -> None:
    report = evaluated(
        tmp_path,
        retrieval_policy=CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.1),
    )

    failing = {check.name for check in report.checks if not check.passed}

    assert failing == {"retrieving candidates again reproduces the candidate sets"}


def test_mixing_embedding_spaces_in_one_channel_is_a_compatibility_failure(tmp_path: Path) -> None:
    inputs = build_inputs()
    other = AppearanceEvidence(
        policy=channel_policy("entity-appearance-comparison-v1"),
        measurement=appearance_measurement(
            embedding_space_id="sha256:another-space",
            contributions_a=(
                FeatureContribution(
                    physical_observation_id="frame-0001",
                    feature_refs=(feature_ref(1, space="sha256:another-space"),),
                ),
            ),
            contributions_b=(
                FeatureContribution(
                    physical_observation_id="frame-0002",
                    feature_refs=(feature_ref(2, space="sha256:another-space"),),
                ),
            ),
        ),
        findings=(),
    )
    rep = representation_evidence()
    assert rep.measurement is not None
    other_space = "sha256:another-representation-space"
    rep_other = dataclasses.replace(
        rep,
        measurement=dataclasses.replace(
            rep.measurement,
            representation_space_id=other_space,
            representations_a=tuple(
                dataclasses.replace(item, representation_space_id=other_space)
                for item in rep.measurement.representations_a
            ),
            representations_b=tuple(
                dataclasses.replace(item, representation_space_id=other_space)
                for item in rep.measurement.representations_b
            ),
        ),
    )
    resolutions = list(inputs.resolutions)
    for index, (appearance, representation) in enumerate(
        ((appearance_evidence(), rep), (other, rep_other))
    ):
        resolutions[index] = dataclasses.replace(
            resolutions[index],
            evidence=dataclasses.replace(
                resolutions[index].evidence,
                appearance=appearance,
                point_representation=representation,
            ),
        )

    report = evaluated(tmp_path, dataclasses.replace(inputs, resolutions=tuple(resolutions)))

    failing = {check.name: check for check in report.checks if not check.passed}
    assert list(failing) == ["no measured channel mixes embedding or representation spaces"]
    messages = failing["no measured channel mixes embedding or representation spaces"].failures
    assert any("appearance" in item for item in messages)
    assert any("point_representation" in item for item in messages)


# --- split diagnostics --------------------------------------------------------------------------


def test_split_diagnostics_are_evaluated_apart_from_merge_quality() -> None:
    points = (
        lattice((0.0, 0.0, 0.0), 0.6)
        + lattice((5.0, 0.0, 0.0), 0.6)
        + lattice((20.0, 0.0, 0.0), 0.6)
    )
    source = scene_source(dict(enumerate(points)))
    two_pieces = entity_over("multi", source, list(range(0, 54)), support_number=1)
    one_piece = entity_over("single", source, list(range(54, 81)), support_number=2)
    other_single = entity_over("other", source, list(range(54, 81)), support_number=3)
    policy = SplitDetectionPolicy(
        connectivity_radius_m=0.6,
        min_partition_points=5,
        min_partition_fraction=0.1,
        min_gap_m=1.0,
        max_dominant_fraction=0.9,
    )
    candidates = detect_split_candidates(
        [two_pieces, one_piece, other_single], source=source, policy=policy
    )

    result = evaluate_splits(
        candidates,
        [
            SplitReference(entity_ref=two_pieces.reference, contains_multiple_objects=True),
            SplitReference(entity_ref=one_piece.reference, contains_multiple_objects=False),
            SplitReference(entity_ref=other_single.reference, contains_multiple_objects=False),
        ],
    )

    assert result.multi_object_entities == 1
    assert dict(result.multi_object_outcomes) == {"suggested": 1}
    assert result.suggested_recall == 1.0
    assert dict(result.single_object_outcomes) == {"not_flagged": 2}
    assert result.false_suggestion_rate == 0.0
    assert evaluate_splits((), []).suggested_recall is None


# --- channel ablations --------------------------------------------------------------------------

GEOMETRY = GeometryComparisonPolicy(
    min_shared_support_jaccard=0.5,
    min_bounds_iou=0.5,
    min_bounds_containment=0.9,
    min_conflict_gap_m=0.5,
    min_extent_ratio=0.3,
)


def ablation_scene() -> dict[str, Entity]:
    """a and b: one pallet seen twice. c and d: a pallet and a person that overlap in space."""

    def labeled(name: str, x: float, number: int, label: str) -> Entity:
        return entity_at(
            name,
            (x, 0.0, 0.0),
            size=0.6,
            support_number=number,
            hypotheses=(
                make_hypothesis("hypothesis-0000", label, fused_evidence_id=fused_id(number)),
            ),
        )

    return {
        "a": labeled("a", 0.0, 1, "pallet"),
        "b": labeled("b", 0.1, 2, "pallet"),
        "c": labeled("c", 5.0, 3, "pallet"),
        "d": labeled("d", 5.1, 4, "person"),
    }


def arms() -> list[ResolutionArm]:
    semantic = SemanticCompatibilityPolicy(refinement_modifiers=())
    return [
        ResolutionArm(
            name="A geometry",
            builder=MatchEvidenceBuilder(ComparisonChannels(geometry=GEOMETRY)),
            policy=ConservativeResolutionPolicy(
                use_channels=(MatchChannel.GEOMETRY,), min_supporting_channels=1
            ),
        ),
        ResolutionArm(
            name="B geometry+semantic",
            builder=MatchEvidenceBuilder(ComparisonChannels(geometry=GEOMETRY, semantic=semantic)),
            policy=ConservativeResolutionPolicy(
                use_channels=(MatchChannel.GEOMETRY, MatchChannel.SEMANTIC),
                min_supporting_channels=1,
            ),
        ),
    ]


def test_a_channel_ablation_changes_only_the_channels_and_keeps_failures_visible() -> None:
    entities = ablation_scene()
    sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.1)
    )
    reference = reference_of({"P1": ["a", "b"], "P2": ["c"], "P3": ["d"]}, [("P2", "P3")])

    first, second = evaluate_channel_ablation(
        arms=arms(),
        entities=list(entities.values()),
        candidate_sets=sets,
        reference=reference,
        links=links_of(entities, *"abcd"),
        resolution_run_id=RUN,
    )

    assert isinstance(first, ArmEvaluation)
    assert (first.name, second.name) == ("A geometry", "B geometry+semantic")
    assert first.channels == ("geometry",) and second.channels == ("geometry", "semantic")
    # Só a geometria funde o par sobreposto que os labels dizem ser objetos diferentes.
    assert (first.identity.false_merge_entities, first.identity.false_merges) == (1, 1)
    assert (second.identity.false_merge_entities, second.identity.false_merges) == (0, 0)
    assert dict(first.decision_counts) == {"match": 2}
    assert dict(second.decision_counts) == {"match": 1, "unresolved": 1}
    # A fusão correta de a e b é preservada nos dois braços: só o canal mudou.
    assert first.identity.true_merges == second.identity.true_merges == 1
    assert first.resolve_seconds >= 0.0 and second.materialize_seconds >= 0.0


def test_ablation_arm_names_must_be_unique() -> None:
    entities = ablation_scene()
    sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.1)
    )
    duplicated = [arms()[0], arms()[0]]

    with pytest.raises(EntityResolutionEvaluationError, match="unique"):
        evaluate_channel_ablation(
            arms=duplicated,
            entities=list(entities.values()),
            candidate_sets=sets,
            reference=reference_of({"P1": ["a"]}),
            links=[],
            resolution_run_id=RUN,
        )


def test_the_writer_and_the_evaluator_agree_on_a_written_ablation_run(tmp_path: Path) -> None:
    entities = ablation_scene()
    sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.1)
    )
    arm = arms()[1]
    resolutions = resolve_candidate_pairs(entities.values(), sets, arm.builder, arm.policy)
    materialization = materialize_resolved_entities(
        entities.values(), [item.decision for item in resolutions], resolution_run_id=RUN
    )
    directory = tmp_path / "artifact"
    EntityResolutionRunWriter(
        output_dir=directory, run_id=RUN, lineage=LINEAGE, code_version="v"
    ).write(candidate_sets=sets, resolutions=resolutions, materialization=materialization)

    report = evaluate_entity_resolution(
        EntityResolutionRunReader(directory),
        entities=entities.values(),
        reference=reference_of({"P1": ["a", "b"], "P2": ["c"], "P3": ["d"]}, [("P2", "P3")]),
        links=links_of(entities, *"abcd"),
        retrieval_policy=CandidateRetrievalPolicy(centroid_radius_m=1.0, bounds_margin_m=0.1),
        reference_set_id="synthetic-1",
        split_reference=[],
    )

    assert report.passed
    assert (report.identity.false_merge_rate, report.identity.duplicate_rate) == (0.0, 0.0)
    assert report.retrieval.retrieval_recall == 1.0

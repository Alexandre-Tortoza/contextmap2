"""Relation evaluation: false, missed, unresolved and unretrieved relations, kept apart."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from relation_builders import entity_ref
from relation_run_fixture import Run, build_run, write_run

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.evaluation import (
    EvaluationReport,
    LabelNormalization,
    MetricStatus,
    ObservationRef,
    PredicateRule,
    ReferenceSampleId,
    ReferenceSetIdentity,
    RelationAnnotation,
    RelationAnnotationSet,
    RelationPredicateEvaluation,
    RelationStatus,
    SpatialRelationsEvaluationError,
    SpatialRelationsEvaluationReport,
    default_metric_registry,
    evaluate_spatial_relations,
    spatial_relations_evaluation_report,
)
from contextmap.evaluation.spatial_relations import _consistency
from contextmap.ingestion import SourceObservationId
from contextmap.spatial_relations import (
    RelationPredicate,
    RelationState,
    SpatialRelationsRunReader,
)

P = RelationPredicate
HOLDS = RelationStatus.HOLDS
NOT = RelationStatus.DOES_NOT_HOLD
AMBIGUOUS = RelationStatus.AMBIGUOUS
UNKNOWN = RelationStatus.UNKNOWN

FLOOR, CRATE, PALLET, HOVER = "id-floor", "id-crate", "id-pallet", "id-hover"
IDENTITIES = {
    entity_ref(1): FLOOR,
    entity_ref(2): CRATE,
    entity_ref(3): PALLET,
    entity_ref(4): HOVER,
}
ANCHOR = (
    ObservationRef(
        sample_id=ReferenceSampleId("sample-0000"), observation_id=SourceObservationId("frame-0000")
    ),
)
RULES = (
    PredicateRule(predicate="next to", symmetric=True),
    PredicateRule(predicate="touching", symmetric=True),
    PredicateRule(predicate="beside", symmetric=True),
    PredicateRule(predicate="above", inverse="below"),
    PredicateRule(predicate="below", inverse="above"),
    PredicateRule(predicate="on top of", inverse="supports"),
    PredicateRule(predicate="supports", inverse="on top of"),
    PredicateRule(predicate="leaning against"),
)


def _relation(
    number: int, subject: str, predicate: str, obj: str, status: RelationStatus
) -> RelationAnnotation:
    return RelationAnnotation(
        relation_id=f"rel-{number:02d}",
        subject_identity_id=subject,
        predicate=predicate,
        object_identity_id=obj,
        status=status,
        anchors=ANCHOR,
    )


def _reference(
    *relations: RelationAnnotation, rules: tuple[PredicateRule, ...] = RULES
) -> RelationAnnotationSet:
    return RelationAnnotationSet(
        normalization=LabelNormalization(policy_id="casefold-exact/1"),
        predicate_rules=rules,
        relations=relations,
    )


CORE = (
    _relation(1, CRATE, "on top of", FLOOR, HOLDS),
    _relation(2, CRATE, "above", FLOOR, HOLDS),
    _relation(3, FLOOR, "next to", CRATE, HOLDS),
    _relation(4, HOVER, "on top of", FLOOR, HOLDS),
    _relation(5, CRATE, "leaning against", FLOOR, NOT),
    _relation(6, HOVER, "touching", FLOOR, NOT),
    _relation(7, CRATE, "touching", HOVER, NOT),
    _relation(8, PALLET, "next to", FLOOR, HOLDS),
    _relation(9, FLOOR, "above", HOVER, NOT),
    _relation(10, CRATE, "next to", HOVER, NOT),
)
EXTRA = (
    _relation(11, "id-ghost", "next to", FLOOR, HOLDS),
    _relation(12, CRATE, "above", HOVER, AMBIGUOUS),
    _relation(13, FLOOR, "leaning against", HOVER, UNKNOWN),
    _relation(14, PALLET, "beside", CRATE, HOLDS),
)


@pytest.fixture(scope="module")
def run() -> Run:
    return build_run()


@pytest.fixture(scope="module")
def artifact(run: Run, tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("evaluation") / "relations"
    write_run(run, directory)
    return directory


def _evaluate(
    artifact: Path,
    reference: RelationAnnotationSet,
    identities: dict[ResolvedEntityReference, str] | None = None,
    **options: object,
) -> SpatialRelationsEvaluationReport:
    return evaluate_spatial_relations(
        SpatialRelationsRunReader(artifact),
        reference=reference,
        identity_of_entity=IDENTITIES if identities is None else identities,
        **options,  # type: ignore[arg-type]
    )


def _counts(item: RelationPredicateEvaluation) -> dict[str, int]:
    return {
        "holds": item.annotated_holds,
        "not": item.annotated_does_not_hold,
        "tp": item.true_positives,
        "fp": item.false_positives,
        "tn": item.true_negatives,
        "unresolved": item.missed_unresolved,
        "rejected": item.missed_rejected,
        "not_retrieved": item.missed_not_retrieved,
        "unresolved_negative": item.unresolved_on_negatives,
        "unannotated": item.unannotated_supported,
    }


# --- per predicate, apart ---


def test_every_canonical_predicate_has_its_own_evaluation_and_there_is_no_aggregate(
    artifact: Path,
) -> None:
    report = _evaluate(artifact, _reference(*CORE, *EXTRA))
    assert [item.predicate for item in report.predicates] == sorted(P, key=lambda item: item.value)
    assert not hasattr(report, "f1")
    assert not hasattr(report, "score")


def test_the_counts_of_each_predicate_are_exact(artifact: Path) -> None:
    report = _evaluate(artifact, _reference(*CORE, *EXTRA))
    assert _counts(report.predicate(P.ON_TOP_OF)) == {
        "holds": 2, "not": 0, "tp": 1, "fp": 0, "tn": 0, "unresolved": 1, "rejected": 0,
        "not_retrieved": 0, "unresolved_negative": 0, "unannotated": 0,
    }  # fmt: skip
    assert _counts(report.predicate(P.ABOVE)) == {
        "holds": 1, "not": 1, "tp": 1, "fp": 0, "tn": 1, "unresolved": 0, "rejected": 0,
        "not_retrieved": 0, "unresolved_negative": 0, "unannotated": 1,
    }  # fmt: skip
    assert _counts(report.predicate(P.BELOW)) == {
        "holds": 1, "not": 1, "tp": 1, "fp": 0, "tn": 1, "unresolved": 0, "rejected": 0,
        "not_retrieved": 0, "unresolved_negative": 0, "unannotated": 1,
    }  # fmt: skip
    assert _counts(report.predicate(P.NEXT_TO)) == {
        "holds": 4, "not": 2, "tp": 2, "fp": 2, "tn": 0, "unresolved": 0, "rejected": 0,
        "not_retrieved": 2, "unresolved_negative": 0, "unannotated": 2,
    }  # fmt: skip
    assert _counts(report.predicate(P.TOUCHING)) == {
        "holds": 0, "not": 4, "tp": 0, "fp": 0, "tn": 2, "unresolved": 0, "rejected": 0,
        "not_retrieved": 0, "unresolved_negative": 2, "unannotated": 2,
    }  # fmt: skip
    assert _counts(report.predicate(P.LEANING_AGAINST)) == {
        "holds": 0, "not": 1, "tp": 0, "fp": 0, "tn": 1, "unresolved": 0, "rejected": 0,
        "not_retrieved": 0, "unresolved_negative": 0, "unannotated": 0,
    }  # fmt: skip
    assert _counts(report.predicate(P.INSIDE)) == dict.fromkeys(
        _counts(report.predicate(P.INSIDE)), 0
    )


def test_false_missed_unresolved_and_unretrieved_are_separately_measurable(artifact: Path) -> None:
    report = _evaluate(artifact, _reference(*CORE, *EXTRA))
    next_to = report.predicate(P.NEXT_TO)
    assert next_to.precision == 0.5
    assert next_to.recall == 0.5
    assert next_to.f1 == 0.5
    assert next_to.false_relation_rate == 1.0
    assert next_to.missed_relation_rate == 0.5
    assert next_to.candidate_retrieval_recall == 0.5
    assert next_to.unresolved_rate == 0.0
    on_top = report.predicate(P.ON_TOP_OF)
    assert (on_top.precision, on_top.recall) == (1.0, 0.5)
    assert on_top.f1 == pytest.approx(2 / 3)
    assert on_top.unresolved_rate == 0.5
    assert on_top.false_relation_rate is None
    assert on_top.candidate_retrieval_recall == 1.0
    touching = report.predicate(P.TOUCHING)
    assert touching.false_relation_rate == 0.0
    assert touching.unresolved_rate == 0.5
    assert touching.precision is None and touching.recall is None and touching.f1 is None


def test_a_predicate_with_no_annotation_has_undefined_rates_never_zero(artifact: Path) -> None:
    inside = _evaluate(artifact, _reference(*CORE)).predicate(P.INSIDE)
    for value in (
        inside.precision,
        inside.recall,
        inside.f1,
        inside.false_relation_rate,
        inside.missed_relation_rate,
        inside.unresolved_rate,
        inside.candidate_retrieval_recall,
    ):
        assert value is None


def test_retrieval_misses_are_kept_apart_with_the_reason(artifact: Path) -> None:
    report = _evaluate(artifact, _reference(*CORE))
    assert report.retrieval_misses == (("pair_not_enumerated_or_predicate_not_selected", 2),)
    excluded = _evaluate(artifact, _reference(_relation(1, FLOOR, "on top of", CRATE, HOLDS)))
    assert excluded.predicate(P.ON_TOP_OF).missed_not_retrieved == 1
    assert excluded.retrieval_misses == (("excluded:subject_not_on_directed_side", 1),)


# --- failures that are not relation failures ---


def test_a_reference_without_an_entity_is_an_entity_failure_not_a_relation_error(
    artifact: Path,
) -> None:
    with_ghost = _evaluate(artifact, _reference(*CORE, EXTRA[0]))
    without = _evaluate(artifact, _reference(*CORE))
    assert with_ghost.predicates == without.predicates
    assert with_ghost.unmatched.reference_without_entity == 2
    assert without.unmatched.reference_without_entity == 0


def test_an_identity_matched_to_several_entities_is_skipped_and_never_repaired(
    artifact: Path,
) -> None:
    duplicated = {**IDENTITIES, entity_ref(3): CRATE}
    report = _evaluate(artifact, _reference(*CORE), identities=duplicated)
    assert report.unmatched.identities_with_several_entities == (CRATE,)
    baseline = _evaluate(artifact, _reference(*CORE))
    assert (
        report.predicate(P.NEXT_TO).annotated_holds < baseline.predicate(P.NEXT_TO).annotated_holds
    )


def test_ambiguous_unknown_and_conflicting_reference_is_counted_never_scored(
    artifact: Path,
) -> None:
    report = _evaluate(artifact, _reference(*CORE, EXTRA[1], EXTRA[2]))
    assert report.unmatched.ambiguous_reference == 2
    assert report.unmatched.unknown_reference == 1
    assert report.predicate(P.LEANING_AGAINST).annotated_holds == 0
    conflicting = _evaluate(
        artifact,
        _reference(
            _relation(1, CRATE, "on top of", FLOOR, HOLDS),
            _relation(2, CRATE, "on top of", FLOOR, NOT),
        ),
    )
    assert conflicting.unmatched.conflicting_reference == 1
    assert conflicting.predicate(P.ON_TOP_OF).annotated_holds == 0


def test_unmapped_wordings_are_reported_and_not_scored(artifact: Path) -> None:
    report = _evaluate(artifact, _reference(*CORE, EXTRA[3]))
    assert report.unmatched.unmapped_predicates == ("beside",)
    assert report.predicates == _evaluate(artifact, _reference(*CORE)).predicates


def test_unannotated_predictions_are_reported_not_scored_as_false_positives(
    artifact: Path,
) -> None:
    report = _evaluate(artifact, _reference(*CORE))
    assert report.predicate(P.TOUCHING).unannotated_supported == 2
    assert report.predicate(P.TOUCHING).false_positives == 0


def test_a_reference_vocabulary_that_contradicts_the_taxonomy_is_refused(artifact: Path) -> None:
    symmetric_above = (PredicateRule(predicate="above", symmetric=True),)
    with pytest.raises(SpatialRelationsEvaluationError, match="symmetric"):
        _evaluate(artifact, _reference(rules=symmetric_above))
    wrong_inverse = (
        PredicateRule(predicate="above", inverse="next to"),
        PredicateRule(predicate="next to", inverse="above"),
    )
    with pytest.raises(SpatialRelationsEvaluationError, match="inverse"):
        _evaluate(artifact, _reference(rules=wrong_inverse))
    omitted = (PredicateRule(predicate="above"), PredicateRule(predicate="next to"))
    _evaluate(artifact, _reference(rules=omitted))


# --- consistency ---


def test_the_relations_of_a_decided_run_are_internally_consistent(artifact: Path) -> None:
    report = _evaluate(artifact, _reference(*CORE))
    assert report.consistency_violations == ()


def _relations(run: Run) -> dict:  # type: ignore[type-arg]
    return {
        (item.subject_entity_ref, item.predicate, item.object_entity_ref): item
        for item in run.decisions.relations
    }


def test_an_inverse_that_disagrees_is_a_violation(run: Run) -> None:
    relations = _relations(run)
    key = (entity_ref(1), P.BELOW, entity_ref(2))
    relations[key] = dataclasses.replace(relations[key], state=RelationState.REJECTED)
    (violation,) = _consistency(relations)
    assert violation.kind == "inverse"
    assert len(violation.relation_ids) == 2


def test_a_symmetric_twin_that_disagrees_is_a_violation(run: Run) -> None:
    relations = _relations(run)
    key = (entity_ref(2), P.NEXT_TO, entity_ref(1))
    relations[key] = dataclasses.replace(relations[key], state=RelationState.REJECTED)
    (violation,) = _consistency(relations)
    assert violation.kind == "symmetric"


def test_a_directed_predicate_supported_both_ways_is_a_violation(run: Run) -> None:
    relations = _relations(run)
    forward = (entity_ref(2), P.ON_TOP_OF, entity_ref(1))
    backward = (entity_ref(1), P.ON_TOP_OF, entity_ref(2))
    relations[backward] = dataclasses.replace(
        relations[forward],
        relation_id=relations[forward].relation_id.replace("relation", "relation-x"),  # type: ignore[arg-type]
        subject_entity_ref=entity_ref(1),
        object_entity_ref=entity_ref(2),
    )
    kinds = {item.kind for item in _consistency(relations)}
    assert kinds == {"mutual_support"}


# --- reproducibility ---


def test_the_report_names_the_run_the_resolution_the_policies_and_the_reference(
    artifact: Path,
) -> None:
    reader = SpatialRelationsRunReader(artifact)
    report = _evaluate(artifact, _reference(*CORE), code_version="abc123")
    manifest = reader.manifest
    assert report.spatial_relations_run_id == manifest.run_id
    assert report.entity_resolution_run_id == manifest.lineage.entity_resolution_run_id
    assert report.entity_resolution_artifact_digest == "sha256:resolution"
    assert report.geometric_map_id == manifest.lineage.geometric_map_id
    assert report.taxonomy_version == manifest.taxonomy_version
    assert (
        report.policies["geometric"]["fingerprint"]
        == (manifest.policies["geometric"]["fingerprint"])
    )
    assert report.policies["decision"]["id"] == manifest.policies["decision"]["policy_id"]
    assert report.reference_normalization == "casefold-exact/1"
    assert report.reference_relation_count == len(CORE)
    assert (report.evaluator_id, report.evaluator_version) == ("spatial-relations-evaluator", "1")
    assert report.code_version == "abc123"
    assert report.configuration_digest().startswith("sha256:")


def test_the_same_inputs_give_the_same_report_and_the_artifact_is_untouched(
    artifact: Path,
) -> None:
    def digest() -> str:
        joined = b"".join(
            path.read_bytes() for path in sorted(artifact.rglob("*")) if path.is_file()
        )
        return hashlib.sha256(joined).hexdigest()

    before = digest()
    first = _evaluate(artifact, _reference(*CORE, *EXTRA))
    second = _evaluate(artifact, _reference(*EXTRA[::-1], *CORE[::-1]))
    assert first.to_dict() == second.to_dict()
    assert digest() == before


def test_the_report_is_plain_json_with_every_count_and_rate(artifact: Path) -> None:
    record = _evaluate(artifact, _reference(*CORE, *EXTRA)).to_dict()
    record = json.loads(json.dumps(record))
    next_to = next(item for item in record["predicates"] if item["predicate"] == "next_to")
    assert next_to["false_positives"] == 2
    assert next_to["precision"] == 0.5
    assert next_to["candidate_retrieval_recall"] == 0.5
    assert record["retrieval_misses"][0]["count"] == 2
    assert record["unmatched"]["unmapped_predicates"] == ["beside"]


# --- the common envelope ---


REFERENCE_SET = ReferenceSetIdentity(
    reference_set_id="synthetic-relations", version="1.0.0", digest="sha256:reference"
)


def _envelope(artifact: Path, reference: RelationAnnotationSet) -> EvaluationReport:
    report = _evaluate(artifact, reference)
    return spatial_relations_evaluation_report(
        report, registry=default_metric_registry(), reference_set=REFERENCE_SET
    )


def test_the_envelope_reports_the_registry_metrics_per_predicate_never_as_one_score(
    artifact: Path,
) -> None:
    envelope = _envelope(artifact, _reference(*CORE))
    results = envelope.quality_metrics
    assert {item.metric for item in results} == {
        "relations.f1",
        "relations.negative_violation.rate",
    }
    assert all(item.strata for item in results)
    predicates = {dict(item.strata)["predicate"] for item in results}
    assert predicates == {"above", "below", "leaning_against", "next_to", "on_top_of", "touching"}
    f1 = {dict(item.strata)["predicate"]: item for item in results if item.metric == "relations.f1"}
    assert f1["next_to"].value == 0.5
    assert f1["touching"].status is MetricStatus.NOT_APPLICABLE
    assert f1["touching"].value is None
    assert envelope.performance_metrics == ()


def test_the_envelope_carries_the_reproducibility_metadata(artifact: Path) -> None:
    metadata = _envelope(artifact, _reference(*CORE)).reproducibility
    assert metadata.reference_set == REFERENCE_SET
    assert {item.kind for item in metadata.input_artifacts} == {
        "spatial_relations_run",
        "entity_resolution_run",
    }
    assert metadata.evaluator.evaluator_id == "spatial-relations-evaluator"
    assert metadata.configuration_digest is not None


def test_no_annotated_population_is_not_applicable_and_never_zero(artifact: Path) -> None:
    envelope = _envelope(artifact, _reference(EXTRA[3]))
    assert {item.status for item in envelope.quality_metrics} == {MetricStatus.NOT_APPLICABLE}
    assert all(item.strata == () for item in envelope.quality_metrics)

"""Annotation quality-assurance tests: malformed fixtures fail, ambiguity does not."""

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from qa_builders import (
    A_CELLS,
    B_CELLS,
    EXACT,
    POLICY,
    clean_sets,
    correspondence,
    frame,
    frame_regions,
    geometry_set,
    identity_set,
    mask,
    occurrence,
    ref,
    relation,
    relations_set,
    semantic_record,
    semantics_set,
    visibility_set,
    write_qa_reference,
)
from reference_set_builders import sample_id

from contextmap.evaluation.annotation_qa import (
    QA_REPORT_SCHEMA,
    AnnotationQaError,
    AnnotationQaPolicy,
    AnnotationQaReport,
    certify_reference_set,
    check_annotation_quality,
)
from contextmap.evaluation.annotations import (
    CASEFOLD_ALIAS_POLICY,
    AnnotationFamily,
    AnnotationSet,
    Coverage,
    GeometryAnnotationSet,
    GeometryCorrespondence,
    IdentityAnnotationSet,
    IdentityOccurrence,
    IdentityScope,
    LabelAlias,
    LabelNormalization,
    PhysicalIdentity,
    RegionAnnotation,
    RegionAnnotationSet,
    RelationAnnotation,
    RelationAnnotationSet,
    RelationStatus,
    SemanticAnnotationRecord,
    SemanticAnnotationSet,
    SemanticStatus,
    VisibilityAnnotation,
    VisibilityAnnotationSet,
    VisibilityLevel,
)
from contextmap.evaluation.reference_integrity import (
    IntegritySeverity,
    ReferenceSetIntegrityError,
)
from contextmap.evaluation.reference_set import (
    ReferenceSplit,
    SplitRole,
    SplitScheme,
    SplitUnit,
    write_reference_set,
)
from contextmap.visual_perception import BoundingBox

Value = AnnotationSet | tuple[AnnotationSet, str]


def _qa(tmp_path: Path, sets: Mapping[str, Value], **overrides: Any) -> AnnotationQaReport:
    manifest = write_qa_reference(tmp_path, sets, **overrides)
    return check_annotation_quality(manifest, tmp_path, policy=POLICY)


def _blockers(report: AnnotationQaReport) -> set[str]:
    return {item.code for item in report.findings if item.severity is IntegritySeverity.BLOCKER}


def _warnings(report: AnnotationQaReport) -> set[str]:
    return {item.code for item in report.findings if item.severity is IntegritySeverity.WARNING}


def _with(extra: Mapping[str, Value] | None = None, **changes: Value) -> dict[str, Value]:
    sets: dict[str, Value] = dict(clean_sets())
    sets.update(extra or {})
    sets.update(changes)
    return sets


# ------------------------------------------------------------------------- clean set


def test_a_consistent_reference_set_passes_with_no_qa_finding(tmp_path: Path) -> None:
    report = _qa(tmp_path, clean_sets())

    assert report.is_valid
    assert report.integrity.is_valid
    assert _blockers(report) == set()
    assert report.findings == ()
    assert report.permissible == ()
    assert report.disagreements == ()
    assert report.annotation_files_checked == 7
    assert report.policy == POLICY


def test_the_report_is_machine_readable_and_versioned(tmp_path: Path) -> None:
    report = _qa(tmp_path, clean_sets())

    record = json.loads(json.dumps(report.to_record()))

    assert record["schema"] == QA_REPORT_SCHEMA
    assert record["valid"] is True
    assert record["policy"]["policy_id"] == "annotation-qa/1"
    assert record["reference_set"]["digest"] == report.reference_set.digest
    for group in ("integrity", "blockers", "warnings", "permissible", "disagreements"):
        assert group in record


def test_the_policy_has_no_meaningless_tolerances() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        AnnotationQaPolicy(
            policy_id="p",
            duplicate_pixel_tolerance_px=0.0,
            duplicate_point_tolerance_m=0.1,
            region_match_iou=0.5,
        )
    with pytest.raises(ValueError, match="IoU"):
        AnnotationQaPolicy(
            policy_id="p",
            duplicate_pixel_tolerance_px=0.5,
            duplicate_point_tolerance_m=0.1,
            region_match_iou=0.0,
        )


# ------------------------------------------------------------- regions: size and shape


def _frame0(**changes: object) -> RegionAnnotationSet:
    return RegionAnnotationSet(frames=(replace(frame_regions(0), **changes), frame_regions(1)))  # type: ignore[arg-type]


def _regions(*regions: RegionAnnotation, **changes: object) -> RegionAnnotationSet:
    return _frame0(regions=regions, **changes)


A0 = RegionAnnotation(region_id="a-0", mask=mask(*A_CELLS))
B0 = RegionAnnotation(region_id="b-0", mask=mask(*B_CELLS))


@pytest.mark.parametrize(
    ("code", "regions"),
    [
        (
            "mask-size-mismatch",
            _regions(RegionAnnotation(region_id="a-0", mask=mask(0, 1, width=3, height=3)), B0),
        ),
        ("area-size-mismatch", _frame0(valid_areas=(mask(0, 1, width=3, height=3),))),
        ("empty-mask", _regions(RegionAnnotation(region_id="a-0", mask=mask()), B0)),
        (
            "box-out-of-image",
            _regions(
                A0,
                B0,
                RegionAnnotation(
                    region_id="c-0", box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=2)
                ),
            ),
        ),
        (
            "mask-outside-box",
            _regions(
                RegionAnnotation(
                    region_id="a-0",
                    mask=mask(*A_CELLS),
                    box=BoundingBox(x_min=2, y_min=0, x_max=4, y_max=1),
                ),
                B0,
            ),
        ),
        (
            "duplicate-region-geometry",
            _regions(A0, RegionAnnotation(region_id="b-0", mask=mask(*A_CELLS))),
        ),
        ("region-in-exclusion-area", _frame0(exclusion_areas=(mask(*A_CELLS),))),
        ("region-outside-valid-area", _frame0(valid_areas=(mask(*B_CELLS),))),
    ],
)
def test_malformed_region_annotations_are_blockers(
    tmp_path: Path, code: str, regions: RegionAnnotationSet
) -> None:
    report = _qa(tmp_path, _with(regions=regions))

    assert code in _blockers(report)
    assert not report.is_valid


@pytest.mark.parametrize(
    ("code", "regions"),
    [
        ("region-overlaps-exclusion-area", _frame0(exclusion_areas=(mask(1),))),
        ("region-extends-outside-valid-area", _frame0(valid_areas=(mask(0, 10, 11),))),
    ],
)
def test_partial_overlaps_with_declared_areas_are_warnings_not_blockers(
    tmp_path: Path, code: str, regions: RegionAnnotationSet
) -> None:
    report = _qa(tmp_path, _with(regions=regions))

    assert code in _warnings(report)
    assert code not in _blockers(report)
    assert report.is_valid


# ------------------------------------------------------------------------- geometry


def _geometry(
    *extra: GeometryCorrespondence, first_calibration: str = "cam0"
) -> GeometryAnnotationSet:
    return replace(
        geometry_set(),
        correspondences=(
            correspondence(0, calibration=first_calibration),
            correspondence(1),
            *extra,
        ),
    )


@pytest.mark.parametrize(
    ("code", "geometry"),
    [
        ("geometry-unknown-calibration", _geometry(first_calibration="cam9")),
        (
            "pixel-outside-image",
            replace(
                geometry_set(),
                correspondences=(correspondence(0, pixel=(10.0, 10.0)), correspondence(1)),
            ),
        ),
        (
            "point-frame-inconsistent",
            _geometry(correspondence(0, name="corr-0b", pixel=(3.0, 2.0), point_frame="base_link")),
        ),
        (
            "conflicting-correspondence",
            _geometry(correspondence(0, name="corr-0b", pixel=(1.2, 1.0), point=(9.0, 9.0, 9.0))),
        ),
    ],
)
def test_inconsistent_geometry_annotations_are_blockers(
    tmp_path: Path, code: str, geometry: GeometryAnnotationSet
) -> None:
    report = _qa(tmp_path, _with(geometry=geometry))

    assert code in _blockers(report)
    assert not report.is_valid


def test_a_repeated_correspondence_is_a_warning(tmp_path: Path) -> None:
    geometry = _geometry(
        correspondence(0, name="corr-0b", pixel=(1.1, 1.0), point=(2.01, 0.0, 1.0))
    )

    report = _qa(tmp_path, _with(geometry=geometry))

    assert "duplicate-correspondence" in _warnings(report)
    assert report.is_valid


# -------------------------------------------------------------------------- identity


def _identities(
    *,
    a: PhysicalIdentity | None = None,
    b: PhysicalIdentity | None = None,
    scope: tuple[IdentityScope, ...] | None = None,
) -> IdentityAnnotationSet:
    base = identity_set()
    first, second = base.identities
    return replace(
        base,
        scope=base.scope if scope is None else scope,
        identities=(a or first, b or second),
    )


def _identity_with_a(*occurrences: IdentityOccurrence) -> IdentityAnnotationSet:
    base = identity_set()
    return _identities(a=replace(base.identities[0], occurrences=occurrences))


def test_an_occurrence_of_an_unknown_region_is_a_blocker(tmp_path: Path) -> None:
    identities = _identity_with_a(occurrence(0, "ghost"), occurrence(1, "a-1"))

    report = _qa(tmp_path, _with(identity=identities))

    assert "identity-unknown-region" in _blockers(report)


def test_a_region_cannot_be_two_identities(tmp_path: Path) -> None:
    base = identity_set()
    identities = _identities(
        b=replace(base.identities[1], occurrences=(occurrence(0, "a-0"), occurrence(1, "b-1")))
    )

    report = _qa(tmp_path, _with(identity=identities))

    assert "region-claimed-by-two-identities" in _blockers(report)
    assert not report.is_valid


def test_one_identity_in_two_regions_of_an_observation_is_a_warning(tmp_path: Path) -> None:
    identities = _identity_with_a(occurrence(0, "a-0"), occurrence(0, None), occurrence(1, "a-1"))

    report = _qa(tmp_path, _with(identity=identities))

    assert "identity-repeated-in-observation" in _warnings(report)
    assert report.is_valid


def test_an_occurrence_outside_the_annotated_scope_is_a_warning(tmp_path: Path) -> None:
    base = identity_set()
    identities = _identities(scope=base.scope[:1])

    report = _qa(tmp_path, _with(identity=identities))

    assert "occurrence-outside-scope" in _warnings(report)


def test_one_identity_with_disjoint_concepts_across_observations_is_flagged(tmp_path: Path) -> None:
    semantics = SemanticAnnotationSet(
        normalization=EXACT,
        records=(
            semantic_record(0, "a-0", ("pallet",)),
            semantic_record(0, "b-0", ("crate",)),
            semantic_record(1, "a-1", ("forklift",)),
            semantic_record(1, "b-1", ("crate",)),
        ),
    )

    report = _qa(tmp_path, _with(semantics=semantics))

    assert "identity-semantic-conflict" in _warnings(report)
    assert report.is_valid


# ------------------------------------------------------------------------- relations


def _relations(*extra: RelationAnnotation) -> RelationAnnotationSet:
    return relations_set(*extra)


def test_a_relation_between_unknown_identities_is_a_blocker(tmp_path: Path) -> None:
    relations = _relations(
        relation("rel-4", "ghost", "next to", "pallet-B", RelationStatus.HOLDS, ref(0))
    )

    report = _qa(tmp_path, _with(relations=relations))

    assert "relation-unknown-identity" in _blockers(report)


def test_a_relation_anchored_where_an_identity_is_not_seen_is_a_blocker(tmp_path: Path) -> None:
    relations = _relations(
        relation("rel-4", "pallet-A", "supports", "pallet-B", RelationStatus.HOLDS, ref(2))
    )

    report = _qa(tmp_path, _with(relations=relations))

    assert "relation-anchor-without-identity" in _blockers(report)


def test_a_symmetric_predicate_cannot_hold_one_way_only(tmp_path: Path) -> None:
    relations = _relations(
        relation("rel-4", "pallet-B", "next to", "pallet-A", RelationStatus.DOES_NOT_HOLD, ref(0))
    )

    report = _qa(tmp_path, _with(relations=relations))

    assert "symmetry-violation" in _blockers(report)


def test_an_inverse_predicate_cannot_contradict_its_inverse(tmp_path: Path) -> None:
    base = relations_set()
    relations = replace(
        base,
        relations=(
            relation("rel-2", "pallet-A", "on top of", "pallet-B", RelationStatus.DOES_NOT_HOLD),
            relation("rel-3", "pallet-B", "supports", "pallet-A", RelationStatus.HOLDS),
        ),
    )

    report = _qa(tmp_path, _with(relations=relations))

    assert "inverse-violation" in _blockers(report)
    assert "symmetry-violation" not in _blockers(report)


def test_a_consistent_inverse_pair_is_accepted(tmp_path: Path) -> None:
    report = _qa(tmp_path, clean_sets())

    assert not {"inverse-violation", "symmetry-violation"} & _blockers(report)


def test_the_same_relation_with_two_statuses_is_a_conflict_and_with_one_a_duplicate(
    tmp_path: Path,
) -> None:
    conflicting = _relations(
        relation("rel-4", "pallet-A", "next to", "pallet-B", RelationStatus.DOES_NOT_HOLD, ref(0))
    )
    repeated = _relations(
        relation("rel-4", "pallet-A", "next to", "pallet-B", RelationStatus.HOLDS, ref(0))
    )

    conflict_report = _qa(tmp_path / "conflict", _with(relations=conflicting))
    duplicate_report = _qa(tmp_path / "duplicate", _with(relations=repeated))

    assert "conflicting-relation" in _blockers(conflict_report)
    assert "duplicate-relation" in _warnings(duplicate_report)
    assert duplicate_report.is_valid


# ----------------------------------------------------- semantics and visibility


def test_a_semantic_record_of_an_unknown_region_is_a_blocker(tmp_path: Path) -> None:
    semantics = replace(
        semantics_set(),
        records=(*semantics_set().records, semantic_record(0, "ghost", ("pallet",))),
    )

    report = _qa(tmp_path, _with(semantics=semantics))

    assert "semantic-unknown-region" in _blockers(report)


def _visibility(**annotation: Any) -> VisibilityAnnotationSet:
    return visibility_set(
        VisibilityAnnotation(
            sample_id=sample_id(0),
            observation_id=frame(0),
            **annotation,
        )
    )


@pytest.mark.parametrize(
    ("code", "extra"),
    [
        (
            "visibility-unknown-target",
            {"region_id": "ghost", "level": VisibilityLevel.FULLY_VISIBLE},
        ),
        ("visibility-unknown-target", {"identity_id": "ghost", "level": VisibilityLevel.UNKNOWN}),
        (
            "visibility-level-fraction-mismatch",
            {
                "region_id": "b-0",
                "level": VisibilityLevel.FULLY_VISIBLE,
                "occluded_fraction": 0.4,
            },
        ),
        (
            "visibility-level-fraction-mismatch",
            {"region_id": "b-0", "level": VisibilityLevel.UNKNOWN, "occluded_fraction": 0.4},
        ),
        (
            "visibility-contradicts-occurrence",
            {"identity_id": "pallet-A", "level": VisibilityLevel.NOT_VISIBLE},
        ),
    ],
)
def test_inconsistent_visibility_annotations_are_blockers(
    tmp_path: Path, code: str, extra: dict[str, object]
) -> None:
    report = _qa(tmp_path, _with(visibility=_visibility(**extra)))

    assert code in _blockers(report)


# ------------------------------------------------ ambiguity is permissible, not a blocker


def test_ambiguity_and_unknown_are_reported_apart_from_blockers(tmp_path: Path) -> None:
    semantics = SemanticAnnotationSet(
        normalization=EXACT,
        records=(
            semantic_record(0, "a-0", ("pallet",)),
            semantic_record(0, "b-0", ("crate", "box"), SemanticStatus.AMBIGUOUS),
            SemanticAnnotationRecord(
                sample_id=sample_id(0),
                observation_id=frame(0),
                region_id=None,
                status=SemanticStatus.UNKNOWN,
                concepts=(),
            ),
            semantic_record(1, "a-1", ("pallet",)),
            semantic_record(1, "b-1", ("crate",)),
        ),
    )
    relations = _relations(
        relation("rel-4", "pallet-A", "supports", "pallet-B", RelationStatus.AMBIGUOUS, ref(0))
    )
    visibility = _visibility(region_id="b-0", level=VisibilityLevel.UNKNOWN)
    base_identity = identity_set()
    identity = _identities(
        scope=(
            base_identity.scope[0],
            IdentityScope(
                sample_id=sample_id(1), observation_id=frame(1), coverage=Coverage.PARTIAL
            ),
        )
    )

    report = _qa(
        tmp_path,
        _with(semantics=semantics, relations=relations, visibility=visibility, identity=identity),
    )

    kinds = {item.kind for item in report.permissible}
    assert {
        "ambiguous-semantics",
        "unknown-semantics",
        "ambiguous-relation",
        "unknown-visibility",
        "partial-identity-coverage",
    } <= kinds
    assert _blockers(report) == set()
    assert report.is_valid


# ---------------------------------------------------------- inter-annotator disagreement


def _second_semantics(*records: SemanticAnnotationRecord) -> tuple[SemanticAnnotationSet, str]:
    return SemanticAnnotationSet(normalization=EXACT, records=records), "annotator-b"


def test_disagreement_between_annotators_stays_visible_and_is_not_a_blocker(
    tmp_path: Path,
) -> None:
    second = _second_semantics(
        semantic_record(0, "a-0", ("pallet",)),
        semantic_record(0, "b-0", ("box",)),
        semantic_record(1, "a-1", ("pallet", "skid"), SemanticStatus.AMBIGUOUS),
        semantic_record(1, "b-1", ("crate", "crate box")),
    )

    report = _qa(tmp_path, _with({"semantics-b": second}))

    summary = next(
        item for item in report.disagreements if item.family is AnnotationFamily.SEMANTICS
    )
    kinds = {item.kind for item in summary.disagreements}
    assert summary.annotators == ("annotator-a", "annotator-b")
    assert summary.annotation_ids == ("ann-semantics", "ann-semantics-b")
    assert (summary.compared, summary.agreements) == (4, 1)
    assert kinds == {"disjoint-concepts", "status-differs", "partial-concept-overlap"}
    assert _blockers(report) == set()
    assert report.is_valid


def test_no_annotator_is_silently_chosen(tmp_path: Path) -> None:
    second = _second_semantics(semantic_record(0, "b-0", ("box",)))

    report = _qa(tmp_path, _with({"semantics-b": second}))

    record = report.to_record()
    forbidden = {"resolved", "chosen", "winner", "selected", "consensus", "majority"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    assert not keys(record) & forbidden
    disagreement = report.disagreements[0].disagreements[0]
    assert {annotation_id for annotation_id, _ in disagreement.positions} == {
        "ann-semantics",
        "ann-semantics-b",
    }


def test_agreeing_annotators_yield_a_summary_without_disagreements(tmp_path: Path) -> None:
    second = _second_semantics(*semantics_set().records)

    report = _qa(tmp_path, _with({"semantics-b": second}))

    summary = report.disagreements[0]
    assert (summary.compared, summary.agreements, summary.disagreements) == (4, 4, ())


def test_the_same_annotator_twice_is_a_duplicate_or_a_conflict_never_a_disagreement(
    tmp_path: Path,
) -> None:
    identical = (semantics_set(), "annotator-a")
    conflicting = _second_semantics(semantic_record(0, "b-0", ("box",)))
    conflicting = (conflicting[0], "annotator-a")

    duplicate_report = _qa(tmp_path / "duplicate", _with({"semantics-b": identical}))
    conflict_report = _qa(tmp_path / "conflict", _with({"semantics-b": conflicting}))

    assert "duplicate-annotation-record" in _warnings(duplicate_report)
    assert duplicate_report.disagreements == ()
    assert "conflicting-annotation-records" in _blockers(conflict_report)
    assert conflict_report.disagreements == ()


def test_relations_and_visibility_disagreements_are_summarized(tmp_path: Path) -> None:
    second_relations = replace(
        relations_set(),
        relations=(
            relation(
                "rel-1", "pallet-A", "next to", "pallet-B", RelationStatus.DOES_NOT_HOLD, ref(0)
            ),
            relation("rel-2", "pallet-A", "on top of", "pallet-B", RelationStatus.DOES_NOT_HOLD),
        ),
    )
    second_visibility = visibility_set()
    second_visibility = replace(
        second_visibility,
        annotations=(
            replace(
                second_visibility.annotations[0],
                level=VisibilityLevel.PARTIALLY_OCCLUDED,
                occluded_fraction=0.2,
            ),
        ),
    )

    report = _qa(
        tmp_path,
        _with(
            {
                "relations-b": (second_relations, "annotator-b"),
                "visibility-b": (second_visibility, "annotator-b"),
            }
        ),
    )

    by_family = {item.family: item for item in report.disagreements}
    assert (
        by_family[AnnotationFamily.RELATIONS].compared,
        by_family[AnnotationFamily.RELATIONS].agreements,
    ) == (2, 1)
    assert by_family[AnnotationFamily.RELATIONS].disagreements[0].kind == "status-differs"
    assert by_family[AnnotationFamily.VISIBILITY].disagreements[0].kind == "level-differs"
    assert report.is_valid


def test_region_annotators_are_compared_by_overlap_not_by_name(tmp_path: Path) -> None:
    second = RegionAnnotationSet(
        frames=(
            replace(
                frame_regions(0),
                regions=(RegionAnnotation(region_id="x-0", mask=mask(*A_CELLS)),),
            ),
        )
    )

    report = _qa(tmp_path, _with({"regions-b": (second, "annotator-b")}))

    summary = next(item for item in report.disagreements if item.family is AnnotationFamily.REGIONS)
    assert (summary.compared, summary.agreements) == (1, 0)
    assert summary.disagreements[0].kind == "region-sets-differ"
    detail = dict(summary.disagreements[0].positions)
    assert "b-0" in detail["ann-regions"]
    assert report.is_valid


def test_annotators_that_normalize_differently_are_not_silently_compared(tmp_path: Path) -> None:
    aliased = SemanticAnnotationSet(
        normalization=LabelNormalization(
            policy_id=CASEFOLD_ALIAS_POLICY, aliases=(LabelAlias(key="pallet", aliases=("skid",)),)
        ),
        records=(semantic_record(0, "a-0", ("skid",)),),
    )

    report = _qa(tmp_path, _with({"semantics-b": (aliased, "annotator-b")}))

    assert "normalization-differs" in _warnings(report)
    assert report.disagreements == ()


# ------------------------------------------------------- integrity and certification


def _leaking_scheme() -> SplitScheme:
    return SplitScheme(
        scheme_id="regions-by-sequence",
        task="region_discovery",
        unit=SplitUnit.SEQUENCE,
        rationale="frames of one sequence are temporally correlated",
        splits=(
            ReferenceSplit(name="tuning", role=SplitRole.TUNING, sample_ids=(sample_id(0),)),
            ReferenceSplit(name="test", role=SplitRole.TEST, sample_ids=(sample_id(1),)),
        ),
    )


def test_the_split_integrity_report_is_part_of_the_qa_report(tmp_path: Path) -> None:
    report = _qa(tmp_path, clean_sets(), split_schemes=(_leaking_scheme(),))

    assert not report.integrity.is_valid
    assert "split-leakage" in {item.code for item in report.integrity.blockers}
    assert _blockers(report) == set()
    assert not report.is_valid


def test_unreadable_files_do_not_stop_the_other_checks(tmp_path: Path) -> None:
    manifest = write_qa_reference(tmp_path, clean_sets())
    (tmp_path / "annotations" / "semantics.json").unlink()

    report = check_annotation_quality(manifest, tmp_path, policy=POLICY)

    assert "annotation-file-missing" in {item.code for item in report.integrity.blockers}
    assert report.annotation_files_checked == 6
    assert not report.is_valid


def test_only_a_reference_set_that_passes_integrity_and_qa_is_certified(tmp_path: Path) -> None:
    manifest = write_qa_reference(tmp_path, clean_sets())
    write_reference_set(tmp_path, manifest)

    certified = certify_reference_set(tmp_path, policy=POLICY)

    assert certified.validated.manifest == manifest
    assert certified.qa.is_valid
    assert certified.policy == POLICY


def test_certification_refuses_qa_blockers_and_integrity_blockers(tmp_path: Path) -> None:
    flawed = tmp_path / "flawed"
    manifest = write_qa_reference(
        flawed,
        _with(
            semantics=replace(
                semantics_set(),
                records=(*semantics_set().records, semantic_record(0, "ghost", ("pallet",))),
            )
        ),
    )
    write_reference_set(flawed, manifest)
    leaking = tmp_path / "leaking"
    leaking_manifest = write_qa_reference(leaking, clean_sets(), split_schemes=(_leaking_scheme(),))
    write_reference_set(leaking, leaking_manifest)

    with pytest.raises(AnnotationQaError, match="semantic-unknown-region") as raised:
        certify_reference_set(flawed, policy=POLICY)
    with pytest.raises(ReferenceSetIntegrityError, match="split-leakage"):
        certify_reference_set(leaking, policy=POLICY)

    assert not raised.value.report.is_valid

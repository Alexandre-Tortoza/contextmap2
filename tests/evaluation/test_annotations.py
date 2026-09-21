"""Multi-level reference annotation schema tests."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from contextmap.evaluation.annotations import (
    CASEFOLD_ALIAS_POLICY,
    CASEFOLD_EXACT_POLICY,
    AnnotationError,
    AnnotationFamily,
    AnnotationSet,
    ConceptAttribute,
    Coverage,
    DistinctIdentityPair,
    FrameRegionAnnotation,
    GeometryAnnotationSet,
    GeometryCorrespondence,
    IdentityAnnotationSet,
    IdentityOccurrence,
    IdentityScope,
    LabelAlias,
    LabelNormalization,
    ObservationRef,
    PhysicalIdentity,
    PredicateRule,
    RegionAnnotation,
    RegionAnnotationSet,
    RelationAnnotation,
    RelationAnnotationSet,
    RelationStatus,
    SceneContextAnnotation,
    SceneContextAnnotationSet,
    SemanticAnnotationRecord,
    SemanticAnnotationSet,
    SemanticStatus,
    VisibilityAnnotation,
    VisibilityAnnotationSet,
    VisibilityLevel,
    decode_annotation_set,
    encode_annotation_set,
    ground_truth_regions,
    read_annotation_set,
    write_annotation_set,
)
from contextmap.evaluation.reference_set import ReferenceSampleId
from contextmap.evaluation.semantic_interpretation import MATCHING_POLICY, SemanticAnnotation
from contextmap.ingestion import CalibrationReferenceId, SourceObservationId
from contextmap.visual_perception import BoundingBox, InlineMask

SAMPLE = ReferenceSampleId("sample-0000")
FRAME = SourceObservationId("frame-0000")
SCAN = SourceObservationId("scan-0000")


def _mask(width: int = 4, height: int = 3, *, foreground: tuple[int, ...] = (0, 1)) -> InlineMask:
    return InlineMask(
        width=width,
        height=height,
        data=tuple(index in foreground for index in range(width * height)),
    )


def _regions() -> RegionAnnotationSet:
    return RegionAnnotationSet(
        frames=(
            FrameRegionAnnotation(
                sample_id=SAMPLE,
                observation_id=FRAME,
                image_width=4,
                image_height=3,
                coverage=Coverage.COMPLETE,
                regions=(
                    RegionAnnotation(region_id="chair-1", mask=_mask()),
                    RegionAnnotation(
                        region_id="table-1",
                        box=BoundingBox(x_min=1, y_min=1, x_max=3, y_max=3),
                    ),
                ),
                valid_areas=(_mask(foreground=tuple(range(12))),),
                exclusion_areas=(_mask(foreground=(11,)),),
            ),
        )
    )


def _semantics() -> SemanticAnnotationSet:
    return SemanticAnnotationSet(
        normalization=LabelNormalization(policy_id=CASEFOLD_EXACT_POLICY),
        records=(
            SemanticAnnotationRecord(
                sample_id=SAMPLE,
                observation_id=FRAME,
                region_id="chair-1",
                status=SemanticStatus.LABELED,
                concepts=("Office Chair", "chair"),
                rejected_concepts=("stool",),
                attributes=(ConceptAttribute(name="color", value="Dark Grey"),),
            ),
            SemanticAnnotationRecord(
                sample_id=SAMPLE,
                observation_id=FRAME,
                region_id="table-1",
                status=SemanticStatus.AMBIGUOUS,
                concepts=("desk", "table"),
            ),
            SemanticAnnotationRecord(
                sample_id=SAMPLE,
                observation_id=FRAME,
                region_id=None,
                status=SemanticStatus.UNKNOWN,
                concepts=(),
            ),
        ),
    )


def _geometry() -> GeometryAnnotationSet:
    return GeometryAnnotationSet(
        correspondences=(
            GeometryCorrespondence(
                correspondence_id="corr-1",
                sample_id=SAMPLE,
                image_observation_id=FRAME,
                calibration_id=CalibrationReferenceId("cam0"),
                pixel_uv=(120.5, 88.0),
                point_frame_id="lidar",
                point_m=(2.0, -0.5, 1.25),
                point_source_observation_id=SCAN,
                pixel_tolerance_px=1.5,
                point_tolerance_m=0.02,
            ),
        )
    )


def _identity() -> IdentityAnnotationSet:
    other_frame = SourceObservationId("frame-0001")
    return IdentityAnnotationSet(
        scope=(
            IdentityScope(sample_id=SAMPLE, observation_id=FRAME, coverage=Coverage.COMPLETE),
            IdentityScope(
                sample_id=ReferenceSampleId("sample-0001"),
                observation_id=other_frame,
                coverage=Coverage.PARTIAL,
            ),
        ),
        identities=(
            PhysicalIdentity(
                identity_id="chair-A",
                occurrences=(
                    IdentityOccurrence(sample_id=SAMPLE, observation_id=FRAME, region_id="chair-1"),
                    IdentityOccurrence(
                        sample_id=ReferenceSampleId("sample-0001"),
                        observation_id=other_frame,
                        region_id="chair-9",
                    ),
                ),
            ),
            PhysicalIdentity(
                identity_id="chair-B",
                occurrences=(
                    IdentityOccurrence(sample_id=SAMPLE, observation_id=FRAME, region_id="chair-2"),
                ),
            ),
        ),
        distinct_pairs=(DistinctIdentityPair(first="chair-A", second="chair-B"),),
    )


def _relations() -> RelationAnnotationSet:
    return RelationAnnotationSet(
        normalization=LabelNormalization(policy_id=CASEFOLD_EXACT_POLICY),
        predicate_rules=(
            PredicateRule(predicate="next to", symmetric=True),
            PredicateRule(predicate="on top of", inverse="supports"),
            PredicateRule(predicate="supports", inverse="on top of"),
        ),
        relations=(
            RelationAnnotation(
                relation_id="rel-1",
                subject_identity_id="chair-A",
                predicate="next to",
                object_identity_id="chair-B",
                status=RelationStatus.HOLDS,
                anchors=(ObservationRef(sample_id=SAMPLE, observation_id=FRAME),),
            ),
            RelationAnnotation(
                relation_id="rel-2",
                subject_identity_id="chair-A",
                predicate="on top of",
                object_identity_id="chair-B",
                status=RelationStatus.DOES_NOT_HOLD,
                anchors=(ObservationRef(sample_id=SAMPLE, observation_id=FRAME),),
            ),
        ),
    )


def _visibility() -> VisibilityAnnotationSet:
    return VisibilityAnnotationSet(
        annotations=(
            VisibilityAnnotation(
                sample_id=SAMPLE,
                observation_id=FRAME,
                region_id="chair-1",
                level=VisibilityLevel.PARTIALLY_OCCLUDED,
                occluded_fraction=0.35,
            ),
            VisibilityAnnotation(
                sample_id=SAMPLE,
                observation_id=FRAME,
                identity_id="chair-B",
                level=VisibilityLevel.UNKNOWN,
            ),
            VisibilityAnnotation(
                sample_id=SAMPLE,
                observation_id=FRAME,
                level=VisibilityLevel.FULLY_VISIBLE,
            ),
        )
    )


def _scene() -> SceneContextAnnotationSet:
    return SceneContextAnnotationSet(
        normalization=LabelNormalization(policy_id=CASEFOLD_EXACT_POLICY),
        annotations=(
            SceneContextAnnotation(
                sample_id=SAMPLE,
                observation_id=None,
                attributes=(
                    ConceptAttribute(name="environment", value="Indoor office"),
                    ConceptAttribute(name="lighting", value="dim"),
                ),
            ),
        ),
    )


ALL_SETS: list[Callable[[], AnnotationSet]] = [
    _regions,
    _semantics,
    _geometry,
    _identity,
    _relations,
    _visibility,
    _scene,
]


@pytest.mark.parametrize("build", ALL_SETS)
def test_every_family_round_trips_through_json(build: Callable[[], AnnotationSet]) -> None:
    annotation_set = build()

    document = json.loads(json.dumps(encode_annotation_set(annotation_set)))

    assert decode_annotation_set(document) == annotation_set
    assert document["schema"] == annotation_set.family.schema


def test_schema_identifiers_are_versioned_per_family() -> None:
    assert AnnotationFamily.REGIONS.schema == "contextmap.reference.regions/v1"
    assert {family.schema for family in AnnotationFamily} == {
        f"contextmap.reference.{name}/v1"
        for name in (
            "regions",
            "semantics",
            "geometry",
            "identity",
            "relations",
            "visibility",
            "scene_context",
        )
    }
    for family in AnnotationFamily:
        assert AnnotationFamily.from_schema(family.schema) is family
    with pytest.raises(AnnotationError, match="schema"):
        AnnotationFamily.from_schema("contextmap.reference.regions/v2")
    document = encode_annotation_set(_regions())
    document["schema"] = "contextmap.reference.regions/v2"
    with pytest.raises(AnnotationError, match="schema"):
        decode_annotation_set(document)


def test_decoding_reports_missing_fields_as_annotation_errors() -> None:
    document = encode_annotation_set(_semantics())
    del document["records"][0]["status"]  # type: ignore[index]

    with pytest.raises(AnnotationError, match="status"):
        decode_annotation_set(document)


def test_normalization_is_explicit_and_never_implies_synonymy() -> None:
    exact = LabelNormalization(policy_id=CASEFOLD_EXACT_POLICY)

    assert exact.key("  Trash   CAN ") == "trash can"
    assert exact.key("bin") != exact.key("trash can")
    assert CASEFOLD_EXACT_POLICY == MATCHING_POLICY

    aliased = LabelNormalization(
        policy_id=CASEFOLD_ALIAS_POLICY,
        aliases=(LabelAlias(key="trash can", aliases=("Bin", "garbage can")),),
    )
    assert aliased.key("BIN") == aliased.key("Trash Can") == "trash can"
    assert aliased.key("basket") == "basket"


def test_normalization_rejects_unknown_policies_and_ambiguous_aliases() -> None:
    with pytest.raises(ValueError, match="policy"):
        LabelNormalization(policy_id="stemming/9")
    with pytest.raises(ValueError, match="alias"):
        LabelNormalization(
            policy_id=CASEFOLD_EXACT_POLICY, aliases=(LabelAlias(key="a", aliases=("b",)),)
        )
    with pytest.raises(ValueError, match="alias"):
        LabelNormalization(
            policy_id=CASEFOLD_ALIAS_POLICY,
            aliases=(
                LabelAlias(key="trash can", aliases=("bin",)),
                LabelAlias(key="recycling bin", aliases=("Bin",)),
            ),
        )


def test_literal_annotator_labels_survive_round_trip() -> None:
    restored = decode_annotation_set(encode_annotation_set(_semantics()))

    assert isinstance(restored, SemanticAnnotationSet)
    assert restored.records[0].concepts == ("Office Chair", "chair")
    assert restored.records[0].attributes[0].value == "Dark Grey"


def test_regions_need_geometry_and_unique_identities() -> None:
    with pytest.raises(ValueError, match="geometry"):
        RegionAnnotation(region_id="empty")
    frame = _regions().frames[0]
    with pytest.raises(ValueError, match="region"):
        replace(frame, regions=(frame.regions[0], frame.regions[0]))
    with pytest.raises(ValueError, match="frame"):
        RegionAnnotationSet(frames=(frame, frame))


def test_partial_and_complete_region_coverage_stay_distinct() -> None:
    frame = _regions().frames[0]
    partial = RegionAnnotationSet(frames=(replace(frame, coverage=Coverage.PARTIAL),))

    restored = decode_annotation_set(encode_annotation_set(partial))

    assert isinstance(restored, RegionAnnotationSet)
    assert restored.frames[0].coverage is Coverage.PARTIAL
    assert restored != _regions()


def test_region_annotations_feed_the_region_discovery_evaluator() -> None:
    truth = ground_truth_regions(_regions().frames[0])

    assert [region.region_id for region in truth] == ["chair-1"]
    assert truth[0].mask == _mask()


def test_semantic_status_defines_how_many_concepts_a_record_needs() -> None:
    labeled = _semantics().records[0]
    with pytest.raises(ValueError, match="labeled"):
        replace(labeled, concepts=())
    with pytest.raises(ValueError, match="ambiguous"):
        replace(_semantics().records[1], concepts=("desk",))
    with pytest.raises(ValueError, match="unknown"):
        replace(_semantics().records[2], concepts=("desk",))


def test_semantic_concepts_are_checked_under_the_declared_normalization() -> None:
    exact = LabelNormalization(policy_id=CASEFOLD_EXACT_POLICY)
    records = _semantics().records
    duplicate = replace(records[0], concepts=("Chair", " chair "))
    contradictory = replace(records[0], rejected_concepts=("CHAIR",))

    with pytest.raises(ValueError, match="unique"):
        SemanticAnnotationSet(normalization=exact, records=(duplicate,))
    with pytest.raises(ValueError, match="rejected"):
        SemanticAnnotationSet(normalization=exact, records=(contradictory,))
    aliased = LabelNormalization(
        policy_id=CASEFOLD_ALIAS_POLICY,
        aliases=(LabelAlias(key="chair", aliases=("office chair",)),),
    )
    with pytest.raises(ValueError, match="unique"):
        SemanticAnnotationSet(normalization=aliased, records=(records[0],))


def test_semantic_targets_are_unique_and_missing_differs_from_unknown() -> None:
    annotations = _semantics()
    with pytest.raises(ValueError, match="target"):
        SemanticAnnotationSet(
            normalization=annotations.normalization,
            records=(annotations.records[0], annotations.records[0]),
        )

    unknown = annotations.find(SAMPLE, FRAME, None)
    absent = annotations.find(SAMPLE, FRAME, "no-such-region")

    assert unknown is not None and unknown.status is SemanticStatus.UNKNOWN
    assert absent is None


def test_semantic_records_feed_the_semantic_interpretation_evaluator() -> None:
    annotations = _semantics()

    labeled = annotations.to_semantic_annotation(annotations.records[0])
    ambiguous = annotations.to_semantic_annotation(annotations.records[1])

    assert labeled == SemanticAnnotation(
        acceptable_hypotheses=("Office Chair", "chair"), ambiguity_expected=False
    )
    assert ambiguous.ambiguity_expected is True
    with pytest.raises(AnnotationError, match="unknown"):
        annotations.to_semantic_annotation(annotations.records[2])


def test_alias_policies_are_expanded_for_the_exact_matching_evaluator() -> None:
    annotations = SemanticAnnotationSet(
        normalization=LabelNormalization(
            policy_id=CASEFOLD_ALIAS_POLICY,
            aliases=(LabelAlias(key="trash can", aliases=("Bin", "garbage can")),),
        ),
        records=(
            SemanticAnnotationRecord(
                sample_id=SAMPLE,
                observation_id=FRAME,
                region_id="bin-1",
                status=SemanticStatus.LABELED,
                concepts=("Bin",),
            ),
        ),
    )

    expanded = annotations.to_semantic_annotation(annotations.records[0])

    assert expanded.acceptable_hypotheses == ("Bin", "trash can", "garbage can")


def test_geometry_correspondences_state_units_frames_and_calibration() -> None:
    correspondence = _geometry().correspondences[0]

    with pytest.raises(ValueError, match="finite"):
        replace(correspondence, point_m=(float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError, match="tolerance"):
        replace(correspondence, pixel_tolerance_px=-1.0)
    with pytest.raises(ValueError, match="frame"):
        replace(correspondence, point_frame_id=" ")
    with pytest.raises(ValueError, match="correspondence"):
        GeometryAnnotationSet(correspondences=(correspondence, correspondence))


def test_identity_is_explicit_and_never_inferred_from_labels() -> None:
    annotations = _identity()
    twin = replace(
        annotations.identities[1],
        identity_id="chair-C",
        occurrences=(
            IdentityOccurrence(sample_id=SAMPLE, observation_id=FRAME, region_id="chair-3"),
        ),
        description="an office chair",
    )

    kept_apart = replace(
        annotations,
        identities=(
            replace(annotations.identities[0], description="an office chair"),
            twin,
            annotations.identities[1],
        ),
        distinct_pairs=(),
    )

    assert [item.identity_id for item in kept_apart.identities] == ["chair-A", "chair-C", "chair-B"]
    with pytest.raises(ValueError, match="identity"):
        replace(
            annotations,
            identities=(annotations.identities[0], annotations.identities[0]),
        )


def test_identity_distinct_pairs_reference_declared_identities() -> None:
    annotations = _identity()

    with pytest.raises(ValueError, match="unknown identity"):
        replace(annotations, distinct_pairs=(DistinctIdentityPair(first="chair-A", second="x"),))
    with pytest.raises(ValueError, match="itself"):
        DistinctIdentityPair(first="chair-A", second="chair-A")
    with pytest.raises(ValueError, match="occurrence"):
        replace(
            annotations.identities[0],
            occurrences=(annotations.identities[0].occurrences[0],) * 2,
        )
    with pytest.raises(ValueError, match="observation"):
        PhysicalIdentity(identity_id="ghost", occurrences=())


def test_identity_scope_marks_which_observations_are_annotated_exhaustively() -> None:
    annotations = _identity()

    assert annotations.coverage_of(SAMPLE, FRAME) is Coverage.COMPLETE
    assert annotations.coverage_of(SAMPLE, SourceObservationId("frame-0001")) is None
    assert (
        annotations.coverage_of(ReferenceSampleId("sample-0001"), SourceObservationId("frame-0001"))
        is Coverage.PARTIAL
    )


def test_relations_declare_status_predicates_and_physical_anchors() -> None:
    relations = _relations()
    relation = relations.relations[0]

    with pytest.raises(ValueError, match="anchor"):
        replace(relation, anchors=())
    with pytest.raises(ValueError, match="itself"):
        replace(relation, object_identity_id="chair-A")
    with pytest.raises(ValueError, match="relation"):
        replace(relations, relations=(relation, relation))
    unknown = replace(relation, relation_id="rel-3", status=RelationStatus.UNKNOWN)
    assert replace(relations, relations=(relation, unknown)).relations[1].status is (
        RelationStatus.UNKNOWN
    )


def test_predicate_rules_are_coherent() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        PredicateRule(predicate="next to", symmetric=True, inverse="beside")
    with pytest.raises(ValueError, match="inverse"):
        PredicateRule(predicate="on top of", inverse="on top of")
    relations = _relations()
    with pytest.raises(ValueError, match="predicate"):
        replace(relations, predicate_rules=(relations.predicate_rules[0],) * 2)
    with pytest.raises(ValueError, match="inverse"):
        replace(relations, predicate_rules=relations.predicate_rules[:2])
    with pytest.raises(ValueError, match="inverse"):
        replace(
            relations,
            predicate_rules=(
                relations.predicate_rules[0],
                relations.predicate_rules[1],
                PredicateRule(predicate="supports", inverse="beneath"),
                PredicateRule(predicate="beneath", inverse="supports"),
            ),
        )
    with pytest.raises(ValueError, match="undeclared predicate"):
        replace(
            relations,
            relations=(replace(relations.relations[0], predicate="behind"),),
        )


def test_visibility_levels_distinguish_unknown_from_not_annotated() -> None:
    annotations = _visibility()

    assert annotations.annotations[1].level is VisibilityLevel.UNKNOWN
    with pytest.raises(ValueError, match="fraction"):
        replace(annotations.annotations[0], occluded_fraction=1.5)
    with pytest.raises(ValueError, match="one target"):
        replace(annotations.annotations[0], identity_id="chair-A")
    with pytest.raises(ValueError, match="target"):
        VisibilityAnnotationSet(annotations=(annotations.annotations[0],) * 2)


def test_scene_context_attributes_are_unique_per_annotation() -> None:
    annotation = _scene().annotations[0]

    with pytest.raises(ValueError, match="attribute"):
        replace(
            annotation,
            attributes=(
                ConceptAttribute(name="lighting", value="dim"),
                ConceptAttribute(name="lighting", value="bright"),
            ),
        )
    with pytest.raises(ValueError, match="attribute"):
        replace(annotation, attributes=())


@pytest.mark.parametrize("build", ALL_SETS)
def test_annotations_are_linkable_to_physical_observations(
    build: Callable[[], AnnotationSet],
) -> None:
    references = build().observation_references()

    assert references
    assert all(isinstance(item, ObservationRef) for item in references)
    assert all(item.sample_id == SAMPLE or item.sample_id == "sample-0001" for item in references)
    assert len(set(references)) == len(references)


def test_written_files_are_immutable_and_hash_matches_the_manifest_entry(tmp_path: Path) -> None:
    path = tmp_path / "annotations" / "regions.json"
    annotation_set = _regions()

    content_hash = write_annotation_set(path, annotation_set)

    assert content_hash == f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    assert read_annotation_set(path) == annotation_set
    with pytest.raises(FileExistsError):
        write_annotation_set(path, annotation_set)


def test_reader_rejects_documents_that_are_not_annotation_sets(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(AnnotationError, match="JSON"):
        read_annotation_set(path)

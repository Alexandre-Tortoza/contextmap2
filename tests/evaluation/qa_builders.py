"""Builders for annotation-QA tests: a clean multi-family reference set written to disk."""

from collections.abc import Mapping
from pathlib import Path

from reference_set_builders import make_valid_manifest, sample_id

from contextmap.evaluation.annotation_qa import AnnotationQaPolicy
from contextmap.evaluation.annotations import (
    CASEFOLD_EXACT_POLICY,
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
    write_annotation_set,
)
from contextmap.evaluation.reference_set import (
    AnnotationFileEntry,
    AnnotationProvenance,
    ProvenanceOrigin,
    ReferenceSampleId,
    ReferenceSetManifest,
    ReferenceTrust,
)
from contextmap.ingestion import CalibrationReferenceId, SourceObservationId
from contextmap.visual_perception import InlineMask

WIDTH = 4
HEIGHT = 3
A_CELLS = (0, 1)
B_CELLS = (10, 11)
EXACT = LabelNormalization(policy_id=CASEFOLD_EXACT_POLICY)

POLICY = AnnotationQaPolicy(
    policy_id="annotation-qa/1",
    duplicate_pixel_tolerance_px=0.5,
    duplicate_point_tolerance_m=0.05,
    region_match_iou=0.5,
)


def mask(*cells: int, width: int = WIDTH, height: int = HEIGHT) -> InlineMask:
    return InlineMask(
        width=width, height=height, data=tuple(index in cells for index in range(width * height))
    )


def frame(index: int) -> SourceObservationId:
    return SourceObservationId(f"frame-{index:04d}")


def scan(index: int) -> SourceObservationId:
    return SourceObservationId(f"scan-{index:04d}")


def ref(index: int) -> ObservationRef:
    return ObservationRef(sample_id=sample_id(index), observation_id=frame(index))


def frame_regions(index: int) -> FrameRegionAnnotation:
    return FrameRegionAnnotation(
        sample_id=sample_id(index),
        observation_id=frame(index),
        image_width=WIDTH,
        image_height=HEIGHT,
        coverage=Coverage.COMPLETE,
        regions=(
            RegionAnnotation(region_id=f"a-{index}", mask=mask(*A_CELLS)),
            RegionAnnotation(region_id=f"b-{index}", mask=mask(*B_CELLS)),
        ),
    )


def regions_set() -> RegionAnnotationSet:
    return RegionAnnotationSet(frames=(frame_regions(0), frame_regions(1)))


def semantic_record(
    index: int,
    region: str,
    concepts: tuple[str, ...],
    status: SemanticStatus = SemanticStatus.LABELED,
) -> SemanticAnnotationRecord:
    return SemanticAnnotationRecord(
        sample_id=sample_id(index),
        observation_id=frame(index),
        region_id=region,
        status=status,
        concepts=concepts,
    )


def semantics_set() -> SemanticAnnotationSet:
    return SemanticAnnotationSet(
        normalization=EXACT,
        records=(
            semantic_record(0, "a-0", ("pallet",)),
            semantic_record(0, "b-0", ("crate",)),
            semantic_record(1, "a-1", ("pallet",)),
            semantic_record(1, "b-1", ("crate",)),
        ),
    )


def correspondence(
    index: int,
    *,
    name: str | None = None,
    calibration: str = "cam0",
    pixel: tuple[float, float] = (1.0, 1.0),
    point: tuple[float, float, float] | None = None,
    point_frame: str = "velodyne",
) -> GeometryCorrespondence:
    return GeometryCorrespondence(
        correspondence_id=name or f"corr-{index}",
        sample_id=sample_id(index),
        image_observation_id=frame(index),
        calibration_id=CalibrationReferenceId(calibration),
        pixel_uv=pixel,
        point_frame_id=point_frame,
        point_m=point or (2.0 + index, 0.0, 1.0),
        point_source_observation_id=scan(index),
        pixel_tolerance_px=0.5,
        point_tolerance_m=0.02,
    )


def geometry_set() -> GeometryAnnotationSet:
    return GeometryAnnotationSet(correspondences=(correspondence(0), correspondence(1)))


def occurrence(index: int, region: str | None) -> IdentityOccurrence:
    return IdentityOccurrence(
        sample_id=sample_id(index), observation_id=frame(index), region_id=region
    )


def identity_set() -> IdentityAnnotationSet:
    return IdentityAnnotationSet(
        scope=tuple(
            IdentityScope(
                sample_id=sample_id(index), observation_id=frame(index), coverage=Coverage.COMPLETE
            )
            for index in (0, 1)
        ),
        identities=(
            PhysicalIdentity(
                identity_id="pallet-A", occurrences=(occurrence(0, "a-0"), occurrence(1, "a-1"))
            ),
            PhysicalIdentity(
                identity_id="pallet-B", occurrences=(occurrence(0, "b-0"), occurrence(1, "b-1"))
            ),
        ),
        distinct_pairs=(DistinctIdentityPair(first="pallet-A", second="pallet-B"),),
    )


def relation(
    name: str,
    subject: str,
    predicate: str,
    obj: str,
    status: RelationStatus,
    *anchors: ObservationRef,
) -> RelationAnnotation:
    return RelationAnnotation(
        relation_id=name,
        subject_identity_id=subject,
        predicate=predicate,
        object_identity_id=obj,
        status=status,
        anchors=anchors or (ref(0),),
    )


def relations_set(*extra: RelationAnnotation) -> RelationAnnotationSet:
    return RelationAnnotationSet(
        normalization=EXACT,
        predicate_rules=(
            PredicateRule(predicate="next to", symmetric=True),
            PredicateRule(predicate="on top of", inverse="supports"),
            PredicateRule(predicate="supports", inverse="on top of"),
        ),
        relations=(
            relation(
                "rel-1", "pallet-A", "next to", "pallet-B", RelationStatus.HOLDS, ref(0), ref(1)
            ),
            relation("rel-2", "pallet-A", "on top of", "pallet-B", RelationStatus.DOES_NOT_HOLD),
            relation("rel-3", "pallet-B", "supports", "pallet-A", RelationStatus.DOES_NOT_HOLD),
            *extra,
        ),
    )


def visibility_set(*extra: VisibilityAnnotation) -> VisibilityAnnotationSet:
    return VisibilityAnnotationSet(
        annotations=(
            VisibilityAnnotation(
                sample_id=sample_id(0),
                observation_id=frame(0),
                region_id="a-0",
                level=VisibilityLevel.FULLY_VISIBLE,
                occluded_fraction=0.0,
            ),
            VisibilityAnnotation(
                sample_id=sample_id(0),
                observation_id=frame(0),
                identity_id="pallet-B",
                level=VisibilityLevel.PARTIALLY_OCCLUDED,
                occluded_fraction=0.3,
            ),
            *extra,
        )
    )


def scene_set() -> SceneContextAnnotationSet:
    return SceneContextAnnotationSet(
        normalization=EXACT,
        annotations=(
            SceneContextAnnotation(
                sample_id=sample_id(0),
                observation_id=None,
                attributes=(ConceptAttribute(name="lighting", value="uniform"),),
            ),
        ),
    )


def clean_sets() -> dict[str, AnnotationSet]:
    """Return one consistent annotation set per family."""
    return {
        "regions": regions_set(),
        "semantics": semantics_set(),
        "geometry": geometry_set(),
        "identity": identity_set(),
        "relations": relations_set(),
        "visibility": visibility_set(),
        "scene_context": scene_set(),
    }


def write_qa_reference(
    root: Path,
    sets: Mapping[str, AnnotationSet | tuple[AnnotationSet, str]],
    *,
    default_annotator: str = "annotator-a",
    **overrides: object,
) -> ReferenceSetManifest:
    """Write every annotation file under ``root`` and return the matching manifest.

    A value may be ``(annotation_set, annotator)`` to record another annotator;
    the file stem is the mapping key, so several files of one family can coexist.
    """
    entries: list[AnnotationFileEntry] = []
    provenance: dict[str, AnnotationProvenance] = {}
    for stem, value in sets.items():
        annotation_set, annotator = (
            value if isinstance(value, tuple) else (value, default_annotator)
        )
        digest = write_annotation_set(root / "annotations" / f"{stem}.json", annotation_set)
        provenance_id = f"prov-{annotator}"
        provenance.setdefault(
            provenance_id,
            AnnotationProvenance(
                provenance_id=provenance_id,
                origin=ProvenanceOrigin.MANUAL_ANNOTATION,
                annotator=annotator,
                method="manual annotation, protocol v2",
            ),
        )
        samples = tuple(
            dict.fromkeys(
                ReferenceSampleId(item.sample_id)
                for item in annotation_set.observation_references()
            )
        )
        entries.append(
            AnnotationFileEntry(
                annotation_id=f"ann-{stem}",
                schema=annotation_set.family.schema,
                path=f"annotations/{stem}.json",
                content_hash=digest,
                trust=ReferenceTrust.TRUSTED_GROUND_TRUTH,
                provenance_id=provenance_id,
                sample_ids=samples,
            )
        )
    return make_valid_manifest(
        annotations=tuple(entries), provenance=tuple(provenance.values()), **overrides
    )

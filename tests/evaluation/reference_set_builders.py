"""Builders for reference-set manifests and their on-disk annotation files in tests."""

import hashlib
from dataclasses import replace
from pathlib import Path

from contextmap.evaluation.annotations import (
    Coverage,
    FrameRegionAnnotation,
    RegionAnnotation,
    RegionAnnotationSet,
    write_annotation_set,
)
from contextmap.evaluation.reference_set import (
    AnnotationFileEntry,
    AnnotationProvenance,
    CalibrationIdentity,
    ProvenanceOrigin,
    ProvenanceReview,
    ReferenceSample,
    ReferenceSampleId,
    ReferenceSetManifest,
    ReferenceSource,
    ReferenceSplit,
    ReferenceTrust,
    SampleGroup,
    SampleStratum,
    SampleTimeSpan,
    SourceKind,
    SplitRole,
    SplitScheme,
    SplitUnit,
    StratumDefinition,
)
from contextmap.ingestion import CalibrationReferenceId, SourceObservationId
from contextmap.shared import SourceTimestamp
from contextmap.visual_perception import InlineMask

REGIONS_SCHEMA = "contextmap.reference.regions/v1"


def content_hash(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def sample_id(index: int) -> ReferenceSampleId:
    return ReferenceSampleId(f"sample-{index:04d}")


def make_source(source_id: str = "seq-a", *, redistributable: bool = True) -> ReferenceSource:
    return ReferenceSource(
        source_id=source_id,
        kind=SourceKind.SEQUENCE_ARTIFACT,
        identity=f"sequence-artifact-{source_id}",
        content_hash=content_hash(source_id),
        license="CC-BY-4.0",
        redistributable=redistributable,
    )


def make_calibration(calibration_id: str = "cam0", source_id: str = "seq-a") -> CalibrationIdentity:
    return CalibrationIdentity(
        calibration_id=CalibrationReferenceId(calibration_id),
        source_id=source_id,
        content_hash=content_hash(calibration_id),
    )


def make_sample(
    index: int,
    *,
    source_id: str = "seq-a",
    calibration_id: str = "cam0",
    second: float | None = None,
    scene: str = "scene-1",
    with_time_span: bool = True,
) -> ReferenceSample:
    at = float(index if second is None else second)
    whole = int(at)
    nanos = round((at - whole) * 1_000_000_000)
    clock = f"dataset:{source_id}"
    span = SampleTimeSpan(
        start=SourceTimestamp(seconds=whole, nanoseconds=nanos, clock_id=clock),
        end=SourceTimestamp(seconds=whole, nanoseconds=nanos + 100_000_000, clock_id=clock),
    )
    return ReferenceSample(
        sample_id=sample_id(index),
        source_id=source_id,
        observation_ids=(
            SourceObservationId(f"frame-{index:04d}"),
            SourceObservationId(f"scan-{index:04d}"),
        ),
        calibration_ids=(CalibrationReferenceId(calibration_id),),
        time_span=span if with_time_span else None,
        content_hash=content_hash(f"sample-{index}"),
        strata=(SampleStratum(name="visibility", value="clear"),),
        groups=(SampleGroup(unit=SplitUnit.SCENE, key=scene),),
    )


def make_provenance(
    origin: ProvenanceOrigin = ProvenanceOrigin.MANUAL_ANNOTATION,
    *,
    provenance_id: str = "prov-manual",
    seeded_from_artifacts: tuple[str, ...] = (),
    reviewed: bool = False,
) -> AnnotationProvenance:
    return AnnotationProvenance(
        provenance_id=provenance_id,
        origin=origin,
        annotator="annotator-team-a",
        method="manual mask painting, protocol v2",
        tool_version="labeler 1.4",
        seeded_from_artifacts=seeded_from_artifacts,
        review=ProvenanceReview(reviewer="reviewer-b", method="full pass") if reviewed else None,
    )


def make_annotation(
    *,
    trust: ReferenceTrust = ReferenceTrust.TRUSTED_GROUND_TRUTH,
    annotation_id: str = "ann-regions",
    path: str = "annotations/regions.json",
    file_hash: str | None = None,
    provenance_id: str = "prov-manual",
    schema: str = REGIONS_SCHEMA,
    samples: tuple[int, ...] = (0, 1),
) -> AnnotationFileEntry:
    return AnnotationFileEntry(
        annotation_id=annotation_id,
        schema=schema,
        path=path,
        content_hash=file_hash or content_hash(annotation_id),
        trust=trust,
        provenance_id=provenance_id,
        sample_ids=tuple(sample_id(index) for index in samples),
    )


def make_scheme() -> SplitScheme:
    """Return a structurally valid scheme that leaks (one sequence in both splits)."""
    return SplitScheme(
        scheme_id="regions-by-sequence",
        task="region_discovery",
        unit=SplitUnit.SEQUENCE,
        rationale="frames of one sequence are temporally correlated",
        splits=(
            ReferenceSplit(name="tuning", role=SplitRole.TUNING, sample_ids=(sample_id(0),)),
            ReferenceSplit(
                name="test", role=SplitRole.TEST, sample_ids=(sample_id(2), sample_id(1))
            ),
        ),
    )


def _strata() -> tuple[StratumDefinition, ...]:
    return (
        StratumDefinition(
            name="visibility",
            values=("clear", "occluded"),
            description="occlusion level of the annotated subject",
        ),
    )


def make_manifest(**overrides: object) -> ReferenceSetManifest:
    """Return a structurally valid manifest; it is *not* free of integrity problems."""
    fields: dict[str, object] = {
        "reference_set_id": "ref-set-alpha",
        "version": "1.0.0",
        "sources": (make_source(),),
        "calibrations": (make_calibration(),),
        "stratum_definitions": _strata(),
        "samples": (make_sample(0), make_sample(1), make_sample(2)),
        "provenance": (make_provenance(),),
        "annotations": (make_annotation(),),
        "split_schemes": (make_scheme(),),
    }
    fields.update(overrides)
    return ReferenceSetManifest(**fields)  # type: ignore[arg-type]


def make_valid_manifest(**overrides: object) -> ReferenceSetManifest:
    """Return a manifest without any integrity finding.

    Sequence ``seq-a`` (samples 0, 1) is tuned on and sequence ``seq-b``
    (samples 2, 3) is the test split, so no sequence straddles the split.
    """
    scheme = SplitScheme(
        scheme_id="regions-by-sequence",
        task="region_discovery",
        unit=SplitUnit.SEQUENCE,
        rationale="frames of one sequence are temporally correlated, so a sequence is one unit",
        splits=(
            ReferenceSplit(
                name="tuning", role=SplitRole.TUNING, sample_ids=(sample_id(0), sample_id(1))
            ),
            ReferenceSplit(
                name="test", role=SplitRole.TEST, sample_ids=(sample_id(2), sample_id(3))
            ),
        ),
    )
    fields: dict[str, object] = {
        "sources": (make_source("seq-a"), make_source("seq-b")),
        "calibrations": (make_calibration("cam0", "seq-a"), make_calibration("cam1", "seq-b")),
        "samples": (
            make_sample(0, source_id="seq-a", scene="scene-a"),
            make_sample(1, source_id="seq-a", scene="scene-a", second=30),
            make_sample(2, source_id="seq-b", calibration_id="cam1", scene="scene-b"),
            make_sample(3, source_id="seq-b", calibration_id="cam1", scene="scene-b", second=30),
        ),
        "annotations": (make_annotation(samples=(0, 1, 2, 3)),),
        "split_schemes": (scheme,),
    }
    fields.update(overrides)
    return make_manifest(**fields)


def regions_for(*indices: int) -> RegionAnnotationSet:
    """Return a region annotation set with one masked region per listed sample."""
    return RegionAnnotationSet(
        frames=tuple(
            FrameRegionAnnotation(
                sample_id=sample_id(index),
                observation_id=SourceObservationId(f"frame-{index:04d}"),
                image_width=4,
                image_height=3,
                coverage=Coverage.COMPLETE,
                regions=(
                    RegionAnnotation(
                        region_id=f"object-{index}",
                        mask=InlineMask(
                            width=4, height=3, data=tuple(cell in (0, 1) for cell in range(12))
                        ),
                    ),
                ),
            )
            for index in indices
        )
    )


def write_valid_reference_set(root: Path, **overrides: object) -> ReferenceSetManifest:
    """Write the regions annotation file under ``root`` and return a matching valid manifest."""
    file_hash = write_annotation_set(root / "annotations" / "regions.json", regions_for(0, 1, 2, 3))
    annotation = replace(make_annotation(samples=(0, 1, 2, 3)), content_hash=file_hash)
    return make_valid_manifest(annotations=(annotation,), **overrides)

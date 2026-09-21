"""Reference-set manifest contract tests."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from contextmap.evaluation.reference_set import (
    REFERENCE_SET_SCHEMA,
    AnnotationFileEntry,
    AnnotationProvenance,
    CalibrationIdentity,
    ProvenanceOrigin,
    ProvenanceReview,
    ReferenceSample,
    ReferenceSampleId,
    ReferenceSetError,
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
    decode_reference_set,
    encode_reference_set,
    read_reference_set,
    require_version_bump_on_change,
    verify_annotation_files,
    write_reference_set,
)
from contextmap.ingestion import CalibrationReferenceId, SourceObservationId
from contextmap.shared import SourceTimestamp


def _hash(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def _source() -> ReferenceSource:
    return ReferenceSource(
        source_id="seq-a",
        kind=SourceKind.SEQUENCE_ARTIFACT,
        identity="sequence-artifact-0001",
        content_hash=_hash("seq-a"),
        license="CC-BY-4.0",
        redistributable=True,
    )


def _calibration() -> CalibrationIdentity:
    return CalibrationIdentity(
        calibration_id=CalibrationReferenceId("cam0"),
        source_id="seq-a",
        content_hash=_hash("cam0"),
    )


def _sample(index: int) -> ReferenceSample:
    start = SourceTimestamp(seconds=index, nanoseconds=0, clock_id="dataset:seq-a")
    end = SourceTimestamp(seconds=index, nanoseconds=100_000_000, clock_id="dataset:seq-a")
    return ReferenceSample(
        sample_id=ReferenceSampleId(f"sample-{index:04d}"),
        source_id="seq-a",
        observation_ids=(
            SourceObservationId(f"frame-{index:04d}"),
            SourceObservationId(f"scan-{index:04d}"),
        ),
        calibration_ids=(CalibrationReferenceId("cam0"),),
        time_span=SampleTimeSpan(start=start, end=end),
        content_hash=_hash(f"sample-{index}"),
        strata=(SampleStratum(name="visibility", value="clear"),),
        groups=(SampleGroup(unit=SplitUnit.SCENE, key="scene-1"),),
    )


def _provenance(
    origin: ProvenanceOrigin = ProvenanceOrigin.MANUAL_ANNOTATION,
) -> AnnotationProvenance:
    return AnnotationProvenance(
        provenance_id="prov-manual",
        origin=origin,
        annotator="annotator-team-a",
        method="manual mask painting, protocol v2",
        tool_version="labeler 1.4",
    )


def _annotation(
    *, trust: ReferenceTrust = ReferenceTrust.TRUSTED_GROUND_TRUTH
) -> AnnotationFileEntry:
    return AnnotationFileEntry(
        annotation_id="ann-regions",
        schema="contextmap.reference.regions/v1",
        path="annotations/regions.json",
        content_hash=_hash("regions"),
        trust=trust,
        provenance_id="prov-manual",
        sample_ids=(ReferenceSampleId("sample-0000"), ReferenceSampleId("sample-0001")),
    )


def _scheme() -> SplitScheme:
    return SplitScheme(
        scheme_id="regions-by-sequence",
        task="region_discovery",
        unit=SplitUnit.SEQUENCE,
        rationale="frames of one sequence are temporally correlated",
        splits=(
            ReferenceSplit(
                name="tuning",
                role=SplitRole.TUNING,
                sample_ids=(ReferenceSampleId("sample-0000"),),
            ),
            ReferenceSplit(
                name="test",
                role=SplitRole.TEST,
                sample_ids=(ReferenceSampleId("sample-0002"), ReferenceSampleId("sample-0001")),
            ),
        ),
    )


def _manifest(**overrides: object) -> ReferenceSetManifest:
    fields: dict[str, object] = {
        "reference_set_id": "ref-set-alpha",
        "version": "1.0.0",
        "sources": (_source(),),
        "calibrations": (_calibration(),),
        "stratum_definitions": (
            StratumDefinition(
                name="visibility",
                values=("clear", "occluded"),
                description="occlusion level of the annotated subject",
            ),
        ),
        "samples": (_sample(0), _sample(1), _sample(2)),
        "provenance": (_provenance(),),
        "annotations": (_annotation(),),
        "split_schemes": (_scheme(),),
    }
    fields.update(overrides)
    return ReferenceSetManifest(**fields)  # type: ignore[arg-type]


def test_manifest_round_trips_with_a_stable_digest() -> None:
    manifest = _manifest()

    document = encode_reference_set(manifest)
    restored = decode_reference_set(json.loads(json.dumps(document)))

    assert document["schema"] == REFERENCE_SET_SCHEMA
    assert restored == manifest
    assert restored.digest() == manifest.digest() == document["digest"]
    assert manifest.digest().startswith("sha256:")
    assert manifest.identity().digest == manifest.digest()
    assert manifest.identity().version == "1.0.0"


def test_digest_changes_when_annotations_or_selections_change() -> None:
    base = _manifest()

    reordered = _manifest(split_schemes=(_reordered_scheme(),))
    other_annotation = _manifest(
        annotations=(replace(_annotation(), content_hash=_hash("edited")),)
    )
    other_sample = _manifest(
        samples=(replace(_sample(0), content_hash=_hash("edited")), _sample(1), _sample(2))
    )

    assert (
        len({base.digest(), reordered.digest(), other_annotation.digest(), other_sample.digest()})
        == 4
    )


def _reordered_scheme() -> SplitScheme:
    scheme = _scheme()
    test = replace(scheme.splits[1], sample_ids=tuple(reversed(scheme.splits[1].sample_ids)))
    return replace(scheme, splits=(scheme.splits[0], test))


def test_selection_reproduces_the_exact_ordered_samples() -> None:
    manifest = _manifest()

    selected = manifest.selection("regions-by-sequence", "test")

    assert [sample.sample_id for sample in selected] == ["sample-0002", "sample-0001"]
    assert selected[0].observation_ids == (
        SourceObservationId("frame-0002"),
        SourceObservationId("scan-0002"),
    )


def test_selection_rejects_unknown_scheme_or_split() -> None:
    manifest = _manifest()

    with pytest.raises(ReferenceSetError):
        manifest.selection("missing", "test")
    with pytest.raises(ReferenceSetError):
        manifest.selection("regions-by-sequence", "missing")


def test_samples_are_bound_to_physical_observation_identities() -> None:
    with pytest.raises(ValueError, match="observation"):
        replace(_sample(0), observation_ids=())
    with pytest.raises(ValueError, match="unique"):
        replace(
            _sample(0),
            observation_ids=(SourceObservationId("frame-0000"), SourceObservationId("frame-0000")),
        )


def test_sample_and_annotation_identities_must_be_unique() -> None:
    with pytest.raises(ValueError, match="sample"):
        _manifest(samples=(_sample(0), _sample(0), _sample(1), _sample(2)))
    with pytest.raises(ValueError, match="annotation"):
        _manifest(annotations=(_annotation(), _annotation()))


def test_references_must_resolve_inside_the_manifest() -> None:
    with pytest.raises(ValueError, match="source"):
        _manifest(samples=(replace(_sample(0), source_id="missing"), _sample(1), _sample(2)))
    with pytest.raises(ValueError, match="calibration"):
        _manifest(
            samples=(
                replace(_sample(0), calibration_ids=(CalibrationReferenceId("missing"),)),
                _sample(1),
                _sample(2),
            )
        )
    with pytest.raises(ValueError, match="provenance"):
        _manifest(annotations=(replace(_annotation(), provenance_id="missing"),))
    with pytest.raises(ValueError, match="annotated sample"):
        _manifest(
            annotations=(replace(_annotation(), sample_ids=(ReferenceSampleId("sample-9999"),)),)
        )
    with pytest.raises(ValueError, match="split sample"):
        scheme = _scheme()
        broken = replace(scheme.splits[0], sample_ids=(ReferenceSampleId("sample-9999"),))
        _manifest(split_schemes=(replace(scheme, splits=(broken, scheme.splits[1])),))


def test_annotations_are_required_to_declare_provenance_and_trust() -> None:
    document = encode_reference_set(_manifest())
    del document["annotations"][0]["trust"]  # type: ignore[index]
    with pytest.raises(ReferenceSetError, match="trust"):
        decode_reference_set(document)

    document = encode_reference_set(_manifest())
    del document["annotations"][0]["provenance_id"]  # type: ignore[index]
    with pytest.raises(ReferenceSetError, match="provenance_id"):
        decode_reference_set(document)


def test_trust_is_never_inferred_from_file_names() -> None:
    entry = replace(_annotation(trust=ReferenceTrust.DIAGNOSTIC_ONLY), path="reference/gt.pcd")

    manifest = _manifest(annotations=(entry,))

    assert manifest.annotations[0].trust is ReferenceTrust.DIAGNOSTIC_ONLY
    assert entry.path == "reference/gt.pcd"


@pytest.mark.parametrize(
    "trust",
    [
        ReferenceTrust.TRUSTED_GROUND_TRUTH,
        ReferenceTrust.APPROXIMATE_ANNOTATION,
        ReferenceTrust.DERIVED_MEASUREMENT,
    ],
)
def test_model_inference_can_only_be_diagnostic(trust: ReferenceTrust) -> None:
    provenance = replace(_provenance(ProvenanceOrigin.MODEL_INFERENCE), provenance_id="prov-model")
    entry = replace(_annotation(trust=trust), provenance_id="prov-model")

    with pytest.raises(ValueError, match="model"):
        _manifest(provenance=(provenance,), annotations=(entry,))


def test_model_inference_is_accepted_as_explicit_diagnostic_data() -> None:
    provenance = replace(_provenance(ProvenanceOrigin.MODEL_INFERENCE), provenance_id="prov-model")
    entry = replace(_annotation(trust=ReferenceTrust.DIAGNOSTIC_ONLY), provenance_id="prov-model")

    manifest = _manifest(provenance=(provenance,), annotations=(entry,))

    assert manifest.annotations[0].trust is ReferenceTrust.DIAGNOSTIC_ONLY


def test_provenance_records_seeding_artifacts_and_review() -> None:
    reviewed = AnnotationProvenance(
        provenance_id="prov-assisted",
        origin=ProvenanceOrigin.MANUAL_ANNOTATION,
        annotator="annotator-team-a",
        method="masks seeded by a model run and corrected by hand",
        tool_version=None,
        seeded_from_artifacts=("perception-run-0007",),
        review=ProvenanceReview(reviewer="reviewer-b", method="full pass against source frames"),
    )

    manifest = _manifest(
        provenance=(_provenance(), reviewed),
    )
    restored = decode_reference_set(encode_reference_set(manifest))

    assert restored.provenance[1].seeded_from_artifacts == ("perception-run-0007",)
    assert restored.provenance[1].review == reviewed.review


def test_strata_must_use_declared_definitions() -> None:
    with pytest.raises(ValueError, match="stratum"):
        _manifest(
            samples=(
                replace(_sample(0), strata=(SampleStratum(name="visibility", value="foggy"),)),
                _sample(1),
                _sample(2),
            )
        )
    with pytest.raises(ValueError, match="stratum"):
        _manifest(
            samples=(
                replace(_sample(0), strata=(SampleStratum(name="range", value="near"),)),
                _sample(1),
                _sample(2),
            )
        )


def test_hashes_and_paths_are_validated() -> None:
    with pytest.raises(ValueError, match="sha256"):
        replace(_source(), content_hash="md5:abc")
    for path in ("/abs/regions.json", "../regions.json", "annotations/../../x.json", ""):
        with pytest.raises(ValueError, match="path"):
            replace(_annotation(), path=path)


def test_time_span_requires_ordered_timestamps_of_one_clock() -> None:
    start = SourceTimestamp(seconds=2, nanoseconds=0, clock_id="dataset:seq-a")
    earlier = SourceTimestamp(seconds=1, nanoseconds=0, clock_id="dataset:seq-a")
    other_clock = SourceTimestamp(seconds=3, nanoseconds=0, clock_id="other")

    with pytest.raises(ValueError, match="before"):
        SampleTimeSpan(start=start, end=earlier)
    with pytest.raises(ValueError, match="clock"):
        SampleTimeSpan(start=start, end=other_clock)


def test_writer_is_immutable_and_reader_verifies_the_digest(tmp_path: Path) -> None:
    manifest = _manifest()
    root = tmp_path / "ref-set-alpha" / "1.0.0"

    write_reference_set(root, manifest)

    assert read_reference_set(root) == manifest
    with pytest.raises(FileExistsError):
        write_reference_set(root, manifest)

    path = root / "manifest.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["samples"][0]["content_hash"] = _hash("tampered")
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ReferenceSetError, match="digest"):
        read_reference_set(root)


def test_reader_rejects_unknown_schema(tmp_path: Path) -> None:
    document = encode_reference_set(_manifest())
    document["schema"] = "contextmap.reference-set/v0"

    with pytest.raises(ReferenceSetError, match="schema"):
        decode_reference_set(document)


def test_annotation_files_are_verified_against_declared_hashes(tmp_path: Path) -> None:
    content = b'{"records": []}'
    entry = replace(_annotation(), content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}")
    manifest = _manifest(annotations=(entry,))
    target = tmp_path / "annotations" / "regions.json"

    with pytest.raises(ReferenceSetError, match="missing"):
        verify_annotation_files(manifest, tmp_path)

    target.parent.mkdir()
    target.write_bytes(content)
    verify_annotation_files(manifest, tmp_path)

    target.write_bytes(b'{"records": [1]}')
    with pytest.raises(ReferenceSetError, match="hash"):
        verify_annotation_files(manifest, tmp_path)


def test_version_must_change_when_content_changes() -> None:
    previous = _manifest()
    edited = _manifest(annotations=(replace(_annotation(), content_hash=_hash("edited")),))

    require_version_bump_on_change(previous, previous)
    require_version_bump_on_change(previous, replace(edited, version="1.1.0"))
    with pytest.raises(ReferenceSetError, match="version"):
        require_version_bump_on_change(previous, edited)
    with pytest.raises(ReferenceSetError, match="reference_set_id"):
        require_version_bump_on_change(previous, replace(edited, reference_set_id="other"))

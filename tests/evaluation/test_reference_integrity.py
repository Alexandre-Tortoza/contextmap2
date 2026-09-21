"""Reference-set integrity and leakage-prevention tests."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from reference_set_builders import (
    REGIONS_SCHEMA,
    content_hash,
    make_annotation,
    make_provenance,
    make_sample,
    make_valid_manifest,
    regions_for,
    sample_id,
    write_valid_reference_set,
)

from contextmap.evaluation.annotations import (
    AnnotationSet,
    GeometryAnnotationSet,
    GeometryCorrespondence,
    write_annotation_set,
)
from contextmap.evaluation.reference_integrity import (
    IntegritySeverity,
    ReferenceSetIntegrityError,
    ReferenceSetIntegrityReport,
    ValidatedReferenceSet,
    open_validated_reference_set,
    require_valid_reference_set,
    validate_reference_set,
)
from contextmap.evaluation.reference_set import (
    AnnotationFileEntry,
    ProvenanceOrigin,
    ReferenceSetError,
    ReferenceSplit,
    ReferenceTrust,
    SampleGroup,
    SplitRole,
    SplitScheme,
    SplitUnit,
    write_reference_set,
)
from contextmap.ingestion import CalibrationReferenceId, SourceObservationId


def _codes(report: ReferenceSetIntegrityReport, severity: IntegritySeverity) -> set[str]:
    return {item.code for item in report.findings if item.severity is severity}


def _blockers(report: ReferenceSetIntegrityReport) -> set[str]:
    return _codes(report, IntegritySeverity.BLOCKER)


def _warnings(report: ReferenceSetIntegrityReport) -> set[str]:
    return _codes(report, IntegritySeverity.WARNING)


def _scheme(
    *,
    unit: SplitUnit,
    tuning: tuple[int, ...],
    test: tuple[int, ...],
    adjacency_window_ns: int | None = None,
) -> SplitScheme:
    return SplitScheme(
        scheme_id="regions-by-sequence",
        task="region_discovery",
        unit=unit,
        rationale="chosen so related samples stay together",
        splits=(
            ReferenceSplit(
                name="tuning",
                role=SplitRole.TUNING,
                sample_ids=tuple(sample_id(index) for index in tuning),
            ),
            ReferenceSplit(
                name="test",
                role=SplitRole.TEST,
                sample_ids=tuple(sample_id(index) for index in test),
            ),
        ),
        adjacency_window_ns=adjacency_window_ns,
    )


def test_a_clean_reference_set_has_no_findings() -> None:
    manifest = make_valid_manifest()

    report = validate_reference_set(manifest)

    assert report.is_valid
    assert report.findings == ()
    assert report.reference_set == manifest.identity()
    assert not report.files_checked


def test_duplicate_sample_content_is_a_blocker() -> None:
    samples = list(make_valid_manifest().samples)
    samples[3] = replace(samples[3], content_hash=samples[0].content_hash)

    report = validate_reference_set(make_valid_manifest(samples=tuple(samples)))

    assert "duplicate-sample-content" in _blockers(report)
    assert not report.is_valid


def test_a_sample_in_two_splits_of_one_scheme_is_a_blocker() -> None:
    scheme = _scheme(unit=SplitUnit.SEQUENCE, tuning=(0, 1), test=(0, 2, 3))

    report = validate_reference_set(make_valid_manifest(split_schemes=(scheme,)))

    assert "split-overlap" in _blockers(report)


def test_a_sequence_straddling_the_split_is_leakage() -> None:
    scheme = _scheme(unit=SplitUnit.SEQUENCE, tuning=(0, 1, 2), test=(3,))

    report = validate_reference_set(make_valid_manifest(split_schemes=(scheme,)))

    leak = next(item for item in report.findings if item.code == "split-leakage")
    assert leak.severity is IntegritySeverity.BLOCKER
    assert "sequence:seq-b" in leak.subjects


def test_scene_leakage_and_missing_group_keys_are_detected() -> None:
    base = make_valid_manifest()
    samples = list(base.samples)
    samples[2] = replace(samples[2], groups=(SampleGroup(unit=SplitUnit.SCENE, key="scene-a"),))
    scheme = _scheme(unit=SplitUnit.SCENE, tuning=(0, 1), test=(2, 3))

    leaking = validate_reference_set(
        make_valid_manifest(samples=tuple(samples), split_schemes=(scheme,))
    )
    assert "split-leakage" in _blockers(leaking)
    assert any("scene:scene-a" in item.subjects for item in leaking.findings)

    samples[2] = replace(samples[2], groups=())
    keyless = validate_reference_set(
        make_valid_manifest(samples=tuple(samples), split_schemes=(scheme,))
    )
    assert "missing-group-key" in _blockers(keyless)


def test_scene_units_keep_distinct_scenes_apart_without_leakage() -> None:
    scheme = _scheme(unit=SplitUnit.SCENE, tuning=(0, 1), test=(2, 3))

    report = validate_reference_set(make_valid_manifest(split_schemes=(scheme,)))

    assert report.is_valid
    assert report.findings == ()


def test_sequence_shared_across_splits_warns_when_the_unit_is_not_the_sequence() -> None:
    base = make_valid_manifest()
    samples = list(base.samples)
    samples[1] = replace(samples[1], groups=(SampleGroup(unit=SplitUnit.SCENE, key="scene-c"),))
    scheme = _scheme(unit=SplitUnit.SCENE, tuning=(0,), test=(1, 2, 3))

    report = validate_reference_set(
        make_valid_manifest(samples=tuple(samples), split_schemes=(scheme,))
    )

    assert report.is_valid
    assert "sequence-shared-across-splits" in _warnings(report)


def test_temporally_adjacent_samples_in_different_splits_leak() -> None:
    base = make_valid_manifest()
    samples = list(base.samples)
    samples[1] = make_sample(1, source_id="seq-a", scene="scene-c", second=3)
    scheme = _scheme(
        unit=SplitUnit.SCENE,
        tuning=(0,),
        test=(1, 2, 3),
        adjacency_window_ns=5_000_000_000,
    )
    adjacent = validate_reference_set(
        make_valid_manifest(samples=tuple(samples), split_schemes=(scheme,))
    )

    samples[1] = make_sample(1, source_id="seq-a", scene="scene-c", second=30)
    distant = validate_reference_set(
        make_valid_manifest(samples=tuple(samples), split_schemes=(scheme,))
    )

    assert "adjacent-split-leakage" in _blockers(adjacent)
    assert "adjacent-split-leakage" not in _blockers(distant)


def test_adjacency_needs_a_time_span() -> None:
    base = make_valid_manifest()
    samples = list(base.samples)
    samples[0] = make_sample(0, source_id="seq-a", scene="scene-a", with_time_span=False)
    scheme = _scheme(unit=SplitUnit.SEQUENCE, tuning=(0, 1), test=(2, 3), adjacency_window_ns=1_000)

    report = validate_reference_set(
        make_valid_manifest(samples=tuple(samples), split_schemes=(scheme,))
    )

    assert "missing-time-span" in _blockers(report)


def test_shared_physical_observations_are_flagged() -> None:
    base = make_valid_manifest()
    samples = list(base.samples)
    samples[2] = replace(samples[2], observation_ids=samples[0].observation_ids)

    across = validate_reference_set(make_valid_manifest(samples=tuple(samples)))
    assert "split-observation-overlap" in _blockers(across)
    assert "shared-observation" in _warnings(across)

    samples = list(base.samples)
    samples[1] = replace(
        samples[1],
        observation_ids=(SourceObservationId("frame-0000"), SourceObservationId("scan-0001")),
    )
    within = validate_reference_set(make_valid_manifest(samples=tuple(samples)))
    assert "split-observation-overlap" not in _blockers(within)
    assert "shared-observation" in _warnings(within)


def test_calibration_must_belong_to_the_source_of_the_sample() -> None:
    samples = list(make_valid_manifest().samples)
    samples[2] = replace(samples[2], calibration_ids=(CalibrationReferenceId("cam0"),))

    report = validate_reference_set(make_valid_manifest(samples=tuple(samples)))

    assert "calibration-source-mismatch" in _blockers(report)


def test_split_policy_must_exist_and_be_usable() -> None:
    assert "no-split-scheme" in _blockers(
        validate_reference_set(make_valid_manifest(split_schemes=()))
    )

    no_test = SplitScheme(
        scheme_id="s",
        task="region_discovery",
        unit=SplitUnit.SEQUENCE,
        rationale="sequence is the unit",
        splits=(
            ReferenceSplit(
                name="tuning",
                role=SplitRole.TUNING,
                sample_ids=tuple(sample_id(index) for index in range(4)),
            ),
            ReferenceSplit(name="dev", role=SplitRole.DEVELOPMENT, sample_ids=()),
        ),
    )
    report = validate_reference_set(make_valid_manifest(split_schemes=(no_test,)))
    assert {"no-test-split", "empty-split"} <= _warnings(report)


def test_unsplit_and_unannotated_samples_are_reported() -> None:
    scheme = _scheme(unit=SplitUnit.SEQUENCE, tuning=(0, 1), test=(2,))
    annotation = make_annotation(samples=(0, 1, 2))

    report = validate_reference_set(
        make_valid_manifest(split_schemes=(scheme,), annotations=(annotation,))
    )

    assert {"sample-not-in-any-split", "sample-without-annotation"} <= _warnings(report)
    assert report.is_valid


def test_model_seeded_references_need_an_explicit_review() -> None:
    seeded = make_provenance(seeded_from_artifacts=("perception-run-0007",))
    unreviewed = validate_reference_set(make_valid_manifest(provenance=(seeded,)))
    assert "unreviewed-model-seeding" in _blockers(unreviewed)

    reviewed = validate_reference_set(
        make_valid_manifest(
            provenance=(
                make_provenance(seeded_from_artifacts=("perception-run-0007",), reviewed=True),
            )
        )
    )
    assert reviewed.is_valid
    assert "model-seeded-reviewed" in _warnings(reviewed)

    diagnostic = validate_reference_set(
        make_valid_manifest(
            provenance=(
                make_provenance(ProvenanceOrigin.MODEL_INFERENCE, seeded_from_artifacts=("run-1",)),
            ),
            annotations=(
                make_annotation(trust=ReferenceTrust.DIAGNOSTIC_ONLY, samples=(0, 1, 2, 3)),
            ),
        )
    )
    assert diagnostic.is_valid
    assert "diagnostic-annotation" in _warnings(diagnostic)


def test_provenance_audit_is_independent_of_model_outputs() -> None:
    manual = make_valid_manifest()
    audited = validate_reference_set(manual).provenance_audit[0]
    assert audited.independent_of_model_output
    assert audited.annotation_id == "ann-regions"
    assert audited.origin is ProvenanceOrigin.MANUAL_ANNOTATION

    seeded = make_valid_manifest(
        provenance=(make_provenance(seeded_from_artifacts=("perception-run-0007",), reviewed=True),)
    )
    entry = validate_reference_set(seeded).provenance_audit[0]
    assert not entry.independent_of_model_output
    assert entry.reviewed
    assert entry.seeded_from_artifacts == ("perception-run-0007",)


def test_unused_provenance_and_annotation_hygiene() -> None:
    other = make_provenance(provenance_id="prov-unused")
    duplicate_path = make_annotation(annotation_id="ann-2", provenance_id="prov-manual")
    same_content = make_annotation(
        annotation_id="ann-3", path="annotations/other.json", file_hash=content_hash("ann-regions")
    )
    bad_schema = make_annotation(
        annotation_id="ann-4", path="annotations/x.json", schema=f"{REGIONS_SCHEMA[:-2]}v9"
    )
    empty = make_annotation(annotation_id="ann-5", path="annotations/empty.json", samples=())

    report = validate_reference_set(
        make_valid_manifest(
            provenance=(make_provenance(), other),
            annotations=(
                make_annotation(samples=(0, 1, 2, 3)),
                duplicate_path,
                same_content,
                bad_schema,
                empty,
            ),
        )
    )

    assert "duplicate-annotation-path" in _blockers(report)
    assert "unknown-annotation-schema" in _blockers(report)
    assert {"unused-provenance", "duplicate-annotation-content", "annotation-without-samples"} <= (
        _warnings(report)
    )


def test_files_are_checked_against_the_manifest_when_a_root_is_given(tmp_path: Path) -> None:
    manifest = write_valid_reference_set(tmp_path)

    report = validate_reference_set(manifest, tmp_path)

    assert report.is_valid
    assert report.findings == ()
    assert report.files_checked


def test_missing_modified_and_unreadable_files_are_blockers(tmp_path: Path) -> None:
    manifest = write_valid_reference_set(tmp_path)
    target = tmp_path / "annotations" / "regions.json"

    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert "annotation-file-hash-mismatch" in _blockers(validate_reference_set(manifest, tmp_path))

    target.unlink()
    assert "annotation-file-missing" in _blockers(validate_reference_set(manifest, tmp_path))

    target.write_text("{not json", encoding="utf-8")
    broken = replace(manifest.annotations[0], content_hash=_file_hash(target))
    unreadable = validate_reference_set(make_valid_manifest(annotations=(broken,)), tmp_path)
    assert "annotation-unreadable" in _blockers(unreadable)


def _file_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_annotations_must_link_to_the_samples_and_observations_of_the_manifest(
    tmp_path: Path,
) -> None:
    valid = write_valid_reference_set(tmp_path)
    wrong_observation = regions_for(0)
    frame = replace(wrong_observation.frames[0], observation_id=SourceObservationId("frame-9999"))
    unknown_sample = replace(regions_for(3).frames[0], sample_id=sample_id(42))
    undeclared = regions_for(2, 3)

    def entry(
        name: str, annotation_set: AnnotationSet, samples: tuple[int, ...]
    ) -> AnnotationFileEntry:
        path = tmp_path / "annotations" / f"{name}.json"
        digest = write_annotation_set(path, annotation_set)
        return make_annotation(
            annotation_id=name,
            path=f"annotations/{name}.json",
            file_hash=digest,
            samples=samples,
        )

    entries = (
        entry("wrong-observation", replace(wrong_observation, frames=(frame,)), (0,)),
        entry("unknown-sample", replace(regions_for(3), frames=(unknown_sample,)), (3,)),
        entry("undeclared", undeclared, (2,)),
    )

    report = validate_reference_set(
        make_valid_manifest(annotations=(*valid.annotations, *entries)),
        tmp_path,
    )

    assert {
        "annotation-unknown-observation",
        "annotation-unknown-sample",
        "annotation-sample-not-declared",
    } <= _blockers(report)


def test_geometry_annotations_must_use_the_calibration_of_their_sample(tmp_path: Path) -> None:
    valid = write_valid_reference_set(tmp_path)
    geometry = GeometryAnnotationSet(
        correspondences=(
            GeometryCorrespondence(
                correspondence_id="corr-1",
                sample_id=sample_id(0),
                image_observation_id=SourceObservationId("frame-0000"),
                calibration_id=CalibrationReferenceId("cam1"),
                pixel_uv=(1.0, 2.0),
                point_frame_id="lidar",
                point_m=(1.0, 2.0, 3.0),
            ),
        )
    )
    path = tmp_path / "annotations" / "geometry.json"
    digest = write_annotation_set(path, geometry)
    entry = make_annotation(
        annotation_id="ann-geometry",
        path="annotations/geometry.json",
        file_hash=digest,
        schema="contextmap.reference.geometry/v1",
        samples=(0,),
    )

    report = validate_reference_set(
        make_valid_manifest(annotations=(*valid.annotations, entry)), tmp_path
    )

    assert "annotation-calibration-mismatch" in _blockers(report)


def test_a_file_whose_schema_differs_from_its_entry_is_a_blocker(tmp_path: Path) -> None:
    valid = write_valid_reference_set(tmp_path)
    wrong = replace(valid.annotations[0], schema="contextmap.reference.semantics/v1")

    report = validate_reference_set(make_valid_manifest(annotations=(wrong,)), tmp_path)

    assert "annotation-schema-mismatch" in _blockers(report)


def test_evaluation_tooling_refuses_invalid_reference_sets(tmp_path: Path) -> None:
    valid = write_valid_reference_set(tmp_path)
    leaking = _scheme(unit=SplitUnit.SEQUENCE, tuning=(0, 1, 2), test=(3,))
    invalid = make_valid_manifest(annotations=valid.annotations, split_schemes=(leaking,))

    with pytest.raises(ReferenceSetIntegrityError, match="split-leakage") as raised:
        require_valid_reference_set(invalid, tmp_path)
    assert not raised.value.report.is_valid

    accepted = require_valid_reference_set(valid, tmp_path)
    assert isinstance(accepted, ValidatedReferenceSet)
    assert accepted.manifest == valid
    assert accepted.report.is_valid and accepted.report.files_checked


def test_a_validated_reference_set_cannot_carry_a_failing_or_foreign_report(
    tmp_path: Path,
) -> None:
    valid = write_valid_reference_set(tmp_path)
    good = validate_reference_set(valid, tmp_path)
    failing = validate_reference_set(make_valid_manifest(split_schemes=()), tmp_path)
    other = replace(valid, version="2.0.0")

    with pytest.raises(ReferenceSetIntegrityError):
        ValidatedReferenceSet(manifest=valid, report=failing, root=tmp_path)
    with pytest.raises(ReferenceSetError, match="another reference set"):
        ValidatedReferenceSet(manifest=other, report=good, root=tmp_path)
    with pytest.raises(ReferenceSetError, match="files"):
        ValidatedReferenceSet(manifest=valid, report=validate_reference_set(valid), root=tmp_path)


def test_open_validated_reference_set_reads_verifies_and_validates(tmp_path: Path) -> None:
    manifest = write_valid_reference_set(tmp_path)
    write_reference_set(tmp_path, manifest)

    opened = open_validated_reference_set(tmp_path)

    assert opened.manifest == manifest
    assert opened.report.reference_set.digest == manifest.digest()

    document = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    document["version"] = "9.9.9"
    (tmp_path / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ReferenceSetError, match="digest"):
        open_validated_reference_set(tmp_path)


def test_the_report_is_machine_readable() -> None:
    samples = list(make_valid_manifest().samples)
    samples[3] = replace(samples[3], content_hash=samples[0].content_hash)
    report = validate_reference_set(make_valid_manifest(samples=tuple(samples)))

    record = json.loads(json.dumps(report.to_record()))

    assert record["valid"] is False
    assert record["reference_set"]["digest"] == report.reference_set.digest
    assert record["blockers"][0]["code"] == "duplicate-sample-content"
    assert record["blockers"][0]["subjects"]
    assert record["provenance_audit"][0]["annotation_id"] == "ann-regions"
    assert record["files_checked"] is False

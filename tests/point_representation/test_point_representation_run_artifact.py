import dataclasses
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pointrep_builders import make_representation, make_space, make_support, radius_policy
from pointrep_fakes import FakeEncoder, RejectingSmallSupportsEncoder
from pointrep_geometry import MAP_ID, LinearScanSource, line_of_points, points_from

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.point_representation import (
    CenteringMode,
    EncodedRepresentation,
    FailedSupport,
    FailureReason,
    PointEncoder,
    PointRepresentationId,
    PointRepresentationRunId,
    RepresentationService,
    ScaleNormalization,
    representation_space_fingerprint,
)
from contextmap.point_representation.backends.geometric_descriptor import (
    GeometricDescriptorEncoder,
)
from contextmap.point_representation.run_artifact import (
    SCHEMA_VERSION,
    IncompleteRunArtifactError,
    PointRepresentationDebugLevel,
    PointRepresentationRunManifest,
    PointRepresentationRunReader,
    PointRepresentationRunWriter,
    RunArtifactError,
    allocate_run_index,
    rebuild_run_registry,
)

RUN_ID = PointRepresentationRunId("run-0001")
SEQUENCE = "corridor-02"
Outcome = EncodedRepresentation | FailedSupport


def ref(index: int, *, map_id: MapId = MAP_ID) -> GeometryReference:
    return GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))


@dataclass
class BuiltRun:
    run_dir: Path
    manifest: PointRepresentationRunManifest
    outcomes: list[Outcome]
    encoder: PointEncoder
    service: RepresentationService
    writer: PointRepresentationRunWriter


def make_writer(
    workspace: Path,
    encoder: PointEncoder,
    source: LinearScanSource,
    *,
    run_index: int = 1,
    debug_level: PointRepresentationDebugLevel = PointRepresentationDebugLevel.NONE,
    association_context_id: str | None = None,
) -> PointRepresentationRunWriter:
    return PointRepresentationRunWriter(
        workspace_root=workspace,
        sequence_name=SEQUENCE,
        run_id=RUN_ID,
        run_index=run_index,
        selection_label="centers",
        backend_label="encoder",
        geometric_map=source.geometric_map,
        space=encoder.representation_space(),
        encoder_identity=encoder.encoder_identity(),
        code_version="test-version",
        association_context_id=association_context_id,
        debug_level=debug_level,
    )


def build_run(
    workspace: Path,
    *,
    encoder: PointEncoder | None = None,
    source: LinearScanSource | None = None,
    centers: Sequence[int] = (5, 0, 2),
    run_index: int = 1,
    debug_level: PointRepresentationDebugLevel = PointRepresentationDebugLevel.NONE,
    association_context_id: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> BuiltRun:
    source = source if source is not None else LinearScanSource(line_of_points(11))
    chosen = encoder if encoder is not None else FakeEncoder()
    service = RepresentationService(source, chosen, run_id=RUN_ID, code_version="test-version")
    writer = make_writer(
        workspace,
        chosen,
        source,
        run_index=run_index,
        debug_level=debug_level,
        association_context_id=association_context_id,
    )
    outcomes = list(service.represent([ref(index) for index in centers]))
    for outcome in outcomes:
        writer.add(outcome)
    manifest = writer.finalize(metrics=service.metrics, backend_diagnostics=diagnostics)
    run_dir = (
        workspace
        / "runs"
        / "point-representation"
        / SEQUENCE
        / f"run-{run_index:04d}__centers__encoder"
    )
    return BuiltRun(run_dir, manifest, outcomes, chosen, service, writer)


def represented(built: BuiltRun) -> list[EncodedRepresentation]:
    return [o for o in built.outcomes if isinstance(o, EncodedRepresentation)]


# --- Round trip ---------------------------------------------------------------------


def test_a_run_round_trips_representations_vectors_and_the_space(tmp_path: Path) -> None:
    built = build_run(tmp_path)

    reader = PointRepresentationRunReader(built.run_dir)

    persisted = list(reader.iter_representations())
    assert persisted == [
        dataclasses.replace(
            outcome.representation, payload_reference=f"outputs/payloads/vectors.f32#{row}"
        )
        for row, outcome in enumerate(represented(built))
    ]
    for outcome in represented(built):
        assert reader.vector(outcome.representation.representation_id) == outcome.values
    assert reader.space() == built.encoder.representation_space()


def test_a_representation_is_read_by_identity_or_by_geometry(tmp_path: Path) -> None:
    built = build_run(tmp_path)
    reader = PointRepresentationRunReader(built.run_dir)
    second = represented(built)[1].representation

    assert reader.representation(second.representation_id).geometry_reference == ref(0)
    assert reader.representation_for(ref(0)) == reader.representation(second.representation_id)
    assert reader.representation_for(ref(9)) is None
    with pytest.raises(RunArtifactError, match="unknown"):
        reader.representation(PointRepresentationId("nope"))


def test_a_persisted_representation_still_names_its_exact_support(tmp_path: Path) -> None:
    built = build_run(tmp_path)
    reader = PointRepresentationRunReader(built.run_dir)

    persisted = reader.representation_for(ref(5))

    assert persisted is not None
    assert persisted.support.geometry_refs == (ref(5), ref(4), ref(6), ref(3), ref(7))
    assert persisted.support.policy == radius_policy(0.6)


def test_partial_descriptors_survive_with_their_undefined_components(tmp_path: Path) -> None:
    encoder = GeometricDescriptorEncoder(radius_policy(0.3))
    source = LinearScanSource(points_from([(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (50.0, 0.0, 0.0)]))

    built = build_run(tmp_path, encoder=encoder, source=source, centers=(0, 2))

    reader = PointRepresentationRunReader(built.run_dir)
    assert built.manifest.dtype == "float64"
    assert built.manifest.payload_path == "outputs/payloads/vectors.f64"
    for outcome in represented(built):
        persisted = reader.representation(outcome.representation.representation_id)
        assert persisted.undefined_components == outcome.representation.undefined_components
        assert persisted.is_partial
        assert reader.vector(persisted.representation_id) == outcome.values
    assert built.manifest.partial_count == 2


def test_failed_supports_are_persisted_explicitly_and_never_as_vectors(tmp_path: Path) -> None:
    built = build_run(tmp_path, encoder=RejectingSmallSupportsEncoder(minimum=4))

    reader = PointRepresentationRunReader(built.run_dir)

    failed = [o for o in built.outcomes if isinstance(o, FailedSupport)]
    assert len(failed) == 1
    assert reader.failed_supports() == failed
    assert reader.failed_supports()[0].reason is FailureReason.UNENCODABLE_SUPPORT
    assert len(list(reader.iter_representations())) == 2
    assert reader.representation_for(ref(0)) is None
    assert built.manifest.failed_count == 1
    assert dict(built.manifest.failed_by_reason) == {"unencodable_support": 1}


def test_a_run_of_only_failures_is_a_valid_explicit_run(tmp_path: Path) -> None:
    built = build_run(tmp_path, encoder=RejectingSmallSupportsEncoder(minimum=99))

    reader = PointRepresentationRunReader(built.run_dir)

    assert list(reader.iter_representations()) == []
    assert len(reader.failed_supports()) == 3
    assert reader.verify_integrity() == []


# --- Lineage -------------------------------------------------------------------------


def test_the_manifest_records_the_lineage_needed_to_reproduce_the_run(tmp_path: Path) -> None:
    encoder = GeometricDescriptorEncoder(
        radius_policy(
            0.6, centering=CenteringMode.CENTROID, scale=ScaleNormalization.SUPPORT_RADIUS
        )
    )
    built = build_run(tmp_path, encoder=encoder, association_context_id="association-run-0003")

    manifest = PointRepresentationRunReader(built.run_dir).manifest

    assert manifest.schema_version == SCHEMA_VERSION
    assert (manifest.run_id, manifest.run_index, manifest.sequence_name) == (RUN_ID, 1, SEQUENCE)
    assert manifest.geometric_map_id == MAP_ID
    assert manifest.geometric_map_frame == "map"
    assert manifest.geometric_map_point_count == 11
    assert manifest.sequence_artifact_id == "sequence-0001"
    assert manifest.association_context_id == "association-run-0003"
    assert manifest.support_policy == encoder.representation_space().support_semantics
    assert manifest.support_policy.preparation.centering is CenteringMode.CENTROID
    assert manifest.representation_space_id == representation_space_fingerprint(
        encoder.representation_space()
    )
    assert manifest.encoder == encoder.encoder_identity()
    assert manifest.code_version == "test-version"
    assert (manifest.representation_count, manifest.failed_count) == (3, 0)
    assert manifest.debug_level == "none"


def test_without_an_association_the_lineage_says_so(tmp_path: Path) -> None:
    assert build_run(tmp_path).manifest.association_context_id is None


def test_the_center_selection_identity_ignores_request_order(tmp_path: Path) -> None:
    first = build_run(tmp_path / "a", centers=(5, 0, 2)).manifest
    second = build_run(tmp_path / "b", centers=(2, 5, 0)).manifest
    other = build_run(tmp_path / "c", centers=(5, 0, 3)).manifest

    assert first.center_selection_id == second.center_selection_id
    assert first.center_selection_id != other.center_selection_id


def test_the_artifact_does_not_duplicate_geometry_or_other_evidence(tmp_path: Path) -> None:
    built = build_run(tmp_path)

    for path in built.run_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
            text = path.read_text(encoding="utf-8")
            assert "coordinates_m" not in text, path
            assert '"label"' not in text and '"claim' not in text, path


# --- Metrics --------------------------------------------------------------------------


def test_counts_support_sizes_and_norms_are_recorded_as_metrics(tmp_path: Path) -> None:
    built = build_run(tmp_path, encoder=RejectingSmallSupportsEncoder(minimum=4))
    reader = PointRepresentationRunReader(built.run_dir)

    counts = reader.read_record("metrics/counts.json")
    sizes = reader.read_record("metrics/support-size.json")
    norms = reader.read_record("metrics/norms.json")

    assert counts == {
        "requested": 3,
        "represented": 2,
        "partial": 0,
        "failed": 1,
        "failed_by_reason": {"unencodable_support": 1},
    }
    assert (sizes["count"], sizes["minimum"], sizes["maximum"]) == (3, 3, 5)
    assert norms["count"] == 2
    assert norms["maximum"] == pytest.approx(math.hypot(5.0, 0.0, 0.0))
    assert norms["non_finite_values"] == 0


def test_runtime_and_backend_diagnostics_are_recorded_apart_from_quality(tmp_path: Path) -> None:
    diagnostics = {"peak_memory_bytes": 123456, "device": "cuda:0", "out_of_memory": 0}
    built = build_run(tmp_path, diagnostics=diagnostics)

    runtime = PointRepresentationRunReader(built.run_dir).read_record("metrics/runtime.json")

    assert runtime["backend"] == diagnostics
    assert runtime["support_extraction_seconds"] >= 0.0
    assert runtime["encoding_seconds"] >= 0.0


# --- Lazy payloads and integrity -------------------------------------------------------


def test_reading_one_vector_never_touches_the_others(tmp_path: Path) -> None:
    built = build_run(tmp_path)
    payload = built.run_dir / "outputs" / "payloads" / "vectors.f32"
    data = bytearray(payload.read_bytes())
    data[0:12] = b"\xff" * 12  # corrompe só a linha 0
    payload.write_bytes(bytes(data))
    reader = PointRepresentationRunReader(built.run_dir)
    second = represented(built)[1]

    assert reader.vector(second.representation.representation_id) == second.values
    assert any("vectors.f32" in problem for problem in reader.verify_integrity())


def test_a_truncated_payload_is_an_explicit_error(tmp_path: Path) -> None:
    built = build_run(tmp_path)
    payload = built.run_dir / "outputs" / "payloads" / "vectors.f32"
    payload.write_bytes(payload.read_bytes()[:-4])
    reader = PointRepresentationRunReader(built.run_dir)
    last = represented(built)[-1].representation.representation_id

    with pytest.raises(RunArtifactError, match="payload"):
        reader.vector(last)


def test_a_payload_reference_outside_the_run_payload_is_rejected(tmp_path: Path) -> None:
    built = build_run(tmp_path)
    records = built.run_dir / "outputs" / "representations.jsonl"
    rid = represented(built)[0].representation.representation_id
    # Mesmo comprimento: os deslocamentos do índice continuam válidos.
    text = records.read_text(encoding="utf-8").replace(
        "outputs/payloads/vectors.f32#0", "outputs/payloads/vectorz.f32#0", 1
    )
    records.write_text(text, encoding="utf-8")

    with pytest.raises(RunArtifactError, match="payload_reference"):
        PointRepresentationRunReader(built.run_dir).vector(rid)


@pytest.mark.parametrize(
    "relative_path",
    [
        "outputs/representations.jsonl",
        "outputs/representation-index.jsonl",
        "outputs/representation-spaces.json",
        "outputs/geometry-representation-index.jsonl",
        "outputs/failed-supports.jsonl",
        "outputs/payloads/vectors.f32",
        "metrics/counts.json",
    ],
)
def test_the_inventory_detects_a_missing_or_altered_contractual_file(
    tmp_path: Path, relative_path: str
) -> None:
    built = build_run(tmp_path)
    reader = PointRepresentationRunReader(built.run_dir)
    assert reader.verify_integrity() == []
    inventoried = {entry.path for entry in reader.manifest.file_inventory}
    assert relative_path in inventoried

    (built.run_dir / relative_path).write_bytes(b"tampered")

    assert any(relative_path in problem for problem in reader.verify_integrity())


def test_an_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    built = build_run(tmp_path)
    manifest_path = built.run_dir / "manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["schema_version"] = "9.9.9"
    manifest_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RunArtifactError, match="schema_version"):
        PointRepresentationRunReader(built.run_dir)


def test_a_directory_without_a_manifest_is_not_a_run(tmp_path: Path) -> None:
    (tmp_path / "half-written").mkdir()

    with pytest.raises(IncompleteRunArtifactError):
        PointRepresentationRunReader(tmp_path / "half-written")


def test_a_run_is_readable_without_numpy_or_a_model_library(tmp_path: Path) -> None:
    built = build_run(tmp_path)
    rid = represented(built)[0].representation.representation_id
    code = (
        "import sys;"
        "from pathlib import Path;"
        "from contextmap.point_representation.run_artifact import PointRepresentationRunReader;"
        f"reader = PointRepresentationRunReader(Path({str(built.run_dir)!r}));"
        f"reader.vector({str(rid)!r});"
        "assert reader.verify_integrity() == [];"
        "bad = [m for m in ('numpy', 'torch', 'rosbags') if m in sys.modules];"
        "assert not bad, bad"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


# --- Debug is never contractual ------------------------------------------------------------


def test_no_debug_level_writes_no_debug_directory(tmp_path: Path) -> None:
    assert not (build_run(tmp_path).run_dir / "debug").exists()


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (
            PointRepresentationDebugLevel.STANDARD,
            {"debug/support-traces.jsonl", "debug/representation-norms.csv"},
        ),
        (
            PointRepresentationDebugLevel.FULL,
            {
                "debug/support-traces.jsonl",
                "debug/representation-norms.csv",
                "debug/component-statistics.json",
            },
        ),
    ],
)
def test_debug_evidence_is_written_but_never_inventoried(
    tmp_path: Path, level: PointRepresentationDebugLevel, expected: set[str]
) -> None:
    built = build_run(tmp_path, debug_level=level)

    written = {
        str(path.relative_to(built.run_dir))
        for path in (built.run_dir / "debug").rglob("*")
        if path.is_file()
    }
    assert written == expected
    assert built.manifest.debug_level == level.value
    assert not any(entry.path.startswith("debug/") for entry in built.manifest.file_inventory)


def test_removing_debug_never_invalidates_the_run(tmp_path: Path) -> None:
    built = build_run(tmp_path, debug_level=PointRepresentationDebugLevel.FULL)
    shutil.rmtree(built.run_dir / "debug")

    reader = PointRepresentationRunReader(built.run_dir)

    assert reader.verify_integrity() == []
    assert len(list(reader.iter_representations())) == 3
    with pytest.raises(RunArtifactError, match="contractual"):
        reader.read_record("debug/support-traces.jsonl")


# --- Immutability, atomicity and identity ---------------------------------------------------


def test_a_finished_run_is_never_overwritten(tmp_path: Path) -> None:
    built = build_run(tmp_path)
    before = (built.run_dir / "manifest.json").read_bytes()
    source = LinearScanSource(line_of_points(11))
    encoder = FakeEncoder()
    writer = make_writer(tmp_path, encoder, source)
    service = RepresentationService(source, encoder, run_id=RUN_ID, code_version="test-version")
    for outcome in service.represent([ref(5)]):
        writer.add(outcome)

    with pytest.raises(RunArtifactError, match="exists"):
        writer.finalize(metrics=service.metrics)

    assert (built.run_dir / "manifest.json").read_bytes() == before
    assert not any(path.name.startswith(".tmp-") for path in built.run_dir.parent.iterdir())


def test_a_writer_finalizes_once(tmp_path: Path) -> None:
    built = build_run(tmp_path)

    with pytest.raises(RunArtifactError, match="finalized"):
        built.writer.finalize(metrics=built.service.metrics)
    with pytest.raises(RunArtifactError, match="finalized"):
        built.writer.add(built.outcomes[0])


def test_run_indexes_count_only_valid_runs_and_the_registry_is_rebuildable(tmp_path: Path) -> None:
    assert allocate_run_index(workspace_root=tmp_path, sequence_name=SEQUENCE) == 1
    first = build_run(tmp_path)
    sequence_dir = first.run_dir.parent
    (sequence_dir / ".tmp-run-0002__centers__encoder-abc12345").mkdir()

    assert allocate_run_index(workspace_root=tmp_path, sequence_name=SEQUENCE) == 2
    registry = json.loads((sequence_dir / "runs.json").read_text(encoding="utf-8"))
    assert [run["run_index"] for run in registry["runs"]] == [1]

    (sequence_dir / "runs.json").unlink()
    rebuild_run_registry(workspace_root=tmp_path, sequence_name=SEQUENCE)
    assert json.loads((sequence_dir / "runs.json").read_text(encoding="utf-8")) == registry

    (first.run_dir / "outputs" / "payloads" / "vectors.f32").write_bytes(b"corrupt")
    assert allocate_run_index(workspace_root=tmp_path, sequence_name=SEQUENCE) == 1


# --- Writer validation -----------------------------------------------------------------------


def _writer_and_outcome(
    tmp_path: Path,
) -> tuple[PointRepresentationRunWriter, EncodedRepresentation]:
    source = LinearScanSource(line_of_points(11))
    encoder = FakeEncoder()
    service = RepresentationService(source, encoder, run_id=RUN_ID, code_version="test-version")
    (outcome,) = service.represent([ref(5)])
    assert isinstance(outcome, EncodedRepresentation)
    return make_writer(tmp_path, encoder, source), outcome


def test_a_repeated_anchor_is_rejected(tmp_path: Path) -> None:
    writer, outcome = _writer_and_outcome(tmp_path)
    writer.add(outcome)

    with pytest.raises(RunArtifactError, match="already"):
        writer.add(outcome)


def test_a_representation_of_another_space_is_rejected(tmp_path: Path) -> None:
    writer, outcome = _writer_and_outcome(tmp_path)
    other = dataclasses.replace(outcome.representation, representation_space_id="sha256:other")

    with pytest.raises(RunArtifactError, match="representation space"):
        writer.add(EncodedRepresentation(representation=other, values=outcome.values))


def test_a_representation_of_another_encoder_is_rejected(tmp_path: Path) -> None:
    writer, outcome = _writer_and_outcome(tmp_path)
    other = dataclasses.replace(
        outcome.representation,
        encoder_identity=dataclasses.replace(
            outcome.representation.encoder_identity, backend_version="2"
        ),
    )

    with pytest.raises(RunArtifactError, match="encoder"):
        writer.add(EncodedRepresentation(representation=other, values=outcome.values))


def test_a_representation_of_another_map_is_rejected(tmp_path: Path) -> None:
    writer, outcome = _writer_and_outcome(tmp_path)
    foreign_support = make_support(0, (0, 1), policy=radius_policy(0.6), map_id=MapId("other-map"))
    foreign = dataclasses.replace(
        make_representation(space=make_space(policy=radius_policy(0.6)), support=foreign_support),
        representation_space_id=outcome.representation.representation_space_id,
        encoder_identity=outcome.representation.encoder_identity,
        shape=(3,),
        payload_reference=None,
    )

    with pytest.raises(RunArtifactError, match="map"):
        writer.add(EncodedRepresentation(representation=foreign, values=(1.0, 2.0, 3.0)))


def test_a_payload_reference_is_assigned_by_the_writer_only(tmp_path: Path) -> None:
    writer, outcome = _writer_and_outcome(tmp_path)
    preset = dataclasses.replace(outcome.representation, payload_reference="outputs/x#0")

    with pytest.raises(RunArtifactError, match="payload"):
        writer.add(EncodedRepresentation(representation=preset, values=outcome.values))


@pytest.mark.parametrize("values", [(1.0, 2.0), (1.0, math.nan, 3.0), (1.0, math.inf, 3.0)])
def test_a_vector_that_breaks_the_space_is_rejected(
    tmp_path: Path, values: tuple[float, ...]
) -> None:
    writer, outcome = _writer_and_outcome(tmp_path)

    with pytest.raises(RunArtifactError, match="vector"):
        writer.add(EncodedRepresentation(representation=outcome.representation, values=values))


def test_a_vector_outside_the_float32_range_is_rejected_not_stored_as_infinity(
    tmp_path: Path,
) -> None:
    writer, outcome = _writer_and_outcome(tmp_path)

    with pytest.raises(RunArtifactError, match="float32"):
        writer.add(
            EncodedRepresentation(representation=outcome.representation, values=(1e300, 0.0, 0.0))
        )


def test_a_run_whose_outcomes_do_not_match_its_metrics_is_not_finalized(tmp_path: Path) -> None:
    source = LinearScanSource(line_of_points(11))
    encoder = FakeEncoder()
    service = RepresentationService(source, encoder, run_id=RUN_ID, code_version="test-version")
    writer = make_writer(tmp_path, encoder, source)
    stream = service.represent([ref(5), ref(0)])
    writer.add(next(stream))  # o run parou no meio: só um dos dois resultados chegou

    with pytest.raises(RunArtifactError, match="incomplete"):
        writer.finalize(metrics=service.metrics)

    assert not (tmp_path / "runs").exists()


def test_a_failed_support_of_another_map_is_rejected(tmp_path: Path) -> None:
    writer, _ = _writer_and_outcome(tmp_path)
    foreign = FailedSupport(
        support=make_support(0, (0, 1), policy=radius_policy(0.6), map_id=MapId("other-map")),
        reason=FailureReason.UNENCODABLE_SUPPORT,
        detail="x",
    )

    with pytest.raises(RunArtifactError, match="map"):
        writer.add(foreign)

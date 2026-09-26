import dataclasses
import json
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from projection_builders import SEQUENCE_ID
from run_builders import (
    ENHANCED,
    NATIVE,
    OCCLUSION,
    CollectingSink,
    frame_id,
    frame_input,
    make_request,
)

import contextmap.sensor_association as sensor_association
from contextmap.geometric_mapping import MapId, geometry_id_for
from contextmap.sensor_association import (
    CandidateGeometryPolicy,
    IncompleteRunArtifactError,
    RunArtifactError,
    SensorAssociationDebugLevel,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SpatialObservation,
    VisibilityDiagnostics,
)
from contextmap.sensor_association.service import (
    FrameAssociation,
    SensorAssociationOutcome,
    SensorAssociationRequest,
    SensorAssociationService,
)
from contextmap.visual_perception import RegionId

SEQUENCE_NAME = "corridor-fixture"
A, B = RegionId("region-A"), RegionId("region-B")


def _request(*channels: object) -> SensorAssociationRequest:
    return make_request(channels=list(channels))  # type: ignore[arg-type]


@dataclasses.dataclass(frozen=True)
class _Written:
    """One persisted run, plus the frames the sink saw while it was written."""

    reader: SensorAssociationRunReader
    run_dir: Path
    outcome: SensorAssociationOutcome
    frames: tuple[FrameAssociation, ...]

    @property
    def observations(self) -> list[SpatialObservation]:
        return [o for frame in self.frames for o in frame.observations]


def _writer(
    root: Path,
    *,
    run_index: int = 1,
    debug_level: SensorAssociationDebugLevel = SensorAssociationDebugLevel.NONE,
) -> SensorAssociationRunWriter:
    return SensorAssociationRunWriter(
        output_dir=root / f"run-{run_index:04d}",
        sequence_name=SEQUENCE_NAME,
        run_id=SensorAssociationRunId(f"assoc-run-{run_index:04d}"),
        run_index=run_index,
        debug_level=debug_level,
    )


def _write(
    root: Path,
    request: SensorAssociationRequest,
    *,
    run_index: int = 1,
    debug_level: SensorAssociationDebugLevel = SensorAssociationDebugLevel.NONE,
    runtime_s: float | None = None,
) -> _Written:
    """Grava em ``root/run-NNNN``: o chamador decide o diretório final, o writer não calcula."""
    run_dir = root / f"run-{run_index:04d}"
    sink = CollectingSink()
    with _writer(root, run_index=run_index, debug_level=debug_level).transaction() as run:
        sink = CollectingSink(run)
        outcome = SensorAssociationService().run(request, sink=sink)
        run.finalize(outcome, runtime_s=runtime_s)
    return _Written(
        reader=SensorAssociationRunReader(run_dir),
        run_dir=run_dir,
        outcome=outcome,
        frames=tuple(sink.frames),
    )


# --- Layout and lineage -----------------------------------------------------


def test_the_run_has_the_documented_layout(tmp_path: Path) -> None:
    run_dir = _write(tmp_path, _request(NATIVE, ENHANCED), runtime_s=1.5).run_dir

    for relative in (
        "README.md",
        "manifest.json",
        "outputs/spatial-observations.jsonl",
        "outputs/observation-index.jsonl",
        "outputs/geometry-support.u32",
        "outputs/observation-quality.jsonl",
        "outputs/projection-records.jsonl",
        "outputs/visibility-records.jsonl",
        "outputs/dense-feature-associations.jsonl",
        "outputs/dense-feature-cells.bin",
        "metrics/frame-diagnostics.jsonl",
        "metrics/summary.json",
        "metrics/runtime.json",
    ):
        assert (run_dir / relative).is_file(), relative
    assert not (run_dir / "debug").exists()


def test_a_run_without_dense_channels_writes_no_dense_files(tmp_path: Path) -> None:
    run_dir = _write(tmp_path, _request()).run_dir

    assert not (run_dir / "outputs/dense-feature-associations.jsonl").exists()
    assert not (run_dir / "outputs/dense-feature-cells.bin").exists()
    assert not (run_dir / "metrics/runtime.json").exists()


def test_the_manifest_names_the_upstream_artifacts_and_the_configuration(tmp_path: Path) -> None:
    written = _write(tmp_path, _request(NATIVE, ENHANCED))
    manifest = written.reader.manifest
    outcome = written.outcome

    assert manifest.run_id == "assoc-run-0001"
    assert manifest.sequence_artifact_id == SEQUENCE_ID
    assert manifest.selection_id == "full-sequence"
    assert manifest.geometric_map_id == MapId("map-0001")
    assert manifest.trajectory_id == outcome.trajectory_id
    assert manifest.perception_run_ids == ("run-0001",)
    assert manifest.calibration_identity == outcome.calibration_identity
    assert manifest.visibility_policy["policy_id"] == OCCLUSION.policy_id
    assert manifest.visibility_policy["fingerprint"] == OCCLUSION.fingerprint()
    assert manifest.membership_policy_id == "mask-membership-v1"
    assert manifest.configuration_fingerprint == outcome.configuration_fingerprint
    assert manifest.code_version == "test"
    assert (manifest.frame_count, manifest.rejected_frame_count) == (2, 0)
    assert manifest.observation_count == 4
    assert manifest.schema_version == "0.2.0"
    assert manifest.debug_level == "none"


def test_the_manifest_reveals_exactly_which_feature_maps_the_run_consumed(tmp_path: Path) -> None:
    reader = _write(tmp_path, _request(NATIVE, ENHANCED)).reader

    channels = {channel["channel_id"]: channel for channel in reader.manifest.dense_channels}
    native, enhanced = channels["dino-native"], channels["dino-enhanced"]
    assert native["interpolation"] == "nearest"
    assert enhanced["interpolation"] == "bilinear"
    (native_source,) = native["feature_sources"]
    (enhanced_source,) = enhanced["feature_sources"]
    assert native_source["source_artifact_id"] == "perception-artifact-0001"
    assert native_source["embedding_space_id"] == "dinov2:b14"
    assert native_source["enhancement"] is None
    assert enhanced_source["enhancement"] is not None
    assert native_source["sampling_fingerprint"] != enhanced_source["sampling_fingerprint"]
    assert native_source["frame_count"] == enhanced_source["frame_count"] == 2


def test_native_and_enhanced_runs_share_upstream_artifacts_yet_stay_identifiable(
    tmp_path: Path,
) -> None:
    native = _write(tmp_path, _request(NATIVE), run_index=1).reader
    enhanced = _write(tmp_path, _request(ENHANCED), run_index=2).reader

    assert native.manifest.run_id != enhanced.manifest.run_id
    assert native.manifest.configuration_fingerprint != enhanced.manifest.configuration_fingerprint
    assert native.manifest.dense_channels != enhanced.manifest.dense_channels
    for shared in (
        "geometric_map_id",
        "sequence_artifact_id",
        "calibration_identity",
        "trajectory_id",
    ):
        assert getattr(native.manifest, shared) == getattr(enhanced.manifest, shared)
    assert native.manifest.perception_run_ids == enhanced.manifest.perception_run_ids


# --- Reading back -----------------------------------------------------------


def test_the_spatial_observations_round_trip_with_their_geometry_support(tmp_path: Path) -> None:
    written = _write(tmp_path, _request(NATIVE))

    assert list(written.reader.observations()) == written.observations


def test_one_observation_is_read_by_identity_without_loading_the_others(tmp_path: Path) -> None:
    written = _write(tmp_path, _request())
    reader = written.reader
    wanted = written.frames[1].observations[1]

    assert reader.observation(wanted.spatial_observation_id) == wanted
    assert reader.observations_of_frame(frame_id(0)) == written.frames[0].observations
    with pytest.raises(RunArtifactError, match="unknown"):
        reader.observation(wanted.spatial_observation_id + "-missing")  # type: ignore[operator]


def test_the_region_geometry_index_is_a_compact_columnar_table(tmp_path: Path) -> None:
    written = _write(tmp_path, _request())
    reader, run_dir = written.reader, written.run_dir
    total = sum(len(o.geometry_support) for o in written.observations)

    assert (run_dir / "outputs/geometry-support.u32").stat().st_size == 4 * total
    observation = written.frames[0].observations[0]
    assert (
        reader.geometry_support(observation.spatial_observation_id) == observation.geometry_support
    )
    first = observation.geometry_support[0]
    assert first.geometry_id == geometry_id_for(map_id=MapId("map-0001"), index=0)


def test_an_observation_that_disagrees_with_its_membership_is_never_persisted(
    tmp_path: Path,
) -> None:
    class _Forging:
        """Hands the run a first frame whose observation contradicts its membership."""

        def __init__(self, inner: object) -> None:
            self._inner = inner
            self._first = True

        def accept(self, frame: FrameAssociation) -> None:
            if self._first:
                self._first = False
                tampered = dataclasses.replace(
                    frame.observations[0],
                    geometry_support=(),
                    visibility=VisibilityDiagnostics(counts={}),
                )
                frame = dataclasses.replace(frame, observations=(tampered, *frame.observations[1:]))
            self._inner.accept(frame)  # type: ignore[attr-defined]

    with (
        pytest.raises(RunArtifactError, match="geometry support"),
        _writer(tmp_path).transaction() as run,
    ):
        SensorAssociationService().run(_request(), sink=_Forging(run))

    assert list(tmp_path.iterdir()) == []


_BEYOND_U32 = 2**32


class _WideIndices:
    """Hands the run frames whose global indices from ``first_row`` on sit past 2**32.

    Only the identities move, and consistently: the observations are rebuilt from the shifted
    rows, so the one thing wrong with the frame is an index the ``<u4`` tables cannot hold. No
    map of that size is materialized.
    """

    def __init__(self, inner: object, *, first_row: int) -> None:
        self._inner = inner
        self._first_row = first_row

    def accept(self, frame: FrameAssociation) -> None:
        projection = frame.resolution.frame
        indices = projection.global_indices.copy()
        indices[self._first_row :] += _BEYOND_U32
        wide = dataclasses.replace(projection, global_indices=indices)
        observations = tuple(
            dataclasses.replace(
                observation,
                geometry_support=wide.map_references(np.asarray(region.associated_indices)),
            )
            for region, observation in zip(
                frame.membership.regions, frame.observations, strict=True
            )
        )
        frame = dataclasses.replace(
            frame,
            resolution=dataclasses.replace(frame.resolution, frame=wide),
            observations=observations,
        )
        self._inner.accept(frame)  # type: ignore[attr-defined]


def test_a_support_index_past_32_bits_is_refused_instead_of_wrapped(tmp_path: Path) -> None:
    # A linha 3 é suporte da região B: o cast silencioso a gravaria como o índice 3.
    with (
        pytest.raises(RunArtifactError, match="32-bit outputs/geometry-support"),
        _writer(tmp_path).transaction() as run,
    ):
        SensorAssociationService().run(_request(), sink=_WideIndices(run, first_row=3))

    assert list(tmp_path.iterdir()) == []


def test_a_dense_eligible_index_past_32_bits_is_refused_instead_of_wrapped(
    tmp_path: Path,
) -> None:
    # A linha 4 é visível e elegível, mas não está em região nenhuma: só o caminho denso a grava.
    with (
        pytest.raises(RunArtifactError, match="32-bit outputs/dense-feature-cells"),
        _writer(tmp_path).transaction() as run,
    ):
        SensorAssociationService().run(_request(NATIVE), sink=_WideIndices(run, first_row=4))

    assert list(tmp_path.iterdir()) == []


def test_the_geometry_to_region_index_keeps_every_overlapping_region(tmp_path: Path) -> None:
    written = _write(tmp_path, _request())
    reader = written.reader
    frame = written.frames[0]

    assert reader.regions_of(frame_id(0), frame.resolution.frame.map_reference(1)) == (A, B)
    assert reader.regions_of(frame_id(0), frame.resolution.frame.map_reference(0)) == (A,)
    assert reader.regions_of(frame_id(0), frame.resolution.frame.map_reference(4)) == ()


def test_the_quality_is_read_back_by_observation(tmp_path: Path) -> None:
    written = _write(tmp_path, _request())
    first = written.frames[0]

    quality = written.reader.quality(first.observations[0].spatial_observation_id)

    assert quality == first.qualities[0]


def test_the_dense_associations_round_trip_as_indices_and_weights(tmp_path: Path) -> None:
    written = _write(tmp_path, _request(NATIVE, ENHANCED))
    reader, run_dir = written.reader, written.run_dir
    global_indices = written.frames[0].resolution.frame.global_indices

    for channel_id in ("dino-native", "dino-enhanced"):
        samples = written.frames[0].dense_samples[channel_id]
        record = reader.dense_association(frame_id(0), channel_id)
        assert record.channel_id == channel_id
        assert list(record.eligible_indices) == global_indices[samples.eligible_indices].tolist()
        assert list(record.sampled) == samples.sampled.tolist()
        assert list(record.cell_rows) == samples.cell_rows.ravel().tolist()
        assert list(record.cell_cols) == samples.cell_cols.ravel().tolist()
        assert list(record.weights) == pytest.approx(samples.weights.ravel().tolist())
        assert record.terms == samples.cell_rows.shape[1]
        assert record.provenance["feature_id"] == str(samples.provenance.feature_id)
        assert record.provenance["payload_reference"] == samples.provenance.payload_reference
    native = reader.dense_association(frame_id(0), "dino-native")
    enhanced = reader.dense_association(frame_id(0), "dino-enhanced")
    assert native.provenance["enhancement"] is None
    assert enhanced.provenance["enhancement"]["source_feature_id"] == "dense-native-0"
    # Nenhum vetor de feature é persistido: só índices e pesos.
    assert not list((run_dir / "outputs").glob("*.npy"))


def test_the_frame_records_keep_the_projection_visibility_and_diagnostics(tmp_path: Path) -> None:
    outcome = _request(NATIVE)
    reader = _write(tmp_path, outcome).reader

    projection = reader.read_records("outputs/projection-records.jsonl")
    visibility = reader.read_records("outputs/visibility-records.jsonl")
    diagnostics = reader.read_records("metrics/frame-diagnostics.jsonl")

    assert [r["source_observation_id"] for r in projection] == [frame_id(0), frame_id(1)]
    assert projection[0]["image_transform"]["transform_id"].startswith("sha256:")
    assert projection[0]["pose_ref"]["lookup_outcome"] == "exact"
    assert projection[0]["calibration_ref"]["camera_model_kind"] == "pinhole"
    assert visibility[0]["state_counts"]["occluded"] == 1
    assert visibility[0]["membership"]["associated_count"] == 3
    assert diagnostics[0]["definitions_version"] == "association-diagnostics-v3"
    assert diagnostics[0]["findings"] == []


def test_the_summary_aggregates_the_run_and_lists_the_rejected_frames(tmp_path: Path) -> None:
    request = make_request(frames=[frame_input(0), frame_input(1, time_ns=10_000_000_000)])
    reader = _write(tmp_path, request).reader

    summary = reader.read_record("metrics/summary.json")
    assert summary["frame_count"] == 1
    assert summary["rejected_frames"] == [
        {"source_observation_id": frame_id(1), "rejection": "out_of_range"}
    ]
    assert reader.manifest.rejected_frame_count == 1
    assert summary["observation_count"] == 2
    assert summary["state_counts"]["occluded"] == 1


def test_the_summary_totals_the_range_limit_diagnostics_over_the_frames(tmp_path: Path) -> None:
    # Com 3,5 m o ponto de fundo (alcance 8,7 m) sai; as três associadas, a z = 3 m, ficam além
    # do piso 3,5 * cos(theta_max), cerca de 2,73 m, nos dois frames.
    written = _write(tmp_path, make_request(candidates=CandidateGeometryPolicy(max_range_m=3.5)))
    frames = [frame.diagnostics for frame in written.frames]

    summary = written.reader.read_record("metrics/summary.json")

    assert [d.range_limit_candidate_support_count for d in frames] == [3, 3]
    assert summary["range_limit_candidate_support_count"] == 6
    slacks = [d.min_range_slack_m for d in frames]
    assert all(slack is not None for slack in slacks)
    assert summary["min_range_slack_m"] == pytest.approx(min(s for s in slacks if s is not None))


def test_without_a_range_limit_the_summary_says_the_diagnostics_do_not_apply(
    tmp_path: Path,
) -> None:
    summary = _write(tmp_path, _request()).reader.read_record("metrics/summary.json")

    assert summary["range_limit_candidate_support_count"] is None
    assert summary["min_range_slack_m"] is None


# --- Immutability, atomicity and integrity ----------------------------------


def test_a_finalized_run_is_never_overwritten(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    with writer.transaction() as run:
        run.finalize(SensorAssociationService().run(_request(), sink=run))
    before = (tmp_path / "run-0001" / "manifest.json").read_bytes()

    with pytest.raises(RunArtifactError, match="finalized"), writer.transaction():
        pass
    with pytest.raises(RunArtifactError, match="exists"), _writer(tmp_path).transaction():
        pass

    assert (tmp_path / "run-0001" / "manifest.json").read_bytes() == before


def test_the_run_is_written_exactly_where_the_caller_says_and_nothing_else_is_created(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ws" / "corridor-02" / "run-0001" / "sensor_association"

    writer = SensorAssociationRunWriter(
        output_dir=target,
        sequence_name=SEQUENCE_NAME,
        run_id=SensorAssociationRunId("association-run"),
        run_index=1,
    )
    with writer.transaction() as run:
        run.finalize(SensorAssociationService().run(_request(), sink=run))

    manifest = SensorAssociationRunReader(target).manifest
    assert manifest.run_id == SensorAssociationRunId("association-run")
    # Sem registro `runs.json` e sem `runs/<capability>/<sequência>/`: só o diretório do artifact.
    assert sorted(path.name for path in target.parent.iterdir()) == ["sensor_association"]
    assert sorted(path.name for path in (tmp_path / "ws").iterdir()) == ["corridor-02"]


def test_the_run_id_and_index_are_recorded_as_supplied_and_never_allocated(
    tmp_path: Path,
) -> None:
    reader = _write(tmp_path, _request(), run_index=7).reader

    manifest = reader.manifest
    assert (manifest.run_id, manifest.run_index) == (SensorAssociationRunId("assoc-run-0007"), 7)
    assert not (tmp_path / "run-0001").exists()


def test_an_interrupted_write_never_looks_like_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "contextmap.sensor_association.run_artifact.encode_observation_quality", explode
    )
    with pytest.raises(RuntimeError, match="disk full"), _writer(tmp_path).transaction() as run:
        SensorAssociationService().run(_request(), sink=run)

    assert list(tmp_path.iterdir()) == []


def test_integrity_detects_a_missing_a_resized_and_a_corrupted_file(tmp_path: Path) -> None:
    written = _write(tmp_path, _request(NATIVE))
    reader, run_dir = written.reader, written.run_dir
    assert reader.verify_integrity() == []

    support = run_dir / "outputs/geometry-support.u32"
    original = support.read_bytes()
    support.write_bytes(original[:-4])
    assert any("size mismatch" in p for p in reader.verify_integrity())
    support.write_bytes(b"\x00" * len(original))
    assert any("hash mismatch" in p for p in reader.verify_integrity())
    support.unlink()
    assert any("missing file" in p for p in reader.verify_integrity())


def test_an_unknown_schema_or_a_missing_manifest_is_refused(tmp_path: Path) -> None:
    run_dir = _write(tmp_path, _request()).run_dir
    manifest_path = run_dir / "manifest.json"
    record = json.loads(manifest_path.read_text())
    record["schema_version"] = "9.9.9"
    manifest_path.write_text(json.dumps(record))

    with pytest.raises(RunArtifactError, match="schema"):
        SensorAssociationRunReader(run_dir)
    manifest_path.unlink()
    with pytest.raises(IncompleteRunArtifactError):
        SensorAssociationRunReader(run_dir)


def test_a_downstream_stage_can_only_read_contractual_records(tmp_path: Path) -> None:
    reader = _write(tmp_path, _request(), debug_level=SensorAssociationDebugLevel.STANDARD).reader

    with pytest.raises(RunArtifactError, match="contractual"):
        reader.read_record("debug/frames/frame-0000/distributions.json")
    with pytest.raises(RunArtifactError, match="contractual"):
        reader.read_records("debug/frames/frame-0000/samples.csv")
    with pytest.raises(RunArtifactError, match="contractual"):
        reader.read_record("manifest.json")


# --- Debug evidence ---------------------------------------------------------


def test_standard_debug_explains_the_samples_and_their_distributions(tmp_path: Path) -> None:
    written = _write(tmp_path, _request(), debug_level=SensorAssociationDebugLevel.STANDARD)
    reader, run_dir = written.reader, written.run_dir
    frame_dir = run_dir / "debug" / "frames" / frame_id(0)

    lines = (frame_dir / "samples.csv").read_text().splitlines()
    assert lines[0] == "geometry_index,state,prepared_u,prepared_v,depth_m,support_depth_m,regions"
    rows = {line.split(",")[0]: line.split(",") for line in lines[1:]}
    assert rows["1"][1] == "associated" and rows["1"][6] == "region-A;region-B"
    assert rows["2"][1] == "occluded"
    assert rows["4"][1] == "visible_unassigned" and rows["4"][6] == ""
    distributions = json.loads((frame_dir / "distributions.json").read_text())
    assert distributions["state_counts"]["occluded"] == 1
    assert distributions["associated_depth_m"]["count"] == 3
    assert sum(distributions["associated_depth_histogram"]["counts"]) == 3
    assert sum(sum(row) for row in distributions["associated_by_image_region"]) == 3
    assert {r["region_id"] for r in distributions["support_density_by_region"]} == {
        "region-A",
        "region-B",
    }
    assert not (frame_dir / "overlay.png").exists()
    assert reader.manifest.debug_level == "standard"


def test_full_debug_adds_the_overlay_the_sampling_coordinates_and_the_feature_sources(
    tmp_path: Path,
) -> None:
    run_dir = _write(
        tmp_path, _request(NATIVE, ENHANCED), debug_level=SensorAssociationDebugLevel.FULL
    ).run_dir
    frame_dir = run_dir / "debug" / "frames" / frame_id(0)

    width, height, pixels = _decode_png((frame_dir / "overlay.png").read_bytes())
    assert (width, height) == (640, 480)
    assert pixels[100][100] == (40, 180, 60)  # associado
    assert pixels[300][400] == (230, 200, 40)  # visível, sem região
    native_csv = (frame_dir / "dense-sampling-dino-native.csv").read_text().splitlines()
    assert native_csv[0] == "geometry_index,sampled,cell0_row,cell0_col,cell0_weight"
    enhanced_csv = (frame_dir / "dense-sampling-dino-enhanced.csv").read_text().splitlines()
    assert enhanced_csv[0].endswith("cell3_weight")
    sources = json.loads((run_dir / "debug" / "feature-sources.json").read_text())
    assert {entry["channel_id"] for entry in sources} == {"dino-native", "dino-enhanced"}
    assert {entry["source_observation_id"] for entry in sources} == {frame_id(0), frame_id(1)}


def test_debug_files_are_never_inventoried_so_removing_them_keeps_the_run_valid(
    tmp_path: Path,
) -> None:
    written = _write(tmp_path, _request(NATIVE), debug_level=SensorAssociationDebugLevel.FULL)
    reader, run_dir = written.reader, written.run_dir

    assert all(not entry.path.startswith("debug/") for entry in reader.manifest.file_inventory)
    assert all(
        entry.path not in ("manifest.json", "README.md") for entry in reader.manifest.file_inventory
    )
    shutil.rmtree(run_dir / "debug")
    assert reader.verify_integrity() == []
    assert list(reader.observations())


def test_no_debug_writes_nothing_beyond_the_contractual_files(tmp_path: Path) -> None:
    run_dir = _write(tmp_path, _request(NATIVE)).run_dir

    assert not (run_dir / "debug").exists()


# --- Independence of heavy dependencies -------------------------------------


def test_the_artifact_opens_without_numpy_ros_or_model_libraries(tmp_path: Path) -> None:
    run_dir = _write(tmp_path, _request(NATIVE, ENHANCED)).run_dir
    code = (
        "import sys;"
        "from contextmap.sensor_association import SensorAssociationRunReader;"
        f"r = SensorAssociationRunReader(__import__('pathlib').Path({str(run_dir)!r}));"
        "assert len(list(r.observations())) == 4;"
        "assert r.dense_association('frame-0000', 'dino-native').terms == 1;"
        "assert r.verify_integrity() == [];"
        "bad = [m for m in ('numpy', 'rosbags', 'torch', 'open3d') if m in sys.modules];"
        "assert not bad, bad"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


def test_the_run_is_reachable_from_the_public_api() -> None:
    public = set(sensor_association.__all__)

    assert {
        "SensorAssociationRunReader",
        "SensorAssociationRunWriter",
        "SensorAssociationRunManifest",
        "SensorAssociationRunId",
        "SensorAssociationDebugLevel",
        "SensorAssociationService",
        "SensorAssociationRequest",
        "SensorAssociationOutcome",
    } <= public
    assert not {"allocate_run_index", "rebuild_run_registry"} & public


def _decode_png(data: bytes) -> tuple[int, int, list[list[tuple[int, int, int]]]]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    cursor = 8
    width = height = 0
    idat = b""
    while cursor < len(data):
        (length,) = struct.unpack(">I", data[cursor : cursor + 4])
        kind = data[cursor + 4 : cursor + 8]
        body = data[cursor + 8 : cursor + 8 + length]
        if kind == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif kind == b"IDAT":
            idat += body
        cursor += 12 + length
    raw = zlib.decompress(idat)
    stride = 1 + 3 * width
    rows = []
    for y in range(height):
        line = raw[y * stride + 1 : (y + 1) * stride]
        rows.append([(line[3 * x], line[3 * x + 1], line[3 * x + 2]) for x in range(width)])
    return width, height, rows

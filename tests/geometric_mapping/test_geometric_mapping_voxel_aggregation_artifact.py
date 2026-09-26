import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from voxel_builders import unit_policy, write_raw_artifact

from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    GeometryBlockSource,
    MapArtifactError,
    MapId,
    VoxelAggregationArtifactReader,
    VoxelAggregationArtifactWriter,
    VoxelAggregationRunId,
    aggregate_geometry,
)
from contextmap.shared import Vector3

REVISIT: tuple[tuple[Vector3, ...], ...] = (
    ((0.25, 0.25, 0.25), (0.75, 0.50, 0.50), (0.50, 0.75, 0.25), (1.50, 0.50, 0.50)),
    ((0.40, 0.40, 0.40),),
    ((0.60, 0.20, 0.80),),
)
ELSEWHERE: tuple[tuple[Vector3, ...], ...] = (
    ((5.25, 0.25, 0.25), (5.75, 0.50, 0.50), (5.50, 0.75, 0.25), (6.50, 0.50, 0.50)),
    ((5.40, 0.40, 0.40),),
    ((5.60, 0.20, 0.80),),
)
RUN_ID = VoxelAggregationRunId("voxel-run-0001")


def _tree_digest(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _derive(tmp_path: Path, *, raw: Path | None = None, block_points: int = 1_000_000) -> Path:
    source = raw if raw is not None else write_raw_artifact(tmp_path / "raw", REVISIT)
    output = tmp_path / "derived"
    VoxelAggregationArtifactWriter(output_dir=output, run_id=RUN_ID).finalize(
        source_dir=source, policy=unit_policy(), code_version="test", block_points=block_points
    )
    return output


# --- The raw artifact stays the canonical evidence -----------------------------------


def test_deriving_an_aggregation_leaves_every_byte_of_the_raw_artifact_unchanged(
    tmp_path: Path,
) -> None:
    raw = write_raw_artifact(tmp_path / "raw", REVISIT)
    before = _tree_digest(raw)

    _derive(tmp_path, raw=raw)

    assert _tree_digest(raw) == before
    with GeometricMapArtifactReader(raw) as reader:
        assert reader.verify_integrity() == []


def test_the_derived_artifact_is_never_written_inside_the_raw_artifact(tmp_path: Path) -> None:
    raw = write_raw_artifact(tmp_path / "raw", REVISIT)

    with pytest.raises(MapArtifactError, match="inside"):
        VoxelAggregationArtifactWriter(output_dir=raw / "voxel", run_id=RUN_ID).finalize(
            source_dir=raw, policy=unit_policy(), code_version="test"
        )


def test_an_existing_derived_artifact_is_never_replaced(tmp_path: Path) -> None:
    raw = write_raw_artifact(tmp_path / "raw", REVISIT)
    output = _derive(tmp_path, raw=raw)

    with pytest.raises(MapArtifactError, match="exists"):
        VoxelAggregationArtifactWriter(output_dir=output, run_id=RUN_ID).finalize(
            source_dir=raw, policy=unit_policy(), code_version="test"
        )


# --- What the derived artifact records ------------------------------------------------


def test_reading_the_artifact_back_reproduces_the_aggregation_exactly(tmp_path: Path) -> None:
    output = _derive(tmp_path)

    with GeometricMapArtifactReader(tmp_path / "raw") as raw:
        expected = aggregate_geometry(raw.geometry(), unit_policy())
    read = VoxelAggregationArtifactReader(output).aggregation()

    assert read.equivalence_problems(expected) == []
    for name in ("keys", "centroids_m", "offset_sums_m", "point_counts", "scan_counts"):
        assert np.array_equal(getattr(read, name), getattr(expected, name)), name


def test_the_manifest_records_the_grid_the_fingerprint_and_separate_counts(
    tmp_path: Path,
) -> None:
    manifest = VoxelAggregationArtifactReader(_derive(tmp_path)).manifest

    assert manifest.policy_fingerprint == unit_policy().fingerprint()
    assert manifest.aggregation_rule == "inter-scan-voxel-centroid-1.0m"
    assert manifest.grid == unit_policy().grid.to_record()
    assert manifest.aggregate_count == 2
    assert manifest.source_point_count == 6
    assert manifest.contribution_count == 4
    assert manifest.scan_count == 3
    assert manifest.observation_count == 3
    assert manifest.derived_map_id == MapId(f"{manifest.source_map_id}--{RUN_ID}")


def test_the_lineage_names_the_exact_raw_artifact_it_was_derived_from(tmp_path: Path) -> None:
    raw = write_raw_artifact(tmp_path / "raw", REVISIT)
    reader = VoxelAggregationArtifactReader(_derive(tmp_path, raw=raw))

    lineage = reader.read_record("lineage.json")

    with GeometricMapArtifactReader(raw) as source:
        geometry_entry = next(
            entry
            for entry in source.manifest.file_inventory
            if entry.path == "outputs/geometry.bin"
        )
        assert lineage["source"] == {
            "run_id": str(source.manifest.run_id),
            "map_id": str(source.manifest.map_id),
            "schema_version": source.manifest.schema_version,
            "configuration_fingerprint": source.manifest.configuration_fingerprint,
            "geometry_content_hash": geometry_entry.content_hash,
            "aggregation_rule": None,
            "sequence_artifact_id": str(source.manifest.sequence_artifact_id),
            "selection_id": source.manifest.selection_id,
            "trajectory_id": str(source.manifest.trajectory_id),
            "calibration_identity": source.manifest.calibration_identity,
        }


def test_the_chunk_size_is_recorded_but_is_not_part_of_the_policy_identity(
    tmp_path: Path,
) -> None:
    raw = write_raw_artifact(tmp_path / "raw", REVISIT)
    fine = VoxelAggregationArtifactReader(_derive(tmp_path / "a", raw=raw, block_points=1))
    coarse = VoxelAggregationArtifactReader(_derive(tmp_path / "b", raw=raw, block_points=64))

    assert fine.read_record("config.json")["execution"] == {"block_points": 1}
    assert fine.manifest.policy_fingerprint == coarse.manifest.policy_fingerprint
    assert fine.aggregation().equivalence_problems(coarse.aggregation()) == []


def test_the_metrics_report_scale_and_the_cost_of_the_lineage(tmp_path: Path) -> None:
    metrics = VoxelAggregationArtifactReader(_derive(tmp_path)).read_record(
        "metrics/aggregation.json"
    )

    assert metrics["source_point_count"] == 6
    assert metrics["aggregate_count"] == 2
    assert metrics["reduction_ratio"] == pytest.approx(2 / 6)
    assert metrics["aggregate_bytes"] == 2 * 148
    assert metrics["lineage_bytes"] == 4 * 8
    assert metrics["lineage_bytes_per_aggregate"] == pytest.approx(16.0)
    assert metrics["explicit_point_lineage_bytes"] == 6 * 8
    assert metrics["points_per_aggregate"]["maximum"] == 5
    assert metrics["scans_per_aggregate"]["maximum"] == 3


def test_the_packed_records_follow_the_documented_layout(tmp_path: Path) -> None:
    from contextmap.geometric_mapping.voxel_aggregation_artifact import (
        AGGREGATE_DTYPE_FIELDS,
        AGGREGATE_RECORD,
        CONTRIBUTION_DTYPE_FIELDS,
        CONTRIBUTION_RECORD,
    )

    output = _derive(tmp_path)

    assert np.dtype(AGGREGATE_DTYPE_FIELDS).itemsize == AGGREGATE_RECORD.size == 148
    assert np.dtype(CONTRIBUTION_DTYPE_FIELDS).itemsize == CONTRIBUTION_RECORD.size == 8
    first = AGGREGATE_RECORD.unpack_from((output / "outputs/aggregates.bin").read_bytes(), 0)
    # chave | centroide | soma dos offsets | mínimo | máximo | primeiro/último ns | contagens.
    assert first[0:3] == (0, 0, 0)
    assert first[-3:] == (5, 3, 3)
    assert CONTRIBUTION_RECORD.unpack_from(
        (output / "outputs/contributions.bin").read_bytes(), 0
    ) == (0, 3)


# --- Reading and auditing -------------------------------------------------------------


def test_the_artifact_opens_as_the_block_source_of_the_derived_map(tmp_path: Path) -> None:
    reader = VoxelAggregationArtifactReader(_derive(tmp_path))

    geometry = reader.geometry()

    assert isinstance(geometry, GeometryBlockSource)
    assert geometry.geometric_map.map_id == reader.manifest.derived_map_id
    assert geometry.geometric_map.aggregation_rule == reader.manifest.aggregation_rule
    assert sum(len(block) for block in geometry.iter_blocks()) == 2


def test_a_finished_artifact_verifies_its_inventory_and_its_lineage(tmp_path: Path) -> None:
    raw = write_raw_artifact(tmp_path / "raw", REVISIT)
    reader = VoxelAggregationArtifactReader(_derive(tmp_path, raw=raw))

    with GeometricMapArtifactReader(raw) as source:
        assert reader.verify_integrity(source=source) == []
    assert reader.verify_integrity() == []


def test_lineage_verification_refuses_another_raw_artifact(tmp_path: Path) -> None:
    reader = VoxelAggregationArtifactReader(_derive(tmp_path))
    other = write_raw_artifact(tmp_path / "other", ELSEWHERE)

    with GeometricMapArtifactReader(other) as source:
        problems = reader.verify_integrity(source=source)

    assert any("geometry" in problem for problem in problems)


def test_a_tampered_aggregate_payload_is_detected(tmp_path: Path) -> None:
    output = _derive(tmp_path)
    payload = output / "outputs/aggregates.bin"
    data = bytearray(payload.read_bytes())
    data[30] ^= 0xFF
    payload.write_bytes(bytes(data))

    problems = VoxelAggregationArtifactReader(output).verify_integrity()

    assert any("aggregates.bin" in problem for problem in problems)


def test_a_missing_manifest_is_an_incomplete_artifact(tmp_path: Path) -> None:
    output = _derive(tmp_path)
    (output / "manifest.json").unlink()

    with pytest.raises(MapArtifactError, match="manifest"):
        VoxelAggregationArtifactReader(output)


def test_an_unknown_grid_indexing_is_refused(tmp_path: Path) -> None:
    output = _derive(tmp_path)
    config = output / "config.json"
    record = json.loads(config.read_text(encoding="utf-8"))
    record["policy"]["grid"]["indexing"] = "round-nearest"
    config.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(MapArtifactError, match="indexing"):
        VoxelAggregationArtifactReader(output).aggregation()


def test_only_contractual_json_records_can_be_read(tmp_path: Path) -> None:
    reader = VoxelAggregationArtifactReader(_derive(tmp_path))

    with pytest.raises(MapArtifactError, match="contractual"):
        reader.read_record("outputs/aggregates.bin")

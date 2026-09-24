"""``ContextMapExecutor`` must not accept a ``sequence`` input that is not the one the geometric
map it opens was actually built over.

Before this fix, ``ContextMapExecutor.execute()`` took the ``artifact_id`` of the sequence
lineage entry from the geometry manifest (``geometry_manifest.sequence_artifact_id``) but
computed ``content_identity`` from whatever directory the caller passed as the ``sequence``
input, without ever opening it to check the two agree -- an artifact/config bug (a stale
selection, a caller wiring the wrong run) could pin a real, verifiable digest to the wrong
identity, and the writer's own structural-dependency check never catches it because
``ArtifactKind.SEQUENCE`` is not structural.

This test reuses the real fixture builders the serialization tests already use
(``context_map_serialization_geometry``/``context_map_serialization_upstream``), so the
geometry/entity-resolution/spatial-relations chain is a real, self-consistent one -- only the
``sequence`` input is deliberately swapped for a mismatched one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from context_map_serialization_geometry import build_geometry_artifact
from context_map_serialization_upstream import write_relations_run, write_resolution_run

from contextmap.artifact import ContextMapArtifactReader, artifact_digest
from contextmap.ingestion import (
    FrameId,
    ImuObservation,
    SensorId,
    SequenceArtifactId,
    SequenceArtifactWriter,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.ingestion.sequence_provenance import SequenceProvenance
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import ContextMapExecutor, ExecutorError
from contextmap.shared import SourceTimestamp
from contextmap.spatial_relations import AxisDirection

_CLOCK = "fixture:header"


def _write_sequence(directory: Path, *, artifact_id: str) -> None:
    with SequenceArtifactWriter(
        output_dir=directory, sequence_name="test", artifact_id=SequenceArtifactId(artifact_id)
    ) as writer:
        writer.add_observation(
            ImuObservation(
                observation_id=SourceObservationId("imu-0000"),
                sensor_id=SensorId("imu"),
                frame_id=FrameId("imu"),
                timestamp=SourceTimestamp(seconds=0, nanoseconds=0, clock_id=_CLOCK),
                provenance=SourceProvenance(source_type="fixture", source_path="fixtures/imu"),
            )
        )
        writer.set_provenance(SequenceProvenance(source_type="fixture", source_path="fixtures/imu"))
        writer.finalize()


def _chain(workspace: Path) -> tuple[Path, Path, Path]:
    """Build a real, self-consistent geometry/resolution/relations chain under ``workspace``."""
    geometry_dir, geometry_manifest = build_geometry_artifact(workspace / "geometric_mapping")
    resolution_dir = workspace / "entity_resolution"
    write_resolution_run(
        resolution_dir,
        run_id="res-0001",
        geometric_map_id=str(geometry_manifest.map_id),
        semantic_map_id="semantic-map-0001",
    )
    relations_dir = workspace / "spatial_relations"
    write_relations_run(relations_dir, run_id="rel-0001", resolution_dir=resolution_dir)
    return geometry_dir, resolution_dir, relations_dir


def _request(workspace: Path, *, sequence_dir: Path, output_dir: Path) -> StageRequest:
    geometry_dir, resolution_dir, relations_dir = _chain(workspace)

    def _ref(stage_id: str, artifact_id: str, location: Path) -> tuple[ArtifactRef, ...]:
        return (
            ArtifactRef(
                stage_id=stage_id,
                contract="test",
                artifact_id=artifact_id,
                content_hash=f"sha256:{artifact_id}",
                location=location.relative_to(workspace).as_posix(),
            ),
        )

    return StageRequest(
        stage_id="context_map",
        inputs={
            "sequence": _ref("ingestion", "sequence", sequence_dir),
            "geometry": _ref("geometric_mapping", "geometry", geometry_dir),
            "entities": _ref("entity_resolution", "entities", resolution_dir),
            "relations": _ref("spatial_relations", "relations", relations_dir),
        },
        components={},
        config_digest="test",
        output_dir=output_dir,
        workspace=workspace,
    )


def test_a_sequence_that_does_not_match_the_geometry_s_own_lineage_is_refused(
    tmp_path: Path,
) -> None:
    wrong_sequence_dir = tmp_path / "ingestion-wrong"
    # The real geometry fixture always claims "sequence-0001" (see `_chain`).
    _write_sequence(wrong_sequence_dir, artifact_id="sequence-9999")

    request = _request(
        tmp_path,
        sequence_dir=wrong_sequence_dir,
        output_dir=tmp_path / "context_map",
    )

    with pytest.raises(ExecutorError, match="sequence"):
        ContextMapExecutor().execute(request)


def test_the_matching_sequence_is_accepted_and_pinned_by_its_own_digest(tmp_path: Path) -> None:
    correct_sequence_dir = tmp_path / "ingestion-correct"
    _write_sequence(correct_sequence_dir, artifact_id="sequence-0001")  # matches the geometry

    request = _request(
        tmp_path,
        sequence_dir=correct_sequence_dir,
        output_dir=tmp_path / "context_map",
    )

    ref = ContextMapExecutor().execute(request)

    with ContextMapArtifactReader.open(tmp_path / "context_map") as reader:
        sequence_entry = next(
            item for item in reader.context_map().lineage if item.artifact_id == "sequence-0001"
        )
    assert sequence_entry.content_identity == artifact_digest(correct_sequence_dir)
    assert ref.contract == "ContextMapArtifact"


def test_the_manifest_records_a_real_configuration_fingerprint_not_none(tmp_path: Path) -> None:
    """PR #438 review: ``runtime.provenance_identity`` requires every stage artifact to record
    an effective configuration digest; ``ContextMapExecutor`` used to hardcode ``None``."""
    sequence_dir = tmp_path / "ingestion-correct"
    _write_sequence(sequence_dir, artifact_id="sequence-0001")
    request = _request(tmp_path, sequence_dir=sequence_dir, output_dir=tmp_path / "context_map")

    ContextMapExecutor(up_axis=AxisDirection.POSITIVE_Z).execute(request)

    with ContextMapArtifactReader.open(tmp_path / "context_map") as reader:
        fingerprint = reader.manifest.configuration_fingerprint
    assert fingerprint is not None
    assert fingerprint.startswith("sha256:")


def test_a_different_up_axis_changes_the_configuration_fingerprint(tmp_path: Path) -> None:
    # Two independent workspaces: `_request`/`_chain` write a fresh geometry/resolution/
    # relations chain per workspace, and an immutable artifact is never overwritten.
    workspace_z, workspace_x = tmp_path / "z", tmp_path / "x"
    sequence_dir_z, sequence_dir_x = workspace_z / "ingestion", workspace_x / "ingestion"
    _write_sequence(sequence_dir_z, artifact_id="sequence-0001")
    _write_sequence(sequence_dir_x, artifact_id="sequence-0001")

    request_z = _request(
        workspace_z, sequence_dir=sequence_dir_z, output_dir=workspace_z / "context_map"
    )
    ContextMapExecutor(up_axis=AxisDirection.POSITIVE_Z).execute(request_z)
    request_x = _request(
        workspace_x, sequence_dir=sequence_dir_x, output_dir=workspace_x / "context_map"
    )
    ContextMapExecutor(up_axis=AxisDirection.POSITIVE_X).execute(request_x)

    with ContextMapArtifactReader.open(workspace_z / "context_map") as reader_z:
        fingerprint_z = reader_z.manifest.configuration_fingerprint
    with ContextMapArtifactReader.open(workspace_x / "context_map") as reader_x:
        fingerprint_x = reader_x.manifest.configuration_fingerprint
    assert fingerprint_z != fingerprint_x

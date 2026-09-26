"""The context stages over an observation selection of the sequence (issue #497).

Visual Perception applies ``inputs.observation_selection``; Sensor Association inherits it from
the perception run it consumes, so its lineage names the selection actually processed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime_foundation import foundation_refs
from runtime_perception import stub_perception_executor

from contextmap.ingestion import (
    FrameId,
    FrameRangeSelection,
    FullSequenceSelection,
    ImageEncoding,
    ImageObservation,
    SensorId,
    SequenceArtifactId,
    SequenceArtifactWriter,
    SourceObservationId,
    SourceProvenance,
    selection_identity,
)
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import (
    ExecutorError,
    SensorAssociationExecutor,
    inventory_digest,
)
from contextmap.sensor_association import (
    CandidateGeometryPolicy,
    DiagnosticTolerances,
    OcclusionPolicy,
    SensorAssociationRunReader,
)
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import LookupPolicy
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
)

SEQUENCE_ID = "sequence-0001"


def _sequence(workspace: Path, frames: int = 4) -> ArtifactRef:
    output = workspace / "S1" / "run-0001" / "ingestion"
    with SequenceArtifactWriter(
        output_dir=output, sequence_name="S1", artifact_id=SequenceArtifactId(SEQUENCE_ID)
    ) as writer:
        for index in range(frames):
            writer.add_observation(
                ImageObservation(
                    observation_id=SourceObservationId(f"frame-{index:04d}"),
                    sensor_id=SensorId("camera_1"),
                    frame_id=FrameId("camera_1_optical"),
                    timestamp=SourceTimestamp(seconds=index, nanoseconds=0, clock_id="fixture"),
                    provenance=SourceProvenance(source_type="fixture", source_path="images"),
                    width=2,
                    height=2,
                    encoding=ImageEncoding.BGR8,
                    data=bytes([10, 20, 30] * 4),
                )
            )
        manifest = writer.finalize()
    return ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id=SEQUENCE_ID,
        content_hash=inventory_digest(manifest.file_inventory),
        location="S1/run-0001/ingestion",
    )


def _perceive(workspace: Path, selection: dict[str, Any] | None) -> Path:
    output = workspace / "S1" / "run-0001" / "visual_perception"
    stub_perception_executor().execute(
        StageRequest(
            stage_id="visual_perception",
            inputs={"sequence": (_sequence(workspace),)},
            components={},
            config_digest="sha256:test",
            output_dir=output,
            workspace=workspace,
            observation_selection=selection,
        )
    )
    return output


class TestVisualPerceptionOverASelection:
    @pytest.mark.parametrize(
        "selection",
        [
            {"kind": "frame_range", "start_frame_index": 0},
            {"kind": "no_such_kind"},
            # Campos de outra seleção, como uma fusão de camadas de configuração produziria.
            {
                "kind": "frame_range",
                "start_frame_index": 0,
                "end_frame_index": 1,
                "observation_ids": ["frame-0000"],
            },
            {"kind": "explicit_ids", "observation_ids": ["no-such-frame"]},
        ],
    )
    def test_an_invalid_selection_fails_before_anything_is_perceived(
        self, tmp_path: Path, selection: dict[str, Any]
    ) -> None:
        with pytest.raises(ExecutorError, match="observation_selection"):
            _perceive(tmp_path, selection)

        assert not (tmp_path / "S1" / "run-0001" / "visual_perception").exists()

    def test_only_the_selected_images_are_perceived(self, tmp_path: Path) -> None:
        pytest.importorskip("PIL")  # as imagens preparadas são PNG; Pillow não é dependência base

        output = _perceive(
            tmp_path, {"kind": "frame_range", "start_frame_index": 1, "end_frame_index": 3}
        )

        reader = PerceptionRunReader(output)
        perceived = sorted(str(result.source_observation_id) for result in reader.list_results())
        assert perceived == ["frame-0001", "frame-0002"]
        assert reader.manifest.selection_id == selection_identity(
            SequenceArtifactId(SEQUENCE_ID),
            FrameRangeSelection(start_frame_index=1, end_frame_index=3),
        )

    def test_without_a_selection_the_whole_sequence_is_perceived(self, tmp_path: Path) -> None:
        pytest.importorskip("PIL")

        reader = PerceptionRunReader(_perceive(tmp_path, None))

        assert len(reader.list_results()) == 4
        assert reader.manifest.selection_id == selection_identity(
            SequenceArtifactId(SEQUENCE_ID), FullSequenceSelection()
        )


def test_sensor_association_records_the_selection_of_its_perception_run(tmp_path: Path) -> None:
    refs = foundation_refs(tmp_path)
    selection_id = "sha256:" + "a" * 64
    perception_dir = tmp_path / "S1" / "run-0001" / "visual_perception"
    manifest = PerceptionRunWriter(
        output_dir=perception_dir,
        sequence_name="S1",
        run_id=PerceptionRunId("perception-1"),
        run_index=1,
        sequence_artifact_id=SequenceArtifactId(refs["sequence"].artifact_id),
        selection_id=selection_id,
        enabled_capabilities=frozenset({"semantic_interpreter"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:test",
    ).finalize()
    perception = ArtifactRef(
        stage_id="visual_perception",
        contract="PerceptionRunArtifact",
        artifact_id="perception-1",
        content_hash=inventory_digest(manifest.file_inventory),
        location="S1/run-0001/visual_perception",
    )
    executor = SensorAssociationExecutor(
        candidates=CandidateGeometryPolicy(max_range_m=None),
        occlusion=OcclusionPolicy(
            cell_size_px=4, neighborhood_radius_cells=0, depth_margin_m=0.1, depth_margin_ratio=0.02
        ),
        tolerances=DiagnosticTolerances(
            max_pose_time_delta_ns=1_000_000,
            max_map_window_offset_ns=None,
            max_reprojection_p95_px=None,
            max_reprojection_invalid_rate=None,
        ),
        pose_policy=LookupPolicy.exact(),
    )
    output = tmp_path / "S1" / "run-0001" / "sensor_association"

    executor.execute(
        StageRequest(
            stage_id="sensor_association",
            inputs={
                "sequence": (refs["sequence"],),
                "perception": (perception,),
                "trajectory": (refs["state_estimation"],),
                "geometry": (refs["geometry"],),
            },
            components={},
            config_digest="sha256:test",
            output_dir=output,
            workspace=tmp_path,
        )
    )

    assert SensorAssociationRunReader(output).manifest.selection_id == selection_id

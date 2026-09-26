"""Real sequence, trajectory and map artifacts, written the way the runtime writes them.

The Spatial Foundation is validated with the capabilities' public readers, so its tests need
the artifacts themselves, not doubles: the synthetic CI sequence, a real State Estimation run
over it and a real Geometric Map built from both, each in its own stage directory.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from contextmap.evaluation.ci_fixtures import CI_FIXTURE_ID, build_synthetic_sequence
from contextmap.geometric_mapping import MotionCorrectionPolicy, ScanDisposition
from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    ImageObservation,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
)
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import (
    GeometricMappingExecutor,
    StateEstimationExecutor,
    inventory_digest,
)
from contextmap.state_estimation import (
    BODY_ENDPOINT,
    GeometryRequirements,
    LookupPolicy,
    StaticRelationRequirement,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
)

DATASET = "S1"
_CAMERA_CALIBRATION = CalibrationReferenceId("front_camera-calib")


def _request(
    workspace: Path, run: str, stage_id: str, inputs: dict[str, ArtifactRef], digest: str
) -> StageRequest:
    return StageRequest(
        stage_id=stage_id,
        inputs={name: (ref,) for name, ref in inputs.items()},
        components={},
        config_digest=digest,
        output_dir=workspace / DATASET / run / stage_id,
        workspace=workspace,
    )


class SyntheticIngestion:
    """An ingestion stage that publishes the synthetic CI sequence where the runtime says."""

    def execute(self, request: StageRequest) -> ArtifactRef:
        assert request.output_dir is not None and request.workspace is not None
        return _publish_sequence(request.output_dir, request.workspace, "seq-A")


def write_sequence(
    workspace: Path, *, run: str = "run-0001", artifact_id: str = "seq-A"
) -> ArtifactRef:
    """Publish the synthetic CI sequence under ``<run>/ingestion``."""
    return _publish_sequence(workspace / DATASET / run / "ingestion", workspace, artifact_id)


def _publish_sequence(output: Path, workspace: Path, artifact_id: str) -> ArtifactRef:
    sequence = build_synthetic_sequence()
    with SequenceArtifactWriter(
        output_dir=output, sequence_name=CI_FIXTURE_ID, artifact_id=SequenceArtifactId(artifact_id)
    ) as writer:
        writer.set_calibration(sequence.calibration)
        for observation in sequence.observations:
            if isinstance(observation, ImageObservation):
                observation = dataclasses.replace(observation, calibration_id=_CAMERA_CALIBRATION)
            writer.add_observation(observation)
        writer.finalize()
    manifest = SequenceArtifactReader(output).manifest
    return ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id=artifact_id,
        content_hash=inventory_digest(manifest.file_inventory),
        location=output.relative_to(workspace).as_posix(),
    )


def estimate(
    workspace: Path, sequence: ArtifactRef, *, run: str = "run-0001", digest: str = "se-1"
) -> ArtifactRef:
    """Run the real State Estimation executor over ``sequence`` with external poses."""
    return state_estimation_executor().execute(
        _request(workspace, run, "state_estimation", {"sequence": sequence}, digest)
    )


def state_estimation_executor() -> StateEstimationExecutor:
    """The real State Estimation executor, over the synthetic sequence's external poses."""
    return StateEstimationExecutor(
        ExternalPoseEstimator(
            ExternalPoseConfig(reference_frame=FrameId("odom"), body_frame=FrameId("base_link"))
        ),
        downstream=(
            GeometryRequirements(
                capability="geometric_mapping",
                modalities=frozenset({"lidar"}),
                static_relations=(
                    StaticRelationRequirement(from_endpoint=BODY_ENDPOINT, to_endpoint="lidar"),
                ),
            ),
        ),
    )


def build_map(
    workspace: Path,
    sequence: ArtifactRef,
    trajectory: ArtifactRef,
    *,
    run: str = "run-0001",
    digest: str = "gm-1",
) -> ArtifactRef:
    """Run the real Geometric Mapping executor over ``sequence`` and ``trajectory``."""
    inputs = {"sequence": sequence, "trajectory": trajectory}
    return geometric_mapping_executor().execute(
        _request(workspace, run, "geometric_mapping", inputs, digest)
    )


def geometric_mapping_executor() -> GeometricMappingExecutor:
    """The real Geometric Mapping executor, with exact pose lookup."""
    return GeometricMappingExecutor(
        pose_lookup=LookupPolicy.exact(),
        motion_correction=MotionCorrectionPolicy(
            raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN
        ),
        code_version="test",
    )


def foundation_refs(
    workspace: Path, *, run: str = "run-0001", artifact_id: str = "seq-A"
) -> dict[str, ArtifactRef]:
    """Write one complete, consistent foundation and return its three references."""
    sequence = write_sequence(workspace, run=run, artifact_id=artifact_id)
    trajectory = estimate(workspace, sequence, run=run)
    geometry = build_map(workspace, sequence, trajectory, run=run)
    return {"sequence": sequence, "state_estimation": trajectory, "geometry": geometry}

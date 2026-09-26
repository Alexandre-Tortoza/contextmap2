"""Run Sensor Association over one arm's geometry for the paired 2D→3D comparison (#624).

The association is the current consumer, unchanged: only the geometry it reads differs. For
arm A it is the raw map; for a derived arm it is the ``VoxelAggregationArtifact`` read as a
block source of its derived map. Frames, regions, masks, calibration, trajectory, candidate,
occlusion and pose policies are identical for every arm, and are the ones the #564 scaling
experiment froze from the canonical ``corridor-245-90s`` run, so the population is reused,
not reinvented.

Usage:
    python associate.py <arm-id> <output-dir> --raw-map DIR [--voxel-aggregation DIR]
        --sequence DIR --trajectory DIR --perception DIR [--window frozen|all]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    GeometryBlockSource,
    VoxelAggregationArtifactReader,
)
from contextmap.ingestion import ImageObservation, SequenceArtifactReader
from contextmap.sensor_association import (
    AssociationFrameInput,
    CandidateGeometryPolicy,
    DiagnosticTolerances,
    OcclusionPolicy,
    SensorAssociationRequest,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
)
from contextmap.state_estimation import LookupPolicy, StateEstimationRunReader, TrajectoryLookup
from contextmap.visual_perception import (
    ArtifactReference,
    PerceptionRunReader,
    SourceImage,
    prepare_image,
)

# A mesma janela congelada e as mesmas políticas do experimento #564 (ver o README dele).
FROZEN_WINDOW = tuple(f"camera_1_image_raw-{index:06d}" for index in range(168, 253, 6))
OCCLUSION = OcclusionPolicy(
    cell_size_px=4, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02
)
TOLERANCES = DiagnosticTolerances(
    max_pose_time_delta_ns=250_000_000,
    max_map_window_offset_ns=1_000_000_000,
    max_reprojection_p95_px=None,
    max_reprojection_invalid_rate=None,
)
POSE_POLICY = LookupPolicy.interpolated(max_interpolation_gap_ns=450_000_000)


def _frame(image: ImageObservation, result: object) -> AssociationFrameInput:
    prepared = prepare_image(
        SourceImage(
            source_observation_id=str(image.observation_id),
            image=ArtifactReference(
                uri=f"sequence://{image.observation_id}",
                sha256=hashlib.sha256(image.data).hexdigest(),
                media_type=f"image/{image.encoding.value}",
            ),
            width=image.width,
            height=image.height,
        ),
        operations=(),
    )
    return AssociationFrameInput(
        observation=image,
        prepared_image=prepared,
        perception_result=result,  # type: ignore[arg-type]
        dense_maps={},
    )


def main() -> int:
    """Associate the frozen window over one arm's geometry and print a summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--raw-map", type=Path, required=True)
    parser.add_argument("--voxel-aggregation", type=Path, default=None)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument("--window", default="frozen", choices=("frozen", "all"))
    parser.add_argument("--max-range-m", type=float, default=20.0)
    options = parser.parse_args()

    sequence = SequenceArtifactReader(options.sequence)
    calibration = sequence.read_calibration()
    if calibration is None:
        raise SystemExit("the sequence artifact carries no calibration")
    trajectory = StateEstimationRunReader(options.trajectory)
    perception = PerceptionRunReader(options.perception)
    selected = set(FROZEN_WINDOW) if options.window == "frozen" else None
    offsets = {
        str(entry.observation_id): entry.offset
        for entry in sequence.iter_index()
        if entry.modality == "image" and (selected is None or str(entry.observation_id) in selected)
    }
    wanted = {
        str(result.source_observation_id)
        for result in perception.iter_results()
        if str(result.source_observation_id) in offsets
    }
    if not wanted:
        raise SystemExit("the selection matched no perceived frame")

    def frames() -> Iterator[AssociationFrameInput]:
        for result in perception.iter_results():
            identity = str(result.source_observation_id)
            if identity in wanted:
                image = sequence.observation_at(offsets[identity])
                if isinstance(image, ImageObservation):
                    yield _frame(image, result)

    started = time.perf_counter()
    with GeometricMapArtifactReader(options.raw_map) as raw:
        geometry: GeometryBlockSource = (
            raw.geometry()
            if options.voxel_aggregation is None
            else VoxelAggregationArtifactReader(options.voxel_aggregation).geometry()
        )
        request = SensorAssociationRequest(
            sequence_artifact_id=sequence.manifest.artifact_id,
            selection_id=f"geometric-aggregation-raw-vs-voxel-20260926/{options.window}",
            geometry=geometry,
            trajectory=TrajectoryLookup(trajectory.trajectory()),
            pose_policy=POSE_POLICY,
            calibration=calibration,
            candidate_policy=CandidateGeometryPolicy(max_range_m=options.max_range_m),
            occlusion_policy=OCCLUSION,
            tolerances=TOLERANCES,
            frames=frames(),
            state_estimation_run_id=trajectory.manifest.run_id,
            code_version=f"geometric-aggregation-{options.arm}",
            mask_loader=perception.mask_store(),
        )
        writer = SensorAssociationRunWriter(
            output_dir=options.output_dir,
            sequence_name=sequence.manifest.sequence_name,
            run_id=SensorAssociationRunId(f"geometric-aggregation-{options.arm}"),
            run_index=1,
        )
        with writer.transaction() as run:
            outcome = SensorAssociationService().run(request, sink=run)
            manifest = run.finalize(outcome, runtime_s=time.perf_counter() - started)
    reader = SensorAssociationRunReader(options.output_dir)
    print(
        json.dumps(
            {
                "arm": options.arm,
                "geometric_map_id": str(manifest.geometric_map_id),
                "window": options.window,
                "selected_frames": len(wanted),
                "frame_count": manifest.frame_count,
                "rejected_frame_count": manifest.rejected_frame_count,
                "observation_count": manifest.observation_count,
                "wall_time_seconds": round(time.perf_counter() - started, 3),
                "integrity_problems": reader.verify_integrity(),
                "run_dir": str(options.output_dir),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

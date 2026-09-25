"""Run one Sensor Association arm over the real corridor-02 inputs (issue #564).

One arm is one (candidate policy, persistence mode) pair executed over the same frozen
upstream artifacts and the same frozen frame window, into its own output directory. The
arm never touches an existing artifact.

Persistence mode is not a flag: it is whichever API the checked-out revision has. On the
batch revision (before #563) the writer exposes ``finalize(outcome)`` and the service
retains every frame; on the streaming revision it exposes ``transaction()`` and the service
takes a ``sink``. The script detects which and reports what it used, so the same file can be
copied into a worktree at the baseline commit.

Usage:
    python arm.py <arm-id> <output-dir> [--max-range-m M] [--frames N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from contextmap.ingestion import ImageObservation, SequenceArtifactReader
from contextmap.sensor_association import (
    DiagnosticTolerances,
    OcclusionPolicy,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationRequest,
    SensorAssociationService,
)
from contextmap.geometric_mapping import GeometricMapArtifactReader
from contextmap.state_estimation import LookupPolicy, StateEstimationRunReader, TrajectoryLookup
from contextmap.visual_perception import (
    ArtifactReference,
    PerceptionRunReader,
    SourceImage,
    prepare_image,
)
from contextmap.sensor_association.service import AssociationFrameInput

WORKSPACE = Path("/home/alexmrtr/Projects/contextmap2/outputs")
SEQUENCE = WORKSPACE / "ingest-real/sequences/corridor-02/d8ef485b87af4452b224c9611ba0c621"
TRAJECTORY = WORKSPACE / "corridor-245-90s/run-0001/state_estimation"
GEOMETRY = WORKSPACE / "corridor-245-90s/run-0001/geometric_mapping"
PERCEPTION = (
    WORKSPACE / "corridor-245-90s/visual_perception/workspace/corridor-02/run-0001/visual_perception"
)

# A janela congelada do run canônico corridor-245-90s: as mesmas 15 imagens físicas que
# `outputs/corridor-245-90s/run-0001/sensor_association` associou (11 aceitas, 4 rejeitadas
# pela política de pose). Reusada, não reinventada.
FROZEN_WINDOW = tuple(f"camera_1_image_raw-{index:06d}" for index in range(168, 253, 6))

# As mesmas políticas que o run canônico registrou no seu manifest.
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
    digest = hashlib.sha256(image.data).hexdigest()
    prepared = prepare_image(
        SourceImage(
            source_observation_id=str(image.observation_id),
            image=ArtifactReference(
                uri=f"sequence://{image.observation_id}",
                sha256=digest,
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


def _candidate_policy(max_range_m: float | None) -> object | None:
    """The candidate policy of this revision, or ``None`` when it has no such concept."""
    try:
        from contextmap.sensor_association import CandidateGeometryPolicy
    except ImportError:
        return None
    return CandidateGeometryPolicy(max_range_m=max_range_m)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-range-m", type=float, default=None)
    parser.add_argument("--frames", type=int, default=None, help="cap on selected frames")
    parser.add_argument(
        "--window",
        default="frozen",
        choices=("frozen", "all"),
        help="'frozen' is the canonical 15-frame window; 'all' is every perceived frame",
    )
    options = parser.parse_args()

    sequence = SequenceArtifactReader(SEQUENCE)
    calibration = sequence.read_calibration()
    if calibration is None:
        raise SystemExit("the sequence artifact carries no calibration")
    trajectory = StateEstimationRunReader(TRAJECTORY)
    perception = PerceptionRunReader(PERCEPTION)

    # Só as imagens são decodificadas: list_observations() decodificaria também os 364 MB de
    # pointcloud e os 20781 registros de IMU, que esta associação nunca usa (#511).
    selected = set(FROZEN_WINDOW) if options.window == "frozen" else None
    images: dict[str, ImageObservation] = {}
    for entry in sequence.iter_index():
        if entry.modality != "image":
            continue
        if selected is not None and str(entry.observation_id) not in selected:
            continue
        observation = sequence.observation_at(entry.offset)
        if isinstance(observation, ImageObservation):
            images[str(observation.observation_id)] = observation

    results = [
        result
        for result in perception.iter_results()
        if str(result.source_observation_id) in images
    ]
    results.sort(key=lambda result: str(result.source_observation_id))
    if options.frames is not None:
        results = results[: options.frames]
    frames = tuple(_frame(images[str(result.source_observation_id)], result) for result in results)
    if not frames:
        raise SystemExit("the selection matched no perceived frame")

    candidate_policy = _candidate_policy(options.max_range_m)
    extra: dict[str, object] = {}
    if candidate_policy is not None:
        extra["candidate_policy"] = candidate_policy
    elif options.max_range_m is not None:
        raise SystemExit("this revision has no candidate policy; --max-range-m is not available")

    writer_kwargs = {
        "output_dir": options.output_dir,
        "sequence_name": sequence.manifest.sequence_name,
        "run_id": SensorAssociationRunId(f"sa-scaling-{options.arm}"),
        "run_index": 1,
    }
    streaming = hasattr(SensorAssociationRunWriter, "transaction")
    started = time.perf_counter()
    with GeometricMapArtifactReader(GEOMETRY) as geometry:
        request = SensorAssociationRequest(
            sequence_artifact_id=sequence.manifest.artifact_id,
            selection_id=f"sensor-association-scaling-20260925/{options.window}",
            geometry=geometry.geometry(),
            trajectory=TrajectoryLookup(trajectory.trajectory()),
            pose_policy=POSE_POLICY,
            calibration=calibration,
            occlusion_policy=OCCLUSION,
            tolerances=TOLERANCES,
            frames=frames,
            state_estimation_run_id=trajectory.manifest.run_id,
            code_version=f"scaling-{options.arm}",
            mask_loader=perception.mask_store(),
            **extra,  # type: ignore[arg-type]
        )
        if streaming:
            writer = SensorAssociationRunWriter(**writer_kwargs)  # type: ignore[arg-type]
            with writer.transaction() as run:
                outcome = SensorAssociationService().run(request, sink=run)
                manifest = run.finalize(outcome, runtime_s=time.perf_counter() - started)
        else:
            outcome = SensorAssociationService().run(request)
            manifest = SensorAssociationRunWriter(**writer_kwargs).finalize(  # type: ignore[arg-type]
                outcome, runtime_s=time.perf_counter() - started
            )
    elapsed = time.perf_counter() - started

    reader = SensorAssociationRunReader(options.output_dir)
    print(
        json.dumps(
            {
                "arm": options.arm,
                "persistence": "streaming" if streaming else "batch",
                "max_range_m": options.max_range_m,
                "window": options.window,
                "selected_frames": len(frames),
                "frame_count": manifest.frame_count,
                "rejected_frame_count": manifest.rejected_frame_count,
                "observation_count": manifest.observation_count,
                "wall_time_seconds": round(elapsed, 3),
                "integrity_problems": reader.verify_integrity(),
                "run_dir": str(options.output_dir),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    sys.exit(main())

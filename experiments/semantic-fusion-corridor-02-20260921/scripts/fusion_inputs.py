"""Loads the real upstream evidence of the fusion arms: association runs, perception results, map."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import (
    PERCEPTION_BASE,
    SELECTION,
    VAL,
    decode_window_observations,
)

from contextmap.sensor_association import (
    ObservationQuality,
    SensorAssociationRunReader,
    SpatialObservation,
    SpatialObservationId,
)
from contextmap.shared import SourceTimestamp
from contextmap.visual_perception import (
    BackendProvenance,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRun,
    PerceptionRunId,
    PerceptionRunReader,
)


@dataclass
class UpstreamEvidence:
    """Everything the fusion arms consume, read once from the persisted upstream artifacts."""

    association_dirs: list[Path]
    association_run_ids: tuple[str, ...]
    perception_dirs: dict[str, Path]
    perception_runs: list[PerceptionRun]
    perception_readers: dict[str, PerceptionRunReader]
    observations: list[SpatialObservation]
    qualities: dict[SpatialObservationId, ObservationQuality]
    results: dict[PerceptionResultId, PerceptionResult]
    timestamps: dict[Any, SourceTimestamp]
    sequence_artifact_id: str
    map_id: str
    manifests: dict[str, dict[str, Any]] = field(default_factory=dict)


def rebuild_perception_run(directory: Path, results: list[PerceptionResult]) -> PerceptionRun:
    """W4: rebuild ``PerceptionRun`` from the manifest and the provenance stored on the results."""
    manifest = json.loads((directory / "manifest.json").read_text())
    provenance: dict[str, BackendProvenance] = {}
    for result in results:
        for region in result.regions:
            provenance.setdefault("region_discovery", region.provenance)
        for feature in result.features:
            provenance.setdefault(f"{feature.scope.value}_feature_extraction", feature.provenance)
    return PerceptionRun(
        run_id=PerceptionRunId(manifest["run_id"]),
        run_index=manifest["run_index"],
        sequence_artifact_id=manifest["sequence_artifact_id"],
        selection_id=manifest["selection_id"],
        enabled_capabilities=frozenset(manifest["enabled_capabilities"]),
        backend_provenance=provenance,
        code_version=None,
    )


def load_upstream(association_root: Path) -> UpstreamEvidence:
    """Read every association run under ``association_root`` and the perception runs it names."""
    directories = sorted(association_root.glob("runs/sensor-association/corridor-02/run-*"))
    if not directories:
        raise SystemExit(f"no association run under {association_root}")
    perception_by_id = {
        json.loads((path / "manifest.json").read_text())["run_id"]: path
        for path in PERCEPTION_BASE.glob("run-*")
    }
    observations: list[SpatialObservation] = []
    qualities: dict[SpatialObservationId, ObservationQuality] = {}
    used_perception: dict[str, Path] = {}
    association_ids: list[str] = []
    manifests: dict[str, dict[str, Any]] = {}
    sequence_id = map_id = ""
    for directory in directories:
        reader = SensorAssociationRunReader(directory)
        manifest = reader.manifest
        association_ids.append(str(manifest.run_id))
        sequence_id = manifest.sequence_artifact_id
        map_id = str(manifest.geometric_map_id)
        for run_id in manifest.perception_run_ids:
            used_perception[str(run_id)] = perception_by_id[str(run_id)]
        for observation in reader.observations():
            observations.append(observation)
            qualities[observation.spatial_observation_id] = reader.quality(
                observation.spatial_observation_id
            )
        manifests[str(manifest.run_id)] = json.loads((directory / "manifest.json").read_text())
    perception_readers = {run_id: PerceptionRunReader(path) for run_id, path in used_perception.items()}
    results: dict[PerceptionResultId, PerceptionResult] = {}
    runs: list[PerceptionRun] = []
    for run_id, reader in perception_readers.items():
        listed = reader.list_results()
        for result in listed:
            results[result.result_id] = result
        runs.append(rebuild_perception_run(used_perception[run_id], listed))
    frame_ids = {str(o.source_observation_id) for o in observations}
    images = decode_window_observations("image", {item["observation_id"] for item in SELECTION["images"]["selected"]})
    timestamps = {image.observation_id: image.timestamp for image in images if str(image.observation_id) in frame_ids}
    return UpstreamEvidence(
        association_dirs=directories,
        association_run_ids=tuple(sorted(association_ids)),
        perception_dirs=used_perception,
        perception_runs=sorted(runs, key=lambda run: run.run_id),
        perception_readers=perception_readers,
        observations=observations,
        qualities=qualities,
        results=results,
        timestamps=timestamps,
        sequence_artifact_id=sequence_id,
        map_id=map_id,
        manifests=manifests,
    )


def default_association_root() -> Path:
    """The full association stage of this validation."""
    return VAL / "sensor_association"

"""Shared helpers of the Semantic Fusion real-data validation drivers (corridor-02, 90 s window).

Code under test: the ``wt-milestone-semantic-fusion`` worktree of ContextMap2 (dev @ ac1eb59 plus the
issue-121 branch). Nothing here is imported by ``src/``: these are validation drivers.

Read-only inputs (main checkout): datasets/corridor-02/*, outputs/ingest-full/..., the
PerceptionRunArtifacts of outputs/validation/2026-09-21/visual_perception, selection.json.

WORKAROUNDS (driver only, never in src; each one is reported as a finding):
  W1. corridor-02-gt.txt is not in the SequenceArtifact (0 external_pose observations): the driver
      builds ExternalPoseMeasurement objects from the TUM file itself.
  W2. Issue #374: SequenceArtifactReader loads every payload, so the window's observations are
      decoded one by one from ``index.jsonl`` with the private ``_decode_observation``.
  W3. The persisted calibration has ``camera_model=None`` for the RGB camera ("MEI intrinsics not
      representable"), written before the canonical MEI model existed. The driver builds a
      ``MeiCameraModel`` from datasets/corridor-02/corridor-02-Intrinsics.yaml and derives a NEW
      calibration set (different identity), so state estimation, geometric mapping and sensor
      association are all re-run under that same identity (the projector refuses a mismatch).
  W4. PerceptionRunManifest does not persist ``PerceptionRun.backend_provenance``; the driver
      rebuilds it from the provenance stored on each result's regions and features.

BUNDLE (experiments/semantic-fusion-corridor-02-20260921): the only differences from the executed copy
are the path constants ``MAIN``, ``VAL`` and ``SELECTION`` below and the ``os`` import they use
(absolute personal paths removed); see the bundle README.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationSet,
    MeiCameraModel,
    SequenceArtifactReader,
)
from contextmap.ingestion import sequence_artifact as _sa
from contextmap.ingestion.calibration import compute_content_hash

NS = 10**9
# Raiz do checkout que contém os insumos não versionados (datasets/ e outputs/). Por padrão é o checkout
# deste bundle (experiments/<bundle>/scripts/common.py -> raiz do repositório).
MAIN = Path(os.environ.get("CONTEXTMAP_INPUT_ROOT", Path(__file__).resolve().parents[3]))
DATE = "20260921"
# Onde as etapas gravam os artifacts e os relatórios (não versionado).
VAL = Path(
    os.environ.get(
        "CONTEXTMAP_VALIDATION_DIR", MAIN / f"workspace/corridor-02/validation-semantic-fusion-{DATE}"
    )
)
SRC_VAL = MAIN / "outputs/validation/2026-09-21"
SEQ_DIR = MAIN / "outputs/ingest-full/sequences/corridor-02/e145f73f8d894f18b96ef1f55ca308c2"
GT = MAIN / "datasets/corridor-02/corridor-02-gt.txt"
INTRINSICS = MAIN / "datasets/corridor-02/corridor-02-Intrinsics.yaml"
PERCEPTION_BASE = SRC_VAL / "visual_perception/workspace/runs/visual-perception/corridor-02"
SEQUENCE_NAME = "corridor-02"
CLOCK = "corridor-02-header"
CAMERA_CALIBRATION_ID = "cal-camera_1_image_raw"
# A seleção é a versionada no bundle, byte a byte a mesma de outputs/validation/2026-09-21/selection.json.
SELECTION = json.loads((Path(__file__).resolve().parents[1] / "selection.json").read_text())
CODE_SHA = SELECTION["code_sha"]


def stage_root(stage: str) -> Path:
    """Workspace root handed to the (legacy) writers of one stage."""
    root = VAL / stage
    root.mkdir(parents=True, exist_ok=True)
    return root


def peak_rss_mb() -> int:
    """Peak resident set size of this process, in MB."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM"):
            return int(line.split()[1]) // 1024
    return -1


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(stage: str, name: str, payload: Any) -> Path:
    """Write a JSON report next to the stage outputs."""
    target = stage_root(stage) / name
    target.write_text(json.dumps(payload, indent=1, default=str, sort_keys=True))
    return target


def mei_calibration() -> tuple[CalibrationSet, dict[str, Any]]:
    """W3: the sequence calibration with the RGB camera's MEI model from the intrinsics YAML.

    Returns:
        The derived calibration set and a record of exactly what was derived from what.
    """
    base = SequenceArtifactReader(SEQ_DIR).read_calibration()
    assert base is not None
    intrinsics = yaml.safe_load(INTRINSICS.read_text())["rgb_camera"]
    model = MeiCameraModel(
        width=int(intrinsics["image_width"]),
        height=int(intrinsics["image_height"]),
        fx=float(intrinsics["projection_parameters"]["gamma1"]),
        fy=float(intrinsics["projection_parameters"]["gamma2"]),
        cx=float(intrinsics["projection_parameters"]["u0"]),
        cy=float(intrinsics["projection_parameters"]["v0"]),
        xi=float(intrinsics["mirror_parameters"]["xi"]),
        distortion_coefficients=(
            float(intrinsics["distortion_parameters"]["k1"]),
            float(intrinsics["distortion_parameters"]["k2"]),
            float(intrinsics["distortion_parameters"]["p1"]),
            float(intrinsics["distortion_parameters"]["p2"]),
        ),
    )
    original = base.entries[CAMERA_CALIBRATION_ID]
    entry = CalibrationEntry(
        calibration_id=original.calibration_id,
        sensor_id=original.sensor_id,
        frame_id=original.frame_id,
        camera_model=model,
        provenance=CalibrationProvenance(
            source_type="dataset",
            source_path="datasets/corridor-02/corridor-02-Intrinsics.yaml",
            original_values={"model_type": intrinsics["model_type"]},
            conversions_applied=(
                "MEI model derived by the validation driver: the persisted SequenceArtifact "
                "calibration had camera_model=None",
            ),
        ),
        content_hash=compute_content_hash(
            sensor_id=original.sensor_id, frame_id=original.frame_id, camera_model=model
        ),
    )
    entries = dict(base.entries)
    entries[CAMERA_CALIBRATION_ID] = entry
    derived = CalibrationSet(
        entries=entries, static_transforms=base.static_transforms, schema_version=base.schema_version
    )
    record = {
        "source_file": str(INTRINSICS.relative_to(MAIN)),
        "source_sha256": sha256_file(INTRINSICS),
        "mei": {
            "width": model.width,
            "height": model.height,
            "fx": model.fx,
            "fy": model.fy,
            "cx": model.cx,
            "cy": model.cy,
            "xi": model.xi,
            "distortion_coefficients": list(model.distortion_coefficients),
        },
        "camera_entry_content_hash_before": original.content_hash,
        "camera_entry_content_hash_after": entry.content_hash,
    }
    return derived, record


def decode_window_observations(modality: str, observation_ids: set[str]) -> list[Any]:
    """W2: decode selected observations from ``index.jsonl`` without touching the other payloads."""
    needle = f'"modality": "{modality}"'
    found = []
    with (SEQ_DIR / "index.jsonl").open() as handle:
        for line in handle:
            if needle not in line:
                continue
            record = json.loads(line)
            if record["observation_id"] in observation_ids:
                found.append(_sa._decode_observation(record, SEQ_DIR))
    return found

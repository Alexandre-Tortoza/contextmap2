"""Run the geometry half of one arm of the raw vs voxel-aggregation experiment (#624).

Arm A (no ``--cell-m``) is the raw ``GeometricMapArtifact`` exactly as it is on disk: it is
read, verified and timed, never rewritten. Arms B/C/D derive a separate
``VoxelAggregationArtifact`` from the same raw artifact with the same grid origin, the same
representative and the same lineage; the resolution is the only thing that changes.

For every arm the script reports, apart from quality: counts, bytes on disk, the time to
build (derived arms), to read everything back and to answer a fixed set of box queries. For
derived arms it adds the geometric fidelity report (pre-registered in
``src/contextmap/evaluation/docs/voxel_aggregation.md``), the lineage verification against
the raw map and the chunking-invariance check. The JSON goes to stdout (for ``_run_one.py``)
and to ``<output-dir>/arm-report.json``.

Usage:
    python arm.py <arm-id> <output-dir> --raw-map DIR [--cell-m R] [--origin X Y Z]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from contextmap.evaluation import encode_aggregation_fidelity_report, evaluate_aggregation_fidelity
from contextmap.geometric_mapping import (
    DEFAULT_BLOCK_POINTS,
    Bounds3D,
    GeometricMapArtifactReader,
    GeometryBlockSource,
    InterScanVoxelPolicy,
    VoxelAggregationArtifactReader,
    VoxelAggregationArtifactWriter,
    VoxelAggregationRunId,
    VoxelGridSpec,
    aggregate_geometry,
)

# Fixados antes de olhar resultados (#624): faixas de alcance ao sensor e as consultas.
RANGE_EDGES_M = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
QUERY_BOX_COUNT = 16
QUERY_BOX_SIDE_M = 2.0
# Um segundo tamanho de bloco, pequeno e sem relação com o tamanho de um scan, para a
# verificação de invariância a chunking sobre o dado real.
CHUNKING_CHECK_BLOCK_POINTS = 4_099


def _query_boxes(bounds: Bounds3D) -> list[Bounds3D]:
    """Boxes centered at evenly spaced points of the map's diagonal: fixed by the raw map only."""
    low, high = bounds.minimum_m, bounds.maximum_m
    half = QUERY_BOX_SIDE_M / 2.0
    boxes = []
    for position in range(QUERY_BOX_COUNT):
        t = (position + 0.5) / QUERY_BOX_COUNT
        center = [low[axis] + t * (high[axis] - low[axis]) for axis in range(3)]
        boxes.append(
            Bounds3D(
                frame_id=bounds.frame_id,
                minimum_m=(center[0] - half, center[1] - half, center[2] - half),
                maximum_m=(center[0] + half, center[1] + half, center[2] + half),
            )
        )
    return boxes


def _timed_reads(geometry: GeometryBlockSource, boxes: list[Bounds3D]) -> dict[str, Any]:
    started = time.perf_counter()
    rows = sum(len(block) for block in geometry.iter_blocks())
    full_read_s = time.perf_counter() - started
    started = time.perf_counter()
    selected = sum(len(block) for box in boxes for block in geometry.iter_blocks(bounds=box))
    return {
        "elements": rows,
        "full_read_seconds": round(full_read_s, 4),
        "box_queries": len(boxes),
        "box_query_seconds": round(time.perf_counter() - started, 4),
        "box_query_elements": selected,
    }


def _disk_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _raw_arm(raw_dir: Path) -> dict[str, Any]:
    with GeometricMapArtifactReader(raw_dir) as reader:
        geometry = reader.geometry()
        manifest = reader.manifest
        return {
            "representation": "raw",
            "aggregation_rule": manifest.aggregation_rule,
            "point_count": manifest.point_count,
            "source_point_count": manifest.source_point_count,
            "scan_count": sum(1 for scan in geometry.scans if scan.geometry_count),
            "observation_count": len(geometry.geometric_map.source_observation_ids),
            "reduction_ratio": manifest.point_count / manifest.source_point_count,
            "disk_bytes": _disk_bytes(raw_dir),
            "integrity_problems": reader.verify_integrity(),
            "reads": _timed_reads(geometry, _query_boxes(geometry.geometric_map.bounds)),
        }


def _derived_arm(
    arm: str,
    raw_dir: Path,
    output_dir: Path,
    policy: InterScanVoxelPolicy,
    block_points: int,
    code_version: str | None,
) -> dict[str, Any]:
    derived_dir = output_dir / "voxel_aggregation"
    started = time.perf_counter()
    VoxelAggregationArtifactWriter(
        output_dir=derived_dir, run_id=VoxelAggregationRunId(f"voxel-{arm}")
    ).finalize(
        source_dir=raw_dir, policy=policy, code_version=code_version, block_points=block_points
    )
    build_s = time.perf_counter() - started

    started = time.perf_counter()
    derived = VoxelAggregationArtifactReader(derived_dir)
    aggregation = derived.aggregation()
    load_s = time.perf_counter() - started
    with GeometricMapArtifactReader(raw_dir) as raw:
        raw_geometry = raw.geometry()
        reads = _timed_reads(derived.geometry(), _query_boxes(raw_geometry.geometric_map.bounds))
        reads["load_seconds"] = round(load_s, 4)
        fidelity = evaluate_aggregation_fidelity(
            raw_geometry, aggregation, range_edges_m=RANGE_EDGES_M
        )
        chunked = aggregate_geometry(raw_geometry, policy, block_points=CHUNKING_CHECK_BLOCK_POINTS)
        integrity = derived.verify_integrity(source=raw)
    return {
        "representation": "voxel_aggregation",
        "artifact_dir": str(derived_dir),
        "manifest": {
            key: value
            for key, value in vars(derived.manifest).items()
            if key not in {"file_inventory"}
        },
        "config": derived.read_record("config.json"),
        "metrics": derived.read_record("metrics/aggregation.json"),
        "disk_bytes": _disk_bytes(derived_dir),
        "build_seconds": round(build_s, 3),
        "reads": reads,
        "fidelity": encode_aggregation_fidelity_report(fidelity),
        "integrity_problems": integrity,
        "chunking_check": {
            "block_points": [block_points, CHUNKING_CHECK_BLOCK_POINTS],
            "problems": aggregation.equivalence_problems(chunked),
        },
    }


def main() -> int:
    """Run one arm and print its report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--raw-map", type=Path, required=True)
    parser.add_argument("--cell-m", type=float, default=None, help="omit for the raw arm A")
    parser.add_argument("--origin", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--block-points", type=int, default=DEFAULT_BLOCK_POINTS)
    parser.add_argument("--code-version", default=None, help="commit recorded in the lineage")
    options = parser.parse_args()

    options.output_dir.mkdir(parents=True, exist_ok=False)
    if options.cell_m is None:
        report = {"arm": options.arm, **_raw_arm(options.raw_map)}
    else:
        with GeometricMapArtifactReader(options.raw_map) as raw:
            frame = raw.manifest.map_frame
        policy = InterScanVoxelPolicy(
            grid=VoxelGridSpec(
                frame_id=frame, origin_m=tuple(options.origin), cell_m=options.cell_m
            )
        )
        report = {
            "arm": options.arm,
            "grid": policy.grid.to_record(),
            "policy_fingerprint": policy.fingerprint(),
            **_derived_arm(
                options.arm,
                options.raw_map,
                options.output_dir,
                policy,
                options.block_points,
                options.code_version,
            ),
        }
    text = json.dumps(report, indent=1, default=str)
    (options.output_dir / "arm-report.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    # Um braço que não verifica não é evidência: o relatório fica, mas o status diz que falhou.
    invalid = report.get("integrity_problems") or report.get("chunking_check", {}).get("problems")
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())

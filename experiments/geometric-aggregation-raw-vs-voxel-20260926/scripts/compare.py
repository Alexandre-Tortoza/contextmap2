"""Compare one derived arm's association against the raw arm's, region by region (#624).

The metric definitions are fixed in ``contextmap.evaluation.voxel_aggregation`` and
documented in ``src/contextmap/evaluation/docs/voxel_aggregation.md`` before any real run
was inspected; this script only feeds them the paired runs.

Usage:
    python compare.py <arm-id> <output-file> --raw-map DIR --voxel-aggregation DIR
        --raw-association DIR --arm-association DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contextmap.evaluation import (
    compare_paired_associations,
    encode_association_stability_report,
    region_supports,
)
from contextmap.geometric_mapping import GeometricMapArtifactReader, VoxelAggregationArtifactReader
from contextmap.sensor_association import SensorAssociationRunReader

# Fixado antes de olhar resultados: uma região com até 20 pontos brutos de suporte é "fina".
THIN_SUPPORT_MAX_POINTS = 20


def main() -> int:
    """Compare one arm's association with the raw arm's and write the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--raw-map", type=Path, required=True)
    parser.add_argument("--voxel-aggregation", type=Path, required=True)
    parser.add_argument("--raw-association", type=Path, required=True)
    parser.add_argument("--arm-association", type=Path, required=True)
    options = parser.parse_args()

    aggregated = VoxelAggregationArtifactReader(options.voxel_aggregation).geometry()
    with GeometricMapArtifactReader(options.raw_map) as raw:
        raw_geometry = raw.geometry()
        report = compare_paired_associations(
            raw_supports=region_supports(
                SensorAssociationRunReader(options.raw_association).observations()
            ),
            aggregate_supports=region_supports(
                SensorAssociationRunReader(options.arm_association).observations()
            ),
            raw_geometry=raw_geometry,
            aggregated=aggregated,
            membership=aggregated.aggregation.membership(raw_geometry),
            thin_support_max_points=THIN_SUPPORT_MAX_POINTS,
        )
    record = {"arm": options.arm, **encode_association_stability_report(report)}
    options.output_file.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    summary = {key: value for key, value in record.items() if key != "regions"}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

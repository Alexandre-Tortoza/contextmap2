"""Sensor Association run unchanged over a derived voxel aggregation, paired with the raw run.

Contract evidence only: the synthetic sequence scans the same four landmarks three times, so
the raw map holds three copies of each landmark and each region is supported by exactly
those copies. The aggregation must collapse the copies without losing a region. Nothing here
is evidence about real data; the real comparison is the #624 experiment.
"""

from __future__ import annotations

from pathlib import Path

from chain import RUN_A, associate, synthetic_chain

from contextmap.evaluation import compare_paired_associations, region_supports
from contextmap.geometric_mapping import (
    InterScanVoxelPolicy,
    VoxelAggregationArtifactReader,
    VoxelAggregationArtifactWriter,
    VoxelAggregationRunId,
    VoxelGridSpec,
)
from contextmap.ingestion import FrameId

# Os marcos ficam em múltiplos de 0,1 m: a origem em meio voxel os mantém longe das fronteiras.
POLICY = InterScanVoxelPolicy(
    grid=VoxelGridSpec(frame_id=FrameId("odom"), origin_m=(0.025, 0.025, 0.025), cell_m=0.05)
)


def test_association_over_the_aggregates_keeps_every_region_the_raw_map_supports(
    tmp_path: Path,
) -> None:
    with synthetic_chain(tmp_path / "chain") as chain:
        derived_dir = tmp_path / "voxel"
        VoxelAggregationArtifactWriter(
            output_dir=derived_dir, run_id=VoxelAggregationRunId("voxel-0.05m")
        ).finalize(
            source_dir=tmp_path / "chain" / "geometric_mapping",
            policy=POLICY,
            code_version="test",
        )
        derived = VoxelAggregationArtifactReader(derived_dir)
        assert derived.verify_integrity(source=chain.geometry) == []
        aggregated = derived.geometry()
        raw_geometry = chain.geometry.geometry()

        _, raw_run = chain.associations[0]
        _, aggregate_run = associate(
            tmp_path / "voxel-association",
            chain.sequence,
            chain.trajectory,
            aggregated,
            chain.perception_results,
            RUN_A,
            1,
        )
        report = compare_paired_associations(
            raw_supports=region_supports(raw_run.observations()),
            aggregate_supports=region_supports(aggregate_run.observations()),
            raw_geometry=raw_geometry,
            aggregated=aggregated,
            membership=aggregated.aggregation.membership(raw_geometry),
            thin_support_max_points=3,
        )

        assert aggregate_run.verify_integrity() == []
        assert aggregate_run.manifest.geometric_map_id == aggregated.geometric_map.map_id
        assert chain.geometry.verify_integrity() == []

    assert report.paired_region_count == raw_run.manifest.observation_count
    assert (report.raw_only_region_count, report.aggregate_only_region_count) == (0, 0)
    assert report.region_retention_rate == 1.0
    for region in report.regions:
        if region.raw_support_count:
            # Três cópias (uma por scan) do marco da região colapsam em um único agregado.
            assert (region.raw_support_count, region.aggregate_support_count) == (3, 1)
            assert region.disagreement == 0.0
            assert region.support_centroid_shift_m is not None
            assert region.support_centroid_shift_m < 1e-5

"""Public evaluation contracts and reproducible report helpers."""

from .region_discovery import (
    EvaluatedDiscoveryFrame,
    EvaluationRunDescriptor,
    GroundTruthRegion,
    ReferenceFrame,
    RegionDiscoveryComparison,
    RegionDiscoveryEvaluationReport,
    RegionDiscoveryEvaluator,
    RegionDiscoveryReferenceSet,
    compare_region_discovery_reports,
    write_region_discovery_reference_set,
    write_region_discovery_report,
)

__all__ = [
    "EvaluatedDiscoveryFrame",
    "EvaluationRunDescriptor",
    "GroundTruthRegion",
    "ReferenceFrame",
    "RegionDiscoveryComparison",
    "RegionDiscoveryEvaluationReport",
    "RegionDiscoveryEvaluator",
    "RegionDiscoveryReferenceSet",
    "compare_region_discovery_reports",
    "write_region_discovery_reference_set",
    "write_region_discovery_report",
]

"""Public API for deterministic ContextMap evaluation reports.

Evaluation measures quality, regressions, and cost without mutating pipeline
outputs. See ``src/contextmap/evaluation/docs/README.md``.
"""

from contextmap.evaluation.feature_extraction import (
    FeatureCostReport,
    FeatureEvaluationContext,
    FeatureEvaluationError,
    FeatureEvaluationReport,
    FeatureNumericalReport,
    FeatureResolutionComparisonReport,
    FeatureSpatialReport,
    assert_repeatable_feature_outputs,
    compare_feature_map_resolutions,
    encode_feature_evaluation_report,
    encode_feature_resolution_comparison_report,
    evaluate_feature_payload,
)
from contextmap.evaluation.region_discovery import (
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
    "FeatureCostReport",
    "FeatureEvaluationContext",
    "FeatureEvaluationError",
    "FeatureEvaluationReport",
    "FeatureNumericalReport",
    "FeatureResolutionComparisonReport",
    "FeatureSpatialReport",
    "GroundTruthRegion",
    "ReferenceFrame",
    "RegionDiscoveryComparison",
    "RegionDiscoveryEvaluationReport",
    "RegionDiscoveryEvaluator",
    "RegionDiscoveryReferenceSet",
    "assert_repeatable_feature_outputs",
    "compare_feature_map_resolutions",
    "compare_region_discovery_reports",
    "encode_feature_evaluation_report",
    "encode_feature_resolution_comparison_report",
    "evaluate_feature_payload",
    "write_region_discovery_reference_set",
    "write_region_discovery_report",
]

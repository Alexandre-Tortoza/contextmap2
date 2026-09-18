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

__all__ = [
    "FeatureCostReport",
    "FeatureEvaluationContext",
    "FeatureEvaluationError",
    "FeatureEvaluationReport",
    "FeatureNumericalReport",
    "FeatureResolutionComparisonReport",
    "FeatureSpatialReport",
    "assert_repeatable_feature_outputs",
    "compare_feature_map_resolutions",
    "encode_feature_evaluation_report",
    "encode_feature_resolution_comparison_report",
    "evaluate_feature_payload",
]

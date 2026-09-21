"""One rich, coherent fused evidence and its support, for serialization and artifact tests."""

from __future__ import annotations

from dataclasses import dataclass

from evidence_builders import ClaimSpec, Scenario, View, build_scenario, make_score
from fusion_builders import (
    make_point_representation_ref,
    make_region_feature,
    result_id,
    spatial_id,
)
from quality_builders import make_quality

from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    EvidenceChannel,
    FusedEvidence,
    FusionSupport,
    QualityAwareAccumulationPolicy,
    QualityInput,
    QualityRamp,
    accumulate_quality_aware_evidence,
)
from contextmap.visual_perception import HypothesisRole

POLICY = QualityAwareAccumulationPolicy(
    definitions_version="observation-quality-v1",
    ramps=(QualityRamp(quality_input=QualityInput.SUPPORT_DEPTH_MEDIAN_M, good=5.0, bad=25.0),),
    neutral_factor=0.5,
    baseline=BaselineAccumulationPolicy(
        abstention_labels=frozenset({"unknown"}),
        channels=frozenset(
            {
                EvidenceChannel.SEMANTIC_CLAIMS,
                EvidenceChannel.SEMANTIC_SCORES,
                EvidenceChannel.VISUAL_FEATURES,
                EvidenceChannel.POINT_REPRESENTATION,
            }
        ),
    ),
)


@dataclass(frozen=True)
class Rich:
    scenario: Scenario
    support: FusionSupport
    evidence: FusedEvidence


def make_rich(*, geometry: range = range(20)) -> Rich:
    """Two frames that contradict each other, an abstention, scores, features and quality."""
    features = (make_region_feature(feature="feature-0001"),)
    views = [
        View(
            "run-a",
            "frame-0120",
            (
                ClaimSpec("door", confidence=0.9),
                ClaimSpec("frame", role=HypothesisRole.ALTERNATIVE),
            ),
            features=features,
            geometry=geometry,
        ),
        View(
            "run-b",
            "frame-0120",
            (ClaimSpec("door", confidence=None),),
            features=features,
            geometry=geometry,
        ),
        View(
            "run-a",
            "frame-0121",
            (ClaimSpec("cabinet", confidence=0.0),),
            features=features,
            geometry=geometry,
        ),
        View("run-a", "frame-0122", (ClaimSpec("unknown"),), features=features, geometry=geometry),
    ]
    scenario = build_scenario(views, geometry_points=max(1_000, len(geometry) + 10))
    scores = {result_id(views[0].run, views[0].frame): [make_score(views[0], 0, 0.31)]}
    qualities = {
        spatial_id(view.run, view.frame): make_quality(
            spatial_id(view.run, view.frame), frame=view.frame, depth_median_m=8.0 + index
        )
        for index, view in enumerate(views)
    }
    evidence = accumulate_quality_aware_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        observation_qualities=qualities,
        semantic_scores=scores,
        point_representation_refs=[make_point_representation_ref(geometry_index=3)],
        policy=POLICY,
        code_version="test",
    )
    return Rich(scenario=scenario, support=scenario.support, evidence=evidence)

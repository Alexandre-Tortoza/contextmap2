"""Several supports, their fused evidence and the lineage of a fusion run, for artifact tests."""

from __future__ import annotations

from dataclasses import dataclass

from evidence_builders import ClaimSpec, View, build_scenario_parts, make_score
from fusion_builders import MAP_ID, make_point_representation_ref, make_region_feature, spatial_id
from observation_builders import make_perception_run
from quality_builders import make_quality

from contextmap.point_representation import PointRepresentationRunId
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    EvidenceChannel,
    ExcludedObservation,
    FusionOutcome,
    FusionRunLineage,
    QualityAwareAccumulationPolicy,
    QualityInput,
    QualityRamp,
    accumulate_quality_aware_evidence,
)
from contextmap.visual_perception import HypothesisRole, PerceptionRunId

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

LINEAGE = FusionRunLineage(
    sequence_artifact_id="sequence-0001",
    geometric_map_id=MAP_ID,
    association_run_ids=("association-run-0001",),
    perception_run_ids=(PerceptionRunId("run-a"), PerceptionRunId("run-b")),
    point_representation_run_ids=(PointRepresentationRunId("representation-run-0001"),),
)


@dataclass(frozen=True)
class RunFixture:
    outcomes: tuple[FusionOutcome, ...]
    excluded: tuple[ExcludedObservation, ...]


def make_run_fixture() -> RunFixture:
    """Three supports over disjoint geometry: contradiction, agreement and pure abstention."""
    features = (make_region_feature(feature="feature-0001"),)
    views = [
        # support 1: two frames contradict each other, two runs interpret the first
        View(
            "run-a",
            "frame-0120",
            (
                ClaimSpec("door", confidence=0.9),
                ClaimSpec("frame", role=HypothesisRole.ALTERNATIVE),
            ),
            features=features,
            geometry=range(0, 20),
        ),
        View(
            "run-b",
            "frame-0120",
            (ClaimSpec("door", confidence=None), ClaimSpec("unknown")),
            features=features,
            geometry=range(0, 20),
        ),
        View(
            "run-a",
            "frame-0121",
            (ClaimSpec("cabinet", confidence=0.0),),
            features=features,
            geometry=range(0, 20),
        ),
        # support 2: agreement over another area
        View(
            "run-a", "frame-0130", (ClaimSpec("pallet", confidence=0.7),), geometry=range(100, 120)
        ),
        View(
            "run-a", "frame-0131", (ClaimSpec("pallet", confidence=0.6),), geometry=range(100, 120)
        ),
        # support 3: only an abstention
        View("run-a", "frame-0140", (ClaimSpec("unknown"),), geometry=range(200, 220)),
    ]
    parts = build_scenario_parts(views, geometry_points=1_000)
    qualities = {
        spatial_id(view.run, view.frame, view.region): make_quality(
            spatial_id(view.run, view.frame, view.region), frame=view.frame, depth_median_m=8.0
        )
        for view in views
    }
    scores = {
        result: [make_score(views[0], 0, 0.31)]
        for result in [next(iter(r for r in parts.results if r == "run-a--frame-0120"))]
    }
    outcomes = tuple(
        FusionOutcome(
            support=support,
            evidence=accumulate_quality_aware_evidence(
                support,
                observations=parts.observations,
                grouping=parts.grouping,
                perception_results=parts.results,
                observation_qualities=qualities,
                semantic_scores=scores,
                point_representation_refs=[make_point_representation_ref(geometry_index=3)],
                policy=POLICY,
                code_version="test",
            ),
        )
        for support in parts.supports
    )
    return RunFixture(outcomes=outcomes, excluded=parts.excluded)


__all__ = ["LINEAGE", "POLICY", "RunFixture", "make_perception_run", "make_run_fixture"]

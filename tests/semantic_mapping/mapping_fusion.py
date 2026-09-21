"""Real fused evidence for Semantic Mapping tests, built through Semantic Fusion itself."""

from __future__ import annotations

from collections.abc import Sequence

from evidence_builders import ClaimSpec, View, build_scenario

from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    FusedEvidence,
    FusionSupport,
    accumulate_baseline_evidence,
)

__all__ = ["ClaimSpec", "View", "fuse"]


def fuse(
    views: Sequence[View], *, policy: BaselineAccumulationPolicy | None = None
) -> tuple[FusionSupport, FusedEvidence]:
    """Fuse the evidence of views that see the same geometry into one support."""
    scenario = build_scenario(views)
    evidence = accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        policy=policy,
        code_version="test",
    )
    return scenario.support, evidence

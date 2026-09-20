"""Five fusion runs over the same upstream evidence, differing only in policy or channels."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_builders import ClaimSpec, View, build_scenario_parts, make_score
from fusion_builders import (
    make_point_representation_ref,
    make_region_feature,
    result_id,
    spatial_id,
)
from fusion_run_fixtures import LINEAGE
from quality_builders import make_quality

from contextmap.evaluation import FusionArmRole, FusionStratificationProfile, ReferenceAnnotation
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    EvidenceChannel,
    FusionOutcome,
    QualityAwareAccumulationPolicy,
    QualityInput,
    QualityRamp,
    SemanticFusionRunId,
    SemanticFusionRunWriter,
    accumulate_baseline_evidence,
    accumulate_quality_aware_evidence,
)
from contextmap.sensor_association import ObservationQuality, SpatialObservationId
from contextmap.visual_perception import HypothesisRole

CLAIMS = EvidenceChannel.SEMANTIC_CLAIMS
ABSTAIN = frozenset({"unknown"})

PROFILE = FusionStratificationProfile(
    range_edges_m=(5.0, 15.0),
    visible_share_edges=(0.5,),
    support_density_edges=(0.05,),
    border_distance_edges_px=(30.0,),
    physical_observation_edges=(2.0, 3.0),
)


@dataclass(frozen=True)
class EvalFixture:
    root: Path
    runs: dict[str, Path]
    roles: dict[str, FusionArmRole]
    qualities: dict[SpatialObservationId, ObservationQuality]
    annotations: tuple[ReferenceAnnotation, ...]
    door_view_id: SpatialObservationId
    support_count: int


def _views() -> list[View]:
    features = (make_region_feature(feature="feature-0001"),)
    # Um segundo espaço de embedding em S2: os espaços devem continuar separados no relatório.
    clip = dataclasses.replace(
        make_region_feature(feature="feature-0002"), embedding_space_id="clip-vit-l14"
    )
    views = [
        # S1: two frames disagree; the first is interpreted twice (repeated inference)
        View(
            "run-a",
            "frame-0120",
            (ClaimSpec("door", confidence=0.9),),
            features=features,
            geometry=range(0, 20),
        ),
        View(
            "run-b",
            "frame-0120",
            (ClaimSpec("door", confidence=None),),
            features=features,
            geometry=range(0, 20),
        ),
        View(
            "run-a",
            "frame-0121",
            (ClaimSpec("cabinet", confidence=0.6),),
            features=features,
            geometry=range(0, 20),
        ),
        # S2: three frames agree, at growing distance
        View(
            "run-a",
            "frame-0130",
            (ClaimSpec("pallet", confidence=0.7),),
            features=features,
            geometry=range(100, 120),
        ),
        View(
            "run-a",
            "frame-0131",
            (ClaimSpec("pallet", confidence=0.6),),
            features=features,
            geometry=range(100, 120),
        ),
        View(
            "run-a",
            "frame-0132",
            (ClaimSpec("pallet", confidence=0.5),),
            features=features,
            geometry=range(100, 120),
        ),
        # S3: only an abstention
        View(
            "run-a",
            "frame-0140",
            (ClaimSpec("unknown"),),
            features=features,
            geometry=range(200, 220),
        ),
        # S4: one interpretation offering an alternative
        View(
            "run-a",
            "frame-0150",
            (ClaimSpec("door"), ClaimSpec("cabinet", role=HypothesisRole.ALTERNATIVE)),
            features=features,
            geometry=range(300, 320),
        ),
        View(
            "run-a", "frame-0151", (ClaimSpec("door"),), features=features, geometry=range(300, 320)
        ),
    ]
    return [
        dataclasses.replace(view, features=(*features, clip))
        if view.frame in {"frame-0130", "frame-0131", "frame-0132"}
        else view
        for view in views
    ]


_QUALITY: dict[tuple[str, str], dict[str, Any]] = {
    ("run-a", "frame-0120"): {
        "depth_median_m": 3.0,
        "visible_share": 0.9,
        "border_median_px": 60.0,
    },
    ("run-b", "frame-0120"): {
        "depth_median_m": 3.0,
        "visible_share": 0.9,
        "border_median_px": 60.0,
    },
    ("run-a", "frame-0121"): {
        "depth_median_m": 20.0,
        "visible_share": 0.4,
        "border_median_px": 10.0,
    },
    ("run-a", "frame-0130"): {"depth_median_m": 4.0, "visible_share": 0.9},
    ("run-a", "frame-0131"): {"depth_median_m": 12.0, "visible_share": 0.6},
    ("run-a", "frame-0132"): {"depth_median_m": 20.0, "visible_share": 0.3},
    ("run-a", "frame-0140"): {"depth_median_m": 6.0},
    ("run-a", "frame-0150"): {"depth_median_m": 30.0},
    ("run-a", "frame-0151"): {"depth_median_m": 30.0},
}

_POLICY = QualityAwareAccumulationPolicy(
    definitions_version="observation-quality-v1",
    ramps=(QualityRamp(quality_input=QualityInput.SUPPORT_DEPTH_MEDIAN_M, good=5.0, bad=25.0),),
    neutral_factor=0.5,
    baseline=BaselineAccumulationPolicy(abstention_labels=ABSTAIN),
)


def make_eval_fixture(root: Path, *, extra_frame: bool = False) -> EvalFixture:
    """Write the five arms; ``extra_frame`` changes the evidence base, to test drift."""
    views = _views()
    if extra_frame:
        views.append(View("run-a", "frame-0160", (ClaimSpec("box"),), geometry=range(400, 420)))
    parts = build_scenario_parts(views, geometry_points=1_000)
    qualities = {
        spatial_id(v.run, v.frame, v.region): make_quality(
            spatial_id(v.run, v.frame, v.region),
            frame=v.frame,
            **_QUALITY.get((v.run, v.frame), {"depth_median_m": 10.0}),
        )
        for v in views
    }
    scores = {result_id("run-a", "frame-0120"): [make_score(views[0], 0, 0.31)]}
    first = make_point_representation_ref(geometry_index=3)
    second = dataclasses.replace(
        make_point_representation_ref(representation_id="repr-0002", geometry_index=105),
        representation_space_id="sha256:space-ptv3",
    )
    structure = [first, second]

    def baseline(channels: frozenset[EvidenceChannel]) -> list[FusionOutcome]:
        policy = BaselineAccumulationPolicy(abstention_labels=ABSTAIN, channels=channels)
        return [
            FusionOutcome(
                support=support,
                evidence=accumulate_baseline_evidence(
                    support,
                    observations=parts.observations,
                    grouping=parts.grouping,
                    perception_results=parts.results,
                    semantic_scores=scores,
                    observation_quality_refs={},
                    point_representation_refs=structure,
                    policy=policy,
                ),
            )
            for support in parts.supports
        ]

    arms: dict[str, tuple[FusionArmRole, list[FusionOutcome]]] = {
        "baseline": (FusionArmRole.BASELINE_CONTROL, baseline(frozenset({CLAIMS}))),
        "quality_aware": (
            FusionArmRole.QUALITY_AWARE,
            [
                FusionOutcome(
                    support=support,
                    evidence=accumulate_quality_aware_evidence(
                        support,
                        observations=parts.observations,
                        grouping=parts.grouping,
                        perception_results=parts.results,
                        observation_qualities=qualities,
                        policy=_POLICY,
                    ),
                )
                for support in parts.supports
            ],
        ),
        "with_scores": (
            FusionArmRole.CHANNEL_ABLATION,
            baseline(frozenset({CLAIMS, EvidenceChannel.SEMANTIC_SCORES})),
        ),
        "with_visual": (
            FusionArmRole.CHANNEL_ABLATION,
            baseline(frozenset({CLAIMS, EvidenceChannel.VISUAL_FEATURES})),
        ),
        "with_visual_and_3d": (
            FusionArmRole.CHANNEL_ABLATION,
            baseline(
                frozenset(
                    {CLAIMS, EvidenceChannel.VISUAL_FEATURES, EvidenceChannel.POINT_REPRESENTATION}
                )
            ),
        ),
    }
    runs: dict[str, Path] = {}
    roles: dict[str, FusionArmRole] = {}
    for index, (arm, (role, outcomes)) in enumerate(arms.items(), start=1):
        writer = SemanticFusionRunWriter(
            workspace_root=root,
            sequence_name="sequence-0001",
            run_id=SemanticFusionRunId(f"fusion-run-{index:04d}"),
            run_index=index,
            selection_label="all-frames",
            policy_label=arm.replace("_", "-"),
            lineage=LINEAGE,
            code_version="test",
        )
        writer.write(outcomes, excluded=parts.excluded)
        runs[arm] = (
            root
            / "runs"
            / "semantic-fusion"
            / "sequence-0001"
            / f"run-{index:04d}__all-frames__{arm.replace('_', '-')}"
        )
        roles[arm] = role
    door_view = spatial_id("run-a", "frame-0120")
    annotations = (
        ReferenceAnnotation(spatial_observation_id=door_view, label="Door"),
        ReferenceAnnotation(
            spatial_observation_id=spatial_id("run-a", "frame-0130"), label="pallet"
        ),
        ReferenceAnnotation(
            spatial_observation_id=spatial_id("run-a", "frame-0150"), label="cabinet"
        ),
    )
    return EvalFixture(
        root=root,
        runs=runs,
        roles=roles,
        qualities=qualities,
        annotations=annotations,
        door_view_id=door_view,
        support_count=len(parts.supports),
    )


__all__ = ["PROFILE", "EvalFixture", "make_eval_fixture"]

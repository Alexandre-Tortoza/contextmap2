"""Stage 4 (real chain, real association): Semantic Fusion arms over the persisted upstream runs.

Every arm consumes the same association runs, perception results, geometric map and support policy;
only the fusion configuration (policy or evidence channels) changes. Thresholds are declared once at
the top of this file and are never retuned per arm. Correctness is N/A: there are no annotations.

Usage: s04_fusion.py [--association-root DIR] [--stage NAME]

BUNDLE (experiments/semantic-fusion-corridor-02-20260921): the only differences from the executed copy
are the ``CODE_SHA`` lookup and the ``import contextmap`` it uses (no fixed worktree path); see the
bundle README.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from common import CODE_SHA, SEQUENCE_NAME, VAL, dump_json, peak_rss_mb, sha256_file, stage_root
from fusion_inputs import UpstreamEvidence, default_association_root, load_upstream

import contextmap
from contextmap.evaluation import (
    FusionArmRole,
    FusionStratificationProfile,
    compare_semantic_fusion_reports,
    encode_semantic_fusion_comparison,
    encode_semantic_fusion_report,
    evaluate_semantic_fusion,
)
from contextmap.geometric_mapping import GeometricMapArtifactReader, GeometryReference, GeometrySource
from contextmap.point_representation import (
    CenteringMode,
    CoordinatePreparation,
    NeighborhoodMethod,
    PointRepresentationDebugLevel,
    PointRepresentationRunId,
    PointRepresentationRunReader,
    PointRepresentationRunWriter,
    RepresentationService,
    ScaleNormalization,
    SupportPolicy,
    SupportType,
)
from contextmap.point_representation import allocate_run_index as allocate_pr_index
from contextmap.point_representation.backends.geometric_descriptor import GeometricDescriptorEncoder
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    EvidenceChannel,
    FusedEvidence,
    FusionOutcome,
    FusionRunLineage,
    FusionSupport,
    GeometryOverlapSupportPolicy,
    PointRepresentationRef,
    QualityAwareAccumulationPolicy,
    QualityInput,
    QualityRamp,
    SemanticFusionRunId,
    SemanticFusionRunReader,
    SemanticFusionRunWriter,
    accumulate_baseline_evidence,
    accumulate_quality_aware_evidence,
    allocate_fusion_run_index,
    build_fusion_supports,
    group_by_physical_observation,
)
from contextmap.visual_perception import PerceptionRunId

# Código sob teste nas etapas de fusão e avaliação: HEAD do checkout que fornece o pacote `contextmap`
# (as etapas a montante usam o code_sha da seleção). No original era o HEAD de um worktree fixo.
CODE_SHA = subprocess.check_output(
    ["git", "-C", str(Path(contextmap.__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
    text=True,
).strip()

# --- Declared once, before any fusion outcome was looked at (AGENTS.md 5 / issue 121: no hidden retuning) --------
SUPPORT_POLICY = GeometryOverlapSupportPolicy(min_geometry_count=5, min_overlap=0.3)
SUPPORT_SWEEP = ((5, 0.1), (5, 0.2), (5, 0.3), (5, 0.5), (20, 0.3))  # structure-only sensitivity, never used to pick
ABSTENTION = frozenset({"unknown"})
NEAR_TIE_MARGIN = 0
# Quality ramps: (input, good, bad). visible_share is deliberately NOT ramped, see the report (degenerate on a
# dense accumulated map: its median is ~2e-4 because the footprint counts every occluded point behind the surface).
QUALITY_RAMPS = {
    "quality_aware_depth_border": (
        QualityRamp(quality_input=QualityInput.SUPPORT_DEPTH_MEDIAN_M, good=4.0, bad=30.0),
        QualityRamp(quality_input=QualityInput.BORDER_DISTANCE_MEDIAN_PX, good=60.0, bad=10.0),
    ),
    "quality_aware_depth_only": (
        QualityRamp(quality_input=QualityInput.SUPPORT_DEPTH_MEDIAN_M, good=4.0, bad=30.0),
    ),
}
NEUTRAL_FACTOR = 0.5
QUALITY_DEFINITIONS = "observation-quality-v1"
PROFILE = FusionStratificationProfile(
    range_edges_m=(3.0, 6.0, 12.0),
    visible_share_edges=(0.001, 0.05),
    support_density_edges=(0.002, 0.02),
    border_distance_edges_px=(20.0, 60.0, 120.0),
    physical_observation_edges=(2.0, 3.0, 5.0),
)
ANCHOR_CANDIDATES = 200  # geometry elements probed per support to pick the anchor nearest to the centroid
REPRESENTATION_SUPPORT = SupportPolicy(
    support_type=SupportType.NEIGHBORHOOD,
    method=NeighborhoodMethod.RADIUS,
    radius_m=0.5,
    k=None,
    max_neighbors=500,
    preparation=CoordinatePreparation(centering=CenteringMode.CENTER, scale_normalization=ScaleNormalization.NONE),
)

CLAIMS = EvidenceChannel.SEMANTIC_CLAIMS
SCORES = EvidenceChannel.SEMANTIC_SCORES
VISUAL = EvidenceChannel.VISUAL_FEATURES
STRUCTURE = EvidenceChannel.POINT_REPRESENTATION

parser = argparse.ArgumentParser()
parser.add_argument("--association-root", type=Path, default=default_association_root())
parser.add_argument("--stage", default="semantic_fusion")
args = parser.parse_args()
STAGE = args.stage
started = time.time()

up: UpstreamEvidence = load_upstream(args.association_root)
gm_dir = next((VAL / "geometric_mapping/runs/geometric-mapping/corridor-02").glob("run-0001__*"))
print(
    f"{len(up.association_dirs)} association runs, {len(up.observations)} spatial observations, "
    f"{len(up.perception_runs)} perception runs, {len(up.timestamps)} frames; HWM {peak_rss_mb()} MB",
    flush=True,
)


def bucket(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)

    def pick(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": pick(0.25),
        "median": statistics.median(ordered),
        "p75": pick(0.75),
        "p90": pick(0.9),
        "max": ordered[-1],
    }


def structure_of(build: Any, grouping: Any) -> dict[str, Any]:
    """Structural statistics of a support build: how much multi-view geometry consistency exists."""
    frame_of = {o.spatial_observation_id: o.source_observation_id for o in up.observations}
    physical = [len({frame_of[i] for i in s.spatial_observation_ids}) for s in build.supports]
    observations = [len(s.spatial_observation_ids) for s in build.supports]
    return {
        "supports": len(build.supports),
        "excluded_observations": len(build.excluded),
        "observations_per_support": bucket([float(v) for v in observations]),
        "physical_observations_per_support": bucket([float(v) for v in physical]),
        "supports_with_two_or_more_physical_observations": sum(1 for v in physical if v >= 2),
        "supports_with_one_physical_observation": sum(1 for v in physical if v == 1),
        "geometry_per_support": bucket([float(len(s.geometry_support)) for s in build.supports]),
        "largest_support_observations": max(observations, default=0),
        "largest_support_share_of_observations": (max(observations, default=0) / max(1, sum(observations))),
    }


with GeometricMapArtifactReader(gm_dir) as gm_reader:
    geometry: GeometrySource = gm_reader.geometry()

    # ---- 1. physical-observation grouping (repeated inference stays one physical observation) -----------------
    grouping = group_by_physical_observation(
        up.observations, selected_runs=up.perception_runs, acquisition_timestamps=up.timestamps
    )
    grouping_report = {
        "physical_observation_count": grouping.physical_observation_count,
        "inference_result_count": grouping.inference_result_count,
        "perception_run_count": grouping.perception_run_count,
        "inference_variant_count": grouping.inference_variant_count,
        "selected_run_ids": list(grouping.selected_run_ids),
    }
    print("grouping:", grouping_report, flush=True)

    # ---- 2. supports under the declared policy + structure-only sensitivity sweep -----------------------------
    t = time.time()
    build = build_fusion_supports(
        up.observations,
        geometry=geometry,
        acquisition_timestamps=up.timestamps,
        policy=SUPPORT_POLICY,
        code_version=CODE_SHA,
    )
    support_seconds = time.time() - t
    primary_structure = structure_of(build, grouping)
    print("supports:", json.dumps(primary_structure, default=str), flush=True)
    sweep = {}
    for min_count, min_overlap in SUPPORT_SWEEP:
        alt = build_fusion_supports(
            up.observations,
            geometry=geometry,
            acquisition_timestamps=up.timestamps,
            policy=GeometryOverlapSupportPolicy(min_geometry_count=min_count, min_overlap=min_overlap),
            code_version=CODE_SHA,
        )
        sweep[f"min_geometry={min_count},min_overlap={min_overlap}"] = structure_of(alt, grouping)

    # ---- 3. optional 3D channel: a real geometric-descriptor representation anchored inside each support -------
    def anchor_of(support: FusionSupport) -> GeometryReference:
        candidates = support.geometry_support
        stride = max(1, len(candidates) // ANCHOR_CANDIDATES)
        cx, cy, cz = support.centroid_m
        best = min(
            candidates[::stride],
            key=lambda ref: (
                sum(
                    (a - b) ** 2
                    for a, b in zip(geometry.get(ref).coordinates_m, (cx, cy, cz), strict=True)
                ),
                ref.geometry_id,
            ),
        )
        return best

    encoder = GeometricDescriptorEncoder(REPRESENTATION_SUPPORT)
    pr_root = stage_root("point_representation")
    pr_index = allocate_pr_index(workspace_root=pr_root, sequence_name=SEQUENCE_NAME)
    pr_run_id = PointRepresentationRunId(f"pr-geometric-descriptor-{STAGE}")
    anchors = [anchor_of(support) for support in build.supports]
    unique_anchors = list(dict.fromkeys(anchors))
    service = RepresentationService(geometry, encoder, run_id=pr_run_id, code_version=CODE_SHA)
    pr_writer = PointRepresentationRunWriter(
        workspace_root=pr_root,
        sequence_name=SEQUENCE_NAME,
        run_id=pr_run_id,
        run_index=pr_index,
        selection_label="support-anchors",
        backend_label="geometric-descriptor",
        geometric_map=geometry.geometric_map,
        space=encoder.representation_space(),
        encoder_identity=encoder.encoder_identity(),
        code_version=CODE_SHA,
    )
    t = time.time()
    structure_refs: list[PointRepresentationRef] = []
    for outcome in service.represent(unique_anchors):
        pr_writer.add(outcome)
        if hasattr(outcome, "representation"):
            rep = outcome.representation
            structure_refs.append(
                PointRepresentationRef(
                    representation_id=rep.representation_id,
                    run_id=pr_run_id,
                    representation_space_id=rep.representation_space_id,
                    geometry_reference=rep.geometry_reference,
                )
            )
    pr_manifest = pr_writer.finalize(metrics=service.metrics)
    pr_seconds = time.time() - t
    pr_dir = next(pr_root.rglob(f"run-{pr_index:04d}__*"))
    pr_problems = PointRepresentationRunReader(pr_dir).verify_integrity()
    print(
        f"point representation: {len(structure_refs)}/{len(unique_anchors)} represented in {pr_seconds:.1f}s, "
        f"integrity {pr_problems}",
        flush=True,
    )

    # ---- 4. arms ----------------------------------------------------------------------------------------------
    observations_by_id = {o.spatial_observation_id: o for o in up.observations}
    lineage_base = dict(
        sequence_artifact_id=up.sequence_artifact_id,
        geometric_map_id=geometry.geometric_map.map_id,
        association_run_ids=up.association_run_ids,
        perception_run_ids=tuple(sorted(PerceptionRunId(str(r.run_id)) for r in up.perception_runs)),
    )

    def baseline_policy(channels: set[EvidenceChannel]) -> BaselineAccumulationPolicy:
        return BaselineAccumulationPolicy(
            abstention_labels=ABSTENTION, near_tie_margin=NEAR_TIE_MARGIN, channels=frozenset(channels)
        )

    def baseline_arm(policy: BaselineAccumulationPolicy, order: list[FusionSupport] | None = None) -> list[FusedEvidence]:
        return [
            accumulate_baseline_evidence(
                support,
                observations=observations_by_id,
                grouping=grouping,
                perception_results=up.results,
                semantic_scores={} if SCORES in policy.channels else None,
                point_representation_refs=structure_refs if STRUCTURE in policy.channels else None,
                policy=policy,
                code_version=CODE_SHA,
            )
            for support in (order or build.supports)
        ]

    def quality_arm(ramps: tuple[QualityRamp, ...]) -> list[FusedEvidence]:
        policy = QualityAwareAccumulationPolicy(
            definitions_version=QUALITY_DEFINITIONS,
            ramps=ramps,
            neutral_factor=NEUTRAL_FACTOR,
            baseline=baseline_policy({CLAIMS}),
        )
        return [
            accumulate_quality_aware_evidence(
                support,
                observations=observations_by_id,
                grouping=grouping,
                perception_results=up.results,
                observation_qualities=up.qualities,
                policy=policy,
                code_version=CODE_SHA,
            )
            for support in build.supports
        ]

    arm_specs: list[tuple[str, FusionArmRole, Any]] = [
        ("baseline_uniform", FusionArmRole.BASELINE_CONTROL, lambda: baseline_arm(baseline_policy({CLAIMS}))),
        ("ch_claims_scores", FusionArmRole.CHANNEL_ABLATION, lambda: baseline_arm(baseline_policy({CLAIMS, SCORES}))),
        ("ch_claims_visual", FusionArmRole.CHANNEL_ABLATION, lambda: baseline_arm(baseline_policy({CLAIMS, VISUAL}))),
        (
            "ch_claims_visual_3d",
            FusionArmRole.CHANNEL_ABLATION,
            lambda: baseline_arm(baseline_policy({CLAIMS, VISUAL, STRUCTURE})),
        ),
        (
            "ch_all",
            FusionArmRole.CHANNEL_ABLATION,
            lambda: baseline_arm(baseline_policy({CLAIMS, SCORES, VISUAL, STRUCTURE})),
        ),
        *[
            (name, FusionArmRole.QUALITY_AWARE, (lambda r=ramps: quality_arm(r)))
            for name, ramps in QUALITY_RAMPS.items()
        ],
    ]

    fusion_root = stage_root(STAGE)
    lineage = FusionRunLineage(
        **lineage_base,
        point_representation_run_ids=(pr_run_id,),
    )
    run_dirs: dict[str, Path] = {}
    arm_runtime: dict[str, dict[str, Any]] = {}
    for name, role, make in arm_specs:
        t = time.time()
        evidences = make()
        outcomes = [FusionOutcome(support=s, evidence=e) for s, e in zip(build.supports, evidences, strict=True)]
        index = allocate_fusion_run_index(workspace_root=fusion_root, sequence_name=SEQUENCE_NAME)
        writer = SemanticFusionRunWriter(
            workspace_root=fusion_root,
            sequence_name=SEQUENCE_NAME,
            run_id=SemanticFusionRunId(f"fusion-{name}"),
            run_index=index,
            selection_label="ts-w336-20frames",
            policy_label=name.replace("_", "-"),
            lineage=lineage,
            code_version=CODE_SHA,
        )
        accumulate_s = time.time() - t
        t = time.time()
        writer.write(
            outcomes,
            excluded=build.excluded,
            runtime={"accumulate_seconds": round(accumulate_s, 2), "peak_rss_mb": peak_rss_mb()},
        )
        write_s = time.time() - t
        run_dir = next(fusion_root.rglob(f"run-{index:04d}__*"))
        run_dirs[name] = run_dir
        arm_runtime[name] = {"accumulate_seconds": round(accumulate_s, 2), "write_seconds": round(write_s, 2)}
        print(f"arm {name}: accumulate {accumulate_s:.1f}s write {write_s:.1f}s -> {run_dir.name}", flush=True)

    # ---- 5. evaluation and controlled comparison (one control, same evidence base) -----------------------------
    reports = {}
    for name, role, _ in arm_specs:
        reports[name] = evaluate_semantic_fusion(
            SemanticFusionRunReader(run_dirs[name]),
            arm_id=name,
            arm_role=role,
            profile=PROFILE,
            annotations=None,
            qualities=up.qualities,
        )
    comparison = compare_semantic_fusion_reports(list(reports.values()))

    # ---- 6. determinism and input-order invariance on real evidence ---------------------------------------------
    def contractual_digest(run_dir: Path) -> dict[str, str]:
        return {
            str(path.relative_to(run_dir)): sha256_file(path)
            for path in sorted((run_dir / "outputs").glob("*"))
            if path.is_file()
        }

    def rerun_baseline(label: str, shuffle_seed: int | None) -> Path:
        observations = list(up.observations)
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(observations)
        rebuilt = build_fusion_supports(
            observations,
            geometry=geometry,
            acquisition_timestamps=up.timestamps,
            policy=SUPPORT_POLICY,
            code_version=CODE_SHA,
        )
        regrouped = group_by_physical_observation(
            observations,
            selected_runs=list(reversed(up.perception_runs)) if shuffle_seed is not None else up.perception_runs,
            acquisition_timestamps=up.timestamps,
        )
        items = sorted(observations_by_id.items())
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(items)
        by_id = dict(items)
        evidences = [
            accumulate_baseline_evidence(
                s,
                observations=by_id,
                grouping=regrouped,
                perception_results=up.results,
                policy=baseline_policy({CLAIMS}),
                code_version=CODE_SHA,
            )
            for s in rebuilt.supports
        ]
        index = allocate_fusion_run_index(workspace_root=fusion_root, sequence_name=SEQUENCE_NAME)
        writer = SemanticFusionRunWriter(
            workspace_root=fusion_root,
            sequence_name=SEQUENCE_NAME,
            run_id=SemanticFusionRunId(f"fusion-{label}"),
            run_index=index,
            selection_label="ts-w336-20frames",
            policy_label=label,
            lineage=lineage,
            code_version=CODE_SHA,
        )
        writer.write(
            [FusionOutcome(support=s, evidence=e) for s, e in zip(rebuilt.supports, evidences, strict=True)],
            excluded=rebuilt.excluded,
        )
        return next(fusion_root.rglob(f"run-{index:04d}__*"))

    same_order = rerun_baseline("baseline-rerun-same-order", None)
    shuffled = rerun_baseline("baseline-rerun-shuffled-order", 1234)
    reference_digest = contractual_digest(run_dirs["baseline_uniform"])
    determinism = {
        "same_order_outputs_byte_identical": contractual_digest(same_order) == reference_digest,
        "shuffled_input_order_outputs_byte_identical": contractual_digest(shuffled) == reference_digest,
        "files_compared": sorted(reference_digest),
    }
    print("determinism:", determinism, flush=True)

    # ---- 6b. repeated inference on real evidence: one SAM2 run vs the same run plus its bit-identical rerun ----------
    def frames_of(support: FusionSupport) -> set[str]:
        return {str(observations_by_id[i].source_observation_id) for i in support.spatial_observation_ids}

    def geometry_key(support: FusionSupport) -> str:
        return hashlib.sha256("|".join(str(r.geometry_id) for r in support.geometry_support).encode()).hexdigest()

    def evidence_of(run_ids: set[str]) -> dict[str, tuple[FusionSupport, FusedEvidence]]:
        chosen = [o for o in up.observations if str(o.provenance.perception_run_id) in run_ids]
        runs = [r for r in up.perception_runs if str(r.run_id) in run_ids]
        regrouped = group_by_physical_observation(chosen, selected_runs=runs, acquisition_timestamps=up.timestamps)
        rebuilt = build_fusion_supports(
            chosen, geometry=geometry, acquisition_timestamps=up.timestamps, policy=SUPPORT_POLICY, code_version=CODE_SHA
        )
        return {
            geometry_key(support): (
                support,
                accumulate_baseline_evidence(
                    support,
                    observations=observations_by_id,
                    grouping=regrouped,
                    perception_results=up.results,
                    policy=baseline_policy({CLAIMS}),
                    code_version=CODE_SHA,
                ),
            )
            for support in rebuilt.supports
        }

    once = evidence_of({"vp-sam2-sel"})
    twice = evidence_of({"vp-sam2-sel", "vp-sam2-rerun"})
    matched = [key for key in twice if key in once]
    repeated_inference = {
        "single_run": "vp-sam2-sel",
        "duplicate_run": "vp-sam2-rerun (bit-identical rerun of the same backend and configuration)",
        "supports_single": len(once),
        "supports_with_duplicate": len(twice),
        "supports_with_identical_geometry": len(matched),
        "same_physical_frames": sum(1 for k in matched if frames_of(once[k][0]) == frames_of(twice[k][0])),
        "same_physical_observation_count": sum(
            1 for k in matched if once[k][1].physical_observation_count == twice[k][1].physical_observation_count
        ),
        "contributions_exactly_doubled": sum(
            1 for k in matched if len(twice[k][1].contributions) == 2 * len(once[k][1].contributions)
        ),
        "inference_results_exactly_doubled": sum(
            1 for k in matched if twice[k][1].inference_result_count == 2 * once[k][1].inference_result_count
        ),
    }
    print("repeated inference check:", repeated_inference, flush=True)

# ---- 7. report ---------------------------------------------------------------------------------------------------
report = {
    "label": "real chain (ExternalPose GT trajectory, real LiDAR map, MEI association, existing SAM2/SAM3 regions). "
    "correctness N/A: no annotations, no semantic claims in any perception run",
    "code_sha": CODE_SHA,
    "declared_before_outcomes": {
        "support_policy": dataclasses.asdict(SUPPORT_POLICY),
        "abstention_labels": sorted(ABSTENTION),
        "near_tie_margin": NEAR_TIE_MARGIN,
        "quality_ramps": {
            name: [(r.quality_input.value, r.good, r.bad) for r in ramps] for name, ramps in QUALITY_RAMPS.items()
        },
        "neutral_factor": NEUTRAL_FACTOR,
        "stratification_profile": dataclasses.asdict(PROFILE),
        "anchor_candidates_per_support": ANCHOR_CANDIDATES,
        "representation_support": {"radius_m": REPRESENTATION_SUPPORT.radius_m, "max_neighbors": REPRESENTATION_SUPPORT.max_neighbors},
    },
    "upstream": {
        "association_run_ids": list(up.association_run_ids),
        "association_dirs": [str(p) for p in up.association_dirs],
        "perception_dirs": {k: str(v) for k, v in up.perception_dirs.items()},
        "geometric_map_dir": str(gm_dir),
        "geometric_map_id": str(up.map_id),
        "sequence_artifact_id": up.sequence_artifact_id,
        "point_representation_run": {
            "dir": str(pr_dir),
            "run_id": str(pr_run_id),
            "represented": len(structure_refs),
            "requested": len(unique_anchors),
            "integrity_problems": pr_problems,
            "seconds": round(pr_seconds, 1),
        },
    },
    "grouping": grouping_report,
    "supports": {"primary": primary_structure, "build_seconds": round(support_seconds, 1), "sensitivity_structure_only": sweep},
    "arms": {
        name: {
            "run_dir": str(run_dirs[name]),
            "runtime": arm_runtime[name],
            "evaluation": encode_semantic_fusion_report(report_),
        }
        for name, report_ in reports.items()
    },
    "comparison": encode_semantic_fusion_comparison(comparison),
    "determinism": determinism,
    "repeated_inference_check": repeated_inference,
    "peak_rss_mb": peak_rss_mb(),
    "total_seconds": round(time.time() - started, 1),
}
target = dump_json(STAGE, "report.json", report)
print("report:", target, "peak RSS", peak_rss_mb(), "MB", "total", round(time.time() - started, 1), "s")

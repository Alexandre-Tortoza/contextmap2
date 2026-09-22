"""Ad hoc (driver-side, not part of the harness) real measurement of the visual evidence channel.

Question: does the region CLIP embedding carry multi-view information that agrees with the geometric
support? Pairs of regions of the SAME support seen in DIFFERENT physical frames are compared, by cosine
similarity inside one embedding space, against random pairs of regions of DIFFERENT supports and frames.
No label, claim or annotation is involved: it is an annotation-free objective proxy, labelled as such.

Usage: s05_visual_consistency.py [--stage semantic_fusion]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import PERCEPTION_BASE, VAL, dump_json

from contextmap.semantic_fusion import SemanticFusionRunReader
from contextmap.visual_perception import FeatureScope, PerceptionRunReader

parser = argparse.ArgumentParser()
parser.add_argument("--stage", default="semantic_fusion")
args = parser.parse_args()
root = VAL / args.stage
run_dir = next(root.glob("runs/semantic-fusion/corridor-02/run-*__ts-w336-20frames__ch-claims-visual"))
reader = SemanticFusionRunReader(run_dir)
perception_dirs = {
    json.loads((p / "manifest.json").read_text())["run_id"]: p for p in PERCEPTION_BASE.glob("run-*")
}
stores = {
    str(run_id): PerceptionRunReader(perception_dirs[str(run_id)]).feature_store()
    for run_id in reader.manifest.lineage.perception_run_ids
}

rng = random.Random(20260921)
# Fator de contribuição do braço ciente de qualidade (declarado a priori): só para estratificar os pares.
weighted_dir = next(root.glob("runs/semantic-fusion/corridor-02/run-*__ts-w336-20frames__quality-aware-depth-border"))
factor_of: dict[str, float] = {}
for outcome in SemanticFusionRunReader(weighted_dir).iter_outcomes():
    assert outcome.evidence.weighting is not None
    for weight in outcome.evidence.weighting.contributions:
        factor_of[str(weight.contribution_id)] = weight.factor

by_support: dict[str, list[tuple[str, np.ndarray, float]]] = defaultdict(list)
space_of: dict[str, str] = {}
for outcome in reader.iter_outcomes():
    for contribution in outcome.evidence.contributions:
        # Uma só run de SAM2: vp-sam2-rerun repete a inferência com embeddings bit a bit idênticos (repetição
        # de inferência não é uma segunda vista), então contá-la duplicaria cada par.
        if str(contribution.perception_run_id) != "vp-sam2-sel":
            continue
        for ref in contribution.visual_feature_refs:
            if ref.scope is not FeatureScope.REGION:
                continue
            vector = stores[str(contribution.perception_run_id)].load(
                contribution.physical_observation_id, ref.feature_id
            ).astype(np.float64)
            vector = vector / np.linalg.norm(vector)
            by_support[str(outcome.support.fusion_support_id)].append(
                (
                    str(contribution.physical_observation_id),
                    vector,
                    factor_of[str(contribution.contribution_id)],
                )
            )
            space_of[ref.embedding_space_id] = ref.embedding_space_id
assert len(space_of) == 1, f"one embedding space expected, got {sorted(space_of)}"

within: list[float] = []
within_factor: list[float] = []
for members in by_support.values():
    pairs = [
        (a, b)
        for a, b in itertools.combinations(members, 2)
        if a[0] != b[0]  # distinct physical frames only: repeated inference is not a second view
    ]
    rng.shuffle(pairs)
    for a, b in pairs[:200]:
        within.append(float(a[1] @ b[1]))
        within_factor.append(min(a[2], b[2]))

flat = [(support, frame, vector) for support, members in by_support.items() for frame, vector, _ in members]
null: list[float] = []
attempts = 0
while len(null) < len(within) and attempts < 50 * max(1, len(within)):
    attempts += 1
    (sa, fa, va), (sb, fb, vb) = rng.sample(flat, 2)
    if sa != sb and fa != fb:
        null.append(float(va @ vb))


def summarize(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda p: ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]  # noqa: E731
    return {"count": len(ordered), "p10": pick(0.1), "median": statistics.median(ordered), "p90": pick(0.9)}


auc = None
if within and null:
    sorted_null = np.sort(np.asarray(null))
    w = np.asarray(within)
    below = np.searchsorted(sorted_null, w, side="left")
    equal = np.searchsorted(sorted_null, w, side="right") - below
    auc = float((below + 0.5 * equal).mean() / len(sorted_null))
by_factor: dict[str, list[float]] = {"factor_0": [], "0<factor<0.5": [], "0.5<=factor<1": [], "factor_1": []}
for cosine, factor in zip(within, within_factor, strict=True):
    key = (
        "factor_0"
        if factor == 0.0
        else "0<factor<0.5"
        if factor < 0.5
        else "0.5<=factor<1"
        if factor < 1.0
        else "factor_1"
    )
    by_factor[key].append(cosine)
result = {
    "label": "ad hoc driver measurement (real data, annotation-free proxy); NOT a harness metric",
    "same_support_cross_frame_cosine_by_min_pair_factor": {k: summarize(v) for k, v in by_factor.items()},
    "factor_source": "quality_aware_depth_border arm (declared a priori); the factor only stratifies pairs here",
    "embedding_space_id": next(iter(space_of)),
    "supports_with_region_features": len(by_support),
    "supports_with_two_or_more_frames": sum(1 for m in by_support.values() if len({item[0] for item in m}) >= 2),
    "same_support_cross_frame_cosine": summarize(within),
    "different_support_cross_frame_cosine": summarize(null),
    "probability_same_support_pair_more_similar_than_random_pair": auc,
    "single_run_used": "vp-sam2-sel (vp-sam2-rerun duplicates it bit for bit; vp-sam3-building has no region features)",
    "note": "1.0 = perfect separation, 0.5 = the region embedding carries no information about the geometric support",
}
print(json.dumps(result, indent=1))
dump_json(args.stage, "visual_consistency.json", result)

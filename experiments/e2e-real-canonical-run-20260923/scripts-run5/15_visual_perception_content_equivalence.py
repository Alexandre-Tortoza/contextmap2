"""Real content comparison of two independent PerceptionRunArtifact reruns over the SAME 20 real
corridor-02 frames (PR #438 review, fifth round): completing scenario 1.0.4's
reproducibility.rerun_equivalence gate requires a real repetition of the perception part of the
canonical pipeline, not just reusing the one real PerceptionRunArtifact this campaign has always
cited.

Unlike the ContextMap comparator, no geometry-based canonicalization is needed here: both runs
process the exact same real SourceObservations (the same 20 frame ids from the same real
SequenceArtifact), so source_observation_id is already a stable, run-independent join key.

Canonicalization, deliberately excluding raw floating-point payloads: SAM2/DINOv2/CLIP embeddings
are real tensors whose bit-for-bit reproducibility across independent GPU processes is not
guaranteed even under identical code/config (CUDA kernels are not required to be deterministic
unless explicitly configured to be, which this pipeline does not do) -- comparing them exactly
would conflate that known, orthogonal GPU non-determinism with the actual claim under test: does
this pipeline discover the same regions and produce the same semantic content. So this compares:

- per-frame region count, and each region's bounding box + region_kind + is_accepted (real,
  deterministic SAM2 output; embeddings are not compared);
- per-frame, per-region-or-scene semantic claims: (hypothesis, role, category, confidence),
  canonicalized by (source_observation_id, region_id or None) -- this is the actual output a
  ContextMap eventually consumes, and Qwen's generation is exactly the part whose determinism
  was, until this round, a documented assumption rather than measured evidence.
"""

from __future__ import annotations

from pathlib import Path

from contextmap.visual_perception import PerceptionRunReader

WORKSPACE = Path(__file__).resolve().parents[3] / (
    "outputs/e2e-real/visual_perception/workspace/corridor-02"
)


def _claim_key(claim: object) -> tuple[str, object, str, str, object, object]:
    return (
        str(claim.source_observation_id),  # type: ignore[attr-defined]
        claim.region_id,  # type: ignore[attr-defined]
        claim.hypothesis,  # type: ignore[attr-defined]
        claim.role.value,  # type: ignore[attr-defined]
        claim.category,  # type: ignore[attr-defined]
        claim.confidence,  # type: ignore[attr-defined]
    )


def main() -> None:
    dir_a = WORKSPACE / "run-0001/visual_perception"
    dir_b = WORKSPACE / "run-0002/visual_perception"
    reader_a, reader_b = PerceptionRunReader(dir_a), PerceptionRunReader(dir_b)
    results_a = {result.source_observation_id: result for result in reader_a.list_results()}
    results_b = {result.source_observation_id: result for result in reader_b.list_results()}

    frame_ids = sorted(set(results_a) | set(results_b))
    print(f"comparing {len(frame_ids)} real frames: {dir_a} vs {dir_b}")

    region_mismatches = 0
    claim_only_a = 0
    claim_only_b = 0
    claim_match = 0
    per_frame_rows: list[dict[str, object]] = []
    for frame_id in frame_ids:
        result_a, result_b = results_a.get(frame_id), results_b.get(frame_id)
        if result_a is None or result_b is None:
            region_mismatches += 1
            print(f"  [{frame_id}] MISSING on one side (A={result_a is not None} B={result_b is not None})")
            continue
        regions_a = [(r.bounding_box, r.region_kind, r.is_accepted) for r in result_a.regions]
        regions_b = [(r.bounding_box, r.region_kind, r.is_accepted) for r in result_b.regions]
        region_ok = regions_a == regions_b

        claims_a = {_claim_key(c) for c in result_a.claims}
        claims_b = {_claim_key(c) for c in result_b.claims}
        only_a, only_b, both = claims_a - claims_b, claims_b - claims_a, claims_a & claims_b
        claim_only_a += len(only_a)
        claim_only_b += len(only_b)
        claim_match += len(both)
        if not region_ok:
            region_mismatches += 1
        row = {
            "frame": frame_id, "region_count_a": len(regions_a), "region_count_b": len(regions_b),
            "regions_identical": region_ok, "claims_a": len(claims_a), "claims_b": len(claims_b),
            "claims_matching": len(both), "claims_only_a": len(only_a), "claims_only_b": len(only_b),
        }
        per_frame_rows.append(row)
        print(
            f"  [{frame_id}] regions: {'OK' if region_ok else 'MISMATCH'} "
            f"({len(regions_a)}/{len(regions_b)}) | claims: match={len(both)} "
            f"only_a={len(only_a)} only_b={len(only_b)}"
        )

    print()
    print(f"region mismatches (frames): {region_mismatches} / {len(frame_ids)}")
    print(f"claims: matching={claim_match} only_in_A={claim_only_a} only_in_B={claim_only_b}")
    total_claims = claim_match + claim_only_a + claim_only_b
    agreement = claim_match / total_claims if total_claims else 1.0
    print(f"claim agreement rate: {agreement:.1%} ({claim_match}/{total_claims})")


if __name__ == "__main__":
    main()

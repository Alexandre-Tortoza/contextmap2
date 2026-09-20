"""The baseline multi-view semantic evidence accumulation policy.

``baseline-evidence-accumulation-v1`` turns the spatial observations of one
:class:`FusionSupport` into one :class:`FusedEvidence`. It is the canonical control
arm: transparent, deterministic and deliberately conservative.

What it does:

* every spatial observation of the support is one :class:`EvidenceContribution`, which
  references its claims, scorer outputs, features and, when provided, its observation
  quality;
* claims are grouped into hypotheses by their **label key**: the text under Unicode NFKC
  normalisation, case-folded, with whitespace collapsed. That is a typographic rule only:
  ``"Pallet"`` and ``"pallet "`` are one hypothesis, ``"pallet"`` and ``"wooden pallet"``
  are two, and no synonym, plural or taxonomy is assumed. The label shown is the most
  common whitespace-collapsed spelling, ties broken by the smallest;
* each claim is listed under every hypothesis with its stance. A claim that proposes the
  hypothesis is ``SUPPORTING``. Any other claim is ``AMBIGUOUS`` when it comes from the same
  contribution as a supporting claim (alternatives of one interpretation) or has the
  ``ALTERNATIVE`` role, and ``CONFLICTING`` otherwise;
* each claim carries typed signals: its own confidence (``None`` when unscored, never
  zero) and one signal per scorer output. Nothing is averaged or combined;
* hypotheses are numbered by label key, so the order implies no ranking. The number of
  independent supporters is a *count of distinct physical observations*
  (:meth:`FusedEvidence.supporting_physical_observations`), so repeated inference over
  one frame and one claim over many geometry points each count once.

What it deliberately does not do: weight by observation quality (the references are kept,
never used), reason about abstention or unknown labels, build uncertainty records, pick a
winner, infer entity identity, or apply prior knowledge.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion.grouping import PhysicalObservationGrouping
from contextmap.semantic_fusion.models import (
    EvidenceContribution,
    EvidenceContributionId,
    EvidenceStance,
    FusedEvidence,
    FusedEvidenceProvenance,
    FusedHypothesis,
    FusedHypothesisId,
    FusionSupport,
    HypothesisEvidence,
    ObservationQualityRef,
    PhysicalObservationGroup,
    PointRepresentationRef,
    ScoreReference,
    SupportSignal,
    SupportSignalKind,
    _producer_key,
    _time_bounds_of,
    evidence_contribution_id_for,
    fused_evidence_id_for,
)
from contextmap.sensor_association import SpatialObservation, SpatialObservationId
from contextmap.visual_perception import (
    ClaimId,
    HypothesisRole,
    PerceptionResult,
    PerceptionResultId,
    SemanticClaim,
    SemanticSupport,
)

BASELINE_ACCUMULATION_POLICY_ID = "baseline-evidence-accumulation-v1"
"""Versioned identity of the baseline policy, including its label-key rule."""


@dataclass(frozen=True)
class _Claim:
    """One claim of one contribution, with the signals that came with it."""

    contribution_id: EvidenceContributionId
    claim: SemanticClaim
    key: str
    signals: tuple[SupportSignal, ...]


def label_key(text: str) -> str:
    """Reduce a hypothesis text to the key under which the baseline compares labels.

    Args:
        text: The hypothesis text as proposed.

    Returns:
        The text under NFKC normalisation, case-folded, with whitespace collapsed. Only
        typographic differences are removed; no synonym or taxonomy is applied.
    """
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def accumulate_baseline_evidence(
    support: FusionSupport,
    *,
    observations: Mapping[SpatialObservationId, SpatialObservation],
    grouping: PhysicalObservationGrouping,
    perception_results: Mapping[PerceptionResultId, PerceptionResult],
    semantic_scores: Mapping[PerceptionResultId, Sequence[SemanticSupport]] | None = None,
    observation_quality_refs: Mapping[SpatialObservationId, ObservationQualityRef] | None = None,
    point_representation_refs: Iterable[PointRepresentationRef] = (),
    code_version: str | None = None,
) -> FusedEvidence:
    """Accumulate the evidence of one support under the baseline policy.

    Evidence channels are opt-in: scorer outputs, quality references and structural
    references take part only when the caller passes them.

    Args:
        support: The support to accumulate over.
        observations: The spatial observations, by identity; every observation of the
            support must be present, and any other is ignored.
        grouping: The physical-observation grouping of the same observations, which
            supplies the acquisition time of each physical observation.
        perception_results: The perception results that own the claims, by identity.
        semantic_scores: Scorer outputs per perception result. A score for a claim that is
            not attached to an observation of the support is ignored.
        observation_quality_refs: References to the measured quality of each observation.
            The baseline keeps them and never weights by them.
        point_representation_refs: Static 3D structure. Only references anchored inside the
            support are kept, each once.
        code_version: Code revision to record in the provenance, when known.

    Returns:
        The fused evidence; it does not depend on the order of any input.

    Raises:
        ValueError: If an observation or perception result is missing, a claim is not in its
            result, an observation belongs to another map or has geometry outside the
            support, the grouping does not cover a physical observation or does not list an
            observation, or a scorer scored one claim twice.
    """
    scores = {} if semantic_scores is None else semantic_scores
    quality_refs = {} if observation_quality_refs is None else observation_quality_refs

    contributions: list[EvidenceContribution] = []
    claims: list[_Claim] = []
    for observation_id in support.spatial_observation_ids:
        observation = _observation(support, observations, observation_id)
        result = perception_results.get(observation.perception_result_id)
        if result is None:
            raise ValueError(
                f"no perception result {observation.perception_result_id!r} for spatial "
                f"observation {observation_id!r}"
            )
        contribution, view_claims = _contribution(
            support,
            observation,
            result,
            scores.get(observation.perception_result_id, ()),
            quality_refs.get(observation_id),
        )
        contributions.append(contribution)
        claims.extend(view_claims)

    groups = _groups(contributions, grouping)
    support_geometry = set(support.geometry_support)
    structure = sorted(
        {ref for ref in point_representation_refs if ref.geometry_reference in support_geometry},
        key=lambda ref: (ref.run_id, ref.representation_id),
    )
    return FusedEvidence(
        fused_evidence_id=fused_evidence_id_for(fusion_support_id=support.fusion_support_id),
        fusion_support_id=support.fusion_support_id,
        physical_observation_groups=groups,
        contributions=tuple(sorted(contributions, key=lambda item: item.contribution_id)),
        hypotheses=_hypotheses(claims),
        point_representation_refs=tuple(structure),
        temporal_summary=_time_bounds_of(
            [group.acquisition_timestamp for group in groups], owner="the physical observations"
        ),
        provenance=FusedEvidenceProvenance(
            grouping_policy_id=grouping.grouping_policy_id,
            fusion_policy_id=BASELINE_ACCUMULATION_POLICY_ID,
            code_version=code_version,
        ),
    )


def _observation(
    support: FusionSupport,
    observations: Mapping[SpatialObservationId, SpatialObservation],
    observation_id: SpatialObservationId,
) -> SpatialObservation:
    observation = observations.get(observation_id)
    if observation is None:
        raise ValueError(
            f"no spatial observation {observation_id!r} for support {support.fusion_support_id!r}"
        )
    if observation.provenance.geometric_map_id != support.geometric_map_id:
        raise ValueError(
            f"spatial observation {observation_id!r} references map "
            f"{observation.provenance.geometric_map_id!r}, but the support is over "
            f"{support.geometric_map_id!r}"
        )
    outside = set(observation.geometry_support) - set(support.geometry_support)
    if outside:
        raise ValueError(
            f"spatial observation {observation_id!r} has {len(outside)} geometry elements "
            f"outside support {support.fusion_support_id!r}"
        )
    return observation


def _contribution(
    support: FusionSupport,
    observation: SpatialObservation,
    result: PerceptionResult,
    result_scores: Sequence[SemanticSupport],
    quality: ObservationQualityRef | None,
) -> tuple[EvidenceContribution, list[_Claim]]:
    contribution_id = evidence_contribution_id_for(
        fusion_support_id=support.fusion_support_id,
        spatial_observation_id=observation.spatial_observation_id,
    )
    result_claims = {claim.claim_id: claim for claim in result.claims}
    scores_of_claim = _scores_by_claim(observation, result_scores)
    claim_refs = sorted(observation.semantic_claim_refs, key=lambda ref: ref.claim_id)
    selected: list[SemanticClaim] = []
    for ref in claim_refs:
        claim = result_claims.get(ref.claim_id)
        if claim is None:
            raise ValueError(
                f"perception result {result.result_id!r} does not contain claim {ref.claim_id!r} "
                f"of spatial observation {observation.spatial_observation_id!r}"
            )
        selected.append(claim)

    score_refs = sorted(
        (
            ScoreReference(claim_id=claim.claim_id, scorer=score.provenance)
            for claim in selected
            for score in scores_of_claim.get(claim.claim_id, ())
        ),
        key=lambda ref: (ref.claim_id, *_producer_key(ref.scorer)),
    )
    contribution = EvidenceContribution(
        contribution_id=contribution_id,
        physical_observation_id=observation.source_observation_id,
        perception_result_id=observation.perception_result_id,
        perception_run_id=observation.provenance.perception_run_id,
        spatial_observation_id=observation.spatial_observation_id,
        region_id=observation.region_id,
        geometry_support=observation.geometry_support,
        claim_refs=tuple(claim_refs),
        score_refs=tuple(score_refs),
        visual_feature_refs=tuple(
            sorted(observation.visual_feature_refs, key=lambda ref: ref.feature_id)
        ),
        observation_quality=quality,
    )
    view_claims = [
        _Claim(
            contribution_id=contribution_id,
            claim=claim,
            key=label_key(claim.hypothesis),
            signals=_signals(claim, scores_of_claim.get(claim.claim_id, ())),
        )
        for claim in selected
    ]
    return contribution, view_claims


def _scores_by_claim(
    observation: SpatialObservation, result_scores: Sequence[SemanticSupport]
) -> dict[ClaimId, list[SemanticSupport]]:
    by_claim: dict[ClaimId, list[SemanticSupport]] = {}
    for score in result_scores:
        by_claim.setdefault(score.claim_id, []).append(score)
    for claim_id, claim_scores in by_claim.items():
        scorers = [_producer_key(score.provenance) for score in claim_scores]
        if len(set(scorers)) != len(scorers):
            raise ValueError(
                f"more than one score from the same scorer for claim {claim_id!r} of spatial "
                f"observation {observation.spatial_observation_id!r}"
            )
    return by_claim


def _signals(claim: SemanticClaim, scores: Sequence[SemanticSupport]) -> tuple[SupportSignal, ...]:
    signals = [
        SupportSignal(
            kind=SupportSignalKind.CLAIM_CONFIDENCE,
            producer=claim.provenance.backend,
            value=claim.confidence,
        ),
        *(
            SupportSignal(
                kind=SupportSignalKind.SCORER_SUPPORT,
                producer=score.provenance,
                value=score.support_score,
            )
            for score in scores
        ),
    ]
    return tuple(sorted(signals, key=lambda item: (item.kind.value, *_producer_key(item.producer))))


def _groups(
    contributions: Sequence[EvidenceContribution], grouping: PhysicalObservationGrouping
) -> tuple[PhysicalObservationGroup, ...]:
    grouped = {group.physical_observation_id: group for group in grouping.groups}
    by_frame: dict[SourceObservationId, list[EvidenceContribution]] = {}
    for contribution in contributions:
        by_frame.setdefault(contribution.physical_observation_id, []).append(contribution)
    groups: list[PhysicalObservationGroup] = []
    for frame in sorted(by_frame):
        placed = grouped.get(frame)
        if placed is None:
            raise ValueError(f"the grouping does not cover physical observation {frame!r}")
        members = by_frame[frame]
        for member in members:
            if member.spatial_observation_id not in placed.spatial_observation_ids:
                raise ValueError(
                    f"the grouping does not list spatial observation "
                    f"{member.spatial_observation_id!r} under physical observation {frame!r}"
                )
        groups.append(
            PhysicalObservationGroup(
                physical_observation_id=frame,
                acquisition_timestamp=placed.acquisition_timestamp,
                spatial_observation_ids=tuple(
                    sorted(member.spatial_observation_id for member in members)
                ),
                perception_result_ids=tuple(
                    sorted({member.perception_result_id for member in members})
                ),
                perception_run_ids=tuple(sorted({member.perception_run_id for member in members})),
            )
        )
    return tuple(groups)


def _hypotheses(claims: Sequence[_Claim]) -> tuple[FusedHypothesis, ...]:
    ordered = sorted(claims, key=lambda item: (item.contribution_id, item.claim.claim_id))
    by_key: dict[str, list[_Claim]] = {}
    for item in ordered:
        by_key.setdefault(item.key, []).append(item)
    hypotheses: list[FusedHypothesis] = []
    for number, key in enumerate(sorted(by_key), start=1):
        supporters = {item.contribution_id for item in by_key[key]}
        hypotheses.append(
            FusedHypothesis(
                hypothesis_id=FusedHypothesisId(f"hypothesis-{number:04d}"),
                label=_spelling(by_key[key]),
                evidence=tuple(
                    HypothesisEvidence(
                        contribution_id=item.contribution_id,
                        claim_id=item.claim.claim_id,
                        stance=_stance(item, key, supporters),
                        role=item.claim.role,
                        signals=item.signals,
                    )
                    for item in ordered
                ),
            )
        )
    return tuple(hypotheses)


def _stance(
    item: _Claim, key: str, supporting_contributions: set[EvidenceContributionId]
) -> EvidenceStance:
    if item.key == key:
        return EvidenceStance.SUPPORTING
    if (
        item.contribution_id in supporting_contributions
        or item.claim.role is HypothesisRole.ALTERNATIVE
    ):
        return EvidenceStance.AMBIGUOUS
    return EvidenceStance.CONFLICTING


def _spelling(supporters: Sequence[_Claim]) -> str:
    """The most common whitespace-collapsed spelling among the claims of a label key."""
    counts = Counter(" ".join(item.claim.hypothesis.split()) for item in supporters)
    return min(counts, key=lambda spelling: (-counts[spelling], spelling))

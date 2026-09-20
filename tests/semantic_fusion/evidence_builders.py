"""Scenario builders: perception evidence, scores and a support built from real observations."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field

from fusion_builders import MAP_ID, make_interpreter, make_scorer, result_id
from geometry_fake import InMemoryGeometrySource
from observation_builders import make_perception_run, make_spatial_observation, timestamps

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import (
    EvidenceReference,
    FusionSupport,
    GeometryOverlapSupportPolicy,
    PhysicalObservationGrouping,
    build_fusion_supports,
    evidence_contribution_id_for,
    group_by_physical_observation,
)
from contextmap.sensor_association import (
    SemanticClaimRef,
    SpatialObservation,
    SpatialObservationId,
    spatial_observation_id_for,
)
from contextmap.visual_perception import (
    BackendProvenance,
    BoundingBox2D,
    ClaimId,
    HypothesisRole,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    Region2D,
    RegionId,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticSupport,
)

_REGION_BACKEND = BackendProvenance(
    backend_id="fake_region_discovery",
    capability="region_discovery",
    provider="fake",
    model="fake",
    version="0",
)


@dataclass(frozen=True)
class ClaimSpec:
    """One claim of one view, described by what fusion reads."""

    label: str
    role: HypothesisRole = HypothesisRole.PRIMARY
    confidence: float | None = None
    claim_id: str | None = None


@dataclass(frozen=True)
class View:
    """One region of one frame under one run, with the claims the interpreter made."""

    run: str
    frame: str
    claims: Sequence[ClaimSpec] = ()
    region: str = "region-0001"
    geometry: Sequence[int] = tuple(range(20))
    interpreter: BackendProvenance = field(default_factory=make_interpreter)


@dataclass(frozen=True)
class Scenario:
    """Everything the baseline accumulation reads, coherent by construction."""

    support: FusionSupport
    observations: dict[SpatialObservationId, SpatialObservation]
    grouping: PhysicalObservationGrouping
    results: dict[PerceptionResultId, PerceptionResult]


def claim_id_of(view: View, index: int) -> ClaimId:
    spec = view.claims[index]
    return ClaimId(spec.claim_id or f"{view.run}--{view.frame}--claim-{index:04d}")


def make_claim(view: View, index: int) -> SemanticClaim:
    spec = view.claims[index]
    return SemanticClaim(
        claim_id=claim_id_of(view, index),
        source_observation_id=SourceObservationId(view.frame),
        perception_result_id=result_id(view.run, view.frame),
        hypothesis=spec.label,
        role=spec.role,
        provenance=SemanticInferenceProvenance(
            backend=view.interpreter,
            task_identity="task",
            prompt_template_id="prompt-v1",
            output_schema_version="schema-v1",
        ),
        confidence=spec.confidence,
        region_id=RegionId(view.region),
    )


def make_score(
    view: View, index: int, value: float, *, scorer: str = "clip_scorer"
) -> SemanticSupport:
    return SemanticSupport(
        claim_id=claim_id_of(view, index),
        support_score=value,
        provenance=make_scorer(scorer),
    )


def build_scenario(
    views: Sequence[View], *, geometry_points: int = 1_000, min_overlap: float = 0.5
) -> Scenario:
    """Observations, results, grouping and the single support that joins them."""
    observations: dict[SpatialObservationId, SpatialObservation] = {}
    claims_of_result: dict[PerceptionResultId, list[SemanticClaim]] = {}
    regions_of_result: dict[PerceptionResultId, set[str]] = {}
    for view in views:
        claims = [make_claim(view, index) for index in range(len(view.claims))]
        observation = make_spatial_observation(
            view.run, view.frame, view.region, support=tuple(view.geometry)
        )
        observation = _with_claims(observation, claims)
        observations[observation.spatial_observation_id] = observation
        key = result_id(view.run, view.frame)
        claims_of_result.setdefault(key, []).extend(claims)
        regions_of_result.setdefault(key, set()).add(view.region)

    runs = sorted({view.run: view for view in views}.values(), key=lambda view: view.run)
    frames = sorted({view.frame for view in views})
    grouping = group_by_physical_observation(
        observations.values(),
        selected_runs=[make_perception_run(v.run, interpreter=v.interpreter) for v in runs],
        acquisition_timestamps=timestamps(frames),
    )
    build = build_fusion_supports(
        observations.values(),
        geometry=InMemoryGeometrySource(
            MAP_ID, {i: (i * 0.1, 0.0, 0.0) for i in range(geometry_points)}
        ),
        acquisition_timestamps=timestamps(frames),
        policy=GeometryOverlapSupportPolicy(min_geometry_count=1, min_overlap=min_overlap),
    )
    assert len(build.supports) == 1, "the scenario must be a single support"
    results = {
        key: _result(key, claims, regions_of_result[key])
        for key, claims in claims_of_result.items()
    }
    return Scenario(
        support=build.supports[0], observations=observations, grouping=grouping, results=results
    )


def _with_claims(
    observation: SpatialObservation, claims: Sequence[SemanticClaim]
) -> SpatialObservation:
    refs = tuple(
        sorted((SemanticClaimRef(claim_id=c.claim_id) for c in claims), key=lambda r: r.claim_id)
    )
    return dataclasses.replace(observation, semantic_claim_refs=refs)


def _result(
    key: PerceptionResultId, claims: Sequence[SemanticClaim], regions: set[str]
) -> PerceptionResult:
    first = claims[0] if claims else None
    frame = first.source_observation_id if first else SourceObservationId(str(key).split("--")[-1])
    run = PerceptionRunId(str(key).split("--")[0])
    return PerceptionResult(
        result_id=key,
        source_observation_id=frame,
        run_id=run,
        sequence_artifact_id="sequence-0001",
        created_at="2026-01-01T00:00:00Z",
        regions=tuple(
            Region2D(
                region_id=RegionId(region),
                bounding_box=BoundingBox2D(x=0.0, y=0.0, width=10.0, height=10.0),
                provenance=_REGION_BACKEND,
            )
            for region in sorted(regions)
        ),
        claims=tuple(claims),
    )


def reference_to(scenario: Scenario, view: View, index: int | None = 0) -> EvidenceReference:
    """The reference that names one claim of one view, or the view itself when ``index`` is None."""
    return EvidenceReference(
        contribution_id=evidence_contribution_id_for(
            fusion_support_id=scenario.support.fusion_support_id,
            spatial_observation_id=spatial_observation_id_for(
                perception_result_id=result_id(view.run, view.frame),
                region_id=RegionId(view.region),
            ),
        ),
        claim_id=None if index is None else claim_id_of(view, index),
    )

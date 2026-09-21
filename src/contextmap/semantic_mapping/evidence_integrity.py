"""Reference integrity and provenance traversal for entity evidence.

An entity only *references* its evidence, so the references must be checkable. This module
validates them against the persisted artifacts they name, and walks them, without ever loading a
perception or model runtime:

* :func:`validate_entity_evidence` reports, instead of raising, every way a reference can be
  wrong: the artifact is missing, has another identity or version, changed since the entity was
  materialized, fails its own integrity check or comes from an incompatible map lineage; the
  fused evidence is missing; the observations, features or 3D representations the entity lists
  are not what the fused evidence carries; geometry lies outside the entity's support;
* :func:`trace_entity_evidence` walks ``Entity -> FusedEvidence -> SpatialObservation ->
  PerceptionResult / Region2D / SemanticClaim -> SourceObservation`` and returns identities only;
* :func:`trace_geometry_sources` walks ``Entity -> EntityGeometry -> GeometryReference ->
  GeometricMapArtifact -> source LiDAR observation``.

Nothing is copied and nothing is re-run: perception and fusion are never invoked.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from contextmap.geometric_mapping import GeometrySource
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import (
    EvidenceContributionId,
    FusedEvidence,
    FusedEvidenceId,
    FusionRunArtifactError,
    FusionSupportId,
    SemanticFusionRunId,
    SemanticFusionRunManifest,
)
from contextmap.semantic_mapping.evidence import (
    EntityFeatureRef,
    FusedEvidenceRef,
    feature_refs_of,
    fusion_artifact_digest,
)
from contextmap.semantic_mapping.geometry import (
    EntityGeometry,
    GeometryResolutionError,
    resolve_geometry,
)
from contextmap.semantic_mapping.models import Entity, EntityReference
from contextmap.sensor_association import SpatialObservationId
from contextmap.shared import SourceTimestamp
from contextmap.visual_perception import ClaimId, PerceptionResultId, PerceptionRunId, RegionId


@runtime_checkable
class FusedEvidenceSource(Protocol):
    """Read boundary of one persisted Semantic Fusion run.

    A ``SemanticFusionRunReader`` satisfies it. Nothing else is needed from the run.
    """

    @property
    def manifest(self) -> SemanticFusionRunManifest:
        """The manifest of the run."""
        ...

    def fused_evidence(self, support_id: FusionSupportId) -> FusedEvidence:
        """Read the evidence fused over one support.

        Args:
            support_id: A support of the run.

        Returns:
            The fused evidence.

        Raises:
            KeyError: If the run has no such support.
        """
        ...

    def verify_integrity(self) -> list[str]:
        """Check the run's files against its inventory; empty means intact."""
        ...


class EvidenceIntegrityKind(Enum):
    """Why an evidence reference of an entity cannot be trusted.

    Attributes:
        MISSING_ARTIFACT: A referenced fusion run was not offered.
        ARTIFACT_MISMATCH: The offered run has another identity or schema version.
        STALE_REFERENCE: The run's content is not what the entity was materialized from.
        CORRUPT_ARTIFACT: The run fails its own integrity check.
        MISSING_REFERENCE: The fused evidence is not in the run, or is another one.
        INCOMPATIBLE_LINEAGE: The run was built over another geometric map than the entity's, or
            over another sequence than the one the entity was materialized from.
        OBSERVATION_MISMATCH: The observations listed differ from the fused evidence's.
        FEATURE_MISMATCH: The visual features listed differ from the fused evidence's.
        POINT_REPRESENTATION_MISMATCH: The 3D representations listed differ from the fused
            evidence's.
        GEOMETRY_OUTSIDE_SUPPORT: The fused evidence sees geometry the entity does not support.
        MISSING_GEOMETRY: A geometry reference of the entity does not resolve in the map.
    """

    MISSING_ARTIFACT = "missing_artifact"
    ARTIFACT_MISMATCH = "artifact_mismatch"
    STALE_REFERENCE = "stale_reference"
    CORRUPT_ARTIFACT = "corrupt_artifact"
    MISSING_REFERENCE = "missing_reference"
    INCOMPATIBLE_LINEAGE = "incompatible_lineage"
    OBSERVATION_MISMATCH = "observation_mismatch"
    FEATURE_MISMATCH = "feature_mismatch"
    POINT_REPRESENTATION_MISMATCH = "point_representation_mismatch"
    GEOMETRY_OUTSIDE_SUPPORT = "geometry_outside_support"
    MISSING_GEOMETRY = "missing_geometry"


@dataclass(frozen=True, kw_only=True)
class EvidenceIntegrityIssue:
    """One way an evidence reference of an entity is wrong.

    Attributes:
        entity: The entity whose reference is wrong.
        kind: What is wrong.
        detail: A deterministic, human-readable explanation.
    """

    entity: EntityReference
    kind: EvidenceIntegrityKind
    detail: str


def validate_entity_evidence(
    entity: Entity,
    *,
    fusion_runs: Mapping[SemanticFusionRunId, FusedEvidenceSource],
    geometry: GeometrySource | None = None,
) -> tuple[EvidenceIntegrityIssue, ...]:
    """Check every evidence reference of an entity against the artifacts it names.

    The checks never raise on a bad reference; they report it. An empty result means the
    references resolve, the artifacts are what the entity was materialized from and the entity
    lists exactly the observations, features and 3D representations its fused evidence carries.

    Args:
        entity: The entity to check.
        fusion_runs: The persisted fusion runs to check against, by run identity. Only the
            runs the entity references are read; nothing else is loaded.
        geometry: The read boundary of the geometric map, to also check that every geometry
            reference of the entity resolves; ``None`` skips that check.

    Returns:
        The issues found, in a deterministic order.
    """
    checker = _Checker(entity)
    for ref in entity.evidence.fused_evidence:
        checker.check(ref, fusion_runs.get(ref.fusion_run_id))
    checker.compare_lists()
    if geometry is not None:
        checker.check_geometry(geometry)
    return tuple(checker.issues)


class _VerifiedOnce:
    """Runs the integrity check of a run once, however many entities reference it."""

    def __init__(self, source: FusedEvidenceSource) -> None:
        self._source = source
        self._problems: list[str] | None = None

    @property
    def manifest(self) -> SemanticFusionRunManifest:
        """The manifest of the run."""
        return self._source.manifest

    def fused_evidence(self, support_id: FusionSupportId) -> FusedEvidence:
        """Read the evidence fused over one support."""
        return self._source.fused_evidence(support_id)

    def verify_integrity(self) -> list[str]:
        """Check the run's files once and remember the result."""
        if self._problems is None:
            self._problems = self._source.verify_integrity()
        return self._problems


def validate_evidence_of_entities(
    entities: Iterable[Entity],
    *,
    fusion_runs: Mapping[SemanticFusionRunId, FusedEvidenceSource],
    geometry: GeometrySource | None = None,
) -> tuple[EvidenceIntegrityIssue, ...]:
    """Check the evidence references of many entities, verifying each run's files only once.

    Args:
        entities: The entities to check.
        fusion_runs: The persisted fusion runs to check against, by run identity.
        geometry: The read boundary of the geometric map, to also check the geometry.

    Returns:
        The issues found for every entity, in the order the entities were given.
    """
    verified = {run_id: _VerifiedOnce(source) for run_id, source in fusion_runs.items()}
    issues: list[EvidenceIntegrityIssue] = []
    for entity in entities:
        issues.extend(validate_entity_evidence(entity, fusion_runs=verified, geometry=geometry))
    return tuple(issues)


class _Checker:
    """Collects the facts of the fused evidence and compares them with the entity's links."""

    def __init__(self, entity: Entity) -> None:
        self._entity = entity
        self._support = set(entity.geometry.geometry_refs)
        self._complete = True
        self._spatial: set[SpatialObservationId] = set()
        self._physical: set[SourceObservationId] = set()
        self._features: set[EntityFeatureRef] = set()
        self._representations: set[tuple[str, str]] = set()
        self.issues: list[EvidenceIntegrityIssue] = []

    def check(self, ref: FusedEvidenceRef, source: FusedEvidenceSource | None) -> None:
        """Check one fused evidence reference and collect what its evidence carries."""
        if source is None:
            self._issue(
                EvidenceIntegrityKind.MISSING_ARTIFACT,
                f"fusion run {ref.fusion_run_id!r} was not offered",
                blocks_comparison=True,
            )
            return
        manifest = source.manifest
        if manifest.run_id != ref.fusion_run_id or manifest.schema_version != (
            ref.fusion_schema_version
        ):
            self._issue(
                EvidenceIntegrityKind.ARTIFACT_MISMATCH,
                f"the entity references run {ref.fusion_run_id!r} at schema "
                f"{ref.fusion_schema_version!r}, the offered run is {manifest.run_id!r} at "
                f"{manifest.schema_version!r}",
                blocks_comparison=True,
            )
            return
        if fusion_artifact_digest(manifest) != ref.fusion_artifact_digest:
            self._issue(
                EvidenceIntegrityKind.STALE_REFERENCE,
                f"the content of fusion run {ref.fusion_run_id!r} changed since the entity was "
                f"materialized",
                blocks_comparison=True,
            )
            return
        for problem in source.verify_integrity():
            self._issue(EvidenceIntegrityKind.CORRUPT_ARTIFACT, problem)
        if manifest.lineage.geometric_map_id != self._entity.geometry.geometric_map_id:
            self._issue(
                EvidenceIntegrityKind.INCOMPATIBLE_LINEAGE,
                f"fusion run {ref.fusion_run_id!r} was built over map "
                f"{manifest.lineage.geometric_map_id!r}, but the entity's geometry is in "
                f"{self._entity.geometry.geometric_map_id!r}",
            )
        if manifest.lineage.sequence_artifact_id != ref.sequence_artifact_id:
            self._issue(
                EvidenceIntegrityKind.INCOMPATIBLE_LINEAGE,
                f"fusion run {ref.fusion_run_id!r} was built over sequence "
                f"{manifest.lineage.sequence_artifact_id!r}, but the entity was materialized from "
                f"sequence {ref.sequence_artifact_id!r}",
            )
        self._collect(ref, source)

    def compare_lists(self) -> None:
        """Compare what the entity lists with what the fused evidence carries."""
        if not self._complete:
            return
        links = self._entity.evidence
        self._compare(
            EvidenceIntegrityKind.OBSERVATION_MISMATCH,
            "spatial observations",
            set(links.spatial_observation_ids),
            self._spatial,
        )
        self._compare(
            EvidenceIntegrityKind.OBSERVATION_MISMATCH,
            "physical observations",
            set(links.physical_observation_ids),
            self._physical,
        )
        self._compare(
            EvidenceIntegrityKind.FEATURE_MISMATCH,
            "visual features",
            set(links.visual_feature_refs),
            self._features,
        )
        self._compare(
            EvidenceIntegrityKind.POINT_REPRESENTATION_MISMATCH,
            "3D representations",
            {(ref.run_id, ref.representation_id) for ref in links.point_representation_refs},
            self._representations,
        )

    def check_geometry(self, geometry: GeometrySource) -> None:
        """Check that every geometry reference of the entity resolves in the map."""
        try:
            resolve_geometry(self._entity.geometry.geometry_refs, source=geometry)
        except GeometryResolutionError as error:
            self._issue(EvidenceIntegrityKind.MISSING_GEOMETRY, str(error))

    def _collect(self, ref: FusedEvidenceRef, source: FusedEvidenceSource) -> None:
        try:
            fused = source.fused_evidence(ref.fusion_support_id)
        except (KeyError, FusionRunArtifactError) as error:
            self._issue(
                EvidenceIntegrityKind.MISSING_REFERENCE,
                f"fusion run {ref.fusion_run_id!r} has no readable support "
                f"{ref.fusion_support_id!r}: {error}",
                blocks_comparison=True,
            )
            return
        if fused.fused_evidence_id != ref.fused_evidence_id:
            self._issue(
                EvidenceIntegrityKind.MISSING_REFERENCE,
                f"support {ref.fusion_support_id!r} holds fused evidence "
                f"{fused.fused_evidence_id!r}, not {ref.fused_evidence_id!r}",
                blocks_comparison=True,
            )
            return
        for contribution in fused.contributions:
            self._spatial.add(contribution.spatial_observation_id)
            self._physical.add(contribution.physical_observation_id)
            outside = [item for item in contribution.geometry_support if item not in self._support]
            if outside:
                self._issue(
                    EvidenceIntegrityKind.GEOMETRY_OUTSIDE_SUPPORT,
                    f"contribution {contribution.contribution_id!r} sees {len(outside)} "
                    f"geometry element(s) outside the entity's support",
                )
        self._features.update(feature_refs_of(fused))
        self._representations.update(
            (ref.run_id, ref.representation_id) for ref in fused.point_representation_refs
        )

    def _compare(
        self,
        kind: EvidenceIntegrityKind,
        what: str,
        listed: AbstractSet[object],
        carried: AbstractSet[object],
    ) -> None:
        if listed != carried:
            self._issue(
                kind,
                f"the entity lists {len(listed)} {what}, the fused evidence carries "
                f"{len(carried)}; {len(listed - carried)} listed are not carried and "
                f"{len(carried - listed)} carried are not listed",
            )

    def _issue(
        self, kind: EvidenceIntegrityKind, detail: str, *, blocks_comparison: bool = False
    ) -> None:
        # Sem os fatos da evidência fundida, comparar as listas só geraria falsos positivos.
        if blocks_comparison:
            self._complete = False
        self.issues.append(
            EvidenceIntegrityIssue(entity=self._entity.reference, kind=kind, detail=detail)
        )


class EvidenceTraceError(Exception):
    """Raised when the provenance of an entity cannot be traversed."""


@dataclass(frozen=True, kw_only=True)
class ContributionTrace:
    """One view that contributed to an entity, as identities along the provenance chain.

    Attributes:
        fused_evidence_id: The fused evidence the view belongs to.
        contribution_id: The view.
        spatial_observation_id: The spatial observation it comes from.
        perception_run_id: The perception run that interpreted it.
        perception_result_id: The perception result that holds the claims.
        region_id: The observed region.
        claim_ids: The claims the view made about the region, sorted.
        physical_observation_id: The physical frame: the end of the chain.
    """

    fused_evidence_id: FusedEvidenceId
    contribution_id: EvidenceContributionId
    spatial_observation_id: SpatialObservationId
    perception_run_id: PerceptionRunId
    perception_result_id: PerceptionResultId
    region_id: RegionId
    claim_ids: tuple[ClaimId, ...]
    physical_observation_id: SourceObservationId


@dataclass(frozen=True, kw_only=True)
class EntityEvidenceTrace:
    """The provenance chain of an entity, from its fused evidence to the physical frames.

    Attributes:
        entity: The entity that was traced.
        contributions: Every contributing view, sorted by fused evidence and contribution.
    """

    entity: EntityReference
    contributions: tuple[ContributionTrace, ...]

    @property
    def physical_observation_ids(self) -> tuple[SourceObservationId, ...]:
        """The distinct physical frames the entity rests on, sorted."""
        return tuple(sorted({item.physical_observation_id for item in self.contributions}))


def trace_entity_evidence(
    entity: Entity, *, fusion_runs: Mapping[SemanticFusionRunId, FusedEvidenceSource]
) -> EntityEvidenceTrace:
    """Walk an entity back to the views and physical frames that support it.

    The traversal reads only the fused evidence the entity references and returns identities: no
    image, mask, embedding or fusion payload is copied and no inference is run.

    Args:
        entity: The entity to trace.
        fusion_runs: The persisted fusion runs the entity references, by run identity.

    Returns:
        The contributing views, each with its spatial observation, perception result, region,
        claims and physical frame.

    Raises:
        EvidenceTraceError: If a referenced run was not offered or its evidence cannot be read.
    """
    contributions: list[ContributionTrace] = []
    for ref in entity.evidence.fused_evidence:
        source = fusion_runs.get(ref.fusion_run_id)
        if source is None:
            raise EvidenceTraceError(f"fusion run {ref.fusion_run_id!r} was not offered")
        try:
            fused = source.fused_evidence(ref.fusion_support_id)
        except (KeyError, FusionRunArtifactError) as error:
            raise EvidenceTraceError(
                f"cannot read fused evidence of support {ref.fusion_support_id!r}: {error}"
            ) from error
        contributions.extend(
            ContributionTrace(
                fused_evidence_id=fused.fused_evidence_id,
                contribution_id=item.contribution_id,
                spatial_observation_id=item.spatial_observation_id,
                perception_run_id=item.perception_run_id,
                perception_result_id=item.perception_result_id,
                region_id=item.region_id,
                claim_ids=tuple(claim.claim_id for claim in item.claim_refs),
                physical_observation_id=item.physical_observation_id,
            )
            for item in fused.contributions
        )
    return EntityEvidenceTrace(
        entity=entity.reference,
        contributions=tuple(
            sorted(contributions, key=lambda item: (item.fused_evidence_id, item.contribution_id))
        ),
    )


@dataclass(frozen=True, kw_only=True)
class GeometrySourceTrace:
    """The geometry of an entity that came from one physical observation.

    Attributes:
        source_observation_id: The physical (LiDAR) observation the points were measured in.
        point_count: How many of the entity's geometry elements come from it.
        first_acquired: Acquisition time of the earliest of those points.
        last_acquired: Acquisition time of the latest.
    """

    source_observation_id: SourceObservationId
    point_count: int
    first_acquired: SourceTimestamp
    last_acquired: SourceTimestamp


def trace_geometry_sources(
    geometry: EntityGeometry, *, source: GeometrySource
) -> tuple[GeometrySourceTrace, ...]:
    """Walk an entity's geometry back to the physical observations it was measured in.

    Args:
        geometry: The geometry of an entity.
        source: The read boundary of the geometric map its references belong to.

    Returns:
        One trace per source observation, sorted by observation.

    Raises:
        GeometryResolutionError: If a geometry reference cannot be resolved.
    """
    counts: dict[SourceObservationId, int] = defaultdict(int)
    stamps: dict[SourceObservationId, list[SourceTimestamp]] = defaultdict(list)
    for point in resolve_geometry(geometry.geometry_refs, source=source):
        counts[point.source_observation_id] += 1
        stamps[point.source_observation_id].append(point.acquisition_timestamp)
    return tuple(
        GeometrySourceTrace(
            source_observation_id=observation,
            point_count=counts[observation],
            first_acquired=min(stamps[observation], key=SourceTimestamp.total_nanoseconds),
            last_acquired=max(stamps[observation], key=SourceTimestamp.total_nanoseconds),
        )
        for observation in sorted(counts)
    )

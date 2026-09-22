"""The evidence chain behind an entity.

An entity is a compact record that *references* the evidence that created it. It never copies
images, masks, embeddings or fusion payloads: it names the fused evidence, the observations that
contributed, the visual features and the 3D representations attached to it, and everything
upstream stays where it was produced. Following the references, a researcher can walk

``Entity -> FusedEvidence -> SpatialObservation -> PerceptionResult / Region2D / SemanticClaim
-> SourceObservation``

and, independently, ``Entity -> EntityGeometry -> GeometryReference -> GeometricMapArtifact ->
source LiDAR observation``, without the entity embedding any of it.

A reference to a persisted artifact carries its identity **and version**, and a digest of the
artifact's inventory, so a reference that no longer points at what it was materialized from is
detected instead of trusted. Optional channels may be absent: an entity whose fused evidence has
no visual features or no 3D representations is valid, with empty lists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import (
    FusedEvidence,
    FusedEvidenceId,
    FusionSupportId,
    PointRepresentationRef,
    SemanticFusionRunId,
    SemanticFusionRunManifest,
)
from contextmap.semantic_mapping._checks import require_canonical, require_present
from contextmap.sensor_association import SpatialObservationId
from contextmap.visual_perception import (
    FeatureId,
    FeatureScope,
    PerceptionResultId,
    PerceptionRunId,
    RegionId,
)

_Id = TypeVar("_Id", bound=str)


def fusion_artifact_digest(manifest: SemanticFusionRunManifest) -> str:
    """Digest the identity and the inventory of a persisted Semantic Fusion run.

    The digest covers the run identity, its schema version and the size and hash of every
    contractual file, so it is independent of the run's internal layout and changes if anything
    the entity was materialized from changes.

    Args:
        manifest: The manifest of the run.

    Returns:
        ``sha256:`` followed by the digest.
    """
    canonical = json.dumps(
        {
            "run_id": str(manifest.run_id),
            "schema_version": manifest.schema_version,
            "files": sorted([entry.path, entry.content_hash] for entry in manifest.file_inventory),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


@dataclass(frozen=True, kw_only=True)
class FusedEvidenceRef:
    """Reference to the evidence fused over one support, inside one persisted fusion run.

    Attributes:
        fusion_run_id: The Semantic Fusion run artifact that owns the evidence.
        fusion_schema_version: The schema version of that artifact.
        fusion_artifact_digest: Digest of the artifact's identity and inventory when the entity
            was materialized; see :func:`fusion_artifact_digest`.
        sequence_artifact_id: The canonical sequence that run was built over when the entity was
            materialized. The digest does not cover the run's lineage, so validation compares
            this identity with the run's manifest explicitly.
        fused_evidence_id: The fused evidence, local to that run.
        fusion_support_id: The support the evidence was accumulated over.
    """

    fusion_run_id: SemanticFusionRunId
    fusion_schema_version: str
    fusion_artifact_digest: str
    sequence_artifact_id: str
    fused_evidence_id: FusedEvidenceId
    fusion_support_id: FusionSupportId

    def __post_init__(self) -> None:
        """Require every identity.

        Raises:
            ValueError: If an identity, the version or the digest is empty.
        """
        require_present(
            self,
            "fusion_run_id",
            "fusion_schema_version",
            "fusion_artifact_digest",
            "sequence_artifact_id",
            "fused_evidence_id",
            "fusion_support_id",
        )


@dataclass(frozen=True, kw_only=True)
class EntityFeatureRef:
    """Reference to a visual feature that describes an entity's evidence.

    The vector stays in Visual Perception's feature store; only the identity travels, with the
    embedding space, so features of different spaces are never mixed by accident.

    The identity is the whole triple ``(perception_run_id, perception_result_id, feature_id)``:
    a result id is local to its run and a feature id to its result, so two runs may reuse the
    same pair and only the run tells them apart.

    Attributes:
        perception_run_id: The perception run that produced the feature.
        perception_result_id: The perception result that owns it, local to that run.
        feature_id: The feature, local to that result.
        embedding_space_id: The embedding space its vector lives in.
        scope: Whether the feature is dense, global or per-region.
        region_id: The region a ``REGION``-scoped feature belongs to; ``None`` otherwise.
    """

    perception_run_id: PerceptionRunId
    perception_result_id: PerceptionResultId
    feature_id: FeatureId
    embedding_space_id: str
    scope: FeatureScope
    region_id: RegionId | None

    def __post_init__(self) -> None:
        """Validate the identities and that the region matches the scope.

        Raises:
            ValueError: If an identity is empty, a region-scoped feature has no region, or
                another scope names one.
        """
        require_present(
            self, "perception_run_id", "perception_result_id", "feature_id", "embedding_space_id"
        )
        if (self.scope is FeatureScope.REGION) != (self.region_id is not None):
            raise ValueError("region_id must be present exactly for region-scoped features")


@dataclass(frozen=True, kw_only=True)
class EntityEvidenceLinks:
    """Where the evidence that supports an entity can be found.

    Attributes:
        fused_evidence: The fused evidence the entity was materialized from, sorted by run and
            evidence and unique; never empty, because an entity with no evidence behind it
            cannot be audited.
        spatial_observation_ids: The spatial observations that contributed, sorted and unique.
        physical_observation_ids: The physical frames that contributed, sorted and unique:
            the end of the provenance chain, however many inference runs interpreted each.
        visual_feature_refs: Region and global features of the contributing views, sorted by
            perception run, result and feature and unique; empty when the channel is absent.
        point_representation_refs: The 3D representations attached to the support, sorted by
            run and representation and unique; empty when the channel is absent.
    """

    fused_evidence: tuple[FusedEvidenceRef, ...]
    spatial_observation_ids: tuple[SpatialObservationId, ...]
    physical_observation_ids: tuple[SourceObservationId, ...]
    visual_feature_refs: tuple[EntityFeatureRef, ...] = ()
    point_representation_refs: tuple[PointRepresentationRef, ...] = ()

    def __post_init__(self) -> None:
        """Validate that the entity has evidence and lists it canonically.

        Raises:
            ValueError: If there is no fused evidence or observation, or a collection is not
                sorted and unique, which also refuses a duplicated reference.
        """
        if not self.fused_evidence:
            raise ValueError("fused_evidence must not be empty: an entity needs evidence")
        if not self.spatial_observation_ids or not self.physical_observation_ids:
            raise ValueError("an entity must list the observations that contributed")
        require_canonical(
            "fused_evidence",
            self.fused_evidence,
            lambda ref: (ref.fusion_run_id, ref.fused_evidence_id),
        )
        require_canonical(
            "spatial_observation_ids", self.spatial_observation_ids, lambda item: (item,)
        )
        require_canonical(
            "physical_observation_ids", self.physical_observation_ids, lambda item: (item,)
        )
        require_canonical("visual_feature_refs", self.visual_feature_refs, _feature_key)
        require_canonical(
            "point_representation_refs",
            self.point_representation_refs,
            lambda ref: (ref.run_id, ref.representation_id),
        )


def feature_refs_of(evidence: FusedEvidence) -> tuple[EntityFeatureRef, ...]:
    """List the visual features of the views that contributed to some fused evidence.

    Args:
        evidence: The evidence fused over one support.

    Returns:
        One reference per feature, each once, sorted by perception run, result and feature.
        Two runs that reuse the same result and feature ids keep one reference each.
    """
    found = {
        _feature_key(ref): ref
        for ref in (
            EntityFeatureRef(
                perception_run_id=item.perception_run_id,
                perception_result_id=item.perception_result_id,
                feature_id=feature.feature_id,
                embedding_space_id=feature.embedding_space_id,
                scope=feature.scope,
                region_id=feature.region_id,
            )
            for item in evidence.contributions
            for feature in item.visual_feature_refs
        )
    }
    return tuple(found[key] for key in sorted(found))


def evidence_links_from_fused_evidence(
    evidence: FusedEvidence, *, manifest: SemanticFusionRunManifest
) -> EntityEvidenceLinks:
    """Link an entity to the fused evidence it is materialized from, without copying it.

    Args:
        evidence: The evidence fused over one support.
        manifest: The manifest of the fusion run that owns the evidence, which gives the
            artifact identity, version and digest the reference carries.

    Returns:
        The links: the fused evidence, every contributing spatial and physical observation,
        the visual features and the 3D representations, each listed once. Channels the fused
        evidence does not carry stay empty.
    """
    return EntityEvidenceLinks(
        fused_evidence=(
            FusedEvidenceRef(
                fusion_run_id=manifest.run_id,
                fusion_schema_version=manifest.schema_version,
                fusion_artifact_digest=fusion_artifact_digest(manifest),
                sequence_artifact_id=manifest.lineage.sequence_artifact_id,
                fused_evidence_id=evidence.fused_evidence_id,
                fusion_support_id=evidence.fusion_support_id,
            ),
        ),
        spatial_observation_ids=_sorted_unique(
            item.spatial_observation_id for item in evidence.contributions
        ),
        physical_observation_ids=_sorted_unique(
            group.physical_observation_id for group in evidence.physical_observation_groups
        ),
        visual_feature_refs=feature_refs_of(evidence),
        point_representation_refs=evidence.point_representation_refs,
    )


def _feature_key(ref: EntityFeatureRef) -> tuple[str, str, str]:
    """The identity of a feature reference: run, result and feature, in that order."""
    return (ref.perception_run_id, ref.perception_result_id, ref.feature_id)


def _sorted_unique(items: Iterable[_Id]) -> tuple[_Id, ...]:
    return tuple(sorted(set(items)))

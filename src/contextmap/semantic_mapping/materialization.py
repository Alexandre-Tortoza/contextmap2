"""Materialization of entities from fused evidence, without any cross-support resolution.

Semantic Fusion accumulates evidence over spatial supports and creates no persistent object.
This module turns selected fused evidence into canonical :class:`Entity` records under one
explicit, versioned and deterministic baseline, ``one-support-one-entity-v1``:

* **one selected support, one entity** -- each :class:`FusionOutcome` the caller selects becomes
  exactly one entity. Nothing else is ever asked: *is support A the same physical object as
  support B?* belongs to Entity Resolution, so two supports stay two entities however similar their
  labels, geometry or features are, and no entity is ever merged, split, re-identified or updated
  from another;
* **identity** -- ``support-derived-entity-id-v1``: the entity id is a pure function of the fusion
  support id (``entity--<support id>``). It is stable within one selected fusion artifact, does not
  depend on the order candidates arrive in or on which others were rejected, and is an
  *artifact-local* identity: it says nothing about any other semantic map;
* **content** -- exact geometry support and its derived summaries, the semantic state with every
  alternative, conflict, abstention and unscored signal, the evidence links and the temporal
  state, each built by the rule its own issue defines;
* **invalid candidates** -- a candidate that cannot become a valid entity is not dropped: it is
  reported as a :class:`CandidateRejection` with the reason.

The service reads only what the caller selects: it never loads every available fusion run, never
rewrites the fused evidence and never runs perception or fusion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from contextmap.geometric_mapping import GeometrySource
from contextmap.semantic_fusion import (
    FusedEvidenceId,
    FusionOutcome,
    FusionSupportId,
    SemanticFusionRunManifest,
)
from contextmap.semantic_mapping._checks import require_canonical, require_present
from contextmap.semantic_mapping.evidence import evidence_links_from_fused_evidence
from contextmap.semantic_mapping.geometry import (
    EmptyGeometrySupportError,
    GeometryResolutionError,
    GeometrySummaryPolicy,
    summarize_geometry,
)
from contextmap.semantic_mapping.models import Entity, EntityId, EntityProvenance, SemanticMapId
from contextmap.semantic_mapping.state_mapping import (
    CLASS_ATTRIBUTE_DERIVATION_ID,
    PRIMARY_HYPOTHESIS_POLICY_ID,
    SEMANTIC_STATE_MAPPING_RULE_ID,
    semantic_state_from_fused_evidence,
)
from contextmap.semantic_mapping.temporal import (
    TEMPORAL_SUMMARY_RULE_ID,
    TemporalEvidenceError,
    summarize_temporal_state,
)

ENTITY_MATERIALIZATION_POLICY_ID = "one-support-one-entity-v1"
"""Versioned identity of the baseline materialization described in this module."""

ENTITY_ID_POLICY_ID = "support-derived-entity-id-v1"
"""Versioned identity of the rule that allocates entity ids."""


class MaterializationInputError(ValueError):
    """Raised when the selection given to the service is inconsistent as a whole."""


class RejectionReason(Enum):
    """Why a selected candidate could not become an entity.

    Attributes:
        EMPTY_GEOMETRY_SUPPORT: The candidate has no 3D support.
        UNRESOLVABLE_GEOMETRY: Its geometry references cannot be resolved against the map.
        INVALID_TEMPORAL_EVIDENCE: Its temporal evidence is missing or inconsistent.
        INVALID_ENTITY: Its parts do not satisfy the entity contract.
    """

    EMPTY_GEOMETRY_SUPPORT = "empty_geometry_support"
    UNRESOLVABLE_GEOMETRY = "unresolvable_geometry"
    INVALID_TEMPORAL_EVIDENCE = "invalid_temporal_evidence"
    INVALID_ENTITY = "invalid_entity"


def entity_id_for(*, fusion_support_id: FusionSupportId) -> EntityId:
    """Compute the deterministic, artifact-local identity of the entity of one support.

    Args:
        fusion_support_id: The support the entity is materialized from.

    Returns:
        A pure function of the input, so the identity does not depend on the order of the
        candidates or on which others were rejected.
    """
    return EntityId(f"entity--{fusion_support_id}")


@dataclass(frozen=True, kw_only=True)
class EntityMaterializationPolicy:
    """Explicit configuration of the baseline materialization.

    Attributes:
        geometry: How the spatial summaries of every entity are derived.
    """

    geometry: GeometrySummaryPolicy

    def fingerprint(self) -> str:
        """Hash every rule and threshold that shapes an entity, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "policy_id": ENTITY_MATERIALIZATION_POLICY_ID,
                "identity_policy_id": ENTITY_ID_POLICY_ID,
                "semantic_state_rule_id": SEMANTIC_STATE_MAPPING_RULE_ID,
                "primary_policy_id": PRIMARY_HYPOTHESIS_POLICY_ID,
                "class_derivation_id": CLASS_ATTRIBUTE_DERIVATION_ID,
                "temporal_rule_id": TEMPORAL_SUMMARY_RULE_ID,
                "geometry": self.geometry.fingerprint(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


@dataclass(frozen=True, kw_only=True)
class CandidateRejection:
    """A selected candidate that could not become an entity, and why.

    Attributes:
        fusion_support_id: The support of the candidate.
        fused_evidence_id: The fused evidence of the candidate.
        reason: What kind of problem it is.
        detail: A deterministic, human-readable explanation.
    """

    fusion_support_id: FusionSupportId
    fused_evidence_id: FusedEvidenceId
    reason: RejectionReason
    detail: str

    def __post_init__(self) -> None:
        """Require the identities and an explanation.

        Raises:
            ValueError: If an identity or the detail is empty.
        """
        require_present(self, "fusion_support_id", "fused_evidence_id", "detail")


@dataclass(frozen=True, kw_only=True)
class EntityMaterialization:
    """The result of materializing a selection: every candidate is an entity or a rejection.

    Attributes:
        semantic_map_id: The semantic map the entities belong to.
        entities: The materialized entities, sorted by id.
        rejections: The candidates that could not become entities, sorted by support.
    """

    semantic_map_id: SemanticMapId
    entities: tuple[Entity, ...]
    rejections: tuple[CandidateRejection, ...] = ()

    def __post_init__(self) -> None:
        """Validate the ordering and that no candidate is both an entity and a rejection.

        Raises:
            ValueError: If the map identity is empty, a collection is not sorted and unique, an
                entity belongs to another semantic map, or a support is both materialized and
                rejected.
        """
        require_present(self, "semantic_map_id")
        require_canonical("entities", self.entities, lambda item: (item.entity_id,))
        require_canonical(
            "rejections",
            self.rejections,
            lambda item: (item.fusion_support_id,),
            detail="by support ",
        )
        for entity in self.entities:
            if entity.semantic_map_id != self.semantic_map_id:
                raise ValueError(
                    f"entity {entity.entity_id!r} belongs to semantic map "
                    f"{entity.semantic_map_id!r}, not {self.semantic_map_id!r}"
                )
        materialized = {
            ref.fusion_support_id
            for entity in self.entities
            for ref in entity.evidence.fused_evidence
        }
        if materialized & {item.fusion_support_id for item in self.rejections}:
            raise ValueError("a support cannot be both materialized and rejected")


def materialize_entities(
    outcomes: Iterable[FusionOutcome],
    *,
    fusion_manifest: SemanticFusionRunManifest,
    geometry: GeometrySource,
    semantic_map_id: SemanticMapId,
    policy: EntityMaterializationPolicy,
    code_version: str | None = None,
) -> EntityMaterialization:
    """Materialize the selected fused evidence as entities, one entity per selected support.

    No candidate is compared with another: the result is a function of each candidate alone, so
    identical immutable inputs and configuration always reproduce identical entities and
    identities, whatever the order of the selection.

    Args:
        outcomes: The selected supports with their fused evidence, in any order; only what is
            passed is read.
        fusion_manifest: The manifest of the fusion run that owns them, which supplies the
            artifact identity, version and digest the evidence links carry.
        geometry: The read boundary of the geometric map the supports were built over.
        semantic_map_id: The identity of the semantic map being built.
        policy: The materialization configuration.
        code_version: Code revision to record in the provenance, when known.

    Returns:
        The entities and the rejected candidates, each candidate in exactly one of them.

    Raises:
        MaterializationInputError: If a support is selected twice, or a support was built over
            another map than the fusion run's lineage or the geometry source.
    """
    selected = sorted(outcomes, key=lambda item: item.support.fusion_support_id)
    _require_coherent_selection(selected, fusion_manifest, geometry)
    provenance = EntityProvenance(
        materialization_policy_id=ENTITY_MATERIALIZATION_POLICY_ID,
        identity_policy_id=ENTITY_ID_POLICY_ID,
        configuration_fingerprint=policy.fingerprint(),
        code_version=code_version,
    )
    entities: list[Entity] = []
    rejections: list[CandidateRejection] = []
    for outcome in selected:
        try:
            entities.append(
                _materialize(
                    outcome,
                    fusion_manifest=fusion_manifest,
                    geometry=geometry,
                    semantic_map_id=semantic_map_id,
                    policy=policy,
                    provenance=provenance,
                )
            )
        except EmptyGeometrySupportError as error:
            rejections.append(_rejection(outcome, RejectionReason.EMPTY_GEOMETRY_SUPPORT, error))
        except GeometryResolutionError as error:
            rejections.append(_rejection(outcome, RejectionReason.UNRESOLVABLE_GEOMETRY, error))
        except TemporalEvidenceError as error:
            rejections.append(_rejection(outcome, RejectionReason.INVALID_TEMPORAL_EVIDENCE, error))
        except ValueError as error:
            rejections.append(_rejection(outcome, RejectionReason.INVALID_ENTITY, error))
    return EntityMaterialization(
        semantic_map_id=semantic_map_id,
        entities=tuple(entities),
        rejections=tuple(rejections),
    )


def _require_coherent_selection(
    selected: list[FusionOutcome],
    fusion_manifest: SemanticFusionRunManifest,
    geometry: GeometrySource,
) -> None:
    identities = [item.support.fusion_support_id for item in selected]
    repeated = sorted({item for item in identities if identities.count(item) > 1})
    if repeated:
        raise MaterializationInputError(f"supports are selected more than once: {repeated!r}")
    served = geometry.geometric_map.map_id
    if fusion_manifest.lineage.geometric_map_id != served:
        raise MaterializationInputError(
            f"the fusion run was built over map {fusion_manifest.lineage.geometric_map_id!r}, but "
            f"the geometry source serves {served!r}"
        )
    for item in selected:
        if item.support.geometric_map_id != served:
            raise MaterializationInputError(
                f"support {item.support.fusion_support_id!r} is over map "
                f"{item.support.geometric_map_id!r}, but the geometry source serves {served!r}"
            )


def _materialize(
    outcome: FusionOutcome,
    *,
    fusion_manifest: SemanticFusionRunManifest,
    geometry: GeometrySource,
    semantic_map_id: SemanticMapId,
    policy: EntityMaterializationPolicy,
    provenance: EntityProvenance,
) -> Entity:
    support, evidence = outcome.support, outcome.evidence
    return Entity(
        entity_id=entity_id_for(fusion_support_id=support.fusion_support_id),
        semantic_map_id=semantic_map_id,
        geometry=summarize_geometry(
            support.geometry_support, source=geometry, policy=policy.geometry
        ),
        semantic_state=semantic_state_from_fused_evidence(evidence),
        evidence=evidence_links_from_fused_evidence(evidence, manifest=fusion_manifest),
        temporal_state=summarize_temporal_state(evidence.physical_observation_groups),
        provenance=provenance,
    )


def _rejection(
    outcome: FusionOutcome, reason: RejectionReason, error: Exception
) -> CandidateRejection:
    return CandidateRejection(
        fusion_support_id=outcome.support.fusion_support_id,
        fused_evidence_id=outcome.evidence.fused_evidence_id,
        reason=reason,
        detail=str(error) or type(error).__name__,
    )

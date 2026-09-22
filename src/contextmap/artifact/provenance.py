"""Lineage and provenance of the ContextMap.

Information in the map has different epistemic status depending on how it was produced: a
sensor measurement, a model interpretation, a deterministic geometric computation, an
accumulation over several views, a human annotation and, in the future, reasoning over prior
knowledge are not the same kind of fact. :class:`DerivationKind` records that distinction and
:class:`EvidenceOrigin` records it together with *everything* the result was derived from, so a
category never pretends to explain a result by itself.

None of this is a confidence. A derivation kind says how information was produced, never how
much to trust it, and no number in this module measures belief. The support values that
justify a hypothesis stay in the upstream artifacts, which the origin points to exactly.

The map lists every upstream artifact it cites, with the exact identities needed to audit it, in
its lineage. Every reference in the map resolves to that table, which is the provenance
closure: a consumer walks from a result to its evidence without loading debug output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from contextmap.artifact._checks import (
    require_artifact_identity,
    require_canonical,
    require_optional_present,
    require_present,
)
from contextmap.artifact.metadata import PolicyRef
from contextmap.artifact.references import UpstreamRecordRef

_CONTENT_IDENTITY_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class ProvenanceError(ValueError):
    """Raised when an origin claims a derivation its evidence does not support."""


class DerivationKind(Enum):
    """How a piece of information was produced.

    This is provenance metadata, not a confidence score: it does not rank kinds by
    trustworthiness and no kind implies a value of belief.

    Attributes:
        SENSOR_OBSERVED: Grounded directly in a canonical sensor observation or in geometry
            measured from one. A model output is never observed, even if it consumed an image.
        MODEL_INFERRED: Produced by a perception, language or scoring model.
        GEOMETRY_DERIVED: Computed deterministically from geometry by a versioned rule.
        MULTIVIEW_FUSED: Accumulated from several evidence contributions by a versioned rule.
        HUMAN_ANNOTATED: An explicit human annotation supplied as an input of the pipeline.
        PRIOR_KNOWLEDGE: Reserved for reasoning over explicit prior knowledge. Nothing in
            Solution 1 emits it yet; the schema only keeps it distinguishable.
    """

    SENSOR_OBSERVED = "sensor_observed"
    MODEL_INFERRED = "model_inferred"
    GEOMETRY_DERIVED = "geometry_derived"
    MULTIVIEW_FUSED = "multiview_fused"
    HUMAN_ANNOTATED = "human_annotated"
    PRIOR_KNOWLEDGE = "prior_knowledge"


class ArtifactKind(Enum):
    """What an upstream artifact is, which fixes what its records mean.

    An evaluation-only reference set and debug output are deliberately absent: neither is ever
    a dependency of a map, so neither can be cited.

    Attributes:
        SEQUENCE: A canonical sequence; its records are physical observations.
        STATE_ESTIMATION_RUN: A pose trajectory.
        GEOMETRIC_MAP: The persistent geometry; its records are geometry elements.
        PERCEPTION_RUN: Visual inference; its records are model outputs.
        SENSOR_ASSOCIATION_RUN: Spatial observations that anchor 2D evidence in geometry.
        POINT_REPRESENTATION_RUN: Optional 3D representation evidence.
        SEMANTIC_FUSION_RUN: Multi-view accumulated evidence.
        SEMANTIC_MAP: Entities materialized before identity resolution.
        ENTITY_RESOLUTION_RUN: Resolved entities and their resolution decisions.
        SPATIAL_RELATIONS_RUN: Relations and the evidence behind them.
        HUMAN_ANNOTATION_SET: Human annotations explicitly supplied as a pipeline input.
    """

    SEQUENCE = "sequence"
    STATE_ESTIMATION_RUN = "state_estimation_run"
    GEOMETRIC_MAP = "geometric_map"
    PERCEPTION_RUN = "perception_run"
    SENSOR_ASSOCIATION_RUN = "sensor_association_run"
    POINT_REPRESENTATION_RUN = "point_representation_run"
    SEMANTIC_FUSION_RUN = "semantic_fusion_run"
    SEMANTIC_MAP = "semantic_map"
    ENTITY_RESOLUTION_RUN = "entity_resolution_run"
    SPATIAL_RELATIONS_RUN = "spatial_relations_run"
    HUMAN_ANNOTATION_SET = "human_annotation_set"

    @property
    def is_physical_observation(self) -> bool:
        """Whether the records of this kind are physical observations, not inferences."""
        return self is ArtifactKind.SEQUENCE

    @property
    def is_structural(self) -> bool:
        """Whether resolving the map needs this artifact.

        A structural dependency is needed to resolve geometry, entities or relations. Every
        other kind is an optional evidence dependency, needed only for deep inspection.
        """
        return self in _STRUCTURAL_KINDS


_STRUCTURAL_KINDS = frozenset(
    {
        ArtifactKind.GEOMETRIC_MAP,
        ArtifactKind.ENTITY_RESOLUTION_RUN,
        ArtifactKind.SPATIAL_RELATIONS_RUN,
    }
)


@dataclass(frozen=True, kw_only=True)
class UpstreamArtifact:
    """One immutable upstream artifact the map cites, with the identities needed to audit it.

    Attributes:
        artifact_id: Exact identity of the artifact or run; never a path.
        kind: What the artifact is.
        content_identity: ``"sha256:<hex>"`` of the artifact's content, so a reference names
            exactly this artifact and a stale or altered copy is detectable.
        configuration_fingerprint: Hash of the effective configuration that produced it;
            ``None`` when nothing was configurable.
        code_version: Code revision that produced it; ``None`` when not known.
        model_identities: The models and checkpoints behind its inferences, sorted and unique;
            empty for an artifact that ran no model.
    """

    artifact_id: str
    kind: ArtifactKind
    content_identity: str
    configuration_fingerprint: str | None
    code_version: str | None
    model_identities: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the identities.

        Raises:
            ValueError: If the artifact identity is blank or a path, the content identity is not
                a SHA-256 digest, an optional identity is set but blank, or the model
                identities are blank, unsorted or repeated.
        """
        require_artifact_identity(self, "artifact_id")
        require_present(self, "content_identity")
        if _CONTENT_IDENTITY_PATTERN.fullmatch(self.content_identity) is None:
            raise ValueError(
                f"content_identity must be 'sha256:<64 hex digits>', got {self.content_identity!r}"
            )
        require_optional_present(self, "configuration_fingerprint", "code_version")
        if any(not model.strip() for model in self.model_identities):
            raise ValueError("model_identities must not contain a blank identity")
        require_canonical("model_identities", self.model_identities, lambda item: (item,))


_POLICY_REQUIRED = frozenset(
    {
        DerivationKind.GEOMETRY_DERIVED,
        DerivationKind.MULTIVIEW_FUSED,
        DerivationKind.PRIOR_KNOWLEDGE,
    }
)


@dataclass(frozen=True, kw_only=True)
class EvidenceOrigin:
    """How a result was produced and everything it was derived from.

    Attributes:
        kind: The derivation category. It never explains the result on its own.
        derived_from: Every upstream record the result rests on, sorted and unique; at least
            one. Each names an artifact listed in the map's lineage.
        policy: The versioned rule that derived the result; required for geometry-derived,
            fused and prior-knowledge results, where a rule is what produced them.
    """

    kind: DerivationKind
    derived_from: tuple[UpstreamRecordRef, ...]
    policy: PolicyRef | None

    def __post_init__(self) -> None:
        """Validate that the origin cites its evidence and, where required, its rule.

        Raises:
            ValueError: If no evidence is cited or the evidence is not sorted and unique.
            ProvenanceError: If a derivation by rule names no policy.
        """
        if not self.derived_from:
            raise ValueError(
                "derived_from must cite at least one upstream record: a derivation category "
                "alone does not explain a result"
            )
        require_canonical(
            "derived_from", self.derived_from, lambda item: (item.artifact_id, item.record_id)
        )
        if self.kind in _POLICY_REQUIRED and self.policy is None:
            raise ProvenanceError(
                f"a {self.kind.name} origin must record the policy that derived it"
            )

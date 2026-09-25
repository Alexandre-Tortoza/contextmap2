"""Capability ports: substitution points for Visual Perception backends.

A port is a typed `Protocol` for one real substitution point — a concrete
model (SAM2, SAM3, Florence-2, DINOv2, DINOv3, CLIP, AlphaCLIP, Qwen,
Gemini, LocateAnything, ...) is an adapter that satisfies one or more of these ports.
Ports never encode a mandatory execution order between capabilities;
:mod:`contextmap.visual_perception.service` resolves the actual stage
graph and decides ordering. No port constructs a concrete backend or
imports a model SDK — that is the composition root's job.

One model may satisfy more than one port through distinct adapters (e.g.
Florence-2 as a :class:`RegionDiscovery` adapter and, separately, as a
:class:`SemanticInterpreter` adapter); satisfying two ports is never a
reason to merge the ports themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from contextmap.visual_perception.dense_region_association import DenseFeatureMap
from contextmap.visual_perception.grounding import (
    RegionGroundingCapabilities,
    RegionGroundingExecution,
    RegionGroundingRequest,
)
from contextmap.visual_perception.models import (
    BackendProvenance,
    FeatureScope,
    PreparedImage,
    Region2D,
    SemanticClaim,
    SemanticScore,
    VisualFeature,
)
from contextmap.visual_perception.semantic_backend import SemanticInterpretationExecution
from contextmap.visual_perception.semantic_requests import (
    SemanticInterpretationRequest,
    SemanticInterpreterCapabilities,
)


@runtime_checkable
class RegionDiscovery(Protocol):
    """Capability port: discover 2D region candidates from a prepared image."""

    def backend_provenance(self) -> BackendProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            The provenance to attach to every region this call produces.
        """
        ...

    def discover(self, image: PreparedImage) -> Sequence[Region2D]:
        """Discover region candidates in a prepared image.

        Args:
            image: The prepared image to search.

        Returns:
            Discovered regions, accepted and/or rejected (see
            :attr:`~contextmap.visual_perception.models.Region2D.is_accepted`).
        """
        ...


@runtime_checkable
class RegionGrounding(Protocol):
    """Capability port: localize what an explicit language query refers to in an image.

    Distinct from :class:`RegionDiscovery`, which proposes regions from the image alone:
    here the query is a first-class, validated, persisted inference input of each
    request, never backend configuration.
    """

    def backend_provenance(self) -> BackendProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            Provenance whose capability is ``"region_grounding"``.
        """
        ...

    def capabilities(self) -> RegionGroundingCapabilities:
        """Declare the query policies (task and geometry) this backend serves."""
        ...

    def ground(self, request: RegionGroundingRequest) -> RegionGroundingExecution:
        """Answer one grounding request.

        Args:
            request: The image, the query and the configuration fingerprint.

        Returns:
            The request, rendered prompt, raw response, parsed outputs and diagnostics.

        Raises:
            GroundingRequestError: Before any inference, when the backend does not declare
                the requested policy, task or geometry.
        """
        ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Capability port: extract visual features from a prepared image.

    A dense/global extractor only ever receives ``image``; a
    region-scoped extractor also receives ``regions``. Which one applies
    is declared by :meth:`required_scope`, not encoded in a shared
    mandatory signature.
    """

    def backend_provenance(self) -> BackendProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            The provenance to attach to every feature this call produces.
        """
        ...

    def required_scope(self) -> FeatureScope:
        """Report which scope this extractor produces.

        Returns:
            ``DENSE`` or ``GLOBAL`` for an extractor that only needs
            ``image``; ``REGION`` for one that also needs ``regions``.
        """
        ...

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        """Extract visual features.

        Args:
            image: The prepared image to extract from.
            regions: Regions to extract per-region features for. Ignored
                by an extractor whose :meth:`required_scope` is not
                ``REGION``; required (non-empty) when it is.

        Returns:
            Extracted features, matching :meth:`required_scope`.
        """
        ...


@runtime_checkable
class FeatureResolutionEnhancement(Protocol):
    """Capability port: increase a dense feature map's spatial resolution."""

    def backend_provenance(self) -> BackendProvenance:
        """Report the selected enhancement model and effective configuration."""
        ...

    def enhance(self, source: DenseFeatureMap) -> DenseFeatureMap:
        """Produce a separately identified dense map from one source artifact.

        Args:
            source: Immutable source dense map, including payload/artifact
                references and exact sampling semantics.

        Returns:
            Enhanced map with complete enhancement lineage. Representation
            changes must use a distinct embedding-space fingerprint.
        """
        ...


@runtime_checkable
class SemanticInterpreter(Protocol):
    """Capability port: interpret a scene and/or specific regions semantically."""

    def backend_provenance(self) -> BackendProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            The provenance to attach to every claim/context this call
            produces.
        """
        ...

    def capabilities(self) -> SemanticInterpreterCapabilities:
        """Declare supported modes and canonical evidence before execution."""
        ...

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        """Interpret one validated canonical semantic request.

        Args:
            request: Exact scene/region evidence selection, prompt identity,
                output schema, and configuration fingerprint.

        Returns:
            Raw, parsed, provenance, and diagnostic records for the call.
        """
        ...


@runtime_checkable
class SemanticScorer(Protocol):
    """Capability port: score semantic claims against compatible visual evidence.

    Scoring never mutates a claim; it produces separate
    :class:`~contextmap.visual_perception.models.SemanticScore` records.
    """

    def backend_provenance(self) -> BackendProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            The provenance to attach to every support judgement this call
            produces.
        """
        ...

    def score(
        self,
        claims: Sequence[SemanticClaim],
        features: Sequence[VisualFeature],
    ) -> Sequence[SemanticScore]:
        """Score claims against visual evidence.

        Args:
            claims: Claims to score.
            features: Canonical global or region features to compare with the
                claim hypotheses. Numerical payloads remain backend-internal.

        Returns:
            Zero or more explicit semantic scores. A claim with no compatible
            feature remains unscored; no zero-valued placeholder is created.
        """
        ...

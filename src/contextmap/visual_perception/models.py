"""Public visual evidence model for the Visual Perception capability.

These types represent the output of processing one physical
:class:`~contextmap.ingestion.SourceObservation` with one configured
:class:`PerceptionRun`, without conflating the physical observation itself
with the inference result produced over it. Reprocessing the same
observation in another run always produces a new, distinct
:class:`PerceptionResult`; it never mutates the `SourceObservation` or a
prior result. See ``src/contextmap/visual_perception/docs/contracts.md``
for field-level ownership and worked examples.

A region, feature, or claim identity (:data:`RegionId`, :data:`FeatureId`,
:data:`ClaimId`) is local visual evidence scoped to the
:class:`PerceptionResult` that produced it — never a persistent 3D entity
identity, and never comparable across two different results without a
later, explicit association performed by a downstream capability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import NewType

from contextmap.ingestion import SourceObservationId

PerceptionRunId = NewType("PerceptionRunId", str)
"""Identity of one configured Visual Perception execution."""

PerceptionResultId = NewType("PerceptionResultId", str)
"""Identity of one PerceptionRun's output for one SourceObservation."""

RegionId = NewType("RegionId", str)
"""Identity of a Region2D, local to the owning PerceptionResult."""

FeatureId = NewType("FeatureId", str)
"""Identity of a VisualFeature, local to the owning PerceptionResult."""

ClaimId = NewType("ClaimId", str)
"""Identity of a SemanticClaim, local to the owning PerceptionResult."""


class HypothesisRole(Enum):
    """Whether a :class:`SemanticClaim` is the leading or a secondary hypothesis."""

    PRIMARY = "primary"
    ALTERNATIVE = "alternative"


class FeatureScope(Enum):
    """What a :class:`VisualFeature` is a representation of."""

    DENSE = "dense"
    GLOBAL = "global"
    REGION = "region"


@dataclass(frozen=True, kw_only=True)
class BackendProvenance:
    """Traceability for one backend invocation producing a piece of evidence.

    Attributes:
        backend_id: Stable identity of the adapter, e.g. ``"sam2"``,
            ``"dinov3"``, ``"fake_region_discovery"``.
        capability: Capability port this backend satisfies, e.g.
            ``"region_discovery"``, ``"feature_extractor"``.
        provider: Model family/provider name.
        model: Model/checkpoint identity.
        version: Backend/model version string.
        configuration_fingerprint: Deterministic hash of the effective
            backend configuration, for reproducibility. ``None`` when the
            backend has no configurable parameters worth fingerprinting.
    """

    backend_id: str
    capability: str
    provider: str
    model: str
    version: str
    configuration_fingerprint: str | None = None


@dataclass(frozen=True, kw_only=True)
class BoundingBox2D:
    """An axis-aligned image-space bounding box.

    Attributes:
        x: Left edge, in pixels.
        y: Top edge, in pixels.
        width: Box width, in pixels.
        height: Box height, in pixels.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate the box has positive extent.

        Raises:
            ValueError: If ``width`` or ``height`` is not positive.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")


@dataclass(frozen=True, kw_only=True)
class Region2D:
    """A 2D region of interest proposed within one :class:`PerceptionResult`.

    Attributes:
        region_id: Identity local to the owning ``PerceptionResult``.
        bounding_box: Image-space bounding geometry.
        provenance: Backend that produced this region.
        mask_reference: Reference to a mask payload (e.g. a debug/output
            file path) — never the raw mask array inline. ``None`` for a
            box-only region.
        region_kind: Free-form category hint from the producing backend,
            e.g. ``"object"``, ``"surface"``. ``None`` when unclassified.
        is_accepted: Whether this region survived normalization/merge and
            belongs to the frozen canonical set. A rejected candidate can
            still be recorded, with ``is_accepted=False``, for audit.
        rejection_reason: Human-readable reason; set only when
            ``is_accepted`` is ``False``.
    """

    region_id: RegionId
    bounding_box: BoundingBox2D
    provenance: BackendProvenance
    mask_reference: str | None = None
    region_kind: str | None = None
    is_accepted: bool = True
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate rejection metadata is consistent with ``is_accepted``.

        Raises:
            ValueError: If ``rejection_reason`` is set while
                ``is_accepted`` is ``True``.
        """
        if self.is_accepted and self.rejection_reason is not None:
            raise ValueError("rejection_reason must be None when is_accepted is True")


@dataclass(frozen=True, kw_only=True)
class VisualFeature:
    """A visual feature vector/map produced by feature extraction.

    The full embedding-space compatibility contract (model family,
    checkpoint, dimension, normalization) is owned by the Feature
    Extraction capability; here, ``embedding_space_id`` is only an opaque
    reference — the same pattern Ingestion used for ``calibration_id``
    before the full calibration contract existed.

    Attributes:
        feature_id: Identity local to the owning ``PerceptionResult``.
        scope: Whether this feature is dense (a spatial map over the
            whole image), global (one vector for the whole image), or
            region (scoped to one ``Region2D``).
        embedding_space_id: Opaque reference to the embedding space this
            feature's values live in. Two features are only comparable
            when this reference matches.
        shape: Tensor shape of the feature payload, e.g. ``(768,)`` for a
            global vector or ``(64, 64, 384)`` for a dense map.
        dtype: Payload element type, e.g. ``"float32"``.
        payload_reference: Reference to the feature payload (e.g. a
            debug/output file path) — never the raw tensor inline.
        provenance: Backend that produced this feature.
        region_id: The ``Region2D`` this feature is scoped to. Required
            when ``scope`` is ``REGION``, ``None`` otherwise.
        normalization: Human-readable normalization applied, e.g.
            ``"l2"``, ``"none"``. ``None`` when not documented.
    """

    feature_id: FeatureId
    scope: FeatureScope
    embedding_space_id: str
    shape: tuple[int, ...]
    dtype: str
    payload_reference: str
    provenance: BackendProvenance
    region_id: RegionId | None = None
    normalization: str | None = None

    def __post_init__(self) -> None:
        """Validate ``region_id`` presence matches ``scope``.

        Raises:
            ValueError: If ``scope`` is ``REGION`` without a
                ``region_id``, or ``region_id`` is set while ``scope`` is
                not ``REGION``.
        """
        if self.scope is FeatureScope.REGION and self.region_id is None:
            raise ValueError("region_id is required when scope is REGION")
        if self.scope is not FeatureScope.REGION and self.region_id is not None:
            raise ValueError("region_id must be None unless scope is REGION")


@dataclass(frozen=True, kw_only=True)
class SemanticClaim:
    """A candidate semantic interpretation of visual evidence.

    A claim preserves uncertainty rather than collapsing directly to a
    hard label: ``confidence=None`` explicitly means "unscored", never
    equivalent to a score of ``0.0`` or ``1.0``.

    Attributes:
        claim_id: Identity local to the owning ``PerceptionResult``.
        text: The claim's label/text.
        role: Whether this is the ``PRIMARY`` interpretation or an
            ``ALTERNATIVE`` hypothesis.
        provenance: Backend/prompt that produced this claim.
        category: Optional coarse category for the claim.
        confidence: Optional score in ``[0, 1]``. ``None`` means unscored.
        region_id: The ``Region2D`` this claim describes, when
            region-scoped. ``None`` for a scene-level claim.
    """

    claim_id: ClaimId
    text: str
    role: HypothesisRole
    provenance: BackendProvenance
    category: str | None = None
    confidence: float | None = None
    region_id: RegionId | None = None

    def __post_init__(self) -> None:
        """Validate ``confidence`` is a valid score when present.

        Raises:
            ValueError: If ``confidence`` is set and outside ``[0, 1]``.
        """
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True, kw_only=True)
class SceneContext:
    """Scene-level semantic evidence.

    Scene-level evidence must never silently override region-level
    evidence; a consumer decides how to reconcile the two explicitly.

    Attributes:
        claims: Scene-level semantic claims. Every claim must have
            ``region_id=None``.
        provenance: Backend that produced this scene context.
    """

    claims: Sequence[SemanticClaim]
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        """Validate no scene-level claim references a region.

        Raises:
            ValueError: If any claim in ``claims`` has a non-``None``
                ``region_id``.
        """
        if any(claim.region_id is not None for claim in self.claims):
            raise ValueError("SceneContext claims must not reference a region_id")


@dataclass(frozen=True, kw_only=True)
class PerceptionRun:
    """One configured execution of Visual Perception over a sequence selection.

    Attributes:
        run_id: Identity of this run.
        run_index: Monotonic index within its sequence-artifact +
            capability scope (``docs/ARTIFACTS.md`` run-identity
            convention); not a global identity.
        sequence_artifact_id: Canonical sequence this run processed
            (:class:`contextmap.ingestion.SequenceArtifactId`, as ``str``).
        selection_id: Deterministic identity of the sequence selection
            processed (:func:`contextmap.ingestion.selection_identity`).
        enabled_capabilities: Capability names enabled for this run, e.g.
            ``{"region_discovery", "feature_extraction"}``.
        backend_provenance: One :class:`BackendProvenance` per enabled
            capability, keyed by capability name.
        code_version: Perception code identity, when available.
    """

    run_id: PerceptionRunId
    run_index: int
    sequence_artifact_id: str
    selection_id: str
    enabled_capabilities: frozenset[str]
    backend_provenance: Mapping[str, BackendProvenance]
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Validate ``run_index`` is non-negative.

        Raises:
            ValueError: If ``run_index`` is negative.
        """
        if self.run_index < 0:
            raise ValueError("run_index must be >= 0")


@dataclass(frozen=True, kw_only=True)
class PerceptionResult:
    """The output of one :class:`PerceptionRun` for one physical `SourceObservation`.

    A ``PerceptionResult`` never represents accumulated evidence across
    multiple runs or multiple physical observations: reprocessing the
    same ``SourceObservation`` in another run always produces a new,
    distinct ``PerceptionResult``, never a mutation of a prior one.

    Attributes:
        result_id: Identity of this result.
        source_observation_id: The physical observation this result was
            produced for (owned by Ingestion; never mutated by
            perception).
        run_id: The ``PerceptionRun`` that produced this result.
        sequence_artifact_id: Canonical sequence the source observation
            belongs to, duplicated here (also on ``PerceptionRun``) so a
            result is traceable without loading its run.
        created_at: ISO 8601 UTC timestamp of when this result was
            produced.
        regions: Accepted and rejected ``Region2D`` candidates.
        features: Extracted ``VisualFeature``s.
        claims: Semantic claims not scoped to ``scene_context``.
        scene_context: Scene-level evidence, when produced.
    """

    result_id: PerceptionResultId
    source_observation_id: SourceObservationId
    run_id: PerceptionRunId
    sequence_artifact_id: str
    created_at: str
    regions: Sequence[Region2D] = ()
    features: Sequence[VisualFeature] = ()
    claims: Sequence[SemanticClaim] = ()
    scene_context: SceneContext | None = None

    def __post_init__(self) -> None:
        """Validate region identities are unique and every reference resolves.

        Raises:
            ValueError: If ``regions`` has a duplicate ``region_id``, or a
                feature/claim references a ``region_id`` absent from
                ``regions``.
        """
        region_ids = {region.region_id for region in self.regions}
        if len(region_ids) != len(self.regions):
            raise ValueError("duplicate region_id in PerceptionResult")
        for feature in self.features:
            if feature.region_id is not None and feature.region_id not in region_ids:
                raise ValueError(f"feature references unknown region_id: {feature.region_id!r}")
        for claim in self.claims:
            if claim.region_id is not None and claim.region_id not in region_ids:
                raise ValueError(f"claim references unknown region_id: {claim.region_id!r}")


@dataclass(frozen=True, kw_only=True)
class PreparedImage:
    """The image-preparation output consumed by discovery/extraction/interpretation ports.

    This is Visual Perception Core's canonical contract for "an image
    ready for capability backends to consume" — any image-preparation
    implementation (resize, rectification, cropping, valid-area masking,
    ...) must produce this shape. The transformations actually applied
    are recorded for audit, but their specific parameters/implementation
    are not part of this core contract.

    Attributes:
        source_observation_id: The physical observation this was
            prepared from.
        payload_reference: Reference to the prepared image payload
            (e.g. a debug/output file path) — never raw pixel data
            inline.
        width: Prepared image width in pixels.
        height: Prepared image height in pixels.
        transformations: Human-readable, ordered record of the
            preparation steps applied, e.g. ``("resize", "rectify")``.
            Empty when the source observation was used unmodified.
    """

    source_observation_id: SourceObservationId
    payload_reference: str
    width: int
    height: int
    transformations: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate the prepared image has positive extent.

        Raises:
            ValueError: If ``width`` or ``height`` is not positive.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")


@dataclass(frozen=True, kw_only=True)
class SemanticSupport:
    """A scorer's assessment of how well visual evidence supports one SemanticClaim.

    Scoring never mutates the original claim: it produces a separate,
    referenceable judgement, keeping the claim itself immutable evidence.

    Attributes:
        claim_id: The claim being scored.
        support_score: Score in ``[0, 1]``: how well the visual evidence
            supports the claim.
        provenance: Backend that produced this support judgement.
    """

    claim_id: ClaimId
    support_score: float
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        """Validate ``support_score`` is a valid score.

        Raises:
            ValueError: If ``support_score`` is outside ``[0, 1]``.
        """
        if not 0.0 <= self.support_score <= 1.0:
            raise ValueError(f"support_score must be in [0, 1], got {self.support_score}")

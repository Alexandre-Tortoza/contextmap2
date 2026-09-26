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
from math import isfinite
from typing import NewType

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.region_models import (
    ArtifactReference,
    CoordinateConvention,
    InlineMask,
    JsonScalar,
    RegionProvenance,
)

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

ScoreId = NewType("ScoreId", str)
"""Identity of one semantic scoring judgement within a PerceptionResult."""


class HypothesisRole(Enum):
    """Whether a :class:`SemanticClaim` is the leading or a secondary hypothesis."""

    PRIMARY = "primary"
    ALTERNATIVE = "alternative"


class SemanticRegionKind(Enum):
    """Whether a semantic hypothesis describes a countable thing or amorphous stuff."""

    THING = "thing"
    STUFF = "stuff"


class SemanticScoreType(Enum):
    """Numerical semantics of a :class:`SemanticScore` value."""

    COSINE_SIMILARITY = "cosine_similarity"


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

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        """Validate the box has positive extent.

        Raises:
            ValueError: If ``width`` or ``height`` is not positive.
        """
        values = (self.x, self.y, self.width, self.height)
        if not all(isfinite(value) for value in values):
            raise ValueError("bounding box coordinates and dimensions must be finite")
        if self.x < 0 or self.y < 0:
            raise ValueError("x and y must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    @property
    def x_min(self) -> float:
        """Return the left edge in pixels."""
        return self.x

    @property
    def y_min(self) -> float:
        """Return the top edge in pixels."""
        return self.y

    @property
    def x_max(self) -> float:
        """Return the exclusive right edge in pixels."""
        return self.x + self.width

    @property
    def y_max(self) -> float:
        """Return the exclusive bottom edge in pixels."""
        return self.y + self.height


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
            belongs to the frozen canonical set. No production discovery
            backend emits ``is_accepted=False``: the canonical path records
            each rejected candidate as a ``RejectedRegionCandidate`` in the
            frame's ``RegionDiscoveryAudit``, persisted in the perception run.
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
    source_observation_id: SourceObservationId | None = None
    image_width: int | None = None
    image_height: int | None = None
    area_pixels: float | None = None
    contributor_candidate_ids: tuple[str, ...] = ()
    discovery_provenance: tuple[RegionProvenance, ...] = ()
    mask: InlineMask | None = None
    coordinate_convention: CoordinateConvention = CoordinateConvention.PIXEL_XY_TOP_LEFT

    def __post_init__(self) -> None:
        """Validate rejection metadata, image space, and geometry.

        Raises:
            ValueError: If ``rejection_reason`` is set while
                ``is_accepted`` is ``True``, if the image dimensions are
                partial or not positive, if ``bounding_box`` extends past
                the image when its dimensions are known, or if the
                area, contributor ids, or inline mask are invalid.
        """
        if self.is_accepted and self.rejection_reason is not None:
            raise ValueError("rejection_reason must be None when is_accepted is True")
        if (self.image_width is None) != (self.image_height is None):
            raise ValueError("image_width and image_height must be provided together")
        if (
            self.image_width is not None
            and self.image_height is not None
            and (self.image_width <= 0 or self.image_height <= 0)
        ):
            raise ValueError("region image dimensions must be positive")
        # Mesma regra de RegionCandidate: a caixa é semiaberta, então tocar a borda é válido.
        if (
            self.image_width is not None
            and self.image_height is not None
            and (
                self.bounding_box.x_max > self.image_width
                or self.bounding_box.y_max > self.image_height
            )
        ):
            raise ValueError(
                "bounding box extends outside the region image: "
                f"x_max={self.bounding_box.x_max}, y_max={self.bounding_box.y_max} for a "
                f"{self.image_width}x{self.image_height} image"
            )
        if self.area_pixels is not None and (
            not isfinite(self.area_pixels) or self.area_pixels <= 0
        ):
            raise ValueError("area_pixels must be positive and finite")
        if any(not candidate_id for candidate_id in self.contributor_candidate_ids):
            raise ValueError("contributor candidate ids must not be empty")
        if len(set(self.contributor_candidate_ids)) != len(self.contributor_candidate_ids):
            raise ValueError("contributor candidate ids must be unique")
        if (
            self.mask is not None
            and self.image_width is not None
            and (self.mask.width != self.image_width or self.mask.height != self.image_height)
        ):
            raise ValueError("inline mask dimensions must match the region image dimensions")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible canonical region representation."""
        mask = None if self.mask is None else self.mask.to_dict()
        return {
            "region_id": str(self.region_id),
            "bounding_box": {
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
            },
            "provenance": {
                "backend_id": self.provenance.backend_id,
                "capability": self.provenance.capability,
                "provider": self.provenance.provider,
                "model": self.provenance.model,
                "version": self.provenance.version,
                "configuration_fingerprint": self.provenance.configuration_fingerprint,
            },
            "mask_reference": self.mask_reference,
            "region_kind": self.region_kind,
            "is_accepted": self.is_accepted,
            "rejection_reason": self.rejection_reason,
            "source_observation_id": (
                None if self.source_observation_id is None else str(self.source_observation_id)
            ),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "area_pixels": self.area_pixels,
            "contributor_candidate_ids": list(self.contributor_candidate_ids),
            "discovery_provenance": [item.to_dict() for item in self.discovery_provenance],
            "mask": mask,
            "coordinate_convention": self.coordinate_convention.value,
        }


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
class SemanticAttribute:
    """One backend-provided attribute attached to a semantic hypothesis."""

    name: str
    value: JsonScalar

    def __post_init__(self) -> None:
        """Reject unnamed attributes."""
        if not self.name.strip():
            raise ValueError("semantic attribute name must not be empty")


@dataclass(frozen=True, kw_only=True)
class SemanticEvidenceReference:
    """Identify one immutable input or intermediate supporting semantic evidence."""

    evidence_type: str
    evidence_id: str

    def __post_init__(self) -> None:
        """Require both evidence type and identity for auditability."""
        if not self.evidence_type.strip():
            raise ValueError("semantic evidence type must not be empty")
        if not self.evidence_id.strip():
            raise ValueError("semantic evidence id must not be empty")


@dataclass(frozen=True, kw_only=True)
class SemanticInferenceProvenance:
    """Trace a semantic output through backend, task, prompt, schema, and raw response."""

    backend: BackendProvenance
    task_identity: str
    prompt_template_id: str
    output_schema_version: str
    raw_response_reference: str | None = None

    def __post_init__(self) -> None:
        """Require versioned semantic policy identities."""
        if self.backend.capability != "semantic_interpreter":
            raise ValueError("semantic inference backend capability must be semantic_interpreter")
        for field_name, value in (
            ("task_identity", self.task_identity),
            ("prompt_template_id", self.prompt_template_id),
            ("output_schema_version", self.output_schema_version),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.raw_response_reference is not None and not self.raw_response_reference.strip():
            raise ValueError("raw_response_reference must not be empty when provided")


@dataclass(frozen=True, kw_only=True)
class SemanticClaim:
    """A candidate semantic interpretation of visual evidence.

    A claim preserves uncertainty rather than collapsing directly to a
    hard label: ``confidence=None`` explicitly means "unscored", never
    equivalent to a score of ``0.0`` or ``1.0``.

    Attributes:
        claim_id: Identity local to the owning ``PerceptionResult``.
        source_observation_id: Physical observation interpreted by the backend.
        perception_result_id: Inference result that owns this claim.
        hypothesis: The backend-proposed label or semantic hypothesis.
        role: Whether this is the ``PRIMARY`` interpretation or an
            ``ALTERNATIVE`` hypothesis.
        provenance: Backend, task, prompt, schema, and raw-response traceability.
        category: Optional coarse category for the claim.
        confidence: Optional score in ``[0, 1]``. ``None`` means unscored.
        region_id: The frozen ``Region2D`` this claim describes, when
            region-scoped. ``None`` for a scene-level claim.
    """

    claim_id: ClaimId
    source_observation_id: SourceObservationId
    perception_result_id: PerceptionResultId
    hypothesis: str
    role: HypothesisRole
    provenance: SemanticInferenceProvenance
    category: str | None = None
    region_kind: SemanticRegionKind | None = None
    attributes: tuple[SemanticAttribute, ...] = ()
    confidence: float | None = None
    region_id: RegionId | None = None
    evidence_references: tuple[SemanticEvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        """Validate ``confidence`` is a valid score when present.

        Raises:
            ValueError: If ``confidence`` is set and outside ``[0, 1]``.
        """
        if not self.hypothesis.strip():
            raise ValueError("hypothesis must not be empty")
        if self.confidence is not None and (
            not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        attribute_names = [attribute.name for attribute in self.attributes]
        if len(set(attribute_names)) != len(attribute_names):
            raise ValueError("semantic attribute names must be unique")
        evidence_keys = [
            (reference.evidence_type, reference.evidence_id)
            for reference in self.evidence_references
        ]
        if len(set(evidence_keys)) != len(evidence_keys):
            raise ValueError("semantic evidence references must be unique")


@dataclass(frozen=True, kw_only=True)
class SceneContext:
    """Scene-level semantic evidence.

    Scene-level evidence must never silently override region-level
    evidence; a consumer decides how to reconcile the two explicitly.

    Attributes:
        source_observation_id: Physical observation interpreted by the backend.
        perception_result_id: Inference result that owns this context.
        claims: Scene-level semantic claims. Every claim must have
            ``region_id=None``.
        provenance: Backend, task, prompt, schema, and raw-response traceability.
    """

    source_observation_id: SourceObservationId
    perception_result_id: PerceptionResultId
    provenance: SemanticInferenceProvenance
    scene_type: str | None = None
    environment: str | None = None
    layout: str | None = None
    lighting: str | None = None
    visibility: str | None = None
    navigability: str | None = None
    claims: Sequence[SemanticClaim] = ()
    evidence_references: tuple[SemanticEvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        """Validate no scene-level claim references a region.

        Raises:
            ValueError: If any claim in ``claims`` has a non-``None``
                ``region_id``.
        """
        if any(claim.region_id is not None for claim in self.claims):
            raise ValueError("SceneContext claims must not reference a region_id")
        if any(claim.source_observation_id != self.source_observation_id for claim in self.claims):
            raise ValueError("SceneContext claim source_observation_id must match its context")
        if any(claim.perception_result_id != self.perception_result_id for claim in self.claims):
            raise ValueError("SceneContext claim perception_result_id must match its context")
        evidence_keys = [
            (reference.evidence_type, reference.evidence_id)
            for reference in self.evidence_references
        ]
        if len(set(evidence_keys)) != len(evidence_keys):
            raise ValueError("scene evidence references must be unique")


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
    semantic_scores: Sequence[SemanticScore] = ()

    def __post_init__(self) -> None:
        """Validate local identities are unique and every reference resolves.

        Raises:
            ValueError: If regions, features, or claims contain a
                duplicate local identity, or a feature/claim references
                a ``region_id`` absent from ``regions``.
        """
        region_ids = {region.region_id for region in self.regions}
        if len(region_ids) != len(self.regions):
            raise ValueError("duplicate region_id in PerceptionResult")
        feature_ids = {feature.feature_id for feature in self.features}
        if len(feature_ids) != len(self.features):
            raise ValueError("duplicate feature_id in PerceptionResult")
        claim_ids = {claim.claim_id for claim in self.claims}
        if len(claim_ids) != len(self.claims):
            raise ValueError("duplicate claim_id in PerceptionResult")
        for feature in self.features:
            if feature.region_id is not None and feature.region_id not in region_ids:
                raise ValueError(f"feature references unknown region_id: {feature.region_id!r}")
        for claim in self.claims:
            if claim.region_id is not None and claim.region_id not in region_ids:
                raise ValueError(f"claim references unknown region_id: {claim.region_id!r}")
            if claim.source_observation_id != self.source_observation_id:
                raise ValueError("claim source_observation_id must match PerceptionResult")
            if claim.perception_result_id != self.result_id:
                raise ValueError("claim perception_result_id must match PerceptionResult")
        if self.scene_context is not None:
            if self.scene_context.source_observation_id != self.source_observation_id:
                raise ValueError("scene context source_observation_id must match PerceptionResult")
            if self.scene_context.perception_result_id != self.result_id:
                raise ValueError("scene context perception_result_id must match PerceptionResult")
        all_claim_ids = set(claim_ids)
        if self.scene_context is not None:
            all_claim_ids.update(claim.claim_id for claim in self.scene_context.claims)
        score_ids = {score.score_id for score in self.semantic_scores}
        if len(score_ids) != len(self.semantic_scores):
            raise ValueError("duplicate score_id in PerceptionResult")
        features_by_id = {feature.feature_id: feature for feature in self.features}
        for score in self.semantic_scores:
            if score.claim_id not in all_claim_ids:
                raise ValueError(f"semantic score references unknown claim_id: {score.claim_id!r}")
            scored_feature = features_by_id.get(score.feature_id)
            if scored_feature is None:
                raise ValueError(
                    f"semantic score references unknown feature_id: {score.feature_id!r}"
                )
            if score.embedding_space_id != scored_feature.embedding_space_id:
                raise ValueError("semantic score embedding_space_id must match its feature")
            if score.source_observation_id != self.source_observation_id:
                raise ValueError("semantic score source_observation_id must match PerceptionResult")
            if score.perception_result_id != self.result_id:
                raise ValueError("semantic score perception_result_id must match PerceptionResult")


@dataclass(frozen=True, slots=True)
class ValidRegion:
    """Explicitly identify pixels eligible for visual processing."""

    mask: InlineMask
    reason: str
    source: str

    def __post_init__(self) -> None:
        """Require auditable constraint motivation and source."""
        if not self.reason or not self.source:
            raise ValueError("valid region reason and source must not be empty")


@dataclass(frozen=True, slots=True)
class ExclusionRegion:
    """Explicitly identify pixels excluded from visual processing."""

    name: str
    mask: InlineMask
    reason: str
    source: str

    def __post_init__(self) -> None:
        """Require a stable diagnostic name, motivation, and source."""
        if not self.name or not self.reason or not self.source:
            raise ValueError("exclusion name, reason, and source must not be empty")


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    """Describe one applied image transformation in execution order."""

    operation: str
    provenance_source: str
    parameters: tuple[tuple[str, JsonScalar], ...]
    input_dimensions: tuple[int, int]
    output_dimensions: tuple[int, int]
    output_image: ArtifactReference
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the structured transformation record."""
        if not self.operation:
            raise ValueError("transformation operation must not be empty")
        if not self.provenance_source:
            raise ValueError("transformation provenance source must not be empty")
        for width, height in (self.input_dimensions, self.output_dimensions):
            if width <= 0 or height <= 0:
                raise ValueError("transformation dimensions must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible audit record."""
        return {
            "operation": self.operation,
            "provenance_source": self.provenance_source,
            "parameters": [{"name": name, "value": value} for name, value in self.parameters],
            "input_dimensions": list(self.input_dimensions),
            "output_dimensions": list(self.output_dimensions),
            "output_image": self.output_image.to_dict(),
            "warnings": list(self.warnings),
        }


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
        transformations: Structured, ordered record of preparation steps.
        payload_artifact: Optional content-addressed metadata for the payload.
        valid_region: Optional explicit eligibility mask.
        exclusion_regions: Optional named exclusion masks.
    """

    source_observation_id: SourceObservationId
    payload_reference: str
    width: int
    height: int
    transformations: Sequence[TransformationRecord] = ()
    payload_artifact: ArtifactReference | None = None
    valid_region: ValidRegion | None = None
    exclusion_regions: Sequence[ExclusionRegion] = ()

    def __post_init__(self) -> None:
        """Validate the prepared image has positive extent.

        Raises:
            ValueError: If ``width`` or ``height`` is not positive.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not self.payload_reference:
            raise ValueError("payload_reference must not be empty")
        if (
            self.payload_artifact is not None
            and self.payload_artifact.uri != self.payload_reference
        ):
            raise ValueError("payload artifact uri must match payload_reference")
        constraints = (() if self.valid_region is None else (self.valid_region.mask,)) + tuple(
            region.mask for region in self.exclusion_regions
        )
        if any(mask.width != self.width or mask.height != self.height for mask in constraints):
            raise ValueError("spatial constraints must match final prepared image dimensions")
        names = [region.name for region in self.exclusion_regions]
        if len(set(names)) != len(names):
            raise ValueError("exclusion regions must have unique names")

    def to_dict(self) -> dict[str, object]:
        """Return an inspectable representation for manifests and diagnostics."""
        return {
            "source_observation_id": str(self.source_observation_id),
            "payload_reference": self.payload_reference,
            "payload_artifact": (
                None if self.payload_artifact is None else self.payload_artifact.to_dict()
            ),
            "width": self.width,
            "height": self.height,
            "transformations": [record.to_dict() for record in self.transformations],
            "valid_region": (
                None
                if self.valid_region is None
                else {
                    "mask": self.valid_region.mask.to_dict(),
                    "reason": self.valid_region.reason,
                    "source": self.valid_region.source,
                }
            ),
            "exclusion_regions": [
                {
                    "name": region.name,
                    "mask": region.mask.to_dict(),
                    "reason": region.reason,
                    "source": region.source,
                }
                for region in self.exclusion_regions
            ],
        }


@dataclass(frozen=True, kw_only=True)
class SemanticScore:
    """One scorer's support judgement for a claim and visual feature.

    ``value`` keeps the scale named by ``score_type``. In particular, cosine
    similarity is not remapped to ``[0, 1]`` and is never presented as a
    probability. A claim with no score remains valid and is represented by the
    absence of a ``SemanticScore`` record, not by a zero value.
    """

    score_id: ScoreId
    claim_id: ClaimId
    feature_id: FeatureId
    score_type: SemanticScoreType
    value: float
    calibrated_probability: float | None
    embedding_space_id: str
    source_observation_id: SourceObservationId
    perception_result_id: PerceptionResultId
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        """Validate numerical semantics and scorer provenance."""
        for field_name, value in (
            ("score_id", str(self.score_id)),
            ("claim_id", str(self.claim_id)),
            ("feature_id", str(self.feature_id)),
            ("embedding_space_id", self.embedding_space_id),
            ("source_observation_id", str(self.source_observation_id)),
            ("perception_result_id", str(self.perception_result_id)),
            ("scorer_id", self.provenance.backend_id),
            ("scorer_model", self.provenance.model),
            ("scorer_version", self.provenance.version),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if not isfinite(self.value):
            raise ValueError("semantic score value must be finite")
        if self.score_type is SemanticScoreType.COSINE_SIMILARITY and not -1.0 <= self.value <= 1.0:
            raise ValueError("cosine similarity must be in [-1, 1]")
        if self.calibrated_probability is not None and (
            not isfinite(self.calibrated_probability)
            or not 0.0 <= self.calibrated_probability <= 1.0
        ):
            raise ValueError("calibrated_probability must be finite and in [0, 1]")
        if self.provenance.capability != "semantic_scorer":
            raise ValueError("semantic score provenance capability must be semantic_scorer")


@dataclass(frozen=True, kw_only=True)
class SemanticSupport:
    """Legacy normalized scorer evidence retained for downstream compatibility.

    New semantic scorer adapters emit :class:`SemanticScore`, whose raw value keeps
    its declared score semantics. This contract remains separate for existing
    downstream evidence explicitly produced as normalized support in [0, 1];
    raw cosine values must never be coerced into it implicitly.
    """

    claim_id: ClaimId
    support_score: float
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        """Validate the normalized support value and scorer provenance."""
        if not 0.0 <= self.support_score <= 1.0:
            raise ValueError(f"support_score must be in [0, 1], got {self.support_score}")
        if self.provenance.capability != "semantic_scorer":
            raise ValueError("semantic support provenance capability must be semantic_scorer")

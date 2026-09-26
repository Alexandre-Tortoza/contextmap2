"""Versioned capability and compatibility matrix of the perception experiment (#522).

The experiment of milestone #22 varies backends *and* pipeline compositions. Before any model
is loaded, an experiment manifest must know, for each model family, which native operation a
ContextMap2 role actually exposes, whether it is selectable today, and which producer ->
consumer edges are admissible. This module freezes that knowledge as data:

* a :class:`Capability` is one native operation of one model family (or one ContextMap2
  operation that consumes it), with its role, status, evidence contracts, controls,
  provenance, limitations, compatible roles, comparison group and metric family;
* a :class:`Composition` is one producer -> consumer edge, classified as supported, planned,
  blocked, incompatible or not scientifically comparable, with the reason;
* two entries are comparable only when they share a comparison group, which encodes task
  semantics, input conditioning and output geometry together: emitting the same geometry
  is never enough.

The matrix describes admissible experimental compositions; it does not choose a canonical
pipeline. The tests tie every ``supported`` entry to the runtime catalog and composition root.
See ``src/contextmap/evaluation/docs/capability-matrix.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from contextmap.evaluation._validation import require_text, require_unique
from contextmap.evaluation.metrics import EvaluationStage


class CapabilityMatrixError(ValueError):
    """Raised when the matrix is inconsistent or a composition is not admissible."""


class ImplementationStatus(Enum):
    """Whether ContextMap2 can run a native operation today."""

    SUPPORTED = "supported"
    """Selectable through the runtime configuration today (catalog and composition root)."""

    PLANNED = "planned"
    """An open milestone issue will make it selectable; not selectable yet."""

    BLOCKED = "blocked"
    """Needs upstream code or weights that are not publicly available."""

    OUT_OF_SCOPE = "out_of_scope"
    """Native to the model but not part of the milestone #22 experiment."""


class CompositionStatus(Enum):
    """Whether a producer -> consumer edge is admissible in an experimental pipeline."""

    SUPPORTED = "supported"
    """The runtime delivers the producer's evidence to the consumer today."""

    PLANNED = "planned"
    """An open milestone issue will deliver it; excluded until then."""

    BLOCKED = "blocked"
    """Waits for upstream assets; excluded until then."""

    INCOMPATIBLE = "incompatible"
    """The consumer's contract cannot use the producer's evidence."""

    NOT_COMPARABLE = "not_scientifically_comparable"
    """It runs, but its result cannot stand in for the path it appears to substitute."""


class Role(Enum):
    """The ContextMap2 role an operation fills, behind its own evidence contract."""

    REGION_DISCOVERY = "region_discovery"
    REGION_GROUNDING = "region_grounding"
    REGION_REFINEMENT = "region_refinement"
    DENSE_FEATURES = "dense_features"
    REGION_FEATURES = "region_features"
    SEMANTIC_VIEW_ASSEMBLY = "semantic_view_assembly"
    SEMANTIC_INTERPRETATION = "semantic_interpretation"
    SEMANTIC_SCORING = "semantic_scoring"
    SENSOR_ASSOCIATION = "sensor_association"
    DENSE_FEATURE_SAMPLING = "dense_feature_sampling"
    SEMANTIC_FUSION = "semantic_fusion"
    PROPOSAL_3D = "proposal_3d"
    METRIC_VERIFICATION = "metric_verification"


class Geometry(Enum):
    """The geometry an operation outputs."""

    NONE = "none"
    BOX = "box"
    MASK = "mask"
    BOX_OR_MASK = "box_or_mask"
    """The native parser decides per response; the geometry type is not controlled."""
    POINT = "point"
    DENSE_GRID = "dense_grid"
    POINT_SET_3D = "point_set_3d"
    BOX_3D = "box_3d"


@dataclass(frozen=True, kw_only=True)
class RuntimeBinding:
    """Where the runtime selects an operation.

    Attributes:
        stage_id: The stage of the canonical preset that runs it.
        component_id: The variation point, or ``None`` for an operation the stage itself owns.
        backend_id: The backend of that variation point.
        settings: ``(parameter, accepted values)`` that select this native operation among
            the ones the backend offers (a Florence-2 ``task``, a SAM3 ``strategy``, the
            ``policy_id`` of a LocateAnything query in its ``query_set``).
    """

    stage_id: str
    component_id: str | None = None
    backend_id: str | None = None
    settings: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        """Require a stage, and a backend exactly when a component is named."""
        require_text("stage_id", self.stage_id)
        if (self.component_id is None) != (self.backend_id is None):
            raise ValueError("a runtime binding names a component and its backend together")
        if self.settings and self.component_id is None:
            raise ValueError("only a backend has settings that select a native operation")


@dataclass(frozen=True, kw_only=True)
class Control:
    """A prompt/task or model/runtime control and whether it can be set today.

    Attributes:
        name: The configuration parameter (or reserved group) when supported; the control's
            name when it is still planned.
        status: Supported, planned or blocked.
        issue: The issue that delivers a planned or blocked control.
    """

    name: str
    status: ImplementationStatus
    issue: int | None = None

    def __post_init__(self) -> None:
        """Require the delivering issue of a control that is not available yet."""
        require_text("control name", self.name)
        if self.status in _AWAITING and self.issue is None:
            raise ValueError(f"control {self.name!r} is {self.status.value} and needs its issue")


_AWAITING = frozenset({ImplementationStatus.PLANNED, ImplementationStatus.BLOCKED})


@dataclass(frozen=True, kw_only=True)
class Capability:
    """One native operation and what ContextMap2 does with it.

    Attributes:
        capability_id: ``<family>.<operation>``.
        backend: The model family (``ContextMap2`` for a stage's own operation).
        native_operation: What the model natively does, in its own terms.
        role: The ContextMap2 role it fills; ``None`` when out of scope.
        adapter: Dotted path of the class or function that implements the role, when it exists.
        status: Supported, planned, blocked or out of scope.
        runtime: Where the runtime selects it. Required when supported; for a planned entry
            it names the identity the delivering issue introduces.
        input_evidence: The contracts it consumes.
        output_evidence: The contracts it produces.
        geometry: The geometry of its output.
        prompt_controls: What conditions the model's task.
        model_controls: Model, checkpoint and runtime settings.
        provenance: The recorded fields that make one output traceable.
        limitations: What the evidence cannot be used for, or what is not validated.
        upstream_roles: Roles whose evidence it can consume.
        downstream_roles: Roles that can consume its evidence.
        comparison_group: Task semantics + input conditioning + output geometry. Only entries
            of the same group are compared directly; ``None`` means comparable with nothing.
        metric_family: The evaluation stage whose registry metrics measure it.
        status_reason: Why it is not supported (what is missing, blocking or excluded).
        issue: The issue that delivers a planned entry or unblocks a blocked one.
    """

    capability_id: str
    backend: str
    native_operation: str
    role: Role | None
    adapter: str | None
    status: ImplementationStatus
    runtime: RuntimeBinding | None
    input_evidence: tuple[str, ...]
    output_evidence: tuple[str, ...]
    geometry: Geometry
    prompt_controls: tuple[Control, ...] = ()
    model_controls: tuple[Control, ...] = ()
    provenance: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    upstream_roles: tuple[Role, ...] = ()
    downstream_roles: tuple[Role, ...] = ()
    comparison_group: str | None = None
    metric_family: EvaluationStage | None = None
    status_reason: str | None = None
    issue: int | None = None

    def __post_init__(self) -> None:
        """Tie the status to what the entry must declare."""
        require_text("capability_id", self.capability_id)
        require_text("backend", self.backend)
        require_text("native_operation", self.native_operation)
        if self.status is ImplementationStatus.SUPPORTED:
            if self.runtime is None or self.role is None or self.adapter is None:
                raise ValueError(
                    f"supported capability {self.capability_id!r} needs its runtime binding, "
                    "role and adapter"
                )
            if self.status_reason is not None:
                raise ValueError(
                    f"supported capability {self.capability_id!r} has no status_reason"
                )
            return
        if not self.status_reason:
            raise ValueError(
                f"{self.status.value} capability {self.capability_id!r} needs a status_reason"
            )
        if self.status in _AWAITING and self.issue is None:
            raise ValueError(
                f"{self.status.value} capability {self.capability_id!r} needs its issue"
            )
        if self.status is ImplementationStatus.BLOCKED and self.runtime is not None:
            raise ValueError(f"blocked capability {self.capability_id!r} has no runtime binding")
        if self.status is ImplementationStatus.OUT_OF_SCOPE and (
            self.role is not None
            or self.runtime is not None
            or self.upstream_roles
            or self.downstream_roles
            or self.comparison_group is not None
        ):
            raise ValueError(
                f"capability {self.capability_id!r} is out of scope: it takes no role, runtime "
                "binding, compatible role or comparison group"
            )


@dataclass(frozen=True, kw_only=True)
class Composition:
    """One producer -> consumer edge of an experimental pipeline.

    Attributes:
        producer: The capability whose evidence flows.
        consumer: The capability that consumes it.
        status: Supported, planned, blocked, incompatible or not scientifically comparable.
        note: How the evidence flows, or why it cannot.
        issue: The issue that delivers a planned edge or unblocks a blocked one.
        comparison_group: The group of the evidence the edge produces, when that evidence is
            compared with another capability or edge (a refinement chain, for example).
        metric_family: The evaluation stage that measures the edge's effect.
    """

    producer: str
    consumer: str
    status: CompositionStatus
    note: str
    issue: int | None = None
    comparison_group: str | None = None
    metric_family: EvaluationStage | None = None

    def __post_init__(self) -> None:
        """Require a note, and the issue of an edge that is not available yet."""
        require_text("producer", self.producer)
        require_text("consumer", self.consumer)
        require_text(f"note of {self.composition_id}", self.note)
        if (
            self.status in {CompositionStatus.PLANNED, CompositionStatus.BLOCKED}
            and self.issue is None
        ):
            raise ValueError(
                f"{self.status.value} composition {self.composition_id} needs its issue"
            )

    @property
    def composition_id(self) -> str:
        """Return ``<producer>-><consumer>``."""
        return f"{self.producer}->{self.consumer}"


@dataclass(frozen=True, kw_only=True)
class CapabilityMatrix:
    """The versioned matrix: every capability and every declared composition.

    Attributes:
        version: Changes whenever an entry changes.
        upstream_checked_on: The date upstream repositories and model cards were consulted.
        capabilities: Every capability, in documentation order.
        compositions: Every declared producer -> consumer edge. An undeclared edge is not
            admissible.
    """

    version: str
    upstream_checked_on: str
    capabilities: tuple[Capability, ...]
    compositions: tuple[Composition, ...]

    def __post_init__(self) -> None:
        """Reject duplicates, unknown endpoints and edges inconsistent with their endpoints."""
        require_text("version", self.version)
        require_text("upstream_checked_on", self.upstream_checked_on)
        try:
            require_unique("capability", (item.capability_id for item in self.capabilities))
            require_unique("composition", (item.composition_id for item in self.compositions))
        except ValueError as error:
            raise CapabilityMatrixError(f"{error}: an entry is declared twice") from error
        for composition in self.compositions:
            producer = self.capability(composition.producer)
            consumer = self.capability(composition.consumer)
            _check_composition(composition, producer, consumer)

    def capability(self, capability_id: str) -> Capability:
        """Return one capability.

        Raises:
            CapabilityMatrixError: If the matrix does not know it.
        """
        for item in self.capabilities:
            if item.capability_id == capability_id:
                return item
        raise CapabilityMatrixError(f"unknown capability {capability_id!r}")

    def composition(self, producer: str, consumer: str) -> Composition:
        """Return the declared edge from ``producer`` to ``consumer``.

        Raises:
            CapabilityMatrixError: If the edge is undeclared, which makes it inadmissible.
        """
        for item in self.compositions:
            if (item.producer, item.consumer) == (producer, consumer):
                return item
        raise CapabilityMatrixError(
            f"undeclared composition {producer} -> {consumer}: it is not an admissible edge"
        )

    def require_supported_composition(self, producer: str, consumer: str) -> None:
        """Refuse, before any model loads, an edge the runtime cannot deliver today.

        Raises:
            CapabilityMatrixError: If the edge is undeclared or not supported, with its status,
                delivering issue and reason.
        """
        composition = self.composition(producer, consumer)
        if composition.status is not CompositionStatus.SUPPORTED:
            issue = "" if composition.issue is None else f" (#{composition.issue})"
            raise CapabilityMatrixError(
                f"composition {composition.composition_id} is {composition.status.value}"
                f"{issue}: {composition.note}"
            )

    def comparable(self, first: str, second: str) -> bool:
        """Return whether two capabilities or edges (``producer->consumer``) compare directly.

        They must share a comparison group; an entry without one compares with nothing.
        """
        groups = [self._group(item) for item in (first, second)]
        return groups[0] is not None and groups[0] == groups[1]

    def _group(self, entry: str) -> str | None:
        if "->" in entry:
            producer, consumer = entry.split("->", 1)
            return self.composition(producer, consumer).comparison_group
        return self.capability(entry).comparison_group


def _check_composition(
    composition: Composition, producer: Capability, consumer: Capability
) -> None:
    """Require an edge whose status and roles agree with its two endpoints."""
    endpoints = (producer, consumer)
    if any(item.status is ImplementationStatus.OUT_OF_SCOPE for item in endpoints):
        raise CapabilityMatrixError(
            f"composition {composition.composition_id} uses an out-of-scope capability"
        )
    runs = {CompositionStatus.SUPPORTED, CompositionStatus.NOT_COMPARABLE}
    if composition.status in runs and any(
        item.status is not ImplementationStatus.SUPPORTED for item in endpoints
    ):
        raise CapabilityMatrixError(
            f"composition {composition.composition_id} is {composition.status.value}, so both "
            "endpoints must be supported"
        )
    if composition.status is CompositionStatus.INCOMPATIBLE:
        return
    if (
        consumer.role not in producer.downstream_roles
        or producer.role not in consumer.upstream_roles
    ):
        raise CapabilityMatrixError(
            f"composition {composition.composition_id} connects roles its endpoints do not "
            "declare compatible"
        )


# ----------------------------------------------------------------------------- the data


def _set(*names: str) -> tuple[Control, ...]:
    """Return controls that the backend configuration accepts today."""
    return tuple(Control(name=name, status=ImplementationStatus.SUPPORTED) for name in names)


def _planned(name: str, issue: int) -> Control:
    return Control(name=name, status=ImplementationStatus.PLANNED, issue=issue)


_VP = "visual_perception"
_BACKENDS = "contextmap.visual_perception.backends"
_REGION_DISCOVERY = "visual_perception.region_discovery"
_GROUNDING = "visual_perception.region_grounding"
_DENSE = "visual_perception.dense_features"
_REGION_FEATURES = "visual_perception.region_features"
_SEMANTIC = "visual_perception.semantic_interpretation"

_REGION_CONSUMERS = (
    Role.SENSOR_ASSOCIATION,
    Role.REGION_FEATURES,
    Role.SEMANTIC_VIEW_ASSEMBLY,
)
_REGION_PROVENANCE = (
    "BackendProvenance(backend_id, model, version, configuration_fingerprint)",
    "RegionProvenance(discovery pass, native proposal id)",
    "BackendScore(name, value, semantics)",
)
_FLORENCE_TEXT_PROVENANCE = (
    *_REGION_PROVENANCE,
    "NativeRegionText(task, prompt, text) -> RegionSemanticHint (stage evidence only)",
)
_FLORENCE_MODEL = _set(
    "checkpoint", "precision", "box_threshold", "generation_settings", "pass_config"
)
_SAM_PROVIDER = (
    "the model runtime is supplied by the caller (resources.providers); no bundled loader"
)
_GROUNDING_PROVENANCE = (
    "RegionGroundingExecution(request_id, query, rendered_prompt, raw_response)",
    "BackendProvenance(model, revision, configuration_fingerprint)",
    "runtime_identity and native diagnostics (never confidence)",
)
_LA_MODEL = _set(
    "model",
    "revision",
    "dtype",
    "generation_mode",
    "max_new_tokens",
    "temperature",
    "text_attention",
    "vision_attention",
    "runtime",
)
_LA_LIMITATIONS = (
    "runs next to the mandatory region_discovery backend; grounded regions are appended after "
    "the perception stage graph, so they reach neither region features nor region interpretation",
    "no calibrated confidence (#573); weights are non-commercial (NVIDIA License)",
    "never executed on real weights here",
)
_DENSE_PROVENANCE = (
    "EmbeddingSpace(family, checkpoint@revision, layer, dimension, normalization)",
    "DenseFeatureSampling(origin, stride, support) and coordinate_transform_id",
)
_INTERPRETATION_PROVENANCE = (
    "SemanticInterpretationRequest(views with sha256, prompt_template_id, output schema)",
    "RenderedSemanticPrompt.fingerprint and raw response",
    "BackendProvenance and effective configuration",
)
_SCORING_LIMITATIONS = (
    "cosine similarity in its own embedding space: never a probability, never pooled with "
    "another scorer's values (#530 evaluates it)",
)

CAPABILITY_MATRIX_VERSION = "1.0.0"
"""Version of :data:`CAPABILITY_MATRIX`; a changed entry is a new version."""

CAPABILITY_MATRIX = CapabilityMatrix(
    version=CAPABILITY_MATRIX_VERSION,
    upstream_checked_on="2026-09-25",
    capabilities=(
        # ------------------------------------------------------------------------- SAM2
        Capability(
            capability_id="sam2.automatic_mask_generation",
            backend="SAM2",
            native_operation="automatic mask generation (point-grid prompts, IoU/stability filter)",
            role=Role.REGION_DISCOVERY,
            adapter=f"{_BACKENDS}.sam2.Sam2RegionDiscovery",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(stage_id=_VP, component_id=_REGION_DISCOVERY, backend_id="sam2"),
            input_evidence=("PreparedImage",),
            output_evidence=("Region2D (mask)",),
            geometry=Geometry.MASK,
            model_controls=_set(
                "checkpoint",
                "precision",
                "predicted_iou_threshold",
                "stability_threshold",
                "automatic_mask_settings",
                "pass_config",
                "normalization_config",
            ),
            provenance=_REGION_PROVENANCE,
            limitations=(
                "unconditioned: proposes whatever is salient, with no category or label",
                "predicted_iou and stability_score are SAM-native, not probabilities",
                _SAM_PROVIDER,
            ),
            downstream_roles=_REGION_CONSUMERS,
            comparison_group="automatic_region_proposal.mask",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="sam2.box_prompt_refinement",
            backend="SAM2",
            native_operation="promptable image segmentation from a box or points (image predictor)",
            role=Role.REGION_REFINEMENT,
            adapter=None,
            status=ImplementationStatus.PLANNED,
            runtime=None,
            input_evidence=("PreparedImage", "Region2D (box) from region grounding"),
            output_evidence=("Region2D (mask) with contributor lineage",),
            geometry=Geometry.MASK,
            prompt_controls=(_planned("refinement_policy", 568),),
            model_controls=(_planned("checkpoint", 568),),
            provenance=("contributor proposal ids", "grounding and refinement provenance"),
            limitations=("uses the proposal only as a segmentation prompt; infers no label",),
            upstream_roles=(Role.REGION_GROUNDING,),
            downstream_roles=_REGION_CONSUMERS,
            comparison_group="prompted_mask_refinement",
            metric_family=EvaluationStage.REGION_DISCOVERY,
            status_reason="no refinement stage exists yet; the runtime identity is not decided",
            issue=568,
        ),
        Capability(
            capability_id="sam2.video_tracking",
            backend="SAM2",
            native_operation="video object segmentation with memory (masklets across frames)",
            role=None,
            adapter=None,
            status=ImplementationStatus.OUT_OF_SCOPE,
            runtime=None,
            input_evidence=("frame sequence", "prompts on one frame"),
            output_evidence=("masklets with object ids",),
            geometry=Geometry.MASK,
            status_reason=(
                "a masklet is cross-observation identity; ContextMap2 keeps perception per frame "
                "and assigns identity in Entity Resolution"
            ),
        ),
        # ------------------------------------------------------------------------- SAM3
        Capability(
            capability_id="sam3.text_concept_segmentation",
            backend="SAM3",
            native_operation="promptable concept segmentation of every instance of a noun phrase",
            role=Role.REGION_DISCOVERY,
            adapter=f"{_BACKENDS}.sam3.Sam3RegionDiscovery",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_REGION_DISCOVERY,
                backend_id="sam3",
                settings=(("strategy", ("text_prompt",)),),
            ),
            input_evidence=("PreparedImage", "concept phrase (configuration)"),
            output_evidence=("Region2D (mask)",),
            geometry=Geometry.MASK,
            prompt_controls=_set("prompt"),
            model_controls=_set(
                "checkpoint", "precision", "score_threshold", "mask_threshold", "pass_config"
            ),
            provenance=(*_REGION_PROVENANCE, "prompt and query id in the proposal provenance"),
            limitations=(
                "one concept phrase per run; prompt sets run as separate runs (#525)",
                "the bundled Sam3ImageProcessorRuntime executes only text_prompt; the automatic, "
                "point_grid, tracker and pcs strategies are rejected explicitly",
                "gated checkpoint; the official image model runs only under bfloat16",
                _SAM_PROVIDER,
            ),
            downstream_roles=_REGION_CONSUMERS,
            comparison_group="category_conditioned_instance_masks",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="sam3.exemplar_prompts",
            backend="SAM3",
            native_operation="concept segmentation from visual exemplars (positive/negative boxes)",
            role=None,
            adapter=None,
            status=ImplementationStatus.OUT_OF_SCOPE,
            runtime=None,
            input_evidence=("PreparedImage", "exemplar boxes"),
            output_evidence=("instance masks",),
            geometry=Geometry.MASK,
            status_reason=(
                "no milestone issue; exemplars would need a prompt-evidence contract of their own "
                "(the 'pcs' strategy is enumerated but not executable)"
            ),
        ),
        Capability(
            capability_id="sam3.video_tracking",
            backend="SAM3",
            native_operation="concept detection and tracking in video (video predictor)",
            role=None,
            adapter=None,
            status=ImplementationStatus.OUT_OF_SCOPE,
            runtime=None,
            input_evidence=("frame sequence", "text or visual prompts"),
            output_evidence=("tracked masklets",),
            geometry=Geometry.MASK,
            status_reason=(
                "cross-observation identity belongs to Entity Resolution, not to perception"
            ),
        ),
        # -------------------------------------------------------------------- Florence-2
        Capability(
            capability_id="florence2.region_proposal",
            backend="Florence-2",
            native_operation="<REGION_PROPOSAL> class-agnostic boxes",
            role=Role.REGION_DISCOVERY,
            adapter=f"{_BACKENDS}.florence2.Florence2RegionDiscovery",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_REGION_DISCOVERY,
                backend_id="florence2",
                settings=(("task", ("<REGION_PROPOSAL>",)),),
            ),
            input_evidence=("PreparedImage",),
            output_evidence=("Region2D (box only)",),
            geometry=Geometry.BOX,
            prompt_controls=_set("task"),
            model_controls=_FLORENCE_MODEL,
            provenance=_REGION_PROVENANCE,
            limitations=("box only: no mask for 2D->3D membership or AlphaCLIP", _SAM_PROVIDER),
            downstream_roles=(Role.REGION_FEATURES, Role.SEMANTIC_VIEW_ASSEMBLY),
            comparison_group="automatic_region_proposal.box",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="florence2.object_detection",
            backend="Florence-2",
            native_operation="<OD> boxes with a generated category name",
            role=Role.REGION_DISCOVERY,
            adapter=f"{_BACKENDS}.florence2.Florence2RegionDiscovery",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_REGION_DISCOVERY,
                backend_id="florence2",
                settings=(("task", ("<OD>",)),),
            ),
            input_evidence=("PreparedImage",),
            output_evidence=("Region2D (box only)", "RegionSemanticHint (category text)"),
            geometry=Geometry.BOX,
            prompt_controls=_set("task"),
            model_controls=_FLORENCE_MODEL,
            provenance=_FLORENCE_TEXT_PROVENANCE,
            limitations=(
                "labels come from the model's training vocabulary, not from a query",
                "the category text is a stage hint, never a SemanticClaim, and does not reach the "
                "PerceptionRunArtifact",
                "box only: no mask for 2D->3D membership or AlphaCLIP",
            ),
            downstream_roles=(Role.REGION_FEATURES, Role.SEMANTIC_VIEW_ASSEMBLY),
            comparison_group="generated_label_detection.box",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="florence2.dense_region_caption",
            backend="Florence-2",
            native_operation="<DENSE_REGION_CAPTION> boxes with a generated description",
            role=Role.REGION_DISCOVERY,
            adapter=f"{_BACKENDS}.florence2.Florence2RegionDiscovery",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_REGION_DISCOVERY,
                backend_id="florence2",
                settings=(("task", ("<DENSE_REGION_CAPTION>",)),),
            ),
            input_evidence=("PreparedImage",),
            output_evidence=("Region2D (box only)", "RegionSemanticHint (description text)"),
            geometry=Geometry.BOX,
            prompt_controls=_set("task"),
            model_controls=_FLORENCE_MODEL,
            provenance=_FLORENCE_TEXT_PROVENANCE,
            limitations=(
                "the description is a stage hint, never a SemanticClaim",
                "box only: no mask for 2D->3D membership or AlphaCLIP",
            ),
            downstream_roles=(Role.REGION_FEATURES, Role.SEMANTIC_VIEW_ASSEMBLY),
            comparison_group="dense_region_caption.box",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="florence2.open_vocabulary_detection",
            backend="Florence-2",
            native_operation="<OPEN_VOCABULARY_DETECTION> boxes or polygons for a text input",
            role=Role.REGION_DISCOVERY,
            adapter=f"{_BACKENDS}.florence2.Florence2RegionDiscovery",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_REGION_DISCOVERY,
                backend_id="florence2",
                settings=(("task", ("<OPEN_VOCABULARY_DETECTION>",)),),
            ),
            input_evidence=("PreparedImage", "text input (configuration)"),
            output_evidence=("Region2D (box only or rasterized polygon mask)",),
            geometry=Geometry.BOX_OR_MASK,
            prompt_controls=_set("task", "prompt"),
            model_controls=_FLORENCE_MODEL,
            provenance=_FLORENCE_TEXT_PROVENANCE,
            limitations=(
                "the official parser returns boxes or polygons per response: the geometry type is "
                "not controlled by the experiment",
                "one text input per run, fixed in configuration",
            ),
            downstream_roles=_REGION_CONSUMERS,
            comparison_group="open_vocabulary_detection.box_or_mask",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="florence2.caption_phrase_grounding",
            backend="Florence-2",
            native_operation="<CAPTION_TO_PHRASE_GROUNDING> boxes for the phrases of a caption",
            role=Role.REGION_DISCOVERY,
            adapter=f"{_BACKENDS}.florence2.Florence2RegionDiscovery",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_REGION_DISCOVERY,
                backend_id="florence2",
                settings=(("task", ("<CAPTION_TO_PHRASE_GROUNDING>",)),),
            ),
            input_evidence=("PreparedImage", "caption (configuration)"),
            output_evidence=("Region2D (box only)", "RegionSemanticHint (phrase)"),
            geometry=Geometry.BOX,
            prompt_controls=_set("task", "prompt"),
            model_controls=_FLORENCE_MODEL,
            provenance=_FLORENCE_TEXT_PROVENANCE,
            limitations=(
                "decomposes a caption into phrases: not the multi-instance referring task of "
                "LocateAnything phrase grounding",
                "box only: no mask for 2D->3D membership or AlphaCLIP",
            ),
            downstream_roles=(Role.REGION_FEATURES, Role.SEMANTIC_VIEW_ASSEMBLY),
            comparison_group="caption_phrase_grounding.box",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="florence2.referring_expression_segmentation",
            backend="Florence-2",
            native_operation="<REFERRING_EXPRESSION_SEGMENTATION> polygon mask of a text referent",
            role=Role.REGION_DISCOVERY,
            adapter=f"{_BACKENDS}.florence2.Florence2RegionDiscovery",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_REGION_DISCOVERY,
                backend_id="florence2",
                settings=(("task", ("<REFERRING_EXPRESSION_SEGMENTATION>",)),),
            ),
            input_evidence=("PreparedImage", "referring expression (configuration)"),
            output_evidence=("Region2D (rasterized polygon mask)",),
            geometry=Geometry.MASK,
            prompt_controls=_set("task", "prompt"),
            model_controls=_FLORENCE_MODEL,
            provenance=_REGION_PROVENANCE,
            limitations=(
                "polygons are rasterized by pixel centre",
                "one expression per run, fixed in configuration",
            ),
            downstream_roles=_REGION_CONSUMERS,
            comparison_group="referring_expression_segmentation.mask",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="florence2.region_to_segmentation",
            backend="Florence-2",
            native_operation="<REGION_TO_SEGMENTATION> polygon mask of an input box",
            role=Role.REGION_DISCOVERY,
            adapter=f"{_BACKENDS}.florence2.Florence2RegionDiscovery",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_REGION_DISCOVERY,
                backend_id="florence2",
                settings=(("task", ("<REGION_TO_SEGMENTATION>",)),),
            ),
            input_evidence=("PreparedImage", "region box as <loc_*> tokens (configuration)"),
            output_evidence=("Region2D (rasterized polygon mask)",),
            geometry=Geometry.MASK,
            prompt_controls=_set("task", "prompt"),
            model_controls=_FLORENCE_MODEL,
            provenance=_REGION_PROVENANCE,
            limitations=(
                "the input box is a static configuration value, so every frame segments the same "
                "box; as a per-proposal refiner (the #568 role) it is not wired and not planned",
            ),
            downstream_roles=_REGION_CONSUMERS,
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="florence2.scene_caption",
            backend="Florence-2",
            native_operation="<CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION> scene text",
            role=Role.SEMANTIC_INTERPRETATION,
            adapter=f"{_BACKENDS}.florence2_semantic.Florence2SemanticInterpreter",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_SEMANTIC,
                backend_id="florence2",
                settings=(
                    ("task", ("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>")),
                ),
            ),
            input_evidence=("SemanticInterpretationRequest (SCENE, one FULL_FRAME view)",),
            output_evidence=("SemanticClaim (the caption as one primary hypothesis)",),
            geometry=Geometry.NONE,
            prompt_controls=_set("task"),
            model_controls=_set(
                "checkpoint", "revision", "precision", "max_new_tokens", "temperature"
            ),
            provenance=_INTERPRETATION_PROVENANCE,
            limitations=(
                "task-native prompt (florence2-task-prompt/1:<task>): cannot share a prompt policy "
                "with instruction-following VLMs",
                "no alternatives and no structured scene fields",
            ),
            upstream_roles=(Role.SEMANTIC_VIEW_ASSEMBLY,),
            downstream_roles=(Role.SEMANTIC_FUSION,),
            comparison_group="scene_caption.task_native",
            metric_family=EvaluationStage.SEMANTIC_INTERPRETATION,
        ),
        Capability(
            capability_id="florence2.region_category",
            backend="Florence-2",
            native_operation="<REGION_TO_CATEGORY> category of a region",
            role=Role.SEMANTIC_INTERPRETATION,
            adapter=f"{_BACKENDS}.florence2_semantic.Florence2SemanticInterpreter",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_SEMANTIC,
                backend_id="florence2",
                settings=(("task", ("<REGION_TO_CATEGORY>",)),),
            ),
            input_evidence=(
                "SemanticInterpretationRequest (REGION, one TIGHT_CROP or MASKED_SUBJECT view)",
            ),
            output_evidence=("SemanticClaim (one primary hypothesis)",),
            geometry=Geometry.NONE,
            prompt_controls=_set("task"),
            model_controls=_set(
                "checkpoint", "revision", "precision", "max_new_tokens", "temperature"
            ),
            provenance=_INTERPRETATION_PROVENANCE,
            limitations=(
                "accepts exactly one region-filling view; no multi-view or scene context",
                "task-native prompt: not comparable with instruction-following VLMs",
            ),
            upstream_roles=(Role.SEMANTIC_VIEW_ASSEMBLY,),
            downstream_roles=(Role.SEMANTIC_FUSION,),
            comparison_group="region_category.task_native",
            metric_family=EvaluationStage.SEMANTIC_INTERPRETATION,
        ),
        Capability(
            capability_id="florence2.region_description",
            backend="Florence-2",
            native_operation="<REGION_TO_DESCRIPTION> description of a region",
            role=Role.SEMANTIC_INTERPRETATION,
            adapter=f"{_BACKENDS}.florence2_semantic.Florence2SemanticInterpreter",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_SEMANTIC,
                backend_id="florence2",
                settings=(("task", ("<REGION_TO_DESCRIPTION>",)),),
            ),
            input_evidence=(
                "SemanticInterpretationRequest (REGION, one TIGHT_CROP or MASKED_SUBJECT view)",
            ),
            output_evidence=("SemanticClaim (the description as one primary hypothesis)",),
            geometry=Geometry.NONE,
            prompt_controls=_set("task"),
            model_controls=_set(
                "checkpoint", "revision", "precision", "max_new_tokens", "temperature"
            ),
            provenance=_INTERPRETATION_PROVENANCE,
            limitations=(
                "a sentence, not a concept: exact-match concept metrics rarely apply",
                "task-native prompt: not comparable with instruction-following VLMs",
            ),
            upstream_roles=(Role.SEMANTIC_VIEW_ASSEMBLY,),
            downstream_roles=(Role.SEMANTIC_FUSION,),
            comparison_group="region_description.task_native",
            metric_family=EvaluationStage.SEMANTIC_INTERPRETATION,
        ),
        Capability(
            capability_id="florence2.ocr",
            backend="Florence-2",
            native_operation="<OCR>, <OCR_WITH_REGION>, <REGION_TO_OCR> text reading",
            role=None,
            adapter=None,
            status=ImplementationStatus.OUT_OF_SCOPE,
            runtime=None,
            input_evidence=("image or region",),
            output_evidence=("text, quad boxes",),
            geometry=Geometry.BOX,
            status_reason="scene-text reading is not part of the Solution 1 map evidence",
        ),
        # ------------------------------------------------------------ dense visual features
        Capability(
            capability_id="dinov2.dense_patch_features",
            backend="DINOv2",
            native_operation="self-supervised patch tokens (CLS and registers removed)",
            role=Role.DENSE_FEATURES,
            adapter=f"{_BACKENDS}.dinov2.DinoV2DenseFeatureBackend",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(stage_id=_VP, component_id=_DENSE, backend_id="dinov2"),
            input_evidence=("PreparedImage",),
            output_evidence=("VisualFeature (dense)", "DenseFeatureMap (native grid)"),
            geometry=Geometry.DENSE_GRID,
            model_controls=_set(
                "checkpoint", "revision", "precision", "input_width", "input_height", "l2_normalize"
            ),
            provenance=_DENSE_PROVENANCE,
            limitations=(
                "patch 14 grid at native resolution; an input that is not a multiple of the patch "
                "leaves an uncovered border",
                "the embedding space is compatible with no other backend or checkpoint",
            ),
            downstream_roles=(Role.DENSE_FEATURE_SAMPLING,),
            comparison_group="dense_visual_features",
            metric_family=EvaluationStage.FEATURE_EXTRACTION,
        ),
        Capability(
            capability_id="dinov3.dense_patch_features",
            backend="DINOv3",
            native_operation="self-supervised patch tokens (CLS and 4 registers removed)",
            role=Role.DENSE_FEATURES,
            adapter=f"{_BACKENDS}.dinov3.DinoV3DenseFeatureBackend",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(stage_id=_VP, component_id=_DENSE, backend_id="dinov3"),
            input_evidence=("PreparedImage",),
            output_evidence=("VisualFeature (dense)", "DenseFeatureMap (native grid)"),
            geometry=Geometry.DENSE_GRID,
            model_controls=_set(
                "checkpoint", "revision", "precision", "input_width", "input_height", "l2_normalize"
            ),
            provenance=_DENSE_PROVENANCE,
            limitations=(
                "gated checkpoint (DINOv3 License)",
                "direct resize may distort the aspect ratio; the experiment fixes the input size",
            ),
            downstream_roles=(Role.DENSE_FEATURE_SAMPLING,),
            comparison_group="dense_visual_features",
            metric_family=EvaluationStage.FEATURE_EXTRACTION,
        ),
        Capability(
            capability_id="siglip2.dense_patch_features",
            backend="SigLIP2",
            native_operation="vision-encoder patch tokens (last_hidden_state, FixRes checkpoints)",
            role=Role.DENSE_FEATURES,
            adapter=f"{_BACKENDS}.siglip2.Siglip2FeatureBackend",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(stage_id=_VP, component_id=_DENSE, backend_id="siglip2"),
            input_evidence=("PreparedImage",),
            output_evidence=("VisualFeature (dense)", "DenseFeatureMap (native grid)"),
            geometry=Geometry.DENSE_GRID,
            model_controls=_set(
                "checkpoint", "revision", "precision", "input_size", "resize_filter", "l2_normalize"
            ),
            provenance=(
                *_DENSE_PROVENANCE,
                "preprocessing identity qualifies EmbeddingSpace.layer",
            ),
            limitations=(
                "square FixRes input: non-square frames are distorted; NaFlex is refused",
                "the Eagle 2.5 dynamic tiling is not reproduced; Eagle hidden states are not used",
                "never executed on real weights here",
            ),
            downstream_roles=(Role.DENSE_FEATURE_SAMPLING,),
            comparison_group="dense_visual_features",
            metric_family=EvaluationStage.FEATURE_EXTRACTION,
        ),
        Capability(
            capability_id="siglip2.global_embedding",
            backend="SigLIP2",
            native_operation="attention-pooled image embedding (pooler_output)",
            role=None,
            adapter=None,
            status=ImplementationStatus.OUT_OF_SCOPE,
            runtime=None,
            input_evidence=("PreparedImage",),
            output_evidence=("VisualFeature (global)",),
            geometry=Geometry.NONE,
            status_reason=(
                "the adapter implements the GLOBAL scope, but the dense_features slot fixes the "
                "dense scope and no milestone issue plans a global or scoring use"
            ),
        ),
        # ------------------------------------------------------------ region features, scoring
        Capability(
            capability_id="clip.region_embedding",
            backend="CLIP",
            native_operation="image projection of a region crop",
            role=Role.REGION_FEATURES,
            adapter=f"{_BACKENDS}.clip.ClipVisualFeatureBackend",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(stage_id=_VP, component_id=_REGION_FEATURES, backend_id="clip"),
            input_evidence=("PreparedImage", "Region2D (box used as crop)"),
            output_evidence=("VisualFeature (region)",),
            geometry=Geometry.NONE,
            prompt_controls=_set("crop_policy", "context_padding_fraction"),
            model_controls=_set("checkpoint", "revision", "precision", "l2_normalize"),
            provenance=("EmbeddingSpace(clip, checkpoint@revision)", "ClipView(crop box, policy)"),
            limitations=(
                "ignores the mask: background inside the box enters the embedding",
                "the slot fixes the region scope; the global CLIP feature the scene scorer needs "
                "is not produced by the runtime",
            ),
            upstream_roles=(Role.REGION_DISCOVERY, Role.REGION_GROUNDING, Role.REGION_REFINEMENT),
            downstream_roles=(Role.SEMANTIC_SCORING,),
            comparison_group="region_crop_embedding",
            metric_family=EvaluationStage.FEATURE_EXTRACTION,
        ),
        Capability(
            capability_id="alphaclip.region_embedding",
            backend="AlphaCLIP",
            native_operation="image projection conditioned on an alpha (mask) channel",
            role=Role.REGION_FEATURES,
            adapter=f"{_BACKENDS}.alphaclip.AlphaClipRegionFeatureBackend",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP, component_id=_REGION_FEATURES, backend_id="alphaclip"
            ),
            input_evidence=("PreparedImage", "Region2D (mask)"),
            output_evidence=("VisualFeature (region)",),
            geometry=Geometry.NONE,
            prompt_controls=_set("view_policy", "context_padding_fraction"),
            model_controls=_set(
                "base_checkpoint_path", "alpha_checkpoint_path", "precision", "l2_normalize"
            ),
            provenance=(
                "EmbeddingSpace(alphaclip, base+alpha fingerprint)",
                "AlphaClipView(mask hash)",
            ),
            limitations=(
                "requires a mask: a box-only region fails explicitly",
                "no verified checkpoint source yet; never executed on real weights (#71)",
            ),
            upstream_roles=(Role.REGION_DISCOVERY, Role.REGION_REFINEMENT),
            downstream_roles=(Role.SEMANTIC_SCORING,),
            comparison_group="mask_conditioned_region_embedding",
            metric_family=EvaluationStage.FEATURE_EXTRACTION,
        ),
        Capability(
            capability_id="clip.semantic_scoring",
            backend="CLIP",
            native_operation="text-image cosine similarity of scene claims and the global feature",
            role=Role.SEMANTIC_SCORING,
            adapter=f"{_BACKENDS}.semantic_scoring.ClipSemanticScorer",
            status=ImplementationStatus.PLANNED,
            runtime=None,
            input_evidence=("SemanticClaim (scene)", "VisualFeature (global, same CLIP space)"),
            output_evidence=("SemanticScore (cosine similarity)",),
            geometry=Geometry.NONE,
            provenance=("SemanticScore(claim, feature, embedding space, scorer provenance)",),
            limitations=_SCORING_LIMITATIONS,
            upstream_roles=(Role.SEMANTIC_INTERPRETATION, Role.REGION_FEATURES),
            downstream_roles=(Role.SEMANTIC_FUSION,),
            comparison_group="claim_support.clip_global",
            status_reason=(
                "the scorer runs only in visual_perception's own stage graph; no runtime "
                "component selects it and no runtime slot produces global CLIP features"
            ),
            issue=527,
        ),
        Capability(
            capability_id="alphaclip.semantic_scoring",
            backend="AlphaCLIP",
            native_operation="text-image cosine similarity of region claims and the same region",
            role=Role.SEMANTIC_SCORING,
            adapter=f"{_BACKENDS}.semantic_scoring.AlphaClipSemanticScorer",
            status=ImplementationStatus.PLANNED,
            runtime=None,
            input_evidence=(
                "SemanticClaim (region)",
                "VisualFeature (region, same AlphaCLIP space)",
            ),
            output_evidence=("SemanticScore (cosine similarity)",),
            geometry=Geometry.NONE,
            provenance=("SemanticScore(claim, feature, embedding space, scorer provenance)",),
            limitations=_SCORING_LIMITATIONS,
            upstream_roles=(Role.SEMANTIC_INTERPRETATION, Role.REGION_FEATURES),
            downstream_roles=(Role.SEMANTIC_FUSION,),
            comparison_group="claim_support.alphaclip_region",
            status_reason=(
                "the scorer runs only in visual_perception's own stage graph; no runtime "
                "component selects it"
            ),
            issue=527,
        ),
        # ------------------------------------------------------------ semantic interpretation
        Capability(
            capability_id="contextmap2.semantic_view_assembly",
            backend="ContextMap2",
            native_operation="content-addressed semantic views of the frame or of one region",
            role=Role.SEMANTIC_VIEW_ASSEMBLY,
            adapter="contextmap.runtime.executors.VisualPerceptionExecutor",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(stage_id=_VP),
            input_evidence=("PreparedImage", "Region2D"),
            output_evidence=("SemanticVisualView (sha256)",),
            geometry=Geometry.NONE,
            prompt_controls=(_planned("view_policy", 524), _planned("request_policy", 544)),
            provenance=("SemanticVisualView(kind, payload_reference, sha256, region_id)",),
            limitations=(
                "today one FULL_FRAME view per scene request and one TIGHT_CROP (region box) per "
                "region request; masked/contextual and multi-view policies arrive with #524",
            ),
            upstream_roles=(Role.REGION_DISCOVERY, Role.REGION_GROUNDING, Role.REGION_REFINEMENT),
            downstream_roles=(Role.SEMANTIC_INTERPRETATION,),
        ),
        *(
            Capability(
                capability_id=f"{family}.{mode}_interpretation",
                backend=backend,
                native_operation=f"{mode} interpretation from an instruction and ordered images",
                role=Role.SEMANTIC_INTERPRETATION,
                adapter=f"{_BACKENDS}.{adapter}",
                status=ImplementationStatus.SUPPORTED,
                runtime=RuntimeBinding(stage_id=_VP, component_id=_SEMANTIC, backend_id=family),
                input_evidence=(
                    f"SemanticInterpretationRequest ({mode.upper()}, ordered SemanticVisualView[])",
                ),
                output_evidence=(
                    ("SemanticClaim", "SceneContext") if mode == "scene" else ("SemanticClaim",)
                ),
                geometry=Geometry.NONE,
                prompt_controls=(
                    *_set("prompt_policy"),
                    _planned("view_policy", 524),
                    *((_planned("scene_context", 529),) if mode == "region" else ()),
                    _planned("request_policy", 544),
                ),
                model_controls=model_controls,
                provenance=_INTERPRETATION_PROVENANCE,
                limitations=limitations,
                upstream_roles=(Role.SEMANTIC_VIEW_ASSEMBLY,),
                downstream_roles=(Role.SEMANTIC_SCORING, Role.SEMANTIC_FUSION),
                comparison_group=f"{mode}_interpretation.instruction_following",
                metric_family=EvaluationStage.SEMANTIC_INTERPRETATION,
            )
            for family, backend, adapter, model_controls, limitations in (
                (
                    "qwen",
                    "Qwen",
                    "qwen.QwenSemanticInterpreter",
                    _set(
                        "model",
                        "revision",
                        "precision",
                        "quantization",
                        "max_new_tokens",
                        "temperature",
                        "min_pixels",
                        "max_pixels",
                    ),
                    (
                        "local; nf4/int8 quantization changes outputs and is part of the identity",
                        "self-reported confidence is never promoted (UNSCORED_ONLY)",
                        "no scene-context conditioning yet (#529)",
                    ),
                ),
                (
                    "gemini",
                    "Gemini",
                    "gemini.GeminiSemanticInterpreter",
                    _set(
                        "model",
                        "temperature",
                        "thinking_budget",
                        "structured_output",
                        "max_retries",
                        "timeout_s",
                    ),
                    (
                        "remote: frames leave the machine; needs GEMINI_API_KEY and consent",
                        "the provider model is not pinnable to a commit; validated only with a "
                        "simulated transport",
                    ),
                ),
                (
                    "eagle2_5",
                    "Eagle 2.5",
                    "eagle2_5.EagleSemanticInterpreter",
                    _set(
                        "model",
                        "revision",
                        "precision",
                        "max_new_tokens",
                        "temperature",
                        "max_dynamic_tiles",
                        "min_dynamic_tiles",
                        "use_thumbnail",
                    ),
                    (
                        "8B model (SigLIP2 encoder + Qwen2.5-7B); 448x448 dynamic tiles bound "
                        "the visual tokens per view",
                        "single observation only: video/multi-observation requests are out of "
                        "scope (#571); never executed on real weights here",
                    ),
                ),
            )
            for mode in ("scene", "region")
        ),
        Capability(
            capability_id="eagle2_5.long_context_video",
            backend="Eagle 2.5",
            native_operation="long-context video and multi-image reasoning (up to 512 frames)",
            role=None,
            adapter=None,
            status=ImplementationStatus.OUT_OF_SCOPE,
            runtime=None,
            input_evidence=("frame sequence or several observations",),
            output_evidence=("text",),
            geometry=Geometry.NONE,
            status_reason=(
                "multi-observation semantic requests are deferred to #571, outside the "
                "single-observation matrix"
            ),
        ),
        # -------------------------------------------------------------------- LocateAnything
        Capability(
            capability_id="locateanything.category_detection",
            backend="LocateAnything",
            native_operation="detect every instance of an ordered category set (PBD boxes)",
            role=Role.REGION_GROUNDING,
            adapter=f"{_BACKENDS}.locateanything.LocateAnythingRegionGrounding",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_GROUNDING,
                backend_id="locateanything",
                settings=(("policy_id", ("locateanything.category-detection/1",)),),
            ),
            input_evidence=("PreparedImage", "GroundingQuery (CATEGORY_DETECTION, BOX)"),
            output_evidence=("Region2D (box only)", "GroundingOutput label (never a claim)"),
            geometry=Geometry.BOX,
            prompt_controls=_set("query_set"),
            model_controls=_LA_MODEL,
            provenance=_GROUNDING_PROVENANCE,
            limitations=_LA_LIMITATIONS,
            downstream_roles=(
                Role.REGION_REFINEMENT,
                Role.REGION_FEATURES,
                Role.SEMANTIC_VIEW_ASSEMBLY,
            ),
            comparison_group="category_conditioned_detection.box",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="locateanything.phrase_grounding",
            backend="LocateAnything",
            native_operation="ground every instance matching a free phrase (PBD boxes)",
            role=Role.REGION_GROUNDING,
            adapter=f"{_BACKENDS}.locateanything.LocateAnythingRegionGrounding",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_GROUNDING,
                backend_id="locateanything",
                settings=(("policy_id", ("locateanything.phrase-grounding/1",)),),
            ),
            input_evidence=("PreparedImage", "GroundingQuery (PHRASE_GROUNDING, BOX)"),
            output_evidence=("Region2D (box only)", "GroundingOutput label (never a claim)"),
            geometry=Geometry.BOX,
            prompt_controls=_set("query_set"),
            model_controls=_LA_MODEL,
            provenance=_GROUNDING_PROVENANCE,
            limitations=_LA_LIMITATIONS,
            downstream_roles=(
                Role.REGION_REFINEMENT,
                Role.REGION_FEATURES,
                Role.SEMANTIC_VIEW_ASSEMBLY,
            ),
            comparison_group="phrase_grounding.box",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="locateanything.pointing",
            backend="LocateAnything",
            native_operation="point to what a phrase refers to",
            role=Role.REGION_GROUNDING,
            adapter=f"{_BACKENDS}.locateanything.LocateAnythingRegionGrounding",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(
                stage_id=_VP,
                component_id=_GROUNDING,
                backend_id="locateanything",
                settings=(("policy_id", ("locateanything.pointing/1",)),),
            ),
            input_evidence=("PreparedImage", "GroundingQuery (PHRASE_GROUNDING, POINT)"),
            output_evidence=("GroundingPoint (grounding stream only)",),
            geometry=Geometry.POINT,
            prompt_controls=_set("query_set"),
            model_controls=_LA_MODEL,
            provenance=_GROUNDING_PROVENANCE,
            limitations=(
                "a point never becomes a box: there is no canonical point contract, so points "
                "reach no downstream stage",
                *_LA_LIMITATIONS[1:],
            ),
            comparison_group="phrase_pointing.point",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Capability(
            capability_id="locateanything.visual_prompt_grounding",
            backend="LocateAnything",
            native_operation="detection with an image crop as the query (visual prompt)",
            role=Role.REGION_GROUNDING,
            adapter=None,
            status=ImplementationStatus.BLOCKED,
            runtime=None,
            input_evidence=("PreparedImage", "visual prompt crop"),
            output_evidence=("Region2D (box only)",),
            geometry=Geometry.BOX,
            downstream_roles=(Role.REGION_REFINEMENT,),
            status_reason=(
                "the released nvidia/LocateAnything-3B weights do not support visual-prompt "
                "inference; upstream ships only a LoRA fine-tuning recipe"
            ),
            issue=574,
        ),
        Capability(
            capability_id="locateanything.text_gui_layout",
            backend="LocateAnything",
            native_operation="scene-text detection, GUI element grounding, layout grounding",
            role=None,
            adapter=None,
            status=ImplementationStatus.OUT_OF_SCOPE,
            runtime=None,
            input_evidence=("image", "optional query"),
            output_evidence=("boxes or points",),
            geometry=Geometry.BOX,
            status_reason="documents and GUIs are not robotic map evidence",
        ),
        # ------------------------------------------------------------------ 3D proposals
        Capability(
            capability_id="locateanything3d.open_vocabulary_3d_proposal",
            backend="LocateAnything3D",
            native_operation="open-vocabulary 3D box proposals from images",
            role=Role.PROPOSAL_3D,
            adapter=None,
            status=ImplementationStatus.BLOCKED,
            runtime=None,
            input_evidence=("image", "text query"),
            output_evidence=("model-predicted 3D boxes",),
            geometry=Geometry.BOX_3D,
            limitations=(
                "model-predicted geometry never replaces LiDAR/depth: every proposal needs metric "
                "verification before it can support a spatial observation",
            ),
            downstream_roles=(Role.METRIC_VERIFICATION,),
            status_reason=(
                "the public NVlabs/LocateAnything3D repository holds only a README title and no "
                "NVIDIA weights are published"
            ),
            issue=575,
        ),
        Capability(
            capability_id="contextmap2.metric_3d_verification",
            backend="ContextMap2",
            native_operation="verify a model-predicted 3D proposal against LiDAR/depth geometry",
            role=Role.METRIC_VERIFICATION,
            adapter=None,
            status=ImplementationStatus.BLOCKED,
            runtime=None,
            input_evidence=("3D proposal", "GeometricMap", "calibration and pose"),
            output_evidence=("SpatialObservation",),
            geometry=Geometry.POINT_SET_3D,
            upstream_roles=(Role.PROPOSAL_3D,),
            downstream_roles=(Role.SEMANTIC_FUSION,),
            status_reason="its design waits for a reproducible LocateAnything3D release",
            issue=575,
        ),
        # ------------------------------------------------------- ContextMap2 downstream stages
        Capability(
            capability_id="contextmap2.mask_membership_association",
            backend="ContextMap2",
            native_operation="assign visible projected map points to the frozen region masks",
            role=Role.SENSOR_ASSOCIATION,
            adapter="contextmap.sensor_association.membership.associate_regions",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(stage_id="sensor_association"),
            input_evidence=(
                "Region2D (mask or mask_reference)",
                "GeometricMap",
                "pose",
                "calibration",
            ),
            output_evidence=("SpatialObservation",),
            geometry=Geometry.POINT_SET_3D,
            provenance=("SpatialObservation provenance (perception run, region, pose, policies)",),
            limitations=(
                "membership is defined by the mask: a box-only region is reported as "
                "NO_INLINE_MASK and supports no point",
            ),
            upstream_roles=(Role.REGION_DISCOVERY, Role.REGION_REFINEMENT),
            downstream_roles=(Role.SEMANTIC_FUSION,),
            metric_family=EvaluationStage.SENSOR_ASSOCIATION,
        ),
        Capability(
            capability_id="contextmap2.dense_feature_sampling",
            backend="ContextMap2",
            native_operation="map each visible point to the cells of a dense feature map",
            role=Role.DENSE_FEATURE_SAMPLING,
            adapter="contextmap.sensor_association.dense_sampling.sample_dense_features",
            status=ImplementationStatus.PLANNED,
            runtime=None,
            input_evidence=("DenseFeatureMap", "visibility of the projected points"),
            output_evidence=("DenseFeatureSamples",),
            geometry=Geometry.POINT_SET_3D,
            provenance=("sampling policy id", "interpolation policy"),
            upstream_roles=(Role.DENSE_FEATURES,),
            downstream_roles=(Role.SEMANTIC_FUSION,),
            metric_family=EvaluationStage.SENSOR_ASSOCIATION,
            status_reason=(
                "the function works on DINO and SigLIP2 grids alike, but SensorAssociationExecutor "
                "passes no dense channel; no implementation issue owns that wiring yet"
            ),
            issue=577,
        ),
        Capability(
            capability_id="contextmap2.semantic_fusion",
            backend="ContextMap2",
            native_operation="accumulate evidence over supports grouped by physical observation",
            role=Role.SEMANTIC_FUSION,
            adapter="contextmap.semantic_fusion.build_fusion_supports",
            status=ImplementationStatus.SUPPORTED,
            runtime=RuntimeBinding(stage_id="semantic_fusion"),
            input_evidence=("SpatialObservation", "SemanticClaim", "SemanticScore"),
            output_evidence=("FusionSupport", "FusedEvidence"),
            geometry=Geometry.POINT_SET_3D,
            provenance=("support and accumulation policy ids", "physical observation grouping"),
            limitations=(
                "repeated inference over one physical observation stays correlated, never new "
                "independent evidence",
            ),
            upstream_roles=(
                Role.SENSOR_ASSOCIATION,
                Role.DENSE_FEATURE_SAMPLING,
                Role.SEMANTIC_INTERPRETATION,
                Role.SEMANTIC_SCORING,
                Role.METRIC_VERIFICATION,
            ),
            metric_family=EvaluationStage.SEMANTIC_FUSION,
        ),
    ),
    compositions=(
        # ------------------------------------------------ grounding -> refinement -> association
        Composition(
            producer="locateanything.category_detection",
            consumer="sam2.box_prompt_refinement",
            status=CompositionStatus.PLANNED,
            note="grounded boxes become SAM2 box prompts with contributor lineage",
            issue=568,
            comparison_group="category_conditioned_instance_masks",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Composition(
            producer="locateanything.phrase_grounding",
            consumer="sam2.box_prompt_refinement",
            status=CompositionStatus.PLANNED,
            note="phrase-grounded boxes become SAM2 box prompts with contributor lineage",
            issue=568,
            comparison_group="phrase_conditioned_instance_masks",
            metric_family=EvaluationStage.REGION_DISCOVERY,
        ),
        Composition(
            producer="locateanything.pointing",
            consumer="sam2.box_prompt_refinement",
            status=CompositionStatus.INCOMPATIBLE,
            note="a point has no canonical contract and stays in the grounding stream",
        ),
        Composition(
            producer="sam2.box_prompt_refinement",
            consumer="contextmap2.mask_membership_association",
            status=CompositionStatus.PLANNED,
            note="refined masks are ordinary mask-backed Region2D for membership",
            issue=568,
            metric_family=EvaluationStage.SENSOR_ASSOCIATION,
        ),
        Composition(
            producer="locateanything.category_detection",
            consumer="contextmap2.mask_membership_association",
            status=CompositionStatus.INCOMPATIBLE,
            note=(
                "a grounded box has no mask: membership reports NO_INLINE_MASK and the region "
                "supports no point; box-only association (#528 P1) needs a box-membership policy "
                "no milestone issue implements"
            ),
        ),
        Composition(
            producer="florence2.object_detection",
            consumer="contextmap2.mask_membership_association",
            status=CompositionStatus.INCOMPATIBLE,
            note="a box-only region has no mask: membership reports NO_INLINE_MASK",
        ),
        Composition(
            producer="florence2.open_vocabulary_detection",
            consumer="contextmap2.mask_membership_association",
            status=CompositionStatus.NOT_COMPARABLE,
            note=(
                "it runs, but the parser returns boxes (NO_INLINE_MASK) or polygons per response, "
                "so the association result mixes geometry types the experiment does not control"
            ),
            metric_family=EvaluationStage.SENSOR_ASSOCIATION,
        ),
        *(
            Composition(
                producer=producer,
                consumer="contextmap2.mask_membership_association",
                status=CompositionStatus.SUPPORTED,
                note="the mask (inline or via the run's mask loader) defines membership",
                metric_family=EvaluationStage.SENSOR_ASSOCIATION,
            )
            for producer in (
                "sam2.automatic_mask_generation",
                "sam3.text_concept_segmentation",
                "florence2.referring_expression_segmentation",
            )
        ),
        # -------------------------------------------------------------------- dense features
        *(
            Composition(
                producer=producer,
                consumer="contextmap2.dense_feature_sampling",
                status=CompositionStatus.PLANNED,
                note=(
                    "the backend-neutral DenseFeatureMap is sampled identically for every grid, "
                    "but the runtime association executor passes no dense channel yet"
                ),
                issue=577,
                metric_family=EvaluationStage.FEATURE_EXTRACTION,
            )
            for producer in (
                "dinov2.dense_patch_features",
                "dinov3.dense_patch_features",
                "siglip2.dense_patch_features",
            )
        ),
        # ------------------------------------------------------------------ region features
        *(
            Composition(
                producer=producer,
                consumer="alphaclip.region_embedding",
                status=CompositionStatus.SUPPORTED,
                note="AlphaCLIP reads the frozen mask of the same region as its alpha channel",
                metric_family=EvaluationStage.FEATURE_EXTRACTION,
            )
            for producer in ("sam2.automatic_mask_generation", "sam3.text_concept_segmentation")
        ),
        Composition(
            producer="florence2.object_detection",
            consumer="alphaclip.region_embedding",
            status=CompositionStatus.INCOMPATIBLE,
            note="AlphaCLIP needs the region mask; a box-only region fails explicitly",
        ),
        Composition(
            producer="locateanything.category_detection",
            consumer="alphaclip.region_embedding",
            status=CompositionStatus.INCOMPATIBLE,
            note="AlphaCLIP needs the region mask; a grounded box has none",
        ),
        *(
            Composition(
                producer=producer,
                consumer="clip.region_embedding",
                status=CompositionStatus.SUPPORTED,
                note="CLIP embeds the crop of the region box (the mask is ignored)",
                metric_family=EvaluationStage.FEATURE_EXTRACTION,
            )
            for producer in ("sam2.automatic_mask_generation", "florence2.object_detection")
        ),
        Composition(
            producer="locateanything.category_detection",
            consumer="clip.region_embedding",
            status=CompositionStatus.PLANNED,
            note=(
                "contract-compatible, but grounded regions are appended after the perception "
                "stage graph and reach no feature extractor; no implementation issue owns it yet"
            ),
            issue=577,
            metric_family=EvaluationStage.FEATURE_EXTRACTION,
        ),
        # ------------------------------------------------------- views -> interpretation
        *(
            Composition(
                producer=producer,
                consumer="contextmap2.semantic_view_assembly",
                status=CompositionStatus.SUPPORTED,
                note="each accepted region gets one TIGHT_CROP view of its box",
            )
            for producer in ("sam2.automatic_mask_generation", "florence2.object_detection")
        ),
        Composition(
            producer="locateanything.category_detection",
            consumer="contextmap2.semantic_view_assembly",
            status=CompositionStatus.PLANNED,
            note=(
                "grounded regions are appended after region interpretation ran; no "
                "implementation issue owns the reordering yet"
            ),
            issue=577,
        ),
        *(
            Composition(
                producer="contextmap2.semantic_view_assembly",
                consumer=consumer,
                status=CompositionStatus.SUPPORTED,
                note=note,
                metric_family=EvaluationStage.SEMANTIC_INTERPRETATION,
            )
            for consumer, note in (
                ("qwen.scene_interpretation", "one FULL_FRAME view per frame"),
                ("qwen.region_interpretation", "one TIGHT_CROP view per region"),
                ("gemini.scene_interpretation", "one FULL_FRAME view, sent to the provider"),
                ("gemini.region_interpretation", "one TIGHT_CROP view, sent to the provider"),
                ("eagle2_5.scene_interpretation", "one FULL_FRAME view, tiled by the processor"),
                ("eagle2_5.region_interpretation", "one TIGHT_CROP view, tiled by the processor"),
                ("florence2.scene_caption", "one FULL_FRAME view"),
                ("florence2.region_category", "one TIGHT_CROP view, the only kind it accepts"),
            )
        ),
        # ---------------------------------------------------------------- semantic scoring
        Composition(
            producer="qwen.region_interpretation",
            consumer="alphaclip.semantic_scoring",
            status=CompositionStatus.PLANNED,
            note="region claims scored against the AlphaCLIP feature of the same region",
            issue=527,
        ),
        Composition(
            producer="alphaclip.region_embedding",
            consumer="alphaclip.semantic_scoring",
            status=CompositionStatus.PLANNED,
            note="the region feature must be in the scorer's AlphaCLIP space",
            issue=527,
        ),
        Composition(
            producer="qwen.scene_interpretation",
            consumer="clip.semantic_scoring",
            status=CompositionStatus.PLANNED,
            note="scene claims need a global CLIP feature, which no runtime slot produces yet",
            issue=527,
        ),
        Composition(
            producer="clip.region_embedding",
            consumer="clip.semantic_scoring",
            status=CompositionStatus.INCOMPATIBLE,
            note="ClipSemanticScorer scores only GLOBAL features; region CLIP features are skipped",
        ),
        # ------------------------------------------------------------------- fusion, 3D
        Composition(
            producer="contextmap2.mask_membership_association",
            consumer="contextmap2.semantic_fusion",
            status=CompositionStatus.SUPPORTED,
            note="spatial observations form fusion supports grouped by physical observation",
            metric_family=EvaluationStage.SEMANTIC_FUSION,
        ),
        Composition(
            producer="locateanything3d.open_vocabulary_3d_proposal",
            consumer="contextmap2.metric_3d_verification",
            status=CompositionStatus.BLOCKED,
            note="a proposal is a hypothesis until LiDAR/depth geometry verifies it",
            issue=575,
        ),
        Composition(
            producer="contextmap2.metric_3d_verification",
            consumer="contextmap2.semantic_fusion",
            status=CompositionStatus.BLOCKED,
            note="only a verified proposal may become a spatial observation",
            issue=575,
        ),
    ),
)
"""The capability and compatibility matrix of milestone #22."""

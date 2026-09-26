"""SAM3 adapter for the canonical Region Discovery capability."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from importlib import import_module
from math import isfinite
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

from ..discovery import (
    AuditedRegions,
    BackendDiagnostics,
    DiscoveryInput,
    DiscoveryOutput,
    DiscoveryPassConfig,
    discover_canonical_regions,
    validate_materialized_discovery_image,
)
from ..models import BackendProvenance, PreparedImage, Region2D
from ..normalization import NormalizationConfig
from ..region_models import (
    BackendScore,
    BoundingBox,
    InlineMask,
    JsonScalar,
    RegionCandidate,
    RegionProvenance,
    mask_bounding_box,
)
from ._model_placement import verify_model_placement

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SUPPORTED_PRECISIONS = frozenset({"float32", "float16", "bfloat16"})


class Sam3Strategy(StrEnum):
    """Explicit SAM3 discovery/query strategies supported by the adapter."""

    AUTOMATIC = "automatic"
    TEXT_PROMPT = "text_prompt"
    POINT_GRID = "point_grid"
    TRACKER = "tracker"
    PCS = "pcs"


@dataclass(frozen=True, slots=True)
class Sam3Config:
    """Effective configuration for one SAM3 Region Discovery adapter.

    ``precision`` is the inference precision the official runtime applies: ``float32``
    runs the SDK as loaded, while ``float16`` and ``bfloat16`` run it under
    ``torch.autocast``. The official SAM3 image model requires ``bfloat16``.
    ``model_version`` has no default: it enters provenance and the digest, so a run never
    records a placeholder instead of the checkpoint version that produced it.
    """

    checkpoint: str
    model_version: str
    device: str = "cpu"
    precision: str = "float32"
    strategy: Sam3Strategy = Sam3Strategy.AUTOMATIC
    score_threshold: float = 0.0
    mask_threshold: float = 0.5
    prompt: str | None = None
    strategy_settings: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate strategy-specific configuration without hidden defaults."""
        if not self.checkpoint or not self.model_version or not self.device or not self.precision:
            raise ValueError("SAM3 checkpoint, version, device, and precision must not be empty")
        if self.precision not in _SUPPORTED_PRECISIONS:
            supported = ", ".join(sorted(_SUPPORTED_PRECISIONS))
            raise ValueError(f"SAM3 precision must be one of: {supported}")
        _validate_unit_threshold(self.score_threshold, "score_threshold")
        _validate_unit_threshold(self.mask_threshold, "mask_threshold")
        if self.strategy is Sam3Strategy.TEXT_PROMPT and not self.prompt:
            raise ValueError("SAM3 text_prompt strategy requires a prompt")
        if self.strategy is Sam3Strategy.AUTOMATIC and self.prompt is not None:
            raise ValueError("SAM3 automatic strategy does not consume a text prompt")
        keys = [key for key, _ in self.strategy_settings]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("SAM3 strategy setting names must be non-empty and unique")

    @property
    def digest(self) -> str:
        """Return a deterministic digest of the effective SAM3 configuration."""
        payload = {
            "checkpoint": self.checkpoint,
            "model_version": self.model_version,
            "device": self.device,
            "precision": self.precision,
            "strategy": self.strategy.value,
            "score_threshold": self.score_threshold,
            "mask_threshold": self.mask_threshold,
            "prompt": self.prompt,
            "strategy_settings": list(self.strategy_settings),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(serialized.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Sam3NativeProposal:
    """SDK-isolated SAM3 proposal: Python scalars and the mask as an immutable ``InlineMask``."""

    proposal_id: str
    box: tuple[float, float, float, float]
    mask: InlineMask
    score_name: str
    score: float
    query_id: str
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate native scalar output before canonical normalization."""
        if not self.proposal_id or not self.score_name or not self.query_id:
            raise ValueError("SAM3 proposal, score, and query identities must not be empty")
        if not isfinite(self.score):
            raise ValueError("SAM3 native score must be finite")


@dataclass(frozen=True, slots=True)
class Sam3NativeOutput:
    """Structured scalar output returned by the internal SAM3 runtime seam."""

    proposals: tuple[Sam3NativeProposal, ...]
    warnings: tuple[str, ...] = ()
    metadata: tuple[tuple[str, JsonScalar], ...] = ()


class Sam3Runtime(Protocol):
    """Internal seam around a loaded SAM3 model/runtime."""

    def predict(self, discovery_input: DiscoveryInput, config: Sam3Config) -> Sam3NativeOutput:
        """Run the configured SAM3 strategy and return scalar-only proposals."""
        ...


class _Sam3ImageProcessor(Protocol):
    """Minimum official SAM3 image processor surface used by the runtime."""

    @property
    def model(self) -> object:
        """Loaded SAM3 model the processor runs, whose placement the runtime verifies."""
        ...

    def set_image(self, image: object) -> object:
        """Encode one image and return its inference state."""
        ...

    def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
        """Apply the configured proposal threshold to the processor state."""
        ...

    def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
        """Run text-conditioned image inference."""
        ...


class Sam3ImageProcessorRuntime:
    """Execute the official SAM3 image processor text-prompt API."""

    def __init__(
        self,
        *,
        processor: _Sam3ImageProcessor,
        image_loader: Callable[[DiscoveryInput], object],
        autocast: Callable[[Sam3Config], AbstractContextManager[object]] | None = None,
    ) -> None:
        """Bind a loaded processor to a pass-aware image loader.

        Args:
            processor: Official image processor already built around the model.
            image_loader: Materializes the exact image of one discovery pass.
            autocast: Builds the precision context the SDK runs in for one
                configuration. Defaults to ``torch.autocast`` for ``float16`` and
                ``bfloat16`` and to no context for ``float32``; tests inject a
                recording context. The SDK calls always run inside
                ``torch.inference_mode()`` as well, whatever the precision.
        """
        self._processor = processor
        self._image_loader = image_loader
        self._autocast = autocast or _torch_autocast

    def predict(self, discovery_input: DiscoveryInput, config: Sam3Config) -> Sam3NativeOutput:
        """Run supported official image inference without a strategy fallback.

        Raises:
            ValueError: If the strategy is not ``text_prompt``, if the prompt is missing,
                or if the processor's model is not on ``config.device`` or holds weights
                the configured precision does not accept: ``float32`` weights, or weights
                already in the configured autocast dtype.
        """
        if config.strategy is not Sam3Strategy.TEXT_PROMPT:
            raise ValueError("official SAM3 image runtime supports only text_prompt strategy")
        if config.prompt is None:
            raise ValueError("official SAM3 image runtime requires a text prompt")
        # O autocast realiza float16/bfloat16 sobre pesos float32 (o SDK oficial roda assim sob
        # bfloat16, #338) ou já no próprio dtype; pesos em outro dtype reduzido tornariam falsa
        # a precisão registrada. Em float32 não há autocast, então só pesos float32 servem.
        verify_model_placement(
            self._processor.model,
            device=config.device,
            precision=config.precision,
            backend="SAM3",
            accepted_dtypes=frozenset({"float32", config.precision}),
        )

        image = self._image_loader(discovery_input)
        validate_materialized_discovery_image(image, discovery_input)
        with _torch_inference_mode(), self._autocast(config):
            state = self._processor.set_image(image)
            state = self._processor.set_confidence_threshold(config.score_threshold, state=state)
            if state is None:
                raise ValueError("SAM3 processor returned no inference state")
            output = self._processor.set_text_prompt(state=state, prompt=config.prompt)
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        proposals = _parse_image_processor_output(
            output,
            width=width,
            height=height,
            mask_threshold=config.mask_threshold,
        )
        return Sam3NativeOutput(
            proposals=proposals,
            metadata=(("runtime_api", "Sam3Processor.set_text_prompt"),),
        )


class Sam3RegionDiscovery:
    """Adapt configured SAM3 output to canonical RegionCandidate values."""

    def __init__(
        self,
        *,
        config: Sam3Config,
        runtime: Sam3Runtime,
        pass_config: DiscoveryPassConfig | None = None,
        normalization_config: NormalizationConfig | None = None,
    ) -> None:
        """Build the adapter with explicit configuration and no fallback runtime."""
        self._config = config
        self._runtime = runtime
        self._pass_config = pass_config
        self._normalization_config = normalization_config

    def backend_provenance(self) -> BackendProvenance:
        """Return SAM3 identity in the Visual Perception Core contract."""
        return BackendProvenance(
            backend_id="sam3",
            capability="region_discovery",
            provider="facebook",
            model=self._config.checkpoint,
            version=self._config.model_version,
            configuration_fingerprint=self._config.digest,
        )

    def discover(self, image: PreparedImage) -> tuple[Region2D, ...]:
        """Discover and normalize canonical regions through the public port."""
        return self.discover_audited(image).regions

    def discover_audited(self, image: PreparedImage) -> AuditedRegions:
        """Discover canonical regions together with the audit of every decision behind them."""
        return discover_canonical_regions(
            image,
            self,
            pass_config=self._pass_config,
            normalization_config=self._normalization_config,
        )

    def discover_candidates(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        """Run exactly the selected SAM3 strategy for one discovery pass."""
        started = perf_counter()
        native_output = self._runtime.predict(discovery_input, self._config)
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        candidates = tuple(
            self._normalize(proposal, discovery_input, width, height)
            for proposal in native_output.proposals
            if proposal.score >= self._config.score_threshold
        )
        duration_ms = (perf_counter() - started) * 1000
        metadata = (
            *native_output.metadata,
            ("raw_proposal_count", len(native_output.proposals)),
            ("filtered_proposal_count", len(native_output.proposals) - len(candidates)),
            ("checkpoint", self._config.checkpoint),
            ("model_version", self._config.model_version),
            ("device", self._config.device),
            ("precision", self._config.precision),
            ("strategy", self._config.strategy.value),
            ("prompt", self._config.prompt),
            ("config_digest", self._config.digest),
        )
        return DiscoveryOutput(
            candidates=candidates,
            diagnostics=BackendDiagnostics(
                duration_ms=duration_ms,
                proposal_count=len(candidates),
                warnings=native_output.warnings,
                metadata=metadata,
            ),
        )

    def _normalize(
        self,
        proposal: Sam3NativeProposal,
        discovery_input: DiscoveryInput,
        width: int,
        height: int,
    ) -> RegionCandidate:
        """Convert one scalar SAM3 proposal without semantic promotion.

        A proposal with mask pixels uses the tight half-open box of its mask as
        candidate geometry. The native box comes from a box head independent of
        the mask head, so it may leave the image or fail to contain the mask; it is
        kept as audit metadata instead of geometry. A proposal without mask pixels
        keeps the native box clamped to the pass, when any part of it is inside, and
        is rejected explicitly by normalization.
        """
        if (proposal.mask.width, proposal.mask.height) != (width, height):
            raise ValueError("SAM3 proposal mask dimensions must match discovery pass dimensions")
        mask_box = mask_bounding_box(proposal.mask)
        bounding_box = mask_box or _clamp_box(proposal.box, width=width, height=height)
        native_metadata: tuple[tuple[str, JsonScalar], ...] = (
            *proposal.metadata,
            ("native_box_x_min", proposal.box[0]),
            ("native_box_y_min", proposal.box[1]),
            ("native_box_x_max", proposal.box[2]),
            ("native_box_y_max", proposal.box[3]),
        )
        if mask_box is not None:
            contains_mask = (
                proposal.box[0] <= mask_box.x_min
                and proposal.box[1] <= mask_box.y_min
                and proposal.box[2] >= mask_box.x_max
                and proposal.box[3] >= mask_box.y_max
            )
            native_metadata = (*native_metadata, ("native_box_contains_mask", contains_mask))
        prompt = self._config.prompt
        query_parts = [self._config.strategy.value]
        if prompt:
            query_parts.append(prompt)
        query_parts.append(proposal.query_id)
        return RegionCandidate(
            candidate_id=proposal.proposal_id,
            source_observation_id=discovery_input.prepared_image.source_observation_id,
            perception_run_id=discovery_input.perception_run_id,
            perception_result_id=discovery_input.perception_result_id,
            image_width=width,
            image_height=height,
            bounding_box=bounding_box,
            mask=proposal.mask,
            score=BackendScore(
                name=proposal.score_name,
                value=proposal.score,
                semantics=(
                    f"SAM3-native {proposal.score_name}; not calibrated across discovery backends"
                ),
            ),
            provenance=RegionProvenance(
                backend_id="sam3",
                backend_version=self._config.model_version,
                checkpoint=self._config.checkpoint,
                config_digest=self._config.digest,
                discovery_pass_id=discovery_input.discovery_pass.pass_id,
                native_proposal_id=proposal.proposal_id,
                query=":".join(query_parts),
            ),
            native_metadata=native_metadata,
        )


def _torch_inference_mode() -> AbstractContextManager[object]:
    """Return ``torch.inference_mode()``, so the SDK calls never record autograd state."""
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as error:
        raise RuntimeError("SAM3 inference requires torch inference_mode; install torch") from error
    return cast(AbstractContextManager[object], torch.inference_mode())


def _torch_autocast(config: Sam3Config) -> AbstractContextManager[object]:
    """Return the ``torch.autocast`` context that realizes the configured precision."""
    if config.precision == "float32":
        return nullcontext()
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"SAM3 {config.precision} inference requires torch autocast; install torch"
        ) from error
    return cast(
        AbstractContextManager[object],
        torch.autocast(
            device_type=config.device.split(":", 1)[0],
            dtype=getattr(torch, config.precision),
        ),
    )


def _clamp_box(
    box: tuple[float, float, float, float], *, width: int, height: int
) -> BoundingBox | None:
    """Clip a native box to the pass, or return ``None`` when nothing of it is inside."""
    x_min, y_min = max(box[0], 0.0), max(box[1], 0.0)
    x_max, y_max = min(box[2], float(width)), min(box[3], float(height))
    if x_max <= x_min or y_max <= y_min:
        return None
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _validate_unit_threshold(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"SAM3 {name} must be between zero and one")


def _parse_image_processor_output(
    output: Mapping[str, object], *, width: int, height: int, mask_threshold: float
) -> tuple[Sam3NativeProposal, ...]:
    """Detach documented boxes, masks, and scores from SAM3 tensors."""
    boxes = _native_sequence(output.get("boxes"), "boxes")
    scores = _native_sequence(output.get("scores"), "scores")
    masks = _native_array(output.get("masks_logits", output.get("masks")), "masks")
    if masks.ndim == 0:
        raise TypeError("SAM3 masks must be a sequence")
    if not (len(boxes) == len(scores) == len(masks)):
        raise ValueError("SAM3 boxes, scores, and masks must have equal proposal counts")

    proposals: list[Sam3NativeProposal] = []
    for index, (box_value, score_value, mask_value) in enumerate(
        zip(boxes, scores, masks, strict=True)
    ):
        box = _numeric_sequence(box_value, "box", length=4)
        proposals.append(
            Sam3NativeProposal(
                proposal_id=f"sam3-{index:06d}",
                box=cast(tuple[float, float, float, float], box),
                mask=_probability_mask(
                    mask_value,
                    width=width,
                    height=height,
                    threshold=mask_threshold,
                ),
                score_name="concept_score",
                score=_finite_number(score_value, "score"),
                query_id=f"text-prompt-{index:06d}",
                metadata=(("mask_threshold", mask_threshold),),
            )
        )
    return tuple(proposals)


def _probability_mask(
    value: NDArray[Any], *, width: int, height: int, threshold: float
) -> InlineMask:
    """Threshold one proposal's probabilities (or boolean mask) into a binary mask.

    Raises:
        TypeError: If the values are not numeric.
        ValueError: If the dimensions differ from the discovery pass, or a value is not finite.
    """
    import numpy as np

    pixels = value
    if pixels.ndim == 3 and pixels.shape[0] == 1 and pixels.shape[1] == height:
        pixels = pixels[0]  # eixo de canal único do SDK
    if pixels.shape != (height, width):
        raise ValueError("SAM3 mask dimensions must match the discovery pass")
    if pixels.dtype.kind not in "biuf":
        raise TypeError("SAM3 mask value must be numeric")
    # Em float64, como a comparação com o float do Python sempre foi: em float32 um limiar como
    # 0.7 arredonda para baixo e o pixel igual a float32(0.7) viraria primeiro plano.
    probabilities = pixels.astype(np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("SAM3 mask value must be finite")
    return InlineMask(probabilities >= threshold)


def _native_array(value: object, name: str) -> NDArray[Any]:
    """Detach an SDK value to a NumPy array, never as a Python object per element.

    A PyTorch tensor may live on the GPU and in ``bfloat16``, which NumPy cannot hold: it is moved
    to the CPU in float64 (exact for every float and boolean dtype), as the other backends detach
    their tensors. Arrays, nested sequences and ``tolist()``-only values are read directly.

    Raises:
        TypeError: If the value is not a sequence.
        ValueError: If its nested sequences are ragged.
    """
    import numpy as np

    detach = getattr(value, "detach", None)
    if callable(detach):
        return np.asarray(detach().to("cpu").double().numpy())
    native = value
    if not hasattr(value, "__array__") and hasattr(value, "tolist"):
        native = value.tolist()
    if not hasattr(native, "__array__") and (
        not isinstance(native, Sequence) or isinstance(native, (str, bytes))
    ):
        raise TypeError(f"SAM3 {name} must be a sequence")
    try:
        return np.asarray(native)
    except ValueError as error:  # propostas ou linhas de tamanhos diferentes
        raise ValueError("SAM3 mask dimensions must match the discovery pass") from error


def _native_sequence(value: object, name: str) -> Sequence[object]:
    native = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(native, Sequence) or isinstance(native, (str, bytes)):
        raise TypeError(f"SAM3 {name} must be a sequence")
    return cast(Sequence[object], native)


def _numeric_sequence(value: object, name: str, *, length: int) -> tuple[float, ...]:
    native = _native_sequence(value, name)
    if len(native) != length:
        raise ValueError(f"SAM3 {name} must contain {length} numbers")
    return tuple(_finite_number(item, name) for item in native)


def _finite_number(value: object, name: str) -> float:
    # bool é subclasse de int, mas um bool nativo não é score nem coordenada: rejeitado como no
    # SAM2 e no Florence-2 (#617).
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"SAM3 {name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"SAM3 {name} must be finite")
    return result

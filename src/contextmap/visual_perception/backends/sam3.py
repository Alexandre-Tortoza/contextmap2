"""SAM3 adapter for the canonical Region Discovery capability."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from time import perf_counter
from typing import Protocol, cast

from ..discovery import (
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
)


class Sam3Strategy(StrEnum):
    """Explicit SAM3 discovery/query strategies supported by the adapter."""

    AUTOMATIC = "automatic"
    TEXT_PROMPT = "text_prompt"
    POINT_GRID = "point_grid"
    TRACKER = "tracker"
    PCS = "pcs"


@dataclass(frozen=True, slots=True)
class Sam3Config:
    """Effective configuration for one SAM3 Region Discovery adapter."""

    checkpoint: str
    model_version: str = "unknown"
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
    """SDK-isolated SAM3 proposal normalized to Python scalar containers."""

    proposal_id: str
    box: tuple[float, float, float, float]
    mask: tuple[bool, ...]
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
    ) -> None:
        """Bind a loaded processor to a pass-aware image loader."""
        self._processor = processor
        self._image_loader = image_loader

    def predict(self, discovery_input: DiscoveryInput, config: Sam3Config) -> Sam3NativeOutput:
        """Run supported official image inference without a strategy fallback."""
        if config.strategy is not Sam3Strategy.TEXT_PROMPT:
            raise ValueError("official SAM3 image runtime supports only text_prompt strategy")
        if config.prompt is None:
            raise ValueError("official SAM3 image runtime requires a text prompt")

        image = self._image_loader(discovery_input)
        validate_materialized_discovery_image(image, discovery_input)
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
        if len(proposal.mask) != width * height:
            raise ValueError("SAM3 proposal mask length must match discovery pass dimensions")
        mask_box = _mask_bounding_box(proposal.mask, width=width, height=height)
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
            mask=InlineMask(width=width, height=height, data=proposal.mask),
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


def _mask_bounding_box(mask: tuple[bool, ...], *, width: int, height: int) -> BoundingBox | None:
    """Return the tight half-open box of the true pixels, or ``None`` for an empty mask."""
    rows = [index for index in range(height) if any(mask[index * width : (index + 1) * width])]
    if not rows:
        return None
    x_min = width
    x_max = 0
    for index in rows:
        row = mask[index * width : (index + 1) * width]
        x_min = min(x_min, row.index(True))
        x_max = max(x_max, width - row[::-1].index(True))
    return BoundingBox(x_min=x_min, y_min=rows[0], x_max=x_max, y_max=rows[-1] + 1)


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
    mask_value = output.get("masks_logits", output.get("masks"))
    masks = _native_sequence(mask_value, "masks")
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
    value: object, *, width: int, height: int, threshold: float
) -> tuple[bool, ...]:
    native = value.tolist() if hasattr(value, "tolist") else value
    rows = _native_sequence(native, "mask")
    if len(rows) == 1:
        possible_rows = _native_sequence(rows[0], "mask channel")
        if len(possible_rows) == height:
            rows = possible_rows
    if len(rows) != height:
        raise ValueError("SAM3 mask dimensions must match the discovery pass")

    flattened: list[bool] = []
    for row in rows:
        pixels = _native_sequence(row, "mask row")
        if len(pixels) != width:
            raise ValueError("SAM3 mask dimensions must match the discovery pass")
        flattened.extend(_finite_number(pixel, "mask value") >= threshold for pixel in pixels)
    return tuple(flattened)


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
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        raise TypeError(f"SAM3 {name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"SAM3 {name} must be finite")
    return result

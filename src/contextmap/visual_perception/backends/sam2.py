"""SAM2 adapter for the canonical Region Discovery capability."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
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


@dataclass(frozen=True, slots=True)
class Sam2Config:
    """Effective configuration for one SAM2 Region Discovery adapter."""

    checkpoint: str
    model_version: str = "unknown"
    device: str = "cpu"
    precision: str = "float32"
    predicted_iou_threshold: float = 0.0
    stability_threshold: float = 0.0
    automatic_mask_settings: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate explicit backend selection and native thresholds."""
        if not self.checkpoint or not self.model_version or not self.device or not self.precision:
            raise ValueError("SAM2 checkpoint, version, device, and precision must not be empty")
        _validate_unit_threshold(self.predicted_iou_threshold, "predicted_iou_threshold")
        _validate_unit_threshold(self.stability_threshold, "stability_threshold")
        keys = [key for key, _ in self.automatic_mask_settings]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("SAM2 automatic mask setting names must be non-empty and unique")

    @property
    def digest(self) -> str:
        """Return a deterministic digest of the effective SAM2 configuration."""
        payload = {
            "checkpoint": self.checkpoint,
            "model_version": self.model_version,
            "device": self.device,
            "precision": self.precision,
            "predicted_iou_threshold": self.predicted_iou_threshold,
            "stability_threshold": self.stability_threshold,
            "automatic_mask_settings": list(self.automatic_mask_settings),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(serialized.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Sam2NativeProposal:
    """SDK-isolated SAM2 proposal normalized to Python scalar containers."""

    proposal_id: str
    box: tuple[float, float, float, float]
    mask: tuple[bool, ...]
    predicted_iou: float
    stability_score: float
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate native scalar output before canonical normalization."""
        if not self.proposal_id:
            raise ValueError("SAM2 proposal_id must not be empty")
        if not isfinite(self.predicted_iou) or not isfinite(self.stability_score):
            raise ValueError("SAM2 native scores must be finite")


class Sam2Runtime(Protocol):
    """Internal seam around a loaded SAM2 model/runtime."""

    def predict(
        self, discovery_input: DiscoveryInput, config: Sam2Config
    ) -> tuple[Sam2NativeProposal, ...]:
        """Run SAM2 and return scalar-only native proposals."""
        ...


class _AutomaticMaskGenerator(Protocol):
    """Minimum official SAM2 automatic-mask generator surface used here."""

    def generate(self, image: object) -> list[dict[str, object]]:
        """Return official automatic-mask records for one image."""
        ...


class Sam2AutomaticMaskRuntime:
    """Execute and isolate the official SAM2 automatic-mask generator API."""

    def __init__(
        self,
        *,
        mask_generator: _AutomaticMaskGenerator,
        image_loader: Callable[[DiscoveryInput], object],
        config_digest: str,
    ) -> None:
        """Bind a configured generator to an explicit prepared-image loader."""
        if not config_digest:
            raise ValueError("SAM2 runtime configuration digest must not be empty")
        self._mask_generator = mask_generator
        self._image_loader = image_loader
        self._config_digest = config_digest

    @classmethod
    def from_model(
        cls,
        *,
        model: object,
        image_loader: Callable[[DiscoveryInput], object],
        config: Sam2Config,
    ) -> Sam2AutomaticMaskRuntime:
        """Construct the official generator lazily from a loaded SAM2 model.

        The optional dependency remains inside this infrastructure module. The
        supplied loader must materialize the configured discovery pass as an HWC
        uint8 image accepted by SAM2.
        """
        settings = dict(config.automatic_mask_settings)
        reserved = {"model", "pred_iou_thresh", "stability_score_thresh", "output_mode"}
        conflicts = reserved.intersection(settings)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"SAM2 automatic mask settings duplicate owned fields: {names}")
        module = import_module("sam2.automatic_mask_generator")
        generator_type = module.SAM2AutomaticMaskGenerator
        generator = generator_type(
            model=model,
            pred_iou_thresh=config.predicted_iou_threshold,
            stability_score_thresh=config.stability_threshold,
            output_mode="binary_mask",
            **settings,
        )
        return cls(
            mask_generator=generator,
            image_loader=image_loader,
            config_digest=config.digest,
        )

    def predict(
        self, discovery_input: DiscoveryInput, config: Sam2Config
    ) -> tuple[Sam2NativeProposal, ...]:
        """Run official SAM2 inference and detach every SDK-native value."""
        if config.digest != self._config_digest:
            raise ValueError("SAM2 runtime configuration digest does not match the active config")
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        image = self._image_loader(discovery_input)
        validate_materialized_discovery_image(image, discovery_input)
        records = self._mask_generator.generate(image)
        return tuple(
            _parse_automatic_mask_record(record, index=index, width=width, height=height)
            for index, record in enumerate(records)
        )


class Sam2RegionDiscovery:
    """Adapt SAM2 automatic-mask output to canonical RegionCandidate values."""

    def __init__(
        self,
        *,
        config: Sam2Config,
        runtime: Sam2Runtime,
        pass_config: DiscoveryPassConfig | None = None,
        normalization_config: NormalizationConfig | None = None,
    ) -> None:
        """Build the adapter with explicit effective configuration and runtime."""
        self._config = config
        self._runtime = runtime
        self._pass_config = pass_config
        self._normalization_config = normalization_config

    def backend_provenance(self) -> BackendProvenance:
        """Return SAM2 identity in the Visual Perception Core contract."""
        return BackendProvenance(
            backend_id="sam2",
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
        """Run SAM2 for one pass and normalize accepted native proposals."""
        started = perf_counter()
        native_proposals = self._runtime.predict(discovery_input, self._config)
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        candidates = tuple(
            self._normalize(proposal, discovery_input, width, height)
            for proposal in native_proposals
            if proposal.predicted_iou >= self._config.predicted_iou_threshold
            and proposal.stability_score >= self._config.stability_threshold
        )
        duration_ms = (perf_counter() - started) * 1000
        return DiscoveryOutput(
            candidates=candidates,
            diagnostics=BackendDiagnostics(
                duration_ms=duration_ms,
                proposal_count=len(candidates),
                warnings=(),
                metadata=(
                    ("raw_proposal_count", len(native_proposals)),
                    ("filtered_proposal_count", len(native_proposals) - len(candidates)),
                    ("checkpoint", self._config.checkpoint),
                    ("model_version", self._config.model_version),
                    ("device", self._config.device),
                    ("precision", self._config.precision),
                    ("config_digest", self._config.digest),
                ),
            ),
        )

    def _normalize(
        self,
        proposal: Sam2NativeProposal,
        discovery_input: DiscoveryInput,
        width: int,
        height: int,
    ) -> RegionCandidate:
        """Convert one scalar SAM2 proposal without leaking native model objects."""
        if len(proposal.mask) != width * height:
            raise ValueError("SAM2 proposal mask length must match discovery pass dimensions")
        return RegionCandidate(
            candidate_id=proposal.proposal_id,
            source_observation_id=discovery_input.prepared_image.source_observation_id,
            perception_run_id=discovery_input.perception_run_id,
            perception_result_id=discovery_input.perception_result_id,
            image_width=width,
            image_height=height,
            bounding_box=BoundingBox(
                x_min=proposal.box[0],
                y_min=proposal.box[1],
                x_max=proposal.box[2],
                y_max=proposal.box[3],
            ),
            mask=InlineMask(width=width, height=height, data=proposal.mask),
            score=BackendScore(
                name="predicted_iou",
                value=proposal.predicted_iou,
                semantics=(
                    "SAM2-native predicted mask IoU; not calibrated across discovery backends"
                ),
            ),
            provenance=RegionProvenance(
                backend_id="sam2",
                backend_version=self._config.model_version,
                checkpoint=self._config.checkpoint,
                config_digest=self._config.digest,
                discovery_pass_id=discovery_input.discovery_pass.pass_id,
                native_proposal_id=proposal.proposal_id,
            ),
            native_metadata=(
                ("stability_score", proposal.stability_score),
                *proposal.metadata,
            ),
        )


def _validate_unit_threshold(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"SAM2 {name} must be between zero and one")


def _parse_automatic_mask_record(
    record: Mapping[str, object], *, index: int, width: int, height: int
) -> Sam2NativeProposal:
    """Convert one documented SAM2 automatic-mask record to scalar containers."""
    box = _numeric_sequence(record.get("bbox"), "bbox", length=4)
    x, y, box_width, box_height = box
    area = _finite_number(record.get("area"), "area")
    return Sam2NativeProposal(
        proposal_id=f"sam2-{index:06d}",
        box=(x, y, x + box_width, y + box_height),
        mask=_binary_mask(record.get("segmentation"), width=width, height=height),
        predicted_iou=_finite_number(record.get("predicted_iou"), "predicted_iou"),
        stability_score=_finite_number(record.get("stability_score"), "stability_score"),
        metadata=(("area_pixels", area),),
    )


def _binary_mask(value: object, *, width: int, height: int) -> tuple[bool, ...]:
    native = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(native, Sequence) or isinstance(native, (str, bytes)):
        raise TypeError("SAM2 segmentation must be a two-dimensional binary mask")
    rows = cast(Sequence[object], native)
    if len(rows) != height:
        raise ValueError("SAM2 segmentation mask dimensions must match the discovery pass")
    flattened: list[bool] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != width:
            raise ValueError("SAM2 segmentation mask dimensions must match the discovery pass")
        for item in row:
            if item not in (False, True, 0, 1):
                raise TypeError("SAM2 segmentation mask must contain binary values")
            flattened.append(bool(item))
    return tuple(flattened)


def _numeric_sequence(value: object, name: str, *, length: int) -> tuple[float, ...]:
    native = value.tolist() if hasattr(value, "tolist") else value
    if (
        not isinstance(native, Sequence)
        or isinstance(native, (str, bytes))
        or len(native) != length
    ):
        raise ValueError(f"SAM2 {name} must contain {length} numbers")
    return tuple(_finite_number(item, name) for item in native)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"SAM2 {name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"SAM2 {name} must be finite")
    return result

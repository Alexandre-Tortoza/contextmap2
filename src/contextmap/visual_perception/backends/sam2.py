"""SAM2 adapter for the canonical Region Discovery capability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from time import perf_counter
from typing import Protocol

from ..discovery import BackendDiagnostics, DiscoveryInput, DiscoveryOutput
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


class Sam2RegionDiscovery:
    """Adapt SAM2 automatic-mask output to canonical RegionCandidate values."""

    def __init__(self, *, config: Sam2Config, runtime: Sam2Runtime) -> None:
        """Build the adapter with explicit effective configuration and runtime."""
        self._config = config
        self._runtime = runtime

    def discover(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        """Run SAM2 for one pass and normalize accepted native proposals."""
        started = perf_counter()
        native_proposals = self._runtime.predict(discovery_input, self._config)
        width = int(discovery_input.discovery_pass.window.width)
        height = int(discovery_input.discovery_pass.window.height)
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
            native_metadata=(("stability_score", proposal.stability_score),),
        )


def _validate_unit_threshold(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"SAM2 {name} must be between zero and one")

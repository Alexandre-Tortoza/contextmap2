"""Florence-2 adapter dedicated to the Region Discovery capability."""

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
class Florence2Config:
    """Effective configuration for Florence-2 region discovery only."""

    checkpoint: str
    task: str
    model_version: str = "unknown"
    device: str = "cpu"
    precision: str = "float32"
    prompt: str | None = None
    box_threshold: float = 0.0
    generation_settings: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate model and selected region-producing task."""
        if not self.checkpoint or not self.model_version or not self.device or not self.precision:
            raise ValueError(
                "Florence-2 checkpoint, version, device, and precision must not be empty"
            )
        if not self.task:
            raise ValueError("Florence-2 region discovery task must not be empty")
        if not isfinite(self.box_threshold) or not 0 <= self.box_threshold <= 1:
            raise ValueError("Florence-2 box_threshold must be between zero and one")
        keys = [key for key, _ in self.generation_settings]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Florence-2 generation setting names must be non-empty and unique")

    @property
    def digest(self) -> str:
        """Return a deterministic digest of the effective Florence-2 configuration."""
        payload = {
            "checkpoint": self.checkpoint,
            "task": self.task,
            "model_version": self.model_version,
            "device": self.device,
            "precision": self.precision,
            "prompt": self.prompt,
            "box_threshold": self.box_threshold,
            "generation_settings": list(self.generation_settings),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(serialized.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Florence2NativeRegion:
    """Parsed Florence-2 geometry kept inside the infrastructure adapter."""

    proposal_id: str
    box: tuple[float, float, float, float]
    score: float | None
    mask: tuple[bool, ...] | None = None
    parsed_text: str | None = None
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate parser output before canonical normalization."""
        if not self.proposal_id:
            raise ValueError("Florence-2 proposal_id must not be empty")
        if self.score is not None and not isfinite(self.score):
            raise ValueError("Florence-2 native score must be finite when present")


@dataclass(frozen=True, slots=True)
class Florence2NativeOutput:
    """Structured output of the Florence-2 region-task parser."""

    regions: tuple[Florence2NativeRegion, ...]
    warnings: tuple[str, ...] = ()
    parsing_diagnostics: tuple[tuple[str, JsonScalar], ...] = ()


class Florence2Runtime(Protocol):
    """Internal seam around Florence model execution and native parsing."""

    def predict(
        self, discovery_input: DiscoveryInput, config: Florence2Config
    ) -> Florence2NativeOutput:
        """Execute the configured Florence region task and parse its output."""
        ...


class Florence2RegionDiscovery:
    """Adapt Florence-2 region tasks without producing semantic claims."""

    def __init__(self, *, config: Florence2Config, runtime: Florence2Runtime) -> None:
        """Build the region-only adapter with explicit model runtime."""
        self._config = config
        self._runtime = runtime

    def discover(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        """Run one Florence-2 region task and normalize parsed geometry."""
        started = perf_counter()
        native_output = self._runtime.predict(discovery_input, self._config)
        width = int(discovery_input.discovery_pass.window.width)
        height = int(discovery_input.discovery_pass.window.height)
        accepted = tuple(
            region
            for region in native_output.regions
            if region.score is None or region.score >= self._config.box_threshold
        )
        candidates = tuple(
            self._normalize(region, discovery_input, width, height) for region in accepted
        )
        duration_ms = (perf_counter() - started) * 1000
        metadata = (
            *native_output.parsing_diagnostics,
            ("raw_proposal_count", len(native_output.regions)),
            ("filtered_proposal_count", len(native_output.regions) - len(candidates)),
            ("checkpoint", self._config.checkpoint),
            ("model_version", self._config.model_version),
            ("device", self._config.device),
            ("precision", self._config.precision),
            ("task", self._config.task),
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
        region: Florence2NativeRegion,
        discovery_input: DiscoveryInput,
        width: int,
        height: int,
    ) -> RegionCandidate:
        """Convert one parsed Florence region without semantic promotion."""
        mask = None
        if region.mask is not None:
            if len(region.mask) != width * height:
                raise ValueError(
                    f"Florence-2 proposal {region.proposal_id} mask length must match "
                    "discovery pass dimensions"
                )
            mask = InlineMask(width=width, height=height, data=region.mask)

        score = None
        if region.score is not None:
            score = BackendScore(
                name="region_score",
                value=region.score,
                semantics=(
                    "Florence-2 task-native region score; not calibrated across discovery backends"
                ),
            )

        native_metadata = region.metadata
        if region.parsed_text is not None:
            native_metadata = (*native_metadata, ("parsed_text", region.parsed_text))
        query = self._config.task
        if self._config.prompt:
            query = f"{query}:{self._config.prompt}"
        return RegionCandidate(
            candidate_id=region.proposal_id,
            source_observation_id=discovery_input.prepared_image.source_observation_id,
            perception_run_id=discovery_input.perception_run_id,
            perception_result_id=discovery_input.perception_result_id,
            image_width=width,
            image_height=height,
            bounding_box=BoundingBox(
                x_min=region.box[0],
                y_min=region.box[1],
                x_max=region.box[2],
                y_max=region.box[3],
            ),
            mask=mask,
            score=score,
            provenance=RegionProvenance(
                backend_id="florence2_region_discovery",
                backend_version=self._config.model_version,
                checkpoint=self._config.checkpoint,
                config_digest=self._config.digest,
                discovery_pass_id=discovery_input.discovery_pass.pass_id,
                native_proposal_id=region.proposal_id,
                query=query,
            ),
            native_metadata=native_metadata,
        )

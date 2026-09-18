"""SAM3 adapter for the canonical Region Discovery capability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
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


class Sam3RegionDiscovery:
    """Adapt configured SAM3 output to canonical RegionCandidate values."""

    def __init__(self, *, config: Sam3Config, runtime: Sam3Runtime) -> None:
        """Build the adapter with explicit configuration and no fallback runtime."""
        self._config = config
        self._runtime = runtime

    def discover(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        """Run exactly the selected SAM3 strategy for one discovery pass."""
        started = perf_counter()
        native_output = self._runtime.predict(discovery_input, self._config)
        width = int(discovery_input.discovery_pass.window.width)
        height = int(discovery_input.discovery_pass.window.height)
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
        """Convert one scalar SAM3 proposal without semantic promotion."""
        if len(proposal.mask) != width * height:
            raise ValueError("SAM3 proposal mask length must match discovery pass dimensions")
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
            bounding_box=BoundingBox(
                x_min=proposal.box[0],
                y_min=proposal.box[1],
                x_max=proposal.box[2],
                y_max=proposal.box[3],
            ),
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
            native_metadata=proposal.metadata,
        )


def _validate_unit_threshold(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"SAM3 {name} must be between zero and one")

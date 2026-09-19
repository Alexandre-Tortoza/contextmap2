"""Backend-neutral discovery passes and deterministic coordinate remapping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from math import isfinite
from typing import Protocol, runtime_checkable

from .models import BackendProvenance, PreparedImage, Region2D
from .normalization import NormalizationConfig, normalize_regions
from .region_models import (
    BoundingBox,
    InlineMask,
    JsonScalar,
    RegionCandidate,
    RejectedRegionCandidate,
    RejectionReason,
)


class PassKind(StrEnum):
    """Kinds of spatial inputs presented to a discovery backend."""

    FULL_FRAME = "full_frame"
    TILE = "tile"


class BorderPolicy(StrEnum):
    """Policies for proposals that touch an internal tile border."""

    KEEP = "keep"
    REJECT_INTERNAL_BORDER = "reject_internal_border"


@dataclass(frozen=True, slots=True)
class BackendDiagnostics:
    """Canonical timing and diagnostic summary emitted by a backend call."""

    duration_ms: float
    proposal_count: int
    warnings: tuple[str, ...]
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate timing, counts, and inspectable metadata."""
        if self.duration_ms < 0:
            raise ValueError("backend duration_ms must be non-negative")
        if self.proposal_count < 0:
            raise ValueError("backend proposal_count must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible diagnostic record."""
        return {
            "duration_ms": self.duration_ms,
            "proposal_count": self.proposal_count,
            "warnings": list(self.warnings),
            "metadata": [{"name": key, "value": value} for key, value in self.metadata],
        }


@dataclass(frozen=True, slots=True)
class DiscoveryPass:
    """One image window and scale presented to a discovery backend."""

    pass_id: str
    kind: PassKind
    window: BoundingBox
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate stable identity and scale."""
        if not self.pass_id:
            raise ValueError("discovery pass id must not be empty")
        if not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("discovery pass scale must be positive and finite")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible pass definition."""
        return {
            "pass_id": self.pass_id,
            "kind": self.kind.value,
            "window": self.window.to_dict(),
            "scale": self.scale,
            "input_dimensions": [self.input_width, self.input_height],
            "input_to_prepared_scale": [
                self.window.width / self.input_width,
                self.window.height / self.input_height,
            ],
        }

    @property
    def input_width(self) -> int:
        """Return the scaled model-input width for this pass."""
        return max(1, round(self.window.width * self.scale))

    @property
    def input_height(self) -> int:
        """Return the scaled model-input height for this pass."""
        return max(1, round(self.window.height * self.scale))


@dataclass(frozen=True, slots=True)
class TilingConfig:
    """Configure one deterministic overlapping tile grid."""

    tile_width: int
    tile_height: int
    overlap_x: int = 0
    overlap_y: int = 0
    border_policy: BorderPolicy = BorderPolicy.KEEP
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Reject grids that cannot advance deterministically."""
        if self.tile_width <= 0 or self.tile_height <= 0:
            raise ValueError("tile dimensions must be positive")
        if self.overlap_x < 0 or self.overlap_y < 0:
            raise ValueError("tile overlaps must be non-negative")
        if self.overlap_x >= self.tile_width or self.overlap_y >= self.tile_height:
            raise ValueError("tile overlap must be smaller than its tile dimension")
        if not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("tile scale must be positive and finite")


@dataclass(frozen=True, slots=True)
class DiscoveryPassConfig:
    """Configure full-frame and optional tiled discovery passes."""

    include_full_frame: bool = True
    tiling: TilingConfig | None = None
    additional_tilings: tuple[TilingConfig, ...] = ()
    max_candidates_per_pass: int | None = None

    def __post_init__(self) -> None:
        """Require at least one pass and valid optional budgets."""
        if not self.include_full_frame and self.tiling is None and not self.additional_tilings:
            raise ValueError("discovery configuration must enable at least one pass")
        if self.max_candidates_per_pass is not None and self.max_candidates_per_pass <= 0:
            raise ValueError("max_candidates_per_pass must be positive when configured")


@dataclass(frozen=True, slots=True)
class DiscoveryInput:
    """Canonical input for one replaceable Region Discovery backend call."""

    prepared_image: PreparedImage
    discovery_pass: DiscoveryPass
    perception_run_id: str
    perception_result_id: str

    def __post_init__(self) -> None:
        """Require inference identities distinct from the physical frame."""
        if not self.perception_run_id or not self.perception_result_id:
            raise ValueError("perception run and result ids must not be empty")


@dataclass(frozen=True, slots=True)
class DiscoveryOutput:
    """Canonical candidate and diagnostics output from one backend invocation."""

    candidates: tuple[RegionCandidate, ...]
    diagnostics: BackendDiagnostics

    def __post_init__(self) -> None:
        """Keep reported and materialized proposal counts consistent."""
        if self.diagnostics.proposal_count != len(self.candidates):
            raise ValueError("backend proposal_count must equal the number of candidates")


@runtime_checkable
class RegionCandidateDiscovery(Protocol):
    """Internal adapter boundary that proposes candidates for one pass."""

    def backend_provenance(self) -> BackendProvenance:
        """Report the backend and effective configuration identity."""
        ...

    def discover_candidates(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        """Produce backend-neutral candidates for one prepared image pass."""
        ...


@dataclass(frozen=True, slots=True)
class DiscoveryRunResult:
    """Globally remapped proposals, rejections, passes, and backend diagnostics."""

    candidates: tuple[RegionCandidate, ...]
    rejected: tuple[RejectedRegionCandidate, ...]
    passes: tuple[DiscoveryPass, ...]
    diagnostics: tuple[BackendDiagnostics, ...]


def discover_canonical_regions(
    prepared_image: PreparedImage,
    backend: RegionCandidateDiscovery,
    *,
    pass_config: DiscoveryPassConfig | None = None,
    normalization_config: NormalizationConfig | None = None,
) -> tuple[Region2D, ...]:
    """Execute pass-level discovery and return the one canonical Region2D contract.

    The core port scopes returned region identities through the eventual
    ``PerceptionResult``. The internal candidate scope used here is deterministic
    and exists only to validate that a single discovery execution is not mixed.
    """
    provenance = backend.backend_provenance()
    fingerprint = provenance.configuration_fingerprint or provenance.version
    scope = f"{provenance.backend_id}:{fingerprint}"
    discovery = run_discovery_passes(
        prepared_image=prepared_image,
        backend=backend,
        perception_run_id=scope,
        perception_result_id=f"{scope}:{prepared_image.source_observation_id}",
        config=pass_config,
    )
    return normalize_regions(
        discovery.candidates,
        prepared_image,
        provenance,
        normalization_config,
    ).regions


def validate_materialized_discovery_image(
    image: object,
    discovery_input: DiscoveryInput,
) -> None:
    """Require a loader to materialize the exact crop/resize for one pass.

    PIL-compatible images expose ``size`` as ``(width, height)`` while
    array-compatible HWC images expose ``shape`` as ``(height, width, ...)``.

    Args:
        image: Model-ready image returned by the configured loader.
        discovery_input: Pass whose scaled dimensions the image must match.

    Raises:
        TypeError: If the image exposes no inspectable spatial dimensions.
        ValueError: If its materialized dimensions differ from the pass.
    """
    dimensions = _materialized_image_dimensions(image)
    expected = (
        discovery_input.discovery_pass.input_width,
        discovery_input.discovery_pass.input_height,
    )
    if dimensions != expected:
        raise ValueError(
            "materialized discovery image dimensions must match the discovery pass: "
            f"expected {expected}, received {dimensions}"
        )


def _materialized_image_dimensions(image: object) -> tuple[int, int]:
    size = _dimension_pair(getattr(image, "size", None))
    if size is not None:
        return size

    shape = _dimension_pair(getattr(image, "shape", None))
    if shape is not None:
        return shape[1], shape[0]

    raise TypeError(
        "materialized discovery image must expose size=(width, height) "
        "or shape=(height, width, ...)"
    )


def _dimension_pair(value: object) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    first, second = value[0], value[1]
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in (first, second)
    ):
        return None
    return first, second


def build_discovery_passes(
    prepared_image: PreparedImage,
    config: DiscoveryPassConfig | None = None,
) -> tuple[DiscoveryPass, ...]:
    """Build full-frame and tile passes in deterministic row-major order."""
    if config is None:
        config = DiscoveryPassConfig()
    passes: list[DiscoveryPass] = []
    if config.include_full_frame:
        passes.append(
            DiscoveryPass(
                pass_id="full-frame",
                kind=PassKind.FULL_FRAME,
                window=BoundingBox(
                    x_min=0,
                    y_min=0,
                    x_max=prepared_image.width,
                    y_max=prepared_image.height,
                ),
            )
        )

    tilings = (() if config.tiling is None else (config.tiling,)) + config.additional_tilings
    for tiling_index, tiling in enumerate(tilings):
        x_positions = _tile_positions(prepared_image.width, tiling.tile_width, tiling.overlap_x)
        y_positions = _tile_positions(prepared_image.height, tiling.tile_height, tiling.overlap_y)
        tile_index = 0
        prefix = "tile" if len(tilings) == 1 else f"tile-s{tiling_index:02d}"
        for y_min in y_positions:
            for x_min in x_positions:
                x_max = min(x_min + tiling.tile_width, prepared_image.width)
                y_max = min(y_min + tiling.tile_height, prepared_image.height)
                passes.append(
                    DiscoveryPass(
                        pass_id=f"{prefix}-{tile_index:04d}",
                        kind=PassKind.TILE,
                        window=BoundingBox(
                            x_min=x_min,
                            y_min=y_min,
                            x_max=x_max,
                            y_max=y_max,
                        ),
                        scale=tiling.scale,
                    )
                )
                tile_index += 1
    return tuple(passes)


def run_discovery_passes(
    *,
    prepared_image: PreparedImage,
    backend: RegionCandidateDiscovery,
    perception_run_id: str,
    perception_result_id: str,
    config: DiscoveryPassConfig | None = None,
) -> DiscoveryRunResult:
    """Execute configured passes and remap every accepted proposal globally."""
    if config is None:
        config = DiscoveryPassConfig()
    passes = build_discovery_passes(prepared_image, config)
    candidates: list[RegionCandidate] = []
    rejected: list[RejectedRegionCandidate] = []
    diagnostics: list[BackendDiagnostics] = []

    tiling_by_pass = _tiling_by_pass(passes, config)
    for discovery_pass in passes:
        discovery_input = DiscoveryInput(
            prepared_image=prepared_image,
            discovery_pass=discovery_pass,
            perception_run_id=perception_run_id,
            perception_result_id=perception_result_id,
        )
        output = backend.discover_candidates(discovery_input)
        diagnostics.append(output.diagnostics)
        pass_candidates = output.candidates
        if config.max_candidates_per_pass is not None:
            kept = pass_candidates[: config.max_candidates_per_pass]
            for candidate in pass_candidates[config.max_candidates_per_pass :]:
                rejected.append(
                    _rejection(
                        discovery_pass,
                        candidate,
                        RejectionReason.REGION_BUDGET_EXCEEDED,
                        "candidate exceeded configured per-pass budget",
                    )
                )
            pass_candidates = kept

        for candidate in pass_candidates:
            _validate_backend_candidate(candidate, discovery_input)
            tiling = tiling_by_pass.get(discovery_pass.pass_id)
            if tiling is not None and _reject_for_internal_border(
                candidate, discovery_pass, prepared_image, tiling.border_policy
            ):
                rejected.append(
                    _rejection(
                        discovery_pass,
                        candidate,
                        RejectionReason.TILE_BORDER_TRUNCATION,
                        "candidate touches an internal tile border",
                    )
                )
                continue
            candidates.append(_remap_candidate(candidate, discovery_pass, prepared_image))

    return DiscoveryRunResult(
        candidates=tuple(candidates),
        rejected=tuple(rejected),
        passes=passes,
        diagnostics=tuple(diagnostics),
    )


def _tile_positions(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    last_start = length - tile_size
    positions = list(range(0, last_start + 1, tile_size - overlap))
    if positions[-1] != last_start:
        positions.append(last_start)
    return tuple(positions)


def _tiling_by_pass(
    passes: tuple[DiscoveryPass, ...], config: DiscoveryPassConfig
) -> dict[str, TilingConfig]:
    tilings = (() if config.tiling is None else (config.tiling,)) + config.additional_tilings
    mapping: dict[str, TilingConfig] = {}
    if not tilings:
        return mapping
    tile_passes = [item for item in passes if item.kind is PassKind.TILE]
    cursor = 0
    for tiling in tilings:
        # Cada grid pode ser reconhecido pelo número determinístico de janelas.
        pass_count = len(
            _tile_positions(
                int(max(item.window.x_max for item in passes)),
                tiling.tile_width,
                tiling.overlap_x,
            )
        ) * len(
            _tile_positions(
                int(max(item.window.y_max for item in passes)),
                tiling.tile_height,
                tiling.overlap_y,
            )
        )
        for item in tile_passes[cursor : cursor + pass_count]:
            mapping[item.pass_id] = tiling
        cursor += pass_count
    return mapping


def _validate_backend_candidate(
    candidate: RegionCandidate, discovery_input: DiscoveryInput
) -> None:
    discovery_pass = discovery_input.discovery_pass
    expected_dimensions = (discovery_pass.input_width, discovery_pass.input_height)
    if (candidate.image_width, candidate.image_height) != expected_dimensions:
        raise ValueError("backend candidate dimensions must match discovery pass dimensions")
    if candidate.source_observation_id != discovery_input.prepared_image.source_observation_id:
        raise ValueError("backend candidate source observation does not match discovery input")
    if candidate.perception_run_id != discovery_input.perception_run_id:
        raise ValueError("backend candidate perception run does not match discovery input")
    if candidate.perception_result_id != discovery_input.perception_result_id:
        raise ValueError("backend candidate perception result does not match discovery input")
    if candidate.provenance.discovery_pass_id != discovery_pass.pass_id:
        raise ValueError("backend candidate provenance does not match discovery pass")


def _reject_for_internal_border(
    candidate: RegionCandidate,
    discovery_pass: DiscoveryPass,
    prepared_image: PreparedImage,
    policy: BorderPolicy,
) -> bool:
    if policy is BorderPolicy.KEEP or candidate.bounding_box is None:
        return False
    box = candidate.bounding_box
    window = discovery_pass.window
    touches_left = box.x_min <= 0 and window.x_min > 0
    touches_top = box.y_min <= 0 and window.y_min > 0
    touches_right = box.x_max >= discovery_pass.input_width and window.x_max < prepared_image.width
    touches_bottom = (
        box.y_max >= discovery_pass.input_height and window.y_max < prepared_image.height
    )
    return touches_left or touches_top or touches_right or touches_bottom


def _remap_candidate(
    candidate: RegionCandidate,
    discovery_pass: DiscoveryPass,
    prepared_image: PreparedImage,
) -> RegionCandidate:
    x_offset = int(discovery_pass.window.x_min)
    y_offset = int(discovery_pass.window.y_min)
    x_scale = discovery_pass.window.width / discovery_pass.input_width
    y_scale = discovery_pass.window.height / discovery_pass.input_height
    box = candidate.bounding_box
    remapped_box = None
    if box is not None:
        remapped_box = BoundingBox(
            x_min=box.x_min * x_scale + x_offset,
            y_min=box.y_min * y_scale + y_offset,
            x_max=box.x_max * x_scale + x_offset,
            y_max=box.y_max * y_scale + y_offset,
        )
    mask = candidate.mask
    if isinstance(mask, InlineMask):
        mask = _resize_mask(
            mask,
            int(discovery_pass.window.width),
            int(discovery_pass.window.height),
        )
        mask = _expand_mask(mask, prepared_image.width, prepared_image.height, x_offset, y_offset)
    return replace(
        candidate,
        candidate_id=f"{discovery_pass.pass_id}/{candidate.candidate_id}",
        image_width=prepared_image.width,
        image_height=prepared_image.height,
        bounding_box=remapped_box,
        mask=mask,
    )


def _resize_mask(mask: InlineMask, output_width: int, output_height: int) -> InlineMask:
    if (mask.width, mask.height) == (output_width, output_height):
        return mask
    data = tuple(
        mask.value_at(
            min(mask.width - 1, int(x * mask.width / output_width)),
            min(mask.height - 1, int(y * mask.height / output_height)),
        )
        for y in range(output_height)
        for x in range(output_width)
    )
    return InlineMask(width=output_width, height=output_height, data=data)


def _expand_mask(
    mask: InlineMask,
    output_width: int,
    output_height: int,
    x_offset: int,
    y_offset: int,
) -> InlineMask:
    data = [False] * (output_width * output_height)
    for y in range(mask.height):
        for x in range(mask.width):
            if mask.value_at(x, y):
                data[(y + y_offset) * output_width + x + x_offset] = True
    return InlineMask(width=output_width, height=output_height, data=tuple(data))


def _rejection(
    discovery_pass: DiscoveryPass,
    candidate: RegionCandidate,
    reason: RejectionReason,
    detail: str,
) -> RejectedRegionCandidate:
    return RejectedRegionCandidate(
        candidate_id=f"{discovery_pass.pass_id}/{candidate.candidate_id}",
        reason=reason,
        detail=detail,
        discovery_pass_id=discovery_pass.pass_id,
    )

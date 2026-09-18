"""Deterministic geometric filtering, merge, and Region2D geometry freeze."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import ceil, floor, isfinite

from .image_preparation import PreparedImage
from .region_models import (
    BoundingBox,
    InlineMask,
    Region2D,
    RegionCandidate,
    RegionIdentity,
    RegionProvenance,
    RejectedRegionCandidate,
    RejectionReason,
)


class MergeKind(StrEnum):
    """Backend-independent reasons for consolidating two proposals."""

    IOU_DUPLICATE = "iou_duplicate"
    CONTAINMENT = "containment"


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Explicit geometric policies applied before Region2D geometry freeze."""

    minimum_area_pixels: float = 1.0
    maximum_area_pixels: float | None = None
    minimum_valid_fraction: float = 1.0
    maximum_exclusion_fraction: float = 0.0
    duplicate_iou_threshold: float = 0.8
    containment_threshold: float = 0.95
    maximum_regions: int | None = None

    def __post_init__(self) -> None:
        """Validate policy ranges before any candidate is processed."""
        if not isfinite(self.minimum_area_pixels) or self.minimum_area_pixels <= 0:
            raise ValueError("minimum_area_pixels must be positive and finite")
        if self.maximum_area_pixels is not None and (
            not isfinite(self.maximum_area_pixels)
            or self.maximum_area_pixels < self.minimum_area_pixels
        ):
            raise ValueError("maximum_area_pixels must be finite and at least the minimum")
        for name, value in (
            ("minimum_valid_fraction", self.minimum_valid_fraction),
            ("maximum_exclusion_fraction", self.maximum_exclusion_fraction),
            ("duplicate_iou_threshold", self.duplicate_iou_threshold),
            ("containment_threshold", self.containment_threshold),
        ):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.maximum_regions is not None and self.maximum_regions <= 0:
            raise ValueError("maximum_regions must be positive when configured")

    @property
    def digest(self) -> str:
        """Return a deterministic identity for the effective geometric policy."""
        payload = {
            "minimum_area_pixels": self.minimum_area_pixels,
            "maximum_area_pixels": self.maximum_area_pixels,
            "minimum_valid_fraction": self.minimum_valid_fraction,
            "maximum_exclusion_fraction": self.maximum_exclusion_fraction,
            "duplicate_iou_threshold": self.duplicate_iou_threshold,
            "containment_threshold": self.containment_threshold,
            "maximum_regions": self.maximum_regions,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(serialized.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class MergeDecision:
    """Record why one proposal contributed to another canonical region."""

    representative_candidate_id: str
    merged_candidate_id: str
    kind: MergeKind
    iou: float
    containment_fraction: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible merge diagnostic."""
        return {
            "representative_candidate_id": self.representative_candidate_id,
            "merged_candidate_id": self.merged_candidate_id,
            "kind": self.kind.value,
            "iou": self.iou,
            "containment_fraction": self.containment_fraction,
        }


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Frozen regions plus every filtering and merge decision."""

    regions: tuple[Region2D, ...]
    rejected: tuple[RejectedRegionCandidate, ...]
    merge_decisions: tuple[MergeDecision, ...]
    config_digest: str


@dataclass(frozen=True, slots=True)
class _Geometry:
    candidate: RegionCandidate
    bounding_box: BoundingBox
    pixels: frozenset[tuple[int, int]]
    area_pixels: float


@dataclass(slots=True)
class _RegionGroup:
    representative: _Geometry
    contributors: list[RegionCandidate]


def normalize_regions(
    candidates: tuple[RegionCandidate, ...],
    prepared_image: PreparedImage,
    config: NormalizationConfig | None = None,
) -> NormalizationResult:
    """Validate, filter, merge, and freeze backend-neutral region proposals.

    Args:
        candidates: Globally remapped proposals from any discovery backend.
        prepared_image: Prepared image and optional declared spatial constraints.
        config: Version-visible geometric normalization policy.

    Returns:
        Immutable canonical regions and complete rejection/merge diagnostics.

    Raises:
        ValueError: If candidate identities or image contexts are inconsistent.
    """
    if config is None:
        config = NormalizationConfig()
    _validate_candidate_set(candidates, prepared_image)
    rejected: list[RejectedRegionCandidate] = []
    valid_geometry: list[_Geometry] = []

    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        geometry = _candidate_geometry(candidate)
        if geometry is None:
            rejected.append(
                _reject(
                    candidate,
                    RejectionReason.INVALID_GEOMETRY,
                    "candidate geometry is empty or cannot be inspected",
                )
            )
            continue
        if geometry.area_pixels < config.minimum_area_pixels:
            rejected.append(
                _reject(
                    candidate,
                    RejectionReason.AREA_BELOW_MINIMUM,
                    f"area {geometry.area_pixels} is below configured minimum",
                )
            )
            continue
        if (
            config.maximum_area_pixels is not None
            and geometry.area_pixels > config.maximum_area_pixels
        ):
            rejected.append(
                _reject(
                    candidate,
                    RejectionReason.AREA_ABOVE_MAXIMUM,
                    f"area {geometry.area_pixels} exceeds configured maximum",
                )
            )
            continue
        if _outside_valid_region(geometry, prepared_image, config.minimum_valid_fraction):
            rejected.append(
                _reject(
                    candidate,
                    RejectionReason.OUTSIDE_VALID_REGION,
                    "candidate does not satisfy configured valid-region coverage",
                )
            )
            continue
        if _overlaps_exclusion(geometry, prepared_image, config.maximum_exclusion_fraction):
            rejected.append(
                _reject(
                    candidate,
                    RejectionReason.EXCLUSION_OVERLAP,
                    "candidate exceeds configured exclusion overlap",
                )
            )
            continue
        valid_geometry.append(geometry)

    groups: list[_RegionGroup] = []
    merge_decisions: list[MergeDecision] = []
    for geometry in valid_geometry:
        match = _find_duplicate(geometry, groups, config)
        if match is None:
            groups.append(_RegionGroup(representative=geometry, contributors=[geometry.candidate]))
            continue
        group, kind, iou, containment = match
        group.contributors.append(geometry.candidate)
        merge_decisions.append(
            MergeDecision(
                representative_candidate_id=group.representative.candidate.candidate_id,
                merged_candidate_id=geometry.candidate.candidate_id,
                kind=kind,
                iou=iou,
                containment_fraction=containment,
            )
        )
        rejected.append(
            _reject(
                geometry.candidate,
                RejectionReason.MERGED_DUPLICATE,
                f"merged into {group.representative.candidate.candidate_id}",
            )
        )

    if config.maximum_regions is not None and len(groups) > config.maximum_regions:
        for group in groups[config.maximum_regions :]:
            rejected.append(
                _reject(
                    group.representative.candidate,
                    RejectionReason.REGION_BUDGET_EXCEEDED,
                    "canonical region exceeded configured maximum_regions budget",
                )
            )
        groups = groups[: config.maximum_regions]

    regions = tuple(_freeze_group(group, index + 1) for index, group in enumerate(groups))
    return NormalizationResult(
        regions=regions,
        rejected=tuple(rejected),
        merge_decisions=tuple(merge_decisions),
        config_digest=config.digest,
    )


def _validate_candidate_set(
    candidates: tuple[RegionCandidate, ...], prepared_image: PreparedImage
) -> None:
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate ids must be unique before normalization")
    if not candidates:
        return
    expected_run = candidates[0].perception_run_id
    expected_result = candidates[0].perception_result_id
    for candidate in candidates:
        if candidate.source_observation_id != prepared_image.source_observation_id:
            raise ValueError("candidate source observation does not match prepared image")
        if (candidate.image_width, candidate.image_height) != (
            prepared_image.width,
            prepared_image.height,
        ):
            raise ValueError("candidate dimensions do not match prepared image")
        if candidate.perception_run_id != expected_run:
            raise ValueError(
                "candidates from different perception runs cannot be normalized together"
            )
        if candidate.perception_result_id != expected_result:
            raise ValueError(
                "candidates from different perception results cannot be normalized together"
            )


def _candidate_geometry(candidate: RegionCandidate) -> _Geometry | None:
    if isinstance(candidate.mask, InlineMask):
        pixels = frozenset(
            (x, y)
            for y in range(candidate.mask.height)
            for x in range(candidate.mask.width)
            if candidate.mask.value_at(x, y)
        )
        if not pixels:
            return None
        mask_box = _pixels_bounding_box(pixels)
        if candidate.bounding_box is not None and not _contains_box(
            candidate.bounding_box, mask_box
        ):
            return None
        bounding_box = candidate.bounding_box or mask_box
        return _Geometry(candidate, bounding_box, pixels, float(len(pixels)))

    if candidate.bounding_box is None:
        return None
    pixels = _box_pixels(candidate.bounding_box)
    if not pixels:
        return None
    return _Geometry(
        candidate,
        candidate.bounding_box,
        pixels,
        candidate.bounding_box.area,
    )


def _pixels_bounding_box(pixels: frozenset[tuple[int, int]]) -> BoundingBox:
    xs = [x for x, _ in pixels]
    ys = [y for _, y in pixels]
    return BoundingBox(min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def _contains_box(container: BoundingBox, contained: BoundingBox) -> bool:
    return (
        container.x_min <= contained.x_min
        and container.y_min <= contained.y_min
        and container.x_max >= contained.x_max
        and container.y_max >= contained.y_max
    )


def _box_pixels(box: BoundingBox) -> frozenset[tuple[int, int]]:
    return frozenset(
        (x, y)
        for y in range(floor(box.y_min), ceil(box.y_max))
        for x in range(floor(box.x_min), ceil(box.x_max))
        if box.x_min <= x + 0.5 < box.x_max and box.y_min <= y + 0.5 < box.y_max
    )


def _outside_valid_region(
    geometry: _Geometry, prepared_image: PreparedImage, minimum_fraction: float
) -> bool:
    if prepared_image.valid_region is None:
        return False
    foreground = geometry.pixels
    valid_count = sum(prepared_image.valid_region.mask.value_at(x, y) for x, y in foreground)
    return valid_count / len(foreground) < minimum_fraction


def _overlaps_exclusion(
    geometry: _Geometry, prepared_image: PreparedImage, maximum_fraction: float
) -> bool:
    if not prepared_image.exclusion_regions:
        return False
    excluded_count = sum(
        any(region.mask.value_at(x, y) for region in prepared_image.exclusion_regions)
        for x, y in geometry.pixels
    )
    return excluded_count / len(geometry.pixels) > maximum_fraction


def _find_duplicate(
    geometry: _Geometry,
    groups: list[_RegionGroup],
    config: NormalizationConfig,
) -> tuple[_RegionGroup, MergeKind, float, float] | None:
    for group in groups:
        representative = group.representative
        intersection = len(geometry.pixels & representative.pixels)
        union = len(geometry.pixels | representative.pixels)
        iou = intersection / union if union else 0.0
        minimum_area = min(len(geometry.pixels), len(representative.pixels))
        containment = intersection / minimum_area if minimum_area else 0.0
        if iou >= config.duplicate_iou_threshold:
            return group, MergeKind.IOU_DUPLICATE, iou, containment
        if containment >= config.containment_threshold:
            return group, MergeKind.CONTAINMENT, iou, containment
    return None


def _freeze_group(group: _RegionGroup, region_index: int) -> Region2D:
    representative = group.representative
    candidate = representative.candidate
    contributors = tuple(item.candidate_id for item in group.contributors)
    provenance: tuple[RegionProvenance, ...] = tuple(item.provenance for item in group.contributors)
    return Region2D(
        identity=RegionIdentity(
            perception_run_id=candidate.perception_run_id,
            perception_result_id=candidate.perception_result_id,
            region_id=f"region-{region_index:04d}",
        ),
        source_observation_id=candidate.source_observation_id,
        image_width=candidate.image_width,
        image_height=candidate.image_height,
        bounding_box=representative.bounding_box,
        mask=candidate.mask,
        area_pixels=representative.area_pixels,
        contributor_candidate_ids=contributors,
        provenance=provenance,
    )


def _reject(
    candidate: RegionCandidate, reason: RejectionReason, detail: str
) -> RejectedRegionCandidate:
    return RejectedRegionCandidate(
        candidate_id=candidate.candidate_id,
        reason=reason,
        detail=detail,
        discovery_pass_id=candidate.provenance.discovery_pass_id,
    )

"""Grounding-to-mask refinement: grounding proposals used only as segmentation prompts.

A grounded box or point locates *where* a query applies, but a box also covers
background, and projecting every depth sample inside it would weaken 2D→3D association.
Refinement is a separate evidence stage::

    grounding proposal -> segmentation prompt -> refined mask-backed Region2D

Each :class:`RefinementPrompt` keeps the identity of the proposal it came from and the
grounding provenance. The refiner's mask becomes a *new* ``Region2D`` with its own identity
(a function of the proposal and the refiner configuration) and the proposal as its only
contributor; the grounding evidence itself is never touched. The acceptance rule
(:data:`REFINEMENT_ACCEPTANCE_POLICY`) is owned here, not by any backend: an empty mask,
or a mask that does not cover its own prompt, is an explicit rejection and produces no
region, so the proposal is never silently replaced. See ``docs/region-refinement.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from math import floor, isfinite
from types import MappingProxyType
from typing import Any, NewType, cast

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.grounding import (
    GROUNDING_CAPABILITY,
    GroundingGeometry,
    GroundingPoint,
    GroundingRequestId,
    RegionGroundingExecution,
    decode_prepared_image_reference,
    encode_prepared_image_reference,
    validate_grounding_image,
)
from contextmap.visual_perception.identity import (
    grounding_point_id_for,
    grounding_region_id_for,
    refined_region_id_for,
)
from contextmap.visual_perception.models import (
    BackendProvenance,
    BoundingBox2D,
    PerceptionResult,
    PerceptionResultId,
    PreparedImage,
    Region2D,
)
from contextmap.visual_perception.region_models import BackendScore, InlineMask, JsonScalar
from contextmap.visual_perception.serialization import (
    decode_bounding_box,
    decode_provenance,
    decode_region,
    encode_bounding_box,
    encode_provenance,
    encode_region,
)

RefinementRequestId = NewType("RefinementRequestId", str)
"""Content identity of one refinement request."""

REFINEMENT_CAPABILITY = "region_refinement"
"""Capability name every refiner reports in its ``BackendProvenance``."""

REFINEMENT_ACCEPTANCE_POLICY = "refinement-acceptance/1"
"""Version of the rule that accepts or rejects a refiner mask for its prompt.

A mask is accepted when it has at least one foreground pixel and covers its prompt: for a
box, at least one foreground pixel centre lies inside the box; for a point, the pixel that
contains the point is foreground. Pixels are ``PIXEL_XY_TOP_LEFT``: pixel ``(x, y)`` spans
``[x, x + 1) x [y, y + 1)``, and a point on the far image edge belongs to the last pixel.
"""


class RefinementRejectionReason(Enum):
    """Why a refiner mask did not become a refined region."""

    EMPTY_MASK = "empty_mask"
    PROMPT_NOT_COVERED = "prompt_not_covered"


class RefinementRequestError(ValueError):
    """Raised, before any inference, when a refiner cannot serve a request."""


@dataclass(frozen=True, kw_only=True)
class RefinementPrompt:
    """One grounding proposal, used only as a segmentation prompt.

    Attributes:
        proposal_id: Identity of the proposal: the grounded box region's ``RegionId``, or a
            grounded point's identity (:func:`grounding_point_id_for`).
        grounding_request_id: Grounding request that produced the proposal.
        output_index: Position of the proposal in that grounding answer.
        grounding_provenance: The grounding backend that produced the proposal.
        box: The proposal box, in prepared-image pixels, for a box prompt.
        point: The proposal point, in prepared-image pixels, for a point prompt.
    """

    proposal_id: str
    grounding_request_id: GroundingRequestId
    output_index: int
    grounding_provenance: BackendProvenance
    box: BoundingBox2D | None = None
    point: GroundingPoint | None = None

    def __post_init__(self) -> None:
        """Require one geometry and a traceable grounding origin."""
        if not self.proposal_id.strip() or not str(self.grounding_request_id).strip():
            raise ValueError("a refinement prompt needs its proposal and grounding request ids")
        if self.output_index < 0:
            raise ValueError("refinement prompt output_index must be non-negative")
        if (self.box is None) == (self.point is None):
            raise ValueError("a refinement prompt carries exactly one geometry: box or point")
        if self.grounding_provenance.capability != GROUNDING_CAPABILITY:
            raise ValueError(f"a refinement prompt must come from a {GROUNDING_CAPABILITY} backend")

    @property
    def geometry(self) -> GroundingGeometry:
        """Return the geometry family of the prompt."""
        return GroundingGeometry.BOX if self.box is not None else GroundingGeometry.POINT


def refinement_prompts_from(execution: RegionGroundingExecution) -> tuple[RefinementPrompt, ...]:
    """Turn every accepted output of a grounding answer into a prompt, in answer order.

    Rejected spans have no geometry and produce no prompt.
    """
    request = execution.request
    prompts: list[RefinementPrompt] = []
    for output in execution.outputs:
        identity = grounding_region_id_for if output.box is not None else grounding_point_id_for
        prompts.append(
            RefinementPrompt(
                proposal_id=str(
                    identity(
                        result_id=request.perception_result_id,
                        request_id=str(request.request_id),
                        index=output.output_index,
                    )
                ),
                grounding_request_id=request.request_id,
                output_index=output.output_index,
                grounding_provenance=execution.provenance,
                box=output.box,
                point=output.point,
            )
        )
    return tuple(prompts)


@dataclass(frozen=True, kw_only=True)
class RegionRefinementCapabilities:
    """Declare which prompt geometries a refiner accepts."""

    prompt_geometries: frozenset[GroundingGeometry]

    def __post_init__(self) -> None:
        """Require at least one prompt geometry."""
        if not self.prompt_geometries:
            raise ValueError("a refiner must accept at least one prompt geometry")


@dataclass(frozen=True, kw_only=True)
class RegionRefinementRequest:
    """Refine every prompt of one prepared image with one refiner configuration.

    Attributes:
        perception_result_id: Result that owns the refined evidence.
        image: The exact prepared image, content-addressed and unconstrained, as for
            grounding.
        prompts: The proposals to refine, each one a prompt of its own.
        configuration_fingerprint: Fingerprint of the refiner configuration.
    """

    perception_result_id: PerceptionResultId
    image: PreparedImage
    prompts: tuple[RefinementPrompt, ...]
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        """Reject an unowned or unaddressed request and prompts outside the image."""
        if not str(self.perception_result_id).strip():
            raise ValueError("refinement request perception_result_id must not be empty")
        if not self.configuration_fingerprint.strip():
            raise ValueError("refinement request configuration_fingerprint must not be empty")
        validate_grounding_image(self.image, "refinement request")
        if not self.prompts:
            raise ValueError("a refinement request needs at least one prompt")
        proposals = [prompt.proposal_id for prompt in self.prompts]
        if len(set(proposals)) != len(proposals):
            raise ValueError("refinement prompts must be unique proposals")
        for prompt in self.prompts:
            _validate_prompt_inside_image(prompt, self.image.width, self.image.height)

    @property
    def source_observation_id(self) -> SourceObservationId:
        """Return the physical observation the request is about."""
        return self.image.source_observation_id

    @property
    def request_id(self) -> RefinementRequestId:
        """Return the digest of the image, the prompts and the refiner configuration."""
        document = {
            "image": encode_prepared_image_reference(self.image),
            "prompts": [_encode_prompt(prompt) for prompt in self.prompts],
            "configuration_fingerprint": self.configuration_fingerprint,
        }
        canonical = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
        return RefinementRequestId(
            f"refinement-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
        )


def validate_refinement_request(
    request: RegionRefinementRequest, capabilities: RegionRefinementCapabilities
) -> None:
    """Refuse, before inference, a prompt geometry the refiner does not accept.

    Raises:
        RefinementRequestError: Naming every unsupported prompt geometry.
    """
    unsupported = {prompt.geometry for prompt in request.prompts} - capabilities.prompt_geometries
    if unsupported:
        names = sorted(geometry.value for geometry in unsupported)
        raise RefinementRequestError(f"the refiner does not accept {names} prompts")


@dataclass(frozen=True, kw_only=True)
class RefinementOutcome:
    """What the refiner produced for one prompt: a refined region or a rejection.

    Attributes:
        prompt: The prompt answered.
        region: The refined mask-backed region, when accepted. Its only contributor is
            ``prompt.proposal_id``. Before persistence it carries the inline mask; once
            persisted, the mask is referenced (``mask_reference``) like any region's.
        rejection_reason: Why the mask was rejected, when it was.
        rejection_detail: Human-readable detail of the rejection.
        native_scores: Refiner-native scores with their own semantics (e.g. SAM2's
            predicted IoU), never a calibrated probability.
        diagnostics: Acceptance diagnostics, with native names and values.
    """

    prompt: RefinementPrompt
    region: Region2D | None = None
    rejection_reason: RefinementRejectionReason | None = None
    rejection_detail: str | None = None
    native_scores: tuple[BackendScore, ...] = ()
    diagnostics: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Require either an accepted region of this prompt or an explained rejection."""
        if (self.region is None) == (self.rejection_reason is None):
            raise ValueError("a refinement outcome is either a refined region or a rejection")
        if (self.rejection_reason is None) != (self.rejection_detail is None):
            raise ValueError("a refinement rejection needs its detail, and only a rejection")
        if self.rejection_detail is not None and not self.rejection_detail.strip():
            raise ValueError("a refinement rejection detail must not be empty")
        if self.region is not None and self.region.contributor_candidate_ids != (
            self.prompt.proposal_id,
        ):
            raise ValueError("a refined region's only contributor must be its prompt's proposal")
        for name, _value in self.diagnostics:
            if not name:
                raise ValueError("refinement diagnostic names must not be empty")


@dataclass(frozen=True, kw_only=True)
class RefinementDiagnostics:
    """Runtime diagnostics of one refinement call."""

    latency_ms: float
    peak_memory_bytes: int | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject negative measurements."""
        if not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("refinement latency_ms must be finite and non-negative")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("refinement peak_memory_bytes must be non-negative")


@dataclass(frozen=True, kw_only=True)
class RegionRefinementExecution:
    """Auditable record of one refinement call: one outcome per prompt, in prompt order."""

    request: RegionRefinementRequest
    provenance: BackendProvenance
    outcomes: tuple[RefinementOutcome, ...]
    diagnostics: RefinementDiagnostics
    effective_configuration: Mapping[str, JsonScalar]
    runtime_identity: Mapping[str, JsonScalar] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Bind the record to its refiner and answer every prompt exactly once, in order."""
        if self.provenance.capability != REFINEMENT_CAPABILITY:
            raise ValueError(
                f"refinement execution provenance capability must be {REFINEMENT_CAPABILITY}"
            )
        if self.provenance.configuration_fingerprint != self.request.configuration_fingerprint:
            raise ValueError(
                "refinement execution provenance configuration fingerprint does not match "
                "the request it served"
            )
        if tuple(item.prompt for item in self.outcomes) != self.request.prompts:
            raise ValueError("a refinement execution answers every request prompt, in order")
        for outcome in self.outcomes:
            if outcome.region is not None and outcome.region.provenance != self.provenance:
                raise ValueError("a refined region must carry the refiner's provenance")

    @property
    def request_id(self) -> RefinementRequestId:
        """Return the served request's identity."""
        return self.request.request_id

    @property
    def regions(self) -> tuple[Region2D, ...]:
        """Return the accepted refined regions, in prompt order."""
        return tuple(item.region for item in self.outcomes if item.region is not None)

    @property
    def rejected(self) -> tuple[RefinementOutcome, ...]:
        """Return the outcomes whose mask was rejected."""
        return tuple(item for item in self.outcomes if item.region is None)


def refinement_outcome(
    *,
    request: RegionRefinementRequest,
    prompt: RefinementPrompt,
    provenance: BackendProvenance,
    mask: InlineMask,
    native_scores: tuple[BackendScore, ...] = (),
    diagnostics: tuple[tuple[str, JsonScalar], ...] = (),
) -> RefinementOutcome:
    """Apply :data:`REFINEMENT_ACCEPTANCE_POLICY` to one refiner mask.

    Args:
        request: The request the mask answers (image space, owner, observation).
        prompt: The prompt the mask answers.
        provenance: The refiner.
        mask: The refiner mask, in the prepared image's pixel space.
        native_scores: Refiner-native scores, kept verbatim.
        diagnostics: Refiner diagnostics to keep next to the acceptance diagnostics.

    Returns:
        The refined region, or an explicit rejection.

    Raises:
        ValueError: If the mask is not in the prepared image's pixel space: that is a
            refiner contract violation, not a rejectable answer.
    """
    width, height = request.image.width, request.image.height
    if (mask.width, mask.height) != (width, height):
        raise ValueError(
            f"refiner mask dimensions {mask.width}x{mask.height} differ from the prepared "
            f"image {width}x{height}"
        )
    area = mask.area
    acceptance: list[tuple[str, JsonScalar]] = [("mask_area_px", area)]
    if prompt.box is not None:
        inside = _pixels_inside_box(mask, prompt.box)
        acceptance += [
            ("mask_px_inside_prompt_box", inside),
            ("mask_px_outside_prompt_box", area - inside),
        ]
        covered = inside > 0
    else:
        point = cast(GroundingPoint, prompt.point)
        covered = mask.value_at(min(floor(point.x), width - 1), min(floor(point.y), height - 1))
        acceptance.append(("prompt_point_covered", covered))
    all_diagnostics = (*acceptance, *diagnostics)
    if area == 0:
        return RefinementOutcome(
            prompt=prompt,
            rejection_reason=RefinementRejectionReason.EMPTY_MASK,
            rejection_detail="the refiner returned a mask without foreground pixels",
            native_scores=native_scores,
            diagnostics=all_diagnostics,
        )
    if not covered:
        return RefinementOutcome(
            prompt=prompt,
            rejection_reason=RefinementRejectionReason.PROMPT_NOT_COVERED,
            rejection_detail=f"the refiner mask does not cover its {prompt.geometry.value} prompt",
            native_scores=native_scores,
            diagnostics=all_diagnostics,
        )
    return RefinementOutcome(
        prompt=prompt,
        region=Region2D(
            region_id=refined_region_id_for(
                result_id=request.perception_result_id,
                proposal_id=prompt.proposal_id,
                configuration_fingerprint=request.configuration_fingerprint,
            ),
            bounding_box=_tight_box(mask),
            provenance=provenance,
            mask=mask,
            source_observation_id=request.source_observation_id,
            image_width=width,
            image_height=height,
            area_pixels=float(area),
            contributor_candidate_ids=(prompt.proposal_id,),
        ),
        native_scores=native_scores,
        diagnostics=all_diagnostics,
    )


def with_refined_regions(
    result: PerceptionResult, executions: Sequence[RegionRefinementExecution]
) -> PerceptionResult:
    """Return ``result`` with every accepted refined region appended; nothing is replaced.

    Raises:
        ValueError: If an execution belongs to another result or observation.
    """
    refined: list[Region2D] = []
    for execution in executions:
        request = execution.request
        if request.perception_result_id != result.result_id:
            raise ValueError(
                "refinement execution perception_result_id does not match the result: "
                f"{request.perception_result_id!r} != {result.result_id!r}"
            )
        if request.source_observation_id != result.source_observation_id:
            raise ValueError("refinement execution source_observation_id does not match")
        refined.extend(execution.regions)
    if not refined:
        return result
    return replace(result, regions=(*result.regions, *refined))


# --- serialization ---------------------------------------------------------------------


def encode_region_refinement_execution(execution: RegionRefinementExecution) -> dict[str, Any]:
    """Encode an execution; refined masks are never inlined, only ``mask_reference``."""
    request = execution.request
    return {
        "request_id": str(request.request_id),
        "request": {
            "perception_result_id": str(request.perception_result_id),
            "image": encode_prepared_image_reference(request.image),
            "prompts": [_encode_prompt(prompt) for prompt in request.prompts],
            "configuration_fingerprint": request.configuration_fingerprint,
        },
        "provenance": encode_provenance(execution.provenance),
        "acceptance_policy": REFINEMENT_ACCEPTANCE_POLICY,
        "outcomes": [
            {
                "proposal_id": outcome.prompt.proposal_id,
                "region": None if outcome.region is None else encode_region(outcome.region),
                "rejection_reason": (
                    None if outcome.rejection_reason is None else outcome.rejection_reason.value
                ),
                "rejection_detail": outcome.rejection_detail,
                "native_scores": [score.to_dict() for score in outcome.native_scores],
                "diagnostics": [
                    {"name": name, "value": value} for name, value in outcome.diagnostics
                ],
            }
            for outcome in execution.outcomes
        ],
        "diagnostics": {
            "latency_ms": execution.diagnostics.latency_ms,
            "peak_memory_bytes": execution.diagnostics.peak_memory_bytes,
            "warnings": list(execution.diagnostics.warnings),
        },
        "effective_configuration": dict(execution.effective_configuration),
        "runtime_identity": dict(execution.runtime_identity),
    }


def decode_region_refinement_execution(record: Mapping[str, Any]) -> RegionRefinementExecution:
    """Decode an execution and verify its recorded identity and acceptance policy.

    Raises:
        ValueError: If the record was written under another acceptance policy, or its
            ``request_id`` does not match its inputs.
    """
    if record["acceptance_policy"] != REFINEMENT_ACCEPTANCE_POLICY:
        raise ValueError(
            f"refinement record uses acceptance policy {record['acceptance_policy']!r}, "
            f"not {REFINEMENT_ACCEPTANCE_POLICY!r}"
        )
    raw_request = record["request"]
    request = RegionRefinementRequest(
        perception_result_id=PerceptionResultId(raw_request["perception_result_id"]),
        image=decode_prepared_image_reference(raw_request["image"]),
        prompts=tuple(_decode_prompt(item) for item in raw_request["prompts"]),
        configuration_fingerprint=raw_request["configuration_fingerprint"],
    )
    if record["request_id"] != request.request_id:
        raise ValueError("refinement request_id does not match the digest of its inputs")
    raw_diagnostics = record["diagnostics"]
    return RegionRefinementExecution(
        request=request,
        provenance=decode_provenance(record["provenance"]),
        outcomes=tuple(
            RefinementOutcome(
                prompt=prompt,
                region=None if item["region"] is None else decode_region(item["region"]),
                rejection_reason=(
                    None
                    if item["rejection_reason"] is None
                    else RefinementRejectionReason(item["rejection_reason"])
                ),
                rejection_detail=item["rejection_detail"],
                native_scores=tuple(
                    BackendScore.from_dict(score) for score in item["native_scores"]
                ),
                diagnostics=tuple(
                    (entry["name"], cast(JsonScalar, entry["value"]))
                    for entry in item["diagnostics"]
                ),
            )
            for prompt, item in zip(request.prompts, record["outcomes"], strict=True)
        ),
        diagnostics=RefinementDiagnostics(
            latency_ms=raw_diagnostics["latency_ms"],
            peak_memory_bytes=raw_diagnostics["peak_memory_bytes"],
            warnings=tuple(raw_diagnostics["warnings"]),
        ),
        effective_configuration=MappingProxyType(dict(record["effective_configuration"])),
        runtime_identity=MappingProxyType(dict(record["runtime_identity"])),
    )


def _encode_prompt(prompt: RefinementPrompt) -> dict[str, Any]:
    return {
        "proposal_id": prompt.proposal_id,
        "grounding_request_id": str(prompt.grounding_request_id),
        "output_index": prompt.output_index,
        "grounding_provenance": encode_provenance(prompt.grounding_provenance),
        "box": None if prompt.box is None else encode_bounding_box(prompt.box),
        "point": None if prompt.point is None else {"x": prompt.point.x, "y": prompt.point.y},
    }


def _decode_prompt(record: Mapping[str, Any]) -> RefinementPrompt:
    raw_point = record["point"]
    return RefinementPrompt(
        proposal_id=record["proposal_id"],
        grounding_request_id=GroundingRequestId(record["grounding_request_id"]),
        output_index=record["output_index"],
        grounding_provenance=decode_provenance(record["grounding_provenance"]),
        box=None if record["box"] is None else decode_bounding_box(record["box"]),
        point=None if raw_point is None else GroundingPoint(x=raw_point["x"], y=raw_point["y"]),
    )


def _validate_prompt_inside_image(prompt: RefinementPrompt, width: int, height: int) -> None:
    if prompt.box is not None:
        box = prompt.box
        inside = box.x_min < width and box.y_min < height
        inside = inside and box.x_max <= width + 1e-6 and box.y_max <= height + 1e-6
    else:
        point = cast(GroundingPoint, prompt.point)
        inside = point.x <= width and point.y <= height
    if not inside:
        raise ValueError(
            f"refinement prompt {prompt.proposal_id!r} lies outside the {width}x{height} image"
        )


def _pixels_inside_box(mask: InlineMask, box: BoundingBox2D) -> int:
    """Count foreground pixels whose centre lies inside the half-open prompt box."""
    count = 0
    for y in range(mask.height):
        if not box.y_min <= y + 0.5 < box.y_max:
            continue
        row = mask.data[y * mask.width : (y + 1) * mask.width]
        count += sum(1 for x, value in enumerate(row) if value and box.x_min <= x + 0.5 < box.x_max)
    return count


def _tight_box(mask: InlineMask) -> BoundingBox2D:
    """Return the half-open pixel box of the foreground of a non-empty mask."""
    columns: list[int] = []
    rows: list[int] = []
    for y in range(mask.height):
        row = mask.data[y * mask.width : (y + 1) * mask.width]
        xs = [x for x, value in enumerate(row) if value]
        if xs:
            rows.append(y)
            columns.extend((xs[0], xs[-1]))
    x_min, x_max = min(columns), max(columns) + 1
    y_min, y_max = rows[0], rows[-1] + 1
    return BoundingBox2D(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

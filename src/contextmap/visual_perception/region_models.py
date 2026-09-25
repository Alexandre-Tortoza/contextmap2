"""Backend-neutral contracts for two-dimensional region evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None


class CoordinateConvention(StrEnum):
    """Coordinate conventions supported by canonical visual geometry."""

    PIXEL_XY_TOP_LEFT = "pixel_xy_top_left"


class RejectionReason(StrEnum):
    """Backend-independent reasons for rejecting a region proposal."""

    INVALID_GEOMETRY = "invalid_geometry"
    OUTSIDE_VALID_REGION = "outside_valid_region"
    EXCLUSION_OVERLAP = "exclusion_overlap"
    AREA_BELOW_MINIMUM = "area_below_minimum"
    AREA_ABOVE_MAXIMUM = "area_above_maximum"
    TILE_BORDER_TRUNCATION = "tile_border_truncation"
    MERGED_DUPLICATE = "merged_duplicate"
    REGION_BUDGET_EXCEEDED = "region_budget_exceeded"


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Reference a persisted payload without embedding it in a public contract."""

    uri: str
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        """Validate the content-addressed reference."""
        if not self.uri:
            raise ValueError("artifact uri must not be empty")
        invalid_character = any(character not in "0123456789abcdef" for character in self.sha256)
        if len(self.sha256) != 64 or invalid_character:
            raise ValueError("artifact sha256 must contain 64 lowercase hexadecimal characters")
        if not self.media_type:
            raise ValueError("artifact media_type must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {"uri": self.uri, "sha256": self.sha256, "media_type": self.media_type}

    @classmethod
    def from_dict(cls, value: object) -> ArtifactReference:
        """Restore an artifact reference from serialized data."""
        data = _mapping(value, "artifact reference")
        return cls(
            uri=_string(data, "uri"),
            sha256=_string(data, "sha256"),
            media_type=_string(data, "media_type"),
        )


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Half-open pixel-space box using ``[x_min, x_max) x [y_min, y_max)``."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        """Reject non-finite and empty boxes."""
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(isfinite(value) for value in values):
            raise ValueError("bounding box coordinates must be finite")
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError("bounding box minimum coordinates must be non-negative")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must have positive width and height")

    @property
    def width(self) -> float:
        """Return box width in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Return box height in pixels."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Return box area in square pixels."""
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-compatible representation."""
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    @classmethod
    def from_dict(cls, value: object) -> BoundingBox:
        """Restore a bounding box from serialized data."""
        data = _mapping(value, "bounding box")
        return cls(
            x_min=_number(data, "x_min"),
            y_min=_number(data, "y_min"),
            x_max=_number(data, "x_max"),
            y_max=_number(data, "y_max"),
        )


@dataclass(frozen=True, slots=True)
class InlineMask:
    """Small serialization-friendly row-major binary mask."""

    width: int
    height: int
    data: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate mask shape and values."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("mask dimensions must be positive")
        if len(self.data) != self.width * self.height:
            raise ValueError("mask data length must equal width multiplied by height")
        if any(type(value) is not bool for value in self.data):
            raise TypeError("mask data must contain bool values")

    @property
    def area(self) -> int:
        """Return the number of foreground pixels."""
        return sum(self.data)

    def value_at(self, x: int, y: int) -> bool:
        """Return whether a pixel is foreground."""
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError("mask coordinate outside image bounds")
        return self.data[y * self.width + x]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "storage": "inline",
            "width": self.width,
            "height": self.height,
            "data": [int(value) for value in self.data],
        }

    @classmethod
    def from_dict(cls, value: object) -> InlineMask:
        """Restore an inline mask from serialized data."""
        data = _mapping(value, "inline mask")
        raw_values = data.get("data")
        invalid_values = isinstance(raw_values, list) and any(
            item not in (0, 1, False, True) for item in raw_values
        )
        if not isinstance(raw_values, list) or invalid_values:
            raise TypeError("inline mask data must be a list of binary values")
        return cls(
            width=_integer(data, "width"),
            height=_integer(data, "height"),
            data=tuple(bool(item) for item in raw_values),
        )


MaskGeometry: TypeAlias = InlineMask


@dataclass(frozen=True, slots=True)
class BackendScore:
    """A backend-native score with its own named semantics."""

    name: str
    value: float
    semantics: str

    def __post_init__(self) -> None:
        """Validate that score meaning is explicit."""
        if not self.name or not self.semantics:
            raise ValueError("backend score name and semantics must not be empty")
        if not isfinite(self.value):
            raise ValueError("backend score value must be finite")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {"name": self.name, "value": self.value, "semantics": self.semantics}

    @classmethod
    def from_dict(cls, value: object) -> BackendScore:
        """Restore a backend-native score from serialized data."""
        data = _mapping(value, "backend score")
        return cls(
            name=_string(data, "name"),
            value=_number(data, "value"),
            semantics=_string(data, "semantics"),
        )


@dataclass(frozen=True, slots=True)
class RegionProvenance:
    """Identify the backend proposal and effective discovery configuration."""

    backend_id: str
    backend_version: str
    checkpoint: str
    config_digest: str
    discovery_pass_id: str
    native_proposal_id: str
    query: str | None = None

    def __post_init__(self) -> None:
        """Require every reproducibility-bearing identity."""
        required = (
            self.backend_id,
            self.backend_version,
            self.checkpoint,
            self.config_digest,
            self.discovery_pass_id,
            self.native_proposal_id,
        )
        if any(not value for value in required):
            raise ValueError("region provenance identity fields must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "checkpoint": self.checkpoint,
            "config_digest": self.config_digest,
            "discovery_pass_id": self.discovery_pass_id,
            "native_proposal_id": self.native_proposal_id,
            "query": self.query,
        }

    @classmethod
    def from_dict(cls, value: object) -> RegionProvenance:
        """Restore provenance from serialized data."""
        data = _mapping(value, "region provenance")
        query = data.get("query")
        if query is not None and not isinstance(query, str):
            raise TypeError("region provenance query must be a string or null")
        return cls(
            backend_id=_string(data, "backend_id"),
            backend_version=_string(data, "backend_version"),
            checkpoint=_string(data, "checkpoint"),
            config_digest=_string(data, "config_digest"),
            discovery_pass_id=_string(data, "discovery_pass_id"),
            native_proposal_id=_string(data, "native_proposal_id"),
            query=query,
        )


@dataclass(frozen=True, slots=True)
class NativeRegionText:
    """Text a region-producing task emitted natively together with one proposal.

    This is evidence of what one backend inference returned next to one geometry, for
    example a Florence-2 ``<OD>`` category or a ``<DENSE_REGION_CAPTION>`` description. It
    is never a label, a ``SemanticClaim`` or a belief, carries no confidence, and does not
    enter ``Region2D``. A proposal whose parser returned no text carries no instance.

    Attributes:
        task: Backend task identity that produced both the geometry and the text.
        text: Verbatim text the backend parser attached to this proposal.
        prompt: Task input that conditioned the output (for example an open-vocabulary
            query), or ``None`` when the task takes no input.
    """

    task: str
    text: str
    prompt: str | None = None

    def __post_init__(self) -> None:
        """Require a task identity and text that actually says something."""
        if not self.task.strip():
            raise ValueError("native region text task must not be empty")
        if not self.text.strip():
            raise ValueError(
                "native region text must contain non-whitespace text; a proposal without "
                "text carries no NativeRegionText"
            )
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("native region text prompt must be None or non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {"task": self.task, "text": self.text, "prompt": self.prompt}

    @classmethod
    def from_dict(cls, value: object) -> NativeRegionText:
        """Restore native proposal text from serialized data."""
        data = _mapping(value, "native region text")
        prompt = data.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise TypeError("native region text prompt must be a string or null")
        return cls(task=_string(data, "task"), text=_string(data, "text"), prompt=prompt)


@dataclass(frozen=True, slots=True)
class RegionCandidate:
    """A backend proposal before canonical filtering, merge, and geometry freeze.

    ``native_text`` is the text the backend task emitted with this exact proposal, kept
    typed so it follows the candidate through pass remapping; normalization never copies
    it into ``Region2D``.
    """

    candidate_id: str
    source_observation_id: str
    perception_run_id: str
    perception_result_id: str
    image_width: int
    image_height: int
    provenance: RegionProvenance
    bounding_box: BoundingBox | None = None
    mask: MaskGeometry | None = None
    mask_reference: ArtifactReference | None = None
    score: BackendScore | None = None
    native_metadata: tuple[tuple[str, JsonScalar], ...] = ()
    native_text: NativeRegionText | None = None
    coordinate_convention: CoordinateConvention = CoordinateConvention.PIXEL_XY_TOP_LEFT

    def __post_init__(self) -> None:
        """Validate geometry, image space, identity, and inspectable metadata."""
        identities = (
            self.candidate_id,
            self.source_observation_id,
            self.perception_run_id,
            self.perception_result_id,
        )
        if any(not identity for identity in identities):
            raise ValueError("candidate identity fields must not be empty")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("candidate image dimensions must be positive")
        if self.bounding_box is None and self.mask is None:
            raise ValueError("candidate must contain at least one geometry representation")
        if self.mask is not None and not isinstance(self.mask, InlineMask):
            raise TypeError("candidate mask must be materialized as InlineMask")
        if self.mask_reference is not None and self.mask is None:
            raise ValueError("candidate mask_reference requires a materialized mask")
        _validate_geometry_bounds(
            bounding_box=self.bounding_box,
            mask=self.mask,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        for key, value in self.native_metadata:
            if not key:
                raise ValueError("native metadata keys must not be empty")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError("native metadata values must be JSON scalar values")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation without native SDK objects."""
        return {
            "candidate_id": self.candidate_id,
            "source_observation_id": self.source_observation_id,
            "perception_run_id": self.perception_run_id,
            "perception_result_id": self.perception_result_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "coordinate_convention": self.coordinate_convention.value,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "mask": _mask_to_dict(self.mask),
            "mask_reference": (
                None if self.mask_reference is None else self.mask_reference.to_dict()
            ),
            "score": self.score.to_dict() if self.score else None,
            "provenance": self.provenance.to_dict(),
            "native_metadata": [
                {"name": name, "value": value} for name, value in self.native_metadata
            ],
            "native_text": None if self.native_text is None else self.native_text.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> RegionCandidate:
        """Restore a canonical candidate from serialized data."""
        data = _mapping(value, "region candidate")
        raw_metadata = data.get("native_metadata", [])
        if not isinstance(raw_metadata, list):
            raise TypeError("native_metadata must be a list")
        metadata: list[tuple[str, JsonScalar]] = []
        for item in raw_metadata:
            entry = _mapping(item, "native metadata entry")
            scalar = entry.get("value")
            if not isinstance(scalar, (str, int, float, bool, type(None))):
                raise TypeError("native metadata values must be JSON scalar values")
            metadata.append((_string(entry, "name"), cast(JsonScalar, scalar)))
        raw_box = data.get("bounding_box")
        raw_score = data.get("score")
        raw_mask_reference = data.get("mask_reference")
        raw_native_text = data.get("native_text")
        return cls(
            candidate_id=_string(data, "candidate_id"),
            source_observation_id=_string(data, "source_observation_id"),
            perception_run_id=_string(data, "perception_run_id"),
            perception_result_id=_string(data, "perception_result_id"),
            image_width=_integer(data, "image_width"),
            image_height=_integer(data, "image_height"),
            coordinate_convention=CoordinateConvention(_string(data, "coordinate_convention")),
            bounding_box=BoundingBox.from_dict(raw_box) if raw_box is not None else None,
            mask=_mask_from_dict(data.get("mask")),
            mask_reference=(
                ArtifactReference.from_dict(raw_mask_reference)
                if raw_mask_reference is not None
                else None
            ),
            score=BackendScore.from_dict(raw_score) if raw_score is not None else None,
            provenance=RegionProvenance.from_dict(data.get("provenance")),
            native_metadata=tuple(metadata),
            native_text=(
                None if raw_native_text is None else NativeRegionText.from_dict(raw_native_text)
            ),
        )


@dataclass(frozen=True, slots=True)
class RejectedRegionCandidate:
    """Machine-readable record of a proposal rejected before geometry freeze."""

    candidate_id: str
    reason: RejectionReason
    detail: str
    discovery_pass_id: str

    def __post_init__(self) -> None:
        """Require a traceable candidate, pass, and diagnostic detail."""
        if not self.candidate_id or not self.detail or not self.discovery_pass_id:
            raise ValueError("rejected candidate identity and detail must not be empty")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible rejection record."""
        return {
            "candidate_id": self.candidate_id,
            "reason": self.reason.value,
            "detail": self.detail,
            "discovery_pass_id": self.discovery_pass_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> RejectedRegionCandidate:
        """Restore a rejection record from serialized data."""
        data = _mapping(value, "rejected candidate")
        return cls(
            candidate_id=_string(data, "candidate_id"),
            reason=RejectionReason(_string(data, "reason")),
            detail=_string(data, "detail"),
            discovery_pass_id=_string(data, "discovery_pass_id"),
        )


def _validate_geometry_bounds(
    *,
    bounding_box: BoundingBox | None,
    mask: MaskGeometry | None,
    image_width: int,
    image_height: int,
) -> None:
    if bounding_box is not None and (
        bounding_box.x_max > image_width or bounding_box.y_max > image_height
    ):
        raise ValueError("bounding box extends outside image bounds")
    if isinstance(mask, InlineMask) and (mask.width != image_width or mask.height != image_height):
        raise ValueError("inline mask dimensions must match image bounds")


def _mask_to_dict(mask: MaskGeometry | None) -> dict[str, object] | None:
    if mask is None:
        return None
    return mask.to_dict()


def _mask_from_dict(value: object) -> MaskGeometry | None:
    if value is None:
        return None
    data = _mapping(value, "mask")
    storage = _string(data, "storage")
    if storage == "inline":
        return InlineMask.from_dict(data)
    raise ValueError(f"unsupported mask storage: {storage}")


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return cast(dict[str, object], value)


def _string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _integer(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if type(value) is not int:
        raise TypeError(f"{key} must be an integer")
    return cast(int, value)


def _number(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{key} must be a number")
    return float(value)

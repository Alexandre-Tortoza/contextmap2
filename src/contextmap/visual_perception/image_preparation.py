"""Auditable image preparation and optional spatial constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .region_models import ArtifactReference, BoundingBox, InlineMask, JsonScalar


@dataclass(frozen=True, slots=True)
class SourceImage:
    """Immutable image reference extracted from one physical observation."""

    source_observation_id: str
    image: ArtifactReference
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate source identity and image dimensions."""
        if not self.source_observation_id:
            raise ValueError("source_observation_id must not be empty")
        _validate_dimensions(self.width, self.height)


@dataclass(frozen=True, slots=True)
class ValidRegion:
    """Explicitly identify pixels eligible for region discovery."""

    mask: InlineMask
    reason: str
    source: str

    def __post_init__(self) -> None:
        """Require auditable constraint motivation and source."""
        if not self.reason or not self.source:
            raise ValueError("valid region reason and source must not be empty")


@dataclass(frozen=True, slots=True)
class ExclusionRegion:
    """Explicitly identify pixels excluded from region discovery."""

    name: str
    mask: InlineMask
    reason: str
    source: str

    def __post_init__(self) -> None:
        """Require a stable diagnostic name, motivation, and source."""
        if not self.name or not self.reason or not self.source:
            raise ValueError("exclusion name, reason, and source must not be empty")


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    """Describe one applied image transformation in execution order."""

    operation: str
    parameters: tuple[tuple[str, JsonScalar], ...]
    input_dimensions: tuple[int, int]
    output_dimensions: tuple[int, int]
    output_image: ArtifactReference
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the structured transformation record."""
        if not self.operation:
            raise ValueError("transformation operation must not be empty")
        _validate_dimensions(*self.input_dimensions)
        _validate_dimensions(*self.output_dimensions)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible audit record."""
        return {
            "operation": self.operation,
            "parameters": [{"name": name, "value": value} for name, value in self.parameters],
            "input_dimensions": list(self.input_dimensions),
            "output_dimensions": list(self.output_dimensions),
            "output_image": self.output_image.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Canonical image-space input presented to discovery backends."""

    source_observation_id: str
    image: ArtifactReference
    width: int
    height: int
    transformations: tuple[TransformationRecord, ...]
    valid_region: ValidRegion | None = None
    exclusion_regions: tuple[ExclusionRegion, ...] = ()

    def __post_init__(self) -> None:
        """Validate final image space and optional constraints."""
        if not self.source_observation_id:
            raise ValueError("source_observation_id must not be empty")
        _validate_dimensions(self.width, self.height)
        _validate_constraints(
            width=self.width,
            height=self.height,
            valid_region=self.valid_region,
            exclusion_regions=self.exclusion_regions,
        )

    def to_dict(self) -> dict[str, object]:
        """Return an inspectable representation for manifests and diagnostics."""
        return {
            "source_observation_id": self.source_observation_id,
            "image": self.image.to_dict(),
            "width": self.width,
            "height": self.height,
            "transformations": [record.to_dict() for record in self.transformations],
            "valid_region": _valid_region_to_dict(self.valid_region),
            "exclusion_regions": [
                _exclusion_region_to_dict(region) for region in self.exclusion_regions
            ],
        }


@dataclass(frozen=True, slots=True)
class ResizeOperation:
    """Resize an image to explicit output dimensions."""

    width: int
    height: int
    output_image: ArtifactReference

    def __post_init__(self) -> None:
        """Validate requested output dimensions."""
        _validate_dimensions(self.width, self.height)


@dataclass(frozen=True, slots=True)
class CropOperation:
    """Crop an image using the current prepared-image coordinate space."""

    box: BoundingBox
    output_image: ArtifactReference


@dataclass(frozen=True, slots=True)
class RectifyOperation:
    """Record externally materialized image rectification."""

    calibration_id: str
    output_image: ArtifactReference

    def __post_init__(self) -> None:
        """Require the calibration identity used for rectification."""
        if not self.calibration_id:
            raise ValueError("rectification calibration_id must not be empty")


@dataclass(frozen=True, slots=True)
class NormalizeOperation:
    """Record externally materialized model-input normalization."""

    method: str
    output_image: ArtifactReference

    def __post_init__(self) -> None:
        """Require an explicit normalization method."""
        if not self.method:
            raise ValueError("normalization method must not be empty")


PreparationOperation: TypeAlias = (
    ResizeOperation | CropOperation | RectifyOperation | NormalizeOperation
)


def prepare_image(
    source: SourceImage,
    *,
    operations: tuple[PreparationOperation, ...] = (),
    valid_region: ValidRegion | None = None,
    exclusion_regions: tuple[ExclusionRegion, ...] = (),
) -> PreparedImage:
    """Apply an explicit preparation plan without mutating the source observation.

    Pixel-rewriting operations carry the immutable output payload reference produced
    by the image codec/transform adapter. This domain function owns coordinate-space
    validation and provenance rather than image-library-specific processing.

    Args:
        source: Immutable source image reference.
        operations: Ordered, explicitly configured preparation operations.
        valid_region: Optional eligibility mask in the final image space.
        exclusion_regions: Optional named exclusion masks in the final image space.

    Returns:
        Prepared image with ordered transformation provenance.

    Raises:
        ValueError: If an operation or constraint is incompatible with its image space.
    """
    width = source.width
    height = source.height
    image = source.image
    records: list[TransformationRecord] = []

    for operation in operations:
        input_dimensions = (width, height)
        if isinstance(operation, ResizeOperation):
            width, height = operation.width, operation.height
            parameters: tuple[tuple[str, JsonScalar], ...] = (
                ("width", width),
                ("height", height),
            )
            operation_name = "resize"
        elif isinstance(operation, CropOperation):
            if operation.box.x_max > width or operation.box.y_max > height:
                raise ValueError("crop extends outside current image bounds")
            width = _integer_extent(operation.box.width, "crop width")
            height = _integer_extent(operation.box.height, "crop height")
            parameters = (
                ("x_min", operation.box.x_min),
                ("y_min", operation.box.y_min),
                ("x_max", operation.box.x_max),
                ("y_max", operation.box.y_max),
            )
            operation_name = "crop"
        elif isinstance(operation, RectifyOperation):
            parameters = (("calibration_id", operation.calibration_id),)
            operation_name = "rectify"
        else:
            parameters = (("method", operation.method),)
            operation_name = "normalize"

        image = operation.output_image
        records.append(
            TransformationRecord(
                operation=operation_name,
                parameters=parameters,
                input_dimensions=input_dimensions,
                output_dimensions=(width, height),
                output_image=image,
            )
        )

    return PreparedImage(
        source_observation_id=source.source_observation_id,
        image=image,
        width=width,
        height=height,
        transformations=tuple(records),
        valid_region=valid_region,
        exclusion_regions=exclusion_regions,
    )


def _integer_extent(value: float, name: str) -> int:
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer number of pixels")
    return int(value)


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")


def _validate_constraints(
    *,
    width: int,
    height: int,
    valid_region: ValidRegion | None,
    exclusion_regions: tuple[ExclusionRegion, ...],
) -> None:
    constraints = (() if valid_region is None else (valid_region.mask,)) + tuple(
        region.mask for region in exclusion_regions
    )
    if any(mask.width != width or mask.height != height for mask in constraints):
        raise ValueError("spatial constraints must match final prepared image dimensions")
    names = [region.name for region in exclusion_regions]
    if len(set(names)) != len(names):
        raise ValueError("exclusion regions must have unique names")


def _valid_region_to_dict(region: ValidRegion | None) -> dict[str, object] | None:
    if region is None:
        return None
    return {"mask": region.mask.to_dict(), "reason": region.reason, "source": region.source}


def _exclusion_region_to_dict(region: ExclusionRegion) -> dict[str, object]:
    return {
        "name": region.name,
        "mask": region.mask.to_dict(),
        "reason": region.reason,
        "source": region.source,
    }

"""Auditable image preparation and optional spatial constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from contextmap.ingestion import SourceObservationId

from .models import ExclusionRegion, PreparedImage, TransformationRecord, ValidRegion
from .region_models import ArtifactReference, BoundingBox, JsonScalar


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
class ResizeOperation:
    """Resize an image to explicit output dimensions."""

    width: int
    height: int
    output_image: ArtifactReference
    provenance_source: str

    def __post_init__(self) -> None:
        """Validate requested output dimensions."""
        _validate_dimensions(self.width, self.height)
        _validate_operation_source(self.provenance_source)


@dataclass(frozen=True, slots=True)
class CropOperation:
    """Crop an image using the current prepared-image coordinate space."""

    box: BoundingBox
    output_image: ArtifactReference
    provenance_source: str

    def __post_init__(self) -> None:
        """Require the configuration or policy that selected the crop."""
        _validate_operation_source(self.provenance_source)


@dataclass(frozen=True, slots=True)
class RectifyOperation:
    """Record externally materialized image rectification."""

    calibration_id: str
    output_image: ArtifactReference
    provenance_source: str

    def __post_init__(self) -> None:
        """Require the calibration identity used for rectification."""
        if not self.calibration_id:
            raise ValueError("rectification calibration_id must not be empty")
        _validate_operation_source(self.provenance_source)


@dataclass(frozen=True, slots=True)
class NormalizeOperation:
    """Record externally materialized model-input normalization."""

    method: str
    output_image: ArtifactReference
    provenance_source: str

    def __post_init__(self) -> None:
        """Require an explicit normalization method."""
        if not self.method:
            raise ValueError("normalization method must not be empty")
        _validate_operation_source(self.provenance_source)


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
                provenance_source=operation.provenance_source,
                parameters=parameters,
                input_dimensions=input_dimensions,
                output_dimensions=(width, height),
                output_image=image,
            )
        )

    return PreparedImage(
        source_observation_id=SourceObservationId(source.source_observation_id),
        payload_reference=image.uri,
        payload_artifact=image,
        width=width,
        height=height,
        transformations=tuple(records),
        valid_region=valid_region,
        exclusion_regions=exclusion_regions,
    )


def _integer_extent(value: float, name: str) -> int:
    if not float(value).is_integer():
        raise ValueError(f"{name} must be an integer number of pixels")
    return int(value)


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")


def _validate_operation_source(value: str) -> None:
    if not value:
        raise ValueError("transformation provenance source must not be empty")

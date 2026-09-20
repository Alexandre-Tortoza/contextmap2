"""Raw-image to prepared-image pixel mapping, from the recorded preparation chain.

Visual Perception prepares an image (crop, resize, ...) and records every step in
the :class:`~contextmap.visual_perception.PreparedImage`. Regions and masks live in
the *prepared* image, while a camera model projects to *raw* pixels, so
association must map one to the other with the exact recorded steps. It never
assumes the two spaces are identical and never infers a space from image
dimensions alone: the chain must start at the raw image the calibration describes,
be contiguous, and end at the prepared image.

Steps are applied about pixel *edges*: a pixel centered at integer ``c`` covers
``[c - 0.5, c + 0.5)``, so its edge coordinate is ``c + 0.5``. A crop subtracts its
origin and a resize multiplies by the size ratio in that space, which is how a
half-size image is really sampled, and the result is converted back to a pixel
center.

A rectification is rejected: the record keeps only the calibration id, not the
remap, so the raw pixel cannot be carried into the rectified image without
recreating preprocessing parameters that were not recorded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.sensor_association.errors import AssociationInputError
from contextmap.visual_perception import PreparedImage, TransformationRecord

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SUPPORTED_OPERATIONS = ("crop", "resize", "normalize")


@dataclass(frozen=True, kw_only=True)
class ImageTransformStep:
    """One recorded preparation step, reduced to its effect on pixel coordinates.

    In edge coordinates a step maps ``e_out = (e_in - offset_px) * scale`` per axis.

    Attributes:
        operation: ``"crop"``, ``"resize"`` or ``"normalize"``.
        input_size: ``(width, height)`` the step consumes, in pixels.
        output_size: ``(width, height)`` the step produces, in pixels.
        offset_px: Origin removed from the input, in input pixels; ``(0, 0)`` unless a crop.
        scale: Factor applied per axis after the offset; ``(1, 1)`` unless a resize.
    """

    operation: str
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    offset_px: tuple[float, float]
    scale: tuple[float, float]


@dataclass(frozen=True, kw_only=True)
class RawToPreparedTransform:
    """The exact map from raw-image pixels to prepared-image pixels.

    Attributes:
        raw_size: ``(width, height)`` of the raw image, in pixels.
        prepared_size: ``(width, height)`` of the prepared image, in pixels.
        steps: The recorded steps in execution order; empty when nothing was done.
        transform_id: ``"sha256:<hex>"`` of the raw size, the steps and the prepared
            size, so equal chains share an identity.
    """

    raw_size: tuple[int, int]
    prepared_size: tuple[int, int]
    steps: tuple[ImageTransformStep, ...]
    transform_id: str

    def map_pixels(self, raw_pixels: NDArray[Any]) -> NDArray[Any]:
        """Carry raw-image pixel centers into the prepared image.

        Args:
            raw_pixels: ``(N, 2)`` continuous ``(u, v)``, pixel center at the integer;
                ``NaN`` stays ``NaN``.

        Returns:
            ``(N, 2)`` prepared-image pixels in the same convention. They are not
            clipped: a pixel outside the prepared image keeps its outside coordinates.
        """
        import numpy as np

        edges = np.asarray(raw_pixels, dtype=np.float64) + 0.5
        for step in self.steps:
            edges = (edges - np.array(step.offset_px)) * np.array(step.scale)
        prepared: NDArray[Any] = edges - 0.5
        return prepared

    def in_prepared_image(self, prepared_pixels: NDArray[Any]) -> NDArray[Any]:
        """Tell which prepared pixels fall inside ``[-0.5, size - 0.5)`` on both axes.

        Args:
            prepared_pixels: ``(N, 2)`` prepared-image pixels; ``NaN`` is never inside.

        Returns:
            ``(N,)`` booleans.
        """
        import numpy as np

        pixels = np.asarray(prepared_pixels, dtype=np.float64)
        width, height = self.prepared_size
        u, v = pixels[:, 0], pixels[:, 1]
        inside: NDArray[Any] = (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
        return inside


def raw_to_prepared_transform(
    *, raw_size: tuple[int, int], prepared_image: PreparedImage
) -> RawToPreparedTransform:
    """Reproduce the coordinate effect of a prepared image's recorded chain.

    Args:
        raw_size: ``(width, height)`` of the raw image the calibration describes.
        prepared_image: The prepared image and its ordered transformation records.

    Returns:
        The transform from raw pixels to prepared pixels.

    Raises:
        AssociationInputError: If the chain does not start at the raw image, is not
            contiguous, does not end at the prepared image, holds a record that
            disagrees with its own parameters, or uses an operation that cannot be
            reproduced (a rectification, or an unknown one).
    """
    steps: list[ImageTransformStep] = []
    current = raw_size
    for index, record in enumerate(prepared_image.transformations):
        if record.input_dimensions != current:
            where = "the raw image" if index == 0 else f"step {index - 1}"
            raise AssociationInputError(
                f"the image transform chain is not contiguous: {record.operation!r} starts from "
                f"{record.input_dimensions} but {where} leaves {current}"
                + (" (the raw image the calibration describes)" if index == 0 else "")
            )
        step = _step_for(record)
        steps.append(step)
        current = step.output_size
    prepared_size = (prepared_image.width, prepared_image.height)
    if current != prepared_size:
        raise AssociationInputError(
            f"the image transform chain ends at {current} but the prepared image is {prepared_size}"
        )
    return RawToPreparedTransform(
        raw_size=raw_size,
        prepared_size=prepared_size,
        steps=tuple(steps),
        transform_id=_transform_id(raw_size, tuple(steps), prepared_size),
    )


def _step_for(record: TransformationRecord) -> ImageTransformStep:
    operation = record.operation
    input_size, output_size = record.input_dimensions, record.output_dimensions
    parameters = dict(record.parameters)
    if operation == "resize":
        return ImageTransformStep(
            operation=operation,
            input_size=input_size,
            output_size=output_size,
            offset_px=(0.0, 0.0),
            scale=(output_size[0] / input_size[0], output_size[1] / input_size[1]),
        )
    if operation == "crop":
        x_min, y_min = _number(parameters, "x_min"), _number(parameters, "y_min")
        x_max, y_max = _number(parameters, "x_max"), _number(parameters, "y_max")
        if (x_max - x_min, y_max - y_min) != (float(output_size[0]), float(output_size[1])):
            raise AssociationInputError(
                f"the crop box ({x_min}, {y_min})-({x_max}, {y_max}) does not produce the "
                f"recorded output size {output_size}"
            )
        return ImageTransformStep(
            operation=operation,
            input_size=input_size,
            output_size=output_size,
            offset_px=(x_min, y_min),
            scale=(1.0, 1.0),
        )
    if operation == "normalize":
        if input_size != output_size:
            raise AssociationInputError(
                f"a normalization must not change the image size, got {input_size} -> {output_size}"
            )
        return ImageTransformStep(
            operation=operation,
            input_size=input_size,
            output_size=output_size,
            offset_px=(0.0, 0.0),
            scale=(1.0, 1.0),
        )
    if operation == "rectify":
        raise AssociationInputError(
            "a rectification cannot be reproduced: the record keeps only the calibration id, "
            "not the remap or the rectified camera model, so raw pixels cannot be carried "
            "into the rectified image"
        )
    raise AssociationInputError(
        f"unsupported image transformation {operation!r}; supported: {_SUPPORTED_OPERATIONS}"
    )


def _number(parameters: dict[str, Any], name: str) -> float:
    value = parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AssociationInputError(f"the crop record has no numeric {name!r} parameter")
    return float(value)


def _transform_id(
    raw_size: tuple[int, int],
    steps: tuple[ImageTransformStep, ...],
    prepared_size: tuple[int, int],
) -> str:
    payload = {
        "raw_size": list(raw_size),
        "steps": [
            {
                "operation": step.operation,
                "input_size": list(step.input_size),
                "output_size": list(step.output_size),
                "offset_px": list(step.offset_px),
                "scale": list(step.scale),
            }
            for step in steps
        ],
        "prepared_size": list(prepared_size),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"

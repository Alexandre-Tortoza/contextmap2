"""Shared revision validation and image preprocessing for Hugging Face image runtimes."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_COMMIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def validate_huggingface_commit_revision(revision: str) -> None:
    """Require an immutable, full lowercase Git commit SHA."""
    if _COMMIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("revision must be a full 40-character lowercase hexadecimal commit SHA")


def preprocess_pixel_values(
    *,
    processor: Any,
    images: Sequence[Any],
    width: int,
    height: int,
    resample: Any,
) -> Any:
    """Resize with Pillow and leave only rescaling and normalization to the processor.

    The resize is owned by the adapter because the Hugging Face processor resizes
    with an implementation that depends on the installed backend (torchvision or
    PIL) and gives different pixels for the same image. Rescaling and
    normalization are elementwise and agree across backends to float precision.

    Args:
        processor: Hugging Face image processor supplying mean, std and rescale.
        images: RGB Pillow images to resize directly, without cropping.
        width: Model input width in pixels.
        height: Model input height in pixels.
        resample: Pillow resampling filter used for the direct resize.

    Returns:
        The processor's ``pixel_values`` batch, one entry per input image.
    """
    # O resize direto fica no Pillow para que o vetor não dependa do backend do processor.
    resized = [image.resize((width, height), resample) for image in images]
    inputs = processor(
        images=resized,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    return inputs["pixel_values"]

"""Versioned evidence-view policies that materialize exact semantic views from a frozen region.

A :class:`SemanticViewPolicy` says which visual views a region request carries, in which
order, and how each is cut from the prepared image. Materializing it is deterministic:
identical pixels, region and policy always produce byte-identical, content-addressed
payloads, and every view records how it was built (:class:`SemanticViewConstruction`).
Nothing is added that the policy does not declare: in particular, a full frame reaches a
region request only when the policy names it.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.models import Region2D
from contextmap.visual_perception.semantic_requests import (
    SemanticInterpretationMode,
    SemanticInterpreterCapabilities,
    SemanticViewConstruction,
    SemanticVisualView,
    VisualViewKind,
)

if TYPE_CHECKING:
    import numpy as np

SEMANTIC_VIEW_POLICY_VERSION = "semantic-views/1"
"""Version of the construction rules every :class:`SemanticViewPolicy` applies.

Version 1 fixes: crop windows round outward (floor of the minimum, ceiling of the maximum
edge) and are clamped to the image, never padded; views keep the source pixel grid (no
resampling); payloads are PNG, 8-bit RGB, no interlace, filter type 0 on every row and
zlib level 9. A change to any of these is a new version.
"""

_VIEW_ROOT = "outputs/semantic-views"
_REGION_BOUND = frozenset(
    {VisualViewKind.MASKED_SUBJECT, VisualViewKind.TIGHT_CROP, VisualViewKind.CONTEXTUAL_CROP}
)
_FIXED_RULES = {
    "crop_rounding": "outward-floor-ceil/clamped-to-image",
    "resize": "none",
    "encoding": "png/rgb8/filter-0/zlib-9",
}


def _validate_rgb(name: str, value: tuple[int, int, int]) -> None:
    if len(value) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
        for item in value
    ):
        raise ValueError(f"{name} must be three integers in 0..255: {value!r}")


@dataclass(frozen=True, kw_only=True)
class SemanticContextBoundary:
    """Outline drawn around the region's tight box inside a contextual crop.

    The outline lies just outside the tight box, so the subject's own pixels are never
    painted over; it is clipped where the crop ends.

    Attributes:
        rgb: Outline color.
        width_px: Outline thickness in pixels.
    """

    rgb: tuple[int, int, int]
    width_px: int

    def __post_init__(self) -> None:
        """Reject an invalid color or a non-positive thickness."""
        _validate_rgb("context boundary rgb", self.rgb)
        if self.width_px < 1:
            raise ValueError("context boundary width_px must be at least 1")


@dataclass(frozen=True, kw_only=True)
class SemanticViewPolicy:
    """Which views a region request carries, in which order, and how each is built.

    A scene request always carries exactly one full-frame view, built by the same rules. A
    parameter exists only for the view it shapes, so two policies that build the same views
    have the same identity (:meth:`fingerprint`).

    Attributes:
        region_views: Ordered, unique view kinds of every region request. At least one of
            them shows the region itself (masked subject, tight or contextual crop).
        mask_fill_rgb: Color of every pixel outside the region mask in the masked-subject
            view; required exactly when that view is declared.
        context_margin_ratio: Margin added on each side of the region box in the contextual
            crop, as a fraction of the box's own width (left/right) and height (top/bottom);
            required exactly when that view is declared.
        context_boundary: Outline of the region box in the contextual crop, or ``None`` to
            draw nothing. Only meaningful when the contextual crop is declared.
    """

    region_views: tuple[VisualViewKind, ...]
    mask_fill_rgb: tuple[int, int, int] | None = None
    context_margin_ratio: float | None = None
    context_boundary: SemanticContextBoundary | None = None

    def __post_init__(self) -> None:
        """Reject an empty, repeated or region-less view set and dead or missing parameters."""
        if not self.region_views:
            raise ValueError("a view policy must declare at least one region view")
        if len(set(self.region_views)) != len(self.region_views):
            raise ValueError("region views must be unique")
        if not _REGION_BOUND & set(self.region_views):
            raise ValueError(
                "a region request needs at least one region-bound view "
                "(masked_subject, tight_crop or contextual_crop)"
            )
        masked = VisualViewKind.MASKED_SUBJECT in self.region_views
        if masked != (self.mask_fill_rgb is not None):
            raise ValueError("mask_fill_rgb is required exactly when masked_subject is declared")
        if self.mask_fill_rgb is not None:
            _validate_rgb("mask_fill_rgb", self.mask_fill_rgb)
        contextual = VisualViewKind.CONTEXTUAL_CROP in self.region_views
        if contextual != (self.context_margin_ratio is not None):
            raise ValueError(
                "context_margin_ratio is required exactly when contextual_crop is declared"
            )
        if self.context_boundary is not None and not contextual:
            raise ValueError("context_boundary applies only when contextual_crop is declared")
        if self.context_margin_ratio is not None:
            if not math.isfinite(self.context_margin_ratio):
                raise ValueError("context_margin_ratio must be finite")
            if self.context_margin_ratio <= 0:
                raise ValueError("context_margin_ratio must be positive")

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible record of every rule this policy applies."""
        boundary = self.context_boundary
        return {
            "version": SEMANTIC_VIEW_POLICY_VERSION,
            "region_views": [kind.value for kind in self.region_views],
            "scene_views": [VisualViewKind.FULL_FRAME.value],
            "mask_fill_rgb": None if self.mask_fill_rgb is None else list(self.mask_fill_rgb),
            "context_margin_ratio": self.context_margin_ratio,
            "context_boundary": (
                None
                if boundary is None
                else {"rgb": list(boundary.rgb), "width_px": boundary.width_px}
            ),
            **_FIXED_RULES,
        }

    def fingerprint(self) -> str:
        """Return the ``"sha256:<hex>"`` identity of :meth:`to_document`."""
        canonical = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class MaterializedSemanticView:
    """One view record and the exact bytes it identifies by ``view.sha256``."""

    view: SemanticVisualView
    payload: bytes


def materialize_scene_view(
    image: np.ndarray,
    *,
    source_observation_id: SourceObservationId,
    source_image_sha256: str,
    policy: SemanticViewPolicy,
) -> MaterializedSemanticView:
    """Build the one full-frame view of a scene request.

    Args:
        image: Prepared image pixels, ``(height, width, 3)`` ``uint8`` RGB.
        source_observation_id: Observation the image was prepared from.
        source_image_sha256: SHA-256 of the encoded prepared image.
        policy: The run's view policy.

    Returns:
        The full-frame view, identical to the full frame a region request of the same image
        carries when its policy declares one.

    Raises:
        ValueError: If the image is not 8-bit RGB.
    """
    _require_rgb8(image)
    height, width = image.shape[:2]
    return _materialized(
        image,
        kind=VisualViewKind.FULL_FRAME,
        bounds=(0, 0, width, height),
        view_id=f"v-{source_observation_id}-full_frame",
        file_stem=f"{source_observation_id}__full_frame",
        source_observation_id=source_observation_id,
        source_image_sha256=source_image_sha256,
        region=None,
        policy=policy,
    )


def materialize_region_views(
    image: np.ndarray,
    *,
    source_observation_id: SourceObservationId,
    source_image_sha256: str,
    region: Region2D,
    policy: SemanticViewPolicy,
) -> tuple[MaterializedSemanticView, ...]:
    """Build exactly the views ``policy`` declares for one frozen region, in its order.

    Args:
        image: Prepared image pixels, ``(height, width, 3)`` ``uint8`` RGB, the image the
            region was discovered in.
        source_observation_id: Observation the image was prepared from.
        source_image_sha256: SHA-256 of the encoded prepared image.
        region: The frozen region; its geometry is read, never changed.
        policy: The run's view policy.

    Returns:
        One materialized view per declared kind, in declared order.

    Raises:
        ValueError: If the region belongs to another observation, the image is not 8-bit
            RGB or does not have the region's image dimensions, the region window lies
            outside the image, or a masked subject is
            requested for a region without an inline mask of the image's size.
    """
    _require_rgb8(image)
    height, width = image.shape[:2]
    if region.source_observation_id not in (None, source_observation_id):
        raise ValueError(
            f"region {region.region_id!r} belongs to observation "
            f"{region.source_observation_id!r}, not {source_observation_id!r}"
        )
    if region.image_width is not None and (region.image_width, region.image_height) != (
        width,
        height,
    ):
        raise ValueError(
            f"region {region.region_id!r} was discovered in a {region.image_width}x"
            f"{region.image_height} image; the image has dimensions {width}x{height}"
        )
    tight = _window(region, margin_ratio=0.0, width=width, height=height)
    materialized: list[MaterializedSemanticView] = []
    for kind in policy.region_views:
        if kind is VisualViewKind.FULL_FRAME:
            materialized.append(
                materialize_scene_view(
                    image,
                    source_observation_id=source_observation_id,
                    source_image_sha256=source_image_sha256,
                    policy=policy,
                )
            )
            continue
        bounds = tight
        if kind is VisualViewKind.CONTEXTUAL_CROP:
            assert policy.context_margin_ratio is not None  # garantido pela política.
            bounds = _window(
                region, margin_ratio=policy.context_margin_ratio, width=width, height=height
            )
        materialized.append(
            _materialized(
                image,
                kind=kind,
                bounds=bounds,
                view_id=f"v-{source_observation_id}-{region.region_id}-{kind.value}",
                file_stem=f"{source_observation_id}__{region.region_id}__{kind.value}",
                source_observation_id=source_observation_id,
                source_image_sha256=source_image_sha256,
                region=region,
                policy=policy,
                tight=tight,
            )
        )
    return tuple(materialized)


def check_view_policy_supported(
    policy: SemanticViewPolicy, capabilities: SemanticInterpreterCapabilities
) -> None:
    """Refuse, before any inference, requests an interpreter declares it cannot consume.

    Only the modes the interpreter supports are checked: region views of an interpreter
    without region mode are never sent to it.

    Raises:
        ValueError: If a declared view kind is not supported, the region view set exceeds
            the interpreter's view bound, or it lacks a view kind the interpreter requires.
    """
    shapes: list[tuple[str, tuple[VisualViewKind, ...]]] = []
    if SemanticInterpretationMode.SCENE in capabilities.supported_modes:
        shapes.append(("scene", (VisualViewKind.FULL_FRAME,)))
    if SemanticInterpretationMode.REGION in capabilities.supported_modes:
        shapes.append(("region", policy.region_views))
    for mode, kinds in shapes:
        unsupported = sorted(
            kind.value for kind in kinds if kind not in capabilities.supported_view_kinds
        )
        if unsupported:
            raise ValueError(
                f"{mode} requests of this view policy carry view kinds the semantic "
                f"interpreter does not support: {unsupported}"
            )
        maximum = capabilities.max_visual_views
        if maximum is not None and len(kinds) > maximum:
            raise ValueError(
                f"the semantic interpreter accepts at most {maximum} visual view(s); "
                f"{mode} requests of this view policy carry {len(kinds)}"
            )
        missing = sorted(kind.value for kind in capabilities.required_view_kinds - set(kinds))
        if missing:
            raise ValueError(
                f"{mode} requests of this view policy lack view kinds the semantic "
                f"interpreter requires: {missing}"
            )


def _require_rgb8(image: np.ndarray) -> None:
    import numpy as np

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"semantic views are cut from an 8-bit RGB image; got dtype {image.dtype} "
            f"and shape {image.shape}"
        )


def _window(
    region: Region2D, *, margin_ratio: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Round a (possibly expanded) region box outward and clamp it to the image."""
    box = region.bounding_box
    margin_x = margin_ratio * box.width
    margin_y = margin_ratio * box.height
    x_min = max(0, math.floor(box.x_min - margin_x))
    y_min = max(0, math.floor(box.y_min - margin_y))
    x_max = min(width, math.ceil(box.x_max + margin_x))
    y_max = min(height, math.ceil(box.y_max + margin_y))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"region {region.region_id!r} lies outside the {width}x{height} image")
    return x_min, y_min, x_max, y_max


def _materialized(
    image: np.ndarray,
    *,
    kind: VisualViewKind,
    bounds: tuple[int, int, int, int],
    view_id: str,
    file_stem: str,
    source_observation_id: SourceObservationId,
    source_image_sha256: str,
    region: Region2D | None,
    policy: SemanticViewPolicy,
    tight: tuple[int, int, int, int] | None = None,
) -> MaterializedSemanticView:
    x_min, y_min, x_max, y_max = bounds
    pixels = image[y_min:y_max, x_min:x_max].copy()
    if kind is VisualViewKind.MASKED_SUBJECT:
        assert policy.mask_fill_rgb is not None  # garantido pela política.
        subject = _region_mask(region, image)[y_min:y_max, x_min:x_max]
        pixels[~subject] = policy.mask_fill_rgb
    if kind is VisualViewKind.CONTEXTUAL_CROP and policy.context_boundary is not None:
        assert tight is not None
        _draw_outline(pixels, tight=tight, origin=(x_min, y_min), boundary=policy.context_boundary)
    payload = _encode_png(pixels)
    return MaterializedSemanticView(
        view=SemanticVisualView(
            view_id=view_id,
            kind=kind,
            payload_reference=f"{_VIEW_ROOT}/{file_stem}.png",
            source_observation_id=source_observation_id,
            sha256=hashlib.sha256(payload).hexdigest(),
            region_id=None if region is None else region.region_id,
            construction=SemanticViewConstruction(
                policy_fingerprint=policy.fingerprint(),
                source_image_sha256=source_image_sha256,
                pixel_bounds=bounds,
            ),
        ),
        payload=payload,
    )


def _region_mask(region: Region2D | None, image: np.ndarray) -> np.ndarray:
    """Return the region's full-image boolean mask; a box-only region is never masked."""
    import numpy as np

    height, width = image.shape[:2]
    if region is None or region.mask is None:
        name = None if region is None else region.region_id
        raise ValueError(
            f"masked_subject needs the region's inline mask; region {name!r} has none "
            "(select a mask-producing region discovery backend or another view)"
        )
    mask = region.mask
    if (mask.width, mask.height) != (width, height):
        raise ValueError(
            f"region {region.region_id!r} inline mask dimensions {mask.width}x{mask.height} "
            f"do not match the {width}x{height} image"
        )
    return np.asarray(mask.data, dtype=bool).reshape(height, width)


def _draw_outline(
    pixels: np.ndarray,
    *,
    tight: tuple[int, int, int, int],
    origin: tuple[int, int],
    boundary: SemanticContextBoundary,
) -> None:
    """Paint a ring of ``width_px`` just outside the tight box, clipped to the crop."""
    import numpy as np

    crop_height, crop_width = pixels.shape[:2]
    x_min, y_min = tight[0] - origin[0], tight[1] - origin[1]
    x_max, y_max = tight[2] - origin[0], tight[3] - origin[1]
    width = boundary.width_px
    outer_x_min, outer_y_min = max(0, x_min - width), max(0, y_min - width)
    outer_x_max, outer_y_max = min(crop_width, x_max + width), min(crop_height, y_max + width)
    ring = np.zeros((crop_height, crop_width), dtype=bool)
    ring[outer_y_min:outer_y_max, outer_x_min:outer_x_max] = True
    ring[y_min:y_max, x_min:x_max] = False
    pixels[ring] = boundary.rgb


def _encode_png(pixels: np.ndarray) -> bytes:
    """Encode RGB8 pixels as a PNG whose bytes depend only on the pixels (policy version 1)."""
    import numpy as np

    height, width = pixels.shape[:2]
    rows = np.zeros((height, 1 + 3 * width), dtype=np.uint8)
    rows[:, 1:] = pixels.reshape(height, 3 * width)  # coluna 0: filtro 0 em toda linha.
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 9))
        + _chunk(b"IEND", b"")
    )


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

"""Tests for versioned semantic evidence-view policies and exact view materialization (#524)."""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import replace

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    SEMANTIC_VIEW_POLICY_VERSION,
    BackendProvenance,
    BoundingBox2D,
    InlineMask,
    Region2D,
    RegionId,
    SemanticContextBoundary,
    SemanticInterpretationMode,
    SemanticInterpreterCapabilities,
    SemanticViewPolicy,
    VisualViewKind,
    check_view_policy_supported,
    materialize_region_views,
    materialize_scene_view,
)

SOURCE = SourceObservationId("frame-0007")
SOURCE_SHA = "a" * 64
FULL = VisualViewKind.FULL_FRAME
MASKED = VisualViewKind.MASKED_SUBJECT
TIGHT = VisualViewKind.TIGHT_CROP
CONTEXT = VisualViewKind.CONTEXTUAL_CROP


def _image(width: int = 8, height: int = 6) -> np.ndarray:
    """Every pixel is unique, so a wrong crop or fill can never look right by accident."""
    ys, xs = np.mgrid[0:height, 0:width]
    return np.stack([xs * 10, ys * 10, xs + ys], axis=-1).astype(np.uint8)


def _decode_png(payload: bytes) -> np.ndarray:
    """Decode the RGB8, filter-0 PNGs this policy version writes (test helper, not a codec)."""
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    offset, chunks = 8, {}
    idat = b""
    while offset < len(payload):
        (length,) = struct.unpack(">I", payload[offset : offset + 4])
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", payload[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(kind + data)
        if kind == b"IDAT":
            idat += data
        else:
            chunks[kind] = data
        offset += 12 + length
    width, height, depth, color, _, _, _ = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
    assert (depth, color) == (8, 2)
    raw = zlib.decompress(idat)
    stride = 1 + 3 * width
    rows = [raw[row * stride : (row + 1) * stride] for row in range(height)]
    assert all(row[0] == 0 for row in rows)
    return np.frombuffer(b"".join(row[1:] for row in rows), dtype=np.uint8).reshape(
        height, width, 3
    )


def _region(
    box: BoundingBox2D, *, mask: InlineMask | None = None, width: int = 8, height: int = 6
) -> Region2D:
    return Region2D(
        region_id=RegionId("run-0001--frame-0007--region-0000"),
        bounding_box=box,
        provenance=BackendProvenance(
            backend_id="fake",
            capability="region_discovery",
            provider="fake",
            model="m",
            version="1",
        ),
        source_observation_id=SOURCE,
        image_width=width,
        image_height=height,
        mask=mask,
    )


def _mask(pixels: set[tuple[int, int]], width: int = 8, height: int = 6) -> InlineMask:
    return InlineMask(
        width=width,
        height=height,
        data=tuple((x, y) in pixels for y in range(height) for x in range(width)),
    )


def _policy(*views: VisualViewKind, **overrides: object) -> SemanticViewPolicy:
    values: dict[str, object] = {"region_views": views}
    if MASKED in views:
        values["mask_fill_rgb"] = (1, 2, 3)
    if CONTEXT in views:
        values["context_margin_ratio"] = 0.5
    values.update(overrides)
    return SemanticViewPolicy(**values)  # type: ignore[arg-type]


def _materialize(
    policy: SemanticViewPolicy, region: Region2D, image: np.ndarray | None = None
) -> tuple:
    return materialize_region_views(
        _image() if image is None else image,
        source_observation_id=SOURCE,
        source_image_sha256=SOURCE_SHA,
        region=region,
        policy=policy,
    )


BOX = BoundingBox2D(x=2, y=1, width=3, height=2)
MASK = _mask({(2, 1), (3, 1), (3, 2)})


class TestMaterialization:
    def test_identical_inputs_and_policy_give_byte_identical_content_addressed_views(
        self,
    ) -> None:
        policy = _policy(MASKED, TIGHT, CONTEXT)

        first = _materialize(policy, _region(BOX, mask=MASK))
        second = _materialize(policy, _region(BOX, mask=MASK))

        assert first == second
        for item in first:
            assert item.view.sha256 == hashlib.sha256(item.payload).hexdigest()

    def test_tight_crop_rounds_outward_and_clamps_at_the_image_edge(self) -> None:
        fractional = BoundingBox2D(x=1.5, y=0.25, width=2.2, height=1.1)
        at_edge = BoundingBox2D(x=6.5, y=4, width=5, height=5)

        (inner,) = _materialize(_policy(TIGHT), _region(fractional))
        (edge,) = _materialize(_policy(TIGHT), _region(at_edge))

        assert inner.view.construction.pixel_bounds == (1, 0, 4, 2)
        assert np.array_equal(_decode_png(inner.payload), _image()[0:2, 1:4])
        assert edge.view.construction.pixel_bounds == (6, 4, 8, 6)
        assert np.array_equal(_decode_png(edge.payload), _image()[4:6, 6:8])

    def test_masked_subject_keeps_the_mask_and_fills_the_rest_with_the_policy_color(
        self,
    ) -> None:
        (masked,) = _materialize(_policy(MASKED), _region(BOX, mask=MASK))

        pixels = _decode_png(masked.payload)
        assert masked.view.construction.pixel_bounds == (2, 1, 5, 3)
        expected = np.full((2, 3, 3), (1, 2, 3), dtype=np.uint8)
        for x, y in ((2, 1), (3, 1), (3, 2)):
            expected[y - 1, x - 2] = _image()[y, x]
        assert np.array_equal(pixels, expected)

    def test_contextual_crop_expands_by_the_ratio_and_is_clamped_not_padded(self) -> None:
        (inner,) = _materialize(_policy(CONTEXT), _region(BOX))
        (edge,) = _materialize(
            _policy(CONTEXT), _region(BoundingBox2D(x=0, y=0, width=2, height=2))
        )

        # 0.5 x (3, 2) de margem por lado: x em [0.5, 6.5) -> [0, 7), y em [0, 4).
        assert inner.view.construction.pixel_bounds == (0, 0, 7, 4)
        assert np.array_equal(_decode_png(inner.payload), _image()[0:4, 0:7])
        assert edge.view.construction.pixel_bounds == (0, 0, 3, 3)

    def test_the_boundary_outlines_the_tight_box_around_the_subject_only(self) -> None:
        policy = _policy(
            CONTEXT,
            context_boundary=SemanticContextBoundary(rgb=(255, 0, 255), width_px=1),
        )

        (view,) = _materialize(policy, _region(BOX))

        pixels = _decode_png(view.payload)
        crop = _image()[0:4, 0:7]
        outline = {(x, 0) for x in range(1, 6)} | {(x, 3) for x in range(1, 6)}
        outline |= {(1, y) for y in range(4)} | {(5, y) for y in range(4)}
        for y in range(4):
            for x in range(7):
                expected = (255, 0, 255) if (x, y) in outline else tuple(crop[y, x])
                assert tuple(pixels[y, x]) == expected, (x, y)

    @pytest.mark.parametrize(
        "views",
        [
            (MASKED,),
            (TIGHT,),
            (CONTEXT,),
            (FULL, TIGHT),
            (MASKED, TIGHT),
            (MASKED, CONTEXT),
            (TIGHT, CONTEXT),
            (MASKED, TIGHT, CONTEXT),
            (FULL, MASKED, TIGHT, CONTEXT),
        ],
    )
    def test_each_variant_yields_exactly_its_declared_views_in_order(
        self, views: tuple[VisualViewKind, ...]
    ) -> None:
        materialized = _materialize(_policy(*views), _region(BOX, mask=MASK))

        assert tuple(item.view.kind for item in materialized) == views
        assert len({item.view.view_id for item in materialized}) == len(views)

    def test_every_view_keeps_its_observation_region_and_construction_lineage(self) -> None:
        policy = _policy(FULL, MASKED, TIGHT, CONTEXT)

        materialized = _materialize(policy, _region(BOX, mask=MASK))

        for item in materialized:
            view = item.view
            assert view.source_observation_id == SOURCE
            assert view.construction is not None
            assert view.construction.policy_fingerprint == policy.fingerprint()
            assert view.construction.source_image_sha256 == SOURCE_SHA
            assert view.payload_reference.startswith("outputs/semantic-views/")
            if view.kind is FULL:
                assert view.region_id is None
                assert view.construction.pixel_bounds == (0, 0, 8, 6)
            else:
                assert view.region_id == RegionId("run-0001--frame-0007--region-0000")

    def test_the_full_frame_of_a_region_request_is_the_scene_view(self) -> None:
        policy = _policy(FULL, TIGHT)

        scene = materialize_scene_view(
            _image(), source_observation_id=SOURCE, source_image_sha256=SOURCE_SHA, policy=policy
        )
        full, _ = _materialize(policy, _region(BOX))

        assert full == scene
        assert np.array_equal(_decode_png(scene.payload), _image())

    def test_a_masked_subject_needs_the_region_inline_mask(self) -> None:
        with pytest.raises(ValueError, match="inline mask"):
            _materialize(_policy(MASKED), _region(BOX))

    def test_a_region_outside_the_image_or_of_another_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            _materialize(_policy(TIGHT), _region(BoundingBox2D(x=9, y=1, width=1, height=1)))
        with pytest.raises(ValueError, match="dimensions"):
            _materialize(_policy(TIGHT), _region(BOX), image=_image(width=9))

    def test_a_region_of_another_observation_is_refused(self) -> None:
        other = replace(_region(BOX), source_observation_id=SourceObservationId("frame-0008"))

        with pytest.raises(ValueError, match="frame-0008"):
            _materialize(_policy(TIGHT), other)

    def test_only_an_rgb8_image_is_accepted(self) -> None:
        with pytest.raises(ValueError, match="RGB"):
            _materialize(_policy(TIGHT), _region(BOX), image=_image()[:, :, 0])


class TestPolicy:
    def test_the_fingerprint_is_deterministic_and_follows_every_parameter(self) -> None:
        base = _policy(MASKED, CONTEXT)

        assert base.fingerprint() == _policy(MASKED, CONTEXT).fingerprint()
        variants = {
            _policy(CONTEXT, MASKED).fingerprint(),
            _policy(MASKED, CONTEXT, mask_fill_rgb=(0, 0, 0)).fingerprint(),
            _policy(MASKED, CONTEXT, context_margin_ratio=0.25).fingerprint(),
            _policy(
                MASKED,
                CONTEXT,
                context_boundary=SemanticContextBoundary(rgb=(255, 0, 0), width_px=2),
            ).fingerprint(),
        }
        assert base.fingerprint() not in variants
        assert len(variants) == 4

    def test_the_document_records_the_fixed_rules_of_its_version(self) -> None:
        document = _policy(TIGHT).to_document()

        assert document["version"] == SEMANTIC_VIEW_POLICY_VERSION
        assert document["region_views"] == ["tight_crop"]
        assert {"crop_rounding", "resize", "encoding"} <= set(document)
        assert _policy(TIGHT).fingerprint().startswith("sha256:")

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ({"region_views": ()}, "at least one"),
            ({"region_views": (TIGHT, TIGHT)}, "unique"),
            ({"region_views": (FULL,)}, "region-bound"),
            ({"region_views": (MASKED,)}, "mask_fill_rgb"),
            ({"region_views": (CONTEXT,)}, "context_margin_ratio"),
            ({"region_views": (TIGHT,), "mask_fill_rgb": (0, 0, 0)}, "mask_fill_rgb"),
            ({"region_views": (TIGHT,), "context_margin_ratio": 0.5}, "context_margin_ratio"),
            (
                {
                    "region_views": (TIGHT,),
                    "context_boundary": SemanticContextBoundary(rgb=(0, 0, 0), width_px=1),
                },
                "context_boundary",
            ),
            ({"region_views": (MASKED,), "mask_fill_rgb": (0, 0, 256)}, "0..255"),
            ({"region_views": (CONTEXT,), "context_margin_ratio": 0.0}, "positive"),
            ({"region_views": (CONTEXT,), "context_margin_ratio": float("inf")}, "finite"),
        ],
    )
    def test_an_ambiguous_or_dead_parameter_is_refused(
        self, values: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SemanticViewPolicy(**values)  # type: ignore[arg-type]

    def test_a_boundary_has_a_color_and_a_positive_width(self) -> None:
        with pytest.raises(ValueError, match="width_px"):
            SemanticContextBoundary(rgb=(0, 0, 0), width_px=0)


class TestPreflight:
    @staticmethod
    def _single_region_filling_view() -> SemanticInterpreterCapabilities:
        """What Florence-2's region tasks declare."""
        return SemanticInterpreterCapabilities(
            supported_modes=frozenset({SemanticInterpretationMode.REGION}),
            supported_view_kinds=frozenset({TIGHT, MASKED}),
            accepts_visual_features=False,
            accepts_scene_context=False,
            max_visual_views=1,
        )

    def test_a_single_region_filling_view_is_accepted(self) -> None:
        check_view_policy_supported(_policy(TIGHT), self._single_region_filling_view())
        check_view_policy_supported(_policy(MASKED), self._single_region_filling_view())

    @pytest.mark.parametrize(
        ("views", "message"),
        [((MASKED, TIGHT), "at most 1"), ((CONTEXT,), "contextual_crop"), ((FULL, TIGHT), "full")],
    )
    def test_a_combination_the_interpreter_cannot_consume_fails_before_inference(
        self, views: tuple[VisualViewKind, ...], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            check_view_policy_supported(_policy(*views), self._single_region_filling_view())

    def test_region_views_are_not_checked_for_an_interpreter_without_region_mode(self) -> None:
        scene_only = replace(
            self._single_region_filling_view(),
            supported_modes=frozenset({SemanticInterpretationMode.SCENE}),
            supported_view_kinds=frozenset({FULL}),
        )

        check_view_policy_supported(_policy(MASKED, TIGHT, CONTEXT), scene_only)

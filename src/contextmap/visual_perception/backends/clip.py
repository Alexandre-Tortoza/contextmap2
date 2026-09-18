"""CLIP visual-feature adapter for global images and frozen region views.

This adapter performs image encoding only. Text encoding, semantic scoring,
and label selection are deliberately absent even though CLIP's projected image
space is language-aligned.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.embedding_space import (
    EmbeddingSpace,
    embedding_space_fingerprint,
)
from contextmap.visual_perception.identity import feature_id_for, perception_result_id_for
from contextmap.visual_perception.models import (
    BackendProvenance,
    BoundingBox2D,
    FeatureScope,
    PerceptionRunId,
    PreparedImage,
    Region2D,
    RegionId,
    VisualFeature,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ClipBackendError(RuntimeError):
    """Base class for explicit CLIP visual-backend failures."""


class ClipDependencyError(ClipBackendError):
    """Raised when an optional CLIP runtime dependency is unavailable."""


class ClipDeviceError(ClipBackendError):
    """Raised when the configured device/precision cannot be used."""


class ClipModelLoadError(ClipBackendError):
    """Raised when the configured CLIP checkpoint cannot load."""


class ClipInferenceError(ClipBackendError):
    """Raised for invalid views, preprocessing, inference, or output metadata."""


@dataclass(frozen=True, kw_only=True)
class ClipConfig:
    """Effective configuration of the CLIP visual adapter.

    Attributes:
        checkpoint: CLIP model/checkpoint identity.
        revision: Exact repository revision or commit.
        scope: Configured output mode, ``GLOBAL`` or ``REGION``.
        device: Requested ``"cpu"``, ``"cuda"``, or ``"mps"`` device.
        precision: Inference/payload dtype, ``"float32"`` or ``"float16"``.
        input_width: Direct-resize model input width.
        input_height: Direct-resize model input height.
        local_files_only: Prohibit implicit downloads when true.
        l2_normalize: Normalize projected image embeddings before persistence.
        crop_policy: ``"tight_box"`` or ``"context_box"``.
        context_padding_fraction: Fraction of region width/height added on each
            side for ``context_box``.
        payload_prefix: Artifact-relative feature directory.
        code_version: Adapter/view-policy version.
    """

    checkpoint: str
    revision: str
    scope: FeatureScope
    device: str = "cpu"
    precision: str = "float32"
    input_width: int = 224
    input_height: int = 224
    local_files_only: bool = True
    l2_normalize: bool = True
    crop_policy: str = "tight_box"
    context_padding_fraction: float = 0.0
    payload_prefix: str = "features"
    code_version: str = "1"

    def __post_init__(self) -> None:
        """Validate model, execution, scope, and view-policy settings."""
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        if not self.revision:
            raise ValueError("revision must not be empty")
        if self.scope not in {FeatureScope.GLOBAL, FeatureScope.REGION}:
            raise ValueError("scope must be GLOBAL or REGION")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if self.precision not in {"float32", "float16"}:
            raise ValueError("precision must be one of: float32, float16")
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("input_width and input_height must be positive")
        if self.crop_policy not in {"tight_box", "context_box"}:
            raise ValueError("crop_policy must be tight_box or context_box")
        if self.context_padding_fraction < 0.0:
            raise ValueError("context_padding_fraction must be non-negative")
        if self.crop_policy == "tight_box" and self.context_padding_fraction != 0.0:
            raise ValueError("context_padding_fraction must be zero for tight_box")
        prefix = Path(self.payload_prefix)
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("payload_prefix must be an artifact-relative path")
        if not self.code_version:
            raise ValueError("code_version must not be empty")


@dataclass(frozen=True, kw_only=True)
class ClipView:
    """Exact source view supplied to the CLIP image encoder.

    Attributes:
        source_region_id: Frozen source region, or ``None`` for global mode.
        crop_box: Pixel box cropped from the prepared image.
        crop_policy: Version-visible crop/context policy.
        source_image_width: Prepared-image width.
        source_image_height: Prepared-image height.
        model_input_width: Width after processor resize.
        model_input_height: Height after processor resize.
        coordinate_transform_id: Hash of source, crop, and resize geometry.
    """

    source_region_id: RegionId | None
    crop_box: BoundingBox2D
    crop_policy: str
    source_image_width: int
    source_image_height: int
    model_input_width: int
    model_input_height: int
    coordinate_transform_id: str


@dataclass(frozen=True, kw_only=True)
class ClipNativeOutput:
    """SDK-free batch of projected CLIP image embeddings.

    Attributes:
        array: NumPy array shaped ``(view_count, projection_dimension)``.
        elapsed_seconds: Decode, preprocessing, and inference wall time.
        warnings: Runtime warnings that do not invalidate the output.
    """

    array: NDArray[Any]
    elapsed_seconds: float
    warnings: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate output rank, dimensions, and timing."""
        if self.array.ndim != 2 or any(dimension <= 0 for dimension in self.array.shape):
            raise ValueError("array must have shape (view_count, projection_dimension)")
        if self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be non-negative")


@dataclass(frozen=True, kw_only=True)
class ClipDiagnostics:
    """Execution diagnostics for one visual encoding call.

    Attributes:
        elapsed_seconds: Runtime-reported decode/preprocess/inference time.
        view_count: Number of encoded global/region views.
        warnings: Runtime and crop-boundary warnings.
    """

    elapsed_seconds: float
    view_count: int
    warnings: Sequence[str]


@dataclass(frozen=True, kw_only=True)
class ClipExtraction:
    """Canonical features plus arrays, views, compatibility, and diagnostics.

    Attributes:
        features: One global feature or one feature per input region.
        embedding_space: Exact CLIP projected image space identity.
        array: Batch payload; each row is persisted separately.
        views: Exact source view corresponding to each array row.
        diagnostics: Timing and warning evidence.
    """

    features: Sequence[VisualFeature]
    embedding_space: EmbeddingSpace
    array: NDArray[Any]
    views: Sequence[ClipView]
    diagnostics: ClipDiagnostics


class ClipRuntime(Protocol):
    """Internal replaceable boundary around CLIP image encoding."""

    def encode(self, image: PreparedImage, views: Sequence[ClipView]) -> ClipNativeOutput:
        """Encode the exact declared views without text/scoring side effects."""
        ...


class FeaturePayloadSink(Protocol):
    """Minimal atomic feature-payload sink used by the adapter."""

    def add_feature_payload(
        self,
        feature: VisualFeature,
        source_observation_id: SourceObservationId,
        array: NDArray[Any],
    ) -> None:
        """Queue one vector for run-artifact finalization."""
        ...


class ClipVisualFeatureBackend:
    """FeatureExtractor adapter for global or region CLIP image embeddings."""

    def __init__(
        self,
        *,
        config: ClipConfig,
        run_id: PerceptionRunId,
        payload_sink: FeaturePayloadSink,
        runtime: ClipRuntime | None = None,
        prepared_image_root: Path | None = None,
    ) -> None:
        """Create a configured CLIP visual adapter."""
        if not str(run_id):
            raise ValueError("run_id must not be empty")
        if runtime is None and prepared_image_root is None:
            raise ValueError("prepared_image_root is required for the default runtime")
        self._config = config
        self._run_id = run_id
        self._payload_sink = payload_sink
        self._runtime = runtime or HuggingFaceClipRuntime(
            config=config,
            prepared_image_root=prepared_image_root,
        )

    def backend_provenance(self) -> BackendProvenance:
        """Report base checkpoint and effective adapter configuration."""
        return BackendProvenance(
            backend_id="clip_visual_huggingface",
            capability="feature_extractor",
            provider="huggingface",
            model=self._config.checkpoint,
            version=self._config.revision,
            configuration_fingerprint=_configuration_fingerprint(self._config),
        )

    def required_scope(self) -> FeatureScope:
        """Return the configured global or region scope."""
        return self._config.scope

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        """Encode/persist the configured visual scope and return metadata."""
        return self.extract_visual(image, regions).features

    def extract_visual(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> ClipExtraction:
        """Build explicit views, encode them, and queue canonical payloads."""
        import numpy as np

        views, view_warnings = _build_views(image=image, regions=regions, config=self._config)
        native = self._runtime.encode(image, views)
        if native.array.shape[0] != len(views):
            raise ClipInferenceError(
                f"runtime returned {native.array.shape[0]} embeddings for {len(views)} views"
            )
        if str(native.array.dtype) != self._config.precision:
            raise ClipInferenceError(
                f"runtime dtype {native.array.dtype!s} does not match "
                f"precision {self._config.precision!r}"
            )

        array = native.array
        normalization = "none"
        if self._config.l2_normalize:
            norms = np.linalg.norm(array, axis=-1, keepdims=True)
            array = np.divide(array, norms, out=np.zeros_like(array), where=norms != 0)
            normalization = "l2"
        embedding_space = EmbeddingSpace(
            family="clip",
            model=self._config.checkpoint,
            version=self._config.revision,
            checkpoint=f"{self._config.checkpoint}@{self._config.revision}",
            layer="image_projection",
            dimension=int(array.shape[1]),
            normalization=normalization,
        )
        result_id = perception_result_id_for(
            run_id=self._run_id,
            source_observation_id=image.source_observation_id,
        )
        features: list[VisualFeature] = []
        for index, view in enumerate(views):
            feature_id = feature_id_for(result_id=result_id, index=index)
            feature = VisualFeature(
                feature_id=feature_id,
                scope=self._config.scope,
                embedding_space_id=embedding_space_fingerprint(embedding_space),
                shape=(int(array.shape[1]),),
                dtype=str(array.dtype),
                normalization=normalization,
                payload_reference=f"{self._config.payload_prefix}/{feature_id}.npy",
                provenance=_view_provenance(self._config, view),
                region_id=view.source_region_id,
            )
            self._payload_sink.add_feature_payload(
                feature,
                image.source_observation_id,
                array[index],
            )
            features.append(feature)

        return ClipExtraction(
            features=tuple(features),
            embedding_space=embedding_space,
            array=array,
            views=views,
            diagnostics=ClipDiagnostics(
                elapsed_seconds=native.elapsed_seconds,
                view_count=len(views),
                warnings=tuple(native.warnings) + view_warnings,
            ),
        )


class HuggingFaceClipRuntime:
    """Lazy Transformers CLIP image-encoder runtime."""

    def __init__(self, *, config: ClipConfig, prepared_image_root: Path | None) -> None:
        """Create a runtime without importing SDKs or loading weights."""
        if prepared_image_root is None:
            raise ValueError("prepared_image_root must not be None")
        self._config = config
        self._prepared_image_root = prepared_image_root.resolve()
        self._torch: Any = None
        self._image_module: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._torch_dtype: Any = None

    def encode(self, image: PreparedImage, views: Sequence[ClipView]) -> ClipNativeOutput:
        """Decode exact crops and run only the CLIP image projection."""
        self._ensure_loaded()
        if not views:
            raise ClipInferenceError("at least one ClipView is required")
        image_path = _resolve_within_root(self._prepared_image_root, image.payload_reference)
        started_at = time.perf_counter()
        try:
            with self._image_module.open(image_path) as loaded_image:
                rgb_image = loaded_image.convert("RGB")
                if rgb_image.size != (image.width, image.height):
                    raise ClipInferenceError(
                        f"prepared image metadata {(image.width, image.height)} does not match "
                        f"decoded payload {rgb_image.size}"
                    )
                crops = [
                    rgb_image.crop(
                        (
                            view.crop_box.x,
                            view.crop_box.y,
                            view.crop_box.x + view.crop_box.width,
                            view.crop_box.y + view.crop_box.height,
                        )
                    )
                    for view in views
                ]
                inputs = self._processor(
                    images=crops,
                    return_tensors="pt",
                    do_resize=True,
                    size={"height": self._config.input_height, "width": self._config.input_width},
                    do_center_crop=False,
                )
            pixel_values = inputs["pixel_values"].to(
                device=self._config.device,
                dtype=self._torch_dtype,
            )
            with self._torch.inference_mode():
                encoded = self._model.get_image_features(pixel_values=pixel_values)
            if hasattr(encoded, "pooler_output"):
                encoded = encoded.pooler_output
            array = encoded.detach().to("cpu").numpy()
        except ClipInferenceError:
            raise
        except Exception as error:
            raise ClipInferenceError(
                f"CLIP visual inference failed for {image.payload_reference!r}: {error}"
            ) from error
        return ClipNativeOutput(
            array=array,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    def _ensure_loaded(self) -> None:
        """Import SDKs, validate device, and load exact processor/model lazily."""
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as error:
            raise ClipDependencyError(
                "CLIP requires torch, transformers, and Pillow in the runtime environment"
            ) from error

        if self._config.device == "cuda" and not torch.cuda.is_available():
            raise ClipDeviceError("configured CUDA device is unavailable")
        if self._config.device == "mps" and not torch.backends.mps.is_available():
            raise ClipDeviceError("configured MPS device is unavailable")
        if self._config.device == "cpu" and self._config.precision == "float16":
            raise ClipDeviceError("float16 CLIP inference is not supported on CPU")
        torch_dtype = getattr(torch, self._config.precision)
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                self._config.checkpoint,
                revision=self._config.revision,
                local_files_only=self._config.local_files_only,
            )
            model = transformers.CLIPModel.from_pretrained(
                self._config.checkpoint,
                revision=self._config.revision,
                local_files_only=self._config.local_files_only,
                torch_dtype=torch_dtype,
            )
            model = model.to(self._config.device)
            model.eval()
        except Exception as error:
            mode = "local cache" if self._config.local_files_only else "configured model source"
            raise ClipModelLoadError(
                f"could not load {self._config.checkpoint}@{self._config.revision} from {mode}: "
                f"{error}"
            ) from error
        self._torch = torch
        self._image_module = image_module
        self._processor = processor
        self._model = model
        self._torch_dtype = torch_dtype


def _build_views(
    *, image: PreparedImage, regions: Sequence[Region2D], config: ClipConfig
) -> tuple[tuple[ClipView, ...], tuple[str, ...]]:
    """Create deterministic global or region crop views without changing regions."""
    if config.scope is FeatureScope.GLOBAL:
        box = BoundingBox2D(x=0, y=0, width=image.width, height=image.height)
        return (_make_view(image=image, region_id=None, box=box, config=config),), ()
    if not regions:
        raise ClipInferenceError("region-scoped CLIP requires at least one region")
    if any(not region.is_accepted for region in regions):
        raise ClipInferenceError("region-scoped CLIP accepts only frozen accepted regions")

    views: list[ClipView] = []
    warnings: list[str] = []
    for region in regions:
        box, was_clipped = _crop_box(image=image, region=region, config=config)
        views.append(_make_view(image=image, region_id=region.region_id, box=box, config=config))
        if was_clipped:
            warnings.append(f"{region.region_id}: crop clipped to image")
    return tuple(views), tuple(warnings)


def _crop_box(
    *, image: PreparedImage, region: Region2D, config: ClipConfig
) -> tuple[BoundingBox2D, bool]:
    """Expand/clip one region box according to the configured view policy."""
    source = region.bounding_box
    padding_x = source.width * config.context_padding_fraction
    padding_y = source.height * config.context_padding_fraction
    left = math.floor(source.x - padding_x)
    top = math.floor(source.y - padding_y)
    right = math.ceil(source.x + source.width + padding_x)
    bottom = math.ceil(source.y + source.height + padding_y)
    clipped_left = max(0, left)
    clipped_top = max(0, top)
    clipped_right = min(image.width, right)
    clipped_bottom = min(image.height, bottom)
    if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
        raise ClipInferenceError(f"region {region.region_id!r} has no support inside the image")
    return (
        BoundingBox2D(
            x=clipped_left,
            y=clipped_top,
            width=clipped_right - clipped_left,
            height=clipped_bottom - clipped_top,
        ),
        (left, top, right, bottom) != (clipped_left, clipped_top, clipped_right, clipped_bottom),
    )


def _make_view(
    *, image: PreparedImage, region_id: RegionId | None, box: BoundingBox2D, config: ClipConfig
) -> ClipView:
    """Build a view and deterministic image→crop→model transform identity."""
    policy = (
        f"context_box:{config.context_padding_fraction:g}"
        if config.crop_policy == "context_box"
        else "tight_box"
    )
    payload = {
        "source_observation_id": image.source_observation_id,
        "source_image": {"width": image.width, "height": image.height},
        "region_id": region_id,
        "crop": {"x": box.x, "y": box.y, "width": box.width, "height": box.height},
        "crop_policy": policy,
        "model_input": {"width": config.input_width, "height": config.input_height},
        "resize": "direct_bicubic_by_clip_processor",
        "center_crop": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return ClipView(
        source_region_id=region_id,
        crop_box=box,
        crop_policy=policy,
        source_image_width=image.width,
        source_image_height=image.height,
        model_input_width=config.input_width,
        model_input_height=config.input_height,
        coordinate_transform_id=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    )


def _configuration_fingerprint(config: ClipConfig) -> str:
    """Hash the effective model/execution/view configuration."""
    payload = {
        "checkpoint": config.checkpoint,
        "revision": config.revision,
        "scope": config.scope.value,
        "device": config.device,
        "precision": config.precision,
        "input_width": config.input_width,
        "input_height": config.input_height,
        "local_files_only": config.local_files_only,
        "l2_normalize": config.l2_normalize,
        "crop_policy": config.crop_policy,
        "context_padding_fraction": config.context_padding_fraction,
        "payload_prefix": config.payload_prefix,
        "code_version": config.code_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _view_provenance(config: ClipConfig, view: ClipView) -> BackendProvenance:
    """Add exact per-feature view identity to the base configuration hash."""
    payload = {
        "configuration_fingerprint": _configuration_fingerprint(config),
        "coordinate_transform_id": view.coordinate_transform_id,
        "source_region_id": view.source_region_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return BackendProvenance(
        backend_id="clip_visual_huggingface",
        capability="feature_extractor",
        provider="huggingface",
        model=config.checkpoint,
        version=config.revision,
        configuration_fingerprint=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    )


def _resolve_within_root(root: Path, reference: str) -> Path:
    """Resolve a prepared-image payload without allowing path escape."""
    candidate = (root / reference).resolve()
    if not candidate.is_relative_to(root):
        raise ClipInferenceError(f"prepared image reference escapes its root: {reference!r}")
    if not candidate.is_file():
        raise ClipInferenceError(f"prepared image payload does not exist: {reference!r}")
    return candidate

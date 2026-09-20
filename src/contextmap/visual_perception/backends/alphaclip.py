"""AlphaCLIP masked-region visual feature adapter.

The adapter consumes frozen region masks through an injected decoder, preserves
their exact transformation lineage, and exposes only canonical VisualFeature
metadata plus NumPy payloads. Text encoding and semantic scoring are absent.
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
from contextmap.visual_perception.backends._feature_values import (
    validate_and_normalize_feature_values,
)
from contextmap.visual_perception.dense_region_association import box_mask_shape
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


class AlphaClipBackendError(RuntimeError):
    """Base class for explicit AlphaCLIP adapter failures."""


class AlphaClipDependencyError(AlphaClipBackendError):
    """Raised when an optional AlphaCLIP dependency is unavailable."""


class AlphaClipDeviceError(AlphaClipBackendError):
    """Raised when the configured device/precision cannot be used."""


class AlphaClipModelLoadError(AlphaClipBackendError):
    """Raised when local base/alpha checkpoints cannot be verified or loaded."""


class AlphaClipInferenceError(AlphaClipBackendError):
    """Raised for invalid masks/views, inference, or output metadata."""


@dataclass(frozen=True, kw_only=True)
class AlphaClipConfig:
    """Effective configuration of an AlphaCLIP region extractor.

    Attributes:
        model_name: AlphaCLIP/OpenAI CLIP architecture identity.
        base_checkpoint_path: Local base CLIP checkpoint under checkpoint root.
        alpha_checkpoint_path: Local AlphaCLIP checkpoint under checkpoint root.
        checkpoint_fingerprint: Expected combined SHA-256 identity of both files.
        device: Requested ``"cpu"``, ``"cuda"``, or ``"mps"`` device.
        precision: Inference/payload dtype, ``"float32"`` or ``"float16"``.
        input_width: Model input width.
        input_height: Model input height.
        view_policy: ``"full_image"`` or ``"context_box"``.
        context_padding_fraction: Region-relative context added on each side.
        image_interpolation: RGB resize interpolation; currently ``"bicubic"``.
        mask_interpolation: Mask resize interpolation; must be ``"nearest"``.
        l2_normalize: Normalize image embeddings before persistence.
        payload_prefix: Artifact-relative feature payload directory.
        code_version: Adapter/mask-transform policy version.
    """

    model_name: str
    base_checkpoint_path: str
    alpha_checkpoint_path: str
    checkpoint_fingerprint: str
    device: str = "cpu"
    precision: str = "float32"
    input_width: int = 224
    input_height: int = 224
    view_policy: str = "full_image"
    context_padding_fraction: float = 0.0
    image_interpolation: str = "bicubic"
    mask_interpolation: str = "nearest"
    l2_normalize: bool = True
    payload_prefix: str = "features"
    code_version: str = "1"

    def __post_init__(self) -> None:
        """Validate checkpoint, execution, and mask/view policy settings."""
        for name, value in (
            ("model_name", self.model_name),
            ("base_checkpoint_path", self.base_checkpoint_path),
            ("alpha_checkpoint_path", self.alpha_checkpoint_path),
            ("checkpoint_fingerprint", self.checkpoint_fingerprint),
            ("code_version", self.code_version),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")
        for name, reference in (
            ("base_checkpoint_path", self.base_checkpoint_path),
            ("alpha_checkpoint_path", self.alpha_checkpoint_path),
            ("payload_prefix", self.payload_prefix),
        ):
            path = Path(reference)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be an artifact-relative path")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if self.precision not in {"float32", "float16"}:
            raise ValueError("precision must be one of: float32, float16")
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("input_width and input_height must be positive")
        if self.view_policy not in {"full_image", "context_box"}:
            raise ValueError("view_policy must be full_image or context_box")
        if not math.isfinite(self.context_padding_fraction) or self.context_padding_fraction < 0.0:
            raise ValueError("context_padding_fraction must be finite and non-negative")
        if self.view_policy == "full_image" and self.context_padding_fraction != 0.0:
            raise ValueError("context_padding_fraction must be zero for full_image")
        if self.image_interpolation != "bicubic":
            raise ValueError("image_interpolation must be bicubic")
        if self.mask_interpolation != "nearest":
            raise ValueError("mask_interpolation must be nearest")
        if not self.payload_prefix:
            raise ValueError("payload_prefix must not be empty")


@dataclass(frozen=True, kw_only=True)
class AlphaClipView:
    """Auditable transformation of one frozen region/mask into model input.

    Attributes:
        source_region_id: Frozen region supporting this feature.
        source_mask_reference: Original region mask payload reference.
        mask_content_hash: Hash of the decoded box-local boolean mask.
        crop_box: Prepared-image view supplied to the model.
        view_policy: Version-visible full/context policy.
        source_image_width: Prepared-image width.
        source_image_height: Prepared-image height.
        model_input_width: RGB/alpha input width after resize.
        model_input_height: RGB/alpha input height after resize.
        image_resize_interpolation: RGB interpolation policy.
        mask_resize_interpolation: Alpha-mask interpolation policy.
        coordinate_transform_id: Hash of all source/view/resize metadata.
    """

    source_region_id: RegionId
    source_mask_reference: str
    mask_content_hash: str
    crop_box: BoundingBox2D
    view_policy: str
    source_image_width: int
    source_image_height: int
    model_input_width: int
    model_input_height: int
    image_resize_interpolation: str
    mask_resize_interpolation: str
    coordinate_transform_id: str


@dataclass(frozen=True, kw_only=True)
class AlphaClipRequest:
    """One runtime input pairing view metadata with its view-local mask.

    Attributes:
        view: Auditable source/view transformation.
        mask: Boolean mask in ``view.crop_box`` coordinates.
    """

    view: AlphaClipView
    mask: NDArray[Any]


@dataclass(frozen=True, kw_only=True)
class AlphaClipNativeOutput:
    """SDK-free projected AlphaCLIP embeddings and runtime diagnostics.

    Attributes:
        array: NumPy array shaped ``(request_count, projection_dimension)``.
        elapsed_seconds: Runtime wall time.
        peak_memory_bytes: Peak accelerator allocation when available.
        warnings: Non-fatal runtime diagnostics.
    """

    array: NDArray[Any]
    elapsed_seconds: float
    peak_memory_bytes: int | None
    warnings: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate output rank and diagnostics."""
        if self.array.ndim != 2 or any(dimension <= 0 for dimension in self.array.shape):
            raise ValueError("array must have shape (request_count, projection_dimension)")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be non-negative")


@dataclass(frozen=True, kw_only=True)
class AlphaClipDiagnostics:
    """Timing, memory, and warning evidence for one adapter call.

    Attributes:
        elapsed_seconds: Runtime decode/preprocess/inference wall time.
        peak_memory_bytes: Peak accelerator memory when available.
        request_count: Number of encoded regions.
        warnings: Runtime and view-boundary warnings.
    """

    elapsed_seconds: float
    peak_memory_bytes: int | None
    request_count: int
    warnings: Sequence[str]


@dataclass(frozen=True, kw_only=True)
class AlphaClipExtraction:
    """Canonical masked-region features with views and diagnostics.

    Attributes:
        features: One canonical region feature per request.
        embedding_space: Distinct AlphaCLIP projection-space identity.
        array: Batch payload, persisted one row per feature.
        views: Exact region/mask transformations corresponding to rows.
        diagnostics: Timing, memory, and warning evidence.
    """

    features: Sequence[VisualFeature]
    embedding_space: EmbeddingSpace
    array: NDArray[Any]
    views: Sequence[AlphaClipView]
    diagnostics: AlphaClipDiagnostics


class RegionMaskSource(Protocol):
    """Decoder boundary for Region2D mask payload references."""

    def load_box_local_mask(self, region: Region2D) -> NDArray[Any]:
        """Load a boolean mask shaped exactly as the region bounding box."""
        ...


class AlphaClipRuntime(Protocol):
    """Internal replaceable boundary around AlphaCLIP model inference."""

    def encode(
        self, image: PreparedImage, requests: Sequence[AlphaClipRequest]
    ) -> AlphaClipNativeOutput:
        """Encode RGB views conditioned on their explicit alpha masks."""
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


class AlphaClipRegionFeatureBackend:
    """FeatureExtractor adapter for mask-conditioned AlphaCLIP region vectors."""

    def __init__(
        self,
        *,
        config: AlphaClipConfig,
        run_id: PerceptionRunId,
        feature_stage_id: str,
        mask_source: RegionMaskSource,
        payload_sink: FeaturePayloadSink,
        runtime: AlphaClipRuntime | None = None,
        prepared_image_root: Path | None = None,
        checkpoint_root: Path | None = None,
    ) -> None:
        """Create a configured AlphaCLIP adapter."""
        if not str(run_id):
            raise ValueError("run_id must not be empty")
        if not feature_stage_id:
            raise ValueError("feature_stage_id must not be empty")
        if runtime is None and (prepared_image_root is None or checkpoint_root is None):
            raise ValueError(
                "prepared_image_root and checkpoint_root are required for the default runtime"
            )
        self._config = config
        self._run_id = run_id
        self._feature_stage_id = feature_stage_id
        self._mask_source = mask_source
        self._payload_sink = payload_sink
        self._runtime = runtime or OfficialAlphaClipRuntime(
            config=config,
            prepared_image_root=prepared_image_root,
            checkpoint_root=checkpoint_root,
        )

    def backend_provenance(self) -> BackendProvenance:
        """Report AlphaCLIP checkpoint and effective configuration."""
        return BackendProvenance(
            backend_id="alphaclip_official",
            capability="feature_extractor",
            provider="SunzeY/AlphaCLIP",
            model=self._config.model_name,
            version=self._config.code_version,
            configuration_fingerprint=_configuration_fingerprint(self._config),
        )

    def required_scope(self) -> FeatureScope:
        """Declare frozen-region feature extraction."""
        return FeatureScope.REGION

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        """Encode/persist masked region features and return canonical metadata."""
        return self.extract_masked(image, regions).features

    def extract_masked(
        self, image: PreparedImage, regions: Sequence[Region2D]
    ) -> AlphaClipExtraction:
        """Decode frozen masks, build views, run AlphaCLIP, and queue payloads."""
        if not regions:
            raise AlphaClipInferenceError("AlphaCLIP requires at least one region")
        if any(not region.is_accepted for region in regions):
            raise AlphaClipInferenceError("AlphaCLIP accepts only frozen accepted regions")

        requests: list[AlphaClipRequest] = []
        view_warnings: list[str] = []
        for region in regions:
            request, warnings = _request_for(
                image=image,
                region=region,
                mask=self._mask_source.load_box_local_mask(region),
                config=self._config,
            )
            requests.append(request)
            view_warnings.extend(warnings)

        native = self._runtime.encode(image, tuple(requests))
        if native.array.shape[0] != len(requests):
            raise AlphaClipInferenceError(
                f"runtime returned {native.array.shape[0]} embeddings for {len(requests)} requests"
            )
        if str(native.array.dtype) != self._config.precision:
            raise AlphaClipInferenceError(
                f"runtime dtype {native.array.dtype!s} does not match "
                f"precision {self._config.precision!r}"
            )
        array, normalization = validate_and_normalize_feature_values(
            native.array,
            l2_normalize=self._config.l2_normalize,
            error_type=AlphaClipInferenceError,
        )

        embedding_space = EmbeddingSpace(
            family="alphaclip",
            model=self._config.model_name,
            version=self._config.code_version,
            checkpoint=self._config.checkpoint_fingerprint,
            layer="alpha_conditioned_image_projection",
            dimension=int(array.shape[1]),
            normalization=normalization,
        )
        result_id = perception_result_id_for(
            run_id=self._run_id,
            source_observation_id=image.source_observation_id,
        )
        features: list[VisualFeature] = []
        for index, request in enumerate(requests):
            feature_id = feature_id_for(
                result_id=result_id,
                producer_id=self._feature_stage_id,
                index=index,
            )
            feature = VisualFeature(
                feature_id=feature_id,
                scope=FeatureScope.REGION,
                embedding_space_id=embedding_space_fingerprint(embedding_space),
                shape=(int(array.shape[1]),),
                dtype=str(array.dtype),
                normalization=normalization,
                payload_reference=f"{self._config.payload_prefix}/{feature_id}.npy",
                provenance=_view_provenance(self._config, request.view),
                region_id=request.view.source_region_id,
            )
            self._payload_sink.add_feature_payload(
                feature,
                image.source_observation_id,
                array[index],
            )
            features.append(feature)

        return AlphaClipExtraction(
            features=tuple(features),
            embedding_space=embedding_space,
            array=array,
            views=tuple(request.view for request in requests),
            diagnostics=AlphaClipDiagnostics(
                elapsed_seconds=native.elapsed_seconds,
                peak_memory_bytes=native.peak_memory_bytes,
                request_count=len(requests),
                warnings=tuple(native.warnings) + tuple(view_warnings),
            ),
        )


class OfficialAlphaClipRuntime:
    """Lazy wrapper around the official ``alpha_clip`` package."""

    def __init__(
        self,
        *,
        config: AlphaClipConfig,
        prepared_image_root: Path | None,
        checkpoint_root: Path | None,
    ) -> None:
        """Create a runtime without importing SDKs or loading checkpoints."""
        if prepared_image_root is None or checkpoint_root is None:
            raise ValueError("prepared_image_root and checkpoint_root must not be None")
        self._config = config
        self._prepared_image_root = prepared_image_root.resolve()
        self._checkpoint_root = checkpoint_root.resolve()
        self._torch: Any = None
        self._image_module: Any = None
        self._model: Any = None

    def encode(
        self, image: PreparedImage, requests: Sequence[AlphaClipRequest]
    ) -> AlphaClipNativeOutput:
        """Run the official RGB+alpha visual encoder on explicit requests."""
        self._ensure_loaded()
        if not requests:
            raise AlphaClipInferenceError("at least one AlphaClipRequest is required")
        image_path = _resolve_file(
            self._prepared_image_root,
            image.payload_reference,
            error_type=AlphaClipInferenceError,
        )
        started_at = time.perf_counter()
        peak_memory_bytes: int | None = None
        try:
            with self._image_module.open(image_path) as loaded_image:
                rgb_image = loaded_image.convert("RGB")
                if rgb_image.size != (image.width, image.height):
                    raise AlphaClipInferenceError(
                        f"prepared image metadata {(image.width, image.height)} does not match "
                        f"decoded payload {rgb_image.size}"
                    )
                image_tensors = []
                for request in requests:
                    box = request.view.crop_box
                    crop = rgb_image.crop((box.x, box.y, box.x + box.width, box.y + box.height))
                    resized = crop.resize(
                        (self._config.input_width, self._config.input_height),
                        resample=self._image_module.Resampling.BICUBIC,
                    )
                    normalized = _normalized_rgb_array(
                        resized,
                        expected_width=self._config.input_width,
                        expected_height=self._config.input_height,
                    )
                    image_tensors.append(self._torch.from_numpy(normalized))
            image_batch = self._torch.stack(image_tensors).to(self._config.device)
            mask_arrays = [
                self._torch.from_numpy(request.mask.astype("float32"))[None, None]
                for request in requests
            ]
            alpha_batch = self._torch.cat(mask_arrays, dim=0)
            alpha_batch = self._torch.nn.functional.interpolate(
                alpha_batch,
                size=(self._config.input_height, self._config.input_width),
                mode="nearest",
            )
            alpha_batch = (alpha_batch - 0.5) / 0.26
            alpha_batch = alpha_batch.to(self._config.device)
            if self._config.precision == "float16":
                image_batch = image_batch.half()
                alpha_batch = alpha_batch.half()
            else:
                image_batch = image_batch.float()
                alpha_batch = alpha_batch.float()

            _validate_batch_geometry(image_batch, alpha_batch)

            if self._config.device == "cuda":
                self._torch.cuda.reset_peak_memory_stats()
            with self._torch.inference_mode():
                encoded = self._model.visual(image_batch, alpha_batch)
            if self._config.device == "cuda":
                peak_memory_bytes = int(self._torch.cuda.max_memory_allocated())
            array = encoded.detach().to("cpu").numpy()
        except AlphaClipInferenceError:
            raise
        except Exception as error:
            raise AlphaClipInferenceError(
                f"AlphaCLIP inference failed for {image.payload_reference!r}: {error}"
            ) from error
        return AlphaClipNativeOutput(
            array=array,
            elapsed_seconds=time.perf_counter() - started_at,
            peak_memory_bytes=peak_memory_bytes,
        )

    def _ensure_loaded(self) -> None:
        """Import SDKs, verify local checkpoints, and load the model lazily."""
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            alpha_clip = importlib.import_module("alpha_clip")
            image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as error:
            raise AlphaClipDependencyError(
                "AlphaCLIP requires torch, alpha_clip, and Pillow in the runtime environment"
            ) from error

        if self._config.device == "cuda" and not torch.cuda.is_available():
            raise AlphaClipDeviceError("configured CUDA device is unavailable")
        if self._config.device == "mps" and not torch.backends.mps.is_available():
            raise AlphaClipDeviceError("configured MPS device is unavailable")
        if self._config.device == "cpu" and self._config.precision == "float16":
            raise AlphaClipDeviceError("float16 AlphaCLIP inference is not supported on CPU")

        base_path = _resolve_file(
            self._checkpoint_root,
            self._config.base_checkpoint_path,
            error_type=AlphaClipModelLoadError,
        )
        alpha_path = _resolve_file(
            self._checkpoint_root,
            self._config.alpha_checkpoint_path,
            error_type=AlphaClipModelLoadError,
        )
        actual_fingerprint = _combined_checkpoint_fingerprint(base_path, alpha_path)
        if actual_fingerprint != self._config.checkpoint_fingerprint:
            raise AlphaClipModelLoadError(
                f"checkpoint fingerprint mismatch: expected "
                f"{self._config.checkpoint_fingerprint}, found {actual_fingerprint}"
            )
        try:
            model, _preprocess = alpha_clip.load(
                str(base_path),
                alpha_vision_ckpt_pth=str(alpha_path),
                device=self._config.device,
            )
            model = model.half() if self._config.precision == "float16" else model.float()
            model.eval()
        except Exception as error:
            raise AlphaClipModelLoadError(
                f"could not load local AlphaCLIP checkpoints for {self._config.model_name}: {error}"
            ) from error
        self._torch = torch
        self._image_module = image_module
        self._model = model


def _request_for(
    *,
    image: PreparedImage,
    region: Region2D,
    mask: NDArray[Any],
    config: AlphaClipConfig,
) -> tuple[AlphaClipRequest, tuple[str, ...]]:
    """Validate/place a box-local mask and derive the configured RGB+alpha view."""
    import numpy as np

    if region.mask_reference is None:
        raise AlphaClipInferenceError(f"region {region.region_id!r} has no mask_reference")
    box = region.bounding_box
    expected_shape = box_mask_shape(box)
    if tuple(mask.shape) != expected_shape:
        raise AlphaClipInferenceError(
            f"mask shape {tuple(mask.shape)} does not match region box {expected_shape}"
        )
    if not np.issubdtype(mask.dtype, np.bool_):
        raise AlphaClipInferenceError(f"mask dtype must be boolean, got {mask.dtype!s}")
    if not np.any(mask):
        raise AlphaClipInferenceError(f"region {region.region_id!r} mask is empty")

    full_mask = np.zeros((image.height, image.width), dtype=np.bool_)
    raster_left = math.floor(box.x)
    raster_top = math.floor(box.y)
    target_left = max(0, raster_left)
    target_top = max(0, raster_top)
    target_right = min(image.width, raster_left + expected_shape[1])
    target_bottom = min(image.height, raster_top + expected_shape[0])
    source_left = target_left - raster_left
    source_top = target_top - raster_top
    copy_width = target_right - target_left
    copy_height = target_bottom - target_top
    if copy_width <= 0 or copy_height <= 0:
        raise AlphaClipInferenceError(f"region {region.region_id!r} has no support inside image")
    full_mask[
        target_top : target_top + copy_height,
        target_left : target_left + copy_width,
    ] = mask[
        source_top : source_top + copy_height,
        source_left : source_left + copy_width,
    ]
    if not np.any(full_mask):
        raise AlphaClipInferenceError(f"region {region.region_id!r} has no true mask inside image")

    warnings: list[str] = []
    if config.view_policy == "full_image":
        crop_box = BoundingBox2D(x=0, y=0, width=image.width, height=image.height)
        view_mask = full_mask
        policy = "full_image"
    else:
        crop_box, was_clipped = _context_box(image=image, box=box, config=config)
        view_mask = full_mask[
            crop_box.y : crop_box.y + crop_box.height,
            crop_box.x : crop_box.x + crop_box.width,
        ]
        policy = f"context_box:{config.context_padding_fraction:g}"
        if was_clipped:
            warnings.append(f"{region.region_id}: context view clipped to image")

    mask_hash = f"sha256:{hashlib.sha256(mask.tobytes()).hexdigest()}"
    transform_payload = {
        "source_observation_id": image.source_observation_id,
        "source_image": {"width": image.width, "height": image.height},
        "region_id": region.region_id,
        "region_box": {"x": box.x, "y": box.y, "width": box.width, "height": box.height},
        "source_mask_reference": region.mask_reference,
        "mask_content_hash": mask_hash,
        "crop": {
            "x": crop_box.x,
            "y": crop_box.y,
            "width": crop_box.width,
            "height": crop_box.height,
        },
        "view_policy": policy,
        "model_input": {"width": config.input_width, "height": config.input_height},
        "image_interpolation": config.image_interpolation,
        "mask_interpolation": config.mask_interpolation,
    }
    encoded = json.dumps(transform_payload, sort_keys=True, separators=(",", ":")).encode()
    view = AlphaClipView(
        source_region_id=region.region_id,
        source_mask_reference=region.mask_reference,
        mask_content_hash=mask_hash,
        crop_box=crop_box,
        view_policy=policy,
        source_image_width=image.width,
        source_image_height=image.height,
        model_input_width=config.input_width,
        model_input_height=config.input_height,
        image_resize_interpolation=config.image_interpolation,
        mask_resize_interpolation=config.mask_interpolation,
        coordinate_transform_id=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    )
    return AlphaClipRequest(view=view, mask=view_mask), tuple(warnings)


def _context_box(
    *, image: PreparedImage, box: BoundingBox2D, config: AlphaClipConfig
) -> tuple[BoundingBox2D, bool]:
    """Expand and clip a region box for the configured context view."""
    padding_x = box.width * config.context_padding_fraction
    padding_y = box.height * config.context_padding_fraction
    left = math.floor(box.x - padding_x)
    top = math.floor(box.y - padding_y)
    right = math.ceil(box.x + box.width + padding_x)
    bottom = math.ceil(box.y + box.height + padding_y)
    clipped = (max(0, left), max(0, top), min(image.width, right), min(image.height, bottom))
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise AlphaClipInferenceError("context view has no support inside image")
    return (
        BoundingBox2D(
            x=clipped[0],
            y=clipped[1],
            width=clipped[2] - clipped[0],
            height=clipped[3] - clipped[1],
        ),
        (left, top, right, bottom) != clipped,
    )


def _configuration_fingerprint(config: AlphaClipConfig) -> str:
    """Hash every effective model/execution/mask transformation parameter."""
    payload = {
        "model_name": config.model_name,
        "base_checkpoint_path": config.base_checkpoint_path,
        "alpha_checkpoint_path": config.alpha_checkpoint_path,
        "checkpoint_fingerprint": config.checkpoint_fingerprint,
        "device": config.device,
        "precision": config.precision,
        "input_width": config.input_width,
        "input_height": config.input_height,
        "view_policy": config.view_policy,
        "context_padding_fraction": config.context_padding_fraction,
        "image_interpolation": config.image_interpolation,
        "mask_interpolation": config.mask_interpolation,
        "l2_normalize": config.l2_normalize,
        "payload_prefix": config.payload_prefix,
        "code_version": config.code_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _normalized_rgb_array(image: Any, *, expected_width: int, expected_height: int) -> NDArray[Any]:
    """Apply only CLIP photometric normalization to an explicitly resized RGB view."""
    import numpy as np

    array = np.asarray(image, dtype=np.float32)
    expected_shape = (expected_height, expected_width, 3)
    if tuple(array.shape) != expected_shape:
        raise AlphaClipInferenceError(
            f"resized RGB shape {tuple(array.shape)} does not match {expected_shape}"
        )
    mean = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
    std = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
    normalized = (array / np.float32(255.0) - mean) / std
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


def _validate_batch_geometry(image_batch: Any, alpha_batch: Any) -> None:
    """Require RGB and alpha batches to share batch and spatial dimensions."""
    image_shape = tuple(int(dimension) for dimension in image_batch.shape)
    alpha_shape = tuple(int(dimension) for dimension in alpha_batch.shape)
    if (
        len(image_shape) != 4
        or len(alpha_shape) != 4
        or image_shape[0] != alpha_shape[0]
        or image_shape[2:] != alpha_shape[2:]
    ):
        raise AlphaClipInferenceError(
            "RGB and alpha batch geometry must match before inference: "
            f"RGB={image_shape}, alpha={alpha_shape}"
        )


def _view_provenance(config: AlphaClipConfig, view: AlphaClipView) -> BackendProvenance:
    """Combine backend configuration with exact region/mask/view identity."""
    payload = {
        "configuration_fingerprint": _configuration_fingerprint(config),
        "coordinate_transform_id": view.coordinate_transform_id,
        "source_region_id": view.source_region_id,
        "mask_content_hash": view.mask_content_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return BackendProvenance(
        backend_id="alphaclip_official",
        capability="feature_extractor",
        provider="SunzeY/AlphaCLIP",
        model=config.model_name,
        version=config.code_version,
        configuration_fingerprint=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    )


def _combined_checkpoint_fingerprint(base_path: Path, alpha_path: Path) -> str:
    """Hash base and alpha checkpoint bytes in an unambiguous order."""
    digest = hashlib.sha256()
    for label, path in ((b"base\0", base_path), (b"alpha\0", alpha_path)):
        digest.update(label)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _resolve_file(root: Path, reference: str, *, error_type: type[AlphaClipBackendError]) -> Path:
    """Resolve a required local file without allowing path escape."""
    candidate = (root / reference).resolve()
    if not candidate.is_relative_to(root):
        raise error_type(f"reference escapes configured root: {reference!r}")
    if not candidate.is_file():
        raise error_type(f"required local file does not exist: {reference!r}")
    return candidate

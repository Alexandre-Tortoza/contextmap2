"""DINOv2 dense-feature backend with canonical metadata and payload output.

The backend has no import-time dependency on PyTorch, Transformers, Pillow,
or NumPy. :class:`HuggingFaceDinoV2Runtime` loads those libraries and the
configured checkpoint only when inference is requested. Tests inject a
deterministic runtime, so contract behavior is verified without downloading
model weights or requiring an accelerator.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.dense_region_association import (
    DenseFeatureMap,
    DenseFeatureSampling,
)
from contextmap.visual_perception.embedding_space import (
    EmbeddingSpace,
    embedding_space_fingerprint,
)
from contextmap.visual_perception.identity import perception_result_id_for
from contextmap.visual_perception.models import (
    BackendProvenance,
    FeatureId,
    FeatureScope,
    PerceptionResultId,
    PerceptionRunId,
    PreparedImage,
    Region2D,
    VisualFeature,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DinoV2BackendError(RuntimeError):
    """Base class for explicit DINOv2 backend failures."""


class DinoV2DependencyError(DinoV2BackendError):
    """Raised when an optional DINOv2 runtime dependency is unavailable."""


class DinoV2DeviceError(DinoV2BackendError):
    """Raised when the configured execution device cannot be used."""


class DinoV2ModelLoadError(DinoV2BackendError):
    """Raised when the configured model/checkpoint cannot be loaded."""


class DinoV2InferenceError(DinoV2BackendError):
    """Raised when preprocessing, inference, or native-output validation fails."""


@dataclass(frozen=True, kw_only=True)
class DinoV2Config:
    """Effective configuration of one DINOv2 dense-feature backend.

    Attributes:
        checkpoint: Hugging Face model/checkpoint identity.
        revision: Exact model repository revision or commit.
        device: Requested PyTorch device: ``"cpu"``, ``"cuda"``, or ``"mps"``.
        precision: Inference and persisted payload dtype: ``"float32"`` or
            ``"float16"``.
        input_width: Model input width after deterministic direct resize.
        input_height: Model input height after deterministic direct resize.
        local_files_only: Whether model loading is prohibited from downloading.
            Defaults to ``True`` so execution never performs an implicit download.
        l2_normalize: Whether each patch vector is L2-normalized before persistence.
        payload_prefix: Artifact-relative directory for feature payloads.
        code_version: Version of this adapter's mapping policy.
    """

    checkpoint: str
    revision: str
    device: str = "cpu"
    precision: str = "float32"
    input_width: int = 224
    input_height: int = 224
    local_files_only: bool = True
    l2_normalize: bool = False
    payload_prefix: str = "features"
    code_version: str = "1"

    def __post_init__(self) -> None:
        """Validate reproducibility and execution settings.

        Raises:
            ValueError: If an identity, device, precision, dimension, or path
                setting is invalid.
        """
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        if not self.revision:
            raise ValueError("revision must not be empty")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if self.precision not in {"float32", "float16"}:
            raise ValueError("precision must be one of: float32, float16")
        if self.input_width <= 0:
            raise ValueError("input_width must be positive")
        if self.input_height <= 0:
            raise ValueError("input_height must be positive")
        prefix = Path(self.payload_prefix)
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("payload_prefix must be an artifact-relative path")
        if not self.code_version:
            raise ValueError("code_version must not be empty")


@dataclass(frozen=True, kw_only=True)
class DinoV2NativeOutput:
    """DINOv2 runtime output after native patch-token reshaping.

    This type remains inside the backend package; no PyTorch tensor crosses the
    runtime boundary.

    Attributes:
        array: NumPy array shaped ``(grid_height, grid_width, channels)``.
        model_input_width: Width actually supplied to the model.
        model_input_height: Height actually supplied to the model.
        patch_width: Native model patch width in model-input pixels.
        patch_height: Native model patch height in model-input pixels.
        register_token_count: Non-spatial register tokens removed before
            reshaping the patch grid.
    """

    array: NDArray[Any]
    model_input_width: int
    model_input_height: int
    patch_width: int
    patch_height: int
    register_token_count: int = 0

    def __post_init__(self) -> None:
        """Validate the native output has spatial and channel dimensions."""
        if self.array.ndim != 3 or any(dimension <= 0 for dimension in self.array.shape):
            raise ValueError("array must have shape (grid_height, grid_width, channels)")
        for name, value in (
            ("model_input_width", self.model_input_width),
            ("model_input_height", self.model_input_height),
            ("patch_width", self.patch_width),
            ("patch_height", self.patch_height),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.register_token_count < 0:
            raise ValueError("register_token_count must be non-negative")


@dataclass(frozen=True, kw_only=True)
class DinoV2Extraction:
    """Payload-rich result used by runtime composition and direct consumers.

    Attributes:
        dense_map: Canonical feature metadata and image-to-grid geometry.
        embedding_space: Full compatibility identity for the patch vectors.
        array: Native-resolution NumPy payload already sent to the configured
            artifact payload sink.
    """

    dense_map: DenseFeatureMap
    embedding_space: EmbeddingSpace
    array: NDArray[Any]


class DinoV2Runtime(Protocol):
    """Internal replaceable boundary around model-specific inference."""

    def infer(self, image: PreparedImage) -> DinoV2NativeOutput:
        """Produce native patch tokens for one prepared image."""
        ...


class FeaturePayloadSink(Protocol):
    """Minimal payload persistence surface required by the backend."""

    def add_feature_payload(
        self,
        feature: VisualFeature,
        source_observation_id: SourceObservationId,
        array: NDArray[Any],
    ) -> None:
        """Queue a feature payload for atomic run-artifact finalization."""
        ...


class DinoV2DenseFeatureBackend:
    """FeatureExtractor adapter producing native-resolution DINOv2 patch maps."""

    def __init__(
        self,
        *,
        config: DinoV2Config,
        run_id: PerceptionRunId,
        feature_stage_id: str,
        source_artifact_id: str,
        payload_sink: FeaturePayloadSink,
        runtime: DinoV2Runtime | None = None,
        prepared_image_root: Path | None = None,
    ) -> None:
        """Create a configured DINOv2 backend.

        Args:
            config: Effective model and execution configuration.
            run_id: Perception run owning produced feature identities.
            feature_stage_id: Pipeline stage identity used to namespace features
                within the complete ``PerceptionResult``.
            source_artifact_id: Run/artifact owning persisted feature payloads.
            payload_sink: Usually ``PerceptionRunWriter``; receives arrays before
                atomic artifact finalization.
            runtime: Optional injected runtime for tests or alternate SDK wiring.
            prepared_image_root: Root used to resolve image payload references by
                the default Hugging Face runtime.

        Raises:
            ValueError: If identities are empty or the default runtime has no
                prepared-image root.
        """
        if not str(run_id):
            raise ValueError("run_id must not be empty")
        if not feature_stage_id:
            raise ValueError("feature_stage_id must not be empty")
        if not source_artifact_id:
            raise ValueError("source_artifact_id must not be empty")
        if runtime is None and prepared_image_root is None:
            raise ValueError("prepared_image_root is required for the default runtime")
        self._config = config
        self._run_id = run_id
        self._feature_stage_id = feature_stage_id
        self._source_artifact_id = source_artifact_id
        self._payload_sink = payload_sink
        self._runtime = runtime or HuggingFaceDinoV2Runtime(
            config=config,
            prepared_image_root=prepared_image_root,
        )

    def backend_provenance(self) -> BackendProvenance:
        """Return exact backend/model/configuration provenance."""
        fingerprint = _configuration_fingerprint(self._config)
        return BackendProvenance(
            backend_id="dinov2_huggingface",
            capability="feature_extractor",
            provider="huggingface",
            model=self._config.checkpoint,
            version=self._config.revision,
            configuration_fingerprint=fingerprint,
        )

    def required_scope(self) -> FeatureScope:
        """Declare that this backend emits a whole-image dense map."""
        return FeatureScope.DENSE

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        """Extract, queue, and return one canonical dense feature metadata record.

        ``regions`` is intentionally unused because dense extraction depends only
        on the prepared image.
        """
        return (self.extract_dense(image).dense_map.feature,)

    def extract_dense(self, image: PreparedImage) -> DinoV2Extraction:
        """Extract a canonical dense map and queue its numerical payload.

        Args:
            image: Canonical prepared RGB input.

        Returns:
            Native-resolution payload, dense-map metadata, and embedding-space
            identity. The payload has also been sent to ``payload_sink``.

        Raises:
            DinoV2InferenceError: If native output contradicts configuration or
                cannot represent a valid patch grid.
            DinoV2BackendError: For explicit dependency, device, model, or
                inference failures from the runtime.
        """
        import numpy as np

        native = self._runtime.infer(image)
        _validate_native_output(native, self._config)
        array = native.array
        normalization = "none"
        if self._config.l2_normalize:
            norms = np.linalg.norm(array, axis=-1, keepdims=True)
            array = np.divide(array, norms, out=np.zeros_like(array), where=norms != 0)
            normalization = "l2"

        channels = int(array.shape[2])
        embedding_space = EmbeddingSpace(
            family="dinov2",
            model=self._config.checkpoint,
            version=self._config.revision,
            checkpoint=f"{self._config.checkpoint}@{self._config.revision}",
            layer=(
                "last_hidden_state.patch_tokens_after_cls_and_"
                f"{native.register_token_count}_registers"
            ),
            dimension=channels,
            normalization=normalization,
        )
        result_id = perception_result_id_for(
            run_id=self._run_id,
            source_observation_id=image.source_observation_id,
        )
        feature_id = _feature_id_for_stage(
            result_id=result_id,
            feature_stage_id=self._feature_stage_id,
            index=0,
        )
        feature = VisualFeature(
            feature_id=feature_id,
            scope=FeatureScope.DENSE,
            embedding_space_id=embedding_space_fingerprint(embedding_space),
            shape=tuple(array.shape),
            dtype=str(array.dtype),
            normalization=normalization,
            payload_reference=f"{self._config.payload_prefix}/{feature_id}.npy",
            provenance=self.backend_provenance(),
        )
        sampling = _sampling_for(image=image, native=native, config=self._config)
        dense_map = DenseFeatureMap(
            feature=feature,
            sampling=sampling,
            source_artifact_id=self._source_artifact_id,
        )
        self._payload_sink.add_feature_payload(feature, image.source_observation_id, array)
        return DinoV2Extraction(
            dense_map=dense_map,
            embedding_space=embedding_space,
            array=array,
        )


class HuggingFaceDinoV2Runtime:
    """Lazy PyTorch/Transformers implementation of DINOv2 native inference."""

    def __init__(self, *, config: DinoV2Config, prepared_image_root: Path | None) -> None:
        """Create a lazy runtime without importing SDKs or loading weights."""
        if prepared_image_root is None:
            raise ValueError("prepared_image_root must not be None")
        self._config = config
        self._prepared_image_root = prepared_image_root.resolve()
        self._torch: Any = None
        self._image_module: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._torch_dtype: Any = None

    def infer(self, image: PreparedImage) -> DinoV2NativeOutput:
        """Load lazily, preprocess one image, and return NumPy patch tokens.

        Raises:
            DinoV2DependencyError: If PyTorch, Transformers, or Pillow is absent.
            DinoV2DeviceError: If the configured device is unavailable.
            DinoV2ModelLoadError: If the configured checkpoint cannot load.
            DinoV2InferenceError: If image decoding/preprocessing/inference fails.
        """
        self._ensure_loaded()
        image_path = _resolve_within_root(self._prepared_image_root, image.payload_reference)
        try:
            with self._image_module.open(image_path) as loaded_image:
                rgb_image = loaded_image.convert("RGB")
                if rgb_image.size != (image.width, image.height):
                    raise DinoV2InferenceError(
                        f"prepared image metadata {(image.width, image.height)} does not match "
                        f"decoded payload {rgb_image.size}"
                    )
                inputs = self._processor(
                    images=rgb_image,
                    return_tensors="pt",
                    do_resize=True,
                    size={"height": self._config.input_height, "width": self._config.input_width},
                    do_center_crop=False,
                )
            pixel_values = inputs["pixel_values"].to(
                device=self._config.device,
                dtype=self._torch_dtype,
            )
            with self._torch.no_grad():
                output = self._model(pixel_values=pixel_values)
            patch_width, patch_height = _patch_dimensions(self._model.config.patch_size)
            grid_width = int(pixel_values.shape[3]) // patch_width
            grid_height = int(pixel_values.shape[2]) // patch_height
            patch_tokens, register_count = _patch_tokens_and_register_count(
                output.last_hidden_state,
                self._model.config,
            )
            expected_tokens = grid_width * grid_height
            if int(patch_tokens.shape[1]) != expected_tokens:
                raise DinoV2InferenceError(
                    f"model returned {patch_tokens.shape[1]} patch tokens; expected "
                    f"{expected_tokens} for grid {(grid_height, grid_width)}"
                )
            array = (
                patch_tokens[0]
                .reshape(grid_height, grid_width, int(patch_tokens.shape[2]))
                .detach()
                .to("cpu")
                .numpy()
            )
        except DinoV2InferenceError:
            raise
        except Exception as error:
            raise DinoV2InferenceError(
                f"DINOv2 inference failed for {image.payload_reference!r}: {error}"
            ) from error

        return DinoV2NativeOutput(
            array=array,
            model_input_width=int(pixel_values.shape[3]),
            model_input_height=int(pixel_values.shape[2]),
            patch_width=patch_width,
            patch_height=patch_height,
            register_token_count=register_count,
        )

    def _ensure_loaded(self) -> None:
        """Import optional SDKs, validate the device, and load configured weights."""
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as error:
            raise DinoV2DependencyError(
                "DINOv2 requires torch, transformers, and Pillow in the runtime environment"
            ) from error

        if self._config.device == "cuda" and not torch.cuda.is_available():
            raise DinoV2DeviceError("configured CUDA device is unavailable")
        if self._config.device == "mps" and not torch.backends.mps.is_available():
            raise DinoV2DeviceError("configured MPS device is unavailable")
        if self._config.device == "cpu" and self._config.precision == "float16":
            raise DinoV2DeviceError("float16 DINOv2 inference is not supported on CPU")

        torch_dtype = getattr(torch, self._config.precision)
        try:
            processor = transformers.AutoImageProcessor.from_pretrained(
                self._config.checkpoint,
                revision=self._config.revision,
                local_files_only=self._config.local_files_only,
            )
            model = transformers.AutoModel.from_pretrained(
                self._config.checkpoint,
                revision=self._config.revision,
                local_files_only=self._config.local_files_only,
                torch_dtype=torch_dtype,
            )
            model = model.to(self._config.device)
            model.eval()
        except Exception as error:
            mode = "local cache" if self._config.local_files_only else "configured model source"
            raise DinoV2ModelLoadError(
                f"could not load {self._config.checkpoint}@{self._config.revision} from {mode}: "
                f"{error}"
            ) from error

        self._torch = torch
        self._image_module = image_module
        self._processor = processor
        self._model = model
        self._torch_dtype = torch_dtype


def _configuration_fingerprint(config: DinoV2Config) -> str:
    """Hash the complete effective backend configuration."""
    payload = {
        "checkpoint": config.checkpoint,
        "revision": config.revision,
        "device": config.device,
        "precision": config.precision,
        "input_width": config.input_width,
        "input_height": config.input_height,
        "local_files_only": config.local_files_only,
        "l2_normalize": config.l2_normalize,
        "payload_prefix": config.payload_prefix,
        "preprocessing": "huggingface_direct_resize_no_crop_v1",
        "code_version": config.code_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _feature_id_for_stage(
    *, result_id: PerceptionResultId, feature_stage_id: str, index: int
) -> FeatureId:
    """Namespace a feature identity by the composing pipeline stage."""
    stage_digest = hashlib.sha256(feature_stage_id.encode("utf-8")).hexdigest()
    return FeatureId(f"{result_id}--feature-stage-{stage_digest}-{index:04d}")


def _patch_tokens_and_register_count(last_hidden_state: Any, model_config: Any) -> tuple[Any, int]:
    """Remove CLS and the register-token count declared by the model config."""
    register_count = getattr(model_config, "num_register_tokens", 0)
    if isinstance(register_count, bool) or not isinstance(register_count, int):
        raise DinoV2InferenceError("model num_register_tokens must be an integer")
    if register_count < 0:
        raise DinoV2InferenceError("model num_register_tokens must be non-negative")
    return last_hidden_state[:, 1 + register_count :, :], register_count


def _validate_native_output(native: DinoV2NativeOutput, config: DinoV2Config) -> None:
    """Validate SDK output against configured preprocessing and patch geometry."""
    if (native.model_input_width, native.model_input_height) != (
        config.input_width,
        config.input_height,
    ):
        raise DinoV2InferenceError(
            "runtime model input dimensions do not match configured direct resize: "
            f"{(native.model_input_width, native.model_input_height)} vs "
            f"{(config.input_width, config.input_height)}"
        )
    expected_grid = (
        native.model_input_height // native.patch_height,
        native.model_input_width // native.patch_width,
    )
    if tuple(native.array.shape[:2]) != expected_grid:
        raise DinoV2InferenceError(
            f"runtime grid shape {tuple(native.array.shape[:2])} does not match "
            f"input/patch geometry {expected_grid}"
        )
    if str(native.array.dtype) != config.precision:
        raise DinoV2InferenceError(
            f"runtime dtype {native.array.dtype!s} does not match precision {config.precision!r}"
        )


def _sampling_for(
    *, image: PreparedImage, native: DinoV2NativeOutput, config: DinoV2Config
) -> DenseFeatureSampling:
    """Map native patch footprints back into prepared-image coordinates."""
    scale_x = image.width / native.model_input_width
    scale_y = image.height / native.model_input_height
    transform_payload = {
        "prepared_image": {
            "width": image.width,
            "height": image.height,
            "transformations": list(image.transformations),
        },
        "model_input": {
            "width": native.model_input_width,
            "height": native.model_input_height,
            "resize": "direct_bilinear_by_huggingface_processor",
            "center_crop": False,
        },
        "patch": {"width": native.patch_width, "height": native.patch_height},
        "token_layout": {
            "class_token_count": 1,
            "register_token_count": native.register_token_count,
        },
        "config_fingerprint": _configuration_fingerprint(config),
    }
    encoded = json.dumps(transform_payload, sort_keys=True, separators=(",", ":")).encode()
    transform_id = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    return DenseFeatureSampling(
        grid_width=int(native.array.shape[1]),
        grid_height=int(native.array.shape[0]),
        source_image_width=image.width,
        source_image_height=image.height,
        origin_x=0.0,
        origin_y=0.0,
        stride_x=native.patch_width * scale_x,
        stride_y=native.patch_height * scale_y,
        support_width=native.patch_width * scale_x,
        support_height=native.patch_height * scale_y,
        coordinate_transform_id=transform_id,
    )


def _patch_dimensions(patch_size: Any) -> tuple[int, int]:
    """Normalize a scalar or two-axis model patch size into width and height."""
    if isinstance(patch_size, int):
        return patch_size, patch_size
    if isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
        patch_height, patch_width = patch_size
        return int(patch_width), int(patch_height)
    raise DinoV2InferenceError(f"unsupported DINOv2 patch_size: {patch_size!r}")


def _resolve_within_root(root: Path, reference: str) -> Path:
    """Resolve an artifact-relative prepared-image reference safely."""
    candidate = (root / reference).resolve()
    if not candidate.is_relative_to(root):
        raise DinoV2InferenceError(f"prepared image reference escapes its root: {reference!r}")
    if not candidate.is_file():
        raise DinoV2InferenceError(f"prepared image payload does not exist: {reference!r}")
    return candidate

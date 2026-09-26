"""DINOv3 dense-feature backend with explicit register-token handling.

PyTorch, Transformers, Pillow, NumPy, and model weights are loaded lazily.
Contract tests inject a deterministic runtime and therefore require no network,
accelerator, or real checkpoint.
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
from contextmap.visual_perception.backends._feature_values import (
    validate_and_normalize_feature_values,
)
from contextmap.visual_perception.backends._huggingface import (
    preprocess_pixel_values,
    validate_huggingface_commit_revision,
)
from contextmap.visual_perception.dense_region_association import (
    DenseFeatureMap,
    DenseFeatureSampling,
)
from contextmap.visual_perception.embedding_space import (
    EmbeddingSpace,
    embedding_space_fingerprint,
)
from contextmap.visual_perception.identity import feature_id_for, perception_result_id_for
from contextmap.visual_perception.models import (
    BackendProvenance,
    FeatureScope,
    PerceptionRunId,
    PreparedImage,
    Region2D,
    VisualFeature,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DinoV3BackendError(RuntimeError):
    """Base class for explicit DINOv3 backend failures."""


class DinoV3DependencyError(DinoV3BackendError):
    """Raised when an optional DINOv3 runtime dependency is unavailable."""


class DinoV3DeviceError(DinoV3BackendError):
    """Raised when the configured device/precision cannot be used."""


class DinoV3ModelLoadError(DinoV3BackendError):
    """Raised when the configured DINOv3 checkpoint cannot load."""


class DinoV3InferenceError(DinoV3BackendError):
    """Raised for preprocessing, inference, or native-output failures."""


@dataclass(frozen=True, kw_only=True)
class DinoV3Config:
    """Effective configuration of one DINOv3 dense extractor.

    Attributes:
        checkpoint: Hugging Face model/checkpoint identity.
        revision: Immutable repository revision as a full Git commit SHA.
        device: Requested ``"cpu"``, ``"cuda"``, or ``"mps"`` device.
        precision: Inference/payload dtype, ``"float32"`` or ``"float16"``.
        input_width: Width after deterministic direct resize.
        input_height: Height after deterministic direct resize.
        local_files_only: Prohibit implicit model downloads when true.
        l2_normalize: L2-normalize each patch vector before persistence.
        payload_prefix: Artifact-relative feature payload directory.
        code_version: Adapter mapping-policy version.
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
        """Validate execution identity and configuration."""
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        validate_huggingface_commit_revision(self.revision)
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if self.precision not in {"float32", "float16"}:
            raise ValueError("precision must be one of: float32, float16")
        if self.input_width <= 0:
            raise ValueError("input_width must be positive")
        if self.input_height <= 0:
            raise ValueError("input_height must be positive")
        if not self.payload_prefix:
            raise ValueError("payload_prefix must not be empty")
        prefix = Path(self.payload_prefix)
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("payload_prefix must be an artifact-relative path")
        if not self.code_version:
            raise ValueError("code_version must not be empty")


@dataclass(frozen=True, kw_only=True)
class DinoV3NativeOutput:
    """SDK-free patch output after removing CLS and register tokens.

    Attributes:
        array: Native patch map shaped ``(height, width, channels)``.
        model_input_width: Width actually supplied to the model.
        model_input_height: Height actually supplied to the model.
        patch_width: Native patch width in model-input pixels.
        patch_height: Native patch height in model-input pixels.
        register_token_count: Register tokens removed before reshaping.
    """

    array: NDArray[Any]
    model_input_width: int
    model_input_height: int
    patch_width: int
    patch_height: int
    register_token_count: int

    def __post_init__(self) -> None:
        """Validate spatial dimensions and register-token metadata."""
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
class DinoV3Extraction:
    """Payload-rich DINOv3 result for runtime composition.

    Attributes:
        dense_map: Canonical dense feature metadata and sampling geometry.
        embedding_space: Full patch-vector compatibility identity.
        array: Native-resolution patch payload sent to the artifact sink.
        register_token_count: Non-spatial tokens excluded from ``array``.
    """

    dense_map: DenseFeatureMap
    embedding_space: EmbeddingSpace
    array: NDArray[Any]
    register_token_count: int


class DinoV3Runtime(Protocol):
    """Internal model-specific inference boundary."""

    def infer(self, image: PreparedImage) -> DinoV3NativeOutput:
        """Return native spatial patch tokens for one prepared image."""
        ...


class FeaturePayloadSink(Protocol):
    """Minimal atomic feature-payload sink used by the adapter."""

    def add_feature_payload(
        self,
        feature: VisualFeature,
        source_observation_id: SourceObservationId,
        array: NDArray[Any],
    ) -> None:
        """Queue one payload for run-artifact finalization."""
        ...


class DinoV3DenseFeatureBackend:
    """FeatureExtractor adapter for native-resolution DINOv3 patch maps."""

    def __init__(
        self,
        *,
        config: DinoV3Config,
        run_id: PerceptionRunId,
        feature_stage_id: str,
        source_artifact_id: str,
        payload_sink: FeaturePayloadSink,
        runtime: DinoV3Runtime | None = None,
        prepared_image_root: Path | None = None,
    ) -> None:
        """Create a configured backend with an injected or default runtime."""
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
        self._runtime = runtime or HuggingFaceDinoV3Runtime(
            config=config,
            prepared_image_root=prepared_image_root,
        )

    def backend_provenance(self) -> BackendProvenance:
        """Report model, revision, adapter, and effective configuration."""
        return BackendProvenance(
            backend_id="dinov3_huggingface",
            capability="feature_extractor",
            provider="huggingface",
            model=self._config.checkpoint,
            version=self._config.revision,
            configuration_fingerprint=_configuration_fingerprint(self._config),
        )

    def required_scope(self) -> FeatureScope:
        """Declare whole-image dense extraction."""
        return FeatureScope.DENSE

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        """Extract/persist one dense feature; region inputs are unused."""
        return (self.extract_dense(image).dense_map.feature,)

    def extract_dense(self, image: PreparedImage) -> DinoV3Extraction:
        """Extract and queue one canonical native-resolution DINOv3 map."""
        native = self._runtime.infer(image)
        _validate_native_output(native, self._config)
        array, normalization = validate_and_normalize_feature_values(
            native.array,
            l2_normalize=self._config.l2_normalize,
            error_type=DinoV3InferenceError,
        )

        embedding_space = EmbeddingSpace(
            family="dinov3",
            model=self._config.checkpoint,
            version=self._config.revision,
            checkpoint=f"{self._config.checkpoint}@{self._config.revision}",
            layer="last_hidden_state.patch_tokens_after_registers",
            dimension=int(array.shape[2]),
            normalization=normalization,
        )
        result_id = perception_result_id_for(
            run_id=self._run_id,
            source_observation_id=image.source_observation_id,
        )
        feature_id = feature_id_for(
            result_id=result_id,
            producer_id=self._feature_stage_id,
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
        dense_map = DenseFeatureMap(
            feature=feature,
            sampling=_sampling_for(image=image, native=native, config=self._config),
            source_artifact_id=self._source_artifact_id,
        )
        self._payload_sink.add_feature_payload(feature, image.source_observation_id, array)
        return DinoV3Extraction(
            dense_map=dense_map,
            embedding_space=embedding_space,
            array=array,
            register_token_count=native.register_token_count,
        )


class HuggingFaceDinoV3Runtime:
    """Lazy Hugging Face implementation with explicit register-token removal."""

    def __init__(self, *, config: DinoV3Config, prepared_image_root: Path | None) -> None:
        """Create the runtime without importing SDKs or loading weights."""
        if prepared_image_root is None:
            raise ValueError("prepared_image_root must not be None")
        self._config = config
        self._prepared_image_root = prepared_image_root.resolve()
        self._torch: Any = None
        self._image_module: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._torch_dtype: Any = None

    def infer(self, image: PreparedImage) -> DinoV3NativeOutput:
        """Decode/preprocess/infer and return only spatial patch tokens."""
        self._ensure_loaded()
        image_path = _resolve_within_root(self._prepared_image_root, image.payload_reference)
        try:
            with self._image_module.open(image_path) as loaded_image:
                rgb_image = loaded_image.convert("RGB")
                if rgb_image.size != (image.width, image.height):
                    raise DinoV3InferenceError(
                        f"prepared image metadata {(image.width, image.height)} does not match "
                        f"decoded payload {rgb_image.size}"
                    )
                pixel_values = preprocess_pixel_values(
                    processor=self._processor,
                    images=[rgb_image],
                    width=self._config.input_width,
                    height=self._config.input_height,
                    resample=self._image_module.Resampling.BILINEAR,
                ).to(
                    device=self._config.device,
                    dtype=self._torch_dtype,
                )
            with self._torch.inference_mode():
                output = self._model(pixel_values=pixel_values)
            patch_width, patch_height = _patch_dimensions(self._model.config.patch_size)
            register_count = int(self._model.config.num_register_tokens)
            grid_width = int(pixel_values.shape[3]) // patch_width
            grid_height = int(pixel_values.shape[2]) // patch_height
            patch_tokens = output.last_hidden_state[:, 1 + register_count :, :]
            expected_tokens = grid_width * grid_height
            if int(patch_tokens.shape[1]) != expected_tokens:
                raise DinoV3InferenceError(
                    f"model returned {patch_tokens.shape[1]} patch tokens after excluding "
                    f"{register_count} registers; expected {expected_tokens}"
                )
            array = (
                patch_tokens[0]
                .reshape(grid_height, grid_width, int(patch_tokens.shape[2]))
                .detach()
                .to("cpu")
                .numpy()
            )
        except DinoV3InferenceError:
            raise
        except Exception as error:
            raise DinoV3InferenceError(
                f"DINOv3 inference failed for {image.payload_reference!r}: {error}"
            ) from error

        return DinoV3NativeOutput(
            array=array,
            model_input_width=int(pixel_values.shape[3]),
            model_input_height=int(pixel_values.shape[2]),
            patch_width=patch_width,
            patch_height=patch_height,
            register_token_count=register_count,
        )

    def _ensure_loaded(self) -> None:
        """Import optional SDKs, validate device, and load exact weights lazily."""
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as error:
            raise DinoV3DependencyError(
                "DINOv3 requires torch, transformers, and Pillow (plus torchvision with "
                "transformers 5.x) in the runtime environment"
            ) from error

        if self._config.device == "cuda" and not torch.cuda.is_available():
            raise DinoV3DeviceError("configured CUDA device is unavailable")
        if self._config.device == "mps" and not torch.backends.mps.is_available():
            raise DinoV3DeviceError("configured MPS device is unavailable")
        if self._config.device == "cpu" and self._config.precision == "float16":
            raise DinoV3DeviceError("float16 DINOv3 inference is not supported on CPU")

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
                dtype=torch_dtype,
            )
            model = model.to(self._config.device)
            model.eval()
        except ImportError as error:
            raise DinoV3DependencyError(
                f"DINOv3 could not import a package required by the Hugging Face image "
                f"processor or model (transformers 5.x needs torchvision): {error}"
            ) from error
        except Exception as error:
            mode = "local cache" if self._config.local_files_only else "configured model source"
            raise DinoV3ModelLoadError(
                f"could not load {self._config.checkpoint}@{self._config.revision} from {mode}: "
                f"{error}"
            ) from error

        self._torch = torch
        self._image_module = image_module
        self._processor = processor
        self._model = model
        self._torch_dtype = torch_dtype


def _configuration_fingerprint(config: DinoV3Config) -> str:
    """Hash every effective scientific/execution parameter."""
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
        "preprocessing": "pillow_direct_bilinear_resize_processor_normalize_no_crop_v2",
        "token_policy": "drop_cls_and_configured_registers_v1",
        "code_version": config.code_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validate_native_output(native: DinoV3NativeOutput, config: DinoV3Config) -> None:
    """Validate runtime output against configured preprocessing geometry."""
    if (native.model_input_width, native.model_input_height) != (
        config.input_width,
        config.input_height,
    ):
        raise DinoV3InferenceError("runtime model input dimensions do not match configuration")
    expected_grid = (
        native.model_input_height // native.patch_height,
        native.model_input_width // native.patch_width,
    )
    if tuple(native.array.shape[:2]) != expected_grid:
        raise DinoV3InferenceError(
            f"runtime grid shape {tuple(native.array.shape[:2])} does not match "
            f"input/patch geometry {expected_grid}"
        )
    if str(native.array.dtype) != config.precision:
        raise DinoV3InferenceError(
            f"runtime dtype {native.array.dtype!s} does not match precision {config.precision!r}"
        )


def _sampling_for(
    *, image: PreparedImage, native: DinoV3NativeOutput, config: DinoV3Config
) -> DenseFeatureSampling:
    """Map the native patch grid back to prepared-image coordinates."""
    scale_x = image.width / native.model_input_width
    scale_y = image.height / native.model_input_height
    transform_payload = {
        "prepared_image": {
            "width": image.width,
            "height": image.height,
            "transformations": [record.to_dict() for record in image.transformations],
        },
        "model_input": {
            "width": native.model_input_width,
            "height": native.model_input_height,
            "resize": "direct_bilinear_by_pillow",
            "center_crop": False,
        },
        "patch": {"width": native.patch_width, "height": native.patch_height},
        "register_token_count": native.register_token_count,
        "config_fingerprint": _configuration_fingerprint(config),
    }
    encoded = json.dumps(transform_payload, sort_keys=True, separators=(",", ":")).encode()
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
        coordinate_transform_id=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    )


def _patch_dimensions(patch_size: Any) -> tuple[int, int]:
    """Normalize scalar/two-axis patch metadata to width and height."""
    if isinstance(patch_size, int):
        return patch_size, patch_size
    if isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
        patch_height, patch_width = patch_size
        return int(patch_width), int(patch_height)
    raise DinoV3InferenceError(f"unsupported DINOv3 patch_size: {patch_size!r}")


def _resolve_within_root(root: Path, reference: str) -> Path:
    """Resolve a prepared-image payload without allowing path escape."""
    candidate = (root / reference).resolve()
    if not candidate.is_relative_to(root):
        raise DinoV3InferenceError(f"prepared image reference escapes its root: {reference!r}")
    if not candidate.is_file():
        raise DinoV3InferenceError(f"prepared image payload does not exist: {reference!r}")
    return candidate

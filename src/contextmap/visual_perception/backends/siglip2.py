"""Direct SigLIP2 vision-encoder feature backend for Eagle-compatible experiments.

The adapter runs a fixed-resolution ("FixRes") SigLIP2 checkpoint through the
Transformers ``SiglipVisionModel`` — the same vision tower Eagle 2.5 uses — and
never extracts anything through Eagle's language decoder. Each adapter instance
has exactly one scope: ``DENSE`` publishes the post-layernorm patch tokens as a
native-resolution :class:`DenseFeatureMap`; ``GLOBAL`` publishes the vector of
the attention-pooling head. Region-scoped features are not implemented.

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

PREPROCESSING_POLICY = "processor_rescale_normalize_no_crop_v1"
"""Version of the pixel path after the Pillow resize: processor rescale and normalization only."""

RESIZE_FILTERS = frozenset({"bicubic", "bilinear"})
"""Pillow filters of the direct square resize.

``bicubic`` is the Eagle 2.5 image transform; ``bilinear`` is the resampling the
published SigLIP2 processor configuration declares.
"""

_OUTPUT_SELECTION = {
    FeatureScope.DENSE: "last_hidden_state.post_layernorm_patch_tokens",
    FeatureScope.GLOBAL: "pooler_output.attention_pooling_head",
}
"""Which encoder output one adapter scope reads."""


class Siglip2BackendError(RuntimeError):
    """Base class for explicit SigLIP2 backend failures."""


class Siglip2DependencyError(Siglip2BackendError):
    """Raised when an optional SigLIP2 runtime dependency is unavailable."""


class Siglip2DeviceError(Siglip2BackendError):
    """Raised when the configured device/precision cannot be used."""


class Siglip2ModelLoadError(Siglip2BackendError):
    """Raised when the configured checkpoint cannot load or is not a FixRes SigLIP2 model."""


class Siglip2InferenceError(Siglip2BackendError):
    """Raised for preprocessing, inference, or native-output failures."""


class Siglip2ScopeError(Siglip2BackendError):
    """Raised when an operation asks for a scope the adapter instance was not configured for."""


@dataclass(frozen=True, kw_only=True)
class Siglip2Config:
    """Effective configuration of one SigLIP2 feature extractor.

    Attributes:
        checkpoint: Hugging Face identity of a fixed-resolution SigLIP2 checkpoint.
        revision: Immutable repository revision as a full Git commit SHA.
        scope: The only scope this instance produces, ``DENSE`` or ``GLOBAL``.
        input_size: Side of the square model input, in pixels. It must equal the
            checkpoint's native ``image_size``: position embeddings are never
            interpolated.
        device: Requested ``"cpu"``, ``"cuda"``, or ``"mps"`` device.
        precision: Inference/payload dtype, ``"float32"`` or ``"float16"``.
        resize_filter: Pillow filter of the direct square resize, ``"bicubic"``
            (Eagle 2.5 image transform) or ``"bilinear"`` (SigLIP2 processor).
        local_files_only: Prohibit implicit model downloads when true.
        l2_normalize: L2-normalize each output vector before persistence.
        payload_prefix: Artifact-relative feature payload directory.
        code_version: Adapter mapping-policy version.
    """

    checkpoint: str
    revision: str
    scope: FeatureScope
    input_size: int
    device: str = "cpu"
    precision: str = "float32"
    resize_filter: str = "bicubic"
    local_files_only: bool = True
    l2_normalize: bool = False
    payload_prefix: str = "features"
    code_version: str = "1"

    def __post_init__(self) -> None:
        """Validate execution identity, scope, and preprocessing configuration."""
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        validate_huggingface_commit_revision(self.revision)
        if self.scope not in _OUTPUT_SELECTION:
            raise ValueError(
                "scope must be DENSE or GLOBAL; region-scoped SigLIP2 features are not implemented"
            )
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if self.precision not in {"float32", "float16"}:
            raise ValueError("precision must be one of: float32, float16")
        if self.resize_filter not in RESIZE_FILTERS:
            raise ValueError(f"resize_filter must be one of: {', '.join(sorted(RESIZE_FILTERS))}")
        if not self.payload_prefix:
            raise ValueError("payload_prefix must not be empty")
        prefix = Path(self.payload_prefix)
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("payload_prefix must be an artifact-relative path")
        if not self.code_version:
            raise ValueError("code_version must not be empty")

    @property
    def preprocessing_id(self) -> str:
        """Identity of the whole pixel path from the prepared image to the model input."""
        return (
            f"pillow_direct_{self.resize_filter}_resize_{self.input_size}x{self.input_size}"
            f"_{PREPROCESSING_POLICY}"
        )


@dataclass(frozen=True, kw_only=True)
class Siglip2NativeOutput:
    """SDK-free encoder output for one prepared image.

    Attributes:
        array: ``(grid_height, grid_width, channels)`` patch tokens for a ``DENSE``
            instance, or the ``(channels,)`` pooled vector for a ``GLOBAL`` one.
        model_input_width: Width actually supplied to the model.
        model_input_height: Height actually supplied to the model.
        patch_width: Native patch width in model-input pixels.
        patch_height: Native patch height in model-input pixels.
    """

    array: NDArray[Any]
    model_input_width: int
    model_input_height: int
    patch_width: int
    patch_height: int

    def __post_init__(self) -> None:
        """Validate array rank and positive model-input/patch geometry."""
        if self.array.ndim not in {1, 3} or any(dimension <= 0 for dimension in self.array.shape):
            raise ValueError(
                "array must have shape (grid_height, grid_width, channels) or (channels,)"
            )
        for name, value in (
            ("model_input_width", self.model_input_width),
            ("model_input_height", self.model_input_height),
            ("patch_width", self.patch_width),
            ("patch_height", self.patch_height),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, kw_only=True)
class Siglip2DenseExtraction:
    """Payload-rich ``DENSE`` result for runtime composition and direct consumers.

    Attributes:
        dense_map: Canonical dense feature metadata and sampling geometry.
        embedding_space: Full patch-vector compatibility identity.
        array: Native-resolution patch payload sent to the artifact sink.
    """

    dense_map: DenseFeatureMap
    embedding_space: EmbeddingSpace
    array: NDArray[Any]


@dataclass(frozen=True, kw_only=True)
class Siglip2GlobalExtraction:
    """Payload-rich ``GLOBAL`` result for direct consumers.

    Attributes:
        feature: Canonical whole-image feature.
        embedding_space: Full pooled-vector compatibility identity.
        array: ``(channels,)`` payload sent to the artifact sink.
    """

    feature: VisualFeature
    embedding_space: EmbeddingSpace
    array: NDArray[Any]


class Siglip2Runtime(Protocol):
    """Internal model-specific inference boundary."""

    def infer(self, image: PreparedImage) -> Siglip2NativeOutput:
        """Return the configured scope's native encoder output for one prepared image."""
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


class Siglip2FeatureBackend:
    """FeatureExtractor adapter for native SigLIP2 patch maps or pooled image vectors."""

    def __init__(
        self,
        *,
        config: Siglip2Config,
        run_id: PerceptionRunId,
        feature_stage_id: str,
        source_artifact_id: str,
        payload_sink: FeaturePayloadSink,
        runtime: Siglip2Runtime | None = None,
        prepared_image_root: Path | None = None,
    ) -> None:
        """Create a configured backend with an injected or default runtime.

        Args:
            config: Effective model, scope, preprocessing and execution configuration.
            run_id: Perception run owning produced feature identities.
            feature_stage_id: Pipeline stage identity namespacing the features.
            source_artifact_id: Run/artifact owning persisted feature payloads.
            payload_sink: Usually ``PerceptionRunWriter``; receives arrays before
                atomic artifact finalization.
            runtime: Optional injected runtime for tests or alternate SDK wiring.
            prepared_image_root: Root resolving prepared-image payload references
                for the default Hugging Face runtime.

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
        self._runtime = runtime or HuggingFaceSiglip2Runtime(
            config=config,
            prepared_image_root=prepared_image_root,
        )

    def backend_provenance(self) -> BackendProvenance:
        """Report model, revision, adapter, and effective configuration."""
        return BackendProvenance(
            backend_id="siglip2_huggingface",
            capability="feature_extractor",
            provider="huggingface",
            model=self._config.checkpoint,
            version=self._config.revision,
            configuration_fingerprint=_configuration_fingerprint(self._config),
        )

    def required_scope(self) -> FeatureScope:
        """Return the one configured scope, ``DENSE`` or ``GLOBAL``."""
        return self._config.scope

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        """Extract/persist the configured scope's one feature; region inputs are unused."""
        if self._config.scope is FeatureScope.DENSE:
            return (self.extract_dense(image).dense_map.feature,)
        return (self.extract_global(image).feature,)

    def extract_dense(self, image: PreparedImage) -> Siglip2DenseExtraction:
        """Extract and queue one canonical native-resolution patch map.

        Raises:
            Siglip2ScopeError: If this instance is not configured for ``DENSE``.
            Siglip2InferenceError: If the runtime output disagrees with the
                configured geometry/precision or holds invalid values.
        """
        self._require_scope(FeatureScope.DENSE)
        native, array, embedding_space, feature = self._encode(image)
        dense_map = DenseFeatureMap(
            feature=feature,
            sampling=_sampling_for(image=image, native=native, config=self._config),
            source_artifact_id=self._source_artifact_id,
        )
        self._payload_sink.add_feature_payload(feature, image.source_observation_id, array)
        return Siglip2DenseExtraction(
            dense_map=dense_map, embedding_space=embedding_space, array=array
        )

    def extract_global(self, image: PreparedImage) -> Siglip2GlobalExtraction:
        """Extract and queue one canonical attention-pooled image vector.

        Raises:
            Siglip2ScopeError: If this instance is not configured for ``GLOBAL``.
            Siglip2InferenceError: If the runtime output disagrees with the
                configured geometry/precision or holds invalid values.
        """
        self._require_scope(FeatureScope.GLOBAL)
        _, array, embedding_space, feature = self._encode(image)
        self._payload_sink.add_feature_payload(feature, image.source_observation_id, array)
        return Siglip2GlobalExtraction(
            feature=feature, embedding_space=embedding_space, array=array
        )

    def _require_scope(self, scope: FeatureScope) -> None:
        """Refuse an extraction of a scope this instance does not produce."""
        if self._config.scope is not scope:
            raise Siglip2ScopeError(
                f"this SigLIP2 instance is configured for {self._config.scope.name}, "
                f"not {scope.name}"
            )

    def _encode(
        self, image: PreparedImage
    ) -> tuple[Siglip2NativeOutput, NDArray[Any], EmbeddingSpace, VisualFeature]:
        """Run the encoder and build the validated payload with its canonical identity.

        Nothing reaches the payload sink here: the caller queues the payload only
        after every contract built from it is valid.
        """
        native = self._runtime.infer(image)
        _validate_native_output(native, self._config)
        array, normalization = validate_and_normalize_feature_values(
            native.array,
            l2_normalize=self._config.l2_normalize,
            error_type=Siglip2InferenceError,
        )
        embedding_space = _embedding_space(
            self._config, dimension=int(array.shape[-1]), normalization=normalization
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
            scope=self._config.scope,
            embedding_space_id=embedding_space_fingerprint(embedding_space),
            shape=tuple(array.shape),
            dtype=str(array.dtype),
            normalization=normalization,
            payload_reference=f"{self._config.payload_prefix}/{feature_id}.npy",
            provenance=self.backend_provenance(),
        )
        return native, array, embedding_space, feature


class HuggingFaceSiglip2Runtime:
    """Lazy Transformers ``SiglipVisionModel`` runtime for FixRes SigLIP2 checkpoints."""

    def __init__(self, *, config: Siglip2Config, prepared_image_root: Path | None) -> None:
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

    def infer(self, image: PreparedImage) -> Siglip2NativeOutput:
        """Decode, preprocess, encode, and return the configured scope's output."""
        self._ensure_loaded()
        image_path = _resolve_within_root(self._prepared_image_root, image.payload_reference)
        try:
            with self._image_module.open(image_path) as loaded_image:
                rgb_image = loaded_image.convert("RGB")
                if rgb_image.size != (image.width, image.height):
                    raise Siglip2InferenceError(
                        f"prepared image metadata {(image.width, image.height)} does not match "
                        f"decoded payload {rgb_image.size}"
                    )
                pixel_values = preprocess_pixel_values(
                    processor=self._processor,
                    images=[rgb_image],
                    width=self._config.input_size,
                    height=self._config.input_size,
                    resample=getattr(
                        self._image_module.Resampling, self._config.resize_filter.upper()
                    ),
                ).to(
                    device=self._config.device,
                    dtype=self._torch_dtype,
                )
            with self._torch.inference_mode():
                output = self._model(pixel_values=pixel_values)
            patch_width, patch_height = _axis_pair(self._model.config.patch_size, "patch_size")
            if self._config.scope is FeatureScope.DENSE:
                array = _patch_grid(
                    output.last_hidden_state,
                    grid_width=int(pixel_values.shape[3]) // patch_width,
                    grid_height=int(pixel_values.shape[2]) // patch_height,
                )
            else:
                if output.pooler_output is None:
                    raise Siglip2InferenceError(
                        "checkpoint has no attention pooling head (vision_use_head=False); "
                        "it cannot produce GLOBAL SigLIP2 features"
                    )
                array = output.pooler_output[0].detach().to("cpu").numpy()
        except Siglip2InferenceError:
            raise
        except Exception as error:
            raise Siglip2InferenceError(
                f"SigLIP2 inference failed for {image.payload_reference!r}: {error}"
            ) from error

        return Siglip2NativeOutput(
            array=array,
            model_input_width=int(pixel_values.shape[3]),
            model_input_height=int(pixel_values.shape[2]),
            patch_width=patch_width,
            patch_height=patch_height,
        )

    def _ensure_loaded(self) -> None:
        """Import optional SDKs, validate device and checkpoint, and load weights lazily."""
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as error:
            raise Siglip2DependencyError(
                "SigLIP2 requires torch, transformers, and Pillow (plus torchvision with "
                "transformers 5.x) in the runtime environment"
            ) from error

        if self._config.device == "cuda" and not torch.cuda.is_available():
            raise Siglip2DeviceError("configured CUDA device is unavailable")
        if self._config.device == "mps" and not torch.backends.mps.is_available():
            raise Siglip2DeviceError("configured MPS device is unavailable")
        if self._config.device == "cpu" and self._config.precision == "float16":
            raise Siglip2DeviceError("float16 SigLIP2 inference is not supported on CPU")

        torch_dtype = getattr(torch, self._config.precision)
        source = {
            "revision": self._config.revision,
            "local_files_only": self._config.local_files_only,
        }
        try:
            # A configuração vem antes dos pesos: um checkpoint NaFlex ou de outra resolução
            # nativa é recusado sem carregar nenhum tensor.
            checkpoint_config = transformers.AutoConfig.from_pretrained(
                self._config.checkpoint, **source
            )
            _require_fixed_resolution_checkpoint(checkpoint_config, self._config)
            processor = transformers.AutoImageProcessor.from_pretrained(
                self._config.checkpoint, **source
            )
            model = transformers.SiglipVisionModel.from_pretrained(
                self._config.checkpoint, dtype=torch_dtype, **source
            )
            model = model.to(self._config.device)
            model.eval()
        except Siglip2ModelLoadError:
            raise
        except ImportError as error:
            raise Siglip2DependencyError(
                f"SigLIP2 could not import a package required by the Hugging Face image "
                f"processor or model (transformers 5.x needs torchvision): {error}"
            ) from error
        except Exception as error:
            mode = "local cache" if self._config.local_files_only else "configured model source"
            raise Siglip2ModelLoadError(
                f"could not load {self._config.checkpoint}@{self._config.revision} from {mode}: "
                f"{error}"
            ) from error

        self._torch = torch
        self._image_module = image_module
        self._processor = processor
        self._model = model
        self._torch_dtype = torch_dtype


def _require_fixed_resolution_checkpoint(checkpoint_config: Any, config: Siglip2Config) -> None:
    """Accept only a FixRes checkpoint whose native resolution is the configured input.

    NaFlex checkpoints (``model_type="siglip2"``) need an aspect-preserving,
    processor-owned resize into a padded patch sequence with attention masks and
    per-image spatial shapes; this adapter implements only the fixed square path.
    """
    model_type = getattr(checkpoint_config, "model_type", None)
    if model_type != "siglip":
        raise Siglip2ModelLoadError(
            f"{config.checkpoint} has model_type {model_type!r}; this adapter only runs "
            f"fixed-resolution SigLIP2 checkpoints (model_type 'siglip'), not NaFlex ones"
        )
    native = _axis_pair(checkpoint_config.vision_config.image_size, "image_size")
    if native != (config.input_size, config.input_size):
        raise Siglip2ModelLoadError(
            f"{config.checkpoint} has native image_size {native[0]}x{native[1]} but input_size is "
            f"{config.input_size}; this adapter never interpolates position embeddings"
        )


def _patch_grid(tokens: Any, *, grid_width: int, grid_height: int) -> NDArray[Any]:
    """Reshape row-major patch tokens (SigLIP has no CLS token) into the native grid."""
    expected = grid_width * grid_height
    if int(tokens.shape[1]) != expected:
        raise Siglip2InferenceError(
            f"model returned {tokens.shape[1]} patch tokens; expected {expected} for a "
            f"{grid_height}x{grid_width} grid"
        )
    array: NDArray[Any] = (
        tokens[0].reshape(grid_height, grid_width, int(tokens.shape[2])).detach().to("cpu").numpy()
    )
    return array


def _embedding_space(
    config: Siglip2Config, *, dimension: int, normalization: str
) -> EmbeddingSpace:
    """Identify the vector space of one scope, including the input pixel path.

    ``EmbeddingSpace`` has no dedicated preprocessing field, so the preprocessing
    identity qualifies ``layer``: the same encoder output read under another
    resize or input resolution is a different space.
    """
    return EmbeddingSpace(
        family="siglip2",
        model=config.checkpoint,
        version=config.revision,
        checkpoint=f"{config.checkpoint}@{config.revision}",
        layer=f"{_OUTPUT_SELECTION[config.scope]}|preprocessing={config.preprocessing_id}",
        dimension=dimension,
        normalization=normalization,
    )


def _configuration_fingerprint(config: Siglip2Config) -> str:
    """Hash every effective scientific/execution parameter."""
    payload = {
        "checkpoint": config.checkpoint,
        "revision": config.revision,
        "scope": config.scope.value,
        "output_selection": _OUTPUT_SELECTION[config.scope],
        "input_size": config.input_size,
        "device": config.device,
        "precision": config.precision,
        "preprocessing": config.preprocessing_id,
        "local_files_only": config.local_files_only,
        "l2_normalize": config.l2_normalize,
        "payload_prefix": config.payload_prefix,
        "code_version": config.code_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validate_native_output(native: Siglip2NativeOutput, config: Siglip2Config) -> None:
    """Validate runtime output against the configured scope, geometry, and precision."""
    if (native.model_input_width, native.model_input_height) != (
        config.input_size,
        config.input_size,
    ):
        raise Siglip2InferenceError("runtime model input dimensions do not match configuration")
    if config.scope is FeatureScope.DENSE:
        expected_grid = (
            native.model_input_height // native.patch_height,
            native.model_input_width // native.patch_width,
        )
        if native.array.ndim != 3 or tuple(native.array.shape[:2]) != expected_grid:
            raise Siglip2InferenceError(
                f"runtime grid shape {tuple(native.array.shape)} does not match "
                f"input/patch geometry {expected_grid}"
            )
    elif native.array.ndim != 1:
        raise Siglip2InferenceError(
            f"runtime global output shape {tuple(native.array.shape)} is not one pooled vector"
        )
    if str(native.array.dtype) != config.precision:
        raise Siglip2InferenceError(
            f"runtime dtype {native.array.dtype!s} does not match precision {config.precision!r}"
        )


def _sampling_for(
    *, image: PreparedImage, native: Siglip2NativeOutput, config: Siglip2Config
) -> DenseFeatureSampling:
    """Map the native patch grid back to prepared-image coordinates.

    Each cell's stride and support are ``patch * image / model_input`` per axis,
    computed with a single rounding so a synthetic geometry is reproduced exactly.
    """
    stride_x = native.patch_width * image.width / native.model_input_width
    stride_y = native.patch_height * image.height / native.model_input_height
    transform_payload = {
        "prepared_image": {
            "width": image.width,
            "height": image.height,
            "transformations": list(image.transformations),
        },
        "model_input": {
            "width": native.model_input_width,
            "height": native.model_input_height,
            "preprocessing": config.preprocessing_id,
        },
        "patch": {"width": native.patch_width, "height": native.patch_height},
        "token_policy": "all_last_hidden_state_tokens_are_patches_row_major_v1",
    }
    encoded = json.dumps(transform_payload, sort_keys=True, separators=(",", ":")).encode()
    return DenseFeatureSampling(
        grid_width=int(native.array.shape[1]),
        grid_height=int(native.array.shape[0]),
        source_image_width=image.width,
        source_image_height=image.height,
        origin_x=0.0,
        origin_y=0.0,
        stride_x=stride_x,
        stride_y=stride_y,
        # Patches ViT não se sobrepõem: o suporte de cada célula é o seu próprio passo.
        support_width=stride_x,
        support_height=stride_y,
        coordinate_transform_id=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    )


def _axis_pair(value: Any, name: str) -> tuple[int, int]:
    """Normalize scalar/two-axis ``(height, width)`` model metadata to ``(width, height)``."""
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = value
        return int(width), int(height)
    raise Siglip2InferenceError(f"unsupported SigLIP2 {name}: {value!r}")


def _resolve_within_root(root: Path, reference: str) -> Path:
    """Resolve a prepared-image payload without allowing path escape."""
    candidate = (root / reference).resolve()
    if not candidate.is_relative_to(root):
        raise Siglip2InferenceError(f"prepared image reference escapes its root: {reference!r}")
    if not candidate.is_file():
        raise Siglip2InferenceError(f"prepared image payload does not exist: {reference!r}")
    return candidate

"""Qwen adapter for canonical scene and region Semantic Interpretation."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

from contextmap.visual_perception.backends._huggingface import (
    validate_huggingface_commit_revision,
)
from contextmap.visual_perception.backends._semantic_views import read_view_payload
from contextmap.visual_perception.models import (
    BackendProvenance,
    SemanticInferenceProvenance,
)
from contextmap.visual_perception.region_models import JsonScalar
from contextmap.visual_perception.semantic_backend import (
    SemanticBackendDiagnostics,
    SemanticInterpretationExecution,
    semantic_failure_from_parse_error,
)
from contextmap.visual_perception.semantic_prompt import (
    SemanticConfidencePolicy,
    SemanticPromptTemplate,
    SemanticResponseParseError,
    parse_semantic_response,
    render_semantic_prompt,
)
from contextmap.visual_perception.semantic_requests import (
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreterCapabilities,
    SemanticVisualView,
    VisualViewKind,
    validate_semantic_request,
)


class QwenBackendError(RuntimeError):
    """Base class for explicit Qwen runtime failures."""


class QwenDependencyError(QwenBackendError):
    """Raised when an optional Qwen runtime dependency is unavailable."""


class QwenDeviceError(QwenBackendError):
    """Raised when the configured device, precision, or quantization cannot be used."""


class QwenModelLoadError(QwenBackendError):
    """Raised when the configured checkpoint cannot load as configured."""


class QwenInferenceError(QwenBackendError):
    """Raised for view loading, generation, or decoding failures."""


@dataclass(frozen=True, kw_only=True)
class QwenSemanticConfig:
    """Effective local Qwen generation configuration.

    Attributes:
        model: Hugging Face checkpoint identity, for example ``Qwen/Qwen3-VL-4B-Instruct``.
        device: Torch device string. Quantization requires a CUDA device.
        precision: Torch dtype name for the non-quantized weights and the 4-bit compute
            dtype: ``float32``, ``float16`` or ``bfloat16``.
        max_new_tokens: Generation token limit.
        temperature: ``0`` selects greedy decoding; a positive value samples with it, using
            the checkpoint's own ``top_p``/``top_k`` defaults.
        quantization: ``None`` loads the checkpoint at ``precision``; ``"4bit"`` applies
            bitsandbytes NF4 (weights in 4 bits, compute in ``precision``); ``"8bit"``
            applies bitsandbytes LLM.int8. Quantization changes outputs and cost, so it is
            part of the configuration fingerprint.
        revision: Immutable Hugging Face commit SHA. The transformers runtime requires it
            so that every output is traceable to exact weights; the seam itself does not.
    """

    model: str
    device: str
    precision: str
    max_new_tokens: int
    temperature: float
    quantization: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        """Validate model/runtime settings before model construction."""
        for name, value in (
            ("model", self.model),
            ("device", self.device),
            ("precision", self.precision),
        ):
            if not value.strip():
                raise ValueError(f"Qwen {name} must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("Qwen max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("Qwen temperature must be non-negative")
        if self.quantization not in {None, "4bit", "8bit"}:
            raise ValueError("Qwen quantization must be None, '4bit', or '8bit'")
        if self.revision is not None:
            validate_huggingface_commit_revision(self.revision)

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return a secret-free, JSON-compatible effective configuration."""
        return cast_config(asdict(self))


@dataclass(frozen=True, kw_only=True)
class QwenGenerationResponse:
    """SDK-neutral result returned by the injected local Qwen runtime."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    peak_memory_bytes: int | None = None
    warnings: tuple[str, ...] = ()


class QwenRuntime(Protocol):
    """Internal boundary isolating transformers/torch/Qwen native objects."""

    def generate(
        self,
        *,
        visual_views: tuple[SemanticVisualView, ...],
        prompt: str,
        config: QwenSemanticConfig,
    ) -> QwenGenerationResponse:
        """Generate one structured response from the exact canonical views.

        An implementation must obtain each view's bytes through ``read_view_payload`` (or
        an equivalent check) and must not decode a payload whose SHA-256 differs from
        ``SemanticVisualView.sha256``: the request identifies its evidence by that hash.
        """
        ...


class QwenSemanticInterpreter:
    """Interpret canonical semantic requests with an explicitly configured Qwen runtime."""

    def __init__(self, *, config: QwenSemanticConfig, runtime: QwenRuntime) -> None:
        """Construct the adapter without loading a model outside the injected runtime."""
        self._config = config
        self._runtime = runtime
        encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        self.configuration_fingerprint = (
            "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        )

    def backend_provenance(self) -> BackendProvenance:
        """Return Qwen model and effective-configuration identity."""
        return BackendProvenance(
            backend_id="qwen_semantic",
            capability="semantic_interpreter",
            provider="qwen",
            model=self._config.model,
            version="1",
            configuration_fingerprint=self.configuration_fingerprint,
        )

    def capabilities(self) -> SemanticInterpreterCapabilities:
        """Declare the canonical evidence this adapter maps to Qwen inputs."""
        return SemanticInterpreterCapabilities(
            supported_modes=frozenset(
                {SemanticInterpretationMode.SCENE, SemanticInterpretationMode.REGION}
            ),
            supported_view_kinds=frozenset(VisualViewKind),
            accepts_visual_features=False,
            accepts_scene_context=False,
        )

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        """Render, execute, and parse one request without fabricating confidence."""
        validate_semantic_request(request, self.capabilities())
        if request.configuration_fingerprint != self.configuration_fingerprint:
            raise ValueError("Qwen request configuration fingerprint does not match adapter")
        template = SemanticPromptTemplate.default_for(request.mode)
        rendered = render_semantic_prompt(
            request,
            template,
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        )
        started = time.monotonic()
        response = self._runtime.generate(
            visual_views=request.visual_views,
            prompt=rendered.text,
            config=self._config,
        )
        latency_ms = (time.monotonic() - started) * 1000
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity=f"qwen-{request.mode.value}-interpretation",
            prompt_template_id=template.template_id,
            output_schema_version=template.output_schema_version,
            raw_response_reference=(
                f"debug/40-semantic-interpretation/{request.request_id}/raw-response.txt"
            ),
        )
        diagnostics = SemanticBackendDiagnostics(
            latency_ms=latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            peak_memory_bytes=response.peak_memory_bytes,
            warnings=response.warnings,
        )
        configuration: Mapping[str, JsonScalar] = MappingProxyType(self._config.to_dict())
        try:
            parsed = parse_semantic_response(
                response.text,
                request,
                provenance,
                confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
            )
        except SemanticResponseParseError as error:
            # A resposta foi realmente observada: preserva-la e o ponto. Reduzi-la a
            # str(error) perderia o que o modelo disse, que e a evidencia que permite
            # diagnosticar truncamento, drift de schema ou prompt mal especificado.
            raise semantic_failure_from_parse_error(
                error,
                request=request,
                rendered_prompt=rendered,
                raw_response=response.text,
                provenance=provenance,
                diagnostics=diagnostics,
                effective_configuration=configuration,
            ) from error
        return SemanticInterpretationExecution(
            request=request,
            rendered_prompt=rendered,
            raw_response=response.text,
            parsed=parsed,
            diagnostics=diagnostics,
            effective_configuration=configuration,
        )


_PRECISIONS = frozenset({"float32", "float16", "bfloat16"})
_DEVICE_TYPES = frozenset({"cpu", "cuda", "mps"})


class HuggingFaceQwenRuntime:
    """Lazy transformers implementation of :class:`QwenRuntime`.

    The runtime is bound to one effective configuration and loads the exact pinned
    checkpoint on the first request. Quantization is applied through bitsandbytes at load
    time and verified afterwards, so a configuration that says ``4bit`` can never silently
    run in full precision. Nothing leaves the process: weights come from the local
    Hugging Face cache unless ``local_files_only`` is disabled explicitly.
    """

    def __init__(
        self,
        *,
        config: QwenSemanticConfig,
        view_root: Path,
        local_files_only: bool = True,
    ) -> None:
        """Bind the runtime to a configuration without importing SDKs or loading weights.

        Args:
            config: Effective configuration; it must pin an immutable ``revision``.
            view_root: Directory containing the run-relative ``outputs/semantic-views/``.
            local_files_only: Refuse implicit downloads when true.

        Raises:
            ValueError: If the configuration does not pin a checkpoint revision.
        """
        if config.revision is None:
            raise ValueError("Qwen revision must be pinned to a commit SHA for the runtime")
        self._config = config
        self._view_root = view_root
        self._local_files_only = local_files_only
        self._torch: Any = None
        self._image_module: Any = None
        self._processor: Any = None
        self._model: Any = None

    def generate(
        self,
        *,
        visual_views: tuple[SemanticVisualView, ...],
        prompt: str,
        config: QwenSemanticConfig,
    ) -> QwenGenerationResponse:
        """Generate one response from the exact views followed by the canonical prompt.

        Every view is decoded from bytes whose SHA-256 was verified against
        ``SemanticVisualView.sha256`` first, so a payload that changed after the request was
        built is rejected instead of being interpreted.

        Raises:
            ValueError: If ``config`` differs from the configuration the runtime is bound to.
            QwenBackendError: For dependency, device, load, or inference failures, including
                a view payload that is missing, escapes the view root, or does not match its
                recorded SHA-256.
        """
        if config != self._config:
            raise ValueError("Qwen runtime was built for another configuration")
        self.load()
        torch = self._torch
        on_cuda = config.device.startswith("cuda")
        try:
            content: list[dict[str, Any]] = []
            for view in visual_views:
                payload = read_view_payload(self._view_root, view)
                with self._image_module.open(io.BytesIO(payload)) as image:
                    content.append({"type": "image", "image": image.convert("RGB")})
            content.append({"type": "text", "text": prompt})
            inputs = self._processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(config.device)
            prompt_tokens = int(inputs["input_ids"].shape[1])
            if on_cuda:
                torch.cuda.reset_peak_memory_stats(config.device)
            with torch.inference_mode():
                generated = self._model.generate(**inputs, **_generation_settings(config))
            peak_memory = int(torch.cuda.max_memory_allocated(config.device)) if on_cuda else None
            new_tokens = generated[:, prompt_tokens:]
            text = self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        except Exception as error:
            references = [view.payload_reference for view in visual_views]
            raise QwenInferenceError(
                f"Qwen generation failed for views {references}: {error}"
            ) from error
        output_tokens = int(new_tokens.shape[1])
        warnings = (
            (
                f"generation reached max_new_tokens={config.max_new_tokens}; "
                "the structured response may be truncated",
            )
            if output_tokens >= config.max_new_tokens
            else ()
        )
        return QwenGenerationResponse(
            text=text,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            peak_memory_bytes=peak_memory,
            warnings=warnings,
        )

    def load(self) -> None:
        """Import optional SDKs, validate the device, and load the exact weights once.

        ``generate`` calls this lazily. A caller that measures latency calls it first, so
        that the one-time load is not attributed to the first request.

        Raises:
            QwenBackendError: For dependency, device, or checkpoint loading failures.
        """
        if self._model is not None:
            return
        config = self._config
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as error:
            raise QwenDependencyError(
                "Qwen requires torch, transformers, and Pillow in the runtime environment"
            ) from error
        if config.precision not in _PRECISIONS:
            raise QwenDeviceError(f"Qwen precision must be one of {sorted(_PRECISIONS)}")
        # O índice (cuda:1) continua permitido; o tipo segue o conjunto dos demais adapters.
        device_type = config.device.split(":", 1)[0]
        if device_type not in _DEVICE_TYPES:
            raise QwenDeviceError(
                f"Qwen device type must be one of: cpu, cuda, mps; got {config.device!r}"
            )
        on_cuda = device_type == "cuda"
        if on_cuda and not torch.cuda.is_available():
            raise QwenDeviceError("configured CUDA device is unavailable")
        if device_type == "mps" and not torch.backends.mps.is_available():
            raise QwenDeviceError("configured MPS device is unavailable")
        if config.quantization is not None and not on_cuda:
            raise QwenDeviceError("bitsandbytes quantization requires a CUDA device")
        if config.device == "cpu" and config.precision == "float16":
            raise QwenDeviceError("float16 Qwen inference is not supported on CPU")

        dtype = getattr(torch, config.precision)
        load_kwargs: dict[str, Any] = {
            "revision": config.revision,
            "local_files_only": self._local_files_only,
            "dtype": dtype,
        }
        if config.quantization == "4bit":
            load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        elif config.quantization == "8bit":
            load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
        if config.quantization is not None:
            # O bitsandbytes quantiza durante o load, então o modelo já nasce no device.
            load_kwargs["device_map"] = {"": config.device}
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                config.model,
                revision=config.revision,
                local_files_only=self._local_files_only,
            )
            model = transformers.AutoModelForImageTextToText.from_pretrained(
                config.model, **load_kwargs
            )
            if config.quantization is None:
                model = model.to(config.device)
            model = model.eval()
        except ImportError as error:
            raise QwenDependencyError(
                "Qwen could not import a package required by the Hugging Face processor "
                f"or by bitsandbytes quantization: {error}"
            ) from error
        except Exception as error:
            source = "local cache" if self._local_files_only else "configured model source"
            raise QwenModelLoadError(
                f"could not load {config.model}@{config.revision} from {source}: {error}"
            ) from error
        applied = getattr(model.config, "quantization_config", None)
        if config.quantization is not None and applied is None:
            raise QwenModelLoadError(
                f"{config.quantization} quantization was requested for {config.model} but was "
                "not applied by the loaded model"
            )
        self._torch = torch
        self._image_module = image_module
        self._processor = processor
        self._model = model


def _generation_settings(config: QwenSemanticConfig) -> dict[str, Any]:
    """Map the configured temperature to explicit greedy or sampling generation settings."""
    if config.temperature > 0:
        return {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": True,
            "temperature": config.temperature,
        }
    # Decoding guloso: anula os defaults de amostragem do generation_config do checkpoint,
    # que de outra forma seriam aplicados e emitiriam avisos.
    return {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }


def cast_config(value: dict[str, object]) -> dict[str, JsonScalar]:
    """Narrow dataclass serialization to the scalar Qwen config contract."""
    return {key: cast(JsonScalar, item) for key, item in value.items()}

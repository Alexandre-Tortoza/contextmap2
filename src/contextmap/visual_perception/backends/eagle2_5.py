"""Eagle 2.5 adapter for canonical scene and region Semantic Interpretation.

Eagle 2.5 (NVlabs) is a long-context VLM: a SigLIP2 vision tower whose image tiles are
projected into a Qwen2.5 language model. This adapter consumes the backend-neutral
:class:`~contextmap.visual_perception.SemanticInterpretationRequest` exactly like the Qwen
and Gemini adapters: it renders the prompt policy the request selects, hands the exact
views and that prompt to an injected runtime, and parses the answer through the canonical
parser. No Eagle-specific evidence type exists; only the runtime seam knows the model.
"""

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
    SemanticResponseParseError,
    parse_semantic_response,
    render_semantic_prompt,
    semantic_prompt_template,
)
from contextmap.visual_perception.semantic_requests import (
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreterCapabilities,
    SemanticVisualView,
    VisualViewKind,
    validate_semantic_request,
)


class EagleBackendError(RuntimeError):
    """Base class for explicit Eagle 2.5 runtime failures."""


class EagleDependencyError(EagleBackendError):
    """Raised when a package the runtime or the checkpoint's remote code imports is missing."""


class EagleDeviceError(EagleBackendError):
    """Raised when the configured device or precision cannot be used."""


class EagleModelLoadError(EagleBackendError):
    """Raised when the configured checkpoint cannot load as configured."""


class EagleInferenceError(EagleBackendError):
    """Raised for view loading, visual-budget, generation, or decoding failures."""


@dataclass(frozen=True, kw_only=True)
class EagleSemanticConfig:
    """Effective Eagle 2.5 generation and visual-input configuration.

    The tiling fields are the visual-input budget the ``eagle_2_5_vl`` processor exposes.
    Each view is resized onto a grid of ``448 x 448`` tiles whose count lies between
    ``min_dynamic_tiles`` and ``max_dynamic_tiles`` (the grid that best preserves the view's
    area and aspect ratio), plus one whole-view thumbnail tile when ``use_thumbnail`` is set
    and the grid has more than one tile. Every tile costs a fixed number of language tokens
    (256 for the published checkpoints), so the budget bounds both pixels and context. The
    runtime always passes these values explicitly: the checkpoint's own processor defaults
    never apply silently.

    Attributes:
        model: Hugging Face checkpoint identity, for example ``nvidia/Eagle2.5-8B``.
        device: Torch device string.
        precision: Torch dtype name of the weights: ``float32``, ``float16`` or ``bfloat16``.
        max_new_tokens: Generation token limit.
        temperature: ``0`` selects greedy decoding; a positive value samples with it.
        max_dynamic_tiles: Upper bound of the dynamic tiles of one view.
        min_dynamic_tiles: Lower bound of the dynamic tiles of one view.
        use_thumbnail: Append the whole-view thumbnail tile to a multi-tile view.
        revision: Immutable Hugging Face commit SHA. It also pins the checkpoint's remote
            code, which the transformers runtime executes; the runtime requires it, the seam
            itself does not.
    """

    model: str
    device: str
    precision: str
    max_new_tokens: int
    temperature: float
    max_dynamic_tiles: int
    min_dynamic_tiles: int = 1
    use_thumbnail: bool = True
    revision: str | None = None

    def __post_init__(self) -> None:
        """Validate model, generation, and visual-budget settings before model construction."""
        for name, value in (
            ("model", self.model),
            ("device", self.device),
            ("precision", self.precision),
        ):
            if not value.strip():
                raise ValueError(f"Eagle {name} must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("Eagle max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("Eagle temperature must be non-negative")
        if self.max_dynamic_tiles < 1:
            raise ValueError("Eagle max_dynamic_tiles must be at least 1")
        if not 1 <= self.min_dynamic_tiles <= self.max_dynamic_tiles:
            raise ValueError("Eagle min_dynamic_tiles must be between 1 and max_dynamic_tiles")
        if self.revision is not None:
            validate_huggingface_commit_revision(self.revision)

    @property
    def max_tiles_per_view(self) -> int:
        """Return the most tiles one view can become under this budget, thumbnail included."""
        # O processor só acrescenta o thumbnail quando a grade escolhida tem mais de uma tile.
        thumbnail = 1 if self.use_thumbnail and self.max_dynamic_tiles > 1 else 0
        return self.max_dynamic_tiles + thumbnail

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return a secret-free, JSON-compatible effective configuration."""
        return {key: cast(JsonScalar, value) for key, value in asdict(self).items()}


@dataclass(frozen=True, kw_only=True)
class EagleGenerationResponse:
    """SDK-neutral result returned by the injected Eagle 2.5 runtime."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    peak_memory_bytes: int | None = None
    warnings: tuple[str, ...] = ()


class EagleRuntime(Protocol):
    """Internal boundary isolating transformers/torch and Eagle's remote-code objects."""

    def generate(
        self,
        *,
        visual_views: tuple[SemanticVisualView, ...],
        prompt: str,
        config: EagleSemanticConfig,
    ) -> EagleGenerationResponse:
        """Generate one structured response from the exact canonical views, in order.

        An implementation must obtain each view's bytes through ``read_view_payload`` (or
        an equivalent check) and must not decode a payload whose SHA-256 differs from
        ``SemanticVisualView.sha256``: the request identifies its evidence by that hash. It
        must apply the visual budget of ``config`` and must not add instructions of its own.
        """
        ...


class EagleSemanticInterpreter:
    """Interpret canonical semantic requests with an explicitly configured Eagle 2.5 runtime."""

    def __init__(self, *, config: EagleSemanticConfig, runtime: EagleRuntime) -> None:
        """Construct the adapter without loading a model outside the injected runtime."""
        self._config = config
        self._runtime = runtime
        encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        self.configuration_fingerprint = (
            "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        )

    def backend_provenance(self) -> BackendProvenance:
        """Return Eagle model and effective-configuration identity."""
        return BackendProvenance(
            backend_id="eagle2_5_semantic",
            capability="semantic_interpreter",
            provider="nvidia",
            model=self._config.model,
            version="1",
            configuration_fingerprint=self.configuration_fingerprint,
        )

    def capabilities(self) -> SemanticInterpreterCapabilities:
        """Declare the canonical evidence this adapter maps to Eagle 2.5 inputs.

        Every view kind is an ordinary image to the model, and several views become several
        images of one message. Visual features and scene-context conditioning are not mapped
        to model inputs, so requests carrying them are refused before inference.
        """
        return SemanticInterpreterCapabilities(
            supported_modes=frozenset(
                {SemanticInterpretationMode.SCENE, SemanticInterpretationMode.REGION}
            ),
            supported_view_kinds=frozenset(VisualViewKind),
            accepts_visual_features=False,
            accepts_scene_context=False,
        )

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        """Render the request's own prompt policy, execute it, and parse the response.

        The prompt is the catalog template ``request.prompt_template_id`` names, rendered for
        this request; the adapter has no default of its own. Confidence is never fabricated:
        a number the model reports about itself is rejected as uncalibrated.

        Raises:
            ValueError: Before inference, if the request is unsupported, was built for another
                configuration, or names a prompt template that is unknown or whose mode or
                output schema differs from the request's.
            EagleBackendError: If the runtime cannot load the model or run the request.
            SemanticInterpretationFailedError: If the observed response cannot be parsed; it
                carries the raw response, the rendered prompt and the diagnostics.
        """
        validate_semantic_request(request, self.capabilities())
        if request.configuration_fingerprint != self.configuration_fingerprint:
            raise ValueError("Eagle request configuration fingerprint does not match adapter")
        # A política de prompt é a que o request seleciona, nunca um padrão do backend:
        # identidade desconhecida, modo ou schema divergente falham aqui, antes da inferência.
        rendered = render_semantic_prompt(
            request,
            semantic_prompt_template(request.prompt_template_id),
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
            task_identity=f"eagle2_5-{request.mode.value}-interpretation",
            prompt_template_id=rendered.template_id,
            output_schema_version=rendered.output_schema_version,
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
            # A resposta foi observada: ela é evidência e sobrevive à falha do parser, com o
            # prompt, a proveniência e os diagnostics que permitem auditá-la.
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


class HuggingFaceEagleRuntime:
    """Lazy transformers implementation of :class:`EagleRuntime`.

    It follows the loading and inference path NVlabs documents for Eagle 2.5:
    ``AutoProcessor``/``AutoModel`` with ``trust_remote_code=True``, the checkpoint's chat
    template, and the processor's dynamic tiling. The checkpoint's Python code is executed,
    so it is only ever loaded at the pinned immutable revision, and by default only from the
    local Hugging Face cache: nothing is downloaded implicitly.
    """

    def __init__(
        self,
        *,
        config: EagleSemanticConfig,
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
            raise ValueError("Eagle revision must be pinned to a commit SHA for the runtime")
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
        config: EagleSemanticConfig,
    ) -> EagleGenerationResponse:
        """Generate one response from the exact views, in order, followed by the prompt.

        Every view is decoded from bytes whose SHA-256 was verified against
        ``SemanticVisualView.sha256`` first. The tiles the processor produced are checked
        against the configured budget before the model runs.

        Raises:
            ValueError: If ``config`` differs from the configuration the runtime is bound to.
            EagleBackendError: For dependency, device, load, budget, or inference failures,
                including a view payload that is missing, escapes the view root, or does not
                match its recorded SHA-256.
        """
        if config != self._config:
            raise ValueError("Eagle runtime was built for another configuration")
        self.load()
        references = [view.payload_reference for view in visual_views]
        try:
            images = [self._decode(view) for view in visual_views]
            # A ordem do conteúdo é a ordem do request: o chat template numera as imagens
            # (<image-1>, <image-2>, ...) e o processor associa a k-ésima à k-ésima da lista.
            content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
            text = self._processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[text],
                images=images,
                return_tensors="pt",
                images_kwargs=_tiling_settings(config),
            ).to(config.device)
        except Exception as error:
            raise EagleInferenceError(
                f"Eagle could not prepare views {references}: {error}"
            ) from error
        _require_tiles_within_budget(inputs, config, view_count=len(visual_views))
        return self._run(inputs, config, references)

    def load(self) -> None:
        """Import optional SDKs, validate the device, and load the exact weights once.

        ``generate`` calls this lazily. A caller that measures latency calls it first, so
        that the one-time load is not attributed to the first request.

        Raises:
            EagleBackendError: For dependency, device, or checkpoint loading failures.
        """
        if self._model is not None:
            return
        config = self._config
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as error:
            raise EagleDependencyError(
                "Eagle requires torch, transformers, and Pillow in the runtime environment"
            ) from error
        if config.precision not in _PRECISIONS:
            raise EagleDeviceError(f"Eagle precision must be one of {sorted(_PRECISIONS)}")
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise EagleDeviceError("configured CUDA device is unavailable")
        if config.device == "cpu" and config.precision == "float16":
            raise EagleDeviceError("float16 Eagle inference is not supported on CPU")
        source = {
            "revision": config.revision,
            "local_files_only": self._local_files_only,
            "trust_remote_code": True,
        }
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                config.model, use_fast=True, **source
            )
            # ``torch_dtype`` é o nome que o código remoto e a versão de transformers
            # documentados pelo Eagle 2.5 (4.51-4.55) aceitam.
            model = transformers.AutoModel.from_pretrained(
                config.model, torch_dtype=getattr(torch, config.precision), **source
            )
            model = model.to(config.device).eval()
        except ImportError as error:
            raise EagleDependencyError(
                f"Eagle could not import a package its processor or remote code needs: {error}"
            ) from error
        except Exception as error:
            location = "local cache" if self._local_files_only else "configured model source"
            raise EagleModelLoadError(
                f"could not load {config.model}@{config.revision} from {location}: {error}"
            ) from error
        self._torch = torch
        self._image_module = image_module
        self._processor = processor
        self._model = model

    def _decode(self, view: SemanticVisualView) -> Any:
        """Decode the verified bytes of one view as an RGB image."""
        payload = read_view_payload(self._view_root, view)
        with self._image_module.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")

    def _run(
        self, inputs: Any, config: EagleSemanticConfig, references: list[str]
    ) -> EagleGenerationResponse:
        """Generate from prepared inputs and decode the answer with its token diagnostics."""
        torch = self._torch
        on_cuda = config.device.startswith("cuda")
        try:
            prompt_tokens = int(inputs["input_ids"].shape[1])
            if on_cuda:
                torch.cuda.reset_peak_memory_stats(config.device)
            with torch.inference_mode():
                generated = self._model.generate(**inputs, **_generation_settings(config))
            peak_memory = int(torch.cuda.max_memory_allocated(config.device)) if on_cuda else None
            # O generate do Eagle entrega inputs_embeds ao LLM, então devolve só os tokens
            # novos: não há prompt a cortar (os exemplos oficiais decodificam tudo).
            output_tokens = int(generated.shape[1])
            text = self._processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        except Exception as error:
            raise EagleInferenceError(
                f"Eagle generation failed for views {references}: {error}"
            ) from error
        warnings = (
            (
                f"generation reached max_new_tokens={config.max_new_tokens}; "
                "the structured response may be truncated",
            )
            if output_tokens >= config.max_new_tokens
            else ()
        )
        return EagleGenerationResponse(
            text=text,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            peak_memory_bytes=peak_memory,
            warnings=warnings,
        )


def _tiling_settings(config: EagleSemanticConfig) -> dict[str, Any]:
    """Map the configured visual budget to the ``images_kwargs`` of the Eagle processor."""
    return {
        "max_dynamic_tiles": config.max_dynamic_tiles,
        "min_dynamic_tiles": config.min_dynamic_tiles,
        "use_thumbnail": config.use_thumbnail,
    }


def _require_tiles_within_budget(
    inputs: Any, config: EagleSemanticConfig, *, view_count: int
) -> None:
    """Refuse processor output whose tile count the configured budget does not allow.

    The processor is remote code: this check makes a budget it ignored an explicit failure
    instead of a silently larger, costlier input.

    Raises:
        EagleInferenceError: If there are fewer tiles than views or more than the budget.
    """
    tiles = int(inputs["pixel_values"].shape[0])
    allowed = view_count * config.max_tiles_per_view
    if not view_count <= tiles <= allowed:
        raise EagleInferenceError(
            f"Eagle processor produced {tiles} tiles for {view_count} views; the configured "
            f"budget allows at least {view_count} and at most {allowed} "
            f"(max_dynamic_tiles={config.max_dynamic_tiles}, "
            f"use_thumbnail={config.use_thumbnail})"
        )


def _generation_settings(config: EagleSemanticConfig) -> dict[str, Any]:
    """Map the configured temperature to explicit greedy or sampling generation settings."""
    if config.temperature > 0:
        return {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": True,
            "temperature": config.temperature,
        }
    # Decoding guloso: anula os defaults de amostragem que o generation_config do LLM
    # poderia trazer, para que temperature=0 signifique exatamente argmax.
    return {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }

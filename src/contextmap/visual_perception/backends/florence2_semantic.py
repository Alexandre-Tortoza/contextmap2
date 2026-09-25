"""Florence-2 adapter for canonical Semantic Interpretation requests.

Florence-2 is driven by task tokens and answers in plain text, while the canonical
boundary exchanges ``semantic-response/1`` JSON. The adapter therefore owns two explicit,
versioned rules. ``florence2-task-prompt/1`` is its task-native prompt policy: the model
input is the configured task token (plus the whole-view box for region tasks), never a
free-form instruction template. ``florence2-task-envelope/1`` maps the answer: the task
text becomes exactly one primary claim, verbatim, and the resulting JSON goes through the
shared parser like the output of any other interpreter. Nothing is inferred from the text.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import io
import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from contextmap.visual_perception.backends._huggingface import (
    validate_huggingface_commit_revision,
)
from contextmap.visual_perception.backends._semantic_views import read_view_payload
from contextmap.visual_perception.models import BackendProvenance, SemanticInferenceProvenance
from contextmap.visual_perception.region_models import JsonScalar
from contextmap.visual_perception.semantic_backend import (
    SemanticBackendDiagnostics,
    SemanticInterpretationExecution,
    semantic_failure_from_parse_error,
)
from contextmap.visual_perception.semantic_prompt import (
    SEMANTIC_RESPONSE_SCHEMA,
    RenderedSemanticPrompt,
    SemanticConfidencePolicy,
    SemanticParseDiagnostic,
    SemanticResponseParseError,
    parse_semantic_response,
)
from contextmap.visual_perception.semantic_requests import (
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreterCapabilities,
    SemanticVisualView,
    VisualViewKind,
    validate_semantic_request,
)

TASK_ENVELOPE_POLICY = "florence2-task-envelope/1"
"""Versioned rule that turns Florence-2 task text into a canonical response."""

TASK_PROMPT_POLICY = "florence2-task-prompt/1"
"""Versioned task-native prompt policy: the model input is the task token plus task input.

A request to Florence-2 names ``"<TASK_PROMPT_POLICY>:<task>"`` as its prompt template, so
the recorded prompt is the text the model actually consumed.
"""

_WHOLE_VIEW_REGION = "<loc_0><loc_0><loc_999><loc_999>"
_LOCATION_TOKEN = re.compile(r"<loc_\d+>")
_PRECISIONS = frozenset({"float32", "float16", "bfloat16"})


class Florence2SemanticError(RuntimeError):
    """Base class for explicit Florence-2 semantic runtime failures."""


class Florence2DependencyError(Florence2SemanticError):
    """Raised when an optional Florence-2 runtime dependency is unavailable."""


class Florence2DeviceError(Florence2SemanticError):
    """Raised when the configured device or precision cannot be used."""


class Florence2ModelLoadError(Florence2SemanticError):
    """Raised when the configured checkpoint cannot load."""


class Florence2InferenceError(Florence2SemanticError):
    """Raised for view loading, generation, or task-parsing failures."""


@dataclass(frozen=True, kw_only=True)
class Florence2SemanticTask:
    """What one Florence-2 task can serve in the canonical semantic boundary.

    Attributes:
        mode: The only canonical mode the task answers.
        view_kinds: Views the task can interpret. A region task receives the whole view as
            its region, so it only accepts views where the region fills the image; the
            request does not carry the region box inside a full frame or a contextual crop.
        task_input: Text appended to the task token: nothing for scene tasks, and the
            whole-view region box for region tasks.
    """

    mode: SemanticInterpretationMode
    view_kinds: frozenset[VisualViewKind]
    task_input: str = ""


_SCENE_TASK = Florence2SemanticTask(
    mode=SemanticInterpretationMode.SCENE,
    view_kinds=frozenset({VisualViewKind.FULL_FRAME}),
)
_REGION_TASK = Florence2SemanticTask(
    mode=SemanticInterpretationMode.REGION,
    view_kinds=frozenset({VisualViewKind.TIGHT_CROP, VisualViewKind.MASKED_SUBJECT}),
    task_input=_WHOLE_VIEW_REGION,
)

FLORENCE2_SEMANTIC_TASKS: Mapping[str, Florence2SemanticTask] = MappingProxyType(
    {
        "<CAPTION>": _SCENE_TASK,
        "<DETAILED_CAPTION>": _SCENE_TASK,
        "<MORE_DETAILED_CAPTION>": _SCENE_TASK,
        "<REGION_TO_CATEGORY>": _REGION_TASK,
        "<REGION_TO_DESCRIPTION>": _REGION_TASK,
    }
)
"""Text-answering Florence-2 tasks that can serve a canonical semantic mode.

Geometry-producing tasks (``<OD>``, ``<REGION_PROPOSAL>``, ...) are deliberately absent:
they belong to ``Florence2RegionDiscovery`` and their outputs are not semantic claims.
"""


@dataclass(frozen=True, kw_only=True)
class Florence2SemanticConfig:
    """Secret-free Florence-2 semantic task and generation configuration.

    Attributes:
        checkpoint: Transformers-native Florence-2 checkpoint, for example
            ``florence-community/Florence-2-large``. The runtime never executes remote code.
        revision: Immutable Hugging Face commit SHA.
        task: One key of :data:`FLORENCE2_SEMANTIC_TASKS`. Tasks answer different questions
            (a category, a description, a caption), so the task is persisted with every claim.
        supported_modes: Must equal the single mode the task serves.
        device: Torch device string.
        precision: ``float32``, ``float16`` or ``bfloat16``.
        max_new_tokens: Generation token limit.
        temperature: ``0`` selects deterministic decoding; positive values sample.
    """

    checkpoint: str
    revision: str
    task: str
    supported_modes: frozenset[SemanticInterpretationMode]
    device: str
    precision: str
    max_new_tokens: int
    temperature: float

    def __post_init__(self) -> None:
        """Validate task, immutable checkpoint identity, and generation settings."""
        for name, value in (
            ("checkpoint", self.checkpoint),
            ("task", self.task),
            ("device", self.device),
            ("precision", self.precision),
        ):
            if not value.strip():
                raise ValueError(f"Florence-2 {name} must not be empty")
        validate_huggingface_commit_revision(self.revision)
        if not self.supported_modes:
            raise ValueError("Florence-2 must declare at least one supported mode")
        task = FLORENCE2_SEMANTIC_TASKS.get(self.task)
        if task is None:
            raise ValueError(
                f"Florence-2 task {self.task!r} is not a supported semantic task "
                f"{sorted(FLORENCE2_SEMANTIC_TASKS)}; geometry-producing tasks belong to "
                "Florence-2 Region Discovery"
            )
        if self.supported_modes != frozenset({task.mode}):
            declared = sorted(mode.value for mode in self.supported_modes)
            raise ValueError(
                f"Florence-2 task {self.task} serves only {task.mode.value} mode, "
                f"but the configuration declares {declared}"
            )
        if self.max_new_tokens <= 0:
            raise ValueError("Florence-2 max_new_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Florence-2 temperature must be finite and non-negative")

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return a stable, SDK-free effective configuration."""
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "task": self.task,
            "supported_modes": ",".join(sorted(mode.value for mode in self.supported_modes)),
            "device": self.device,
            "precision": self.precision,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


@dataclass(frozen=True, kw_only=True)
class Florence2SemanticResponse:
    """SDK-neutral Florence-2 generation result.

    Attributes:
        text: The task-native answer after the official task parser, for example ``"door"``.
            It is not canonical JSON: the adapter maps it with :data:`TASK_ENVELOPE_POLICY`.
    """

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    peak_memory_bytes: int | None = None
    warnings: tuple[str, ...] = ()


class Florence2SemanticRuntime(Protocol):
    """Internal boundary around Florence-2 processor/model native objects."""

    def generate(
        self,
        *,
        visual_views: tuple[SemanticVisualView, ...],
        task_prompt: str,
        config: Florence2SemanticConfig,
    ) -> Florence2SemanticResponse:
        """Run ``task_prompt`` (task token plus task input) and return the task-native text.

        An implementation must obtain each view's bytes through ``read_view_payload`` (or
        an equivalent check) and must not decode a payload whose SHA-256 differs from
        ``SemanticVisualView.sha256``: the request identifies its evidence by that hash.
        """
        ...


class Florence2SemanticInterpreter:
    """Interpret scene/region evidence through a distinct Florence-2 capability adapter.

    Attributes:
        configuration_fingerprint: Identity of the effective configuration; a request must
            carry it.
        prompt_template_id: The task-native prompt policy (:data:`TASK_PROMPT_POLICY` and the
            configured task) every request must name. Florence-2 cannot consume a free-form
            instruction template, so a request naming one is refused, never ignored.
        output_schema_version: The schema :data:`TASK_ENVELOPE_POLICY` produces.
    """

    def __init__(
        self, *, config: Florence2SemanticConfig, runtime: Florence2SemanticRuntime
    ) -> None:
        """Bind configuration to a model lifecycle supplied by the composition root."""
        self._config = config
        self._runtime = runtime
        self._task = FLORENCE2_SEMANTIC_TASKS[config.task]
        encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        self.configuration_fingerprint = (
            "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        )
        self.prompt_template_id = f"{TASK_PROMPT_POLICY}:{config.task}"
        self.output_schema_version = SEMANTIC_RESPONSE_SCHEMA
        # O prompt nativo depende só da task configurada, nunca do request: renderizado uma vez.
        task_prompt = f"{config.task}{self._task.task_input}"
        self._rendered_prompt = RenderedSemanticPrompt(
            template_id=self.prompt_template_id,
            output_schema_version=self.output_schema_version,
            text=task_prompt,
            fingerprint="sha256:" + hashlib.sha256(task_prompt.encode("utf-8")).hexdigest(),
        )

    def backend_provenance(self) -> BackendProvenance:
        """Return checkpoint/task/config identity for every produced output."""
        return BackendProvenance(
            backend_id="florence2_semantic",
            capability="semantic_interpreter",
            provider="microsoft",
            model=f"{self._config.checkpoint}@{self._config.revision}",
            version="1",
            configuration_fingerprint=self.configuration_fingerprint,
        )

    def capabilities(self) -> SemanticInterpreterCapabilities:
        """Declare the single mode, the views the task can interpret, and one view at most."""
        return SemanticInterpreterCapabilities(
            supported_modes=self._config.supported_modes,
            supported_view_kinds=self._task.view_kinds,
            accepts_visual_features=False,
            accepts_scene_context=False,
            max_visual_views=1,
        )

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        """Run the configured task and parse it through the shared canonical boundary.

        Raises:
            ValueError: Before inference, if the request is unsupported (including more than
                the one view the capabilities declare), was built for another
                configuration, names a prompt policy other than
                :attr:`prompt_template_id`, or asks for an output schema the task envelope
                does not produce.
            SemanticInterpretationFailedError: If the mapped response cannot be parsed.
        """
        validate_semantic_request(request, self.capabilities())
        if request.configuration_fingerprint != self.configuration_fingerprint:
            raise ValueError("Florence-2 request configuration fingerprint does not match adapter")
        if request.prompt_template_id != self.prompt_template_id:
            raise ValueError(
                f"Florence-2 consumes only its task-native prompt policy "
                f"{self.prompt_template_id!r}; the request names {request.prompt_template_id!r}"
            )
        if request.requested_output_schema != self.output_schema_version:
            raise ValueError(
                f"Florence-2 answers through {TASK_ENVELOPE_POLICY} in "
                f"{self.output_schema_version!r}; the request asks for "
                f"{request.requested_output_schema!r}"
            )
        rendered = self._rendered_prompt
        started = time.monotonic()
        response = self._runtime.generate(
            visual_views=request.visual_views,
            task_prompt=rendered.text,
            config=self._config,
        )
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity=f"florence2-{self._config.task}-{request.mode.value}",
            prompt_template_id=rendered.template_id,
            output_schema_version=rendered.output_schema_version,
            raw_response_reference=(
                f"debug/40-semantic-interpretation/{request.request_id}/raw-response.txt"
            ),
        )
        task_text = response.text.strip()
        try:
            parsed = parse_semantic_response(
                _canonical_response_json(request.mode, task_text),
                request,
                provenance,
                confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
            )
        except SemanticResponseParseError as error:
            # raw_response e o texto da task, como no caminho de sucesso: o envelope canonico
            # e reconstruivel por TASK_ENVELOPE_POLICY e nao e o que o modelo respondeu.
            raise semantic_failure_from_parse_error(
                error,
                request=request,
                rendered_prompt=rendered,
                raw_response=response.text,
                provenance=provenance,
                diagnostics=SemanticBackendDiagnostics(
                    latency_ms=(time.monotonic() - started) * 1000,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    peak_memory_bytes=response.peak_memory_bytes,
                    warnings=response.warnings,
                ),
                effective_configuration=MappingProxyType(self._config.to_dict()),
            ) from error
        # O hash e o raw_response registram o que o modelo respondeu (o texto da task), não
        # o envelope intermediário; o envelope é reconstruível por TASK_ENVELOPE_POLICY.
        parsed = dataclasses.replace(
            parsed,
            raw_response_sha256=hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
            diagnostics=(
                *parsed.diagnostics,
                SemanticParseDiagnostic(
                    code="wrapped_task_text",
                    message=(
                        f"{TASK_ENVELOPE_POLICY}: task text wrapped verbatim as one primary "
                        "claim; no label, attribute, or scene field was inferred from it."
                    ),
                ),
            ),
        )
        warnings: list[str] = []
        if not task_text:
            warnings.append("task produced empty text; mapped to an explicit abstention")
        warnings.extend(response.warnings)
        configuration: Mapping[str, JsonScalar] = MappingProxyType(self._config.to_dict())
        return SemanticInterpretationExecution(
            request=request,
            rendered_prompt=rendered,
            raw_response=response.text,
            parsed=parsed,
            diagnostics=SemanticBackendDiagnostics(
                latency_ms=(time.monotonic() - started) * 1000,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                peak_memory_bytes=response.peak_memory_bytes,
                warnings=tuple(warnings),
            ),
            effective_configuration=configuration,
        )


def _canonical_response_json(mode: SemanticInterpretationMode, task_text: str) -> str:
    """Wrap verbatim task text as one primary claim, or abstain when there is none.

    Only ``hypothesis`` carries information. ``category``, ``region_kind``, attributes and
    every scene field stay empty because the task text does not provide them, and an empty
    answer is an explicit abstention instead of an invented claim. The text is embedded
    with ``json.dumps``, so it can never add claims, alternatives, or fields.
    """
    if not task_text:
        payload: dict[str, object] = {"abstained": True, "claims": [], "scene_context": None}
    else:
        payload = {
            "abstained": False,
            "claims": [
                {
                    "hypothesis": task_text,
                    "role": "primary",
                    "category": None,
                    "region_kind": None,
                    "attributes": {},
                    "confidence": None,
                }
            ],
            # Objeto vazio: nenhum campo de cena é inferido a partir do texto da task.
            "scene_context": {} if mode is SemanticInterpretationMode.SCENE else None,
        }
    return json.dumps(payload, sort_keys=True)


class HuggingFaceFlorence2SemanticRuntime:
    """Lazy transformers implementation of :class:`Florence2SemanticRuntime`.

    Loads the transformers-native Florence-2 port at the pinned revision (no remote code,
    local cache only unless ``local_files_only`` is disabled) and runs one task per call.
    The official task parser produces the text; the echoed region tokens of the region
    tasks are removed because they repeat the input box and are not part of the answer.
    """

    def __init__(
        self,
        *,
        config: Florence2SemanticConfig,
        view_root: Path,
        local_files_only: bool = True,
    ) -> None:
        """Bind the runtime to a configuration without importing SDKs or loading weights.

        Args:
            config: Effective configuration.
            view_root: Directory containing the run-relative ``outputs/semantic-views/``.
            local_files_only: Refuse implicit downloads when true.
        """
        self._config = config
        self._view_root = view_root
        self._local_files_only = local_files_only
        self._torch: Any = None
        self._image_module: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._dtype: Any = None

    def generate(
        self,
        *,
        visual_views: tuple[SemanticVisualView, ...],
        task_prompt: str,
        config: Florence2SemanticConfig,
    ) -> Florence2SemanticResponse:
        """Run one Florence-2 task on the single view and return its parsed text.

        The view is decoded from bytes whose SHA-256 was verified against
        ``SemanticVisualView.sha256`` first, so a payload that changed after the request was
        built is rejected instead of being interpreted.

        Raises:
            ValueError: If ``config`` differs from the configuration the runtime is bound to.
            Florence2SemanticError: For dependency, device, load, or inference failures,
                including a view payload that is missing, escapes the view root, or does not
                match its recorded SHA-256.
        """
        if config != self._config:
            raise ValueError("Florence-2 runtime was built for another configuration")
        if len(visual_views) != 1:
            raise Florence2InferenceError("Florence-2 tasks consume exactly one visual view")
        self.load()
        torch = self._torch
        on_cuda = config.device.startswith("cuda")
        try:
            payload = read_view_payload(self._view_root, visual_views[0])
            with self._image_module.open(io.BytesIO(payload)) as image:
                rgb = image.convert("RGB")
            inputs = self._processor(text=task_prompt, images=rgb, return_tensors="pt").to(
                config.device, self._dtype
            )
            prompt_tokens = int(inputs["input_ids"].shape[1])
            if on_cuda:
                torch.cuda.reset_peak_memory_stats(config.device)
            generation: dict[str, Any] = {
                "max_new_tokens": config.max_new_tokens,
                "do_sample": config.temperature > 0,
            }
            if config.temperature > 0:
                generation["temperature"] = config.temperature
            with torch.inference_mode():
                generated = self._model.generate(**inputs, **generation)
            peak_memory = int(torch.cuda.max_memory_allocated(config.device)) if on_cuda else None
            decoded = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = self._processor.post_process_generation(
                decoded, task=config.task, image_size=rgb.size
            )
            task_text = parsed[config.task]
            if not isinstance(task_text, str):
                raise Florence2InferenceError(
                    f"task {config.task} did not return text: {type(task_text).__name__}"
                )
        except Florence2SemanticError:
            raise
        except Exception as error:
            references = [view.payload_reference for view in visual_views]
            raise Florence2InferenceError(
                f"Florence-2 {config.task} failed for views {references}: {error}"
            ) from error
        return Florence2SemanticResponse(
            text=_LOCATION_TOKEN.sub("", task_text).strip(),
            input_tokens=prompt_tokens,
            output_tokens=int(generated.shape[1]),
            peak_memory_bytes=peak_memory,
        )

    def load(self) -> None:
        """Import optional SDKs, validate the device, and load the exact weights once.

        Raises:
            Florence2SemanticError: For dependency, device, or checkpoint loading failures.
        """
        if self._model is not None:
            return
        config = self._config
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as error:
            raise Florence2DependencyError(
                "Florence-2 requires torch, transformers, and Pillow in the runtime environment"
            ) from error
        if config.precision not in _PRECISIONS:
            raise Florence2DeviceError(f"Florence-2 precision must be one of {sorted(_PRECISIONS)}")
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise Florence2DeviceError("configured CUDA device is unavailable")
        if config.device == "cpu" and config.precision == "float16":
            raise Florence2DeviceError("float16 Florence-2 inference is not supported on CPU")
        dtype = getattr(torch, config.precision)
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                config.checkpoint,
                revision=config.revision,
                local_files_only=self._local_files_only,
            )
            model = transformers.Florence2ForConditionalGeneration.from_pretrained(
                config.checkpoint,
                revision=config.revision,
                local_files_only=self._local_files_only,
                dtype=dtype,
            )
            model = model.to(config.device).eval()
        except ImportError as error:
            raise Florence2DependencyError(
                "Florence-2 could not import a package required by the Hugging Face "
                f"processor or model: {error}"
            ) from error
        except Exception as error:
            source = "local cache" if self._local_files_only else "configured model source"
            raise Florence2ModelLoadError(
                f"could not load {config.checkpoint}@{config.revision} from {source}: {error}"
            ) from error
        self._torch = torch
        self._image_module = image_module
        self._processor = processor
        self._model = model
        self._dtype = dtype

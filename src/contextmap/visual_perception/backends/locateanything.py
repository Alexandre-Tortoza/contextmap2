"""NVIDIA LocateAnything adapter for the prompt-conditioned Region Grounding capability.

LocateAnything (NVlabs/Eagle, ``Embodied/``) answers a language query with generated text in
which every geometry is an atomic block of coordinate tokens normalized to ``[0, 1000]``:

- box: ``<ref>label</ref><box><x1><y1><x2><y2></box>``;
- point: ``<box><x><y></box>``;
- explicit no object: ``<box>none</box>``;
- end of response: ``<|im_end|>``.

This module owns the mapping from a canonical :class:`RegionGroundingRequest` to the
upstream prompt templates, and from the verbatim answer back to canonical evidence. It never
imports an SDK: model execution sits behind the :class:`LocateAnythingRuntime` seam, so the
parser, the policies and the provenance are testable with a deterministic fake. The parser
is total: every span of the answer ends up as an output, an explicit no-match, or an
explicitly rejected span with its reason; nothing is silently dropped or repaired. The
upstream worker exposes no calibrated per-box score, so no output carries a confidence;
decoder statistics are kept verbatim as native diagnostics.

:class:`TransformersLocateAnythingRuntime` is the bundled implementation of the seam. It
imports torch, transformers, Pillow and the model's remote code lazily, loads the pinned
revision once per runtime (one per resolved run), validates device and attention paths
before loading, refuses any silent upstream fallback, and reports library versions.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import os
import platform
import re
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path, PurePosixPath
from time import perf_counter
from types import MappingProxyType
from typing import Any, Protocol

from contextmap.visual_perception.backends._huggingface import (
    validate_huggingface_commit_revision,
)
from contextmap.visual_perception.grounding import (
    GROUNDING_CAPABILITY,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingPoint,
    GroundingQuery,
    GroundingQueryPolicy,
    GroundingRejectionReason,
    GroundingRequestError,
    GroundingTask,
    RegionGroundingCapabilities,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RejectedGroundingOutput,
    validate_grounding_query,
)
from contextmap.visual_perception.models import BackendProvenance, BoundingBox2D, PreparedImage
from contextmap.visual_perception.region_models import JsonScalar

CATEGORY_DETECTION_POLICY = "locateanything.category-detection/1"
"""Upstream ``detect()`` template: every instance of each category, joined by ``</c>``."""

PHRASE_GROUNDING_POLICY = "locateanything.phrase-grounding/1"
"""Upstream ``ground_multi()`` template: every instance a free-form phrase refers to."""

POINTING_POLICY = "locateanything.pointing/1"
"""Upstream ``point()`` template: one point per referred instance."""

PARSER_VERSION = "locateanything-response/1"
"""Version of the answer grammar and pixel conversion; part of the configuration identity."""

NORMALIZED_COORDINATE_MAX = 1000
"""Upstream coordinate tokens span ``<0>``..``<1000>``; 1000 maps to the image edge."""

_CATEGORY_SEPARATOR = "</c>"
_PROMPT_TEMPLATES: Mapping[str, str] = MappingProxyType(
    {
        # Textos idênticos aos do worker oficial (inclusive "matches" no detect()): o modelo
        # foi treinado com eles, e qualquer mudança exige outra identidade de política.
        CATEGORY_DETECTION_POLICY: (
            "Locate all the instances that matches the following description: {query}."
        ),
        PHRASE_GROUNDING_POLICY: (
            "Locate all the instances that match the following description: {query}."
        ),
        POINTING_POLICY: "Point to: {query}.",
    }
)
_CAPABILITIES = RegionGroundingCapabilities(
    query_policies=(
        GroundingQueryPolicy(
            policy_id=CATEGORY_DETECTION_POLICY,
            task=GroundingTask.CATEGORY_DETECTION,
            geometry=GroundingGeometry.BOX,
        ),
        GroundingQueryPolicy(
            policy_id=PHRASE_GROUNDING_POLICY,
            task=GroundingTask.PHRASE_GROUNDING,
            geometry=GroundingGeometry.BOX,
        ),
        GroundingQueryPolicy(
            policy_id=POINTING_POLICY,
            task=GroundingTask.PHRASE_GROUNDING,
            geometry=GroundingGeometry.POINT,
        ),
    )
)

_BLOCK = r"(?:(?!<ref>|</ref>|<box>|</box>).)*"
_TOKEN = re.compile(
    rf"<ref>(?P<ref>{_BLOCK})</ref>"
    rf"|<box>(?P<box>{_BLOCK})</box>"
    r"|(?P<end><\|im_end\|>|<\|endoftext\|>)"
    r"|(?P<space>\s+)",
    re.DOTALL,
)
_RESYNC = re.compile(r"<ref>|<box>|<\|im_end\|>|<\|endoftext\|>")
_COORDINATE_SEQUENCE = re.compile(r"(?:<\d+>)+")
_COORDINATE = re.compile(r"<(\d+)>")
_STATISTIC = re.compile(r"([A-Za-z_][\w()]*)=([^;\s]+)")
_DEVICE = re.compile(r"cpu|cuda(?::\d+)?")
_DTYPES = frozenset({"bfloat16", "float16", "float32"})
# Caminhos de atenção aceitos por runtime: "auto" nunca é aceito, porque o upstream o resolve
# com fallback silencioso para SDPA quando a biblioteca pedida não está instalada.
_STANDARD_TEXT_ATTENTION = frozenset({"sdpa", "eager", "magi"})
_BATCH_TEXT_ATTENTION = frozenset({"sdpa", "eager", "magi", "la_flash"})
_VISION_ATTENTION = frozenset({"sdpa", "eager", "flash_attention_2"})
_CUDA_ONLY_ATTENTION = frozenset({"magi", "la_flash", "flash_attention_2"})
_FLASH_ATTENTION = frozenset({"la_flash", "flash_attention_2"})
_BATCH_SCHEDULERS = frozenset({"eager", "hold_ar", "ar_first", "pipeline", "adaptive"})
_MAGI_MIN_COMPUTE_CAPABILITY = 9
"""MagiAttention runs only on Hopper (sm90) or Blackwell GPUs, per the upstream README."""


class LocateAnythingGenerationMode(StrEnum):
    """Upstream decoding modes: parallel boxes, autoregressive, or parallel with fallback."""

    FAST = "fast"
    SLOW = "slow"
    HYBRID = "hybrid"


class LocateAnythingRuntimeMode(StrEnum):
    """Which upstream inference path executes the model.

    ``STANDARD`` is the worker's ``AutoModel`` + remote-code ``generate()`` path;
    ``BATCH`` is the Hugging Face release's ``batch_utils`` hybrid scheduler.
    """

    STANDARD = "standard"
    BATCH = "batch"


@dataclass(frozen=True, kw_only=True)
class LocateAnythingConfig:
    """Effective LocateAnything inference configuration.

    Attributes:
        model: Hugging Face model identity, e.g. ``"nvidia/LocateAnything-3B"``.
        revision: Immutable full commit SHA of the model repository (weights and remote
            code); a branch or tag name is refused so a checkpoint cannot change silently.
        device: ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``.
        dtype: Weight/compute dtype: ``"bfloat16"`` (upstream default), ``"float16"`` or
            ``"float32"``.
        generation_mode: Explicit upstream decoding mode.
        max_new_tokens: Generation token limit (the model card suggests 8192 to avoid
            truncated dense answers).
        temperature: ``0`` decodes greedily; a positive value samples with it.
        top_p: Nucleus sampling mass, in ``(0, 1]``.
        top_k: Top-k cutoff; ``0`` disables it, as upstream.
        repetition_penalty: Upstream sampler repetition penalty; ``1.0`` disables it.
        text_attention: Language-decoder attention: ``"sdpa"``, ``"eager"`` or ``"magi"``
            (Hopper/Blackwell only), plus ``"la_flash"`` for the batch runtime. Explicit
            because the upstream default silently falls back to SDPA.
        vision_attention: MoonViT attention: ``"sdpa"``, ``"eager"`` or
            ``"flash_attention_2"``. Explicit for the same reason.
        runtime: Standard worker path or the release batch runtime.
        scheduler: Batch hybrid scheduler (batch runtime only, required there).
        group_size: Batch scheduler group size (batch runtime only, required there);
            ``0`` lets the upstream runtime choose, as upstream documents.
        local_files_only: Refuse downloads and load only from the local Hugging Face cache
            (the default, for reproducible offline runs). An asset policy, not an inference
            setting, so it is recorded but not part of the fingerprint.
    """

    model: str
    revision: str
    device: str
    dtype: str
    generation_mode: LocateAnythingGenerationMode
    max_new_tokens: int
    temperature: float
    text_attention: str
    vision_attention: str
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1
    runtime: LocateAnythingRuntimeMode = LocateAnythingRuntimeMode.STANDARD
    scheduler: str | None = None
    group_size: int | None = None
    local_files_only: bool = True

    def __post_init__(self) -> None:
        """Validate identity and generation settings before any runtime exists."""
        if not self.model.strip():
            raise ValueError("LocateAnything model must not be empty")
        validate_huggingface_commit_revision(self.revision)
        if _DEVICE.fullmatch(self.device) is None:
            raise ValueError("LocateAnything device must be 'cpu', 'cuda' or 'cuda:<index>'")
        if self.dtype not in _DTYPES:
            raise ValueError(f"LocateAnything dtype must be one of {sorted(_DTYPES)}")
        if self.device == "cpu" and self.dtype == "float16":
            raise ValueError("float16 LocateAnything inference is not supported on CPU")
        if self.max_new_tokens <= 0:
            raise ValueError("LocateAnything max_new_tokens must be positive")
        if not isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("LocateAnything temperature must be finite and non-negative")
        if not isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("LocateAnything top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("LocateAnything top_k must be non-negative (0 disables it)")
        if not isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("LocateAnything repetition_penalty must be positive")
        if self.model.startswith("/") and PurePosixPath(self.model).name != self.revision:
            raise ValueError(
                "a local LocateAnything model directory must be the Hugging Face snapshot of "
                "the pinned revision (.../snapshots/<revision>)"
            )
        self._validate_runtime_paths()

    def _validate_runtime_paths(self) -> None:
        """Refuse attention/runtime/device combinations the upstream code cannot honor."""
        batch = self.runtime is LocateAnythingRuntimeMode.BATCH
        text_choices = _BATCH_TEXT_ATTENTION if batch else _STANDARD_TEXT_ATTENTION
        if self.text_attention not in text_choices:
            raise ValueError(
                f"text_attention {self.text_attention!r} is not supported by the "
                f"{self.runtime.value} runtime; choose one of {sorted(text_choices)}"
            )
        if self.vision_attention not in _VISION_ATTENTION:
            raise ValueError(
                f"vision_attention {self.vision_attention!r} is not supported; choose one of "
                f"{sorted(_VISION_ATTENTION)}"
            )
        cuda_only = {self.text_attention, self.vision_attention} & _CUDA_ONLY_ATTENTION
        if self.device == "cpu" and cuda_only:
            raise ValueError(f"attention {sorted(cuda_only)} requires a CUDA device")
        if not batch:
            if self.scheduler is not None:
                raise ValueError("scheduler applies only to the batch runtime")
            if self.group_size is not None:
                raise ValueError("group_size applies only to the batch runtime")
            return
        if self.device == "cpu":
            raise ValueError("the LocateAnything batch runtime requires a CUDA device")
        if self.generation_mode is not LocateAnythingGenerationMode.HYBRID:
            raise ValueError("the LocateAnything batch runtime supports only hybrid generation")
        if self.dtype != "bfloat16":
            raise ValueError(
                "the LocateAnything batch runtime loads the release weights in bfloat16 and "
                "exposes no dtype control"
            )
        if self.scheduler not in _BATCH_SCHEDULERS:
            raise ValueError(f"batch runtime scheduler must be one of {sorted(_BATCH_SCHEDULERS)}")
        if self.group_size is None or self.group_size < 0:
            raise ValueError(
                "batch runtime group_size must be set and non-negative (0 lets the upstream "
                "runtime choose)"
            )

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return the JSON-compatible effective configuration."""
        values = asdict(self)
        values["generation_mode"] = self.generation_mode.value
        values["runtime"] = self.runtime.value
        return values

    @property
    def fingerprint(self) -> str:
        """Return the digest of every setting that can change the evidence produced.

        ``local_files_only`` is left out: with a pinned revision it decides where the
        assets come from, never which assets or how they run.
        """
        document = {
            **{key: value for key, value in self.to_dict().items() if key != "local_files_only"},
            "parser_version": PARSER_VERSION,
        }
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class LocateAnythingGeneration:
    """SDK-free result of one LocateAnything generation.

    Attributes:
        text: The verbatim decoded answer, special tokens included.
        decode_steps: Upstream sampling history, verbatim: one ``(decoder, text)`` pair per
            decoding step, ``decoder`` being ``"mtp"`` or ``"ar"``. Empty when the runtime
            does not expose it.
        statistics: Upstream statistics text, verbatim, when exposed.
        peak_memory_bytes: Peak accelerator memory of the call, when measured.
        runtime_identity: Library/device identity of the runtime that generated the text.
        native_diagnostics: Runtime measurements of this call with native names, e.g.
            ``runtime.cold_load_ms`` on the call that loaded the model.
        warnings: Runtime warnings.
    """

    text: str
    decode_steps: tuple[tuple[str, str], ...] = ()
    statistics: str | None = None
    peak_memory_bytes: int | None = None
    runtime_identity: tuple[tuple[str, JsonScalar], ...] = ()
    native_diagnostics: tuple[tuple[str, JsonScalar], ...] = ()
    warnings: tuple[str, ...] = ()


class LocateAnythingRuntime(Protocol):
    """Internal seam isolating torch, transformers and the upstream remote code."""

    def generate(
        self, *, image: PreparedImage, prompt: str, config: LocateAnythingConfig
    ) -> LocateAnythingGeneration:
        """Generate one answer for the exact prepared image and rendered prompt."""
        ...


@dataclass(frozen=True, kw_only=True)
class LocateAnythingParse:
    """Every span of one answer: accepted outputs, explicit no-match, explicit rejections.

    Attributes:
        outputs: Accepted box/point outputs in answer order.
        rejected_outputs: Spans explicitly rejected, sharing the output index space.
        no_match_labels: Labels answered with ``<box>none</box>`` (``None`` if unlabelled).
        terminated: Whether the answer carried an end-of-response token.
        warnings: Parser warnings, e.g. an unusable decode history.
    """

    outputs: tuple[GroundingOutput, ...]
    rejected_outputs: tuple[RejectedGroundingOutput, ...]
    no_match_labels: tuple[str | None, ...]
    terminated: bool
    warnings: tuple[str, ...] = ()


def validate_locateanything_query(query: GroundingQuery) -> None:
    """Check, without any model, that LocateAnything can serve a query as written.

    Raises:
        GroundingRequestError: If the policy, task or geometry is not declared, or the
            query cannot be rendered with the upstream template.
    """
    validate_grounding_query(query, _CAPABILITIES)
    render_locateanything_prompt(query)


def render_locateanything_prompt(query: GroundingQuery) -> str:
    """Render a query with the upstream template of its policy.

    Raises:
        GroundingRequestError: If the policy is not a LocateAnything policy, or a
            category contains the upstream separator ``</c>`` and would be split.
    """
    template = _PROMPT_TEMPLATES.get(query.policy_id)
    if template is None:
        raise GroundingRequestError(f"no LocateAnything template for {query.policy_id!r}")
    if query.task is GroundingTask.CATEGORY_DETECTION:
        for category in query.categories:
            if _CATEGORY_SEPARATOR in category:
                raise GroundingRequestError(
                    f"category {category!r} contains the upstream separator "
                    f"{_CATEGORY_SEPARATOR!r} and would be split into several categories"
                )
        return template.format(query=_CATEGORY_SEPARATOR.join(query.categories))
    return template.format(query=query.text)


def parse_locateanything_response(
    text: str,
    *,
    image_width: int,
    image_height: int,
    decode_steps: tuple[tuple[str, str], ...] = (),
) -> LocateAnythingParse:
    """Parse a verbatim answer into pixel-space outputs and explicit rejections.

    Coordinates are converted as upstream ``parse_boxes()``/``parse_points()`` do:
    ``pixel = normalized / 1000 * image_size`` in the prepared image the model received,
    so ``0`` and ``1000`` map exactly to the image edges.

    Args:
        text: The answer, verbatim.
        image_width: Width of the prepared image the model received.
        image_height: Height of the prepared image the model received.
        decode_steps: Optional upstream sampling history; when it rebuilds ``text``
            exactly, each output records which decoder produced it.

    Returns:
        The parse, in answer order.
    """
    step_spans = _decode_step_spans(text, decode_steps)
    warnings: list[str] = []
    if decode_steps and step_spans is None:
        warnings.append(
            "decode history does not rebuild the answer; per-output decoder is not reported"
        )
    outputs: list[GroundingOutput] = []
    rejected: list[RejectedGroundingOutput] = []
    no_match_labels: list[str | None] = []
    label: str | None = None
    terminated = False
    index = 0
    position = 0

    def reject(span: str, reason: GroundingRejectionReason, detail: str) -> None:
        nonlocal index
        rejected.append(
            RejectedGroundingOutput(
                output_index=index, native_text=span, reason=reason, detail=detail, label=label
            )
        )
        index += 1

    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            resync = _RESYNC.search(text, position + 1)
            end = len(text) if resync is None else resync.start()
            span = text[position:end]
            if terminated:
                reject(
                    span,
                    GroundingRejectionReason.UNRECOGNIZED_TEXT,
                    "text after the end-of-response token",
                )
            elif span.startswith("<box>"):
                reject(
                    span, GroundingRejectionReason.MALFORMED_GEOMETRY, "unclosed or nested <box>"
                )
            else:
                reject(
                    span,
                    GroundingRejectionReason.UNRECOGNIZED_TEXT,
                    "text outside the LocateAnything answer grammar",
                )
            position = end
            continue
        kind = match.lastgroup
        span = match.group(0)
        position = match.end()
        if kind in {"space", "end"}:
            terminated = terminated or kind == "end"
            continue
        if terminated:
            reject(
                span,
                GroundingRejectionReason.UNRECOGNIZED_TEXT,
                "text after the end-of-response token",
            )
            continue
        if kind == "ref":
            label = match.group("ref")
            continue
        content = match.group("box")
        if content == "none":
            no_match_labels.append(label)
            continue
        problem = _geometry_problem(content)
        if problem is not None:
            reject(span, *problem)
            continue
        values = [int(value) for value in _COORDINATE.findall(content)]
        native_diagnostics: tuple[tuple[str, JsonScalar], ...] = ()
        if step_spans is not None:
            native_diagnostics = (("decoder", _decoder_of(match.start(), match.end(), step_spans)),)
        outputs.append(
            _output(
                index=index,
                span=span,
                label=label,
                values=values,
                width=image_width,
                height=image_height,
                native_diagnostics=native_diagnostics,
            )
        )
        index += 1
    return LocateAnythingParse(
        outputs=tuple(outputs),
        rejected_outputs=tuple(rejected),
        no_match_labels=tuple(no_match_labels),
        terminated=terminated,
        warnings=tuple(warnings),
    )


class LocateAnythingRegionGrounding:
    """Serve canonical grounding requests with an explicitly configured LocateAnything runtime."""

    def __init__(self, *, config: LocateAnythingConfig, runtime: LocateAnythingRuntime) -> None:
        """Bind the adapter to one configuration and runtime; nothing is loaded here."""
        self._config = config
        self._runtime = runtime

    def backend_provenance(self) -> BackendProvenance:
        """Return the model, its pinned revision and the configuration fingerprint."""
        return BackendProvenance(
            backend_id="locateanything",
            capability=GROUNDING_CAPABILITY,
            provider="nvidia",
            model=self._config.model,
            version=self._config.revision,
            configuration_fingerprint=self._config.fingerprint,
        )

    def capabilities(self) -> RegionGroundingCapabilities:
        """Declare category detection and phrase grounding (boxes), and pointing (points)."""
        return _CAPABILITIES

    def ground(self, request: RegionGroundingRequest) -> RegionGroundingExecution:
        """Validate, render, generate and parse one request.

        Raises:
            GroundingRequestError: Before any inference, for an unsupported policy, task or
                geometry, a request fingerprinted for another configuration, or a query
                that the upstream template cannot represent.
        """
        validate_grounding_query(request.query, _CAPABILITIES)
        if request.configuration_fingerprint != self._config.fingerprint:
            raise GroundingRequestError(
                "grounding request configuration fingerprint does not match this "
                "LocateAnything configuration"
            )
        prompt = render_locateanything_prompt(request.query)
        started = perf_counter()
        generation = self._runtime.generate(image=request.image, prompt=prompt, config=self._config)
        latency_ms = (perf_counter() - started) * 1000
        parsed = parse_locateanything_response(
            generation.text,
            image_width=request.image.width,
            image_height=request.image.height,
            decode_steps=generation.decode_steps,
        )
        warnings = [*generation.warnings, *parsed.warnings]
        if not parsed.terminated:
            warnings.append(
                "answer has no end-of-response token; it may be truncated at "
                f"max_new_tokens={self._config.max_new_tokens}"
            )
        warnings.extend(_geometry_mismatch_warnings(parsed.outputs, request.query.geometry))
        return RegionGroundingExecution(
            request=request,
            provenance=self.backend_provenance(),
            rendered_prompt=prompt,
            raw_response=generation.text,
            outputs=parsed.outputs,
            rejected_outputs=parsed.rejected_outputs,
            no_match_labels=parsed.no_match_labels,
            diagnostics=GroundingDiagnostics(
                latency_ms=latency_ms,
                peak_memory_bytes=generation.peak_memory_bytes,
                warnings=tuple(warnings),
                native=(
                    *generation.native_diagnostics,
                    *_statistics_diagnostics(generation.statistics),
                ),
            ),
            effective_configuration=MappingProxyType(
                {**self._config.to_dict(), "parser_version": PARSER_VERSION}
            ),
            runtime_identity=MappingProxyType(dict(generation.runtime_identity)),
        )


def _geometry_problem(content: str) -> tuple[GroundingRejectionReason, str] | None:
    """Explain why a ``<box>`` content is not a valid box or point, or return ``None``."""
    if _COORDINATE_SEQUENCE.fullmatch(content) is None:
        return (
            GroundingRejectionReason.MALFORMED_GEOMETRY,
            "box content must be integer coordinate tokens or 'none'",
        )
    values = [int(value) for value in _COORDINATE.findall(content)]
    if len(values) not in {2, 4}:
        return (
            GroundingRejectionReason.MALFORMED_GEOMETRY,
            f"a box needs four coordinates and a point two; found {len(values)}",
        )
    if any(value > NORMALIZED_COORDINATE_MAX for value in values):
        return (
            GroundingRejectionReason.COORDINATE_OUT_OF_RANGE,
            f"normalized coordinates must be in [0, {NORMALIZED_COORDINATE_MAX}]: {values}",
        )
    if len(values) == 4 and (values[2] <= values[0] or values[3] <= values[1]):
        return (
            GroundingRejectionReason.INVERTED_GEOMETRY,
            f"a box needs x2 > x1 and y2 > y1: {values}",
        )
    return None


def _output(
    *,
    index: int,
    span: str,
    label: str | None,
    values: list[int],
    width: int,
    height: int,
    native_diagnostics: tuple[tuple[str, JsonScalar], ...],
) -> GroundingOutput:
    """Convert validated normalized coordinates into one pixel-space output."""
    xs = [value * width / NORMALIZED_COORDINATE_MAX for value in values[0::2]]
    ys = [value * height / NORMALIZED_COORDINATE_MAX for value in values[1::2]]
    if len(values) == 2:
        return GroundingOutput(
            output_index=index,
            native_text=span,
            label=label,
            point=GroundingPoint(x=xs[0], y=ys[0]),
            native_diagnostics=native_diagnostics,
        )
    return GroundingOutput(
        output_index=index,
        native_text=span,
        label=label,
        box=BoundingBox2D(x=xs[0], y=ys[0], width=xs[1] - xs[0], height=ys[1] - ys[0]),
        native_diagnostics=native_diagnostics,
    )


def _decode_step_spans(
    text: str, steps: tuple[tuple[str, str], ...]
) -> list[tuple[int, int, str]] | None:
    """Locate each decoding step in the answer, or ``None`` if the steps do not rebuild it."""
    if not steps or "".join(piece for _, piece in steps) != text:
        return None
    spans: list[tuple[int, int, str]] = []
    start = 0
    for decoder, piece in steps:
        spans.append((start, start + len(piece), decoder))
        start += len(piece)
    return spans


def _decoder_of(start: int, end: int, spans: list[tuple[int, int, str]]) -> str:
    """Name the decoders that produced a span, in first-use order (``"mtp+ar"`` = fallback)."""
    decoders: list[str] = []
    for span_start, span_end, decoder in spans:
        if span_start < end and span_end > start and decoder not in decoders:
            decoders.append(decoder)
    return "+".join(decoders)


def _statistics_diagnostics(statistics: str | None) -> tuple[tuple[str, JsonScalar], ...]:
    """Keep upstream statistics verbatim, plus their ``name=value`` pairs when recognizable."""
    if statistics is None:
        return ()
    diagnostics: list[tuple[str, JsonScalar]] = [("stats.raw", statistics)]
    # Só o formato "Statistic Info, k=v; ..." do generate() oficial é decomposto; qualquer
    # outro (ex.: o dict do runtime batch serializado) fica apenas bruto, sem adivinhação.
    if "Statistic Info" in statistics:
        diagnostics.extend(
            (f"stats.{name}", _scalar(value)) for name, value in _STATISTIC.findall(statistics)
        )
    return tuple(diagnostics)


def _scalar(value: str) -> JsonScalar:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        number = float(value)
    except ValueError:
        return value
    return number if isfinite(number) else value


def _geometry_mismatch_warnings(
    outputs: tuple[GroundingOutput, ...], requested: GroundingGeometry
) -> list[str]:
    """Report outputs whose geometry differs from the request; they are kept, not converted."""
    mismatched = [output for output in outputs if output.geometry is not requested]
    if not mismatched:
        return []
    family = mismatched[0].geometry.value
    return [
        f"{len(mismatched)} {family} output(s) answered a {requested.value} request; "
        "kept as returned, never converted"
    ]


class LocateAnythingBackendError(RuntimeError):
    """Base class for explicit LocateAnything runtime failures."""


class LocateAnythingDependencyError(LocateAnythingBackendError):
    """Raised when a package the configured runtime path needs is not installed."""


class LocateAnythingDeviceError(LocateAnythingBackendError):
    """Raised when the configured device cannot run the configured attention path."""


class LocateAnythingModelLoadError(LocateAnythingBackendError):
    """Raised when the pinned checkpoint cannot load exactly as configured."""


class LocateAnythingInferenceError(LocateAnythingBackendError):
    """Raised for image loading or generation failures."""


@dataclass(frozen=True, kw_only=True)
class _LoadedModel:
    """Everything one loaded runtime keeps; SDK objects never leave this module."""

    torch: Any
    image_module: Any
    tokenizer: Any
    processor: Any
    model: Any
    identity: tuple[tuple[str, JsonScalar], ...]
    load_ms: float
    batch: Any = None


class TransformersLocateAnythingRuntime:
    """Lazy torch/transformers implementation of :class:`LocateAnythingRuntime`.

    Bound to one configuration and one prepared-image root. Nothing is imported or loaded
    at construction; the first :meth:`generate` (or an explicit :meth:`load`) imports the
    SDKs, checks the device and attention packages, loads the pinned revision once, and
    verifies that the loaded model uses exactly the configured attention paths, dtype and
    commit. The upstream remote code runs with ``trust_remote_code=True`` at the pinned
    revision, from the local cache unless ``local_files_only`` is disabled.
    """

    def __init__(
        self,
        *,
        config: LocateAnythingConfig,
        prepared_image_root: Path,
        environ: MutableMapping[str, str] | None = None,
    ) -> None:
        """Bind the runtime without importing any SDK.

        Args:
            config: Effective configuration; revision and paths are already validated.
            prepared_image_root: Directory that resolves ``PreparedImage.payload_reference``.
            environ: Process environment the batch runtime is configured through (its
                upstream knobs are environment variables); defaults to ``os.environ``.
        """
        self._config = config
        self._root = prepared_image_root
        self._environ = os.environ if environ is None else environ
        self._loaded: _LoadedModel | None = None

    def load(self) -> None:
        """Import, validate and load the pinned model once.

        :meth:`generate` calls this lazily; a caller that measures latency calls it first
        so the one-time load is not attributed to the first request.

        Raises:
            LocateAnythingBackendError: For a missing package, an unusable device, or a
                checkpoint that does not load exactly as configured.
        """
        if self._loaded is not None:
            return
        started = perf_counter()
        config = self._config
        torch = _import("torch", "PyTorch")
        transformers = _import("transformers", "Transformers")
        image_module = _import("PIL.Image", "Pillow")
        identity: list[tuple[str, JsonScalar]] = [
            ("runtime", config.runtime.value),
            ("python", platform.python_version()),
            ("torch", _version(torch)),
            ("torch.cuda", getattr(getattr(torch, "version", None), "cuda", None)),
            ("transformers", _version(transformers)),
            *self._check_device_and_attention(torch),
        ]
        commit: str | None
        if config.runtime is LocateAnythingRuntimeMode.BATCH:
            tokenizer, processor, model, batch, commit = self._load_batch(identity)
        else:
            tokenizer, processor, model, commit = self._load_standard(transformers, torch)
            batch = None
        expected_dtype = getattr(torch, config.dtype)
        loaded_dtype = getattr(model, "dtype", expected_dtype)
        if loaded_dtype != expected_dtype:
            raise LocateAnythingModelLoadError(
                f"LocateAnything was configured with dtype {config.dtype} but loaded as "
                f"{loaded_dtype}"
            )
        on_cuda = config.device.startswith("cuda")
        identity.append(
            ("device_name", torch.cuda.get_device_name(config.device) if on_cuda else "cpu")
        )
        identity.append(("model_commit", commit))
        self._loaded = _LoadedModel(
            torch=torch,
            image_module=image_module,
            tokenizer=tokenizer,
            processor=processor,
            model=model,
            batch=batch,
            identity=tuple(identity),
            load_ms=(perf_counter() - started) * 1000,
        )

    def generate(
        self, *, image: PreparedImage, prompt: str, config: LocateAnythingConfig
    ) -> LocateAnythingGeneration:
        """Generate one answer for the exact prepared image and rendered prompt.

        Raises:
            ValueError: If ``config`` is not the configuration this runtime was built for.
            LocateAnythingBackendError: For load, image or generation failures, including an
                image whose bytes no longer match its recorded SHA-256.
        """
        if config != self._config:
            raise ValueError("LocateAnything runtime was built for another configuration")
        cold = self._loaded is None
        self.load()
        loaded = self._loaded
        assert loaded is not None  # load() acabou de preencher ou falhou com erro explícito.
        pil_image = self._open_image(loaded, image)
        torch = loaded.torch
        on_cuda = config.device.startswith("cuda")
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(config.device)
        try:
            if loaded.batch is None:
                text, steps, statistics = self._generate_standard(loaded, pil_image, prompt)
            else:
                text, steps, statistics = self._generate_batch(loaded, pil_image, prompt)
        except LocateAnythingBackendError:
            raise
        except Exception as error:
            raise LocateAnythingInferenceError(
                f"LocateAnything generation failed: {error}"
            ) from error
        peak = int(torch.cuda.max_memory_allocated(config.device)) if on_cuda else None
        return LocateAnythingGeneration(
            text=text,
            decode_steps=steps,
            statistics=statistics,
            peak_memory_bytes=peak,
            runtime_identity=loaded.identity,
            native_diagnostics=(("runtime.cold_load_ms", loaded.load_ms),) if cold else (),
        )

    def _check_device_and_attention(self, torch: Any) -> list[tuple[str, JsonScalar]]:
        """Fail before loading when the device or an attention package cannot serve the path."""
        config = self._config
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise LocateAnythingDeviceError(
                f"configured CUDA device {config.device} is unavailable"
            )
        attentions = {config.text_attention, config.vision_attention}
        versions: list[tuple[str, JsonScalar]] = []
        if attentions & _FLASH_ATTENTION:
            flash = _import(
                "flash_attn", f"FlashAttention for {sorted(attentions & _FLASH_ATTENTION)}"
            )
            versions.append(("flash_attn", _version(flash)))
        if "magi" in attentions:
            magi = _import("magi_attention", "MagiAttention for text_attention='magi'")
            versions.append(("magi_attention", _version(magi)))
            major, _minor = torch.cuda.get_device_capability(config.device)
            if major < _MAGI_MIN_COMPUTE_CAPABILITY:
                raise LocateAnythingDeviceError(
                    "MagiAttention runs only on Hopper or Blackwell GPUs (compute capability "
                    f">= {_MAGI_MIN_COMPUTE_CAPABILITY}.0); {config.device} is {major}.{_minor}"
                )
        return versions

    def _load_standard(self, transformers: Any, torch: Any) -> tuple[Any, Any, Any, str | None]:
        """Load the upstream worker path with the configured attention and verify it."""
        config = self._config
        options = {
            "revision": config.revision,
            "local_files_only": config.local_files_only,
            "trust_remote_code": True,
        }
        try:
            model_config = transformers.AutoConfig.from_pretrained(config.model, **options)
            # O upstream lê a atenção destas três chaves; defini-las explicitamente impede o
            # default "magi"/"flash_attention_2" com fallback silencioso para SDPA.
            model_config._attn_implementation = config.text_attention
            model_config.text_config._attn_implementation = config.text_attention
            model_config.vision_config._attn_implementation = config.vision_attention
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, **options)
            processor = transformers.AutoProcessor.from_pretrained(config.model, **options)
            model = transformers.AutoModel.from_pretrained(
                config.model, config=model_config, dtype=getattr(torch, config.dtype), **options
            )
            model = model.to(config.device).eval()
        except ImportError as error:
            raise LocateAnythingDependencyError(
                f"the LocateAnything remote code needs a missing package: {error}"
            ) from error
        except Exception as error:
            source = "local cache" if config.local_files_only else "configured model source"
            raise LocateAnythingModelLoadError(
                f"could not load {config.model}@{config.revision} from {source}: {error}"
            ) from error
        for part, requested in (
            ("text_config", config.text_attention),
            ("vision_config", config.vision_attention),
        ):
            loaded = getattr(getattr(model.config, part, None), "_attn_implementation", None)
            if loaded != requested:
                raise LocateAnythingModelLoadError(
                    f"{part} attention {requested!r} was requested but the loaded model uses "
                    f"{loaded!r}; refusing the upstream fallback"
                )
        commit = getattr(model.config, "_commit_hash", None)
        if commit is not None and commit != config.revision:
            raise LocateAnythingModelLoadError(
                f"loaded commit {commit} differs from the pinned revision {config.revision}"
            )
        return tokenizer, processor, model, commit

    def _load_batch(self, identity: list[tuple[str, JsonScalar]]) -> tuple[Any, Any, Any, Any, str]:
        """Load the release batch runtime from the pinned snapshot, strictly."""
        config = self._config
        if config.model.startswith("/"):
            snapshot = Path(config.model)
        else:
            hub = _import("huggingface_hub", "to resolve the pinned snapshot")
            identity.append(("huggingface_hub", _version(hub)))
            try:
                snapshot = Path(
                    hub.snapshot_download(
                        repo_id=config.model,
                        revision=config.revision,
                        local_files_only=config.local_files_only,
                    )
                )
            except Exception as error:
                raise LocateAnythingModelLoadError(
                    f"could not resolve {config.model}@{config.revision}: {error}"
                ) from error
        if snapshot.name != config.revision:
            raise LocateAnythingModelLoadError(
                f"snapshot {snapshot} does not belong to the pinned revision {config.revision}"
            )
        # O runtime batch vem com o repositório do modelo (batch_utils/, kernel_utils/) e é
        # configurado só por variáveis de ambiente; STRICT=1 faz ele falhar em vez de cair
        # para SDPA quando o caminho de atenção pedido não está disponível.
        if str(snapshot) not in sys.path:
            sys.path.insert(0, str(snapshot))
        self._environ.update(
            {
                "LA_FLASH_MODEL": str(snapshot),
                "LA_FLASH_ATTN": config.text_attention,
                "LA_FLASH_VISION_ATTN": config.vision_attention,
                "LA_FLASH_HYBRID_SCHEDULER": str(config.scheduler),
                "LA_FLASH_HYBRID_GROUP_SIZE": str(config.group_size),
                "LA_FLASH_STRICT_ATTN": "1",
            }
        )
        batch = _import("batch_utils", "the batch runtime of the Hugging Face model release")
        try:
            tokenizer, processor, model = batch.load()
        except Exception as error:
            raise LocateAnythingModelLoadError(
                f"the LocateAnything batch runtime could not load {snapshot}: {error}"
            ) from error
        return tokenizer, processor, model, batch, config.revision

    def _open_image(self, loaded: _LoadedModel, image: PreparedImage) -> Any:
        """Read the prepared image, verify its SHA-256 and decode it as RGB."""
        artifact = image.payload_artifact
        if artifact is None:
            raise LocateAnythingInferenceError(
                "prepared image has no recorded sha256 to verify before inference"
            )
        reference = PurePosixPath(image.payload_reference)
        if reference.is_absolute() or ".." in reference.parts:
            raise LocateAnythingInferenceError(
                f"prepared image {image.payload_reference!r} points outside the prepared-image root"
            )
        path = self._root / reference
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise LocateAnythingInferenceError(
                f"could not read prepared image {path}: {error}"
            ) from error
        if hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise LocateAnythingInferenceError(
                f"prepared image {image.payload_reference!r} does not match its recorded sha256"
            )
        return loaded.image_module.open(io.BytesIO(payload)).convert("RGB")

    def _generate_standard(
        self, loaded: _LoadedModel, image: Any, prompt: str
    ) -> tuple[str, tuple[tuple[str, str], ...], str | None]:
        """Run the upstream worker ``_predict_standard`` call with the configured settings."""
        config = self._config
        processor = loaded.processor
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
        text = processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = processor.process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(
            config.device
        )
        # O generate() oficial imprime as estatísticas com verbose=True; elas já voltam no
        # retorno, então a saída padrão é descartada em vez de poluir o log do run.
        with loaded.torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            response = loaded.model.generate(
                pixel_values=inputs["pixel_values"].to(getattr(loaded.torch, config.dtype)),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=loaded.tokenizer,
                max_new_tokens=config.max_new_tokens,
                use_cache=True,
                generation_mode=config.generation_mode.value,
                temperature=config.temperature,
                do_sample=config.temperature > 0,
                top_p=config.top_p,
                top_k=None if config.top_k == 0 else config.top_k,
                repetition_penalty=config.repetition_penalty,
                verbose=True,
            )
        answer = response[0] if isinstance(response, tuple) else response
        if not isinstance(answer, str):
            raise LocateAnythingInferenceError("LocateAnything generate() returned no text answer")
        if isinstance(response, tuple) and len(response) >= 3:
            steps = tuple((str(decoder), str(piece)) for decoder, piece in response[1])
            return answer, steps, None if response[2] is None else str(response[2])
        return answer, (), None

    def _generate_batch(
        self, loaded: _LoadedModel, image: Any, prompt: str
    ) -> tuple[str, tuple[tuple[str, str], ...], str | None]:
        """Run one request through the release batch hybrid scheduler."""
        config = self._config
        with loaded.torch.no_grad():
            answers = loaded.batch.generate_batch_hybrid(
                [(image, prompt)],
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=None if config.top_k == 0 else config.top_k,
                repetition_penalty=config.repetition_penalty,
                max_new_tokens=config.max_new_tokens,
                scheduler=config.scheduler,
                group_size=config.group_size,
            )
        if len(answers) != 1 or not isinstance(answers[0], str):
            raise LocateAnythingInferenceError("batch runtime did not return one text answer")
        stats = loaded.batch.get_last_hybrid_stats()
        statistics = None if stats is None else json.dumps(stats, sort_keys=True, default=str)
        return answers[0], (), statistics


def _import(name: str, purpose: str) -> Any:
    """Import one optional module, turning its absence into an explicit dependency error."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        raise LocateAnythingDependencyError(
            f"LocateAnything needs {name!r} ({purpose}) in the runtime environment"
        ) from error


def _version(module: Any) -> JsonScalar:
    version = getattr(module, "__version__", None)
    return None if version is None else str(version)

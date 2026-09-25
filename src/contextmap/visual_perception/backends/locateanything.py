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
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from time import perf_counter
from types import MappingProxyType
from typing import Protocol

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


class LocateAnythingGenerationMode(StrEnum):
    """Upstream decoding modes: parallel boxes, autoregressive, or parallel with fallback."""

    FAST = "fast"
    SLOW = "slow"
    HYBRID = "hybrid"


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
    """

    model: str
    revision: str
    device: str
    dtype: str
    generation_mode: LocateAnythingGenerationMode
    max_new_tokens: int
    temperature: float
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1

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

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return the JSON-compatible effective configuration."""
        values = asdict(self)
        values["generation_mode"] = self.generation_mode.value
        return values

    @property
    def fingerprint(self) -> str:
        """Return the digest of every setting that can change the evidence produced."""
        document = {**self.to_dict(), "parser_version": PARSER_VERSION}
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
        warnings: Runtime warnings.
    """

    text: str
    decode_steps: tuple[tuple[str, str], ...] = ()
    statistics: str | None = None
    peak_memory_bytes: int | None = None
    runtime_identity: tuple[tuple[str, JsonScalar], ...] = ()
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
                native=_statistics_diagnostics(generation.statistics),
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

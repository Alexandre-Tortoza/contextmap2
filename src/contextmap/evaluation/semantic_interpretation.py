"""Reproducible, backend-neutral Semantic Interpretation evaluation reports.

The evaluator consumes canonical executions and never changes a claim, picks a winner, or
applies Semantic Fusion. Annotations are partial: an absent annotation means "not annotated",
so its claims are *unassessed*, never wrong. Only ``rejected_hypotheses`` are negative truth.
Quality, cost, outcomes and repeat stability are separate blocks of the report.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from contextmap.visual_perception import (
    BackendProvenance,
    HypothesisRole,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticResponseParseError,
)

MATCHING_POLICY = "casefold-exact/1"
"""Versioned open-vocabulary matching policy used by the baseline evaluator."""

_BACKEND_COMPARISON_CONTEXT_FIELDS = (
    "reference_set_version",
    "selection_id",
    "perception_run_id",
    "evaluator_version",
)
"""Context identities that make two reports the same experiment run by different backends.

They fix the reference expectations, the selected frames, the perception run whose regions were
interpreted (a ``region_id`` is only meaningful inside it) and the evaluator that scored them.
``evaluation_id``, ``artifact_id`` and ``pipeline_configuration_digest`` are excluded on purpose:
they identify each backend's own evaluation, output and pipeline, so they differ by design.
"""

PERCENTILE_METHOD = "nearest-rank"
"""Percentile convention of every latency percentile: the ceil(p * n)-th sorted value."""

SCENE_CONTEXT_FIELDS = (
    "scene_type",
    "environment",
    "layout",
    "lighting",
    "visibility",
    "navigability",
)
"""The structured ``SceneContext`` fields a scene annotation may constrain."""


class SemanticEvaluationError(ValueError):
    """Raised when semantic evidence cannot form a controlled comparison."""


class StratumSource(Enum):
    """Where a stratum label comes from, so derived labels are never read as annotations."""

    ANNOTATION = "annotation"
    DERIVED = "derived"


@dataclass(frozen=True, kw_only=True)
class SemanticStratum:
    """One stratification label of a request, for example ``visibility=occluded``."""

    scheme: str
    label: str
    source: StratumSource

    def __post_init__(self) -> None:
        """Require a scheme and a label."""
        if not self.scheme.strip() or not self.label.strip():
            raise ValueError("stratum scheme and label must not be empty")


@dataclass(frozen=True, kw_only=True)
class SemanticEvaluationContext:
    """Reproducibility identities shared by all samples in one report."""

    evaluation_id: str
    reference_set_version: str
    selection_id: str
    perception_run_id: str
    artifact_id: str
    pipeline_configuration_digest: str
    evaluator_version: str

    def __post_init__(self) -> None:
        """Require every lineage/version identity."""
        for name, value in self.__dict__.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, kw_only=True)
class SceneContextAnnotation:
    """Acceptable values of the scene fields an annotator constrained.

    Attributes:
        acceptable: Field name to literal acceptable values. Fields that are absent were
            not annotated and are not scored.
    """

    acceptable: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Reject unknown fields and fields without an acceptable value."""
        if not self.acceptable:
            raise ValueError("a scene context annotation needs at least one field")
        for name, values in self.acceptable.items():
            if name not in SCENE_CONTEXT_FIELDS:
                raise ValueError(f"unknown scene field {name!r}; use {SCENE_CONTEXT_FIELDS}")
            if not values or any(not _normalize(value) for value in values):
                raise ValueError(f"scene field {name!r} needs at least one non-empty value")


@dataclass(frozen=True, kw_only=True)
class SemanticAnnotation:
    """Versioned reference expectations for one request.

    Attributes:
        acceptable_hypotheses: Concepts an annotator accepts, matched with
            ``casefold-exact/1``. Empty when the annotation only constrains other things.
        ambiguity_expected: Whether preserving alternatives is the right answer.
        rejected_hypotheses: Concepts known **not** to apply. They are the only negative
            truth: a claim outside ``acceptable_hypotheses`` is merely unsupported.
        abstention_expected: The annotator could not tell, so abstaining is right and the
            correctness of any claim is not applicable.
        scene_context: Acceptable scene field values, for scene requests.
        visibility_stratum: Visibility/occlusion label used to stratify the report.
    """

    acceptable_hypotheses: tuple[str, ...] = ()
    ambiguity_expected: bool = False
    rejected_hypotheses: tuple[str, ...] = ()
    abstention_expected: bool = False
    scene_context: SceneContextAnnotation | None = None
    visibility_stratum: str | None = None

    def __post_init__(self) -> None:
        """Reject empty, duplicated or contradictory expectations."""
        acceptable = tuple(_normalize(value) for value in self.acceptable_hypotheses)
        rejected = tuple(_normalize(value) for value in self.rejected_hypotheses)
        if any(not value for value in (*acceptable, *rejected)):
            raise ValueError("hypotheses must be non-empty concepts")
        if len(set(acceptable)) != len(acceptable):
            raise ValueError("acceptable_hypotheses must be unique under the matching policy")
        if len(set(rejected)) != len(rejected):
            raise ValueError("rejected_hypotheses must be unique under the matching policy")
        if set(acceptable) & set(rejected):
            raise ValueError("a hypothesis cannot be both acceptable and rejected")
        if self.abstention_expected:
            if acceptable or self.ambiguity_expected or self.scene_context is not None:
                raise ValueError(
                    "an expected abstention excludes acceptable hypotheses, ambiguity and "
                    "scene context"
                )
        elif not acceptable and self.scene_context is None:
            raise ValueError(
                "an annotation must state acceptable hypotheses, a scene context, or an "
                "expected abstention"
            )
        if self.visibility_stratum is not None and not self.visibility_stratum.strip():
            raise ValueError("visibility_stratum must not be empty")


@dataclass(frozen=True, kw_only=True)
class SemanticEvaluationInput:
    """One execution, evidence variant, and its optional reference annotation.

    Attributes:
        execution: The canonical execution to evaluate.
        evidence_variant_id: Label of the evidence configuration (for example a crop kind).
        annotation: Reference expectations; ``None`` leaves the claims unassessed.
        backend: Explicit provenance, required only for an abstained execution.
        strata: Extra stratification labels of the request.
        repeat_index: ``0`` for the primary run. Repeats measure stability and cost; they are
            not independent physical evidence, so quality counts only repeat ``0``.
    """

    execution: SemanticInterpretationExecution
    evidence_variant_id: str
    annotation: SemanticAnnotation | None = None
    backend: BackendProvenance | None = None
    strata: tuple[SemanticStratum, ...] = ()
    repeat_index: int = 0

    def __post_init__(self) -> None:
        """Require a non-negative repeat index."""
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")


@dataclass(frozen=True, kw_only=True)
class SemanticEvaluationFailure:
    """A request failure classified independently of semantic quality.

    Attributes:
        failure_kind: ``parser`` (the response did not satisfy the canonical schema) or
            ``backend`` (the model, runtime or provider failed).
        raw_response: The rejected text for a parser failure, so truncation or schema drift
            can be told from a model that answered wrongly.
        mode: ``scene`` or ``region``.
        source_observation_id: Frame the failed request interpreted.
        region_id: Region the failed request interpreted; ``None`` for a scene request.

    ``mode``, ``source_observation_id`` and ``region_id`` are the physical identity of the
    failed request. A single report does not need them, but ``compare_semantic_backends``
    rejects a failure that lacks them, because it could not tell which input failed.
    """

    request_id: str
    evidence_variant_id: str
    failure_kind: str
    message: str
    mode: str | None = None
    source_observation_id: str | None = None
    region_id: str | None = None
    raw_response: str | None = None
    strata: tuple[SemanticStratum, ...] = ()
    repeat_index: int = 0

    def __post_init__(self) -> None:
        """Validate explicit parser/backend failure classification."""
        if self.failure_kind not in {"parser", "backend"}:
            raise ValueError("failure_kind must be parser or backend")
        if not self.request_id or not self.evidence_variant_id or not self.message:
            raise ValueError("semantic evaluation failure fields must not be empty")
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")

    @classmethod
    def from_error(
        cls,
        request: SemanticInterpretationRequest,
        *,
        evidence_variant_id: str,
        error: Exception,
        repeat_index: int = 0,
        strata: tuple[SemanticStratum, ...] = (),
    ) -> SemanticEvaluationFailure:
        """Classify an exception raised while interpreting ``request``."""
        is_parser = isinstance(error, SemanticResponseParseError)
        return cls(
            request_id=str(request.request_id),
            evidence_variant_id=evidence_variant_id,
            failure_kind="parser" if is_parser else "backend",
            message=str(error) if is_parser else f"{type(error).__name__}: {error}",
            mode=request.mode.value,
            source_observation_id=str(request.source_observation_id),
            region_id=None if request.region_id is None else str(request.region_id),
            raw_response=error.raw_response
            if isinstance(error, SemanticResponseParseError)
            else None,
            strata=strata,
            repeat_index=repeat_index,
        )


@dataclass(frozen=True, kw_only=True)
class SceneFieldOutcome:
    """How one annotated scene field was answered: correct, incorrect or missing."""

    field: str
    outcome: str


@dataclass(frozen=True, kw_only=True)
class SemanticSampleReport:
    """Traceable quality/cost observations for one canonical request execution."""

    request_id: str
    source_observation_id: str
    perception_result_id: str
    region_id: str | None
    mode: str
    repeat_index: int
    evidence_variant_id: str
    evidence_channels: tuple[str, ...]
    backend: BackendProvenance
    prompt_template_id: str
    prompt_fingerprint: str
    raw_response_sha256: str
    annotated: bool
    claim_count: int
    acceptable_claim_count: int | None
    unsupported_claim_count: int | None
    rejected_claim_count: int | None
    alternative_claim_count: int
    ambiguity_preserved: bool | None
    abstained: bool
    abstention_expected: bool | None
    duplicate_claim_count: int
    primary_hypothesis: str | None
    answer_digest: str
    scene_field_outcomes: tuple[SceneFieldOutcome, ...]
    strata: tuple[SemanticStratum, ...]
    latency_ms: float
    retries: int
    input_tokens: int | None
    output_tokens: int | None
    peak_memory_bytes: int | None


@dataclass(frozen=True, kw_only=True)
class SceneFieldQuality:
    """Aggregated correctness of one annotated scene field."""

    field: str
    correct: int
    incorrect: int
    missing: int


@dataclass(frozen=True, kw_only=True)
class SemanticQualityReport:
    """Aggregated semantic quality of the primary run, without runtime/cost conflation.

    Rates are ``None`` when nothing could be assessed: with no annotation there is no
    "acceptable" or "unsupported", only claims.
    """

    claim_count: int
    annotated_sample_count: int
    assessed_claim_count: int
    acceptable_claim_rate: float | None
    unsupported_claim_rate: float | None
    rejected_claim_rate: float | None
    ambiguity_preservation_rate: float | None
    abstention_count: int
    expected_abstention_count: int
    correct_abstention_count: int
    unexpected_abstention_count: int
    duplicate_claim_count: int
    scene_context: tuple[SceneFieldQuality, ...]
    distinct_hypothesis_count: int
    dominant_hypothesis: str | None
    dominant_hypothesis_share: float | None


@dataclass(frozen=True, kw_only=True)
class SemanticModeCost:
    """Latency and resources of one mode (scene or region)."""

    mode: str
    request_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    peak_memory_bytes: int | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, kw_only=True)
class SemanticCostReport:
    """Runtime, retries, usage, and memory of every execution, kept apart from quality.

    Cost measures every execution including repeats; ``peak_memory_bytes`` is the process
    peak the backend reported, which for a local model includes its weights.
    """

    request_count: int
    total_latency_ms: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    percentile_method: str
    retry_count: int
    input_tokens: int | None
    output_tokens: int | None
    peak_memory_bytes: int | None
    by_mode: tuple[SemanticModeCost, ...]


@dataclass(frozen=True, kw_only=True)
class SemanticOutcomeReport:
    """What happened to the requests of one mode in the primary run."""

    mode: str
    requests: int
    interpreted: int
    abstained: int
    parser_failures: int
    backend_failures: int


@dataclass(frozen=True, kw_only=True)
class RepeatConsistencyReport:
    """Stability of repeated executions of the same request and evidence variant.

    Attributes:
        compared_request_count: Requests with at least two executions or failures.
        identical_raw_response_count: Requests whose repeats returned the same raw text.
        identical_answer_count: Requests whose repeats returned the same normalized answer.
        divergent_requests: ``request_id::variant`` of requests whose answers differ,
            including a repeat that failed where another succeeded.
    """

    compared_request_count: int
    identical_raw_response_count: int
    identical_answer_count: int
    divergent_requests: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class SemanticStratumReport:
    """Primary-run outcomes inside one stratum."""

    scheme: str
    label: str
    source: StratumSource
    sample_count: int
    failure_count: int
    claim_count: int
    acceptable_claim_rate: float | None
    unsupported_claim_rate: float | None
    rejected_claim_rate: float | None
    abstention_count: int


@dataclass(frozen=True, kw_only=True)
class SemanticEvaluationReport:
    """Common report schema for Qwen, Gemini, Florence-2, and future adapters."""

    context: SemanticEvaluationContext
    matching_policy: str
    samples: tuple[SemanticSampleReport, ...]
    failures: tuple[SemanticEvaluationFailure, ...]
    quality: SemanticQualityReport
    cost: SemanticCostReport
    outcomes: tuple[SemanticOutcomeReport, ...]
    consistency: RepeatConsistencyReport
    strata: tuple[SemanticStratumReport, ...]


@dataclass(frozen=True, kw_only=True)
class SemanticRequestOutcomes:
    """The outcome of one request per compared backend, in report order.

    Attributes:
        source_observation_id: Frame every compared backend interpreted for this request.
        region_id: Region every compared backend interpreted, ``None`` for a scene request.
        mode: ``scene`` or ``region``, shared by every compared backend.
    """

    request_id: str
    evidence_variant_id: str
    source_observation_id: str
    region_id: str | None
    mode: str
    outcomes: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class SemanticBackendComparison:
    """Reports aligned over the same attempted requests, physical inputs and evidence variants.

    Attributes:
        outcomes: Per request, ``claims``, ``abstained``, ``parser_failure`` or
            ``backend_failure`` for each report.
        primary_comparable_count: Requests where every backend produced a primary hypothesis.
        primary_agreement_count: Of those, requests where all primaries match under the
            matching policy. Agreement between backends is not correctness.
    """

    request_ids: tuple[str, ...]
    reports: tuple[SemanticEvaluationReport, ...]
    outcomes: tuple[SemanticRequestOutcomes, ...]
    primary_comparable_count: int
    primary_agreement_count: int


@dataclass(frozen=True, kw_only=True)
class EvidenceVariantEntry:
    """One evidence variant measured on the regions every compared variant covers."""

    variant_id: str
    evidence_channels: tuple[str, ...]
    prompt_template_ids: tuple[str, ...]
    sample_count: int
    failure_count: int
    abstention_count: int
    acceptable_claim_rate: float | None
    unsupported_claim_rate: float | None
    rejected_claim_rate: float | None
    latency_p50_ms: float | None
    acceptable_claim_rate_delta: float | None
    unsupported_claim_rate_delta: float | None
    primary_comparable_count: int | None
    primary_agreement_count: int | None


@dataclass(frozen=True, kw_only=True)
class EvidenceVariantComparison:
    """A controlled comparison of evidence variants against a baseline variant."""

    baseline_variant_id: str
    paired_region_count: int
    entries: tuple[EvidenceVariantEntry, ...]


def evaluate_semantic_interpretation(
    *,
    context: SemanticEvaluationContext,
    inputs: tuple[SemanticEvaluationInput, ...],
    failures: tuple[SemanticEvaluationFailure, ...] = (),
) -> SemanticEvaluationReport:
    """Evaluate canonical executions with an explicit matching convention."""
    if not inputs and not failures:
        raise SemanticEvaluationError("semantic evaluation requires inputs or failures")
    keys = [
        (item.execution.request.request_id, item.evidence_variant_id, item.repeat_index)
        for item in inputs
    ] + [(item.request_id, item.evidence_variant_id, item.repeat_index) for item in failures]
    duplicated = [key for key, count in Counter(keys).items() if count > 1]
    if duplicated:
        raise SemanticEvaluationError(f"duplicate request/variant/repeat: {duplicated[0]}")
    samples = tuple(_evaluate_sample(item) for item in inputs)
    return SemanticEvaluationReport(
        context=context,
        matching_policy=MATCHING_POLICY,
        samples=samples,
        failures=failures,
        quality=_quality(samples),
        cost=_cost(samples),
        outcomes=_outcomes(samples, failures),
        consistency=_consistency(samples, failures),
        strata=_strata(samples, failures),
    )


def compare_semantic_backends(
    reports: tuple[SemanticEvaluationReport, ...],
) -> SemanticBackendComparison:
    """Align backend reports only when they attempted the same physical inputs.

    A request is aligned by its request id, evidence variant, source observation, region and
    mode, so two reports that reuse the same request ids for other frames or regions are not
    comparable. A failure is an attempt like any other and must carry the same identity.

    Args:
        reports: One report per backend, in the order the outcomes are listed.

    Returns:
        The per-request outcomes and the agreement between primary hypotheses.

    Raises:
        SemanticEvaluationError: If there are fewer than two reports; the reports differ in
            reference set, selection, perception run, evaluator version or matching policy;
            a failure lacks the physical identity needed to align it; or the reports did not
            attempt exactly the same requests over the same observations, regions, modes and
            evidence variants.
    """
    if len(reports) < 2:
        raise SemanticEvaluationError("backend comparison requires at least two reports")
    _require_shared_experiment(reports)
    attempts = [_attempts(report) for report in reports]
    for index, other in enumerate(attempts[1:], start=1):
        if other.keys() != attempts[0].keys():
            raise SemanticEvaluationError(_misalignment(attempts[0].keys(), other.keys(), index))
    keys = sorted(attempts[0], key=_alignment_order)
    outcomes = tuple(
        SemanticRequestOutcomes(
            request_id=key.request_id,
            evidence_variant_id=key.evidence_variant_id,
            source_observation_id=key.source_observation_id,
            region_id=key.region_id,
            mode=key.mode,
            outcomes=tuple(report_attempts[key].outcome for report_attempts in attempts),
        )
        for key in keys
    )
    comparable = agreement = 0
    for key in keys:
        primaries = [report_attempts[key].primary_hypothesis for report_attempts in attempts]
        if all(primary is not None for primary in primaries):
            comparable += 1
            agreement += len(set(primaries)) == 1
    return SemanticBackendComparison(
        request_ids=tuple(sorted({key.request_id for key in keys})),
        reports=reports,
        outcomes=outcomes,
        primary_comparable_count=comparable,
        primary_agreement_count=agreement,
    )


def compare_evidence_variants(
    report: SemanticEvaluationReport, *, baseline_variant_id: str
) -> EvidenceVariantComparison:
    """Compare evidence variants of one backend on the regions every variant covers.

    Pairing is by physical observation and region, so the same backend and model answered
    the same region under each evidence configuration. The channels of each variant are read
    from the requests (view kinds, features, scene context), not from the variant label.
    """
    primary = [sample for sample in report.samples if sample.repeat_index == 0]
    primary_failures = [item for item in report.failures if item.repeat_index == 0]
    variants = sorted(
        {sample.evidence_variant_id for sample in primary}
        | {item.evidence_variant_id for item in primary_failures}
    )
    if baseline_variant_id not in variants:
        raise SemanticEvaluationError(f"baseline variant {baseline_variant_id!r} was not evaluated")
    covered: dict[str, set[tuple[str, str | None]]] = {variant: set() for variant in variants}
    for sample in primary:
        covered[sample.evidence_variant_id].add((sample.source_observation_id, sample.region_id))
    for item in primary_failures:
        if item.source_observation_id is not None:
            covered[item.evidence_variant_id].add((item.source_observation_id, item.region_id))
    paired = set.intersection(*covered.values())
    if not paired:
        raise SemanticEvaluationError("no region is present in every evidence variant")
    ordered = [
        baseline_variant_id,
        *(variant for variant in variants if variant != baseline_variant_id),
    ]
    entries: list[EvidenceVariantEntry] = []
    baseline_rates: tuple[float | None, float | None] = (None, None)
    baseline_primaries: dict[tuple[str, str | None], str | None] = {}
    for variant in ordered:
        samples = tuple(
            sample
            for sample in primary
            if sample.evidence_variant_id == variant
            and (sample.source_observation_id, sample.region_id) in paired
        )
        failure_count = sum(
            1
            for item in primary_failures
            if item.evidence_variant_id == variant
            and (item.source_observation_id, item.region_id) in paired
        )
        quality = _quality(samples) if samples else None
        acceptable = None if quality is None else quality.acceptable_claim_rate
        unsupported = None if quality is None else quality.unsupported_claim_rate
        is_baseline = variant == baseline_variant_id
        primaries = {
            (sample.source_observation_id, sample.region_id): sample.primary_hypothesis
            for sample in samples
        }
        if is_baseline:
            baseline_rates = (acceptable, unsupported)
            baseline_primaries = primaries
        # Sensibilidade à evidência sem anotação: a mesma região mantém a hipótese primária?
        comparable = [
            key
            for key, value in primaries.items()
            if value is not None and baseline_primaries.get(key) is not None
        ]
        entries.append(
            EvidenceVariantEntry(
                variant_id=variant,
                evidence_channels=tuple(
                    sorted({channel for sample in samples for channel in sample.evidence_channels})
                ),
                prompt_template_ids=tuple(
                    sorted({sample.prompt_template_id for sample in samples})
                ),
                sample_count=len(samples),
                failure_count=failure_count,
                abstention_count=sum(sample.abstained for sample in samples),
                acceptable_claim_rate=acceptable,
                unsupported_claim_rate=unsupported,
                rejected_claim_rate=None if quality is None else quality.rejected_claim_rate,
                latency_p50_ms=_percentile([sample.latency_ms for sample in samples], 50),
                acceptable_claim_rate_delta=_delta(acceptable, baseline_rates[0], is_baseline),
                unsupported_claim_rate_delta=_delta(unsupported, baseline_rates[1], is_baseline),
                primary_comparable_count=None if is_baseline else len(comparable),
                primary_agreement_count=(
                    None
                    if is_baseline
                    else sum(primaries[key] == baseline_primaries[key] for key in comparable)
                ),
            )
        )
    return EvidenceVariantComparison(
        baseline_variant_id=baseline_variant_id,
        paired_region_count=len(paired),
        entries=tuple(entries),
    )


def encode_semantic_evaluation_report(report: SemanticEvaluationReport) -> dict[str, Any]:
    """Encode a report into JSON primitives, keeping every identity and identifier."""
    encoded = _plain(report)
    assert isinstance(encoded, dict)
    return encoded


def encode_semantic_backend_comparison(comparison: SemanticBackendComparison) -> dict[str, Any]:
    """Encode a backend comparison, with the full report of every backend."""
    encoded = _plain(comparison)
    assert isinstance(encoded, dict)
    return encoded


def encode_evidence_variant_comparison(comparison: EvidenceVariantComparison) -> dict[str, Any]:
    """Encode an evidence-variant comparison into JSON primitives."""
    encoded = _plain(comparison)
    assert isinstance(encoded, dict)
    return encoded


def _evaluate_sample(item: SemanticEvaluationInput) -> SemanticSampleReport:
    execution = item.execution
    request = execution.request
    annotation = item.annotation
    claims = tuple(execution.parsed.claims)
    if execution.parsed.scene_context is not None:
        claims += tuple(execution.parsed.scene_context.claims)
    normalized = tuple(_normalize(claim.hypothesis) for claim in claims)
    acceptable_set = (
        {_normalize(value) for value in annotation.acceptable_hypotheses} if annotation else set()
    )
    rejected_set = (
        {_normalize(value) for value in annotation.rejected_hypotheses} if annotation else set()
    )
    assessed = bool(acceptable_set)
    acceptable_count = sum(value in acceptable_set for value in normalized)
    alternatives = sum(claim.role is HypothesisRole.ALTERNATIVE for claim in claims)
    primaries = [
        value
        for claim, value in zip(claims, normalized, strict=True)
        if claim.role is HypothesisRole.PRIMARY
    ]
    scene_outcomes = _scene_outcomes(annotation, execution)
    strata = _sample_strata(item)
    channels = {f"view:{view.kind.value}" for view in request.visual_views}
    if request.visual_features:
        channels.add("visual_features")
    if request.scene_context_reference is not None:
        channels.add("scene_context")
    if request.supporting_metadata:
        channels.add("supporting_metadata")
    return SemanticSampleReport(
        request_id=str(request.request_id),
        source_observation_id=str(request.source_observation_id),
        perception_result_id=str(request.perception_result_id),
        region_id=None if request.region_id is None else str(request.region_id),
        mode=request.mode.value,
        repeat_index=item.repeat_index,
        evidence_variant_id=item.evidence_variant_id,
        evidence_channels=tuple(sorted(channels)),
        backend=item.backend or _backend_from_outputs(execution),
        prompt_template_id=execution.rendered_prompt.template_id,
        prompt_fingerprint=execution.rendered_prompt.fingerprint,
        raw_response_sha256=execution.parsed.raw_response_sha256,
        annotated=annotation is not None,
        claim_count=len(claims),
        acceptable_claim_count=acceptable_count if assessed else None,
        unsupported_claim_count=len(claims) - acceptable_count if assessed else None,
        rejected_claim_count=(
            sum(value in rejected_set for value in normalized) if rejected_set else None
        ),
        alternative_claim_count=alternatives,
        ambiguity_preserved=(
            alternatives > 0 if annotation is not None and annotation.ambiguity_expected else None
        ),
        abstained=execution.parsed.abstained,
        abstention_expected=None if annotation is None else annotation.abstention_expected,
        duplicate_claim_count=len(normalized) - len(set(normalized)),
        primary_hypothesis=primaries[0] if primaries else None,
        answer_digest=_answer_digest(execution, normalized, claims),
        scene_field_outcomes=scene_outcomes,
        strata=strata,
        latency_ms=execution.diagnostics.latency_ms,
        retries=execution.diagnostics.retries,
        input_tokens=execution.diagnostics.input_tokens,
        output_tokens=execution.diagnostics.output_tokens,
        peak_memory_bytes=execution.diagnostics.peak_memory_bytes,
    )


def _sample_strata(item: SemanticEvaluationInput) -> tuple[SemanticStratum, ...]:
    """Merge explicit strata with the annotation's visibility label, rejecting conflicts."""
    strata = item.strata
    annotation = item.annotation
    if annotation is not None and annotation.visibility_stratum is not None:
        if any(stratum.scheme == "visibility" for stratum in strata):
            raise SemanticEvaluationError(
                "visibility is stratified by both the annotation and an explicit stratum for "
                f"request {item.execution.request.request_id}"
            )
        strata = (
            *strata,
            SemanticStratum(
                scheme="visibility",
                label=annotation.visibility_stratum,
                source=StratumSource.ANNOTATION,
            ),
        )
    return strata


def _scene_outcomes(
    annotation: SemanticAnnotation | None, execution: SemanticInterpretationExecution
) -> tuple[SceneFieldOutcome, ...]:
    if annotation is None or annotation.scene_context is None:
        return ()
    context = execution.parsed.scene_context
    outcomes = []
    for name, acceptable in sorted(annotation.scene_context.acceptable.items()):
        value = None if context is None else getattr(context, name)
        if value is None:
            outcome = "missing"
        elif _normalize(value) in {_normalize(candidate) for candidate in acceptable}:
            outcome = "correct"
        else:
            outcome = "incorrect"
        outcomes.append(SceneFieldOutcome(field=name, outcome=outcome))
    return tuple(outcomes)


def _answer_digest(
    execution: SemanticInterpretationExecution, normalized: tuple[str, ...], claims: Sequence[Any]
) -> str:
    """Hash the normalized answer so repeats can be compared without their raw text."""
    context = execution.parsed.scene_context
    scene = (
        {}
        if context is None
        else {
            name: _normalize(value)
            for name in SCENE_CONTEXT_FIELDS
            if (value := getattr(context, name)) is not None
        }
    )
    answer = {
        "abstained": execution.parsed.abstained,
        "claims": sorted(
            [claim.role.value, hypothesis]
            for claim, hypothesis in zip(claims, normalized, strict=True)
        ),
        "scene": scene,
    }
    return "sha256:" + hashlib.sha256(json.dumps(answer, sort_keys=True).encode()).hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _quality(samples: Sequence[SemanticSampleReport]) -> SemanticQualityReport:
    primary = [sample for sample in samples if sample.repeat_index == 0]
    assessed = [sample for sample in primary if sample.acceptable_claim_count is not None]
    rejected = [sample for sample in primary if sample.rejected_claim_count is not None]
    assessed_claims = sum(sample.claim_count for sample in assessed)
    ambiguity = [
        sample.ambiguity_preserved for sample in primary if sample.ambiguity_preserved is not None
    ]
    expected = [sample for sample in primary if sample.abstention_expected]
    hypotheses = Counter(
        sample.primary_hypothesis for sample in primary if sample.primary_hypothesis is not None
    )
    dominant = (
        min(hypotheses, key=lambda value: (-hypotheses[value], value)) if hypotheses else None
    )
    scene_fields: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in primary:
        for outcome in sample.scene_field_outcomes:
            scene_fields[outcome.field][outcome.outcome] += 1
    return SemanticQualityReport(
        claim_count=sum(sample.claim_count for sample in primary),
        annotated_sample_count=sum(sample.annotated for sample in primary),
        assessed_claim_count=assessed_claims,
        acceptable_claim_rate=_rate(
            sum(sample.acceptable_claim_count or 0 for sample in assessed), assessed_claims
        ),
        unsupported_claim_rate=_rate(
            sum(sample.unsupported_claim_count or 0 for sample in assessed), assessed_claims
        ),
        rejected_claim_rate=_rate(
            sum(sample.rejected_claim_count or 0 for sample in rejected),
            sum(sample.claim_count for sample in rejected),
        ),
        ambiguity_preservation_rate=_rate(sum(ambiguity), len(ambiguity)),
        abstention_count=sum(sample.abstained for sample in primary),
        expected_abstention_count=len(expected),
        correct_abstention_count=sum(sample.abstained for sample in expected),
        unexpected_abstention_count=sum(
            sample.abstained
            for sample in primary
            if sample.annotated and sample.abstention_expected is False
        ),
        duplicate_claim_count=sum(sample.duplicate_claim_count for sample in primary),
        scene_context=tuple(
            SceneFieldQuality(
                field=name,
                correct=counts["correct"],
                incorrect=counts["incorrect"],
                missing=counts["missing"],
            )
            for name, counts in sorted(scene_fields.items())
        ),
        distinct_hypothesis_count=len(hypotheses),
        dominant_hypothesis=dominant,
        dominant_hypothesis_share=(
            None if dominant is None else hypotheses[dominant] / sum(hypotheses.values())
        ),
    )


def _cost(samples: Sequence[SemanticSampleReport]) -> SemanticCostReport:
    latencies = [sample.latency_ms for sample in samples]
    by_mode: dict[str, list[SemanticSampleReport]] = defaultdict(list)
    for sample in samples:
        by_mode[sample.mode].append(sample)
    return SemanticCostReport(
        request_count=len(samples),
        total_latency_ms=sum(latencies),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        percentile_method=PERCENTILE_METHOD,
        retry_count=sum(sample.retries for sample in samples),
        input_tokens=_optional_sum(sample.input_tokens for sample in samples),
        output_tokens=_optional_sum(sample.output_tokens for sample in samples),
        peak_memory_bytes=_optional_max(sample.peak_memory_bytes for sample in samples),
        by_mode=tuple(
            SemanticModeCost(
                mode=mode,
                request_count=len(group),
                latency_p50_ms=_required(_percentile([s.latency_ms for s in group], 50)),
                latency_p95_ms=_required(_percentile([s.latency_ms for s in group], 95)),
                peak_memory_bytes=_optional_max(s.peak_memory_bytes for s in group),
                input_tokens=_optional_sum(s.input_tokens for s in group),
                output_tokens=_optional_sum(s.output_tokens for s in group),
            )
            for mode, group in sorted(by_mode.items())
        ),
    )


def _outcomes(
    samples: Sequence[SemanticSampleReport], failures: Sequence[SemanticEvaluationFailure]
) -> tuple[SemanticOutcomeReport, ...]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        if sample.repeat_index == 0:
            counts[sample.mode]["abstained" if sample.abstained else "interpreted"] += 1
    for failure in failures:
        if failure.repeat_index == 0:
            counts[failure.mode or "unknown"][f"{failure.failure_kind}_failures"] += 1
    return tuple(
        SemanticOutcomeReport(
            mode=mode,
            requests=sum(group.values()),
            interpreted=group["interpreted"],
            abstained=group["abstained"],
            parser_failures=group["parser_failures"],
            backend_failures=group["backend_failures"],
        )
        for mode, group in sorted(counts.items())
    )


def _consistency(
    samples: Sequence[SemanticSampleReport], failures: Sequence[SemanticEvaluationFailure]
) -> RepeatConsistencyReport:
    raw: dict[tuple[str, str], list[str]] = defaultdict(list)
    answers: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sample in samples:
        key = (sample.request_id, sample.evidence_variant_id)
        raw[key].append(sample.raw_response_sha256)
        answers[key].append(sample.answer_digest)
    for failure in failures:
        key = (failure.request_id, failure.evidence_variant_id)
        raw[key].append(
            hashlib.sha256(failure.raw_response.encode()).hexdigest()
            if failure.raw_response is not None
            else f"failure:{failure.failure_kind}:{failure.message}"
        )
        answers[key].append(f"failure:{failure.failure_kind}")
    compared = sorted(key for key, values in raw.items() if len(values) > 1)
    return RepeatConsistencyReport(
        compared_request_count=len(compared),
        identical_raw_response_count=sum(len(set(raw[key])) == 1 for key in compared),
        identical_answer_count=sum(len(set(answers[key])) == 1 for key in compared),
        divergent_requests=tuple(
            f"{request_id}::{variant}"
            for request_id, variant in compared
            if len(set(answers[(request_id, variant)])) > 1
        ),
    )


def _strata(
    samples: Sequence[SemanticSampleReport], failures: Sequence[SemanticEvaluationFailure]
) -> tuple[SemanticStratumReport, ...]:
    grouped_samples: dict[tuple[str, str, StratumSource], list[SemanticSampleReport]] = defaultdict(
        list
    )
    failure_counts: Counter[tuple[str, str, StratumSource]] = Counter()
    for sample in samples:
        if sample.repeat_index == 0:
            for stratum in sample.strata:
                grouped_samples[(stratum.scheme, stratum.label, stratum.source)].append(sample)
    for failure in failures:
        if failure.repeat_index == 0:
            for stratum in failure.strata:
                failure_counts[(stratum.scheme, stratum.label, stratum.source)] += 1
    reports = []
    for key in sorted(
        set(grouped_samples) | set(failure_counts), key=lambda k: (k[0], k[1], k[2].value)
    ):
        group = grouped_samples.get(key, [])
        quality = _quality(group)
        reports.append(
            SemanticStratumReport(
                scheme=key[0],
                label=key[1],
                source=key[2],
                sample_count=len(group),
                failure_count=failure_counts[key],
                claim_count=quality.claim_count,
                acceptable_claim_rate=quality.acceptable_claim_rate,
                unsupported_claim_rate=quality.unsupported_claim_rate,
                rejected_claim_rate=quality.rejected_claim_rate,
                abstention_count=quality.abstention_count,
            )
        )
    return tuple(reports)


@dataclass(frozen=True)
class _RequestKey:
    """What a backend comparison aligns on: the request, its evidence and its physical input."""

    request_id: str
    evidence_variant_id: str
    source_observation_id: str
    region_id: str | None
    mode: str


@dataclass(frozen=True)
class _Attempt:
    """The outcome of one attempted request in one report."""

    outcome: str
    primary_hypothesis: str | None


def _alignment_order(key: _RequestKey) -> tuple[str, str, str, str, str]:
    return (
        key.request_id,
        key.evidence_variant_id,
        key.source_observation_id,
        key.mode,
        key.region_id or "",
    )


def _describe(key: _RequestKey) -> str:
    return (
        f"request {key.request_id!r} (variant {key.evidence_variant_id!r}, "
        f"observation {key.source_observation_id!r}, region {key.region_id!r}, "
        f"mode {key.mode!r})"
    )


def _require_shared_experiment(reports: Sequence[SemanticEvaluationReport]) -> None:
    """Reject reports whose reference set, selection, run, evaluator or matcher differ."""
    first = reports[0]
    for index, report in enumerate(reports[1:], start=1):
        for name in _BACKEND_COMPARISON_CONTEXT_FIELDS:
            expected = getattr(first.context, name)
            found = getattr(report.context, name)
            if expected != found:
                raise SemanticEvaluationError(
                    f"backend reports must share {name}: report 0 has {expected!r}, "
                    f"report {index} has {found!r}"
                )
        if report.matching_policy != first.matching_policy:
            raise SemanticEvaluationError(
                f"backend reports must share matching_policy: report 0 has "
                f"{first.matching_policy!r}, report {index} has {report.matching_policy!r}"
            )


def _attempts(report: SemanticEvaluationReport) -> dict[_RequestKey, _Attempt]:
    """Map every primary-run attempt, success or failure, to its physical alignment key."""
    attempts = {
        _RequestKey(
            request_id=sample.request_id,
            evidence_variant_id=sample.evidence_variant_id,
            source_observation_id=sample.source_observation_id,
            region_id=sample.region_id,
            mode=sample.mode,
        ): _Attempt(
            outcome="abstained" if sample.abstained else "claims",
            primary_hypothesis=sample.primary_hypothesis,
        )
        for sample in report.samples
        if sample.repeat_index == 0
    }
    for failure in report.failures:
        if failure.repeat_index == 0:
            attempts[_failure_key(failure)] = _Attempt(
                outcome=f"{failure.failure_kind}_failure", primary_hypothesis=None
            )
    return attempts


def _failure_key(failure: SemanticEvaluationFailure) -> _RequestKey:
    """Build the alignment key of a failure, which must name the physical input it failed on."""
    if failure.mode is None or failure.source_observation_id is None:
        raise SemanticEvaluationError(
            f"failure of request {failure.request_id!r} (variant "
            f"{failure.evidence_variant_id!r}) lacks the physical identity a comparison needs: "
            "mode and source_observation_id (and region_id for a region request)"
        )
    if (failure.mode == SemanticInterpretationMode.REGION.value) != (failure.region_id is not None):
        raise SemanticEvaluationError(
            f"failure of request {failure.request_id!r} (variant "
            f"{failure.evidence_variant_id!r}) has an inconsistent physical identity: "
            f"region_id must be present exactly for region mode, got mode {failure.mode!r} "
            f"and region_id {failure.region_id!r}"
        )
    return _RequestKey(
        request_id=failure.request_id,
        evidence_variant_id=failure.evidence_variant_id,
        source_observation_id=failure.source_observation_id,
        region_id=failure.region_id,
        mode=failure.mode,
    )


def _misalignment(first: Iterable[_RequestKey], other: Iterable[_RequestKey], index: int) -> str:
    """Name the first request each side attempted that the other did not."""
    first_keys, other_keys = set(first), set(other)
    only_first = min(first_keys - other_keys, key=_alignment_order, default=None)
    only_other = min(other_keys - first_keys, key=_alignment_order, default=None)
    differences = []
    if only_first is not None:
        differences.append(f"only report 0 has {_describe(only_first)}")
    if only_other is not None:
        differences.append(f"only report {index} has {_describe(only_other)}")
    return (
        "backend reports must evaluate the same requests over the same physical inputs "
        "(request id, evidence variant, source observation, region and mode): "
        + "; ".join(differences)
    )


def _backend_from_outputs(execution: SemanticInterpretationExecution) -> BackendProvenance:
    if execution.parsed.claims:
        return execution.parsed.claims[0].provenance.backend
    context = execution.parsed.scene_context
    if context is not None:
        return context.provenance.backend
    raise SemanticEvaluationError(
        "abstained execution requires explicit backend provenance in SemanticEvaluationInput"
    )


def _percentile(values: Sequence[float], percent: int) -> float | None:
    """Return the nearest-rank percentile: the ceil(percent/100 * n)-th sorted value."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percent / 100 * len(ordered))) - 1]


def _required(value: float | None) -> float:
    if value is None:
        raise SemanticEvaluationError("a percentile needs at least one measurement")
    return value


def _delta(value: float | None, baseline: float | None, is_baseline: bool) -> float | None:
    if is_baseline or value is None or baseline is None:
        return None
    return value - baseline


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _optional_sum(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _optional_max(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _plain(value: object) -> object:
    """Convert report dataclasses, enums, mappings and sequences into JSON primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value

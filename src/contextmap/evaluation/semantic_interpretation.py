"""Reproducible, backend-neutral Semantic Interpretation evaluation reports."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from contextmap.visual_perception import (
    BackendProvenance,
    HypothesisRole,
    SemanticInterpretationExecution,
)

MATCHING_POLICY = "casefold-exact/1"
"""Versioned open-vocabulary matching policy used by the baseline evaluator."""


class SemanticEvaluationError(ValueError):
    """Raised when semantic evidence cannot form a controlled comparison."""


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
class SemanticAnnotation:
    """Versioned acceptable concepts and ambiguity expectation for one request."""

    acceptable_hypotheses: tuple[str, ...]
    ambiguity_expected: bool = False

    def __post_init__(self) -> None:
        """Reject empty or duplicated normalized concepts."""
        normalized = tuple(_normalize(value) for value in self.acceptable_hypotheses)
        if not normalized or any(not value for value in normalized):
            raise ValueError("acceptable_hypotheses must contain non-empty concepts")
        if len(set(normalized)) != len(normalized):
            raise ValueError("acceptable_hypotheses must be unique under the matching policy")


@dataclass(frozen=True, kw_only=True)
class SemanticEvaluationInput:
    """One execution, evidence ablation variant, and its reference annotation."""

    execution: SemanticInterpretationExecution
    evidence_variant_id: str
    annotation: SemanticAnnotation
    backend: BackendProvenance | None = None


@dataclass(frozen=True, kw_only=True)
class SemanticEvaluationFailure:
    """A request failure classified independently of semantic quality."""

    request_id: str
    evidence_variant_id: str
    failure_kind: str
    message: str

    def __post_init__(self) -> None:
        """Validate explicit parser/backend failure classification."""
        if self.failure_kind not in {"parser", "backend"}:
            raise ValueError("failure_kind must be parser or backend")
        if not self.request_id or not self.evidence_variant_id or not self.message:
            raise ValueError("semantic evaluation failure fields must not be empty")


@dataclass(frozen=True, kw_only=True)
class SemanticSampleReport:
    """Traceable quality/cost observations for one canonical request."""

    request_id: str
    source_observation_id: str
    perception_result_id: str
    region_id: str | None
    evidence_variant_id: str
    backend: BackendProvenance
    prompt_template_id: str
    prompt_fingerprint: str
    acceptable_claim_count: int
    unsupported_claim_count: int
    alternative_claim_count: int
    ambiguity_preserved: bool | None
    abstained: bool
    duplicate_claim_count: int
    latency_ms: float
    retries: int
    input_tokens: int | None
    output_tokens: int | None
    peak_memory_bytes: int | None


@dataclass(frozen=True, kw_only=True)
class SemanticQualityReport:
    """Aggregated semantic quality without runtime/cost conflation."""

    claim_count: int
    acceptable_claim_rate: float
    unsupported_claim_rate: float
    ambiguity_preservation_rate: float | None
    abstention_count: int
    duplicate_claim_count: int


@dataclass(frozen=True, kw_only=True)
class SemanticCostReport:
    """Runtime, retries, usage, and memory kept separate from quality."""

    request_count: int
    total_latency_ms: float
    retry_count: int
    input_tokens: int | None
    output_tokens: int | None
    peak_memory_bytes: int | None


@dataclass(frozen=True, kw_only=True)
class SemanticEvaluationReport:
    """Common report schema for Qwen, Gemini, Florence-2, and future adapters."""

    context: SemanticEvaluationContext
    matching_policy: str
    samples: tuple[SemanticSampleReport, ...]
    failures: tuple[SemanticEvaluationFailure, ...]
    quality: SemanticQualityReport
    cost: SemanticCostReport


@dataclass(frozen=True, kw_only=True)
class SemanticBackendComparison:
    """Reports aligned over the same physical requests and evidence variants."""

    request_ids: tuple[str, ...]
    reports: tuple[SemanticEvaluationReport, ...]


def evaluate_semantic_interpretation(
    *,
    context: SemanticEvaluationContext,
    inputs: tuple[SemanticEvaluationInput, ...],
    failures: tuple[SemanticEvaluationFailure, ...] = (),
) -> SemanticEvaluationReport:
    """Evaluate canonical executions with an explicit matching convention."""
    if not inputs and not failures:
        raise SemanticEvaluationError("semantic evaluation requires inputs or failures")
    samples = tuple(_evaluate_sample(item) for item in inputs)
    claim_count = sum(
        sample.acceptable_claim_count + sample.unsupported_claim_count for sample in samples
    )
    acceptable = sum(sample.acceptable_claim_count for sample in samples)
    unsupported = sum(sample.unsupported_claim_count for sample in samples)
    ambiguity = tuple(
        sample.ambiguity_preserved for sample in samples if sample.ambiguity_preserved is not None
    )
    input_tokens = _optional_sum(sample.input_tokens for sample in samples)
    output_tokens = _optional_sum(sample.output_tokens for sample in samples)
    memory_values = [
        sample.peak_memory_bytes for sample in samples if sample.peak_memory_bytes is not None
    ]
    return SemanticEvaluationReport(
        context=context,
        matching_policy=MATCHING_POLICY,
        samples=samples,
        failures=failures,
        quality=SemanticQualityReport(
            claim_count=claim_count,
            acceptable_claim_rate=0.0 if claim_count == 0 else acceptable / claim_count,
            unsupported_claim_rate=0.0 if claim_count == 0 else unsupported / claim_count,
            ambiguity_preservation_rate=(
                None if not ambiguity else sum(ambiguity) / len(ambiguity)
            ),
            abstention_count=sum(sample.abstained for sample in samples),
            duplicate_claim_count=sum(sample.duplicate_claim_count for sample in samples),
        ),
        cost=SemanticCostReport(
            request_count=len(samples),
            total_latency_ms=sum(sample.latency_ms for sample in samples),
            retry_count=sum(sample.retries for sample in samples),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            peak_memory_bytes=max(memory_values) if memory_values else None,
        ),
    )


def compare_semantic_backends(
    reports: tuple[SemanticEvaluationReport, ...],
) -> SemanticBackendComparison:
    """Align backend reports only when they evaluate the same request/variant set."""
    if len(reports) < 2:
        raise SemanticEvaluationError("backend comparison requires at least two reports")
    expected = tuple(
        sorted((sample.request_id, sample.evidence_variant_id) for sample in reports[0].samples)
    )
    for report in reports[1:]:
        found = tuple(
            sorted((sample.request_id, sample.evidence_variant_id) for sample in report.samples)
        )
        if found != expected:
            raise SemanticEvaluationError(
                "backend reports must evaluate the same request and evidence variants"
            )
    return SemanticBackendComparison(
        request_ids=tuple(sorted({request_id for request_id, _variant in expected})),
        reports=reports,
    )


def _evaluate_sample(item: SemanticEvaluationInput) -> SemanticSampleReport:
    execution = item.execution
    request = execution.request
    claims = tuple(execution.parsed.claims)
    if execution.parsed.scene_context is not None:
        claims += tuple(execution.parsed.scene_context.claims)
    backend = item.backend or _backend_from_outputs(execution)
    acceptable = {_normalize(value) for value in item.annotation.acceptable_hypotheses}
    normalized_claims = tuple(_normalize(claim.hypothesis) for claim in claims)
    acceptable_count = sum(value in acceptable for value in normalized_claims)
    alternatives = sum(claim.role is HypothesisRole.ALTERNATIVE for claim in claims)
    ambiguity_preserved = alternatives > 0 if item.annotation.ambiguity_expected else None
    return SemanticSampleReport(
        request_id=str(request.request_id),
        source_observation_id=str(request.source_observation_id),
        perception_result_id=str(request.perception_result_id),
        region_id=None if request.region_id is None else str(request.region_id),
        evidence_variant_id=item.evidence_variant_id,
        backend=backend,
        prompt_template_id=execution.rendered_prompt.template_id,
        prompt_fingerprint=execution.rendered_prompt.fingerprint,
        acceptable_claim_count=acceptable_count,
        unsupported_claim_count=len(claims) - acceptable_count,
        alternative_claim_count=alternatives,
        ambiguity_preserved=ambiguity_preserved,
        abstained=execution.parsed.abstained,
        duplicate_claim_count=len(normalized_claims) - len(set(normalized_claims)),
        latency_ms=execution.diagnostics.latency_ms,
        retries=execution.diagnostics.retries,
        input_tokens=execution.diagnostics.input_tokens,
        output_tokens=execution.diagnostics.output_tokens,
        peak_memory_bytes=execution.diagnostics.peak_memory_bytes,
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


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _optional_sum(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None

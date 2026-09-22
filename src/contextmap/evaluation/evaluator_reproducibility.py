"""Reproducibility of evaluators: identical inputs must give equivalent reports.

An evaluator that changes its answer between two runs on the same artifacts and
the same reference set makes every comparison meaningless, so this module runs
an evaluator repeatedly and compares the reports it produced. Two reports are
*equivalent* when everything except resource measurements matches: the
reproducibility metadata, every quality metric, the metric statuses and sample
counts, and the stage report.

Resource measurements (wall time, memory, storage) vary between executions by
nature; their **values** are excluded and the check says so explicitly. Any
other nondeterminism that is mathematically unavoidable (an unordered floating
point reduction, a randomized tie-break) has to be **declared** with a reason
(:class:`NondeterministicField`); an undeclared difference is a failure. See
``src/contextmap/evaluation/docs/annotation-qa.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from contextmap.evaluation._validation import require_text
from contextmap.evaluation.report_schema import EvaluationReport, encode_evaluation_report

_QUALITY_METRIC = "quality_metric:"
_STAGE_REPORT = "stage_report:"
_MASKED = "<declared nondeterministic>"
_EXCLUDED = "<resource measurement>"

PERFORMANCE_EXCLUSION_REASON = (
    "resource measurements (time, memory, storage) vary between executions by nature; their "
    "values are excluded, while their presence, status and sample count are still compared"
)


class EvaluatorNondeterminismError(ValueError):
    """Raised when an evaluator gave different reports for identical inputs."""


@dataclass(frozen=True, kw_only=True)
class NondeterministicField:
    """A field that is mathematically unavoidable to vary, with the reason.

    Attributes:
        location: ``quality_metric:<metric name>`` (the value of that metric) or
            ``stage_report:<dotted.key.path>`` (a key of the stage report).
        reason: Why the field cannot be made deterministic.
    """

    location: str
    reason: str

    def __post_init__(self) -> None:
        """Require a known location form and a documented reason."""
        require_text("nondeterminism reason", self.reason)
        for prefix in (_QUALITY_METRIC, _STAGE_REPORT):
            if self.location.startswith(prefix) and self.location[len(prefix) :].strip():
                return
        raise ValueError(
            f"nondeterministic location {self.location!r} must be "
            f"'{_QUALITY_METRIC}<metric>' or '{_STAGE_REPORT}<dotted.path>'"
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"location": self.location, "reason": self.reason}


@dataclass(frozen=True, kw_only=True)
class ReproducibilityCheck:
    """The outcome of running an evaluator repeatedly on identical inputs.

    Attributes:
        repetitions: How many times the evaluator ran.
        differences: Locations where reports differ, empty when reproducible.
        declared: The declared unavoidable nondeterminism the comparison excused.
        performance_values_excluded: Whether resource values were left out of the comparison.
        performance_exclusion_reason: Why they were.
    """

    repetitions: int
    differences: tuple[str, ...]
    declared: tuple[NondeterministicField, ...]
    performance_values_excluded: bool = True
    performance_exclusion_reason: str = PERFORMANCE_EXCLUSION_REASON

    @property
    def reproducible(self) -> bool:
        """Return whether every run gave an equivalent report."""
        return not self.differences

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "reproducible": self.reproducible,
            "repetitions": self.repetitions,
            "differences": list(self.differences),
            "declared_nondeterminism": [item.to_record() for item in self.declared],
            "performance_values_excluded": self.performance_values_excluded,
            "performance_exclusion_reason": self.performance_exclusion_reason,
        }


def _normalize(
    report: EvaluationReport, declared: Sequence[NondeterministicField]
) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(json.dumps(encode_evaluation_report(report)))
    for result in document["performance_metrics"]:
        if result["status"] == "value":
            result["value"] = _EXCLUDED
    for field in declared:
        if field.location.startswith(_QUALITY_METRIC):
            name = field.location[len(_QUALITY_METRIC) :]
            for result in document["quality_metrics"]:
                if result["metric"] == name and result["status"] == "value":
                    result["value"] = _MASKED
        else:
            keys = field.location[len(_STAGE_REPORT) :].split(".")
            node: Any = document["stage_report"]
            for key in keys[:-1]:
                node = node.get(key) if isinstance(node, dict) else None
            if isinstance(node, dict) and keys[-1] in node:
                node[keys[-1]] = _MASKED
    return document


def _differences(first: Any, second: Any, path: str) -> Iterator[str]:
    if isinstance(first, dict) and isinstance(second, dict):
        for key in sorted(set(first) | set(second)):
            child = f"{path}.{key}" if path else str(key)
            if key not in first or key not in second:
                yield child
            else:
                yield from _differences(first[key], second[key], child)
    elif isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            yield f"{path}.length"
        for index, (left, right) in enumerate(zip(first, second, strict=False)):
            yield from _differences(left, right, f"{path}[{index}]")
    elif first != second:
        yield path


def compare_evaluation_reports(
    first: EvaluationReport,
    second: EvaluationReport,
    *,
    nondeterministic: Sequence[NondeterministicField] = (),
) -> tuple[str, ...]:
    """Return where two reports differ, ignoring resource values and declared fields.

    An empty result means the reports are equivalent.
    """
    left = _normalize(first, nondeterministic)
    right = _normalize(second, nondeterministic)
    return tuple(_differences(left, right, ""))


def check_evaluator_reproducibility(
    evaluate: Callable[[], EvaluationReport],
    *,
    repetitions: int = 2,
    nondeterministic: Sequence[NondeterministicField] = (),
) -> ReproducibilityCheck:
    """Run an evaluator repeatedly on identical inputs and compare the reports.

    ``evaluate`` must close over identical artifacts and an identical reference
    set: it is the evaluator, not its inputs, that is under test.

    Raises:
        ValueError: If fewer than two repetitions are requested.
    """
    if repetitions < 2:
        raise ValueError("repetitions must be at least 2 to compare a run with another")
    reports = [evaluate() for _ in range(repetitions)]
    differences: set[str] = set()
    for report in reports[1:]:
        differences.update(
            compare_evaluation_reports(reports[0], report, nondeterministic=nondeterministic)
        )
    return ReproducibilityCheck(
        repetitions=repetitions,
        differences=tuple(sorted(differences)),
        declared=tuple(nondeterministic),
    )


def require_reproducible(check: ReproducibilityCheck) -> None:
    """Refuse an evaluator that was not reproducible.

    Raises:
        EvaluatorNondeterminismError: Naming where the reports differed.
    """
    if not check.reproducible:
        raise EvaluatorNondeterminismError(
            f"the evaluator gave different reports for identical inputs at: "
            f"{list(check.differences)}; declare the nondeterminism with a reason if it is "
            "mathematically unavoidable"
        )

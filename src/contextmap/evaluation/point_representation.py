"""Deterministic evaluation of Point Representation arms.

Point Representation is optional and must justify its cost. This harness runs
the same geometry and the same centers through each *arm* of an ablation and
reports what each one produced, without turning it into a verdict:

    off                          no point representation
    deterministic descriptor     the geometry-only control
    learned 3D (for example PTv3)
    pretrained / distilled       a separately identified backend, optional

Representation-level evidence (coverage, repeatability, norm distribution,
sensitivity to controlled geometry variations), compute cost and downstream
results are separate sections of every report. There is no single score and no
winner: quality and cost are never mixed, a learned backend is never picked from
one metric, and arms with different representation spaces are never compared
numerically against each other. Sensitivity compares an arm with itself, in one
space, under a controlled variation of the geometry.

Downstream effects (fusion consistency, entity resolution) belong to later
capabilities: the harness only carries the metrics a caller measured for each
arm and refuses to compare arms whose held-fixed downstream configuration
differs. Repeated encodings of identical geometry are never treated as
independent physical observations.

The harness imports no NumPy, model library or GPU runtime; backends are passed
in already constructed. Composition remains the caller's responsibility until
the global runtime is materialized.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMap,
    GeometryPoint,
    GeometryReference,
    GeometrySource,
    MapId,
)
from contextmap.point_representation import (
    EncodedRepresentation,
    EncoderIdentity,
    FailedSupport,
    PointEncoder,
    PointRepresentationRunId,
    RepresentationMetrics,
    RepresentationService,
    SupportPolicy,
    center_selection_id,
    encode_support_policy,
    representation_space_fingerprint,
)
from contextmap.shared import Vector3

_PTV3_BACKEND_ID = "ptv3"


class RepresentationEvaluationError(ValueError):
    """Raised when an evaluation or comparison would not be a controlled one."""


class RepresentationArmRole(Enum):
    """Which arm of the ablation an encoder plays.

    Attributes:
        OFF: No point representation; the baseline for downstream comparison.
        DETERMINISTIC_DESCRIPTOR: The geometry-only control.
        LEARNED_3D: A generic learned 3D encoder, for example PTv3 inference.
        PRETRAINED_DISTILLED: A separately identified pretrained/distilled
            encoder (Sonata/Vernata-style); optional and never implied by
            running PTv3 alone.
    """

    OFF = "off"
    DETERMINISTIC_DESCRIPTOR = "deterministic_descriptor"
    LEARNED_3D = "learned_3d"
    PRETRAINED_DISTILLED = "pretrained_distilled"


@dataclass(frozen=True, kw_only=True)
class RepresentationArm:
    """One arm of an ablation.

    Attributes:
        arm_id: Identity of the arm within the comparison.
        role: The arm's role.
        encoder: The already constructed encoder; ``None`` exactly for ``OFF``.
    """

    arm_id: str
    role: RepresentationArmRole
    encoder: PointEncoder | None

    def __post_init__(self) -> None:
        """Validate the arm.

        Raises:
            ValueError: If ``arm_id`` is empty, an ``OFF`` arm has an encoder, any
                other arm has none, or PTv3 is labeled pretrained/distilled.
        """
        if not self.arm_id:
            raise ValueError("an arm needs a non-empty arm_id")
        if self.role is RepresentationArmRole.OFF:
            if self.encoder is not None:
                raise ValueError("an 'off' arm has no encoder")
            return
        if self.encoder is None:
            raise ValueError(f"a {self.role.value} arm needs an encoder")
        if (
            self.role is RepresentationArmRole.PRETRAINED_DISTILLED
            and self.encoder.encoder_identity().backend_id == _PTV3_BACKEND_ID
        ):
            raise ValueError(
                "PTv3 alone is not a pretrained or distilled (Sonata/Vernata-style) "
                "representation; label it learned_3d"
            )


@dataclass(frozen=True, kw_only=True)
class RepresentationEvaluationContext:
    """Identities held fixed across the arms of one evaluation.

    Attributes:
        evaluation_id: Identity of this evaluation execution.
        code_version: Code revision that ran it.
        downstream_configuration_fingerprint: Identity of the downstream
            configuration (perception, association, geometry, claims) held fixed.
    """

    evaluation_id: str
    code_version: str
    downstream_configuration_fingerprint: str

    def __post_init__(self) -> None:
        """Require every identity.

        Raises:
            ValueError: If an identity is empty.
        """
        for name in ("evaluation_id", "code_version", "downstream_configuration_fingerprint"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, kw_only=True)
class RepresentationArmIdentity:
    """What identifies an arm exactly, so it can be reproduced.

    Attributes:
        arm_id: Identity of the arm.
        role: The arm's role.
        encoder: Backend, version, configuration fingerprint and checkpoint
            hash; ``None`` for ``OFF``.
        representation_space_id: Fingerprint of the arm's space; ``None`` for ``OFF``.
        support_policy: The support policy the arm ran under; ``None`` for ``OFF``.
    """

    arm_id: str
    role: RepresentationArmRole
    encoder: EncoderIdentity | None
    representation_space_id: str | None
    support_policy: SupportPolicy | None


@dataclass(frozen=True, kw_only=True)
class RepresentationDistribution:
    """Count, minimum, median, 95th percentile (nearest rank) and maximum of a measurement."""

    count: int
    minimum: float | None
    median: float | None
    p95: float | None
    maximum: float | None


@dataclass(frozen=True, kw_only=True)
class RepresentationCoverageReport:
    """What the arm produced for the requested centers.

    Attributes:
        requested: Centers asked for.
        represented: Representations produced, partial ones included.
        partial: Representations with undefined components.
        failed_by_reason: Failed supports per reason; never counted as representations.
        undefined_component_fraction: Fraction of all components of the produced
            representations that are undefined; ``None`` when none were produced.
    """

    requested: int
    represented: int
    partial: int
    failed_by_reason: Mapping[str, int]
    undefined_component_fraction: float | None


@dataclass(frozen=True, kw_only=True)
class RepresentationRepeatabilityReport:
    """Whether running the same input twice gives the same representations.

    Attributes:
        compared: Centers represented in both runs.
        max_abs_difference: Largest absolute component difference between runs.
        tolerance: The tolerance ``repeatable`` was judged against.
        identical_coverage: Whether the same centers were represented, failed
            and partial in both runs.
        repeatable: ``max_abs_difference <= tolerance`` with identical coverage;
            ``None`` when nothing could be compared.
    """

    compared: int
    max_abs_difference: float
    tolerance: float
    identical_coverage: bool
    repeatable: bool | None


@dataclass(frozen=True, kw_only=True)
class RepresentationSensitivityReport:
    """How much an arm's representations move under one controlled geometry variation.

    Both sides are in the arm's own space, so the comparison is between
    compatible vectors; only components defined on both sides are compared.

    Attributes:
        variation: Name of the variation, with its parameters.
        description: What the variation does.
        compared: Centers represented and comparable on both sides.
        unmatched: Requested centers that could not be compared.
        relative_l2_change: ``‖v - v'‖ / ‖v‖`` per center.
        cosine_distance: ``1 - cos(v, v')`` per center.
        component_mean_absolute_change: Mean absolute change per named
            component, only for an interpretable descriptor; ``None`` for an
            opaque (learned) space.
    """

    variation: str
    description: str
    compared: int
    unmatched: int
    relative_l2_change: RepresentationDistribution
    cosine_distance: RepresentationDistribution
    component_mean_absolute_change: Mapping[str, float] | None


@dataclass(frozen=True, kw_only=True)
class RepresentationCostReport:
    """What producing the representations cost, apart from every quality measure.

    Attributes:
        support_extraction_seconds: Wall time extracting supports.
        encoding_seconds: Wall time inside the encoder.
        seconds_per_representation: Total wall time per produced representation;
            ``None`` when none were produced.
        payload_bytes: Storage the produced vectors occupy.
        backend_diagnostics: Backend-specific figures supplied by the caller,
            for example peak device memory.
    """

    support_extraction_seconds: float
    encoding_seconds: float
    seconds_per_representation: float | None
    payload_bytes: int
    backend_diagnostics: Mapping[str, str | int | float | None]


@dataclass(frozen=True, kw_only=True)
class RepresentationArmReport:
    """Everything one arm produced, in sections that are never mixed.

    Attributes:
        identity: What identifies the arm exactly.
        evaluation_id: Identity of the evaluation execution.
        code_version: Code revision that ran it.
        geometric_map_id: The map that was evaluated.
        center_selection_id: Identity of the fixed set of centers.
        downstream_configuration_fingerprint: The downstream configuration held fixed.
        coverage: What was produced; ``None`` for ``OFF``.
        repeatability: Repeatability; ``None`` for ``OFF``.
        distribution: Distribution of the vector L2 norms over the defined
            components; ``None`` when no vector was produced.
        sensitivity: One report per controlled variation.
        cost: Compute and storage cost; ``None`` for ``OFF``.
        downstream: Downstream metrics the caller measured for this arm,
            carried as given and never combined.
    """

    identity: RepresentationArmIdentity
    evaluation_id: str
    code_version: str
    geometric_map_id: MapId
    center_selection_id: str
    downstream_configuration_fingerprint: str
    coverage: RepresentationCoverageReport | None
    repeatability: RepresentationRepeatabilityReport | None
    distribution: RepresentationDistribution | None
    sensitivity: tuple[RepresentationSensitivityReport, ...]
    cost: RepresentationCostReport | None
    downstream: Mapping[str, float]


@dataclass(frozen=True, kw_only=True)
class RepresentationAblationReport:
    """Arms evaluated under the same controlled variables, side by side.

    There is deliberately no score, rank or winner: the reader compares the
    separate sections and decides whether to keep the capability optional,
    adopt one backend or defer it.

    Attributes:
        geometric_map_id: The map every arm was evaluated on.
        center_selection_id: The fixed set of centers.
        downstream_configuration_fingerprint: The downstream configuration held fixed.
        arms: One report per arm, in the order given.
    """

    geometric_map_id: MapId
    center_selection_id: str
    downstream_configuration_fingerprint: str
    arms: tuple[RepresentationArmReport, ...]


@dataclass(frozen=True, kw_only=True)
class GeometryVariation:
    """A named, controlled variation of the geometry an arm is evaluated on.

    Attributes:
        name: Name with its parameters, e.g. ``"noise(sigma_m=0.005, seed=1)"``.
        description: What the variation does.
        apply: Builds the varied geometry from a source and the centers; the
            source is never modified.
    """

    name: str
    description: str
    apply: Callable[[GeometrySource, Collection[GeometryReference]], GeometrySource]

    def __post_init__(self) -> None:
        """Require a name and a description.

        Raises:
            ValueError: If either is empty.
        """
        if not self.name or not self.description:
            raise ValueError("a geometry variation needs a name and a description")


class _VariedGeometry:
    """An in-memory :class:`GeometrySource` over a varied copy of another map's geometry."""

    def __init__(self, base: GeometricMap, points: Sequence[GeometryPoint]) -> None:
        if not points:
            raise RepresentationEvaluationError("the variation left no geometry")
        self._points = tuple(points)
        self._by_id = {point.geometry_id: point for point in self._points}
        # O índice espacial do mapa original não descreve esta cópia.
        self._map = replace(
            base,
            point_count=len(self._points),
            bounds=Bounds3D.enclosing(
                (point.coordinates_m for point in self._points), frame_id=base.frame_id
            ),
            spatial_index=None,
        )

    @property
    def geometric_map(self) -> GeometricMap:
        return self._map

    def get(self, reference: GeometryReference) -> GeometryPoint:
        if reference.map_id != self._map.map_id or reference.geometry_id not in self._by_id:
            raise KeyError(reference)
        return self._by_id[reference.geometry_id]

    def iter_geometry(self) -> Iterator[GeometryPoint]:
        return iter(self._points)

    def query_bounds(self, bounds: Bounds3D) -> Iterator[GeometryPoint]:
        if bounds.frame_id != self._map.frame_id:
            raise ValueError("query bounds are expressed in another frame than the map")
        return (
            point
            for point in self._points
            if bounds.contains(point.coordinates_m, frame_id=point.map_frame)
        )


def translation_variation(offset_m: Vector3) -> GeometryVariation:
    """A rigid translation of every map coordinate, in meters.

    Args:
        offset_m: The ``(x, y, z)`` offset.

    Raises:
        ValueError: If a component of the offset is not finite.
    """
    if not all(math.isfinite(value) for value in offset_m):
        raise ValueError(f"the variation offset must be finite, got {offset_m!r}")
    dx, dy, dz = offset_m

    def apply(source: GeometrySource, centers: Collection[GeometryReference]) -> GeometrySource:
        return _VariedGeometry(
            source.geometric_map,
            [
                replace(
                    point,
                    coordinates_m=(
                        point.coordinates_m[0] + dx,
                        point.coordinates_m[1] + dy,
                        point.coordinates_m[2] + dz,
                    ),
                )
                for point in source.iter_geometry()
            ],
        )

    return GeometryVariation(
        name=f"translation(offset_m={offset_m!r})",
        description="Translate every map coordinate rigidly; a spatially consistent "
        "representation should not change.",
        apply=apply,
    )


def rotation_about_z_variation(degrees: float) -> GeometryVariation:
    """A rigid rotation of the map about its z axis through the origin.

    Args:
        degrees: The rotation angle.

    Raises:
        ValueError: If the angle is not finite.
    """
    if not math.isfinite(degrees):
        raise ValueError(f"the variation degrees must be finite, got {degrees!r}")
    cosine, sine = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))

    def apply(source: GeometrySource, centers: Collection[GeometryReference]) -> GeometrySource:
        return _VariedGeometry(
            source.geometric_map,
            [
                replace(
                    point,
                    coordinates_m=(
                        point.coordinates_m[0] * cosine - point.coordinates_m[1] * sine,
                        point.coordinates_m[0] * sine + point.coordinates_m[1] * cosine,
                        point.coordinates_m[2],
                    ),
                )
                for point in source.iter_geometry()
            ],
        )

    return GeometryVariation(
        name=f"rotation_about_z(degrees={degrees!r})",
        description="Rotate the map about its z axis; the shape of a local support is "
        "unchanged, so a rotation-invariant representation should not change.",
        apply=apply,
    )


def noise_variation(*, sigma_m: float, seed: int) -> GeometryVariation:
    """Independent Gaussian noise on every coordinate, reproducible from a seed.

    Args:
        sigma_m: Standard deviation in meters; positive and finite.
        seed: Seed of the deterministic noise.

    Raises:
        ValueError: If ``sigma_m`` is not positive and finite.
    """
    if not math.isfinite(sigma_m) or sigma_m <= 0:
        raise ValueError(f"the variation sigma_m must be positive and finite, got {sigma_m!r}")

    def apply(source: GeometrySource, centers: Collection[GeometryReference]) -> GeometrySource:
        rng = random.Random(seed)
        return _VariedGeometry(
            source.geometric_map,
            [
                replace(
                    point,
                    coordinates_m=(
                        point.coordinates_m[0] + rng.gauss(0.0, sigma_m),
                        point.coordinates_m[1] + rng.gauss(0.0, sigma_m),
                        point.coordinates_m[2] + rng.gauss(0.0, sigma_m),
                    ),
                )
                for point in source.iter_geometry()
            ],
        )

    return GeometryVariation(
        name=f"noise(sigma_m={sigma_m!r}, seed={seed})",
        description="Add independent Gaussian noise to every coordinate: sensitivity to "
        "sensor noise and to the support boundary it moves.",
        apply=apply,
    )


def subsample_variation(keep_fraction: float, *, seed: int) -> GeometryVariation:
    """Keep a random fraction of the geometry, always keeping the centers.

    Args:
        keep_fraction: Fraction of elements kept, in ``(0, 1]``.
        seed: Seed of the deterministic selection.

    Raises:
        ValueError: If ``keep_fraction`` is outside ``(0, 1]``.
    """
    if not 0 < keep_fraction <= 1:
        raise ValueError(f"the variation keep_fraction must be in (0, 1], got {keep_fraction!r}")

    def apply(source: GeometrySource, centers: Collection[GeometryReference]) -> GeometrySource:
        rng = random.Random(seed)
        protected = set(centers)
        kept: list[GeometryPoint] = []
        for point in source.iter_geometry():
            # O sorteio é feito para todo ponto: o resultado não depende de quais são protegidos.
            draw = rng.random()
            if point.reference in protected or draw < keep_fraction:
                kept.append(point)
        return _VariedGeometry(source.geometric_map, kept)

    return GeometryVariation(
        name=f"subsample(keep_fraction={keep_fraction!r}, seed={seed})",
        description="Keep a random fraction of the geometry, centers included: sensitivity "
        "to point density.",
        apply=apply,
    )


def evaluate_representation_arm(
    arm: RepresentationArm,
    *,
    source: GeometrySource,
    centers: Sequence[GeometryReference],
    context: RepresentationEvaluationContext,
    variations: Sequence[GeometryVariation] = (),
    repeatability_tolerance: float = 0.0,
    downstream: Mapping[str, float] | None = None,
    backend_diagnostics: Mapping[str, str | int | float | None] | None = None,
) -> RepresentationArmReport:
    """Evaluate one arm on a fixed geometry and a fixed set of centers.

    Args:
        arm: The arm to evaluate.
        source: The persistent geometry, never modified.
        centers: The fixed centers every arm of the comparison is evaluated on.
        context: The identities held fixed across arms.
        variations: Controlled geometry variations to measure sensitivity under.
        repeatability_tolerance: Largest absolute component difference between
            two identical runs still judged repeatable; ``0.0`` demands identical
            output, which a deterministic encoder should meet.
        downstream: Downstream metrics measured for this arm, carried as given.
        backend_diagnostics: Backend-specific cost figures, such as peak device memory.

    Returns:
        The arm's report; for ``OFF`` only the identity, the fixed variables and
        the downstream metrics.

    Raises:
        RepresentationEvaluationError: If there are no centers, the tolerance or
            a downstream metric is not finite, or a variation leaves no geometry.
    """
    requested = tuple(centers)
    if not requested:
        raise RepresentationEvaluationError("an evaluation needs at least one center")
    if not math.isfinite(repeatability_tolerance) or repeatability_tolerance < 0:
        raise RepresentationEvaluationError("repeatability_tolerance must be finite and >= 0")
    downstream_metrics = _finite_metrics(downstream)
    map_id = source.geometric_map.map_id
    fixed: dict[str, Any] = {
        "evaluation_id": context.evaluation_id,
        "code_version": context.code_version,
        "geometric_map_id": map_id,
        "center_selection_id": center_selection_id(map_id=map_id, centers=requested),
        "downstream_configuration_fingerprint": context.downstream_configuration_fingerprint,
        "downstream": downstream_metrics,
    }
    encoder = arm.encoder
    if encoder is None:
        return RepresentationArmReport(
            identity=RepresentationArmIdentity(
                arm_id=arm.arm_id,
                role=arm.role,
                encoder=None,
                representation_space_id=None,
                support_policy=None,
            ),
            coverage=None,
            repeatability=None,
            distribution=None,
            sensitivity=(),
            cost=None,
            **fixed,
        )

    space = encoder.representation_space()
    baseline, baseline_metrics = _run(source, encoder, requested, context)
    repeat, _ = _run(source, encoder, requested, context)
    represented = _represented(baseline)
    return RepresentationArmReport(
        identity=RepresentationArmIdentity(
            arm_id=arm.arm_id,
            role=arm.role,
            encoder=encoder.encoder_identity(),
            representation_space_id=representation_space_fingerprint(space),
            support_policy=space.support_semantics,
        ),
        coverage=RepresentationCoverageReport(
            requested=baseline_metrics.requested,
            represented=baseline_metrics.represented,
            partial=baseline_metrics.partial,
            failed_by_reason={
                reason.value: count for reason, count in baseline_metrics.failed_by_reason.items()
            },
            undefined_component_fraction=_undefined_fraction(represented, space.dimension),
        ),
        repeatability=_repeatability(baseline, repeat, repeatability_tolerance),
        distribution=_norm_distribution(represented),
        sensitivity=tuple(
            _sensitivity(variation, source, encoder, requested, context, represented)
            for variation in variations
        ),
        cost=RepresentationCostReport(
            support_extraction_seconds=baseline_metrics.support_extraction_seconds,
            encoding_seconds=baseline_metrics.encoding_seconds,
            seconds_per_representation=(
                (baseline_metrics.support_extraction_seconds + baseline_metrics.encoding_seconds)
                / baseline_metrics.represented
                if baseline_metrics.represented
                else None
            ),
            payload_bytes=baseline_metrics.represented * space.vector_bytes,
            backend_diagnostics=dict(backend_diagnostics or {}),
        ),
        **fixed,
    )


def compare_representation_arms(
    reports: Sequence[RepresentationArmReport],
) -> RepresentationAblationReport:
    """Put arm reports side by side after checking they were evaluated like with like.

    Args:
        reports: One report per arm.

    Returns:
        The ablation report; it holds the reports as given and computes nothing
        across them.

    Raises:
        RepresentationEvaluationError: If there is no report, an arm repeats, or the
            geometric map, the center selection or the held-fixed downstream
            configuration differs between arms.
    """
    if not reports:
        raise RepresentationEvaluationError("a comparison needs at least one arm report")
    first = reports[0]
    seen: set[str] = set()
    for report in reports:
        arm_id = report.identity.arm_id
        if arm_id in seen:
            raise RepresentationEvaluationError(f"duplicate arm_id in the comparison: {arm_id!r}")
        seen.add(arm_id)
        for label, expected, found in (
            ("geometric map", first.geometric_map_id, report.geometric_map_id),
            ("center selection", first.center_selection_id, report.center_selection_id),
            (
                "downstream configuration",
                first.downstream_configuration_fingerprint,
                report.downstream_configuration_fingerprint,
            ),
        ):
            if found != expected:
                raise RepresentationEvaluationError(
                    f"arm {arm_id!r} used another {label} than arm {first.identity.arm_id!r}: "
                    f"{found} != {expected}; a comparison must hold it fixed"
                )
    return RepresentationAblationReport(
        geometric_map_id=first.geometric_map_id,
        center_selection_id=first.center_selection_id,
        downstream_configuration_fingerprint=first.downstream_configuration_fingerprint,
        arms=tuple(reports),
    )


def encode_representation_arm_report(report: RepresentationArmReport) -> dict[str, Any]:
    """Encode an arm report as JSON-compatible data with every identity needed to reproduce it."""
    identity = report.identity
    return {
        "identity": {
            "arm_id": identity.arm_id,
            "role": identity.role.value,
            "encoder": None
            if identity.encoder is None
            else {
                "backend_id": identity.encoder.backend_id,
                "backend_version": identity.encoder.backend_version,
                "configuration_fingerprint": identity.encoder.configuration_fingerprint,
                "checkpoint_hash": identity.encoder.checkpoint_hash,
            },
            "representation_space_id": identity.representation_space_id,
            "support_policy": None
            if identity.support_policy is None
            else encode_support_policy(identity.support_policy),
        },
        "evaluation_id": report.evaluation_id,
        "code_version": report.code_version,
        "geometric_map_id": str(report.geometric_map_id),
        "center_selection_id": report.center_selection_id,
        "downstream_configuration_fingerprint": report.downstream_configuration_fingerprint,
        "representation": None
        if report.coverage is None
        else {
            "coverage": _encode(report.coverage),
            "repeatability": _encode(report.repeatability),
            "distribution": None if report.distribution is None else _encode(report.distribution),
            "sensitivity": [_encode(study) for study in report.sensitivity],
        },
        "cost": None if report.cost is None else _encode(report.cost),
        "downstream": dict(report.downstream),
    }


def encode_representation_ablation_report(report: RepresentationAblationReport) -> dict[str, Any]:
    """Encode an ablation report as JSON-compatible data; arm sections stay separate."""
    return {
        "geometric_map_id": str(report.geometric_map_id),
        "center_selection_id": report.center_selection_id,
        "downstream_configuration_fingerprint": report.downstream_configuration_fingerprint,
        "arms": [encode_representation_arm_report(arm) for arm in report.arms],
    }


def _run(
    source: GeometrySource,
    encoder: PointEncoder,
    centers: Sequence[GeometryReference],
    context: RepresentationEvaluationContext,
) -> tuple[list[EncodedRepresentation | FailedSupport], RepresentationMetrics]:
    service = RepresentationService(
        source,
        encoder,
        run_id=PointRepresentationRunId(context.evaluation_id),
        code_version=context.code_version,
    )
    outcomes = list(service.represent(centers))
    return outcomes, service.metrics


def _represented(
    outcomes: Sequence[EncodedRepresentation | FailedSupport],
) -> dict[GeometryReference, EncodedRepresentation]:
    return {
        outcome.representation.geometry_reference: outcome
        for outcome in outcomes
        if isinstance(outcome, EncodedRepresentation)
    }


def _undefined_fraction(
    represented: Mapping[GeometryReference, EncodedRepresentation], dimension: int
) -> float | None:
    if not represented:
        return None
    undefined = sum(len(item.representation.undefined_components) for item in represented.values())
    return undefined / (len(represented) * dimension)


def _repeatability(
    first: Sequence[EncodedRepresentation | FailedSupport],
    second: Sequence[EncodedRepresentation | FailedSupport],
    tolerance: float,
) -> RepresentationRepeatabilityReport:
    left, right = _represented(first), _represented(second)
    common = [center for center in left if center in right]
    largest = max(
        (
            abs(a - b)
            for center in common
            for a, b in zip(left[center].values, right[center].values, strict=True)
        ),
        default=0.0,
    )
    identical = left.keys() == right.keys() and all(
        left[center].representation.undefined_components
        == right[center].representation.undefined_components
        for center in common
    )
    return RepresentationRepeatabilityReport(
        compared=len(common),
        max_abs_difference=largest,
        tolerance=tolerance,
        identical_coverage=identical,
        repeatable=(largest <= tolerance and identical) if common else None,
    )


def _norm_distribution(
    represented: Mapping[GeometryReference, EncodedRepresentation],
) -> RepresentationDistribution | None:
    norms = [_defined_norm(item) for item in represented.values()]
    present = [norm for norm in norms if norm is not None]
    return _distribution(present) if present else None


def _defined_norm(item: EncodedRepresentation) -> float | None:
    undefined = set(item.representation.undefined_components)
    defined = [value for index, value in enumerate(item.values) if index not in undefined]
    return math.hypot(*defined) if defined else None


def _sensitivity(
    variation: GeometryVariation,
    source: GeometrySource,
    encoder: PointEncoder,
    centers: Sequence[GeometryReference],
    context: RepresentationEvaluationContext,
    baseline: Mapping[GeometryReference, EncodedRepresentation],
) -> RepresentationSensitivityReport:
    varied_outcomes, _ = _run(variation.apply(source, centers), encoder, centers, context)
    varied = _represented(varied_outcomes)
    names = encoder.representation_space().feature_names
    relative: list[float] = []
    cosine: list[float] = []
    component_changes: dict[int, list[float]] = {}
    for center, before in baseline.items():
        after = varied.get(center)
        if after is None:
            continue
        undefined = set(before.representation.undefined_components) | set(
            after.representation.undefined_components
        )
        indexes = [index for index in range(len(before.values)) if index not in undefined]
        a = [before.values[index] for index in indexes]
        b = [after.values[index] for index in indexes]
        norm_a, norm_b = math.hypot(*a), math.hypot(*b)
        if not indexes or norm_a == 0.0 or norm_b == 0.0:
            continue
        relative.append(math.hypot(*(x - y for x, y in zip(a, b, strict=True))) / norm_a)
        # A clamp evita que o arredondamento produza uma distância cosseno ligeiramente negativa.
        cosine.append(
            max(0.0, 1.0 - sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b))
        )
        for index, x, y in zip(indexes, a, b, strict=True):
            component_changes.setdefault(index, []).append(abs(x - y))
    return RepresentationSensitivityReport(
        variation=variation.name,
        description=variation.description,
        compared=len(relative),
        unmatched=len(centers) - len(relative),
        relative_l2_change=_distribution(relative),
        cosine_distance=_distribution(cosine),
        component_mean_absolute_change=(
            {
                names[index]: statistics.fmean(changes)
                for index, changes in sorted(component_changes.items())
            }
            if names
            else None
        ),
    )


def _distribution(values: Sequence[float]) -> RepresentationDistribution:
    if not values:
        return RepresentationDistribution(
            count=0, minimum=None, median=None, p95=None, maximum=None
        )
    ordered = sorted(values)
    return RepresentationDistribution(
        count=len(ordered),
        minimum=ordered[0],
        median=statistics.median(ordered),
        p95=ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        maximum=ordered[-1],
    )


def _finite_metrics(metrics: Mapping[str, float] | None) -> dict[str, float]:
    checked: dict[str, float] = {}
    for name, value in (metrics or {}).items():
        if not name or not math.isfinite(value):
            raise RepresentationEvaluationError(
                f"downstream metric {name!r} must have a name and a finite value, got {value!r}"
            )
        checked[name] = float(value)
    return checked


def _encode(report: Any) -> dict[str, Any]:
    """Encode a flat report dataclass, turning nested reports and mappings into plain data."""
    record: dict[str, Any] = {}
    for name, value in vars(report).items():
        if hasattr(value, "__dataclass_fields__"):
            record[name] = _encode(value)
        elif isinstance(value, Mapping):
            record[name] = dict(value)
        else:
            record[name] = value
    return record

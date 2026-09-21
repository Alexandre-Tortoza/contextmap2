"""Geometric comparison of two entities: one evidence channel, never the sole authority.

The geometry channel compares the persistent 3D support of two entities in their common map
frame and reports *documented signals*, each with its own units and never folded into one number:

* ``centroid_distance_m`` and ``bounds_gap_m`` -- where the supports are;
* ``bounds_iou`` and the two ``bounds_overlap_fraction`` values -- how much volume the boxes share;
* ``support_jaccard`` -- how many geometry elements the supports share (the same authoritative
  points, not merely nearby ones);
* ``extent_ratio`` -- whether the boxes are of comparable size;
* ``support_distance`` -- nearest-point statistics between the supports (optional, needs the map's
  ``GeometrySource``);
* ``orientation_angle_rad`` -- the angle between the principal axes (when both entities have one).

The measurement is then read through the explicit, versioned rules of
``entity-geometry-comparison-v1``, each recorded as a :class:`~contextmap.entity_resolution.Finding`
with the metric, the observed value and the threshold. A high geometric score is only geometry: it
never says the entities are the same object.

Scale and robustness are handled explicitly rather than hidden in one threshold:

* thresholds on ratios (IoU, containment, Jaccard, extent ratio) are scale free; the two that are
  metric distances (``min_conflict_gap_m``, the support distance) are declared by the profile and
  have no defaults, so nothing is hardcoded to one dataset;
* a **small object inside a large background entity** (a floor, a wall) is fully contained in it but
  is not evidence of identity, so containment supports a match only when the boxes are also of
  comparable size, and a size mismatch that containment explains (partial visibility, background)
  is neutral rather than conflicting;
* **flat bounds** have no volumetric overlap: the measure is absent, not zero;
* **sparse and disconnected** supports are flagged as caveats of the measurement instead of being
  read as if they were reliable;
* coordinates of different geometric maps or frames are never compared: a failed gate makes the
  channel unavailable.

NumPy is imported only when the optional nearest-point statistics are requested, so reading and
writing resolutions never needs it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from contextmap.entity_resolution._checks import require_finite
from contextmap.entity_resolution._spatial import bounds_gap as _bounds_gap
from contextmap.entity_resolution._spatial import (
    box_volume,
    extent_ratio,
    intersection_volume,
)
from contextmap.entity_resolution.channels import (
    EvidenceStatus,
    Finding,
    GeometryEvidence,
    GeometryMeasurement,
    SupportDistance,
    Unavailability,
    UnavailableReason,
)
from contextmap.entity_resolution.evidence import evaluate_comparison_gates
from contextmap.entity_resolution.models import PolicyRef, reference_order
from contextmap.geometric_mapping import GeometryReference, GeometrySource
from contextmap.semantic_mapping import Entity, EntityGeometry, resolve_geometry

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

GEOMETRY_COMPARISON_POLICY_ID = "entity-geometry-comparison-v1"
"""Versioned identity of the geometric metrics and rules described in this module."""

# Limita a memória do produto de distâncias: linhas por bloco vezes pontos do outro lado.
_MAX_DISTANCE_BLOCK_ELEMENTS = 1_000_000


def _require_ratio(owner: object, name: str) -> None:
    value = getattr(owner, name)
    if not (math.isfinite(value) and 0.0 < value <= 1.0):
        raise ValueError(f"{name} must be within (0, 1], got {value!r}")


@dataclass(frozen=True, kw_only=True)
class SupportDistancePolicy:
    """Whether and how to compare the supports point by point.

    Attributes:
        max_points_per_side: Most points taken from each support; a larger support is sampled at
            evenly spaced positions of its sorted geometry references, deterministically, and the
            sample size is recorded on the measurement.
        max_mean_distance_m: Mean nearest-point distance, in meters, at or under which the
            supports are considered close.
    """

    max_points_per_side: int
    max_mean_distance_m: float

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: If the sample size is not positive or the distance is not finite and
                positive.
        """
        if self.max_points_per_side < 1:
            raise ValueError(
                f"max_points_per_side must be at least 1, got {self.max_points_per_side}"
            )
        if not (math.isfinite(self.max_mean_distance_m) and self.max_mean_distance_m > 0.0):
            raise ValueError(
                f"max_mean_distance_m must be finite and positive, got {self.max_mean_distance_m!r}"
            )


@dataclass(frozen=True, kw_only=True)
class GeometryComparisonPolicy:
    """Explicit thresholds of the geometric rules.

    There are no defaults: the thresholds are scientific choices, tied to the scale and density of
    a scene, that a profile declares and justifies with validation data.

    Attributes:
        min_shared_support_jaccard: Share of the union of the supports that is common, in
            ``(0, 1]``, from which shared support is evidence for the match.
        min_bounds_iou: Intersection over union of the boxes, in ``(0, 1]``, from which the overlap
            supports the match.
        min_bounds_containment: Share of one box's volume inside the other, in ``(0, 1]``, from
            which one box is considered contained in the other.
        min_conflict_gap_m: Distance between the boxes, in meters, from which their separation
            argues against the match.
        min_extent_ratio: Ratio between the smaller and the larger side of the boxes, in
            ``(0, 1]``, under which the sizes are considered not comparable.
        support_distance: Nearest-point statistics, or ``None`` to skip them and never resolve the
            geometry.
        max_orientation_angle_rad: Angle between the principal axes, in ``[0, pi / 2]``, above
            which the orientations argue against the match; ``None`` to only measure it.
    """

    min_shared_support_jaccard: float
    min_bounds_iou: float
    min_bounds_containment: float
    min_conflict_gap_m: float
    min_extent_ratio: float
    support_distance: SupportDistancePolicy | None = None
    max_orientation_angle_rad: float | None = None

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ValueError: If a ratio is outside ``(0, 1]``, the gap is not finite and positive, or
                the angle is outside ``[0, pi / 2]``.
        """
        for name in (
            "min_shared_support_jaccard",
            "min_bounds_iou",
            "min_bounds_containment",
            "min_extent_ratio",
        ):
            _require_ratio(self, name)
        if not (math.isfinite(self.min_conflict_gap_m) and self.min_conflict_gap_m > 0.0):
            raise ValueError(
                f"min_conflict_gap_m must be finite and positive, got {self.min_conflict_gap_m!r}"
            )
        angle = self.max_orientation_angle_rad
        if angle is not None:
            require_finite(self, "max_orientation_angle_rad")
            if not 0.0 <= angle <= math.pi / 2:
                raise ValueError(
                    f"max_orientation_angle_rad must be within [0, pi / 2], got {angle!r}"
                )

    def fingerprint(self) -> str:
        """Hash the policy identity and every threshold, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        distance = self.support_distance
        canonical = json.dumps(
            {
                "policy_id": GEOMETRY_COMPARISON_POLICY_ID,
                "min_shared_support_jaccard": self.min_shared_support_jaccard,
                "min_bounds_iou": self.min_bounds_iou,
                "min_bounds_containment": self.min_bounds_containment,
                "min_conflict_gap_m": self.min_conflict_gap_m,
                "min_extent_ratio": self.min_extent_ratio,
                "support_distance": None
                if distance is None
                else {
                    "max_points_per_side": distance.max_points_per_side,
                    "max_mean_distance_m": distance.max_mean_distance_m,
                },
                "max_orientation_angle_rad": self.max_orientation_angle_rad,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def ref(self) -> PolicyRef:
        """The policy identity and configuration, as recorded on the evidence."""
        return PolicyRef(
            policy_id=GEOMETRY_COMPARISON_POLICY_ID, configuration_fingerprint=self.fingerprint()
        )


def compare_geometry(
    entity_a: Entity,
    entity_b: Entity,
    policy: GeometryComparisonPolicy,
    *,
    source: GeometrySource | None = None,
) -> GeometryEvidence:
    """Compare the persistent 3D support of two entities.

    The result does not depend on the order of the arguments: the pair is put in canonical order
    and the measurements refer to it.

    Args:
        entity_a: One entity.
        entity_b: The other entity.
        policy: The thresholds of the rules.
        source: The read boundary of the geometric map both supports belong to; required exactly
            when the policy asks for nearest-point statistics.

    Returns:
        The geometric evidence: the measurement and the finding of every rule, or an unavailable
        channel when a hard gate blocks the comparison.

    Raises:
        ValueError: If the policy asks for nearest-point statistics and no source is given.
        GeometryResolutionError: If the policy asks for them and a reference cannot be resolved
            against ``source``; an invalid reference fails loudly instead of being skipped.
    """
    first, second = sorted((entity_a, entity_b), key=lambda item: reference_order(item.reference))
    failed = [gate for gate in evaluate_comparison_gates(first, second) if not gate.passed]
    if failed:
        return GeometryEvidence(
            policy=policy.ref(),
            unavailable=Unavailability(
                reason=UnavailableReason.BLOCKED_BY_GATE,
                detail="; ".join(f"{gate.gate_id}: {gate.detail}" for gate in failed),
            ),
        )
    if policy.support_distance is not None and source is None:
        raise ValueError("the policy asks for support distances, which need a GeometrySource")
    measurement = _measure(first, second, policy, source)
    return GeometryEvidence(
        policy=policy.ref(), measurement=measurement, findings=_findings(measurement, policy)
    )


def _measure(
    first: Entity,
    second: Entity,
    policy: GeometryComparisonPolicy,
    source: GeometrySource | None,
) -> GeometryMeasurement:
    geometry_a, geometry_b = first.geometry, second.geometry
    volume_a, volume_b = box_volume(geometry_a.bounds), box_volume(geometry_b.bounds)
    shared = len(
        {ref.geometry_id for ref in geometry_a.geometry_refs}
        & {ref.geometry_id for ref in geometry_b.geometry_refs}
    )
    count_a, count_b = len(geometry_a.geometry_refs), len(geometry_b.geometry_refs)
    overlap = intersection_volume(geometry_a.bounds, geometry_b.bounds)
    flat = volume_a == 0.0 or volume_b == 0.0
    distance = None
    if policy.support_distance is not None and source is not None:
        distance = _support_distance(geometry_a, geometry_b, policy.support_distance, source)
    return GeometryMeasurement(
        geometric_map_id=geometry_a.geometric_map_id,
        map_frame=str(geometry_a.map_frame),
        centroid_distance_m=math.dist(geometry_a.centroid_m, geometry_b.centroid_m),
        bounds_gap_m=_bounds_gap(geometry_a.bounds, geometry_b.bounds),
        bounds_iou=None if flat else _clamp_unit(overlap / (volume_a + volume_b - overlap)),
        bounds_overlap_fraction_a=None if volume_a == 0.0 else _clamp_unit(overlap / volume_a),
        bounds_overlap_fraction_b=None if volume_b == 0.0 else _clamp_unit(overlap / volume_b),
        support_count_a=count_a,
        support_count_b=count_b,
        shared_support_count=shared,
        support_jaccard=shared / (count_a + count_b - shared),
        extent_ratio=extent_ratio(geometry_a.bounds, geometry_b.bounds),
        support_distance=distance,
        orientation_angle_rad=_orientation_angle(geometry_a, geometry_b),
        caveats=tuple(
            sorted(
                {f"a:{kind.value}" for kind in geometry_a.diagnostic_kinds()}
                | {f"b:{kind.value}" for kind in geometry_b.diagnostic_kinds()}
            )
        ),
    )


def _clamp_unit(value: float) -> float:
    # O arredondamento pode empurrar uma razão exata de 1 para 1 + epsilon.
    return min(1.0, max(0.0, value))


def _orientation_angle(first: EntityGeometry, second: EntityGeometry) -> float | None:
    if first.orientation is None or second.orientation is None:
        return None
    axis_a, axis_b = first.orientation.axes[0], second.orientation.axes[0]
    cosine = abs(sum(a * b for a, b in zip(axis_a, axis_b, strict=True)))
    # O eixo principal não tem sentido: o ângulo vai de 0 a 90 graus.
    return math.acos(min(1.0, cosine))


def _sample(references: tuple[GeometryReference, ...], limit: int) -> tuple[GeometryReference, ...]:
    """Evenly spaced references of a support already sorted by geometry id."""
    if len(references) <= limit:
        return references
    positions = (
        sorted({round(index * (len(references) - 1) / (limit - 1)) for index in range(limit)})
        if limit > 1
        else [0]
    )
    return tuple(references[position] for position in positions)


def _support_distance(
    first: EntityGeometry,
    second: EntityGeometry,
    policy: SupportDistancePolicy,
    source: GeometrySource,
) -> SupportDistance:
    import numpy as np

    sampled_a = _sample(first.geometry_refs, policy.max_points_per_side)
    sampled_b = _sample(second.geometry_refs, policy.max_points_per_side)
    points_a = np.array(
        [point.coordinates_m for point in resolve_geometry(sampled_a, source=source)]
    )
    points_b = np.array(
        [point.coordinates_m for point in resolve_geometry(sampled_b, source=source)]
    )
    a_to_b = _nearest_distances(points_a, points_b)
    b_to_a = _nearest_distances(points_b, points_a)
    return SupportDistance(
        a_to_b_mean_m=float(a_to_b.mean()),
        b_to_a_mean_m=float(b_to_a.mean()),
        hausdorff_m=float(max(a_to_b.max(), b_to_a.max())),
        points_a=len(sampled_a),
        points_b=len(sampled_b),
    )


def _nearest_distances(
    origin: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Distance from each origin point to its nearest target point, in blocks to bound memory."""
    import numpy as np

    rows = max(1, _MAX_DISTANCE_BLOCK_ELEMENTS // len(target))
    nearest = np.empty(len(origin))
    for start in range(0, len(origin), rows):
        block = origin[start : start + rows]
        differences = block[:, None, :] - target[None, :, :]
        nearest[start : start + rows] = np.sqrt((differences**2).sum(axis=2)).min(axis=1)
    return nearest


def _findings(
    measurement: GeometryMeasurement, policy: GeometryComparisonPolicy
) -> tuple[Finding, ...]:
    contained_fraction = _largest(
        measurement.bounds_overlap_fraction_a, measurement.bounds_overlap_fraction_b
    )
    contained = (
        contained_fraction is not None and contained_fraction >= policy.min_bounds_containment
    )
    comparable_size = (
        measurement.extent_ratio is not None and measurement.extent_ratio >= policy.min_extent_ratio
    )
    findings = [
        _shared_support(measurement, policy),
        _bounds_iou(measurement, policy),
        _containment(contained_fraction, contained, comparable_size, measurement, policy),
        _separation(measurement, policy),
        _extent_mismatch(contained, measurement, policy),
    ]
    if measurement.support_distance is not None and policy.support_distance is not None:
        findings.append(_support_proximity(measurement.support_distance, policy.support_distance))
    if policy.max_orientation_angle_rad is not None:
        findings.append(_orientation(measurement, policy.max_orientation_angle_rad))
    return tuple(findings)


def _largest(first: float | None, second: float | None) -> float | None:
    present = [value for value in (first, second) if value is not None]
    return max(present) if present else None


def _shared_support(measurement: GeometryMeasurement, policy: GeometryComparisonPolicy) -> Finding:
    jaccard = measurement.support_jaccard
    threshold = policy.min_shared_support_jaccard
    supporting = jaccard >= threshold
    union = measurement.support_count_a + measurement.support_count_b
    union -= measurement.shared_support_count
    return Finding(
        rule_id="shared-support",
        status=EvidenceStatus.SUPPORTING if supporting else EvidenceStatus.NEUTRAL,
        detail=(
            f"the supports share {measurement.shared_support_count} of {union} geometry "
            f"elements; sharing none is not evidence against"
        ),
        metric="support_jaccard",
        observed=jaccard,
        threshold=threshold,
    )


def _bounds_iou(measurement: GeometryMeasurement, policy: GeometryComparisonPolicy) -> Finding:
    iou = measurement.bounds_iou
    if iou is None:
        return Finding(
            rule_id="bounds-iou",
            status=EvidenceStatus.NEUTRAL,
            detail="not evaluable: a box is flat, so it has no volumetric overlap",
        )
    supporting = iou >= policy.min_bounds_iou
    return Finding(
        rule_id="bounds-iou",
        status=EvidenceStatus.SUPPORTING if supporting else EvidenceStatus.NEUTRAL,
        detail=f"the boxes overlap by an intersection over union of {iou:.3f}",
        metric="bounds_iou",
        observed=iou,
        threshold=policy.min_bounds_iou,
    )


def _containment(
    fraction: float | None,
    contained: bool,
    comparable_size: bool,
    measurement: GeometryMeasurement,
    policy: GeometryComparisonPolicy,
) -> Finding:
    if fraction is None:
        return Finding(
            rule_id="bounds-containment",
            status=EvidenceStatus.NEUTRAL,
            detail="not evaluable: both boxes are flat",
        )
    supporting = contained and comparable_size
    if supporting:
        detail = "one box lies inside the other and both are of comparable size (partial view)"
    elif contained:
        detail = (
            f"one box lies inside the other but the sizes differ (extent ratio "
            f"{measurement.extent_ratio}, under {policy.min_extent_ratio}): a large background "
            f"entity containing a small one is not evidence of identity"
        )
    else:
        detail = "neither box lies inside the other"
    return Finding(
        rule_id="bounds-containment",
        status=EvidenceStatus.SUPPORTING if supporting else EvidenceStatus.NEUTRAL,
        detail=detail,
        metric="bounds_overlap_fraction",
        observed=fraction,
        threshold=policy.min_bounds_containment,
    )


def _separation(measurement: GeometryMeasurement, policy: GeometryComparisonPolicy) -> Finding:
    gap = measurement.bounds_gap_m
    conflicting = gap >= policy.min_conflict_gap_m
    return Finding(
        rule_id="bounds-separation",
        status=EvidenceStatus.CONFLICTING if conflicting else EvidenceStatus.NEUTRAL,
        detail=f"the boxes are {gap:.3f} m apart",
        metric="bounds_gap_m",
        observed=gap,
        threshold=policy.min_conflict_gap_m,
    )


def _extent_mismatch(
    contained: bool, measurement: GeometryMeasurement, policy: GeometryComparisonPolicy
) -> Finding:
    ratio = measurement.extent_ratio
    if ratio is None:
        return Finding(
            rule_id="extent-mismatch",
            status=EvidenceStatus.NEUTRAL,
            detail="not evaluable: a box is flat on an axis on which the other is not",
        )
    mismatch = ratio < policy.min_extent_ratio
    if mismatch and contained:
        status = EvidenceStatus.NEUTRAL
        detail = (
            f"extent ratio {ratio:.3f}, but one box lies inside the other: partial visibility or "
            f"a large background entity explains the difference"
        )
    elif mismatch:
        status = EvidenceStatus.CONFLICTING
        detail = (
            f"extent ratio {ratio:.3f}: the boxes differ in size and neither contains the other"
        )
    else:
        status = EvidenceStatus.NEUTRAL
        detail = f"extent ratio {ratio:.3f}: the boxes are of comparable size"
    return Finding(
        rule_id="extent-mismatch",
        status=status,
        detail=detail,
        metric="extent_ratio",
        observed=ratio,
        threshold=policy.min_extent_ratio,
    )


def _support_proximity(distance: SupportDistance, policy: SupportDistancePolicy) -> Finding:
    mean = max(distance.a_to_b_mean_m, distance.b_to_a_mean_m)
    close = mean <= policy.max_mean_distance_m
    return Finding(
        rule_id="support-proximity",
        status=EvidenceStatus.SUPPORTING if close else EvidenceStatus.NEUTRAL,
        detail=(
            f"mean nearest-point distance {mean:.3f} m over {distance.points_a} and "
            f"{distance.points_b} sampled points"
        ),
        metric="support_mean_distance_m",
        observed=mean,
        threshold=policy.max_mean_distance_m,
    )


def _orientation(measurement: GeometryMeasurement, threshold_rad: float) -> Finding:
    angle = measurement.orientation_angle_rad
    if angle is None:
        return Finding(
            rule_id="orientation-mismatch",
            status=EvidenceStatus.NEUTRAL,
            detail="not evaluable: an entity has no well-defined orientation",
        )
    return Finding(
        rule_id="orientation-mismatch",
        status=EvidenceStatus.CONFLICTING if angle > threshold_rad else EvidenceStatus.NEUTRAL,
        detail=f"the principal axes differ by {math.degrees(angle):.1f} degrees",
        metric="orientation_angle_rad",
        observed=angle,
        threshold=threshold_rad,
    )

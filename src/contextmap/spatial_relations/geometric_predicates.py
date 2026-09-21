"""Geometric predicate evaluators: proximity, direction and topology from entity bounds.

The first reliable relation channel comes from the geometry of resolved entities, measured
explicitly instead of guessed from language. Every evaluator reads only the axis-aligned bounds,
the support diagnostics and the declared frame conventions of two entities, and emits a
:class:`~contextmap.spatial_relations.RelationEvidence` in the ``GEOMETRY`` channel that keeps the
exact numbers, the thresholds they were compared with and the geometry they were taken over. No
semantic label can reach the arithmetic, and no signal is folded into an undocumented score:
evidence stays inspectable independently of any later reconciliation.

Every comparison uses one rule. ``boundary_tolerance_m`` is the uncertainty of a bounds face, and a
comparison is decided only when it holds for every value within that tolerance of the measurement:

* the condition holds across the whole band, so the comparison *meets* the threshold;
* the condition fails across the whole band, so the comparison *fails* it;
* otherwise the measurement is *within tolerance* of the threshold and the verdict is ambiguous.

An evidence record is ``SUPPORTS`` when every condition meets its threshold, ``CONFLICTS`` when any
fails, and ``AMBIGUOUS`` otherwise, with one ``WITHIN_TOLERANCE`` caveat for each measurement that
could not be decided. Two further rules keep unreliable geometry from looking decisive:

* a sparse or disconnected support is never decisive, because its bounds may not describe the
  object: the record is ``AMBIGUOUS`` with an ``UNRELIABLE_GEOMETRY`` caveat;
* an interpenetration depth is not measurable along a flat axis, so ``NEXT_TO`` and
  ``INTERSECTS`` over a flat support are ``AMBIGUOUS`` with a ``DEGENERATE_GEOMETRY`` caveat.

Only one direction of each inverse pair is evaluated (``ABOVE``, ``IN_FRONT_OF``, ``INSIDE``);
``BELOW``, ``BEHIND`` and ``CONTAINS`` are generated from it. Contact and support predicates need
point-level measurements and belong to another channel.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.semantic_mapping import (
    EntityGeometry,
    GeometryDiagnosticKind,
    geometry_set_digest,
)
from contextmap.spatial_relations._bounds import (
    axis_overlap_m,
    bounds_gap_m,
    cross_section_axes,
    directed_interval,
)
from contextmap.spatial_relations._checks import require_finite
from contextmap.spatial_relations.candidates import RelationCandidateSet
from contextmap.spatial_relations.evidence import (
    EvidenceCaveat,
    EvidenceCaveatKind,
    MeasuredGeometry,
    Quantity,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceProvenance,
    RelationEvidenceStatus,
    evidence_id_for,
)
from contextmap.spatial_relations.frame_conventions import AxisDirection, FrameConventions
from contextmap.spatial_relations.taxonomy import (
    TAXONOMY_VERSION,
    FrameRequirement,
    RelationPredicate,
    predicate_spec,
)

GEOMETRIC_POLICY_ID = "bounds-geometric-predicates-v1"
"""Versioned identity of the evaluation rules described in this module."""

GEOMETRIC_PREDICATES = frozenset(
    {
        RelationPredicate.NEXT_TO,
        RelationPredicate.ABOVE,
        RelationPredicate.IN_FRONT_OF,
        RelationPredicate.INSIDE,
        RelationPredicate.INTERSECTS,
    }
)
"""The predicates this channel evaluates directly."""

_RULE_IDS = {
    RelationPredicate.NEXT_TO: "bounds-next-to-v1",
    RelationPredicate.ABOVE: "bounds-above-v1",
    RelationPredicate.IN_FRONT_OF: "bounds-in-front-of-v1",
    RelationPredicate.INSIDE: "bounds-inside-v1",
    RelationPredicate.INTERSECTS: "bounds-intersects-v1",
}
_AXIS_LETTERS = "xyz"


@dataclass(frozen=True, kw_only=True)
class GeometricPredicatePolicy:
    """Explicit thresholds of the geometric evaluators.

    There are no defaults: the thresholds are scientific choices that a profile declares, and none
    is specific to a dataset.

    Attributes:
        boundary_tolerance_m: Uncertainty of a bounds face, in meters. A comparison whose outcome
            depends on where inside this tolerance the true face lies is ambiguous.
        next_to_max_gap_m: Largest gap between the bounds, in meters, that still counts as next to.
        adjacent_penetration_m: Deepest interpenetration of the bounds, in meters, that still
            counts as adjacent rather than intersecting. It also bounds how far a subject may
            sink into an object and still be above it. It splits ``NEXT_TO`` from
            ``INTERSECTS`` so that both are never supported for the same pair.
        containment_slack_m: How far a face of the subject may protrude beyond the object's face,
            in meters, and still count as inside; it absorbs noise in partially observed bounds.
        directional_overlap_fraction: Share, in ``(0, 1]``, of the smaller extent that the
            cross-sections of subject and object must share on each axis perpendicular to the
            direction of ``ABOVE`` or ``IN_FRONT_OF``.
    """

    boundary_tolerance_m: float
    next_to_max_gap_m: float
    adjacent_penetration_m: float
    containment_slack_m: float
    directional_overlap_fraction: float

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ValueError: If the tolerance or the gap is not finite and positive, the penetration or
                the slack is not finite and non-negative, or the overlap fraction is not within
                ``(0, 1]``.
        """
        for name in ("boundary_tolerance_m", "next_to_max_gap_m"):
            value: float = getattr(self, name)
            require_finite(name, value)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        for name in ("adjacent_penetration_m", "containment_slack_m"):
            value = getattr(self, name)
            require_finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must not be negative, got {value!r}")
        require_finite("directional_overlap_fraction", self.directional_overlap_fraction)
        if not 0.0 < self.directional_overlap_fraction <= 1.0:
            raise ValueError(
                f"directional_overlap_fraction must be within (0, 1], got "
                f"{self.directional_overlap_fraction!r}"
            )

    def fingerprint(self) -> str:
        """Hash the policy identity and thresholds, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "policy_id": GEOMETRIC_POLICY_ID,
                "boundary_tolerance_m": self.boundary_tolerance_m,
                "next_to_max_gap_m": self.next_to_max_gap_m,
                "adjacent_penetration_m": self.adjacent_penetration_m,
                "containment_slack_m": self.containment_slack_m,
                "directional_overlap_fraction": self.directional_overlap_fraction,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


class _Verdict(Enum):
    """The outcome of one comparison against one threshold."""

    MEETS = "meets"
    FAILS = "fails"
    WITHIN_TOLERANCE = "within_tolerance"


@dataclass(frozen=True)
class _Check:
    """One condition of a predicate, decided under the boundary tolerance."""

    name: str
    verdict: _Verdict
    value: float
    threshold: float
    tolerance: float

    def caveat(self) -> EvidenceCaveat:
        return EvidenceCaveat(
            kind=EvidenceCaveatKind.WITHIN_TOLERANCE,
            detail=(
                f"{self.name} {self.value:.6g} is within {self.tolerance:.6g} m of its "
                f"threshold {self.threshold:.6g}"
            ),
        )


def _at_least(name: str, value: float, threshold: float, tolerance: float) -> _Check:
    if value - tolerance >= threshold:
        verdict = _Verdict.MEETS
    elif value + tolerance < threshold:
        verdict = _Verdict.FAILS
    else:
        verdict = _Verdict.WITHIN_TOLERANCE
    return _Check(name, verdict, value, threshold, tolerance)


def _at_most(name: str, value: float, threshold: float, tolerance: float) -> _Check:
    if value + tolerance <= threshold:
        verdict = _Verdict.MEETS
    elif value - tolerance > threshold:
        verdict = _Verdict.FAILS
    else:
        verdict = _Verdict.WITHIN_TOLERANCE
    return _Check(name, verdict, value, threshold, tolerance)


@dataclass(frozen=True)
class _Assessment:
    """The numbers and conditions of one predicate over one pair, before they become evidence."""

    measurements: list[Quantity]
    thresholds: list[Quantity]
    checks: list[_Check]
    measures_penetration_depth: bool = False


def evaluate_geometric_predicate(
    predicate: RelationPredicate,
    *,
    subject_entity_ref: ResolvedEntityReference,
    subject_geometry: EntityGeometry,
    object_entity_ref: ResolvedEntityReference,
    object_geometry: EntityGeometry,
    policy: GeometricPredicatePolicy,
    conventions: FrameConventions,
) -> RelationEvidence:
    """Measure one geometric predicate for one directed pair.

    Args:
        predicate: A predicate of the geometric channel, evaluated in the given direction.
        subject_entity_ref: The entity the statement is about.
        subject_geometry: Its geometry.
        object_entity_ref: The entity it is related to.
        object_geometry: Its geometry.
        policy: The thresholds.
        conventions: The axes the run declared for the map frame.

    Returns:
        The evidence: the status, the exact measurements, the thresholds, the geometry measured
        and the caveats that limit it.

    Raises:
        ValueError: If the predicate is derived (evaluate its inverse and invert the relation), or
            belongs to the contact channel, or the entities are not a relatable pair.
        IncompatibleFrameError: If a geometry is not in the frame of ``conventions`` or the two
            belong to different geometric maps.
        UndeclaredAxisError: If the predicate needs an axis that the conventions do not declare.
    """
    _require_geometric(predicate)
    spec = predicate_spec(predicate)
    conventions.require_evaluable(spec.frame_requirement, subject_geometry, object_geometry)
    assessment = _ASSESSORS[predicate](subject_geometry, object_geometry, policy, conventions)
    status, caveats = _decide(assessment, subject_geometry, object_geometry)
    uses_axes = spec.frame_requirement is not FrameRequirement.MAP_FRAME
    return RelationEvidence(
        evidence_id=evidence_id_for(
            channel=RelationEvidenceChannel.GEOMETRY,
            subject_entity_ref=subject_entity_ref,
            predicate=predicate,
            object_entity_ref=object_entity_ref,
        ),
        channel=RelationEvidenceChannel.GEOMETRY,
        subject_entity_ref=subject_entity_ref,
        predicate=predicate,
        object_entity_ref=object_entity_ref,
        status=status,
        measurements=tuple(sorted(assessment.measurements, key=lambda item: item.name)),
        thresholds=tuple(sorted(assessment.thresholds, key=lambda item: item.name)),
        geometry=(
            _measured_geometry("object", object_geometry),
            _measured_geometry("subject", subject_geometry),
        ),
        caveats=tuple(sorted(caveats, key=lambda item: (item.kind.value, item.detail))),
        provenance=RelationEvidenceProvenance(
            rule_id=_RULE_IDS[predicate],
            configuration_fingerprint=policy.fingerprint(),
            taxonomy_version=TAXONOMY_VERSION,
            map_frame=conventions.map_frame,
            geometric_map_id=subject_geometry.geometric_map_id,
            frame_conventions_fingerprint=conventions.fingerprint() if uses_axes else None,
        ),
    )


def evaluate_geometric_candidates(
    candidates: RelationCandidateSet,
    *,
    entities: Mapping[ResolvedEntityReference, EntityGeometry],
    policy: GeometricPredicatePolicy,
    conventions: FrameConventions,
) -> tuple[RelationEvidence, ...]:
    """Evaluate the geometric candidates of a candidate set, in candidate order.

    Candidates of other channels (contact and support predicates) are left to their own
    evaluators; nothing is dropped from ``candidates``.

    Args:
        candidates: The output of candidate generation.
        entities: The geometry of every resolved entity the candidates name.
        policy: The thresholds.
        conventions: The axes the run declared; they must be the ones the candidates were
            generated under.

    Returns:
        One evidence record per geometric candidate.

    Raises:
        ValueError: If the candidates were generated under other frame conventions.
    """
    if candidates.provenance.frame_conventions_fingerprint != conventions.fingerprint():
        raise ValueError(
            "the candidates were generated under other frame conventions than the ones given"
        )
    return tuple(
        evaluate_geometric_predicate(
            candidate.predicate,
            subject_entity_ref=candidate.subject_entity_ref,
            subject_geometry=entities[candidate.subject_entity_ref],
            object_entity_ref=candidate.object_entity_ref,
            object_geometry=entities[candidate.object_entity_ref],
            policy=policy,
            conventions=conventions,
        )
        for candidate in candidates.candidates
        if candidate.predicate in GEOMETRIC_PREDICATES
    )


def _require_geometric(predicate: RelationPredicate) -> None:
    if predicate in GEOMETRIC_PREDICATES:
        return
    spec = predicate_spec(predicate)
    if spec.is_derived and spec.inverse is not None:
        raise ValueError(
            f"{predicate.name} is derived: evaluate {spec.inverse.name} and invert the relation"
        )
    raise ValueError(f"{predicate.name} belongs to the contact channel, not to bounds geometry")


def _decide(
    assessment: _Assessment, subject: EntityGeometry, obj: EntityGeometry
) -> tuple[RelationEvidenceStatus, list[EvidenceCaveat]]:
    verdicts = [check.verdict for check in assessment.checks]
    if _Verdict.FAILS in verdicts:
        status = RelationEvidenceStatus.CONFLICTS
    elif all(verdict is _Verdict.MEETS for verdict in verdicts):
        status = RelationEvidenceStatus.SUPPORTS
    else:
        status = RelationEvidenceStatus.AMBIGUOUS
    caveats = [
        check.caveat() for check in assessment.checks if check.verdict is _Verdict.WITHIN_TOLERANCE
    ]
    unreliable = _reliability_caveats(subject, obj, assessment.measures_penetration_depth)
    if unreliable:
        # Limites de um suporte esparso, desconexo ou plano podem não descrever o objeto: nunca
        # decisivos, mas as medições e a razão continuam registradas.
        status = RelationEvidenceStatus.AMBIGUOUS
        caveats.extend(unreliable)
    return status, caveats


def _reliability_caveats(
    subject: EntityGeometry, obj: EntityGeometry, measures_penetration_depth: bool
) -> list[EvidenceCaveat]:
    caveats: list[EvidenceCaveat] = []
    for role, geometry in (("subject", subject), ("object", obj)):
        kinds = geometry.diagnostic_kinds()
        if GeometryDiagnosticKind.SPARSE_SUPPORT in kinds:
            caveats.append(
                EvidenceCaveat(
                    kind=EvidenceCaveatKind.UNRELIABLE_GEOMETRY,
                    detail=(
                        f"the {role} support is sparse ({geometry.statistics.point_count} "
                        f"points): its bounds may not describe the object"
                    ),
                )
            )
        if GeometryDiagnosticKind.DISCONNECTED_SUPPORT in kinds:
            caveats.append(
                EvidenceCaveat(
                    kind=EvidenceCaveatKind.UNRELIABLE_GEOMETRY,
                    detail=(
                        f"the {role} support is disconnected "
                        f"({geometry.statistics.component_count} components): its bounds may not "
                        f"describe one object"
                    ),
                )
            )
        if measures_penetration_depth and GeometryDiagnosticKind.DEGENERATE_EXTENT in kinds:
            caveats.append(
                EvidenceCaveat(
                    kind=EvidenceCaveatKind.DEGENERATE_GEOMETRY,
                    detail=(
                        f"the {role} support is flat on some axis, so an interpenetration depth "
                        f"cannot be measured along it"
                    ),
                )
            )
    return caveats


def _measured_geometry(role: str, geometry: EntityGeometry) -> MeasuredGeometry:
    return MeasuredGeometry(
        role=role,
        point_count=geometry.statistics.point_count,
        geometry_digest=geometry_set_digest(geometry.geometry_refs),
    )


def _quantity(name: str, value: float, unit: str) -> Quantity:
    return Quantity(name=name, value=value, unit=unit)


def _axis_overlaps(subject: EntityGeometry, obj: EntityGeometry) -> list[float]:
    return [axis_overlap_m(subject.bounds, obj.bounds, axis) for axis in range(3)]


def _policy_thresholds(policy: GeometricPredicatePolicy, *names: str) -> list[Quantity]:
    values = {
        "adjacent_penetration": policy.adjacent_penetration_m,
        "boundary_tolerance": policy.boundary_tolerance_m,
        "containment_slack": policy.containment_slack_m,
        "next_to_max_gap": policy.next_to_max_gap_m,
    }
    return [_quantity(name, values[name], "m") for name in names]


def _assess_next_to(
    subject: EntityGeometry,
    obj: EntityGeometry,
    policy: GeometricPredicatePolicy,
    conventions: FrameConventions,
) -> _Assessment:
    gap = bounds_gap_m(subject.bounds, obj.bounds)
    depth = min(_axis_overlaps(subject, obj))
    tolerance = policy.boundary_tolerance_m
    return _Assessment(
        measurements=[
            _quantity("bounds_gap", gap, "m"),
            _quantity("min_axis_overlap", depth, "m"),
        ],
        thresholds=_policy_thresholds(
            policy, "adjacent_penetration", "boundary_tolerance", "next_to_max_gap"
        ),
        checks=[
            _at_most("bounds_gap", gap, policy.next_to_max_gap_m, tolerance),
            _at_most("min_axis_overlap", depth, policy.adjacent_penetration_m, tolerance),
        ],
        measures_penetration_depth=True,
    )


def _assess_intersects(
    subject: EntityGeometry,
    obj: EntityGeometry,
    policy: GeometricPredicatePolicy,
    conventions: FrameConventions,
) -> _Assessment:
    overlaps = _axis_overlaps(subject, obj)
    depth = min(overlaps)
    return _Assessment(
        measurements=[
            *(
                _quantity(f"axis_overlap_{letter}", overlap, "m")
                for letter, overlap in zip(_AXIS_LETTERS, overlaps, strict=True)
            ),
            _quantity("min_axis_overlap", depth, "m"),
        ],
        thresholds=_policy_thresholds(policy, "adjacent_penetration", "boundary_tolerance"),
        checks=[
            _at_least(
                "min_axis_overlap",
                depth,
                policy.adjacent_penetration_m,
                policy.boundary_tolerance_m,
            )
        ],
        measures_penetration_depth=True,
    )


def _assess_inside(
    subject: EntityGeometry,
    obj: EntityGeometry,
    policy: GeometricPredicatePolicy,
    conventions: FrameConventions,
) -> _Assessment:
    margin = min(
        min(
            subject.bounds.minimum_m[axis] - obj.bounds.minimum_m[axis],
            obj.bounds.maximum_m[axis] - subject.bounds.maximum_m[axis],
        )
        for axis in range(3)
    )
    return _Assessment(
        measurements=[_quantity("containment_margin", margin, "m")],
        thresholds=_policy_thresholds(policy, "boundary_tolerance", "containment_slack"),
        checks=[
            _at_least(
                "containment_margin",
                margin,
                -policy.containment_slack_m,
                policy.boundary_tolerance_m,
            )
        ],
    )


def _assess_directional(
    subject: EntityGeometry,
    obj: EntityGeometry,
    policy: GeometricPredicatePolicy,
    direction: AxisDirection,
    clearance_name: str,
) -> _Assessment:
    """Assess "the subject lies further along ``direction`` than the object, over the same spot".

    Two conditions, both decided under the boundary tolerance: the subject's low face along the
    direction does not sink into the object's high face by more than the adjacent penetration,
    and on each axis perpendicular to the direction the two share at least the configured fraction
    of the smaller extent.
    """
    tolerance = policy.boundary_tolerance_m
    subject_low, _ = directed_interval(subject.bounds, direction)
    _, object_high = directed_interval(obj.bounds, direction)
    clearance = subject_low - object_high
    measurements = [_quantity(clearance_name, clearance, "m")]
    thresholds = _policy_thresholds(policy, "adjacent_penetration", "boundary_tolerance")
    thresholds.append(
        _quantity("directional_overlap_fraction", policy.directional_overlap_fraction, "ratio")
    )
    checks = [_at_least(clearance_name, clearance, -policy.adjacent_penetration_m, tolerance)]
    for axis in cross_section_axes(direction):
        letter = _AXIS_LETTERS[axis]
        overlap = axis_overlap_m(subject.bounds, obj.bounds, axis)
        smaller_extent = min(subject.extent_m[axis], obj.extent_m[axis])
        required = policy.directional_overlap_fraction * smaller_extent
        measurements.append(_quantity(f"footprint_overlap_{letter}", overlap, "m"))
        if smaller_extent > 0.0:
            measurements.append(
                _quantity(f"footprint_overlap_fraction_{letter}", overlap / smaller_extent, "ratio")
            )
        thresholds.append(_quantity(f"required_footprint_overlap_{letter}", required, "m"))
        checks.append(_at_least(f"footprint_overlap_{letter}", overlap, required, tolerance))
    return _Assessment(measurements=measurements, thresholds=thresholds, checks=checks)


def _assess_above(
    subject: EntityGeometry,
    obj: EntityGeometry,
    policy: GeometricPredicatePolicy,
    conventions: FrameConventions,
) -> _Assessment:
    return _assess_directional(
        subject, obj, policy, _declared(conventions.up_axis), "vertical_clearance"
    )


def _assess_in_front_of(
    subject: EntityGeometry,
    obj: EntityGeometry,
    policy: GeometricPredicatePolicy,
    conventions: FrameConventions,
) -> _Assessment:
    return _assess_directional(
        subject, obj, policy, _declared(conventions.forward_axis), "forward_clearance"
    )


def _declared(axis: AxisDirection | None) -> AxisDirection:
    """Return a declared axis; ``require_evaluable`` has already refused an undeclared one."""
    if axis is None:
        raise ValueError("the axis is not declared")
    return axis


_ASSESSORS = {
    RelationPredicate.NEXT_TO: _assess_next_to,
    RelationPredicate.INTERSECTS: _assess_intersects,
    RelationPredicate.INSIDE: _assess_inside,
    RelationPredicate.ABOVE: _assess_above,
    RelationPredicate.IN_FRONT_OF: _assess_in_front_of,
}

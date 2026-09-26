"""Contact and support evidence: touching, on top of and leaning against.

A distance between centroids or between boxes cannot tell touching from merely nearby, or resting
from hovering. These predicates need *point-level* measurements, so the evaluators resolve the
points of both entities from the geometric map and measure them directly: the nearest support
distance between the two point sets, how many points of each entity are in contact and so how large
a contact can be measured, how the subject's lowest face sits against the object's highest face,
how much of the subject's footprint the object supports, and how far the subject is tilted from the
declared up axis. Every accepted relation therefore traces to measurements and to the exact
geometry subsets used, which the evidence lists by identity.

The conditions of each predicate:

* ``TOUCHING`` (symmetric): the nearest support distance is within the contact distance, and each
  entity has enough points in contact for a contact area to be measurable.
* ``ON_TOP_OF``: touching, the subject's lowest face is within a tolerance of the object's highest
  face along the declared up axis, and the object supports at least a fraction of the subject's
  extent on each horizontal axis.
* ``LEANING_AGAINST``: touching, the subject's heights overlap the object's, and the subject's
  dominant axis is tilted from the up axis by an angle between a lower and an upper bound (neither
  upright nor lying flat). The tilt needs the orientation of the subject; without one the evidence
  is unavailable, never guessed.

Comparisons follow the decision rule of ``_assessment``: a comparison that does not hold or fail
across the whole tolerance band is ambiguous. There is no physics simulation, no inferred gravity
and no label anywhere: the up axis is the one the run declared, and no surface normals exist
upstream, so none are used.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.geometric_mapping import GeometryPoint, GeometryReference, GeometrySource
from contextmap.semantic_mapping import EntityGeometry, geometry_set_digest, resolve_geometry
from contextmap.spatial_relations._assessment import (
    Assessment,
    Check,
    Verdict,
    at_least,
    at_most,
    decide,
    measured_geometry,
    quantity,
)
from contextmap.spatial_relations._bounds import (
    axis_overlap_m,
    bounds_gap_m,
    cross_section_axes,
    directed_interval,
    widest_spread_axis,
)
from contextmap.spatial_relations._checks import require_finite
from contextmap.spatial_relations.candidates import RelationCandidate, RelationCandidateSet
from contextmap.spatial_relations.evidence import (
    EvidenceCaveat,
    EvidenceCaveatKind,
    MeasuredGeometry,
    Quantity,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceProvenance,
    evidence_id_for,
)
from contextmap.spatial_relations.frame_conventions import (
    AxisDirection,
    FrameConventions,
    IncompatibleFrameError,
)
from contextmap.spatial_relations.taxonomy import (
    TAXONOMY_VERSION,
    FrameRequirement,
    RelationPredicate,
    predicate_spec,
)

CONTACT_POLICY_ID = "point-contact-predicates-v1"
"""Versioned identity of the evaluation rules described in this module."""

CONTACT_PREDICATES = frozenset(
    {
        RelationPredicate.TOUCHING,
        RelationPredicate.ON_TOP_OF,
        RelationPredicate.LEANING_AGAINST,
    }
)
"""The predicates this channel evaluates directly."""

_RULE_IDS = {
    RelationPredicate.TOUCHING: "point-contact-touching-v1",
    RelationPredicate.ON_TOP_OF: "point-contact-on-top-of-v1",
    RelationPredicate.LEANING_AGAINST: "point-contact-leaning-against-v1",
}
_AXIS_LETTERS = "xyz"

# Um subconjunto de contato pequeno é listado por referência; um grande guarda só a contagem e o
# digest, para o registro não crescer com a densidade da nuvem.
_MAX_LISTED_REFERENCES = 256


@dataclass(frozen=True, kw_only=True)
class ContactPredicatePolicy:
    """Explicit thresholds of the contact and support evaluators.

    There are no defaults: the thresholds are scientific choices that a profile declares, and none
    is specific to a dataset.

    Attributes:
        contact_distance_m: Distance, in meters, at or below which two points are in contact.
        contact_tolerance_m: Uncertainty of a point position, in meters. A comparison whose
            outcome depends on where inside this tolerance the true position lies is ambiguous.
        min_contact_points: Points each entity needs in contact for a contact area to be
            measurable; fewer makes the contact ambiguous.
        support_height_tolerance_m: How far the subject's lowest face may lie from the object's
            highest face, in meters, for ``ON_TOP_OF``.
        support_footprint_fraction: Share, in ``(0, 1]``, of the subject's extent that the object
            must support on each horizontal axis for ``ON_TOP_OF``.
        leaning_min_tilt_deg: Smallest tilt of the subject from the up axis, in degrees, that is
            leaning rather than upright.
        leaning_max_tilt_deg: Largest such tilt, in degrees, that is leaning rather than lying
            flat; below ninety.
        tilt_tolerance_deg: Uncertainty of a measured tilt, in degrees.
        leaning_min_vertical_overlap_m: How much the heights of subject and object must overlap,
            in meters, for the subject to lean on the object's side.
    """

    contact_distance_m: float
    contact_tolerance_m: float
    min_contact_points: int
    support_height_tolerance_m: float
    support_footprint_fraction: float
    leaning_min_tilt_deg: float
    leaning_max_tilt_deg: float
    tilt_tolerance_deg: float
    leaning_min_vertical_overlap_m: float

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ValueError: If a distance or tolerance is not finite and positive, a tolerance or
                overlap that may be zero is negative, the fraction is outside ``(0, 1]``, the
                point count is below one, or the tilt bounds are not ``0 < min < max < 90``.
        """
        for name in (
            "contact_distance_m",
            "contact_tolerance_m",
            "tilt_tolerance_deg",
        ):
            value: float = getattr(self, name)
            require_finite(name, value)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        for name in ("support_height_tolerance_m", "leaning_min_vertical_overlap_m"):
            value = getattr(self, name)
            require_finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must not be negative, got {value!r}")
        if self.min_contact_points < 1:
            raise ValueError(
                f"min_contact_points must be at least 1, got {self.min_contact_points}"
            )
        require_finite("support_footprint_fraction", self.support_footprint_fraction)
        if not 0.0 < self.support_footprint_fraction <= 1.0:
            raise ValueError(
                f"support_footprint_fraction must be within (0, 1], got "
                f"{self.support_footprint_fraction!r}"
            )
        require_finite("leaning_min_tilt_deg", self.leaning_min_tilt_deg)
        require_finite("leaning_max_tilt_deg", self.leaning_max_tilt_deg)
        if not 0.0 < self.leaning_min_tilt_deg < self.leaning_max_tilt_deg:
            raise ValueError(
                "leaning_min_tilt_deg must be positive and below leaning_max_tilt_deg, got "
                f"{self.leaning_min_tilt_deg!r} and {self.leaning_max_tilt_deg!r}"
            )
        if not self.leaning_max_tilt_deg < 90.0:
            raise ValueError(
                f"leaning_max_tilt_deg must be below 90, got {self.leaning_max_tilt_deg!r}"
            )

    @property
    def search_radius_m(self) -> float:
        """The farthest a point pair can lie and still decide the nearest-distance condition."""
        return self.contact_distance_m + self.contact_tolerance_m

    def fingerprint(self) -> str:
        """Hash the policy identity and thresholds, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "policy_id": CONTACT_POLICY_ID,
                "contact_distance_m": self.contact_distance_m,
                "contact_tolerance_m": self.contact_tolerance_m,
                "min_contact_points": self.min_contact_points,
                "support_height_tolerance_m": self.support_height_tolerance_m,
                "support_footprint_fraction": self.support_footprint_fraction,
                "leaning_min_tilt_deg": self.leaning_min_tilt_deg,
                "leaning_max_tilt_deg": self.leaning_max_tilt_deg,
                "tilt_tolerance_deg": self.tilt_tolerance_deg,
                "leaning_min_vertical_overlap_m": self.leaning_min_vertical_overlap_m,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


@dataclass(frozen=True)
class _Contact:
    """What the points of two entities say about their contact.

    Attributes:
        nearest_m: The smallest distance between a subject point and an object point, when some
            pair lies within the search radius; ``None`` otherwise.
        pairs_within_search_radius: Point pairs within the search radius.
        subject_points: The subject points within the contact distance of an object point.
        object_points: The object points within the contact distance of a subject point.
    """

    nearest_m: float | None
    pairs_within_search_radius: int
    subject_points: tuple[GeometryPoint, ...]
    object_points: tuple[GeometryPoint, ...]


_PointGrid = Mapping[tuple[int, int, int], Sequence[GeometryPoint]]
"""Points bucketed by the cubic cell they fall in, keyed by integer cell coordinates."""


def evaluate_contact_predicate(
    predicate: RelationPredicate,
    *,
    subject_entity_ref: ResolvedEntityReference,
    subject_geometry: EntityGeometry,
    object_entity_ref: ResolvedEntityReference,
    object_geometry: EntityGeometry,
    geometry_source: GeometrySource,
    policy: ContactPredicatePolicy,
    conventions: FrameConventions,
) -> RelationEvidence:
    """Measure one contact or support predicate for one directed pair.

    Args:
        predicate: A predicate of the contact channel, evaluated in the given direction.
        subject_entity_ref: The entity the statement is about.
        subject_geometry: Its geometry.
        object_entity_ref: The entity it is related to.
        object_geometry: Its geometry.
        geometry_source: The read boundary of the geometric map the entities belong to, from which
            their points are resolved.
        policy: The thresholds.
        conventions: The axes the run declared for the map frame.

    Returns:
        The evidence: the status, the exact measurements, the thresholds, the geometry measured
        (the whole supports and the points in contact, by identity) and the caveats.

    Raises:
        ValueError: If the predicate does not belong to the contact channel, or the entities are
            not a relatable pair.
        IncompatibleFrameError: If a geometry or the source is not in the frame of
            ``conventions``, or the two entities belong to different geometric maps.
        UndeclaredAxisError: If the predicate needs an axis that the conventions do not declare.
        GeometryResolutionError: If the points of an entity cannot be resolved from the source.
    """
    _require_contact(predicate)
    _require_source_frame(geometry_source, conventions)
    subject_points = resolve_geometry(subject_geometry.geometry_refs, source=geometry_source)
    object_points = resolve_geometry(object_geometry.geometry_refs, source=geometry_source)
    return _evaluate(
        predicate,
        subject_entity_ref,
        subject_geometry,
        subject_points,
        object_entity_ref,
        object_geometry,
        _contact_grid(object_points, policy.search_radius_m),
        policy,
        conventions,
    )


def evaluate_contact_candidates(
    candidates: RelationCandidateSet,
    *,
    entities: Mapping[ResolvedEntityReference, EntityGeometry],
    geometry_source: GeometrySource,
    policy: ContactPredicatePolicy,
    conventions: FrameConventions,
) -> tuple[RelationEvidence, ...]:
    """Evaluate the contact candidates of a candidate set, returned in candidate order.

    Candidates of the geometric channel are left to their own evaluator; nothing is dropped from
    ``candidates``. The points of an entity are resolved once, however many candidates name it,
    and bucketed into a contact grid at most once, the first time it is an object. Both are
    released right after the last contact candidate that names the entity, and none is ever
    resolved twice.

    Candidate order follows entity identities, which say nothing about where the entities are, so
    evaluating in that order would keep almost every entity of a dense scene resident. The
    candidates are instead evaluated in sweep order (see :func:`_sweep_order`): an entity is then
    needed only while the sweep crosses it and its neighbors, and only the entities around the
    sweep front stay resident. Every evaluation is a pure function of its pair, so the order
    changes what is resident, never the evidence.

    Args:
        candidates: The output of candidate generation.
        entities: The geometry of every resolved entity the candidates name.
        geometry_source: The read boundary of the geometric map the entities belong to.
        policy: The thresholds.
        conventions: The axes the run declared; they must be the ones the candidates were
            generated under.

    Returns:
        One evidence record per contact candidate.

    Raises:
        ValueError: If the candidates were generated under other frame conventions.
    """
    if candidates.provenance.frame_conventions_fingerprint != conventions.fingerprint():
        raise ValueError(
            "the candidates were generated under other frame conventions than the ones given"
        )
    _require_source_frame(geometry_source, conventions)
    points: dict[ResolvedEntityReference, tuple[GeometryPoint, ...]] = {}
    grids: dict[ResolvedEntityReference, _PointGrid] = {}

    def resolved(reference: ResolvedEntityReference) -> tuple[GeometryPoint, ...]:
        if reference not in points:
            points[reference] = resolve_geometry(
                entities[reference].geometry_refs, source=geometry_source
            )
        return points[reference]

    def grid(reference: ResolvedEntityReference) -> _PointGrid:
        # O raio de busca é o da política, fixo nesta chamada: a entidade basta como chave.
        if reference not in grids:
            grids[reference] = _contact_grid(resolved(reference), policy.search_radius_m)
        return grids[reference]

    contact = [item for item in candidates.candidates if item.predicate in CONTACT_PREDICATES]
    order = _sweep_order(contact, entities)
    # Posição, na varredura, do último candidato que nomeia cada entidade: depois dele, nem os
    # pontos nem a grade dela servem a outro candidato, então são liberados.
    last_use = {
        reference: position
        for position, index in enumerate(order)
        for reference in (contact[index].subject_entity_ref, contact[index].object_entity_ref)
    }
    evidence: dict[int, RelationEvidence] = {}
    for position, index in enumerate(order):
        candidate = contact[index]
        subject, obj = candidate.subject_entity_ref, candidate.object_entity_ref
        evidence[index] = _evaluate(
            candidate.predicate,
            subject,
            entities[subject],
            resolved(subject),
            obj,
            entities[obj],
            grid(obj),
            policy,
            conventions,
        )
        for reference in (subject, obj):
            if last_use[reference] == position:
                points.pop(reference, None)
                grids.pop(reference, None)
    return tuple(evidence[index] for index in range(len(contact)))


def _sweep_order(
    candidates: Sequence[RelationCandidate],
    entities: Mapping[ResolvedEntityReference, EntityGeometry],
) -> list[int]:
    """Order candidates by where a sweep along the scene reaches both of their entities.

    The sweep runs along the axis on which the entities are most spread out, and a candidate is
    reached once the sweep has passed the lower face of both of its entities. An entity is then
    first needed at its own lower face and last needed at the lower face of its farthest
    neighbor, which lies within the candidate reach of its upper face, so the entities needed at
    any point are the ones the sweep front is crossing.

    Args:
        candidates: The candidates to order.
        entities: The geometry of every entity they name.

    Returns:
        The indexes of ``candidates``, in sweep order; ties keep candidate order.
    """
    if not candidates:
        return []
    named = {item.subject_entity_ref for item in candidates} | {
        item.object_entity_ref for item in candidates
    }
    axis = widest_spread_axis([entities[reference].bounds for reference in named])

    def reached(index: int) -> float:
        candidate = candidates[index]
        return max(
            entities[candidate.subject_entity_ref].bounds.minimum_m[axis],
            entities[candidate.object_entity_ref].bounds.minimum_m[axis],
        )

    return sorted(range(len(candidates)), key=lambda index: (reached(index), index))


def _require_contact(predicate: RelationPredicate) -> None:
    if predicate not in CONTACT_PREDICATES:
        raise ValueError(
            f"{predicate.name} is not a contact predicate: geometric predicates are evaluated "
            f"by evaluate_geometric_predicate"
        )


def _require_source_frame(source: GeometrySource, conventions: FrameConventions) -> None:
    if source.geometric_map.frame_id != conventions.map_frame:
        raise IncompatibleFrameError(
            f"the geometry source is expressed in frame {source.geometric_map.frame_id!r}, but "
            f"the conventions describe {conventions.map_frame!r}"
        )


def _evaluate(
    predicate: RelationPredicate,
    subject_entity_ref: ResolvedEntityReference,
    subject_geometry: EntityGeometry,
    subject_points: Sequence[GeometryPoint],
    object_entity_ref: ResolvedEntityReference,
    object_geometry: EntityGeometry,
    object_grid: _PointGrid,
    policy: ContactPredicatePolicy,
    conventions: FrameConventions,
) -> RelationEvidence:
    spec = predicate_spec(predicate)
    conventions.require_evaluable(spec.frame_requirement, subject_geometry, object_geometry)
    contact = _measure_contact(
        subject_points, object_grid, policy.contact_distance_m, policy.search_radius_m
    )
    assessment = _ASSESSORS[predicate](
        subject_geometry, object_geometry, contact, policy, conventions
    )
    status, caveats = decide(assessment, subject_geometry, object_geometry)
    geometry = [
        measured_geometry("object", object_geometry),
        measured_geometry("subject", subject_geometry),
        *assessment.measured_geometry,
    ]
    uses_axes = spec.frame_requirement is not FrameRequirement.MAP_FRAME
    return RelationEvidence(
        evidence_id=evidence_id_for(
            channel=RelationEvidenceChannel.CONTACT,
            subject_entity_ref=subject_entity_ref,
            predicate=predicate,
            object_entity_ref=object_entity_ref,
        ),
        channel=RelationEvidenceChannel.CONTACT,
        subject_entity_ref=subject_entity_ref,
        predicate=predicate,
        object_entity_ref=object_entity_ref,
        status=status,
        measurements=tuple(sorted(assessment.measurements, key=lambda item: item.name)),
        thresholds=tuple(sorted(assessment.thresholds, key=lambda item: item.name)),
        geometry=tuple(sorted(geometry, key=lambda item: item.role)),
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


def _contact_grid(points: Sequence[GeometryPoint], search_radius_m: float) -> _PointGrid:
    """Bucket the points of an object in cubic cells whose side is the search radius.

    The grid depends only on the points and the radius, so it can serve every subject measured
    against the same object with the same radius.
    """
    grid: dict[tuple[int, int, int], list[GeometryPoint]] = defaultdict(list)
    for point in points:
        grid[_cell(point.coordinates_m, search_radius_m)].append(point)
    return grid


def _measure_contact(
    subject_points: Sequence[GeometryPoint],
    object_grid: _PointGrid,
    contact_distance_m: float,
    search_radius_m: float,
) -> _Contact:
    """Find the point pairs within the search radius with a uniform grid.

    The object points are bucketed by :func:`_contact_grid` in cells of the same search radius,
    so every subject point only meets the object points of the 27 cells around it. The result is
    exact for every pair within the search radius and does not depend on the order the points are
    given in.
    """
    nearest: float | None = None
    pairs = 0
    in_contact_subject: dict[str, GeometryPoint] = {}
    in_contact_object: dict[str, GeometryPoint] = {}
    for subject_point in subject_points:
        cx, cy, cz = _cell(subject_point.coordinates_m, search_radius_m)
        for dx, dy, dz in product((-1, 0, 1), repeat=3):
            for object_point in object_grid.get((cx + dx, cy + dy, cz + dz), ()):
                distance = math.dist(subject_point.coordinates_m, object_point.coordinates_m)
                if distance > search_radius_m:
                    continue
                pairs += 1
                nearest = distance if nearest is None else min(nearest, distance)
                if distance <= contact_distance_m:
                    in_contact_subject[subject_point.geometry_id] = subject_point
                    in_contact_object[object_point.geometry_id] = object_point
    return _Contact(
        nearest_m=nearest,
        pairs_within_search_radius=pairs,
        subject_points=tuple(in_contact_subject[key] for key in sorted(in_contact_subject)),
        object_points=tuple(in_contact_object[key] for key in sorted(in_contact_object)),
    )


def _cell(coordinates_m: tuple[float, float, float], size_m: float) -> tuple[int, int, int]:
    x, y, z = coordinates_m
    return (math.floor(x / size_m), math.floor(y / size_m), math.floor(z / size_m))


def _contact_subset(role: str, points: Sequence[GeometryPoint]) -> MeasuredGeometry:
    references = tuple(
        GeometryReference(map_id=point.map_id, geometry_id=point.geometry_id) for point in points
    )
    return MeasuredGeometry(
        role=role,
        point_count=len(references),
        geometry_digest=geometry_set_digest(references),
        geometry_refs=references if len(references) <= _MAX_LISTED_REFERENCES else (),
    )


def _contact_assessment(
    subject: EntityGeometry,
    obj: EntityGeometry,
    contact: _Contact,
    policy: ContactPredicatePolicy,
) -> Assessment:
    """The measurements and conditions every contact predicate shares: are they in contact?"""
    tolerance = policy.contact_tolerance_m
    measurements: list[Quantity] = [
        quantity("bounds_gap", bounds_gap_m(subject.bounds, obj.bounds), "m"),
        quantity("pairs_within_search_radius", float(contact.pairs_within_search_radius), "count"),
    ]
    thresholds = [
        quantity("contact_distance", policy.contact_distance_m, "m"),
        quantity("contact_tolerance", tolerance, "m"),
        quantity("min_contact_points", float(policy.min_contact_points), "count"),
    ]
    checks: list[Check] = []
    subsets: list[MeasuredGeometry] = []
    if contact.nearest_m is None:
        # Nenhum par dentro do raio de busca: a menor distância é maior que o contato mais a
        # tolerância, o que basta para falhar a condição sem inventar um valor exato.
        checks.append(
            Check(
                "nearest_support_distance",
                Verdict.FAILS,
                policy.search_radius_m,
                policy.contact_distance_m,
                tolerance,
            )
        )
        return Assessment(measurements=measurements, thresholds=thresholds, checks=checks)
    measurements.append(quantity("nearest_support_distance", contact.nearest_m, "m"))
    nearest = at_most(
        "nearest_support_distance", contact.nearest_m, policy.contact_distance_m, tolerance
    )
    checks.append(nearest)
    measurements.append(
        quantity("contact_points_subject", float(len(contact.subject_points)), "count")
    )
    measurements.append(
        quantity("contact_points_object", float(len(contact.object_points)), "count")
    )
    if contact.subject_points:
        subsets.append(_contact_subset("subject_contact_points", contact.subject_points))
    if contact.object_points:
        subsets.append(_contact_subset("object_contact_points", contact.object_points))
    if nearest.verdict is Verdict.MEETS:
        for role, count in (
            ("subject", len(contact.subject_points)),
            ("object", len(contact.object_points)),
        ):
            checks.append(_count_check(role, count, policy.min_contact_points))
    return Assessment(
        measurements=measurements, thresholds=thresholds, checks=checks, measured_geometry=subsets
    )


def _count_check(role: str, count: int, minimum: int) -> Check:
    """Decide whether an entity has enough points in contact for a contact area to be measurable."""
    enough = count >= minimum
    return Check(
        f"contact_points_{role}",
        Verdict.MEETS if enough else Verdict.WITHIN_TOLERANCE,
        float(count),
        float(minimum),
        0.0,
        unit="points",
        caveat=None
        if enough
        else EvidenceCaveat(
            kind=EvidenceCaveatKind.UNRELIABLE_GEOMETRY,
            detail=(
                f"only {count} {role} point(s) are within the contact distance: too few to "
                f"measure a contact area (at least {minimum} are needed)"
            ),
        ),
    )


def _assess_touching(
    subject: EntityGeometry,
    obj: EntityGeometry,
    contact: _Contact,
    policy: ContactPredicatePolicy,
    conventions: FrameConventions,
) -> Assessment:
    return _contact_assessment(subject, obj, contact, policy)


def _assess_on_top_of(
    subject: EntityGeometry,
    obj: EntityGeometry,
    contact: _Contact,
    policy: ContactPredicatePolicy,
    conventions: FrameConventions,
) -> Assessment:
    assessment = _contact_assessment(subject, obj, contact, policy)
    up = _declared(conventions.up_axis)
    tolerance = policy.contact_tolerance_m
    subject_low, _ = directed_interval(subject.bounds, up)
    _, object_high = directed_interval(obj.bounds, up)
    height_error = subject_low - object_high
    measurements = [*assessment.measurements, quantity("support_height_error", height_error, "m")]
    thresholds = [
        *assessment.thresholds,
        quantity("support_height_tolerance", policy.support_height_tolerance_m, "m"),
        quantity("support_footprint_fraction", policy.support_footprint_fraction, "ratio"),
    ]
    checks = [
        *assessment.checks,
        at_most(
            "abs_support_height_error",
            abs(height_error),
            policy.support_height_tolerance_m,
            tolerance,
        ),
    ]
    for axis in cross_section_axes(up):
        letter = _AXIS_LETTERS[axis]
        overlap = axis_overlap_m(subject.bounds, obj.bounds, axis)
        extent = subject.extent_m[axis]
        required = policy.support_footprint_fraction * extent
        measurements.append(quantity(f"support_footprint_overlap_{letter}", overlap, "m"))
        if extent > 0.0:
            measurements.append(
                quantity(f"support_footprint_overlap_fraction_{letter}", overlap / extent, "ratio")
            )
        thresholds.append(quantity(f"required_support_footprint_overlap_{letter}", required, "m"))
        checks.append(at_least(f"support_footprint_overlap_{letter}", overlap, required, tolerance))
    return Assessment(
        measurements=measurements,
        thresholds=thresholds,
        checks=checks,
        measured_geometry=assessment.measured_geometry,
    )


def _assess_leaning_against(
    subject: EntityGeometry,
    obj: EntityGeometry,
    contact: _Contact,
    policy: ContactPredicatePolicy,
    conventions: FrameConventions,
) -> Assessment:
    assessment = _contact_assessment(subject, obj, contact, policy)
    up = _declared(conventions.up_axis)
    overlap = axis_overlap_m(subject.bounds, obj.bounds, up.axis_index)
    measurements = [*assessment.measurements, quantity("vertical_overlap", overlap, "m")]
    thresholds = [
        *assessment.thresholds,
        quantity("leaning_min_tilt", policy.leaning_min_tilt_deg, "deg"),
        quantity("leaning_max_tilt", policy.leaning_max_tilt_deg, "deg"),
        quantity("leaning_min_vertical_overlap", policy.leaning_min_vertical_overlap_m, "m"),
        quantity("tilt_tolerance", policy.tilt_tolerance_deg, "deg"),
    ]
    checks = [
        *assessment.checks,
        at_least(
            "vertical_overlap",
            overlap,
            policy.leaning_min_vertical_overlap_m,
            policy.contact_tolerance_m,
        ),
    ]
    missing: list[EvidenceCaveat] = []
    if subject.orientation is None:
        missing.append(
            EvidenceCaveat(
                kind=EvidenceCaveatKind.MISSING_INPUT,
                detail=(
                    "the subject has no derived orientation, so its tilt from the up axis "
                    "cannot be measured"
                ),
            )
        )
    else:
        tilt = _tilt_from_up_deg(subject.orientation.axes[0], up)
        measurements.append(quantity("tilt_from_up", tilt, "deg"))
        checks.append(
            at_least(
                "tilt_from_up", tilt, policy.leaning_min_tilt_deg, policy.tilt_tolerance_deg, "deg"
            )
        )
        checks.append(
            at_most(
                "tilt_from_up", tilt, policy.leaning_max_tilt_deg, policy.tilt_tolerance_deg, "deg"
            )
        )
    return Assessment(
        measurements=measurements,
        thresholds=thresholds,
        checks=checks,
        missing_inputs=missing,
        measured_geometry=assessment.measured_geometry,
    )


def _tilt_from_up_deg(axis: tuple[float, float, float], up: AxisDirection) -> float:
    """The angle in degrees, within ``[0, 90]``, between a dominant axis and the up axis.

    The sign of a principal axis is arbitrary, so only its inclination counts: an axis and its
    opposite lean the same way.
    """
    cosine = abs(sum(a * b for a, b in zip(axis, up.vector, strict=True)))
    return math.degrees(math.acos(min(1.0, cosine)))


def _declared(axis: AxisDirection | None) -> AxisDirection:
    """Return a declared axis; ``require_evaluable`` has already refused an undeclared one."""
    if axis is None:
        raise ValueError("the axis is not declared")
    return axis


_ASSESSORS = {
    RelationPredicate.TOUCHING: _assess_touching,
    RelationPredicate.ON_TOP_OF: _assess_on_top_of,
    RelationPredicate.LEANING_AGAINST: _assess_leaning_against,
}

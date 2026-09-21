"""The 3D support of an entity, and the spatial summaries derived from it.

An entity keeps the *exact* persistent geometry that justifies its spatial existence, as
references into an immutable geometric map. That geometry stays authoritative: centroid, bounds,
extent, statistics and orientation are summaries *derived* from it, expressed in the same map
frame, and they record how they were derived (algorithm, input set, numerical conventions,
filtering) so they can be recomputed and compared. Nothing here copies XYZ into the entity, and a
single centroid is never the only spatial representation.

Recomputing a summary never changes an entity's identity: identity is allocated by the
materialization policy, not by geometry values.

The baseline algorithm, ``entity-geometry-summary-v1``:

* **centroid** -- the arithmetic mean of the support, summed exactly (``math.fsum``);
* **bounds and extent** -- the tight axis-aligned box of the support and its side lengths;
* **statistics** -- the number of points, the volume of the bounds and the density in points per
  cubic meter (``None`` when the box is flat, so a planar support has no density);
* **connectivity** -- points closer than ``connectivity_radius_m`` are linked and the connected
  components of those links are counted; more than one component is a disconnected support. The
  cost grows roughly with the square of the local density, so the policy bounds it with
  ``max_connectivity_points``: a larger support is not linked and says so instead of taking
  unbounded time;
* **orientation** -- optional. The principal axes of the covariance of the support, computed only
  when the policy asks for it and only when they are well defined: enough points and three
  clearly distinct variances. Otherwise the orientation is absent and the diagnostic says why.

The support is never filtered or downsampled; that is recorded rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import product

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometryPoint,
    GeometryReference,
    GeometrySource,
    MapId,
)
from contextmap.ingestion import FrameId
from contextmap.semantic_mapping._checks import require_canonical, require_present
from contextmap.shared import Vector3

GEOMETRY_SUMMARY_ALGORITHM_ID = "entity-geometry-summary-v1"

DEFAULT_MAX_CONNECTIVITY_POINTS = 10_000
"""Largest support whose connected components are counted unless the policy says otherwise.

The pairwise comparison inside each 27-cell neighborhood is roughly quadratic in the local density.
One synthetic run in pure Python (random points in a 2 x 2 x 1 m box, 0.3 m radius) took about 4.5 s
at 10,000 points and 42 s at 30,000, and no real support has been measured: the limit keeps one
entity in the order of seconds. It is a cost guard, not a scientific threshold, so a profile that
accepts the cost raises it explicitly, and the value enters the configuration fingerprint.
"""
"""Versioned identity of the baseline spatial summary described in this module."""

_ORIENTATION_METHOD = "pca-covariance-jacobi-v1"
_NUMERICAL_CONVENTIONS = (
    "float64; centroid=arithmetic mean (fsum), clamped into the bounds; bounds=min/max; "
    "extent=max-min; volume=extent product; covariance normalized by n; axes sorted by "
    "descending variance, sign fixed so the largest component is positive, third axis is the "
    "cross product of the first two"
)
_NO_FILTERING = "none: the whole support is used, nothing is filtered or downsampled"

_TOLERANCE = 1e-9
_UNIT_TOLERANCE = 1e-6


class EmptyGeometrySupportError(ValueError):
    """Raised when an entity would have no 3D support."""


class GeometryResolutionError(Exception):
    """Raised when geometry references cannot be resolved against the map that owns them."""


class GeometryDiagnosticKind(Enum):
    """Why a spatial summary should be read with care.

    Attributes:
        SPARSE_SUPPORT: The support has fewer points than the policy considers reliable.
        DISCONNECTED_SUPPORT: The support falls apart into several connected components.
        DEGENERATE_EXTENT: The bounds are flat on some axis, so volume and density are not
            meaningful.
        ORIENTATION_NOT_JUSTIFIED: An orientation was requested but its principal axes are not
            well defined for this support.
        CONNECTIVITY_NOT_COMPUTED: The support has more points than the policy's connectivity
            limit, so its connected components were not counted; it may or may not be connected.
    """

    SPARSE_SUPPORT = "sparse_support"
    DISCONNECTED_SUPPORT = "disconnected_support"
    DEGENERATE_EXTENT = "degenerate_extent"
    ORIENTATION_NOT_JUSTIFIED = "orientation_not_justified"
    CONNECTIVITY_NOT_COMPUTED = "connectivity_not_computed"


@dataclass(frozen=True, kw_only=True)
class GeometryDiagnostic:
    """A caveat about a spatial summary, stated instead of hidden.

    Attributes:
        kind: What kind of caveat this is.
        detail: A deterministic, human-readable explanation.
    """

    kind: GeometryDiagnosticKind
    detail: str

    def __post_init__(self) -> None:
        """Require an explanation.

        Raises:
            ValueError: If the detail is empty.
        """
        require_present(self, "detail")


@dataclass(frozen=True, kw_only=True)
class SupportStatistics:
    """Size, density and connectivity of the support.

    Attributes:
        point_count: Geometry elements in the support.
        volume_m3: Volume of the bounds, in cubic meters.
        density_per_m3: ``point_count`` over ``volume_m3``; ``None`` when the bounds are flat.
        component_count: Connected components at the policy's connectivity radius; ``None`` when
            the support is above the policy's connectivity limit and they were not counted,
            which is not the same as one component.
        largest_component_fraction: Share of the points in the largest component, in ``(0, 1]``;
            ``None`` exactly when ``component_count`` is.
    """

    point_count: int
    volume_m3: float
    density_per_m3: float | None
    component_count: int | None
    largest_component_fraction: float | None

    def __post_init__(self) -> None:
        """Validate that the figures are coherent with each other.

        Raises:
            ValueError: If the count or components are not positive, only one of the two
                connectivity figures is present, a figure is not finite or negative, the density
                does not match the volume, or the largest component fraction is outside
                ``(0, 1]``.
        """
        if (self.component_count is None) != (self.largest_component_fraction is None):
            raise ValueError(
                "component_count and largest_component_fraction must be both present or both None"
            )
        if self.point_count < 1 or (self.component_count is not None and self.component_count < 1):
            raise ValueError("point_count and component_count must be at least 1")
        if self.component_count is not None and self.component_count > self.point_count:
            raise ValueError("component_count cannot exceed point_count")
        if not (math.isfinite(self.volume_m3) and self.volume_m3 >= 0.0):
            raise ValueError(f"volume_m3 must be finite and not negative, got {self.volume_m3!r}")
        if self.volume_m3 == 0.0:
            if self.density_per_m3 is not None:
                raise ValueError("density_per_m3 must be None when the volume is zero")
        elif self.density_per_m3 is None or not math.isclose(
            self.density_per_m3, self.point_count / self.volume_m3, rel_tol=1e-9
        ):
            raise ValueError("density_per_m3 must equal point_count over volume_m3")
        if self.largest_component_fraction is not None and not (
            0.0 < self.largest_component_fraction <= 1.0
        ):
            raise ValueError(
                f"largest_component_fraction must be within (0, 1], got "
                f"{self.largest_component_fraction!r}"
            )


@dataclass(frozen=True, kw_only=True)
class EntityOrientation:
    """The principal axes of an entity's support, when they are well defined.

    Attributes:
        axes: Three unit axes in the map frame, orthogonal and right-handed, sorted by
            descending variance.
        variances_m2: The variance of the support along each axis, in square meters,
            non-increasing.
        method: The versioned method that derived the axes.
    """

    axes: tuple[Vector3, Vector3, Vector3]
    variances_m2: Vector3
    method: str

    def __post_init__(self) -> None:
        """Validate that the axes are an orthonormal right-handed frame and the variances ordered.

        Raises:
            ValueError: If the method is empty, an axis is not a finite unit vector, the axes
                are not orthogonal or right-handed, or the variances are negative, not finite
                or not sorted in descending order.
        """
        require_present(self, "method")
        first, second, third = self.axes
        for axis in self.axes:
            if not all(math.isfinite(value) for value in axis) or not math.isclose(
                _norm(axis), 1.0, abs_tol=_UNIT_TOLERANCE
            ):
                raise ValueError(f"orientation axes must be finite unit vectors, got {axis!r}")
        if any(
            abs(_dot(left, right)) > _UNIT_TOLERANCE
            for left, right in ((first, second), (first, third), (second, third))
        ):
            raise ValueError("orientation axes must be orthogonal")
        if _dot(_cross(first, second), third) < 1.0 - _UNIT_TOLERANCE:
            raise ValueError("orientation axes must be right-handed")
        if not all(math.isfinite(value) and value >= 0.0 for value in self.variances_m2):
            raise ValueError(
                f"variances must be finite and not negative, got {self.variances_m2!r}"
            )
        if not (self.variances_m2[0] >= self.variances_m2[1] >= self.variances_m2[2]):
            raise ValueError("variances must be sorted in descending order")


@dataclass(frozen=True, kw_only=True)
class SpatialSummaryProvenance:
    """How the derived spatial summaries of an entity were computed.

    Attributes:
        algorithm_id: Versioned algorithm.
        map_frame: The frame every derived value is expressed in.
        input_geometry_count: Geometry elements the summaries were computed from.
        input_geometry_digest: Digest of the exact set of geometry identities used.
        numerical_conventions: The numerical conventions the values follow.
        filtering: Whether any filtering or downsampling was applied, stated explicitly.
        configuration_fingerprint: Hash of the policy the summaries were computed under.
    """

    algorithm_id: str
    map_frame: FrameId
    input_geometry_count: int
    input_geometry_digest: str
    numerical_conventions: str
    filtering: str
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        """Require every field to say something.

        Raises:
            ValueError: If a text field is empty or the input set is empty.
        """
        require_present(
            self,
            "algorithm_id",
            "map_frame",
            "input_geometry_digest",
            "numerical_conventions",
            "filtering",
            "configuration_fingerprint",
        )
        if self.input_geometry_count < 1:
            raise ValueError("input_geometry_count must be at least 1")


@dataclass(frozen=True, kw_only=True)
class OrientationPolicy:
    """When an orientation is mathematically justified.

    There are no defaults: the thresholds are scientific choices that a profile declares.

    Attributes:
        min_points: Fewest points the support needs for its axes to mean anything.
        min_variance_ratio: How much larger than the next each variance must be, so the axes
            are distinct; ``1`` accepts any ordering and larger values demand a clearer one.
    """

    min_points: int
    min_variance_ratio: float

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ValueError: If ``min_points`` is below three or the ratio is not finite and above
                one.
        """
        if self.min_points < 3:
            raise ValueError(f"min_points must be at least 3, got {self.min_points}")
        if not (math.isfinite(self.min_variance_ratio) and self.min_variance_ratio > 1.0):
            raise ValueError(
                f"min_variance_ratio must be finite and above 1, got {self.min_variance_ratio!r}"
            )


@dataclass(frozen=True, kw_only=True)
class GeometrySummaryPolicy:
    """Explicit configuration of the baseline spatial summary.

    There are no defaults: the thresholds are scientific choices that a profile declares.

    Attributes:
        sparse_point_threshold: Supports with fewer points are flagged as sparse.
        connectivity_radius_m: Distance under which two points are linked, in meters.
        orientation: When to derive an orientation, or ``None`` to never derive one.
        max_connectivity_points: Largest support whose connected components are counted. A
            larger support is not linked, so its components are ``None`` and a diagnostic says
            so; it bounds the cost of the connectivity check (see
            :data:`DEFAULT_MAX_CONNECTIVITY_POINTS`).
    """

    sparse_point_threshold: int
    connectivity_radius_m: float
    orientation: OrientationPolicy | None = None
    max_connectivity_points: int = DEFAULT_MAX_CONNECTIVITY_POINTS

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ValueError: If the sparse threshold or the connectivity limit is below one or the
                radius is not finite and positive.
        """
        if self.sparse_point_threshold < 1:
            raise ValueError(
                f"sparse_point_threshold must be at least 1, got {self.sparse_point_threshold}"
            )
        if self.max_connectivity_points < 1:
            raise ValueError(
                f"max_connectivity_points must be at least 1, got {self.max_connectivity_points}"
            )
        if not (math.isfinite(self.connectivity_radius_m) and self.connectivity_radius_m > 0.0):
            raise ValueError(
                f"connectivity_radius_m must be finite and positive, got "
                f"{self.connectivity_radius_m!r}"
            )

    def fingerprint(self) -> str:
        """Hash the algorithm identity and thresholds, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "algorithm_id": GEOMETRY_SUMMARY_ALGORITHM_ID,
                "sparse_point_threshold": self.sparse_point_threshold,
                "connectivity_radius_m": self.connectivity_radius_m,
                "max_connectivity_points": self.max_connectivity_points,
                "orientation": None
                if self.orientation is None
                else {
                    "min_points": self.orientation.min_points,
                    "min_variance_ratio": self.orientation.min_variance_ratio,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def geometry_set_digest(references: Sequence[GeometryReference]) -> str:
    """Digest the exact set of geometry identities of a support.

    Args:
        references: The support, sorted by ``geometry_id``.

    Returns:
        ``sha256:`` followed by the digest of the identities, so a summary can be tied to the
        exact set of geometry it was computed from.
    """
    joined = "\n".join(reference.geometry_id for reference in references)
    return f"sha256:{hashlib.sha256(joined.encode()).hexdigest()}"


@dataclass(frozen=True, kw_only=True)
class EntityGeometry:
    """The persistent geometry that supports one entity, with reproducible summaries.

    Attributes:
        geometry_refs: References to the supporting geometry, sorted by ``geometry_id``,
            unique and never empty: an entity without real 3D support is not an entity. This
            is the authoritative support; every other field is derived from it.
        map_frame: The frame of the map the geometry belongs to.
        centroid_m: Centroid of the support, in meters, in ``map_frame``.
        bounds: Tight axis-aligned box of the support, in ``map_frame``.
        extent_m: Side lengths of ``bounds``, in meters.
        statistics: Size, density and connectivity of the support.
        summary: How the summaries were derived.
        orientation: Principal axes of the support, only when the policy asked for them and they
            are well defined.
        diagnostics: Caveats about the summaries, sorted by kind and unique.
    """

    geometry_refs: tuple[GeometryReference, ...]
    map_frame: FrameId
    centroid_m: Vector3
    bounds: Bounds3D
    extent_m: Vector3
    statistics: SupportStatistics
    summary: SpatialSummaryProvenance
    orientation: EntityOrientation | None = None
    diagnostics: tuple[GeometryDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        """Validate that the support is present and every summary agrees with it.

        Raises:
            EmptyGeometrySupportError: If the support is empty.
            ValueError: If the frame is empty, the support is not sorted by geometry and unique
                or spans more than one map, a summary is not in ``map_frame``, the centroid is not
                finite or lies outside the bounds, the extent is not the bounds' size, the
                statistics or the provenance disagree with the support, or the diagnostics
                contradict the statistics.
        """
        require_present(self, "map_frame")
        if not self.geometry_refs:
            raise EmptyGeometrySupportError(
                "geometry_refs must not be empty: an entity needs 3D support"
            )
        require_canonical(
            "geometry_refs",
            self.geometry_refs,
            lambda reference: (reference.geometry_id,),
            detail="by geometry_id ",
        )
        maps = {reference.map_id for reference in self.geometry_refs}
        if len(maps) > 1:
            raise ValueError(f"geometry_refs must belong to one map, got {sorted(maps)!r}")
        self._require_summaries_in_frame()
        self._require_summaries_match_support()
        self._require_diagnostics_match()

    @property
    def geometric_map_id(self) -> MapId:
        """The immutable geometric map the support belongs to."""
        return self.geometry_refs[0].map_id

    def diagnostic_kinds(self) -> frozenset[GeometryDiagnosticKind]:
        """The kinds of caveat attached to the summaries."""
        return frozenset(item.kind for item in self.diagnostics)

    def _require_summaries_in_frame(self) -> None:
        if self.bounds.frame_id != self.map_frame:
            raise ValueError(
                f"bounds are expressed in {self.bounds.frame_id!r}, not the map frame "
                f"{self.map_frame!r}"
            )
        if self.summary.map_frame != self.map_frame:
            raise ValueError(
                f"the summary was derived in {self.summary.map_frame!r}, not the map frame "
                f"{self.map_frame!r}"
            )

    def _require_summaries_match_support(self) -> None:
        if not all(math.isfinite(value) for value in self.centroid_m):
            raise ValueError(f"centroid_m must be finite, got {self.centroid_m!r}")
        if not self.bounds.contains(self.centroid_m, frame_id=self.bounds.frame_id):
            raise ValueError(
                f"centroid_m {self.centroid_m!r} must lie inside the bounds "
                f"{self.bounds.minimum_m!r}..{self.bounds.maximum_m!r}"
            )
        expected = tuple(
            high - low
            for low, high in zip(self.bounds.minimum_m, self.bounds.maximum_m, strict=True)
        )
        if any(
            not math.isclose(got, want, abs_tol=_TOLERANCE)
            for got, want in zip(self.extent_m, expected, strict=True)
        ):
            raise ValueError(
                f"extent_m {self.extent_m!r} must be the size of the bounds, {expected!r}"
            )
        count = len(self.geometry_refs)
        if self.statistics.point_count != count or self.summary.input_geometry_count != count:
            raise ValueError(
                f"statistics and summary must describe the {count} geometry references, got "
                f"{self.statistics.point_count} and {self.summary.input_geometry_count}"
            )
        if self.summary.input_geometry_digest != geometry_set_digest(self.geometry_refs):
            raise ValueError("the summary was not derived from exactly these geometry references")
        volume = math.prod(self.extent_m)
        if not math.isclose(self.statistics.volume_m3, volume, rel_tol=1e-9, abs_tol=_TOLERANCE):
            raise ValueError(
                f"volume_m3 {self.statistics.volume_m3!r} must be the volume of the bounds, "
                f"{volume!r}"
            )

    def _require_diagnostics_match(self) -> None:
        require_canonical(
            "diagnostics", self.diagnostics, lambda item: (item.kind.value,), detail="by kind "
        )
        kinds = self.diagnostic_kinds()
        components = self.statistics.component_count
        disconnected = components is not None and components > 1
        if disconnected != (GeometryDiagnosticKind.DISCONNECTED_SUPPORT in kinds):
            raise ValueError(
                "the disconnected_support diagnostic must be present exactly when the support "
                "has more than one connected component"
            )
        if (components is None) != (GeometryDiagnosticKind.CONNECTIVITY_NOT_COMPUTED in kinds):
            raise ValueError(
                "the connectivity_not_computed diagnostic must be present exactly when the "
                "connected components were not counted"
            )
        flat = any(side == 0.0 for side in self.extent_m)
        if flat != (GeometryDiagnosticKind.DEGENERATE_EXTENT in kinds):
            raise ValueError(
                "the degenerate_extent diagnostic must be present exactly when the bounds are "
                "flat on some axis"
            )
        if (
            self.orientation is not None
            and GeometryDiagnosticKind.ORIENTATION_NOT_JUSTIFIED in kinds
        ):
            raise ValueError(
                "an orientation cannot coexist with an orientation_not_justified diagnostic"
            )


def resolve_geometry(
    references: Sequence[GeometryReference], *, source: GeometrySource
) -> tuple[GeometryPoint, ...]:
    """Resolve references to their authoritative geometry, in the order given.

    Args:
        references: References into the map ``source`` serves.
        source: The read boundary of the geometric map.

    Returns:
        The geometry of every reference, from the persisted map.

    Raises:
        GeometryResolutionError: If a reference belongs to another map, does not exist in this
            one, or resolves to geometry expressed in a frame other than the map's.
    """
    geometric_map = source.geometric_map
    foreign = sorted({reference.map_id for reference in references} - {geometric_map.map_id})
    if foreign:
        raise GeometryResolutionError(
            f"references belong to map(s) {foreign!r}, but the source serves "
            f"{geometric_map.map_id!r}"
        )
    points: list[GeometryPoint] = []
    missing: list[str] = []
    for reference in references:
        try:
            points.append(source.get(reference))
        except KeyError:
            missing.append(reference.geometry_id)
    if missing:
        raise GeometryResolutionError(
            f"{len(missing)} geometry reference(s) do not exist in map {geometric_map.map_id!r}: "
            f"{missing[:3]!r}"
        )
    wrong_frame = sorted({point.map_frame for point in points} - {geometric_map.frame_id})
    if wrong_frame:
        raise GeometryResolutionError(
            f"geometry is expressed in {wrong_frame!r}, not the map frame "
            f"{geometric_map.frame_id!r}"
        )
    return tuple(points)


def summarize_geometry(
    references: Sequence[GeometryReference],
    *,
    source: GeometrySource,
    policy: GeometrySummaryPolicy,
) -> EntityGeometry:
    """Build the geometry of an entity from its supporting references.

    Args:
        references: The support, in any order; it is sorted into canonical order.
        source: The read boundary of the geometric map the references belong to.
        policy: The thresholds of the baseline summary.

    Returns:
        The entity geometry, with every derived summary in the map frame.

    Raises:
        EmptyGeometrySupportError: If there is no support.
        GeometryResolutionError: If a reference cannot be resolved against the map.
        ValueError: If a reference is repeated.
    """
    if not references:
        raise EmptyGeometrySupportError("an entity needs at least one geometry reference")
    ordered = tuple(sorted(references, key=lambda reference: reference.geometry_id))
    points = resolve_geometry(ordered, source=source)
    frame = source.geometric_map.frame_id
    coordinates = [point.coordinates_m for point in points]
    bounds = Bounds3D.enclosing(coordinates, frame_id=frame)
    centroid = _centroid(coordinates, bounds)
    extent = _vector(
        high - low for low, high in zip(bounds.minimum_m, bounds.maximum_m, strict=True)
    )
    volume = math.prod(extent)
    # O custo da conectividade cresce ~quadraticamente com a densidade local: acima do limite da
    # política os componentes não são contados, e o diagnóstico diz isso em vez de demorar sem teto.
    sizes = (
        None
        if len(points) > policy.max_connectivity_points
        else _component_sizes(coordinates, policy.connectivity_radius_m)
    )
    statistics = SupportStatistics(
        point_count=len(points),
        volume_m3=volume,
        density_per_m3=None if volume == 0.0 else len(points) / volume,
        component_count=None if sizes is None else len(sizes),
        largest_component_fraction=None if sizes is None else max(sizes) / len(points),
    )
    diagnostics = {item.kind: item for item in _diagnostics(statistics, extent, policy)}
    orientation = None
    if policy.orientation is not None:
        orientation, reason = _orientation(coordinates, centroid, policy.orientation)
        if orientation is None and reason is not None:
            item = GeometryDiagnostic(
                kind=GeometryDiagnosticKind.ORIENTATION_NOT_JUSTIFIED, detail=reason
            )
            diagnostics[item.kind] = item
    return EntityGeometry(
        geometry_refs=ordered,
        map_frame=frame,
        centroid_m=centroid,
        bounds=bounds,
        extent_m=extent,
        statistics=statistics,
        summary=SpatialSummaryProvenance(
            algorithm_id=GEOMETRY_SUMMARY_ALGORITHM_ID,
            map_frame=frame,
            input_geometry_count=len(ordered),
            input_geometry_digest=geometry_set_digest(ordered),
            numerical_conventions=_NUMERICAL_CONVENTIONS,
            filtering=_NO_FILTERING,
            configuration_fingerprint=policy.fingerprint(),
        ),
        orientation=orientation,
        diagnostics=tuple(sorted(diagnostics.values(), key=lambda item: item.kind.value)),
    )


def verify_geometry_summary(
    geometry: EntityGeometry, *, source: GeometrySource, policy: GeometrySummaryPolicy
) -> list[str]:
    """Recompute the summaries of an entity from the persistent geometry and compare.

    Args:
        geometry: The geometry of an entity.
        source: The read boundary of the geometric map it references.
        policy: The policy the summaries are expected to follow.

    Returns:
        Human-readable problems; empty means the summaries are the ones the persistent
        geometry and the policy produce. An unresolvable support is reported, not raised.
    """
    try:
        expected = summarize_geometry(geometry.geometry_refs, source=source, policy=policy)
    except GeometryResolutionError as error:
        return [f"the support cannot be resolved: {error}"]
    problems: list[str] = []
    if geometry.summary.configuration_fingerprint != expected.summary.configuration_fingerprint:
        problems.append("the summary was computed under a different policy")
    _compare_vector("centroid_m", geometry.centroid_m, expected.centroid_m, problems)
    _compare_vector(
        "bounds.minimum_m", geometry.bounds.minimum_m, expected.bounds.minimum_m, problems
    )
    _compare_vector(
        "bounds.maximum_m", geometry.bounds.maximum_m, expected.bounds.maximum_m, problems
    )
    _compare_vector("extent_m", geometry.extent_m, expected.extent_m, problems)
    if geometry.statistics.point_count != expected.statistics.point_count:
        problems.append("statistics.point_count differs from the persistent geometry")
    if geometry.statistics.component_count != expected.statistics.component_count:
        problems.append("statistics.component_count differs from the persistent geometry")
    if (geometry.orientation is None) != (expected.orientation is None):
        problems.append("the presence of an orientation differs from the persistent geometry")
    elif geometry.orientation is not None and expected.orientation is not None:
        _compare_vector(
            "orientation.variances_m2",
            geometry.orientation.variances_m2,
            expected.orientation.variances_m2,
            problems,
        )
    if geometry.diagnostic_kinds() != expected.diagnostic_kinds():
        problems.append("the diagnostics differ from the persistent geometry")
    return problems


def _compare_vector(name: str, got: Vector3, want: Vector3, problems: list[str]) -> None:
    if any(not math.isclose(a, b, abs_tol=_TOLERANCE) for a, b in zip(got, want, strict=True)):
        problems.append(f"{name} {got!r} differs from the persistent geometry, {want!r}")


def _vector(values: Iterable[float]) -> Vector3:
    x, y, z = values
    return (x, y, z)


def _centroid(coordinates: Sequence[Vector3], bounds: Bounds3D) -> Vector3:
    count = len(coordinates)
    mean = tuple(math.fsum(point[axis] for point in coordinates) / count for axis in range(3))
    # A média está matematicamente dentro da caixa; o clamp só corrige o erro de arredondamento.
    x, y, z = (
        min(max(value, low), high)
        for value, low, high in zip(mean, bounds.minimum_m, bounds.maximum_m, strict=True)
    )
    return (x, y, z)


def _diagnostics(
    statistics: SupportStatistics, extent: Vector3, policy: GeometrySummaryPolicy
) -> list[GeometryDiagnostic]:
    found: list[GeometryDiagnostic] = []
    if statistics.point_count < policy.sparse_point_threshold:
        found.append(
            GeometryDiagnostic(
                kind=GeometryDiagnosticKind.SPARSE_SUPPORT,
                detail=(
                    f"{statistics.point_count} points, fewer than the "
                    f"{policy.sparse_point_threshold} the policy considers reliable"
                ),
            )
        )
    if statistics.component_count is None:
        found.append(
            GeometryDiagnostic(
                kind=GeometryDiagnosticKind.CONNECTIVITY_NOT_COMPUTED,
                detail=(
                    f"{statistics.point_count} points, above the connectivity limit of "
                    f"{policy.max_connectivity_points} the policy accepts: the connected "
                    f"components were not counted"
                ),
            )
        )
    elif statistics.component_count > 1:
        found.append(
            GeometryDiagnostic(
                kind=GeometryDiagnosticKind.DISCONNECTED_SUPPORT,
                detail=(
                    f"{statistics.component_count} connected components at "
                    f"{policy.connectivity_radius_m} m; the largest holds "
                    f"{statistics.largest_component_fraction:.3f} of the points"
                ),
            )
        )
    if any(side == 0.0 for side in extent):
        found.append(
            GeometryDiagnostic(
                kind=GeometryDiagnosticKind.DEGENERATE_EXTENT,
                detail=(
                    "the bounds are flat on at least one axis: volume and density are not "
                    "meaningful"
                ),
            )
        )
    return found


def _component_sizes(coordinates: Sequence[Vector3], radius_m: float) -> list[int]:
    """Size of each connected component when points closer than ``radius_m`` are linked.

    Points are bucketed in a grid of cell size ``radius_m``, so only the 27 neighboring cells
    of a point can hold a neighbor within the radius.
    """
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (x, y, z) in enumerate(coordinates):
        cells[
            (math.floor(x / radius_m), math.floor(y / radius_m), math.floor(z / radius_m))
        ].append(index)
    parent = list(range(len(coordinates)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    radius_squared = radius_m * radius_m
    for (cx, cy, cz), members in cells.items():
        for dx, dy, dz in product((-1, 0, 1), repeat=3):
            for other in cells.get((cx + dx, cy + dy, cz + dz), ()):
                for member in members:
                    if (
                        other > member
                        and _squared_distance(coordinates[member], coordinates[other])
                        <= radius_squared
                    ):
                        parent[find(other)] = find(member)
    sizes: dict[int, int] = defaultdict(int)
    for index in range(len(coordinates)):
        sizes[find(index)] += 1
    return sorted(sizes.values(), reverse=True)


def _orientation(
    coordinates: Sequence[Vector3], centroid: Vector3, policy: OrientationPolicy
) -> tuple[EntityOrientation | None, str | None]:
    """Derive the principal axes, or say why they are not well defined."""
    count = len(coordinates)
    if count < policy.min_points:
        return None, f"{count} points, fewer than the {policy.min_points} an orientation needs"
    covariance = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(row, 3):
            value = (
                math.fsum(
                    (point[row] - centroid[row]) * (point[column] - centroid[column])
                    for point in coordinates
                )
                / count
            )
            covariance[row][column] = covariance[column][row] = value
    variances, vectors = _symmetric_eigen(covariance)
    order = sorted(range(3), key=lambda index: -variances[index])
    ordered_variances = [max(variances[index], 0.0) for index in order]
    for larger, smaller in ((0, 1), (1, 2)):
        if not _distinct(ordered_variances[larger], ordered_variances[smaller], policy):
            return None, (
                f"variances {tuple(ordered_variances)!r} are not distinct enough: the ratio "
                f"between consecutive variances must be at least {policy.min_variance_ratio}"
            )
    first = _canonical_sign(vectors[order[0]])
    second = _canonical_sign(vectors[order[1]])
    third = _cross(first, second)
    return (
        EntityOrientation(
            axes=(first, second, third),
            variances_m2=_vector(ordered_variances),
            method=_ORIENTATION_METHOD,
        ),
        None,
    )


def _distinct(larger: float, smaller: float, policy: OrientationPolicy) -> bool:
    if larger <= 0.0:
        return False
    return smaller == 0.0 or larger / smaller >= policy.min_variance_ratio


def _canonical_sign(axis: Vector3) -> Vector3:
    """Flip an axis so its largest component is positive, so equivalent axes encode identically."""
    dominant = max(axis, key=abs)
    return axis if dominant >= 0.0 else (-axis[0], -axis[1], -axis[2])


def _symmetric_eigen(matrix: list[list[float]]) -> tuple[list[float], list[Vector3]]:
    """Eigenvalues and unit eigenvectors of a symmetric 3x3 matrix, by cyclic Jacobi rotations."""
    a = [row[:] for row in matrix]
    v = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
    scale = sum(a[i][i] ** 2 for i in range(3)) or 1.0
    for _ in range(64):
        if a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2 <= 1e-32 * scale:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if a[p][q] == 0.0:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            other = 3 - p - q
            app, aqq, apq = a[p][p], a[q][q], a[p][q]
            a[p][p] = app - t * apq
            a[q][q] = aqq + t * apq
            a[p][q] = a[q][p] = 0.0
            aop, aoq = a[other][p], a[other][q]
            a[other][p] = a[p][other] = c * aop - s * aoq
            a[other][q] = a[q][other] = s * aop + c * aoq
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
    eigenvalues = [a[i][i] for i in range(3)]
    eigenvectors = [(v[0][i], v[1][i], v[2][i]) for i in range(3)]
    return eigenvalues, eigenvectors


def _squared_distance(left: Vector3, right: Vector3) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))

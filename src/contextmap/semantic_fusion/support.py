"""Deterministic construction of :class:`FusionSupport` over persistent geometry.

Semantic Fusion needs a spatial unit to accumulate evidence over. Fusing at every
geometry point would create millions of redundant states, and creating entities
here would collapse the boundary with Semantic Mapping. :func:`build_fusion_supports`
therefore groups the spatial observations that see *overlapping geometry* and asserts
nothing else: no label, no class and no object identity.

The baseline policy, ``geometry-jaccard-support-v1``:

* **overlap measure** -- the Jaccard index of the geometry sets two observations
  reference, ``|A and B| / |A or B|``. Only the identity of the geometry counts, never a
  label, a claim or a score;
* **merge** -- two observations with an overlap of at least ``min_overlap`` (inclusive)
  are linked, and the connected components of those links are the supports. The
  linkage is transitive, so a chain of partly overlapping views can join views that do
  not overlap each other directly;
* **split** -- an observation belongs to exactly one support and a support is never split;
  disjoint geometry is never merged, so identical claims do not join separate places;
* **nesting** -- Jaccard is symmetric, so a small region nested in a large one has a low
  overlap and stays a separate support. A false merge would create spurious conflicts
  between, say, a door and its handle, whereas a false split only accumulates less
  evidence, which Entity Resolution can later refine;
* **minimum support** -- an observation with fewer than ``min_geometry_count`` geometry
  elements is not assigned to any support and is reported in
  :attr:`FusionSupportBuild.excluded`, never dropped silently;
* **disconnected geometry** -- the geometry of one region is not analysed for
  connectivity: a region whose points are far apart stays in one support and its bounds
  cover them all;
* **ordering and identity** -- observations are ordered by identity, supports by their
  first observation, and ``support-000001``, ``support-000002``, ... follow that order,
  so identical input and configuration always rebuild identical supports.

Pairwise overlap uses one bit set per observation over the geometry that at least two
observations reference, so the cost is quadratic in the number of observations and
linear in the number of shared geometry elements per pair.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise

from contextmap.geometric_mapping import Bounds3D, GeometryId, GeometryReference, GeometrySource
from contextmap.ingestion import FrameId, SourceObservationId
from contextmap.semantic_fusion.models import (
    FusionSupport,
    FusionSupportId,
    FusionSupportProvenance,
    _require_canonical,
)
from contextmap.sensor_association import SpatialObservation, SpatialObservationId
from contextmap.shared import SourceTimestamp, Vector3
from contextmap.state_estimation import TimeBounds

GEOMETRY_OVERLAP_SUPPORT_POLICY_ID = "geometry-jaccard-support-v1"
"""Versioned identity of the baseline support policy described in this module."""


@dataclass(frozen=True, kw_only=True)
class GeometryOverlapSupportPolicy:
    """Explicit configuration of the baseline support policy.

    There are no defaults: the thresholds are scientific choices that a profile declares.

    Attributes:
        min_geometry_count: Fewest geometry elements an observation needs to take part.
        min_overlap: Jaccard overlap, in ``(0, 1]``, at which two observations are linked;
            the threshold itself links them.
    """

    min_geometry_count: int
    min_overlap: float

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ValueError: If ``min_geometry_count`` is below one or ``min_overlap`` is not in
                ``(0, 1]``.
        """
        if self.min_geometry_count < 1:
            raise ValueError(
                f"min_geometry_count must be at least 1, got {self.min_geometry_count}"
            )
        if not (math.isfinite(self.min_overlap) and 0.0 < self.min_overlap <= 1.0):
            raise ValueError(f"min_overlap must be within (0, 1], got {self.min_overlap!r}")

    def fingerprint(self) -> str:
        """Hash the policy identity and thresholds, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "policy_id": GEOMETRY_OVERLAP_SUPPORT_POLICY_ID,
                "min_geometry_count": self.min_geometry_count,
                "min_overlap": self.min_overlap,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


@dataclass(frozen=True, kw_only=True)
class ExcludedObservation:
    """A spatial observation that did not take part in any support, and why.

    Attributes:
        spatial_observation_id: The excluded observation.
        geometry_count: How many geometry elements it references.
        minimum_geometry_count: The policy minimum it fell short of.
    """

    spatial_observation_id: SpatialObservationId
    geometry_count: int
    minimum_geometry_count: int


@dataclass(frozen=True, kw_only=True)
class FusionSupportBuild:
    """The supports built from a set of spatial observations, and what was left out.

    Every input observation is in exactly one support or in ``excluded``.

    Attributes:
        supports: The supports, sorted by identity.
        excluded: Observations with too little geometry, sorted by observation.
    """

    supports: tuple[FusionSupport, ...]
    excluded: tuple[ExcludedObservation, ...]

    def __post_init__(self) -> None:
        """Validate the ordering and that no observation is placed twice.

        Raises:
            ValueError: If a collection is not sorted and unique, an observation is in more
                than one support, or an excluded observation is also in a support.
        """
        _require_canonical("supports", self.supports, lambda item: (item.fusion_support_id,))
        _require_canonical("excluded", self.excluded, lambda item: (item.spatial_observation_id,))
        placed: set[SpatialObservationId] = set()
        for support in self.supports:
            for observation_id in support.spatial_observation_ids:
                if observation_id in placed:
                    raise ValueError(
                        f"spatial observation {observation_id!r} is in more than one support"
                    )
                placed.add(observation_id)
        for item in self.excluded:
            if item.spatial_observation_id in placed:
                raise ValueError(
                    f"excluded spatial observation {item.spatial_observation_id!r} is also in "
                    f"a support"
                )

    def support_id_of(self) -> dict[SpatialObservationId, FusionSupportId]:
        """Index the support each supported observation belongs to.

        Returns:
            A mapping from spatial observation to its support; excluded observations are
            absent.
        """
        return {
            observation_id: support.fusion_support_id
            for support in self.supports
            for observation_id in support.spatial_observation_ids
        }


def build_fusion_supports(
    observations: Iterable[SpatialObservation],
    *,
    geometry: GeometrySource,
    acquisition_timestamps: Mapping[SourceObservationId, SourceTimestamp],
    policy: GeometryOverlapSupportPolicy,
    code_version: str | None = None,
) -> FusionSupportBuild:
    """Group spatial observations over overlapping geometry into fusion supports.

    Args:
        observations: The spatial observations to group, in any order.
        geometry: The read boundary of the map the observations reference; used to resolve
            each geometry element once for the bounds and centroid of a support.
        acquisition_timestamps: When each physical observation was acquired, in one clock
            domain.
        policy: The thresholds of the baseline policy.
        code_version: Code revision to record in the provenance, when known.

    Returns:
        The supports and the observations left out. The result does not depend on the order
        of ``observations``.

    Raises:
        ValueError: If an observation is repeated or belongs to another map than
            ``geometry``, references geometry that the map does not contain, a supported
            physical observation has no acquisition timestamp, or the timestamps span more
            than one clock domain.
    """
    ordered = sorted(observations, key=lambda item: item.spatial_observation_id)
    map_id = geometry.geometric_map.map_id
    for previous, current in pairwise(ordered):
        if previous.spatial_observation_id == current.spatial_observation_id:
            raise ValueError(f"duplicate spatial observation {current.spatial_observation_id!r}")
    for observation in ordered:
        if observation.provenance.geometric_map_id != map_id:
            raise ValueError(
                f"spatial observation {observation.spatial_observation_id!r} references map "
                f"{observation.provenance.geometric_map_id!r}, but the geometry source serves "
                f"map {map_id!r}"
            )

    eligible = [item for item in ordered if len(item.geometry_support) >= policy.min_geometry_count]
    excluded = tuple(
        ExcludedObservation(
            spatial_observation_id=item.spatial_observation_id,
            geometry_count=len(item.geometry_support),
            minimum_geometry_count=policy.min_geometry_count,
        )
        for item in ordered
        if len(item.geometry_support) < policy.min_geometry_count
    )

    components = _overlap_components(eligible, policy.min_overlap)
    resolver = _CoordinateResolver(geometry)
    provenance = FusionSupportProvenance(
        support_policy_id=GEOMETRY_OVERLAP_SUPPORT_POLICY_ID,
        configuration_fingerprint=policy.fingerprint(),
        code_version=code_version,
    )
    supports = tuple(
        _support_of(
            FusionSupportId(f"support-{number:06d}"),
            [eligible[index] for index in component],
            resolver=resolver,
            acquisition_timestamps=acquisition_timestamps,
            provenance=provenance,
        )
        for number, component in enumerate(components, start=1)
    )
    return FusionSupportBuild(supports=supports, excluded=excluded)


def _overlap_components(
    observations: list[SpatialObservation], min_overlap: float
) -> list[list[int]]:
    """Link observations by Jaccard overlap and return the components, by first member."""
    counts = [len(item.geometry_support) for item in observations]
    references = [
        {reference.geometry_id for reference in item.geometry_support} for item in observations
    ]
    occurrences: dict[GeometryId, int] = {}
    for geometry_ids in references:
        for geometry_id in geometry_ids:
            occurrences[geometry_id] = occurrences.get(geometry_id, 0) + 1
    # Só a geometria vista por mais de uma observação pode contribuir para uma interseção.
    shared = {
        geometry_id: bit
        for bit, geometry_id in enumerate(
            sorted(geometry_id for geometry_id, seen in occurrences.items() if seen > 1)
        )
    }
    masks = [_bit_set(geometry_ids, shared) for geometry_ids in references]

    parent = list(range(len(observations)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(observations)):
        for right in range(left + 1, len(observations)):
            intersection = (masks[left] & masks[right]).bit_count()
            if intersection == 0:
                continue
            union = counts[left] + counts[right] - intersection
            if intersection / union >= min_overlap:
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    # A menor posição é a raiz: a primeira observação identifica a componente.
                    parent[max(root_left, root_right)] = min(root_left, root_right)

    members: dict[int, list[int]] = {}
    for index in range(len(observations)):
        members.setdefault(find(index), []).append(index)
    return [members[root] for root in sorted(members)]


def _bit_set(geometry_ids: set[GeometryId], bit_of: Mapping[GeometryId, int]) -> int:
    buffer = bytearray((len(bit_of) + 7) // 8)
    for geometry_id in geometry_ids:
        bit = bit_of.get(geometry_id)
        if bit is not None:
            buffer[bit >> 3] |= 1 << (bit & 7)
    return int.from_bytes(buffer, "little")


class _CoordinateResolver:
    """Resolve each geometry element of the map at most once."""

    def __init__(self, geometry: GeometrySource) -> None:
        self._geometry = geometry
        self._cache: dict[GeometryReference, Vector3] = {}

    def coordinates(self, reference: GeometryReference) -> Vector3:
        cached = self._cache.get(reference)
        if cached is None:
            cached = self._geometry.get(reference).coordinates_m
            self._cache[reference] = cached
        return cached

    @property
    def frame_id(self) -> FrameId:
        return self._geometry.geometric_map.frame_id


def _support_of(
    support_id: FusionSupportId,
    members: list[SpatialObservation],
    *,
    resolver: _CoordinateResolver,
    acquisition_timestamps: Mapping[SourceObservationId, SourceTimestamp],
    provenance: FusionSupportProvenance,
) -> FusionSupport:
    references = sorted(
        {reference for item in members for reference in item.geometry_support},
        key=lambda reference: reference.geometry_id,
    )
    coordinates: list[Vector3] = []
    for reference in references:
        try:
            coordinates.append(resolver.coordinates(reference))
        except KeyError as error:
            owner = next(item for item in members if reference in item.geometry_support)
            raise ValueError(
                f"spatial observation {owner.spatial_observation_id!r} references geometry "
                f"{reference.geometry_id!r}, which is not in map {reference.map_id!r}"
            ) from error
    count = len(coordinates)
    centroid = (
        math.fsum(point[0] for point in coordinates) / count,
        math.fsum(point[1] for point in coordinates) / count,
        math.fsum(point[2] for point in coordinates) / count,
    )
    stamps: list[SourceTimestamp] = []
    for item in members:
        stamp = acquisition_timestamps.get(item.source_observation_id)
        if stamp is None:
            raise ValueError(
                f"no acquisition timestamp for physical observation {item.source_observation_id!r}"
            )
        stamps.append(stamp)
    clocks = {stamp.clock_id for stamp in stamps}
    if len(clocks) != 1:
        raise ValueError(
            f"the observations of {support_id!r} span more than one clock domain: "
            f"{sorted(clocks)!r}"
        )
    return FusionSupport(
        fusion_support_id=support_id,
        geometric_map_id=references[0].map_id,
        geometry_support=tuple(references),
        spatial_observation_ids=tuple(item.spatial_observation_id for item in members),
        bounds=Bounds3D.enclosing(coordinates, frame_id=resolver.frame_id),
        centroid_m=centroid,
        time_bounds=TimeBounds(
            start=min(stamps, key=SourceTimestamp.total_nanoseconds),
            end=max(stamps, key=SourceTimestamp.total_nanoseconds),
        ),
        provenance=provenance,
    )

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
observations reference. Candidate pairs come from an inverted index of that shared geometry,
so only observations that share at least one element are compared: the cost follows the
input size plus the pairs that actually intersect, and is quadratic only when every
observation overlaps every other, where those pairs are real work.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import pairwise

from contextmap.geometric_mapping import Bounds3D, GeometryId, GeometryReference, GeometrySource
from contextmap.ingestion import FrameId, SourceObservationId
from contextmap.semantic_fusion.models import (
    FusionSupport,
    FusionSupportId,
    FusionSupportProvenance,
    _require_canonical,
    _time_bounds_of,
)
from contextmap.sensor_association import SpatialObservation, SpatialObservationId
from contextmap.shared import SourceTimestamp, Vector3

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
    # Índice invertido: quem vê cada elemento (geometry_support já é único por observação).
    holders: defaultdict[GeometryId, list[int]] = defaultdict(list)
    for index, item in enumerate(observations):
        for reference in item.geometry_support:
            holders[reference.geometry_id].append(index)
    # Só a geometria vista por mais de uma observação pode contribuir para uma interseção. Os
    # elementos vistos pelo mesmo conjunto de observações formam uma classe, e cada classe ocupa
    # uma faixa contígua de bits: o bit set de uma observação é o OR das faixas das suas classes,
    # e seus parceiros, o OR dos conjuntos de observações dessas classes. Um bit por elemento,
    # como antes, então a interseção de um par é a mesma.
    class_sizes: dict[tuple[int, ...], int] = {}
    for seen_by in holders.values():
        if len(seen_by) > 1:
            key = tuple(seen_by)
            class_sizes[key] = class_sizes.get(key, 0) + 1
    spans: list[list[tuple[int, int]]] = [[] for _ in observations]
    partners = [0] * len(observations)
    width = 0
    for members_of_class, size in class_sizes.items():
        class_bits = _index_bit_set(members_of_class)
        for index in members_of_class:
            spans[index].append((width, size))
            partners[index] |= class_bits
        width += size
    masks = [_span_bit_set(item, width) for item in spans]

    parent = list(range(len(observations)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(observations)):
        # Só os pares que compartilham geometria podem se ligar: nenhum par disjunto é avaliado.
        for right in _set_bits(partners[left] >> (left + 1), offset=left + 1):
            if _overlap(masks[left], masks[right], counts[left], counts[right]) >= min_overlap:
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    # A menor posição é a raiz: a primeira observação identifica a componente.
                    parent[max(root_left, root_right)] = min(root_left, root_right)

    members: dict[int, list[int]] = {}
    for index in range(len(observations)):
        members.setdefault(find(index), []).append(index)
    return [members[root] for root in sorted(members)]


def _overlap(left_mask: int, right_mask: int, left_count: int, right_count: int) -> float:
    """Jaccard index of two observations, from their shared-geometry bit sets and sizes."""
    intersection = (left_mask & right_mask).bit_count()
    return intersection / (left_count + right_count - intersection)


def _index_bit_set(indexes: tuple[int, ...]) -> int:
    """Bit set with one bit per observation position in ``indexes``."""
    buffer = bytearray((max(indexes) >> 3) + 1)
    for index in indexes:
        buffer[index >> 3] |= 1 << (index & 7)
    return int.from_bytes(buffer, "little")


def _span_bit_set(spans: list[tuple[int, int]], width: int) -> int:
    """Bit set of ``width`` bits with ``[start, start + size)`` set for every span, in one pass."""
    buffer = bytearray((width + 7) >> 3)
    for start, size in spans:
        if size == 1:
            buffer[start >> 3] |= 1 << (start & 7)
            continue
        end = start + size
        # Bits soltos até o próximo byte inteiro, bytes inteiros no meio, bits soltos no fim.
        head = min(end, (start + 7) & ~7)
        tail = max(head, end & ~7)
        for bit in (*range(start, head), *range(tail, end)):
            buffer[bit >> 3] |= 1 << (bit & 7)
        buffer[head >> 3 : tail >> 3] = b"\xff" * ((tail - head) >> 3)
    return int.from_bytes(buffer, "little")


def _set_bits(bits: int, *, offset: int) -> Iterator[int]:
    """Yield ``offset`` plus the position of every set bit of ``bits``, in increasing order."""
    while bits:
        lowest = bits & -bits
        yield offset + lowest.bit_length() - 1
        bits ^= lowest


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
    bounds = Bounds3D.enclosing(coordinates, frame_id=resolver.frame_id)
    count = len(coordinates)
    mean = tuple(math.fsum(point[axis] for point in coordinates) / count for axis in range(3))
    # A média está matematicamente dentro da caixa; o clamp só corrige o erro de arredondamento
    # que, num eixo degenerado (ex.: piso em z constante), a deixaria 1 ulp fora dos bounds.
    x, y, z = (
        min(max(value, low), high)
        for value, low, high in zip(mean, bounds.minimum_m, bounds.maximum_m, strict=True)
    )
    stamps: list[SourceTimestamp] = []
    for item in members:
        stamp = acquisition_timestamps.get(item.source_observation_id)
        if stamp is None:
            raise ValueError(
                f"no acquisition timestamp for physical observation {item.source_observation_id!r}"
            )
        stamps.append(stamp)
    return FusionSupport(
        fusion_support_id=support_id,
        geometric_map_id=references[0].map_id,
        geometry_support=tuple(references),
        spatial_observation_ids=tuple(item.spatial_observation_id for item in members),
        bounds=bounds,
        centroid_m=(x, y, z),
        time_bounds=_time_bounds_of(stamps, owner=f"the observations of {support_id!r}"),
        provenance=provenance,
    )

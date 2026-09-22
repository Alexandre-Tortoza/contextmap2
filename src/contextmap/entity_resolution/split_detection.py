"""Optional detection of entities that may hold more than one object: split candidates.

Semantic Mapping materializes one entity per fusion support, and a support can occasionally cover
more than one physical object, producing an over-merged entity before resolution begins. Splitting
is harder than merging, so it is an **explicit, optional capability that only reports**: the
detector emits :class:`SplitCandidate` diagnostics, never splits an entity, never runs inside
pairwise resolution and never mutates a source entity. Nothing here materializes a split, so the
baseline merge resolution is unchanged whether or not detection is ever called.

The baseline, ``entity-split-detection-v1``, is geometric and deterministic. It resolves the exact
geometry references of an entity against the map, links points closer than
``connectivity_radius_m`` and takes the connected components as candidate partitions, each kept as
the exact references it is made of. An entity whose support is one connected piece is not a
candidate. A disconnected entity is **not** split just for being disconnected; the signals decide:

* ``disconnected-components``: at least two *significant* partitions (at least
  ``min_partition_points`` points and ``min_partition_fraction`` of the support) support a split,
  so a sparse entity whose pieces are noise does not qualify;
* ``partition-gap``: the smallest gap between the bounds of two significant partitions is at least
  ``min_gap_m`` (supporting) or smaller (conflicting: the pieces may be one sparse object);
* ``dominant-partition``: one partition holds more than ``max_dominant_fraction`` of the support
  (conflicting: the rest looks like outliers around one object).

The status follows the signals, conservatively: ``suggested`` needs the components signal and no
conflicting signal, ``unresolved`` means it holds but another signal argues against, and
``rejected`` means there are not two significant partitions. Semantic or appearance
discontinuity and clustering
beyond connectivity are not implemented, and the semantic label is never a split criterion on its
own. Any future automatic materialization would need its own configuration, version and evaluation.
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

from contextmap.entity_resolution._checks import require_canonical
from contextmap.entity_resolution._spatial import bounds_gap
from contextmap.entity_resolution.channels import EvidenceStatus, Finding
from contextmap.entity_resolution.models import PolicyRef, reference_order
from contextmap.geometric_mapping import Bounds3D, GeometryReference, GeometrySource
from contextmap.semantic_mapping import Entity, EntityReference, resolve_geometry

SPLIT_DETECTION_POLICY_ID = "entity-split-detection-v1"
"""Versioned identity of the geometric split signals described in this module."""


class SplitStatus(Enum):
    """What the signals say about splitting an entity.

    Attributes:
        SUGGESTED: The signals support splitting and nothing argues against it.
        REJECTED: There are not two significant partitions, so there is nothing to split.
        UNRESOLVED: The components support a split but another signal argues against it.
    """

    SUGGESTED = "suggested"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, kw_only=True)
class SplitPartition:
    """One proposed part of an entity, as the exact geometry it is made of.

    Attributes:
        geometry_refs: The authoritative references of the part, sorted by ``geometry_id`` and
            unique, never empty; a subset of the entity's support, never a copy of coordinates.
        bounds: The tight axis-aligned box of the part, in the map frame.
    """

    geometry_refs: tuple[GeometryReference, ...]
    bounds: Bounds3D

    def __post_init__(self) -> None:
        """Validate that the part is a canonical, non-empty set of references.

        Raises:
            ValueError: If the part is empty, or its references are not sorted and unique.
        """
        if not self.geometry_refs:
            raise ValueError("geometry_refs must not be empty")
        require_canonical(
            "geometry_refs",
            self.geometry_refs,
            lambda reference: (reference.geometry_id,),
            detail="by geometry_id ",
        )


@dataclass(frozen=True, kw_only=True)
class SplitDetectionPolicy:
    """Explicit thresholds of the geometric split signals.

    There are no defaults: the thresholds are scientific choices, tied to the scale and density of a
    scene, that a profile declares and justifies with evaluation.

    Attributes:
        connectivity_radius_m: Distance under which two points are linked, in meters.
        min_partition_points: Fewest points a partition needs to be significant.
        min_partition_fraction: Smallest share of the support a partition needs to be significant,
            in ``(0, 1)``.
        min_gap_m: Smallest gap between two significant partitions that supports a split, in
            meters.
        max_dominant_fraction: Share of the support above which one partition dominates, in
            ``(0.5, 1]``; a dominated entity is not split.
    """

    connectivity_radius_m: float
    min_partition_points: int
    min_partition_fraction: float
    min_gap_m: float
    max_dominant_fraction: float

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ValueError: If a distance is not finite and positive, the point count is below one, or
                a fraction is outside its range.
        """
        for name in ("connectivity_radius_m", "min_gap_m"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if self.min_partition_points < 1:
            raise ValueError("min_partition_points must be at least 1")
        if not (
            math.isfinite(self.min_partition_fraction) and 0.0 < self.min_partition_fraction < 1.0
        ):
            raise ValueError("min_partition_fraction must be within (0, 1)")
        if not (
            math.isfinite(self.max_dominant_fraction) and 0.5 < self.max_dominant_fraction <= 1.0
        ):
            raise ValueError("max_dominant_fraction must be within (0.5, 1]")

    def fingerprint(self) -> str:
        """Hash the policy identity and thresholds, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "policy_id": SPLIT_DETECTION_POLICY_ID,
                "connectivity_radius_m": self.connectivity_radius_m,
                "min_partition_points": self.min_partition_points,
                "min_partition_fraction": self.min_partition_fraction,
                "min_gap_m": self.min_gap_m,
                "max_dominant_fraction": self.max_dominant_fraction,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def ref(self) -> PolicyRef:
        """The policy identity and configuration, as recorded on a candidate."""
        return PolicyRef(
            policy_id=SPLIT_DETECTION_POLICY_ID, configuration_fingerprint=self.fingerprint()
        )


@dataclass(frozen=True, kw_only=True)
class SplitCandidate:
    """A diagnostic that an entity may hold more than one object; never a split.

    Attributes:
        entity_ref: The entity, exactly as its semantic map names it; it is not modified.
        partitions: The connected pieces of its support, each as exact references, sorted by their
            first reference; at least two.
        signals: The findings behind the status: ``supporting`` supports a split, ``conflicting``
            argues against it.
        status: What the signals say.
        policy: The policy and configuration that produced it.
    """

    entity_ref: EntityReference
    partitions: tuple[SplitPartition, ...]
    signals: tuple[Finding, ...]
    status: SplitStatus
    policy: PolicyRef

    def __post_init__(self) -> None:
        """Validate that the partitions are disjoint, ordered, and the status follows the signals.

        Raises:
            ValueError: If there are fewer than two partitions, they are unsorted or share a
                reference, or the status does not follow from the signals.
        """
        if len(self.partitions) < 2:
            raise ValueError("a split candidate needs at least two partitions")
        require_canonical(
            "partitions", self.partitions, lambda item: (item.geometry_refs[0].geometry_id,)
        )
        references = [ref for item in self.partitions for ref in item.geometry_refs]
        if len(set(references)) != len(references):
            raise ValueError("partitions must not share a geometry reference")
        if self.status is not _status_of(self.signals):
            raise ValueError(f"status {self.status.value!r} does not follow from the signals")


def _status_of(signals: Sequence[Finding]) -> SplitStatus:
    by_rule = {item.rule_id: item.status for item in signals}
    if by_rule.get("disconnected-components") is not EvidenceStatus.SUPPORTING:
        return SplitStatus.REJECTED
    if any(item.status is EvidenceStatus.CONFLICTING for item in signals):
        return SplitStatus.UNRESOLVED
    return SplitStatus.SUGGESTED


def detect_split_candidates(
    entities: Iterable[Entity], *, source: GeometrySource, policy: SplitDetectionPolicy
) -> tuple[SplitCandidate, ...]:
    """Flag the entities whose support falls apart into several pieces.

    Only reports: no entity is split, merged or modified. An entity with a single connected support
    is not a candidate, and a disconnected one is judged by the signals instead of split by default.

    Args:
        entities: The entities to inspect, in any order; all over the map ``source`` serves.
        source: The read boundary of the geometric map the supports belong to.
        policy: The thresholds of the signals.

    Returns:
        One candidate per entity with at least two connected pieces, sorted by entity reference.

    Raises:
        GeometryResolutionError: If a reference of an entity cannot be resolved against ``source``.
    """
    candidates = []
    for entity in sorted(entities, key=lambda item: reference_order(item.reference)):
        parts = _partitions(entity, source, policy.connectivity_radius_m)
        if len(parts) < 2:
            continue
        signals = _signals(parts, policy)
        candidates.append(
            SplitCandidate(
                entity_ref=entity.reference,
                partitions=tuple(parts),
                signals=signals,
                status=_status_of(signals),
                policy=policy.ref(),
            )
        )
    return tuple(candidates)


def _partitions(entity: Entity, source: GeometrySource, radius_m: float) -> list[SplitPartition]:
    """The connected components of the entity's support, as exact references and bounds."""
    points = resolve_geometry(entity.geometry.geometry_refs, source=source)
    coordinates = [point.coordinates_m for point in points]
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

    limit = radius_m * radius_m
    for (cx, cy, cz), members in cells.items():
        for dx, dy, dz in product((-1, 0, 1), repeat=3):
            for other in cells.get((cx + dx, cy + dy, cz + dz), ()):
                for member in members:
                    if (
                        other > member
                        and _squared(coordinates[member], coordinates[other]) <= limit
                    ):
                        parent[find(other)] = find(member)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(coordinates)):
        groups[find(index)].append(index)
    frame = entity.geometry.map_frame
    parts = [
        SplitPartition(
            geometry_refs=tuple(entity.geometry.geometry_refs[index] for index in sorted(members)),
            bounds=Bounds3D.enclosing([coordinates[index] for index in members], frame_id=frame),
        )
        for members in groups.values()
    ]
    return sorted(parts, key=lambda item: item.geometry_refs[0].geometry_id)


def _squared(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _signals(parts: list[SplitPartition], policy: SplitDetectionPolicy) -> tuple[Finding, ...]:
    total = sum(len(item.geometry_refs) for item in parts)
    significant = [
        item
        for item in parts
        if len(item.geometry_refs) >= policy.min_partition_points
        and len(item.geometry_refs) / total >= policy.min_partition_fraction
    ]
    two_or_more = len(significant) >= 2
    signals = [
        Finding(
            rule_id="disconnected-components",
            status=EvidenceStatus.SUPPORTING if two_or_more else EvidenceStatus.NEUTRAL,
            detail=(
                f"{len(parts)} connected pieces, {len(significant)} significant (at least "
                f"{policy.min_partition_points} points and {policy.min_partition_fraction} of the "
                f"support)"
            ),
            metric="significant_partition_count",
            observed=float(len(significant)),
            threshold=2.0,
        )
    ]
    if two_or_more:
        gap = min(
            bounds_gap(left.bounds, right.bounds)
            for index, left in enumerate(significant)
            for right in significant[index + 1 :]
        )
        far = gap >= policy.min_gap_m
        signals.append(
            Finding(
                rule_id="partition-gap",
                status=EvidenceStatus.SUPPORTING if far else EvidenceStatus.CONFLICTING,
                detail=(
                    f"the closest significant pieces are {gap:.3f} m apart"
                    + ("" if far else ": they may be one sparse object")
                ),
                metric="partition_gap_m",
                observed=gap,
                threshold=policy.min_gap_m,
            )
        )
        dominant = max(len(item.geometry_refs) for item in parts) / total
        signals.append(
            Finding(
                rule_id="dominant-partition",
                status=EvidenceStatus.CONFLICTING
                if dominant > policy.max_dominant_fraction
                else EvidenceStatus.NEUTRAL,
                detail=f"the largest piece holds {dominant:.3f} of the support",
                metric="dominant_fraction",
                observed=dominant,
                threshold=policy.max_dominant_fraction,
            )
        )
    return tuple(signals)

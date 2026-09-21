"""Coordinate semantics of a ContextMap: frame, units, axes and origin.

A consumer must be able to read an ``(x, y, z)`` of the map from this record alone, without
opening state-estimation internals and without inferring anything from a dataset name. Each
statement is explicit, and what is not known is written as ``None`` instead of being omitted.

The most important distinction is the origin. An estimator-local origin is defined by where a
run of an estimator happened to start, so its coordinates mean something only inside that one
artifact. An externally anchored origin is tied to an external reference frame. Coordinates of
two maps are never implied to be comparable unless both are anchored to the same reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from contextmap.artifact._checks import require_present
from contextmap.shared import Vector3

UP_DIRECTION_NORM_TOLERANCE = 1e-6
"""Maximum accepted ``|norm - 1|`` of a declared up direction."""


class LengthUnit(Enum):
    """The unit of every length and coordinate of the map.

    Attributes:
        METER: SI meters, the unit of every persisted coordinate in Solution 1.
    """

    METER = "meter"


class Handedness(Enum):
    """The handedness of the coordinate system.

    Attributes:
        RIGHT_HANDED: ``x cross y`` points along ``z``.
        LEFT_HANDED: ``x cross y`` points against ``z``.
    """

    RIGHT_HANDED = "right_handed"
    LEFT_HANDED = "left_handed"


class AnchorKind(Enum):
    """How the origin of the map frame is defined.

    Attributes:
        ESTIMATOR_LOCAL: Defined by an estimator run (for example, its first pose). Coordinates
            are meaningful only inside this artifact.
        EXTERNALLY_ANCHORED: Tied to a named external reference frame through an explicit
            alignment.
    """

    ESTIMATOR_LOCAL = "estimator_local"
    EXTERNALLY_ANCHORED = "externally_anchored"


@dataclass(frozen=True, kw_only=True)
class MapAnchor:
    """Where the origin of the map frame comes from.

    Attributes:
        kind: Estimator-local or externally anchored.
        origin_definition: Human-readable statement of how the origin is defined and, for an
            external anchor, how the alignment was established.
        reference_frame_id: Identity of the external reference frame; required when the map is
            externally anchored and ``None`` when it is estimator-local.
    """

    kind: AnchorKind
    origin_definition: str
    reference_frame_id: str | None

    def __post_init__(self) -> None:
        """Validate that the anchor is defined and coherent with its kind.

        Raises:
            ValueError: If the origin is not defined, an external anchor names no reference,
                or an estimator-local one claims a reference.
        """
        require_present(self, "origin_definition")
        if self.kind is AnchorKind.EXTERNALLY_ANCHORED:
            if self.reference_frame_id is None or not self.reference_frame_id.strip():
                raise ValueError("an externally anchored origin must name its reference_frame_id")
        elif self.reference_frame_id is not None:
            raise ValueError(
                "an estimator-local origin must not claim a reference_frame_id: no alignment exists"
            )


@dataclass(frozen=True, kw_only=True)
class MapFrame:
    """The coordinate frame every position of the map is expressed in.

    Attributes:
        frame_id: Identity of the map frame; the frame of the geometry the map references.
        unit: Unit of every coordinate.
        handedness: Handedness of the axes.
        up_direction: Unit vector, in the map frame, that points away from gravity; ``None``
            when it is not known. An estimator-local frame does not guarantee that ``z`` is up.
        anchor: How the origin is defined.
    """

    frame_id: str
    unit: LengthUnit
    handedness: Handedness
    up_direction: Vector3 | None
    anchor: MapAnchor

    def __post_init__(self) -> None:
        """Validate the frame identity and the up direction.

        Raises:
            ValueError: If the frame identity is blank or the up direction is not a finite
                unit vector.
        """
        require_present(self, "frame_id")
        if self.up_direction is not None:
            finite = all(math.isfinite(value) for value in self.up_direction)
            if (
                not finite
                or abs(math.hypot(*self.up_direction) - 1.0) > UP_DIRECTION_NORM_TOLERANCE
            ):
                raise ValueError(
                    f"up_direction must be a finite unit vector, got {self.up_direction!r}"
                )

    def is_comparable_with(self, other: MapFrame) -> bool:
        """Check whether coordinates of two maps can be compared directly.

        Comparability is never implied by a shared name: two estimator-local frames are not
        comparable even when both are called ``map``, because no alignment exists between them.
        Two frames are comparable only when both are externally anchored to the same reference
        frame and agree on unit and handedness.

        Args:
            other: The frame of the other map.

        Returns:
            ``True`` only when an alignment through a shared external reference exists.
        """
        return (
            self.anchor.kind is AnchorKind.EXTERNALLY_ANCHORED
            and other.anchor.kind is AnchorKind.EXTERNALLY_ANCHORED
            and self.anchor.reference_frame_id == other.anchor.reference_frame_id
            and self.unit is other.unit
            and self.handedness is other.handedness
        )

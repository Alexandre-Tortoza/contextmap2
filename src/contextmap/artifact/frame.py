"""Coordinate semantics of a ContextMap: frame, units, axes and origin.

A consumer must be able to read an ``(x, y, z)`` of the map from this record alone, without
opening state-estimation internals and without inferring anything from a dataset name. Each
statement is explicit, and what is not known is written as ``None`` instead of being omitted.

The most important distinction is the origin. An estimator-local origin is defined by where a
run of an estimator happened to start, so its coordinates mean something only inside that one
artifact. An externally anchored origin means the map's coordinates are already expressed
exactly in the named external reference frame: there is no separate local origin or basis of
its own that a transform would need to remove. That invariant is what makes two externally
anchored maps directly comparable; without it, sharing the name of a reference would prove
nothing about whether their XYZ values live in the same basis. Coordinates of two maps are
never implied to be comparable unless both satisfy it against the same reference.
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
        EXTERNALLY_ANCHORED: The map's coordinates are already expressed exactly in the named
            external reference frame, with no separate local origin or basis: the map frame
            *is* the reference frame, and :class:`MapFrame` requires ``frame_id`` to equal
            ``reference_frame_id`` to make that identity explicit and verifiable. This is a
            structural guarantee, not a claim about how the alignment was produced upstream.
    """

    ESTIMATOR_LOCAL = "estimator_local"
    EXTERNALLY_ANCHORED = "externally_anchored"


@dataclass(frozen=True, kw_only=True)
class MapAnchor:
    """Where the origin of the map frame comes from.

    Attributes:
        kind: Estimator-local or externally anchored.
        origin_definition: Human-readable statement of how the origin is defined and, for an
            external anchor, how the alignment that put the coordinates in the reference frame
            was established upstream. It documents that history; it is not what makes two maps
            comparable, since free text can drift from the data without being detectable.
        reference_frame_id: Identity of the external reference frame; required when the map is
            externally anchored and ``None`` when it is estimator-local. For an externally
            anchored map, :class:`MapFrame` requires this to equal the map's own ``frame_id``.
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
        frame_id: Identity of the map frame; the frame of the geometry the map references. For
            an externally anchored map this equals ``anchor.reference_frame_id``: the map frame
            *is* the reference frame, never a separate local basis aligned to it.
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
        """Validate the frame identity, the up direction and the external-anchor invariant.

        Raises:
            ValueError: If the frame identity is blank, the up direction is not a finite unit
                vector, or the map is externally anchored but its ``frame_id`` differs from
                ``anchor.reference_frame_id``.
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
        if (
            self.anchor.kind is AnchorKind.EXTERNALLY_ANCHORED
            and self.frame_id != self.anchor.reference_frame_id
        ):
            # EXTERNALLY_ANCHORED significa que as coordenadas já estão no frame externo, sem
            # base local própria: sem esta igualdade, dois mapas citando o mesmo
            # reference_frame_id poderiam ter bases diferentes e ainda assim parecer comparáveis.
            raise ValueError(
                "an externally anchored frame's frame_id must equal its reference_frame_id "
                f"{self.anchor.reference_frame_id!r} (no separate local origin/base is "
                f"representable), got frame_id={self.frame_id!r}"
            )

    def is_comparable_with(self, other: MapFrame) -> bool:
        """Check whether coordinates of two maps can be compared directly.

        Comparability is never implied by a shared name: two estimator-local frames are not
        comparable even when both are called ``map``, because no alignment exists between them.
        Two frames are comparable only when both are externally anchored to the same reference
        frame and agree on unit and handedness. This is sound, not merely a naming coincidence,
        because :class:`MapFrame` requires an externally anchored frame's ``frame_id`` to equal
        its ``reference_frame_id``: matching ``reference_frame_id`` therefore proves the two
        frames share the same identity, with no separate local basis either could diverge in.

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


CANONICAL_MAP_FRAME_DECLARATION_ID = "estimator-local-map-frame-v1"
"""Identity of :func:`estimator_local_map_frame`'s declaration, versioned like any other policy."""


def estimator_local_map_frame(*, frame_id: str, up_direction: Vector3 | None = None) -> MapFrame:
    """The one map-frame declaration Solution 1's canonical profile uses today.

    Every source this repository ingests (ROS) already follows REP-103: metric, right-handed
    coordinates. No capability upstream of ``ContextMap`` surveys or externally anchors its
    output, so the origin is always wherever the state-estimation run happened to start --
    :attr:`AnchorKind.ESTIMATOR_LOCAL`, never :attr:`AnchorKind.EXTERNALLY_ANCHORED`. This does
    not invent that convention per call: it names it once, as an explicit, versioned identity
    (:data:`CANONICAL_MAP_FRAME_DECLARATION_ID`), the same way every other scientific policy in
    this codebase is named rather than silently assumed.

    Args:
        frame_id: Identity of the map frame; the caller's own geometry manifest already knows
            it, so this function never invents one.
        up_direction: The map frame's up axis, when a run declared one (for example Spatial
            Relations' own ``FrameConventions.up_axis.vector``) -- never derived independently,
            so it can never disagree with what the map's own evidence was actually produced
            under. ``None`` when no run declared one: an estimator-local frame never guarantees
            a known up direction (see :class:`MapFrame`).

    Returns:
        A frame declaring :data:`~contextmap.artifact.frame.LengthUnit.METER`,
        :data:`~contextmap.artifact.frame.Handedness.RIGHT_HANDED` and an estimator-local anchor.
    """
    return MapFrame(
        frame_id=frame_id,
        unit=LengthUnit.METER,
        handedness=Handedness.RIGHT_HANDED,
        up_direction=up_direction,
        anchor=MapAnchor(
            kind=AnchorKind.ESTIMATOR_LOCAL,
            origin_definition="pose of the first accepted scan of the state estimation run",
            reference_frame_id=None,
        ),
    )

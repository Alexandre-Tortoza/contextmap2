"""What a map frame declares about direction, and the loud failure when it declares nothing.

Nothing upstream says which way is up. A geometric map is expressed in an arbitrary frame chosen
by state estimation, and a SLAM frame is not gravity-aligned unless something aligned it. Vertical
and depth predicates are therefore meaningless until a run *declares* the axes of its map frame,
and this module is where that declaration lives, versioned and fingerprinted like any other
policy. Nothing here is dataset-specific and no axis is ever guessed from the geometry.

Axes are restricted to the signed principal axes of the map frame. Entity geometry is summarized
as an axis-aligned box in that frame, so an up axis tilted with respect to the box faces could not
be evaluated exactly; a tilted convention would be a new policy version, not a silent
approximation.

Evaluators call :meth:`FrameConventions.require_evaluable` before measuring anything. A geometry in
another frame or another map, or a predicate whose axis was never declared, raises instead of
producing evidence that only looks meaningful.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from contextmap.semantic_mapping import EntityGeometry
from contextmap.shared import Vector3
from contextmap.spatial_relations._checks import require_present
from contextmap.spatial_relations.taxonomy import FrameRequirement

FRAME_CONVENTIONS_POLICY_ID = "map-frame-conventions-v1"
"""Versioned identity of the meaning given to the declared axes."""


class FrameConventionError(ValueError):
    """Base class for evidence that cannot be produced under the declared frame conventions."""


class IncompatibleFrameError(FrameConventionError):
    """Raised when geometry is not expressed in the frame the conventions describe."""


class UndeclaredAxisError(FrameConventionError):
    """Raised when a predicate needs an axis that the conventions do not declare."""


class AxisDirection(Enum):
    """A signed principal axis of a map frame.

    Attributes:
        POSITIVE_X: The ``+x`` axis.
        NEGATIVE_X: The ``-x`` axis.
        POSITIVE_Y: The ``+y`` axis.
        NEGATIVE_Y: The ``-y`` axis.
        POSITIVE_Z: The ``+z`` axis.
        NEGATIVE_Z: The ``-z`` axis.
    """

    POSITIVE_X = "+x"
    NEGATIVE_X = "-x"
    POSITIVE_Y = "+y"
    NEGATIVE_Y = "-y"
    POSITIVE_Z = "+z"
    NEGATIVE_Z = "-z"

    @property
    def axis_index(self) -> int:
        """The index of the axis: ``0`` for x, ``1`` for y and ``2`` for z."""
        return "xyz".index(self.value[1])

    @property
    def sign(self) -> int:
        """``1`` when the direction follows the axis and ``-1`` when it opposes it."""
        return 1 if self.value[0] == "+" else -1

    @property
    def vector(self) -> Vector3:
        """The unit vector of the direction, in the map frame."""
        components = [0.0, 0.0, 0.0]
        components[self.axis_index] = float(self.sign)
        return (components[0], components[1], components[2])

    @property
    def opposite(self) -> AxisDirection:
        """The direction along the same axis with the other sign."""
        flipped = ("-" if self.sign == 1 else "+") + self.value[1]
        return AxisDirection(flipped)


@dataclass(frozen=True, kw_only=True)
class FrameConventions:
    """The axes a run declares for one map frame.

    There are no defaults: which way is up is a fact about the run's map frame, not about the
    algorithm.

    Attributes:
        map_frame: The frame these conventions describe; geometry must be expressed in it.
        up_axis: The direction opposite to gravity, or ``None`` when the run does not declare one.
        forward_axis: The horizontal reference direction of ``IN_FRONT_OF`` and ``BEHIND``, or
            ``None`` when the run does not declare one. It belongs to the map frame, not to the
            orientation of any entity.
    """

    map_frame: str
    up_axis: AxisDirection | None = None
    forward_axis: AxisDirection | None = None

    def __post_init__(self) -> None:
        """Validate that the declaration is coherent.

        Raises:
            ValueError: If the frame is empty, a forward axis is declared without an up axis, or
                the forward axis is not perpendicular to the up axis.
        """
        require_present(self, "map_frame")
        if self.forward_axis is None:
            return
        if self.up_axis is None:
            raise ValueError("a forward axis is only meaningful when an up axis is declared")
        if self.forward_axis.axis_index == self.up_axis.axis_index:
            raise ValueError(
                f"the forward axis {self.forward_axis.value!r} must be perpendicular to the up "
                f"axis {self.up_axis.value!r}"
            )

    def supports(self, requirement: FrameRequirement) -> bool:
        """Whether the declared axes satisfy what a predicate needs.

        Args:
            requirement: What a predicate needs from the map frame.

        Returns:
            ``True`` when every axis the requirement names is declared.
        """
        if requirement is FrameRequirement.MAP_FRAME:
            return True
        if requirement is FrameRequirement.UP_AXIS:
            return self.up_axis is not None
        return self.up_axis is not None and self.forward_axis is not None

    def require_evaluable(self, requirement: FrameRequirement, *geometries: EntityGeometry) -> None:
        """Refuse to measure when the frame cannot carry the predicate.

        Args:
            requirement: What the predicate needs from the map frame.
            *geometries: The geometry the predicate is about, every one of which must be
                expressed in the declared map frame and belong to one geometric map.

        Raises:
            IncompatibleFrameError: If a geometry is expressed in another frame, or the geometries
                belong to different geometric maps: identical frame names in two maps do not make
                their coordinates comparable.
            UndeclaredAxisError: If the predicate needs an axis that is not declared.
        """
        for geometry in geometries:
            if geometry.map_frame != self.map_frame:
                raise IncompatibleFrameError(
                    f"geometry is expressed in frame {geometry.map_frame!r}, but the conventions "
                    f"describe {self.map_frame!r}"
                )
        maps = {geometry.geometric_map_id for geometry in geometries}
        if len(maps) > 1:
            raise IncompatibleFrameError(
                f"geometry belongs to different geometric maps {sorted(maps)!r}; coordinates of "
                f"different maps are not comparable even in a frame with the same name"
            )
        if requirement is FrameRequirement.MAP_FRAME:
            return
        if self.up_axis is None:
            raise UndeclaredAxisError(
                f"the conventions for frame {self.map_frame!r} declare no up axis, so vertical "
                f"relations cannot be evaluated"
            )
        if requirement is FrameRequirement.UP_AND_FORWARD_AXES and self.forward_axis is None:
            raise UndeclaredAxisError(
                f"the conventions for frame {self.map_frame!r} declare no forward axis, so "
                f"depth relations cannot be evaluated"
            )

    def fingerprint(self) -> str:
        """Hash the policy identity and the declared axes, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical declaration.
        """
        canonical = json.dumps(
            {
                "policy_id": FRAME_CONVENTIONS_POLICY_ID,
                "map_frame": self.map_frame,
                "up_axis": None if self.up_axis is None else self.up_axis.value,
                "forward_axis": None if self.forward_axis is None else self.forward_axis.value,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

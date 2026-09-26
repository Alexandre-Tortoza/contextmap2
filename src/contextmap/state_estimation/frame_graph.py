"""Static frame graph built from canonical calibration extrinsics.

The graph connects frames through the *static* transforms Ingestion owns
(:class:`~contextmap.ingestion.CalibrationSet`). It never owns or rewrites
calibration; it only answers two questions with explicit frame direction:

* what is ``T_parent_child`` between two frames, composed along the path;
* do redundant paths between frames agree, or is the graph ambiguous?

Every edge is a :class:`~contextmap.ingestion.RigidTransform` following the
``T_parent_child`` convention (``p_parent = R * p_child + t``); traversing it
from child to parent uses its inverse. A dynamic pose (the body in the map)
is not part of this graph: it is a
:class:`~contextmap.state_estimation.PoseEstimate`.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from contextmap.ingestion import CalibrationSet, FrameId, RigidTransform
from contextmap.shared import (
    Quaternion,
    Vector3,
    compose_rigid,
    invert_rigid,
    quaternion_angle_between,
    quaternion_norm,
)

_IDENTITY_ROTATION: Quaternion = (0.0, 0.0, 0.0, 1.0)

STATIC_ROTATION_NORM_TOLERANCE = 1e-5
"""Largest ``|norm(q) - 1|`` accepted for a static rotation the graph composes.

Inverting and composing a transform assume a unit quaternion, so a larger deviation means
wrong numbers, not a slightly imprecise rotation. The tolerance accepts text-precision
calibration, the same default as the state-estimation preflight's rotation check."""


class FrameGraphError(ValueError):
    """Raised when a frame is unknown or two frames have no static path."""


@dataclass(frozen=True, kw_only=True)
class ResolvedTransform:
    """``T_parent_child`` resolved through the graph.

    Attributes:
        parent_frame: Frame the coordinates are expressed in.
        child_frame: Frame whose coordinates are transformed.
        translation: ``(x, y, z)`` in meters.
        rotation: Unit quaternion ``(x, y, z, w)``.
    """

    parent_frame: FrameId
    child_frame: FrameId
    translation: Vector3
    rotation: Quaternion


@dataclass(frozen=True, kw_only=True)
class LoopInconsistency:
    """A redundant static transform that disagrees with the rest of the graph.

    Attributes:
        frames: ``(parent, child)`` of the transform that closes the loop.
            Which edge of an inconsistent loop is reported depends on the
            traversal, so read it as "this loop does not close".
        translation_error_m: Distance in meters between the position
            predicted through the rest of the loop and the transform.
        rotation_error_rad: Rotation angle in radians between the two.
    """

    frames: tuple[FrameId, FrameId]
    translation_error_m: float
    rotation_error_rad: float


@dataclass(frozen=True)
class _Step:
    neighbor: FrameId
    edge_index: int
    reversed_edge: bool


class StaticFrameGraph:
    """Undirected graph of frames connected by static ``RigidTransform`` edges."""

    def __init__(self, transforms: Sequence[RigidTransform]) -> None:
        """Build the graph.

        Args:
            transforms: Static transforms, each an edge ``parent <-> child``.
        """
        self._transforms = tuple(transforms)
        adjacency: dict[FrameId, list[_Step]] = {}
        for index, transform in enumerate(self._transforms):
            adjacency.setdefault(transform.parent_frame, []).append(
                _Step(neighbor=transform.child_frame, edge_index=index, reversed_edge=False)
            )
            adjacency.setdefault(transform.child_frame, []).append(
                _Step(neighbor=transform.parent_frame, edge_index=index, reversed_edge=True)
            )
        # Ordem determinística: o mesmo grafo sempre produz a mesma árvore e o mesmo resultado.
        self._adjacency = {
            frame: sorted(steps, key=lambda step: (str(step.neighbor), step.edge_index))
            for frame, steps in adjacency.items()
        }

    @classmethod
    def from_calibration(cls, calibration: CalibrationSet) -> StaticFrameGraph:
        """Build the graph from a calibration set's static transforms."""
        return cls(calibration.static_transforms)

    @property
    def frames(self) -> frozenset[FrameId]:
        """Every frame that appears in a static transform."""
        return frozenset(self._adjacency)

    @property
    def edges(self) -> tuple[tuple[FrameId, FrameId], ...]:
        """``(parent, child)`` of every static transform, in input order."""
        return tuple((item.parent_frame, item.child_frame) for item in self._transforms)

    @property
    def loop_count(self) -> int:
        """Number of independent loops, i.e. redundant edges beyond a spanning forest."""
        return len(self._transforms) - len(self._adjacency) + self._component_count()

    def component_of(self, frame: FrameId) -> frozenset[FrameId]:
        """Return the frames connected to ``frame`` through static transforms.

        Raises:
            FrameGraphError: If ``frame`` is not in the graph.
        """
        self._require_frame(frame)
        return frozenset(self._traverse(frame)[0])

    def resolve(self, parent_frame: FrameId, child_frame: FrameId) -> ResolvedTransform:
        """Resolve ``T_parent_child`` by composing the static transforms on the path.

        Args:
            parent_frame: Frame the coordinates are expressed in.
            child_frame: Frame whose coordinates are transformed.

        Returns:
            The composed transform; the identity when both frames are the same.

        Raises:
            FrameGraphError: If a frame is unknown, no static path connects them, or a transform
                on the path does not have a unit rotation quaternion.
        """
        self._require_frame(parent_frame)
        self._require_frame(child_frame)
        _, from_root, reached_by = self._traverse(parent_frame)
        if child_frame not in from_root:
            raise FrameGraphError(
                f"no static path between {parent_frame!r} and {child_frame!r} in the calibration"
            )
        # Só as arestas do caminho entram no resultado: uma extrínseca inválida que este caminho
        # não usa não impede resolvê-lo (o preflight a reporta como calibração não usada).
        frame = child_frame
        while frame != parent_frame:
            transform = self._transforms[reached_by[frame]]
            _require_unit_rotation(transform)
            frame = (
                transform.parent_frame if transform.child_frame == frame else transform.child_frame
            )
        translation, rotation = from_root[child_frame]
        return ResolvedTransform(
            parent_frame=parent_frame,
            child_frame=child_frame,
            translation=translation,
            rotation=rotation,
        )

    def loop_inconsistencies(
        self, *, translation_tolerance_m: float, rotation_tolerance_rad: float
    ) -> tuple[LoopInconsistency, ...]:
        """Find redundant transforms that disagree with the rest of the graph.

        A spanning tree of each component defines every frame relative to a
        root. Any other edge closes a loop; if composing the path through the
        tree does not reproduce that edge, two paths between the same frames
        disagree and the graph is ambiguous.

        Args:
            translation_tolerance_m: Largest accepted position disagreement.
            rotation_tolerance_rad: Largest accepted rotation disagreement.

        Returns:
            One finding per loop that does not close within the tolerances;
            a disagreement that is not finite is always reported.
        """
        findings: list[LoopInconsistency] = []
        visited: set[FrameId] = set()
        for root in sorted(self._adjacency, key=str):
            if root in visited:
                continue
            order, from_root, reached_by = self._traverse(root)
            tree_edges = set(reached_by.values())
            component = set(order)
            visited.update(component)
            for index, transform in enumerate(self._transforms):
                if index in tree_edges or transform.parent_frame not in component:
                    continue
                predicted_translation, predicted_rotation = compose_rigid(
                    outer_translation=from_root[transform.parent_frame][0],
                    outer_rotation=from_root[transform.parent_frame][1],
                    inner_translation=transform.translation,
                    inner_rotation=transform.rotation,
                )
                actual_translation, actual_rotation = from_root[transform.child_frame]
                translation_error = math.dist(predicted_translation, actual_translation)
                rotation_error = quaternion_angle_between(predicted_rotation, actual_rotation)
                # `not (x <= tol)` também captura NaN, que uma comparação `>` deixaria passar.
                if not (
                    translation_error <= translation_tolerance_m
                    and rotation_error <= rotation_tolerance_rad
                ):
                    findings.append(
                        LoopInconsistency(
                            frames=(transform.parent_frame, transform.child_frame),
                            translation_error_m=translation_error,
                            rotation_error_rad=rotation_error,
                        )
                    )
        return tuple(findings)

    def _require_frame(self, frame: FrameId) -> None:
        if frame not in self._adjacency:
            raise FrameGraphError(f"unknown frame {frame!r}: it appears in no static transform")

    def _component_count(self) -> int:
        visited: set[FrameId] = set()
        count = 0
        for root in self._adjacency:
            if root not in visited:
                visited.update(self._traverse(root)[0])
                count += 1
        return count

    def _traverse(
        self, root: FrameId
    ) -> tuple[list[FrameId], dict[FrameId, tuple[Vector3, Quaternion]], dict[FrameId, int]]:
        """Breadth-first spanning tree from ``root``.

        Returns:
            The reached frames, ``T_root_frame`` for each of them, and, for each reached frame
            other than ``root``, the index of the tree edge it was reached through.
        """
        from_root: dict[FrameId, tuple[Vector3, Quaternion]] = {
            root: ((0.0, 0.0, 0.0), _IDENTITY_ROTATION)
        }
        order = [root]
        reached_by: dict[FrameId, int] = {}
        queue = deque([root])
        while queue:
            frame = queue.popleft()
            for step in self._adjacency[frame]:
                if step.neighbor in from_root:
                    continue
                transform = self._transforms[step.edge_index]
                if step.reversed_edge:
                    edge_translation, edge_rotation = invert_rigid(
                        translation=transform.translation, rotation=transform.rotation
                    )
                else:
                    edge_translation, edge_rotation = transform.translation, transform.rotation
                translation, rotation = compose_rigid(
                    outer_translation=from_root[frame][0],
                    outer_rotation=from_root[frame][1],
                    inner_translation=edge_translation,
                    inner_rotation=edge_rotation,
                )
                from_root[step.neighbor] = (translation, rotation)
                reached_by[step.neighbor] = step.edge_index
                order.append(step.neighbor)
                queue.append(step.neighbor)
        return order, from_root, reached_by


def _require_unit_rotation(transform: RigidTransform) -> None:
    """Refuse a transform whose rotation quaternion is not unit within the tolerance.

    Raises:
        FrameGraphError: Naming the transform and the measured norm.
    """
    norm = quaternion_norm(transform.rotation)
    # `not (x <= tol)` também recusa uma norma NaN, que `x > tol` deixaria passar.
    if not abs(norm - 1.0) <= STATIC_ROTATION_NORM_TOLERANCE:
        raise FrameGraphError(
            f"static transform T_{transform.parent_frame}_{transform.child_frame} has a rotation "
            f"quaternion of norm {norm:.6f}, not a unit quaternion within "
            f"{STATIC_ROTATION_NORM_TOLERANCE}: composing or inverting it would give wrong "
            "translations; fix the calibration instead of renormalizing it"
        )

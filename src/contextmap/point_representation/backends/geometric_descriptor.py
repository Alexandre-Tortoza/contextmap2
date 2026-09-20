"""Deterministic geometric descriptor: the geometry-only control encoder.

The descriptor summarizes the local 3D structure of one prepared support with
documented, interpretable quantities derived from its coordinates alone: size,
density and extent, and the eigen-decomposition of the support's covariance
(linearity, planarity, scattering, surface variation, and the direction of the
normal and of the principal axis). It uses no RGB, no visual feature, no label
and no learned model, so it is the control condition against which a learned 3D
representation has to justify itself, and it stays a geometric evidence channel
rather than a semantic classifier.

Everything is plain Python arithmetic on IEEE-754 doubles: sums use
``math.fsum`` (exactly rounded, independent of member order) and the
eigen-decomposition is a cyclic Jacobi iteration with a fixed sweep limit, so
identical input gives an identical vector on one platform and agrees within
about 1e-12 across platforms. No NumPy or model runtime is imported.

Feature definition, version ``1``
    Coordinates are the prepared local coordinates (meters unless the support
    policy normalizes scale). Let ``λ1 >= λ2 >= λ3 >= 0`` be the eigenvalues of
    the population covariance of the members.

    ==== ========================= =============================================
    idx  name                      definition
    ==== ========================= =============================================
    0    ``support_size``          number of members
    1    ``density_per_m3``        ``n / (4/3 π r³)``, ``r`` = farthest member
                                   distance from the center, in meters
    2    ``rms_radius``            ``sqrt(λ1 + λ2 + λ3)``
    3    ``max_radius``            largest member distance to the centroid
    4    ``linearity``             ``(λ1 - λ2) / λ1``
    5    ``planarity``             ``(λ2 - λ3) / λ1``
    6    ``scattering``            ``λ3 / λ1``
    7    ``surface_variation``     ``λ3 / (λ1 + λ2 + λ3)``
    8-10 ``normal_abs_x/y/z``      ``|component|`` of the ``λ3`` eigenvector
    11-13 ``principal_axis_abs_*`` ``|component|`` of the ``λ1`` eigenvector
    ==== ========================= =============================================

    Direction components are absolute values because an eigenvector has no
    sign: this removes the sign ambiguity without an arbitrary convention, and
    it does not assume the map frame's z axis is vertical.

Undefined components
    A component that cannot be defined is declared in ``undefined_components``
    with a zero placeholder; nothing is invented for it:

    * density when the farthest member is at distance zero;
    * the shape features (4-7), the normals and the axes when the rms radius is
      at most :data:`MIN_SUPPORT_EXTENT` (one point, or coincident points);
    * the normal (8-10) when ``λ2 - λ3`` is at most :data:`EIGEN_GAP_TOLERANCE`
      of ``λ1`` (a line, or a scatter with no distinguished normal);
    * the axis (11-13) when ``λ1 - λ2`` is at most that tolerance (an isotropic
      plane or volume has no principal axis).

    A direction is reported only when its eigenvalue is separated from its
    neighbor, so it is numerically unique; how reliable it is as geometry is
    what ``linearity`` and ``planarity`` say.
"""

from __future__ import annotations

import hashlib
import json
import math

from contextmap.point_representation.compatibility import representation_space_fingerprint
from contextmap.point_representation.models import (
    EncoderIdentity,
    PreparedSupport,
    RepresentationSpace,
    SupportPolicy,
    SupportType,
)
from contextmap.point_representation.ports import EncodedVector
from contextmap.shared import Vector3

DESCRIPTOR_VERSION = "1"
"""Version of the feature definition; part of the representation space identity."""

FEATURE_NAMES: tuple[str, ...] = (
    "support_size",
    "density_per_m3",
    "rms_radius",
    "max_radius",
    "linearity",
    "planarity",
    "scattering",
    "surface_variation",
    "normal_abs_x",
    "normal_abs_y",
    "normal_abs_z",
    "principal_axis_abs_x",
    "principal_axis_abs_y",
    "principal_axis_abs_z",
)
"""The ordered feature definition of version ``1``."""

MIN_SUPPORT_EXTENT = 1e-9
"""Rms radius, in prepared-coordinate units, at or below which a support is a single location."""

EIGEN_GAP_TOLERANCE = 1e-6
"""Relative eigenvalue gap (to ``λ1``) below which a direction is not numerically unique."""

_JACOBI_MAX_SWEEPS = 30
_JACOBI_OFF_DIAGONAL_TOLERANCE = 1e-30
"""Relative squared off-diagonal mass at which the Jacobi iteration stops."""

_DENSITY = FEATURE_NAMES.index("density_per_m3")
_SHAPE = range(FEATURE_NAMES.index("linearity"), FEATURE_NAMES.index("surface_variation") + 1)
_NORMAL = range(FEATURE_NAMES.index("normal_abs_x"), FEATURE_NAMES.index("normal_abs_z") + 1)
_AXIS = range(
    FEATURE_NAMES.index("principal_axis_abs_x"), FEATURE_NAMES.index("principal_axis_abs_z") + 1
)


class GeometricDescriptorEncoder:
    """Encodes a prepared support as its geometric descriptor."""

    def __init__(self, support_policy: SupportPolicy) -> None:
        """Bind the descriptor to the support policy it is computed under.

        Args:
            support_policy: The neighborhood policy of the supports it will
                receive; it is part of the representation space identity.

        Raises:
            ValueError: If the policy is a single point, which has no local
                structure to describe.
        """
        if support_policy.support_type is not SupportType.NEIGHBORHOOD:
            raise ValueError(
                "the geometric descriptor needs a neighborhood support policy: a single point "
                "has no local structure"
            )
        self._space = RepresentationSpace(
            family="geometric_descriptor",
            model="local-covariance-shape",
            version=DESCRIPTOR_VERSION,
            checkpoint=None,
            dimension=len(FEATURE_NAMES),
            dtype="float64",
            normalization="none",
            input_definition="xyz-local-prepared",
            support_semantics=support_policy,
            feature_names=FEATURE_NAMES,
        )
        configuration = json.dumps(
            {
                "descriptor_version": DESCRIPTOR_VERSION,
                "eigen_gap_tolerance": EIGEN_GAP_TOLERANCE,
                "min_support_extent": MIN_SUPPORT_EXTENT,
                "representation_space": representation_space_fingerprint(self._space),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._identity = EncoderIdentity(
            backend_id="geometric_descriptor",
            backend_version=DESCRIPTOR_VERSION,
            configuration_fingerprint="sha256:"
            + hashlib.sha256(configuration.encode("utf-8")).hexdigest(),
            checkpoint_hash=None,
        )

    def encoder_identity(self) -> EncoderIdentity:
        """Report the descriptor identity; it has no checkpoint."""
        return self._identity

    def representation_space(self) -> RepresentationSpace:
        """Report the versioned descriptor space under this encoder's support policy."""
        return self._space

    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        """Compute the descriptor of one prepared support.

        Args:
            prepared: A support prepared under this encoder's support policy.

        Returns:
            The 14 features in :data:`FEATURE_NAMES` order, with the components
            that cannot be defined for this support declared undefined.

        Raises:
            ValueError: If the support was prepared under another policy, which
                would label the vector with the wrong representation space.
        """
        if prepared.support.policy != self._space.support_semantics:
            raise ValueError(
                "the support was prepared under a different support policy than the one this "
                "encoder's representation space declares"
            )
        coordinates = prepared.local_coordinates_m
        count = len(coordinates)
        centroid = tuple(
            math.fsum(point[axis] for point in coordinates) / count for axis in range(3)
        )
        deviations = [
            (point[0] - centroid[0], point[1] - centroid[1], point[2] - centroid[2])
            for point in coordinates
        ]
        covariance = [
            [math.fsum(d[row] * d[column] for d in deviations) / count for column in range(3)]
            for row in range(3)
        ]
        rms_radius = math.sqrt(covariance[0][0] + covariance[1][1] + covariance[2][2])

        values = [0.0] * len(FEATURE_NAMES)
        undefined: set[int] = set()
        values[0] = float(count)
        values[2] = rms_radius
        values[3] = max(math.hypot(*deviation) for deviation in deviations)

        farthest_m = prepared.support.statistics.max_distance_m
        volume_m3 = 4.0 / 3.0 * math.pi * farthest_m**3
        if volume_m3 > 0.0:
            values[_DENSITY] = count / volume_m3
        else:
            undefined.add(_DENSITY)

        if rms_radius <= MIN_SUPPORT_EXTENT:
            undefined.update(_SHAPE, _NORMAL, _AXIS)
        else:
            self._describe_shape(covariance, values, undefined)
        return EncodedVector(values=tuple(values), undefined_components=tuple(sorted(undefined)))

    @staticmethod
    def _describe_shape(
        covariance: list[list[float]], values: list[float], undefined: set[int]
    ) -> None:
        (l1, l2, l3), (axis, _, normal) = _eigen_decomposition(covariance)
        values[_SHAPE[0]] = (l1 - l2) / l1
        values[_SHAPE[1]] = (l2 - l3) / l1
        values[_SHAPE[2]] = l3 / l1
        values[_SHAPE[3]] = l3 / (l1 + l2 + l3)
        if l2 - l3 > EIGEN_GAP_TOLERANCE * l1:
            for index, component in zip(_NORMAL, normal, strict=True):
                values[index] = abs(component)
        else:
            undefined.update(_NORMAL)
        if l1 - l2 > EIGEN_GAP_TOLERANCE * l1:
            for index, component in zip(_AXIS, axis, strict=True):
                values[index] = abs(component)
        else:
            undefined.update(_AXIS)


def _eigen_decomposition(
    matrix: list[list[float]],
) -> tuple[tuple[float, float, float], tuple[Vector3, Vector3, Vector3]]:
    """Eigenvalues (descending, clamped at zero) and unit eigenvectors of a symmetric 3x3.

    A cyclic Jacobi iteration: deterministic, with a fixed sweep limit, and
    accurate for the singular and repeated-eigenvalue matrices that degenerate
    supports produce. Eigenvalue ties keep the original index order.
    """
    a = [row[:] for row in matrix]
    vectors = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
    diagonal_mass = a[0][0] ** 2 + a[1][1] ** 2 + a[2][2] ** 2
    for _ in range(_JACOBI_MAX_SWEEPS):
        off_diagonal_mass = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2
        if off_diagonal_mass <= _JACOBI_OFF_DIAGONAL_TOLERANCE * diagonal_mass:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            _rotate(a, vectors, p, q)
    order = sorted(range(3), key=lambda index: -a[index][index])
    eigenvalues = tuple(max(a[index][index], 0.0) for index in order)
    eigenvectors = tuple(
        (vectors[0][index], vectors[1][index], vectors[2][index]) for index in order
    )
    return (
        (eigenvalues[0], eigenvalues[1], eigenvalues[2]),
        (eigenvectors[0], eigenvectors[1], eigenvectors[2]),
    )


def _rotate(a: list[list[float]], vectors: list[list[float]], p: int, q: int) -> None:
    """Zero ``a[p][q]`` with a Jacobi rotation, accumulating it into ``vectors``."""
    apq = a[p][q]
    if apq == 0.0:
        return
    theta = (a[q][q] - a[p][p]) / (2.0 * apq)
    t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
    c = 1.0 / math.sqrt(t * t + 1.0)
    s = t * c
    a[p][p] -= t * apq
    a[q][q] += t * apq
    a[p][q] = a[q][p] = 0.0
    r = 3 - p - q
    arp, arq = a[r][p], a[r][q]
    a[r][p] = a[p][r] = c * arp - s * arq
    a[r][q] = a[q][r] = s * arp + c * arq
    for row in vectors:
        vkp, vkq = row[p], row[q]
        row[p] = c * vkp - s * vkq
        row[q] = s * vkp + c * vkq

"""Canonical contracts for Point Representation.

Point Representation describes the local 3D structure around a persistent
geometry element as a numerical vector. It is a *3D evidence channel*: it is
distinct from the visual evidence Sensor Association projects onto geometry
(``VisualFeature``) and from the semantic hypotheses Visual Perception makes
(``SemanticClaim``). No contract here carries a label, a claim, an entity or a
visual embedding, and nothing concatenates channels into one opaque vector.

Support is part of the identity
    A vector is computed from a *support*, not from a single XYZ. A
    :class:`PointSupport` lists exactly which :class:`GeometryReference` formed
    the input, under which :class:`SupportPolicy`, so a consumer can always
    reconstruct what the vector describes. The policy is also part of the
    :class:`RepresentationSpace`: a vector over a 0.25 m neighborhood is not
    comparable with one over 1 m even from the same encoder.

Comparability
    Two representations are comparable only when their spaces have the same
    fingerprint (see :mod:`contextmap.point_representation.compatibility`).
    Equal dimensionality is never sufficient.

No numerical payload appears here: a representation only *references* its
vector (``payload_reference``), so the contracts are readable with the
standard library alone. Coordinates come from Geometric Mapping in one map
frame, in meters; this module never receives ROS, NumPy or framework tensors.
See ``src/contextmap/point_representation/docs/contracts.md`` for the field
reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import NewType

from contextmap.geometric_mapping import GeometryReference
from contextmap.shared import Vector3

PointRepresentationRunId = NewType("PointRepresentationRunId", str)
"""Identity of one immutable Point Representation run artifact."""

PointRepresentationId = NewType("PointRepresentationId", str)
"""Identity of one representation, local to its run."""

SUPPORTED_DTYPES = ("float32", "float64")
"""Payload element types a representation may declare."""

_ITEM_SIZE_BYTES = {"float32": 4, "float64": 8}


def representation_id_for(*, run_id: PointRepresentationRunId, index: int) -> PointRepresentationId:
    """Compute the deterministic identity of the ``index``-th representation of a run.

    Args:
        run_id: The run that owns the representation.
        index: Zero-based position of the representation in the run.

    Returns:
        A pure function of the inputs, so references survive serialization and
        artifact round-trips without an identity registry.
    """
    return PointRepresentationId(f"{run_id}--repr-{index:09d}")


class SupportType(Enum):
    """What extent of geometry a representation describes.

    Attributes:
        POINT: The center point alone.
        NEIGHBORHOOD: The center plus the geometry around it, selected by a
            :class:`NeighborhoodMethod`.
    """

    POINT = "point"
    NEIGHBORHOOD = "neighborhood"


class NeighborhoodMethod(Enum):
    """How a neighborhood support selects its members.

    In both methods the center is a member of its own support and counts
    toward the limit.

    Attributes:
        RADIUS: Every element within a Euclidean radius of the center.
        K_NEAREST: The ``k`` elements nearest to the center.
    """

    RADIUS = "radius"
    K_NEAREST = "k_nearest"


class CenteringMode(Enum):
    """Which point local coordinates are translated relative to.

    Attributes:
        NONE: Map-frame coordinates, untranslated.
        CENTER: Relative to the support's center element.
        CENTROID: Relative to the mean of the support's members.
    """

    NONE = "none"
    CENTER = "center"
    CENTROID = "centroid"


class ScaleNormalization(Enum):
    """How local coordinates are rescaled after centering.

    Attributes:
        NONE: Coordinates stay in meters.
        SUPPORT_RADIUS: Coordinates are divided by the policy radius, so a
            radius support spans a unit ball; only valid for a radius method.
    """

    NONE = "none"
    SUPPORT_RADIUS = "support_radius"


@dataclass(frozen=True, kw_only=True)
class CoordinatePreparation:
    """The deterministic transform applied to support coordinates before encoding.

    Every normalization is recorded here so it can be undone or audited; an
    orientation canonicalization is deliberately not offered until it has an
    explicit definition.

    Attributes:
        centering: Which point local coordinates are relative to.
        scale_normalization: Whether and how they are rescaled.
    """

    centering: CenteringMode = CenteringMode.CENTER
    scale_normalization: ScaleNormalization = ScaleNormalization.NONE


@dataclass(frozen=True, kw_only=True)
class SupportPolicy:
    """The rule that selects and prepares the geometry a representation is computed from.

    Attributes:
        support_type: A single point or a neighborhood.
        method: Neighborhood selection; ``None`` exactly for a point support.
        radius_m: Radius in meters of a radius method; ``None`` otherwise.
        k: Number of members, center included, of a k-nearest method; ``None``
            otherwise.
        max_neighbors: Deterministic cap on a radius support's members, keeping
            the nearest; ``None`` for no cap. A k-nearest method is already
            capped by ``k``.
        preparation: How the selected coordinates are prepared.
    """

    support_type: SupportType
    method: NeighborhoodMethod | None
    radius_m: float | None
    k: int | None
    max_neighbors: int | None
    preparation: CoordinatePreparation

    def __post_init__(self) -> None:
        """Validate that the parameters match the support type and method.

        Raises:
            ValueError: If a parameter is missing, superfluous, non-positive or
                not finite for the declared type and method, or a scale
                normalization is not defined for it.
        """
        if self.support_type is SupportType.POINT:
            self._require_point_policy()
        elif self.method is NeighborhoodMethod.RADIUS:
            self._require_radius_policy()
        elif self.method is NeighborhoodMethod.K_NEAREST:
            self._require_k_nearest_policy()
        else:
            raise ValueError("a neighborhood support policy needs a method")

    def _require_point_policy(self) -> None:
        if (
            self.method is not None
            or self.radius_m is not None
            or self.k is not None
            or self.max_neighbors is not None
        ):
            raise ValueError("a point support policy takes no method, radius, k or cap")
        if self.preparation.scale_normalization is not ScaleNormalization.NONE:
            raise ValueError("a point support policy cannot normalize scale")

    def _require_radius_policy(self) -> None:
        if self.radius_m is None or not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError(
                f"a radius support policy needs a finite positive radius_m, got {self.radius_m!r}"
            )
        if self.k is not None:
            raise ValueError("a radius support policy takes no k")
        if self.max_neighbors is not None and self.max_neighbors < 1:
            raise ValueError("a support policy max_neighbors must be at least 1")

    def _require_k_nearest_policy(self) -> None:
        if self.k is None or self.k < 1:
            raise ValueError(f"a k-nearest support policy needs k >= 1, got {self.k!r}")
        if self.radius_m is not None or self.max_neighbors is not None:
            raise ValueError(
                "a k-nearest support policy takes no radius or cap; k already bounds it"
            )
        if self.preparation.scale_normalization is ScaleNormalization.SUPPORT_RADIUS:
            raise ValueError("a k-nearest support policy has no radius to normalize scale by")


@dataclass(frozen=True, kw_only=True)
class SupportStatistics:
    """Summary of how a support is spread around its center.

    Distances are Euclidean, in meters, in the map frame, measured from the
    center element (which is at distance zero from itself).

    Attributes:
        count: Members of the support, center included.
        min_distance_m: Smallest member distance.
        max_distance_m: Largest member distance.
        mean_distance_m: Mean member distance.
        near_map_bounds: Whether the support's full extent reaches past the map
            bounds, so it may be truncated by the edge of the map.
        candidate_count: Members that qualified before a ``max_neighbors`` cap;
            equal to ``count`` when nothing was cut, ``None`` when not tracked.
    """

    count: int
    min_distance_m: float
    max_distance_m: float
    mean_distance_m: float
    near_map_bounds: bool
    candidate_count: int | None

    def __post_init__(self) -> None:
        """Validate ordering and finiteness of the summary.

        Raises:
            ValueError: If the count is not positive, a distance is not finite
                or negative, they are not ordered ``min <= mean <= max``, or the
                candidate count is below the count.
        """
        distances = (self.min_distance_m, self.mean_distance_m, self.max_distance_m)
        if self.count < 1:
            raise ValueError("support statistics need at least one member")
        if not all(math.isfinite(value) for value in distances):
            raise ValueError("support statistics distances must be finite")
        if not 0 <= self.min_distance_m <= self.mean_distance_m <= self.max_distance_m:
            raise ValueError(
                f"support statistics must satisfy 0 <= min <= mean <= max, got {distances!r}"
            )
        if self.candidate_count is not None and self.candidate_count < self.count:
            raise ValueError("support statistics candidate_count cannot be below count")


@dataclass(frozen=True, kw_only=True)
class PointSupport:
    """Exactly which geometry a representation was computed from.

    Attributes:
        policy: The rule that produced the support.
        center: The geometry element the representation is anchored to.
        geometry_refs: The supporting elements, center included, in the order
            their prepared coordinates are presented to an encoder.
        map_frame: Frame the geometry is expressed in.
        statistics: How the members are spread around the center.
        query_method: Identity of the spatial query that selected the
            candidates, so a result can be reproduced or compared.
    """

    policy: SupportPolicy
    center: GeometryReference
    geometry_refs: tuple[GeometryReference, ...]
    map_frame: str
    statistics: SupportStatistics
    query_method: str

    def __post_init__(self) -> None:
        """Validate that the support is reconstructable and consistent with its policy.

        Raises:
            ValueError: If the center is not a member, a member repeats or
                belongs to another map, the statistics disagree with the members
                or the policy, or the frame or query method is missing.
        """
        if not self.map_frame or not self.query_method:
            raise ValueError("a support needs a map_frame and a query_method")
        if self.center not in self.geometry_refs:
            raise ValueError("a support must contain its center among geometry_refs")
        if len(set(self.geometry_refs)) != len(self.geometry_refs):
            raise ValueError("a support must not repeat a geometry reference")
        if any(reference.map_id != self.center.map_id for reference in self.geometry_refs):
            raise ValueError("every support geometry reference must belong to the center's map")
        if self.statistics.count != len(self.geometry_refs):
            raise ValueError("support statistics count must equal the number of geometry_refs")
        self._require_policy_limits()

    def _require_policy_limits(self) -> None:
        policy = self.policy
        count = len(self.geometry_refs)
        if policy.support_type is SupportType.POINT and count != 1:
            raise ValueError("a point support contains only its center")
        if policy.k is not None and count > policy.k:
            raise ValueError(f"a support of {count} members exceeds the policy k of {policy.k}")
        if policy.max_neighbors is not None and count > policy.max_neighbors:
            raise ValueError(
                f"a support of {count} members exceeds the policy max_neighbors "
                f"of {policy.max_neighbors}"
            )
        if policy.radius_m is not None and self.statistics.max_distance_m > policy.radius_m:
            raise ValueError(
                f"a support reaches {self.statistics.max_distance_m} m, beyond the policy radius "
                f"of {policy.radius_m} m"
            )

    @property
    def applied_scale_m(self) -> float | None:
        """The divisor applied to local coordinates, in meters; ``None`` when not scaled."""
        if self.policy.preparation.scale_normalization is ScaleNormalization.SUPPORT_RADIUS:
            return self.policy.radius_m
        return None


@dataclass(frozen=True, kw_only=True)
class PreparedSupport:
    """A support with its coordinates prepared for an encoder.

    The local coordinates never replace the references: the support still says
    which persistent geometry they came from.

    Attributes:
        support: The support the coordinates were prepared from.
        local_coordinates_m: One ``(x, y, z)`` per element of
            ``support.geometry_refs``, in the same order, after the support
            policy's centering and scale normalization. Meters unless
            ``support.applied_scale_m`` is set, in which case they are
            multiples of it.
    """

    support: PointSupport
    local_coordinates_m: tuple[Vector3, ...]

    def __post_init__(self) -> None:
        """Validate one finite coordinate per supporting element.

        Raises:
            ValueError: If the counts differ or a coordinate is not finite.
        """
        if len(self.local_coordinates_m) != len(self.support.geometry_refs):
            raise ValueError(
                f"a prepared support needs one coordinate per geometry reference, got "
                f"{len(self.local_coordinates_m)} for {len(self.support.geometry_refs)}"
            )
        if not all(math.isfinite(value) for point in self.local_coordinates_m for value in point):
            raise ValueError("prepared support coordinates must be finite")


@dataclass(frozen=True, kw_only=True)
class RepresentationSpace:
    """Identity of the vector space a :class:`PointRepresentation` lives in.

    Two spaces are the same identity if and only if every field is equal; the
    deterministic string this reduces to is the space fingerprint. The support
    policy is part of the identity because it defines what each vector
    describes.

    Attributes:
        family: Producer family, e.g. ``"geometric_descriptor"`` or ``"ptv3"``.
        model: Model or descriptor identity within the family.
        version: Version of the model or descriptor definition.
        checkpoint: Checkpoint identity for a learned encoder; ``None`` for a
            deterministic descriptor.
        dimension: Vector dimensionality.
        dtype: Element type of the payload; one of :data:`SUPPORTED_DTYPES`.
        normalization: Normalization applied to the vector before persistence,
            e.g. ``"none"`` or ``"l2"``.
        input_definition: What the encoder receives, e.g. ``"xyz-local-prepared"``.
        support_semantics: The support policy the vectors are computed under.
        feature_names: Meaning of each component for an interpretable
            descriptor, in order; empty for a learned encoder.
    """

    family: str
    model: str
    version: str
    checkpoint: str | None = None
    dimension: int
    dtype: str
    normalization: str
    input_definition: str
    support_semantics: SupportPolicy
    feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the identity fields.

        Raises:
            ValueError: If a text field is empty, the dimension is not positive,
                the dtype is unsupported, or ``feature_names`` is neither empty
                nor one distinct name per component.
        """
        if not all((self.family, self.model, self.version, self.normalization)):
            raise ValueError(
                "a representation space needs family, model, version and normalization"
            )
        if not self.input_definition:
            raise ValueError("a representation space needs an input_definition")
        if self.dimension <= 0:
            raise ValueError(
                f"a representation space dimension must be positive, got {self.dimension}"
            )
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"a representation space dtype must be one of {SUPPORTED_DTYPES}, "
                f"got {self.dtype!r}"
            )
        if self.feature_names and (
            len(self.feature_names) != self.dimension
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError(
                "representation space feature_names must be empty or one distinct name per "
                "component"
            )

    @property
    def vector_bytes(self) -> int:
        """Bytes one stored vector of this space occupies in a payload."""
        return self.dimension * _ITEM_SIZE_BYTES[self.dtype]


@dataclass(frozen=True, kw_only=True)
class EncoderIdentity:
    """Which implementation produced a representation.

    Attributes:
        backend_id: Backend identity, e.g. ``"geometric_descriptor"``.
        backend_version: Backend implementation version.
        configuration_fingerprint: Fingerprint of the effective backend
            configuration; never contains secrets.
        checkpoint_hash: Hash of the checkpoint weights for a learned encoder;
            ``None`` for a deterministic descriptor.
    """

    backend_id: str
    backend_version: str
    configuration_fingerprint: str
    checkpoint_hash: str | None = None

    def __post_init__(self) -> None:
        """Validate the identity is complete.

        Raises:
            ValueError: If the backend, its version or the configuration
                fingerprint is empty, or a given checkpoint hash is empty.
        """
        if not (self.backend_id and self.backend_version and self.configuration_fingerprint):
            raise ValueError(
                "encoder identity needs backend_id, backend_version and configuration_fingerprint"
            )
        if self.checkpoint_hash is not None and not self.checkpoint_hash:
            raise ValueError("encoder identity checkpoint_hash must not be empty when given")


@dataclass(frozen=True, kw_only=True)
class RepresentationProvenance:
    """Where a representation's code came from.

    The geometry lineage lives in the support and the geometry map; the
    encoder in :class:`EncoderIdentity`. Run-level lineage (input artifacts,
    configuration) is recorded by the run artifact, not repeated per vector.

    Attributes:
        code_version: Identity of the code that produced the representation.
    """

    code_version: str

    def __post_init__(self) -> None:
        """Validate the code version is present.

        Raises:
            ValueError: If ``code_version`` is empty.
        """
        if not self.code_version:
            raise ValueError("representation provenance needs a code_version")


@dataclass(frozen=True, kw_only=True)
class PointRepresentation:
    """A numerical description of the local 3D structure around one geometry element.

    The vector itself is not embedded: ``payload_reference`` says where it is
    stored, and ``shape``, ``dtype`` and ``normalization`` describe it. It is
    3D evidence, not a label, a claim or a visual feature, and it is not
    semantic truth.

    Attributes:
        representation_id: Identity of this representation, local to its run.
        geometry_reference: The persistent geometry element it is anchored to;
            always the center of ``support``.
        support: Exactly which geometry the vector was computed from.
        representation_space_id: Fingerprint of the
            :class:`RepresentationSpace`; representations are comparable only
            when this is equal.
        shape: Shape of the stored vector.
        dtype: Element type of the stored vector.
        normalization: Normalization applied to the stored vector.
        payload_reference: Run-relative reference to the stored vector;
            ``None`` when the payload was not stored.
        encoder_identity: The encoder that produced it.
        provenance: Where the producing code came from.
        undefined_components: Ascending indexes of components the encoder could
            not define (e.g. a curvature of a degenerate support). Their stored
            values are placeholders that must never be interpreted.
    """

    representation_id: PointRepresentationId
    geometry_reference: GeometryReference
    support: PointSupport
    representation_space_id: str
    shape: tuple[int, ...]
    dtype: str
    normalization: str
    payload_reference: str | None
    encoder_identity: EncoderIdentity
    provenance: RepresentationProvenance
    undefined_components: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate anchoring, payload description and undefined components.

        Raises:
            ValueError: If the anchor is not the support center, the space or
                normalization is missing, the shape or dtype is invalid, the
                payload reference is empty, absolute or escapes the run, or an
                undefined component is out of range or not ascending.
        """
        if self.geometry_reference != self.support.center:
            raise ValueError("representation geometry_reference must be the center of its support")
        if not self.representation_space_id or not self.normalization:
            raise ValueError("a representation needs a representation_space_id and a normalization")
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError(
                f"a representation shape must be positive in every axis, got {self.shape!r}"
            )
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"a representation dtype must be one of {SUPPORTED_DTYPES}, got {self.dtype!r}"
            )
        if self.payload_reference is not None:
            _require_relative_reference(self.payload_reference)
        self._require_valid_undefined_components()

    def _require_valid_undefined_components(self) -> None:
        size = math.prod(self.shape)
        indexes = self.undefined_components
        if any(not 0 <= index < size for index in indexes) or any(
            earlier >= later for earlier, later in pairwise(indexes)
        ):
            raise ValueError(
                f"undefined_components must be ascending indexes below {size}, got {indexes!r}"
            )

    @property
    def is_partial(self) -> bool:
        """Whether some components are undefined and must not be interpreted."""
        return bool(self.undefined_components)


class FailureReason(Enum):
    """Why a support produced no representation.

    Attributes:
        UNENCODABLE_SUPPORT: The encoder declared the support cannot be
            represented, e.g. too few points for a learned model.
        NON_FINITE_OUTPUT: The encoder returned NaN or infinity in a component
            it did not declare undefined.
    """

    UNENCODABLE_SUPPORT = "unencodable_support"
    NON_FINITE_OUTPUT = "non_finite_output"


@dataclass(frozen=True, kw_only=True)
class FailedSupport:
    """A support that could not be represented, recorded instead of a default vector.

    A failure is never turned into a zero or default representation: it is an
    explicit outcome that keeps which geometry was involved and why.

    Attributes:
        support: The support the encoder was asked to represent.
        reason: Why no representation exists.
        detail: The encoder's own account of the failure.
    """

    support: PointSupport
    reason: FailureReason
    detail: str

    def __post_init__(self) -> None:
        """Require an explanation.

        Raises:
            ValueError: If ``detail`` is empty.
        """
        if not self.detail:
            raise ValueError("a failed support needs a detail explaining the failure")


def _require_relative_reference(reference: str) -> None:
    """Reject a payload reference that is empty, absolute or leaves the run directory."""
    path = reference.split("#", maxsplit=1)[0]
    parts = path.split("/")
    if not path or path.startswith("/") or ".." in parts:
        raise ValueError(
            f"a representation payload_reference must be a relative path inside the run, "
            f"got {reference!r}"
        )

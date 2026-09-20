"""Geometry preflight: are the frames, extrinsics and clocks a run needs trustworthy?

Geometry failures are often caused not by the estimator but by missing or
inconsistent frame semantics, static extrinsics or clock metadata. The
preflight fails a run *before* the estimator starts when what that run needs
is missing or invalid, and it records which later capabilities still lack
prerequisites without pretending they are validated.

Requirements are declared per capability, so a LiDAR-inertial run is never
blocked for missing camera intrinsics: those are reported as downstream
readiness instead. The preflight never owns, parses or rewrites calibration
(Ingestion does), never hardcodes a rig or dataset, and does not claim that a
camera-LiDAR calibration is physically correct just because its matrices are
valid: reprojection validation belongs to Sensor Association.

Conventions the checks assume, and the report states: transforms follow
``T_parent_child`` (``p_parent = R * p_child + t``), translations are in
meters and rotations are unit quaternions ordered ``(x, y, z, w)``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from contextmap.ingestion import (
    MODALITY_NAMES,
    CalibrationReferenceId,
    CalibrationSet,
    FrameId,
    RigidTransform,
    SourceObservation,
    observation_modality,
)
from contextmap.shared import (
    RotationMatrix,
    compose_rigid,
    invert_rigid,
    quaternion_angle_between,
    quaternion_norm,
    quaternion_to_rotation_matrix,
)
from contextmap.state_estimation.frame_graph import StaticFrameGraph

BODY_ENDPOINT = "body"
"""Endpoint naming the body frame of the dynamic transform, instead of a modality."""

CONVENTIONS = {
    "transform": "T_parent_child",
    "rotation": "unit quaternion (x, y, z, w)",
    "translation_unit": "m",
}
"""Conventions the checks assume; stated so a report is self-describing."""

_ENDPOINTS = MODALITY_NAMES | {BODY_ENDPOINT}


@dataclass(frozen=True, kw_only=True)
class StaticRelationRequirement:
    """A static transform a capability needs between two frames.

    Attributes:
        from_endpoint: A modality name (``"lidar"``, ``"imu"``, ``"image"``,
            ``"external_pose"``) or :data:`BODY_ENDPOINT`.
        to_endpoint: A modality name or :data:`BODY_ENDPOINT`.
    """

    from_endpoint: str
    to_endpoint: str

    def __post_init__(self) -> None:
        """Reject endpoints that are neither a modality nor the body frame.

        Raises:
            ValueError: If an endpoint is unknown.
        """
        for endpoint in (self.from_endpoint, self.to_endpoint):
            if endpoint not in _ENDPOINTS:
                raise ValueError(
                    f"unknown endpoint {endpoint!r}, expected one of {sorted(_ENDPOINTS)}"
                )


@dataclass(frozen=True, kw_only=True)
class GeometryRequirements:
    """What one capability needs from the sequence and its calibration.

    Attributes:
        capability: Name shown in reports, e.g. ``"state_estimation:external_pose"``.
        modalities: Observation modalities that must be present.
        static_relations: Static transforms that must exist between frames.
        camera_model_modalities: Modalities whose observations must reference
            a calibration entry with a camera model (intrinsics).
        reference_frame: For the *executing* capability only, the reference
            frame of the dynamic transform it publishes.
        body_frame: For the executing capability only, the body frame of the
            dynamic transform; :data:`BODY_ENDPOINT` resolves to it.
    """

    capability: str
    modalities: frozenset[str]
    static_relations: tuple[StaticRelationRequirement, ...] = ()
    camera_model_modalities: frozenset[str] = frozenset()
    reference_frame: FrameId | None = None
    body_frame: FrameId | None = None

    def __post_init__(self) -> None:
        """Reject unknown modality names.

        Raises:
            ValueError: If a modality is not one of the canonical modalities.
        """
        unknown = (self.modalities | self.camera_model_modalities) - MODALITY_NAMES
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")


@dataclass(frozen=True, kw_only=True)
class PreflightTolerances:
    """Documented numeric tolerances of the transform checks.

    Attributes:
        rotation_orthonormality: Largest ``|norm(q) - 1|``, ``|R^T R - I|``
            entry and ``|det(R) - 1|`` accepted for a static rotation. The
            default tolerates text-precision quaternions while rejecting a
            rotation that is not a rotation.
        inverse_round_trip_translation_m: Largest translation left by
            ``T * T^-1``, in meters.
        inverse_round_trip_rotation_rad: Largest rotation left by ``T * T^-1``.
        loop_translation_m: Largest position disagreement between redundant
            paths, in meters (1 mm by default).
        loop_rotation_rad: Largest rotation disagreement between redundant paths.
    """

    rotation_orthonormality: float = 1e-5
    inverse_round_trip_translation_m: float = 1e-9
    inverse_round_trip_rotation_rad: float = 1e-9
    loop_translation_m: float = 1e-3
    loop_rotation_rad: float = 1e-3


class PreflightStatus(Enum):
    """Whether the run may start."""

    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True, kw_only=True)
class PreflightFinding:
    """One reason a run is blocked, or one caveat worth recording.

    Attributes:
        code: Stable identifier, e.g. ``"missing_static_transform"``.
        message: Human-readable explanation naming the frames or inputs involved.
    """

    code: str
    message: str


@dataclass(frozen=True, kw_only=True)
class TransformCheck:
    """Result of one numeric or availability check on a static transform.

    Attributes:
        subject: What was checked, e.g. ``"T_imu_velodyne"`` or ``"imu<-velodyne"``.
        check: ``"finite"``, ``"rotation_orthonormal"``, ``"inverse_round_trip"``,
            ``"relation_available"`` or ``"loop_consistency"``.
        passed: Whether the check passed.
        detail: The numbers behind the result.
    """

    subject: str
    check: str
    passed: bool
    detail: str


@dataclass(frozen=True, kw_only=True)
class ClockCheck:
    """Result of one clock/timestamp metadata check.

    Attributes:
        subject: What was checked, e.g. ``"modality:lidar"``.
        passed: Whether the check passed.
        detail: The clock identities involved.
    """

    subject: str
    passed: bool
    detail: str


@dataclass(frozen=True, kw_only=True)
class FrameGraphSummary:
    """The frames the run relies on.

    Attributes:
        reference_frame: Reference frame of the dynamic transform.
        body_frame: Body frame of the dynamic transform.
        static_frames: Frames connected by static calibration transforms, sorted.
        static_edges: ``(parent, child)`` of each static transform.
        component_count: Independent groups of statically connected frames.
    """

    reference_frame: FrameId
    body_frame: FrameId
    static_frames: tuple[FrameId, ...]
    static_edges: tuple[tuple[FrameId, FrameId], ...]
    component_count: int


@dataclass(frozen=True, kw_only=True)
class DownstreamReadiness:
    """Whether a later capability's prerequisites exist, without blocking this run.

    Attributes:
        capability: The later capability, e.g. ``"sensor_association"``.
        ready: ``True`` when every prerequisite is present and valid.
        missing: What is missing or invalid; empty when ``ready``.
    """

    capability: str
    ready: bool
    missing: tuple[PreflightFinding, ...]


@dataclass(frozen=True, kw_only=True)
class GeometryPreflightReport:
    """Structured readiness of a run's geometry prerequisites.

    Attributes:
        required_inputs: Modalities the executing capability needs.
        available_inputs: Modalities present in the observations.
        frame_graph: The frames the run relies on.
        calibration_identity: Hash of the calibration used, or ``None`` when
            there is none.
        transform_checks: Every transform and relation check performed.
        clock_checks: Every clock/timestamp metadata check performed.
        blockers: Reasons the run must not start.
        warnings: Caveats that do not block, including downstream gaps.
        downstream: Readiness of later capabilities.
        conventions: Conventions the checks assume.
    """

    required_inputs: frozenset[str]
    available_inputs: frozenset[str]
    frame_graph: FrameGraphSummary
    calibration_identity: str | None
    transform_checks: tuple[TransformCheck, ...]
    clock_checks: tuple[ClockCheck, ...]
    blockers: tuple[PreflightFinding, ...]
    warnings: tuple[PreflightFinding, ...]
    downstream: tuple[DownstreamReadiness, ...]
    conventions: dict[str, str]

    @property
    def status(self) -> PreflightStatus:
        """``BLOCKED`` if there is any blocker, otherwise ``READY``."""
        return PreflightStatus.BLOCKED if self.blockers else PreflightStatus.READY


def calibration_identity(calibration: CalibrationSet | None) -> str | None:
    """Compute a deterministic identity of the calibration a run used.

    Args:
        calibration: The canonical calibration, or ``None``.

    Returns:
        ``"sha256:<hex digest>"`` over the static transforms and the entry
        content hashes, independent of their order; ``None`` without calibration.
    """
    if calibration is None:
        return None
    payload = {
        "schema_version": calibration.schema_version,
        "static_transforms": sorted(
            [str(t.parent_frame), str(t.child_frame), list(t.translation), list(t.rotation)]
            for t in calibration.static_transforms
        ),
        "entries": sorted(
            [str(entry_id), entry.content_hash] for entry_id, entry in calibration.entries.items()
        ),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class _Facts:
    """What the observations say about modalities, frames, clocks and calibration ids."""

    modalities: frozenset[str]
    frames: dict[str, set[FrameId]]
    clocks: dict[str, set[str]]
    calibration_ids: dict[str, list[CalibrationReferenceId | None]]


def _gather_facts(observations: Iterable[SourceObservation]) -> _Facts:
    frames: dict[str, set[FrameId]] = {}
    clocks: dict[str, set[str]] = {}
    calibration_ids: dict[str, list[CalibrationReferenceId | None]] = {}
    for observation in observations:
        modality = observation_modality(observation)
        frames.setdefault(modality, set()).add(observation.frame_id)
        clocks.setdefault(modality, set()).add(observation.timestamp.clock_id)
        calibration_ids.setdefault(modality, []).append(observation.calibration_id)
    return _Facts(
        modalities=frozenset(frames),
        frames=frames,
        clocks=clocks,
        calibration_ids=calibration_ids,
    )


def _transform_checks(
    transforms: Sequence[RigidTransform], tolerances: PreflightTolerances
) -> tuple[list[TransformCheck], set[int]]:
    """Check every static transform; return the checks and the indexes that failed."""
    checks: list[TransformCheck] = []
    failed: set[int] = set()
    for index, transform in enumerate(transforms):
        subject = f"T_{transform.parent_frame}_{transform.child_frame}"
        values = (*transform.translation, *transform.rotation)
        finite = all(math.isfinite(value) for value in values)
        checks.append(
            TransformCheck(
                subject=subject,
                check="finite",
                passed=finite,
                detail="all values finite" if finite else f"non-finite values in {values!r}",
            )
        )
        if not finite:
            failed.add(index)
            continue

        # A matriz sozinha não denuncia um quaternion sem norma unitária cuja parte vetorial é
        # nula (vira a identidade), então a norma entra na mesma medida de erro.
        error = max(
            abs(quaternion_norm(transform.rotation) - 1.0),
            _orthonormality_error(quaternion_to_rotation_matrix(transform.rotation)),
        )
        orthonormal = error <= tolerances.rotation_orthonormality
        checks.append(
            TransformCheck(
                subject=subject,
                check="rotation_orthonormal",
                passed=orthonormal,
                detail=(
                    f"max(||q|| - 1, |R^T R - I|, |det R - 1|) = {error:.3e}, "
                    f"tolerance {tolerances.rotation_orthonormality:.1e}"
                ),
            )
        )
        if not orthonormal:
            failed.add(index)
            continue

        inverse_translation, inverse_rotation = invert_rigid(
            translation=transform.translation, rotation=transform.rotation
        )
        round_trip_translation, round_trip_rotation = compose_rigid(
            outer_translation=transform.translation,
            outer_rotation=transform.rotation,
            inner_translation=inverse_translation,
            inner_rotation=inverse_rotation,
        )
        translation_left = math.hypot(*round_trip_translation)
        rotation_left = quaternion_angle_between((0.0, 0.0, 0.0, 1.0), round_trip_rotation)
        round_trip = (
            translation_left <= tolerances.inverse_round_trip_translation_m
            and rotation_left <= tolerances.inverse_round_trip_rotation_rad
        )
        checks.append(
            TransformCheck(
                subject=subject,
                check="inverse_round_trip",
                passed=round_trip,
                detail=f"T * T^-1 leaves {translation_left:.3e} m and {rotation_left:.3e} rad",
            )
        )
        if not round_trip:
            failed.add(index)
    return checks, failed


def _orthonormality_error(matrix: RotationMatrix) -> float:
    """Return the largest deviation of a matrix from a proper rotation."""
    worst = 0.0
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[k][i] * matrix[k][j] for k in range(3))
            worst = max(worst, abs(dot - (1.0 if i == j else 0.0)))
    row0, row1, row2 = matrix
    cross = (
        row1[1] * row2[2] - row1[2] * row2[1],
        row1[2] * row2[0] - row1[0] * row2[2],
        row1[0] * row2[1] - row1[1] * row2[0],
    )
    determinant = sum(a * b for a, b in zip(row0, cross, strict=True))
    return max(worst, abs(determinant - 1.0))


def _relation_findings(
    requirements: GeometryRequirements,
    *,
    facts: _Facts,
    calibration: CalibrationSet | None,
    graph: StaticFrameGraph,
    body_frame: FrameId,
) -> tuple[list[PreflightFinding], list[TransformCheck], set[FrameId]]:
    """Evaluate one capability's requirements; return findings, checks and used frames."""
    findings: list[PreflightFinding] = []
    checks: list[TransformCheck] = []
    touched: set[FrameId] = set()

    for modality in sorted(requirements.modalities - facts.modalities):
        findings.append(
            PreflightFinding(
                code="missing_input",
                message=f"{requirements.capability} requires {modality!r} observations, "
                "but the selection has none",
            )
        )

    def endpoint_frame(endpoint: str) -> FrameId | None:
        if endpoint == BODY_ENDPOINT:
            return body_frame
        frames = facts.frames.get(endpoint, set())
        if len(frames) > 1:
            findings.append(
                PreflightFinding(
                    code="ambiguous_endpoint",
                    message=f"modality {endpoint!r} spans several frames "
                    f"{sorted(map(str, frames))}, so a static relation to it is ambiguous",
                )
            )
            return None
        # Sem observações da modalidade o problema já foi reportado como `missing_input`.
        return next(iter(frames), None)

    for relation in requirements.static_relations:
        first = endpoint_frame(relation.from_endpoint)
        second = endpoint_frame(relation.to_endpoint)
        if first is None or second is None:
            continue
        subject = f"{first}<-{second}"
        if first == second:
            checks.append(
                TransformCheck(
                    subject=subject,
                    check="relation_available",
                    passed=True,
                    detail="same frame, identity",
                )
            )
            continue
        if calibration is None:
            findings.append(
                PreflightFinding(
                    code="missing_calibration",
                    message=f"{requirements.capability} needs the static transform between "
                    f"{first!r} and {second!r}, but the sequence has no calibration",
                )
            )
            continue
        connected = first in graph.frames and second in graph.component_of(first)
        checks.append(
            TransformCheck(
                subject=subject,
                check="relation_available",
                passed=connected,
                detail="static path found" if connected else "no static path",
            )
        )
        if not connected:
            findings.append(
                PreflightFinding(
                    code="missing_static_transform",
                    message=f"{requirements.capability} needs a static transform between "
                    f"{first!r} and {second!r}, but calibration has no path connecting them",
                )
            )
            continue
        touched |= graph.component_of(first)

    for modality in sorted(requirements.camera_model_modalities & facts.modalities):
        entries = calibration.entries if calibration is not None else {}
        lacks_model = any(
            calibration_id is None
            or calibration_id not in entries
            or entries[calibration_id].camera_model is None
            for calibration_id in facts.calibration_ids[modality]
        )
        if lacks_model:
            findings.append(
                PreflightFinding(
                    code="missing_camera_model",
                    message=f"{requirements.capability} needs a camera model (intrinsics) for "
                    f"{modality!r}, but an observation has no calibration entry with one",
                )
            )
    return findings, checks, touched


def run_geometry_preflight(
    *,
    observations: Iterable[SourceObservation],
    calibration: CalibrationSet | None,
    execution: GeometryRequirements,
    downstream: Sequence[GeometryRequirements] = (),
    tolerances: PreflightTolerances | None = None,
) -> GeometryPreflightReport:
    """Check that the frames, extrinsics and clocks a run needs are trustworthy.

    Only the requirements of ``execution`` can block. ``downstream``
    capabilities are evaluated the same way but reported as readiness, so a
    LiDAR-inertial run is not blocked for missing camera intrinsics.

    Args:
        observations: The selected canonical observations.
        calibration: The sequence's canonical calibration, or ``None``.
        execution: What the capability about to run needs; it must declare
            ``reference_frame`` and ``body_frame``.
        downstream: What later capabilities will need.
        tolerances: Numeric tolerances; defaults are documented on
            :class:`PreflightTolerances`.

    Returns:
        The report; ``status`` is ``BLOCKED`` when the run must not start.

    Raises:
        ValueError: If ``execution`` does not declare its dynamic frames.
    """
    if execution.reference_frame is None or execution.body_frame is None:
        raise ValueError("the executing capability must declare its reference_frame and body_frame")
    limits = tolerances if tolerances is not None else PreflightTolerances()
    body_frame = execution.body_frame

    facts = _gather_facts(observations)
    transforms = calibration.static_transforms if calibration is not None else ()
    graph = StaticFrameGraph(transforms)
    transform_checks, failed_indexes = _transform_checks(transforms, limits)
    valid_graph = StaticFrameGraph(
        [t for index, t in enumerate(transforms) if index not in failed_indexes]
    )
    loops = valid_graph.loop_inconsistencies(
        translation_tolerance_m=limits.loop_translation_m,
        rotation_tolerance_rad=limits.loop_rotation_rad,
    )
    for loop in loops:
        transform_checks.append(
            TransformCheck(
                subject=f"{loop.frames[0]}<->{loop.frames[1]}",
                check="loop_consistency",
                passed=False,
                detail=f"redundant paths disagree by {loop.translation_error_m:.3e} m "
                f"and {loop.rotation_error_rad:.3e} rad",
            )
        )
    if valid_graph.loop_count > 0 and not loops:
        transform_checks.append(
            TransformCheck(
                subject="static frame graph",
                check="loop_consistency",
                passed=True,
                detail=f"{valid_graph.loop_count} redundant loop(s) agree within tolerance",
            )
        )

    def graph_findings(touched: set[FrameId]) -> list[PreflightFinding]:
        found: list[PreflightFinding] = []
        for index in sorted(failed_indexes):
            transform = transforms[index]
            if transform.parent_frame in touched:
                found.append(
                    PreflightFinding(
                        code="invalid_transform",
                        message=f"static transform T_{transform.parent_frame}_"
                        f"{transform.child_frame} failed validation",
                    )
                )
        for loop in loops:
            if loop.frames[0] in touched:
                found.append(
                    PreflightFinding(
                        code="ambiguous_frame_graph",
                        message=f"redundant static paths through {loop.frames[0]!r} and "
                        f"{loop.frames[1]!r} disagree by {loop.translation_error_m:.3e} m and "
                        f"{loop.rotation_error_rad:.3e} rad",
                    )
                )
        return found

    findings, relation_checks, touched = _relation_findings(
        execution, facts=facts, calibration=calibration, graph=graph, body_frame=body_frame
    )
    transform_checks.extend(relation_checks)
    blockers = [*findings, *graph_findings(touched)]

    warnings: list[PreflightFinding] = []
    for index in sorted(failed_indexes):
        transform = transforms[index]
        if transform.parent_frame not in touched:
            warnings.append(
                PreflightFinding(
                    code="unused_calibration_invalid",
                    message=f"static transform T_{transform.parent_frame}_{transform.child_frame} "
                    "failed validation, but this run does not use it",
                )
            )
    for loop in loops:
        if loop.frames[0] not in touched:
            warnings.append(
                PreflightFinding(
                    code="unused_calibration_inconsistent",
                    message=f"redundant static paths through {loop.frames[0]!r} and "
                    f"{loop.frames[1]!r} disagree, but this run does not use them",
                )
            )

    clock_checks: list[ClockCheck] = []
    required_present = sorted(execution.modalities & facts.modalities)
    all_clocks: set[str] = set()
    for modality in required_present:
        clocks = facts.clocks[modality]
        all_clocks |= clocks
        if "" in clocks:
            clock_checks.append(
                ClockCheck(
                    subject=f"modality:{modality}", passed=False, detail="empty clock identity"
                )
            )
            blockers.append(
                PreflightFinding(
                    code="missing_clock",
                    message=f"observations of {modality!r} carry no clock identity",
                )
            )
        elif len(clocks) > 1:
            clock_checks.append(
                ClockCheck(
                    subject=f"modality:{modality}",
                    passed=False,
                    detail=f"several clock domains {sorted(clocks)}",
                )
            )
            blockers.append(
                PreflightFinding(
                    code="clock_domain_mismatch",
                    message=f"observations of {modality!r} use several clock domains "
                    f"{sorted(clocks)}",
                )
            )
        else:
            clock_checks.append(
                ClockCheck(subject=f"modality:{modality}", passed=True, detail=next(iter(clocks)))
            )
    if len(required_present) > 1:
        aligned = len(all_clocks) <= 1
        clock_checks.append(
            ClockCheck(
                subject="required modalities",
                passed=aligned,
                detail=f"clock domains {sorted(all_clocks)}",
            )
        )
        if not aligned and "" not in all_clocks:
            blockers.append(
                PreflightFinding(
                    code="clock_domain_mismatch",
                    message="required modalities use different clock domains "
                    f"{sorted(all_clocks)}; clock domains are never compared implicitly",
                )
            )

    readiness: list[DownstreamReadiness] = []
    for requirements in downstream:
        missing, downstream_checks, downstream_touched = _relation_findings(
            requirements, facts=facts, calibration=calibration, graph=graph, body_frame=body_frame
        )
        transform_checks.extend(downstream_checks)
        missing = [*missing, *graph_findings(downstream_touched)]
        readiness.append(
            DownstreamReadiness(
                capability=requirements.capability, ready=not missing, missing=tuple(missing)
            )
        )
        if missing:
            warnings.append(
                PreflightFinding(
                    code="downstream_not_ready",
                    message=f"{requirements.capability} lacks "
                    f"{sorted({finding.code for finding in missing})}",
                )
            )

    return GeometryPreflightReport(
        required_inputs=execution.modalities,
        available_inputs=facts.modalities,
        frame_graph=FrameGraphSummary(
            reference_frame=execution.reference_frame,
            body_frame=body_frame,
            static_frames=tuple(sorted(graph.frames, key=str)),
            static_edges=graph.edges,
            component_count=len({frozenset(graph.component_of(frame)) for frame in graph.frames}),
        ),
        calibration_identity=calibration_identity(calibration),
        transform_checks=tuple(transform_checks),
        clock_checks=tuple(clock_checks),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        downstream=tuple(readiness),
        conventions=dict(CONVENTIONS),
    )

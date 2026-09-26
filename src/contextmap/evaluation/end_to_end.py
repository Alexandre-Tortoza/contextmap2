"""Canonical Solution 1 end-to-end scenario, acceptance matrix and acceptance report.

``Solution 1 validated`` is only meaningful against a frozen definition. This module
owns that definition as versioned, hashable data: the recorded input the run is
frozen on, the canonical backend profile (which stages are required and which
capabilities are ablation-only), and the acceptance matrix, one gate per
stage-specific requirement. It also owns the acceptance report, which records one
result per gate.

Two rules shape the report. There is no overall score or pass flag: a failed,
blocked or unevaluated gate stays visible with the capability that owns it. And a
result is only counted as evidence when it is ``real``: a contract rehearsal over
synthetic fixtures protects regressions but never validates Solution 1. See
``src/contextmap/evaluation/docs/end-to-end.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.evaluation._persistence import canonical_digest, write_immutable_json
from contextmap.evaluation._validation import require_sha256, require_text, require_unique
from contextmap.evaluation.annotations import AnnotationFamily
from contextmap.evaluation.reference_set import ReferenceSetIdentity

SCENARIO_SCHEMA = "contextmap.e2e.scenario/v1"
ACCEPTANCE_REPORT_SCHEMA = "contextmap.e2e.acceptance-report/v1"
SCENARIO_ID = "solution-1-canonical"
SCENARIO_VERSION = "1.0.5"
"""Version of the frozen scenario; any change to a frozen decision needs a new one.

1.0.1 fixed the runtime-selection completeness bug that #540's real catalog wiring for
``entity_resolution``/``spatial_relations`` exposed: those capabilities, and the policy
components ``geometric_mapping``/``sensor_association`` already had, now declare catalog
components and therefore need an explicit backend selected (the runtime never chooses one on
the caller's behalf, even when a policy has a single option). The profile also started
resolving against ``canonical/2`` instead of ``canonical/1``, since that is the preset whose
topology actually included semantic mapping, entity resolution and spatial relations.

1.0.2 fixed the same class of gap for ``semantic_mapping``: it declared a real catalog
component (``semantic_mapping.geometry_summary``, needed to compose a
``SemanticMappingExecutor`` automatically), so the profile needed an explicit selection for it
too.

1.0.3: before v0.1.0 ships there is no released consumer to protect from a runtime topology
change, so ``canonical/2``/``canonical/3`` were collapsed back into ``canonical/1`` -- one single
topology, end to end through ``context_map``, free to keep evolving until the release (see
:data:`contextmap.runtime.catalog.CANONICAL_PROFILE_ID`). ``RUNTIME_PRESET`` now reads
``canonical/1``. No version in this history alters the dataset, the gates, the ablation-only
list or any stage's scientific rationale.

1.0.4 re-freezes the real subject on a new, immutable ``SequenceArtifact``
(``720a486de8d44c16a9d3d2ff9fa7b1a4``) produced through issue #554's dataset-scoped timestamp
policy: the RGB calibration now carries a real ``MeiCameraModel`` instead of the previous
artifact's ``camera_model=None``, and every observation's header timestamp carries an explicit,
auditable constant-offset correction instead of the recording gap simply being ignored. The
90 s window, its pose-gap/path/yaw rationale and the pose side file are unchanged (same physical
segment); only the artifact identity, its selection identity and its observation IDs/timestamps
shifted by the correction's offset. Resolved observation counts inside the window are bit-for-bit
identical to the previous artifact (2160 images, 892 LiDAR scans, 17983 IMU samples), which is
itself evidence that the correction changed no data, only its clock.

1.0.5 is the **release contract** of v0.1.0, and it exists because 1.0.4's campaign produced a
real negative result that 1.0.4 must keep reporting. Two independent real executions of the
identical canonical Visual Perception configuration (SAM2.1-hiera-tiny + DINOv2-base +
CLIP-ViT-L/14 + Qwen3-VL-4B-Instruct nf4, greedy, ``temperature=0.0``) over the same 20 real
corridor-02 frames agreed on only 29 of 90 compared canonical claims, while region discovery was
perfectly reproducible (0/20 frame mismatches) and, given the *same* ``PerceptionRunArtifact``,
all eight downstream stages were exactly reproducible. 1.0.4 therefore fails
``reproducibility.rerun_equivalence`` and **stays failed**: it is the authoritative record of the
campaign that ran against it, and no version of this history may relabel it.

What 1.0.5 changes is what the *release* requires, not what 1.0.4 measured:

* ``reproducibility.rerun_equivalence`` is narrowed to the property the evidence actually
  supports -- repeated runs from the same ``PerceptionRunArtifact`` yield equivalent artifacts,
  metric reports and final map. It no longer asserts that recorded sensors to
  ``ContextMapArtifact`` is reproducible end to end while a generative backend participates in
  the chain, because that is not what was demonstrated.
* ``reproducibility.semantic_rerun_agreement`` is a new :attr:`GateKind.REPORT` gate owned by
  ``visual_perception``: the measured agreement is reported with its denominator and no pass
  threshold, so the negative evidence is counted rather than dropped. A report gate never gates
  a release by itself, which is the existing design of :func:`unmet_required_gates`.
* :class:`ExperimentalComponent` declares the Qwen3-VL semantic interpretation backend
  experimental for this release, naming the property v0.1.0 does not guarantee, the measurement
  and issue #556, which stays open outside the release blocker.
* ``visual_perception.semantic_quality`` now requires ``semantic.parse_failure_rate``. This was
  deliberately deferred from 1.0.4: adding a metric to a frozen gate demands a new version, and
  retroactively requiring it would have mixed two experimental definitions in one campaign. It
  becomes required here because a run that records the contractual failures stream can measure
  it completely.

No dataset, stage, backend selection or ablation-only decision changes in 1.0.5.
"""

CROSS_STAGE = "cross_stage"
"""Capability name of a gate about a boundary; the failing capability is named per result."""

_RUNTIME = "runtime"


class ScenarioError(ValueError):
    """Raised when a scenario, gate, result or report is not what a reader may rely on."""


class EvidenceClass(Enum):
    """Whether evidence came from real recorded data or from a contract rehearsal."""

    REAL = "real"
    FAKE_CONTRACT = "fake_contract"


class GateKind(Enum):
    """How a gate is decided."""

    INVARIANT = "invariant"
    """Binary structural property; a violation fails the gate."""

    REPORT = "report"
    """Metrics must be reported with explicit denominators. There is no pass threshold
    in this scenario version: a threshold is a frozen decision that needs a new version,
    set from a non-held-out split."""


class GateStatus(Enum):
    """The state of one gate in one run."""

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_EVALUATED = "not_evaluated"


# ------------------------------------------------------------------------------ scenario


@dataclass(frozen=True, kw_only=True)
class ComponentSelection:
    """The backend one variation point of a stage is frozen to.

    Attributes:
        component_id: ``"<capability>.<slot>"`` identity of the variation point.
        backend: Backend or policy identity, as the runtime catalog names it.
        model: Model family the backend runs, when it has one. The exact revision or
            checkpoint hash is recorded by the run itself and checked by the
            ``runtime.provenance_identity`` gate.
        external_service: Whether the backend calls a remote service.
    """

    component_id: str
    backend: str
    model: str | None = None
    external_service: bool = False

    def __post_init__(self) -> None:
        """Reject a blank identity."""
        require_text("component_id", self.component_id)
        require_text("backend", self.backend)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "component_id": self.component_id,
            "backend": self.backend,
            "model": self.model,
            "external_service": self.external_service,
        }


@dataclass(frozen=True, kw_only=True)
class ScenarioStage:
    """One required stage of the canonical run.

    Every stage of the scenario is required; what is optional lives in
    :class:`AblationOnlyOption`.

    Attributes:
        stage_id: Runtime stage identity of ``canonical/1``.
        capability: Capability that owns the stage.
        components: Backends the stage's variation points are frozen to.
        rationale: Why the profile is this way.
    """

    stage_id: str
    capability: str
    components: tuple[ComponentSelection, ...] = ()
    rationale: str

    def __post_init__(self) -> None:
        """Reject a blank identity or rationale."""
        require_text("stage_id", self.stage_id)
        require_text("capability", self.capability)
        require_text(f"rationale of {self.stage_id}", self.rationale)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "stage_id": self.stage_id,
            "capability": self.capability,
            "components": [component.to_record() for component in self.components],
            "rationale": self.rationale,
        }


@dataclass(frozen=True, kw_only=True)
class AblationOnlyOption:
    """An implemented option that stays outside the canonical run.

    An option leaves this list only through a controlled end-to-end ablation that
    shows benefit relative to cost, followed by a new scenario version and a passing
    canonical run; never by a paper's claim or a stage-level metric alone.

    Attributes:
        component_id: Variation point or stage the option belongs to.
        option: The variant that is compared, not adopted.
        reason: Why it is not canonical.
    """

    component_id: str
    option: str
    reason: str

    def __post_init__(self) -> None:
        """Reject a blank field."""
        require_text("component_id", self.component_id)
        require_text("option", self.option)
        require_text(f"reason of {self.option}", self.reason)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"component_id": self.component_id, "option": self.option, "reason": self.reason}


@dataclass(frozen=True, kw_only=True)
class ExperimentalComponent:
    """A canonical-run component whose named output property the release does not guarantee.

    This is not an :class:`AblationOnlyOption`: the component runs inside the canonical
    pipeline and its evidence reaches the final map. What this declares is narrower, and a
    release contract has to say it out loud: which property of the component's output the
    scenario does **not** require, the real measurement that motivated the declaration, and
    where the open question is tracked. A consumer of the artifact must not assume the named
    property holds.

    Declaring a component experimental never edits a previous scenario version's verdict. The
    version that measured a violation keeps reporting it; a later version may move the property
    out of its own release contract, which is a new frozen decision and therefore a new digest.

    Attributes:
        component_id: Variation point the component is selected at.
        option: The selected backend.
        unguaranteed_property: The property the scenario does not require of its output.
        measurement: The real measurement behind the declaration, with its denominator.
        follow_up: Where the open question is tracked.
    """

    component_id: str
    option: str
    unguaranteed_property: str
    measurement: str
    follow_up: str

    def __post_init__(self) -> None:
        """Reject a blank field: an unguaranteed property with no measurement is an opinion."""
        for name, value in (
            ("component_id", self.component_id),
            ("option", self.option),
            ("unguaranteed_property", self.unguaranteed_property),
            ("measurement", self.measurement),
            ("follow_up", self.follow_up),
        ):
            require_text(f"{name} of the experimental component {self.option!r}", value)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "component_id": self.component_id,
            "option": self.option,
            "unguaranteed_property": self.unguaranteed_property,
            "measurement": self.measurement,
            "follow_up": self.follow_up,
        }


@dataclass(frozen=True, kw_only=True)
class SelectedImage:
    """One image of the frozen selection and the LiDAR scan nearest to it."""

    observation_id: str
    timestamp_ns: int
    nearest_lidar_id: str

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "observation_id": self.observation_id,
            "timestamp_ns": self.timestamp_ns,
            "nearest_lidar_id": self.nearest_lidar_id,
        }


@dataclass(frozen=True, kw_only=True)
class SourceSelection:
    """The exact slice of the recorded sequence the run is frozen on.

    Attributes:
        clock_id: The single clock every timestamp below is expressed in.
        selection_identity: Digest of the selection, when one was recorded.
        start_ns: Window start in ``clock_id`` nanoseconds.
        end_ns: Window end in ``clock_id`` nanoseconds.
        image_count: Image observations inside the window.
        lidar_scan_count: LiDAR scans inside the window.
        imu_count: IMU samples inside the window.
        selected_images: The images perception runs on.
    """

    clock_id: str
    selection_identity: str | None
    start_ns: int
    end_ns: int
    image_count: int
    lidar_scan_count: int
    imu_count: int
    selected_images: tuple[SelectedImage, ...]

    def __post_init__(self) -> None:
        """Require a coherent window and unique selected images."""
        require_text("clock_id", self.clock_id)
        if self.selection_identity is not None:
            require_sha256("selection_identity", self.selection_identity)
        if self.end_ns <= self.start_ns:
            raise ScenarioError("the selection window must end after it starts")
        require_unique("selected image", (image.observation_id for image in self.selected_images))

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "clock_id": self.clock_id,
            "selection_identity": self.selection_identity,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "image_count": self.image_count,
            "lidar_scan_count": self.lidar_scan_count,
            "imu_count": self.imu_count,
            "selected_images": [image.to_record() for image in self.selected_images],
        }


@dataclass(frozen=True, kw_only=True)
class ExternalInput:
    """A side input that is declared, hashed and never a reference for evaluation.

    Attributes:
        name: Input name.
        role: What the pipeline uses it as (for example ``pose_input``).
        location: Where it lives, relative to the dataset root.
        digest: ``sha256:`` of the content, when it is a file.
    """

    name: str
    role: str
    location: str
    digest: str | None

    def __post_init__(self) -> None:
        """Reject a blank field and a malformed digest."""
        require_text("name", self.name)
        require_text("role", self.role)
        require_text("location", self.location)
        if self.digest is not None:
            require_sha256(f"digest of {self.name}", self.digest)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "name": self.name,
            "role": self.role,
            "location": self.location,
            "digest": self.digest,
        }


@dataclass(frozen=True, kw_only=True)
class ScenarioSubject:
    """The recorded input a scenario is frozen on and how strong its evidence is.

    Attributes:
        subject_id: Stable name of the subject.
        evidence_class: ``REAL`` for a recorded sequence, ``FAKE_CONTRACT`` for
            synthetic fixtures.
        dataset_id: Dataset the sequence comes from.
        sequence_artifact_id: Exact ``SequenceArtifact`` identity, when one exists.
        sequence_manifest_digest: ``sha256:`` of that artifact's manifest.
        selection: The frozen slice of the sequence.
        external_inputs: Declared side inputs.
        reference_set: Reference set the quality gates evaluate against, or ``None``
            while none is annotated. ``None`` blocks the quality gates explicitly.
        note: What the subject is and is not evidence of.
    """

    subject_id: str
    evidence_class: EvidenceClass
    dataset_id: str
    sequence_artifact_id: str | None
    sequence_manifest_digest: str | None
    selection: SourceSelection
    external_inputs: tuple[ExternalInput, ...]
    reference_set: ReferenceSetIdentity | None
    note: str

    def __post_init__(self) -> None:
        """Reject a blank identity or a malformed digest."""
        require_text("subject_id", self.subject_id)
        require_text("dataset_id", self.dataset_id)
        require_text("note", self.note)
        if self.sequence_manifest_digest is not None:
            require_sha256("sequence_manifest_digest", self.sequence_manifest_digest)
        require_unique("external input", (item.name for item in self.external_inputs))

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "subject_id": self.subject_id,
            "evidence_class": self.evidence_class.value,
            "dataset_id": self.dataset_id,
            "sequence_artifact_id": self.sequence_artifact_id,
            "sequence_manifest_digest": self.sequence_manifest_digest,
            "selection": self.selection.to_record(),
            "external_inputs": [item.to_record() for item in self.external_inputs],
            "reference_set": None if self.reference_set is None else self.reference_set.to_record(),
            "note": self.note,
        }


# --------------------------------------------------------------------------------- gates


@dataclass(frozen=True, kw_only=True)
class AcceptanceGate:
    """One stage-specific acceptance requirement.

    Attributes:
        gate_id: ``"<area>.<check>"`` identity, unique inside the matrix.
        capability: Capability that owns a failure of this gate, or
            :data:`CROSS_STAGE` for a boundary gate whose result names the capability
            that broke the contract.
        kind: Whether the gate is an invariant or a report.
        requirement: What must hold.
        evidence: What demonstrates it.
        metrics: Names of the registry metrics that back the gate.
        needs_reference_annotations: Annotation families the gate cannot be decided
            without. An absent family blocks the gate; it never counts as zero.
    """

    gate_id: str
    capability: str
    kind: GateKind
    requirement: str
    evidence: str
    metrics: tuple[str, ...] = ()
    needs_reference_annotations: tuple[AnnotationFamily, ...] = ()

    def __post_init__(self) -> None:
        """Require a gate to name its owner, its requirement and its evidence."""
        require_text("gate_id", self.gate_id)
        for name, value in (
            ("capability", self.capability),
            ("requirement", self.requirement),
            ("evidence", self.evidence),
        ):
            if not value or not value.strip():
                raise ScenarioError(f"gate {self.gate_id!r} needs a non-empty {name}")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "gate_id": self.gate_id,
            "capability": self.capability,
            "kind": self.kind.value,
            "requirement": self.requirement,
            "evidence": self.evidence,
            "metrics": list(self.metrics),
            "needs_reference_annotations": [f.value for f in self.needs_reference_annotations],
        }


@dataclass(frozen=True, kw_only=True)
class E2EScenario:
    """The frozen definition of what ``Solution 1 validated`` means.

    Attributes:
        scenario_id: Stable name of the scenario across versions.
        version: Version; any change to a frozen decision needs a new one.
        subject: The recorded input and the strength of its evidence.
        stages: The canonical run, in topology order.
        ablation_only: Implemented options that stay outside the canonical run.
        experimental: Canonical components whose named output property the release does not
            guarantee.
        gates: The acceptance matrix.
    """

    scenario_id: str
    version: str
    subject: ScenarioSubject
    stages: tuple[ScenarioStage, ...]
    ablation_only: tuple[AblationOnlyOption, ...]
    experimental: tuple[ExperimentalComponent, ...] = ()
    gates: tuple[AcceptanceGate, ...]

    def __post_init__(self) -> None:
        """Require unique stages and gates, and gates that belong to a known owner."""
        require_text("scenario_id", self.scenario_id)
        require_text("version", self.version)
        require_unique("stage", (stage.stage_id for stage in self.stages))
        require_unique("gate", (gate.gate_id for gate in self.gates))
        owners = {stage.capability for stage in self.stages} | {_RUNTIME, CROSS_STAGE}
        for gate in self.gates:
            if gate.capability not in owners:
                raise ScenarioError(
                    f"gate {gate.gate_id!r} is owned by {gate.capability!r}, which is not a "
                    "capability of the scenario"
                )

    def matrix_record(self) -> list[dict[str, Any]]:
        """Return the acceptance matrix alone, shared by every subject."""
        return [gate.to_record() for gate in self.gates]

    @property
    def matrix_digest(self) -> str:
        """Return the digest of the acceptance matrix alone."""
        return canonical_digest(self.matrix_record())

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record the digest is computed over."""
        return {
            "scenario_id": self.scenario_id,
            "version": self.version,
            "subject": self.subject.to_record(),
            "stages": [stage.to_record() for stage in self.stages],
            "ablation_only": [option.to_record() for option in self.ablation_only],
            "experimental": [component.to_record() for component in self.experimental],
            "gates": self.matrix_record(),
        }

    @property
    def digest(self) -> str:
        """Return the digest of the whole scenario."""
        return canonical_digest(self.to_record())


def encode_scenario(scenario: E2EScenario) -> dict[str, Any]:
    """Return the scenario document, including its own digest and its matrix digest."""
    return {
        "schema": SCENARIO_SCHEMA,
        **scenario.to_record(),
        "matrix_digest": scenario.matrix_digest,
        "digest": scenario.digest,
    }


def scenario_snapshot_name(scenario: E2EScenario) -> str:
    """Return the file name of the committed snapshot of a scenario."""
    return f"{scenario.scenario_id}-{scenario.version}-{scenario.subject.subject_id}.json"


RUNTIME_PRESET = "canonical/1"
"""Runtime topology preset the scenario's profile is expressed against.

Pre-v0.1.0, ``canonical/1`` is the repository's one and only topology, end to end through
``context_map`` (see :data:`contextmap.runtime.catalog.CANONICAL_PROFILE_ID`); there is no
released consumer yet to protect from a topology change, so this identity is free to keep
evolving with the pipeline until the release.
"""

_OPTIONAL_STAGES_OFF = ("point_representation",)
"""Optional runtime stages the canonical profile keeps off; they run only as ablations."""


def scenario_runtime_document(scenario: E2EScenario) -> dict[str, Any]:
    """Return the runtime configuration document that selects the scenario's backends.

    The document only *selects*: backend parameters such as checkpoint, revision and
    thresholds belong to the capability that owns the backend and to the run that
    records them, never to the scenario. Writing it as a ``.json`` file and passing it to
    the runtime resolves the canonical profile without a second source of truth.

    Args:
        scenario: The frozen scenario.

    Returns:
        A JSON-compatible document with the topology preset, the optional stages kept
        off, and one backend per variation point of the profile.
    """
    components: dict[str, dict[str, dict[str, str]]] = {}
    for stage in scenario.stages:
        for component in stage.components:
            capability, slot = component.component_id.split(".", 1)
            components.setdefault(capability, {})[slot] = {"backend": component.backend}
    return {
        "pipeline": {
            "preset": RUNTIME_PRESET,
            "stages": {stage_id: False for stage_id in _OPTIONAL_STAGES_OFF},
        },
        "components": components,
    }


# ------------------------------------------------------------------------------ results


@dataclass(frozen=True, kw_only=True)
class GateResult:
    """The state of one gate in one run, with the evidence that decided it.

    Build it with :meth:`passed`, :meth:`failed`, :meth:`blocked` or
    :meth:`not_evaluated`, which enforce what each status must carry.

    Attributes:
        gate_id: The gate this result is about.
        status: The gate state.
        evidence_class: ``real`` or ``fake_contract``; ``None`` when nothing was
            evaluated (blocked or not evaluated).
        evidence_refs: Exact artifact identities, report digests or paths.
        failing_capabilities: Who owns a failure or a blocker.
        detail: What was checked and found.
    """

    gate_id: str
    status: GateStatus
    evidence_class: EvidenceClass | None
    evidence_refs: tuple[str, ...]
    failing_capabilities: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        """Enforce what each status must carry."""
        require_text("gate_id", self.gate_id)
        if self.status in (GateStatus.PASSED, GateStatus.FAILED):
            if self.evidence_class is None or not self.evidence_refs:
                raise ScenarioError(
                    f"{self.status.value} gate {self.gate_id!r} needs an evidence class and "
                    "at least one evidence reference"
                )
        elif self.evidence_class is not None or self.evidence_refs:
            raise ScenarioError(
                f"{self.status.value} gate {self.gate_id!r} evaluated nothing, so it cannot "
                "carry evidence"
            )
        if self.status is GateStatus.PASSED and self.failing_capabilities:
            raise ScenarioError(f"passed gate {self.gate_id!r} cannot name a failing capability")
        if self.status in (GateStatus.FAILED, GateStatus.BLOCKED) and not self.failing_capabilities:
            raise ScenarioError(
                f"{self.status.value} gate {self.gate_id!r} must name a failing capability "
                "(blocked_by)"
            )
        require_text(f"detail of {self.gate_id}", self.detail)

    @classmethod
    def passed(
        cls,
        gate_id: str,
        *,
        evidence_class: EvidenceClass,
        evidence_refs: Sequence[str],
        detail: str,
    ) -> GateResult:
        """Return a passed result."""
        return cls(
            gate_id=gate_id,
            status=GateStatus.PASSED,
            evidence_class=evidence_class,
            evidence_refs=tuple(evidence_refs),
            failing_capabilities=(),
            detail=detail,
        )

    @classmethod
    def failed(
        cls,
        gate_id: str,
        *,
        evidence_class: EvidenceClass,
        evidence_refs: Sequence[str],
        failing_capabilities: Sequence[str],
        detail: str,
    ) -> GateResult:
        """Return a failed result naming the capability that owns the failure."""
        return cls(
            gate_id=gate_id,
            status=GateStatus.FAILED,
            evidence_class=evidence_class,
            evidence_refs=tuple(evidence_refs),
            failing_capabilities=tuple(failing_capabilities),
            detail=detail,
        )

    @classmethod
    def blocked(cls, gate_id: str, *, blocked_by: Sequence[str], detail: str) -> GateResult:
        """Return a result for a gate that cannot be decided yet, naming what blocks it."""
        return cls(
            gate_id=gate_id,
            status=GateStatus.BLOCKED,
            evidence_class=None,
            evidence_refs=(),
            failing_capabilities=tuple(blocked_by),
            detail=detail,
        )

    @classmethod
    def not_evaluated(cls, gate_id: str, *, detail: str) -> GateResult:
        """Return a result for a gate nobody tried to decide in this pass."""
        return cls(
            gate_id=gate_id,
            status=GateStatus.NOT_EVALUATED,
            evidence_class=None,
            evidence_refs=(),
            failing_capabilities=(),
            detail=detail,
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "gate_id": self.gate_id,
            "status": self.status.value,
            "evidence_class": None if self.evidence_class is None else self.evidence_class.value,
            "evidence_refs": list(self.evidence_refs),
            "failing_capabilities": list(self.failing_capabilities),
            "detail": self.detail,
        }


@dataclass(frozen=True, kw_only=True)
class ArtifactRecord:
    """The exact identity of one artifact a report refers to."""

    artifact_id: str
    digest: str

    def __post_init__(self) -> None:
        """Reject a blank identity and a malformed digest."""
        require_text("artifact_id", self.artifact_id)
        require_sha256(f"digest of {self.artifact_id}", self.digest)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"artifact_id": self.artifact_id, "digest": self.digest}


@dataclass(frozen=True, kw_only=True)
class AcceptanceReport:
    """One result per gate of a scenario, for one run.

    There is no aggregate field on purpose: readiness is read from the gates.

    Attributes:
        report_id: Identity of the report.
        scenario_id: Scenario the run was checked against.
        scenario_version: Its version.
        scenario_digest: Its digest.
        matrix_digest: Digest of the acceptance matrix.
        subject_evidence_class: Strength of the scenario's subject.
        run_id: Identity of the run that produced the artifacts.
        code_version: Code identity of the run.
        results: Exactly one result per gate, in matrix order.
        stage_artifacts: The exact artifact of each stage the run produced or reused.
        final_artifact: The final ``ContextMapArtifact``, when the run produced one.
        limitations: Known limitations of the run, stated rather than hidden.
    """

    report_id: str
    scenario_id: str
    scenario_version: str
    scenario_digest: str
    matrix_digest: str
    subject_evidence_class: EvidenceClass
    run_id: str
    code_version: str
    results: tuple[GateResult, ...]
    stage_artifacts: Mapping[str, ArtifactRecord]
    final_artifact: ArtifactRecord | None
    limitations: tuple[str, ...]


def assemble_acceptance_report(
    scenario: E2EScenario,
    *,
    report_id: str,
    run_id: str,
    code_version: str,
    results: Iterable[GateResult],
    stage_artifacts: Mapping[str, ArtifactRecord] | None = None,
    final_artifact: ArtifactRecord | None = None,
    limitations: Sequence[str] = (),
) -> AcceptanceReport:
    """Bind one result per gate to a scenario.

    Args:
        scenario: The frozen scenario the run is checked against.
        report_id: Identity of the report.
        run_id: Identity of the run that produced the artifacts.
        code_version: Code identity of the run.
        results: One result for every gate of the scenario.
        stage_artifacts: Exact artifact of each stage, by stage id.
        final_artifact: The final map artifact, when there is one.
        limitations: Known limitations of the run.

    Returns:
        The report, with results in matrix order.

    Raises:
        ScenarioError: If a gate has no result, has two, or a result names an unknown gate.
    """
    require_text("report_id", report_id)
    require_text("run_id", run_id)
    by_gate: dict[str, GateResult] = {}
    known = {gate.gate_id for gate in scenario.gates}
    for result in results:
        if result.gate_id not in known:
            raise ScenarioError(f"result for unknown gate {result.gate_id!r}")
        if result.gate_id in by_gate:
            raise ScenarioError(f"gate {result.gate_id!r} has a result twice")
        by_gate[result.gate_id] = result
    missing = [gate.gate_id for gate in scenario.gates if gate.gate_id not in by_gate]
    if missing:
        raise ScenarioError(
            f"gates missing a result: {', '.join(missing)}; state "
            "GateResult.not_evaluated() explicitly instead of omitting them"
        )
    stage_ids = {stage.stage_id for stage in scenario.stages}
    for stage_id in stage_artifacts or {}:
        if stage_id not in stage_ids:
            raise ScenarioError(f"artifact recorded for unknown stage {stage_id!r}")
    return AcceptanceReport(
        report_id=report_id,
        scenario_id=scenario.scenario_id,
        scenario_version=scenario.version,
        scenario_digest=scenario.digest,
        matrix_digest=scenario.matrix_digest,
        subject_evidence_class=scenario.subject.evidence_class,
        run_id=run_id,
        code_version=code_version,
        results=tuple(by_gate[gate.gate_id] for gate in scenario.gates),
        stage_artifacts=dict(stage_artifacts or {}),
        final_artifact=final_artifact,
        limitations=tuple(limitations),
    )


@dataclass(frozen=True, kw_only=True)
class UnmetGate:
    """A required gate that the report does not show as met, and why."""

    gate_id: str
    capabilities: tuple[str, ...]
    status: GateStatus
    reason: str


def unmet_required_gates(
    scenario: E2EScenario,
    report: AcceptanceReport,
    *,
    kinds: frozenset[GateKind] | None = None,
) -> tuple[UnmetGate, ...]:
    """Return every required gate the report does not show as met.

    A gate is met only when it passed with ``real`` evidence. Contract evidence over
    synthetic fixtures never satisfies a gate. A failing capability is named for each
    unmet gate.

    Args:
        scenario: The scenario the report was assembled against.
        report: The report.
        kinds: Restrict which :class:`GateKind` values count as required. ``None``
            (the default) keeps every gate required, exactly as before this parameter
            existed. Pass ``frozenset({GateKind.INVARIANT})`` for a release-readiness
            check: a :attr:`GateKind.REPORT` gate has no pass threshold by design (its
            own semantics are metrics-only), so it should never gate a release by
            itself -- only a structural :attr:`GateKind.INVARIANT` gate should.

    Returns:
        The unmet gates in matrix order; empty only when every required gate is met.
    """
    owner = {gate.gate_id: gate.capability for gate in scenario.gates}
    gate_kind = {gate.gate_id: gate.kind for gate in scenario.gates}
    unmet: list[UnmetGate] = []
    for result in report.results:
        if kinds is not None and gate_kind[result.gate_id] not in kinds:
            continue
        capabilities = result.failing_capabilities or (owner[result.gate_id],)
        if result.status is not GateStatus.PASSED:
            reason = f"{result.status.value}: {result.detail}"
        elif result.evidence_class is not EvidenceClass.REAL:
            reason = "passed on contract evidence only; a real run is still required"
        else:
            continue
        unmet.append(
            UnmetGate(
                gate_id=result.gate_id,
                capabilities=capabilities,
                status=result.status,
                reason=reason,
            )
        )
    return tuple(unmet)


def encode_acceptance_report(report: AcceptanceReport) -> dict[str, Any]:
    """Return the report document; it carries every gate and no aggregate label."""
    return {
        "schema": ACCEPTANCE_REPORT_SCHEMA,
        "report_id": report.report_id,
        "scenario": {
            "scenario_id": report.scenario_id,
            "version": report.scenario_version,
            "digest": report.scenario_digest,
            "matrix_digest": report.matrix_digest,
            "evidence_class": report.subject_evidence_class.value,
        },
        "run_id": report.run_id,
        "code_version": report.code_version,
        "stage_artifacts": {
            stage_id: record.to_record() for stage_id, record in report.stage_artifacts.items()
        },
        "final_artifact": None
        if report.final_artifact is None
        else report.final_artifact.to_record(),
        "gates": [result.to_record() for result in report.results],
        "limitations": list(report.limitations),
    }


def write_acceptance_report(path: Path, report: AcceptanceReport) -> None:
    """Atomically publish the report, refusing to replace an existing one."""
    write_immutable_json(path, encode_acceptance_report(report), "acceptance report")

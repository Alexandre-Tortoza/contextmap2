"""Cross-stage invariants of a Solution 1 run: lineage, frames, traceability, identity.

A pipeline can produce plausible output while silently losing provenance, changing
coordinate semantics, duplicating physical observations or attaching evidence to the
wrong geometry. Each stage validates its own artifact; this module validates the
*boundaries* between them, reading only the public manifests and objects of the
capabilities. It never repairs an invalid upstream artifact: a broken boundary becomes
a finding that names the capability that broke the contract.

The checks cover the stages that exist today (ingestion to semantic fusion). Entity
resolution, spatial relations and the final artifact are reported as unverified
boundaries, so a gate that depends on them is blocked instead of passed. See
``src/contextmap/evaluation/docs/end-to-end.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from contextmap.evaluation.end_to_end import EvidenceClass, GateResult
from contextmap.geometric_mapping import (
    GeometricMapArtifactManifest,
    GeometrySource,
)
from contextmap.ingestion import SequenceArtifactManifest
from contextmap.semantic_fusion import FusionOutcome, SemanticFusionRunManifest
from contextmap.sensor_association import (
    SensorAssociationRunManifest,
    SpatialObservation,
    SpatialObservationId,
)
from contextmap.state_estimation import StateEstimationRunManifest
from contextmap.visual_perception import PerceptionResult, PerceptionResultId, PerceptionRun

LINEAGE_CLOSURE = "cross_stage.lineage_closure"
COORDINATE_CONSISTENCY = "cross_stage.coordinate_consistency"
EVIDENCE_TRACEABILITY = "cross_stage.evidence_traceability"
PHYSICAL_OBSERVATION_IDENTITY = "cross_stage.physical_observation_identity"

_STATE_ESTIMATION = "state_estimation"
_GEOMETRIC_MAPPING = "geometric_mapping"
_VISUAL_PERCEPTION = "visual_perception"
_SENSOR_ASSOCIATION = "sensor_association"
_SEMANTIC_FUSION = "semantic_fusion"


@dataclass(frozen=True, kw_only=True)
class CrossStageInputs:
    """The public manifests and objects the cross-stage checks read.

    Attributes:
        sequence: Manifest of the canonical sequence.
        sequence_calibration_identity: Identity of the calibration the sequence artifact
            carries, when the caller supplies it; the map and the associations must have
            used exactly that calibration.
        trajectory: Manifest of the state estimation run.
        geometry: Manifest of the geometric map.
        geometry_source: The read boundary of that map, to resolve references.
        perception_runs: The perception runs the association and fusion consumed.
        perception_results: Their results, by identity.
        associations: Manifests of every association run consumed.
        spatial_observations: Every spatial observation of those runs, by identity.
        fusion: Manifest of the semantic fusion run.
        fusion_outcomes: The supports and fused evidence of that run.
    """

    sequence: SequenceArtifactManifest
    trajectory: StateEstimationRunManifest
    geometry: GeometricMapArtifactManifest
    geometry_source: GeometrySource
    perception_runs: Sequence[PerceptionRun]
    perception_results: Mapping[PerceptionResultId, PerceptionResult]
    associations: Sequence[SensorAssociationRunManifest]
    spatial_observations: Mapping[SpatialObservationId, SpatialObservation]
    fusion: SemanticFusionRunManifest
    fusion_outcomes: Sequence[FusionOutcome]
    sequence_calibration_identity: str | None = None


@dataclass(frozen=True, kw_only=True)
class CrossStageFinding:
    """One broken boundary and the capability that owns the break."""

    gate_id: str
    failing_capability: str
    message: str


@dataclass(frozen=True, kw_only=True)
class CrossStageReport:
    """What the checks found and how many assertions each gate ran.

    Attributes:
        findings: Every broken boundary, in check order.
        checks_run: Number of assertions evaluated per gate; a gate with zero checks
            proved nothing.
        notes: Boundaries that could not be compared because an artifact does not record
            the identity (for example a trajectory that records no calibration). They
            are limitations, stated rather than silently passed.
    """

    findings: tuple[CrossStageFinding, ...]
    checks_run: Mapping[str, int]
    notes: tuple[str, ...] = ()


@dataclass
class _Collector:
    findings: list[CrossStageFinding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def expect(self, gate_id: str, capability: str, holds: bool, message: str) -> None:
        self.counts[gate_id] = self.counts.get(gate_id, 0) + 1
        if not holds:
            self.findings.append(
                CrossStageFinding(gate_id=gate_id, failing_capability=capability, message=message)
            )


def check_cross_stage(inputs: CrossStageInputs) -> CrossStageReport:
    """Check the boundaries between the stage artifacts of one run.

    Args:
        inputs: The manifests and objects of every stage from ingestion to fusion.

    Returns:
        The findings and the number of checks each gate ran. The inputs are not modified.
    """
    collector = _Collector()
    _check_lineage(inputs, collector)
    _check_coordinates(inputs, collector)
    _check_traceability(inputs, collector)
    _check_physical_identity(inputs, collector)
    return CrossStageReport(
        findings=tuple(collector.findings),
        checks_run=dict(collector.counts),
        notes=tuple(collector.notes),
    )


def cross_stage_gate_results(
    report: CrossStageReport,
    *,
    evidence_class: EvidenceClass,
    evidence_refs: Sequence[str],
    unverified_boundaries: Sequence[str] = (),
) -> tuple[GateResult, ...]:
    """Turn a cross-stage report into one gate result per cross-stage gate.

    A gate with findings fails and names the capabilities that own them, whatever else
    is unverified. A gate with no finding but with unverified boundaries is blocked by
    those capabilities: passing it would claim a boundary nobody checked.

    Args:
        report: The result of :func:`check_cross_stage`.
        evidence_class: Whether the checked artifacts came from a real run.
        evidence_refs: Exact identities of the artifacts checked.
        unverified_boundaries: Capabilities whose boundary was not part of the run yet.

    Returns:
        The four cross-stage gate results, in matrix order.
    """
    results: list[GateResult] = []
    for gate_id in (
        LINEAGE_CLOSURE,
        COORDINATE_CONSISTENCY,
        EVIDENCE_TRACEABILITY,
        PHYSICAL_OBSERVATION_IDENTITY,
    ):
        own = [finding for finding in report.findings if finding.gate_id == gate_id]
        checked = report.checks_run.get(gate_id, 0)
        summary = f"{checked} checks, {len(own)} findings"
        if report.notes:
            summary += "; limitations: " + " | ".join(report.notes)
        if own:
            capabilities = tuple(sorted({finding.failing_capability for finding in own}))
            results.append(
                GateResult.failed(
                    gate_id,
                    evidence_class=evidence_class,
                    evidence_refs=evidence_refs,
                    failing_capabilities=capabilities,
                    detail=f"{summary}: " + "; ".join(finding.message for finding in own[:5]),
                )
            )
        elif unverified_boundaries:
            results.append(
                GateResult.blocked(
                    gate_id,
                    blocked_by=unverified_boundaries,
                    detail=(
                        f"boundaries up to semantic_fusion verified ({summary}); "
                        f"not yet verifiable: {', '.join(unverified_boundaries)}"
                    ),
                )
            )
        else:
            results.append(
                GateResult.passed(
                    gate_id,
                    evidence_class=evidence_class,
                    evidence_refs=evidence_refs,
                    detail=summary,
                )
            )
    return tuple(results)


def _check_lineage(inputs: CrossStageInputs, collector: _Collector) -> None:
    def expect(capability: str, what: str, actual: object, expected: object) -> None:
        collector.expect(
            LINEAGE_CLOSURE,
            capability,
            str(actual) == str(expected),
            f"{what} is {actual!r}, expected {expected!r}",
        )

    sequence_id = inputs.sequence.artifact_id
    trajectory, geometry, fusion = inputs.trajectory, inputs.geometry, inputs.fusion
    expect(
        _STATE_ESTIMATION, "state estimation sequence", trajectory.sequence_artifact_id, sequence_id
    )
    expect(_GEOMETRIC_MAPPING, "geometric map sequence", geometry.sequence_artifact_id, sequence_id)
    expect(_SEMANTIC_FUSION, "fusion sequence", fusion.lineage.sequence_artifact_id, sequence_id)
    for run in inputs.perception_runs:
        expect(
            _VISUAL_PERCEPTION,
            f"perception run {run.run_id} sequence",
            run.sequence_artifact_id,
            sequence_id,
        )

    expect(_GEOMETRIC_MAPPING, "map trajectory", geometry.trajectory_id, trajectory.trajectory_id)
    expect(
        _GEOMETRIC_MAPPING,
        "map state estimation run",
        geometry.state_estimation_run_id,
        trajectory.run_id,
    )
    sequence_calibration = inputs.sequence_calibration_identity
    if sequence_calibration is None:
        collector.notes.append(
            "the sequence calibration identity was not supplied, so the calibration the map and "
            "the associations used is not compared with the sequence artifact's"
        )
    else:
        expect(
            _GEOMETRIC_MAPPING,
            "map sequence calibration",
            geometry.calibration_identity,
            sequence_calibration,
        )
        for association in inputs.associations:
            expect(
                _SENSOR_ASSOCIATION,
                f"association run {association.run_id} sequence calibration",
                association.calibration_identity,
                sequence_calibration,
            )
    if trajectory.calibration_identity is None:
        collector.notes.append(
            "the trajectory records no calibration identity (the backend does not consume "
            "calibration), so calibration lineage is verified between the map and the "
            "associations only"
        )
    else:
        expect(
            _GEOMETRIC_MAPPING,
            "map calibration",
            geometry.calibration_identity,
            trajectory.calibration_identity,
        )

    perception_ids = {str(run.run_id) for run in inputs.perception_runs}
    consumed_perception: set[str] = set()
    for association in inputs.associations:
        label = f"association run {association.run_id}"
        expect(
            _SENSOR_ASSOCIATION, f"{label} sequence", association.sequence_artifact_id, sequence_id
        )
        expect(_SENSOR_ASSOCIATION, f"{label} map", association.geometric_map_id, geometry.map_id)
        expect(
            _SENSOR_ASSOCIATION,
            f"{label} trajectory",
            association.trajectory_id,
            trajectory.trajectory_id,
        )
        expect(
            _SENSOR_ASSOCIATION,
            f"{label} state estimation run",
            association.state_estimation_run_id,
            trajectory.run_id,
        )
        expect(
            _SENSOR_ASSOCIATION,
            f"{label} calibration",
            association.calibration_identity,
            geometry.calibration_identity,
        )
        named = {str(run_id) for run_id in association.perception_run_ids}
        collector.expect(
            LINEAGE_CLOSURE,
            _SENSOR_ASSOCIATION,
            named <= perception_ids,
            f"{label} names perception runs {sorted(named - perception_ids)} "
            "that were not supplied",
        )
        consumed_perception |= named

    expect(_SEMANTIC_FUSION, "fusion map", fusion.lineage.geometric_map_id, geometry.map_id)
    supplied_associations = {str(association.run_id) for association in inputs.associations}
    named_associations = {str(run_id) for run_id in fusion.lineage.association_run_ids}
    collector.expect(
        LINEAGE_CLOSURE,
        _SEMANTIC_FUSION,
        named_associations == supplied_associations,
        f"fusion names association runs {sorted(named_associations)}, but the run consumed "
        f"{sorted(supplied_associations)}",
    )
    named_perception = {str(run_id) for run_id in fusion.lineage.perception_run_ids}
    collector.expect(
        LINEAGE_CLOSURE,
        _SEMANTIC_FUSION,
        consumed_perception <= named_perception <= perception_ids,
        f"fusion names perception runs {sorted(named_perception)}; the associations consumed "
        f"{sorted(consumed_perception)} out of {sorted(perception_ids)}",
    )


def _check_coordinates(inputs: CrossStageInputs, collector: _Collector) -> None:
    trajectory, geometry = inputs.trajectory, inputs.geometry
    collector.expect(
        COORDINATE_CONSISTENCY,
        _GEOMETRIC_MAPPING,
        str(geometry.map_frame) == str(trajectory.reference_frame),
        f"map frame {geometry.map_frame!r} differs from the trajectory reference frame "
        f"{trajectory.reference_frame!r}",
    )
    collector.expect(
        COORDINATE_CONSISTENCY,
        _GEOMETRIC_MAPPING,
        geometry.clock_id == trajectory.clock_id,
        f"map clock {geometry.clock_id!r} differs from the trajectory clock "
        f"{trajectory.clock_id!r}",
    )
    for observation in inputs.spatial_observations.values():
        label = f"observation {observation.spatial_observation_id}"
        collector.expect(
            COORDINATE_CONSISTENCY,
            _SENSOR_ASSOCIATION,
            str(observation.provenance.geometric_map_id) == str(geometry.map_id),
            f"{label} records map {observation.provenance.geometric_map_id!r}, "
            f"not {geometry.map_id!r}",
        )
        collector.expect(
            COORDINATE_CONSISTENCY,
            _SENSOR_ASSOCIATION,
            str(observation.pose_ref.trajectory_id) == str(trajectory.trajectory_id),
            f"{label} used trajectory {observation.pose_ref.trajectory_id!r}, "
            f"not {trajectory.trajectory_id!r}",
        )
        collector.expect(
            COORDINATE_CONSISTENCY,
            _SENSOR_ASSOCIATION,
            observation.calibration_ref.calibration_identity == geometry.calibration_identity,
            f"{label} used calibration {observation.calibration_ref.calibration_identity!r}, "
            f"not the map's {geometry.calibration_identity!r}",
        )
        for reference in observation.geometry_support:
            _expect_reference(
                inputs, collector, _SENSOR_ASSOCIATION, f"{label} geometry", reference
            )
    for outcome in inputs.fusion_outcomes:
        support = outcome.support
        collector.expect(
            COORDINATE_CONSISTENCY,
            _SEMANTIC_FUSION,
            str(support.geometric_map_id) == str(geometry.map_id),
            f"support {support.fusion_support_id} is over map {support.geometric_map_id!r}, "
            f"not {geometry.map_id!r}",
        )
        for reference in support.geometry_support:
            _expect_reference(
                inputs,
                collector,
                _SEMANTIC_FUSION,
                f"support {support.fusion_support_id}",
                reference,
            )


def _expect_reference(
    inputs: CrossStageInputs,
    collector: _Collector,
    capability: str,
    label: str,
    reference: object,
) -> None:
    """A reference is valid only if it points into the run's own map and resolves there."""
    map_id = getattr(reference, "map_id", None)
    if str(map_id) != str(inputs.geometry.map_id):
        collector.expect(
            COORDINATE_CONSISTENCY,
            capability,
            False,
            f"{label} references map {map_id!r}, not {inputs.geometry.map_id!r}",
        )
        return
    try:
        inputs.geometry_source.get(reference)  # type: ignore[arg-type]
        resolved = True
    except KeyError:
        resolved = False
    collector.expect(
        COORDINATE_CONSISTENCY,
        capability,
        resolved,
        f"{label} references geometry {getattr(reference, 'geometry_id', None)!r}, "
        "which the map does not contain",
    )


def _check_traceability(inputs: CrossStageInputs, collector: _Collector) -> None:
    claims_by_result = {
        result_id: {str(claim.claim_id) for claim in result.claims}
        for result_id, result in inputs.perception_results.items()
    }
    all_claims = set().union(*claims_by_result.values()) if claims_by_result else set()
    for observation in inputs.spatial_observations.values():
        label = f"observation {observation.spatial_observation_id}"
        result = inputs.perception_results.get(observation.perception_result_id)
        collector.expect(
            EVIDENCE_TRACEABILITY,
            _SENSOR_ASSOCIATION,
            result is not None,
            f"{label} names perception result {observation.perception_result_id!r}, "
            "which is not among the consumed results",
        )
        if result is None:
            continue
        collector.expect(
            EVIDENCE_TRACEABILITY,
            _SENSOR_ASSOCIATION,
            observation.region_id in {region.region_id for region in result.regions},
            f"{label} names region {observation.region_id!r}, which its result does not contain",
        )
        collector.expect(
            EVIDENCE_TRACEABILITY,
            _SENSOR_ASSOCIATION,
            observation.source_observation_id == result.source_observation_id,
            f"{label} is over frame {observation.source_observation_id!r} but its result is over "
            f"{result.source_observation_id!r}",
        )
        referenced = {str(ref.claim_id) for ref in observation.semantic_claim_refs}
        unknown_claims = sorted(referenced - claims_by_result[observation.perception_result_id])
        collector.expect(
            EVIDENCE_TRACEABILITY,
            _SENSOR_ASSOCIATION,
            referenced <= claims_by_result[observation.perception_result_id],
            f"{label} references claims {unknown_claims} "
            "that its perception result does not contain",
        )
    for outcome in inputs.fusion_outcomes:
        support, evidence = outcome.support, outcome.evidence
        label = f"support {support.fusion_support_id}"
        members = [
            inputs.spatial_observations.get(ident) for ident in support.spatial_observation_ids
        ]
        missing = [
            str(ident)
            for ident, member in zip(support.spatial_observation_ids, members, strict=True)
            if member is None
        ]
        collector.expect(
            EVIDENCE_TRACEABILITY,
            _SEMANTIC_FUSION,
            not missing,
            f"{label} is over spatial observations {missing} that no consumed association produced",
        )
        present = [member for member in members if member is not None]
        anchored = {str(ref.geometry_id) for member in present for ref in member.geometry_support}
        collector.expect(
            EVIDENCE_TRACEABILITY,
            _SEMANTIC_FUSION,
            {str(ref.geometry_id) for ref in support.geometry_support} <= anchored,
            f"{label} claims geometry that none of its observations supports",
        )
        # Nenhuma claim das observações do suporte pode sumir: cada uma precisa aparecer na
        # contribuição da sua observação, e a evidência das hipóteses só cita claims que existem.
        for member in present:
            contribution = next(
                (
                    item
                    for item in evidence.contributions
                    if item.spatial_observation_id == member.spatial_observation_id
                ),
                None,
            )
            kept = (
                set()
                if contribution is None
                else {str(ref.claim_id) for ref in contribution.claim_refs}
            )
            wanted = {str(ref.claim_id) for ref in member.semantic_claim_refs}
            collector.expect(
                EVIDENCE_TRACEABILITY,
                _SEMANTIC_FUSION,
                wanted <= kept,
                f"{label} dropped claims {sorted(wanted - kept)} of "
                f"{member.spatial_observation_id}",
            )
        cited = {
            str(item.claim_id) for hypothesis in evidence.hypotheses for item in hypothesis.evidence
        }
        collector.expect(
            EVIDENCE_TRACEABILITY,
            _SEMANTIC_FUSION,
            cited <= all_claims,
            f"{label} cites claims {sorted(cited - all_claims)} that no perception result contains",
        )


def _check_physical_identity(inputs: CrossStageInputs, collector: _Collector) -> None:
    """Repeated inference over one physical frame must stay one physical observation.

    The fused evidence guarantees its own grouping is consistent; what it cannot know is
    whether the grouping matches the association artifact it was built from, which is
    what is compared here.
    """
    for outcome in inputs.fusion_outcomes:
        support, evidence = outcome.support, outcome.evidence
        label = f"evidence of support {support.fusion_support_id}"
        members = {
            str(ident): inputs.spatial_observations[ident]
            for ident in support.spatial_observation_ids
            if ident in inputs.spatial_observations
        }
        frames = {str(member.source_observation_id) for member in members.values()}
        groups = evidence.physical_observation_groups
        collector.expect(
            PHYSICAL_OBSERVATION_IDENTITY,
            _SEMANTIC_FUSION,
            len(groups) == len(frames),
            f"{label} has {len(groups)} physical observation groups for {len(frames)} distinct "
            "frames of its observations: repeated inference was counted as another observation",
        )
        placed = {str(ident) for group in groups for ident in group.spatial_observation_ids}
        collector.expect(
            PHYSICAL_OBSERVATION_IDENTITY,
            _SEMANTIC_FUSION,
            placed == {str(ident) for ident in support.spatial_observation_ids},
            f"{label} groups observations {sorted(placed)} that differ from its support",
        )
        for group in groups:
            for ident in group.spatial_observation_ids:
                member = members.get(str(ident))
                collector.expect(
                    PHYSICAL_OBSERVATION_IDENTITY,
                    _SEMANTIC_FUSION,
                    member is None or member.source_observation_id == group.physical_observation_id,
                    f"{label} groups {ident} under physical observation "
                    f"{group.physical_observation_id!r}, but it is over "
                    f"{None if member is None else member.source_observation_id!r}",
                )

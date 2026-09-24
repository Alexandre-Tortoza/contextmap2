"""Cross-stage invariants over the synthetic chain: lineage, frames, traceability, identity.

Each test either shows the invariants hold across the real artifacts of the chain (fake/contract
evidence: formulas and canned model output) or corrupts exactly one boundary and requires the
evaluator to name the capability that broke it. The evaluator never repairs an artifact.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from chain import RUN_A, SyntheticChain, cross_stage_inputs, synthetic_chain

from contextmap.entity_resolution import (
    EntityResolutionRunId,
    ResolvedEntityId,
    ResolvedEntityReference,
)
from contextmap.evaluation.cross_stage import (
    COORDINATE_CONSISTENCY,
    EVIDENCE_TRACEABILITY,
    LINEAGE_CLOSURE,
    PHYSICAL_OBSERVATION_IDENTITY,
    CrossStageFinding,
    CrossStageInputs,
    check_cross_stage,
    cross_stage_gate_results,
)
from contextmap.evaluation.end_to_end import EvidenceClass, GateStatus
from contextmap.geometric_mapping import GeometryId, GeometryReference, MapId
from contextmap.ingestion import FrameId, SequenceArtifactId, SourceObservationId
from contextmap.semantic_fusion import SemanticFusionRunId
from contextmap.sensor_association import SemanticClaimRef, SpatialObservationId
from contextmap.state_estimation import StateEstimationRunId
from contextmap.visual_perception import ClaimId


@pytest.fixture
def chain(tmp_path: Path) -> Iterator[SyntheticChain]:
    with synthetic_chain(tmp_path) as built:
        yield built


def _findings(inputs: CrossStageInputs) -> tuple[CrossStageFinding, ...]:
    return check_cross_stage(inputs).findings


def test_an_intact_chain_has_no_finding_and_every_gate_ran_real_checks(
    chain: SyntheticChain,
) -> None:
    report = check_cross_stage(cross_stage_inputs(chain))

    assert report.findings == ()
    assert set(report.checks_run) == {
        LINEAGE_CLOSURE,
        COORDINATE_CONSISTENCY,
        EVIDENCE_TRACEABILITY,
        PHYSICAL_OBSERVATION_IDENTITY,
    }
    assert all(count > 0 for count in report.checks_run.values())


def test_repeated_inference_over_one_physical_observation_is_grouped_once(
    chain: SyntheticChain,
) -> None:
    # frame-0000 foi interpretado por duas runs: 4 contribuições, 3 observações físicas.
    pallet = chain.fusion_outcomes[0].evidence

    assert len(pallet.contributions) == 4
    assert len(pallet.physical_observation_groups) == 3
    assert _findings(cross_stage_inputs(chain)) == ()


def test_a_trajectory_of_another_sequence_is_a_state_estimation_lineage_break(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    other = replace(inputs.trajectory, sequence_artifact_id=SequenceArtifactId("another-sequence"))

    findings = _findings(replace(inputs, trajectory=other))

    assert {(f.gate_id, f.failing_capability) for f in findings} >= {
        (LINEAGE_CLOSURE, "state_estimation")
    }


def test_a_map_built_from_another_trajectory_run_breaks_lineage_at_geometry(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    other = replace(inputs.geometry, state_estimation_run_id=StateEstimationRunId("state-run-9999"))

    findings = _findings(replace(inputs, geometry=other))

    assert (LINEAGE_CLOSURE, "geometric_mapping") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_a_map_built_with_another_calibration_than_the_sequence_is_a_lineage_break(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    assert inputs.sequence_calibration_identity is not None
    other = replace(inputs, sequence_calibration_identity="sha256:" + "0" * 64)

    findings = _findings(other)

    assert {(f.gate_id, f.failing_capability) for f in findings} >= {
        (LINEAGE_CLOSURE, "geometric_mapping"),
        (LINEAGE_CLOSURE, "sensor_association"),
    }
    assert any("sequence calibration" in f.message for f in findings)


def test_without_the_sequence_calibration_the_comparison_is_a_stated_limitation(
    chain: SyntheticChain,
) -> None:
    inputs = replace(cross_stage_inputs(chain), sequence_calibration_identity=None)

    report = check_cross_stage(inputs)

    assert report.findings == ()
    assert any("sequence calibration" in note for note in report.notes)


def test_a_fusion_run_that_names_another_map_or_omits_an_association_run_breaks_lineage(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    lineage = replace(
        inputs.fusion.lineage,
        geometric_map_id=MapId("another-map"),
        association_run_ids=inputs.fusion.lineage.association_run_ids[:1],
    )

    findings = _findings(replace(inputs, fusion=replace(inputs.fusion, lineage=lineage)))

    messages = [f.message for f in findings if f.failing_capability == "semantic_fusion"]
    assert any("another-map" in message for message in messages)
    assert any("association" in message for message in messages)


def test_a_map_frame_that_differs_from_the_trajectory_frame_is_a_geometry_error(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)

    findings = _findings(
        replace(inputs, geometry=replace(inputs.geometry, map_frame=FrameId("world")))
    )

    assert (COORDINATE_CONSISTENCY, "geometric_mapping") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_an_observation_that_references_geometry_of_another_map_is_an_association_error(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    key = next(iter(inputs.spatial_observations))
    observation = inputs.spatial_observations[key]
    foreign = tuple(
        GeometryReference(map_id=MapId("another-map"), geometry_id=ref.geometry_id)
        for ref in observation.geometry_support
    )
    observations = {
        **inputs.spatial_observations,
        key: replace(
            observation,
            geometry_support=foreign,
            provenance=replace(observation.provenance, geometric_map_id=MapId("another-map")),
        ),
    }

    findings = _findings(replace(inputs, spatial_observations=observations))

    assert (COORDINATE_CONSISTENCY, "sensor_association") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_geometry_that_does_not_exist_in_the_map_is_reported_not_repaired(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    key = next(iter(inputs.spatial_observations))
    observation = inputs.spatial_observations[key]
    phantom = GeometryReference(
        map_id=inputs.geometry.map_id,
        geometry_id=GeometryId(f"{inputs.geometry.map_id}--geometry-999999"),
    )
    observations = {
        **inputs.spatial_observations,
        key: replace(observation, geometry_support=(*observation.geometry_support[:-1], phantom)),
    }

    findings = _findings(replace(inputs, spatial_observations=observations))

    assert any("999999" in f.message for f in findings)
    assert observations[key].geometry_support[-1] == phantom  # a entrada não foi alterada


def test_an_observation_whose_perception_result_is_missing_is_not_traceable(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    results = {
        key: value
        for key, value in inputs.perception_results.items()
        if str(value.run_id) != str(RUN_A) or "frame-0001" not in str(key)
    }

    findings = _findings(replace(inputs, perception_results=results))

    assert (EVIDENCE_TRACEABILITY, "sensor_association") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_a_claim_the_association_kept_but_fusion_never_saw_is_reported_as_lost(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    key = SpatialObservationId("spatial--run-a--frame-0000--region-pallet")
    observation = inputs.spatial_observations[key]
    result = inputs.perception_results[observation.perception_result_id]
    extra = replace(result.claims[0], claim_id=ClaimId(f"{result.result_id}--claim-9999"))
    results = {
        **inputs.perception_results,
        result.result_id: replace(result, claims=(*result.claims, extra)),
    }
    observations = {
        **inputs.spatial_observations,
        key: replace(
            observation,
            semantic_claim_refs=(
                *observation.semantic_claim_refs,
                SemanticClaimRef(claim_id=extra.claim_id),
            ),
        ),
    }

    findings = _findings(
        replace(inputs, perception_results=results, spatial_observations=observations)
    )

    assert (EVIDENCE_TRACEABILITY, "semantic_fusion") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_repeated_inference_counted_as_another_physical_observation_is_caught(
    chain: SyntheticChain,
) -> None:
    # A associação diz que a observação da run-b é sobre outro frame; a fusão a agrupou antes.
    inputs = cross_stage_inputs(chain)
    key = SpatialObservationId("spatial--run-b--frame-0000--region-pallet")
    observation = inputs.spatial_observations[key]
    observations = {
        **inputs.spatial_observations,
        key: replace(observation, source_observation_id=SourceObservationId("frame-0007")),
    }

    findings = _findings(replace(inputs, spatial_observations=observations))

    physical = [f for f in findings if f.gate_id == PHYSICAL_OBSERVATION_IDENTITY]
    assert {f.failing_capability for f in physical} == {"semantic_fusion"}
    assert any("frame-0007" in f.message for f in physical)
    assert any("repeated inference" in f.message for f in physical)


def test_a_support_over_an_unknown_observation_is_reported(chain: SyntheticChain) -> None:
    inputs = cross_stage_inputs(chain)
    key = SpatialObservationId("spatial--run-a--frame-0000--region-pallet")
    observations = {k: v for k, v in inputs.spatial_observations.items() if k != key}

    findings = _findings(replace(inputs, spatial_observations=observations))

    assert (EVIDENCE_TRACEABILITY, "semantic_fusion") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_gate_results_pass_only_the_gates_with_no_finding_and_name_the_failing_capability(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    broken = replace(inputs, geometry=replace(inputs.geometry, map_frame=FrameId("world")))

    results = {
        result.gate_id: result
        for result in cross_stage_gate_results(
            check_cross_stage(broken),
            evidence_class=EvidenceClass.FAKE_CONTRACT,
            evidence_refs=("chain:synthetic",),
        )
    }

    # The broken map frame also breaks the final artifact's own frame consistency check
    # (issue #178 extends coordinate checks through to the ContextMapArtifact).
    assert results[COORDINATE_CONSISTENCY].status is GateStatus.FAILED
    assert results[COORDINATE_CONSISTENCY].failing_capabilities == ("artifact", "geometric_mapping")
    assert results[LINEAGE_CLOSURE].status is GateStatus.PASSED
    assert results[EVIDENCE_TRACEABILITY].status is GateStatus.PASSED


def test_unverified_boundaries_block_a_gate_instead_of_passing_it(chain: SyntheticChain) -> None:
    report = check_cross_stage(cross_stage_inputs(chain))

    results = cross_stage_gate_results(
        report,
        evidence_class=EvidenceClass.FAKE_CONTRACT,
        evidence_refs=("chain:synthetic",),
        unverified_boundaries=("entity_resolution", "spatial_relations", "artifact"),
    )

    assert {result.status for result in results} == {GateStatus.BLOCKED}
    assert all(
        result.failing_capabilities == ("entity_resolution", "spatial_relations", "artifact")
        for result in results
    )
    assert all("0 findings" in result.detail for result in results)


# --- issue #178: semantic_mapping, entity_resolution, spatial_relations, ContextMapArtifact ------


def test_a_mapping_run_that_names_another_fusion_run_breaks_lineage(chain: SyntheticChain) -> None:
    inputs = cross_stage_inputs(chain)
    other_lineage = replace(
        inputs.mapping.lineage, fusion_run_id=SemanticFusionRunId("another-fusion-run")
    )

    findings = _findings(replace(inputs, mapping=replace(inputs.mapping, lineage=other_lineage)))

    assert (LINEAGE_CLOSURE, "semantic_mapping") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_a_resolution_run_that_names_another_mapping_run_breaks_lineage(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    other_lineage = replace(inputs.resolution.lineage, semantic_mapping_run_id="another-run")

    findings = _findings(
        replace(inputs, resolution=replace(inputs.resolution, lineage=other_lineage))
    )

    assert (LINEAGE_CLOSURE, "entity_resolution") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_a_relations_run_that_names_another_resolution_run_breaks_lineage(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    other_lineage = replace(
        inputs.relations.lineage,
        entity_resolution_run_id=EntityResolutionRunId("another-resolution-run"),
    )

    findings = _findings(
        replace(inputs, relations=replace(inputs.relations, lineage=other_lineage))
    )

    assert (LINEAGE_CLOSURE, "spatial_relations") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_a_context_map_built_from_another_resolution_run_is_a_closure_break(
    chain: SyntheticChain,
) -> None:
    """The map stays internally consistent (its own entities agree with its own lineage); what
    it cannot know is whether that lineage is the run this evaluation actually expects -- the
    same shape as a map built from another trajectory run breaking lineage at geometry."""
    inputs = cross_stage_inputs(chain)
    other = replace(inputs.resolution, run_id=EntityResolutionRunId("another-resolution-run"))

    findings = _findings(replace(inputs, resolution=other))

    assert (LINEAGE_CLOSURE, "artifact") in {(f.gate_id, f.failing_capability) for f in findings}


def test_an_entity_geometry_reference_to_another_map_is_a_mapping_coordinate_error(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    key = next(iter(inputs.entities))
    entity = inputs.entities[key]
    foreign = tuple(
        GeometryReference(map_id=MapId("another-map"), geometry_id=ref.geometry_id)
        for ref in entity.geometry.geometry_refs
    )
    entities = {
        **inputs.entities,
        key: replace(entity, geometry=replace(entity.geometry, geometry_refs=foreign)),
    }

    findings = _findings(replace(inputs, entities=entities))

    assert (COORDINATE_CONSISTENCY, "semantic_mapping") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_a_resolved_entity_whose_member_entity_is_missing_is_not_traceable(
    chain: SyntheticChain,
) -> None:
    """A resolved entity stays internally valid (its id is derived from real members); what it
    cannot know is whether semantic mapping still has each member -- that is what is compared
    here, the same shape as an association naming a perception result that was not supplied."""
    inputs = cross_stage_inputs(chain)
    key = next(iter(inputs.resolved_entities))
    missing_ref = inputs.resolved_entities[key].members[0].entity_ref
    entities = {ref: entity for ref, entity in inputs.entities.items() if ref != missing_ref}

    findings = _findings(replace(inputs, entities=entities))

    assert (EVIDENCE_TRACEABILITY, "entity_resolution") in {
        (f.gate_id, f.failing_capability) for f in findings
    }
    assert any(str(missing_ref.entity_id) in f.message for f in findings)


def test_a_relation_naming_an_unresolved_entity_endpoint_is_not_traceable(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    relation = inputs.relation_records[0]
    phantom = ResolvedEntityReference(
        resolution_run_id=relation.object_entity_ref.resolution_run_id,
        resolved_entity_id=ResolvedEntityId("resolved-entity-does-not-exist"),
    )
    records = (
        replace(relation, object_entity_ref=phantom),
        *inputs.relation_records[1:],
    )

    findings = _findings(replace(inputs, relation_records=records))

    assert (EVIDENCE_TRACEABILITY, "spatial_relations") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_a_context_entity_sourcing_an_unknown_resolved_entity_is_not_traceable(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    entity = inputs.context_map.entities[0]
    phantom_source = ResolvedEntityReference(
        resolution_run_id=entity.source.resolution_run_id,
        resolved_entity_id=ResolvedEntityId("resolved-entity-does-not-exist"),
    )
    entities = (replace(entity, source=phantom_source), *inputs.context_map.entities[1:])

    findings = _findings(
        replace(inputs, context_map=replace(inputs.context_map, entities=entities))
    )

    assert (EVIDENCE_TRACEABILITY, "artifact") in {
        (f.gate_id, f.failing_capability) for f in findings
    }


def test_a_resolved_entity_that_drops_a_members_physical_observation_is_caught(
    chain: SyntheticChain,
) -> None:
    inputs = cross_stage_inputs(chain)
    key = next(iter(inputs.resolved_entities))
    resolved = inputs.resolved_entities[key]
    shrunk = replace(
        resolved.evidence, physical_observation_ids=resolved.evidence.physical_observation_ids[:-1]
    )
    resolved_entities = {**inputs.resolved_entities, key: replace(resolved, evidence=shrunk)}

    findings = _findings(replace(inputs, resolved_entities=resolved_entities))

    assert (PHYSICAL_OBSERVATION_IDENTITY, "entity_resolution") in {
        (f.gate_id, f.failing_capability) for f in findings
    }

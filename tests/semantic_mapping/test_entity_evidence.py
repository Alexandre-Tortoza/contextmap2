import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from fusion_builders import make_region_feature
from mapping_builders import (
    MAP_ID,
    SUMMARY_POLICY,
    geometry_refs,
    make_evidence_links,
    make_fused_evidence_ref,
)
from mapping_fusion import ClaimSpec, FusionRun, View, entity_from_outcome, fuse, write_fusion_run
from mapping_geometry_fake import InMemoryGeometrySource

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    EvidenceChannel,
    EvidenceStance,
    FusedEvidence,
    FusedEvidenceId,
    FusionSupportId,
    SemanticFusionRunId,
    SemanticFusionRunManifest,
    SemanticFusionRunReader,
)
from contextmap.semantic_mapping import (
    Entity,
    EntityEvidenceLinks,
    EntityFeatureRef,
    EvidenceIntegrityIssue,
    EvidenceIntegrityKind,
    EvidenceTraceError,
    FusedEvidenceRef,
    FusedEvidenceSource,
    ObservationRef,
    feature_refs_of,
    fusion_artifact_digest,
    summarize_geometry,
    trace_entity_evidence,
    trace_geometry_sources,
    validate_entity_evidence,
)
from contextmap.semantic_mapping.serialization import decode_entity, encode_entity
from contextmap.sensor_association import SpatialObservationId
from contextmap.shared import SourceTimestamp
from contextmap.visual_perception import (
    FeatureId,
    FeatureScope,
    PerceptionResultId,
    PerceptionRunId,
    RegionId,
)

RUN_ID = SemanticFusionRunId("fusion-run-0001")


@pytest.fixture
def run(tmp_path: Path) -> FusionRun:
    return write_fusion_run(tmp_path)


def _entities(run: FusionRun) -> list[Entity]:
    return [entity_from_outcome(outcome, run) for outcome in run.outcomes]


def _kinds(issues: tuple[EvidenceIntegrityIssue, ...]) -> set[EvidenceIntegrityKind]:
    return {item.kind for item in issues}


def _floats(node: object) -> list[float]:
    """Every floating point number anywhere in a JSON-like record."""
    if isinstance(node, float):
        return [node]
    if isinstance(node, dict):
        return [number for value in node.values() for number in _floats(value)]
    if isinstance(node, list):
        return [number for value in node for number in _floats(value)]
    return []


def _with_ref(entity: Entity, ref: FusedEvidenceRef) -> EntityEvidenceLinks:
    return dataclasses.replace(entity.evidence, fused_evidence=(ref,))


def _entity_with_ref(entity: Entity, **changes: Any) -> Entity:
    (ref,) = entity.evidence.fused_evidence
    return dataclasses.replace(
        entity, evidence=_with_ref(entity, dataclasses.replace(ref, **changes))
    )


class TestLinksFromFusedEvidence:
    def test_the_links_list_what_the_fused_evidence_carries_and_nothing_more(
        self, run: FusionRun
    ) -> None:
        contradiction, agreement, _ = _entities(run)

        links = contradiction.evidence
        assert [ref.fusion_run_id for ref in links.fused_evidence] == [RUN_ID]
        # Três vistas (duas execuções sobre frame-0120 e uma sobre frame-0121), dois frames físicos.
        assert len(links.spatial_observation_ids) == 3
        assert links.physical_observation_ids == ("frame-0120", "frame-0121")
        # Uma feature de região por vista: os resultados de percepção são distintos.
        assert len(links.visual_feature_refs) == 3
        assert {ref.embedding_space_id for ref in links.visual_feature_refs} == {"dinov3-vit-b16"}
        assert len(links.point_representation_refs) == 1
        assert agreement.evidence.physical_observation_ids == ("frame-0130", "frame-0131")

    def test_repeated_inference_over_one_frame_does_not_inflate_the_physical_observations(
        self, run: FusionRun
    ) -> None:
        contradiction = _entities(run)[0]

        assert len(contradiction.evidence.spatial_observation_ids) == 3
        assert len(contradiction.evidence.physical_observation_ids) == 2

    def test_optional_channels_may_be_absent_without_invalidating_the_entity(
        self, run: FusionRun
    ) -> None:
        agreement = _entities(run)[1]

        assert agreement.evidence.visual_feature_refs == ()
        assert agreement.evidence.point_representation_refs == ()
        assert validate_entity_evidence(agreement, fusion_runs={RUN_ID: run.reader}) == ()

    def test_only_identities_are_stored_never_payloads(self, run: FusionRun) -> None:
        record = encode_entity(_entities(run)[0])["evidence"]

        assert set(record) == {
            "fused_evidence",
            "spatial_observation_ids",
            "physical_observation_ids",
            "visual_feature_refs",
            "point_representation_refs",
        }
        # Nenhum número de ponto flutuante: sem vetores, coordenadas nem scores copiados.
        assert not _floats(record)

    def test_a_reference_carries_the_artifact_identity_version_and_digest(
        self, run: FusionRun
    ) -> None:
        ref = _entities(run)[0].evidence.fused_evidence[0]

        assert ref.fusion_run_id == RUN_ID
        assert ref.fusion_schema_version == run.manifest.schema_version
        assert ref.fusion_artifact_digest == fusion_artifact_digest(run.manifest)

    def test_the_artifact_digest_follows_the_content_of_the_run(self, run: FusionRun) -> None:
        changed: SemanticFusionRunManifest = dataclasses.replace(
            run.manifest,
            file_inventory=(
                dataclasses.replace(run.manifest.file_inventory[0], content_hash="sha256:other"),
                *run.manifest.file_inventory[1:],
            ),
        )

        assert fusion_artifact_digest(changed) != fusion_artifact_digest(run.manifest)
        assert fusion_artifact_digest(run.manifest) == fusion_artifact_digest(run.manifest)


class TestFeatureIdentityAcrossPerceptionRuns:
    """``PerceptionResultId`` is local to a run and ``FeatureId`` to a result: only the run
    tells two features that reuse those identities apart."""

    @staticmethod
    def _feature(run: str, result: str = "result-0001") -> EntityFeatureRef:
        return EntityFeatureRef(
            perception_run_id=PerceptionRunId(run),
            perception_result_id=PerceptionResultId(result),
            feature_id=FeatureId("feature-0001"),
            embedding_space_id="dinov3-vit-b16",
            scope=FeatureScope.GLOBAL,
            region_id=None,
        )

    @staticmethod
    def _evidence_where_two_runs_reuse_the_same_ids() -> FusedEvidence:
        features = (make_region_feature(),)
        _, evidence = fuse(
            [
                View("run-a", "frame-0120", (ClaimSpec("door"),), features=features),
                View("run-b", "frame-0120", (ClaimSpec("door"),), features=features),
            ],
            policy=BaselineAccumulationPolicy(
                channels=frozenset(
                    {EvidenceChannel.SEMANTIC_CLAIMS, EvidenceChannel.VISUAL_FEATURES}
                )
            ),
        )
        shared = PerceptionResultId("result-0001")
        return dataclasses.replace(
            evidence,
            contributions=tuple(
                dataclasses.replace(item, perception_result_id=shared)
                for item in evidence.contributions
            ),
            physical_observation_groups=tuple(
                dataclasses.replace(group, perception_result_ids=(shared,))
                for group in evidence.physical_observation_groups
            ),
        )

    def test_features_of_two_runs_that_reuse_result_and_feature_ids_both_survive(self) -> None:
        evidence = self._evidence_where_two_runs_reuse_the_same_ids()

        refs = feature_refs_of(evidence)

        assert [
            (ref.perception_run_id, ref.perception_result_id, ref.feature_id) for ref in refs
        ] == [
            ("run-a", "result-0001", "feature-0001"),
            ("run-b", "result-0001", "feature-0001"),
        ]

    def test_the_links_accept_them_and_order_them_by_run_first(self) -> None:
        both = make_evidence_links(
            features=(self._feature("run-a", "result-0002"), self._feature("run-b", "result-0001"))
        )

        assert [ref.perception_run_id for ref in both.visual_feature_refs] == ["run-a", "run-b"]
        with pytest.raises(ValueError, match="visual_feature_refs must be sorted and unique"):
            make_evidence_links(
                features=(
                    self._feature("run-b", "result-0001"),
                    self._feature("run-a", "result-0002"),
                )
            )

    def test_a_feature_repeated_within_one_run_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="visual_feature_refs must be sorted and unique"):
            make_evidence_links(features=(self._feature("run-a"), self._feature("run-a")))


class TestLinksContract:
    def test_an_entity_needs_evidence_behind_it(self) -> None:
        with pytest.raises(ValueError, match="fused_evidence must not be empty"):
            EntityEvidenceLinks(
                fused_evidence=(),
                spatial_observation_ids=(SpatialObservationId("s"),),
                physical_observation_ids=(SourceObservationId("p"),),
            )

    def test_an_entity_must_list_the_observations_that_contributed(self) -> None:
        with pytest.raises(ValueError, match="observations that contributed"):
            EntityEvidenceLinks(
                fused_evidence=(make_fused_evidence_ref(),),
                spatial_observation_ids=(),
                physical_observation_ids=(SourceObservationId("p"),),
            )

    def test_references_are_canonical_and_never_duplicated(self) -> None:
        ref = make_fused_evidence_ref()

        with pytest.raises(ValueError, match="fused_evidence must be sorted and unique"):
            dataclasses.replace(make_evidence_links(), fused_evidence=(ref, ref))
        with pytest.raises(ValueError, match="spatial_observation_ids must be sorted and unique"):
            make_evidence_links(spatial=("b", "a"))
        with pytest.raises(ValueError, match="physical_observation_ids must be sorted and unique"):
            make_evidence_links(physical=("a", "a"))

    def test_a_duplicated_feature_or_representation_is_refused(self, run: FusionRun) -> None:
        links = _entities(run)[0].evidence
        feature = links.visual_feature_refs[0]
        (representation,) = links.point_representation_refs

        with pytest.raises(ValueError, match="visual_feature_refs must be sorted and unique"):
            dataclasses.replace(links, visual_feature_refs=(feature, feature))
        with pytest.raises(ValueError, match="point_representation_refs must be sorted and unique"):
            dataclasses.replace(links, point_representation_refs=(representation, representation))

    @pytest.mark.parametrize(
        ("scope", "region"),
        [(FeatureScope.REGION, None), (FeatureScope.GLOBAL, RegionId("region-0001"))],
    )
    def test_a_region_feature_needs_its_region_and_other_scopes_must_not_name_one(
        self, scope: FeatureScope, region: RegionId | None
    ) -> None:
        with pytest.raises(ValueError, match="region_id must be present exactly"):
            EntityFeatureRef(
                perception_run_id=PerceptionRunId("run-a"),
                perception_result_id=PerceptionResultId("result"),
                feature_id=FeatureId("feature"),
                embedding_space_id="space",
                scope=scope,
                region_id=region,
            )

    @pytest.mark.parametrize("missing", ["fusion_artifact_digest", "sequence_artifact_id"])
    def test_a_reference_needs_every_identity(self, missing: str) -> None:
        identities = {
            "fusion_run_id": RUN_ID,
            "fusion_schema_version": "0.1.0",
            "fusion_artifact_digest": "sha256:artifact",
            "sequence_artifact_id": "sequence-0001",
            "fused_evidence_id": FusedEvidenceId("fused"),
            "fusion_support_id": FusionSupportId("support"),
        }
        identities[missing] = " "

        with pytest.raises(ValueError, match=missing):
            FusedEvidenceRef(**identities)  # type: ignore[arg-type]

    def test_a_3d_representation_anchored_outside_the_entity_support_is_refused(
        self, run: FusionRun
    ) -> None:
        entity = _entities(run)[0]
        (representation,) = entity.evidence.point_representation_refs
        outside = dataclasses.replace(
            representation,
            geometry_reference=GeometryReference(
                map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=999)
            ),
        )

        with pytest.raises(ValueError, match="anchored outside the geometry support"):
            dataclasses.replace(
                entity,
                evidence=dataclasses.replace(entity.evidence, point_representation_refs=(outside,)),
            )


class TestReferenceIntegrity:
    def test_every_entity_of_a_real_run_validates_clean(self, run: FusionRun) -> None:
        for entity in _entities(run):
            issues = validate_entity_evidence(
                entity, fusion_runs={RUN_ID: run.reader}, geometry=run.geometry
            )
            assert issues == ()

    def test_the_reader_of_a_persisted_run_is_a_fused_evidence_source(self, run: FusionRun) -> None:
        assert isinstance(run.reader, FusedEvidenceSource)

    def test_a_missing_artifact_is_reported_and_nothing_else_is_guessed(
        self, run: FusionRun
    ) -> None:
        issues = validate_entity_evidence(_entities(run)[0], fusion_runs={})

        assert _kinds(issues) == {EvidenceIntegrityKind.MISSING_ARTIFACT}

    def test_another_identity_or_schema_version_is_an_artifact_mismatch(
        self, run: FusionRun
    ) -> None:
        entity = _entities(run)[0]
        old_schema = _entity_with_ref(entity, fusion_schema_version="0.0.1")
        other_run = _entity_with_ref(entity, fusion_run_id=SemanticFusionRunId("other"))

        assert _kinds(validate_entity_evidence(old_schema, fusion_runs={RUN_ID: run.reader})) == {
            EvidenceIntegrityKind.ARTIFACT_MISMATCH
        }
        assert _kinds(
            validate_entity_evidence(
                other_run, fusion_runs={SemanticFusionRunId("other"): run.reader}
            )
        ) == {EvidenceIntegrityKind.ARTIFACT_MISMATCH}

    def test_content_that_changed_since_materialization_is_a_stale_reference(
        self, run: FusionRun
    ) -> None:
        stale = _entity_with_ref(_entities(run)[0], fusion_artifact_digest="sha256:old")

        assert _kinds(validate_entity_evidence(stale, fusion_runs={RUN_ID: run.reader})) == {
            EvidenceIntegrityKind.STALE_REFERENCE
        }

    def test_a_corrupt_artifact_is_reported(self, run: FusionRun) -> None:
        entity = _entities(run)[0]
        target = run.run_dir / "outputs" / "fused-evidence.jsonl"
        target.write_bytes(target.read_bytes() + b"\n")

        issues = validate_entity_evidence(entity, fusion_runs={RUN_ID: run.reader})

        assert EvidenceIntegrityKind.CORRUPT_ARTIFACT in _kinds(issues)

    def test_fused_evidence_that_is_not_in_the_run_is_a_missing_reference(
        self, run: FusionRun
    ) -> None:
        gone = _entity_with_ref(
            _entities(run)[0], fusion_support_id=FusionSupportId("support-9999")
        )

        assert _kinds(validate_entity_evidence(gone, fusion_runs={RUN_ID: run.reader})) == {
            EvidenceIntegrityKind.MISSING_REFERENCE
        }

    def test_a_support_that_holds_other_fused_evidence_is_a_missing_reference(
        self, run: FusionRun
    ) -> None:
        class Renamed:
            """A run whose supports hold fused evidence under another identity."""

            manifest = run.manifest

            def fused_evidence(self, support_id: FusionSupportId) -> FusedEvidence:
                found = run.reader.fused_evidence(support_id)
                return dataclasses.replace(found, fused_evidence_id=FusedEvidenceId("fused--other"))

            def verify_integrity(self) -> list[str]:
                return []

        issues = validate_entity_evidence(_entities(run)[0], fusion_runs={RUN_ID: Renamed()})

        assert _kinds(issues) == {EvidenceIntegrityKind.MISSING_REFERENCE}

    def test_a_run_built_over_another_map_is_an_incompatible_lineage(self, run: FusionRun) -> None:
        entity = _entities(run)[1]
        other = MapId("map-0002")
        source = InMemoryGeometrySource(other, {i: (i * 0.1, 0.0, 0.0) for i in range(1_000)})
        moved = dataclasses.replace(
            entity,
            geometry=summarize_geometry(
                geometry_refs(range(100, 120), map_id=other), source=source, policy=SUMMARY_POLICY
            ),
        )

        assert EvidenceIntegrityKind.INCOMPATIBLE_LINEAGE in _kinds(
            validate_entity_evidence(moved, fusion_runs={RUN_ID: run.reader})
        )

    def test_a_reference_carries_the_sequence_the_fusion_run_was_built_over(
        self, run: FusionRun
    ) -> None:
        ref = _entities(run)[0].evidence.fused_evidence[0]

        assert ref.sequence_artifact_id == run.manifest.lineage.sequence_artifact_id

    def test_a_run_built_over_another_sequence_is_an_incompatible_lineage(
        self, run: FusionRun
    ) -> None:
        class OtherSequence:
            """The same run, map and inventory, whose manifest names another sequence."""

            manifest = dataclasses.replace(
                run.manifest,
                lineage=dataclasses.replace(
                    run.manifest.lineage, sequence_artifact_id="sequence-9999"
                ),
            )

            def fused_evidence(self, support_id: FusionSupportId) -> FusedEvidence:
                return run.reader.fused_evidence(support_id)

            def verify_integrity(self) -> list[str]:
                return []

        # A troca só da sequência não muda o digest: a detecção é uma comparação explícita.
        assert fusion_artifact_digest(OtherSequence.manifest) == fusion_artifact_digest(
            run.manifest
        )

        issues = validate_entity_evidence(_entities(run)[0], fusion_runs={RUN_ID: OtherSequence()})

        assert _kinds(issues) == {EvidenceIntegrityKind.INCOMPATIBLE_LINEAGE}
        assert "sequence" in issues[0].detail

    def test_an_entity_that_expects_another_sequence_is_an_incompatible_lineage(
        self, run: FusionRun
    ) -> None:
        elsewhere = _entity_with_ref(_entities(run)[0], sequence_artifact_id="sequence-9999")

        issues = validate_entity_evidence(elsewhere, fusion_runs={RUN_ID: run.reader})

        assert _kinds(issues) == {EvidenceIntegrityKind.INCOMPATIBLE_LINEAGE}

    def test_listed_observations_must_be_the_ones_the_evidence_carries(
        self, run: FusionRun
    ) -> None:
        entity = _entities(run)[0]
        links = entity.evidence
        fewer = dataclasses.replace(
            entity,
            evidence=dataclasses.replace(
                links, spatial_observation_ids=links.spatial_observation_ids[:-1]
            ),
        )
        temporal = entity.temporal_state
        extra = ObservationRef(
            physical_observation_id=SourceObservationId("frame-9999"),
            acquisition_timestamp=SourceTimestamp(
                seconds=temporal.last_seen.seconds + 100,
                nanoseconds=0,
                clock_id=temporal.last_seen.clock_id,
            ),
            inference_result_count=1,
        )
        stranger = dataclasses.replace(
            entity,
            evidence=dataclasses.replace(
                links,
                physical_observation_ids=(
                    *links.physical_observation_ids,
                    extra.physical_observation_id,
                ),
            ),
            temporal_state=dataclasses.replace(
                temporal,
                last_seen=extra.acquisition_timestamp,
                physical_observation_count=temporal.physical_observation_count + 1,
                inference_result_count=temporal.inference_result_count + 1,
                observation_refs=(*temporal.observation_refs, extra),
            ),
        )

        assert _kinds(validate_entity_evidence(fewer, fusion_runs={RUN_ID: run.reader})) == {
            EvidenceIntegrityKind.OBSERVATION_MISMATCH
        }
        assert _kinds(validate_entity_evidence(stranger, fusion_runs={RUN_ID: run.reader})) == {
            EvidenceIntegrityKind.OBSERVATION_MISMATCH
        }

    def test_listed_features_and_representations_must_be_the_ones_the_evidence_carries(
        self, run: FusionRun
    ) -> None:
        entity = _entities(run)[0]
        no_features = dataclasses.replace(
            entity, evidence=dataclasses.replace(entity.evidence, visual_feature_refs=())
        )
        no_representations = dataclasses.replace(
            entity, evidence=dataclasses.replace(entity.evidence, point_representation_refs=())
        )

        assert _kinds(validate_entity_evidence(no_features, fusion_runs={RUN_ID: run.reader})) == {
            EvidenceIntegrityKind.FEATURE_MISMATCH
        }
        assert _kinds(
            validate_entity_evidence(no_representations, fusion_runs={RUN_ID: run.reader})
        ) == {EvidenceIntegrityKind.POINT_REPRESENTATION_MISMATCH}

    def test_geometry_the_evidence_sees_outside_the_entity_support_is_reported(
        self, run: FusionRun
    ) -> None:
        entity = _entities(run)[1]
        smaller = summarize_geometry(
            entity.geometry.geometry_refs[:5], source=run.geometry, policy=SUMMARY_POLICY
        )
        narrowed = dataclasses.replace(entity, geometry=smaller)

        assert EvidenceIntegrityKind.GEOMETRY_OUTSIDE_SUPPORT in _kinds(
            validate_entity_evidence(narrowed, fusion_runs={RUN_ID: run.reader})
        )

    def test_a_geometry_reference_that_does_not_resolve_is_reported(self, run: FusionRun) -> None:
        entity = _entities(run)[1]
        partial = InMemoryGeometrySource(MAP_ID, {index: (0.0, 0.0, 0.0) for index in range(50)})

        issues = validate_entity_evidence(
            entity, fusion_runs={RUN_ID: run.reader}, geometry=partial
        )

        assert EvidenceIntegrityKind.MISSING_GEOMETRY in _kinds(issues)


class TestProvenanceTraversal:
    def test_an_entity_traces_back_to_its_views_claims_and_physical_frames(
        self, run: FusionRun
    ) -> None:
        entity = _entities(run)[0]

        trace = trace_entity_evidence(entity, fusion_runs={RUN_ID: run.reader})

        assert trace.entity == entity.reference
        assert len(trace.contributions) == 3
        assert {item.spatial_observation_id for item in trace.contributions} == set(
            entity.evidence.spatial_observation_ids
        )
        assert trace.physical_observation_ids == entity.evidence.physical_observation_ids
        assert all(item.claim_ids for item in trace.contributions)
        assert {item.perception_run_id for item in trace.contributions} == {"run-a", "run-b"}

    def test_every_claim_of_the_entity_semantic_state_is_reachable_in_the_trace(
        self, run: FusionRun
    ) -> None:
        entity = _entities(run)[0]
        trace = trace_entity_evidence(entity, fusion_runs={RUN_ID: run.reader})

        claims = {
            (item.contribution_id, claim)
            for item in trace.contributions
            for claim in item.claim_ids
        }
        reachable = [
            (evidence.contribution_id, evidence.claim_id)
            for hypothesis in entity.semantic_state.hypotheses
            for evidence in hypothesis.evidence
            if evidence.stance is not EvidenceStance.ABSTAINING
        ]
        assert reachable and all(pair in claims for pair in reachable)

    def test_the_trace_exposes_the_scorer_references_of_each_view(self, run: FusionRun) -> None:
        entity = _entities(run)[0]
        (ref,) = entity.evidence.fused_evidence
        fused = run.reader.fused_evidence(ref.fusion_support_id)

        trace = trace_entity_evidence(entity, fusion_runs={RUN_ID: run.reader})

        traced = {
            (item.contribution_id, score)
            for item in trace.contributions
            for score in item.score_refs
        }
        carried = {
            (item.contribution_id, score)
            for item in fused.contributions
            for score in item.score_refs
        }
        assert carried, "the fixture must score at least one claim"
        assert traced == carried
        for item in trace.contributions:
            assert {score.claim_id for score in item.score_refs} <= set(item.claim_ids)

    def test_a_trace_needs_the_run_and_the_evidence(self, run: FusionRun) -> None:
        entity = _entities(run)[0]
        gone = _entity_with_ref(entity, fusion_support_id=FusionSupportId("support-9999"))

        with pytest.raises(EvidenceTraceError, match="was not offered"):
            trace_entity_evidence(entity, fusion_runs={})
        with pytest.raises(EvidenceTraceError, match="cannot read"):
            trace_entity_evidence(gone, fusion_runs={RUN_ID: run.reader})

    def test_the_geometry_traces_back_to_the_source_observations(self, run: FusionRun) -> None:
        entity = _entities(run)[0]

        traces = trace_geometry_sources(entity.geometry, source=run.geometry)

        assert sum(item.point_count for item in traces) == len(entity.geometry.geometry_refs)
        assert [item.source_observation_id for item in traces] == ["lidar-frame-01824"]
        first, last = traces[0].first_acquired, traces[0].last_acquired
        assert first.total_nanoseconds() <= last.total_nanoseconds()


class TestLinkageSurvivesReopen:
    def test_an_entity_persisted_and_reloaded_still_validates_against_the_reopened_run(
        self, run: FusionRun
    ) -> None:
        entity = _entities(run)[0]
        reloaded = decode_entity(json.loads(json.dumps(encode_entity(entity))))
        reopened = SemanticFusionRunReader(run.run_dir)

        assert reloaded == entity
        assert validate_entity_evidence(reloaded, fusion_runs={RUN_ID: reopened}) == ()
        traced = trace_entity_evidence(reloaded, fusion_runs={RUN_ID: reopened})
        assert traced.physical_observation_ids == entity.evidence.physical_observation_ids

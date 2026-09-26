import json
from hashlib import sha256

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    AuditedRegions,
    BackendDiagnostics,
    BackendProvenance,
    BoundingBox,
    DiscoveryPassConfig,
    MergeKind,
    NormalizationConfig,
    PreparedImage,
    RegionCandidate,
    RegionDiscoveryAudit,
    RegionProvenance,
    RejectionReason,
)
from contextmap.visual_perception.discovery import (
    DiscoveryInput,
    DiscoveryOutput,
    discover_canonical_regions,
)

_PASSES = DiscoveryPassConfig(max_candidates_per_pass=3)
_NORMALIZATION = NormalizationConfig(minimum_area_pixels=2.0)
_DIAGNOSTICS = BackendDiagnostics(
    duration_ms=2.5,
    proposal_count=4,
    warnings=("fake warning",),
    metadata=(("raw_proposal_count", 5),),
)


def _image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-audit"),
        payload_reference="sequence/frames/frame-audit.png",
        payload_artifact=ArtifactReference(
            uri="sequence/frames/frame-audit.png",
            sha256=sha256(b"audit").hexdigest(),
            media_type="image/png",
        ),
        width=4,
        height=3,
        transformations=(),
    )


class _ProposalsToAudit:
    """Proposes on every pass a region, its duplicate, a sliver and one proposal over budget."""

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake",
            capability="region_discovery",
            provider="test",
            model="none",
            version="1",
            configuration_fingerprint="sha256:fake",
        )

    def discover_candidates(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        boxes = {
            "p-1": BoundingBox(0, 0, width, height),
            "p-2": BoundingBox(0, 0, width, height),
            "p-3": BoundingBox(0, 0, 1, 1),
            "p-4": BoundingBox(0, 0, 2, 2),
        }
        return DiscoveryOutput(
            candidates=tuple(
                RegionCandidate(
                    candidate_id=candidate_id,
                    source_observation_id=discovery_input.prepared_image.source_observation_id,
                    perception_run_id=discovery_input.perception_run_id,
                    perception_result_id=discovery_input.perception_result_id,
                    image_width=width,
                    image_height=height,
                    bounding_box=box,
                    provenance=RegionProvenance(
                        backend_id="fake",
                        backend_version="1",
                        checkpoint="none",
                        config_digest="sha256:fake",
                        discovery_pass_id=discovery_input.discovery_pass.pass_id,
                        native_proposal_id=candidate_id,
                    ),
                )
                for candidate_id, box in boxes.items()
            ),
            diagnostics=_DIAGNOSTICS,
        )


def _audited() -> AuditedRegions:
    return discover_canonical_regions(
        _image(),
        _ProposalsToAudit(),
        pass_config=_PASSES,
        normalization_config=_NORMALIZATION,
    )


def test_canonical_discovery_reports_every_pass_rejection_and_merge() -> None:
    audited = _audited()

    audit = audited.audit
    assert audit.source_observation_id == "frame-audit"
    assert audit.backend == _ProposalsToAudit().backend_provenance()
    assert [discovery_pass.pass_id for discovery_pass in audit.passes] == ["full-frame"]
    assert audit.diagnostics == (_DIAGNOSTICS,)
    assert [(item.candidate_id, item.reason) for item in audit.pass_rejections] == [
        ("full-frame/p-4", RejectionReason.REGION_BUDGET_EXCEEDED)
    ]
    assert [(item.candidate_id, item.reason) for item in audit.normalization_rejections] == [
        ("full-frame/p-3", RejectionReason.AREA_BELOW_MINIMUM),
        ("full-frame/p-2", RejectionReason.MERGED_DUPLICATE),
    ]
    assert [
        (decision.representative_candidate_id, decision.merged_candidate_id, decision.kind)
        for decision in audit.merge_decisions
    ] == [("full-frame/p-1", "full-frame/p-2", MergeKind.IOU_DUPLICATE)]
    assert audit.normalization_config_digest == _NORMALIZATION.digest
    # A região sobrevivente e a auditoria nomeiam os mesmos candidatos.
    (region,) = audited.regions
    assert region.contributor_candidate_ids == ("full-frame/p-1", "full-frame/p-2")


def test_the_audit_record_round_trips_through_json() -> None:
    audit = _audited().audit

    record = json.loads(json.dumps(audit.to_dict()))

    assert set(record) == {
        "source_observation_id",
        "backend",
        "passes",
        "pass_rejections",
        "normalization_config_digest",
        "normalization_rejections",
        "merge_decisions",
    }
    # Cada pass carrega o diagnóstico do backend naquele pass.
    assert record["passes"][0]["diagnostics"] == _DIAGNOSTICS.to_dict()
    assert RegionDiscoveryAudit.from_dict(record) == audit


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_dimensions", [1, 1]),
        ("unexpected", "value"),
    ],
)
def test_a_record_that_is_not_the_canonical_encoding_is_refused(field: str, value: object) -> None:
    record = json.loads(json.dumps(_audited().audit.to_dict()))
    record["passes"][0][field] = value

    with pytest.raises(ValueError, match="canonical"):
        RegionDiscoveryAudit.from_dict(record)


def test_every_pass_needs_its_backend_diagnostics() -> None:
    audit = _audited().audit

    with pytest.raises(ValueError, match="one diagnostic record"):
        RegionDiscoveryAudit(
            source_observation_id=audit.source_observation_id,
            backend=audit.backend,
            passes=audit.passes,
            diagnostics=(),
            pass_rejections=audit.pass_rejections,
            normalization_config_digest=audit.normalization_config_digest,
            normalization_rejections=audit.normalization_rejections,
            merge_decisions=audit.merge_decisions,
        )

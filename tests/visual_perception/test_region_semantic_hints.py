"""Native proposal text linked to canonical regions as evidence, never as belief."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from hashlib import sha256

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BackendProvenance,
    BoundingBox,
    HintContribution,
    NativeRegionText,
    NormalizationConfig,
    PreparedImage,
    RegionCandidate,
    RegionId,
    RegionProvenance,
    RegionSemanticHint,
    SemanticClaim,
    derive_region_semantic_hints,
    normalize_regions,
)

WIDTH = 8
HEIGHT = 6
BACKEND = "florence2_region_discovery"
CHECKPOINT = "florence-community/Florence-2-large"
CONFIG_DIGEST = "sha256:florence-od"


def _backend_provenance() -> BackendProvenance:
    return BackendProvenance(
        backend_id=BACKEND,
        capability="region_discovery",
        provider="microsoft",
        model=CHECKPOINT,
        version="1",
        configuration_fingerprint=CONFIG_DIGEST,
    )


def _prepared() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-1"),
        payload_reference="outputs/frame.png",
        payload_artifact=ArtifactReference(
            uri="outputs/frame.png",
            sha256=sha256(b"frame").hexdigest(),
            media_type="image/png",
        ),
        width=WIDTH,
        height=HEIGHT,
    )


def _candidate(
    candidate_id: str,
    box: BoundingBox,
    text: str | None,
    *,
    discovery_pass: str = "full-frame",
) -> RegionCandidate:
    native_proposal_id = candidate_id.split("/")[-1]
    return RegionCandidate(
        candidate_id=candidate_id,
        source_observation_id="frame-1",
        perception_run_id="run-1",
        perception_result_id="result-1",
        image_width=WIDTH,
        image_height=HEIGHT,
        bounding_box=box,
        provenance=RegionProvenance(
            backend_id=BACKEND,
            backend_version="1",
            checkpoint=CHECKPOINT,
            config_digest=CONFIG_DIGEST,
            discovery_pass_id=discovery_pass,
            native_proposal_id=native_proposal_id,
            query="<OD>",
        ),
        native_text=None if text is None else NativeRegionText(task="<OD>", text=text),
    )


def _hints(
    candidates: tuple[RegionCandidate, ...], config: NormalizationConfig | None = None
) -> dict[str, RegionSemanticHint]:
    normalization = normalize_regions(candidates, _prepared(), _backend_provenance(), config)
    return {
        hint.candidate_id: hint for hint in derive_region_semantic_hints(candidates, normalization)
    }


def test_native_text_requires_task_and_real_text_and_round_trips_on_its_candidate() -> None:
    with pytest.raises(ValueError, match="task"):
        NativeRegionText(task=" ", text="chair")
    with pytest.raises(ValueError, match="text"):
        NativeRegionText(task="<OD>", text="  ")
    with pytest.raises(ValueError, match="prompt"):
        NativeRegionText(task="<OPEN_VOCABULARY_DETECTION>", text="chair", prompt="")
    candidate = _candidate("full-frame/florence2-box-000000", BoundingBox(0, 0, 2, 2), "chair")

    restored = RegionCandidate.from_dict(json.loads(json.dumps(candidate.to_dict())))

    assert restored == candidate
    assert restored.native_text == NativeRegionText(task="<OD>", text="chair")


def test_text_stays_linked_to_its_exact_native_proposal_task_and_region() -> None:
    chair = _candidate("full-frame/florence2-box-000000", BoundingBox(0, 0, 2, 2), "chair")
    table = _candidate("full-frame/florence2-box-000001", BoundingBox(4, 2, 8, 6), "table")
    normalization = normalize_regions((table, chair), _prepared(), _backend_provenance())

    hints = derive_region_semantic_hints((table, chair), normalization)

    region_boxes = {region.region_id: region.bounding_box for region in normalization.regions}
    assert [hint.native_text.text for hint in hints] == ["chair", "table"]
    table_hint = hints[1]
    assert table_hint.candidate_id == "full-frame/florence2-box-000001"
    assert table_hint.provenance == table.provenance
    assert table_hint.provenance.native_proposal_id == "florence2-box-000001"
    assert table_hint.provenance.checkpoint == CHECKPOINT
    assert table_hint.provenance.config_digest == CONFIG_DIGEST
    assert table_hint.native_text.task == "<OD>"
    assert table_hint.source_observation_id == "frame-1"
    assert table_hint.perception_result_id == "result-1"
    assert table_hint.contribution is HintContribution.REPRESENTATIVE
    assert table_hint.region_id is not None
    table_box = region_boxes[table_hint.region_id]
    assert (table_box.x_min, table_box.y_min, table_box.x_max, table_box.y_max) == (4, 2, 8, 6)


def test_merge_never_turns_a_contributor_text_into_the_merged_region_own_text() -> None:
    # "seat" fica contido em "chair": o merge usa a geometria de quem vem primeiro na ordem
    # canônica, e o texto de cada proposta continua preso à sua própria geometria.
    chair = _candidate("full-frame/florence2-box-000000", BoundingBox(0, 0, 4, 4), "chair")
    seat = _candidate("full-frame/florence2-box-000001", BoundingBox(1, 1, 3, 3), "seat")
    lamp = _candidate("full-frame/florence2-box-000002", BoundingBox(6, 0, 8, 2), "lamp")
    config = NormalizationConfig(containment_threshold=0.9)

    hints = _hints((lamp, seat, chair), config)

    assert hints[chair.candidate_id].contribution is HintContribution.REPRESENTATIVE
    assert hints[seat.candidate_id].contribution is HintContribution.MERGED
    assert hints[seat.candidate_id].region_id == hints[chair.candidate_id].region_id
    assert hints[lamp.candidate_id].contribution is HintContribution.REPRESENTATIVE
    assert hints[lamp.candidate_id].region_id != hints[chair.candidate_id].region_id

    # Invertendo a ordem canônica, a geometria congelada passa a ser a do "seat"; o texto
    # "chair" não é promovido a texto da região, apenas registrado como contribuinte fundido.
    small_first = _candidate("full-frame/florence2-box-000000", BoundingBox(1, 1, 3, 3), "seat")
    large_second = _candidate("full-frame/florence2-box-000001", BoundingBox(0, 0, 4, 4), "chair")

    reversed_hints = _hints((large_second, small_first), config)

    assert reversed_hints[small_first.candidate_id].contribution is HintContribution.REPRESENTATIVE
    assert reversed_hints[large_second.candidate_id].contribution is HintContribution.MERGED


def test_every_contributor_of_a_merged_region_remains_an_auditable_hint() -> None:
    full_frame = _candidate("full-frame/florence2-box-000003", BoundingBox(2, 1, 6, 5), "chair")
    tile = _candidate(
        "tile-0001/florence2-box-000000",
        BoundingBox(2, 1, 6, 5),
        "armchair",
        discovery_pass="tile-0001",
    )
    normalization = normalize_regions((tile, full_frame), _prepared(), _backend_provenance())

    hints = derive_region_semantic_hints((tile, full_frame), normalization)

    (region,) = normalization.regions
    assert {hint.region_id for hint in hints} == {region.region_id}
    assert {hint.candidate_id for hint in hints} == set(region.contributor_candidate_ids)
    by_pass = {hint.provenance.discovery_pass_id: hint for hint in hints}
    assert by_pass["full-frame"].native_text.text == "chair"
    assert by_pass["full-frame"].contribution is HintContribution.REPRESENTATIVE
    assert by_pass["tile-0001"].native_text.text == "armchair"
    assert by_pass["tile-0001"].contribution is HintContribution.MERGED
    assert by_pass["tile-0001"].provenance.native_proposal_id == "florence2-box-000000"
    (decision,) = normalization.merge_decisions
    assert decision.merged_candidate_id == by_pass["tile-0001"].candidate_id


def test_text_of_a_proposal_that_reached_no_region_is_kept_without_a_region() -> None:
    tiny = _candidate("full-frame/florence2-box-000000", BoundingBox(0, 0, 1, 1), "cup")
    first = _candidate("full-frame/florence2-box-000001", BoundingBox(2, 0, 4, 2), "shelf")
    inside_cut = _candidate("full-frame/florence2-box-000002", BoundingBox(5, 0, 8, 3), "cabinet")
    handle = _candidate("full-frame/florence2-box-000003", BoundingBox(6, 1, 8, 2), "handle")
    config = NormalizationConfig(
        minimum_area_pixels=2, containment_threshold=0.9, maximum_regions=1
    )

    hints = _hints((tiny, first, inside_cut, handle), config)

    assert hints[tiny.candidate_id].contribution is HintContribution.REJECTED
    assert hints[tiny.candidate_id].region_id is None
    assert hints[first.candidate_id].contribution is HintContribution.REPRESENTATIVE
    # O grupo "cabinet" foi cortado pelo budget: nem ele nem o "handle" fundido nele
    # alcançam uma região, então nenhum dos dois textos pode apontar para uma.
    assert hints[inside_cut.candidate_id].region_id is None
    assert hints[handle.candidate_id].contribution is HintContribution.REJECTED
    assert hints[handle.candidate_id].region_id is None


def test_proposal_without_native_text_yields_no_hint() -> None:
    geometry_only = _candidate("full-frame/florence2-box-000000", BoundingBox(0, 0, 2, 2), None)
    labelled = _candidate("full-frame/florence2-box-000001", BoundingBox(4, 2, 8, 6), "table")

    hints = _hints((geometry_only, labelled))

    assert list(hints) == [labelled.candidate_id]


def test_repeated_derivation_is_deterministic_and_independent_of_input_order() -> None:
    candidates = (
        _candidate("full-frame/florence2-box-000002", BoundingBox(6, 0, 8, 2), "lamp"),
        _candidate("full-frame/florence2-box-000000", BoundingBox(0, 0, 4, 4), "chair"),
        _candidate("full-frame/florence2-box-000001", BoundingBox(1, 1, 3, 3), "seat"),
    )
    config = NormalizationConfig(containment_threshold=0.9)

    def materialize(order: tuple[RegionCandidate, ...]) -> str:
        normalization = normalize_regions(order, _prepared(), _backend_provenance(), config)
        hints = derive_region_semantic_hints(order, normalization)
        return json.dumps([hint.to_dict() for hint in hints], sort_keys=True)

    first = materialize(candidates)

    assert materialize(candidates) == first
    assert materialize(tuple(reversed(candidates))) == first
    assert [item["candidate_id"] for item in json.loads(first)] == sorted(
        candidate.candidate_id for candidate in candidates
    )


def test_hints_are_frame_evidence_and_create_no_claim_entity_or_confidence() -> None:
    candidate = _candidate("full-frame/florence2-box-000000", BoundingBox(0, 0, 2, 2), "chair")
    normalization = normalize_regions((candidate,), _prepared(), _backend_provenance())

    hints = derive_region_semantic_hints((candidate,), normalization)

    assert all(isinstance(hint, RegionSemanticHint) for hint in hints)
    assert not any(isinstance(hint, SemanticClaim) for hint in hints)
    names = {field.name for field in fields(RegionSemanticHint)} | set(hints[0].to_dict())
    assert not names & {"confidence", "hypothesis", "entity_id", "label", "score"}
    # A região congelada continua só geometria: o texto não entra no contrato de Region2D.
    assert "chair" not in json.dumps(normalization.regions[0].to_dict())


def test_derivation_refuses_a_normalization_of_another_candidate_set() -> None:
    chair = _candidate("full-frame/florence2-box-000000", BoundingBox(0, 0, 2, 2), "chair")
    table = _candidate("full-frame/florence2-box-000001", BoundingBox(4, 2, 8, 6), "table")
    normalization = normalize_regions((chair, table), _prepared(), _backend_provenance())

    with pytest.raises(ValueError, match="not among the discovery candidates"):
        derive_region_semantic_hints((chair,), normalization)
    with pytest.raises(ValueError, match="unique"):
        derive_region_semantic_hints((chair, chair, table), normalization)


def test_hint_region_link_is_consistent_with_its_contribution() -> None:
    candidate = _candidate("full-frame/florence2-box-000000", BoundingBox(0, 0, 2, 2), "chair")

    def hint(contribution: HintContribution, region_id: str | None) -> RegionSemanticHint:
        return RegionSemanticHint(
            candidate_id=candidate.candidate_id,
            source_observation_id=candidate.source_observation_id,
            perception_run_id=candidate.perception_run_id,
            perception_result_id=candidate.perception_result_id,
            provenance=candidate.provenance,
            native_text=NativeRegionText(task="<OD>", text="chair"),
            contribution=contribution,
            region_id=None if region_id is None else RegionId(region_id),
        )

    assert hint(HintContribution.MERGED, "region-0001").region_id == "region-0001"
    with pytest.raises(ValueError, match="region_id"):
        hint(HintContribution.REJECTED, "region-0001")
    with pytest.raises(ValueError, match="region_id"):
        hint(HintContribution.MERGED, None)
    with pytest.raises(ValueError, match="region_id"):
        hint(HintContribution.REPRESENTATIVE, None)


def test_derivation_refuses_a_region_whose_merge_lineage_names_no_single_representative() -> None:
    chair = _candidate("full-frame/florence2-box-000000", BoundingBox(0, 0, 4, 4), "chair")
    seat = _candidate("full-frame/florence2-box-000001", BoundingBox(1, 1, 3, 3), "seat")
    normalization = normalize_regions(
        (chair, seat),
        _prepared(),
        _backend_provenance(),
        NormalizationConfig(containment_threshold=0.9),
    )

    # Sem a decisão de merge, as duas propostas pareceriam representantes da mesma região.
    with pytest.raises(ValueError, match="exactly one representative"):
        derive_region_semantic_hints((chair, seat), replace(normalization, merge_decisions=()))

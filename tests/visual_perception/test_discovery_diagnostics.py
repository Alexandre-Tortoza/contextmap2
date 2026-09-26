import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    ArtifactReference,
    BackendProvenance,
    BoundingBox,
    InlineMask,
    NativeRegionText,
    PreparedImage,
    RegionCandidate,
    RegionProvenance,
)
from contextmap.visual_perception.diagnostics import (
    DebugLevel,
    DiscoveryAuditRecord,
    RegionDiscoveryEvidenceWriter,
)
from contextmap.visual_perception.discovery import (
    BackendDiagnostics,
    DiscoveryPass,
    DiscoveryRunResult,
    PassKind,
)
from contextmap.visual_perception.normalization import normalize_regions


def _record() -> DiscoveryAuditRecord:
    prepared = PreparedImage(
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
    box = BoundingBox(1, 0, 3, 2)
    candidate = RegionCandidate(
        candidate_id="full-frame/proposal-1",
        source_observation_id="frame-audit",
        perception_run_id="run-audit",
        perception_result_id="result-audit",
        image_width=4,
        image_height=3,
        bounding_box=box,
        mask=InlineMask(
            width=4,
            height=3,
            data=tuple(
                box.x_min <= x < box.x_max and box.y_min <= y < box.y_max
                for y in range(3)
                for x in range(4)
            ),
        ),
        provenance=RegionProvenance(
            backend_id="sam3",
            backend_version="3",
            checkpoint="facebook/sam3",
            config_digest="sha256:sam3",
            discovery_pass_id="full-frame",
            native_proposal_id="proposal-1",
            query="automatic:query-1",
        ),
    )
    discovery = DiscoveryRunResult(
        candidates=(candidate,),
        rejected=(),
        passes=(
            DiscoveryPass(
                pass_id="full-frame",
                kind=PassKind.FULL_FRAME,
                window=BoundingBox(0, 0, 4, 3),
            ),
        ),
        diagnostics=(
            BackendDiagnostics(
                duration_ms=4.5,
                proposal_count=1,
                warnings=("diagnostic warning",),
            ),
        ),
    )
    normalization = normalize_regions(
        discovery.candidates,
        prepared,
        BackendProvenance(
            backend_id="sam3",
            capability="region_discovery",
            provider="facebook",
            model="facebook/sam3",
            version="3",
            configuration_fingerprint="sha256:sam3",
        ),
    )
    return DiscoveryAuditRecord(
        prepared_image=prepared,
        backend_id="sam3",
        backend_config=(
            ("checkpoint", "facebook/sam3"),
            ("strategy", "automatic"),
        ),
        discovery=discovery,
        normalization=normalization,
    )


def test_none_level_writes_only_contractual_outputs_and_metrics(tmp_path: Path) -> None:
    stage = tmp_path / "20-region-discovery"

    written = RegionDiscoveryEvidenceWriter().write(stage, _record(), DebugLevel.NONE)

    assert written.stage_directory == stage
    assert (stage / "outputs" / "regions.jsonl").is_file()
    assert (stage / "outputs" / "metrics.json").is_file()
    assert (stage / "manifest.json").is_file()
    assert not (stage / "debug").exists()
    region_data = json.loads((stage / "outputs" / "regions.jsonl").read_text().strip())
    assert region_data["region_id"] == "region-0001"
    metrics = json.loads((stage / "outputs" / "metrics.json").read_text())
    assert metrics["raw_candidate_count"] == 1
    assert metrics["accepted_region_count"] == 1
    assert metrics["total_backend_duration_ms"] == 4.5


def test_standard_level_writes_structured_decisions_and_visual_overlays(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "20-region-discovery"

    RegionDiscoveryEvidenceWriter().write(stage, _record(), DebugLevel.STANDARD)

    debug = stage / "debug"
    expected = {
        "prepared-image.json",
        "backend-config.json",
        "passes.json",
        "candidates.jsonl",
        "accepted-regions.jsonl",
        "rejected-regions.jsonl",
        "merge-decisions.jsonl",
        "candidates-overlay.svg",
        "accepted-regions-overlay.svg",
        "rejected-regions-overlay.svg",
    }
    assert expected <= {path.name for path in debug.iterdir()}
    assert "full-frame/proposal-1" in (debug / "candidates-overlay.svg").read_text()
    backend_config = json.loads((debug / "backend-config.json").read_text())
    assert backend_config["strategy"] == "automatic"


def test_full_level_writes_per_region_mask_and_record(tmp_path: Path) -> None:
    stage = tmp_path / "20-region-discovery"

    RegionDiscoveryEvidenceWriter().write(stage, _record(), DebugLevel.FULL)

    region_directory = stage / "debug" / "regions" / "region-0001"
    assert (region_directory / "region.json").is_file()
    assert (region_directory / "mask.pbm").read_text().startswith("P1\n4 3\n")


def test_finalized_stage_is_immutable(tmp_path: Path) -> None:
    stage = tmp_path / "20-region-discovery"
    writer = RegionDiscoveryEvidenceWriter()
    writer.write(stage, _record(), DebugLevel.NONE)
    original_manifest = (stage / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="already finalized"):
        writer.write(stage, _record(), DebugLevel.FULL)

    assert (stage / "manifest.json").read_bytes() == original_manifest
    assert not list(tmp_path.glob(".20-region-discovery.tmp-*"))


def _record_with_native_text() -> DiscoveryAuditRecord:
    record = _record()
    (candidate,) = record.discovery.candidates
    labelled = replace(candidate, native_text=NativeRegionText(task="<OD>", text="monitor"))
    return replace(record, discovery=replace(record.discovery, candidates=(labelled,)))


def test_native_text_hints_are_contractual_and_deterministically_materialized(
    tmp_path: Path,
) -> None:
    first = RegionDiscoveryEvidenceWriter().write(
        tmp_path / "first", _record_with_native_text(), DebugLevel.NONE
    )
    second = RegionDiscoveryEvidenceWriter().write(
        tmp_path / "second", _record_with_native_text(), DebugLevel.STANDARD
    )

    hints_path = first.stage_directory / "outputs" / "region-semantic-hints.jsonl"
    (hint,) = [json.loads(line) for line in hints_path.read_text().splitlines()]
    assert hint["candidate_id"] == "full-frame/proposal-1"
    assert hint["region_id"] == "region-0001"
    assert hint["contribution"] == "representative"
    assert hint["native_text"] == {"task": "<OD>", "text": "monitor", "prompt": None}
    assert hint["provenance"]["native_proposal_id"] == "proposal-1"
    assert hint["provenance"]["checkpoint"] == "facebook/sam3"
    assert hint["provenance"]["config_digest"] == "sha256:sam3"
    assert (
        hints_path.read_bytes()
        == (second.stage_directory / "outputs" / "region-semantic-hints.jsonl").read_bytes()
    )
    manifest = json.loads(first.manifest_path.read_text())
    assert "outputs/region-semantic-hints.jsonl" in {item["path"] for item in manifest["files"]}
    regions = (first.stage_directory / "outputs" / "regions.jsonl").read_text()
    assert "monitor" not in regions


def test_geometry_only_discovery_writes_an_empty_hint_output(tmp_path: Path) -> None:
    written = RegionDiscoveryEvidenceWriter().write(tmp_path / "stage", _record(), DebugLevel.NONE)

    hints_path = written.stage_directory / "outputs" / "region-semantic-hints.jsonl"
    assert hints_path.read_text() == ""

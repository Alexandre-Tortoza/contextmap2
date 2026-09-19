from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    BackendProvenance,
    BoundingBox2D,
    DenseFeatureDiagnostic,
    EmbeddingSpace,
    FeatureDebugLevel,
    FeatureDiagnosticPreview,
    FeatureEventStatus,
    FeatureExtractionDiagnostic,
    FeatureId,
    FeatureScope,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRunId,
    PerceptionRunReader,
    PerceptionRunWriter,
    RegionFeatureDiagnostic,
    RegionId,
    RunArtifactError,
    VisualFeature,
)

_BACKEND = BackendProvenance(
    backend_id="dinov3_huggingface",
    capability="feature_extractor",
    provider="huggingface",
    model="facebook/dinov3-vits16",
    version="commit-abc",
    configuration_fingerprint="sha256:config",
)
_SPACE = EmbeddingSpace(
    family="dinov3",
    model="facebook/dinov3-vits16",
    version="commit-abc",
    checkpoint="facebook/dinov3-vits16@commit-abc",
    layer="last_hidden_state.patch_tokens_after_registers",
    dimension=4,
    normalization="none",
)


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "visual-perception" / "corridor" / "run-0001__frame-0001__features"


def _writer(tmp_path: Path, level: FeatureDebugLevel) -> PerceptionRunWriter:
    return PerceptionRunWriter(
        workspace_root=tmp_path,
        sequence_name="corridor",
        run_id=PerceptionRunId("run-0001"),
        run_index=1,
        sequence_artifact_id="corridor-artifact",
        selection_id="selection-0001",
        enabled_capabilities=frozenset({"feature_extractor"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:pipeline",
        selection_label="frame-0001",
        profile_label="features",
        feature_debug_level=level,
    )


def _dense_diagnostic() -> FeatureExtractionDiagnostic:
    return FeatureExtractionDiagnostic(
        event_id="dense-frame-0001",
        source_observation_id=SourceObservationId("frame-0001"),
        source_prepared_image_reference="prepared/frame-0001.png",
        source_image_width=8,
        source_image_height=8,
        stage_id="dense_feature_extraction",
        status=FeatureEventStatus.SUCCEEDED,
        backend=_BACKEND,
        feature_id=FeatureId("feature-dense"),
        scope=FeatureScope.DENSE,
        embedding_space=_SPACE,
        output_shape=(4, 4, 4),
        dtype="float32",
        normalization="none",
        payload_reference="frame-0001/feature-dense.npy",
        preprocessing=("rectify:v1", "resize:8x8"),
        duration_seconds=0.2,
        peak_memory_bytes=2048,
        warnings=("fixture warning",),
        dense=DenseFeatureDiagnostic(
            source_artifact_id="run-0001",
            grid_width=4,
            grid_height=4,
            stride_x=2.0,
            stride_y=2.0,
            support_width=2.0,
            support_height=2.0,
            coordinate_transform_id="sha256:transform",
        ),
    )


def _region_diagnostic() -> FeatureExtractionDiagnostic:
    return FeatureExtractionDiagnostic(
        event_id="region-frame-0001-region-0001",
        source_observation_id=SourceObservationId("frame-0001"),
        source_prepared_image_reference="prepared/frame-0001.png",
        source_image_width=8,
        source_image_height=8,
        stage_id="region_feature_extraction",
        status=FeatureEventStatus.SUCCEEDED,
        backend=BackendProvenance(
            backend_id="alphaclip_official",
            capability="feature_extractor",
            provider="SunzeY/AlphaCLIP",
            model="ViT-B/16",
            version="1",
            configuration_fingerprint="sha256:alpha-config",
        ),
        feature_id=FeatureId("feature-region"),
        scope=FeatureScope.REGION,
        embedding_space=EmbeddingSpace(
            family="alphaclip",
            model="ViT-B/16",
            version="1",
            checkpoint="sha256:alpha-checkpoint",
            layer="alpha_conditioned_image_projection",
            dimension=4,
            normalization="l2",
        ),
        output_shape=(4,),
        dtype="float32",
        normalization="l2",
        payload_reference="frame-0001/feature-region.npy",
        preprocessing=("context_box:0.25", "mask_resize:nearest"),
        duration_seconds=0.3,
        peak_memory_bytes=4096,
        region=RegionFeatureDiagnostic(
            region_id=RegionId("region-0001"),
            bounding_box=BoundingBox2D(x=1, y=2, width=3, height=4),
            support_box=BoundingBox2D(x=0, y=1, width=5, height=6),
            source_mask_reference="masks/region-0001.npy",
            mask_content_hash="sha256:mask",
            coordinate_transform_id="sha256:view",
            pooling_policy="mask_weighted_mean_v1",
            contributing_cell_count=4,
            total_weight=12,
            coverage_fraction=0.75,
        ),
    )


def test_debug_none_keeps_contractual_payload_and_required_metrics(tmp_path: Path) -> None:
    writer = _writer(tmp_path, FeatureDebugLevel.NONE)
    feature = VisualFeature(
        feature_id=FeatureId("feature-dense"),
        scope=FeatureScope.DENSE,
        embedding_space_id="space-dense",
        shape=(2, 2),
        dtype="float32",
        payload_reference="frame-0001/feature-dense.npy",
        provenance=_BACKEND,
    )
    writer.add_feature_payload(
        feature,
        SourceObservationId("frame-0001"),
        np.ones((2, 2), dtype=np.float32),
    )
    writer.add_result(
        PerceptionResult(
            result_id=PerceptionResultId("run-0001--frame-0001"),
            source_observation_id=SourceObservationId("frame-0001"),
            run_id=PerceptionRunId("run-0001"),
            sequence_artifact_id="corridor-artifact",
            created_at="2026-01-01T00:00:00+00:00",
            features=(feature,),
        )
    )
    writer.add_feature_diagnostic(_dense_diagnostic())
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    assert not (run_dir / "debug").exists()
    metrics = run_dir / "metrics" / "feature-extraction.jsonl"
    record = json.loads(metrics.read_text(encoding="utf-8"))
    assert record["backend_id"] == "dinov3_huggingface"
    assert record["duration_seconds"] == 0.2
    assert record["peak_memory_bytes"] == 2048
    reader = PerceptionRunReader(run_dir)
    np.testing.assert_array_equal(
        reader.feature_store().load(SourceObservationId("frame-0001"), feature.feature_id),
        np.ones((2, 2)),
    )
    assert reader.verify_integrity() == []


def test_standard_debug_writes_common_metadata_and_primary_previews(tmp_path: Path) -> None:
    writer = _writer(tmp_path, FeatureDebugLevel.STANDARD)
    writer.add_feature_diagnostic(_dense_diagnostic())
    writer.add_feature_diagnostic(_region_diagnostic())
    writer.add_feature_preview(
        FeatureDiagnosticPreview(
            relative_path="regions/region-0001/support-preview.jpg",
            content=b"support-preview",
            minimum_level=FeatureDebugLevel.STANDARD,
        )
    )
    writer.add_feature_preview(
        FeatureDiagnosticPreview(
            relative_path="regions/region-0001/mask-preview.png",
            content=b"full-only-mask-preview",
            minimum_level=FeatureDebugLevel.FULL,
        )
    )
    writer.finalize()

    debug_root = _run_dir(tmp_path) / "debug" / "30-feature-extraction"
    summary = json.loads((debug_root / "feature-summary.json").read_text(encoding="utf-8"))
    assert summary["feature_count_by_scope"] == {"dense": 1, "region": 1}
    assert summary["event_count_by_backend"] == {
        "alphaclip_official": 1,
        "dinov3_huggingface": 1,
    }
    spaces = json.loads((debug_root / "embedding-spaces.json").read_text(encoding="utf-8"))
    assert {space["family"] for space in spaces} == {"alphaclip", "dinov3"}
    dense_meta = json.loads((debug_root / "feature-map-meta.json").read_text(encoding="utf-8"))
    assert dense_meta[0]["dense"]["grid_width"] == 4
    region_meta = json.loads(
        (debug_root / "regions" / "region-0001" / "feature-meta.json").read_text(encoding="utf-8")
    )
    assert region_meta["features"][0]["region"]["mask_content_hash"] == "sha256:mask"
    assert (debug_root / "regions" / "region-0001" / "support-preview.jpg").read_bytes() == (
        b"support-preview"
    )
    assert not (debug_root / "regions" / "region-0001" / "mask-preview.png").exists()
    assert not (debug_root / "regions" / "region-0001" / "pooling.json").exists()


def test_full_debug_adds_pooling_evidence_and_full_previews(tmp_path: Path) -> None:
    writer = _writer(tmp_path, FeatureDebugLevel.FULL)
    writer.add_feature_diagnostic(_region_diagnostic())
    writer.add_feature_preview(
        FeatureDiagnosticPreview(
            relative_path="regions/region-0001/mask-preview.png",
            content=b"mask-preview",
            minimum_level=FeatureDebugLevel.FULL,
        )
    )
    writer.finalize()

    region_root = _run_dir(tmp_path) / "debug" / "30-feature-extraction" / "regions" / "region-0001"
    pooling = json.loads((region_root / "pooling.json").read_text(encoding="utf-8"))
    assert pooling == {
        "pooling": [
            {
                "contributing_cell_count": 4,
                "coverage_fraction": 0.75,
                "feature_id": "feature-region",
                "pooling_policy": "mask_weighted_mean_v1",
                "total_weight": 12,
            }
        ]
    }
    assert (region_root / "mask-preview.png").read_bytes() == b"mask-preview"


def test_failures_and_abstentions_are_recorded_without_fake_feature_metadata() -> None:
    failure = FeatureExtractionDiagnostic(
        event_id="failed",
        source_observation_id=SourceObservationId("frame-0001"),
        source_prepared_image_reference="prepared/frame-0001.png",
        source_image_width=8,
        source_image_height=8,
        stage_id="dense_feature_extraction",
        status=FeatureEventStatus.FAILED,
        backend=_BACKEND,
        duration_seconds=0.1,
        failure_reason="checkpoint missing",
    )
    assert failure.feature_id is None

    with pytest.raises(ValueError, match="failure_reason"):
        FeatureExtractionDiagnostic(
            event_id="invalid",
            source_observation_id=SourceObservationId("frame-0001"),
            source_prepared_image_reference="prepared/frame-0001.png",
            source_image_width=8,
            source_image_height=8,
            stage_id="dense_feature_extraction",
            status=FeatureEventStatus.SUCCEEDED,
            backend=_BACKEND,
            failure_reason="must not exist",
        )


def test_preview_paths_cannot_escape_debug_root() -> None:
    with pytest.raises(ValueError, match="relative_path"):
        FeatureDiagnosticPreview(
            relative_path="../outputs/features/payload.npy",
            content=b"not allowed",
            minimum_level=FeatureDebugLevel.STANDARD,
        )


def test_diagnostics_cannot_be_added_after_finalize(tmp_path: Path) -> None:
    writer = _writer(tmp_path, FeatureDebugLevel.NONE)
    writer.finalize()

    with pytest.raises(RunArtifactError, match="after finalize"):
        writer.add_feature_diagnostic(_dense_diagnostic())

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    BackendProvenance,
    BoundingBox2D,
    DenseFeatureDiagnostic,
    DenseFeatureSampling,
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
    Region2D,
    RegionFeatureDiagnostic,
    RegionId,
    RunArtifactError,
    VisualFeature,
    embedding_space_fingerprint,
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


_REGION_BACKEND = BackendProvenance(
    backend_id="alphaclip_official",
    capability="feature_extractor",
    provider="SunzeY/AlphaCLIP",
    model="ViT-B/16",
    version="1",
    configuration_fingerprint="sha256:alpha-config",
)
_REGION_SPACE = EmbeddingSpace(
    family="alphaclip",
    model="ViT-B/16",
    version="1",
    checkpoint="sha256:alpha-checkpoint",
    layer="alpha_conditioned_image_projection",
    dimension=4,
    normalization="l2",
)
_OBSERVATION = SourceObservationId("frame-0001")


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "visual_perception"


def _writer(tmp_path: Path, level: FeatureDebugLevel) -> PerceptionRunWriter:
    return PerceptionRunWriter(
        output_dir=_run_dir(tmp_path),
        sequence_name="corridor",
        run_id=PerceptionRunId("run-0001"),
        run_index=1,
        sequence_artifact_id="corridor-artifact",
        selection_id="selection-0001",
        enabled_capabilities=frozenset({"feature_extractor"}),
        pipeline_preset=CANONICAL_PRESET_V1,
        configuration_digest="sha256:pipeline",
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
            origin_x=1.0,
            origin_y=0.5,
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
        backend=_REGION_BACKEND,
        feature_id=FeatureId("feature-region"),
        scope=FeatureScope.REGION,
        embedding_space=_REGION_SPACE,
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


def _dense_feature() -> VisualFeature:
    """Feature that ``_dense_diagnostic()`` describes exactly (4x4 grid, 4 channels)."""
    return VisualFeature(
        feature_id=FeatureId("feature-dense"),
        scope=FeatureScope.DENSE,
        embedding_space_id=embedding_space_fingerprint(_SPACE),
        shape=(4, 4, 4),
        dtype="float32",
        normalization="none",
        payload_reference="frame-0001/feature-dense.npy",
        provenance=_BACKEND,
    )


def _region_feature() -> VisualFeature:
    """Feature that ``_region_diagnostic()`` describes exactly."""
    return VisualFeature(
        feature_id=FeatureId("feature-region"),
        scope=FeatureScope.REGION,
        embedding_space_id=embedding_space_fingerprint(_REGION_SPACE),
        shape=(4,),
        dtype="float32",
        normalization="l2",
        payload_reference="frame-0001/feature-region.npy",
        provenance=_REGION_BACKEND,
        region_id=RegionId("region-0001"),
    )


def _writer_with_features(
    tmp_path: Path,
    *features: VisualFeature,
    level: FeatureDebugLevel = FeatureDebugLevel.NONE,
    persist_payloads: bool = True,
) -> PerceptionRunWriter:
    """Writer holding frame-0001's result with ``features`` and, by default, their payloads."""
    writer = _writer(tmp_path, level)
    if persist_payloads:
        for feature in features:
            writer.add_feature_payload(
                feature, _OBSERVATION, np.ones(feature.shape, dtype=feature.dtype)
            )
    regions = tuple(
        Region2D(
            region_id=feature.region_id,
            bounding_box=BoundingBox2D(x=1, y=2, width=3, height=4),
            provenance=_REGION_BACKEND,
        )
        for feature in features
        if feature.region_id is not None
    )
    writer.add_result(
        PerceptionResult(
            result_id=PerceptionResultId("run-0001--frame-0001"),
            source_observation_id=_OBSERVATION,
            run_id=PerceptionRunId("run-0001"),
            sequence_artifact_id="corridor-artifact",
            created_at="2026-01-01T00:00:00+00:00",
            regions=regions,
            features=features,
        )
    )
    return writer


def _assert_rejected_atomically(writer: PerceptionRunWriter, tmp_path: Path, message: str) -> None:
    """Finalize must fail with ``message`` and leave neither a run nor a temporary directory."""
    with pytest.raises(RunArtifactError, match=message):
        writer.finalize()
    sequence_dir = _run_dir(tmp_path).parent
    assert not sequence_dir.exists() or not any(sequence_dir.iterdir())


def test_debug_none_keeps_contractual_payload_and_required_metrics(tmp_path: Path) -> None:
    writer = _writer_with_features(tmp_path, _dense_feature())
    feature = _dense_feature()
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
        reader.feature_store().load(_OBSERVATION, feature.feature_id),
        np.ones((4, 4, 4)),
    )
    assert reader.verify_integrity() == []


def test_standard_debug_writes_common_metadata_and_primary_previews(tmp_path: Path) -> None:
    writer = _writer_with_features(
        tmp_path, _dense_feature(), _region_feature(), level=FeatureDebugLevel.STANDARD
    )
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
    writer = _writer_with_features(tmp_path, _region_feature(), level=FeatureDebugLevel.FULL)
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


def test_dense_sampling_is_rebuildable_from_required_metrics_without_debug(tmp_path: Path) -> None:
    """Regression: a real run showed the geometry only in debug and without the grid origin."""
    writer = _writer_with_features(tmp_path, _dense_feature())
    writer.add_feature_diagnostic(_dense_diagnostic())
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    assert not (run_dir / "debug").exists()
    record = json.loads(
        (run_dir / "metrics" / "feature-extraction.jsonl").read_text(encoding="utf-8")
    )

    persisted = record["dense"]
    assert persisted["origin_x"] == 1.0
    assert persisted["origin_y"] == 0.5
    rebuilt = DenseFeatureSampling(
        grid_width=persisted["grid_width"],
        grid_height=persisted["grid_height"],
        source_image_width=persisted["source_image_width"],
        source_image_height=persisted["source_image_height"],
        origin_x=persisted["origin_x"],
        origin_y=persisted["origin_y"],
        stride_x=persisted["stride_x"],
        stride_y=persisted["stride_y"],
        support_width=persisted["support_width"],
        support_height=persisted["support_height"],
        coordinate_transform_id=persisted["coordinate_transform_id"],
    )
    assert rebuilt == DenseFeatureSampling(
        grid_width=4,
        grid_height=4,
        source_image_width=8,
        source_image_height=8,
        origin_x=1.0,
        origin_y=0.5,
        stride_x=2.0,
        stride_y=2.0,
        support_width=2.0,
        support_height=2.0,
        coordinate_transform_id="sha256:transform",
    )
    assert persisted["source_artifact_id"] == "run-0001"


def test_non_dense_metrics_carry_no_dense_geometry(tmp_path: Path) -> None:
    writer = _writer_with_features(tmp_path, _region_feature())
    writer.add_feature_diagnostic(_region_diagnostic())
    writer.finalize()

    record = json.loads(
        (_run_dir(tmp_path) / "metrics" / "feature-extraction.jsonl").read_text(encoding="utf-8")
    )

    assert record["dense"] is None


@pytest.mark.parametrize("axis", ["origin_x", "origin_y"])
def test_dense_diagnostic_rejects_non_finite_origin(axis: str) -> None:
    values: dict[str, float] = {"origin_x": 0.0, "origin_y": 0.0, axis: float("nan")}
    with pytest.raises(ValueError, match=axis):
        DenseFeatureDiagnostic(
            source_artifact_id="run-0001",
            grid_width=4,
            grid_height=4,
            stride_x=2.0,
            stride_y=2.0,
            support_width=2.0,
            support_height=2.0,
            coordinate_transform_id="sha256:transform",
            **values,
        )


def test_finalize_accepts_diagnostics_that_describe_their_persisted_features(
    tmp_path: Path,
) -> None:
    writer = _writer_with_features(tmp_path, _dense_feature(), _region_feature())
    writer.add_feature_diagnostic(_dense_diagnostic())
    writer.add_feature_diagnostic(_region_diagnostic())

    writer.finalize()

    records = (
        (_run_dir(tmp_path) / "metrics" / "feature-extraction.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert [json.loads(record)["feature_id"] for record in records] == [
        "feature-dense",
        "feature-region",
    ]
    assert PerceptionRunReader(_run_dir(tmp_path)).verify_integrity() == []


def test_finalize_rejects_a_dense_diagnostic_declared_for_another_persisted_shape(
    tmp_path: Path,
) -> None:
    """Regression (#423 review): a 4x4/(4, 4, 4) diagnostic was accepted for a (2, 2) feature."""
    writer = _writer_with_features(tmp_path, replace(_dense_feature(), shape=(2, 2)))
    writer.add_feature_diagnostic(_dense_diagnostic())

    _assert_rejected_atomically(writer, tmp_path, "output_shape")


def test_finalize_rejects_a_dense_grid_that_disagrees_with_the_feature_shape(
    tmp_path: Path,
) -> None:
    """The declared numerical shape matches, but the grid does not describe its spatial axes."""
    feature = replace(_dense_feature(), shape=(2, 2, 4))
    writer = _writer_with_features(tmp_path, feature)
    writer.add_feature_diagnostic(replace(_dense_diagnostic(), output_shape=(2, 2, 4)))

    _assert_rejected_atomically(writer, tmp_path, "grid")


def test_finalize_rejects_a_dense_grid_with_swapped_axes(tmp_path: Path) -> None:
    """``feature.shape[:2]`` is ``(grid_height, grid_width)``; a transposed grid is another map."""
    feature = replace(_dense_feature(), shape=(2, 3, 4))
    writer = _writer_with_features(tmp_path, feature)
    diagnostic = _dense_diagnostic()
    assert diagnostic.dense is not None
    writer.add_feature_diagnostic(
        replace(
            diagnostic,
            output_shape=(2, 3, 4),
            dense=replace(diagnostic.dense, grid_width=2, grid_height=3),
        )
    )

    _assert_rejected_atomically(writer, tmp_path, "grid")


def test_finalize_rejects_a_dense_feature_that_is_not_height_width_channels(
    tmp_path: Path,
) -> None:
    feature = replace(_dense_feature(), shape=(4, 4))
    writer = _writer_with_features(tmp_path, feature)
    writer.add_feature_diagnostic(replace(_dense_diagnostic(), output_shape=(4, 4)))

    _assert_rejected_atomically(writer, tmp_path, "height, width, channels")


def test_finalize_rejects_dense_geometry_owned_by_another_run(tmp_path: Path) -> None:
    writer = _writer_with_features(tmp_path, _dense_feature())
    diagnostic = _dense_diagnostic()
    assert diagnostic.dense is not None
    writer.add_feature_diagnostic(
        replace(diagnostic, dense=replace(diagnostic.dense, source_artifact_id="other-run"))
    )

    _assert_rejected_atomically(writer, tmp_path, "source_artifact_id")


@pytest.mark.parametrize(
    ("field_name", "override"),
    [
        pytest.param(
            "scope", {"scope": FeatureScope.GLOBAL, "dense": None}, id="scope-global-vs-dense"
        ),
        pytest.param("dtype", {"dtype": "float64"}, id="dtype"),
        pytest.param("normalization", {"normalization": "l2"}, id="normalization"),
        pytest.param("payload_reference", {"payload_reference": "other.npy"}, id="payload"),
        pytest.param(
            "backend",
            {"backend": replace(_BACKEND, configuration_fingerprint="sha256:other")},
            id="provenance",
        ),
        pytest.param(
            "embedding_space",
            {"embedding_space": replace(_SPACE, layer="other_layer")},
            id="embedding-space-fingerprint",
        ),
    ],
)
def test_finalize_rejects_a_diagnostic_whose_metadata_differs_from_its_feature(
    tmp_path: Path, field_name: str, override: dict[str, Any]
) -> None:
    writer = _writer_with_features(tmp_path, _dense_feature())
    writer.add_feature_diagnostic(replace(_dense_diagnostic(), **override))

    _assert_rejected_atomically(writer, tmp_path, field_name)


def test_finalize_rejects_a_region_diagnostic_bound_to_another_region(tmp_path: Path) -> None:
    writer = _writer_with_features(tmp_path, _region_feature())
    diagnostic = _region_diagnostic()
    assert diagnostic.region is not None
    writer.add_feature_diagnostic(
        replace(diagnostic, region=replace(diagnostic.region, region_id=RegionId("region-0002")))
    )

    _assert_rejected_atomically(writer, tmp_path, "region_id")


@pytest.mark.parametrize(
    "status", [FeatureEventStatus.SUCCEEDED, FeatureEventStatus.WARNING], ids=lambda s: s.value
)
def test_finalize_rejects_a_produced_diagnostic_without_its_feature(
    tmp_path: Path, status: FeatureEventStatus
) -> None:
    writer = _writer_with_features(tmp_path)
    writer.add_feature_diagnostic(replace(_dense_diagnostic(), status=status))

    _assert_rejected_atomically(writer, tmp_path, "exactly one result feature")


def test_finalize_rejects_a_diagnostic_whose_feature_belongs_to_another_observation(
    tmp_path: Path,
) -> None:
    writer = _writer_with_features(tmp_path, _dense_feature())
    writer.add_feature_diagnostic(
        replace(_dense_diagnostic(), source_observation_id=SourceObservationId("frame-0002"))
    )

    _assert_rejected_atomically(writer, tmp_path, "exactly one result feature")


def test_finalize_rejects_two_diagnostics_for_the_same_produced_feature(tmp_path: Path) -> None:
    """Two records with possibly different geometry would make the contractual mapping ambiguous."""
    writer = _writer_with_features(tmp_path, _dense_feature())
    writer.add_feature_diagnostic(_dense_diagnostic())
    writer.add_feature_diagnostic(replace(_dense_diagnostic(), event_id="dense-frame-0001-again"))

    _assert_rejected_atomically(writer, tmp_path, "more than one")


def test_failed_and_abstained_diagnostics_do_not_require_a_feature(tmp_path: Path) -> None:
    writer = _writer_with_features(tmp_path)
    for status in (FeatureEventStatus.FAILED, FeatureEventStatus.ABSTAINED):
        writer.add_feature_diagnostic(
            FeatureExtractionDiagnostic(
                event_id=f"{status.value}-frame-0001",
                source_observation_id=_OBSERVATION,
                source_prepared_image_reference="prepared/frame-0001.png",
                source_image_width=8,
                source_image_height=8,
                stage_id="dense_feature_extraction",
                status=status,
                backend=_BACKEND,
                failure_reason="checkpoint missing",
            )
        )

    writer.finalize()

    records = [
        json.loads(line)
        for line in (_run_dir(tmp_path) / "metrics" / "feature-extraction.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(record["status"], record["feature_id"], record["dense"]) for record in records] == [
        ("failed", None, None),
        ("abstained", None, None),
    ]


def test_diagnostic_does_not_require_the_optional_payload_of_a_persisted_feature(
    tmp_path: Path,
) -> None:
    """Payload persistence is opt-in per feature; the binding is to the result's feature."""
    writer = _writer_with_features(tmp_path, _dense_feature(), persist_payloads=False)
    writer.add_feature_diagnostic(_dense_diagnostic())

    writer.finalize()

    assert PerceptionRunReader(_run_dir(tmp_path)).verify_integrity() == []


_SAMPLING_FIELDS = ("stride_x", "stride_y", "support_width", "support_height")


def _dense_values(**overrides: float) -> dict[str, float]:
    values = {
        "origin_x": 0.0,
        "origin_y": 0.0,
        "stride_x": 2.0,
        "stride_y": 2.0,
        "support_width": 2.0,
        "support_height": 2.0,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize("name", _SAMPLING_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_dense_diagnostic_rejects_non_finite_stride_and_support(name: str, value: float) -> None:
    """These values are contractual persistence: NaN/Inf must not reach ``metrics/``."""
    values = _dense_values(**{name: value})
    with pytest.raises(ValueError, match=f"{name} must be finite"):
        DenseFeatureDiagnostic(
            source_artifact_id="run-0001",
            grid_width=4,
            grid_height=4,
            coordinate_transform_id="sha256:transform",
            **values,
        )


@pytest.mark.parametrize("name", _SAMPLING_FIELDS)
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_dense_diagnostic_rejects_non_positive_stride_and_support(name: str, value: float) -> None:
    values = _dense_values(**{name: value})
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        DenseFeatureDiagnostic(
            source_artifact_id="run-0001",
            grid_width=4,
            grid_height=4,
            coordinate_transform_id="sha256:transform",
            **values,
        )


@pytest.mark.parametrize("name", ("origin_x", "origin_y", *_SAMPLING_FIELDS))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_dense_diagnostic_and_dense_sampling_accept_the_same_geometry(
    name: str, value: float
) -> None:
    """A persisted geometry must always rebuild a valid ``DenseFeatureSampling``."""
    values = _dense_values(**{name: value})

    def diagnostic_accepts() -> bool:
        try:
            DenseFeatureDiagnostic(
                source_artifact_id="run-0001",
                grid_width=4,
                grid_height=4,
                coordinate_transform_id="sha256:transform",
                **values,
            )
        except ValueError:
            return False
        return True

    def sampling_accepts() -> bool:
        try:
            DenseFeatureSampling(
                grid_width=4,
                grid_height=4,
                source_image_width=8,
                source_image_height=8,
                coordinate_transform_id="sha256:transform",
                **values,
            )
        except ValueError:
            return False
        return True

    assert diagnostic_accepts() == sampling_accepts()

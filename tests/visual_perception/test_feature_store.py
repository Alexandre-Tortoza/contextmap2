from pathlib import Path

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    BackendProvenance,
    FeatureId,
    FeaturePayloadIntegrityError,
    FeatureScope,
    FeatureStoreError,
    FeatureStoreReader,
    FeatureStoreWriter,
    RegionId,
    VisualFeature,
    write_feature_index,
)

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="feature_extractor", provider="fake", model="fake", version="0.1"
)


def _dense_feature(feature_id: str = "feature-dense-0000") -> VisualFeature:
    return VisualFeature(
        feature_id=FeatureId(feature_id),
        scope=FeatureScope.DENSE,
        embedding_space_id="fake-dense-space",
        shape=(4, 4, 8),
        dtype="float32",
        payload_reference=f"frame-0124/{feature_id}.npy",
        provenance=_PROVENANCE,
    )


def _region_feature(feature_id: str = "feature-region-0000") -> VisualFeature:
    return VisualFeature(
        feature_id=FeatureId(feature_id),
        scope=FeatureScope.REGION,
        embedding_space_id="fake-region-space",
        shape=(768,),
        dtype="float64",
        payload_reference=f"frame-0124/{feature_id}.npy",
        provenance=_PROVENANCE,
        region_id=RegionId("region-0001"),
    )


def test_written_payload_round_trips_exactly(tmp_path: Path) -> None:
    feature = _dense_feature()
    array = np.arange(4 * 4 * 8, dtype="float32").reshape(4, 4, 8)

    writer = FeatureStoreWriter(tmp_path)
    entry = writer.write(feature, SourceObservationId("frame-0124"), array)
    write_feature_index(tmp_path, writer.entries())

    reader = FeatureStoreReader.open(tmp_path)
    assert reader.feature_ids() == (feature.feature_id,)
    assert reader.entry(feature.feature_id) == entry

    loaded = reader.load(feature.feature_id)
    np.testing.assert_array_equal(loaded, array)
    assert loaded.dtype == array.dtype


def test_multiple_dtypes_and_shapes_round_trip(tmp_path: Path) -> None:
    dense = _dense_feature()
    region = _region_feature()
    dense_array = np.random.default_rng(0).random((4, 4, 8)).astype("float32")
    region_array = np.random.default_rng(1).random(768).astype("float64")

    writer = FeatureStoreWriter(tmp_path)
    writer.write(dense, SourceObservationId("frame-0124"), dense_array)
    writer.write(region, SourceObservationId("frame-0124"), region_array)
    write_feature_index(tmp_path, writer.entries())

    reader = FeatureStoreReader.open(tmp_path)
    np.testing.assert_array_equal(reader.load(dense.feature_id), dense_array)
    np.testing.assert_array_equal(reader.load(region.feature_id), region_array)


def test_opening_a_root_without_an_index_is_empty_not_an_error(tmp_path: Path) -> None:
    reader = FeatureStoreReader.open(tmp_path)
    assert reader.feature_ids() == ()
    with pytest.raises(FeatureStoreError, match="no persisted payload"):
        reader.entry(FeatureId("does-not-exist"))


def test_write_rejects_shape_mismatch(tmp_path: Path) -> None:
    feature = _dense_feature()
    wrong_shape_array = np.zeros((2, 2), dtype="float32")

    writer = FeatureStoreWriter(tmp_path)
    with pytest.raises(FeatureStoreError, match="shape"):
        writer.write(feature, SourceObservationId("frame-0124"), wrong_shape_array)


def test_write_rejects_dtype_mismatch(tmp_path: Path) -> None:
    feature = _dense_feature()
    wrong_dtype_array = np.zeros((4, 4, 8), dtype="float64")

    writer = FeatureStoreWriter(tmp_path)
    with pytest.raises(FeatureStoreError, match="dtype"):
        writer.write(feature, SourceObservationId("frame-0124"), wrong_dtype_array)


def test_write_rejects_payload_reference_escaping_the_store_root(tmp_path: Path) -> None:
    escaping_feature = VisualFeature(
        feature_id=FeatureId("feature-evil"),
        scope=FeatureScope.GLOBAL,
        embedding_space_id="fake-space",
        shape=(4,),
        dtype="float32",
        payload_reference="../outside.npy",
        provenance=_PROVENANCE,
    )
    array = np.zeros(4, dtype="float32")
    writer = FeatureStoreWriter(tmp_path)
    with pytest.raises(FeatureStoreError, match="boundary"):
        writer.write(escaping_feature, SourceObservationId("frame-0124"), array)


def test_load_detects_missing_payload_file(tmp_path: Path) -> None:
    feature = _dense_feature()
    writer = FeatureStoreWriter(tmp_path)
    writer.write(feature, SourceObservationId("frame-0124"), np.zeros((4, 4, 8), dtype="float32"))
    write_feature_index(tmp_path, writer.entries())

    (tmp_path / feature.payload_reference).unlink()

    reader = FeatureStoreReader.open(tmp_path)
    with pytest.raises(FeatureStoreError, match="missing payload"):
        reader.load(feature.feature_id)


def test_load_detects_content_hash_mismatch(tmp_path: Path) -> None:
    feature = _dense_feature()
    writer = FeatureStoreWriter(tmp_path)
    writer.write(feature, SourceObservationId("frame-0124"), np.zeros((4, 4, 8), dtype="float32"))
    write_feature_index(tmp_path, writer.entries())

    payload_path = tmp_path / feature.payload_reference
    tampered = np.ones((4, 4, 8), dtype="float32")
    np.save(payload_path, tampered)

    reader = FeatureStoreReader.open(tmp_path)
    with pytest.raises(FeaturePayloadIntegrityError, match="content hash mismatch"):
        reader.load(feature.feature_id)


def test_load_detects_corrupt_payload(tmp_path: Path) -> None:
    feature = _dense_feature()
    writer = FeatureStoreWriter(tmp_path)
    entry = writer.write(
        feature, SourceObservationId("frame-0124"), np.zeros((4, 4, 8), dtype="float32")
    )

    payload_path = tmp_path / feature.payload_reference
    corrupted = b"not a valid npy file"
    payload_path.write_bytes(corrupted)
    import hashlib

    from contextmap.visual_perception.feature_store import FeaturePayloadEntry

    corrupted_entry = FeaturePayloadEntry(
        feature_id=entry.feature_id,
        source_observation_id=entry.source_observation_id,
        scope=entry.scope,
        embedding_space_id=entry.embedding_space_id,
        shape=entry.shape,
        dtype=entry.dtype,
        normalization=entry.normalization,
        payload_reference=entry.payload_reference,
        content_hash=f"sha256:{hashlib.sha256(corrupted).hexdigest()}",
        size_bytes=len(corrupted),
        provenance=entry.provenance,
    )

    reader = FeatureStoreReader(tmp_path, (corrupted_entry,))
    with pytest.raises(FeaturePayloadIntegrityError, match="unsupported or corrupt"):
        reader.load(feature.feature_id)

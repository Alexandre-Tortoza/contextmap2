from pathlib import Path

import numpy as np
import pytest
from mask_cases import GOLDEN, MASK_STORE_SEEDS, mask_store_case

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import RegionId
from contextmap.visual_perception.mask_store import (
    MASK_INDEX_SCHEMA_VERSION,
    MaskPayloadIntegrityError,
    MaskStoreError,
    MaskStoreReader,
    MaskStoreWriter,
    write_mask_index,
)
from contextmap.visual_perception.region_models import InlineMask


def _checkerboard_mask(width: int = 8, height: int = 6) -> InlineMask:
    data = tuple((x + y) % 2 == 0 for y in range(height) for x in range(width))
    return InlineMask(np.array(data, dtype=bool).reshape(height, width))


def test_written_payload_round_trips_bit_exactly(tmp_path: Path) -> None:
    mask = _checkerboard_mask()

    writer = MaskStoreWriter(tmp_path)
    entry = writer.write(
        region_id=RegionId("region-0000"),
        source_observation_id=SourceObservationId("frame-0001"),
        mask=mask,
    )
    write_mask_index(tmp_path, writer.entries())

    reader = MaskStoreReader.open(tmp_path)
    loaded = reader.load(SourceObservationId("frame-0001"), RegionId("region-0000"))

    assert loaded == mask
    assert entry.width == mask.width
    assert entry.height == mask.height
    assert entry.content_hash.startswith("sha256:")


def test_full_image_mask_payload_is_bit_packed_and_smaller_than_one_byte_per_pixel(
    tmp_path: Path,
) -> None:
    width, height = 640, 480
    data = tuple((x // 40 + y // 40) % 2 == 0 for y in range(height) for x in range(width))
    mask = InlineMask(np.array(data, dtype=bool).reshape(height, width))

    writer = MaskStoreWriter(tmp_path)
    entry = writer.write(
        region_id=RegionId("region-0000"),
        source_observation_id=SourceObservationId("frame-0001"),
        mask=mask,
    )

    # 8 pixels per byte, well under the ~0.92 MB the JSON-inline representation cost (#378).
    assert entry.size_bytes < (width * height) // 8 + 256


def test_index_round_trips_through_disk_with_schema_header(tmp_path: Path) -> None:
    mask = _checkerboard_mask()
    writer = MaskStoreWriter(tmp_path)
    writer.write(
        region_id=RegionId("region-0000"),
        source_observation_id=SourceObservationId("frame-0001"),
        mask=mask,
    )
    write_mask_index(tmp_path, writer.entries())

    header = (tmp_path / "mask-index.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert MASK_INDEX_SCHEMA_VERSION in header

    reopened = MaskStoreReader.open(tmp_path)
    assert reopened.region_keys() == ((SourceObservationId("frame-0001"), RegionId("region-0000")),)


def test_opening_a_root_without_an_index_is_an_empty_reader(tmp_path: Path) -> None:
    reader = MaskStoreReader.open(tmp_path)

    assert reader.region_keys() == ()
    with pytest.raises(MaskStoreError, match="no persisted mask"):
        reader.load(SourceObservationId("frame-0001"), RegionId("region-0000"))


def test_duplicate_region_key_is_rejected(tmp_path: Path) -> None:
    mask = _checkerboard_mask()
    writer = MaskStoreWriter(tmp_path)
    writer.write(
        region_id=RegionId("region-0000"),
        source_observation_id=SourceObservationId("frame-0001"),
        mask=mask,
    )

    with pytest.raises(MaskStoreError, match="duplicate mask payload key"):
        writer.write(
            region_id=RegionId("region-0000"),
            source_observation_id=SourceObservationId("frame-0001"),
            mask=mask,
        )


def test_region_id_is_local_to_source_observation(tmp_path: Path) -> None:
    """The same region_id string in two different frames must not collide (#378)."""
    mask_a = _checkerboard_mask(width=4, height=4)
    mask_b = InlineMask(~mask_a.as_array())
    writer = MaskStoreWriter(tmp_path)
    writer.write(
        region_id=RegionId("region-0000"),
        source_observation_id=SourceObservationId("frame-0001"),
        mask=mask_a,
    )
    writer.write(
        region_id=RegionId("region-0000"),
        source_observation_id=SourceObservationId("frame-0002"),
        mask=mask_b,
    )
    write_mask_index(tmp_path, writer.entries())

    reader = MaskStoreReader.open(tmp_path)
    assert reader.load(SourceObservationId("frame-0001"), RegionId("region-0000")) == mask_a
    assert reader.load(SourceObservationId("frame-0002"), RegionId("region-0000")) == mask_b


def test_corrupted_payload_fails_hash_verification(tmp_path: Path) -> None:
    mask = _checkerboard_mask()
    writer = MaskStoreWriter(tmp_path)
    entry = writer.write(
        region_id=RegionId("region-0000"),
        source_observation_id=SourceObservationId("frame-0001"),
        mask=mask,
    )
    write_mask_index(tmp_path, writer.entries())

    (tmp_path / entry.payload_reference).write_bytes(b"corrupted")

    reader = MaskStoreReader.open(tmp_path)
    with pytest.raises(MaskPayloadIntegrityError, match="content hash mismatch"):
        reader.load(SourceObservationId("frame-0001"), RegionId("region-0000"))


def test_payload_reference_cannot_escape_the_store_root(tmp_path: Path) -> None:
    writer = MaskStoreWriter(tmp_path)
    with pytest.raises(MaskStoreError, match="payload_reference resolves outside"):
        writer.write(
            region_id=RegionId("../../escape"),
            source_observation_id=SourceObservationId("frame-0001"),
            mask=_checkerboard_mask(),
        )


@pytest.mark.parametrize("seed", MASK_STORE_SEEDS)
def test_persisted_mask_bytes_match_the_recorded_payload(tmp_path: Path, seed: int) -> None:
    # #593: o payload empacotado (e seu hash no índice) não muda com a representação em memória.
    mask = mask_store_case(seed)
    writer = MaskStoreWriter(tmp_path)

    entry = writer.write(
        region_id=RegionId("region-0001"),
        source_observation_id=SourceObservationId("frame-1"),
        mask=mask,
    )
    write_mask_index(tmp_path, writer.entries())

    assert entry.content_hash == GOLDEN["mask_store"][str(seed)]
    assert (
        MaskStoreReader.open(tmp_path).load(SourceObservationId("frame-1"), RegionId("region-0001"))
        == mask
    )

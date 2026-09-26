from pathlib import Path

import pytest

from contextmap.ingestion import (
    SequenceProvenance,
    compute_configuration_hash,
    compute_content_identity,
    compute_source_content_hash,
)
from contextmap.ingestion.sequence_provenance import decode_provenance, encode_provenance


def _provenance(**overrides: object) -> SequenceProvenance:
    defaults: dict[str, object] = {
        "source_type": "ros1_bag",
        "source_path": "data/example.bag",
        "source_content_hash": "sha256:aaaa",
        "configuration_hash": "sha256:bbbb",
    }
    defaults.update(overrides)
    return SequenceProvenance(**defaults)  # type: ignore[arg-type]


def test_same_source_and_config_have_the_same_identity() -> None:
    first = _provenance()
    second = _provenance()

    assert compute_content_identity(first) == compute_content_identity(second)


def test_changed_configuration_changes_identity() -> None:
    baseline = _provenance()
    changed_config = _provenance(configuration_hash="sha256:cccc")

    assert compute_content_identity(baseline) != compute_content_identity(changed_config)


def test_changed_source_content_changes_identity() -> None:
    baseline = _provenance()
    changed_source = _provenance(source_content_hash="sha256:dddd")

    assert compute_content_identity(baseline) != compute_content_identity(changed_source)


def test_same_content_under_a_different_path_has_the_same_identity() -> None:
    """Path equality is never assumed to imply content equality, or the reverse."""
    original = _provenance(source_path="data/a.bag")
    moved = _provenance(source_path="data/renamed-copy.bag")

    assert compute_content_identity(original) == compute_content_identity(moved)


def test_source_content_hash_detects_modification_under_the_same_path(tmp_path: Path) -> None:
    source = tmp_path / "example.bag"
    source.write_bytes(b"original content")
    before = compute_source_content_hash(source)

    source.write_bytes(b"modified content")
    after = compute_source_content_hash(source)

    assert before != after


def test_source_content_hash_is_stable_for_unchanged_file(tmp_path: Path) -> None:
    source = tmp_path / "example.bag"
    source.write_bytes(b"stable content")

    assert compute_source_content_hash(source) == compute_source_content_hash(source)


def test_source_content_hash_covers_directory_sources_like_ros2_bags(tmp_path: Path) -> None:
    bag_dir = tmp_path / "ros2_bag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text("version: 1")
    (bag_dir / "data.db3").write_bytes(b"\x00\x01\x02")

    before = compute_source_content_hash(bag_dir)
    (bag_dir / "data.db3").write_bytes(b"\x00\x01\x03")
    after = compute_source_content_hash(bag_dir)

    assert before != after


def test_compute_configuration_hash_is_none_for_empty_config() -> None:
    assert compute_configuration_hash({}) is None


def test_compute_configuration_hash_changes_with_content() -> None:
    a = compute_configuration_hash({"tolerance_seconds": 0.05})
    b = compute_configuration_hash({"tolerance_seconds": 0.1})

    assert a != b


def test_compute_configuration_hash_rejects_a_non_primitive_value_naming_its_key() -> None:
    # Um objeto sem forma JSON viraria str(obj), com endereço de memória: hash não determinístico.
    config = {"window": {"clock_id": "recording_time", "bounds": object()}}

    with pytest.raises(TypeError, match=r"\['window'\]\['bounds'\].*object"):
        compute_configuration_hash(config)


def test_compute_configuration_hash_rejects_a_non_primitive_list_item() -> None:
    config = {"required_topics": ["rgb", Path("lidar")]}

    with pytest.raises(TypeError, match=r"\['required_topics'\]\[1\].*Path"):
        compute_configuration_hash(config)


def test_provenance_round_trips_through_encode_decode() -> None:
    provenance = _provenance(
        adapter_type="ros1_bag",
        code_version="0.1.0",
        calibration_source_hash="sha256:eeee",
        synchronization_policy="nearest_within_tolerance",
        warnings=("missing imu topic",),
        ingestion_config={"tolerance_seconds": 0.05},
    )

    decoded = decode_provenance(encode_provenance(provenance))

    assert decoded == provenance


def test_encoded_provenance_includes_content_identity() -> None:
    provenance = _provenance()

    encoded = encode_provenance(provenance)

    assert encoded["content_identity"] == compute_content_identity(provenance)

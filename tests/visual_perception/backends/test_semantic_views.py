"""Unit tests of the shared semantic view payload reader that every semantic backend uses."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import RegionId, SemanticVisualView, VisualViewKind
from contextmap.visual_perception.backends._semantic_views import read_view_payload

REFERENCE = "outputs/semantic-views/region-0007.png"
PAYLOAD = b"canonical view bytes"


def _view(payload: bytes = PAYLOAD, reference: str = REFERENCE) -> SemanticVisualView:
    return SemanticVisualView(
        view_id="tight-crop",
        kind=VisualViewKind.TIGHT_CROP,
        payload_reference=reference,
        source_observation_id=SourceObservationId("frame-0124"),
        region_id=RegionId("region-0007"),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _write(root: Path, payload: bytes = PAYLOAD, reference: str = REFERENCE) -> Path:
    path = root / reference
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_the_exact_payload_bytes_are_returned_when_they_match_the_recorded_sha256(
    tmp_path: Path,
) -> None:
    _write(tmp_path)

    assert read_view_payload(tmp_path, _view()) == PAYLOAD


def test_a_payload_that_changed_after_the_view_was_built_is_rejected_with_both_hashes(
    tmp_path: Path,
) -> None:
    _write(tmp_path, b"bytes written after the request was built")

    with pytest.raises(ValueError, match="sha256") as caught:
        read_view_payload(tmp_path, _view())

    message = str(caught.value)
    assert hashlib.sha256(PAYLOAD).hexdigest() in message
    assert hashlib.sha256(b"bytes written after the request was built").hexdigest() in message
    assert REFERENCE in message


def test_a_view_with_the_hash_of_another_payload_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path)

    with pytest.raises(ValueError, match="sha256"):
        read_view_payload(tmp_path, _view(b"another payload"))


def test_a_missing_payload_is_a_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_view_payload(tmp_path, _view())


def test_a_link_that_leaves_the_view_root_is_rejected_before_reading(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "outputs" / "semantic-views").mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(PAYLOAD)
    (root / REFERENCE).symlink_to(outside)

    with pytest.raises(ValueError, match="escapes"):
        read_view_payload(root, _view())

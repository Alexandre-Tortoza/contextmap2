"""Tests for the Spatial Foundation: one pinned sequence, trajectory and map (issue #494)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from runtime_foundation import build_map, estimate, foundation_refs, write_sequence

from contextmap.runtime import (
    ArtifactRef,
    SpatialFoundation,
    SpatialFoundationError,
    resolve_spatial_foundation,
)


def _resolve(workspace: Path, refs: dict[str, ArtifactRef]) -> SpatialFoundation:
    return resolve_spatial_foundation(workspace, **refs)


class TestAValidFoundation:
    def test_matching_artifacts_form_one_foundation(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)

        foundation = _resolve(tmp_path, refs)

        assert foundation.identity.startswith("sha256:")
        assert foundation.sequence == refs["sequence"]
        assert foundation.state_estimation == refs["state_estimation"]
        assert foundation.geometry == refs["geometry"]

    def test_equivalent_inputs_give_the_same_identity(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)

        assert _resolve(tmp_path, refs).identity == _resolve(tmp_path, refs).identity

    def test_moving_the_workspace_does_not_change_the_identity(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path / "a")
        shutil.copytree(tmp_path / "a", tmp_path / "b")

        assert _resolve(tmp_path / "a", refs).identity == _resolve(tmp_path / "b", refs).identity

    def test_the_identity_follows_content_not_locations(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)
        moved = tmp_path / "S1" / "elsewhere"
        shutil.copytree(tmp_path / "S1" / "run-0001" / "geometric_mapping", moved)
        relocated = {
            **refs,
            "geometry": dataclasses.replace(refs["geometry"], location="S1/elsewhere"),
        }

        assert _resolve(tmp_path, relocated).identity == _resolve(tmp_path, refs).identity

    def test_another_map_is_another_foundation(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)
        other = build_map(
            tmp_path, refs["sequence"], refs["state_estimation"], run="run-0002", digest="gm-2"
        )

        changed = _resolve(tmp_path, {**refs, "geometry": other})

        assert changed.identity != _resolve(tmp_path, refs).identity

    def test_the_foundation_only_references_its_artifacts(self) -> None:
        names = {field.name for field in dataclasses.fields(SpatialFoundation)}

        assert names == {"identity", "sequence", "state_estimation", "geometry"}


class TestAMismatchedFoundation:
    def test_a_map_of_another_sequence_is_rejected(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)
        other = foundation_refs(tmp_path, run="run-0002", artifact_id="seq-B")

        with pytest.raises(SpatialFoundationError, match="seq-B") as caught:
            _resolve(tmp_path, {**refs, "geometry": other["geometry"]})

        assert any("sequence" in problem for problem in caught.value.problems)

    def test_a_map_of_another_state_estimation_run_is_rejected(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)
        other = estimate(tmp_path, refs["sequence"], run="run-0002", digest="se-2")

        with pytest.raises(SpatialFoundationError, match="state estimation run"):
            _resolve(tmp_path, {**refs, "state_estimation": other})

    def test_a_reference_that_names_another_artifact_is_rejected(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)
        forged = dataclasses.replace(refs["state_estimation"], artifact_id="not-this-run")

        with pytest.raises(SpatialFoundationError, match="not-this-run"):
            _resolve(tmp_path, {**refs, "state_estimation": forged})

    def test_a_reference_of_another_kind_is_rejected(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)

        with pytest.raises(SpatialFoundationError, match="GeometricMapArtifact"):
            _resolve(tmp_path, {**refs, "geometry": refs["state_estimation"]})

    def test_changed_content_is_rejected(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)
        directory = tmp_path / "S1" / "run-0001" / "state_estimation"
        payload = next(path for path in sorted(directory.rglob("*.jsonl")))
        payload.write_text(payload.read_text("utf-8") + "\n", encoding="utf-8")

        with pytest.raises(SpatialFoundationError, match="state_estimation"):
            _resolve(tmp_path, refs)

    def test_a_map_over_part_of_the_sequence_is_not_a_foundation(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)
        manifest = tmp_path / "S1" / "run-0001" / "geometric_mapping" / "manifest.json"
        document = json.loads(manifest.read_text("utf-8"))
        document["selection_id"] = "sha256:" + "0" * 64
        manifest.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(SpatialFoundationError, match="whole sequence"):
            _resolve(tmp_path, refs)

    def test_every_problem_is_reported_at_once(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)
        other = foundation_refs(tmp_path, run="run-0002", artifact_id="seq-B")
        forged = dataclasses.replace(refs["state_estimation"], artifact_id="not-this-run")

        with pytest.raises(SpatialFoundationError) as caught:
            _resolve(tmp_path, {**refs, "state_estimation": forged, "geometry": other["geometry"]})

        roles = sorted(problem.split(":")[0] for problem in caught.value.problems)
        assert roles == ["geometry", "state_estimation"]


def test_the_sequence_writer_helper_is_deterministic(tmp_path: Path) -> None:
    first = write_sequence(tmp_path, run="run-0001")
    second = write_sequence(tmp_path, run="run-0002")

    assert first.content_hash == second.content_hash

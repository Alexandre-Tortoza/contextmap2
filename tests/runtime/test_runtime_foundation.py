"""Tests for the Spatial Foundation: one pinned sequence, trajectory and map (issue #494)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from runtime_documents import effective_from, selected_document
from runtime_foundation import (
    SyntheticIngestion,
    build_map,
    estimate,
    foundation_refs,
    geometric_mapping_executor,
    state_estimation_executor,
    write_sequence,
)

from contextmap.runtime import (
    ArtifactRef,
    RunJournal,
    SpatialFoundation,
    SpatialFoundationError,
    foundation_of_run,
    read_plan_document,
    resolve_plan,
    resolve_spatial_foundation,
    run_plan,
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

    def test_it_round_trips_through_its_document(self, tmp_path: Path) -> None:
        foundation = _resolve(tmp_path, foundation_refs(tmp_path))

        assert SpatialFoundation.from_document(foundation.to_document()) == foundation

    def test_a_document_without_one_of_its_references_is_refused(self, tmp_path: Path) -> None:
        document = _resolve(tmp_path, foundation_refs(tmp_path)).to_document()
        del document["geometry"]

        with pytest.raises(ValueError, match="geometry"):
            SpatialFoundation.from_document(document)

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


class TestTheFoundationOfARun:
    """Issue #502: a completed run that produced or used them names the three artifacts."""

    @staticmethod
    def _run(tmp_path: Path, target: str) -> Path:
        effective = effective_from(tmp_path, selected_document())
        execution = resolve_plan(effective).scope(targets=[target])
        journal = RunJournal.create(tmp_path / "ws", effective, execution)
        run_plan(
            execution,
            {
                "ingestion": SyntheticIngestion(),
                "state_estimation": state_estimation_executor(),
                "geometric_mapping": geometric_mapping_executor(),
            },
            environ={},
            module_available=lambda _name: True,
            journal=journal,
        )
        return journal.directory

    def test_the_run_that_built_the_map_is_a_foundation(self, tmp_path: Path) -> None:
        run = self._run(tmp_path, "geometric_mapping")

        foundation = foundation_of_run(tmp_path / "ws", run)

        record = read_plan_document(run / "execution.json")
        outputs = {stage["stage_id"]: stage["output"] for stage in record["stages"]}
        assert foundation.geometry == ArtifactRef.from_document(outputs["geometric_mapping"])
        assert foundation == resolve_spatial_foundation(
            tmp_path / "ws",
            sequence=foundation.sequence,
            state_estimation=foundation.state_estimation,
            geometry=foundation.geometry,
        )

    def test_a_run_without_a_map_is_not_a_foundation(self, tmp_path: Path) -> None:
        run = self._run(tmp_path, "state_estimation")

        with pytest.raises(SpatialFoundationError, match="geometric_mapping"):
            foundation_of_run(tmp_path / "ws", run)


def test_the_sequence_writer_helper_is_deterministic(tmp_path: Path) -> None:
    first = write_sequence(tmp_path, run="run-0001")
    second = write_sequence(tmp_path, run="run-0002")

    assert first.content_hash == second.content_hash

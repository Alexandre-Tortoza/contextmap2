"""ContextBranch: explicit accumulation of ContextRuns over one foundation (issue #496)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextmap.runtime import (
    ArtifactRef,
    ContextBranch,
    ContextBranchError,
    ContextRun,
    ContextRunId,
    SpatialFoundation,
    SpatialFoundationId,
    append_to_branch,
    create_branch,
    open_branch,
)


def _foundation(name: str = "A") -> SpatialFoundation:
    def ref(stage_id: str, contract: str) -> ArtifactRef:
        return ArtifactRef(
            stage_id=stage_id,
            contract=contract,
            artifact_id=f"{stage_id}-{name}",
            content_hash=f"sha256:{stage_id}-{name}",
            location=f"S1/run-0001/{stage_id}",
        )

    return SpatialFoundation(
        identity=SpatialFoundationId(f"sha256:foundation-{name}"),
        sequence=ref("ingestion", "SequenceArtifact"),
        state_estimation=ref("state_estimation", "StateEstimationRunArtifact"),
        geometry=ref("geometric_mapping", "GeometricMapArtifact"),
    )


def _context(number: int, foundation: SpatialFoundation | None = None) -> ContextRun:
    return ContextRun(
        identity=ContextRunId(f"sha256:context-{number}"),
        foundation=foundation or _foundation(),
        observation_selection={
            "kind": "frame_range",
            "start_frame_index": 10 * number,
            "end_frame_index": 10 * number + 10,
        },
        artifacts=(),
        run=f"S1/run-{number:04d}",
        plan_digest="sha256:plan",
    )


def _branch(tmp_path: Path, name: str = "corridor") -> ContextBranch:
    return create_branch(tmp_path, dataset="S1", name=name, foundation=_foundation())


class TestCreation:
    def test_a_new_branch_is_empty_and_bound_to_its_foundation(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)

        assert branch.name == "corridor"
        assert branch.foundation == _foundation()
        assert branch.revision == 0
        assert branch.members == ()
        assert (tmp_path / "S1" / "branches" / "corridor" / "branch.json").is_file()

    def test_it_reopens_from_its_records(self, tmp_path: Path) -> None:
        created = _branch(tmp_path)

        assert open_branch(tmp_path, dataset="S1", name="corridor") == created

    def test_a_name_is_created_once(self, tmp_path: Path) -> None:
        _branch(tmp_path)

        with pytest.raises(ContextBranchError, match="already exists"):
            _branch(tmp_path)

    @pytest.mark.parametrize("name", ["", "Upper", "../escape", "a/b", "-dash"])
    def test_a_name_is_a_lowercase_slug(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ContextBranchError, match="slug"):
            _branch(tmp_path, name=name)

    def test_an_unknown_branch_cannot_be_opened(self, tmp_path: Path) -> None:
        with pytest.raises(ContextBranchError, match="no branch"):
            open_branch(tmp_path, dataset="S1", name="missing")


class TestAccumulation:
    def test_appending_a_run_is_a_new_revision(self, tmp_path: Path) -> None:
        branch = append_to_branch(tmp_path, _branch(tmp_path), _context(1))

        assert branch.revision == 1
        (member,) = branch.members
        assert (member.revision, member.context_run_id, member.run) == (
            1,
            "sha256:context-1",
            "S1/run-0001",
        )
        assert open_branch(tmp_path, dataset="S1", name="corridor") == branch

    def test_the_same_runs_in_any_order_are_the_same_run_set(self, tmp_path: Path) -> None:
        forward = _branch(tmp_path, "forward")
        backward = _branch(tmp_path, "backward")
        for number in (1, 2, 3):
            forward = append_to_branch(tmp_path, forward, _context(number))
        for number in (3, 2, 1):
            backward = append_to_branch(tmp_path, backward, _context(number))

        assert forward.context_run_ids == backward.context_run_ids
        assert [member.context_run_id for member in forward.members] != [
            member.context_run_id for member in backward.members
        ]

    def test_a_run_is_a_member_once(self, tmp_path: Path) -> None:
        branch = append_to_branch(tmp_path, _branch(tmp_path), _context(1))

        with pytest.raises(ContextBranchError, match="already a member"):
            append_to_branch(tmp_path, branch, _context(1))

    def test_a_run_over_another_foundation_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ContextBranchError, match="foundation"):
            append_to_branch(tmp_path, _branch(tmp_path), _context(1, _foundation("B")))

    def test_growth_never_changes_earlier_records_or_revisions(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)
        for number in (1, 2, 3):
            branch = append_to_branch(tmp_path, branch, _context(number))
        root = tmp_path / "S1" / "branches" / "corridor"
        before = {path.name: path.read_bytes() for path in root.rglob("*.json")}
        at_three = branch.at(3)

        grown = append_to_branch(tmp_path, branch, _context(4))

        after = {path.name: path.read_bytes() for path in root.rglob("*.json")}
        assert {name: after[name] for name in before} == before
        assert grown.at(3) == at_three
        assert grown.revision == 4

    def test_an_earlier_revision_names_only_its_members(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)
        for number in (1, 2, 3):
            branch = append_to_branch(tmp_path, branch, _context(number))

        assert branch.at(2).context_run_ids == ("sha256:context-1", "sha256:context-2")
        with pytest.raises(ContextBranchError, match="revision 4"):
            branch.at(4)

    def test_a_concurrent_append_at_the_same_revision_is_refused(self, tmp_path: Path) -> None:
        stale = _branch(tmp_path)
        append_to_branch(tmp_path, stale, _context(1))

        with pytest.raises(ContextBranchError, match="revision 1"):
            append_to_branch(tmp_path, stale, _context(2))


class TestIntegrity:
    def test_a_member_record_under_another_revision_is_refused(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)
        for number in (1, 2):
            branch = append_to_branch(tmp_path, branch, _context(number))
        members = tmp_path / "S1" / "branches" / "corridor" / "members"
        document = json.loads((members / "000002.json").read_text("utf-8"))
        document["revision"] = 7
        (members / "000002.json").write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(ContextBranchError, match="000002"):
            open_branch(tmp_path, dataset="S1", name="corridor")

    def test_a_missing_revision_is_refused(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)
        for number in (1, 2):
            branch = append_to_branch(tmp_path, branch, _context(number))
        (tmp_path / "S1" / "branches" / "corridor" / "members" / "000001.json").unlink()

        with pytest.raises(ContextBranchError, match="revision 1"):
            open_branch(tmp_path, dataset="S1", name="corridor")

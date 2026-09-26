"""ContextBranch: the explicit accumulation of ContextRuns over one spatial foundation.

A long-lived contextual map grows by appending ContextRuns to a branch. The branch is an
evidence-selection boundary, not belief: it holds no fused state, no confidence and no winner,
only which immutable ContextRuns were appended, in which order, over which foundation.

Its records are the authority and are never rewritten. ``branch.json`` binds the branch name to
its foundation once; each append publishes one new member record, atomically and without
overwriting, so two appends racing for the same revision cannot both succeed. There is no
mutable index to rebuild: opening a branch reads its member records and checks that their
revisions are exactly ``1..N``. See ``docs/runtime-composition.md``, "Contexto incremental".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextmap.runtime._files import publish_text
from contextmap.runtime.context_run import ContextRun, ContextRunId
from contextmap.runtime.foundation import SpatialFoundation

CONTEXT_BRANCH_SCHEMA_VERSION = "0.1.0"
"""Version of ``branch.json`` and of the member records."""

BRANCHES_DIRECTORY = "branches"
"""Where the branches of a dataset live: ``<workspace>/<dataset>/branches/<name>``."""

_BRANCH_FILENAME = "branch.json"
_MEMBERS_DIRECTORY = "members"
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")


class ContextBranchError(ValueError):
    """A branch cannot be created, opened or grown as asked."""


@dataclass(frozen=True, kw_only=True)
class ContextBranchMember:
    """One ContextRun appended to a branch.

    Attributes:
        revision: The branch revision this append created, from 1.
        context_run_id: The appended ContextRun.
        run: Where its record is, relative to the workspace (``<dataset>/run-NNNN``).
    """

    revision: int
    context_run_id: ContextRunId
    run: str


@dataclass(frozen=True, kw_only=True)
class ContextBranch:
    """A named accumulation of ContextRuns over one spatial foundation.

    Attributes:
        name: The branch, unique within its dataset.
        dataset: The dataset directory the branch lives in.
        foundation: The spatial foundation every member was built on.
        members: The appended ContextRuns, in revision order.
    """

    name: str
    dataset: str
    foundation: SpatialFoundation
    members: tuple[ContextBranchMember, ...]

    @property
    def revision(self) -> int:
        """Return the current revision: the number of members."""
        return len(self.members)

    @property
    def context_run_ids(self) -> tuple[ContextRunId, ...]:
        """Return the member set in canonical order, independent of the order of appends."""
        return tuple(sorted(member.context_run_id for member in self.members))

    def at(self, revision: int) -> ContextBranch:
        """Return the branch as it was at ``revision``: a later append never changes it.

        Raises:
            ContextBranchError: If the branch never reached ``revision``.
        """
        if not 0 <= revision <= self.revision:
            raise ContextBranchError(
                f"branch {self.name!r} has no revision {revision}; it is at revision "
                f"{self.revision}"
            )
        return ContextBranch(
            name=self.name,
            dataset=self.dataset,
            foundation=self.foundation,
            members=self.members[:revision],
        )


def create_branch(
    workspace: Path, *, dataset: str, name: str, foundation: SpatialFoundation
) -> ContextBranch:
    """Create an empty branch bound to one spatial foundation.

    Args:
        workspace: The workspace root.
        dataset: The dataset directory, as in ``<workspace>/<dataset>/run-NNNN``.
        name: A lowercase slug, unique within the dataset.
        foundation: The foundation every member must be built on.

    Returns:
        The branch at revision 0.

    Raises:
        ContextBranchError: If the name is not a slug or the branch already exists.
    """
    directory = _directory(workspace, dataset, name)
    document = {
        "schema_version": CONTEXT_BRANCH_SCHEMA_VERSION,
        "name": name,
        "foundation": foundation.to_document(),
    }
    try:
        publish_text(directory, _BRANCH_FILENAME, _text(document))
    except FileExistsError as error:
        raise ContextBranchError(f"branch {name!r} already exists in {dataset!r}") from error
    return ContextBranch(name=name, dataset=dataset, foundation=foundation, members=())


def open_branch(workspace: Path, *, dataset: str, name: str) -> ContextBranch:
    """Reconstruct a branch from its records.

    Args:
        workspace: The workspace root.
        dataset: The dataset directory.
        name: The branch.

    Returns:
        The branch at its current revision.

    Raises:
        ContextBranchError: If the branch does not exist, or its records are unreadable,
            inconsistent or not exactly revisions ``1..N``.
    """
    directory = _directory(workspace, dataset, name)
    path = directory / _BRANCH_FILENAME
    if not path.is_file():
        raise ContextBranchError(f"there is no branch {name!r} in {dataset!r}")
    document = _read(path)
    if document.get("name") != name:
        raise ContextBranchError(f"{path} names branch {document.get('name')!r}, not {name!r}")
    try:
        foundation = SpatialFoundation.from_document(document["foundation"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContextBranchError(f"{path} has no valid foundation: {error}") from error
    members = []
    records = sorted((directory / _MEMBERS_DIRECTORY).glob("[0-9]*.json"))
    for expected, record in enumerate(records, start=1):
        if record.stem != _member_stem(expected):
            raise ContextBranchError(
                f"branch {name!r} lacks the record of revision {expected}; found {record.name}"
            )
        members.append(_member(record, expected))
    return ContextBranch(name=name, dataset=dataset, foundation=foundation, members=tuple(members))


def append_to_branch(
    workspace: Path, branch: ContextBranch, context_run: ContextRun
) -> ContextBranch:
    """Append a ContextRun as the next revision of a branch.

    Nothing already written changes: the append is one new member record. It is published
    without overwriting, so a stale ``branch`` whose next revision was taken meanwhile fails
    instead of hiding the other append.

    Args:
        workspace: The workspace root.
        branch: The branch, at the revision this append builds on.
        context_run: The run to append.

    Returns:
        The branch at its new revision.

    Raises:
        ContextBranchError: If the run was built on another foundation, is already a member,
            or the next revision was appended concurrently.
    """
    if context_run.foundation.identity != branch.foundation.identity:
        raise ContextBranchError(
            f"context run {context_run.identity!r} was built on foundation "
            f"{context_run.foundation.identity!r}, but branch {branch.name!r} accumulates "
            f"evidence over {branch.foundation.identity!r}"
        )
    if context_run.identity in {member.context_run_id for member in branch.members}:
        raise ContextBranchError(
            f"context run {context_run.identity!r} is already a member of {branch.name!r}"
        )
    member = ContextBranchMember(
        revision=branch.revision + 1, context_run_id=context_run.identity, run=context_run.run
    )
    document = {
        "schema_version": CONTEXT_BRANCH_SCHEMA_VERSION,
        "branch": branch.name,
        "revision": member.revision,
        "context_run_id": member.context_run_id,
        "run": member.run,
    }
    members = _directory(workspace, branch.dataset, branch.name) / _MEMBERS_DIRECTORY
    try:
        publish_text(members, f"{_member_stem(member.revision)}.json", _text(document))
    except FileExistsError as error:
        raise ContextBranchError(
            f"revision {member.revision} of branch {branch.name!r} was appended meanwhile; "
            "reopen the branch and append again"
        ) from error
    return ContextBranch(
        name=branch.name,
        dataset=branch.dataset,
        foundation=branch.foundation,
        members=(*branch.members, member),
    )


def _directory(workspace: Path, dataset: str, name: str) -> Path:
    if not _NAME.fullmatch(name):
        raise ContextBranchError(
            f"branch name {name!r} must be a lowercase slug ([a-z0-9][a-z0-9-]*)"
        )
    return workspace / dataset / BRANCHES_DIRECTORY / name


def _member_stem(revision: int) -> str:
    return f"{revision:06d}"


def _member(path: Path, revision: int) -> ContextBranchMember:
    """Read one member record and check it is the record of ``revision``."""
    document = _read(path)
    if document.get("revision") != revision:
        raise ContextBranchError(
            f"{path.name} records revision {document.get('revision')!r}, not {revision}"
        )
    context_run_id, run = document.get("context_run_id"), document.get("run")
    if not isinstance(context_run_id, str) or not isinstance(run, str):
        raise ContextBranchError(f"{path.name} lacks its context run or its run location")
    return ContextBranchMember(
        revision=revision, context_run_id=ContextRunId(context_run_id), run=run
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ContextBranchError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise ContextBranchError(f"{path} is not a branch record")
    if document.get("schema_version") != CONTEXT_BRANCH_SCHEMA_VERSION:
        raise ContextBranchError(
            f"{path} follows schema_version {document.get('schema_version')!r}, this runtime "
            f"reads {CONTEXT_BRANCH_SCHEMA_VERSION!r}"
        )
    return document


def _text(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"

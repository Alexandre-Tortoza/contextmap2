"""ContextRun: one execution that adds context evidence over a spatial foundation.

Incremental context (``docs/runtime-composition.md``, "Contexto incremental (v0.1.1)") grows a
contextual map by running the context stages (Visual Perception and Sensor Association, plus
Point Representation when enabled) over explicit selections of one sequence, always against the
same :class:`~contextmap.runtime.foundation.SpatialFoundation`. A ContextRun is the record of one
such execution: which foundation and which observation selection it used, and which capability
artifacts it produced or reused.

It is orchestration provenance, not evidence: it references the capability artifacts and copies
nothing from them. It is published once, next to the run's journal, only when the run completed;
a failed run keeps its journal as its only record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NewType

from contextmap.runtime._files import publish_text
from contextmap.runtime.artifacts import ArtifactRef
from contextmap.runtime.foundation import SpatialFoundation
from contextmap.runtime.pipeline import ExecutionPlan, ExecutionRecord, PipelinePlan

CONTEXT_RUN_FILENAME = "context_run.json"
CONTEXT_RUN_SCHEMA_VERSION = "0.1.0"
"""Version of the ``context_run.json`` document."""

CONTEXT_STAGES = ("visual_perception", "sensor_association", "point_representation")
"""The stages a ContextRun may execute; every other stage is foundation or materialization."""

_FOUNDATION_ROLES = {
    "ingestion": "sequence",
    "state_estimation": "state_estimation",
    "geometric_mapping": "geometry",
}
"""The foundation stages a ContextRun is given, and the foundation role each one fills."""

ContextRunId = NewType("ContextRunId", str)
"""``"sha256:<hex>"`` over the foundation, the selection and the content of the artifacts."""

Disposition = Literal["produced", "reused"]


class ContextRunError(ValueError):
    """A ContextRun record cannot be published or read as it stands."""


@dataclass(frozen=True, kw_only=True)
class ContextArtifact:
    """One capability artifact a ContextRun references.

    Attributes:
        ref: The artifact, with its content hash and location.
        disposition: ``"produced"`` by this run, or ``"reused"`` from an earlier one.
    """

    ref: ArtifactRef
    disposition: Disposition

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {**self.ref.to_document(), "disposition": self.disposition}


@dataclass(frozen=True, kw_only=True)
class ContextRun:
    """The record of one execution of the context stages over a spatial foundation.

    Attributes:
        identity: Derived from the foundation, the observation selection and the content of
            the referenced artifacts; never from a path, a run number or a clock. Two executions
            that reference the same evidence are the same ContextRun.
        foundation: The spatial foundation the context stages ran against.
        observation_selection: The encoded selection of the observations processed, or
            ``None`` for the whole sequence.
        artifacts: The context artifacts, one per executed context stage, ordered by stage.
        run: Where the runtime run that holds the record is, relative to the workspace
            (``<dataset>/run-NNNN``). A locator, never part of the identity.
        plan_digest: Digest of the resolved plan the run executed.
    """

    identity: ContextRunId
    foundation: SpatialFoundation
    observation_selection: Mapping[str, Any] | None
    artifacts: tuple[ContextArtifact, ...]
    run: str
    plan_digest: str

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible record persisted as ``context_run.json``."""
        return {
            "schema_version": CONTEXT_RUN_SCHEMA_VERSION,
            "context_run_id": self.identity,
            "foundation": self.foundation.to_document(),
            "observation_selection": self.observation_selection,
            "artifacts": [artifact.to_document() for artifact in self.artifacts],
            "run": self.run,
            "plan_digest": self.plan_digest,
        }


def context_scope(plan: PipelinePlan, foundation: SpatialFoundation) -> ExecutionPlan:
    """Scope a plan to the context stages over a provided spatial foundation.

    The foundation stages are supplied, never recomputed: a ContextRun adds evidence to a fixed
    world model. The observation selection comes from the plan (``inputs.observation_selection``);
    the Visual Perception executor resolves it against the foundation's sequence before any work.

    Args:
        plan: The resolved topology.
        foundation: The spatial foundation to run against.

    Returns:
        The execution of the enabled context stages.
    """
    return plan.scope(
        targets=[stage.stage_id for stage in plan.stages if stage.stage_id in CONTEXT_STAGES],
        provided={
            stage_id: getattr(foundation, role) for stage_id, role in _FOUNDATION_ROLES.items()
        },
    )


def publish_context_run(
    run_directory: Path,
    *,
    workspace: Path,
    foundation: SpatialFoundation,
    execution: ExecutionPlan,
    record: ExecutionRecord,
) -> ContextRun:
    """Publish the ContextRun record of a completed run, once.

    Args:
        run_directory: The directory of the runtime run that executed ``execution``.
        workspace: The workspace the run lives in.
        foundation: The spatial foundation the execution was scoped to.
        execution: The scoped execution, from :func:`context_scope`.
        record: Its execution record.

    Returns:
        The published record.

    Raises:
        ContextRunError: If the record holds a stage outside the context, an artifact without
            a content hash or location, or evidence built from another foundation's artifacts.
        FileExistsError: If the run already has a ContextRun record.
    """
    foundation_refs = {
        stage_id: _signature(getattr(foundation, role))
        for stage_id, role in _FOUNDATION_ROLES.items()
    }
    artifacts: list[ContextArtifact] = []
    for stage in record.stages:
        if stage.stage_id not in CONTEXT_STAGES:
            raise ContextRunError(
                f"stage {stage.stage_id!r} is not a context stage; a ContextRun holds only "
                f"{list(CONTEXT_STAGES)}"
            )
        for name, refs in stage.inputs.items():
            for ref in refs:
                expected = foundation_refs.get(ref.stage_id)
                if expected is not None and _signature(ref) != expected:
                    raise ContextRunError(
                        f"stage {stage.stage_id!r} consumed {ref.artifact_id!r} as {name!r}, "
                        f"not the foundation's {expected[2]!r}"
                    )
        if stage.output.content_hash is None or stage.output.location is None:
            raise ContextRunError(
                f"stage {stage.stage_id!r} produced {stage.output.artifact_id!r} without a "
                "content hash and a location; a ContextRun cannot reference it"
            )
        reused = stage.decision is not None and stage.decision.kind == "reused"
        artifacts.append(
            ContextArtifact(ref=stage.output, disposition="reused" if reused else "produced")
        )
    ordered = tuple(sorted(artifacts, key=lambda artifact: artifact.ref.stage_id))
    selection = _selection(execution)
    context = ContextRun(
        identity=_identity(foundation, selection, ordered),
        foundation=foundation,
        observation_selection=selection,
        artifacts=ordered,
        run=run_directory.relative_to(workspace).as_posix(),
        plan_digest=record.plan_digest,
    )
    text = json.dumps(context.to_document(), indent=2, sort_keys=True, ensure_ascii=True)
    publish_text(run_directory, CONTEXT_RUN_FILENAME, text + "\n")
    return context


def read_context_run(run_directory: Path, *, workspace: Path) -> ContextRun:
    """Read the ContextRun record of a run and check it was not altered or moved.

    Args:
        run_directory: The directory of the runtime run that holds the record.
        workspace: The workspace the run lives in.

    Returns:
        The record.

    Raises:
        ContextRunError: If the record is unreadable, follows another schema version, names
            another run, or does not match its own identity.
    """
    path = run_directory / CONTEXT_RUN_FILENAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ContextRunError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise ContextRunError(f"{path} is not a ContextRun record")
    if document.get("schema_version") != CONTEXT_RUN_SCHEMA_VERSION:
        raise ContextRunError(
            f"{path} follows schema_version {document.get('schema_version')!r}, this runtime "
            f"reads {CONTEXT_RUN_SCHEMA_VERSION!r}"
        )
    run = run_directory.relative_to(workspace).as_posix()
    if document.get("run") != run:
        raise ContextRunError(f"{path} is the record of run {document.get('run')!r}, not {run!r}")
    try:
        foundation = SpatialFoundation.from_document(document["foundation"])
        artifacts = tuple(
            ContextArtifact(ref=ArtifactRef.from_document(item), disposition=_disposition(item))
            for item in document["artifacts"]
        )
        selection = document["observation_selection"]
        plan_digest = str(document["plan_digest"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContextRunError(f"{path} is not a valid ContextRun record: {error}") from error
    identity = _identity(foundation, selection, artifacts)
    if document.get("context_run_id") != identity:
        raise ContextRunError(
            f"{path} does not match its identity: it records {document.get('context_run_id')!r} "
            f"but its content gives {identity!r}"
        )
    return ContextRun(
        identity=identity,
        foundation=foundation,
        observation_selection=selection,
        artifacts=artifacts,
        run=run,
        plan_digest=plan_digest,
    )


def _selection(execution: ExecutionPlan) -> Mapping[str, Any] | None:
    """Return the observation selection the execution's observation-scoped stage applies."""
    for stage in execution.stages:
        if stage.observation_selection is not None:
            return stage.observation_selection
    return None


def _signature(ref: ArtifactRef) -> tuple[str, str, str, str | None]:
    """What makes two references name the same artifact; the location is only a locator."""
    return (ref.stage_id, ref.contract, ref.artifact_id, ref.content_hash)


def _disposition(item: Mapping[str, Any]) -> Disposition:
    value = item.get("disposition")
    if value == "produced":
        return "produced"
    if value == "reused":
        return "reused"
    raise ValueError(f"unknown artifact disposition {value!r}")


def _identity(
    foundation: SpatialFoundation,
    selection: Mapping[str, Any] | None,
    artifacts: tuple[ContextArtifact, ...],
) -> ContextRunId:
    """Identify a ContextRun by its foundation, its selection and its artifacts' content."""
    document = {
        "foundation": foundation.identity,
        "observation_selection": selection,
        "artifacts": sorted(
            [artifact.ref.stage_id, artifact.ref.contract, artifact.ref.content_hash]
            for artifact in artifacts
        ),
    }
    return ContextRunId(f"sha256:{hashlib.sha256(_canonical(document).encode()).hexdigest()}")


def _canonical(document: object) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

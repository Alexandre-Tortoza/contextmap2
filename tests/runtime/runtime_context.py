"""An incremental-context workspace driven by the fake World stages (issues #495-#498).

Context runs and builds execute the real runtime (plan, scope, journal, reuse, records) over
fake stages that are pure functions of their inputs and configuration, with a spatial foundation
made of plain references: no artifact is read, so no capability payload is needed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from runtime_documents import effective_from, selected_document
from runtime_worlds import World

from contextmap.runtime import (
    ArtifactRef,
    ContextRun,
    EffectiveConfig,
    PipelinePlan,
    ReusePolicy,
    RunJournal,
    SpatialFoundation,
    SpatialFoundationId,
    StageRequest,
    context_scope,
    publish_context_run,
    resolve_plan,
    run_plan,
)

FRAMES = {"kind": "frame_range", "start_frame_index": 0, "end_frame_index": 10}
LATER = {"kind": "frame_range", "start_frame_index": 10, "end_frame_index": 20}
PROVIDED_RUNTIMES = (
    "visual_perception.region_discovery",
    "visual_perception.dense_features",
    "visual_perception.region_features",
    "visual_perception.semantic_interpretation",
)


def ref(stage_id: str, contract: str, name: str) -> ArtifactRef:
    """A reference to an artifact named ``name``, as if a first run had written it."""
    return ArtifactRef(
        stage_id=stage_id,
        contract=contract,
        artifact_id=name,
        content_hash=f"sha256:{name}",
        location=f"S1/run-0001/{stage_id}",
    )


def foundation(geometry: str = "map-A") -> SpatialFoundation:
    """A spatial foundation made of plain references; ``geometry`` names its map."""
    return SpatialFoundation(
        identity=SpatialFoundationId(f"sha256:foundation-{geometry}"),
        sequence=ref("ingestion", "SequenceArtifact", "seq-A"),
        state_estimation=ref("state_estimation", "StateEstimationRunArtifact", "se-A"),
        geometry=ref("geometric_mapping", "GeometricMapArtifact", geometry),
    )


def document(
    selection: dict[str, Any] | None = None, *, min_overlap: float | None = None
) -> dict[str, Any]:
    """The test configuration, over ``selection`` and with an optional fusion policy change."""
    config = selected_document()
    config["pipeline"]["stages"]["point_representation"] = False
    if selection is not None:
        config["inputs"]["observation_selection"] = selection
    if min_overlap is not None:
        support = config["components"]["semantic_fusion"]["support"]
        support["geometry-jaccard-support-v1"]["min_overlap"] = min_overlap
    return config


class Located:
    """A World stage that also reports where it wrote, as a real executor does."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def execute(self, request: StageRequest) -> ArtifactRef:
        produced: ArtifactRef = self._inner.execute(request)
        assert request.output_dir is not None and request.workspace is not None
        location = request.output_dir.relative_to(request.workspace).as_posix()
        return dataclasses.replace(produced, location=location)


class ContextWorkspace:
    """One workspace, one reuse index and one fake world for context runs and builds."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.workspace = root / "ws"
        self.world = World()
        self.index = self.world.store(root / "index")
        self.last: dict[str, Any] = {}

    def plan(self, config: dict[str, Any] | None = None) -> tuple[EffectiveConfig, PipelinePlan]:
        effective = effective_from(self.root, config or document())
        return effective, resolve_plan(effective)

    def executors(self, plan: PipelinePlan) -> dict[str, Located]:
        return {
            stage.stage_id: Located(self.world.executor(stage.stage_id, stage.output or ""))
            for stage in plan.stages
        }

    def reuse(self) -> ReusePolicy:
        return ReusePolicy(store=self.index, code_identity="code-1")

    def run(
        self,
        over: SpatialFoundation | None = None,
        selection: dict[str, Any] | None = None,
    ) -> ContextRun:
        """Execute and publish one ContextRun over ``over`` (the default foundation)."""
        base = over or foundation()
        effective, plan = self.plan(document(selection))
        execution = context_scope(plan, base)
        journal = RunJournal.create(self.workspace, effective, execution)
        record = run_plan(
            execution,
            self.executors(plan),
            environ={},
            module_available=lambda _name: True,
            provided_runtimes=PROVIDED_RUNTIMES,
            reuse=self.reuse(),
            journal=journal,
        )
        self.last = {
            "run_directory": journal.directory,
            "workspace": self.workspace,
            "foundation": base,
            "execution": execution,
            "record": record,
        }
        return publish_context_run(
            journal.directory,
            workspace=self.workspace,
            foundation=base,
            execution=execution,
            record=record,
        )

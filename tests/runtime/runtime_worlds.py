"""A fake pipeline world shared by the runtime tests: pure stages over content identity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from contextmap.runtime import ArtifactRef, FileArtifactStore, PipelinePlan, StageRequest


class World:
    """Fake stages that behave like pure functions of their inputs and configuration.

    Content is derived from what a stage consumed and how it is configured, so identical
    identity gives identical content; each execution is a distinct run with its own
    artifact id. ``ignore_config`` makes content independent of configuration, to model a
    configuration change that does not change what the stage produces.
    """

    def __init__(self, *, ignore_config: bool = False) -> None:
        self.ignore_config = ignore_config
        self.runs: list[str] = []
        self.existing: set[str] = set()
        self.fail_at: str | None = None
        self.fail_with: BaseException = RuntimeError("out of memory")
        self._count = 0

    def executor(self, stage_id: str, contract: str) -> Any:
        world = self

        class Executor:
            def execute(self, request: StageRequest) -> ArtifactRef:
                if world.fail_at == stage_id:
                    raise world.fail_with
                world._count += 1
                world.runs.append(stage_id)
                parts = [stage_id]
                if not world.ignore_config:
                    parts.append(request.config_digest)
                parts.extend(
                    f"{name}={ref.content_hash}"
                    for name, refs in sorted(request.inputs.items())
                    for ref in refs
                )
                content = "sha256:" + hashlib.sha256("|".join(parts).encode()).hexdigest()
                artifact_id = f"{stage_id}-run{world._count}"
                world.existing.add(artifact_id)
                return ArtifactRef(
                    stage_id=stage_id,
                    contract=contract,
                    artifact_id=artifact_id,
                    content_hash=content,
                )

        return Executor()

    def executors(self, plan: PipelinePlan) -> dict[str, Any]:
        return {
            stage.stage_id: self.executor(stage.stage_id, stage.output or "")
            for stage in plan.stages
        }

    def store(self, root: Path) -> FileArtifactStore:
        return FileArtifactStore(root, verify=lambda ref: ref.artifact_id in self.existing)


def world_executors(world: World) -> dict[str, Any]:
    """One fake executor per canonical stage, for the CLI, which has no plan to build them from."""
    from contextmap.runtime.catalog import CANONICAL_PRESET

    return {
        stage.stage_id: world.executor(stage.stage_id, stage.output or "")
        for stage in CANONICAL_PRESET.stages
    }


def run_cli(*argv: str, **options: Any) -> tuple[int, str, str]:
    """Run the command-line interface in-process and return its exit code and both streams."""
    import io

    from contextmap.runtime.cli import main

    out, err = io.StringIO(), io.StringIO()
    options.setdefault("module_available", lambda _name: True)
    options.setdefault("environ", {})
    code = main(list(argv), stdout=out, stderr=err, **options)
    return code, out.getvalue(), err.getvalue()

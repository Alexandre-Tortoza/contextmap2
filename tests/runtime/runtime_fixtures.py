"""Fixtures the runtime tests import explicitly (no conftest: its module name is not unique)."""

from __future__ import annotations

import dataclasses

import pytest

from contextmap.runtime import catalog


@pytest.fixture
def unavailable_context_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Append a stage whose capability does not exist, as a later preset will declare it.

    The canonical preset declares only what executes today, so the behavior of an unavailable
    stage (reported in the topology, blocked at preflight, never simulated) is exercised on a
    preset with a ``context_map`` stage marked unavailable, and the real catalog is left
    untouched.
    """
    preset = catalog.CANONICAL_PRESET
    later = catalog.StageDeclaration(
        stage_id="context_map",
        capability="artifact",
        available=False,
        unavailable_reason="the artifact capability is not implemented yet (milestone #15)",
        inputs=(
            catalog.StageInput(name="fusion", contract=catalog.FUSION, source="semantic_fusion"),
        ),
        output="ContextMapArtifact",
    )
    monkeypatch.setitem(
        catalog.PRESETS,
        preset.preset_id,
        dataclasses.replace(preset, stages=(*preset.stages, later)),
    )

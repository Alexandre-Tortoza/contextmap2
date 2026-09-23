"""Fixtures the runtime tests import explicitly (no conftest: its module name is not unique)."""

from __future__ import annotations

import dataclasses

import pytest

from contextmap.runtime import catalog


@pytest.fixture
def unavailable_future_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add a stage marked unavailable, as the catalog does for a capability that is missing.

    Every capability of the canonical/1 topology exists today (including ``context_map``, now
    real), so the behavior of an unavailable stage (reported in the topology, blocked at
    preflight, never simulated) is exercised on a preset extended with a fictional
    ``scene_graph`` stage marked unavailable -- no such capability exists at all -- and the
    real catalog is left untouched.
    """
    preset = catalog.CANONICAL_PRESET
    scene_graph_stage = catalog.StageDeclaration(
        stage_id="scene_graph",
        capability="scene_graph",
        available=False,
        unavailable_reason="the scene_graph capability is not implemented yet (milestone #15)",
        inputs=(
            catalog.StageInput(
                name="geometry", contract="GeometricMapArtifact", source="geometric_mapping"
            ),
        ),
        output="SceneGraphArtifact",
    )
    monkeypatch.setitem(
        catalog.PRESETS,
        preset.preset_id,
        dataclasses.replace(preset, stages=(*preset.stages, scene_graph_stage)),
    )

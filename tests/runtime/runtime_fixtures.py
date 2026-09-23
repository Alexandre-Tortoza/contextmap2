"""Fixtures the runtime tests import explicitly (no conftest: its module name is not unique)."""

from __future__ import annotations

import dataclasses

import pytest

from contextmap.runtime import catalog


@pytest.fixture
def unavailable_context_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add a stage marked unavailable, as the catalog does for a capability that is missing.

    Every capability of the canonical/1 topology exists today, so the behavior of an
    unavailable stage (reported in the topology, blocked at preflight, never simulated) is
    exercised on a preset extended with a ``context_map`` stage marked unavailable -- the
    capability is real, but no preset declares it as a stage yet -- and the real catalog is
    left untouched.
    """
    preset = catalog.CANONICAL_PRESET
    context_map_stage = catalog.StageDeclaration(
        stage_id="context_map",
        capability="artifact",
        available=False,
        unavailable_reason="the artifact capability is not implemented yet (milestone #15)",
        inputs=(
            catalog.StageInput(
                name="geometry", contract="GeometricMapArtifact", source="geometric_mapping"
            ),
        ),
        output="ContextMapArtifact",
    )
    monkeypatch.setitem(
        catalog.PRESETS,
        preset.preset_id,
        dataclasses.replace(preset, stages=(*preset.stages, context_map_stage)),
    )

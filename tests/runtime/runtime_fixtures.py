"""Fixtures the runtime tests import explicitly (no conftest: its module name is not unique)."""

from __future__ import annotations

import dataclasses

import pytest

from contextmap.runtime import catalog


@pytest.fixture
def unavailable_context_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare the last stage unavailable, as the catalog does for a capability that is missing.

    Every capability of the canonical topology exists today, so the behavior of an unavailable
    stage (reported in the topology, blocked at preflight, never simulated) is exercised on a
    preset that marks ``context_map`` unavailable, and the real catalog is left untouched.
    """
    preset = catalog.CANONICAL_PRESET
    stages = tuple(
        dataclasses.replace(
            stage,
            available=False,
            unavailable_reason="the artifact capability is not implemented yet (milestone #15)",
        )
        if stage.stage_id == "context_map"
        else stage
        for stage in preset.stages
    )
    monkeypatch.setitem(
        catalog.PRESETS, preset.preset_id, dataclasses.replace(preset, stages=stages)
    )

"""Composition of the local Qwen visual input budget (#526).

The budget is an ordinary, validated parameter of the ``qwen`` backend block: the capability's
own configuration checks it before any model runtime is requested, it reaches the runtime
provider inside the effective ``QwenSemanticConfig``, and it enters the stage identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document

from contextmap.runtime import resolve_plan
from contextmap.runtime.composition import RuntimeProvider, compose
from contextmap.runtime.errors import BackendConfigurationError
from contextmap.visual_perception.backends.qwen import QwenSemanticConfig

REGION = "visual_perception.region_discovery"
INTERPRETER = "visual_perception.semantic_interpretation"
BUDGET = {"min_pixels": 256 * 32 * 32, "max_pixels": 1280 * 32 * 32}


class _Recorder:
    """Records the configuration each caller-supplied runtime was requested with."""

    def __init__(self) -> None:
        self.configs: dict[str, Any] = {}

    def provider(self, component_id: str) -> RuntimeProvider:
        def provide(config: Any, secrets: Any) -> object:
            self.configs[component_id] = config
            return object()

        return provide


def _qwen_document(**budget: int) -> dict[str, Any]:
    document = selected_document()
    document["components"]["visual_perception"]["semantic_interpretation"]["qwen"].update(budget)
    return document


def _compose(tmp_path: Path, document: dict[str, Any], recorder: _Recorder) -> Any:
    return compose(
        effective_from(tmp_path, document),
        providers={REGION: recorder.provider(REGION), INTERPRETER: recorder.provider(INTERPRETER)},
        module_available=lambda _name: True,
        environ={},
    )


def test_a_configured_budget_reaches_the_runtime_and_the_interpreter_identity(
    tmp_path: Path,
) -> None:
    unbudgeted = _compose(tmp_path, selected_document(), _Recorder())
    recorder = _Recorder()

    composed = _compose(tmp_path, _qwen_document(**BUDGET), recorder)

    config = recorder.configs[INTERPRETER]
    assert isinstance(config, QwenSemanticConfig)
    assert (config.min_pixels, config.max_pixels) == (262_144, 1_310_720)
    assert (
        composed.semantic_interpreter.backend_provenance().configuration_fingerprint
        != unbudgeted.semantic_interpreter.backend_provenance().configuration_fingerprint
    )


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        ({"max_pixels": 1_310_720}, "set together"),
        ({"min_pixels": 1_310_720, "max_pixels": 262_144}, "must not exceed max_pixels"),
    ],
)
def test_an_unenforceable_budget_is_refused_before_the_model_runtime_is_requested(
    tmp_path: Path, budget: dict[str, int], message: str
) -> None:
    recorder = _Recorder()

    with pytest.raises(BackendConfigurationError, match=message):
        _compose(tmp_path, _qwen_document(**budget), recorder)

    assert INTERPRETER not in recorder.configs


def test_changing_only_the_budget_changes_only_the_perception_stage_identity(
    tmp_path: Path,
) -> None:
    """A budget ablation recomputes perception; the geometry path upstream stays reusable."""
    base = resolve_plan(effective_from(tmp_path, selected_document()))

    changed = resolve_plan(effective_from(tmp_path, _qwen_document(**BUDGET)))

    differing = {
        stage.stage_id
        for stage in changed.stages
        if stage.config_digest != base.stage(stage.stage_id).config_digest
    }
    assert differing == {"visual_perception"}

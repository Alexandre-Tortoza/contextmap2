"""Composition of the Eagle 2.5 semantic interpreter from resolved configuration (#570)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime_documents import CANONICAL_PROMPT_POLICY, effective_from, selected_document

from contextmap.runtime.composition import (
    ComposedRuntime,
    RuntimeProvider,
    SemanticRequestPrompt,
    compose,
)
from contextmap.runtime.errors import BackendConfigurationError, BackendRuntimeMissingError
from contextmap.visual_perception import SemanticInterpretationMode
from contextmap.visual_perception.backends.eagle2_5 import (
    EagleSemanticConfig,
    EagleSemanticInterpreter,
)

REGION_DISCOVERY = "visual_perception.region_discovery"
INTERPRETER = "visual_perception.semantic_interpretation"
REVISION = "c" * 40


class _Recorder:
    """Stands in for caller-supplied model runtimes and records every request for one."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def provider(self, component_id: str) -> RuntimeProvider:
        def provide(config: Any, secrets: Any) -> object:
            self.calls.append((component_id, config))
            return object()

        return provide


def _eagle_document(**overrides: object) -> dict[str, Any]:
    parameters: dict[str, object] = {
        "model": "nvidia/Eagle2.5-8B",
        "revision": REVISION,
        "precision": "bfloat16",
        "max_new_tokens": 256,
        "temperature": 0.0,
        "max_dynamic_tiles": 6,
        "prompt_policy": dict(CANONICAL_PROMPT_POLICY),
    }
    parameters.update(overrides)
    document = selected_document()
    document["components"]["visual_perception"]["semantic_interpretation"] = {
        "backend": "eagle2_5",
        "eagle2_5": {key: value for key, value in parameters.items() if value is not None},
    }
    return document


def _compose(
    tmp_path: Path,
    document: dict[str, Any],
    *,
    recorder: _Recorder | None = None,
    with_interpreter_runtime: bool = True,
) -> ComposedRuntime:
    recorder = recorder or _Recorder()
    providers = {REGION_DISCOVERY: recorder.provider(REGION_DISCOVERY)}
    if with_interpreter_runtime:
        providers[INTERPRETER] = recorder.provider(INTERPRETER)
    return compose(
        effective_from(tmp_path, document),
        providers=providers,
        module_available=lambda _name: True,
        environ={},
    )


def test_eagle2_5_is_selectable_from_resolved_configuration(tmp_path: Path) -> None:
    recorder = _Recorder()
    document = _eagle_document(use_thumbnail=False)
    document["resources"]["device"] = "cuda"

    composed = _compose(tmp_path, document, recorder=recorder)

    assert isinstance(composed.semantic_interpreter, EagleSemanticInterpreter)
    (component_id, config) = recorder.calls[-1]
    assert component_id == INTERPRETER
    assert config == EagleSemanticConfig(
        model="nvidia/Eagle2.5-8B",
        revision=REVISION,
        device="cuda",
        precision="bfloat16",
        max_new_tokens=256,
        temperature=0.0,
        max_dynamic_tiles=6,
        use_thumbnail=False,
    )
    assert composed.semantic_interpreter.backend_provenance().configuration_fingerprint == (
        EagleSemanticInterpreter(config=config, runtime=object()).configuration_fingerprint  # type: ignore[arg-type]
    )


def test_requests_name_the_configured_prompt_policy(tmp_path: Path) -> None:
    policy = {"scene": "scene/v1", "region": "region-abstention/v1"}

    composed = _compose(tmp_path, _eagle_document(prompt_policy=policy))

    assert composed.semantic_prompts == {
        SemanticInterpretationMode.SCENE: SemanticRequestPrompt(
            template_id="scene/v1", output_schema="semantic-response/1"
        ),
        SemanticInterpretationMode.REGION: SemanticRequestPrompt(
            template_id="region-abstention/v1", output_schema="semantic-response/1"
        ),
    }


@pytest.mark.parametrize(
    ("missing", "message"),
    [("prompt_policy", "prompt_policy"), ("max_dynamic_tiles", "max_dynamic_tiles")],
)
def test_the_prompt_policy_and_the_visual_budget_are_never_defaulted(
    tmp_path: Path, missing: str, message: str
) -> None:
    recorder = _Recorder()

    with pytest.raises(BackendConfigurationError, match=message):
        _compose(tmp_path, _eagle_document(**{missing: None}), recorder=recorder)

    assert [component for component, _ in recorder.calls] == [REGION_DISCOVERY]


def test_without_a_runtime_provider_the_missing_eagle_runtime_is_reported(
    tmp_path: Path,
) -> None:
    with pytest.raises(BackendRuntimeMissingError, match="EagleRuntime"):
        _compose(tmp_path, _eagle_document(), with_interpreter_runtime=False)

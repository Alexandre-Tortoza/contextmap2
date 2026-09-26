"""#544: resolved configuration expresses one semantic request policy, end to end.

The policy is resolved from the effective configuration alone -- no provider, no model, no
backend constructed -- so an experiment manifest can name an arm's treatment by its identity
before anything runs, and the runtime composes exactly that policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import SHA_A, effective_from, selected_document

from contextmap.runtime import ConfigurationError, resolve_effective_config
from contextmap.runtime.composition import compose, resolve_semantic_request_policy
from contextmap.runtime.errors import BackendConfigurationError
from contextmap.runtime.pipeline import resolve_plan
from contextmap.visual_perception import (
    SemanticModePrompt,
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
)

_SEMANTIC = "visual_perception.semantic_interpretation"


def _with_qwen(**groups: object) -> dict[str, Any]:
    document = selected_document()
    document["components"]["visual_perception"]["semantic_interpretation"]["qwen"].update(groups)
    return document


def _conditioned(**extra: object) -> dict[str, Any]:
    return _with_qwen(
        prompt_policy={
            "scene": "scene/v1",
            "region": "region-scene-context/v1",
            "region_scene_context": True,
            **extra,
        },
        view_policy={
            "region_views": ["masked_subject", "tight_crop"],
            "mask_fill_rgb": [0, 0, 0],
        },
    )


def _toml(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                "[components.visual_perception.semantic_interpretation.qwen]",
                'prompt_policy = { region_scene_context = true, region = "region-scene-context/v1"'
                ', scene = "scene/v1" }',
                'view_policy = { mask_fill_rgb = [0, 0, 0], region_views = ["masked_subject", '
                '"tight_crop"] }',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_the_policy_resolves_from_configuration_alone_and_is_the_one_composed(
    tmp_path: Path,
) -> None:
    effective = effective_from(tmp_path, _conditioned())

    resolved = resolve_semantic_request_policy(effective)

    assert resolved == SemanticRequestPolicy.from_prompt_policy(
        SemanticPromptPolicy(
            scene="scene/v1", region="region-scene-context/v1", region_scene_context=True
        ),
        views=SemanticViewPolicy(
            region_views=(VisualViewKind.MASKED_SUBJECT, VisualViewKind.TIGHT_CROP),
            mask_fill_rgb=(0, 0, 0),
        ),
    )
    composed = compose(
        effective,
        providers={
            "visual_perception.region_discovery": lambda _config, _secrets: object(),
            _SEMANTIC: lambda _config, _secrets: object(),
        },
        module_available=lambda _name: True,
        environ={},
    )
    assert composed.semantic_request_policy == resolved


def test_equivalent_configurations_resolve_deterministically_to_one_identity(
    tmp_path: Path,
) -> None:
    """JSON file, TOML file with keys in another order, and dotted overrides agree."""
    base = tmp_path / "base.json"
    base.write_text(json.dumps(selected_document()), encoding="utf-8")
    (tmp_path / "json").mkdir()
    from_json = effective_from(tmp_path / "json", _conditioned())
    from_toml = resolve_effective_config(files=[base, _toml(tmp_path / "policy.toml")])
    from_overrides = resolve_effective_config(
        files=[base],
        overrides=[
            f"components.{_SEMANTIC}.qwen.prompt_policy.region=region-scene-context/v1",
            f"components.{_SEMANTIC}.qwen.prompt_policy.region_scene_context=true",
            f'components.{_SEMANTIC}.qwen.view_policy={{"region_views": '
            '["masked_subject", "tight_crop"], "mask_fill_rgb": [0, 0, 0]}',
        ],
    )

    fingerprints = {
        resolve_semantic_request_policy(effective).fingerprint()
        for effective in (from_json, from_toml, from_overrides)
    }
    stage_digests = {
        resolve_plan(effective).stage("visual_perception").config_digest
        for effective in (from_json, from_toml, from_overrides)
    }
    assert len(fingerprints) == 1
    assert len(stage_digests) == 1
    assert from_json.digest == from_toml.digest == from_overrides.digest


def test_equivalent_policies_deduplicate_by_their_identity(tmp_path: Path) -> None:
    """An explicit ``region_scene_context = false`` is the policy that omits the switch."""
    implicit = _with_qwen(prompt_policy={"scene": "scene/v1", "region": "region/v1"})
    explicit = _with_qwen(
        prompt_policy={"scene": "scene/v1", "region": "region/v1", "region_scene_context": False}
    )
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    first = resolve_semantic_request_policy(effective_from(tmp_path / "a", implicit))
    second = resolve_semantic_request_policy(effective_from(tmp_path / "b", explicit))

    assert first == second
    assert first.fingerprint() == second.fingerprint()


def test_switching_scene_context_changes_only_the_semantic_stage_identity(
    tmp_path: Path,
) -> None:
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    off = effective_from(tmp_path / "off", _conditioned(region_scene_context=False))
    on = effective_from(tmp_path / "on", _conditioned())

    assert (
        resolve_semantic_request_policy(off).fingerprint()
        != resolve_semantic_request_policy(on).fingerprint()
    )
    base, changed = resolve_plan(off), resolve_plan(on)
    assert {
        stage.stage_id
        for stage in changed.stages
        if stage.config_digest != base.stage(stage.stage_id).config_digest
    } == {"visual_perception"}


def test_florence2_resolves_its_task_native_prompt_for_its_mode_only(tmp_path: Path) -> None:
    document = selected_document()
    document["components"]["visual_perception"]["semantic_interpretation"] = {
        "backend": "florence2",
        "florence2": {
            "checkpoint": "florence-community/Florence-2-large",
            "revision": SHA_A,
            "task": "<CAPTION>",
            "supported_modes": ["scene"],
            "precision": "float32",
            "max_new_tokens": 64,
            "temperature": 0.0,
            "view_policy": {"region_views": ["tight_crop"]},
        },
    }

    policy = resolve_semantic_request_policy(effective_from(tmp_path, document))

    assert policy.region is None
    assert policy.scene == SemanticModePrompt(
        template_id="florence2-task-prompt/1:<CAPTION>", output_schema="semantic-response/1"
    )


@pytest.mark.parametrize(
    ("groups", "error", "message"),
    [
        ({"view_policy": None}, BackendConfigurationError, "view_policy"),
        (
            {"prompt_policy": {"scene": "scene/v1", "region": "region/v9"}},
            BackendConfigurationError,
            "unknown semantic prompt template",
        ),
    ],
)
def test_an_incomplete_or_invalid_policy_fails_resolution(
    tmp_path: Path, groups: dict[str, object], error: type[Exception], message: str
) -> None:
    document = _with_qwen(**groups)
    qwen = document["components"]["visual_perception"]["semantic_interpretation"]["qwen"]
    for name in [name for name, value in groups.items() if value is None]:
        del qwen[name]

    with pytest.raises(error, match=message):
        resolve_semantic_request_policy(effective_from(tmp_path, document))


def test_without_a_selected_interpreter_there_is_no_policy_to_resolve(tmp_path: Path) -> None:
    document = selected_document()
    document["components"]["visual_perception"]["semantic_interpretation"] = {"backend": None}

    with pytest.raises(ConfigurationError, match="semantic_interpretation"):
        resolve_semantic_request_policy(effective_from(tmp_path, document))

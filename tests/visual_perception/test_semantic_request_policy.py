"""Tests for the resolved semantic request policy of a run (#544)."""

from __future__ import annotations

import pytest

from contextmap.visual_perception import (
    SEMANTIC_REQUEST_POLICY_VERSION,
    SemanticInterpretationMode,
    SemanticInterpreterCapabilities,
    SemanticModePrompt,
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
    check_request_policy_supported,
)

SCENE = SemanticInterpretationMode.SCENE
REGION = SemanticInterpretationMode.REGION
TIGHT = SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,))
MASKED_TIGHT = SemanticViewPolicy(
    region_views=(VisualViewKind.MASKED_SUBJECT, VisualViewKind.TIGHT_CROP),
    mask_fill_rgb=(0, 0, 0),
)


def _policy(
    region: str = "region/v1", *, views: SemanticViewPolicy = TIGHT, context: bool = False
) -> SemanticRequestPolicy:
    return SemanticRequestPolicy.from_prompt_policy(
        SemanticPromptPolicy(scene="scene/v1", region=region, region_scene_context=context),
        views=views,
    )


def _capabilities(**overrides: object) -> SemanticInterpreterCapabilities:
    values: dict[str, object] = {
        "supported_modes": frozenset({SCENE, REGION}),
        "supported_view_kinds": frozenset(VisualViewKind),
        "accepts_visual_features": False,
        "accepts_scene_context": True,
    }
    values.update(overrides)
    return SemanticInterpreterCapabilities(**values)  # type: ignore[arg-type]


class TestIdentity:
    def test_one_policy_names_prompt_views_scene_context_and_schema_of_every_mode(self) -> None:
        policy = _policy("region-scene-context/v1", views=MASKED_TIGHT, context=True)

        document = policy.to_document()

        assert document["version"] == SEMANTIC_REQUEST_POLICY_VERSION
        assert document["prompts"] == {
            "scene": {"template_id": "scene/v1", "output_schema": "semantic-response/1"},
            "region": {
                "template_id": "region-scene-context/v1",
                "output_schema": "semantic-response/1",
            },
        }
        assert document["region_scene_context"] is True
        assert document["views"] == MASKED_TIGHT.to_document()
        assert policy.prompt_for(REGION) == SemanticModePrompt(
            template_id="region-scene-context/v1", output_schema="semantic-response/1"
        )

    def test_equivalent_selections_resolve_to_one_identity(self) -> None:
        explicit = SemanticRequestPolicy.from_prompt_policy(
            SemanticPromptPolicy(scene="scene/v1", region="region/v1", region_scene_context=False),
            views=SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,)),
        )

        assert explicit == _policy()
        assert explicit.fingerprint() == _policy().fingerprint()
        assert len({_policy().fingerprint() for _ in range(3)}) == 1

    @pytest.mark.parametrize(
        "changed",
        [
            _policy("region-abstention/v1"),
            _policy(views=MASKED_TIGHT),
            _policy("region-scene-context/v1"),
            _policy("region-scene-context/v1", context=True),
        ],
    )
    def test_any_changed_dimension_changes_the_identity(
        self, changed: SemanticRequestPolicy
    ) -> None:
        assert changed.fingerprint() != _policy().fingerprint()

    def test_the_view_order_is_part_of_the_identity(self) -> None:
        reordered = SemanticViewPolicy(
            region_views=(VisualViewKind.TIGHT_CROP, VisualViewKind.MASKED_SUBJECT),
            mask_fill_rgb=(0, 0, 0),
        )

        assert _policy(views=reordered).fingerprint() != _policy(views=MASKED_TIGHT).fingerprint()


class TestValidation:
    def test_a_mode_without_a_prompt_is_absent_not_defaulted(self) -> None:
        policy = SemanticRequestPolicy(
            region=SemanticModePrompt(
                template_id="florence2-task-prompt/1:<REGION_TO_CATEGORY>",
                output_schema="semantic-response/1",
            ),
            views=TIGHT,
        )

        assert policy.prompt_for(SCENE) is None
        assert policy.to_document()["prompts"]["scene"] is None

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ({}, "at least one mode"),
            (
                {
                    "scene": SemanticModePrompt(
                        template_id="scene/v1", output_schema="semantic-response/1"
                    ),
                    "region_scene_context": True,
                },
                "region prompt",
            ),
        ],
    )
    def test_an_empty_or_contradictory_policy_is_refused(
        self, values: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SemanticRequestPolicy(views=TIGHT, **values)  # type: ignore[arg-type]

    def test_a_mode_prompt_needs_an_identity_and_a_schema(self) -> None:
        with pytest.raises(ValueError, match="template_id"):
            SemanticModePrompt(template_id=" ", output_schema="semantic-response/1")


class TestPreflight:
    def test_a_policy_the_interpreter_can_consume_passes(self) -> None:
        check_request_policy_supported(
            _policy("region-scene-context/v1", views=MASKED_TIGHT, context=True), _capabilities()
        )

    @pytest.mark.parametrize(
        ("policy", "capabilities", "message"),
        [
            (_policy(), _capabilities(supported_modes=frozenset({REGION})), "scene"),
            (
                _policy(views=MASKED_TIGHT),
                _capabilities(max_visual_views=1),
                "at most 1",
            ),
            (
                _policy("region-scene-context/v1", context=True),
                _capabilities(accepts_scene_context=False),
                "does not accept scene context",
            ),
        ],
    )
    def test_an_unsupported_combination_is_refused_before_any_inference(
        self,
        policy: SemanticRequestPolicy,
        capabilities: SemanticInterpreterCapabilities,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            check_request_policy_supported(policy, capabilities)

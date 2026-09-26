"""The resolved policy that assembles every semantic request of one run (#544).

A run's semantic requests are shaped by three explicitly selected choices: the prompt policy
of each mode, the ordered evidence views, and whether region requests are conditioned on the
frame's scene context. :class:`SemanticRequestPolicy` is that selection, resolved, with one
identity (:meth:`SemanticRequestPolicy.fingerprint`), so an experiment arm can name exactly the
treatment its requests received and two equivalent selections are recognisably the same. The
backend, its model and its visual-input budget stay backend configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from contextmap.visual_perception.semantic_prompt import SemanticPromptPolicy
from contextmap.visual_perception.semantic_requests import (
    SemanticInterpretationMode,
    SemanticInterpreterCapabilities,
)
from contextmap.visual_perception.semantic_views import (
    SemanticViewPolicy,
    check_view_policy_supported,
)

SEMANTIC_REQUEST_POLICY_VERSION = "semantic-request-policy/1"
"""Version of the request-policy document :meth:`SemanticRequestPolicy.to_document` writes."""


@dataclass(frozen=True, kw_only=True)
class SemanticModePrompt:
    """The prompt policy identity every request of one mode names.

    Attributes:
        template_id: Becomes the request's ``prompt_template_id``: a free-form catalog
            template, or a task-native prompt policy (Florence-2).
        output_schema: Becomes the request's ``requested_output_schema``.
    """

    template_id: str
    output_schema: str

    def __post_init__(self) -> None:
        """Reject an unnamed prompt policy or output schema."""
        if not self.template_id.strip():
            raise ValueError("mode prompt template_id must not be empty")
        if not self.output_schema.strip():
            raise ValueError("mode prompt output_schema must not be empty")


@dataclass(frozen=True, kw_only=True)
class SemanticRequestPolicy:
    """Prompt, ordered views, scene-context switch and output schema of a run's requests.

    A mode without a prompt is not interpreted: its requests are refused, never defaulted.

    Attributes:
        views: Ordered evidence views and how they are built.
        scene: Prompt of scene requests, or ``None`` when scenes are not interpreted.
        region: Prompt of region requests, or ``None`` when regions are not interpreted.
        region_scene_context: Whether every region request carries the scene context its
            frame's scene request produced (#529).
    """

    views: SemanticViewPolicy
    scene: SemanticModePrompt | None = None
    region: SemanticModePrompt | None = None
    region_scene_context: bool = False

    def __post_init__(self) -> None:
        """Require one interpreted mode, and a region prompt for scene-context conditioning."""
        if self.scene is None and self.region is None:
            raise ValueError("a semantic request policy interprets at least one mode")
        if self.region_scene_context and self.region is None:
            raise ValueError("scene-context conditioning of region requests needs a region prompt")

    @classmethod
    def from_prompt_policy(
        cls, prompt_policy: SemanticPromptPolicy, *, views: SemanticViewPolicy
    ) -> SemanticRequestPolicy:
        """Resolve an instruction-following interpreter's configured prompt and view policies."""
        scene = prompt_policy.template_for(SemanticInterpretationMode.SCENE)
        region = prompt_policy.template_for(SemanticInterpretationMode.REGION)
        return cls(
            views=views,
            scene=SemanticModePrompt(
                template_id=scene.template_id, output_schema=scene.output_schema_version
            ),
            region=SemanticModePrompt(
                template_id=region.template_id, output_schema=region.output_schema_version
            ),
            region_scene_context=prompt_policy.region_scene_context,
        )

    def prompt_for(self, mode: SemanticInterpretationMode) -> SemanticModePrompt | None:
        """Return the prompt of one mode, or ``None`` when that mode is not interpreted."""
        return self.scene if mode is SemanticInterpretationMode.SCENE else self.region

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible record of every choice, the view rules included."""
        return {
            "version": SEMANTIC_REQUEST_POLICY_VERSION,
            "prompts": {
                mode.value: (
                    None
                    if (prompt := self.prompt_for(mode)) is None
                    else {"template_id": prompt.template_id, "output_schema": prompt.output_schema}
                )
                for mode in SemanticInterpretationMode
            },
            "region_scene_context": self.region_scene_context,
            "views": self.views.to_document(),
        }

    def fingerprint(self) -> str:
        """Return the ``"sha256:<hex>"`` identity of :meth:`to_document`."""
        canonical = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_request_policy_supported(
    policy: SemanticRequestPolicy, capabilities: SemanticInterpreterCapabilities
) -> None:
    """Refuse, before any inference, a policy an interpreter declares it cannot consume.

    Raises:
        ValueError: If the policy prompts a mode the interpreter does not support, its views
            do not fit the interpreter (:func:`check_view_policy_supported`), or it
            conditions region requests on scene context the interpreter does not accept.
    """
    for mode in SemanticInterpretationMode:
        if policy.prompt_for(mode) is not None and mode not in capabilities.supported_modes:
            raise ValueError(
                f"the request policy prompts {mode.value} requests, which the semantic "
                "interpreter does not support"
            )
    check_view_policy_supported(policy.views, capabilities)
    if policy.region_scene_context and not capabilities.accepts_scene_context:
        raise ValueError(
            "the request policy conditions region requests on scene context, but the semantic "
            "interpreter does not accept scene context"
        )

"""The capability and compatibility matrix of the perception experiment (#522).

The matrix is data, so these tests tie it to reality: a supported entry must be selectable
through the runtime catalog and composition root today, a planned entry must not claim a
backend that already exists, every declared control and native selector must be accepted by
the backend configuration, and the published document must be rendered from the data.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from dataclasses import replace
from pathlib import Path

import pytest

from contextmap.evaluation.capability_matrix import (
    CAPABILITY_MATRIX,
    Capability,
    CapabilityMatrix,
    CapabilityMatrixError,
    Composition,
    CompositionStatus,
    Control,
    ImplementationStatus,
    Role,
)
from contextmap.evaluation.metrics import MetricKind, default_metric_registry
from contextmap.runtime.catalog import CANONICAL_PRESET, COMPONENTS
from contextmap.runtime.composition import composable_backends
from contextmap.visual_perception.backends.florence2 import Florence2Config
from contextmap.visual_perception.backends.florence2_semantic import FLORENCE2_SEMANTIC_TASKS
from contextmap.visual_perception.backends.locateanything import (
    CATEGORY_DETECTION_POLICY,
    PHRASE_GROUNDING_POLICY,
    POINTING_POLICY,
)
from contextmap.visual_perception.backends.sam3 import Sam3Strategy

MATRIX = CAPABILITY_MATRIX
DOC = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "contextmap"
    / "evaluation"
    / "docs"
    / "capability-matrix.md"
)

# Onde cada backend lê seus parâmetros: a dataclass de configuração da própria capability e os
# grupos reservados que a composition root separa antes de construí-la.
_BACKEND_CONFIGURATION = {
    ("visual_perception.region_discovery", "sam2"): (
        "contextmap.visual_perception.backends.sam2.Sam2Config",
        {"pass_config", "normalization_config"},
    ),
    ("visual_perception.region_discovery", "sam3"): (
        "contextmap.visual_perception.backends.sam3.Sam3Config",
        {"pass_config", "normalization_config"},
    ),
    ("visual_perception.region_discovery", "florence2"): (
        "contextmap.visual_perception.backends.florence2.Florence2Config",
        {"pass_config", "normalization_config"},
    ),
    ("visual_perception.region_grounding", "locateanything"): (
        "contextmap.visual_perception.backends.locateanything.LocateAnythingConfig",
        {"query_set"},
    ),
    ("visual_perception.region_refinement", "sam2"): (
        "contextmap.visual_perception.backends.sam2.Sam2RefinementConfig",
        set(),
    ),
    ("visual_perception.dense_features", "dinov2"): (
        "contextmap.visual_perception.backends.dinov2.DinoV2Config",
        set(),
    ),
    ("visual_perception.dense_features", "dinov3"): (
        "contextmap.visual_perception.backends.dinov3.DinoV3Config",
        set(),
    ),
    ("visual_perception.dense_features", "siglip2"): (
        "contextmap.visual_perception.backends.siglip2.Siglip2Config",
        set(),
    ),
    ("visual_perception.region_features", "clip"): (
        "contextmap.visual_perception.backends.clip.ClipConfig",
        set(),
    ),
    ("visual_perception.region_features", "alphaclip"): (
        "contextmap.visual_perception.backends.alphaclip.AlphaClipConfig",
        set(),
    ),
    ("visual_perception.semantic_interpretation", "qwen"): (
        "contextmap.visual_perception.backends.qwen.QwenSemanticConfig",
        {"prompt_policy"},
    ),
    ("visual_perception.semantic_interpretation", "gemini"): (
        "contextmap.visual_perception.backends.gemini.GeminiSemanticConfig",
        {"prompt_policy"},
    ),
    ("visual_perception.semantic_interpretation", "eagle2_5"): (
        "contextmap.visual_perception.backends.eagle2_5.EagleSemanticConfig",
        {"prompt_policy"},
    ),
    ("visual_perception.semantic_interpretation", "florence2"): (
        "contextmap.visual_perception.backends.florence2_semantic.Florence2SemanticConfig",
        set(),
    ),
}


def _symbol(path: str) -> object:
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _stage(stage_id: str):  # type: ignore[no-untyped-def]
    return CANONICAL_PRESET.stage(stage_id)


def _supported() -> list[Capability]:
    return [item for item in MATRIX.capabilities if item.status is ImplementationStatus.SUPPORTED]


# ------------------------------------------------------------------ tied to the runtime


@pytest.mark.parametrize("capability", _supported(), ids=lambda item: item.capability_id)
def test_a_supported_capability_is_selectable_through_the_runtime_today(
    capability: Capability,
) -> None:
    binding = capability.runtime
    assert binding is not None
    stage = _stage(binding.stage_id)
    if binding.component_id is None:
        assert binding.backend_id is None
        return
    assert binding.component_id in stage.components
    assert binding.backend_id in COMPONENTS[binding.component_id].backends
    assert (binding.component_id, binding.backend_id) in composable_backends()


def test_a_planned_capability_never_claims_a_backend_that_exists() -> None:
    planned = [
        item
        for item in MATRIX.capabilities
        if item.status is not ImplementationStatus.SUPPORTED and item.runtime is not None
    ]

    for item in planned:
        binding = item.runtime
        assert binding is not None
        assert item.status is ImplementationStatus.PLANNED
        # O slot já existe; o backend não. Quando ele chegar, a entrada precisa virar supported.
        assert binding.component_id is not None
        assert binding.component_id in COMPONENTS
        assert binding.backend_id not in COMPONENTS[binding.component_id].backends
        assert (binding.component_id, binding.backend_id) not in composable_backends()


def test_the_eagle_family_shares_the_contracts_of_the_existing_backends() -> None:
    for mode in ("scene", "region"):
        eagle = MATRIX.capability(f"eagle2_5.{mode}_interpretation")
        assert eagle.status is ImplementationStatus.SUPPORTED
        assert (
            eagle.comparison_group
            == MATRIX.capability(f"qwen.{mode}_interpretation").comparison_group
        )
        assert eagle.runtime is not None
        assert (eagle.runtime.component_id, eagle.runtime.backend_id) == (
            "visual_perception.semantic_interpretation",
            "eagle2_5",
        )


def _accepts(component_id: str, backend_id: str, name: str, value: str) -> bool:
    if (component_id, backend_id, name) == (
        "visual_perception.region_discovery",
        "sam3",
        "strategy",
    ):
        return value in {item.value for item in Sam3Strategy}
    if (component_id, backend_id, name) == (
        "visual_perception.region_discovery",
        "florence2",
        "task",
    ):
        try:
            Florence2Config(checkpoint="microsoft/Florence-2-large", task=value)
        except ValueError:
            return False
        return True
    if (component_id, backend_id, name) == (
        "visual_perception.semantic_interpretation",
        "florence2",
        "task",
    ):
        return value in FLORENCE2_SEMANTIC_TASKS
    if (component_id, backend_id, name) == (
        "visual_perception.region_grounding",
        "locateanything",
        "query_set.queries.policy_id",
    ):
        return value in {CATEGORY_DETECTION_POLICY, PHRASE_GROUNDING_POLICY, POINTING_POLICY}
    raise AssertionError(f"no validator for selector {name!r} of {component_id}/{backend_id}")


def test_every_native_selector_is_a_value_the_backend_accepts() -> None:
    checked = 0
    for item in MATRIX.capabilities:
        if item.runtime is None or item.runtime.component_id is None:
            continue
        for name, values in item.runtime.settings:
            for value in values:
                assert item.runtime.backend_id is not None
                assert _accepts(item.runtime.component_id, item.runtime.backend_id, name, value), (
                    f"{item.capability_id}: {name}={value!r}"
                )
                checked += 1

    assert checked >= 14


def test_every_supported_control_is_a_parameter_the_backend_reads() -> None:
    for item in _supported():
        binding = item.runtime
        assert binding is not None
        controls = [
            control
            for control in (*item.prompt_controls, *item.model_controls)
            if control.status is ImplementationStatus.SUPPORTED
        ]
        if binding.component_id is None:
            assert not controls, f"{item.capability_id} has no backend to read its controls"
            continue
        path, reserved = _BACKEND_CONFIGURATION[(binding.component_id, str(binding.backend_id))]
        fields = {field.name for field in dataclasses.fields(_symbol(path))}  # type: ignore[arg-type]
        for control in controls:
            assert control.name in fields | reserved, f"{item.capability_id}: {control.name}"


def test_every_named_adapter_exists() -> None:
    for item in MATRIX.capabilities:
        if item.adapter is not None:
            assert _symbol(item.adapter) is not None


def test_every_metric_family_has_quality_metrics_in_the_registry() -> None:
    registry = default_metric_registry()
    families = {
        item.metric_family for item in MATRIX.capabilities if item.metric_family is not None
    } | {item.metric_family for item in MATRIX.compositions if item.metric_family is not None}

    for family in families:
        assert any(
            definition.stage is family and definition.kind is MetricKind.QUALITY
            for definition in registry.definitions
        ), family


# ------------------------------------------------------------------- the issue's inventory


def test_every_required_model_family_is_inventoried_in_the_same_matrix() -> None:
    backends = {item.backend for item in MATRIX.capabilities}

    assert backends >= {
        "SAM2",
        "SAM3",
        "Florence-2",
        "DINOv2",
        "DINOv3",
        "CLIP",
        "AlphaCLIP",
        "Qwen",
        "Gemini",
        "LocateAnything",
        "Eagle 2.5",
        "SigLIP2",
        "LocateAnything3D",
    }


def test_native_operations_are_finer_than_a_generic_region_discovery_label() -> None:
    def operations(backend: str) -> set[str]:
        return {item.capability_id for item in MATRIX.capabilities if item.backend == backend}

    assert len(operations("Florence-2")) >= 9
    assert len(operations("SAM3")) >= 3
    assert len(operations("LocateAnything")) >= 5
    florence_tasks = {
        value
        for item in MATRIX.capabilities
        if item.backend == "Florence-2" and item.runtime is not None
        for _, values in item.runtime.settings
        for value in values
    }
    assert florence_tasks >= {
        "<REGION_PROPOSAL>",
        "<OD>",
        "<DENSE_REGION_CAPTION>",
        "<OPEN_VOCABULARY_DETECTION>",
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        "<REGION_TO_SEGMENTATION>",
        "<REGION_TO_CATEGORY>",
        "<REGION_TO_DESCRIPTION>",
        "<CAPTION>",
    }


@pytest.mark.parametrize(
    ("capability_id", "issue"),
    [
        ("locateanything.visual_prompt_grounding", 574),
        ("locateanything3d.open_vocabulary_3d_proposal", 575),
        ("contextmap2.metric_3d_verification", 575),
    ],
)
def test_capabilities_without_public_assets_stay_blocked_and_unselectable(
    capability_id: str, issue: int
) -> None:
    item = MATRIX.capability(capability_id)

    assert item.status is ImplementationStatus.BLOCKED
    assert item.issue == issue
    assert item.runtime is None
    assert item.status_reason


# ------------------------------------------------------------------------ compositions

REQUIRED_COMPOSITIONS = {
    ("locateanything.category_detection", "sam2.box_prompt_refinement"): (
        CompositionStatus.SUPPORTED
    ),
    ("sam2.box_prompt_refinement", "contextmap2.mask_membership_association"): (
        CompositionStatus.SUPPORTED
    ),
    ("locateanything.category_detection", "contextmap2.mask_membership_association"): (
        CompositionStatus.INCOMPATIBLE
    ),
    ("sam3.text_concept_segmentation", "contextmap2.mask_membership_association"): (
        CompositionStatus.SUPPORTED
    ),
    ("florence2.referring_expression_segmentation", "contextmap2.mask_membership_association"): (
        CompositionStatus.SUPPORTED
    ),
    ("dinov2.dense_patch_features", "contextmap2.dense_feature_sampling"): (
        CompositionStatus.PLANNED
    ),
    ("dinov3.dense_patch_features", "contextmap2.dense_feature_sampling"): (
        CompositionStatus.PLANNED
    ),
    ("siglip2.dense_patch_features", "contextmap2.dense_feature_sampling"): (
        CompositionStatus.PLANNED
    ),
    ("sam2.automatic_mask_generation", "alphaclip.region_embedding"): CompositionStatus.SUPPORTED,
    ("florence2.object_detection", "alphaclip.region_embedding"): CompositionStatus.INCOMPATIBLE,
    ("sam2.automatic_mask_generation", "clip.region_embedding"): CompositionStatus.SUPPORTED,
    ("contextmap2.semantic_view_assembly", "qwen.region_interpretation"): (
        CompositionStatus.SUPPORTED
    ),
    ("contextmap2.semantic_view_assembly", "gemini.scene_interpretation"): (
        CompositionStatus.SUPPORTED
    ),
    ("contextmap2.semantic_view_assembly", "eagle2_5.region_interpretation"): (
        CompositionStatus.SUPPORTED
    ),
    ("qwen.region_interpretation", "alphaclip.semantic_scoring"): CompositionStatus.PLANNED,
    ("clip.region_embedding", "clip.semantic_scoring"): CompositionStatus.INCOMPATIBLE,
    ("contextmap2.mask_membership_association", "contextmap2.semantic_fusion"): (
        CompositionStatus.SUPPORTED
    ),
    ("locateanything3d.open_vocabulary_3d_proposal", "contextmap2.metric_3d_verification"): (
        CompositionStatus.BLOCKED
    ),
    ("contextmap2.metric_3d_verification", "contextmap2.semantic_fusion"): (
        CompositionStatus.BLOCKED
    ),
}


@pytest.mark.parametrize(
    ("pair", "status"), REQUIRED_COMPOSITIONS.items(), ids=lambda value: str(value)
)
def test_every_composition_the_issue_names_is_classified(
    pair: tuple[str, str], status: CompositionStatus
) -> None:
    composition = MATRIX.composition(*pair)

    assert composition.status is status
    assert composition.note


def test_every_composition_status_is_used() -> None:
    assert {item.status for item in MATRIX.compositions} == set(CompositionStatus)


def test_a_box_only_region_never_passes_for_mask_evidence() -> None:
    # Uma caixa não é uma máscara: nada que exija a máscara aceita uma região só com caixa.
    for producer in ("locateanything.category_detection", "florence2.object_detection"):
        assert MATRIX.capability(producer).geometry.value == "box"
        for consumer in ("contextmap2.mask_membership_association", "alphaclip.region_embedding"):
            assert MATRIX.composition(producer, consumer).status is CompositionStatus.INCOMPATIBLE


def test_only_a_supported_composition_passes_the_pre_load_guard() -> None:
    MATRIX.require_supported_composition(
        "sam3.text_concept_segmentation", "contextmap2.mask_membership_association"
    )

    with pytest.raises(CapabilityMatrixError, match="incompatible") as incompatible:
        MATRIX.require_supported_composition(
            "locateanything.category_detection", "contextmap2.mask_membership_association"
        )
    assert "NO_INLINE_MASK" in str(incompatible.value)
    MATRIX.require_supported_composition(
        "locateanything.category_detection", "sam2.box_prompt_refinement"
    )
    with pytest.raises(CapabilityMatrixError, match="#527"):
        MATRIX.require_supported_composition(
            "qwen.region_interpretation", "alphaclip.semantic_scoring"
        )
    with pytest.raises(CapabilityMatrixError, match="undeclared"):
        MATRIX.require_supported_composition(
            "qwen.scene_interpretation", "dinov2.dense_patch_features"
        )
    with pytest.raises(CapabilityMatrixError, match="unknown capability"):
        MATRIX.capability("sam4.everything")


# ------------------------------------------------------------------------ comparability


@pytest.mark.parametrize(
    ("first", "second", "comparable"),
    [
        ("dinov2.dense_patch_features", "siglip2.dense_patch_features", True),
        ("dinov3.dense_patch_features", "dinov2.dense_patch_features", True),
        ("qwen.region_interpretation", "gemini.region_interpretation", True),
        ("qwen.region_interpretation", "eagle2_5.region_interpretation", True),
        ("qwen.scene_interpretation", "eagle2_5.scene_interpretation", True),
        (
            "sam3.text_concept_segmentation",
            "locateanything.category_detection->sam2.box_prompt_refinement",
            True,
        ),
        # Mesma geometria não basta: a tarefa e o condicionamento da entrada precisam casar.
        ("locateanything.category_detection", "florence2.object_detection", False),
        ("florence2.region_proposal", "locateanything.category_detection", False),
        ("sam2.automatic_mask_generation", "sam3.text_concept_segmentation", False),
        ("sam2.automatic_mask_generation", "florence2.region_proposal", False),
        ("locateanything.phrase_grounding", "florence2.caption_phrase_grounding", False),
        ("qwen.region_interpretation", "florence2.region_category", False),
        ("clip.region_embedding", "alphaclip.region_embedding", False),
        ("clip.semantic_scoring", "alphaclip.semantic_scoring", False),
        ("florence2.region_to_segmentation", "florence2.region_to_segmentation", False),
    ],
)
def test_operations_are_comparable_only_when_task_conditioning_and_geometry_align(
    first: str, second: str, comparable: bool
) -> None:
    assert MATRIX.comparable(first, second) is comparable
    assert MATRIX.comparable(second, first) is comparable


# ------------------------------------------------------------------------- invariants


def _capability(**changes: object) -> Capability:
    return replace(MATRIX.capability("sam2.automatic_mask_generation"), **changes)  # type: ignore[arg-type]


def test_a_capability_states_why_it_is_not_supported() -> None:
    with pytest.raises(ValueError, match="status_reason"):
        _capability(status=ImplementationStatus.OUT_OF_SCOPE, runtime=None, role=None)
    with pytest.raises(ValueError, match="issue"):
        _capability(status=ImplementationStatus.PLANNED, status_reason="later", issue=None)
    with pytest.raises(ValueError, match="runtime"):
        _capability(runtime=None)
    with pytest.raises(ValueError, match="issue"):
        Control(name="view_policy", status=ImplementationStatus.PLANNED)


def test_an_out_of_scope_capability_takes_no_role_in_a_pipeline() -> None:
    with pytest.raises(ValueError, match="out of scope"):
        _capability(
            status=ImplementationStatus.OUT_OF_SCOPE,
            runtime=None,
            status_reason="not in this milestone",
        )


def _matrix(**changes: object) -> CapabilityMatrix:
    return replace(MATRIX, **changes)  # type: ignore[arg-type]


def test_the_matrix_refuses_an_inconsistent_composition() -> None:
    unknown = Composition(
        producer="sam2.automatic_mask_generation",
        consumer="sam4.everything",
        status=CompositionStatus.SUPPORTED,
        note="n",
    )
    promoted = replace(
        MATRIX.composition("dinov2.dense_patch_features", "contextmap2.dense_feature_sampling"),
        status=CompositionStatus.SUPPORTED,
        issue=None,
    )
    wrong_role = Composition(
        producer="dinov2.dense_patch_features",
        consumer="qwen.region_interpretation",
        status=CompositionStatus.PLANNED,
        note="n",
        issue=1,
    )

    with pytest.raises(CapabilityMatrixError, match="unknown capability"):
        _matrix(compositions=(*MATRIX.compositions, unknown))
    with pytest.raises(CapabilityMatrixError, match="supported"):
        _matrix(
            compositions=tuple(
                promoted if item.composition_id == promoted.composition_id else item
                for item in MATRIX.compositions
            )
        )
    with pytest.raises(CapabilityMatrixError, match="role"):
        _matrix(compositions=(*MATRIX.compositions, wrong_role))
    with pytest.raises(CapabilityMatrixError, match="twice"):
        _matrix(compositions=(*MATRIX.compositions, MATRIX.compositions[0]))
    with pytest.raises(CapabilityMatrixError, match="twice"):
        _matrix(capabilities=(*MATRIX.capabilities, MATRIX.capabilities[0]))


def test_roles_of_the_matrix_are_all_used() -> None:
    used = {item.role for item in MATRIX.capabilities if item.role is not None}

    assert used == set(Role)


# ------------------------------------------------------------------------ the document

_START = "<!-- {name}:start (gerado de CAPABILITY_MATRIX; não editar à mão) -->"
_END = "<!-- {name}:end -->"


def _cell(value: object) -> str:
    return "—" if value is None or value == "" else str(value).replace("|", "\\|")


def _capability_table() -> str:
    rows = [
        "| id | backend | operação nativa | papel | status | issue | grupo de comparação |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in MATRIX.capabilities:
        issue = None if item.issue is None else f"#{item.issue}"
        role = None if item.role is None else item.role.value
        rows.append(
            f"| `{item.capability_id}` | {_cell(item.backend)} | {_cell(item.native_operation)} "
            f"| {_cell(role)} | {item.status.value} | {_cell(issue)} "
            f"| {_cell(item.comparison_group)} |"
        )
    return "\n".join(rows)


def _composition_table() -> str:
    rows = [
        "| produtor | consumidor | status | issue | grupo de comparação |",
        "|---|---|---|---|---|",
    ]
    for item in MATRIX.compositions:
        issue = None if item.issue is None else f"#{item.issue}"
        rows.append(
            f"| `{item.producer}` | `{item.consumer}` | {item.status.value} | {_cell(issue)} "
            f"| {_cell(item.comparison_group)} |"
        )
    return "\n".join(rows)


@pytest.mark.parametrize(
    ("name", "render"),
    [("capabilities", _capability_table), ("compositions", _composition_table)],
)
def test_the_document_tables_are_rendered_from_the_matrix(name: str, render: object) -> None:
    text = DOC.read_text(encoding="utf-8")
    start = _START.format(name=name)
    end = _END.format(name=name)
    assert start in text and end in text, f"{DOC.name} lacks the {name} markers"
    published = text.split(start, 1)[1].split(end, 1)[0].strip()

    expected = render()  # type: ignore[operator]
    assert published == expected, f"regenerate the {name} table of {DOC.name}:\n{expected}"


def test_the_document_names_the_matrix_version() -> None:
    text = DOC.read_text(encoding="utf-8")

    assert f"`{MATRIX.version}`" in text
    assert MATRIX.upstream_checked_on in text
    assert re.search(r"#57[45]", text)

"""Factorized experiment phases expanded into resolved runtime configurations (#527).

A phase names what it varies; expansion resolves every arm through the runtime's own
configuration and topology resolution (never composing a backend or loading a model), checks
its wiring against the capability matrix, records blocked and duplicate arms with their
reasons and builds the controlled comparison of the runnable ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from experiment_builders import REGISTRY, make_executor, pinned, validated_reference

from contextmap.evaluation.capability_matrix import CAPABILITY_MATRIX
from contextmap.evaluation.experiment_phases import (
    ArmState,
    Factor,
    FactorLevel,
    MatchedParameters,
    PhaseExpansionError,
    PhaseKind,
    PhaseManifest,
    PhaseSpec,
    expand_phase,
    record_phase_outcomes,
    write_phase_manifest,
)
from contextmap.evaluation.experiment_runner import (
    ArmUnavailableError,
    read_verified_document,
    run_experiment,
)
from contextmap.evaluation.experiments import (
    AblationMode,
    DifferenceKind,
    ExperimentPurpose,
    MetricRef,
    SelectionBinding,
    VariationKind,
    arm_differences,
)
from contextmap.evaluation.metrics import EvaluationStage
from contextmap.evaluation.reference_integrity import ValidatedReferenceSet

VP = "components.visual_perception"
SEMANTIC = f"{VP}.semantic_interpretation"
REVISION = "1" * 40
PROMPTS = {"scene": "scene/v1", "region": "region/v1"}
VIEWS = {"region": "tight-crop/1"}
QWEN = {
    f"{SEMANTIC}.backend": "qwen",
    f"{SEMANTIC}.qwen.model": "Qwen/Qwen3-VL-4B-Instruct",
    f"{SEMANTIC}.qwen.revision": REVISION,
    f"{SEMANTIC}.qwen.precision": "bfloat16",
    f"{SEMANTIC}.qwen.max_new_tokens": 256,
    f"{SEMANTIC}.qwen.temperature": 0.0,
    f"{SEMANTIC}.qwen.prompt_policy": PROMPTS,
    f"{SEMANTIC}.qwen.view_policy": VIEWS,
}
EAGLE = {
    f"{SEMANTIC}.backend": "eagle2_5",
    f"{SEMANTIC}.eagle2_5.model": "nvidia/Eagle2.5-8B",
    f"{SEMANTIC}.eagle2_5.revision": "2" * 40,
    f"{SEMANTIC}.eagle2_5.max_new_tokens": 256,
    f"{SEMANTIC}.eagle2_5.temperature": 0.0,
    f"{SEMANTIC}.eagle2_5.max_dynamic_tiles": 6,
    f"{SEMANTIC}.eagle2_5.prompt_policy": PROMPTS,
    f"{SEMANTIC}.eagle2_5.view_policy": VIEWS,
}
DINOV3 = {
    f"{VP}.dense_features.backend": "dinov3",
    f"{VP}.dense_features.dinov3.checkpoint": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    f"{VP}.dense_features.dinov3.revision": REVISION,
}
SIGLIP2 = {
    f"{VP}.dense_features.backend": "siglip2",
    f"{VP}.dense_features.siglip2.checkpoint": "google/siglip2-so400m-patch16-512",
    f"{VP}.dense_features.siglip2.revision": REVISION,
    f"{VP}.dense_features.siglip2.input_size": 512,
}
BASE = {
    "components.ingestion.source_adapter.backend": "ros1_bag",
    f"{VP}.region_discovery.backend": "sam2",
    f"{VP}.region_discovery.sam2.checkpoint": "sam2.1_hiera_tiny.pt",
    **DINOV3,
    f"{VP}.region_features.backend": "alphaclip",
    f"{VP}.region_features.alphaclip.model_name": "ViT-L/14",
    **QWEN,
}
BASE_EDGES = (
    ("sam2.automatic_mask_generation", "contextmap2.semantic_view_assembly"),
    ("contextmap2.semantic_view_assembly", "qwen.region_interpretation"),
    ("sam2.automatic_mask_generation", "alphaclip.region_embedding"),
)
LOCATE = {
    f"{VP}.region_grounding.backend": "locateanything",
    f"{VP}.region_grounding.locateanything.model": "nvidia/LocateAnything-3B",
    f"{VP}.region_grounding.locateanything.revision": REVISION,
    f"{VP}.region_grounding.locateanything.query_set": {
        "queries": [
            {
                "task": "category_detection",
                "policy_id": "locateanything.category-detection/1",
                "geometry": "box",
                "categories": ["chair", "door"],
            }
        ]
    },
}
REFINE = {
    f"{VP}.region_refinement.backend": "sam2",
    f"{VP}.region_refinement.sam2.checkpoint": "sam2.1_hiera_tiny.pt",
}
SEQUENCE = pinned("sequence", "sequence-0001")


def level(value: str, overrides: dict[str, Any] | None = None, **kwargs: Any) -> FactorLevel:
    return FactorLevel(value=value, overrides=tuple((overrides or {}).items()), **kwargs)


def factor(name: str, kind: VariationKind, *levels: FactorLevel) -> Factor:
    return Factor(name=name, kind=kind, levels=levels, baseline=levels[0].value)


PROMPT_FACTOR = factor(
    "prompt_policy",
    VariationKind.POLICY,
    level("region/v1", {f"{SEMANTIC}.qwen.prompt_policy": PROMPTS}),
    level(
        "region-abstention/v1",
        {f"{SEMANTIC}.qwen.prompt_policy": {**PROMPTS, "region": "region-abstention/v1"}},
    ),
)
DENSE_FACTOR = factor(
    "dense_features", VariationKind.BACKEND, level("dinov3", DINOV3), level("siglip2", SIGLIP2)
)
INTERPRETER_FACTOR = factor(
    "semantic_backend",
    VariationKind.BACKEND,
    level(
        "qwen", QWEN, edges=(("contextmap2.semantic_view_assembly", "qwen.region_interpretation"),)
    ),
    level(
        "eagle2_5",
        EAGLE,
        edges=(("contextmap2.semantic_view_assembly", "eagle2_5.region_interpretation"),),
    ),
)
REFINEMENT_FACTOR = factor(
    "refinement",
    VariationKind.TOPOLOGY,
    level("none"),
    level(
        "sam2",
        REFINE,
        edges=(("locateanything.category_detection", "sam2.box_prompt_refinement"),),
    ),
)


@pytest.fixture
def validated(tmp_path: Path) -> ValidatedReferenceSet:
    return validated_reference(tmp_path / "reference")


def phase(validated: ValidatedReferenceSet, **changes: Any) -> PhaseSpec:
    fields: dict[str, Any] = {
        "experiment_id": "perception-ablation",
        "phase_id": "semantic-prompt",
        "kind": PhaseKind.SEMANTIC_PROMPT,
        "version": "1.0.0",
        "description": "controlled phase",
        "purpose": ExperimentPurpose.EVALUATION,
        "evaluated_stage": EvaluationStage.SEMANTIC_INTERPRETATION,
        "selection": SelectionBinding.for_split(validated.manifest, "regions-by-sequence", "test"),
        "base_overrides": tuple(BASE.items()),
        "base_edges": BASE_EDGES,
        "factors": (PROMPT_FACTOR,),
        "targets": ("visual_perception",),
        "pinned": (("ingestion", SEQUENCE),),
        "quality_metrics": (MetricRef(name="semantic.acceptable_claim_rate", version="1"),),
        "resource_capture": (MetricRef(name="runtime.wall_time", version="1"),),
        "registry": REGISTRY.identity(),
    }
    fields.update(changes)
    return PhaseSpec(**fields)


def _arm(manifest: PhaseManifest, value: str) -> Any:
    return next(item for item in manifest.arms if value in dict(item.assignments).values())


# --------------------------------------------------------------------- one-factor phases


def test_a_prompt_phase_resolves_every_arm_and_declares_only_the_prompt_field(
    validated: ValidatedReferenceSet,
) -> None:
    manifest = expand_phase(phase(validated))

    assert [item.arm_id for item in manifest.arms] == ["baseline", "arm-01"]
    assert all(item.state is ArmState.PLANNED for item in manifest.arms)
    assert all(str(item.effective_config_digest).startswith("sha256:") for item in manifest.arms)
    assert len({item.effective_config_digest for item in manifest.arms}) == 2
    experiment = manifest.experiment
    assert experiment is not None
    assert experiment.mode is AblationMode.SELECTED
    (difference,) = arm_differences(experiment.variables, *experiment.arms)
    assert difference.stage_id == "visual_perception.semantic_interpretation"
    assert difference.field == "qwen.prompt_policy.region"
    assert difference.declared_by == ("prompt_policy",)
    assert {item.arm_id: item.topology_digest for item in manifest.arms} == {
        arm.arm_id: arm.topology.digest() for arm in experiment.arms
    }


def test_a_backend_substitution_keeps_the_topology_and_changes_only_the_backend(
    validated: ValidatedReferenceSet,
) -> None:
    manifest = expand_phase(
        phase(
            validated,
            phase_id="dense-features",
            kind=PhaseKind.FEATURE_BACKEND,
            factors=(DENSE_FACTOR,),
            evaluated_stage=EvaluationStage.FEATURE_EXTRACTION,
        )
    )

    experiment = manifest.experiment
    assert experiment is not None
    baseline, siglip = experiment.arms
    assert [item.stage_id for item in baseline.topology.stages] == [
        item.stage_id for item in siglip.topology.stages
    ]
    differences = arm_differences(experiment.variables, baseline, siglip)
    assert {item.stage_id for item in differences} == {"visual_perception.dense_features"}
    assert DifferenceKind.BACKEND in {item.kind for item in differences}
    assert "siglip2.dense_patch_features" in _arm(manifest, "siglip2").capabilities


def test_qwen_and_eagle_are_compared_under_a_matched_prompt_and_view_policy(
    validated: ValidatedReferenceSet,
) -> None:
    matched = MatchedParameters(
        component_id="visual_perception.semantic_interpretation",
        names=("prompt_policy", "view_policy", "max_new_tokens", "temperature"),
    )
    spec = phase(
        validated,
        phase_id="semantic-interpreter",
        kind=PhaseKind.SEMANTIC_INTERPRETER,
        factors=(INTERPRETER_FACTOR,),
        base_edges=BASE_EDGES[::2],
        matched=(matched,),
    )

    manifest = expand_phase(spec)

    assert _arm(manifest, "eagle2_5").state is ArmState.PLANNED
    assert "eagle2_5.region_interpretation" in _arm(manifest, "eagle2_5").capabilities
    other_views = factor(
        "semantic_backend",
        VariationKind.BACKEND,
        INTERPRETER_FACTOR.levels[0],
        level(
            "eagle2_5",
            {**EAGLE, f"{SEMANTIC}.eagle2_5.view_policy": {"region": "masked+tight/1"}},
            edges=INTERPRETER_FACTOR.levels[1].edges,
        ),
    )
    with pytest.raises(PhaseExpansionError, match="view_policy") as error:
        expand_phase(replace(spec, factors=(other_views,)))
    assert "masked+tight/1" in str(error.value) and "tight-crop/1" in str(error.value)


def test_a_backend_substitution_that_changes_the_task_is_refused(
    validated: ValidatedReferenceSet,
) -> None:
    florence = level(
        "florence2",
        {
            f"{SEMANTIC}.backend": "florence2",
            f"{SEMANTIC}.florence2.checkpoint": "florence-community/Florence-2-large",
            f"{SEMANTIC}.florence2.task": "<REGION_TO_CATEGORY>",
        },
        edges=(("contextmap2.semantic_view_assembly", "florence2.region_category"),),
    )

    with pytest.raises(PhaseExpansionError, match="no comparable counterpart"):
        expand_phase(
            phase(
                validated,
                kind=PhaseKind.SEMANTIC_INTERPRETER,
                base_edges=BASE_EDGES[::2],
                factors=(
                    factor(
                        "semantic_backend", VariationKind.BACKEND, level("qwen", QWEN), florence
                    ),
                ),
            )
        )


# ------------------------------------------------------------------ topology and wiring


def test_box_only_grounding_and_grounding_with_refinement_are_a_topology_ablation(
    validated: ValidatedReferenceSet,
) -> None:
    manifest = expand_phase(
        phase(
            validated,
            phase_id="grounding-refinement",
            kind=PhaseKind.GROUNDING_REFINEMENT,
            base_overrides=tuple({**BASE, **LOCATE}.items()),
            factors=(REFINEMENT_FACTOR,),
            evaluated_stage=EvaluationStage.REGION_DISCOVERY,
        )
    )

    experiment = manifest.experiment
    assert experiment is not None
    box_only, refined = experiment.arms
    assert "visual_perception.region_refinement" not in {
        item.stage_id for item in box_only.topology.stages
    }
    assert refined.topology.stage("visual_perception.region_refinement").depends_on == (
        "ingestion.source_adapter",
    )
    # A fiação produtor -> consumidor fica registrada no arm e verificada contra a matriz.
    assert (
        "locateanything.category_detection",
        "sam2.box_prompt_refinement",
    ) in _arm(manifest, "sam2").edges
    (difference,) = arm_differences(experiment.variables, box_only, refined)
    assert (difference.kind, difference.declared_by) == (DifferenceKind.PRESENCE, ("refinement",))
    assert "sam2.box_prompt_refinement" in _arm(manifest, "sam2").capabilities
    assert "locateanything.category_detection" in _arm(manifest, "none").capabilities


def test_a_topology_phase_cannot_hide_a_backend_change(validated: ValidatedReferenceSet) -> None:
    sneaky = replace(
        REFINEMENT_FACTOR,
        levels=(
            REFINEMENT_FACTOR.levels[0],
            replace(
                REFINEMENT_FACTOR.levels[1],
                overrides=(*REFINEMENT_FACTOR.levels[1].overrides, *SIGLIP2.items()),
            ),
        ),
    )

    with pytest.raises(PhaseExpansionError, match="undeclared change"):
        expand_phase(
            phase(
                validated,
                kind=PhaseKind.GROUNDING_REFINEMENT,
                base_overrides=tuple({**BASE, **LOCATE}.items()),
                factors=(sneaky,),
            )
        )


@pytest.mark.parametrize(
    ("edge", "reason"),
    [
        (
            ("locateanything.category_detection", "contextmap2.mask_membership_association"),
            "not in the arm",
        ),
        (("locateanything.category_detection", "alphaclip.region_embedding"), "incompatible"),
        (("dinov3.dense_patch_features", "qwen.region_interpretation"), "undeclared composition"),
        (("florence2.object_detection", "clip.region_embedding"), "not in the arm"),
    ],
)
def test_invalid_wiring_fails_preflight(
    validated: ValidatedReferenceSet, edge: tuple[str, str], reason: str
) -> None:
    with pytest.raises(PhaseExpansionError, match=reason):
        expand_phase(
            phase(
                validated,
                base_overrides=tuple({**BASE, **LOCATE}.items()),
                base_edges=(*BASE_EDGES, edge),
            )
        )


def test_a_selection_the_matrix_does_not_know_fails_preflight(
    validated: ValidatedReferenceSet,
) -> None:
    automatic_sam3 = {
        f"{VP}.region_discovery.backend": "sam3",
        f"{VP}.region_discovery.sam3.checkpoint": "facebook/sam3",
        f"{VP}.region_discovery.sam3.strategy": "automatic",
    }

    with pytest.raises(PhaseExpansionError, match="no capability of the matrix"):
        expand_phase(phase(validated, base_overrides=tuple({**BASE, **automatic_sam3}.items())))


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({f"{VP}.region_discovery.backend": "sam9"}, "sam9"),
        ({f"{VP}.region_discovery.backend": None}, "no backend selected"),
    ],
)
def test_an_arm_that_does_not_resolve_fails_preflight(
    validated: ValidatedReferenceSet, overrides: dict[str, Any], reason: str
) -> None:
    with pytest.raises(PhaseExpansionError, match=reason):
        expand_phase(phase(validated, base_overrides=tuple({**BASE, **overrides}.items())))


# ------------------------------------------------------- blocked, duplicate and selected


def test_the_semantic_scorer_arm_stays_blocked_with_the_matrix_reason(
    validated: ValidatedReferenceSet,
) -> None:
    scorer = factor(
        "semantic_scoring",
        VariationKind.EVIDENCE_CHANNELS,
        level("disabled"),
        level(
            "alphaclip",
            capabilities=("alphaclip.semantic_scoring",),
            edges=(
                ("qwen.region_interpretation", "alphaclip.semantic_scoring"),
                ("alphaclip.region_embedding", "alphaclip.semantic_scoring"),
            ),
        ),
    )

    manifest = expand_phase(phase(validated, kind=PhaseKind.FEATURE_BACKEND, factors=(scorer,)))

    enabled = _arm(manifest, "alphaclip")
    assert enabled.state is ArmState.BLOCKED
    assert 527 in enabled.issues
    reason = CAPABILITY_MATRIX.capability("alphaclip.semantic_scoring").status_reason
    assert reason is not None and reason in str(enabled.reason)
    assert _arm(manifest, "disabled").state is ArmState.PLANNED
    # Um único arm executável não é uma comparação.
    assert manifest.experiment is None


@pytest.mark.parametrize(
    ("capability", "issue"),
    [
        ("locateanything.visual_prompt_grounding", 574),
        ("locateanything3d.open_vocabulary_3d_proposal", 575),
    ],
)
def test_arms_that_need_unpublished_assets_stay_blocked(
    validated: ValidatedReferenceSet, capability: str, issue: int
) -> None:
    blocked = factor(
        "region_source",
        VariationKind.BACKEND,
        level("discovery"),
        level("unpublished", capabilities=(capability,)),
    )

    manifest = expand_phase(phase(validated, kind=PhaseKind.BACKEND_GRID, factors=(blocked,)))

    arm = _arm(manifest, "unpublished")
    assert arm.state is ArmState.BLOCKED
    assert arm.issues == (issue,)
    assert f"#{issue}" in str(arm.reason)


def test_equivalent_effective_configurations_deduplicate_deterministically(
    validated: ValidatedReferenceSet,
) -> None:
    budget = factor(
        "visual_budget",
        VariationKind.CONFIGURATION,
        level("unbounded"),
        level(
            "small", {f"{SEMANTIC}.qwen.min_pixels": 200704, f"{SEMANTIC}.qwen.max_pixels": 401408}
        ),
        level(
            "small-again",
            {f"{SEMANTIC}.qwen.max_pixels": 401408, f"{SEMANTIC}.qwen.min_pixels": 200704},
        ),
    )
    spec = phase(validated, kind=PhaseKind.VISUAL_BUDGET, factors=(budget,))

    first = expand_phase(spec)
    second = expand_phase(spec)

    duplicate = _arm(first, "small-again")
    assert duplicate.state is ArmState.SKIPPED
    assert duplicate.duplicate_of == _arm(first, "small").arm_id
    assert duplicate.effective_config_digest == _arm(first, "small").effective_config_digest
    assert first.digest() == second.digest()
    assert first.experiment is not None
    assert [item.arm_id for item in first.experiment.arms] == ["baseline", "arm-01"]


def test_a_composition_phase_expands_only_the_selected_compositions(
    validated: ValidatedReferenceSet,
) -> None:
    grounding = factor(
        "grounding",
        VariationKind.TOPOLOGY,
        level("none"),
        level("locateanything", LOCATE),
    )
    spec = phase(
        validated,
        phase_id="compositions",
        kind=PhaseKind.PIPELINE_COMPOSITION,
        factors=(grounding, REFINEMENT_FACTOR, DENSE_FACTOR, INTERPRETER_FACTOR),
        base_edges=BASE_EDGES[::2],
        compositions=(
            (
                ("grounding", "locateanything"),
                ("refinement", "sam2"),
                ("dense_features", "dinov3"),
                ("semantic_backend", "qwen"),
            ),
            (
                ("grounding", "locateanything"),
                ("refinement", "sam2"),
                ("dense_features", "siglip2"),
                ("semantic_backend", "eagle2_5"),
            ),
        ),
    )

    manifest = expand_phase(spec)

    assert len(manifest.arms) == 3  # baseline + 2 selected, not the 16 of the product
    assert all(item.state is ArmState.PLANNED for item in manifest.arms)
    with pytest.raises(PhaseExpansionError, match="assign"):
        expand_phase(replace(spec, compositions=((("grounding", "locateanything"),),)))
    with pytest.raises(PhaseExpansionError, match="selected compositions"):
        expand_phase(replace(spec, compositions=()))


def test_a_phase_varies_only_the_kind_of_factor_it_is_named_for(
    validated: ValidatedReferenceSet,
) -> None:
    with pytest.raises(PhaseExpansionError, match="topology"):
        expand_phase(phase(validated, factors=(REFINEMENT_FACTOR,)))
    with pytest.raises(PhaseExpansionError, match="exactly one factor"):
        expand_phase(phase(validated, factors=(PROMPT_FACTOR, replace(PROMPT_FACTOR, name="x"))))
    assert len(PhaseKind) == 11


def test_expansion_imports_no_backend_and_no_model_runtime() -> None:
    script = (
        "import sys, json\n"
        "sys.path.insert(0, 'tests/evaluation')\n"
        "from contextmap.evaluation.experiment_phases import expand_phase\n"
        "import test_experiment_phases as t\n"
        "from experiment_builders import reference_with_semantics\n"
        "ref = reference_with_semantics()\n"
        "class V: manifest = ref\n"
        "expand_phase(t.phase(V))\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in {'torch', 'transformers'}"
        " or m.startswith('contextmap.visual_perception.backends.')]\n"
        "print(json.dumps(heavy))\n"
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, check=True
    )

    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


# ------------------------------------------------------------------ outcomes, persistence


def test_failed_oom_and_skipped_arms_are_preserved_with_blocked_ones(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    budget = factor(
        "visual_budget",
        VariationKind.CONFIGURATION,
        level("unbounded"),
        level("small", {f"{SEMANTIC}.qwen.max_pixels": 401408, f"{SEMANTIC}.qwen.min_pixels": 1}),
        level("large", {f"{SEMANTIC}.qwen.max_pixels": 4014080, f"{SEMANTIC}.qwen.min_pixels": 1}),
        level("scored", capabilities=("alphaclip.semantic_scoring",)),
    )
    manifest = expand_phase(phase(validated, kind=PhaseKind.VISUAL_BUDGET, factors=(budget,)))
    experiment = manifest.experiment
    assert experiment is not None
    values = {arm.arm_id: {"semantic.acceptable_claim_rate": 0.5} for arm in experiment.arms}
    executor, _ = make_executor(
        values,
        failures={
            _arm(manifest, "small").arm_id: ArmUnavailableError("no GPU was free"),
            _arm(manifest, "large").arm_id: RuntimeError("CUDA out of memory"),
        },
    )
    run = run_experiment(
        experiment,
        executor=executor,
        registry=REGISTRY,
        reference_set=validated,
        root=tmp_path / "run",
    )

    recorded = record_phase_outcomes(manifest, run)

    states = {dict(item.assignments)["visual_budget"]: item for item in recorded.arms}
    assert states["unbounded"].state is ArmState.EXECUTED
    assert states["small"].state is ArmState.SKIPPED
    assert "no GPU was free" in str(states["small"].reason)
    assert states["large"].state is ArmState.FAILED
    assert "CUDA out of memory" in str(states["large"].reason)
    assert states["scored"].state is ArmState.BLOCKED
    assert recorded.digest() != manifest.digest()
    with pytest.raises(PhaseExpansionError, match="another experiment"):
        record_phase_outcomes(replace(manifest, experiment=None), run)


def test_the_phase_manifest_persists_what_reproduces_every_arm(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = expand_phase(phase(validated))

    write_phase_manifest(tmp_path, manifest)

    document = read_verified_document(tmp_path / "phase.json")
    assert document["schema"] == "contextmap.experiment-phase/v1"
    assert (document["experiment_id"], document["phase_id"], document["kind"]) == (
        "perception-ablation",
        "semantic-prompt",
        "semantic_prompt",
    )
    assert document["base_configuration_digest"].startswith("sha256:")
    assert document["selection"]["reference_set"] == validated.manifest.identity().to_record()
    assert document["matrix_version"] == CAPABILITY_MATRIX.version
    assert (
        document["factors"][0]["levels"][1]["overrides"][0][0] == f"{SEMANTIC}.qwen.prompt_policy"
    )
    assert document["pinned"] == [["ingestion", SEQUENCE.to_record()]]
    assert document["targets"] == ["visual_perception"]
    assert {item["state"] for item in document["arms"]} == {"planned"}
    assert all(item["effective_config_digest"] for item in document["arms"])
    assert document["experiment"]["digest"] == manifest.experiment.digest()  # type: ignore[union-attr]
    with pytest.raises(FileExistsError):
        write_phase_manifest(tmp_path, manifest)

"""The frozen scenario resolves, unchanged, through the real runtime configuration.

The scenario is the only source of the canonical profile. These tests prove that the document it
yields is a valid runtime configuration: every backend it names exists in the runtime catalog, every
stage it requires is a stage of the runtime topology, and the optional stages stay off.
"""

from __future__ import annotations

import json
from pathlib import Path

from contextmap.evaluation import canonical_real_scenario, scenario_runtime_document
from contextmap.runtime import EffectiveConfig, check_selection, resolve_effective_config


def _resolve(directory: Path) -> EffectiveConfig:
    path = directory / "canonical.json"
    path.write_text(json.dumps(scenario_runtime_document(canonical_real_scenario())))
    return resolve_effective_config(files=[path])


def test_the_scenario_profile_is_a_complete_runtime_selection(tmp_path: Path) -> None:
    effective = _resolve(tmp_path)

    # Todo ponto de variação de um estágio habilitado tem backend: nada ficou por escolher.
    assert check_selection(effective.config) == ()


def test_the_runtime_topology_contains_every_required_stage_and_keeps_optional_ones_off(
    tmp_path: Path,
) -> None:
    scenario = canonical_real_scenario()
    document = _resolve(tmp_path).config.to_document()
    stages = document["pipeline"]["stages"]

    assert document["pipeline"]["preset"] == "canonical/1"
    assert {stage.stage_id for stage in scenario.stages} <= set(stages)
    assert all(stages[stage.stage_id] for stage in scenario.stages)
    # Point Representation existe no runtime e fica desligada: só entra por ablação.
    assert stages["point_representation"] is False
    # pose_ingestion (issue #555) é a bridge opcional de pose auxiliar: corridor-02 não a usa
    # ainda (pose continua fora de banda), então fica desligada como qualquer estágio opt-in.
    assert stages["pose_ingestion"] is False
    assert set(stages) - {stage.stage_id for stage in scenario.stages} == {
        "point_representation",
        "pose_ingestion",
    }


def test_the_backends_the_runtime_resolves_are_the_ones_the_scenario_froze(tmp_path: Path) -> None:
    scenario = canonical_real_scenario()
    components = _resolve(tmp_path).config.to_document()["components"]

    for stage in scenario.stages:
        for component in stage.components:
            capability, slot = component.component_id.split(".", 1)
            assert components[capability][slot]["backend"] == component.backend


def test_the_effective_configuration_digest_does_not_depend_on_where_the_file_lives(
    tmp_path: Path,
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    assert _resolve(tmp_path / "a").digest == _resolve(tmp_path / "b").digest

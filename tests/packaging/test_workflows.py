"""Structural guards for the CI and release workflows.

They check invariants that a release depends on (no publication without
verification, least privilege, one version per action) rather than YAML details.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is a development dependency")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
WORKFLOWS = {path.name: path for path in sorted(WORKFLOW_DIRECTORY.glob("*.yml"))}


def _load(name: str) -> dict[Any, Any]:
    return dict(yaml.safe_load(WORKFLOWS[name].read_text(encoding="utf-8")))


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # O YAML 1.1 do PyYAML lê a chave ``on`` como o booleano True.
    triggers = workflow.get("on", workflow.get(True))
    return dict(triggers) if isinstance(triggers, dict) else {}


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job.get("steps", []))


def _as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _action_references() -> list[tuple[str, str, str]]:
    references = []
    for name in WORKFLOWS:
        for job_name, job in _load(name)["jobs"].items():
            for use in [job.get("uses"), *(step.get("uses") for step in _steps(job))]:
                if use and not use.startswith("./"):
                    action, _, version = use.partition("@")
                    references.append((name, f"{job_name}", f"{action}@{version}"))
    return references


def test_each_action_is_used_at_a_single_version_across_workflows() -> None:
    versions: dict[str, set[str]] = {}
    for _, _, reference in _action_references():
        action, _, version = reference.partition("@")
        repository = "/".join(action.split("/")[:2])
        versions.setdefault(repository, set()).add(version)

    inconsistent = {action: sorted(found) for action, found in versions.items() if len(found) > 1}
    assert not inconsistent, f"actions pinned at more than one version: {inconsistent}"


def test_ci_keeps_the_quality_job_that_branch_protection_names() -> None:
    ci = _load("ci.yml")

    assert "quality" in ci["jobs"], "renaming CI / quality would orphan a required check"
    assert "workflow_call" in _triggers(ci), "the release workflow must be able to reuse CI"


def test_ci_tests_every_python_version_the_package_claims() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    claimed = {
        c.rsplit("::", 1)[1].strip()
        for c in project["classifiers"]
        if c.startswith("Programming Language :: Python :: 3.")
    }
    ci = _load("ci.yml")
    tested = {
        str(step["with"]["python-version"])
        for step in _steps(ci["jobs"]["quality"])
        if "python-version" in step.get("with", {})
    }
    tested |= {
        str(version)
        for version in ci["jobs"]["python-compatibility"]["strategy"]["matrix"]["python-version"]
    }

    assert claimed == tested


def test_release_runs_only_for_tags() -> None:
    triggers = _triggers(_load("release.yml"))

    assert set(triggers) == {"push"}
    assert set(triggers["push"]) == {"tags"}


def test_release_publishes_only_after_verification_and_ci() -> None:
    jobs = _load("release.yml")["jobs"]

    assert _as_list(jobs["checks"]["needs"]) == ["verify"]
    assert jobs["checks"]["uses"] == "./.github/workflows/ci.yml"
    assert set(_as_list(jobs["publish"]["needs"])) == {"verify", "checks"}


def test_release_verify_job_runs_the_tag_gate_on_full_history() -> None:
    steps = _steps(_load("release.yml")["jobs"]["verify"])
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout"))
    commands = " ".join(step.get("run", "") for step in steps)

    assert checkout["with"]["fetch-depth"] == 0
    assert ".github/scripts/verify_release_tag.sh" in commands


def test_only_the_publish_job_can_write_repository_contents() -> None:
    for name in WORKFLOWS:
        workflow = _load(name)
        assert workflow.get("permissions", {}).get("contents") != "write", name
        for job_name, job in workflow["jobs"].items():
            writes = job.get("permissions", {}).get("contents") == "write"
            assert writes == (name == "release.yml" and job_name == "publish"), (name, job_name)


def test_release_publishes_what_ci_built_and_tested() -> None:
    publish = _steps(_load("release.yml")["jobs"]["publish"])
    downloads = [s for s in publish if s.get("uses", "").startswith("actions/download-artifact")]
    uploads = [
        step
        for step in _steps(_load("ci.yml")["jobs"]["package"])
        if step.get("uses", "").startswith("actions/upload-artifact")
    ]

    assert [d["with"]["name"] for d in downloads] == ["dist"]
    assert [u["with"]["name"] for u in uploads] == ["dist"]


def test_workflows_use_no_secret_besides_the_automatic_token() -> None:
    for name, path in WORKFLOWS.items():
        secrets = set(re.findall(r"secrets\.([A-Za-z0-9_]+)", path.read_text(encoding="utf-8")))
        assert secrets <= {"GITHUB_TOKEN"}, (name, secrets)


def test_workflows_that_run_with_base_permissions_never_check_out_pull_request_code() -> None:
    for name in WORKFLOWS:
        workflow = _load(name)
        if "pull_request_target" not in _triggers(workflow):
            continue
        uses = [step.get("uses", "") for job in workflow["jobs"].values() for step in _steps(job)]
        assert not [use for use in uses if use.startswith("actions/checkout")], name


@pytest.mark.parametrize("job_name", ["package", "lightweight-install"])
def test_install_smoke_jobs_do_not_cache_dependencies(job_name: str) -> None:
    # Um cache de pip poderia esconder uma dependência que o pacote não declara.
    for step in _steps(_load("ci.yml")["jobs"][job_name]):
        assert "cache" not in step.get("with", {}), step

"""Integrity of the light reproduction bundles versioned under ``experiments/``.

A bundle keeps what an experiment needs to be reviewed and repeated without the dataset or the
large artifacts: a manifest with identities and full SHA-256 values, the selection, the drivers
and the final reports. These tests check each bundle against its own manifest and never need
the dataset.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPOSITORY_ROOT / "experiments"
SEMANTIC_FUSION_BUNDLE = EXPERIMENTS / "semantic-fusion-corridor-02-20260921"
EVALUATION_DOC = (
    REPOSITORY_ROOT / "src" / "contextmap" / "evaluation" / "docs" / "semantic_fusion.md"
)

MAX_BUNDLE_BYTES = 1_000_000
"""A bundle is light by definition: the dataset and the large artifacts stay out of Git."""

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ABBREVIATED_HASH = re.compile(r"[0-9a-f]{6,}(?:…|\.\.\.)")
_PERSONAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\Users\\|/root/")
_SECRET = re.compile(
    r"\b(?:hf|sk|ghp|gho)_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY"
)


def _bundles() -> list[Path]:
    return sorted(path.parent for path in EXPERIMENTS.glob("*/manifest.json"))


def _manifest(bundle: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    return record


def _payload_files(bundle: Path) -> list[Path]:
    # O bytecode que rodar um driver deixa em __pycache__ não faz parte do bundle (o Git o ignora).
    return sorted(
        path
        for path in bundle.rglob("*")
        if path.is_file() and "__pycache__" not in path.relative_to(bundle).parts
    )


def _entry(bundle: Path, relative: str) -> dict[str, Any]:
    entries = [item for item in _manifest(bundle)["files"] if item["path"] == relative]
    assert len(entries) == 1, f"{relative} must be listed exactly once in the manifest"
    entry: dict[str, Any] = entries[0]
    return entry


def test_the_real_semantic_fusion_run_has_a_reproduction_bundle() -> None:
    assert (SEMANTIC_FUSION_BUNDLE / "manifest.json").is_file()
    assert (SEMANTIC_FUSION_BUNDLE / "README.md").is_file()
    assert (SEMANTIC_FUSION_BUNDLE / "selection.json").is_file()


@pytest.mark.parametrize("bundle", _bundles(), ids=lambda path: path.name)
class TestEveryBundle:
    def test_every_listed_file_matches_its_full_sha256_and_size(self, bundle: Path) -> None:
        for entry in _manifest(bundle)["files"]:
            path = bundle / entry["path"]
            assert path.is_file(), f"{entry['path']} is listed but missing"
            assert _SHA256.fullmatch(entry["sha256"]), f"{entry['path']}: not a full SHA-256"
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
            assert len(data) == entry["size_bytes"], entry["path"]

    def test_every_file_of_the_bundle_is_listed(self, bundle: Path) -> None:
        listed = {entry["path"] for entry in _manifest(bundle)["files"]}
        present = {
            path.relative_to(bundle).as_posix()
            for path in _payload_files(bundle)
            if path.name != "manifest.json"
        }

        assert present == listed

    def test_a_file_that_ran_records_the_full_hash_of_what_was_executed(self, bundle: Path) -> None:
        for entry in _manifest(bundle)["files"]:
            if entry["origin"] == "authored":
                assert "executed_sha256" not in entry, entry["path"]
                continue
            assert entry["origin"] == "executed", entry["path"]
            assert _SHA256.fullmatch(entry["executed_sha256"]), f"{entry['path']}: executed hash"
            # Uma transformação (por exemplo, remover caminhos pessoais) tem de ser nomeada; um
            # arquivo idêntico ao executado não tem transformação.
            changed = entry["executed_sha256"] != entry["sha256"]
            assert changed == bool(entry.get("transformation")), entry["path"]

    def test_the_manifest_never_abbreviates_a_hash(self, bundle: Path) -> None:
        text = (bundle / "manifest.json").read_text(encoding="utf-8")

        assert not _ABBREVIATED_HASH.search(text)

    def test_the_bundle_is_light(self, bundle: Path) -> None:
        total = sum(path.stat().st_size for path in _payload_files(bundle))

        assert total < MAX_BUNDLE_BYTES

    def test_the_bundle_has_no_personal_path_or_secret(self, bundle: Path) -> None:
        for path in _payload_files(bundle):
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(bundle)
            assert not _PERSONAL_PATH.search(text), f"{relative}: personal absolute path"
            assert not _SECRET.search(text), f"{relative}: secret-looking string"


def test_the_manifest_selection_is_the_one_in_selection_json() -> None:
    manifest = _manifest(SEMANTIC_FUSION_BUNDLE)["selection"]
    selection = json.loads((SEMANTIC_FUSION_BUNDLE / "selection.json").read_text(encoding="utf-8"))

    assert manifest["selection_identity"] == selection["selection_identity"]
    assert manifest["frames"] == [
        item["observation_id"] for item in selection["images"]["selected"]
    ]
    assert len(manifest["frames"]) == 20


def test_the_manifest_lists_the_seven_arms_of_the_recorded_report() -> None:
    manifest = _manifest(SEMANTIC_FUSION_BUNDLE)
    report = json.loads(
        (SEMANTIC_FUSION_BUNDLE / "reports/semantic_fusion/report.json").read_text(encoding="utf-8")
    )
    arms = {arm["arm_id"]: arm for arm in manifest["configuration"]["arms"]}

    assert set(arms) == set(report["arms"])
    assert len(arms) == 7
    for name, arm in report["arms"].items():
        lineage = arm["evaluation"]["lineage"]
        assert arms[name]["run_id"] == lineage["run_id"]
        assert arms[name]["fusion_policy_id"] == lineage["fusion_policy_id"]
        assert (
            arms[name]["fusion_configuration_fingerprint"]
            == lineage["fusion_configuration_fingerprint"]
        )


def test_the_evaluation_doc_quotes_the_full_hashes_of_the_bundle() -> None:
    doc = EVALUATION_DOC.read_text(encoding="utf-8")

    assert "experiments/semantic-fusion-corridor-02-20260921" in doc
    assert not _ABBREVIATED_HASH.search(doc)
    for relative in (
        "reports/semantic_fusion/report.json",
        "reports/semantic_fusion/visual_consistency.json",
        "scripts/common.py",
        "scripts/s04_fusion.py",
        "selection.json",
    ):
        entry = _entry(SEMANTIC_FUSION_BUNDLE, relative)
        assert entry["sha256"] in doc, f"{relative}: versioned SHA-256 is not quoted in full"
        assert entry["executed_sha256"] in doc, (
            f"{relative}: executed SHA-256 is not quoted in full"
        )

"""The v0 layout of a ContextMapArtifact directory (issue #155)."""

import pytest

from contextmap.artifact.serialization.errors import ManifestError
from contextmap.artifact.serialization.layout import (
    ARTIFACT_TYPE,
    CONTRACTUAL_FILES,
    FORMAT_VERSION,
    MANIFEST,
    README,
    SUPPORTED_FORMAT_VERSIONS,
    require_contractual_path,
)


def test_the_layout_names_the_files_the_issue_lists() -> None:
    assert ARTIFACT_TYPE == "context_map"
    assert set(CONTRACTUAL_FILES) == {
        "map-metadata.json",
        "geometry/geometry-reference.json",
        "entities/entities.jsonl",
        "relations/relations.jsonl",
        "lineage/lineage.json",
        "indexes/entity-index.jsonl",
        "indexes/relation-index.jsonl",
        "indexes/entity-relation-index.jsonl",
    }
    assert MANIFEST == "manifest.json"
    assert README == "README.md"


def test_layout_files_are_unique_contractual_paths_and_the_manifest_is_not_one_of_them() -> None:
    assert len(set(CONTRACTUAL_FILES)) == len(CONTRACTUAL_FILES)
    for path in CONTRACTUAL_FILES:
        require_contractual_path(path)
    assert MANIFEST not in CONTRACTUAL_FILES
    assert README not in CONTRACTUAL_FILES


def test_only_the_current_format_version_is_supported() -> None:
    assert frozenset({FORMAT_VERSION}) == SUPPORTED_FORMAT_VERSIONS


@pytest.mark.parametrize(
    "path",
    ["", "/etc/passwd", "../up.json", "a/../../up.json", "./x", "a\\b.json", "a//b.json"],
)
def test_contractual_paths_are_plain_relative_posix_paths(path: str) -> None:
    with pytest.raises(ManifestError, match="relative"):
        require_contractual_path(path)


@pytest.mark.parametrize("path", ["debug/x.json", "debug", "debug/deep/x.bin"])
def test_debug_is_never_a_contractual_path(path: str) -> None:
    with pytest.raises(ManifestError, match="debug"):
        require_contractual_path(path)

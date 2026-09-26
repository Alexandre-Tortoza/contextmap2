"""A real ContextMapArtifact on disk for the CLI tests, written by the artifact tests' builders.

The map and its upstream artifacts (a real geometric map, Entity Resolution and Spatial Relations
runs, and one optional evidence run) come from the builders the artifact capability's own
validation tests use, so the CLI is exercised against exactly the artifacts that validator is
tested on. Those builders are imported by base name, which needs their directory on
``sys.path`` (the serialization builders reach Entity Resolution's the same way).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ARTIFACT_TESTS = str(Path(__file__).resolve().parents[1] / "artifact")
if _ARTIFACT_TESTS not in sys.path:
    sys.path.insert(0, _ARTIFACT_TESTS)

from context_map_serialization_builders import World, make_world, write_artifact  # noqa: E402

__all__ = ["World", "written_context_map"]


def written_context_map(root: Path) -> tuple[World, Path]:
    """Write the upstream world under ``root`` and a populated ContextMapArtifact over it.

    Args:
        root: An empty directory that receives every artifact.

    Returns:
        The upstream world and the directory of the written ContextMapArtifact, whose manifest
        locates each upstream artifact by a relative hint.
    """
    world = make_world(root)
    return world, write_artifact(world)[0]

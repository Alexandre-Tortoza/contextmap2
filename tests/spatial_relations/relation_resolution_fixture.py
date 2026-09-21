"""A real Entity Resolution run for the wiring tests, built by Entity Resolution's own fixtures.

Spatial Relations reads a resolution run through Entity Resolution's public reader. The tests use
the run that Entity Resolution's fixtures write with its real writer, and rebuild the one geometric
map its entities were summarized from, so the resolved geometry can be read back exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

for _directory in ("semantic_fusion", "semantic_mapping", "entity_resolution"):
    _path = str(Path(__file__).resolve().parents[1] / _directory)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from resolution_entity_builders import (  # noqa: E402
    box_corners,
    scene_source,
    stable_index_base,
)
from resolution_run_fixtures import (  # noqa: E402
    LINEAGE as ER_LINEAGE,
)
from resolution_run_fixtures import (  # noqa: E402
    RUN as ER_RUN,
)
from resolution_run_fixtures import (  # noqa: E402
    RunInputs,
    build_inputs,
)
from resolution_run_fixtures import (  # noqa: E402
    write_run as write_er_run,
)

from contextmap.geometric_mapping import GeometrySource  # noqa: E402
from contextmap.shared import Vector3  # noqa: E402

__all__ = [
    "ER_LINEAGE",
    "ER_RUN",
    "RunInputs",
    "build_inputs",
    "er_scene_source",
    "write_er_run",
]

_NAMES = "abcdef"


def er_scene_source() -> GeometrySource:
    """The geometric map of the fixture's six entities: a box of eight corners each, 1 m apart."""
    points: dict[int, Vector3] = {}
    for index, name in enumerate(_NAMES):
        base = stable_index_base(name)
        corners = box_corners((index * 1.0, 0.0, 0.0), 0.5)
        points.update(zip(range(base, base + len(corners)), corners, strict=True))
    return scene_source(points)

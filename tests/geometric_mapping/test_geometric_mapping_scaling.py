"""Scaling gates for the geometric map: what a query reads, what a run holds, what it persists.

The map is the largest artifact of the chain (1.05 GB for corridor-02, measured in
``experiments/e2e-real-canonical-run-20260923/resource-profile.md``), and every later stage reads
it. Small fixtures cannot show a cost that follows the size of the map, so these gates grow the map
around a fixed piece of it and check that the work stays with that piece:

* a box query reads the scans its box reaches, however long the corridor is, through both
  boundaries (``query_bounds`` and ``iter_blocks``);
* accumulation keeps one transformed scan alive and hands it to the sink before the next one is
  transformed, whatever the number of scans;
* only the payload, at exactly one 64-byte record per point, grows with the points of a scan:
  the rest of the artifact is per scan.

That the pruned query returns exactly what a full filter returns is already pinned by
``test_geometric_mapping_spatial_access.py`` and ``test_geometric_mapping_bulk_geometry.py``. CI
keeps to counters, retention invariants and broad envelopes; wall-clock figures belong to
``src/contextmap/geometric_mapping/docs/spatial-access.md``.
"""

from __future__ import annotations

import random
import weakref
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import BinaryIO, cast

import pytest
from input_builders import MS, assemble_plan, make_calibration, make_trajectory, rigid
from lidar_builders import make_scan
from map_builders import accumulate, make_scans, open_geometry

import contextmap.geometric_mapping.accumulation as accumulation
from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    GeometryInputPlan,
    MapDebugLevel,
    MapId,
    PackedGeometry,
    TransformedScan,
    accumulate_plan,
)
from contextmap.geometric_mapping.geometry_storage import PACKED_POINT
from contextmap.ingestion import FrameId
from contextmap.shared import Vector3

MAP = FrameId("map")


def _cloud(count: int, *, seed: int = 7) -> tuple[Vector3, ...]:
    """A cloud around the sensor: 4 m along x, 2 m along y and 2 m up."""
    rng = random.Random(seed)
    return tuple(
        (rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 1.0), rng.uniform(0.0, 2.0))
        for _ in range(count)
    )


def _plan(scan_count: int, *, points: Sequence[Vector3]) -> GeometryInputPlan:
    """``scan_count`` scans of the same cloud, one meter apart along +x."""
    calibration = make_calibration((rigid("body", "lidar", (0.5, 0.0, 0.25)),))
    return assemble_plan(
        [
            make_scan(f"scan-{index:04d}", time_ns=index * 100 * MS, points=points)
            for index in range(scan_count)
        ],
        calibration=calibration,
        trajectory=make_trajectory(count=max(5, scan_count), calibration=calibration),
    )


# --- Complexity gate: a query reads what its box reaches, not the map ---------------------------

POINTS_PER_SCAN = 20
# Um metro do corredor perto do começo: só os scans cujas nuvens de 4 m alcançam x em [3, 3,5].
LOCAL_BOX = Bounds3D(frame_id=MAP, minimum_m=(3.0, -1.0, 0.0), maximum_m=(3.5, 1.0, 2.5))


def _corridor(scan_count: int) -> PackedGeometry:
    return open_geometry(*accumulate(make_scans(scan_count, points=_cloud(POINTS_PER_SCAN))))


def _counting_rows_read(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the rows every query reads, through the scan pruning both boundaries share."""
    rows = [0]
    select = PackedGeometry._selected_ranges

    def counted(self: PackedGeometry, bounds: Bounds3D | None) -> Iterator[tuple[int, int, bool]]:
        for first, stop, wholly_inside in select(self, bounds):
            rows[0] += stop - first
            yield first, stop, wholly_inside

    monkeypatch.setattr(PackedGeometry, "_selected_ranges", counted)
    return rows


def _rows_read_by_each_boundary(
    geometry: PackedGeometry, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, int, int]:
    """Rows read by ``query_bounds`` and ``iter_blocks`` for the local box, and points found."""
    rows = _counting_rows_read(monkeypatch)
    found = len(list(geometry.query_bounds(LOCAL_BOX)))
    by_query, rows[0] = rows[0], 0
    blocks = list(geometry.iter_blocks(bounds=LOCAL_BOX))
    by_blocks = rows[0]
    monkeypatch.undo()
    assert sum(len(block) for block in blocks) == found
    return by_query, by_blocks, found


def test_growing_the_corridor_does_not_grow_what_a_local_box_query_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = {
        scan_count: _rows_read_by_each_boundary(_corridor(scan_count), monkeypatch)
        for scan_count in (10, 100, 1000)
    }

    # O mapa cresce 100x; o que cada fronteira lê e encontra na caixa, não.
    assert len(set(measured.values())) == 1
    by_query, by_blocks, found = measured[10]
    assert by_query == by_blocks
    assert 0 < found <= by_query <= 6 * POINTS_PER_SCAN


def test_a_query_without_a_box_reads_the_whole_map_which_is_what_the_index_avoids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _corridor(100)
    rows = _counting_rows_read(monkeypatch)

    blocks = list(geometry.iter_blocks())

    assert rows[0] == sum(len(block) for block in blocks) == 100 * POINTS_PER_SCAN


# --- Retention gate: accumulation streams one scan at a time ------------------------------------


class _StreamWitness:
    """Watches the transformed scans a run produces and the sink it writes them to."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.bytes_written = 0
        self.alive_at_each_write: list[int] = []
        self.written_when_each_scan_arrives: list[int] = []
        self._produced: list[weakref.ref[TransformedScan]] = []
        transform = accumulation.transform_scans

        def watched(plan: GeometryInputPlan) -> Iterator[TransformedScan]:
            for scan in transform(plan):
                self.written_when_each_scan_arrives.append(self.bytes_written)
                self._produced.append(weakref.ref(scan))
                yield scan

        monkeypatch.setattr(accumulation, "transform_scans", watched)

    @property
    def produced(self) -> int:
        return len(self._produced)

    def alive(self) -> int:
        return sum(ref() is not None for ref in self._produced)

    def write(self, payload: bytes) -> int:
        self.alive_at_each_write.append(self.alive())
        self.bytes_written += len(payload)
        return len(payload)


@pytest.mark.parametrize("scan_count", [1, 4, 16])
def test_accumulation_holds_one_transformed_scan_and_writes_it_before_the_next(
    monkeypatch: pytest.MonkeyPatch, scan_count: int
) -> None:
    plan = _plan(scan_count, points=_cloud(POINTS_PER_SCAN))
    witness = _StreamWitness(monkeypatch)

    accumulated = accumulate_plan(
        plan,
        map_id=MapId("map-0001"),
        sink=cast(BinaryIO, witness),
        aggregation=None,
        configuration_fingerprint=None,
        code_version=None,
    )

    scan_bytes = POINTS_PER_SCAN * PACKED_POINT.size
    assert witness.produced == scan_count
    # Enquanto o scan K é escrito, ele é o único vivo: nada de 0..K-1 sobrevive.
    assert witness.alive_at_each_write == [1] * scan_count
    # E o scan K-1 já foi para o sink quando o K é transformado: nada fica acumulado em memória.
    assert witness.written_when_each_scan_arrives == [k * scan_bytes for k in range(scan_count)]
    assert witness.bytes_written == scan_count * scan_bytes
    assert witness.alive() == 0
    assert len(accumulated.scans) == scan_count


# --- Storage gate: only the payload follows the points ------------------------------------------

PAYLOAD = "outputs/geometry.bin"
STORAGE_SCANS = 12


def _persist(workspace: Path, points_per_scan: int) -> dict[str, int]:
    """Persist a run with every piece of debug evidence and return the size of each file."""
    run_dir = workspace / f"points-{points_per_scan}"
    GeometricMapArtifactWriter(
        output_dir=run_dir,
        sequence_name="corridor",
        run_id=GeometricMapRunId(f"points-{points_per_scan}"),
        run_index=1,
        debug_level=MapDebugLevel.FULL,
    ).finalize(
        plan=_plan(STORAGE_SCANS, points=_cloud(points_per_scan)),
        aggregation=None,
        code_version="test",
    )
    return {
        path.relative_to(run_dir).as_posix(): path.stat().st_size
        for path in run_dir.rglob("*")
        if path.is_file()
    }


def test_only_the_payload_grows_with_the_points_of_each_scan(tmp_path: Path) -> None:
    sparse = _persist(tmp_path, 10)
    dense = _persist(tmp_path, 1000)

    assert dense.keys() == sparse.keys()
    # Um registro de 64 bytes por ponto, e nada mais por ponto em lugar nenhum do payload.
    assert sparse[PAYLOAD] == STORAGE_SCANS * 10 * PACKED_POINT.size
    assert dense[PAYLOAD] == STORAGE_SCANS * 1000 * PACKED_POINT.size

    def beyond_payload(files: dict[str, int]) -> int:
        return sum(size for path, size in files.items() if path != PAYLOAD)

    # 100x os pontos; índice de origem, metadados, métricas, manifest e debug são por scan.
    assert beyond_payload(dense) <= 1.25 * beyond_payload(sparse)

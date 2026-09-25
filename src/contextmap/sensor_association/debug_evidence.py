"""Human debug evidence of a Sensor Association run.

Debug files help diagnose range, visibility and spatial-sampling effects without rerunning:
per-point samples, distributions of the associated support, a state-colored projection overlay
and the coordinates the dense features were sampled at. They are written next to the run but
are **never contractual**: they are not inventoried, removing them cannot invalidate the run,
and no downstream stage may depend on them.

Everything is plain CSV or JSON, plus a small PNG overlay built with the standard library, so
inspecting a run needs no visualization stack.
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import TYPE_CHECKING, Any

from contextmap.sensor_association.service import FrameAssociation, SensorAssociationOutcome
from contextmap.shared import AtomicRunDirectory

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Cores do overlay por estado: verde associado, amarelo visível sem região, vermelho ocluído,
# cinza fora do suporte válido, sobre fundo escuro.
_BACKGROUND = (20, 20, 20)
_COLORS = {
    "outside_valid_support": (128, 128, 128),
    "occluded": (220, 50, 50),
    "visible_unassigned": (230, 200, 40),
    "associated": (40, 180, 60),
}
_DRAW_ORDER = ("outside_valid_support", "occluded", "visible_unassigned", "associated")
_HISTOGRAM_BINS = 10
_IMAGE_REGION_GRID = 3


def write_standard_debug(run: AtomicRunDirectory, outcome: SensorAssociationOutcome) -> None:
    """Write per-point samples and the distributions of every frame."""
    for frame in outcome.frames:
        directory = f"debug/frames/{frame.source_observation_id}"
        states = _point_states(frame)
        run.write_text(f"{directory}/samples.csv", _samples_csv(frame, states), contractual=False)
        run.write_text(
            f"{directory}/distributions.json",
            json.dumps(_distributions(frame, states), indent=2, sort_keys=True) + "\n",
            contractual=False,
        )


def write_full_debug(run: AtomicRunDirectory, outcome: SensorAssociationOutcome) -> None:
    """Write the overlays, the dense sampling coordinates and the feature sources."""
    sources: list[dict[str, Any]] = []
    for frame in outcome.frames:
        directory = f"debug/frames/{frame.source_observation_id}"
        run.write_bytes(
            f"{directory}/overlay.png", _overlay_png(frame, _point_states(frame)), contractual=False
        )
        for channel_id, samples in frame.dense_samples.items():
            run.write_text(
                f"{directory}/dense-sampling-{channel_id}.csv",
                _dense_csv(samples, frame.resolution.frame.global_indices),
                contractual=False,
            )
            sources.append(
                {
                    "source_observation_id": str(frame.source_observation_id),
                    "channel_id": channel_id,
                    **samples.provenance.to_record(),
                }
            )
    if sources:
        run.write_text(
            "debug/feature-sources.json",
            json.dumps(sources, indent=2, sort_keys=True) + "\n",
            contractual=False,
        )


def _point_states(frame: FrameAssociation) -> dict[int, str]:
    """State name of every candidate row that reached the prepared image."""
    import numpy as np

    projection = frame.resolution.frame
    assigned = ~frame.membership.visible_unassigned
    states: dict[int, str] = {}
    for index in np.flatnonzero(projection.in_prepared_image).tolist():
        if frame.resolution.outside_valid_support[index]:
            states[index] = "outside_valid_support"
        elif frame.resolution.occluded[index]:
            states[index] = "occluded"
        elif assigned[index]:
            states[index] = "associated"
        else:
            states[index] = "visible_unassigned"
    return states


def _samples_csv(frame: FrameAssociation, states: dict[int, str]) -> str:
    projection = frame.resolution.frame
    resolution = frame.resolution
    lines = ["geometry_index,state,prepared_u,prepared_v,depth_m,support_depth_m,regions"]
    for row, state in states.items():
        u, v = projection.prepared_pixels[row]
        regions = ";".join(str(region) for region in frame.membership.regions_of(row))
        # A coluna nomeia a geometria persistente, não a linha do array de candidatos.
        index = int(projection.global_indices[row])
        lines.append(
            f"{index},{state},{float(u)!r},{float(v)!r},{float(resolution.depth_m[row])!r},"
            f"{float(resolution.support_depth_m[row])!r},{regions}"
        )
    return "\n".join(lines) + "\n"


def _distributions(frame: FrameAssociation, states: dict[int, str]) -> dict[str, Any]:
    import numpy as np

    projection = frame.resolution.frame
    associated = np.array([i for i, state in states.items() if state == "associated"], dtype=int)
    counts = {name: sum(1 for state in states.values() if state == name) for name in _DRAW_ORDER}
    record: dict[str, Any] = {
        "state_counts": {
            **{k.value: v for k, v in frame.resolution.state_counts().items()},
            "visible": frame.resolution.visible_count,
        },
        "prepared_image_state_counts": counts,
        "associated_depth_m": None,
        "associated_depth_histogram": None,
        "associated_by_image_region": [[0] * _IMAGE_REGION_GRID for _ in range(_IMAGE_REGION_GRID)],
        "support_density_by_region": [
            {
                "region_id": str(region.region_id),
                "associated_count": region.associated_count,
                "mask_area_px": region.mask_area_px,
                "density": region.associated_count / region.mask_area_px,
            }
            for region in frame.membership.regions
        ],
    }
    if associated.size:
        depth = frame.resolution.depth_m[associated]
        histogram, edges = np.histogram(depth, bins=_HISTOGRAM_BINS)
        record["associated_depth_m"] = {
            "count": int(depth.size),
            "minimum": float(depth.min()),
            "median": float(np.median(depth)),
            "maximum": float(depth.max()),
        }
        record["associated_depth_histogram"] = {
            "bin_edges_m": [float(edge) for edge in edges],
            "counts": [int(count) for count in histogram],
        }
        width, height = projection.image_transform.prepared_size
        pixels = projection.prepared_pixels[associated] + 0.5
        columns = np.minimum((pixels[:, 0] * _IMAGE_REGION_GRID // width).astype(int), 2)
        rows = np.minimum((pixels[:, 1] * _IMAGE_REGION_GRID // height).astype(int), 2)
        grid = record["associated_by_image_region"]
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            grid[row][column] += 1
    return record


def _overlay_png(frame: FrameAssociation, states: dict[int, str]) -> bytes:
    import numpy as np

    projection = frame.resolution.frame
    width, height = projection.image_transform.prepared_size
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = _BACKGROUND
    for name in _DRAW_ORDER:
        indices = [i for i, state in states.items() if state == name]
        if not indices:
            continue
        # O índice do pixel de centro c é floor(c + 0.5).
        pixels = np.floor(projection.prepared_pixels[indices] + 0.5).astype(int)
        canvas[pixels[:, 1], pixels[:, 0]] = _COLORS[name]
    return _encode_png(canvas)


def _dense_csv(samples: Any, global_indices: NDArray[Any]) -> str:
    terms = samples.cell_rows.shape[1]
    header = "geometry_index,sampled," + ",".join(
        f"cell{t}_row,cell{t}_col,cell{t}_weight" for t in range(terms)
    )
    lines = [header]
    for slot, index in enumerate(global_indices[samples.eligible_indices].tolist()):
        cells = ",".join(
            f"{int(samples.cell_rows[slot, t])},{int(samples.cell_cols[slot, t])},"
            f"{float(samples.weights[slot, t])!r}"
            for t in range(terms)
        )
        lines.append(f"{index},{int(samples.sampled[slot])},{cells}")
    return "\n".join(lines) + "\n"


def _encode_png(rgb: NDArray[Any]) -> bytes:
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )

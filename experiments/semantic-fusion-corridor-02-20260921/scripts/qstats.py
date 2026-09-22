"""Marginal distribution of the real ObservationQuality components (upstream property, not an outcome)."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from contextmap.sensor_association import SensorAssociationRunReader

root = Path(sys.argv[1])
values: dict[str, list[float]] = {}
unavailable: dict[str, int] = {}
geometry_counts: list[int] = []
for directory in sorted(root.glob("runs/sensor-association/corridor-02/run-*")):
    reader = SensorAssociationRunReader(directory)
    n = 0
    for observation in reader.observations():
        quality = reader.quality(observation.spatial_observation_id)
        n += 1
        geometry_counts.append(len(observation.geometry_support))
        if quality.support_depth_m is not None:
            values.setdefault("depth_median_m", []).append(quality.support_depth_m.median)
        if quality.visible_share is not None:
            values.setdefault("visible_share", []).append(quality.visible_share)
        values.setdefault("density", []).append(quality.support_density_per_mask_pixel)
        if quality.border_distance_px is not None:
            values.setdefault("border_min_px", []).append(quality.border_distance_px.minimum)
            values.setdefault("border_median_px", []).append(quality.border_distance_px.median)
        if quality.support_off_axis_angle_rad is not None:
            values.setdefault("off_axis_median_rad", []).append(quality.support_off_axis_angle_rad.median)
        values.setdefault("pose_dt_ms", []).append(quality.pose_ref.time_delta_ns / 1e6)
        for component, reason in quality.unavailable.items():
            unavailable[component.value] = unavailable.get(component.value, 0) + 1
    print(directory.name, "observations", n)


def q(xs: list[float]) -> str:
    xs = sorted(xs)
    pick = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]
    return f"n={len(xs)} min={xs[0]:.4g} p10={pick(.1):.4g} p25={pick(.25):.4g} med={statistics.median(xs):.4g} p75={pick(.75):.4g} p90={pick(.9):.4g} max={xs[-1]:.4g}"


for name, xs in sorted(values.items()):
    print(f"{name}: {q(xs)}")
print("geometry per observation:", q([float(x) for x in geometry_counts]))
print("unavailable components:", unavailable)

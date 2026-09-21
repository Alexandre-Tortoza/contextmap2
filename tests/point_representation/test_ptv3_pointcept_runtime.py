"""Deterministic parts of the real PTv3 runtime: no torch, no CUDA, no checkpoint download.

The forward pass itself needs a GPU, spconv and the Pointcept code, so it is exercised by the
opt-in test in ``test_ptv3_pointcept_real.py``. What is checked here is everything around it that
must hold on any machine: voxelization, pooling, checkpoint identity, compatibility with the
configuration and explicit failure when a dependency is missing.
"""

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pointrep_ptv3_fakes import make_config

from contextmap.point_representation.backends import ptv3_pointcept
from contextmap.point_representation.backends.ptv3 import PTv3RuntimeUnavailableError
from contextmap.point_representation.backends.ptv3_pointcept import (
    NUSCENES_SEMSEG_PTV3M1_BASE,
    PointceptPTv3Runtime,
    check_runtime_compatibility,
    extract_backbone_state,
    pool_voxel_features,
    verify_checkpoint_sha256,
    voxelize,
)

GRID_M = 0.05
CORNER_POINTS = [(0.0, 0.0, 0.0), (0.01, 0.01, 0.01), (0.20, 0.0, 0.0)]


def config_for_nuscenes(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "output_dimension": NUSCENES_SEMSEG_PTV3M1_BASE.output_channels,
        "padding_channels": NUSCENES_SEMSEG_PTV3M1_BASE.in_channels - 3,
    }
    values.update(overrides)
    return make_config(**values)


# --- Optional dependency isolation --------------------------------------------------


def test_the_real_runtime_module_needs_no_model_library_or_numpy_to_import() -> None:
    code = (
        "import sys;"
        "import contextmap.point_representation.backends.ptv3_pointcept as m;"
        "from pathlib import Path;"
        "runtime = m.PointceptPTv3Runtime("
        "pointcept_root=Path('/nonexistent/pointcept'),"
        "checkpoint_path=Path('/nonexistent/model.pth'),"
        "backbone=m.NUSCENES_SEMSEG_PTV3M1_BASE);"
        "bad = [n for n in ('numpy', 'torch', 'spconv', 'torch_scatter', 'pointcept') "
        "if n in sys.modules];"
        "assert not bad, bad"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


# --- Backbone description --------------------------------------------------------------


def test_the_nuscenes_backbone_declares_its_input_and_output_channels() -> None:
    assert NUSCENES_SEMSEG_PTV3M1_BASE.in_channels == 4
    assert NUSCENES_SEMSEG_PTV3M1_BASE.output_channels == 64


def test_inference_is_configured_deterministically_and_without_a_classification_head() -> None:
    kwargs = NUSCENES_SEMSEG_PTV3M1_BASE.constructor_kwargs()

    assert kwargs["shuffle_orders"] is False
    assert kwargs["enable_flash"] is False
    assert kwargs["drop_path"] == 0.0
    assert kwargs["in_channels"] == 4
    assert not {"num_classes", "seg_head", "criteria"} & set(kwargs)


# --- Compatibility of the configuration with the checkpoint ---------------------------------


def test_a_configuration_matching_the_backbone_is_compatible() -> None:
    check_runtime_compatibility(config_for_nuscenes(), NUSCENES_SEMSEG_PTV3M1_BASE)


def test_an_output_dimension_the_backbone_cannot_produce_is_an_incompatible_checkpoint() -> None:
    with pytest.raises(PTv3RuntimeUnavailableError, match="output_dimension"):
        check_runtime_compatibility(
            config_for_nuscenes(output_dimension=32), NUSCENES_SEMSEG_PTV3M1_BASE
        )


def test_padding_that_does_not_complete_the_checkpoint_input_is_incompatible() -> None:
    with pytest.raises(PTv3RuntimeUnavailableError, match="padding_channels"):
        check_runtime_compatibility(
            config_for_nuscenes(padding_channels=0), NUSCENES_SEMSEG_PTV3M1_BASE
        )


def test_sparse_convolution_needs_a_cuda_device_so_cpu_is_refused() -> None:
    with pytest.raises(PTv3RuntimeUnavailableError, match="CUDA"):
        check_runtime_compatibility(config_for_nuscenes(device="cpu"), NUSCENES_SEMSEG_PTV3M1_BASE)


# --- Voxelization ---------------------------------------------------------------------


def test_points_in_the_same_cell_share_one_voxel_at_their_centroid() -> None:
    voxels = voxelize(CORNER_POINTS, GRID_M)

    assert voxels.counts.tolist() == [2, 1]
    assert voxels.inverse.tolist() == [0, 0, 1]
    assert voxels.centroid_m[0].tolist() == pytest.approx([0.005, 0.005, 0.005])
    assert voxels.centroid_m[1].tolist() == pytest.approx([0.20, 0.0, 0.0])


def test_grid_coordinates_are_non_negative_integers_anchored_at_the_minimum_corner() -> None:
    # Pontos longe das fronteiras das células, para não depender de arredondamento.
    voxels = voxelize([(-1.0, 2.0, 0.5), (-0.88, 2.0, 0.5), (-0.47, 2.32, 0.62)], GRID_M)

    assert voxels.grid_coord.dtype == np.int64
    assert voxels.grid_coord.min(axis=0).tolist() == [0, 0, 0]
    assert voxels.grid_coord.tolist() == [[0, 0, 0], [2, 0, 0], [10, 6, 2]]


def test_voxelization_does_not_depend_on_the_order_of_the_points() -> None:
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.5, 0.5, size=(200, 3))

    forward = voxelize([tuple(p) for p in points], GRID_M)
    backward = voxelize([tuple(p) for p in points[::-1]], GRID_M)

    assert np.array_equal(forward.grid_coord, backward.grid_coord)
    assert np.allclose(forward.centroid_m, backward.centroid_m)
    assert np.array_equal(forward.counts, backward.counts)
    assert np.array_equal(forward.inverse[::-1], backward.inverse)


def test_every_point_maps_back_to_the_voxel_that_contains_it() -> None:
    rng = np.random.default_rng(5)
    points = rng.uniform(-0.5, 0.5, size=(120, 3))

    voxels = voxelize([tuple(p) for p in points], GRID_M)

    assert voxels.counts.sum() == len(points)
    assert np.array_equal(np.bincount(voxels.inverse), voxels.counts)
    cells = np.floor((points - points.min(axis=0)) / GRID_M).astype(np.int64)
    assert np.array_equal(voxels.grid_coord[voxels.inverse], cells)


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([], "at least one"),
        ([(0.0, 0.0)], "shape"),
        ([(0.0, 0.0, float("nan"))], "finite"),
        ([(0.0, float("inf"), 0.0)], "finite"),
    ],
)
def test_invalid_coordinates_are_an_explicit_error(points: list[Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        voxelize(points, GRID_M)


# --- Pooling ------------------------------------------------------------------------------


def voxel_features(voxels: Any) -> Any:
    """One recognizable feature row per voxel: (10, 100) for voxel 0, (20, 200) for voxel 1."""
    return np.array([[10.0, 100.0], [20.0, 200.0]])[: len(voxels.counts)]


def test_center_pooling_returns_the_feature_of_the_voxel_holding_the_center_point() -> None:
    voxels = voxelize(CORNER_POINTS, GRID_M)

    assert pool_voxel_features(
        voxel_features(voxels), voxels, center_index=0, pooling="center"
    ).tolist() == [10.0, 100.0]
    assert pool_voxel_features(
        voxel_features(voxels), voxels, center_index=2, pooling="center"
    ).tolist() == [20.0, 200.0]


def test_mean_pooling_averages_over_the_points_not_over_the_voxels() -> None:
    voxels = voxelize(CORNER_POINTS, GRID_M)

    pooled = pool_voxel_features(voxel_features(voxels), voxels, center_index=0, pooling="mean")

    # dois pontos no voxel 0 e um no voxel 1: (2*10 + 1*20) / 3
    assert pooled.tolist() == pytest.approx([40.0 / 3.0, 400.0 / 3.0])


def test_pooling_rejects_an_unknown_mode_and_a_center_outside_the_support() -> None:
    voxels = voxelize(CORNER_POINTS, GRID_M)

    with pytest.raises(ValueError, match="pooling"):
        pool_voxel_features(voxel_features(voxels), voxels, center_index=0, pooling="max")
    with pytest.raises(ValueError, match="center_index"):
        pool_voxel_features(voxel_features(voxels), voxels, center_index=3, pooling="center")


def test_pooling_rejects_features_that_do_not_cover_every_voxel() -> None:
    voxels = voxelize(CORNER_POINTS, GRID_M)

    with pytest.raises(ValueError, match="feature"):
        pool_voxel_features(np.zeros((1, 2)), voxels, center_index=0, pooling="mean")


# --- Checkpoint state ----------------------------------------------------------------------


def test_only_the_backbone_weights_are_kept_and_the_classification_head_is_dropped() -> None:
    state = {
        "module.backbone.embedding.stem.conv.weight": 1,
        "module.backbone.dec.dec0.block1.mlp.0.fc2.bias": 2,
        "module.seg_head.weight": 3,
        "module.seg_head.bias": 4,
    }

    assert extract_backbone_state(state) == {
        "embedding.stem.conv.weight": 1,
        "dec.dec0.block1.mlp.0.fc2.bias": 2,
    }


def test_weights_saved_without_the_distributed_wrapper_prefix_are_accepted() -> None:
    assert extract_backbone_state({"backbone.embedding.x": 1, "seg_head.weight": 2}) == {
        "embedding.x": 1
    }


def test_a_checkpoint_without_backbone_weights_is_incompatible() -> None:
    with pytest.raises(PTv3RuntimeUnavailableError, match="backbone"):
        extract_backbone_state({"module.seg_head.weight": 1})


# --- Checkpoint identity --------------------------------------------------------------------


def test_the_checkpoint_hash_is_verified_against_the_file_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")

    with pytest.raises(PTv3RuntimeUnavailableError, match="does not match"):
        verify_checkpoint_sha256(checkpoint, "sha256:" + "0" * 64)


def test_a_matching_checkpoint_hash_is_accepted(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    digest = "sha256:" + hashlib.sha256(b"weights").hexdigest()

    verify_checkpoint_sha256(checkpoint, digest)


def test_a_missing_checkpoint_file_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(PTv3RuntimeUnavailableError, match="not found"):
        verify_checkpoint_sha256(tmp_path / "absent.pth", "sha256:" + "0" * 64)


# --- The runtime: failures happen before any weight is trusted ---------------------------------


def make_runtime(tmp_path: Path, **overrides: Any) -> PointceptPTv3Runtime:
    values: dict[str, Any] = {
        "pointcept_root": tmp_path / "Pointcept",
        "checkpoint_path": tmp_path / "model.pth",
        "backbone": NUSCENES_SEMSEG_PTV3M1_BASE,
    }
    values.update(overrides)
    return PointceptPTv3Runtime(**values)


COORDINATES = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)]


def test_a_missing_pointcept_checkout_stops_the_run_explicitly(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    with pytest.raises(PTv3RuntimeUnavailableError, match="Pointcept"):
        runtime.infer(coordinates_m=COORDINATES, center_index=0, config=config_for_nuscenes())


def test_weights_that_differ_from_the_configured_hash_are_never_loaded(tmp_path: Path) -> None:
    (tmp_path / "Pointcept" / "pointcept" / "models").mkdir(parents=True)
    (tmp_path / "model.pth").write_bytes(b"not the pinned weights")
    runtime = make_runtime(tmp_path)

    with pytest.raises(PTv3RuntimeUnavailableError, match="does not match"):
        runtime.infer(coordinates_m=COORDINATES, center_index=0, config=config_for_nuscenes())


def test_a_missing_model_library_is_reported_with_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Pointcept" / "pointcept" / "models").mkdir(parents=True)
    weights = b"pinned weights"
    (tmp_path / "model.pth").write_bytes(weights)
    config = config_for_nuscenes(checkpoint_hash="sha256:" + hashlib.sha256(weights).hexdigest())

    def refuse(name: str) -> Any:
        raise ImportError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(ptv3_pointcept, "_import_module", refuse)

    with pytest.raises(PTv3RuntimeUnavailableError, match="torch"):
        make_runtime(tmp_path).infer(coordinates_m=COORDINATES, center_index=0, config=config)


def test_coordinates_that_are_not_finite_are_rejected_before_the_runtime_loads(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)

    with pytest.raises(ValueError, match="finite"):
        runtime.infer(
            coordinates_m=[(0.0, 0.0, float("nan"))],
            center_index=0,
            config=config_for_nuscenes(),
        )


def test_a_center_outside_the_support_is_rejected_before_the_runtime_loads(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)

    with pytest.raises(ValueError, match="center_index"):
        runtime.infer(coordinates_m=COORDINATES, center_index=7, config=config_for_nuscenes())

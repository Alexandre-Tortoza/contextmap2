"""Opt-in checks of the real PTv3 forward pass; skipped on any machine without the model stack.

They need a CUDA GPU, torch, spconv, torch-scatter, a clone of the official Pointcept repository and
the ``nuscenes-semseg-pt-v3m1-0-base`` checkpoint. Point the environment at them::

    CONTEXTMAP_POINTCEPT_ROOT=/path/to/Pointcept \\
    CONTEXTMAP_PTV3_CHECKPOINT=/path/to/model_best.pth \\
    pytest tests/point_representation/test_ptv3_pointcept_real.py

The GPU is shared, so run them under the repository's GPU lock when other jobs may be running.
These are the only tests that run a real PTv3; everything else in this capability uses fakes.
"""

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pointrep_builders import radius_policy
from pointrep_geometry import MAP_ID, LinearScanSource, line_of_points
from pointrep_ptv3_fakes import make_config

from contextmap.geometric_mapping import GeometryReference, geometry_id_for
from contextmap.point_representation import (
    EncodedRepresentation,
    PointRepresentationRunId,
    RepresentationService,
)
from contextmap.point_representation.backends.ptv3 import (
    PTv3Config,
    PTv3OutOfMemoryError,
    PTv3PointEncoder,
)
from contextmap.point_representation.backends.ptv3_pointcept import (
    NUSCENES_SEMSEG_PTV3M1_BASE,
    PointceptPTv3Runtime,
)

POINTCEPT_ROOT = os.environ.get("CONTEXTMAP_POINTCEPT_ROOT")
CHECKPOINT = os.environ.get("CONTEXTMAP_PTV3_CHECKPOINT")

pytestmark = pytest.mark.skipif(
    not (POINTCEPT_ROOT and CHECKPOINT),
    reason="set CONTEXTMAP_POINTCEPT_ROOT and CONTEXTMAP_PTV3_CHECKPOINT to run the real PTv3",
)

# Nuvem determinística de 60 pontos, em metros, com estrutura local (uma "parede" e um "piso").
CLOUD = [(0.01 * i, 0.0, 0.02 * (i % 7)) for i in range(30)] + [
    (0.01 * i, 0.15 + 0.005 * (i % 5), 0.0) for i in range(30)
]


def make_real_config(**overrides: Any) -> PTv3Config:
    assert CHECKPOINT is not None
    digest = hashlib.sha256(Path(CHECKPOINT).read_bytes()).hexdigest()
    values: dict[str, Any] = {
        "variant": "ptv3m1-base",
        "checkpoint": "Pointcept/PointTransformerV3:nuscenes-semseg-pt-v3m1-0-base/model_best.pth",
        "checkpoint_hash": f"sha256:{digest}",
        "output_dimension": NUSCENES_SEMSEG_PTV3M1_BASE.output_channels,
        "padding_channels": NUSCENES_SEMSEG_PTV3M1_BASE.in_channels - 3,
    }
    values.update(overrides)
    return make_config(**values)


@pytest.fixture(scope="module")
def runtime() -> Iterator[PointceptPTv3Runtime]:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    assert POINTCEPT_ROOT is not None and CHECKPOINT is not None
    yield PointceptPTv3Runtime(
        pointcept_root=Path(POINTCEPT_ROOT),
        checkpoint_path=Path(CHECKPOINT),
        backbone=NUSCENES_SEMSEG_PTV3M1_BASE,
        weights_only=False,  # o checkpoint de treino do Pointcept não carrega com weights_only=True
    )


def test_a_real_forward_pass_returns_a_finite_vector_of_the_backbone_width(
    runtime: PointceptPTv3Runtime,
) -> None:
    inference = runtime.infer(coordinates_m=CLOUD, center_index=10, config=make_real_config())

    assert len(inference.vector) == 64
    assert all(abs(value) < 1e6 for value in inference.vector)
    assert any(value != 0.0 for value in inference.vector)


def test_the_same_support_encodes_the_same_vector_within_numerical_tolerance(
    runtime: PointceptPTv3Runtime,
) -> None:
    config = make_real_config()

    first = runtime.infer(coordinates_m=CLOUD, center_index=10, config=config)
    second = runtime.infer(coordinates_m=CLOUD, center_index=10, config=config)

    difference = max(abs(a - b) for a, b in zip(first.vector, second.vector, strict=True))
    assert difference < 1e-4


def test_center_and_mean_pooling_are_different_descriptors_of_the_same_support(
    runtime: PointceptPTv3Runtime,
) -> None:
    center = runtime.infer(coordinates_m=CLOUD, center_index=10, config=make_real_config())
    mean = runtime.infer(
        coordinates_m=CLOUD, center_index=10, config=make_real_config(pooling="mean")
    )

    assert center.vector != mean.vector


def test_the_peak_device_memory_of_a_call_covers_at_least_the_resident_weights(
    runtime: PointceptPTv3Runtime,
) -> None:
    inference = runtime.infer(coordinates_m=CLOUD, center_index=10, config=make_real_config())

    weights_bytes = 46_158_272 * 4
    assert inference.peak_memory_bytes is not None
    assert inference.peak_memory_bytes >= weights_bytes


def test_a_single_point_support_still_encodes(runtime: PointceptPTv3Runtime) -> None:
    inference = runtime.infer(
        coordinates_m=[(0.0, 0.0, 0.0)], center_index=0, config=make_real_config()
    )

    assert len(inference.vector) == 64


def test_the_real_runtime_flows_through_the_encoder_and_the_service(
    runtime: PointceptPTv3Runtime,
) -> None:
    policy = radius_policy(0.6)
    encoder = PTv3PointEncoder(config=make_real_config(), support_policy=policy, runtime=runtime)
    service = RepresentationService(
        LinearScanSource(line_of_points(11)),
        encoder,
        run_id=PointRepresentationRunId("run-0001"),
        code_version="test",
    )
    reference = GeometryReference(
        map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=5)
    )

    (outcome,) = service.represent([reference])

    assert isinstance(outcome, EncodedRepresentation)
    assert outcome.representation.shape == (64,)
    assert outcome.representation.encoder_identity.backend_id == "ptv3"
    assert encoder.telemetry.peak_memory_bytes is not None


def test_running_out_of_device_memory_is_an_explicit_error_and_the_runtime_recovers(
    runtime: PointceptPTv3Runtime,
) -> None:
    torch = pytest.importorskip("torch")
    config = make_real_config()
    runtime.infer(coordinates_m=CLOUD, center_index=0, config=config)  # garante o modelo residente
    torch.cuda.empty_cache()  # sem blocos livres em cache, toda alocação nova passa pelo teto
    torch.cuda.set_per_process_memory_fraction(1e-4)  # só este processo: ~800 KB de teto
    try:
        with pytest.raises(PTv3OutOfMemoryError):
            runtime.infer(coordinates_m=CLOUD, center_index=0, config=config)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)

    assert len(runtime.infer(coordinates_m=CLOUD, center_index=0, config=config).vector) == 64

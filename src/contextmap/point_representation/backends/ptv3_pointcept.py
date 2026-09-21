"""Real PTv3 runtime: the Pointcept backbone behind the :class:`PTv3Runtime` boundary.

:class:`PointceptPTv3Runtime` gives :class:`PTv3PointEncoder` a real forward pass. It is a
*backbone-only* runtime: it builds the Point Transformer V3 encoder-decoder from a Pointcept
checkout, loads only the ``backbone.*`` weights of a supervised segmentation checkpoint and never
builds or applies the classification head. The per-point features it returns are a learned 3D
descriptor of the local structure, not a label; a segmentation checkpoint is used because it is what
the PTv3 authors publish, not because segmentation is wanted here. This is inference with a generic
PTv3, not Sonata- or Vernata-style representation learning.

The module imports no model library at import time (no torch, spconv, torch-scatter, timm or
Pointcept, and not even NumPy): everything is imported when the first support is encoded, so the
capability and the rest of the pipeline work without them and disabling this backend changes no
canonical contract. Pointcept is not pip-installable; it must be a clone of the official repository
at a pinned commit, passed as ``pointcept_root``.

Failure is explicit and there is no fallback. Anything that prevents a forward pass from being
trustworthy raises :class:`PTv3RuntimeUnavailableError` before a weight is used: a missing
dependency or Pointcept checkout, a checkpoint whose SHA-256 differs from the configured
``checkpoint_hash``, a checkpoint that does not fit the declared backbone, a device without CUDA.
Running out of device memory on one support raises :class:`PTv3OutOfMemoryError`.
"""

from __future__ import annotations

import hashlib
import importlib
import pickle
import sys
import types
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmap.point_representation.backends.ptv3 import (
    PTv3Config,
    PTv3Inference,
    PTv3OutOfMemoryError,
    PTv3RuntimeUnavailableError,
)
from contextmap.shared import Vector3

if TYPE_CHECKING:
    from numpy.typing import NDArray

POINTCEPT_RUNTIME_VERSION = "1"
"""Version of this runtime's mapping from a support to the backbone input and back."""

_PTV3_MODULE = "pointcept.models.point_transformer_v3.point_transformer_v3m1_base"
_BACKBONE_PREFIX = "backbone."
_DISTRIBUTED_PREFIX = "module."
_XYZ_CHANNELS = 3
_HASH_CHUNK_BYTES = 1 << 20
# Falhas de leitura de torch.load; qualquer outra exceção é um defeito e deve aparecer como tal.
_UNREADABLE_CHECKPOINT = (
    pickle.UnpicklingError,
    zipfile.BadZipFile,
    EOFError,
    OSError,
    RuntimeError,
)


@dataclass(frozen=True, kw_only=True)
class PointceptBackbone:
    """Architecture of one Pointcept PTv3 (``PT-v3m1``) backbone: the checkpoint's own settings.

    The values must be the ones the checkpoint was trained with (its ``config.py``); loading is
    strict, so a mismatch fails instead of producing a different network.

    Attributes:
        in_channels: Input feature channels the stem convolution expects.
        order: Serialization curves the blocks cycle through.
        stride: Downsampling stride between encoder stages.
        enc_depths: Blocks per encoder stage.
        enc_channels: Channels per encoder stage.
        enc_num_head: Attention heads per encoder stage.
        enc_patch_size: Attention patch size per encoder stage.
        dec_depths: Blocks per decoder stage.
        dec_channels: Channels per decoder stage; the first one is the output feature size.
        dec_num_head: Attention heads per decoder stage.
        dec_patch_size: Attention patch size per decoder stage.
    """

    in_channels: int
    order: tuple[str, ...]
    stride: tuple[int, ...]
    enc_depths: tuple[int, ...]
    enc_channels: tuple[int, ...]
    enc_num_head: tuple[int, ...]
    enc_patch_size: tuple[int, ...]
    dec_depths: tuple[int, ...]
    dec_channels: tuple[int, ...]
    dec_num_head: tuple[int, ...]
    dec_patch_size: tuple[int, ...]

    @property
    def output_channels(self) -> int:
        """Size of the per-point feature the decoder returns."""
        return self.dec_channels[0]

    def constructor_kwargs(self) -> dict[str, Any]:
        """Arguments for ``PointTransformerV3``, configured for deterministic inference.

        Three settings differ from training on purpose. ``shuffle_orders`` is off because Pointcept
        draws a random permutation of the serialization curves on every forward pass, which would
        make the same support encode differently twice; the weights do not depend on which curve a
        block uses, so a fixed assignment is a valid inference configuration. ``enable_flash`` is
        off because FlashAttention is not installed, and the plain patch attention has the same
        parameters. ``drop_path`` is zero because there is no stochastic depth at inference.
        """
        return {
            "in_channels": self.in_channels,
            "order": self.order,
            "stride": self.stride,
            "enc_depths": self.enc_depths,
            "enc_channels": self.enc_channels,
            "enc_num_head": self.enc_num_head,
            "enc_patch_size": self.enc_patch_size,
            "dec_depths": self.dec_depths,
            "dec_channels": self.dec_channels,
            "dec_num_head": self.dec_num_head,
            "dec_patch_size": self.dec_patch_size,
            # Demais valores: os do config.py do checkpoint nuScenes (PT-v3m1 base).
            "mlp_ratio": 4,
            "qkv_bias": True,
            "qk_scale": None,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "pre_norm": True,
            "enable_rpe": False,
            "upcast_attention": False,
            "upcast_softmax": False,
            # Inferência determinística (ver a docstring).
            "drop_path": 0.0,
            "shuffle_orders": False,
            "enable_flash": False,
        }


NUSCENES_SEMSEG_PTV3M1_BASE = PointceptBackbone(
    in_channels=4,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
)
"""Backbone of ``Pointcept/PointTransformerV3`` ``nuscenes-semseg-pt-v3m1-0-base`` (MIT).

Its four input channels are XYZ plus the LiDAR return intensity, so the runtime is configured with
``padding_channels=1``: the intensity channel is filled with zeros and never measured.
"""


@dataclass(frozen=True, eq=False)
class Voxelization:
    """A support reduced to one entry per occupied grid cell.

    Attributes:
        grid_coord: Non-negative integer cell of each voxel, anchored at the support's minimum
            corner, shape ``(V, 3)``, sorted lexicographically.
        centroid_m: Mean of the points inside each voxel, in the input units, shape ``(V, 3)``.
        inverse: Voxel index of each input point, shape ``(N,)``.
        counts: Number of input points in each voxel, shape ``(V,)``.
    """

    grid_coord: NDArray[Any]
    centroid_m: NDArray[Any]
    inverse: NDArray[Any]
    counts: NDArray[Any]


def voxelize(coordinates_m: Sequence[Vector3], grid_size_m: float) -> Voxelization:
    """Merge the points of one support that fall in the same grid cell.

    Sparse convolution needs unique voxel indices, so points sharing a cell are represented by
    their centroid; the mapping back to the input points is kept to pool the features afterwards.
    The result does not depend on the order of the points.

    Args:
        coordinates_m: One ``(x, y, z)`` per supporting element.
        grid_size_m: Cell edge, in the units of ``coordinates_m``.

    Returns:
        The voxels, their centroids and the point-to-voxel mapping.

    Raises:
        ValueError: If there are no points, the shape is not ``(N, 3)`` or a value is not finite.
    """
    import numpy as np

    points = np.asarray(coordinates_m, dtype=np.float64)
    if points.size == 0:
        raise ValueError("a support needs at least one point")
    if points.ndim != 2 or points.shape[1] != _XYZ_CHANNELS:
        raise ValueError(f"coordinates must have shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("coordinates must be finite")
    cells = np.floor((points - points.min(axis=0)) / grid_size_m).astype(np.int64)
    grid_coord, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    centroid_m = np.zeros((len(grid_coord), _XYZ_CHANNELS), dtype=np.float64)
    np.add.at(centroid_m, inverse, points)
    centroid_m /= counts[:, None]
    return Voxelization(
        grid_coord=grid_coord, centroid_m=centroid_m, inverse=inverse, counts=counts
    )


def pool_voxel_features(
    voxel_features: NDArray[Any], voxels: Voxelization, *, center_index: int, pooling: str
) -> NDArray[Any]:
    """Reduce per-voxel features to the one vector that represents the support.

    Args:
        voxel_features: One feature row per voxel, shape ``(V, C)``.
        voxels: The voxelization the features were computed on.
        center_index: Position of the support's center element among the input points.
        pooling: ``"center"`` returns the feature of the voxel holding the center element;
            ``"mean"`` averages over the input points (a voxel counts once per point it holds).

    Returns:
        The pooled vector, shape ``(C,)``.

    Raises:
        ValueError: If ``pooling`` is unknown, ``center_index`` is outside the support or the
            features do not cover every voxel.
    """
    if pooling not in ("center", "mean"):
        raise ValueError(f"pooling must be 'center' or 'mean', got {pooling!r}")
    if not 0 <= center_index < len(voxels.inverse):
        raise ValueError(
            f"center_index {center_index} is outside a support of {len(voxels.inverse)} points"
        )
    if voxel_features.shape[0] != len(voxels.counts):
        raise ValueError(
            f"got {voxel_features.shape[0]} feature rows for {len(voxels.counts)} voxels"
        )
    pooled: NDArray[Any]
    if pooling == "center":
        pooled = voxel_features[voxels.inverse[center_index]]
    else:
        pooled = (voxel_features * voxels.counts[:, None]).sum(axis=0) / voxels.counts.sum()
    return pooled


def check_runtime_compatibility(config: PTv3Config, backbone: PointceptBackbone) -> None:
    """Refuse a configuration that this backbone cannot honour.

    Args:
        config: The effective adapter configuration.
        backbone: The architecture the checkpoint was trained with.

    Raises:
        PTv3RuntimeUnavailableError: If the device is not CUDA, the output dimension is not the
            backbone's feature size, or the declared zero padding does not complete the input.
    """
    if not config.device.startswith("cuda"):
        raise PTv3RuntimeUnavailableError(
            f"the PTv3 backbone needs a CUDA device for sparse convolution, got {config.device!r}"
        )
    if config.output_dimension != backbone.output_channels:
        raise PTv3RuntimeUnavailableError(
            f"output_dimension is {config.output_dimension} but the checkpoint backbone produces "
            f"{backbone.output_channels} features"
        )
    if config.padding_channels + _XYZ_CHANNELS != backbone.in_channels:
        raise PTv3RuntimeUnavailableError(
            f"padding_channels is {config.padding_channels} but the checkpoint backbone expects "
            f"{backbone.in_channels} input channels ({_XYZ_CHANNELS} for XYZ)"
        )


def verify_checkpoint_sha256(path: Path, expected: str) -> None:
    """Require that a checkpoint file is exactly the bytes the configuration pinned.

    Args:
        path: Checkpoint file.
        expected: ``"sha256:<64 hex>"`` from :attr:`PTv3Config.checkpoint_hash`.

    Raises:
        PTv3RuntimeUnavailableError: If the file is missing or its SHA-256 differs.
    """
    if not path.is_file():
        raise PTv3RuntimeUnavailableError(f"checkpoint file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    actual = f"sha256:{digest.hexdigest()}"
    if actual != expected:
        raise PTv3RuntimeUnavailableError(
            f"the checkpoint {path} has hash {actual}, which does not match the configured "
            f"checkpoint_hash {expected}"
        )


def extract_backbone_state(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the backbone weights of a Pointcept segmentation checkpoint.

    Pointcept saves a ``DefaultSegmentor`` under the distributed-training prefix, so the backbone
    lives under ``module.backbone.`` next to the classification head ``module.seg_head.``; the head
    is discarded on purpose, because this stage emits no classification.

    Args:
        state_dict: The ``state_dict`` entry of the checkpoint.

    Returns:
        The backbone weights keyed as ``PointTransformerV3`` names them.

    Raises:
        PTv3RuntimeUnavailableError: If no ``backbone.*`` weight is present.
    """
    backbone: dict[str, Any] = {}
    for key, value in state_dict.items():
        name = key.removeprefix(_DISTRIBUTED_PREFIX)
        if name.startswith(_BACKBONE_PREFIX):
            backbone[name.removeprefix(_BACKBONE_PREFIX)] = value
    if not backbone:
        raise PTv3RuntimeUnavailableError(
            "the checkpoint has no 'backbone.*' weights, so it is not a Pointcept PTv3 model"
        )
    return backbone


def _import_module(name: str) -> Any:
    """Import a module; the seam that lets tests simulate a missing model library."""
    return importlib.import_module(name)


def _require(name: str) -> Any:
    """Import an optional model library or fail with what to install."""
    try:
        return _import_module(name)
    except ImportError as error:
        raise PTv3RuntimeUnavailableError(
            f"the real PTv3 runtime needs the optional dependency {name!r}, which cannot be "
            f"imported ({error}); install it in the model environment"
        ) from error


def _import_pointcept_ptv3(pointcept_root: Path) -> Any:
    """Import only the PTv3 backbone module of a Pointcept checkout.

    Importing ``pointcept.models`` runs an ``__init__`` that pulls every model and the training
    engine (torch_cluster, wandb, tensorboard, compiled ``pointops``), none of which inference
    needs. Instead, lightweight package stubs whose ``__path__`` points at the checkout make the
    backbone's own imports resolve to the real files; only the training-hook base class, which the
    backbone module imports and never uses at inference, is replaced by an empty class. If a real
    or already-stubbed ``pointcept`` is loaded in the process, it is used as it is.
    """
    if "pointcept" not in sys.modules:
        package = pointcept_root / "pointcept"
        for name, path in (
            ("pointcept", package),
            ("pointcept.models", package / "models"),
            ("pointcept.models.point_transformer_v3", package / "models" / "point_transformer_v3"),
            ("pointcept.engines", package / "engines"),
        ):
            stub = types.ModuleType(name)
            stub.__path__ = [str(path)]
            sys.modules[name] = stub
        hooks = types.ModuleType("pointcept.engines.hooks")
        hooks.HookBase = type("HookBase", (), {})  # type: ignore[attr-defined]
        sys.modules["pointcept.engines.hooks"] = hooks
    try:
        return importlib.import_module(_PTV3_MODULE)
    except ImportError as error:
        raise PTv3RuntimeUnavailableError(
            f"the Pointcept PTv3 module cannot be imported from {pointcept_root} ({error})"
        ) from error


@dataclass(frozen=True, eq=False)
class _LoadedBackbone:
    """A backbone resident on the device, with the identity it was loaded under."""

    torch: Any
    model: Any
    device: Any
    device_name: str
    checkpoint_hash: str


class PointceptPTv3Runtime:
    """Runs the Pointcept PTv3 backbone on one support and pools its per-point features.

    The model is built lazily, on the first :meth:`infer`, so constructing the runtime imports and
    loads nothing. A support is voxelized, the centroids (plus zero padding channels) go through the
    backbone, the per-voxel features are mapped back to the support's points and pooled as
    ``config.pooling`` says.

    ``peak_memory_bytes`` in the result is ``torch.cuda.max_memory_allocated`` for the call: the
    peak of tensor memory the process allocated on the device, including the resident weights, and
    excluding the CUDA context and allocator cache that ``nvidia-smi`` also counts.
    """

    def __init__(
        self,
        *,
        pointcept_root: Path,
        checkpoint_path: Path,
        backbone: PointceptBackbone,
        weights_only: bool = True,
    ) -> None:
        """Remember where the code and weights are; nothing is imported or loaded yet.

        Args:
            pointcept_root: Clone of the official Pointcept repository at a pinned commit.
            checkpoint_path: Local checkpoint file whose SHA-256 must equal
                ``PTv3Config.checkpoint_hash``.
            backbone: The architecture the checkpoint was trained with.
            weights_only: Load with ``torch.load(weights_only=True)``. Pointcept training
                checkpoints also pickle optimizer/scheduler state and NumPy scalars, which that
                mode rejects. Passing ``False`` is an explicit decision to unpickle the file, made
                only for a checkpoint of verified provenance; the SHA-256 is always verified first,
                so what is unpickled is exactly the pinned bytes.
        """
        self._pointcept_root = pointcept_root
        self._checkpoint_path = checkpoint_path
        self._backbone = backbone
        self._weights_only = weights_only
        self._loaded: _LoadedBackbone | None = None

    def infer(
        self, *, coordinates_m: Sequence[Vector3], center_index: int, config: PTv3Config
    ) -> PTv3Inference:
        """Encode one support.

        Args:
            coordinates_m: The prepared local coordinates of the support.
            center_index: Position of the center element in ``coordinates_m``.
            config: The effective configuration.

        Returns:
            The pooled vector and the peak device memory of the call.

        Raises:
            ValueError: If the coordinates are empty, not ``(N, 3)`` or not finite, or the center
                is outside the support.
            PTv3OutOfMemoryError: If this support exhausts device memory.
            PTv3RuntimeUnavailableError: If the runtime cannot run at all.
        """
        voxels = voxelize(coordinates_m, config.grid_size_m)
        if not 0 <= center_index < len(coordinates_m):
            raise ValueError(
                f"center_index {center_index} is outside a support of {len(coordinates_m)} points"
            )
        loaded = self._ensure_loaded(config)
        features, peak_memory_bytes = self._forward(loaded, voxels, config)
        pooled = pool_voxel_features(
            features, voxels, center_index=center_index, pooling=config.pooling
        )
        return PTv3Inference(
            vector=tuple(float(value) for value in pooled), peak_memory_bytes=peak_memory_bytes
        )

    def _forward(
        self, loaded: _LoadedBackbone, voxels: Voxelization, config: PTv3Config
    ) -> tuple[NDArray[Any], int]:
        """Run the backbone on the voxels and return their features and the memory peak."""
        import numpy as np

        torch = loaded.torch
        voxel_count = len(voxels.counts)
        padding = np.zeros((voxel_count, config.padding_channels), dtype=np.float64)
        features = np.concatenate([voxels.centroid_m, padding], axis=1).astype(np.float32)
        data = {
            "coord": torch.from_numpy(voxels.centroid_m.astype(np.float32)).to(loaded.device),
            "grid_coord": torch.from_numpy(voxels.grid_coord).to(loaded.device),
            "feat": torch.from_numpy(features).to(loaded.device),
            "offset": torch.tensor([voxel_count], dtype=torch.long, device=loaded.device),
        }
        torch.cuda.reset_peak_memory_stats(loaded.device)
        try:
            with torch.no_grad(), self._autocast(torch, config):
                output = loaded.model(data)
            voxel_features = output.feat.float().cpu().numpy()
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise PTv3OutOfMemoryError(str(error)) from error
        return voxel_features, int(torch.cuda.max_memory_allocated(loaded.device))

    @staticmethod
    def _autocast(torch: Any, config: PTv3Config) -> Any:
        """Mixed-precision context for float16/bfloat16; float32 runs without autocast."""
        if config.precision == "float32":
            return torch.autocast(device_type="cuda", enabled=False)
        dtype = torch.float16 if config.precision == "float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _ensure_loaded(self, config: PTv3Config) -> _LoadedBackbone:
        """Build the backbone on first use, trusting no weight until its hash is verified."""
        check_runtime_compatibility(config, self._backbone)
        if self._loaded is not None:
            if (self._loaded.checkpoint_hash, self._loaded.device_name) != (
                config.checkpoint_hash,
                config.device,
            ):
                raise PTv3RuntimeUnavailableError(
                    "this runtime already loaded another checkpoint or device; build one runtime "
                    "per checkpoint and device"
                )
            return self._loaded
        if not (self._pointcept_root / "pointcept" / "models").is_dir():
            raise PTv3RuntimeUnavailableError(
                f"no Pointcept checkout found at {self._pointcept_root}; clone the official "
                "repository at a pinned commit"
            )
        verify_checkpoint_sha256(self._checkpoint_path, config.checkpoint_hash)
        torch = _require("torch")
        for dependency in ("spconv.pytorch", "torch_scatter", "timm", "addict"):
            _require(dependency)
        if not torch.cuda.is_available():
            raise PTv3RuntimeUnavailableError("CUDA is not available to torch")
        try:
            device = torch.device(config.device)
            torch.zeros(1, device=device)
        except (RuntimeError, AssertionError) as error:
            raise PTv3RuntimeUnavailableError(
                f"device {config.device!r} cannot be used: {error}"
            ) from error
        ptv3 = _import_pointcept_ptv3(self._pointcept_root)
        model = ptv3.PointTransformerV3(**self._backbone.constructor_kwargs())
        state = self._read_backbone_state(torch)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise PTv3RuntimeUnavailableError(
                f"the checkpoint does not fit the declared backbone: {error}"
            ) from error
        model.to(device).eval()
        self._loaded = _LoadedBackbone(
            torch=torch,
            model=model,
            device=device,
            device_name=config.device,
            checkpoint_hash=config.checkpoint_hash,
        )
        return self._loaded

    def _read_backbone_state(self, torch: Any) -> dict[str, Any]:
        """Read the checkpoint on the CPU and keep the backbone weights."""
        try:
            checkpoint = torch.load(
                self._checkpoint_path, map_location="cpu", weights_only=self._weights_only
            )
        except _UNREADABLE_CHECKPOINT as error:
            raise PTv3RuntimeUnavailableError(
                f"the checkpoint cannot be read with weights_only={self._weights_only}: {error}"
            ) from error
        if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
            raise PTv3RuntimeUnavailableError("the checkpoint has no 'state_dict' entry")
        return extract_backbone_state(checkpoint["state_dict"])

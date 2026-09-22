"""A torch double that makes the device run out of memory at a chosen point.

``PointceptPTv3Runtime`` needs a GPU, spconv and the Pointcept code to run for real (see
``test_ptv3_pointcept_real.py``). This double implements only the calls the runtime makes, so the
deterministic suite can check what the runtime does when the device runs out of memory: at each
allocation one support causes, in the forward pass, when copying the result back and while the
weights move to the device. It also reports what was still resident on the device whenever the
runtime cleared the allocator cache, because a cache clear only returns memory nobody references.
"""

from __future__ import annotations

import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

DevicePoint = Literal["coord", "grid_coord", "feat", "offset", "forward", "result", "model_move"]

OUTPUT_CHANNELS = 64
"""Width of the per-voxel features the fake backbone returns (the nuScenes backbone's)."""

_XYZ_COLUMNS = 3


class FakeOutOfMemoryError(RuntimeError):
    """Stands in for ``torch.cuda.OutOfMemoryError``, which is a ``RuntimeError``."""


class FakeTensor:
    """An array that remembers whether it lives on the fake device."""

    def __init__(self, torch: FakeTorch, array: NDArray[Any], *, on_device: bool) -> None:
        self.array = array
        self._torch = torch
        if on_device:
            torch.device_tensors.add(self)

    def to(self, device: object) -> FakeTensor:
        """Copy to the device; the staging allocation that can run out of memory."""
        self._torch.fail_once(_staging_point(self.array))
        return FakeTensor(self._torch, self.array, on_device=True)

    def float(self) -> FakeTensor:
        """Convert to float32; on a device tensor this allocates the converted copy."""
        self._torch.fail_once("result")
        return self

    def cpu(self) -> FakeTensor:
        return FakeTensor(self._torch, self.array, on_device=False)

    def numpy(self) -> NDArray[Any]:
        return self.array


class FakeModel:
    """A backbone whose weights are resident on the device once ``to`` has been called."""

    def __init__(self, torch: FakeTorch) -> None:
        self._torch = torch
        self.loaded_state: dict[str, Any] | None = None

    def load_state_dict(self, state: dict[str, Any], strict: bool) -> None:
        self.loaded_state = state

    def to(self, device: object) -> FakeModel:
        """Move the weights; running out of memory leaves part of them on the device."""
        if str(device).startswith("cuda"):
            self._torch.models_on_device.add(self)
            self._torch.fail_once("model_move")
        else:
            self._torch.models_on_device.discard(self)
        return self

    def eval(self) -> FakeModel:
        return self

    def __call__(self, data: dict[str, FakeTensor]) -> SimpleNamespace:
        self._torch.forward_calls += 1
        self._torch.fail_once("forward")
        coordinates = data["coord"].array
        features = np.repeat(coordinates[:, :1], OUTPUT_CHANNELS, axis=1) + np.arange(
            OUTPUT_CHANNELS, dtype=np.float32
        )
        return SimpleNamespace(feat=FakeTensor(self._torch, features, on_device=True))


class FakeCuda:
    """``torch.cuda``: counts allocator cache clears and what was resident at each one."""

    OutOfMemoryError = FakeOutOfMemoryError

    def __init__(self, torch: FakeTorch) -> None:
        self._torch = torch
        self.empty_cache_calls = 0
        # A cada limpeza: (tensores do suporte vivos no dispositivo, modelos com pesos nele).
        self.residency_at_empty_cache: list[tuple[int, int]] = []

    def is_available(self) -> bool:
        return True

    def reset_peak_memory_stats(self, device: object) -> None:
        pass

    def max_memory_allocated(self, device: object) -> int:
        return 1_000_000

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1
        self.residency_at_empty_cache.append(
            (len(self._torch.device_tensors), len(self._torch.models_on_device))
        )


class FakeTorch:
    """The ``torch`` module as the runtime uses it, with one scripted out-of-memory failure.

    Args:
        out_of_memory_at: The first time the runtime reaches this point the device runs out of
            memory; later attempts succeed, as when another process frees its memory. ``None``
            never fails. The staging points are recognized by what is being copied: ``coord``
            (float32 voxel centroids), ``grid_coord`` (int64 cells), ``feat`` (the padded input
            features) and ``offset`` (the batch offsets tensor).
    """

    long = "long"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self, *, out_of_memory_at: DevicePoint | None = None) -> None:
        self.cuda = FakeCuda(self)
        self.forward_calls = 0
        self.models_built: list[weakref.ReferenceType[FakeModel]] = []
        self.device_tensors: weakref.WeakSet[FakeTensor] = weakref.WeakSet()
        self.models_on_device: weakref.WeakSet[FakeModel] = weakref.WeakSet()
        self.pointcept_module = SimpleNamespace(PointTransformerV3=self._build_model)
        self._out_of_memory_at = out_of_memory_at

    def fail_once(self, point: DevicePoint) -> None:
        """Raise the device out-of-memory error if this is the scripted point, the first time."""
        if self._out_of_memory_at == point:
            self._out_of_memory_at = None
            raise FakeOutOfMemoryError(
                f"CUDA out of memory at {point!r}. Tried to allocate 24.00 MiB."
            )

    def _build_model(self, **constructor_kwargs: Any) -> FakeModel:
        model = FakeModel(self)
        self.models_built.append(weakref.ref(model))
        return model

    def device(self, name: str) -> str:
        return name

    def zeros(self, size: int, *, device: object) -> None:
        return None

    def load(self, path: Path, *, map_location: str, weights_only: bool) -> dict[str, Any]:
        return {"state_dict": {"module.backbone.embedding.weight": 1.0}}

    def from_numpy(self, array: NDArray[Any]) -> FakeTensor:
        return FakeTensor(self, array, on_device=False)

    def tensor(self, values: list[int], *, dtype: object, device: object) -> FakeTensor:
        self.fail_once("offset")
        return FakeTensor(self, np.asarray(values), on_device=True)

    def no_grad(self) -> Any:
        return nullcontext()

    def autocast(self, **options: Any) -> Any:
        return nullcontext()


def _staging_point(array: NDArray[Any]) -> DevicePoint:
    """Name the input being staged from its dtype and width."""
    if array.dtype == np.int64:
        return "grid_coord"
    return "coord" if array.shape[1] == _XYZ_COLUMNS else "feat"

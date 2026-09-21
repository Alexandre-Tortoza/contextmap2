"""Optional PTv3 learned 3D encoder adapter: a boundary, not a training method.

This adapter lets a Point Transformer V3 backbone be selected as a
:class:`PointEncoder` without any downstream change. It consumes only the
canonical prepared support (local XYZ) and returns one vector per support. All
framework detail (torch, CUDA, checkpoint loading, precision, tensor
conversion, serialization/voxelization) lives behind the injected
:class:`PTv3Runtime`; this module imports none of it, so importing it needs no
model library and disabling the backend changes nothing in the canonical
contracts.

What this is, and is not
    This is *inference* with a generic PTv3 encoder as a learned 3D baseline.
    It is not Sonata- or Vernata-style representation learning: those methods
    add self-distillation and, for Vernata, high-resolution 2D-to-3D
    cross-modal supervision, none of which is implemented or implied here. A
    pretrained or distilled encoder must be a separately identified backend, so
    a configuration that names Sonata or Vernata is rejected.

Failure is explicit and there is no fallback to another encoder: a missing
dependency or an incompatible checkpoint (:class:`PTv3RuntimeUnavailableError`)
stops the run, a support that exhausts device memory becomes an explicit failed
support, and a wrongly shaped result is a contract violation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Protocol

from contextmap.point_representation.compatibility import representation_space_fingerprint
from contextmap.point_representation.models import (
    EncoderIdentity,
    PreparedSupport,
    RepresentationSpace,
    SupportPolicy,
    SupportType,
)
from contextmap.point_representation.ports import EncodedVector, UnencodableSupportError
from contextmap.shared import Vector3

PTV3_ADAPTER_VERSION = "1"
"""Version of this adapter's mapping from configuration to a representation space."""

_PRECISIONS = ("float32", "float16", "bfloat16")
_POOLINGS = ("center", "mean")
_NORMALIZATIONS = ("none", "l2")
_CHECKPOINT_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_DISTILLED_FAMILIES = ("sonata", "vernata")


@dataclass(frozen=True, kw_only=True)
class PTv3Config:
    """Effective, secret-free configuration of the PTv3 adapter.

    Attributes:
        variant: Architecture identity, e.g. ``"ptv3-base"``.
        checkpoint: Human-readable checkpoint identifier (not a filesystem path).
        checkpoint_hash: ``"sha256:<64 hex>"`` of the checkpoint weights; part
            of the space identity, so two weights under one name never mix.
        device: Execution device, e.g. ``"cuda:0"`` or ``"cpu"``. Recorded in
            the configuration identity but not in the space identity.
        precision: Inference precision: ``float32``, ``float16`` or ``bfloat16``.
        grid_size_m: Serialization/voxelization grid size, in prepared-coordinate
            units.
        output_dimension: Dimension of the pooled vector the runtime returns.
        pooling: How per-point features reduce to one vector: ``center`` takes
            the feature at the support's center element, ``mean`` averages them.
        normalization: Normalization applied to the returned vector: ``none``
            or ``l2``.
        min_support_points: Supports smaller than this are reported as failed
            without invoking the runtime.
        padding_channels: Number of trailing input channels, beyond XYZ, that the
            checkpoint expects and the runtime fills with zeros (for example the
            LiDAR intensity channel of a model trained on nuScenes). It is part of
            the input definition, so it changes the space identity; no measured
            value is ever supplied for these channels.
    """

    variant: str
    checkpoint: str
    checkpoint_hash: str
    device: str
    precision: str
    grid_size_m: float
    output_dimension: int
    pooling: str
    normalization: str
    min_support_points: int
    padding_channels: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration before any model is built.

        Raises:
            ValueError: If a field is empty, unsupported, non-positive, or not
                finite; the checkpoint hash is malformed; or the variant or
                checkpoint names a pretrained/distilled family.
        """
        for name in ("variant", "checkpoint", "device"):
            if not getattr(self, name).strip():
                raise ValueError(f"PTv3 {name} must not be empty")
        if any(
            family in value.lower()
            for family in _DISTILLED_FAMILIES
            for value in (self.variant, self.checkpoint)
        ):
            raise ValueError(
                "PTv3 is the generic learned 3D baseline; a Sonata/Vernata-style pretrained or "
                "distilled encoder must be a separately identified backend"
            )
        if not _CHECKPOINT_HASH.fullmatch(self.checkpoint_hash):
            raise ValueError("PTv3 checkpoint_hash must be 'sha256:' followed by 64 hex digits")
        for name, allowed in (
            ("precision", _PRECISIONS),
            ("pooling", _POOLINGS),
            ("normalization", _NORMALIZATIONS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"PTv3 {name} must be one of {allowed}, got {getattr(self, name)!r}"
                )
        if not math.isfinite(self.grid_size_m) or self.grid_size_m <= 0:
            raise ValueError("PTv3 grid_size_m must be finite and positive")
        if self.output_dimension <= 0:
            raise ValueError("PTv3 output_dimension must be positive")
        if self.min_support_points < 1:
            raise ValueError("PTv3 min_support_points must be at least 1")
        if self.padding_channels < 0:
            raise ValueError("PTv3 padding_channels must not be negative")

    def to_dict(self) -> dict[str, str | int | float]:
        """Return the secret-free, JSON-compatible effective configuration."""
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class PTv3Inference:
    """SDK-neutral result of one forward pass, with no framework tensor.

    Attributes:
        vector: The pooled vector, ``output_dimension`` plain floats.
        peak_memory_bytes: Peak device memory the runtime observed for this
            call, when it can measure it; ``None`` otherwise.
    """

    vector: tuple[float, ...]
    peak_memory_bytes: int | None = None


class PTv3OutOfMemoryError(RuntimeError):
    """Raised by a runtime when one support exhausts device memory."""


class PTv3RuntimeUnavailableError(RuntimeError):
    """Raised by a runtime that cannot run at all: a missing dependency, device or checkpoint."""


class PTv3Runtime(Protocol):
    """Internal boundary isolating torch, CUDA, PTv3 and checkpoint loading."""

    def infer(
        self, *, coordinates_m: Sequence[Vector3], center_index: int, config: PTv3Config
    ) -> PTv3Inference:
        """Run the encoder on one support and pool its features.

        Args:
            coordinates_m: The prepared local coordinates, one ``(x, y, z)`` per
                supporting element.
            center_index: Position of the center element in ``coordinates_m``.
            config: The effective configuration (precision, grid size, pooling).

        Returns:
            The pooled vector.

        Raises:
            PTv3OutOfMemoryError: If this support exhausts device memory.
            PTv3RuntimeUnavailableError: If the runtime cannot run at all.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class PTv3Telemetry:
    """Runtime cost observed by one adapter, for later ablation against ``off``.

    Attributes:
        encoded: Supports encoded successfully.
        out_of_memory: Supports that exhausted device memory.
        inference_seconds: Wall time spent inside the runtime.
        peak_memory_bytes: Largest peak device memory a call reported; ``None``
            when the runtime never measured it.
    """

    encoded: int
    out_of_memory: int
    inference_seconds: float
    peak_memory_bytes: int | None


class PTv3PointEncoder:
    """Encodes a prepared support with an injected PTv3 runtime."""

    def __init__(
        self, *, config: PTv3Config, support_policy: SupportPolicy, runtime: PTv3Runtime
    ) -> None:
        """Construct the adapter without loading a model outside the injected runtime.

        Args:
            config: The effective configuration.
            support_policy: The neighborhood policy of the supports it will
                receive; it is part of the representation space identity.
            runtime: The runtime that owns torch, CUDA and the checkpoint.

        Raises:
            ValueError: If the policy is a single point, which has no local
                structure for the encoder.
        """
        if support_policy.support_type is not SupportType.NEIGHBORHOOD:
            raise ValueError("the PTv3 backend needs a neighborhood support policy")
        self._config = config
        self._runtime = runtime
        self._space = RepresentationSpace(
            family="ptv3",
            model=f"{config.variant}/{config.pooling}-pooling",
            version=PTV3_ADAPTER_VERSION,
            checkpoint=f"{config.checkpoint}@{config.checkpoint_hash}",
            dimension=config.output_dimension,
            dtype="float32",
            normalization=config.normalization,
            input_definition=_input_definition(config),
            support_semantics=support_policy,
            feature_names=(),
        )
        configuration = json.dumps(
            {
                "adapter_version": PTV3_ADAPTER_VERSION,
                "config": config.to_dict(),
                "representation_space": representation_space_fingerprint(self._space),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._identity = EncoderIdentity(
            backend_id="ptv3",
            backend_version=PTV3_ADAPTER_VERSION,
            configuration_fingerprint="sha256:"
            + hashlib.sha256(configuration.encode("utf-8")).hexdigest(),
            checkpoint_hash=config.checkpoint_hash,
        )
        self._encoded = 0
        self._out_of_memory = 0
        self._inference_seconds = 0.0
        self._peak_memory_bytes: int | None = None

    @property
    def telemetry(self) -> PTv3Telemetry:
        """Runtime and peak device memory observed so far."""
        return PTv3Telemetry(
            encoded=self._encoded,
            out_of_memory=self._out_of_memory,
            inference_seconds=self._inference_seconds,
            peak_memory_bytes=self._peak_memory_bytes,
        )

    def encoder_identity(self) -> EncoderIdentity:
        """Report the adapter identity, including the checkpoint hash."""
        return self._identity

    def representation_space(self) -> RepresentationSpace:
        """Report the space, identified by checkpoint, input, pooling and support policy."""
        return self._space

    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        """Encode one prepared support through the runtime.

        Args:
            prepared: A support prepared under this adapter's support policy.

        Returns:
            One vector of ``output_dimension`` floats.

        Raises:
            UnencodableSupportError: If the support is smaller than
                ``min_support_points``, exhausts device memory, or cannot be
                normalized.
            ValueError: If the support was prepared under another policy, or the
                runtime returned a vector of the wrong dimension.
            PTv3RuntimeUnavailableError: If the runtime cannot run at all; the
                run stops and nothing falls back to another encoder.
        """
        if prepared.support.policy != self._space.support_semantics:
            raise ValueError(
                "the support was prepared under a different support policy than the one this "
                "encoder's representation space declares"
            )
        count = len(prepared.local_coordinates_m)
        if count < self._config.min_support_points:
            raise UnencodableSupportError(
                f"the support has {count} points but the PTv3 backend needs at least "
                f"{self._config.min_support_points}"
            )
        started = perf_counter()
        try:
            inference = self._runtime.infer(
                coordinates_m=prepared.local_coordinates_m,
                center_index=prepared.support.geometry_refs.index(prepared.support.center),
                config=self._config,
            )
        except PTv3OutOfMemoryError as error:
            self._inference_seconds += perf_counter() - started
            self._out_of_memory += 1
            raise UnencodableSupportError(
                f"device out of memory while encoding a {count}-point support: {error}"
            ) from error
        self._inference_seconds += perf_counter() - started
        if len(inference.vector) != self._config.output_dimension:
            raise ValueError(
                f"the PTv3 runtime returned {len(inference.vector)} values for a "
                f"{self._config.output_dimension}-dimensional representation space"
            )
        if inference.peak_memory_bytes is not None:
            self._peak_memory_bytes = max(self._peak_memory_bytes or 0, inference.peak_memory_bytes)
        values = tuple(float(value) for value in inference.vector)
        if self._config.normalization == "l2":
            values = _l2_normalized(values)
        self._encoded += 1
        return EncodedVector(values=values)


def _input_definition(config: PTv3Config) -> str:
    """Describe exactly what the runtime feeds the backbone, so the space identity is honest."""
    definition = f"xyz-local-prepared;grid_size_m={config.grid_size_m};precision={config.precision}"
    if config.padding_channels:
        definition += f";zero_padding_channels={config.padding_channels}"
    return definition


def _l2_normalized(values: tuple[float, ...]) -> tuple[float, ...]:
    """Scale a vector to unit norm; a zero vector has no direction, so it is refused."""
    norm = math.hypot(*values)
    if norm == 0.0:
        raise UnencodableSupportError(
            "the runtime returned a zero vector, which cannot be l2-normalized"
        )
    return tuple(value / norm for value in values)

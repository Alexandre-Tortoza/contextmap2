"""In-memory ingestion fixtures and a fake source adapter for the runtime tests.

The adapter is scripted: it yields real canonical observations, can fail or cancel at a given
observation and reports what its "source" provides, so the ingestion service is exercised
end to end (down to a real ``SequenceArtifact`` on disk) without ROS or a recorded bag.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from contextmap.ingestion import (
    CalibrationSet,
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    SensorId,
    SourceAdapter,
    SourceAdapterCapabilities,
    SourceAdapterConfig,
    SourceAdapterWarning,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
    SourceTopicMapping,
    SynchronizationConfig,
)
from contextmap.runtime import (
    BackendConfigurationError,
    IngestionRequest,
    ValidationPolicy,
)
from contextmap.shared import SourceTimestamp

CLOCK = "fake:header"


def _timestamp(seconds: float) -> SourceTimestamp:
    whole = int(seconds)
    return SourceTimestamp(
        seconds=whole, nanoseconds=round((seconds - whole) * 1_000_000_000), clock_id=CLOCK
    )


def image(index: int, seconds: float, *, data: bytes | None = None) -> ImageObservation:
    return ImageObservation(
        observation_id=SourceObservationId(f"frame-{index:04d}"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=_timestamp(seconds),
        provenance=SourceProvenance(
            source_type="fake", source_path="fake.bag", source_topic="/camera"
        ),
        width=2,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=bytes(range(6)) if data is None else data,
    )


def imu(index: int, seconds: float) -> ImuObservation:
    return ImuObservation(
        observation_id=SourceObservationId(f"imu-{index:04d}"),
        sensor_id=SensorId("imu0"),
        frame_id=FrameId("imu_link"),
        timestamp=_timestamp(seconds),
        provenance=SourceProvenance(
            source_type="fake", source_path="fake.bag", source_topic="/imu"
        ),
        linear_acceleration=(0.0, 0.0, 9.81),
        angular_velocity=(0.0, 0.0, 0.0),
    )


def sequence() -> list[SourceObservation]:
    """Three images and three IMU samples, each pair on the same instant."""
    return [
        image(1, 1.0),
        imu(1, 1.0),
        image(2, 2.0),
        imu(2, 2.0),
        image(3, 3.0),
        imu(3, 3.0),
    ]


class FakeAdapter:
    """A scripted source adapter, conforming to the ingestion ``SourceAdapter`` boundary."""

    def __init__(
        self,
        config: SourceAdapterConfig,
        observations: Sequence[SourceObservation],
        *,
        provides: SourceAdapterCapabilities,
        warnings: Sequence[str] = (),
        fail_after: int | None = None,
        error: BaseException | None = None,
        on_observation: Callable[[int], None] | None = None,
    ) -> None:
        self.config = config
        self._observations = list(observations)
        self._provides = provides
        self._warnings = [
            SourceAdapterWarning(topic=None, message_index=None, reason=w) for w in warnings
        ]
        self._fail_after = fail_after
        self._error = error
        self._on_observation = on_observation
        self.read_calls = 0

    def capabilities(self) -> SourceAdapterCapabilities:
        return self._provides

    def read_observations(self) -> Iterator[SourceObservation]:
        self.read_calls += 1
        for index, observation in enumerate(self._observations):
            if self._fail_after is not None and index == self._fail_after:
                raise self._error or RuntimeError("bag corrupt")
            if self._on_observation is not None:
                self._on_observation(index)
            yield observation

    def read_calibration(self) -> CalibrationSet | None:
        return self.config.calibration

    def warnings(self) -> Sequence[SourceAdapterWarning]:
        return self._warnings


def factory(
    *,
    family: str = "ros1_bag",
    observations: Sequence[SourceObservation] | None = None,
    provides: SourceAdapterCapabilities | None = None,
    built: list[FakeAdapter] | None = None,
    **adapter_options: Any,
) -> Callable[[SourceAdapterConfig], SourceAdapter]:
    """A factory like the composed one: it builds only its configured family, never another."""
    capabilities = provides or SourceAdapterCapabilities(rgb=True, imu=True)

    def build(config: SourceAdapterConfig) -> SourceAdapter:
        if config.source_type != family:
            raise BackendConfigurationError(
                "ingestion.source_adapter",
                family,
                [f"the request asks for {config.source_type!r}; only {family!r} is configured"],
            )
        adapter = FakeAdapter(
            config,
            sequence() if observations is None else observations,
            provides=capabilities,
            **adapter_options,
        )
        if built is not None:
            built.append(adapter)
        return adapter

    return build


def request(
    tmp_path: Path,
    *,
    source_type: str = "ros1_bag",
    name: str = "corridor-02",
    topics: SourceTopicMapping | None = None,
    reference: str = "image",
    tolerance_ns: int = 100_000_000,
    validation: ValidationPolicy | None = None,
    **changes: Any,
) -> IngestionRequest:
    source = tmp_path / "recording.bag"
    source.write_bytes(b"raw-source-bytes")
    values: dict[str, Any] = {
        "source_type": source_type,
        "source_path": str(source),
        "sequence_name": name,
        "output_dir": str(tmp_path / "ws" / "sequences" / name / "artifact-1"),
        "topics": topics or SourceTopicMapping(rgb="/camera", imu="/imu"),
        "synchronization": SynchronizationConfig(
            reference_modality=reference, tolerance_nanoseconds=tolerance_ns
        ),
        "timestamp_clock_id": CLOCK,
    }
    if validation is not None:
        values["validation"] = validation
    values.update(changes)
    return IngestionRequest(**values)

from collections.abc import Iterator, Sequence

import pytest

from contextmap.ingestion import (
    CalibrationSet,
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    MissingRequiredTopicError,
    SensorId,
    SourceAdapter,
    SourceAdapterCapabilities,
    SourceAdapterConfig,
    SourceAdapterWarning,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
    SourceTopicMapping,
    SourceWindow,
)
from contextmap.shared import SourceTimestamp


class FakeSourceAdapter:
    """A minimal, dependency-free SourceAdapter used to test the boundary itself.

    Not a real decoder: it yields the observations it was constructed with,
    demonstrating that downstream code only needs the SourceAdapter
    protocol to consume any source, ROS-backed or not.
    """

    def __init__(
        self,
        *,
        config: SourceAdapterConfig,
        observations: Sequence[SourceObservation],
        available_topics: frozenset[str],
    ) -> None:
        self._config = config
        self._observations = observations
        self._available_topics = available_topics
        self._warnings: list[SourceAdapterWarning] = []

    def capabilities(self) -> SourceAdapterCapabilities:
        return SourceAdapterCapabilities(
            rgb="rgb" in self._available_topics,
            lidar="lidar" in self._available_topics,
            imu="imu" in self._available_topics,
            external_pose="pose" in self._available_topics,
            calibration="camera_info" in self._available_topics,
        )

    def read_observations(self) -> Iterator[SourceObservation]:
        missing_required = self._config.required_topics - self._available_topics
        if missing_required:
            raise MissingRequiredTopicError(
                f"required topics missing from source: {sorted(missing_required)}"
            )
        self._warnings = []
        for observation in self._observations:
            if isinstance(observation, ImuObservation) and "imu" not in self._available_topics:
                self._warnings.append(
                    SourceAdapterWarning(
                        topic=self._config.topics.imu,
                        message_index=None,
                        reason="imu topic not available in source",
                    )
                )
                continue
            yield observation

    def warnings(self) -> Sequence[SourceAdapterWarning]:
        return tuple(self._warnings)

    def read_calibration(self) -> CalibrationSet | None:
        return self._config.calibration


def _image_observation() -> ImageObservation:
    return ImageObservation(
        observation_id=SourceObservationId("frame-0001"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        timestamp=SourceTimestamp(seconds=1, nanoseconds=0, clock_id="clock-a"),
        provenance=SourceProvenance(source_type="fake", source_path="fixtures/fake"),
        width=1,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=b"\x00",
    )


def _imu_observation() -> ImuObservation:
    return ImuObservation(
        observation_id=SourceObservationId("imu-0001"),
        sensor_id=SensorId("imu0"),
        frame_id=FrameId("imu_link"),
        timestamp=SourceTimestamp(seconds=1, nanoseconds=0, clock_id="clock-a"),
        provenance=SourceProvenance(source_type="fake", source_path="fixtures/fake"),
    )


def _consume_generically(adapter: SourceAdapter) -> list[SourceObservationId]:
    """Downstream-style consumption: no branching on which adapter this is."""
    return [observation.observation_id for observation in adapter.read_observations()]


def test_fake_adapter_satisfies_the_source_adapter_protocol() -> None:
    config = SourceAdapterConfig(
        source_type="fake",
        path="fixtures/fake",
        topics=SourceTopicMapping(rgb="/camera"),
        timestamp_clock_id="fake-clock",
    )
    adapter = FakeSourceAdapter(
        config=config, observations=[_image_observation()], available_topics=frozenset({"rgb"})
    )

    assert isinstance(adapter, SourceAdapter)


def test_downstream_code_consumes_any_adapter_without_branching() -> None:
    config = SourceAdapterConfig(
        source_type="fake",
        path="fixtures/fake",
        topics=SourceTopicMapping(rgb="/camera"),
        timestamp_clock_id="fake-clock",
    )
    adapter = FakeSourceAdapter(
        config=config, observations=[_image_observation()], available_topics=frozenset({"rgb"})
    )

    assert _consume_generically(adapter) == ["frame-0001"]


def test_capabilities_reflect_available_topics() -> None:
    config = SourceAdapterConfig(
        source_type="fake",
        path="fixtures/fake",
        topics=SourceTopicMapping(rgb="/camera", lidar="/velodyne"),
        timestamp_clock_id="fake-clock",
    )
    adapter = FakeSourceAdapter(config=config, observations=[], available_topics=frozenset({"rgb"}))

    capabilities = adapter.capabilities()

    assert capabilities.rgb is True
    assert capabilities.lidar is False
    assert capabilities.imu is False


def test_missing_required_topic_raises_before_yielding_anything() -> None:
    config = SourceAdapterConfig(
        source_type="fake",
        path="fixtures/fake",
        topics=SourceTopicMapping(rgb="/camera", lidar="/velodyne"),
        timestamp_clock_id="fake-clock",
        required_topics=frozenset({"lidar"}),
    )
    adapter = FakeSourceAdapter(
        config=config, observations=[_image_observation()], available_topics=frozenset({"rgb"})
    )

    with pytest.raises(MissingRequiredTopicError):
        list(adapter.read_observations())


def test_unsupported_modality_is_reported_as_a_warning_not_silently_dropped() -> None:
    config = SourceAdapterConfig(
        source_type="fake",
        path="fixtures/fake",
        topics=SourceTopicMapping(rgb="/camera", imu="/imu/data"),
        timestamp_clock_id="fake-clock",
    )
    adapter = FakeSourceAdapter(
        config=config,
        observations=[_image_observation(), _imu_observation()],
        available_topics=frozenset({"rgb"}),
    )

    observations = list(adapter.read_observations())

    assert [obs.observation_id for obs in observations] == ["frame-0001"]
    assert len(adapter.warnings()) == 1
    assert adapter.warnings()[0].reason == "imu topic not available in source"


def test_config_rejects_unknown_required_topic_name() -> None:
    with pytest.raises(ValueError, match="required_topics"):
        SourceAdapterConfig(
            source_type="fake",
            path="fixtures/fake",
            topics=SourceTopicMapping(),
            timestamp_clock_id="fake-clock",
            required_topics=frozenset({"radar"}),
        )


def test_source_window_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="end_seconds"):
        SourceWindow(clock_id="clock-a", start_seconds=5.0, end_seconds=2.0)


def test_source_window_rejects_equal_start_and_end() -> None:
    """[t, t) não tem nenhum instante dentro; tratamos como configuração inválida.

    Uma janela sempre errou pelo lado explícito (ver
    ``InvalidSourceWindowError`` para não-sobreposição): resolver
    silenciosamente para uma leitura vazia é o comportamento que
    ``docs/adapters.md`` proíbe. Uma janela de largura zero é sintoma do
    mesmo problema — quase sempre um erro de configuração (dois limites
    iguais por engano), não uma leitura vazia intencional.
    """
    with pytest.raises(ValueError, match="end_seconds"):
        SourceWindow(clock_id="clock-a", start_seconds=1.0, end_seconds=1.0)


@pytest.mark.parametrize("start_seconds", [float("nan"), float("inf"), float("-inf")])
def test_source_window_rejects_non_finite_start(start_seconds: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        SourceWindow(clock_id="clock-a", start_seconds=start_seconds, end_seconds=2.0)


@pytest.mark.parametrize("end_seconds", [float("nan"), float("inf"), float("-inf")])
def test_source_window_rejects_non_finite_end(end_seconds: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        SourceWindow(clock_id="clock-a", start_seconds=0.0, end_seconds=end_seconds)


def test_resolved_window_clock_id_is_deterministic_from_source_type_and_path() -> None:
    config = SourceAdapterConfig(
        source_type="ros1_bag", path="data/example.bag", topics=SourceTopicMapping()
    )

    assert config.resolved_window_clock_id() == "ros1_bag:data/example.bag:recording_time"


def test_resolved_window_clock_id_differs_from_the_header_clock_id() -> None:
    config = SourceAdapterConfig(
        source_type="ros1_bag", path="data/example.bag", topics=SourceTopicMapping()
    )

    assert config.resolved_window_clock_id() != config.resolved_timestamp_clock_id()


def test_config_accepts_an_optional_window() -> None:
    window = SourceWindow(
        clock_id="ros1_bag:data/example.bag:recording_time", start_seconds=0.0, end_seconds=2.0
    )
    config = SourceAdapterConfig(
        source_type="ros1_bag", path="data/example.bag", topics=SourceTopicMapping(), window=window
    )

    assert config.window is window


def test_config_window_defaults_to_none() -> None:
    config = SourceAdapterConfig(
        source_type="ros1_bag", path="data/example.bag", topics=SourceTopicMapping()
    )

    assert config.window is None


def test_generic_adapter_exposes_configured_calibration() -> None:
    calibration = CalibrationSet(entries={}, static_transforms=())
    adapter = FakeSourceAdapter(
        config=SourceAdapterConfig(
            source_type="fake",
            path="fixtures/fake",
            topics=SourceTopicMapping(),
            timestamp_clock_id="fake-clock",
            calibration=calibration,
        ),
        observations=[],
        available_topics=frozenset(),
    )

    assert adapter.read_calibration() is calibration

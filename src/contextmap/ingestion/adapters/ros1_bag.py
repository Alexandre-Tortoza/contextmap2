"""ROS 1 bag source adapter.

Reads a ROS 1 bag and decodes the topics named in the adapter's
:class:`~contextmap.ingestion.source_adapter.SourceAdapterConfig` into
canonical :data:`~contextmap.ingestion.models.SourceObservation` instances,
never exposing a ROS message object across the
:class:`~contextmap.ingestion.source_adapter.SourceAdapter` boundary. Uses
the pure-Python ``rosbags`` library (optional dependency, extra
``contextmap[ros1]``), so no ROS 1 installation is required — see
``src/contextmap/ingestion/docs/adapters.md``.

v0 scope: RGB (``sensor_msgs/Image``), LiDAR (``sensor_msgs/PointCloud2``),
IMU (``sensor_msgs/Imu``), external pose (``nav_msgs/Odometry``), and
camera calibration (``sensor_msgs/CameraInfo``, via
:meth:`Ros1BagSourceAdapter.read_calibration`). Static/dynamic TF
(``tf2_msgs/TFMessage``) is not decoded in v0; callers provide static
extrinsics through ``SourceAdapterConfig.calibration``. Message decoding
shared with the ROS 2 adapter lives in
:mod:`contextmap.ingestion.adapters._ros_common`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from contextmap.ingestion.adapters import _ros_common
from contextmap.ingestion.calibration import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationSet,
    compute_content_hash,
)
from contextmap.ingestion.models import (
    CalibrationReferenceId,
    FrameId,
    SensorId,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.ingestion.source_adapter import (
    MissingRequiredTopicError,
    SourceAdapterCapabilities,
    SourceAdapterConfig,
    SourceAdapterWarning,
)

_SOURCE_TYPE = "ros1_bag"


class Ros1BagSourceAdapter:
    """Reads a ROS 1 bag and yields canonical source observations."""

    def __init__(self, config: SourceAdapterConfig) -> None:
        """Create an adapter for a configured ROS 1 bag.

        Args:
            config: Adapter configuration (path + topic mapping). By
                convention ``config.source_type`` is ``"ros1_bag"``, but
                this is not validated here.
        """
        self._config = config
        self._typestore = get_typestore(Stores.ROS1_NOETIC)
        self._warnings: list[SourceAdapterWarning] = []
        self._content_hash = _ros_common.StreamingContentHash()
        self._calibration_cache: CalibrationSet | None = None
        self._calibration_read = False

    def capabilities(self) -> SourceAdapterCapabilities:
        """Report which configured modalities are actually present in the bag.

        Returns:
            Capabilities reflecting which configured topics exist in the bag.
        """
        available = self._available_topics()
        mapping = self._config.topics
        return SourceAdapterCapabilities(
            rgb=mapping.rgb is not None and mapping.rgb in available,
            lidar=mapping.lidar is not None and mapping.lidar in available,
            imu=mapping.imu is not None and mapping.imu in available,
            external_pose=mapping.pose is not None and mapping.pose in available,
            calibration=(
                self._config.calibration is not None
                or (mapping.camera_info is not None and mapping.camera_info in available)
            ),
        )

    def read_observations(self) -> Iterator[SourceObservation]:
        """Decode the bag and yield canonical observations.

        When ``config.window`` is set, only messages whose recording
        timestamp falls inside it are read at all — the bag's own chunk
        index is used to skip the rest without decompressing it (issue
        #506) — and each configured topic's ``observation_id`` counter
        starts at zero relative to the window, not to the bag's start;
        correlating across two different windows' observations must use
        their physical ``timestamp``, never ``observation_id``. This cost
        guarantee covers the configured modality topics (rgb/lidar/imu/pose)
        and :meth:`content_hash`; it does not extend to
        :meth:`read_calibration`, which is global source metadata read once
        regardless of the window (see :meth:`read_calibration`).

        Yields:
            One :data:`~contextmap.ingestion.models.SourceObservation` per
            successfully decoded message within the configured window (the
            whole bag when none is configured), in bag order. A message
            whose content cannot be decoded (a ``ValueError`` from decoding)
            is skipped and reported through :meth:`warnings`.

        Raises:
            MissingRequiredTopicError: If a topic named in
                ``config.required_topics`` is absent from the bag.
            InvalidSourceWindowError: If ``config.window`` is set and its
                clock does not match this bag's recording-time clock, or it
                does not overlap the bag's recording-time range at all.
        """
        available = self._available_topics()
        self._check_required_topics(available)

        topic_kinds = self._configured_topic_kinds()
        calibration_ids = _ros_common.calibration_ids_by_sensor(self.read_calibration())
        self._warnings = []
        self._content_hash = _ros_common.StreamingContentHash()
        counters: dict[str, int] = {}
        start_ns, stop_ns = _ros_common.resolve_window_bounds(
            self._config, open_reader=lambda: Reader(self._config.path), topic_kinds=topic_kinds
        )

        with Reader(self._config.path) as reader:
            connections = [
                connection for connection in reader.connections if connection.topic in topic_kinds
            ]
            for connection, bag_timestamp, rawdata in reader.messages(
                connections=connections, start=start_ns, stop=stop_ns
            ):
                topic = connection.topic
                self._content_hash.update(
                    topic=topic, timestamp_nanoseconds=bag_timestamp, rawdata=rawdata
                )
                index = counters.get(topic, 0)
                counters[topic] = index + 1
                try:
                    message = self._typestore.deserialize_ros1(rawdata, connection.msgtype)
                    observation = self._decode(
                        kind=topic_kinds[topic],
                        topic=topic,
                        msgtype=connection.msgtype,
                        index=index,
                        message=message,
                        bag_timestamp_nanoseconds=bag_timestamp,
                        calibration_ids=calibration_ids,
                    )
                # Só ValueError é o erro contratual de "conteúdo não decodificável";
                # KeyError/AttributeError indicam erro estrutural ou de programação e
                # devem abortar a leitura em vez de virar um warning por mensagem.
                except ValueError as error:
                    self._warnings.append(
                        SourceAdapterWarning(topic=topic, message_index=index, reason=str(error))
                    )
                    continue
                yield observation

    def warnings(self) -> Sequence[SourceAdapterWarning]:
        """Return warnings from the most recent :meth:`read_observations` call.

        Returns:
            Accumulated warnings, in the order they occurred.
        """
        return tuple(self._warnings)

    def content_hash(self) -> str | None:
        """Return the content hash of exactly what the most recent read actually read.

        Unlike :func:`~contextmap.ingestion.sequence_provenance.compute_source_content_hash`
        (a separate, full pass over the whole source), this is accumulated
        incrementally while :meth:`read_observations` reads each message, so
        it covers only the configured window when one is set — never the
        whole 24 GB bag just to declare provenance for a 90 s slice of it
        (issue #506).

        Returns:
            ``"sha256:<hex digest>"``, or ``None`` if
            :meth:`read_observations` was never called.
        """
        return self._content_hash.hexdigest()

    def read_calibration(self) -> CalibrationSet | None:
        """Return merged configured and source camera calibration.

        Every message on ``camera_info`` must describe the same calibration;
        a change is rejected because the canonical calibration is static for
        the sequence. Static extrinsics supplied in
        ``SourceAdapterConfig.calibration`` are preserved in the result.

        Calibration is deliberately NOT bounded by ``config.window``: unlike
        ``read_observations()``/``content_hash()`` (issue #506, cost
        proportional to the window), ``camera_info`` is treated as global,
        source-wide metadata that describes the whole bag, not a per-window
        artifact — see ``docs/adapters.md``. The scan is performed at most
        once per adapter instance and the result cached, so calling this
        method again (as ``read_observations()`` does internally, and as the
        runtime does again afterwards) never re-scans the bag.

        Returns:
            The merged calibration, or ``None`` when neither configuration
            nor the source provides calibration.
        """
        if not self._calibration_read:
            self._calibration_cache = self._discover_calibration()
            self._calibration_read = True
        return self._calibration_cache

    def _discover_calibration(self) -> CalibrationSet | None:
        """Scan the whole bag's ``camera_info`` topic once; never windowed."""
        topic = self._config.topics.camera_info
        discovered: list[CalibrationEntry] = []
        if topic is not None:
            with Reader(self._config.path) as reader:
                connections = [
                    connection for connection in reader.connections if connection.topic == topic
                ]
                for connection, _timestamp, rawdata in reader.messages(connections=connections):
                    message = self._typestore.deserialize_ros1(rawdata, connection.msgtype)
                    discovered.append(self._camera_calibration_entry(message, topic))
        return _ros_common.merge_calibration(self._config.calibration, tuple(discovered))

    def _camera_calibration_entry(self, message: Any, topic: str) -> CalibrationEntry:
        sensor_id = SensorId(_ros_common.sanitize_topic(self._config.topics.rgb or topic))
        frame_id = FrameId(message.header.frame_id)
        camera_model = _ros_common.build_camera_model(
            width=message.width,
            height=message.height,
            k_matrix=message.K,
            distortion_model_name=message.distortion_model,
            distortion_coefficients=message.D,
        )
        calibration_id = CalibrationReferenceId(f"{sensor_id}-calib")
        return CalibrationEntry(
            calibration_id=calibration_id,
            sensor_id=sensor_id,
            frame_id=frame_id,
            camera_model=camera_model,
            provenance=CalibrationProvenance(
                source_type=_SOURCE_TYPE,
                source_path=self._config.path,
                original_values={
                    "distortion_model": message.distortion_model,
                    "D": [float(value) for value in message.D],
                },
            ),
            content_hash=compute_content_hash(
                sensor_id=sensor_id, frame_id=frame_id, camera_model=camera_model
            ),
        )

    def _available_topics(self) -> frozenset[str]:
        with Reader(self._config.path) as reader:
            return frozenset(connection.topic for connection in reader.connections)

    def _configured_topic_kinds(self) -> dict[str, str]:
        mapping = self._config.topics
        kinds: dict[str, str] = {}
        for kind, topic in (
            ("rgb", mapping.rgb),
            ("lidar", mapping.lidar),
            ("imu", mapping.imu),
            ("pose", mapping.pose),
        ):
            if topic is not None:
                kinds[topic] = kind
        return kinds

    def _check_required_topics(self, available: frozenset[str]) -> None:
        mapping = self._config.topics
        topic_by_field = {
            "rgb": mapping.rgb,
            "camera_info": mapping.camera_info,
            "lidar": mapping.lidar,
            "imu": mapping.imu,
            "pose": mapping.pose,
        }
        missing = [
            field_name
            for field_name in self._config.required_topics
            if topic_by_field[field_name] is None or topic_by_field[field_name] not in available
        ]
        if missing:
            raise MissingRequiredTopicError(
                f"required topics missing from ros1 bag {self._config.path!r}: {sorted(missing)}"
            )

    def _decode(
        self,
        *,
        kind: str,
        topic: str,
        msgtype: str,
        index: int,
        message: Any,
        bag_timestamp_nanoseconds: int,
        calibration_ids: dict[SensorId, CalibrationReferenceId],
    ) -> SourceObservation:
        observation_id = SourceObservationId(f"{_ros_common.sanitize_topic(topic)}-{index:06d}")
        sensor_id = SensorId(_ros_common.sanitize_topic(topic))
        provenance = SourceProvenance(
            source_type=_SOURCE_TYPE,
            source_path=self._config.path,
            source_topic=topic,
            source_message_index=index,
            raw_metadata={
                "msgtype": msgtype,
                "bag_timestamp_nanoseconds": bag_timestamp_nanoseconds,
            },
        )
        timestamp = _ros_common.stamp_to_timestamp(
            message.header.stamp, clock_id=self._config.resolved_timestamp_clock_id()
        )
        calibration_id = calibration_ids.get(sensor_id)

        if kind == "rgb":
            return _ros_common.decode_image(
                message,
                observation_id=observation_id,
                sensor_id=sensor_id,
                timestamp=timestamp,
                provenance=provenance,
                calibration_id=calibration_id,
            )
        if kind == "lidar":
            return _ros_common.decode_lidar(
                message,
                observation_id=observation_id,
                sensor_id=sensor_id,
                timestamp=timestamp,
                provenance=provenance,
                calibration_id=calibration_id,
            )
        if kind == "imu":
            return _ros_common.decode_imu(
                message,
                observation_id=observation_id,
                sensor_id=sensor_id,
                timestamp=timestamp,
                provenance=provenance,
                calibration_id=calibration_id,
            )
        if kind == "pose":
            return _ros_common.decode_pose(
                message,
                observation_id=observation_id,
                sensor_id=sensor_id,
                timestamp=timestamp,
                provenance=provenance,
                calibration_id=calibration_id,
            )
        raise ValueError(f"unhandled configured topic kind: {kind!r}")

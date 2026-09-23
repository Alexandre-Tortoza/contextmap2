"""Integration test: a windowed ingestion's SequenceArtifact declares its window (issue #506).

The issue's acceptance criteria asks for an *output* guarantee, not only an
adapter-level one: the resulting SequenceArtifact's provenance must
unambiguously declare which window it covers, and the source content hash
it records must cover only what was actually read, never the whole source.
Composing SourceWindow, SequenceArtifactWriter and SequenceProvenance this
way is itself an Ingestion-owned concern (SequenceProvenance.ingestion_config
is exactly the free-form slot documented for this); wiring a *real*
production run to always do so is the runtime composition root's job
(out of scope here), but the capability must make it possible and correct,
which this test demonstrates end to end with a real bag file (small,
synthetic content -- not the 24 GB corridor-02 bag) and real hashing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from contextmap.ingestion import (
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SourceAdapterConfig,
    SourceTopicMapping,
    SourceWindow,
)
from contextmap.ingestion.adapters.ros1_bag import Ros1BagSourceAdapter
from contextmap.ingestion.sequence_provenance import (
    SequenceProvenance,
    compute_configuration_hash,
    compute_source_content_hash,
)

_TS = get_typestore(Stores.ROS1_NOETIC)


def _header(seconds: int, frame_id: str) -> Any:
    stamp = _TS.types["builtin_interfaces/msg/Time"](sec=seconds, nanosec=0)
    return _TS.types["std_msgs/msg/Header"](seq=0, stamp=stamp, frame_id=frame_id)


def _build_bag(path: Path, *, image_bag_timestamps_ns: list[int]) -> None:
    types = _TS.types
    with Writer(path) as writer:
        image_conn = writer.add_connection(
            "/camera/image_raw", types["sensor_msgs/msg/Image"].__msgtype__, typestore=_TS
        )
        for index, bag_ts in enumerate(image_bag_timestamps_ns):
            msg = types["sensor_msgs/msg/Image"](
                header=_header(1000 + index, "front_camera_optical"),
                height=1,
                width=1,
                encoding="mono8",
                is_bigendian=0,
                step=1,
                data=np.array([index], dtype=np.uint8),
            )
            raw = _TS.serialize_ros1(msg, image_conn.msgtype)
            writer.write(image_conn, bag_ts, raw)


def test_windowed_sequence_artifact_declares_its_window_and_a_window_scoped_hash(
    tmp_path: Path,
) -> None:
    bag_path = tmp_path / "source.bag"
    _build_bag(
        bag_path,
        image_bag_timestamps_ns=[1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000],
    )
    base_config = SourceAdapterConfig(
        source_type="ros1_bag",
        path=str(bag_path),
        topics=SourceTopicMapping(rgb="/camera/image_raw"),
    )
    window = SourceWindow(
        clock_id=base_config.resolved_window_clock_id(), start_seconds=2.0, end_seconds=4.0
    )
    adapter = Ros1BagSourceAdapter(replace(base_config, window=window))

    ingestion_config: dict[str, object] = {
        "topics": {"rgb": "/camera/image_raw"},
        "window": {
            "clock_id": window.clock_id,
            "start_seconds": window.start_seconds,
            "end_seconds": window.end_seconds,
        },
    }
    output_dir = tmp_path / "sequence"
    with SequenceArtifactWriter(
        output_dir=output_dir,
        sequence_name="windowed-integration",
        artifact_id=SequenceArtifactId("windowed-integration-0001"),
    ) as writer:
        for observation in adapter.read_observations():
            writer.add_observation(observation)
        window_scoped_hash = adapter.content_hash()
        assert window_scoped_hash is not None
        writer.set_provenance(
            SequenceProvenance(
                source_type="ros1_bag",
                source_path=str(bag_path),
                source_content_hash=window_scoped_hash,
                ingestion_config=ingestion_config,
                configuration_hash=compute_configuration_hash(ingestion_config),
                adapter_type="ros1_bag",
            )
        )
        manifest = writer.finalize()

    assert manifest.observation_counts["image"] == 2  # only the windowed messages

    reader = SequenceArtifactReader(output_dir)
    assert reader.verify_integrity() == []
    provenance = reader.read_provenance()
    assert provenance is not None

    # The output unambiguously declares the window it covers.
    declared_window = provenance.ingestion_config["window"]
    assert declared_window == {
        "clock_id": window.clock_id,
        "start_seconds": window.start_seconds,
        "end_seconds": window.end_seconds,
    }

    # The declared source content hash covers only the window that was
    # actually read, never the whole source file.
    whole_file_hash = compute_source_content_hash(bag_path)
    assert provenance.source_content_hash == window_scoped_hash
    assert provenance.source_content_hash != whole_file_hash

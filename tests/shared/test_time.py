import dataclasses

import pytest

from contextmap.shared import SourceTimestamp


def test_to_float_seconds_combines_seconds_and_nanoseconds() -> None:
    timestamp = SourceTimestamp(seconds=10, nanoseconds=500_000_000, clock_id="system")

    assert timestamp.to_float_seconds() == pytest.approx(10.5)


def test_rejects_nanoseconds_out_of_range() -> None:
    with pytest.raises(ValueError, match="nanoseconds"):
        SourceTimestamp(seconds=0, nanoseconds=1_000_000_000, clock_id="system")

    with pytest.raises(ValueError, match="nanoseconds"):
        SourceTimestamp(seconds=0, nanoseconds=-1, clock_id="system")


def test_is_immutable() -> None:
    timestamp = SourceTimestamp(seconds=0, nanoseconds=0, clock_id="system")

    with pytest.raises(dataclasses.FrozenInstanceError):
        timestamp.seconds = 1  # type: ignore[misc]


def test_different_clock_ids_are_not_assumed_comparable() -> None:
    bag_time = SourceTimestamp(seconds=100, nanoseconds=0, clock_id="ros1_bag:/clock")
    sensor_time = SourceTimestamp(seconds=100, nanoseconds=0, clock_id="sensor:imu0")

    assert bag_time != sensor_time
    assert bag_time.clock_id != sensor_time.clock_id

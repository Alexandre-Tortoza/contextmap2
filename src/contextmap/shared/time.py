"""Time primitive shared by every capability that consumes sensor data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_MAX_NANOSECONDS = 999_999_999


@dataclass(frozen=True)
class SourceTimestamp:
    """A point in time expressed in an explicit, named clock domain.

    Timestamps from different sources are not guaranteed to share a clock.
    Two ``SourceTimestamp`` values are only meaningfully comparable when
    their ``clock_id`` matches; converting a source's native clock into a
    normalized/synchronized timeline is a decision owned by ingestion
    (see ``docs/PIPELINE.md``, temporal index section), not by this type.

    Attributes:
        seconds: Whole seconds since the clock's epoch. May be negative for
            clocks whose epoch is not the same as the observation.
        nanoseconds: Sub-second remainder, in the range [0, 999_999_999].
        clock_id: Identifier of the clock domain this timestamp belongs to,
            e.g. "ros1_bag:/clock", "ros2_bag:system_time",
            "dataset:frame_index".
    """

    seconds: int
    nanoseconds: int
    clock_id: str

    def __post_init__(self) -> None:
        """Validate the sub-second remainder.

        Raises:
            ValueError: If ``nanoseconds`` is outside [0, 999_999_999].
        """
        if not 0 <= self.nanoseconds <= _MAX_NANOSECONDS:
            raise ValueError(
                f"nanoseconds must be in [0, {_MAX_NANOSECONDS}], got {self.nanoseconds}"
            )

    def to_float_seconds(self) -> float:
        """Return this timestamp as a single float number of seconds.

        This conversion loses the ability to detect clock-domain mismatches;
        prefer comparing two ``SourceTimestamp`` values with matching
        ``clock_id`` instead of comparing the float result directly.

        Returns:
            The timestamp as seconds since the clock's epoch.
        """
        return self.seconds + self.nanoseconds / 1_000_000_000

    def total_nanoseconds(self) -> int:
        """Return this timestamp as an exact integer number of nanoseconds.

        Returns:
            Nanoseconds since the clock domain's epoch, without floating-point
            precision loss.
        """
        return self.seconds * 1_000_000_000 + self.nanoseconds

    def to_record(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable record of this timestamp.

        Returns:
            ``{"seconds", "nanoseconds", "clock_id"}``; the exact integers, so
            no precision is lost and the clock domain travels with the value.
        """
        return {
            "seconds": self.seconds,
            "nanoseconds": self.nanoseconds,
            "clock_id": self.clock_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SourceTimestamp:
        """Rebuild a timestamp from a record produced by :meth:`to_record`.

        Args:
            record: The persisted record.

        Returns:
            The validated timestamp.

        Raises:
            ValueError: If ``nanoseconds`` is out of range.
        """
        return cls(
            seconds=record["seconds"],
            nanoseconds=record["nanoseconds"],
            clock_id=record["clock_id"],
        )

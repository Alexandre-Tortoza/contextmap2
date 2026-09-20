"""Errors raised when the inputs of a Sensor Association step are incompatible."""

from __future__ import annotations


class AssociationInputError(ValueError):
    """The map, trajectory, calibration, observation or image chain do not fit together.

    Raised for wiring and lineage problems that no amount of retrying fixes: a map
    built from another trajectory, a frame that does not match, a calibration that
    is missing or different, or an image transformation that cannot be reproduced.
    A frame whose pose the lookup policy rejects is *not* an error; it is data.
    """

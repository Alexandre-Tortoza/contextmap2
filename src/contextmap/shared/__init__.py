"""Cross-capability primitives with no single clear domain owner.

Every type exported here must satisfy the admission criteria documented in
``docs/shared-primitives.md`` before it is added: it must be reused by more
than one capability with identical semantics, have no capability that is a
clearly superior domain owner, and stay small, stable, and backend-neutral.
"""

from contextmap.shared.geometry import (
    UNIT_QUATERNION_TOLERANCE,
    Quaternion,
    Vector3,
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_norm,
)
from contextmap.shared.time import SourceTimestamp

__all__ = [
    "UNIT_QUATERNION_TOLERANCE",
    "Quaternion",
    "SourceTimestamp",
    "Vector3",
    "is_unit_quaternion",
    "normalize_quaternion",
    "quaternion_norm",
]

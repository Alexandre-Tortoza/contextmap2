"""Cross-capability primitives with no single clear domain owner.

Every type exported here must satisfy the admission criteria documented in
``docs/shared-primitives.md`` before it is added: it must be reused by more
than one capability with identical semantics, have no capability that is a
clearly superior domain owner, and stay small, stable, and backend-neutral.
"""

from contextmap.shared.geometry import (
    UNIT_QUATERNION_TOLERANCE,
    Quaternion,
    RotationMatrix,
    Vector3,
    compose_rigid,
    invert_rigid,
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_angle_between,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_norm,
    quaternion_to_rotation_matrix,
    rotate_vector,
)
from contextmap.shared.time import SourceTimestamp

__all__ = [
    "UNIT_QUATERNION_TOLERANCE",
    "Quaternion",
    "RotationMatrix",
    "SourceTimestamp",
    "Vector3",
    "compose_rigid",
    "invert_rigid",
    "is_unit_quaternion",
    "normalize_quaternion",
    "quaternion_angle_between",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_norm",
    "quaternion_to_rotation_matrix",
    "rotate_vector",
]

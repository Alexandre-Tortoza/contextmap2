"""Representation-space fingerprint and the rule for when vectors are comparable.

:class:`~contextmap.point_representation.models.PointRepresentation` carries an
opaque ``representation_space_id``; this module defines what it identifies. The
central rule: two representations are comparable (distance, similarity,
averaging, indexing) only when their spaces have the *exact same* fingerprint,
never merely because they share a dimension. A deterministic descriptor and a
learned encoder of the same dimension are not compatible, and neither are the
same encoder under two support radii.
"""

from __future__ import annotations

import hashlib
import json

from contextmap.point_representation.models import PointRepresentation, RepresentationSpace
from contextmap.point_representation.serialization import encode_representation_space


class RepresentationSpaceMismatchError(ValueError):
    """Raised when two incompatible representations or spaces are treated as comparable."""


def representation_space_fingerprint(space: RepresentationSpace) -> str:
    """Compute the deterministic compatibility fingerprint of a space.

    This is the value encoders stamp into every ``representation_space_id``
    they produce. Two spaces yield the same fingerprint if and only if every
    field, including the support policy, is equal.

    Args:
        space: The representation space to fingerprint.

    Returns:
        ``"sha256:<hex digest>"`` of the canonical JSON encoding.
    """
    canonical = json.dumps(
        encode_representation_space(space), sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_compatible_representation_spaces(
    left: RepresentationSpace, right: RepresentationSpace
) -> None:
    """Require that two spaces are the same identity.

    Raises:
        RepresentationSpaceMismatchError: If the fingerprints differ.
    """
    _require_same_fingerprint(
        representation_space_fingerprint(left), representation_space_fingerprint(right)
    )


def ensure_compatible_representations(
    left: PointRepresentation, right: PointRepresentation
) -> None:
    """Require that two representations live in the same space.

    Raises:
        RepresentationSpaceMismatchError: If their ``representation_space_id`` differ.
    """
    _require_same_fingerprint(left.representation_space_id, right.representation_space_id)


def _require_same_fingerprint(left: str, right: str) -> None:
    if left != right:
        raise RepresentationSpaceMismatchError(
            f"representation spaces are not compatible: {left} != {right}; equal dimensionality "
            "does not make vectors comparable"
        )

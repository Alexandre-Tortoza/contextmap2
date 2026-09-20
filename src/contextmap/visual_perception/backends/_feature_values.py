"""Shared numerical validation for concrete feature-extraction adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray


def validate_and_normalize_feature_values(
    array: NDArray[Any],
    *,
    l2_normalize: bool,
    error_type: type[Exception],
) -> tuple[NDArray[Any], str]:
    """Reject invalid model values and apply truthful L2 normalization.

    A zero vector cannot satisfy an ``l2`` normalization declaration. Rejecting
    it here also prevents invalid payloads from reaching an artifact sink before
    the later evaluation boundary can inspect them.
    """
    import numpy as np

    if not bool(np.isfinite(array).all()):
        raise error_type("feature payload values must all be finite")
    if not l2_normalize:
        return array, "none"

    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if bool(np.any(norms == 0)):
        raise error_type("cannot l2-normalize a zero-norm feature vector")
    return np.divide(array, norms), "l2"

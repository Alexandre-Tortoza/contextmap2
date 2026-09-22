"""Plain-Python vector arithmetic shared by the appearance and representation channels.

Vectors are tuples of floats at every public boundary, so no tensor type or NumPy array leaks into a
contract, and reading or writing a resolution never needs NumPy. The dimensions involved (a few
hundred to a few thousand) and the number of vectors per entity make the loops cheap; every sum runs
in a fixed order, so results are reproducible bit for bit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Inner product of two vectors of the same dimension."""
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def norm(values: Sequence[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(dot(values, values))


def unit(values: Sequence[float], *, what: str) -> tuple[float, ...]:
    """Scale a vector to unit length.

    Args:
        values: The vector.
        what: What the vector is, for the error message.

    Returns:
        The vector divided by its norm.

    Raises:
        ValueError: If a component is not finite or the vector has no direction (zero norm), which
            is corrupt evidence and never a valid embedding.
    """
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{what} has a component that is not finite")
    length = norm(values)
    if length == 0.0:
        raise ValueError(f"{what} is the zero vector and has no direction")
    return tuple(value / length for value in values)


def mean(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Component-wise mean of vectors of the same dimension."""
    count = len(vectors)
    return tuple(
        math.fsum(vector[index] for vector in vectors) / count for index in range(len(vectors[0]))
    )


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two vectors that both have a direction, clamped into ``[-1, 1]``."""
    value = dot(left, right) / (norm(left) * norm(right))
    return max(-1.0, min(1.0, value))

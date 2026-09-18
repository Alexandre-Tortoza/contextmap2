"""Canonical embedding space contract and cross-backend compatibility rule.

:class:`~contextmap.visual_perception.models.VisualFeature` deliberately
carries only an opaque ``embedding_space_id`` reference, never a raw
tensor or backend SDK object — the same pattern Ingestion used for
``calibration_id`` before the full calibration contract existed (#41).
This module defines what that reference actually identifies: an
:class:`EmbeddingSpace` (model family, checkpoint, layer, dimension,
normalization) and its deterministic compatibility fingerprint.

The central rule this module enforces: two features are comparable
(cosine similarity, averaging, indexing, scoring) only when their
embedding spaces are the *exact same* fingerprint — never merely because
they happen to share a dimension. DINOv3 and CLIP embeddings of the same
dimension are not compatible; two CLIP checkpoints are not automatically
compatible with each other either. See
``src/contextmap/visual_perception/docs/embedding_space.md`` for worked
examples.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from contextmap.visual_perception.models import VisualFeature

SCHEMA_VERSION = "0.1.0"
"""Embedding space encoding schema version."""


class EmbeddingSpaceMismatchError(ValueError):
    """Raised when two incompatible features/embedding spaces are treated as comparable."""


@dataclass(frozen=True, kw_only=True)
class EmbeddingSpace:
    """Identity of the vector space a :class:`VisualFeature`'s payload lives in.

    Two spaces are the same identity if and only if every field here is
    equal — a coincidentally equal ``dimension`` alone never implies
    compatibility. See :func:`embedding_space_fingerprint` for the
    deterministic string this identity reduces to for
    ``VisualFeature.embedding_space_id``.

    Attributes:
        family: Model family, e.g. ``"dinov2"``, ``"dinov3"``, ``"clip"``,
            ``"alphaclip"``.
        model: Model/architecture identity, e.g. ``"vit-l-14"``.
        version: Backend/model version string.
        checkpoint: Specific checkpoint identity, when the family ships
            more than one (e.g. a CLIP checkpoint name/hash). ``None``
            when the family has exactly one relevant checkpoint.
        layer: Which layer/projection head this space's vectors come
            from, when relevant (e.g. ``"patch_embeddings"``,
            ``"projection_head"``). ``None`` when not applicable.
        dimension: Vector dimensionality.
        normalization: Normalization applied before persistence, e.g.
            ``"l2"``, ``"none"``. ``None`` when not documented.
    """

    family: str
    model: str
    version: str
    checkpoint: str | None = None
    layer: str | None = None
    dimension: int
    normalization: str | None = None

    def __post_init__(self) -> None:
        """Validate ``dimension`` is positive.

        Raises:
            ValueError: If ``dimension`` is not positive.
        """
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")


def encode_embedding_space(space: EmbeddingSpace) -> dict[str, Any]:
    """Encode an :class:`EmbeddingSpace` into a JSON-serializable dict."""
    return {
        "family": space.family,
        "model": space.model,
        "version": space.version,
        "checkpoint": space.checkpoint,
        "layer": space.layer,
        "dimension": space.dimension,
        "normalization": space.normalization,
    }


def decode_embedding_space(record: dict[str, Any]) -> EmbeddingSpace:
    """Decode an :class:`EmbeddingSpace` from :func:`encode_embedding_space`'s output."""
    return EmbeddingSpace(
        family=record["family"],
        model=record["model"],
        version=record["version"],
        checkpoint=record["checkpoint"],
        layer=record["layer"],
        dimension=record["dimension"],
        normalization=record["normalization"],
    )


def embedding_space_fingerprint(space: EmbeddingSpace) -> str:
    """Compute the deterministic compatibility fingerprint of an embedding space.

    This is the value backends stamp into every
    :attr:`~contextmap.visual_perception.models.VisualFeature.embedding_space_id`
    they produce. Two spaces yield the same fingerprint if and only if
    every field of :class:`EmbeddingSpace` is equal.

    Args:
        space: The embedding space to fingerprint.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    payload = json.dumps(encode_embedding_space(space), sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def ensure_compatible_embedding_spaces(a: EmbeddingSpace, b: EmbeddingSpace) -> None:
    """Validate two embedding spaces are the exact same identity.

    Args:
        a: First embedding space.
        b: Second embedding space.

    Raises:
        EmbeddingSpaceMismatchError: If ``a`` and ``b`` are not the same
            identity — regardless of whether they share a ``dimension``.
    """
    fingerprint_a = embedding_space_fingerprint(a)
    fingerprint_b = embedding_space_fingerprint(b)
    if fingerprint_a != fingerprint_b:
        raise EmbeddingSpaceMismatchError(
            f"incompatible embedding spaces: {a.family}/{a.model} ({fingerprint_a}) vs "
            f"{b.family}/{b.model} ({fingerprint_b})"
        )


def ensure_compatible_features(a: VisualFeature, b: VisualFeature) -> None:
    """Validate two persisted features reference the exact same embedding space.

    Use this at the point a downstream operation (cosine similarity,
    averaging, indexing, scoring) is about to treat two features'
    payloads as comparable — the common case where only each feature's
    opaque ``embedding_space_id`` is available, not the full
    :class:`EmbeddingSpace` that produced it.

    Args:
        a: First feature.
        b: Second feature.

    Raises:
        EmbeddingSpaceMismatchError: If ``a.embedding_space_id`` and
            ``b.embedding_space_id`` differ.
    """
    if a.embedding_space_id != b.embedding_space_id:
        raise EmbeddingSpaceMismatchError(
            f"incompatible embedding spaces: feature {a.feature_id!r} "
            f"({a.embedding_space_id!r}) vs feature {b.feature_id!r} ({b.embedding_space_id!r})"
        )

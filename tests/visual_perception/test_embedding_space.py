import pytest

from contextmap.visual_perception import (
    BackendProvenance,
    EmbeddingSpace,
    EmbeddingSpaceMismatchError,
    FeatureId,
    FeatureScope,
    VisualFeature,
    decode_embedding_space,
    embedding_space_fingerprint,
    encode_embedding_space,
    ensure_compatible_embedding_spaces,
    ensure_compatible_features,
)

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="feature_extractor", provider="fake", model="fake", version="0.1"
)


def _dinov3_space() -> EmbeddingSpace:
    return EmbeddingSpace(family="dinov3", model="vit-l-14", version="1", dimension=1024)


def _clip_space(*, checkpoint: str = "openai/clip-vit-l-14") -> EmbeddingSpace:
    return EmbeddingSpace(
        family="clip", model="vit-l-14", version="1", checkpoint=checkpoint, dimension=1024
    )


def _feature(space_id: str, feature_id: str = "feature-0000") -> VisualFeature:
    return VisualFeature(
        feature_id=FeatureId(feature_id),
        scope=FeatureScope.GLOBAL,
        embedding_space_id=space_id,
        shape=(1024,),
        dtype="float32",
        payload_reference=f"features/{feature_id}.bin",
        provenance=_PROVENANCE,
    )


def test_dimension_must_be_positive() -> None:
    with pytest.raises(ValueError, match="dimension"):
        EmbeddingSpace(family="dinov3", model="vit-l-14", version="1", dimension=0)


def test_same_fields_produce_the_same_fingerprint() -> None:
    a = _dinov3_space()
    b = _dinov3_space()
    assert embedding_space_fingerprint(a) == embedding_space_fingerprint(b)


def test_different_family_same_dimension_is_not_compatible() -> None:
    """DINOv3 embedding != CLIP embedding, even with the same dimension."""
    dinov3 = _dinov3_space()
    clip = _clip_space()
    assert dinov3.dimension == clip.dimension
    assert embedding_space_fingerprint(dinov3) != embedding_space_fingerprint(clip)

    with pytest.raises(EmbeddingSpaceMismatchError):
        ensure_compatible_embedding_spaces(dinov3, clip)


def test_different_checkpoint_same_family_is_not_automatically_compatible() -> None:
    """CLIP checkpoint A != automatically compatible with checkpoint B."""
    checkpoint_a = _clip_space(checkpoint="openai/clip-vit-l-14")
    checkpoint_b = _clip_space(checkpoint="laion/clip-vit-l-14")

    with pytest.raises(EmbeddingSpaceMismatchError):
        ensure_compatible_embedding_spaces(checkpoint_a, checkpoint_b)


def test_ensure_compatible_embedding_spaces_accepts_identical_spaces() -> None:
    ensure_compatible_embedding_spaces(_dinov3_space(), _dinov3_space())


def test_ensure_compatible_features_compares_embedding_space_id_only() -> None:
    dinov3_id = embedding_space_fingerprint(_dinov3_space())
    clip_id = embedding_space_fingerprint(_clip_space())

    ensure_compatible_features(_feature(dinov3_id, "a"), _feature(dinov3_id, "b"))

    with pytest.raises(EmbeddingSpaceMismatchError):
        ensure_compatible_features(_feature(dinov3_id, "a"), _feature(clip_id, "b"))


def test_encode_decode_round_trip() -> None:
    space = _clip_space()
    decoded = decode_embedding_space(encode_embedding_space(space))
    assert decoded == space
    assert embedding_space_fingerprint(decoded) == embedding_space_fingerprint(space)

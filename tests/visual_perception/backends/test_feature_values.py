import numpy as np

from contextmap.visual_perception.backends._feature_values import (
    validate_and_normalize_feature_values,
)


def test_l2_normalization_accumulates_float16_values_in_float32() -> None:
    array = np.array([[300.0, 300.0]], dtype=np.float16)

    normalized, policy = validate_and_normalize_feature_values(
        array,
        l2_normalize=True,
        error_type=ValueError,
    )

    assert policy == "l2"
    assert normalized.dtype == np.float16
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(
        normalized,
        np.array([[2**-0.5, 2**-0.5]], dtype=np.float16),
        rtol=1e-3,
    )

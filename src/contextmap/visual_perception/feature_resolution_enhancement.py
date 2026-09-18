"""Validated boundary for optional dense feature-resolution enhancement.

The stage consumes and produces the same :class:`DenseFeatureMap` contract.
Concrete learned implementations remain backend details; this module only
enforces artifact identity, compatibility, spatial, integrity, and provenance
invariants common to every implementation.
"""

from __future__ import annotations

from contextmap.visual_perception.dense_region_association import DenseFeatureMap
from contextmap.visual_perception.ports import FeatureResolutionEnhancement


class FeatureResolutionEnhancementError(ValueError):
    """Raised when an enhancement backend returns an invalid dense map."""


def enhance_feature_resolution(
    source: DenseFeatureMap, *, enhancer: FeatureResolutionEnhancement
) -> DenseFeatureMap:
    """Run one selected enhancer and validate its output against the source.

    Args:
        source: Immutable native dense map selected as the upstream artifact.
        enhancer: Explicitly selected enhancement backend.

    Returns:
        A separately identified enhanced dense map consumable anywhere a
        native :class:`DenseFeatureMap` is accepted.

    Raises:
        FeatureResolutionEnhancementError: If the backend omits or falsifies
            source/output lineage, does not increase resolution, reuses source
            identities, or changes source-image/compatibility semantics
            implicitly.
    """
    output = enhancer.enhance(source)
    lineage = output.enhancement
    if lineage is None:
        raise FeatureResolutionEnhancementError(
            "enhancement output must include resolution-enhancement lineage"
        )

    expected_source_lineage = (
        source.feature.feature_id,
        source.source_artifact_id,
        source.feature.payload_reference,
        source.feature.embedding_space_id,
        source.feature.dtype,
        source.feature.normalization,
        source.sampling.coordinate_transform_id,
        (source.sampling.grid_width, source.sampling.grid_height),
        (source.sampling.source_image_width, source.sampling.source_image_height),
    )
    actual_source_lineage = (
        lineage.source_feature_id,
        lineage.source_artifact_id,
        lineage.source_payload_reference,
        lineage.source_embedding_space_id,
        lineage.source_dtype,
        lineage.source_normalization,
        lineage.source_coordinate_transform_id,
        lineage.input_grid_size,
        lineage.source_image_size,
    )
    if actual_source_lineage != expected_source_lineage:
        raise FeatureResolutionEnhancementError(
            "enhancement lineage does not identify the exact source dense map"
        )

    backend = enhancer.backend_provenance()
    if lineage.backend != backend or output.feature.provenance != backend:
        raise FeatureResolutionEnhancementError(
            "enhancement lineage/output provenance does not match the selected backend"
        )
    output_grid_size = (output.sampling.grid_width, output.sampling.grid_height)
    output_image_size = (
        output.sampling.source_image_width,
        output.sampling.source_image_height,
    )
    if lineage.output_grid_size != output_grid_size:
        raise FeatureResolutionEnhancementError(
            "enhancement lineage output grid does not match DenseFeatureMap sampling"
        )
    if lineage.source_image_size != output_image_size:
        raise FeatureResolutionEnhancementError(
            "enhancement output source image dimensions differ from the source map"
        )
    if lineage.output_embedding_space_id != output.feature.embedding_space_id:
        raise FeatureResolutionEnhancementError(
            "enhancement lineage output embedding space does not match the output feature"
        )

    input_width, input_height = lineage.input_grid_size
    output_width, output_height = lineage.output_grid_size
    if (
        output_width < input_width
        or output_height < input_height
        or (output_width == input_width and output_height == input_height)
    ):
        raise FeatureResolutionEnhancementError(
            "resolution enhancement must increase at least one grid dimension without "
            "decreasing the other"
        )
    if output.feature.feature_id == source.feature.feature_id:
        raise FeatureResolutionEnhancementError(
            "enhancement output must have a distinct feature identity"
        )
    if output.source_artifact_id == source.source_artifact_id:
        raise FeatureResolutionEnhancementError(
            "enhancement output must belong to a distinct immutable artifact"
        )
    if output.feature.payload_reference == source.feature.payload_reference:
        raise FeatureResolutionEnhancementError(
            "enhancement output must use a distinct payload reference"
        )
    if (
        output.feature.embedding_space_id == source.feature.embedding_space_id
        and output.feature.normalization != source.feature.normalization
    ):
        raise FeatureResolutionEnhancementError(
            "changed normalization requires a distinct output embedding space"
        )
    return output

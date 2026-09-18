"""Deterministic identity generation for Visual Perception evidence.

Region, feature, and claim identities are local to the
:class:`~contextmap.visual_perception.models.PerceptionResult` that
produced them; a ``PerceptionResult`` identity is local to the
:class:`~contextmap.visual_perception.models.PerceptionRun` that produced
it. Every generator here is a pure function of its inputs — the same
inputs always produce the same identity — rather than a random value, so
within-run evidence stays reproducible across repeated construction and
survives serialization round-trips without needing a separate identity
registry. See ``src/contextmap/visual_perception/docs/identity.md`` for
the full traceability chain this supports.
"""

from __future__ import annotations

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.models import (
    ClaimId,
    FeatureId,
    PerceptionResultId,
    PerceptionRunId,
    RegionId,
)


def perception_result_id_for(
    *, run_id: PerceptionRunId, source_observation_id: SourceObservationId
) -> PerceptionResultId:
    """Compute the deterministic identity of a run's result for one observation.

    A given ``(run_id, source_observation_id)`` pair always produces the
    same identity — there is exactly one ``PerceptionResult`` per run per
    observation, so this pair is already a natural, stable key.

    Args:
        run_id: The run producing the result.
        source_observation_id: The physical observation being processed.

    Returns:
        The deterministic result identity.
    """
    return PerceptionResultId(f"{run_id}--{source_observation_id}")


def region_id_for(*, result_id: PerceptionResultId, index: int) -> RegionId:
    """Compute the deterministic identity of the ``index``-th region in a result.

    Args:
        result_id: The owning result.
        index: Zero-based position among the regions discovered for this
            result, in the discovery backend's own deterministic order.

    Returns:
        The deterministic region identity.
    """
    return RegionId(f"{result_id}--region-{index:04d}")


def feature_id_for(*, result_id: PerceptionResultId, index: int) -> FeatureId:
    """Compute the deterministic identity of the ``index``-th feature in a result.

    Args:
        result_id: The owning result.
        index: Zero-based position among the features produced for this
            result, in the extraction backend's own deterministic order.

    Returns:
        The deterministic feature identity.
    """
    return FeatureId(f"{result_id}--feature-{index:04d}")


def claim_id_for(*, result_id: PerceptionResultId, index: int) -> ClaimId:
    """Compute the deterministic identity of the ``index``-th claim in a result.

    Args:
        result_id: The owning result.
        index: Zero-based position among the claims produced for this
            result, in the interpretation backend's own deterministic
            order.

    Returns:
        The deterministic claim identity.
    """
    return ClaimId(f"{result_id}--claim-{index:04d}")

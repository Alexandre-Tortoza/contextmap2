"""In-memory stand-ins for the vector sources of the appearance and representation channels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from mapping_builders import make_interpreter

from contextmap.entity_resolution import LoadedFeature
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_mapping import EntityFeatureRef
from contextmap.visual_perception import (
    FeatureId,
    FeatureScope,
    PerceptionRunId,
    RegionId,
    VisualFeature,
    perception_result_id_for,
)

RUN = PerceptionRunId("perception-run-0001")
SPACE_A = "sha256:embedding-space-a"
SPACE_B = "sha256:embedding-space-b"


def frame(second: int) -> str:
    return f"frame-{second:04d}"


def feature_ref(
    second: int,
    index: int = 0,
    *,
    space: str = SPACE_A,
    scope: FeatureScope = FeatureScope.REGION,
    run: PerceptionRunId = RUN,
) -> EntityFeatureRef:
    """A feature of the region of one physical frame, whose result id is derived, not parsed."""
    return EntityFeatureRef(
        perception_run_id=run,
        perception_result_id=perception_result_id_for(
            run_id=run, source_observation_id=SourceObservationId(frame(second))
        ),
        feature_id=FeatureId(f"feature-{second:04d}-{index:02d}"),
        embedding_space_id=space,
        scope=scope,
        region_id=RegionId("region-0001") if scope is FeatureScope.REGION else None,
    )


def visual_feature(
    ref: EntityFeatureRef, *, space: str | None = None, dimension: int
) -> VisualFeature:
    return VisualFeature(
        feature_id=ref.feature_id,
        scope=ref.scope,
        embedding_space_id=space or ref.embedding_space_id,
        shape=(dimension,),
        dtype="float32",
        payload_reference=f"features/{ref.feature_id}.npy",
        provenance=make_interpreter(),
        region_id=ref.region_id,
        normalization="l2",
    )


class InMemoryFeatureSource:
    """Serves vectors by feature id and counts how often each was loaded."""

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        *,
        reported_space: Mapping[str, str] | None = None,
    ) -> None:
        self._vectors = {
            key: tuple(float(item) for item in value) for key, value in vectors.items()
        }
        self._reported_space = dict(reported_space or {})
        self.loads: list[str] = []

    def load(
        self, feature: EntityFeatureRef, physical_observation_id: SourceObservationId
    ) -> LoadedFeature:
        key = str(feature.feature_id)
        self.loads.append(key)
        values = self._vectors[key]
        return LoadedFeature(
            feature=visual_feature(
                feature, space=self._reported_space.get(key), dimension=len(values)
            ),
            values=values,
        )

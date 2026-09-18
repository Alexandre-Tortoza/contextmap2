"""Public contract for the Visual Perception capability.

Visual Perception turns one physical
:class:`~contextmap.ingestion.SourceObservation` into backend-agnostic
visual evidence (regions, features, semantic claims) for one configured
:class:`PerceptionRun`, without deciding persistent 3D entity identity,
final semantic meaning, or 2D→3D projection. See
``src/contextmap/visual_perception/docs/README.md`` for the full
capability documentation.
"""

from contextmap.visual_perception.evidence_set import (
    EvidenceSetError,
    ObservationEvidence,
    PerceptionEvidenceSet,
)
from contextmap.visual_perception.identity import (
    claim_id_for,
    feature_id_for,
    perception_result_id_for,
    region_id_for,
)
from contextmap.visual_perception.models import (
    BackendProvenance,
    BoundingBox2D,
    ClaimId,
    FeatureId,
    FeatureScope,
    HypothesisRole,
    PerceptionResult,
    PerceptionResultId,
    PerceptionRun,
    PerceptionRunId,
    PreparedImage,
    Region2D,
    RegionId,
    SceneContext,
    SemanticClaim,
    SemanticSupport,
    VisualFeature,
)
from contextmap.visual_perception.ports import (
    FeatureExtractor,
    RegionDiscovery,
    SemanticInterpreter,
    SemanticScorer,
)
from contextmap.visual_perception.run_artifact import (
    IncompleteRunArtifactError,
    PerceptionRunReader,
    PerceptionRunWriter,
    RunArtifactError,
    RunArtifactFileEntry,
    RunArtifactManifest,
    allocate_run_index,
    rebuild_run_registry,
)
from contextmap.visual_perception.serialization import (
    decode_perception_result,
    encode_perception_result,
)
from contextmap.visual_perception.service import (
    StageDefinition,
    StageGraphError,
    StageOutcome,
    StageRunner,
    StageStatus,
    assemble_perception_result,
    execute_stage_graph,
)

__all__ = [
    "BackendProvenance",
    "BoundingBox2D",
    "ClaimId",
    "EvidenceSetError",
    "FeatureExtractor",
    "FeatureId",
    "FeatureScope",
    "HypothesisRole",
    "IncompleteRunArtifactError",
    "ObservationEvidence",
    "PerceptionEvidenceSet",
    "PerceptionResult",
    "PerceptionResultId",
    "PerceptionRun",
    "PerceptionRunId",
    "PerceptionRunReader",
    "PerceptionRunWriter",
    "PreparedImage",
    "Region2D",
    "RegionDiscovery",
    "RegionId",
    "RunArtifactError",
    "RunArtifactFileEntry",
    "RunArtifactManifest",
    "SceneContext",
    "SemanticClaim",
    "SemanticInterpreter",
    "SemanticScorer",
    "SemanticSupport",
    "StageDefinition",
    "StageGraphError",
    "StageOutcome",
    "StageRunner",
    "StageStatus",
    "VisualFeature",
    "allocate_run_index",
    "assemble_perception_result",
    "claim_id_for",
    "decode_perception_result",
    "encode_perception_result",
    "execute_stage_graph",
    "feature_id_for",
    "perception_result_id_for",
    "rebuild_run_registry",
    "region_id_for",
]

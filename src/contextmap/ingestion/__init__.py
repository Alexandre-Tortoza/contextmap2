"""Public contract for the ingestion capability.

Ingestion normalizes source-specific robotic data (ROS 1 bags, ROS 2 bags,
recorded datasets) into canonical, backend-agnostic sensor observations, and
persists them as a reusable, immutable sequence artifact. See
``src/contextmap/ingestion/docs/README.md`` for the full capability
documentation.
"""

from contextmap.ingestion.models import (
    MODALITY_NAMES,
    CalibrationReferenceId,
    ExternalPoseMeasurement,
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    SensorId,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
    observation_modality,
)
from contextmap.ingestion.sequence_artifact import (
    IncompleteSequenceArtifactError,
    SequenceArtifactError,
    SequenceArtifactFileEntry,
    SequenceArtifactId,
    SequenceArtifactManifest,
    SequenceArtifactReader,
    SequenceArtifactWriter,
)
from contextmap.ingestion.synchronization import (
    DroppedEvent,
    ModalityAssociation,
    ProcessingObservation,
    SynchronizationConfig,
    SynchronizationDiagnostics,
    synchronize,
)

__all__ = [
    "MODALITY_NAMES",
    "CalibrationReferenceId",
    "DroppedEvent",
    "ExternalPoseMeasurement",
    "FrameId",
    "ImageEncoding",
    "ImageObservation",
    "ImuObservation",
    "IncompleteSequenceArtifactError",
    "LidarObservation",
    "ModalityAssociation",
    "PointFieldDataType",
    "PointFieldDescriptor",
    "ProcessingObservation",
    "SensorId",
    "SequenceArtifactError",
    "SequenceArtifactFileEntry",
    "SequenceArtifactId",
    "SequenceArtifactManifest",
    "SequenceArtifactReader",
    "SequenceArtifactWriter",
    "SourceObservation",
    "SourceObservationId",
    "SourceProvenance",
    "SynchronizationConfig",
    "SynchronizationDiagnostics",
    "observation_modality",
    "synchronize",
]

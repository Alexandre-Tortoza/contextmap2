"""Stage-level metric registry: versioned, machine-readable metric definitions.

Every capability has different quality semantics, so combining region IoU, pose
error, reprojection error, semantic quality, entity identity, relations and
runtime into one score would hide the cause of a failure. The registry names
each metric once, per stage and version, and states what it measures, over
which population, in which unit and range, which annotations it needs, how it
aggregates, what it does when annotations are missing, who computes it and
whether higher or lower is better. Quality and performance metrics are distinct
kinds and never share a score. See ``src/contextmap/evaluation/docs/metrics.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from contextmap.evaluation._persistence import canonical_digest
from contextmap.evaluation._validation import require_text, require_unique
from contextmap.evaluation.annotations import AnnotationError, AnnotationFamily

ALLOWED_UNITS = frozenset(
    {
        "ratio",
        "count",
        "pixels",
        "meters",
        "degrees",
        "seconds",
        "bytes",
        "per_second",
        "score",
    }
)
"""Units a metric may declare. ``score`` is an uncalibrated, arbitrary score.

``probability`` is deliberately absent: an arbitrary support score is never
called a calibrated probability.
"""


class MetricRegistryError(ValueError):
    """Raised when a metric or a registry is not what an evaluator may rely on."""


class MetricCompatibilityError(MetricRegistryError):
    """Raised when the available annotations cannot back a metric as defined."""


class EvaluationStage(Enum):
    """The pipeline stages an evaluation report can be about."""

    INGESTION_INTEGRITY = "ingestion_integrity"
    STATE_ESTIMATION = "state_estimation"
    GEOMETRIC_MAPPING = "geometric_mapping"
    REGION_DISCOVERY = "region_discovery"
    FEATURE_EXTRACTION = "feature_extraction"
    SEMANTIC_INTERPRETATION = "semantic_interpretation"
    SENSOR_ASSOCIATION = "sensor_association"
    POINT_REPRESENTATION = "point_representation"
    SEMANTIC_FUSION = "semantic_fusion"
    ENTITY_RESOLUTION = "entity_resolution"
    SPATIAL_RELATIONS = "spatial_relations"
    ARTIFACT_INTEGRITY = "artifact_integrity"
    RUNTIME = "runtime"


class MetricKind(Enum):
    """Quality says how correct a result is; performance says what it cost."""

    QUALITY = "quality"
    PERFORMANCE = "performance"


class MetricDirection(Enum):
    """Whether a larger value is an improvement."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NOT_ORDERED = "not_ordered"


class MissingDataBehavior(Enum):
    """What a metric does when samples lack the annotation it needs."""

    EXCLUDE_UNANNOTATED = "exclude_unannotated"
    """Restrict the population to annotated samples; not applicable if there are none."""

    NOT_APPLICABLE = "not_applicable"
    """Any gap makes the metric not applicable; it is never reported as zero."""


@dataclass(frozen=True, kw_only=True)
class MetricDefinition:
    """One metric, in one version, for one stage.

    Attributes:
        name: Dotted metric name, e.g. ``region.iou.mean``.
        version: Version of the definition; a change of meaning is a new version.
        stage: The stage the metric evaluates; ``None`` for a cross-cutting
            performance metric that any stage's report may carry.
        kind: Quality or performance.
        description: What the metric measures.
        population: What is averaged over, including the annotated population
            and time window it is computed on.
        unit: One of :data:`ALLOWED_UNITS`.
        minimum: Smallest possible value, when bounded.
        maximum: Largest possible value, when bounded.
        direction: Whether higher or lower is better.
        aggregation: How per-sample values become the reported one.
        required_annotations: Exact annotation schema identifiers the metric
            needs; empty for metrics that need no reference.
        missing_data: What happens when annotations are missing.
        evaluator_id: The evaluator that computes the metric.
        evaluator_version: Version of that evaluator.
    """

    name: str
    version: str
    stage: EvaluationStage | None
    kind: MetricKind
    description: str
    population: str
    unit: str
    minimum: float | None
    maximum: float | None
    direction: MetricDirection
    aggregation: str
    required_annotations: tuple[str, ...]
    missing_data: MissingDataBehavior
    evaluator_id: str
    evaluator_version: str

    def __post_init__(self) -> None:
        """Reject definitions that would mislead an evaluator or a reader."""
        for field, value in (
            ("name", self.name),
            ("version", self.version),
            ("description", self.description),
            ("population", self.population),
            ("aggregation", self.aggregation),
            ("evaluator_id", self.evaluator_id),
            ("evaluator_version", self.evaluator_version),
        ):
            require_text(f"metric {field}", value)
        if self.unit not in ALLOWED_UNITS:
            raise ValueError(
                f"metric {self.name!r} declares unit {self.unit!r}; allowed units are "
                f"{sorted(ALLOWED_UNITS)} (a support score is never a calibrated probability)"
            )
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"metric {self.name!r} has an empty range")
        if self.kind is MetricKind.QUALITY and self.stage is None:
            raise ValueError(f"quality metric {self.name!r} must name the stage it evaluates")
        if self.kind is MetricKind.PERFORMANCE and self.required_annotations:
            raise ValueError(f"performance metric {self.name!r} cannot require annotations")
        for schema in self.required_annotations:
            try:
                AnnotationFamily.from_schema(schema)
            except AnnotationError as error:
                raise ValueError(
                    f"metric {self.name!r} requires an unknown annotation schema {schema!r}"
                ) from error
        require_unique(f"required annotation of {self.name!r}", self.required_annotations)

    @property
    def key(self) -> str:
        """Return ``name/version``, the identity of the definition."""
        return f"{self.name}/{self.version}"

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "name": self.name,
            "version": self.version,
            "stage": None if self.stage is None else self.stage.value,
            "kind": self.kind.value,
            "description": self.description,
            "population": self.population,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "direction": self.direction.value,
            "aggregation": self.aggregation,
            "required_annotations": list(self.required_annotations),
            "missing_data": self.missing_data.value,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> MetricDefinition:
        """Rebuild a definition from :meth:`to_record` output."""
        stage = record["stage"]
        return cls(
            name=record["name"],
            version=record["version"],
            stage=None if stage is None else EvaluationStage(stage),
            kind=MetricKind(record["kind"]),
            description=record["description"],
            population=record["population"],
            unit=record["unit"],
            minimum=record["minimum"],
            maximum=record["maximum"],
            direction=MetricDirection(record["direction"]),
            aggregation=record["aggregation"],
            required_annotations=tuple(record["required_annotations"]),
            missing_data=MissingDataBehavior(record["missing_data"]),
            evaluator_id=record["evaluator_id"],
            evaluator_version=record["evaluator_version"],
        )


@dataclass(frozen=True, kw_only=True)
class MetricRegistryIdentity:
    """The identity a report carries to name the registry its metrics come from."""

    registry_id: str
    registry_version: str
    digest: str

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "digest": self.digest,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> MetricRegistryIdentity:
        """Rebuild an identity from :meth:`to_record` output."""
        return cls(
            registry_id=record["registry_id"],
            registry_version=record["registry_version"],
            digest=record["digest"],
        )


@dataclass(frozen=True, kw_only=True)
class MetricRegistry:
    """An immutable set of metric definitions, versioned and hashed as a whole."""

    registry_id: str
    registry_version: str
    definitions: tuple[MetricDefinition, ...]

    def __post_init__(self) -> None:
        """Require an identity and unique ``name/version`` keys."""
        require_text("registry_id", self.registry_id)
        require_text("registry_version", self.registry_version)
        require_unique("metric definition", (item.key for item in self.definitions))

    def get(self, name: str, version: str) -> MetricDefinition:
        """Return one definition.

        Raises:
            MetricRegistryError: If the metric or that version is unknown.
        """
        for definition in self.definitions:
            if definition.name == name and definition.version == version:
                return definition
        raise MetricRegistryError(f"unknown metric {name}/{version} in {self.registry_id!r}")

    def definitions_for(self, stage: EvaluationStage) -> tuple[MetricDefinition, ...]:
        """Return the definitions that evaluate a stage."""
        return tuple(item for item in self.definitions if item.stage is stage)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record the digest is computed over."""
        return {
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "definitions": [item.to_record() for item in self.definitions],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> MetricRegistry:
        """Rebuild a registry from :meth:`to_record` output."""
        return cls(
            registry_id=record["registry_id"],
            registry_version=record["registry_version"],
            definitions=tuple(MetricDefinition.from_record(item) for item in record["definitions"]),
        )

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of every definition."""
        return canonical_digest(self.to_record())

    def identity(self) -> MetricRegistryIdentity:
        """Return the identity reports cite."""
        return MetricRegistryIdentity(
            registry_id=self.registry_id,
            registry_version=self.registry_version,
            digest=self.digest(),
        )


def require_annotation_compatibility(
    definition: MetricDefinition, available_schemas: Iterable[str]
) -> None:
    """Refuse a metric whose annotation schemas are not among the available ones.

    A schema must match exactly, version included: a metric defined against
    ``regions/v1`` is not computed from ``regions/v2`` annotations.

    Raises:
        MetricCompatibilityError: Naming the missing schemas.
    """
    available = set(available_schemas)
    missing = [schema for schema in definition.required_annotations if schema not in available]
    if missing:
        raise MetricCompatibilityError(
            f"metric {definition.key} needs annotation schemas {missing}, but only "
            f"{sorted(available)} are available"
        )


# --------------------------------------------------------------------- default registry

_EXCLUDE = MissingDataBehavior.EXCLUDE_UNANNOTATED
_HIGHER = MetricDirection.HIGHER_IS_BETTER
_LOWER = MetricDirection.LOWER_IS_BETTER
_NEUTRAL = MetricDirection.NOT_ORDERED


def _quality(
    name: str,
    stage: EvaluationStage,
    description: str,
    *,
    population: str,
    unit: str,
    direction: MetricDirection,
    aggregation: str,
    evaluator: str,
    minimum: float | None = 0.0,
    maximum: float | None = None,
    annotations: tuple[AnnotationFamily, ...] = (),
    missing_data: MissingDataBehavior = MissingDataBehavior.NOT_APPLICABLE,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        version="1",
        stage=stage,
        kind=MetricKind.QUALITY,
        description=description,
        population=population,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        direction=direction,
        aggregation=aggregation,
        required_annotations=tuple(family.schema for family in annotations),
        missing_data=missing_data,
        evaluator_id=evaluator,
        evaluator_version="1",
    )


def _performance(
    name: str,
    description: str,
    *,
    unit: str,
    aggregation: str,
    population: str = "every evaluated sample of the stage",
    direction: MetricDirection = _LOWER,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        version="1",
        stage=None,
        kind=MetricKind.PERFORMANCE,
        description=description,
        population=population,
        unit=unit,
        minimum=0.0,
        maximum=None,
        direction=direction,
        aggregation=aggregation,
        required_annotations=(),
        missing_data=MissingDataBehavior.NOT_APPLICABLE,
        evaluator_id="runtime-profiler",
        evaluator_version="1",
    )


def default_metric_registry() -> MetricRegistry:
    """Return the version-1 registry of Solution 1 stage metrics."""
    stage = EvaluationStage
    regions = AnnotationFamily.REGIONS
    semantics = AnnotationFamily.SEMANTICS
    geometry = AnnotationFamily.GEOMETRY
    identity = AnnotationFamily.IDENTITY
    relations = AnnotationFamily.RELATIONS
    definitions = (
        _quality(
            "ingestion.integrity.violations",
            stage.INGESTION_INTEGRITY,
            "Integrity violations found in the canonical sequence.",
            population="every observation of the evaluated sequence artifact",
            unit="count",
            direction=_LOWER,
            aggregation="sum over observations",
            evaluator="ingestion-integrity-check",
        ),
        _quality(
            "ingestion.modality.coverage",
            stage.INGESTION_INTEGRITY,
            "Fraction of expected modalities present in the sequence.",
            population="modalities the reference set expects for the sequence",
            unit="ratio",
            maximum=1.0,
            direction=_HIGHER,
            aggregation="present modalities divided by expected modalities",
            evaluator="ingestion-integrity-check",
        ),
        _quality(
            "state.ate.rmse",
            stage.STATE_ESTIMATION,
            "Absolute trajectory error against a trusted reference trajectory.",
            population="poses associated with a trusted reference pose within tolerance",
            unit="meters",
            direction=_LOWER,
            aggregation="root mean square over associated poses after the declared alignment",
            evaluator="state-estimation-evaluator",
            missing_data=_EXCLUDE,
        ),
        _quality(
            "state.rpe.translation.rmse",
            stage.STATE_ESTIMATION,
            "Relative pose error of translation over a declared interval.",
            population="pose pairs a declared interval apart with trusted references",
            unit="meters",
            direction=_LOWER,
            aggregation="root mean square over pose pairs",
            evaluator="state-estimation-evaluator",
            missing_data=_EXCLUDE,
        ),
        _quality(
            "state.gap.ratio",
            stage.STATE_ESTIMATION,
            "Fraction of the sequence time not covered by valid poses.",
            population="the time window of the evaluated sequence",
            unit="ratio",
            maximum=1.0,
            direction=_LOWER,
            aggregation="gap duration divided by sequence duration",
            evaluator="state-estimation-evaluator",
        ),
        _quality(
            "geometry.scan_overlap.plane_distance.median",
            stage.GEOMETRIC_MAPPING,
            "Point-to-plane agreement between overlapping scans of the built map.",
            population="overlapping scan pairs of the evaluated map",
            unit="meters",
            direction=_LOWER,
            aggregation="median over overlapping scan pairs",
            evaluator="geometric-mapping-evaluator",
        ),
        _quality(
            "geometry.expected_point.error.max",
            stage.GEOMETRIC_MAPPING,
            "Largest position error of source points with a known global position.",
            population="expected points declared by the protocol",
            unit="meters",
            direction=_LOWER,
            aggregation="maximum over expected points",
            evaluator="geometric-mapping-evaluator",
            missing_data=_EXCLUDE,
        ),
        _quality(
            "region.iou.mean",
            stage.REGION_DISCOVERY,
            "Mean IoU between discovered and annotated regions.",
            population="matched regions of frames with a regions annotation",
            unit="ratio",
            maximum=1.0,
            direction=_HIGHER,
            aggregation="mean over annotated frames of the per-frame mean IoU",
            evaluator="region-discovery-evaluator",
            annotations=(regions,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "region.recall.mean",
            stage.REGION_DISCOVERY,
            "Fraction of annotated regions recovered by a discovered region.",
            population="annotated regions of frames with a regions annotation",
            unit="ratio",
            maximum=1.0,
            direction=_HIGHER,
            aggregation="mean over annotated frames of the per-frame region recall",
            evaluator="region-discovery-evaluator",
            annotations=(regions,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "region.duplicate_rate.mean",
            stage.REGION_DISCOVERY,
            "Fraction of discovered regions that duplicate an annotated one.",
            population="discovered regions of frames with a regions annotation",
            unit="ratio",
            maximum=1.0,
            direction=_LOWER,
            aggregation="mean over annotated frames of the per-frame duplicate rate",
            evaluator="region-discovery-evaluator",
            annotations=(regions,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "feature.finite_ratio",
            stage.FEATURE_EXTRACTION,
            "Fraction of feature values that are finite.",
            population="every value of the evaluated feature payloads",
            unit="ratio",
            maximum=1.0,
            direction=_HIGHER,
            aggregation="finite values divided by all values",
            evaluator="feature-extraction-evaluator",
        ),
        _quality(
            "feature.repeatability.max_abs_diff",
            stage.FEATURE_EXTRACTION,
            "Largest absolute difference between repeated extractions of one input.",
            population="repeated extractions under an identical context",
            unit="score",
            direction=_LOWER,
            aggregation="maximum over repeated pairs",
            evaluator="feature-extraction-evaluator",
        ),
        _quality(
            "semantic.acceptable_claim_rate",
            stage.SEMANTIC_INTERPRETATION,
            "Fraction of claims that match an acceptable annotated concept.",
            population="claims of requests whose target has a labeled or ambiguous record",
            unit="ratio",
            maximum=1.0,
            direction=_HIGHER,
            aggregation="acceptable claims divided by all claims",
            evaluator="semantic-interpretation-evaluator",
            annotations=(semantics,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "semantic.unsupported_claim_rate",
            stage.SEMANTIC_INTERPRETATION,
            "Fraction of claims that match no acceptable annotated concept.",
            population="claims of requests whose target has a labeled or ambiguous record",
            unit="ratio",
            maximum=1.0,
            direction=_LOWER,
            aggregation="unsupported claims divided by all claims",
            evaluator="semantic-interpretation-evaluator",
            annotations=(semantics,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "semantic.ambiguity_preservation_rate",
            stage.SEMANTIC_INTERPRETATION,
            "Fraction of ambiguous targets whose alternatives the interpreter kept.",
            population="requests whose target has an ambiguous record",
            unit="ratio",
            maximum=1.0,
            direction=_HIGHER,
            aggregation="requests that preserved ambiguity divided by ambiguous requests",
            evaluator="semantic-interpretation-evaluator",
            annotations=(semantics,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "association.reprojection_error.median",
            stage.SENSOR_ASSOCIATION,
            "Pixel error between a projected 3D reference point and its annotated pixel.",
            population="trusted 3D-pixel correspondences visible in the evaluated frames",
            unit="pixels",
            direction=_LOWER,
            aggregation="median over correspondences",
            evaluator="sensor-association-evaluator",
            annotations=(geometry,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "association.visible_support.ratio",
            stage.SENSOR_ASSOCIATION,
            "Fraction of considered map points that are visible to the camera.",
            population="map points considered for the evaluated frames",
            unit="ratio",
            maximum=1.0,
            direction=_NEUTRAL,
            aggregation="visible points divided by considered points",
            evaluator="sensor-association-evaluator",
        ),
        _quality(
            "pointrep.repeatability.cosine",
            stage.POINT_REPRESENTATION,
            "Cosine similarity of representations of the same support under identical input.",
            population="supports encoded twice under an identical context",
            unit="ratio",
            minimum=-1.0,
            maximum=1.0,
            direction=_HIGHER,
            aggregation="mean over supports",
            evaluator="point-representation-evaluator",
        ),
        _quality(
            "fusion.reference_recovery.rate",
            stage.SEMANTIC_FUSION,
            "Fraction of annotated supports whose fused evidence keeps the reference concept.",
            population="supports whose observations carry one consistent semantic annotation",
            unit="ratio",
            maximum=1.0,
            direction=_HIGHER,
            aggregation="recovered supports divided by annotated supports",
            evaluator="semantic-fusion-evaluator",
            annotations=(semantics,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "fusion.ambiguity_retention.rate",
            stage.SEMANTIC_FUSION,
            "Fraction of ambiguous or conflicting supports that keep their alternatives.",
            population="supports whose annotation or evidence is ambiguous or conflicting",
            unit="ratio",
            maximum=1.0,
            direction=_HIGHER,
            aggregation="supports that retained alternatives divided by ambiguous supports",
            evaluator="semantic-fusion-evaluator",
            annotations=(semantics,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "entity.false_merge.rate",
            stage.ENTITY_RESOLUTION,
            "Fraction of resolved entities that merge annotated-distinct identities.",
            population="resolved entities in observations with a COMPLETE identity scope",
            unit="ratio",
            maximum=1.0,
            direction=_LOWER,
            aggregation="false-merge entities divided by resolved entities",
            evaluator="entity-resolution-evaluator",
            annotations=(identity,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "entity.duplicate.rate",
            stage.ENTITY_RESOLUTION,
            "Fraction of annotated identities represented by more than one resolved entity.",
            population="annotated identities in observations with a COMPLETE identity scope",
            unit="ratio",
            maximum=1.0,
            direction=_LOWER,
            aggregation="duplicated identities divided by annotated identities",
            evaluator="entity-resolution-evaluator",
            annotations=(identity,),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "relations.f1",
            stage.SPATIAL_RELATIONS,
            "F1 of predicted relations against annotated relations that hold.",
            population="annotated relations with status holds, does_not_hold or ambiguous",
            unit="ratio",
            maximum=1.0,
            direction=_HIGHER,
            aggregation="harmonic mean of precision and recall over decided relations",
            evaluator="spatial-relations-evaluator",
            annotations=(relations, identity),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "relations.negative_violation.rate",
            stage.SPATIAL_RELATIONS,
            "Fraction of explicitly negative annotated relations predicted to hold.",
            population="annotated relations with status does_not_hold",
            unit="ratio",
            maximum=1.0,
            direction=_LOWER,
            aggregation="violated negatives divided by annotated negatives",
            evaluator="spatial-relations-evaluator",
            annotations=(relations, identity),
            missing_data=_EXCLUDE,
        ),
        _quality(
            "artifact.integrity.violations",
            stage.ARTIFACT_INTEGRITY,
            "Integrity violations found in the evaluated artifact.",
            population="every file and reference of the evaluated artifact",
            unit="count",
            direction=_LOWER,
            aggregation="sum over checks",
            evaluator="artifact-integrity-check",
        ),
        _quality(
            "artifact.round_trip.mismatches",
            stage.ARTIFACT_INTEGRITY,
            "Fields that differ after writing and reading the artifact back.",
            population="every field of the round-tripped artifact",
            unit="count",
            direction=_LOWER,
            aggregation="sum over fields",
            evaluator="artifact-integrity-check",
        ),
        _performance(
            "runtime.wall_time",
            "Wall-clock time spent evaluating or running the stage.",
            unit="seconds",
            aggregation="sum over samples",
        ),
        _performance(
            "runtime.peak_memory",
            "Peak resident or device memory during the stage.",
            unit="bytes",
            aggregation="maximum over samples",
        ),
        _performance(
            "runtime.storage_size",
            "Bytes written by the stage.",
            unit="bytes",
            aggregation="sum over artifacts written",
        ),
        _performance(
            "runtime.throughput",
            "Samples processed per second.",
            unit="per_second",
            aggregation="samples divided by wall time",
            direction=_HIGHER,
        ),
    )
    return MetricRegistry(
        registry_id="contextmap-stage-metrics", registry_version="1", definitions=definitions
    )

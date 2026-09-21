"""Versioned, multi-level reference annotation schemas for Solution 1 evaluation.

The pipeline is evaluated at independent levels, so the reference set carries
one purpose-built annotation family per level:

==============  =========================================================
Family          Evaluates
==============  =========================================================
regions         2D region masks/bounds, valid and exclusion areas
semantics       open-vocabulary concepts, ambiguity, abstention, attributes
geometry        trusted 3D <-> pixel correspondences
identity        physical entity identity across observations
relations       spatial relations between annotated identities
visibility      occlusion/visibility strata
scene_context   scene metadata used to stratify results
==============  =========================================================

Three rules run through every family:

* annotations are **partial by nature**: an absent record means *not
  annotated*, never negative truth. Negative truth is always explicit
  (``rejected_concepts``, ``DOES_NOT_HOLD``, ``distinct_pairs``, a
  ``COMPLETE`` region frame);
* ambiguity and unknown are first-class values, not missing data;
* every record is linked to physical observations through
  ``ReferenceSampleId``/``SourceObservationId``, and no model output becomes
  reference truth here (that is decided by the trust level of the manifest).

See ``src/contextmap/evaluation/docs/annotations.md``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

from contextmap.evaluation._persistence import file_digest, write_immutable_json
from contextmap.evaluation._validation import require_text, require_unique
from contextmap.evaluation.reference_set import ReferenceSampleId
from contextmap.evaluation.region_discovery import GroundTruthRegion
from contextmap.evaluation.semantic_interpretation import MATCHING_POLICY, SemanticAnnotation
from contextmap.ingestion import CalibrationReferenceId, SourceObservationId
from contextmap.visual_perception import BoundingBox, InlineMask

CASEFOLD_EXACT_POLICY = MATCHING_POLICY
"""Case and whitespace normalization only; no synonym equivalence is implied."""

CASEFOLD_ALIAS_POLICY = "casefold-alias/1"
"""``casefold-exact/1`` plus an alias table the annotation document declares."""

_POLICIES = frozenset({CASEFOLD_EXACT_POLICY, CASEFOLD_ALIAS_POLICY})


class AnnotationError(ValueError):
    """Raised when an annotation document cannot be read or used as declared."""


class AnnotationFamily(Enum):
    """The annotation levels a reference set can carry."""

    REGIONS = "regions"
    SEMANTICS = "semantics"
    GEOMETRY = "geometry"
    IDENTITY = "identity"
    RELATIONS = "relations"
    VISIBILITY = "visibility"
    SCENE_CONTEXT = "scene_context"

    @property
    def schema(self) -> str:
        """Return the versioned schema identifier of this family."""
        return f"contextmap.reference.{self.value}/v1"

    @classmethod
    def from_schema(cls, schema: str) -> AnnotationFamily:
        """Return the family of a schema identifier.

        Raises:
            AnnotationError: If the schema is not a supported version of a family.
        """
        for family in cls:
            if family.schema == schema:
                return family
        raise AnnotationError(f"unsupported annotation schema {schema!r}")


class Coverage(Enum):
    """Whether everything in scope was annotated.

    Only ``COMPLETE`` lets an evaluator treat an unannotated item as a negative
    (a prediction with no reference counterpart is then a false positive).
    """

    COMPLETE = "complete"
    PARTIAL = "partial"


@dataclass(frozen=True, kw_only=True)
class ObservationRef:
    """A link from an annotation to a physical observation of a sample.

    ``observation_id`` is ``None`` when the annotation applies to the sample as
    a whole.
    """

    sample_id: ReferenceSampleId
    observation_id: SourceObservationId | None

    def __post_init__(self) -> None:
        """Require the sample identity."""
        require_text("sample_id", self.sample_id)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"sample_id": self.sample_id, "observation_id": self.observation_id}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ObservationRef:
        """Rebuild a reference from :meth:`to_record` output."""
        observation = record["observation_id"]
        return cls(
            sample_id=ReferenceSampleId(record["sample_id"]),
            observation_id=None if observation is None else SourceObservationId(observation),
        )


def _fold(label: str) -> str:
    return " ".join(label.casefold().split())


@dataclass(frozen=True, kw_only=True)
class LabelAlias:
    """A comparison key and the literal labels an annotator treats as the same concept."""

    key: str
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require a key and at least one alias, all non-empty."""
        require_text("alias key", self.key)
        if not self.aliases:
            raise ValueError(f"alias key {self.key!r} needs at least one alias")
        for alias in self.aliases:
            require_text("alias", alias)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"key": self.key, "aliases": list(self.aliases)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> LabelAlias:
        """Rebuild an alias group from :meth:`to_record` output."""
        return cls(key=record["key"], aliases=tuple(record["aliases"]))


@dataclass(frozen=True, kw_only=True)
class LabelNormalization:
    """The explicit, versioned rule that turns literal labels into comparison keys.

    Literal annotator labels are always preserved. Two labels are only compared
    as equal through this policy; nothing is forced into a closed taxonomy.

    Attributes:
        policy_id: ``casefold-exact/1`` or ``casefold-alias/1``.
        aliases: Alias groups; only valid for ``casefold-alias/1``.
    """

    policy_id: str
    aliases: tuple[LabelAlias, ...] = ()

    def __post_init__(self) -> None:
        """Reject unknown policies, aliases without the alias policy and conflicts."""
        if self.policy_id not in _POLICIES:
            raise ValueError(f"unknown normalization policy {self.policy_id!r}")
        if self.policy_id == CASEFOLD_EXACT_POLICY and self.aliases:
            raise ValueError(f"the {CASEFOLD_EXACT_POLICY} policy does not take aliases")
        _ = self._lookup

    @cached_property
    def _lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for group in self.aliases:
            key = _fold(group.key)
            for form in (key, *(_fold(alias) for alias in group.aliases)):
                if lookup.setdefault(form, key) != key:
                    raise ValueError(f"alias {form!r} maps to two different keys")
        return lookup

    def key(self, label: str) -> str:
        """Return the comparison key of a literal label under this policy."""
        folded = _fold(label)
        return self._lookup.get(folded, folded)

    def expansions(self, label: str) -> tuple[str, ...]:
        """Return the literal forms an exact matcher must accept to honor the aliases."""
        folded = _fold(label)
        for group in self.aliases:
            if self._lookup.get(folded) == _fold(group.key):
                return (group.key, *group.aliases)
        return (label,)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"policy_id": self.policy_id, "aliases": [item.to_record() for item in self.aliases]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> LabelNormalization:
        """Rebuild a normalization from :meth:`to_record` output."""
        return cls(
            policy_id=record["policy_id"],
            aliases=tuple(LabelAlias.from_record(item) for item in record["aliases"]),
        )


@dataclass(frozen=True, kw_only=True)
class ConceptAttribute:
    """A literal attribute (name and value) an annotator recorded."""

    name: str
    value: str

    def __post_init__(self) -> None:
        """Require a name and a value."""
        require_text("attribute name", self.name)
        require_text("attribute value", self.value)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ConceptAttribute:
        """Rebuild an attribute from :meth:`to_record` output."""
        return cls(name=record["name"], value=record["value"])


def _unique_refs(refs: Iterable[ObservationRef]) -> tuple[ObservationRef, ...]:
    return tuple(dict.fromkeys(refs))


# --------------------------------------------------------------------------- regions


@dataclass(frozen=True, kw_only=True)
class RegionAnnotation:
    """One annotated 2D region, by mask, by box or by both.

    The box uses the half-open pixel convention of ``BoundingBox``.
    """

    region_id: str
    mask: InlineMask | None = None
    box: BoundingBox | None = None

    def __post_init__(self) -> None:
        """Require an identity and some geometry."""
        require_text("region_id", self.region_id)
        if self.mask is None and self.box is None:
            raise ValueError(f"region {self.region_id!r} needs geometry: a mask, a box or both")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        box = self.box
        return {
            "region_id": self.region_id,
            "mask": None if self.mask is None else self.mask.to_dict(),
            "box": None
            if box is None
            else {"x_min": box.x_min, "y_min": box.y_min, "x_max": box.x_max, "y_max": box.y_max},
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RegionAnnotation:
        """Rebuild a region from :meth:`to_record` output."""
        mask, box = record["mask"], record["box"]
        return cls(
            region_id=record["region_id"],
            mask=None if mask is None else InlineMask.from_dict(mask),
            box=None if box is None else BoundingBox(**box),
        )


@dataclass(frozen=True, kw_only=True)
class FrameRegionAnnotation:
    """The region annotation of one image observation.

    Attributes:
        sample_id: The sample the image belongs to.
        observation_id: The annotated image observation.
        image_width: Width of the annotated image, in pixels.
        image_height: Height of the annotated image, in pixels.
        coverage: ``COMPLETE`` when every in-scope region is annotated, so an
            unmatched prediction is a false positive; ``PARTIAL`` otherwise.
        regions: The annotated regions.
        valid_areas: Where the annotation applies (e.g. a fisheye valid
            region); empty means the whole image.
        exclusion_areas: Areas left out of the annotation: predictions there
            are neither correct nor false positives.
    """

    sample_id: ReferenceSampleId
    observation_id: SourceObservationId
    image_width: int
    image_height: int
    coverage: Coverage
    regions: tuple[RegionAnnotation, ...]
    valid_areas: tuple[InlineMask, ...] = ()
    exclusion_areas: tuple[InlineMask, ...] = ()

    def __post_init__(self) -> None:
        """Require identities, positive image size and unique region ids."""
        require_text("sample_id", self.sample_id)
        require_text("observation_id", self.observation_id)
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        require_unique(
            f"region id of frame {self.observation_id!r}",
            (region.region_id for region in self.regions),
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "sample_id": self.sample_id,
            "observation_id": self.observation_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "coverage": self.coverage.value,
            "regions": [item.to_record() for item in self.regions],
            "valid_areas": [item.to_dict() for item in self.valid_areas],
            "exclusion_areas": [item.to_dict() for item in self.exclusion_areas],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FrameRegionAnnotation:
        """Rebuild a frame annotation from :meth:`to_record` output."""
        return cls(
            sample_id=ReferenceSampleId(record["sample_id"]),
            observation_id=SourceObservationId(record["observation_id"]),
            image_width=record["image_width"],
            image_height=record["image_height"],
            coverage=Coverage(record["coverage"]),
            regions=tuple(RegionAnnotation.from_record(item) for item in record["regions"]),
            valid_areas=tuple(InlineMask.from_dict(item) for item in record["valid_areas"]),
            exclusion_areas=tuple(InlineMask.from_dict(item) for item in record["exclusion_areas"]),
        )


@dataclass(frozen=True, kw_only=True)
class RegionAnnotationSet:
    """Region annotations, one entry per annotated image observation."""

    family: ClassVar[AnnotationFamily] = AnnotationFamily.REGIONS

    frames: tuple[FrameRegionAnnotation, ...]

    def __post_init__(self) -> None:
        """Require each (sample, observation) to be annotated once."""
        require_unique(
            "frame (sample, observation)",
            (f"{item.sample_id}/{item.observation_id}" for item in self.frames),
        )

    def observation_references(self) -> tuple[ObservationRef, ...]:
        """Return the distinct physical observations the annotations refer to."""
        return _unique_refs(
            ObservationRef(sample_id=item.sample_id, observation_id=item.observation_id)
            for item in self.frames
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"frames": [item.to_record() for item in self.frames]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RegionAnnotationSet:
        """Rebuild the set from :meth:`to_record` output."""
        return cls(
            frames=tuple(FrameRegionAnnotation.from_record(item) for item in record["frames"])
        )


def ground_truth_regions(frame: FrameRegionAnnotation) -> tuple[GroundTruthRegion, ...]:
    """Return the mask regions of a frame in the form Region Discovery evaluation consumes.

    Regions annotated only by a box carry no mask and are left out.
    """
    return tuple(
        GroundTruthRegion(region_id=region.region_id, mask=region.mask)
        for region in frame.regions
        if region.mask is not None
    )


# ------------------------------------------------------------------------- semantics


class SemanticStatus(Enum):
    """What the annotator could say about a target."""

    LABELED = "labeled"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True, kw_only=True)
class SemanticAnnotationRecord:
    """Open-vocabulary semantic truth for one region, or for the whole observation.

    Attributes:
        sample_id: The sample the target belongs to.
        observation_id: The observation the target is seen in.
        region_id: The annotated region; ``None`` for scene-level semantics.
        status: ``LABELED`` (one or more acceptable concepts), ``AMBIGUOUS``
            (two or more equally plausible concepts, so preserving alternatives
            is the right answer) or ``UNKNOWN`` (the annotator could not tell,
            so abstaining is right and correctness is not applicable).
        concepts: Literal acceptable concepts, as the annotator wrote them.
        rejected_concepts: Concepts explicitly known **not** to apply: the only
            way to express negative semantic truth.
        attributes: Literal attributes recorded for the target.
    """

    sample_id: ReferenceSampleId
    observation_id: SourceObservationId
    region_id: str | None
    status: SemanticStatus
    concepts: tuple[str, ...]
    rejected_concepts: tuple[str, ...] = ()
    attributes: tuple[ConceptAttribute, ...] = ()

    def __post_init__(self) -> None:
        """Require the concept count each status implies."""
        require_text("sample_id", self.sample_id)
        require_text("observation_id", self.observation_id)
        for concept in (*self.concepts, *self.rejected_concepts):
            require_text("concept", concept)
        if self.status is SemanticStatus.LABELED and not self.concepts:
            raise ValueError("a labeled record needs at least one acceptable concept")
        if self.status is SemanticStatus.AMBIGUOUS and len(self.concepts) < 2:
            raise ValueError("an ambiguous record needs at least two plausible concepts")
        if self.status is SemanticStatus.UNKNOWN and self.concepts:
            raise ValueError("an unknown record cannot list acceptable concepts")
        require_unique("attribute name", (item.name for item in self.attributes))

    @property
    def target(self) -> tuple[str, str, str | None]:
        """Return the identity of the annotated target."""
        return (self.sample_id, self.observation_id, self.region_id)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "sample_id": self.sample_id,
            "observation_id": self.observation_id,
            "region_id": self.region_id,
            "status": self.status.value,
            "concepts": list(self.concepts),
            "rejected_concepts": list(self.rejected_concepts),
            "attributes": [item.to_record() for item in self.attributes],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SemanticAnnotationRecord:
        """Rebuild a record from :meth:`to_record` output."""
        return cls(
            sample_id=ReferenceSampleId(record["sample_id"]),
            observation_id=SourceObservationId(record["observation_id"]),
            region_id=record["region_id"],
            status=SemanticStatus(record["status"]),
            concepts=tuple(record["concepts"]),
            rejected_concepts=tuple(record["rejected_concepts"]),
            attributes=tuple(ConceptAttribute.from_record(item) for item in record["attributes"]),
        )


@dataclass(frozen=True, kw_only=True)
class SemanticAnnotationSet:
    """Semantic annotations plus the normalization their concepts are compared under."""

    family: ClassVar[AnnotationFamily] = AnnotationFamily.SEMANTICS

    normalization: LabelNormalization
    records: tuple[SemanticAnnotationRecord, ...]

    def __post_init__(self) -> None:
        """Require unique targets and concepts that are coherent under the normalization."""
        require_unique(
            "semantic target",
            (f"{r.sample_id}/{r.observation_id}/{r.region_id}" for r in self.records),
        )
        for record in self.records:
            keys = [self.normalization.key(concept) for concept in record.concepts]
            require_unique(f"concepts of target {record.target}", keys)
            rejected = {self.normalization.key(concept) for concept in record.rejected_concepts}
            for concept, key in zip(record.concepts, keys, strict=True):
                if key in rejected:
                    raise ValueError(
                        f"concept {concept!r} of target {record.target} is both acceptable "
                        "and rejected"
                    )

    def find(
        self,
        sample_id: ReferenceSampleId,
        observation_id: SourceObservationId,
        region_id: str | None,
    ) -> SemanticAnnotationRecord | None:
        """Return the record of a target, or ``None`` when it was not annotated."""
        target = (sample_id, observation_id, region_id)
        return next((record for record in self.records if record.target == target), None)

    def to_semantic_annotation(self, record: SemanticAnnotationRecord) -> SemanticAnnotation:
        """Return the record in the form Semantic Interpretation evaluation consumes.

        The evaluator matches with ``casefold-exact/1``, so the aliases the
        document declares are expanded into the acceptable hypotheses.

        Raises:
            AnnotationError: For an ``UNKNOWN`` record, which has no acceptable
                hypothesis: correctness is not applicable to it.
        """
        if record.status is SemanticStatus.UNKNOWN:
            raise AnnotationError(
                f"target {record.target} is unknown: no acceptable hypotheses to evaluate against"
            )
        hypotheses: dict[str, str] = {}
        for concept in record.concepts:
            for form in (concept, *self.normalization.expansions(concept)):
                hypotheses.setdefault(_fold(form), form)
        return SemanticAnnotation(
            acceptable_hypotheses=tuple(hypotheses.values()),
            ambiguity_expected=record.status is SemanticStatus.AMBIGUOUS,
        )

    def observation_references(self) -> tuple[ObservationRef, ...]:
        """Return the distinct physical observations the annotations refer to."""
        return _unique_refs(
            ObservationRef(sample_id=item.sample_id, observation_id=item.observation_id)
            for item in self.records
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "normalization": self.normalization.to_record(),
            "records": [item.to_record() for item in self.records],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SemanticAnnotationSet:
        """Rebuild the set from :meth:`to_record` output."""
        return cls(
            normalization=LabelNormalization.from_record(record["normalization"]),
            records=tuple(SemanticAnnotationRecord.from_record(item) for item in record["records"]),
        )


# -------------------------------------------------------------------------- geometry


def _finite(name: str, values: Iterable[float]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be finite")


def _tolerance(name: str, value: float | None) -> None:
    if value is not None and not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} tolerance must be a positive finite number")


@dataclass(frozen=True, kw_only=True)
class GeometryCorrespondence:
    """A trusted correspondence between an image pixel and a 3D point.

    Attributes:
        correspondence_id: Identity inside the annotation set.
        sample_id: The sample the correspondence belongs to.
        image_observation_id: The image observation the pixel lies in.
        calibration_id: The camera calibration the pixel refers to.
        pixel_uv: ``(u, v)`` in image pixels, on the half-open pixel grid.
        point_frame_id: Coordinate frame the 3D point is expressed in.
        point_m: ``(x, y, z)`` in metres, in ``point_frame_id``.
        point_source_observation_id: The scan the point was measured in, when
            it comes from a sensor observation.
        pixel_tolerance_px: Annotator's pixel uncertainty, when known.
        point_tolerance_m: Annotator's 3D uncertainty in metres, when known.
    """

    correspondence_id: str
    sample_id: ReferenceSampleId
    image_observation_id: SourceObservationId
    calibration_id: CalibrationReferenceId
    pixel_uv: tuple[float, float]
    point_frame_id: str
    point_m: tuple[float, float, float]
    point_source_observation_id: SourceObservationId | None = None
    pixel_tolerance_px: float | None = None
    point_tolerance_m: float | None = None

    def __post_init__(self) -> None:
        """Require identities, finite coordinates and positive tolerances."""
        require_text("correspondence_id", self.correspondence_id)
        require_text("sample_id", self.sample_id)
        require_text("image_observation_id", self.image_observation_id)
        require_text("calibration_id", self.calibration_id)
        require_text("point_frame_id", self.point_frame_id)
        _finite("pixel_uv", self.pixel_uv)
        _finite("point_m", self.point_m)
        _tolerance("pixel", self.pixel_tolerance_px)
        _tolerance("point", self.point_tolerance_m)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "correspondence_id": self.correspondence_id,
            "sample_id": self.sample_id,
            "image_observation_id": self.image_observation_id,
            "calibration_id": self.calibration_id,
            "pixel_uv": list(self.pixel_uv),
            "point_frame_id": self.point_frame_id,
            "point_m": list(self.point_m),
            "point_source_observation_id": self.point_source_observation_id,
            "pixel_tolerance_px": self.pixel_tolerance_px,
            "point_tolerance_m": self.point_tolerance_m,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> GeometryCorrespondence:
        """Rebuild a correspondence from :meth:`to_record` output."""
        source = record["point_source_observation_id"]
        u, v = record["pixel_uv"]
        x, y, z = record["point_m"]
        return cls(
            correspondence_id=record["correspondence_id"],
            sample_id=ReferenceSampleId(record["sample_id"]),
            image_observation_id=SourceObservationId(record["image_observation_id"]),
            calibration_id=CalibrationReferenceId(record["calibration_id"]),
            pixel_uv=(float(u), float(v)),
            point_frame_id=record["point_frame_id"],
            point_m=(float(x), float(y), float(z)),
            point_source_observation_id=None if source is None else SourceObservationId(source),
            pixel_tolerance_px=record["pixel_tolerance_px"],
            point_tolerance_m=record["point_tolerance_m"],
        )


@dataclass(frozen=True, kw_only=True)
class GeometryAnnotationSet:
    """Trusted 3D <-> pixel correspondences."""

    family: ClassVar[AnnotationFamily] = AnnotationFamily.GEOMETRY

    correspondences: tuple[GeometryCorrespondence, ...]

    def __post_init__(self) -> None:
        """Require unique correspondence ids."""
        require_unique(
            "correspondence id", (item.correspondence_id for item in self.correspondences)
        )

    def observation_references(self) -> tuple[ObservationRef, ...]:
        """Return the distinct physical observations the annotations refer to."""
        refs: list[ObservationRef] = []
        for item in self.correspondences:
            refs.append(
                ObservationRef(sample_id=item.sample_id, observation_id=item.image_observation_id)
            )
            if item.point_source_observation_id is not None:
                refs.append(
                    ObservationRef(
                        sample_id=item.sample_id, observation_id=item.point_source_observation_id
                    )
                )
        return _unique_refs(refs)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"correspondences": [item.to_record() for item in self.correspondences]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> GeometryAnnotationSet:
        """Rebuild the set from :meth:`to_record` output."""
        return cls(
            correspondences=tuple(
                GeometryCorrespondence.from_record(item) for item in record["correspondences"]
            )
        )


# -------------------------------------------------------------------------- identity


@dataclass(frozen=True, kw_only=True)
class IdentityOccurrence:
    """One appearance of a physical entity in one observation."""

    sample_id: ReferenceSampleId
    observation_id: SourceObservationId
    region_id: str | None

    def __post_init__(self) -> None:
        """Require the observation identity."""
        require_text("sample_id", self.sample_id)
        require_text("observation_id", self.observation_id)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "sample_id": self.sample_id,
            "observation_id": self.observation_id,
            "region_id": self.region_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> IdentityOccurrence:
        """Rebuild an occurrence from :meth:`to_record` output."""
        return cls(
            sample_id=ReferenceSampleId(record["sample_id"]),
            observation_id=SourceObservationId(record["observation_id"]),
            region_id=record["region_id"],
        )


@dataclass(frozen=True, kw_only=True)
class PhysicalIdentity:
    """One physical entity and every observation the annotator saw it in.

    Identity is what the annotator declares here. It is never derived from a
    shared label: two objects with the same concept stay two identities.
    """

    identity_id: str
    occurrences: tuple[IdentityOccurrence, ...]
    description: str = ""

    def __post_init__(self) -> None:
        """Require an identity and unique occurrences."""
        require_text("identity_id", self.identity_id)
        if not self.occurrences:
            raise ValueError(
                f"identity {self.identity_id!r} needs at least one occurrence in an observation"
            )
        require_unique(
            f"occurrence of identity {self.identity_id!r}",
            (f"{o.sample_id}/{o.observation_id}/{o.region_id}" for o in self.occurrences),
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "identity_id": self.identity_id,
            "occurrences": [item.to_record() for item in self.occurrences],
            "description": self.description,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PhysicalIdentity:
        """Rebuild an identity from :meth:`to_record` output."""
        return cls(
            identity_id=record["identity_id"],
            occurrences=tuple(
                IdentityOccurrence.from_record(item) for item in record["occurrences"]
            ),
            description=record["description"],
        )


@dataclass(frozen=True, kw_only=True)
class DistinctIdentityPair:
    """An explicit annotation that two identities are **different** physical objects."""

    first: str
    second: str

    def __post_init__(self) -> None:
        """Require two different identities."""
        require_text("first identity", self.first)
        require_text("second identity", self.second)
        if self.first == self.second:
            raise ValueError("an identity cannot be distinct from itself")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"first": self.first, "second": self.second}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> DistinctIdentityPair:
        """Rebuild a pair from :meth:`to_record` output."""
        return cls(first=record["first"], second=record["second"])


@dataclass(frozen=True, kw_only=True)
class IdentityScope:
    """How exhaustively identities were annotated in one observation."""

    sample_id: ReferenceSampleId
    observation_id: SourceObservationId
    coverage: Coverage

    def __post_init__(self) -> None:
        """Require the observation identity."""
        require_text("sample_id", self.sample_id)
        require_text("observation_id", self.observation_id)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "sample_id": self.sample_id,
            "observation_id": self.observation_id,
            "coverage": self.coverage.value,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> IdentityScope:
        """Rebuild a scope entry from :meth:`to_record` output."""
        return cls(
            sample_id=ReferenceSampleId(record["sample_id"]),
            observation_id=SourceObservationId(record["observation_id"]),
            coverage=Coverage(record["coverage"]),
        )


@dataclass(frozen=True, kw_only=True)
class IdentityAnnotationSet:
    """Physical entity identity across observations.

    Only what is declared is truth: identities group occurrences, and
    ``distinct_pairs`` states explicit "not the same object". A pair that is
    neither grouped nor declared distinct is *not annotated*, not negative.
    ``scope`` says in which observations identities were annotated
    exhaustively (``COMPLETE``), which is what makes an unmatched predicted
    entity a duplicate rather than an unlabeled object.
    """

    family: ClassVar[AnnotationFamily] = AnnotationFamily.IDENTITY

    scope: tuple[IdentityScope, ...]
    identities: tuple[PhysicalIdentity, ...]
    distinct_pairs: tuple[DistinctIdentityPair, ...] = ()

    def __post_init__(self) -> None:
        """Require unique scope, identities and pairs that name declared identities."""
        require_unique(
            "identity scope (sample, observation)",
            (f"{item.sample_id}/{item.observation_id}" for item in self.scope),
        )
        require_unique("identity id", (item.identity_id for item in self.identities))
        declared = {item.identity_id for item in self.identities}
        seen: set[frozenset[str]] = set()
        for pair in self.distinct_pairs:
            for identity_id in (pair.first, pair.second):
                if identity_id not in declared:
                    raise ValueError(f"distinct pair references unknown identity {identity_id!r}")
            key = frozenset((pair.first, pair.second))
            if key in seen:
                raise ValueError(f"distinct pair {sorted(key)} is repeated")
            seen.add(key)

    def coverage_of(
        self, sample_id: ReferenceSampleId, observation_id: SourceObservationId
    ) -> Coverage | None:
        """Return how exhaustively an observation was annotated, or ``None`` if not at all."""
        for item in self.scope:
            if item.sample_id == sample_id and item.observation_id == observation_id:
                return item.coverage
        return None

    def observation_references(self) -> tuple[ObservationRef, ...]:
        """Return the distinct physical observations the annotations refer to."""
        refs = [
            ObservationRef(sample_id=item.sample_id, observation_id=item.observation_id)
            for item in self.scope
        ]
        refs.extend(
            ObservationRef(sample_id=occurrence.sample_id, observation_id=occurrence.observation_id)
            for identity in self.identities
            for occurrence in identity.occurrences
        )
        return _unique_refs(refs)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "scope": [item.to_record() for item in self.scope],
            "identities": [item.to_record() for item in self.identities],
            "distinct_pairs": [item.to_record() for item in self.distinct_pairs],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> IdentityAnnotationSet:
        """Rebuild the set from :meth:`to_record` output."""
        return cls(
            scope=tuple(IdentityScope.from_record(item) for item in record["scope"]),
            identities=tuple(PhysicalIdentity.from_record(item) for item in record["identities"]),
            distinct_pairs=tuple(
                DistinctIdentityPair.from_record(item) for item in record["distinct_pairs"]
            ),
        )


# ------------------------------------------------------------------------- relations


class RelationStatus(Enum):
    """What the annotator asserts about a relation between two identities."""

    HOLDS = "holds"
    DOES_NOT_HOLD = "does_not_hold"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True, kw_only=True)
class PredicateRule:
    """A predicate of the document's vocabulary and its logical properties.

    Attributes:
        predicate: The literal predicate, e.g. ``"next to"``.
        symmetric: ``A p B`` holds exactly when ``B p A`` does.
        inverse: The predicate ``q`` with ``A p B`` exactly when ``B q A``;
            the inverse must itself be declared, with ``p`` as its inverse.
    """

    predicate: str
    symmetric: bool = False
    inverse: str | None = None

    def __post_init__(self) -> None:
        """Reject contradictory logical properties."""
        require_text("predicate", self.predicate)
        if self.symmetric and self.inverse is not None:
            raise ValueError(f"a symmetric predicate {self.predicate!r} cannot declare an inverse")
        if self.inverse is not None:
            require_text("inverse", self.inverse)
            if _fold(self.inverse) == _fold(self.predicate):
                raise ValueError(
                    f"inverse of {self.predicate!r} must differ from it; declare it symmetric"
                )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"predicate": self.predicate, "symmetric": self.symmetric, "inverse": self.inverse}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PredicateRule:
        """Rebuild a rule from :meth:`to_record` output."""
        return cls(
            predicate=record["predicate"], symmetric=record["symmetric"], inverse=record["inverse"]
        )


@dataclass(frozen=True, kw_only=True)
class RelationAnnotation:
    """A relation between two annotated identities, anchored in physical observations.

    Attributes:
        relation_id: Identity inside the annotation set.
        subject_identity_id: The subject, an identity of the identity family.
        predicate: A predicate declared in the set's vocabulary.
        object_identity_id: The object, an identity of the identity family.
        status: Holds, explicitly does not hold, ambiguous or unknown.
        anchors: The observations in which the relation was judged.
    """

    relation_id: str
    subject_identity_id: str
    predicate: str
    object_identity_id: str
    status: RelationStatus
    anchors: tuple[ObservationRef, ...]

    def __post_init__(self) -> None:
        """Require identities, distinct ends and at least one physical anchor."""
        require_text("relation_id", self.relation_id)
        require_text("subject identity", self.subject_identity_id)
        require_text("object identity", self.object_identity_id)
        require_text("predicate", self.predicate)
        if self.subject_identity_id == self.object_identity_id:
            raise ValueError(f"relation {self.relation_id!r} relates an identity to itself")
        if not self.anchors:
            raise ValueError(f"relation {self.relation_id!r} needs at least one observation anchor")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "relation_id": self.relation_id,
            "subject_identity_id": self.subject_identity_id,
            "predicate": self.predicate,
            "object_identity_id": self.object_identity_id,
            "status": self.status.value,
            "anchors": [item.to_record() for item in self.anchors],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RelationAnnotation:
        """Rebuild a relation from :meth:`to_record` output."""
        return cls(
            relation_id=record["relation_id"],
            subject_identity_id=record["subject_identity_id"],
            predicate=record["predicate"],
            object_identity_id=record["object_identity_id"],
            status=RelationStatus(record["status"]),
            anchors=tuple(ObservationRef.from_record(item) for item in record["anchors"]),
        )


@dataclass(frozen=True, kw_only=True)
class RelationAnnotationSet:
    """Relations, their predicate vocabulary and the normalization it is compared under."""

    family: ClassVar[AnnotationFamily] = AnnotationFamily.RELATIONS

    normalization: LabelNormalization
    predicate_rules: tuple[PredicateRule, ...]
    relations: tuple[RelationAnnotation, ...]

    def __post_init__(self) -> None:
        """Require a coherent vocabulary and relations that only use it."""
        key = self.normalization.key
        require_unique("predicate rule", (key(rule.predicate) for rule in self.predicate_rules))
        by_key = {key(rule.predicate): rule for rule in self.predicate_rules}
        for rule in self.predicate_rules:
            if rule.inverse is None:
                continue
            other = by_key.get(key(rule.inverse))
            if other is None or other.inverse is None or key(other.inverse) != key(rule.predicate):
                raise ValueError(
                    f"inverse {rule.inverse!r} of {rule.predicate!r} must be declared "
                    "with the predicate as its own inverse"
                )
        require_unique("relation id", (item.relation_id for item in self.relations))
        for relation in self.relations:
            if key(relation.predicate) not in by_key:
                raise ValueError(
                    f"relation {relation.relation_id!r} uses undeclared predicate "
                    f"{relation.predicate!r}"
                )

    def observation_references(self) -> tuple[ObservationRef, ...]:
        """Return the distinct physical observations the annotations refer to."""
        return _unique_refs(anchor for item in self.relations for anchor in item.anchors)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "normalization": self.normalization.to_record(),
            "predicate_rules": [item.to_record() for item in self.predicate_rules],
            "relations": [item.to_record() for item in self.relations],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RelationAnnotationSet:
        """Rebuild the set from :meth:`to_record` output."""
        return cls(
            normalization=LabelNormalization.from_record(record["normalization"]),
            predicate_rules=tuple(
                PredicateRule.from_record(item) for item in record["predicate_rules"]
            ),
            relations=tuple(RelationAnnotation.from_record(item) for item in record["relations"]),
        )


# ------------------------------------------------------------------------ visibility


class VisibilityLevel(Enum):
    """How visible a target is in an observation; ``UNKNOWN`` is a judgement, not absence."""

    FULLY_VISIBLE = "fully_visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    HEAVILY_OCCLUDED = "heavily_occluded"
    NOT_VISIBLE = "not_visible"
    UNKNOWN = "unknown"


@dataclass(frozen=True, kw_only=True)
class VisibilityAnnotation:
    """Visibility of a region, an identity or the whole observation.

    Attributes:
        sample_id: The sample the target belongs to.
        observation_id: The observation the visibility is judged in.
        level: The visibility level.
        region_id: The region, when the target is a region.
        identity_id: The identity, when the target is a physical entity.
        occluded_fraction: Fraction of the target that is occluded, in [0, 1].
    """

    sample_id: ReferenceSampleId
    observation_id: SourceObservationId
    level: VisibilityLevel
    region_id: str | None = None
    identity_id: str | None = None
    occluded_fraction: float | None = None

    def __post_init__(self) -> None:
        """Require at most one target and a valid occluded fraction."""
        require_text("sample_id", self.sample_id)
        require_text("observation_id", self.observation_id)
        if self.region_id is not None and self.identity_id is not None:
            raise ValueError("a visibility annotation names at most one target, region or identity")
        if self.occluded_fraction is not None and not 0.0 <= self.occluded_fraction <= 1.0:
            raise ValueError("occluded fraction must be in [0, 1]")

    @property
    def target(self) -> tuple[str, str, str | None, str | None]:
        """Return the identity of the annotated target."""
        return (self.sample_id, self.observation_id, self.region_id, self.identity_id)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "sample_id": self.sample_id,
            "observation_id": self.observation_id,
            "level": self.level.value,
            "region_id": self.region_id,
            "identity_id": self.identity_id,
            "occluded_fraction": self.occluded_fraction,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> VisibilityAnnotation:
        """Rebuild an annotation from :meth:`to_record` output."""
        return cls(
            sample_id=ReferenceSampleId(record["sample_id"]),
            observation_id=SourceObservationId(record["observation_id"]),
            level=VisibilityLevel(record["level"]),
            region_id=record["region_id"],
            identity_id=record["identity_id"],
            occluded_fraction=record["occluded_fraction"],
        )


@dataclass(frozen=True, kw_only=True)
class VisibilityAnnotationSet:
    """Occlusion and visibility strata."""

    family: ClassVar[AnnotationFamily] = AnnotationFamily.VISIBILITY

    annotations: tuple[VisibilityAnnotation, ...]

    def __post_init__(self) -> None:
        """Require each visibility target to be annotated once."""
        require_unique("visibility target", (str(item.target) for item in self.annotations))

    def observation_references(self) -> tuple[ObservationRef, ...]:
        """Return the distinct physical observations the annotations refer to."""
        return _unique_refs(
            ObservationRef(sample_id=item.sample_id, observation_id=item.observation_id)
            for item in self.annotations
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"annotations": [item.to_record() for item in self.annotations]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> VisibilityAnnotationSet:
        """Rebuild the set from :meth:`to_record` output."""
        return cls(
            annotations=tuple(
                VisibilityAnnotation.from_record(item) for item in record["annotations"]
            )
        )


# --------------------------------------------------------------------- scene context


@dataclass(frozen=True, kw_only=True)
class SceneContextAnnotation:
    """Scene or context attributes of a sample or of one of its observations.

    ``observation_id`` is ``None`` when the attributes describe the sample as a
    whole.
    """

    sample_id: ReferenceSampleId
    observation_id: SourceObservationId | None
    attributes: tuple[ConceptAttribute, ...]

    def __post_init__(self) -> None:
        """Require at least one attribute, with unique names."""
        require_text("sample_id", self.sample_id)
        if not self.attributes:
            raise ValueError("a scene context annotation needs at least one attribute")
        require_unique("attribute name", (item.name for item in self.attributes))

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "sample_id": self.sample_id,
            "observation_id": self.observation_id,
            "attributes": [item.to_record() for item in self.attributes],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SceneContextAnnotation:
        """Rebuild an annotation from :meth:`to_record` output."""
        observation = record["observation_id"]
        return cls(
            sample_id=ReferenceSampleId(record["sample_id"]),
            observation_id=None if observation is None else SourceObservationId(observation),
            attributes=tuple(ConceptAttribute.from_record(item) for item in record["attributes"]),
        )


@dataclass(frozen=True, kw_only=True)
class SceneContextAnnotationSet:
    """Scene metadata used to stratify results, with the normalization of its values."""

    family: ClassVar[AnnotationFamily] = AnnotationFamily.SCENE_CONTEXT

    normalization: LabelNormalization
    annotations: tuple[SceneContextAnnotation, ...]

    def __post_init__(self) -> None:
        """Require each (sample, observation) to be described once."""
        require_unique(
            "scene context target",
            (f"{item.sample_id}/{item.observation_id}" for item in self.annotations),
        )

    def observation_references(self) -> tuple[ObservationRef, ...]:
        """Return the distinct physical observations the annotations refer to."""
        return _unique_refs(
            ObservationRef(sample_id=item.sample_id, observation_id=item.observation_id)
            for item in self.annotations
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "normalization": self.normalization.to_record(),
            "annotations": [item.to_record() for item in self.annotations],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SceneContextAnnotationSet:
        """Rebuild the set from :meth:`to_record` output."""
        return cls(
            normalization=LabelNormalization.from_record(record["normalization"]),
            annotations=tuple(
                SceneContextAnnotation.from_record(item) for item in record["annotations"]
            ),
        )


# ------------------------------------------------------------------------ documents

AnnotationSet: TypeAlias = (
    RegionAnnotationSet
    | SemanticAnnotationSet
    | GeometryAnnotationSet
    | IdentityAnnotationSet
    | RelationAnnotationSet
    | VisibilityAnnotationSet
    | SceneContextAnnotationSet
)
"""Any annotation family's document content."""

_SET_TYPES: dict[AnnotationFamily, type[AnnotationSet]] = {
    AnnotationFamily.REGIONS: RegionAnnotationSet,
    AnnotationFamily.SEMANTICS: SemanticAnnotationSet,
    AnnotationFamily.GEOMETRY: GeometryAnnotationSet,
    AnnotationFamily.IDENTITY: IdentityAnnotationSet,
    AnnotationFamily.RELATIONS: RelationAnnotationSet,
    AnnotationFamily.VISIBILITY: VisibilityAnnotationSet,
    AnnotationFamily.SCENE_CONTEXT: SceneContextAnnotationSet,
}


def encode_annotation_set(annotation_set: AnnotationSet) -> dict[str, Any]:
    """Return the annotation document, tagged with its versioned schema."""
    return {"schema": annotation_set.family.schema, **annotation_set.to_record()}


def decode_annotation_set(document: Mapping[str, Any]) -> AnnotationSet:
    """Rebuild an annotation set from a document, dispatching on its schema.

    Raises:
        AnnotationError: If the schema is unsupported or a field is missing or invalid.
    """
    try:
        family = AnnotationFamily.from_schema(document["schema"])
        return _SET_TYPES[family].from_record(document)
    except AnnotationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AnnotationError(f"invalid annotation document: {error}") from error


def write_annotation_set(path: Path, annotation_set: AnnotationSet) -> str:
    """Atomically publish an immutable annotation file and return its content hash.

    The returned ``sha256:<hex>`` is what the reference-set manifest records for
    the file.

    Raises:
        FileExistsError: If the file already exists.
    """
    write_immutable_json(path, encode_annotation_set(annotation_set), "annotation file")
    return file_digest(path)


def read_annotation_set(path: Path) -> AnnotationSet:
    """Read an annotation file of any family.

    Raises:
        AnnotationError: If the file is not valid JSON or not a valid annotation document.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AnnotationError(f"annotation file is not valid JSON: {error}") from error
    return decode_annotation_set(document)

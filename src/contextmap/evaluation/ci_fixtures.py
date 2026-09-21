"""Deterministic, model-independent CI fixture subset for cross-module regressions.

Full real-data evaluation needs large files, GPU models or external APIs. CI
still needs a small, stable subset that exercises the contracts and the known
failure modes of the implemented capabilities. This module *generates* that
subset from formulas only (no random numbers, no clock, no network, no model):

* a synthetic canonical sequence: RGB frames, LiDAR scans, external poses and a
  calibration set, with 3D landmarks whose projection is known analytically;
* a versioned reference set (manifest plus one annotation file per family) that
  passes the integrity validation;
* a catalogue of fixture cases, each with stable id, edge cases, expected
  outputs, tolerances, provenance, license and content hash, and an explicit
  coverage matrix that says which requirements the subset does **not** cover.

The subset guards regressions; it is not evidence of real-world quality, and
real-data evaluation stays separate from it. See
``src/contextmap/evaluation/docs/ci-fixtures.md``.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from contextmap.evaluation._persistence import (
    canonical_digest,
    write_immutable_json,
)
from contextmap.evaluation.annotations import (
    CASEFOLD_EXACT_POLICY,
    AnnotationSet,
    ConceptAttribute,
    Coverage,
    DistinctIdentityPair,
    FrameRegionAnnotation,
    GeometryAnnotationSet,
    GeometryCorrespondence,
    IdentityAnnotationSet,
    IdentityOccurrence,
    IdentityScope,
    LabelNormalization,
    ObservationRef,
    PhysicalIdentity,
    PredicateRule,
    RegionAnnotation,
    RegionAnnotationSet,
    RelationAnnotation,
    RelationAnnotationSet,
    RelationStatus,
    SceneContextAnnotation,
    SceneContextAnnotationSet,
    SemanticAnnotationRecord,
    SemanticAnnotationSet,
    SemanticStatus,
    VisibilityAnnotation,
    VisibilityAnnotationSet,
    VisibilityLevel,
    encode_annotation_set,
    write_annotation_set,
)
from contextmap.evaluation.reference_set import (
    AnnotationFileEntry,
    AnnotationProvenance,
    CalibrationIdentity,
    ProvenanceOrigin,
    ReferenceSample,
    ReferenceSampleId,
    ReferenceSetIdentity,
    ReferenceSetManifest,
    ReferenceSource,
    ReferenceSplit,
    ReferenceTrust,
    SampleGroup,
    SampleStratum,
    SampleTimeSpan,
    SourceKind,
    SplitRole,
    SplitScheme,
    SplitUnit,
    StratumDefinition,
    write_reference_set,
)
from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationSet,
    ExternalPoseMeasurement,
    FrameId,
    ImageEncoding,
    ImageObservation,
    LidarObservation,
    PinholeCameraModel,
    PointFieldDataType,
    PointFieldDescriptor,
    RigidTransform,
    SensorId,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import SourceTimestamp
from contextmap.visual_perception import InlineMask

CI_FIXTURE_ID = "contextmap-ci-subset"
"""Stable name of the CI fixture subset across versions."""

CI_FIXTURE_VERSION = "1.0.1"
"""Version of the subset; it changes whenever any generated file changes."""

CATALOGUE_SCHEMA = "contextmap.ci-fixtures/v1"
CATALOGUE_FILENAME = "catalogue.json"

_GENERATOR = "contextmap.evaluation.ci_fixtures"
_LICENSE = "AGPL-3.0 (repository LICENSE)"
_CLOCK = "fixture:header"
_SOURCE_ID = "ci-corridor"
_CAMERA_CALIBRATION = "front_camera-calib"
_LIDAR_CALIBRATION = "velodyne_top-calib"

IMAGE_WIDTH = 16
IMAGE_HEIGHT = 12
FOCAL_PX = 12.0
PRINCIPAL_POINT = (8.0, 6.0)
CAMERA_TRANSLATION_M = (0.1, 0.0, 0.2)
CAMERA_OPTICAL_ROTATION = (-0.5, 0.5, -0.5, 0.5)
LIDAR_TRANSLATION_M = (0.0, 0.0, 0.3)
FRAME_TIMES_S = (0, 1, 2)
"""The robot advances 1 m per second along +x, with identity orientation."""

LANDMARKS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("pallet-corner", (5.0, 0.5, 0.3)),
    ("shelf-post", (6.0, -0.4, 0.1)),
    ("behind-start", (-2.0, 0.0, 0.2)),
    ("far-left", (5.0, 4.0, 0.2)),
)
"""World landmarks; two are visible, one starts behind the camera, one falls outside."""


class FixtureCatalogueError(ValueError):
    """Raised when a fixture catalogue cannot be trusted."""


# ----------------------------------------------------------------- synthetic sequence


@dataclass(frozen=True, kw_only=True)
class SyntheticSequence:
    """The canonical ingestion objects of the synthetic corridor sequence."""

    name: str
    observations: tuple[SourceObservation, ...]
    calibration: CalibrationSet


def frame_ids(index: int) -> tuple[SourceObservationId, SourceObservationId, SourceObservationId]:
    """Return the image, LiDAR and pose observation ids of frame ``index``."""
    return (
        SourceObservationId(f"frame-{index:04d}"),
        SourceObservationId(f"scan-{index:04d}"),
        SourceObservationId(f"odom-{index:04d}"),
    )


def landmark_in_camera(
    world_point: tuple[float, float, float], index: int
) -> tuple[float, float, float]:
    """Return a world point in the camera optical frame at frame ``index``.

    The pose is a pure translation along +x and the optical frame follows the
    REP-103 convention, so the transform is written out by hand rather than
    through any capability: ``x_opt = -dy``, ``y_opt = -dz``, ``z_opt = dx``.
    """
    dx = world_point[0] - FRAME_TIMES_S[index] - CAMERA_TRANSLATION_M[0]
    dy = world_point[1] - CAMERA_TRANSLATION_M[1]
    dz = world_point[2] - CAMERA_TRANSLATION_M[2]
    return (-dy, -dz, dx)


def project_landmark(world_point: tuple[float, float, float], index: int) -> dict[str, Any]:
    """Return the expected pixel of a landmark and why it is (not) visible."""
    x, y, z = landmark_in_camera(world_point, index)
    if z <= 0.0:
        return {"status": "behind_camera", "pixel": None}
    pixel = (
        FOCAL_PX * x / z + PRINCIPAL_POINT[0],
        FOCAL_PX * y / z + PRINCIPAL_POINT[1],
    )
    inside = 0.0 <= pixel[0] < IMAGE_WIDTH and 0.0 <= pixel[1] < IMAGE_HEIGHT
    return {"status": "visible" if inside else "outside_image", "pixel": list(pixel)}


def _landmark_in_lidar(world_point: tuple[float, float, float], index: int) -> tuple[float, ...]:
    return (
        world_point[0] - FRAME_TIMES_S[index] - LIDAR_TRANSLATION_M[0],
        world_point[1] - LIDAR_TRANSLATION_M[1],
        world_point[2] - LIDAR_TRANSLATION_M[2],
    )


def _timestamp(index: int) -> SourceTimestamp:
    return SourceTimestamp(seconds=FRAME_TIMES_S[index], nanoseconds=0, clock_id=_CLOCK)


def _provenance(topic: str) -> SourceProvenance:
    return SourceProvenance(
        source_type="synthetic_fixture",
        source_path=f"{CI_FIXTURE_ID}/{CI_FIXTURE_VERSION}",
        source_topic=topic,
    )


def _image_bytes(index: int) -> bytes:
    return bytes(
        (x * 13 + y * 7 + channel * 31 + index * 5) % 256
        for y in range(IMAGE_HEIGHT)
        for x in range(IMAGE_WIDTH)
        for channel in range(3)
    )


def _lidar_bytes(index: int) -> bytes:
    return b"".join(
        struct.pack("<fff", *_landmark_in_lidar(point, index)) for _, point in LANDMARKS
    )


def build_calibration_set() -> CalibrationSet:
    """Return the calibration of the synthetic rig (pinhole camera, LiDAR, extrinsics)."""
    provenance = CalibrationProvenance(
        source_type="synthetic_fixture", source_path=f"{CI_FIXTURE_ID}/{CI_FIXTURE_VERSION}"
    )
    camera = PinholeCameraModel(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        fx=FOCAL_PX,
        fy=FOCAL_PX,
        cx=PRINCIPAL_POINT[0],
        cy=PRINCIPAL_POINT[1],
    )
    camera_entry = CalibrationEntry(
        calibration_id=CalibrationReferenceId(_CAMERA_CALIBRATION),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        camera_model=camera,
        provenance=provenance,
        content_hash=canonical_digest(_camera_parameters()),
    )
    lidar_entry = CalibrationEntry(
        calibration_id=CalibrationReferenceId(_LIDAR_CALIBRATION),
        sensor_id=SensorId("velodyne_top"),
        frame_id=FrameId("velodyne"),
        camera_model=None,
        provenance=provenance,
        content_hash=canonical_digest({"lidar_translation_m": LIDAR_TRANSLATION_M}),
    )
    identity = (0.0, 0.0, 0.0, 1.0)
    return CalibrationSet(
        entries={
            camera_entry.calibration_id: camera_entry,
            lidar_entry.calibration_id: lidar_entry,
        },
        static_transforms=(
            RigidTransform(
                parent_frame=FrameId("odom"),
                child_frame=FrameId("base_link"),
                translation=(0.0, 0.0, 0.0),
                rotation=identity,
            ),
            RigidTransform(
                parent_frame=FrameId("base_link"),
                child_frame=FrameId("front_camera_optical"),
                translation=CAMERA_TRANSLATION_M,
                rotation=CAMERA_OPTICAL_ROTATION,
            ),
            RigidTransform(
                parent_frame=FrameId("base_link"),
                child_frame=FrameId("velodyne"),
                translation=LIDAR_TRANSLATION_M,
                rotation=identity,
            ),
        ),
    )


def _camera_parameters() -> dict[str, Any]:
    return {
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "focal_px": FOCAL_PX,
        "principal_point_px": PRINCIPAL_POINT,
        "camera_translation_m": CAMERA_TRANSLATION_M,
        "camera_optical_rotation_xyzw": CAMERA_OPTICAL_ROTATION,
    }


def build_synthetic_sequence() -> SyntheticSequence:
    """Return the deterministic synthetic sequence: three frames of RGB, LiDAR and pose."""
    observations: list[SourceObservation] = []
    for index in range(len(FRAME_TIMES_S)):
        frame_id, scan_id, odom_id = frame_ids(index)
        observations.append(
            ImageObservation(
                observation_id=frame_id,
                sensor_id=SensorId("front_camera"),
                frame_id=FrameId("front_camera_optical"),
                timestamp=_timestamp(index),
                provenance=_provenance("/camera/image_raw"),
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                encoding=ImageEncoding.RGB8,
                data=_image_bytes(index),
            )
        )
        observations.append(
            LidarObservation(
                observation_id=scan_id,
                sensor_id=SensorId("velodyne_top"),
                frame_id=FrameId("velodyne"),
                timestamp=_timestamp(index),
                provenance=_provenance("/velodyne_points"),
                point_count=len(LANDMARKS),
                point_step_bytes=12,
                fields=tuple(
                    PointFieldDescriptor(
                        name=name, offset_bytes=offset, data_type=PointFieldDataType.FLOAT32
                    )
                    for name, offset in (("x", 0), ("y", 4), ("z", 8))
                ),
                data=_lidar_bytes(index),
            )
        )
        observations.append(
            ExternalPoseMeasurement(
                observation_id=odom_id,
                sensor_id=SensorId("wheel_odometry"),
                frame_id=FrameId("base_link"),
                timestamp=_timestamp(index),
                provenance=_provenance("/odom"),
                parent_frame=FrameId("odom"),
                translation=(float(FRAME_TIMES_S[index]), 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
        )
    return SyntheticSequence(
        name=CI_FIXTURE_ID,
        observations=tuple(observations),
        calibration=build_calibration_set(),
    )


def _payload_hash(observation: SourceObservation) -> str:
    data = getattr(observation, "data", b"")
    return canonical_digest({"id": observation.observation_id, "data": data.hex()})


# -------------------------------------------------------------------------- annotations


def _mask(*, x: range, y: range) -> InlineMask:
    return InlineMask(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        data=tuple(
            column in x and row in y for row in range(IMAGE_HEIGHT) for column in range(IMAGE_WIDTH)
        ),
    )


def _sample_id(index: int) -> ReferenceSampleId:
    return ReferenceSampleId(f"sample-{index:04d}")


def _frame(index: int) -> SourceObservationId:
    return frame_ids(index)[0]


def build_annotation_sets() -> dict[str, AnnotationSet]:
    """Return the annotation set of each family, keyed by its file stem."""
    region_a = _mask(x=range(2, 9), y=range(2, 8))
    region_b = _mask(x=range(6, 13), y=range(4, 10))
    exclusion = _mask(x=range(13, 16), y=range(9, 12))
    valid = _mask(x=range(1, 15), y=range(1, 11))
    exact = LabelNormalization(policy_id=CASEFOLD_EXACT_POLICY)
    s0, s1, s2 = (_sample_id(index) for index in range(3))
    f0, f1, f2 = (_frame(index) for index in range(3))

    regions = RegionAnnotationSet(
        frames=(
            FrameRegionAnnotation(
                sample_id=s0,
                observation_id=f0,
                image_width=IMAGE_WIDTH,
                image_height=IMAGE_HEIGHT,
                coverage=Coverage.COMPLETE,
                regions=(
                    RegionAnnotation(region_id="region-a", mask=region_a),
                    RegionAnnotation(region_id="region-b", mask=region_b),
                ),
                valid_areas=(valid,),
                exclusion_areas=(exclusion,),
            ),
            FrameRegionAnnotation(
                sample_id=s1,
                observation_id=f1,
                image_width=IMAGE_WIDTH,
                image_height=IMAGE_HEIGHT,
                coverage=Coverage.PARTIAL,
                regions=(
                    RegionAnnotation(
                        region_id="region-a1", mask=_mask(x=range(3, 10), y=range(2, 8))
                    ),
                    RegionAnnotation(
                        region_id="region-b1", mask=_mask(x=range(7, 14), y=range(4, 10))
                    ),
                ),
            ),
        )
    )
    semantics = SemanticAnnotationSet(
        normalization=exact,
        records=(
            SemanticAnnotationRecord(
                sample_id=s0,
                observation_id=f0,
                region_id="region-a",
                status=SemanticStatus.LABELED,
                concepts=("wooden pallet", "pallet"),
                rejected_concepts=("shelf",),
                attributes=(ConceptAttribute(name="material", value="wood"),),
            ),
            SemanticAnnotationRecord(
                sample_id=s0,
                observation_id=f0,
                region_id="region-b",
                status=SemanticStatus.AMBIGUOUS,
                concepts=("crate", "box"),
            ),
            SemanticAnnotationRecord(
                sample_id=s0,
                observation_id=f0,
                region_id=None,
                status=SemanticStatus.UNKNOWN,
                concepts=(),
            ),
        ),
    )
    correspondences = []
    for index in (0, 2):
        for name, point in LANDMARKS[:2]:
            expected = project_landmark(point, index)
            correspondences.append(
                GeometryCorrespondence(
                    correspondence_id=f"corr-{name}-{index}",
                    sample_id=_sample_id(index),
                    image_observation_id=_frame(index),
                    calibration_id=CalibrationReferenceId(_CAMERA_CALIBRATION),
                    pixel_uv=(expected["pixel"][0], expected["pixel"][1]),
                    point_frame_id="velodyne",
                    point_m=(
                        _landmark_in_lidar(point, index)[0],
                        _landmark_in_lidar(point, index)[1],
                        _landmark_in_lidar(point, index)[2],
                    ),
                    point_source_observation_id=frame_ids(index)[1],
                    pixel_tolerance_px=0.5,
                    point_tolerance_m=0.01,
                )
            )
    geometry = GeometryAnnotationSet(correspondences=tuple(correspondences))
    identity = IdentityAnnotationSet(
        scope=(
            IdentityScope(sample_id=s0, observation_id=f0, coverage=Coverage.COMPLETE),
            IdentityScope(sample_id=s1, observation_id=f1, coverage=Coverage.COMPLETE),
            IdentityScope(sample_id=s2, observation_id=f2, coverage=Coverage.PARTIAL),
        ),
        identities=(
            PhysicalIdentity(
                identity_id="pallet-1",
                occurrences=(
                    IdentityOccurrence(sample_id=s0, observation_id=f0, region_id="region-a"),
                    IdentityOccurrence(sample_id=s1, observation_id=f1, region_id="region-a1"),
                    IdentityOccurrence(sample_id=s2, observation_id=f2, region_id=None),
                ),
                description="the pallet the robot approaches",
            ),
            PhysicalIdentity(
                identity_id="pallet-2",
                occurrences=(
                    IdentityOccurrence(sample_id=s0, observation_id=f0, region_id="region-b"),
                    IdentityOccurrence(sample_id=s1, observation_id=f1, region_id="region-b1"),
                ),
                description="a second pallet with the same concept, a different object",
            ),
        ),
        distinct_pairs=(DistinctIdentityPair(first="pallet-1", second="pallet-2"),),
    )
    anchor0 = ObservationRef(sample_id=s0, observation_id=f0)
    anchor1 = ObservationRef(sample_id=s1, observation_id=f1)
    relations = RelationAnnotationSet(
        normalization=exact,
        predicate_rules=(
            PredicateRule(predicate="next to", symmetric=True),
            PredicateRule(predicate="on top of", inverse="supports"),
            PredicateRule(predicate="supports", inverse="on top of"),
        ),
        relations=(
            RelationAnnotation(
                relation_id="rel-next-to",
                subject_identity_id="pallet-1",
                predicate="next to",
                object_identity_id="pallet-2",
                status=RelationStatus.HOLDS,
                anchors=(anchor0, anchor1),
            ),
            RelationAnnotation(
                relation_id="rel-not-on-top",
                subject_identity_id="pallet-1",
                predicate="on top of",
                object_identity_id="pallet-2",
                status=RelationStatus.DOES_NOT_HOLD,
                anchors=(anchor0,),
            ),
            RelationAnnotation(
                relation_id="rel-supports-ambiguous",
                subject_identity_id="pallet-2",
                predicate="supports",
                object_identity_id="pallet-1",
                status=RelationStatus.AMBIGUOUS,
                anchors=(anchor1,),
            ),
        ),
    )
    visibility = VisibilityAnnotationSet(
        annotations=(
            VisibilityAnnotation(
                sample_id=s0,
                observation_id=f0,
                region_id="region-a",
                level=VisibilityLevel.FULLY_VISIBLE,
                occluded_fraction=0.0,
            ),
            VisibilityAnnotation(
                sample_id=s0,
                observation_id=f0,
                region_id="region-b",
                level=VisibilityLevel.PARTIALLY_OCCLUDED,
                occluded_fraction=0.3,
            ),
            VisibilityAnnotation(
                sample_id=s2,
                observation_id=f2,
                identity_id="pallet-2",
                level=VisibilityLevel.UNKNOWN,
            ),
        )
    )
    scene = SceneContextAnnotationSet(
        normalization=exact,
        annotations=(
            SceneContextAnnotation(
                sample_id=s0,
                observation_id=None,
                attributes=(
                    ConceptAttribute(name="environment", value="indoor corridor"),
                    ConceptAttribute(name="lighting", value="uniform"),
                ),
            ),
            SceneContextAnnotation(
                sample_id=s1,
                observation_id=f1,
                attributes=(ConceptAttribute(name="lighting", value="dim"),),
            ),
        ),
    )
    return {
        "regions": regions,
        "semantics": semantics,
        "geometry": geometry,
        "identity": identity,
        "relations": relations,
        "visibility": visibility,
        "scene_context": scene,
    }


# ------------------------------------------------------------------------- reference set


def _sample_hash(index: int) -> str:
    return canonical_digest(
        [
            _payload_hash(observation)
            for observation in build_synthetic_sequence().observations
            if observation.observation_id in frame_ids(index)
        ]
    )


def _samples() -> tuple[ReferenceSample, ...]:
    return tuple(
        ReferenceSample(
            sample_id=_sample_id(index),
            source_id=_SOURCE_ID,
            observation_ids=frame_ids(index),
            calibration_ids=(CalibrationReferenceId(_CAMERA_CALIBRATION),),
            time_span=SampleTimeSpan(start=_timestamp(index), end=_timestamp(index)),
            content_hash=_sample_hash(index),
            strata=(SampleStratum(name="visibility", value="clear"),),
            groups=(SampleGroup(unit=SplitUnit.SCENE, key="synthetic-corridor"),),
        )
        for index in range(len(FRAME_TIMES_S))
    )


def build_reference_set(annotation_hashes: Mapping[str, str]) -> ReferenceSetManifest:
    """Return the reference-set manifest, given the content hash of each annotation file."""
    samples = _samples()
    sequence = build_synthetic_sequence()
    source_hash = canonical_digest(
        [_payload_hash(item) for item in sequence.observations]
        + [canonical_digest(_camera_parameters())]
    )
    entries = []
    for stem, annotation_set in build_annotation_sets().items():
        covered = tuple(
            dict.fromkeys(item.sample_id for item in annotation_set.observation_references())
        )
        entries.append(
            AnnotationFileEntry(
                annotation_id=f"ann-{stem.replace('_', '-')}",
                schema=annotation_set.family.schema,
                path=f"annotations/{stem}.json",
                content_hash=annotation_hashes[stem],
                trust=ReferenceTrust.TRUSTED_GROUND_TRUTH,
                provenance_id="prov-synthetic",
                sample_ids=covered,
            )
        )
    return ReferenceSetManifest(
        reference_set_id=CI_FIXTURE_ID,
        version=CI_FIXTURE_VERSION,
        sources=(
            ReferenceSource(
                source_id=_SOURCE_ID,
                kind=SourceKind.SYNTHETIC_FIXTURE,
                identity=f"{CI_FIXTURE_ID}/{CI_FIXTURE_VERSION}/sequence",
                content_hash=source_hash,
                license=_LICENSE,
                redistributable=True,
            ),
        ),
        calibrations=(
            CalibrationIdentity(
                calibration_id=CalibrationReferenceId(_CAMERA_CALIBRATION),
                source_id=_SOURCE_ID,
                content_hash=canonical_digest(_camera_parameters()),
            ),
        ),
        stratum_definitions=(
            StratumDefinition(
                name="visibility",
                values=("clear", "partially_occluded"),
                description="occlusion level of the main subject, assigned by construction",
            ),
        ),
        samples=samples,
        provenance=(
            AnnotationProvenance(
                provenance_id="prov-synthetic",
                origin=ProvenanceOrigin.SYNTHETIC_GENERATION,
                annotator=_GENERATOR,
                method="analytic construction by the fixture generator; no model is involved",
                tool_version=CI_FIXTURE_VERSION,
            ),
        ),
        annotations=tuple(entries),
        split_schemes=(
            SplitScheme(
                scheme_id="ci-regression",
                task="cross_module_regression",
                unit=SplitUnit.SEQUENCE,
                rationale=(
                    "the subset is never tuned on: it only guards regressions, so it has a "
                    "single held-out split over its one sequence"
                ),
                splits=(
                    ReferenceSplit(
                        name="ci",
                        role=SplitRole.TEST,
                        sample_ids=tuple(sample.sample_id for sample in samples),
                    ),
                ),
            ),
        ),
    )


# --------------------------------------------------------------------------- catalogue


class CoverageStatus(Enum):
    """Whether the subset covers a requirement, and at which level."""

    COVERED = "covered"
    ANNOTATION_LEVEL = "annotation_level"
    NOT_YET_AVAILABLE = "not_yet_available"


@dataclass(frozen=True, kw_only=True)
class FixtureCase:
    """One fixture: stable id, what it exercises, what to expect and how exactly.

    Attributes:
        fixture_id: Stable identity of the case.
        capability: The capability whose behavior the case guards.
        description: What the case checks.
        edge_cases: Named edge cases the case represents explicitly.
        inputs: JSON-compatible inputs (parameters or canned model responses).
        expected: JSON-compatible expected outputs.
        tolerances: Named numeric tolerances for comparing outputs with ``expected``.
        generator: What produced the case.
        license: License of the data.
        redistributable: Whether the data may be redistributed with the repository.
        content_hash: ``sha256:<hex>`` of inputs, expected outputs and tolerances.
    """

    fixture_id: str
    capability: str
    description: str
    edge_cases: tuple[str, ...]
    inputs: Mapping[str, Any]
    expected: Mapping[str, Any]
    tolerances: Mapping[str, float]
    generator: str
    license: str
    redistributable: bool
    content_hash: str

    def __post_init__(self) -> None:
        """Require identity, provenance and a content hash that matches the content."""
        for name in ("fixture_id", "capability", "description", "generator", "license"):
            if not getattr(self, name).strip():
                raise FixtureCatalogueError(f"fixture case {name} must not be empty")
        if self.content_hash != self.compute_hash():
            raise FixtureCatalogueError(
                f"fixture case {self.fixture_id!r} does not match its content hash"
            )

    def compute_hash(self) -> str:
        """Return the hash of the case content."""
        return canonical_digest(
            {"inputs": self.inputs, "expected": self.expected, "tolerances": self.tolerances}
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "fixture_id": self.fixture_id,
            "capability": self.capability,
            "description": self.description,
            "edge_cases": list(self.edge_cases),
            "inputs": self.inputs,
            "expected": self.expected,
            "tolerances": self.tolerances,
            "generator": self.generator,
            "license": self.license,
            "redistributable": self.redistributable,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FixtureCase:
        """Rebuild a case from :meth:`to_record` output, verifying its hash."""
        return cls(
            fixture_id=record["fixture_id"],
            capability=record["capability"],
            description=record["description"],
            edge_cases=tuple(record["edge_cases"]),
            inputs=record["inputs"],
            expected=record["expected"],
            tolerances=record["tolerances"],
            generator=record["generator"],
            license=record["license"],
            redistributable=record["redistributable"],
            content_hash=record["content_hash"],
        )


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _case(
    fixture_id: str,
    capability: str,
    description: str,
    *,
    edge_cases: tuple[str, ...],
    inputs: Mapping[str, Any],
    expected: Mapping[str, Any],
    tolerances: Mapping[str, float] | None = None,
) -> FixtureCase:
    plain_inputs = _plain(inputs)
    plain_expected = _plain(expected)
    tolerance_values = _plain(dict(tolerances or {}))
    content = canonical_digest(
        {"inputs": plain_inputs, "expected": plain_expected, "tolerances": tolerance_values}
    )
    return FixtureCase(
        fixture_id=fixture_id,
        capability=capability,
        description=description,
        edge_cases=edge_cases,
        inputs=plain_inputs,
        expected=plain_expected,
        tolerances=tolerance_values,
        generator=_GENERATOR,
        license=_LICENSE,
        redistributable=True,
        content_hash=content,
    )


@dataclass(frozen=True, kw_only=True)
class CoverageEntry:
    """What the subset does for one requirement, and why when it does nothing."""

    requirement: str
    status: CoverageStatus
    fixture_ids: tuple[str, ...]
    note: str

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "requirement": self.requirement,
            "status": self.status.value,
            "fixture_ids": list(self.fixture_ids),
            "note": self.note,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CoverageEntry:
        """Rebuild an entry from :meth:`to_record` output."""
        return cls(
            requirement=record["requirement"],
            status=CoverageStatus(record["status"]),
            fixture_ids=tuple(record["fixture_ids"]),
            note=record["note"],
        )


@dataclass(frozen=True, kw_only=True)
class FixtureCatalogue:
    """The versioned catalogue of the CI fixture subset.

    Attributes:
        catalogue_id: Stable name of the subset.
        version: Subset version; it changes whenever any generated file does.
        reference_set: The reference set the subset ships with.
        cases: The fixture cases.
        coverage: One entry per requirement of the CI subset, covered or not.
    """

    catalogue_id: str
    version: str
    reference_set: ReferenceSetIdentity
    cases: tuple[FixtureCase, ...]
    coverage: tuple[CoverageEntry, ...]

    def __post_init__(self) -> None:
        """Require unique cases and coverage entries that name existing cases."""
        ids = [case.fixture_id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise FixtureCatalogueError("fixture ids must be unique")
        for entry in self.coverage:
            for fixture_id in entry.fixture_ids:
                if fixture_id not in ids:
                    raise FixtureCatalogueError(
                        f"coverage of {entry.requirement!r} names unknown fixture {fixture_id!r}"
                    )

    def case(self, fixture_id: str) -> FixtureCase:
        """Return a case by identity."""
        for case in self.cases:
            if case.fixture_id == fixture_id:
                return case
        raise FixtureCatalogueError(f"unknown fixture {fixture_id!r}")

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible content the digest is computed over."""
        return {
            "schema": CATALOGUE_SCHEMA,
            "catalogue_id": self.catalogue_id,
            "version": self.version,
            "reference_set": self.reference_set.to_record(),
            "cases": [case.to_record() for case in self.cases],
            "coverage": [entry.to_record() for entry in self.coverage],
        }

    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the whole catalogue."""
        return canonical_digest(self.to_record())


def encode_catalogue(catalogue: FixtureCatalogue) -> dict[str, Any]:
    """Return the catalogue document, including its own digest."""
    return {**catalogue.to_record(), "digest": catalogue.digest()}


def decode_catalogue(document: Mapping[str, Any]) -> FixtureCatalogue:
    """Rebuild a catalogue, verifying its schema, every case hash and its digest.

    Raises:
        FixtureCatalogueError: If anything does not match.
    """
    try:
        if document["schema"] != CATALOGUE_SCHEMA:
            raise FixtureCatalogueError(f"unsupported catalogue schema {document['schema']!r}")
        identity = document["reference_set"]
        catalogue = FixtureCatalogue(
            catalogue_id=document["catalogue_id"],
            version=document["version"],
            reference_set=ReferenceSetIdentity.from_record(identity),
            cases=tuple(FixtureCase.from_record(item) for item in document["cases"]),
            coverage=tuple(CoverageEntry.from_record(item) for item in document["coverage"]),
        )
        declared = document["digest"]
    except FixtureCatalogueError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise FixtureCatalogueError(f"invalid fixture catalogue: {error}") from error
    if catalogue.digest() != declared:
        raise FixtureCatalogueError("fixture catalogue digest mismatch")
    return catalogue


def read_catalogue(root: Path) -> FixtureCatalogue:
    """Read and verify the catalogue under ``root``."""
    text = (root / CATALOGUE_FILENAME).read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise FixtureCatalogueError(f"fixture catalogue is not valid JSON: {error}") from error
    return decode_catalogue(document)


# Canned model responses: the parser is the subject, no model is involved.
_RESPONSE_PRIMARY_WITH_ALTERNATIVE = {
    "abstained": False,
    "claims": [
        {
            "hypothesis": "wooden pallet",
            "role": "primary",
            "category": None,
            "region_kind": "thing",
            "attributes": {},
            "confidence": None,
        },
        {
            "hypothesis": "wooden crate",
            "role": "alternative",
            "category": None,
            "region_kind": "thing",
            "attributes": {},
            "confidence": None,
        },
    ],
    "scene_context": None,
}
_RESPONSE_ABSTENTION = {"abstained": True, "claims": [], "scene_context": None}
_RESPONSE_MEASURED = {
    "abstained": False,
    "claims": [
        {
            "hypothesis": "wooden pallet",
            "role": "primary",
            "category": None,
            "region_kind": "thing",
            "attributes": {},
            "confidence": 0.62,
        }
    ],
    "scene_context": None,
}
_RESPONSE_REPEAT = {
    "abstained": False,
    "claims": [
        {
            "hypothesis": "pallet",
            "role": "primary",
            "category": None,
            "region_kind": "thing",
            "attributes": {},
            "confidence": None,
        }
    ],
    "scene_context": None,
}


def _overlap_expected() -> dict[str, Any]:
    region_a = _mask(x=range(2, 9), y=range(2, 8))
    region_b = _mask(x=range(6, 13), y=range(4, 10))
    inter = sum(a and b for a, b in zip(region_a.data, region_b.data, strict=True))
    union = sum(a or b for a, b in zip(region_a.data, region_b.data, strict=True))
    return {
        "area_a": region_a.area,
        "area_b": region_b.area,
        "intersection": inter,
        "union": union,
        "iou": inter / union,
    }


def build_catalogue(reference_set: ReferenceSetIdentity) -> FixtureCatalogue:
    """Return the catalogue of fixture cases and the explicit coverage matrix."""
    sequence = build_synthetic_sequence()
    counts = {"image": 0, "lidar": 0, "external_pose": 0}
    for observation in sequence.observations:
        if isinstance(observation, ImageObservation):
            counts["image"] += 1
        elif isinstance(observation, LidarObservation):
            counts["lidar"] += 1
        else:
            counts["external_pose"] += 1
    projections = {
        f"{name}@{index}": project_landmark(point, index)
        for index in range(len(FRAME_TIMES_S))
        for name, point in LANDMARKS
    }
    cases = (
        _case(
            "ingestion-canonical-sequence",
            "ingestion",
            "The synthetic RGB/LiDAR/pose sequence round-trips through a sequence artifact "
            "with the same identities, timestamps and payload bytes.",
            edge_cases=("shared-clock-domain", "float32-xyz-point-layout", "rgb8-payload"),
            inputs={
                "sequence": CI_FIXTURE_ID,
                "frame_times_s": list(FRAME_TIMES_S),
                "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
                "landmark_count": len(LANDMARKS),
            },
            expected={
                "observation_counts": counts,
                "observation_ids": [str(item.observation_id) for item in sequence.observations],
                "payload_hashes": {
                    str(item.observation_id): _payload_hash(item) for item in sequence.observations
                },
            },
        ),
        _case(
            "projection-known-landmarks",
            "sensor_association",
            "Known 3D landmarks project to analytically known pixels through the calibrated "
            "pinhole model; one landmark starts behind the camera and one falls outside.",
            edge_cases=("behind-camera", "outside-image", "pose-dependent-projection"),
            inputs={
                "camera": _camera_parameters(),
                "landmarks": {name: list(point) for name, point in LANDMARKS},
                "frame_times_s": list(FRAME_TIMES_S),
            },
            expected={"projections": projections},
            tolerances={"pixel_abs": 1e-6},
        ),
        _case(
            "trajectory-external-pose",
            "state_estimation",
            "External pose measurements become a canonical trajectory with the recorded "
            "translations and identity orientation.",
            edge_cases=("identity-orientation", "constant-velocity"),
            inputs={"frame_times_s": list(FRAME_TIMES_S)},
            expected={
                "translations_m": [[float(second), 0.0, 0.0] for second in FRAME_TIMES_S],
                "pose_observation_ids": [str(frame_ids(i)[2]) for i in range(len(FRAME_TIMES_S))],
            },
            tolerances={"translation_abs_m": 1e-9},
        ),
        _case(
            "regions-overlapping-masks",
            "visual_perception",
            "Two annotated regions overlap; their areas, intersection and IoU are known.",
            edge_cases=("overlapping-regions", "exclusion-area", "valid-area", "partial-coverage"),
            inputs={"frame": str(_frame(0)), "regions": ["region-a", "region-b"]},
            expected=_overlap_expected(),
            tolerances={"iou_abs": 1e-12},
        ),
        _case(
            "semantic-claims-variants",
            "visual_perception",
            "Canned model responses parse into canonical claims: a primary with an "
            "alternative, an abstention, an unscored claim and a measured one; a repeated "
            "inference over one physical observation stays one observation.",
            edge_cases=(
                "semantic-alternative",
                "abstention",
                "unscored-claim",
                "measured-claim",
                "repeated-inference-one-observation",
            ),
            inputs={
                "observation_id": str(_frame(0)),
                "responses": {
                    "primary_with_alternative": _RESPONSE_PRIMARY_WITH_ALTERNATIVE,
                    "abstention": _RESPONSE_ABSTENTION,
                    "measured": _RESPONSE_MEASURED,
                    "repeated_inference": _RESPONSE_REPEAT,
                },
            },
            expected={
                "primary_with_alternative": {
                    "abstained": False,
                    "roles": ["primary", "alternative"],
                    "confidences": [None, None],
                },
                "abstention": {"abstained": True, "roles": [], "confidences": []},
                "measured": {"abstained": False, "roles": ["primary"], "confidences": [0.62]},
                "repeated_inference": {
                    "abstained": False,
                    "roles": ["primary"],
                    "confidences": [None],
                },
                "physical_observation_count": 1,
                "inference_count": 2,
            },
            tolerances={"confidence_abs": 1e-12},
        ),
        _case(
            "identity-duplicate-and-distinct",
            "entity_resolution",
            "Two objects with the same concept are distinct identities, while one object "
            "seen in several frames is a single identity (annotation level).",
            edge_cases=("same-label-distinct-objects", "one-object-many-observations"),
            inputs={"annotation": "identity"},
            expected={
                "identities": {"pallet-1": 3, "pallet-2": 2},
                "distinct_pairs": [["pallet-1", "pallet-2"]],
                "exhaustive_observations": [str(_frame(0)), str(_frame(1))],
                "partial_observations": [str(_frame(2))],
            },
        ),
        _case(
            "relations-symmetry-and-negatives",
            "spatial_relations",
            "A symmetric relation, an explicit negative and an ambiguous relation with "
            "declared inverse predicates (annotation level).",
            edge_cases=("symmetric-predicate", "explicit-negative", "ambiguous-relation"),
            inputs={"annotation": "relations"},
            expected={
                "statuses": {
                    "rel-next-to": "holds",
                    "rel-not-on-top": "does_not_hold",
                    "rel-supports-ambiguous": "ambiguous",
                },
                "symmetric": ["next to"],
                "inverses": {"on top of": "supports", "supports": "on top of"},
            },
        ),
        _case(
            "visibility-and-context-strata",
            "evaluation",
            "Visibility levels, including unknown, and scene attributes used to stratify results.",
            edge_cases=("visibility-unknown", "occluded-fraction", "sample-level-context"),
            inputs={"annotation": ["visibility", "scene_context"]},
            expected={
                "levels": ["fully_visible", "partially_occluded", "unknown"],
                "occluded_fractions": [0.0, 0.3, None],
            },
        ),
        _case(
            "reference-set-integrity",
            "evaluation",
            "The subset's own reference set passes the integrity validation with no finding.",
            edge_cases=("synthetic-provenance", "single-held-out-split"),
            inputs={"reference_set": CI_FIXTURE_ID},
            expected={"valid": True, "blockers": 0, "warnings": 0},
        ),
    )
    coverage = (
        CoverageEntry(
            requirement="canonical RGB/LiDAR/pose/calibration ingestion",
            status=CoverageStatus.COVERED,
            fixture_ids=("ingestion-canonical-sequence", "trajectory-external-pose"),
            note="",
        ),
        CoverageEntry(
            requirement="known 3D transforms/projections",
            status=CoverageStatus.COVERED,
            fixture_ids=("projection-known-landmarks",),
            note="camera model and extrinsic composition; the full frame projector is not run",
        ),
        CoverageEntry(
            requirement="2D masks/regions and overlapping regions",
            status=CoverageStatus.COVERED,
            fixture_ids=("regions-overlapping-masks",),
            note="",
        ),
        CoverageEntry(
            requirement="semantic alternatives/abstention/unscored claims",
            status=CoverageStatus.COVERED,
            fixture_ids=("semantic-claims-variants",),
            note="canned responses through the public parser, no model",
        ),
        CoverageEntry(
            requirement="repeated inference over one physical observation",
            status=CoverageStatus.COVERED,
            fixture_ids=("semantic-claims-variants",),
            note="one physical observation, two inferences",
        ),
        CoverageEntry(
            requirement="multi-view fusion",
            status=CoverageStatus.NOT_YET_AVAILABLE,
            fixture_ids=(),
            note=(
                "needs synthetic Sensor Association outputs (spatial observations with "
                "geometry references); Semantic Fusion is covered by its own capability tests"
            ),
        ),
        CoverageEntry(
            requirement="entity duplicate/distinct cases",
            status=CoverageStatus.ANNOTATION_LEVEL,
            fixture_ids=("identity-duplicate-and-distinct",),
            note="Entity and EntityResolution contracts do not exist yet",
        ),
        CoverageEntry(
            requirement="spatial relation fixtures",
            status=CoverageStatus.ANNOTATION_LEVEL,
            fixture_ids=("relations-symmetry-and-negatives",),
            note="Relation contracts do not exist yet",
        ),
        CoverageEntry(
            requirement="final ContextMapArtifact round-trip",
            status=CoverageStatus.NOT_YET_AVAILABLE,
            fixture_ids=(),
            note="the ContextMapArtifact schema and serialization do not exist yet",
        ),
    )
    return FixtureCatalogue(
        catalogue_id=CI_FIXTURE_ID,
        version=CI_FIXTURE_VERSION,
        reference_set=reference_set,
        cases=cases,
        coverage=coverage,
    )


# ------------------------------------------------------------------------ generation


@dataclass(frozen=True, kw_only=True)
class CiFixtureSubset:
    """Everything the generator produced."""

    manifest: ReferenceSetManifest
    catalogue: FixtureCatalogue
    sequence: SyntheticSequence


def generate_ci_fixture_subset(root: Path) -> CiFixtureSubset:
    """Write the whole subset under ``root`` and return what was generated.

    ``root`` is a version directory (``<subset>/<version>``): it receives
    ``manifest.json``, ``catalogue.json`` and ``annotations/``. Writing is
    immutable: an existing file makes the generator fail rather than overwrite it.
    """
    hashes = {
        stem: write_annotation_set(root / "annotations" / f"{stem}.json", annotation_set)
        for stem, annotation_set in build_annotation_sets().items()
    }
    manifest = build_reference_set(hashes)
    write_reference_set(root, manifest)
    catalogue = build_catalogue(manifest.identity())
    write_immutable_json(
        root / CATALOGUE_FILENAME, encode_catalogue(catalogue), "fixture catalogue"
    )
    return CiFixtureSubset(
        manifest=manifest, catalogue=catalogue, sequence=build_synthetic_sequence()
    )


def annotation_documents() -> dict[str, dict[str, Any]]:
    """Return the annotation documents by file stem, as they are written to disk."""
    return {
        stem: encode_annotation_set(annotation_set)
        for stem, annotation_set in build_annotation_sets().items()
    }

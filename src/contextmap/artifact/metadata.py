"""Top-level metadata of a ContextMap.

The metadata says what a map is, where it came from and how it was created, so a consumer can
interpret the rest of the artifact without opening anything upstream: the coordinate frame and
units, the spatial and temporal extent, the source sequences, and which optional content the
map really carries. It holds no filesystem path, no serializer detail and no backend or model
object, and content is never implied to exist from a file name or layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from contextmap.artifact._checks import (
    require_artifact_identity,
    require_canonical,
    require_optional_present,
    require_present,
)
from contextmap.artifact.frame import MapFrame
from contextmap.geometric_mapping import Bounds3D
from contextmap.shared import SourceTimestamp
from contextmap.spatial_relations import RelationPredicate


@dataclass(frozen=True, kw_only=True, order=True)
class PolicyRef:
    """A versioned rule that produced a result.

    Attributes:
        policy_id: Identity of the rule.
        version: Version of the rule; a different version is a different rule.
    """

    policy_id: str
    version: str

    def __post_init__(self) -> None:
        """Require both parts of the identity.

        Raises:
            ValueError: If the policy id or the version is blank.
        """
        require_present(self, "policy_id", "version")


@dataclass(frozen=True, kw_only=True)
class MapCreation:
    """How the map was created.

    A run that is executed again produces another map with another identity, so the creation
    identity records the rule and the code that assembled this one, never a wall-clock time.

    Attributes:
        assembly_policy: The versioned rule that composed geometry, entities and relations.
        code_version: Code revision that assembled the map; ``None`` when it is not known.
        configuration_fingerprint: Hash of the effective assembly configuration; ``None`` when
            nothing was configurable.
    """

    assembly_policy: PolicyRef
    code_version: str | None
    configuration_fingerprint: str | None

    def __post_init__(self) -> None:
        """Require that an identity that is present is not blank.

        Raises:
            ValueError: If ``code_version`` or ``configuration_fingerprint`` is set but blank.
        """
        require_optional_present(self, "code_version", "configuration_fingerprint")


@dataclass(frozen=True, kw_only=True, order=True)
class SourceSequence:
    """A canonical sequence, and the selection of it, that the map was built from.

    Attributes:
        sequence_artifact_id: Identity of the immutable sequence artifact.
        selection_id: Deterministic identity of the part of the sequence that was used.
    """

    sequence_artifact_id: str
    selection_id: str

    def __post_init__(self) -> None:
        """Require both identities.

        Raises:
            ValueError: If an identity is blank or is a filesystem path.
        """
        require_artifact_identity(self, "sequence_artifact_id")
        require_present(self, "selection_id")


@dataclass(frozen=True, kw_only=True)
class ObservationWindow:
    """The closed time interval covered by the observations the map was built from.

    Both ends are in one clock domain: instants of different clocks are never compared.

    Attributes:
        start: First instant covered.
        end: Last instant covered; not before ``start``.
    """

    start: SourceTimestamp
    end: SourceTimestamp

    def __post_init__(self) -> None:
        """Validate that both ends share a clock and are ordered.

        Raises:
            ValueError: If the clock domains differ or ``end`` precedes ``start``.
        """
        if self.start.clock_id != self.end.clock_id:
            raise ValueError(
                f"an observation window must share one clock domain, got "
                f"{self.start.clock_id!r} and {self.end.clock_id!r}"
            )
        if self.end.total_nanoseconds() < self.start.total_nanoseconds():
            raise ValueError("an observation window end must not precede its start")


class MapCapability(Enum):
    """Optional content a map may carry, declared instead of implied.

    A capability is declared when the stage that produces it contributed to the map, even if it
    found nothing: an empty relation set is different from relations that were never computed.

    Attributes:
        GEOMETRY: Persistent geometry, referenced from the geometric-map artifact.
        ENTITIES: Resolved semantic entities.
        RELATIONS: Spatial relations between entities.
        POINT_REPRESENTATION_EVIDENCE: References to 3D point-representation evidence.
    """

    GEOMETRY = "geometry"
    ENTITIES = "entities"
    POINT_REPRESENTATION_EVIDENCE = "point_representation_evidence"
    RELATIONS = "relations"


@dataclass(frozen=True, kw_only=True)
class DeclaredCapabilities:
    """The optional content a map declares.

    Attributes:
        content: The declared capabilities, sorted and unique; geometry is always among them.
        relation_predicates: The canonical predicates present, sorted by value and unique; empty
            unless relations are declared.
    """

    content: tuple[MapCapability, ...]
    relation_predicates: tuple[RelationPredicate, ...]

    def __post_init__(self) -> None:
        """Validate that the declaration is canonical and coherent.

        Raises:
            ValueError: If the declaration is not sorted and unique, omits geometry, declares
                relations without entities, or lists predicates without relations.
        """
        require_canonical("content", self.content, lambda item: (item.value,), detail="by value ")
        if MapCapability.GEOMETRY not in self.content:
            raise ValueError("a map always declares its geometry capability")
        if MapCapability.RELATIONS in self.content and MapCapability.ENTITIES not in self.content:
            raise ValueError("relations are between entities: declare entities as well")
        require_canonical(
            "relation_predicates",
            self.relation_predicates,
            lambda item: (item.value,),
            detail="by value ",
        )
        if self.relation_predicates and MapCapability.RELATIONS not in self.content:
            raise ValueError("relation_predicates are only declared with the relations capability")


@dataclass(frozen=True, kw_only=True)
class ContextMapMetadata:
    """What a map is and where it came from.

    Attributes:
        creation: How the map was assembled.
        source_sequences: Every sequence selection the map was built from, sorted and unique.
        frame: The coordinate frame, units, axes and origin of every position of the map.
        bounds: Axis-aligned spatial extent of the map, expressed in ``frame``.
        time_bounds: Time window covered by the observations the map was built from.
        capabilities: The optional content the map declares.
    """

    creation: MapCreation
    source_sequences: tuple[SourceSequence, ...]
    frame: MapFrame
    bounds: Bounds3D
    time_bounds: ObservationWindow
    capabilities: DeclaredCapabilities

    def __post_init__(self) -> None:
        """Validate that the sources are declared and the extent is in the map frame.

        Raises:
            ValueError: If there is no source sequence, they are not sorted and unique, or the
                bounds are expressed in another frame: a frame is never inferred.
        """
        if not self.source_sequences:
            raise ValueError("source_sequences must list at least one sequence")
        require_canonical(
            "source_sequences",
            self.source_sequences,
            lambda item: (item.sequence_artifact_id, item.selection_id),
        )
        if self.bounds.frame_id != self.frame.frame_id:
            raise ValueError(
                f"bounds are expressed in frame {self.bounds.frame_id!r}, not in the map frame "
                f"{self.frame.frame_id!r}"
            )

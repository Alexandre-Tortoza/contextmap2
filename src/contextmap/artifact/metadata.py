"""Top-level metadata of a ContextMap.

The metadata says what a map is, where it came from and how it was created, so a consumer can
interpret the rest of the artifact without opening anything upstream. It carries no filesystem
path, no serializer detail and no backend or model object.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextmap.artifact._checks import (
    require_artifact_identity,
    require_canonical,
    require_optional_present,
    require_present,
)


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
class ContextMapMetadata:
    """What a map is and where it came from.

    Attributes:
        creation: How the map was assembled.
        source_sequences: Every sequence selection the map was built from, sorted and unique.
    """

    creation: MapCreation
    source_sequences: tuple[SourceSequence, ...]

    def __post_init__(self) -> None:
        """Validate that the sources are declared and canonical.

        Raises:
            ValueError: If there is no source sequence or they are not sorted and unique.
        """
        if not self.source_sequences:
            raise ValueError("source_sequences must list at least one sequence")
        require_canonical(
            "source_sequences",
            self.source_sequences,
            lambda item: (item.sequence_artifact_id, item.selection_id),
        )

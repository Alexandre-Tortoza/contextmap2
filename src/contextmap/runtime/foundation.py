"""The Spatial Foundation: the one sequence, trajectory and map a contextual map is built on.

Incremental context (``docs/runtime-composition.md``, "Contexto incremental (v0.1.1)") adds
evidence to a contextual map over time. That evidence is only comparable when every piece of it
refers to the same spatial world model, so the runtime pins one ``SequenceArtifact``, one
``StateEstimationRunArtifact`` and one ``GeometricMapArtifact`` and proves they belong together
before any context stage runs.

The foundation is orchestration, not science: it references the three immutable artifacts and
checks the lineage they already declare, through their capabilities' public readers. It copies no
geometry, trajectory or calibration, and it has no artifact of its own.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NewType

from contextmap.ingestion import (
    FullSequenceSelection,
    SequenceArtifactId,
    SequenceArtifactReader,
    selection_identity,
)
from contextmap.runtime.artifacts import ArtifactRef, artifact_directory, inventory_digest
from contextmap.runtime.catalog import GEOMETRY, SEQUENCE, TRAJECTORY

SpatialFoundationId = NewType("SpatialFoundationId", str)
"""``"sha256:<hex>"`` over the contractual content of the three artifacts."""


class SpatialFoundationError(ValueError):
    """The three artifacts do not form one spatial foundation.

    Attributes:
        problems: Every problem found, each prefixed with the role it concerns.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        """Build the error from every problem found."""
        self.problems = tuple(problems)
        listed = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"the artifacts do not form one spatial foundation:\n{listed}")


@dataclass(frozen=True, kw_only=True)
class SpatialFoundation:
    """One validated sequence, trajectory and map that context evidence is built on.

    Attributes:
        identity: Derived from the contractual content of the three artifacts, never from a
            path, a run number or a clock: the same content anywhere is the same foundation.
        sequence: The canonical ``SequenceArtifact``.
        state_estimation: The ``StateEstimationRunArtifact`` holding the trajectory.
        geometry: The ``GeometricMapArtifact`` built from that sequence and trajectory.
    """

    identity: SpatialFoundationId
    sequence: ArtifactRef
    state_estimation: ArtifactRef
    geometry: ArtifactRef

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form embedded in the records that depend on it."""
        return {
            "identity": self.identity,
            **{role: getattr(self, role).to_document() for role in _ROLES},
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> SpatialFoundation:
        """Rebuild a foundation already validated when its record was written.

        It is not validated again: that would read every artifact, geometry included, each
        time a record naming it is opened. The record's own identity check covers it.

        Args:
            document: A mapping produced by :meth:`to_document`.

        Returns:
            The foundation.

        Raises:
            ValueError: If the identity or a reference is missing or malformed.
        """
        identity = document.get("identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("a spatial foundation document needs a text identity")
        refs: dict[str, ArtifactRef] = {}
        for role in _ROLES:
            value = document.get(role)
            if not isinstance(value, Mapping):
                raise ValueError(f"a spatial foundation document lacks its {role} reference")
            refs[role] = ArtifactRef.from_document(value)
        return cls(identity=SpatialFoundationId(identity), **refs)


_ROLES = ("sequence", "state_estimation", "geometry")


def resolve_spatial_foundation(
    workspace: Path,
    *,
    sequence: ArtifactRef,
    state_estimation: ArtifactRef,
    geometry: ArtifactRef,
) -> SpatialFoundation:
    """Validate three artifacts as one spatial foundation and derive its identity.

    The map must have been built from this sequence and this state-estimation run and
    trajectory, in the trajectory's reference frame, over the whole sequence; each reference must
    name the artifact at its location, and each artifact must pass its own inventory check. The
    map's derived spatial index is not recomputed: it is a view of the geometry, and the
    inventory already proves the contractual content.

    Args:
        workspace: The workspace the references' locations are relative to.
        sequence: The canonical ``SequenceArtifact``.
        state_estimation: The ``StateEstimationRunArtifact``.
        geometry: The ``GeometricMapArtifact``.

    Returns:
        The foundation.

    Raises:
        SpatialFoundationError: With every problem found, if the artifacts do not belong
            together or one of them is not intact.
    """
    roles = (
        ("sequence", sequence, SEQUENCE),
        ("state_estimation", state_estimation, TRAJECTORY),
        ("geometry", geometry, GEOMETRY),
    )
    kinds = [
        f"{role}: {ref.artifact_id!r} is a {ref.contract!r}, not a {contract!r}"
        for role, ref, contract in roles
        if ref.contract != contract
    ]
    if kinds:
        raise SpatialFoundationError(kinds)

    # Importados aqui: importar o pacote runtime não pode carregar outras capabilities além da
    # Ingestion (tests/architecture/test_runtime_boundaries.py).
    from contextmap.geometric_mapping import GeometricMapArtifactReader
    from contextmap.state_estimation import StateEstimationRunReader

    sequence_reader = SequenceArtifactReader(artifact_directory(workspace, sequence))
    trajectory_reader = StateEstimationRunReader(artifact_directory(workspace, state_estimation))
    with GeometricMapArtifactReader(artifact_directory(workspace, geometry)) as geometry_reader:
        map_manifest = geometry_reader.manifest
        map_problems = geometry_reader.verify_integrity(check_index=False)
    sequence_manifest = sequence_reader.manifest
    trajectory_manifest = trajectory_reader.manifest

    problems: list[str] = []
    for role, ref, actual in (
        ("sequence", sequence, str(sequence_manifest.artifact_id)),
        ("state_estimation", state_estimation, str(trajectory_manifest.run_id)),
        ("geometry", geometry, str(map_manifest.run_id)),
    ):
        if ref.artifact_id != actual:
            problems.append(
                f"{role}: the reference names {ref.artifact_id!r}, but the artifact at "
                f"{ref.location!r} is {actual!r}"
            )
    for role, found in (
        ("sequence", sequence_reader.verify_integrity()),
        ("state_estimation", trajectory_reader.verify_integrity()),
        ("geometry", map_problems),
    ):
        problems.extend(f"{role}: {problem}" for problem in found)

    sequence_id = sequence_manifest.artifact_id
    run_id = trajectory_manifest.run_id
    map_id = map_manifest.map_id
    if trajectory_manifest.sequence_artifact_id != sequence_id:
        problems.append(
            f"state_estimation: run {run_id!r} estimated sequence "
            f"{trajectory_manifest.sequence_artifact_id!r}, not {sequence_id!r}"
        )
    if map_manifest.sequence_artifact_id != sequence_id:
        problems.append(
            f"geometry: map {map_id!r} was built from sequence "
            f"{map_manifest.sequence_artifact_id!r}, not {sequence_id!r}"
        )
    if map_manifest.state_estimation_run_id != run_id:
        problems.append(
            f"geometry: map {map_id!r} was built from state estimation run "
            f"{map_manifest.state_estimation_run_id!r}, not {run_id!r}"
        )
    if map_manifest.trajectory_id != trajectory_manifest.trajectory_id:
        problems.append(
            f"geometry: map {map_id!r} was built from trajectory {map_manifest.trajectory_id!r}, "
            f"not {trajectory_manifest.trajectory_id!r}"
        )
    if map_manifest.map_frame != trajectory_manifest.reference_frame:
        problems.append(
            f"geometry: map frame {map_manifest.map_frame!r} is not the trajectory's reference "
            f"frame {trajectory_manifest.reference_frame!r}"
        )
    for role, name, of_sequence, selection in (
        (
            "state_estimation",
            run_id,
            trajectory_manifest.sequence_artifact_id,
            trajectory_manifest.selection_id,
        ),
        ("geometry", map_id, map_manifest.sequence_artifact_id, map_manifest.selection_id),
    ):
        if selection != _whole(of_sequence):
            problems.append(
                f"{role}: {name!r} covers selection {selection!r} of its sequence, not the "
                "whole sequence a foundation is built on"
            )
    if problems:
        raise SpatialFoundationError(problems)

    return SpatialFoundation(
        identity=_identity(
            sequence=inventory_digest(sequence_manifest.file_inventory),
            state_estimation=inventory_digest(trajectory_manifest.file_inventory),
            geometry=inventory_digest(map_manifest.file_inventory),
        ),
        sequence=sequence,
        state_estimation=state_estimation,
        geometry=geometry,
    )


def _whole(sequence_artifact_id: SequenceArtifactId) -> str:
    """Return the selection identity of a whole sequence."""
    return selection_identity(sequence_artifact_id, FullSequenceSelection())


def _identity(*, sequence: str, state_estimation: str, geometry: str) -> SpatialFoundationId:
    """Combine the content digests of the three artifacts into the foundation identity."""
    document = {"geometry": geometry, "sequence": sequence, "state_estimation": state_estimation}
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return SpatialFoundationId(f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}")

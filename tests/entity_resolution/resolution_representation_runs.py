"""Real Point Representation runs on disk, with scripted encoders, for the representation tests."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pointrep_builders import radius_policy
from pointrep_fakes import FakeEncoder
from pointrep_geometry import MAP_ID, LinearScanSource, line_of_points

from contextmap.geometric_mapping import GeometryReference, geometry_id_for, geometry_index_of
from contextmap.point_representation import (
    EncodedVector,
    PointEncoder,
    PointRepresentation,
    PointRepresentationRunId,
    PointRepresentationRunReader,
    PointRepresentationRunWriter,
    PreparedSupport,
    RepresentationService,
    representation_space_fingerprint,
)
from contextmap.semantic_fusion import PointRepresentationRef

Script = Mapping[int, tuple[Sequence[float], Sequence[int]]]


class ScriptedEncoder(FakeEncoder):
    """Returns the vector scripted for the center of each support, with its undefined components."""

    def __init__(
        self,
        script: Script,
        *,
        family: str = "scripted",
        checkpoint: str | None = None,
        dimension: int = 4,
    ) -> None:
        super().__init__(radius_policy(0.6), family=family, checkpoint=checkpoint)
        self._space = dataclasses.replace(
            self._space,
            dimension=dimension,
            feature_names=tuple(f"component_{index}" for index in range(dimension)),
        )
        self._script = script

    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        center = prepared.support.center
        index = geometry_index_of(map_id=center.map_id, geometry_id=center.geometry_id)
        values, undefined = self._script[index]
        return EncodedVector(
            values=tuple(float(value) for value in values), undefined_components=tuple(undefined)
        )


@dataclass(frozen=True)
class RealRun:
    """A finalized run, opened, with the reference an entity would hold for each center."""

    reader: PointRepresentationRunReader
    run_id: PointRepresentationRunId
    space_id: str
    refs: dict[int, PointRepresentationRef]
    representations: dict[int, PointRepresentation]


def geometry_ref(index: int) -> GeometryReference:
    return GeometryReference(map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=index))


def write_run(
    workspace: Path,
    script: Script,
    *,
    run: str = "representation-run-0001",
    family: str = "scripted",
    checkpoint: str | None = None,
    dimension: int = 4,
    encoder: PointEncoder | None = None,
) -> RealRun:
    """Encode the scripted centers of a line of points and persist them as a real run."""
    run_id = PointRepresentationRunId(run)
    source = LinearScanSource(line_of_points(40))
    chosen = encoder or ScriptedEncoder(
        script, family=family, checkpoint=checkpoint, dimension=dimension
    )
    service = RepresentationService(source, chosen, run_id=run_id, code_version="test")
    writer = PointRepresentationRunWriter(
        workspace_root=workspace,
        sequence_name="corridor-02",
        run_id=run_id,
        run_index=1,
        selection_label="centers",
        backend_label="scripted",
        geometric_map=source.geometric_map,
        space=chosen.representation_space(),
        encoder_identity=chosen.encoder_identity(),
        code_version="test",
    )
    for outcome in service.represent([geometry_ref(index) for index in sorted(script)]):
        writer.add(outcome)
    writer.finalize(metrics=service.metrics, backend_diagnostics=None)
    run_dir = (
        workspace / "runs" / "point-representation" / "corridor-02" / "run-0001__centers__scripted"
    )
    reader = PointRepresentationRunReader(run_dir)
    representations = {
        geometry_index_of(
            map_id=item.geometry_reference.map_id, geometry_id=item.geometry_reference.geometry_id
        ): item
        for item in reader.iter_representations()
    }
    space_id = representation_space_fingerprint(chosen.representation_space())
    refs = {
        index: PointRepresentationRef(
            representation_id=item.representation_id,
            run_id=run_id,
            representation_space_id=item.representation_space_id,
            geometry_reference=item.geometry_reference,
        )
        for index, item in representations.items()
    }
    return RealRun(reader, run_id, space_id, refs, representations)

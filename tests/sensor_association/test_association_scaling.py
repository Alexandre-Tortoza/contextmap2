"""Scaling and equivalence gates for candidate culling and bounded-memory execution.

Small fixtures hid the failure these gates exist for: on the real 26.63 M-point map the
association projected the whole map into every frame and kept every frame's arrays, so cost
grew as ``frames x global_map_points``. Optimizing that can silently drop valid geometry
while still producing a plausible map, so resource behaviour and scientific equivalence are
checked together, and always by **persistent geometry identity**, never by candidate row.

The arms these gates compare are the ones the code can still express:

``D`` full-map candidates + streaming persistence
``C`` culled candidates + streaming persistence

Batch retention no longer exists after #563, so the arm that had it is not resurrected here:
``D`` isolates streaming against the versioned experiment report's measurement of the old
batch arm, and ``D -> C`` isolates culling. See
``experiments/sensor-association-scaling-20260925/README.md``.

CI keeps to counters, retention invariants and broad envelopes; wall-clock numbers belong to
the experiment report, on real hardware.
"""

from __future__ import annotations

import dataclasses
import gc
from pathlib import Path

import pytest
from projection_builders import (
    ArrayGeometrySource,
    map_point_for_camera_point,
    map_point_for_pixel,
)
from run_builders import (
    ENHANCED,
    NATIVE,
    CollectingSink,
    frame_id,
    frame_input,
    make_request,
)

from contextmap.geometric_mapping import GeometryReference, geometry_id_for
from contextmap.sensor_association import (
    CandidateGeometryPolicy,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
)
from contextmap.sensor_association.service import (
    FrameAssociation,
    SensorAssociationRequest,
)
from contextmap.sensor_association.visibility import VisibilityResolution
from contextmap.shared import Vector3
from contextmap.state_estimation import LookupPolicy

SERVICE = SensorAssociationService()
UNBOUNDED = CandidateGeometryPolicy(max_range_m=None)
FRAME_TIMES = (0, 25_000_000, 50_000_000, 75_000_000, 100_000_000)

# A cena local: o que a câmera realmente vê, a poucos metros.
NEARBY: list[Vector3] = [
    map_point_for_pixel(u, v, z)
    for u, v, z in ((100, 100, 3.0), (150, 100, 3.0), (150, 100, 8.0), (240, 100, 3.0))
]


def _leading_filler(count: int) -> list[Vector3]:
    """Behind-camera geometry placed **before** the scene, so retained global indices are
    neither contiguous nor zero-based.

    Without it every culled run keeps indices ``0..n``, exactly where a row and a global index
    coincide -- the coincidence #562 removed, and the one a test must not rely on.
    """
    return [map_point_for_camera_point((0.0, 0.0, -(50.0 + index * 0.5))) for index in range(count)]


def _far_filler(count: int) -> list[Vector3]:
    """Map bulk far *behind* the camera: real geometry no frame can ever associate.

    Behind-camera geometry is what makes the equivalence arms comparable exactly: it can
    never be in the prepared image, so it is neither region support nor an occluder, and
    excluding it can only change the candidate-stage counts.
    """
    return [
        map_point_for_camera_point((0.0, 0.0, -(200.0 + index * 0.5))) for index in range(count)
    ]


def _request(
    *,
    filler: int = 0,
    leading: int = 0,
    candidates: CandidateGeometryPolicy = UNBOUNDED,
    frames: int = 1,
    channels: tuple[object, ...] = (),
) -> SensorAssociationRequest:
    """A request whose local scene is fixed and whose map grows with ``filler``.

    ``leading`` puts unassociable geometry *before* the scene, which offsets every retained
    global index away from its row.
    """
    base = make_request(channels=list(channels), candidates=candidates)  # type: ignore[arg-type]
    source = ArrayGeometrySource(
        [*_leading_filler(leading), *NEARBY, *_far_filler(filler)], calibration=base.calibration
    )
    return dataclasses.replace(
        base,
        geometry=source,
        pose_policy=LookupPolicy.interpolated(),
        frames=tuple(
            frame_input(index, time_ns=FRAME_TIMES[index], channels=list(channels))  # type: ignore[arg-type]
            for index in range(frames)
        ),
    )


def _write(root: Path, request: SensorAssociationRequest, *, index: int = 1) -> Path:
    run_dir = root / f"run-{index:04d}"
    writer = SensorAssociationRunWriter(
        output_dir=run_dir,
        sequence_name="fixture",
        run_id=SensorAssociationRunId(f"arm-{index:04d}"),
        run_index=index,
    )
    with writer.transaction() as run:
        run.finalize(SERVICE.run(request, sink=run))
    return run_dir


def _without_fingerprint(observation: object) -> object:
    """The observation with its configuration fingerprint blanked, for cross-arm comparison."""
    provenance = dataclasses.replace(
        observation.provenance,  # type: ignore[attr-defined]
        configuration_fingerprint=None,
    )
    return dataclasses.replace(observation, provenance=provenance)  # type: ignore[type-var]


def _support_by_observation(run_dir: Path) -> dict[str, tuple[str, ...]]:
    """Every observation's geometry support, by persistent identity, never by row."""
    reader = SensorAssociationRunReader(run_dir)
    return {
        str(observation.spatial_observation_id): tuple(
            str(reference.geometry_id)
            for reference in reader.geometry_support(str(observation.spatial_observation_id))
        )
        for observation in reader.observations()
    }


# --- Complexity gate: per-frame work follows the candidates, not the map --------------------


@pytest.mark.parametrize("filler", [0, 40, 400, 4000])
def test_growing_the_map_at_fixed_local_density_does_not_grow_per_frame_work(
    filler: int,
) -> None:
    bounded = _request(filler=filler, candidates=CandidateGeometryPolicy(max_range_m=20.0))

    _, frames = _run_collecting(bounded)

    projection = frames[0].resolution.frame
    # O mapa cresce 1x -> 1000x; a população avaliada não.
    assert projection.map_point_count == len(NEARBY) + filler
    assert projection.candidate_count == len(NEARBY)
    # E as alocações por frame são do tamanho dos candidatos, não do mapa.
    assert projection.prepared_pixels.shape == (len(NEARBY), 2)
    assert projection.camera_depth_m.shape == (len(NEARBY),)
    assert frames[0].resolution.support_depth_m.shape == (len(NEARBY),)


def test_the_full_map_arm_grows_with_the_map_which_is_the_failure_being_fixed() -> None:
    small = _request(filler=0)
    large = _request(filler=4000)

    _, small_frames = _run_collecting(small)
    _, large_frames = _run_collecting(large)

    assert small_frames[0].resolution.frame.candidate_count == len(NEARBY)
    assert large_frames[0].resolution.frame.candidate_count == len(NEARBY) + 4000


def test_the_candidate_fraction_falls_as_the_map_grows_around_a_fixed_scene() -> None:
    fractions = []
    for filler in (40, 400, 4000):
        _, frames = _run_collecting(
            _request(filler=filler, candidates=CandidateGeometryPolicy(max_range_m=20.0))
        )
        fractions.append(frames[0].resolution.frame.candidates.candidate_fraction)

    assert fractions == sorted(fractions, reverse=True)
    assert fractions[-1] < fractions[0] / 10


# --- Retention gate: memory does not follow the frame count --------------------------------


def _live_resolutions() -> int:
    gc.collect()
    return sum(1 for obj in gc.get_objects() if isinstance(obj, VisibilityResolution))


def _run_collecting(
    request: SensorAssociationRequest,
) -> tuple[object, tuple[FrameAssociation, ...]]:
    sink = CollectingSink()
    outcome = SERVICE.run(request, sink=sink)
    return outcome, tuple(sink.frames)


class _BoundedWitness:
    """Records how many frame resolutions are alive as each frame is accepted."""

    def __init__(self, downstream: object | None = None) -> None:
        self.live: list[int] = []
        self._downstream = downstream

    def accept(self, frame: FrameAssociation) -> None:
        if self._downstream is not None:
            self._downstream.accept(frame)  # type: ignore[attr-defined]
        self.live.append(_live_resolutions())


@pytest.mark.parametrize("frames", [1, 3, 5])
def test_exactly_one_frame_resolution_is_alive_however_many_frames_the_run_has(
    frames: int,
) -> None:
    witness = _BoundedWitness()

    SERVICE.run(
        _request(frames=frames, candidates=CandidateGeometryPolicy(max_range_m=20.0)), sink=witness
    )

    # Enquanto o frame K é aceito, ele é o único vivo: nada de 0..K-1 sobrevive.
    assert witness.live == [1] * frames
    assert _live_resolutions() == 0


@pytest.mark.parametrize("frames", [1, 3, 5])
def test_persisting_a_run_keeps_the_same_bound(tmp_path: Path, frames: int) -> None:
    writer = SensorAssociationRunWriter(
        output_dir=tmp_path / "run",
        sequence_name="fixture",
        run_id=SensorAssociationRunId("bounded"),
        run_index=1,
    )
    request = _request(frames=frames, candidates=CandidateGeometryPolicy(max_range_m=20.0))

    with writer.transaction() as run:
        witness = _BoundedWitness(run)
        outcome = SERVICE.run(request, sink=witness)
        run.finalize(outcome)

    assert witness.live == [1] * frames
    assert outcome.frame_count == frames
    assert _live_resolutions() == 0


def test_the_outcome_is_the_same_size_whatever_the_frame_count() -> None:
    one = SERVICE.run(_request(frames=1), sink=CollectingSink())
    five = SERVICE.run(_request(frames=5), sink=CollectingSink())

    # O resultado do run é identidade e agregados: dois contadores, não N frames.
    assert (one.frame_count, five.frame_count) == (1, 5)
    assert not hasattr(one, "frames")
    assert len(dataclasses.fields(one)) == len(dataclasses.fields(five))


# --- Equivalence gate: identical where the policy evaluates the same population -------------


def test_a_range_that_keeps_everything_produces_the_same_artifact_content(
    tmp_path: Path,
) -> None:
    full = _write(tmp_path, _request(filler=20, channels=(NATIVE, ENHANCED)), index=1)
    wide = _write(
        tmp_path,
        _request(
            filler=20,
            candidates=CandidateGeometryPolicy(max_range_m=1000.0),
            channels=(NATIVE, ENHANCED),
        ),
        index=2,
    )

    assert _support_by_observation(wide) == _support_by_observation(full)
    full_reader, wide_reader = SensorAssociationRunReader(full), SensorAssociationRunReader(wide)
    assert [str(o.spatial_observation_id) for o in wide_reader.observations()] == [
        str(o.spatial_observation_id) for o in full_reader.observations()
    ]
    for observation in full_reader.observations():
        identity = str(observation.spatial_observation_id)
        assert wide_reader.quality(identity) == full_reader.quality(identity)
        # A observação é idêntica a menos do fingerprint de configuração, que **deve**
        # diferir: a política de candidatos faz parte dele.
        assert _without_fingerprint(wide_reader.observation(identity)) == _without_fingerprint(
            full_reader.observation(identity)
        )


def test_the_same_run_lineage_and_policies_survive_culling(tmp_path: Path) -> None:
    full = SensorAssociationRunReader(_write(tmp_path, _request(filler=20), index=1)).manifest
    culled = SensorAssociationRunReader(
        _write(
            tmp_path,
            _request(filler=20, candidates=CandidateGeometryPolicy(max_range_m=20.0)),
            index=2,
        )
    ).manifest

    for field in (
        "geometric_map_id",
        "trajectory_id",
        "calibration_identity",
        "perception_run_ids",
        "sequence_artifact_id",
        "selection_id",
        "membership_policy_id",
    ):
        assert getattr(culled, field) == getattr(full, field), field
    # A política de candidatos é a única diferença declarada, e ela está no manifest.
    assert culled.candidate_policy["max_range_m"] == 20.0
    assert full.candidate_policy["max_range_m"] is None
    assert culled.candidate_policy["fingerprint"] != full.candidate_policy["fingerprint"]


def test_the_regions_supporting_each_persistent_point_are_unchanged_by_culling(
    tmp_path: Path,
) -> None:
    full = SensorAssociationRunReader(_write(tmp_path, _request(filler=20), index=1))
    culled = SensorAssociationRunReader(
        _write(
            tmp_path,
            _request(filler=20, candidates=CandidateGeometryPolicy(max_range_m=20.0)),
            index=2,
        )
    )
    references = {
        reference
        for observation in full.observations()
        for reference in full.geometry_support(str(observation.spatial_observation_id))
    }

    assert references
    for reference in references:
        assert culled.regions_of(frame_id(0), reference) == full.regions_of(frame_id(0), reference)


def test_the_dense_sample_source_geometry_is_unchanged_by_culling(tmp_path: Path) -> None:
    full = SensorAssociationRunReader(
        _write(tmp_path, _request(filler=20, channels=(NATIVE,)), index=1)
    )
    culled = SensorAssociationRunReader(
        _write(
            tmp_path,
            _request(
                filler=20, candidates=CandidateGeometryPolicy(max_range_m=20.0), channels=(NATIVE,)
            ),
            index=2,
        )
    )

    a = full.dense_association(frame_id(0), "dino-native")
    b = culled.dense_association(frame_id(0), "dino-native")
    # Os índices são identidades globais, então comparam-se diretamente entre braços.
    assert list(b.eligible_indices) == list(a.eligible_indices)
    assert list(b.sampled) == list(a.sampled)
    assert list(b.cell_rows) == list(a.cell_rows)
    assert list(b.weights) == pytest.approx(list(a.weights))


def test_the_only_declared_difference_is_the_evaluated_population(tmp_path: Path) -> None:
    full = SensorAssociationRunReader(_write(tmp_path, _request(filler=20), index=1))
    culled = SensorAssociationRunReader(
        _write(
            tmp_path,
            _request(filler=20, candidates=CandidateGeometryPolicy(max_range_m=20.0)),
            index=2,
        )
    )

    (full_record,) = full.read_records("outputs/projection-records.jsonl")
    (culled_record,) = culled.read_records("outputs/projection-records.jsonl")

    assert full_record["candidates"]["candidate_count"] == len(NEARBY) + 20
    assert culled_record["candidates"]["candidate_count"] == len(NEARBY)
    assert (
        culled_record["candidates"]["map_point_count"]
        == full_record["candidates"]["map_point_count"]
    )
    # Os pontos que saíram eram todos atrás/fora: nenhum deles era suporte de região.
    assert culled_record["stage_counts"]["in_support"] == full_record["stage_counts"]["in_support"]
    assert (
        full_record["stage_counts"]["behind_camera"]
        > (culled_record["stage_counts"]["behind_camera"])
    )


# --- A row is not an index, and the reopened artifact must agree -----------------------------


def test_the_retained_indices_are_offset_and_gapped_not_a_prefix_of_the_map() -> None:
    """The fixture is only meaningful if row != global index; assert that it is."""
    _, frames = _run_collecting(
        _request(leading=7, filler=5, candidates=CandidateGeometryPolicy(max_range_m=20.0))
    )

    projection = frames[0].resolution.frame
    kept = [int(index) for index in projection.global_indices]
    assert kept == [7, 8, 9, 10]
    assert kept != list(range(len(kept)))
    assert projection.map_reference(0) != projection.map_reference(1)


def test_a_reopened_run_resolves_support_and_regions_through_the_offset_indices(
    tmp_path: Path,
) -> None:
    """`regions_of()` binary-searches the persisted support, so its order must be the map's.

    With the retained geometry offset away from row zero, a run that persisted rows instead of
    global indices would still produce well-formed references and a working reverse lookup --
    for the *wrong* geometry. Comparing the reopened artifact against the full-map arm by
    identity is what catches that.
    """
    full = SensorAssociationRunReader(_write(tmp_path, _request(leading=7, filler=5), index=1))
    culled = SensorAssociationRunReader(
        _write(
            tmp_path,
            _request(leading=7, filler=5, candidates=CandidateGeometryPolicy(max_range_m=20.0)),
            index=2,
        )
    )

    supported = {
        str(observation.spatial_observation_id): full.geometry_support(
            str(observation.spatial_observation_id)
        )
        for observation in full.observations()
    }
    assert any(supported.values())
    for identity, references in supported.items():
        assert culled.geometry_support(identity) == references
        # A geometria retida começa no índice 7: nenhuma referência é a do índice zero.
        assert all(not str(r.geometry_id).endswith("-000000000") for r in references)
        # E a ordem persistida é crescente, que é o que a busca binária pressupõe.
        assert list(references) == sorted(references, key=lambda r: str(r.geometry_id))

    for references in supported.values():
        for reference in references:
            assert culled.regions_of(frame_id(0), reference) == full.regions_of(
                frame_id(0), reference
            )
            assert culled.regions_of(frame_id(0), reference) != ()


def test_a_reopened_run_reports_no_region_for_geometry_the_frame_did_not_associate(
    tmp_path: Path,
) -> None:
    culled = SensorAssociationRunReader(
        _write(
            tmp_path,
            _request(leading=7, filler=5, candidates=CandidateGeometryPolicy(max_range_m=20.0)),
            index=1,
        )
    )
    map_id = culled.manifest.geometric_map_id

    # Índice 0 é filler atrás da câmera: existe no mapa, nunca foi suporte de região.
    behind = GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=0))

    assert culled.regions_of(frame_id(0), behind) == ()

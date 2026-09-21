"""Point-representation comparison: compatible spaces only, undefined components never read."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from pointrep_builders import radius_policy
from pointrep_geometry import MAP_ID
from resolution_entity_builders import entity_over, scene_source
from resolution_representation_runs import RealRun, geometry_ref, write_run

from contextmap.entity_resolution import (
    REPRESENTATION_AGGREGATION_ID,
    REPRESENTATION_COMPARISON_POLICY_ID,
    EvidenceStatus,
    LoadedRepresentation,
    PointRepresentationEvidence,
    RepresentationComparator,
    RepresentationComparisonPolicy,
    RepresentationRef,
    RunReaderRepresentationSource,
    UnavailableReason,
)
from contextmap.entity_resolution._codec import from_record, to_record
from contextmap.point_representation import (
    PointRepresentationRunId,
    RepresentationSpaceMismatchError,
)
from contextmap.point_representation.backends.geometric_descriptor import (
    GeometricDescriptorEncoder,
)
from contextmap.semantic_fusion import PointRepresentationRef
from contextmap.semantic_mapping import Entity

LINE = scene_source({index: (index * 0.25, 0.0, 0.0) for index in range(40)}, map_id=MAP_ID)


def policy_for(run: RealRun, **kwargs: Any) -> RepresentationComparisonPolicy:
    values: dict[str, Any] = {
        "representation_space_id": run.space_id,
        "min_supporting_similarity": 0.95,
        "max_conflicting_similarity": 0.2,
        "min_defined_component_fraction": 0.5,
    }
    values.update(kwargs)
    return RepresentationComparisonPolicy(**values)


def entity_of(entity_id: str, first: int, refs: Sequence[PointRepresentationRef]) -> Entity:
    """An entity whose support is ten consecutive points and which holds the given references."""
    ordered = tuple(sorted(refs, key=lambda ref: (ref.run_id, ref.representation_id)))
    return entity_over(entity_id, LINE, range(first, first + 10), representations=ordered)


class CountingSource:
    """Wraps a source, counts loads and optionally tampers with what it returns."""

    def __init__(
        self,
        inner: RunReaderRepresentationSource,
        tamper: Callable[[LoadedRepresentation], LoadedRepresentation] | None = None,
    ) -> None:
        self._inner = inner
        self._tamper = tamper
        self.loads: list[str] = []

    def load(self, ref: RepresentationRef) -> LoadedRepresentation:
        self.loads.append(str(ref.representation_id))
        loaded = self._inner.load(ref)
        return self._tamper(loaded) if self._tamper else loaded


def compare(
    run: RealRun,
    first: Entity,
    second: Entity,
    policy: RepresentationComparisonPolicy | None = None,
    *,
    source: CountingSource | None = None,
) -> PointRepresentationEvidence:
    used = source or CountingSource(RunReaderRepresentationSource({run.run_id: run.reader}))
    return RepresentationComparator(used, policy or policy_for(run)).compare(first, second)


def two_entities(
    run: RealRun, centers: tuple[Sequence[int], Sequence[int]] = ((3,), (23,))
) -> tuple[Entity, Entity]:
    return (
        entity_of("a", 0, [run.refs[index] for index in centers[0]]),
        entity_of("b", 20, [run.refs[index] for index in centers[1]]),
    )


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.hypot(*left) * math.hypot(*right))


# --- compatible spaces produce evidence with provenance -----------------------------------------


def test_compatible_representations_produce_evidence_with_full_provenance(tmp_path: Path) -> None:
    run = write_run(tmp_path, {3: ((1, 2, 3, 4), ()), 23: ((1, 2, 3, 5), ())})
    first, second = two_entities(run)

    evidence = compare(run, first, second)

    measurement = evidence.measurement
    assert measurement is not None
    assert measurement.representation_space_id == run.space_id
    assert measurement.metric == "cosine-similarity"
    assert measurement.aggregation_id == REPRESENTATION_AGGREGATION_ID
    assert measurement.similarity == pytest.approx(cosine((1, 2, 3, 4), (1, 2, 3, 5)), abs=1e-6)
    assert (measurement.dimension, measurement.compared_components) == (4, 4)
    assert measurement.representations_a[0].geometry_reference == geometry_ref(3)
    assert measurement.representations_a[0].representation_space_id == run.space_id
    assert evidence.status is EvidenceStatus.SUPPORTING
    assert evidence.policy.policy_id == REPRESENTATION_COMPARISON_POLICY_ID
    assert evidence.policy.configuration_fingerprint == policy_for(run).fingerprint()


def test_the_evidence_is_reproducible_and_symmetric(tmp_path: Path) -> None:
    run = write_run(tmp_path, {3: ((1, 2, 3, 4), ()), 23: ((4, 3, 2, 1), ())})
    first, second = two_entities(run)

    assert compare(run, first, second) == compare(run, first, second)
    assert compare(run, first, second) == compare(run, second, first)


def test_similarity_is_read_through_explicit_thresholds(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        {
            3: ((1, 0, 0, 0), ()),
            23: ((1, 0.1, 0, 0), ()),
            5: ((0.7, 0.7, 0, 0), ()),
            25: ((0, 1, 0, 0), ()),
        },
    )

    def status(
        a: int, b: int, policy: RepresentationComparisonPolicy | None = None
    ) -> EvidenceStatus:
        first, second = two_entities(run, ((a,), (b,)))
        return compare(run, first, second, policy).status

    assert status(3, 23) is EvidenceStatus.SUPPORTING
    assert status(5, 23) is EvidenceStatus.NEUTRAL
    assert status(3, 25) is EvidenceStatus.CONFLICTING
    assert status(3, 25, policy_for(run, max_conflicting_similarity=None)) is EvidenceStatus.NEUTRAL


# --- optionality and incompatible spaces --------------------------------------------------------


def test_an_entity_without_representations_is_missing_evidence_not_a_zero(tmp_path: Path) -> None:
    run = write_run(tmp_path, {3: ((1, 2, 3, 4), ())})
    first = entity_of("a", 0, [run.refs[3]])
    bare = entity_of("b", 20, [])
    source = CountingSource(RunReaderRepresentationSource({run.run_id: run.reader}))

    evidence = compare(run, first, bare, source=source)

    assert evidence.status is EvidenceStatus.UNAVAILABLE
    assert evidence.measurement is None
    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.MISSING_EVIDENCE
    assert source.loads == []


def test_representations_of_another_space_are_unavailable_even_with_the_same_dimension(
    tmp_path: Path,
) -> None:
    run = write_run(tmp_path / "one", {3: ((1, 2, 3, 4), ())})
    other = write_run(
        tmp_path / "two",
        {23: ((1, 2, 3, 4), ())},
        run="representation-run-0002",
        family="ptv3-fake",
        checkpoint="checkpoint-a",
    )
    first = entity_of("a", 0, [run.refs[3]])
    second = entity_of("b", 20, [other.refs[23]])
    source = CountingSource(
        RunReaderRepresentationSource({run.run_id: run.reader, other.run_id: other.reader})
    )

    evidence = compare(run, first, second, source=source)

    assert run.space_id != other.space_id
    assert evidence.status is EvidenceStatus.UNAVAILABLE
    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.INCOMPATIBLE_DOMAIN
    assert (
        run.space_id in evidence.unavailable.detail
        and other.space_id in evidence.unavailable.detail
    )
    assert source.loads == []


def test_a_store_that_contradicts_its_reference_fails_loudly(tmp_path: Path) -> None:
    run = write_run(tmp_path / "one", {3: ((1, 2, 3, 4), ())})
    other = write_run(
        tmp_path / "two",
        {23: ((1, 2, 3, 4), ())},
        run="representation-run-0002",
        family="ptv3-fake",
        checkpoint="checkpoint-a",
    )
    lying = PointRepresentationRef(
        representation_id=other.refs[23].representation_id,
        run_id=other.run_id,
        representation_space_id=run.space_id,
        geometry_reference=other.refs[23].geometry_reference,
    )
    source = CountingSource(
        RunReaderRepresentationSource({run.run_id: run.reader, other.run_id: other.reader})
    )

    with pytest.raises(RepresentationSpaceMismatchError):
        compare(run, entity_of("a", 0, [run.refs[3]]), entity_of("b", 20, [lying]), source=source)


def test_learned_and_deterministic_spaces_use_the_same_boundary(tmp_path: Path) -> None:
    learned = write_run(
        tmp_path / "learned",
        {3: ((1, 2, 3, 4), ()), 23: ((1, 2, 3, 4), ())},
        family="ptv3-fake",
        checkpoint="checkpoint-a",
    )
    descriptor = write_run(
        tmp_path / "descriptor",
        {3: ((0,), ()), 23: ((0,), ())},
        encoder=GeometricDescriptorEncoder(radius_policy(0.6)),
    )

    for run in (learned, descriptor):
        first, second = two_entities(run)
        evidence = compare(run, first, second)
        assert evidence.measurement is not None
        assert evidence.status is EvidenceStatus.SUPPORTING
        assert evidence.measurement.similarity == pytest.approx(1.0, abs=1e-9)
    # O descritor determinístico não define todos os componentes de um suporte em linha reta.
    descriptor_measurement = compare(descriptor, *two_entities(descriptor)).measurement
    assert descriptor_measurement is not None
    assert descriptor_measurement.compared_components < descriptor_measurement.dimension


# --- undefined components are never interpreted -------------------------------------------------


def test_an_undefined_component_is_left_out_instead_of_read_as_zero(tmp_path: Path) -> None:
    run = write_run(tmp_path, {3: ((1, 1, 0, 99), (3,)), 23: ((1, 1, 0, 1), ())})
    first, second = two_entities(run)

    measurement = compare(run, first, second).measurement

    assert measurement is not None
    assert (measurement.dimension, measurement.compared_components) == (4, 3)
    assert measurement.similarity == pytest.approx(1.0, abs=1e-6)
    # Lida como zero, a mesma comparação daria cosseno 0,816.
    assert cosine((1, 1, 0, 0), (1, 1, 0, 1)) == pytest.approx(0.8165, abs=1e-4)


def test_too_few_commonly_defined_components_make_the_channel_unavailable(tmp_path: Path) -> None:
    run = write_run(tmp_path, {3: ((1, 1, 1, 1), (1, 2, 3)), 23: ((1, 1, 1, 1), ())})
    first, second = two_entities(run)

    evidence = compare(run, first, second)

    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.INSUFFICIENT_EVIDENCE
    assert "1 of 4" in evidence.unavailable.detail


def test_defined_components_without_a_direction_are_insufficient_evidence(tmp_path: Path) -> None:
    run = write_run(tmp_path, {3: ((0, 0, 0, 5), (3,)), 23: ((1, 1, 1, 1), ())})
    first, second = two_entities(run)

    evidence = compare(run, first, second)

    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.INSUFFICIENT_EVIDENCE


def test_a_support_prototype_averages_each_component_over_the_supports_that_define_it(
    tmp_path: Path,
) -> None:
    run = write_run(
        tmp_path,
        {
            3: ((1, 0, 0, 7), (3,)),
            5: ((3, 0, 0, 9), ()),
            23: ((2, 0, 0, 9), ()),
        },
    )
    first, second = two_entities(run, ((3, 5), (23,)))

    measurement = compare(run, first, second).measurement

    assert measurement is not None
    # Média (1 + 3) / 2 = 2 no componente 0; o componente 3 só existe no segundo suporte (9).
    assert measurement.similarity == pytest.approx(1.0, abs=1e-6)
    assert measurement.compared_components == 4
    pairs = [cosine((1, 0, 0), (2, 0, 0)), cosine((3, 0, 0, 9), (2, 0, 0, 9))]
    assert measurement.pair_similarity_min == pytest.approx(min(pairs), abs=1e-6)
    assert measurement.pair_similarity_max == pytest.approx(max(pairs), abs=1e-6)
    assert [item.representation_id for item in measurement.representations_a] == sorted(
        str(run.refs[index].representation_id) for index in (3, 5)
    )


# --- corrupt evidence and reuse -----------------------------------------------------------------


def test_vectors_of_different_lengths_are_refused(tmp_path: Path) -> None:
    run = write_run(tmp_path, {3: ((1, 2, 3, 4), ()), 23: ((1, 2, 3, 4), ())})
    first, second = two_entities(run)
    truncated = CountingSource(
        RunReaderRepresentationSource({run.run_id: run.reader}),
        tamper=lambda loaded: LoadedRepresentation(
            representation=loaded.representation,
            values=loaded.values[:3]
            if loaded.representation.representation_id == run.refs[3].representation_id
            else loaded.values,
        ),
    )

    with pytest.raises(ValueError, match="dimension"):
        compare(run, first, second, source=truncated)


def test_the_representations_of_one_entity_must_share_a_dimension(tmp_path: Path) -> None:
    run = write_run(
        tmp_path, {3: ((1, 2, 3, 4), ()), 5: ((1, 2, 3, 4), ()), 23: ((1, 2, 3, 4), ())}
    )
    first, second = two_entities(run, ((3, 5), (23,)))
    ragged = CountingSource(
        RunReaderRepresentationSource({run.run_id: run.reader}),
        tamper=lambda loaded: LoadedRepresentation(
            representation=loaded.representation,
            values=loaded.values[:2]
            if loaded.representation.representation_id == run.refs[5].representation_id
            else loaded.values,
        ),
    )

    with pytest.raises(ValueError, match="entity 'a'"):
        compare(run, first, second, source=ragged)


def test_a_component_that_is_not_finite_is_corrupt_evidence(tmp_path: Path) -> None:
    run = write_run(tmp_path, {3: ((1, 2, 3, 4), ()), 23: ((1, 2, 3, 4), ())})
    first, second = two_entities(run)
    corrupt = CountingSource(
        RunReaderRepresentationSource({run.run_id: run.reader}),
        tamper=lambda loaded: LoadedRepresentation(
            representation=loaded.representation, values=(math.nan, *loaded.values[1:])
        ),
    )

    with pytest.raises(ValueError, match="finite"):
        compare(run, first, second, source=corrupt)


def test_an_entity_is_loaded_once_however_many_pairs_it_takes_part_in(tmp_path: Path) -> None:
    run = write_run(
        tmp_path, {3: ((1, 2, 3, 4), ()), 23: ((1, 2, 3, 4), ()), 25: ((1, 2, 3, 4), ())}
    )
    first = entity_of("a", 0, [run.refs[3]])
    second = entity_of("b", 20, [run.refs[23]])
    third = entity_of("c", 22, [run.refs[25]])
    source = CountingSource(RunReaderRepresentationSource({run.run_id: run.reader}))
    comparator = RepresentationComparator(source, policy_for(run))

    comparator.compare(first, second)
    comparator.compare(first, third)

    assert sorted(source.loads) == sorted(
        str(run.refs[index].representation_id) for index in (3, 23, 25)
    )


def test_the_run_reader_source_refuses_an_unknown_run(tmp_path: Path) -> None:
    run = write_run(tmp_path, {3: ((1, 2, 3, 4), ())})
    source = RunReaderRepresentationSource({PointRepresentationRunId("another-run"): run.reader})

    ref = run.refs[3]
    reference = RepresentationRef(
        run_id=ref.run_id,
        representation_id=ref.representation_id,
        representation_space_id=ref.representation_space_id,
        geometry_reference=ref.geometry_reference,
    )

    with pytest.raises(ValueError, match="representation run"):
        source.load(reference)


# --- policy -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"representation_space_id": " "},
        {"min_supporting_similarity": 1.5},
        {"min_supporting_similarity": math.nan},
        {"max_conflicting_similarity": 0.99},
        {"max_conflicting_similarity": -1.5},
        {"min_defined_component_fraction": 0.0},
        {"min_defined_component_fraction": 1.2},
    ],
)
def test_the_policy_refuses_impossible_thresholds(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "representation_space_id": "sha256:space",
        "min_supporting_similarity": 0.95,
        "max_conflicting_similarity": 0.2,
        "min_defined_component_fraction": 0.5,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        RepresentationComparisonPolicy(**values)


def test_the_fingerprint_changes_with_the_space_and_every_threshold() -> None:
    def make(**kwargs: Any) -> RepresentationComparisonPolicy:
        values: dict[str, Any] = {
            "representation_space_id": "sha256:space",
            "min_supporting_similarity": 0.95,
            "max_conflicting_similarity": 0.2,
            "min_defined_component_fraction": 0.5,
        }
        values.update(kwargs)
        return RepresentationComparisonPolicy(**values)

    variants = [
        make(representation_space_id="sha256:other"),
        make(min_supporting_similarity=0.9),
        make(max_conflicting_similarity=None),
        make(min_defined_component_fraction=0.75),
    ]

    assert len({make().fingerprint(), *(item.fingerprint() for item in variants)}) == 5


def test_the_evidence_survives_a_json_round_trip(tmp_path: Path) -> None:
    run = write_run(tmp_path, {3: ((1, 2, 3, 4), ()), 23: ((1, 2, 3, 5), ())})
    first, second = two_entities(run)
    evidence = compare(run, first, second)

    record = json.loads(json.dumps(to_record(evidence)))

    assert from_record(PointRepresentationEvidence, record) == evidence

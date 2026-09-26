"""Appearance comparison: compatible spaces only, one vote per physical observation."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from resolution_entity_builders import entity_at
from resolution_feature_fakes import (
    RUN,
    SPACE_A,
    SPACE_B,
    InMemoryFeatureSource,
    feature_ref,
    frame,
    visual_feature,
)

from contextmap.entity_resolution import (
    APPEARANCE_AGGREGATION_ID,
    APPEARANCE_COMPARISON_POLICY_ID,
    AppearanceComparator,
    AppearanceComparisonPolicy,
    AppearanceEvidence,
    EvidenceStatus,
    FeatureStoreVectorSource,
    UnavailableReason,
)
from contextmap.entity_resolution._codec import from_record, to_record
from contextmap.entity_resolution._vectors import dot as raw_dot
from contextmap.entity_resolution._vectors import unit as raw_unit
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_mapping import Entity
from contextmap.visual_perception import (
    EmbeddingSpaceMismatchError,
    FeatureId,
    FeatureScope,
    FeatureStoreReader,
    FeatureStoreWriter,
    PerceptionRunId,
    write_feature_index,
)

POLICY = AppearanceComparisonPolicy(
    embedding_space_id=SPACE_A, min_supporting_similarity=0.8, max_conflicting_similarity=0.2
)


def unit(*values: float) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def entity_with(entity_id: str, features: dict[int, list[Any]], **kwargs: Any) -> Entity:
    """An entity seen in the given frames, with the given features per frame."""
    refs = tuple(
        sorted(
            (
                feature_ref(second, index, **kwargs)
                for second, items in features.items()
                for index, _ in enumerate(items)
            ),
            key=lambda ref: (ref.perception_result_id, ref.feature_id),
        )
    )
    return entity_at(entity_id, seconds=sorted(features), features=refs)


def vectors_of(entity_id: str, features: dict[int, list[Any]]) -> dict[str, Any]:
    return {
        f"feature-{second:04d}-{index:02d}": vector
        for second, items in features.items()
        for index, vector in enumerate(items)
    }


def build(
    first: dict[int, list[Any]], second: dict[int, list[Any]], **kwargs: Any
) -> tuple[Entity, Entity, InMemoryFeatureSource]:
    return (
        entity_with("a", first),
        entity_with("b", second, **kwargs),
        InMemoryFeatureSource({**vectors_of("a", first), **vectors_of("b", second)}),
    )


# --- reproducible evidence with full provenance -------------------------------------------------


def test_compatible_features_produce_evidence_with_full_provenance() -> None:
    first, second, source = build({10: [unit(1, 0, 0)]}, {40: [unit(1, 0.1, 0)]})

    evidence = AppearanceComparator(source, POLICY).compare(first, second)

    measurement = evidence.measurement
    assert measurement is not None
    assert measurement.embedding_space_id == SPACE_A
    assert measurement.metric == "cosine-similarity"
    assert measurement.aggregation_id == APPEARANCE_AGGREGATION_ID
    assert measurement.similarity == pytest.approx(1 / math.sqrt(1.01), rel=1e-9)
    assert [item.physical_observation_id for item in measurement.contributions_a] == [frame(10)]
    assert [item.physical_observation_id for item in measurement.contributions_b] == [frame(40)]
    assert measurement.contributions_a[0].feature_refs == (feature_ref(10),)
    assert evidence.status is EvidenceStatus.SUPPORTING
    assert evidence.policy.policy_id == APPEARANCE_COMPARISON_POLICY_ID
    assert evidence.policy.configuration_fingerprint == POLICY.fingerprint()


def test_the_evidence_is_reproducible_and_symmetric() -> None:
    first, second, source = build({10: [unit(1, 0, 0)]}, {40: [unit(1, 1, 0)]})
    comparator = AppearanceComparator(source, POLICY)

    assert comparator.compare(first, second) == comparator.compare(first, second)
    assert comparator.compare(first, second) == comparator.compare(second, first)


def test_similarity_is_read_through_explicit_thresholds() -> None:
    def status(a: tuple[float, ...], b: tuple[float, ...], policy: Any = POLICY) -> EvidenceStatus:
        first, second, source = build({10: [a]}, {40: [b]})
        return AppearanceComparator(source, policy).compare(first, second).status

    assert status(unit(1, 0, 0), unit(1, 0.1, 0)) is EvidenceStatus.SUPPORTING
    assert status(unit(1, 0, 0), unit(1, 1, 0)) is EvidenceStatus.NEUTRAL
    assert status(unit(1, 0, 0), unit(0, 1, 0)) is EvidenceStatus.CONFLICTING
    # Sem limiar de conflito, uma similaridade baixa (outro ponto de vista) não é evidência contra.
    only_support = AppearanceComparisonPolicy(
        embedding_space_id=SPACE_A, min_supporting_similarity=0.8
    )
    assert status(unit(1, 0, 0), unit(0, 1, 0), only_support) is EvidenceStatus.NEUTRAL


# --- incompatible or missing evidence is unavailable, never a score -----------------------------


def test_features_of_another_space_are_unavailable_even_with_the_same_dimension() -> None:
    first, _, source = build({10: [unit(1, 0, 0)]}, {40: [unit(1, 0, 0)]})
    other_space = entity_with("b", {40: [unit(1, 0, 0)]}, space=SPACE_B)

    evidence = AppearanceComparator(source, POLICY).compare(first, other_space)

    assert evidence.status is EvidenceStatus.UNAVAILABLE
    assert evidence.measurement is None
    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.INCOMPATIBLE_DOMAIN
    assert SPACE_A in evidence.unavailable.detail and SPACE_B in evidence.unavailable.detail
    assert source.loads == []


def test_an_entity_without_region_features_is_missing_evidence_not_a_zero() -> None:
    first, _, source = build({10: [unit(1, 0, 0)]}, {40: [unit(1, 0, 0)]})
    bare = entity_at("b", seconds=(40,))

    evidence = AppearanceComparator(source, POLICY).compare(first, bare)

    assert evidence.status is EvidenceStatus.UNAVAILABLE
    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.MISSING_EVIDENCE
    assert source.loads == []


def test_global_features_do_not_describe_the_entity() -> None:
    first, _, source = build({10: [unit(1, 0, 0)]}, {40: [unit(1, 0, 0)]})
    only_global = entity_at(
        "b",
        seconds=(40,),
        features=(feature_ref(40, scope=FeatureScope.GLOBAL),),
    )

    evidence = AppearanceComparator(source, POLICY).compare(first, only_global)

    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.MISSING_EVIDENCE


def test_a_store_that_contradicts_its_reference_fails_loudly() -> None:
    first = entity_with("a", {10: [None]})
    second = entity_with("b", {40: [None]})
    source = InMemoryFeatureSource(
        {"feature-0010-00": unit(1, 0, 0), "feature-0040-00": unit(1, 0, 0)},
        reported_space={"feature-0040-00": SPACE_B},
    )

    with pytest.raises(EmbeddingSpaceMismatchError):
        AppearanceComparator(source, POLICY).compare(first, second)


def test_vectors_of_different_dimensions_in_one_space_are_refused() -> None:
    first, second, source = build({10: [unit(1, 0, 0)]}, {40: [unit(1, 0)]})

    with pytest.raises(ValueError, match="dimension"):
        AppearanceComparator(source, POLICY).compare(first, second)


@pytest.mark.parametrize("bad", [(0.0, 0.0, 0.0), (math.nan, 0.0, 1.0), (math.inf, 0.0, 1.0)])
def test_a_degenerate_vector_is_corrupt_evidence_and_refused(bad: tuple[float, ...]) -> None:
    first, second, source = build({10: [bad]}, {40: [unit(1, 0, 0)]})

    with pytest.raises(ValueError, match="feature-0010-00"):
        AppearanceComparator(source, POLICY).compare(first, second)


def test_a_feature_that_belongs_to_no_physical_observation_of_the_entity_is_refused() -> None:
    stray = entity_at("b", seconds=(40,), features=(feature_ref(99),))
    first, _, source = build({10: [unit(1, 0, 0)]}, {40: [unit(1, 0, 0)]})

    with pytest.raises(ValueError, match="physical observation"):
        AppearanceComparator(source, POLICY).compare(first, stray)


# --- multiple observations ----------------------------------------------------------------------


def test_repeated_inference_on_one_frame_is_one_observation_not_independent_evidence() -> None:
    once, other, source = build({10: [unit(1, 0, 0)]}, {40: [unit(1, 1, 0)]})
    thrice = entity_with("a", {10: [None, None, None]})
    triple_source = InMemoryFeatureSource(
        {
            "feature-0010-00": unit(1, 0, 0),
            "feature-0010-01": unit(1, 0, 0),
            "feature-0010-02": unit(1, 0, 0),
            "feature-0040-00": unit(1, 1, 0),
        }
    )

    single = AppearanceComparator(source, POLICY).compare(once, other)
    repeated = AppearanceComparator(triple_source, POLICY).compare(thrice, other)

    assert repeated.measurement is not None and single.measurement is not None
    assert len(repeated.measurement.contributions_a) == 1
    assert len(repeated.measurement.contributions_a[0].feature_refs) == 3
    assert repeated.measurement.similarity == pytest.approx(single.measurement.similarity)


def test_each_physical_observation_counts_once_in_the_prototype() -> None:
    frames_a = {10: [unit(1, 0, 0)], 12: [unit(0, 1, 0)]}
    first, second, source = build(frames_a, {40: [unit(1, 1, 0)]})

    evidence = AppearanceComparator(source, POLICY).compare(first, second)

    measurement = evidence.measurement
    assert measurement is not None
    assert [item.physical_observation_id for item in measurement.contributions_a] == [
        frame(10),
        frame(12),
    ]
    # O protótipo é a média dos dois pontos de vista, e o alvo está exatamente entre eles.
    assert measurement.similarity == pytest.approx(1.0)
    assert measurement.pair_similarity_min == pytest.approx(math.cos(math.pi / 4))
    assert measurement.pair_similarity_max == pytest.approx(math.cos(math.pi / 4))


def test_a_view_from_a_different_angle_widens_the_pair_range() -> None:
    first, second, source = build(
        {10: [unit(1, 0, 0)]}, {40: [unit(1, 0.2, 0)], 42: [unit(1, 3, 0)]}
    )

    measurement = AppearanceComparator(source, POLICY).compare(first, second).measurement

    assert measurement is not None
    assert measurement.pair_similarity_min < measurement.pair_similarity_max


def test_identical_prototypes_yield_similarity_one_without_error() -> None:
    # Regressão ER-01: o produto interno cru de um vetor unitário consigo mesmo passa de 1.0
    # por arredondamento, e o clamp assimétrico deixava pair_similarity_min > max.
    vector = unit(1, 1, 1)
    prototype = raw_unit(vector, what="precondition")
    assert raw_dot(prototype, prototype) > 1.0
    first, second, source = build({10: [vector]}, {40: [vector]})

    measurement = AppearanceComparator(source, POLICY).compare(first, second).measurement

    assert measurement is not None
    assert measurement.similarity == 1.0
    assert measurement.pair_similarity_min == 1.0
    assert measurement.pair_similarity_max == 1.0


def test_opposing_prototypes_cancel_and_the_channel_is_unavailable() -> None:
    first, second, source = build(
        {10: [unit(1, 0, 0)], 12: [unit(-1, 0, 0)]}, {40: [unit(1, 0, 0)]}
    )

    evidence = AppearanceComparator(source, POLICY).compare(first, second)

    assert evidence.unavailable is not None
    assert evidence.unavailable.reason is UnavailableReason.INSUFFICIENT_EVIDENCE


def test_an_entity_is_loaded_once_however_many_pairs_it_takes_part_in() -> None:
    first, second, source = build({10: [unit(1, 0, 0)]}, {40: [unit(1, 0, 0)]})
    third = entity_with("c", {50: [unit(1, 0, 0)]})
    source = InMemoryFeatureSource(
        {
            "feature-0010-00": unit(1, 0, 0),
            "feature-0040-00": unit(1, 0, 0),
            "feature-0050-00": unit(1, 0, 0),
        }
    )
    comparator = AppearanceComparator(source, POLICY)

    comparator.compare(first, second)
    comparator.compare(first, third)

    assert sorted(source.loads) == ["feature-0010-00", "feature-0040-00", "feature-0050-00"]


# --- policy -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_space_id": ""},
        {"min_supporting_similarity": 1.5},
        {"min_supporting_similarity": math.nan},
        {"max_conflicting_similarity": 0.9},
        {"max_conflicting_similarity": -1.5},
    ],
)
def test_the_policy_refuses_impossible_thresholds(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "embedding_space_id": SPACE_A,
        "min_supporting_similarity": 0.8,
        "max_conflicting_similarity": 0.2,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        AppearanceComparisonPolicy(**values)


def test_the_fingerprint_changes_with_the_space_and_every_threshold() -> None:
    variants = [
        AppearanceComparisonPolicy(
            embedding_space_id=SPACE_B,
            min_supporting_similarity=0.8,
            max_conflicting_similarity=0.2,
        ),
        AppearanceComparisonPolicy(
            embedding_space_id=SPACE_A,
            min_supporting_similarity=0.7,
            max_conflicting_similarity=0.2,
        ),
        AppearanceComparisonPolicy(embedding_space_id=SPACE_A, min_supporting_similarity=0.8),
    ]

    assert len({POLICY.fingerprint(), *(item.fingerprint() for item in variants)}) == 4


def test_the_evidence_survives_a_json_round_trip() -> None:
    first, second, source = build(
        {10: [unit(1, 0, 0)], 12: [unit(1, 0.1, 0)]}, {40: [unit(1, 0, 0)]}
    )
    evidence = AppearanceComparator(source, POLICY).compare(first, second)

    record = json.loads(json.dumps(to_record(evidence)))

    assert from_record(AppearanceEvidence, record) == evidence


# --- the real feature store ---------------------------------------------------------------------


def write_store(
    root: Path, vectors: dict[str, tuple[float, ...]], *, space: str = SPACE_A
) -> FeatureStoreReader:
    writer = FeatureStoreWriter(root)
    for key, values in vectors.items():
        second = int(key.split("-")[1])
        ref = feature_ref(second, int(key.split("-")[2]), space=space)
        feature = visual_feature(ref, dimension=len(values))
        writer.write(feature, SourceObservationId(frame(second)), np.array(values, dtype="float32"))
    write_feature_index(root, writer.entries())
    return FeatureStoreReader.open(root)


def test_the_real_feature_store_gives_the_same_evidence_as_the_in_memory_source(
    tmp_path: Path,
) -> None:
    first, second, memory = build({10: [unit(1, 0, 0)]}, {40: [unit(1, 0.1, 0)]})
    reader = write_store(
        tmp_path, {"feature-0010-00": unit(1, 0, 0), "feature-0040-00": unit(1, 0.1, 0)}
    )

    real = AppearanceComparator(FeatureStoreVectorSource({RUN: reader}), POLICY).compare(
        first, second
    )
    fake = AppearanceComparator(memory, POLICY).compare(first, second)

    assert real.measurement is not None and fake.measurement is not None
    assert real.measurement.similarity == pytest.approx(fake.measurement.similarity, abs=1e-6)
    assert real.status is fake.status


def test_the_feature_store_source_refuses_an_unknown_run_and_a_non_vector(tmp_path: Path) -> None:
    reader = write_store(tmp_path, {"feature-0010-00": unit(1, 0, 0)})
    source = FeatureStoreVectorSource({RUN: reader})

    with pytest.raises(ValueError, match="perception run"):
        source.load(
            feature_ref(10, run=PerceptionRunId("another-run")), SourceObservationId(frame(10))
        )

    matrix_dir = tmp_path / "matrix"
    writer = FeatureStoreWriter(matrix_dir)
    ref = feature_ref(10)
    matrix = visual_feature(ref, dimension=3)
    dense = replace(matrix, shape=(2, 3))
    writer.write(dense, SourceObservationId(frame(10)), np.zeros((2, 3), dtype="float32"))
    write_feature_index(matrix_dir, writer.entries())
    with pytest.raises(ValueError, match="vector"):
        FeatureStoreVectorSource({RUN: FeatureStoreReader.open(matrix_dir)}).load(
            ref, SourceObservationId(frame(10))
        )
    assert FeatureId("feature-0010-00") == ref.feature_id


def test_the_feature_store_source_refuses_a_scope_that_differs_from_the_reference(
    tmp_path: Path,
) -> None:
    ref = feature_ref(10)
    global_feature = replace(
        visual_feature(ref, dimension=3), scope=FeatureScope.GLOBAL, region_id=None
    )
    writer = FeatureStoreWriter(tmp_path)
    writer.write(global_feature, SourceObservationId(frame(10)), np.ones(3, dtype="float32"))
    write_feature_index(tmp_path, writer.entries())
    source = FeatureStoreVectorSource({RUN: FeatureStoreReader.open(tmp_path)})

    with pytest.raises(ValueError, match="scoped"):
        source.load(ref, SourceObservationId(frame(10)))

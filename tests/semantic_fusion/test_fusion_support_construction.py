import dataclasses
import math
import random
import typing
from collections.abc import Iterable

import pytest
from fusion_builders import MAP_ID, frame_timestamp, geometry_refs, spatial_id
from geometry_fake import InMemoryGeometrySource
from observation_builders import make_spatial_observation, timestamps

from contextmap.geometric_mapping import MapId
from contextmap.semantic_fusion import (
    GEOMETRY_OVERLAP_SUPPORT_POLICY_ID,
    ExcludedObservation,
    FusionSupportBuild,
    GeometryOverlapSupportPolicy,
    build_fusion_supports,
)
from contextmap.semantic_fusion import support as support_module
from contextmap.sensor_association import SpatialObservation

POLICY = GeometryOverlapSupportPolicy(min_geometry_count=3, min_overlap=0.5)
FRAMES = ("frame-0120", "frame-0121", "frame-0122", "frame-0123")


def _source() -> InMemoryGeometrySource:
    return InMemoryGeometrySource(MAP_ID, {i: (i * 0.1, 0.0, 0.0) for i in range(200)})


def _obs(
    frame: str, indexes: Iterable[int], *, run: str = "run-a", region: str = "region-0001"
) -> SpatialObservation:
    return make_spatial_observation(run, frame, region, support=tuple(indexes))


def _build(
    observations: Iterable[SpatialObservation],
    *,
    policy: GeometryOverlapSupportPolicy = POLICY,
    source: InMemoryGeometrySource | None = None,
) -> FusionSupportBuild:
    return build_fusion_supports(
        observations,
        geometry=_source() if source is None else source,
        acquisition_timestamps=timestamps(FRAMES),
        policy=policy,
    )


def _members(build: FusionSupportBuild) -> list[tuple[str, ...]]:
    return [tuple(support.spatial_observation_ids) for support in build.supports]


def test_repeated_observations_of_the_same_area_share_one_support() -> None:
    build = _build([_obs(frame, range(20)) for frame in FRAMES[:3]])

    assert len(build.supports) == 1
    support = build.supports[0]
    assert support.spatial_observation_ids == tuple(
        spatial_id("run-a", frame) for frame in FRAMES[:3]
    )
    assert support.geometry_support == geometry_refs(range(20))
    assert build.excluded == ()


def test_a_support_summarizes_geometry_time_and_policy() -> None:
    build = _build([_obs("frame-0120", range(20)), _obs("frame-0121", range(20))])

    support = build.supports[0]
    assert support.geometric_map_id == MAP_ID
    assert support.bounds.frame_id == "map"
    assert support.bounds.minimum_m == (0.0, 0.0, 0.0)
    assert support.bounds.maximum_m == pytest.approx((1.9, 0.0, 0.0))
    assert support.centroid_m == pytest.approx((0.95, 0.0, 0.0))
    assert support.time_bounds.start == frame_timestamp("frame-0120")
    assert support.time_bounds.end == frame_timestamp("frame-0121")
    assert support.provenance.support_policy_id == GEOMETRY_OVERLAP_SUPPORT_POLICY_ID
    assert support.provenance.configuration_fingerprint == POLICY.fingerprint()


def test_a_flat_support_keeps_its_centroid_inside_the_degenerate_bounds() -> None:
    # Regressão SF-01: fsum([0.1] * 3) / 3 == 0.10000000000000002 > 0.1.
    flat = InMemoryGeometrySource(MAP_ID, {i: (i * 0.1, 0.0, 0.1) for i in range(3)})

    build = _build([_obs("frame-0120", range(3))], source=flat)

    support = build.supports[0]
    assert support.bounds.maximum_m[2] == 0.1
    assert support.centroid_m[2] == 0.1


def test_each_geometry_element_is_resolved_once_however_many_views_see_it() -> None:
    source = _source()

    _build([_obs(frame, range(20)) for frame in FRAMES], source=source)

    assert source.lookups == 20


def test_strong_overlap_merges_and_partial_overlap_depends_on_the_declared_threshold() -> None:
    views = [_obs("frame-0120", range(0, 20)), _obs("frame-0121", range(15, 35))]

    assert len(_build(views).supports) == 2
    loose = GeometryOverlapSupportPolicy(min_geometry_count=3, min_overlap=0.1)
    assert len(_build(views, policy=loose).supports) == 1


def test_the_overlap_threshold_is_inclusive() -> None:
    views = [_obs("frame-0120", range(10)), _obs("frame-0121", range(5))]

    assert len(_build(views).supports) == 1
    strict = GeometryOverlapSupportPolicy(min_geometry_count=3, min_overlap=0.51)
    assert len(_build(views, policy=strict).supports) == 2


def test_disjoint_regions_are_not_merged_even_if_their_claims_match() -> None:
    same_claims = [_obs("frame-0120", range(0, 20)), _obs("frame-0121", range(100, 120))]

    build = _build(same_claims)

    assert len(build.supports) == 2
    assert same_claims[0].semantic_claim_refs == same_claims[1].semantic_claim_refs


def test_claims_never_influence_the_grouping() -> None:
    views = [_obs("frame-0120", range(20)), _obs("frame-0121", range(20))]
    unlabeled = [dataclasses.replace(view, semantic_claim_refs=()) for view in views]

    assert _members(_build(views)) == _members(_build(unlabeled))


def test_a_small_support_nested_in_a_large_one_stays_separate() -> None:
    door_and_handle = [_obs("frame-0120", range(50)), _obs("frame-0121", range(10))]

    build = _build(door_and_handle)

    assert _members(build) == [
        (spatial_id("run-a", "frame-0120"),),
        (spatial_id("run-a", "frame-0121"),),
    ]


def test_overlapping_masks_of_one_observation_share_a_support_when_they_overlap_strongly() -> None:
    whole = _obs("frame-0120", range(0, 20), region="region-0001")
    shifted = _obs("frame-0120", range(2, 22), region="region-0002")
    apart = _obs("frame-0120", range(100, 120), region="region-0003")

    build = _build([whole, shifted, apart])

    assert len(build.supports) == 2
    assert build.supports[0].spatial_observation_ids == (
        spatial_id("run-a", "frame-0120", "region-0001"),
        spatial_id("run-a", "frame-0120", "region-0002"),
    )
    assert build.supports[0].time_bounds.start == build.supports[0].time_bounds.end


def test_overlap_chains_merge_transitively() -> None:
    chain = [
        _obs("frame-0120", range(0, 10)),
        _obs("frame-0121", range(5, 15)),
        _obs("frame-0122", range(10, 20)),
    ]
    policy = GeometryOverlapSupportPolicy(min_geometry_count=3, min_overlap=0.3)

    build = _build(chain, policy=policy)

    assert len(build.supports) == 1
    assert len(build.supports[0].spatial_observation_ids) == 3


def test_an_observation_with_too_little_geometry_is_excluded_explicitly() -> None:
    sparse = _obs("frame-0121", (5, 6))
    empty = _obs("frame-0122", ())

    build = _build([_obs("frame-0120", range(20)), sparse, empty])

    assert len(build.supports) == 1
    assert build.excluded == (
        ExcludedObservation(
            spatial_observation_id=sparse.spatial_observation_id,
            geometry_count=2,
            minimum_geometry_count=3,
        ),
        ExcludedObservation(
            spatial_observation_id=empty.spatial_observation_id,
            geometry_count=0,
            minimum_geometry_count=3,
        ),
    )


def test_every_observation_is_in_exactly_one_support_or_explicitly_excluded() -> None:
    views = [
        _obs("frame-0120", range(20)),
        _obs("frame-0121", range(20)),
        _obs("frame-0122", range(100, 120)),
        _obs("frame-0123", (1,)),
    ]

    build = _build(views)
    support_of = build.support_id_of()

    excluded = {item.spatial_observation_id for item in build.excluded}
    assert set(support_of) | excluded == {view.spatial_observation_id for view in views}
    assert not set(support_of) & excluded
    first, second = spatial_id("run-a", "frame-0120"), spatial_id("run-a", "frame-0121")
    assert support_of[first] == support_of[second]


def test_the_build_is_the_same_whatever_the_order_of_the_inputs() -> None:
    views = [
        _obs(frame, indexes, region=region)
        for frame in FRAMES
        for indexes, region in ((range(0, 20), "region-0001"), (range(100, 120), "region-0002"))
    ]
    expected = _build(views)

    for seed in range(5):
        shuffled = list(views)
        random.Random(seed).shuffle(shuffled)
        assert _build(shuffled) == expected


def test_support_identities_follow_the_order_of_their_first_observation() -> None:
    build = _build([_obs("frame-0121", range(100, 120)), _obs("frame-0120", range(20))])

    assert [support.fusion_support_id for support in build.supports] == [
        "support-000001",
        "support-000002",
    ]
    assert build.supports[0].spatial_observation_ids == (spatial_id("run-a", "frame-0120"),)


def test_the_fingerprint_changes_with_the_configuration_only() -> None:
    same = GeometryOverlapSupportPolicy(min_geometry_count=3, min_overlap=0.5)
    other = GeometryOverlapSupportPolicy(min_geometry_count=3, min_overlap=0.6)

    assert same.fingerprint() == POLICY.fingerprint()
    assert other.fingerprint() != POLICY.fingerprint()
    assert POLICY.fingerprint().startswith("sha256:")


def test_no_observation_is_an_empty_build() -> None:
    build = _build([])

    assert build.supports == ()
    assert build.excluded == ()
    assert build.support_id_of() == {}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_geometry_count": 0, "min_overlap": 0.5}, "min_geometry_count"),
        ({"min_geometry_count": 3, "min_overlap": 0.0}, "min_overlap"),
        ({"min_geometry_count": 3, "min_overlap": 1.1}, "min_overlap"),
        ({"min_geometry_count": 3, "min_overlap": math.nan}, "min_overlap"),
    ],
)
def test_an_impossible_policy_is_rejected(kwargs: dict[str, typing.Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GeometryOverlapSupportPolicy(**kwargs)


def test_an_observation_of_another_map_than_the_geometry_source_is_rejected() -> None:
    foreign = make_spatial_observation("run-a", "frame-0120", map_id=MapId("map-0002"))

    with pytest.raises(ValueError, match="map-0002"):
        _build([foreign])


def test_geometry_missing_from_the_map_is_rejected_with_the_observation() -> None:
    lost = _obs("frame-0120", (0, 1, 999))

    with pytest.raises(ValueError, match=r"spatial--run-a--frame-0120--region-0001.*999"):
        _build([lost])


def test_a_frame_without_an_acquisition_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"no acquisition timestamp.*frame-0130"):
        _build([_obs("frame-0130", range(20))])


def test_the_same_spatial_observation_twice_is_rejected() -> None:
    view = _obs("frame-0120", range(20))

    with pytest.raises(ValueError, match="duplicate spatial observation"):
        _build([view, view])


def test_the_build_rejects_an_observation_in_two_supports() -> None:
    build = _build([_obs("frame-0120", range(20)), _obs("frame-0121", range(100, 120))])
    first, second = build.supports
    twin = dataclasses.replace(second, spatial_observation_ids=first.spatial_observation_ids)

    with pytest.raises(ValueError, match="more than one support"):
        dataclasses.replace(build, supports=(first, twin))


def test_the_build_rejects_supports_out_of_order() -> None:
    build = _build([_obs("frame-0120", range(20)), _obs("frame-0121", range(100, 120))])

    with pytest.raises(ValueError, match="supports must be sorted and unique"):
        dataclasses.replace(build, supports=tuple(reversed(build.supports)))


# --- SF-03: pares candidatos pelo índice de geometria compartilhada ------------------------


def _brute_force_groups(
    observations: list[SpatialObservation], policy: GeometryOverlapSupportPolicy
) -> list[tuple[str, ...]]:
    """Referência de força bruta: Jaccard de todos os pares e componentes conexas."""
    ordered = sorted(observations, key=lambda item: item.spatial_observation_id)
    eligible = [item for item in ordered if len(item.geometry_support) >= policy.min_geometry_count]
    sets = [{ref.geometry_id for ref in item.geometry_support} for item in eligible]
    root = list(range(len(eligible)))

    def find(index: int) -> int:
        while root[index] != index:
            index = root[index]
        return index

    for left in range(len(eligible)):
        for right in range(left + 1, len(eligible)):
            union = len(sets[left] | sets[right])
            if len(sets[left] & sets[right]) / union >= policy.min_overlap:
                low, high = sorted((find(left), find(right)))
                root[high] = low
    groups: dict[int, list[str]] = {}
    for index, item in enumerate(eligible):
        groups.setdefault(find(index), []).append(item.spatial_observation_id)
    return [tuple(groups[key]) for key in sorted(groups)]


def _random_observations(seed: int) -> list[SpatialObservation]:
    rng = random.Random(seed)
    observations = []
    for number in range(30):
        shape = rng.choice(["range", "range", "subset", "tiny"])
        if shape == "range":
            start = rng.randrange(0, 180)
            indexes: Iterable[int] = range(start, min(200, start + rng.randrange(3, 40)))
        elif shape == "subset":
            indexes = sorted(rng.sample(range(200), rng.randrange(3, 30)))
        else:
            indexes = range(rng.randrange(0, 199), 200)[:2]
        frame = rng.choice(FRAMES)
        observations.append(_obs(frame, indexes, region=f"region-{number:04d}"))
    return observations


@pytest.mark.parametrize("seed", range(30))
def test_the_supports_are_those_of_the_all_pairs_reference(seed: int) -> None:
    observations = _random_observations(seed)

    build = _build(observations)

    assert _members(build) == _brute_force_groups(observations, POLICY)
    assert [support.fusion_support_id for support in build.supports] == [
        f"support-{number:06d}" for number in range(1, len(build.supports) + 1)
    ]
    assert {item.spatial_observation_id for item in build.excluded} == {
        item.spatial_observation_id
        for item in observations
        if len(item.geometry_support) < POLICY.min_geometry_count
    }


def test_only_observations_that_share_geometry_are_compared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Gate de escala por contador: N observações disjuntas mais um par que se sobrepõe; o
    # número de pares avaliados segue os pares que se intersectam, não N².
    evaluated = 0
    real = support_module._overlap

    def counting(*args: int) -> float:
        nonlocal evaluated
        evaluated += 1
        return real(*args)

    monkeypatch.setattr(support_module, "_overlap", counting)
    disjoint = [
        _obs("frame-0120", range(3 * n, 3 * n + 3), region=f"region-{n:04d}") for n in range(60)
    ]
    overlapping = _obs("frame-0121", range(0, 3), region="region-overlap")

    build = _build([*disjoint, overlapping])

    assert len(build.supports) == 60
    assert evaluated == 1

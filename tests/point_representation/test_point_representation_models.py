import dataclasses
import math
import subprocess
import sys
from typing import Any

import pytest
from pointrep_builders import (
    ENCODER,
    knn_policy,
    make_representation,
    make_space,
    make_statistics,
    make_support,
    radius_policy,
    ref,
)

import contextmap.point_representation as point_representation
from contextmap.geometric_mapping import MapId
from contextmap.point_representation import (
    CenteringMode,
    CoordinatePreparation,
    NeighborhoodMethod,
    PointRepresentation,
    PointSupport,
    PreparedSupport,
    RepresentationSpace,
    RepresentationSpaceMismatchError,
    ScaleNormalization,
    SupportPolicy,
    SupportStatistics,
    SupportType,
    ensure_compatible_representation_spaces,
    ensure_compatible_representations,
    representation_id_for,
    representation_space_fingerprint,
)


def test_public_api_exports_resolve() -> None:
    for name in point_representation.__all__:
        assert hasattr(point_representation, name), name


def test_the_contracts_are_readable_without_model_or_numeric_libraries() -> None:
    code = (
        "import sys, contextmap.point_representation;"
        "bad = [m for m in ('numpy', 'torch', 'rosbags', 'open3d') if m in sys.modules];"
        "assert not bad, bad"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


# --- Support policy ---------------------------------------------------------


def test_a_radius_policy_declares_its_radius_and_preparation() -> None:
    policy = radius_policy(0.75, max_neighbors=64)

    assert policy.support_type is SupportType.NEIGHBORHOOD
    assert policy.method is NeighborhoodMethod.RADIUS
    assert (policy.radius_m, policy.k, policy.max_neighbors) == (0.75, None, 64)
    assert policy.preparation.centering is CenteringMode.CENTER


def test_a_point_policy_needs_no_neighborhood_parameters() -> None:
    policy = SupportPolicy(
        support_type=SupportType.POINT,
        method=None,
        radius_m=None,
        k=None,
        max_neighbors=None,
        preparation=CoordinatePreparation(),
    )

    assert policy.support_type is SupportType.POINT


@pytest.mark.parametrize(
    "overrides",
    [
        {"radius_m": 0.0},
        {"radius_m": -1.0},
        {"radius_m": math.nan},
        {"radius_m": math.inf},
        {"radius_m": None},
        {"k": 4},
        {"max_neighbors": 0},
    ],
)
def test_invalid_radius_policies_are_rejected(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "support_type": SupportType.NEIGHBORHOOD,
        "method": NeighborhoodMethod.RADIUS,
        "radius_m": 0.5,
        "k": None,
        "max_neighbors": None,
        "preparation": CoordinatePreparation(),
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="policy"):
        SupportPolicy(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"k": 0},
        {"k": None},
        {"radius_m": 0.5},
        {"max_neighbors": 5},
        {
            "preparation": CoordinatePreparation(
                scale_normalization=ScaleNormalization.SUPPORT_RADIUS
            )
        },
    ],
)
def test_invalid_knn_policies_are_rejected(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "support_type": SupportType.NEIGHBORHOOD,
        "method": NeighborhoodMethod.K_NEAREST,
        "radius_m": None,
        "k": 8,
        "max_neighbors": None,
        "preparation": CoordinatePreparation(),
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="policy"):
        SupportPolicy(**values)  # type: ignore[arg-type]


def test_a_point_policy_rejects_neighborhood_parameters_and_scale_normalization() -> None:
    with pytest.raises(ValueError, match="policy"):
        SupportPolicy(
            support_type=SupportType.POINT,
            method=None,
            radius_m=0.5,
            k=None,
            max_neighbors=None,
            preparation=CoordinatePreparation(),
        )
    with pytest.raises(ValueError, match="policy"):
        SupportPolicy(
            support_type=SupportType.POINT,
            method=None,
            radius_m=None,
            k=None,
            max_neighbors=None,
            preparation=CoordinatePreparation(
                scale_normalization=ScaleNormalization.SUPPORT_RADIUS
            ),
        )


def test_a_neighborhood_policy_needs_a_method() -> None:
    with pytest.raises(ValueError, match="policy"):
        SupportPolicy(
            support_type=SupportType.NEIGHBORHOOD,
            method=None,
            radius_m=None,
            k=None,
            max_neighbors=None,
            preparation=CoordinatePreparation(),
        )


# --- Support statistics and PointSupport ------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"count": 0},
        {"min_distance_m": -0.1},
        {"max_distance_m": 0.1, "min_distance_m": 0.2},
        {"mean_distance_m": 9.0},
        {"max_distance_m": math.nan},
        {"candidate_count": 1, "count": 4},
    ],
)
def test_support_statistics_are_validated(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "count": 4,
        "min_distance_m": 0.0,
        "max_distance_m": 0.4,
        "mean_distance_m": 0.2,
        "near_map_bounds": False,
        "candidate_count": None,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="statistics"):
        SupportStatistics(**values)  # type: ignore[arg-type]


def test_a_support_identifies_exactly_which_geometry_formed_it() -> None:
    support = make_support(center_index=2, member_indexes=(2, 0, 5, 7))

    assert support.center == ref(2)
    assert support.geometry_refs == (ref(2), ref(0), ref(5), ref(7))
    assert support.map_frame == "map"
    assert support.statistics.count == 4
    assert support.query_method == "bounds-query+euclidean-filter"


def test_a_support_always_contains_its_center_and_never_repeats_a_reference() -> None:
    with pytest.raises(ValueError, match="support"):
        make_support(center_index=9, member_indexes=(0, 1, 2))
    with pytest.raises(ValueError, match="support"):
        make_support(center_index=0, member_indexes=(0, 1, 1))


def test_every_reference_of_a_support_belongs_to_the_center_map() -> None:
    support_refs = (ref(0), ref(1, map_id=MapId("other-map")))

    with pytest.raises(ValueError, match="support"):
        dataclasses.replace(make_support(), geometry_refs=support_refs)


def test_the_statistics_count_must_match_the_references() -> None:
    with pytest.raises(ValueError, match="support"):
        dataclasses.replace(make_support(), statistics=make_statistics(count=9))


def test_a_point_support_is_just_its_center() -> None:
    policy = SupportPolicy(
        support_type=SupportType.POINT,
        method=None,
        radius_m=None,
        k=None,
        max_neighbors=None,
        preparation=CoordinatePreparation(),
    )

    single = make_support(center_index=3, member_indexes=(3,), policy=policy)

    assert single.geometry_refs == (ref(3),)
    with pytest.raises(ValueError, match="support"):
        make_support(center_index=3, member_indexes=(3, 4), policy=policy)


def test_a_radius_support_cannot_reach_beyond_its_radius() -> None:
    far = dataclasses.replace(make_statistics(), max_distance_m=0.9, mean_distance_m=0.3)

    with pytest.raises(ValueError, match="support"):
        dataclasses.replace(make_support(policy=radius_policy(0.5)), statistics=far)


def test_a_knn_support_never_exceeds_k() -> None:
    with pytest.raises(ValueError, match="support"):
        make_support(member_indexes=(0, 1, 2, 3, 4), policy=knn_policy(3))


def test_the_applied_scale_is_the_radius_exactly_when_scale_is_normalized() -> None:
    scaled = make_support(policy=radius_policy(0.5, scale=ScaleNormalization.SUPPORT_RADIUS))

    assert scaled.applied_scale_m == 0.5
    assert make_support(policy=radius_policy(0.5)).applied_scale_m is None
    assert make_support(policy=knn_policy(4)).applied_scale_m is None


# --- RepresentationSpace ----------------------------------------------------


def test_the_space_fingerprint_is_deterministic() -> None:
    assert representation_space_fingerprint(make_space()) == representation_space_fingerprint(
        make_space()
    )
    assert representation_space_fingerprint(make_space()).startswith("sha256:")


@pytest.mark.parametrize(
    "variant",
    [
        {"family": "ptv3"},
        {"model": "other"},
        {"version": "2"},
        {"checkpoint": "sha256:abc"},
        {"dimension": 5, "feature_names": ()},
        {"dtype": "float64"},
        {"normalization": "l2"},
        {"policy": radius_policy(1.0)},
        {"policy": knn_policy(8)},
        {"feature_names": ("a", "b", "c", "z")},
    ],
)
def test_the_fingerprint_changes_with_every_semantic_field(variant: dict[str, object]) -> None:
    base = representation_space_fingerprint(make_space())

    assert representation_space_fingerprint(make_space(**variant)) != base  # type: ignore[arg-type]


def test_equal_dimensionality_never_implies_compatibility() -> None:
    descriptor = make_space(family="geometric_descriptor", dimension=16, feature_names=())
    learned = make_space(family="ptv3", dimension=16, feature_names=(), checkpoint="sha256:w")

    with pytest.raises(RepresentationSpaceMismatchError):
        ensure_compatible_representation_spaces(descriptor, learned)
    ensure_compatible_representation_spaces(descriptor, descriptor)


def test_the_support_policy_is_part_of_the_space_identity() -> None:
    small = make_space(policy=radius_policy(0.25))
    large = make_space(policy=radius_policy(1.0))

    with pytest.raises(RepresentationSpaceMismatchError):
        ensure_compatible_representation_spaces(small, large)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dimension": 0},
        {"dtype": "float16"},
        {"family": ""},
        {"normalization": ""},
        {"input_definition": ""},
        {"feature_names": ("a", "b")},
        {"feature_names": ("a", "a", "c", "d")},
    ],
)
def test_a_space_is_validated(overrides: dict[str, object]) -> None:
    values = dataclasses.asdict(make_space())
    values["support_semantics"] = radius_policy()
    values["feature_names"] = ("a", "b", "c", "d")
    values.update(overrides)

    with pytest.raises(ValueError, match=r"space|dimension|dtype|feature"):
        RepresentationSpace(**values)


# --- PointRepresentation ----------------------------------------------------


def test_a_representation_is_anchored_to_geometry_and_its_reconstructable_support() -> None:
    representation = make_representation(index=1, support=make_support(1, (1, 2, 3)))

    assert representation.geometry_reference == ref(1)
    assert representation.support.geometry_refs == (ref(1), ref(2), ref(3))
    assert representation.shape == (4,)
    assert representation.dtype == "float32"
    assert representation.encoder_identity == ENCODER
    assert representation.representation_id == representation_id_for(
        run_id=point_representation.PointRepresentationRunId("run-0001"), index=1
    )


def test_the_anchor_must_be_the_center_of_the_support() -> None:
    with pytest.raises(ValueError, match="geometry_reference"):
        dataclasses.replace(make_representation(), geometry_reference=ref(2))


def test_representation_ids_are_deterministic_and_run_scoped() -> None:
    run = point_representation.PointRepresentationRunId("run-0001")

    assert representation_id_for(run_id=run, index=3) == representation_id_for(run_id=run, index=3)
    assert representation_id_for(run_id=run, index=3) != representation_id_for(run_id=run, index=4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"shape": ()},
        {"shape": (0,)},
        {"dtype": "int8"},
        {"representation_space_id": ""},
        {"normalization": ""},
        {"payload_reference": ""},
        {"payload_reference": "/etc/passwd"},
        {"payload_reference": "../escape"},
    ],
)
def test_a_representation_is_validated(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"representation|shape|dtype|payload"):
        dataclasses.replace(make_representation(), **overrides)


def test_a_representation_without_a_stored_payload_says_so() -> None:
    assert make_representation(payload_reference=None).payload_reference is None


def test_undefined_components_are_explicit_and_within_the_dimension() -> None:
    partial = make_representation(undefined_components=(1, 3))

    assert partial.undefined_components == (1, 3)
    assert partial.is_partial
    assert not make_representation().is_partial
    with pytest.raises(ValueError, match="undefined_components"):
        make_representation(undefined_components=(4,))
    with pytest.raises(ValueError, match="undefined_components"):
        make_representation(undefined_components=(2, 1))


def test_representations_of_different_spaces_are_never_compared_implicitly() -> None:
    descriptor = make_representation(space=make_space(family="geometric_descriptor"))
    learned = make_representation(space=make_space(family="ptv3", checkpoint="sha256:w"))

    ensure_compatible_representations(descriptor, descriptor)
    with pytest.raises(RepresentationSpaceMismatchError):
        ensure_compatible_representations(descriptor, learned)


def test_visual_and_semantic_evidence_stay_out_of_the_representation_contract() -> None:
    forbidden = {"label", "claim", "claims", "entity", "visual_feature", "embedding", "features"}

    for contract in (PointRepresentation, PointSupport, RepresentationSpace):
        assert not forbidden & {field.name for field in dataclasses.fields(contract)}


def test_a_prepared_support_carries_local_coordinates_for_every_supporting_point() -> None:
    support = make_support(0, (0, 1, 2))

    prepared = PreparedSupport(
        support=support, local_coordinates_m=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0))
    )

    assert len(prepared.local_coordinates_m) == len(prepared.support.geometry_refs)
    with pytest.raises(ValueError, match="prepared"):
        PreparedSupport(support=support, local_coordinates_m=((0.0, 0.0, 0.0),))
    with pytest.raises(ValueError, match="prepared"):
        PreparedSupport(
            support=support,
            local_coordinates_m=((0.0, 0.0, 0.0), (math.nan, 0.0, 0.0), (0.0, 0.1, 0.0)),
        )

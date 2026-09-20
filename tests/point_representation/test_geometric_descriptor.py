import dataclasses
import math
import random
import subprocess
import sys

import pytest
from pointrep_builders import radius_policy
from pointrep_geometry import MAP_ID, LinearScanSource, points_from

from contextmap.geometric_mapping import GeometryReference, geometry_id_for
from contextmap.point_representation import (
    CenteringMode,
    CoordinatePreparation,
    EncodedRepresentation,
    PointEncoder,
    PointRepresentationRunId,
    PreparedSupport,
    RepresentationService,
    ScaleNormalization,
    SupportExtractor,
    SupportPolicy,
    SupportType,
    representation_space_fingerprint,
)
from contextmap.point_representation.backends.geometric_descriptor import (
    DESCRIPTOR_VERSION,
    FEATURE_NAMES,
    GeometricDescriptorEncoder,
)

WIDE = radius_policy(10.0)
Vector = tuple[float, float, float]


def ref(index: int) -> GeometryReference:
    return GeometryReference(map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=index))


def prepared_from(
    coordinates: list[Vector], *, policy: SupportPolicy = WIDE, center_index: int = 0
) -> PreparedSupport:
    """A real prepared support over ``coordinates`` (every point is inside ``policy``)."""
    source = LinearScanSource(points_from(coordinates))
    return SupportExtractor(source, policy).extract(ref(center_index))


def describe(
    prepared: PreparedSupport, policy: SupportPolicy = WIDE
) -> tuple[dict[str, float], set[str]]:
    """The named feature values and the names of the undefined ones."""
    encoded = GeometricDescriptorEncoder(policy).encode(prepared)
    values = dict(zip(FEATURE_NAMES, encoded.values, strict=True))
    undefined = {FEATURE_NAMES[index] for index in encoded.undefined_components}
    return values, undefined


def grid(nx: int, ny: int, nz: int, spacing_m: float = 0.125) -> list[Vector]:
    """A lattice centered on the origin; dyadic spacing keeps sums exact."""
    return [
        (
            (i - (nx - 1) / 2) * spacing_m,
            (j - (ny - 1) / 2) * spacing_m,
            (k - (nz - 1) / 2) * spacing_m,
        )
        for i in range(nx)
        for j in range(ny)
        for k in range(nz)
    ]


def rotate_about_z(coordinates: list[Vector], degrees: float) -> list[Vector]:
    cosine, sine = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return [(x * cosine - y * sine, x * sine + y * cosine, z) for x, y, z in coordinates]


PLANE = grid(9, 9, 1)
RECTANGLE = grid(9, 5, 1)
LINE = [(t, t, 0.0) for t in (k * 0.125 for k in range(9))]
VOLUME = grid(5, 5, 5)

NORMAL = ("normal_abs_x", "normal_abs_y", "normal_abs_z")
AXIS = ("principal_axis_abs_x", "principal_axis_abs_y", "principal_axis_abs_z")
SHAPE = ("linearity", "planarity", "scattering", "surface_variation")


# --- The versioned descriptor space --------------------------------------------


def test_the_descriptor_space_is_versioned_and_fully_defined() -> None:
    space = GeometricDescriptorEncoder(radius_policy(0.5)).representation_space()

    assert (space.family, space.model, space.version) == (
        "geometric_descriptor",
        "local-covariance-shape",
        DESCRIPTOR_VERSION,
    )
    assert space.checkpoint is None
    assert (space.dimension, space.dtype, space.normalization) == (14, "float64", "none")
    assert space.input_definition == "xyz-local-prepared"
    assert space.feature_names == FEATURE_NAMES
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES) == 14
    assert space.support_semantics == radius_policy(0.5)


def test_the_space_fingerprint_depends_on_the_support_policy() -> None:
    same_a = GeometricDescriptorEncoder(radius_policy(0.5)).representation_space()
    same_b = GeometricDescriptorEncoder(radius_policy(0.5)).representation_space()
    other = GeometricDescriptorEncoder(radius_policy(1.0)).representation_space()

    assert representation_space_fingerprint(same_a) == representation_space_fingerprint(same_b)
    assert representation_space_fingerprint(same_a) != representation_space_fingerprint(other)


def test_the_encoder_identity_is_deterministic_and_has_no_checkpoint() -> None:
    identity = GeometricDescriptorEncoder(radius_policy(0.5)).encoder_identity()

    assert identity == GeometricDescriptorEncoder(radius_policy(0.5)).encoder_identity()
    assert identity.configuration_fingerprint != (
        GeometricDescriptorEncoder(radius_policy(1.0)).encoder_identity().configuration_fingerprint
    )
    assert identity.backend_id == "geometric_descriptor"
    assert identity.backend_version == DESCRIPTOR_VERSION
    assert identity.checkpoint_hash is None
    assert identity.configuration_fingerprint.startswith("sha256:")


def test_a_point_support_policy_is_an_impossible_configuration() -> None:
    point_policy = SupportPolicy(
        support_type=SupportType.POINT,
        method=None,
        radius_m=None,
        k=None,
        max_neighbors=None,
        preparation=CoordinatePreparation(),
    )

    with pytest.raises(ValueError, match="neighborhood"):
        GeometricDescriptorEncoder(point_policy)


def test_a_support_prepared_under_another_policy_is_rejected() -> None:
    prepared = prepared_from(PLANE, policy=radius_policy(1.0))

    with pytest.raises(ValueError, match="policy"):
        GeometricDescriptorEncoder(radius_policy(2.0)).encode(prepared)


def test_the_descriptor_satisfies_the_encoder_port_without_a_model_runtime() -> None:
    assert isinstance(GeometricDescriptorEncoder(WIDE), PointEncoder)
    code = (
        "import sys, contextmap.point_representation.backends.geometric_descriptor;"
        "bad = [m for m in ('numpy', 'torch', 'transformers') if m in sys.modules];"
        "assert not bad, bad"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


# --- Planar, linear and volumetric supports ------------------------------------


def test_a_planar_support_is_planar_with_its_normal_along_the_axis() -> None:
    values, undefined = describe(prepared_from(PLANE))

    assert values["planarity"] == pytest.approx(1.0, abs=1e-9)
    assert values["linearity"] == pytest.approx(0.0, abs=1e-9)
    assert values["scattering"] == pytest.approx(0.0, abs=1e-9)
    assert values["surface_variation"] == pytest.approx(0.0, abs=1e-9)
    assert [values[name] for name in NORMAL] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    # Um plano isotrópico não tem direção principal: nada é inventado para ela.
    assert undefined == set(AXIS)


def test_an_elongated_planar_support_reports_its_principal_axis() -> None:
    values, undefined = describe(prepared_from(RECTANGLE))

    assert [values[name] for name in AXIS] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert [values[name] for name in NORMAL] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert values["planarity"] == pytest.approx(0.3, abs=1e-9)
    assert values["linearity"] == pytest.approx(0.7, abs=1e-9)
    assert undefined == set()


def test_a_linear_support_is_linear_and_has_no_normal() -> None:
    values, undefined = describe(prepared_from(LINE))

    assert values["linearity"] == pytest.approx(1.0, abs=1e-9)
    assert values["planarity"] == pytest.approx(0.0, abs=1e-9)
    assert values["scattering"] == pytest.approx(0.0, abs=1e-9)
    assert [values[name] for name in AXIS] == pytest.approx(
        [math.sqrt(0.5), math.sqrt(0.5), 0.0], abs=1e-9
    )
    assert undefined == set(NORMAL)


def test_a_volumetric_support_is_scattered_and_has_no_direction() -> None:
    values, undefined = describe(prepared_from(VOLUME))

    assert values["scattering"] == pytest.approx(1.0, abs=1e-9)
    assert values["surface_variation"] == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert values["linearity"] == pytest.approx(0.0, abs=1e-9)
    assert values["planarity"] == pytest.approx(0.0, abs=1e-9)
    assert undefined == {*NORMAL, *AXIS}


def test_planar_linear_and_volumetric_supports_are_distinguishable() -> None:
    plane, _ = describe(prepared_from(PLANE))
    line, _ = describe(prepared_from(LINE))
    volume, _ = describe(prepared_from(VOLUME))

    assert plane["planarity"] > 0.9 > max(line["planarity"], volume["planarity"])
    assert line["linearity"] > 0.9 > max(plane["linearity"], volume["linearity"])
    assert volume["scattering"] > 0.9 > max(plane["scattering"], line["scattering"])


def test_a_noisy_plane_keeps_a_well_defined_normal() -> None:
    rng = random.Random(5)
    noisy = [
        (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.gauss(0.0, 0.005)) for _ in range(200)
    ]

    values, undefined = describe(prepared_from(noisy))

    assert values["normal_abs_z"] > 0.999
    assert 0.0 < values["surface_variation"] < 0.01
    assert not set(NORMAL) & undefined


def test_shape_features_are_rotation_invariant_and_the_axis_follows_the_rotation() -> None:
    reference, _ = describe(prepared_from(RECTANGLE))
    rotated, _ = describe(prepared_from(rotate_about_z(RECTANGLE, 30.0)))

    for name in (*SHAPE, "normal_abs_z", "rms_radius", "max_radius"):
        assert rotated[name] == pytest.approx(reference[name], abs=1e-9)
    assert [rotated[name] for name in AXIS] == pytest.approx(
        [math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0], abs=1e-9
    )


@pytest.mark.parametrize("seed", range(25))
def test_eigenvalue_features_match_an_independent_numpy_decomposition(seed: int) -> None:
    numpy = pytest.importorskip("numpy")
    rng = random.Random(seed)
    anisotropic = [(rng.gauss(0, 0.30), rng.gauss(0, 0.12), rng.gauss(0, 0.04)) for _ in range(80)]
    # Uma rotação aleatória descarta o caso trivial de autovetores alinhados aos eixos.
    rotation, _ = numpy.linalg.qr(
        numpy.array([[rng.gauss(0, 1) for _ in range(3)] for _ in range(3)])
    )
    cloud: list[Vector] = []
    for point in anisotropic:
        x, y, z = (float(c) for c in rotation @ numpy.array(point))
        cloud.append((x, y, z))

    values, undefined = describe(prepared_from(cloud))

    centered = numpy.array(cloud) - numpy.mean(cloud, axis=0)
    eigenvalues, eigenvectors = numpy.linalg.eigh(centered.T @ centered / len(cloud))
    l3, l2, l1 = (float(value) for value in eigenvalues)
    assert values["linearity"] == pytest.approx((l1 - l2) / l1, abs=1e-9)
    assert values["planarity"] == pytest.approx((l2 - l3) / l1, abs=1e-9)
    assert values["scattering"] == pytest.approx(l3 / l1, abs=1e-9)
    assert values["surface_variation"] == pytest.approx(l3 / (l1 + l2 + l3), abs=1e-9)
    assert [values[name] for name in NORMAL] == pytest.approx(
        [abs(float(component)) for component in eigenvectors[:, 0]], abs=1e-7
    )
    assert [values[name] for name in AXIS] == pytest.approx(
        [abs(float(component)) for component in eigenvectors[:, 2]], abs=1e-7
    )
    assert undefined == set()


# --- Extent and density ---------------------------------------------------------


def test_extent_and_density_are_measured_in_meters() -> None:
    prepared = prepared_from([(k * 0.125, 0.0, 0.0) for k in range(5)], policy=radius_policy(1.0))

    values, undefined = describe(prepared, radius_policy(1.0))

    xs = [0.0, 0.125, 0.25, 0.375, 0.5]
    mean = sum(xs) / 5
    assert values["support_size"] == 5.0
    assert values["rms_radius"] == pytest.approx(math.sqrt(sum((x - mean) ** 2 for x in xs) / 5))
    assert values["max_radius"] == pytest.approx(0.25)
    # A densidade usa a distância do membro mais distante ao centro (0,5 m), em metros.
    assert values["density_per_m3"] == pytest.approx(5.0 / (4.0 / 3.0 * math.pi * 0.5**3))
    assert "density_per_m3" not in undefined


# --- Degenerate supports ---------------------------------------------------------


def test_a_single_point_leaves_undefined_what_it_cannot_define() -> None:
    encoder = GeometricDescriptorEncoder(radius_policy(0.5))
    prepared = prepared_from([(0.0, 0.0, 0.0), (9.0, 9.0, 9.0)], policy=radius_policy(0.5))

    encoded = encoder.encode(prepared)

    values = dict(zip(FEATURE_NAMES, encoded.values, strict=True))
    undefined = {FEATURE_NAMES[index] for index in encoded.undefined_components}
    assert values["support_size"] == 1.0
    assert values["rms_radius"] == 0.0
    assert values["max_radius"] == 0.0
    assert undefined == {"density_per_m3", *SHAPE, *NORMAL, *AXIS}
    assert all(values[name] == 0.0 for name in undefined)
    assert encoded.undefined_components == tuple(sorted(encoded.undefined_components))


def test_two_points_form_a_line() -> None:
    values, undefined = describe(prepared_from([(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)]))

    assert values["linearity"] == pytest.approx(1.0)
    assert values["planarity"] == pytest.approx(0.0)
    assert [values[name] for name in AXIS] == pytest.approx([1.0, 0.0, 0.0])
    assert undefined == set(NORMAL)


def test_coincident_points_have_no_shape_density_or_direction() -> None:
    values, undefined = describe(prepared_from([(0.5, 0.5, 0.5)] * 4))

    assert values["support_size"] == 4.0
    assert values["rms_radius"] == 0.0
    assert undefined == {"density_per_m3", *SHAPE, *NORMAL, *AXIS}


def test_undefined_components_never_carry_an_invented_value() -> None:
    encoded = GeometricDescriptorEncoder(WIDE).encode(prepared_from(VOLUME))

    assert encoded.undefined_components
    assert all(encoded.values[index] == 0.0 for index in encoded.undefined_components)
    assert all(math.isfinite(value) for value in encoded.values)


# --- Reproducibility and input independence -------------------------------------


def test_identical_support_and_configuration_give_identical_vectors() -> None:
    prepared = prepared_from(rotate_about_z(RECTANGLE, 17.0))
    encoder = GeometricDescriptorEncoder(WIDE)

    assert encoder.encode(prepared) == encoder.encode(prepared)


def test_the_member_order_does_not_change_the_vector() -> None:
    prepared = prepared_from(rotate_about_z(RECTANGLE, 17.0))
    order = list(range(len(prepared.local_coordinates_m)))
    random.Random(1).shuffle(order)
    shuffled = PreparedSupport(
        support=dataclasses.replace(
            prepared.support, geometry_refs=tuple(prepared.support.geometry_refs[i] for i in order)
        ),
        local_coordinates_m=tuple(prepared.local_coordinates_m[i] for i in order),
    )
    encoder = GeometricDescriptorEncoder(WIDE)

    assert encoder.encode(shuffled).values == pytest.approx(
        encoder.encode(prepared).values, abs=1e-12
    )


@pytest.mark.parametrize("centering", [CenteringMode.CENTER, CenteringMode.CENTROID])
def test_the_vector_is_independent_of_where_the_coordinates_are_centered(
    centering: CenteringMode,
) -> None:
    coordinates = [(x + 3.0, y - 2.0, z + 0.5) for x, y, z in rotate_about_z(RECTANGLE, 17.0)]
    baseline_policy = radius_policy(10.0, centering=CenteringMode.NONE)
    policy = radius_policy(10.0, centering=centering)

    baseline, _ = describe(prepared_from(coordinates, policy=baseline_policy), baseline_policy)
    centered, _ = describe(prepared_from(coordinates, policy=policy), policy)

    assert list(centered.values()) == pytest.approx(list(baseline.values()), abs=1e-9)


def test_scale_normalization_rescales_extent_and_keeps_the_shape() -> None:
    scaled_policy = radius_policy(
        2.0, centering=CenteringMode.CENTER, scale=ScaleNormalization.SUPPORT_RADIUS
    )
    plain_policy = radius_policy(2.0)
    coordinates = rotate_about_z(RECTANGLE, 17.0)

    plain, _ = describe(prepared_from(coordinates, policy=plain_policy), plain_policy)
    scaled, _ = describe(prepared_from(coordinates, policy=scaled_policy), scaled_policy)

    for name in (*SHAPE, *NORMAL, "support_size", "density_per_m3"):
        assert scaled[name] == pytest.approx(plain[name], abs=1e-9)
    assert scaled["rms_radius"] == pytest.approx(plain["rms_radius"] / 2.0)
    assert scaled["max_radius"] == pytest.approx(plain["max_radius"] / 2.0)


# --- Through the execution service ------------------------------------------------


def test_the_service_publishes_partial_descriptors_for_degenerate_supports() -> None:
    cube = grid(5, 5, 5, spacing_m=0.25)
    source = LinearScanSource(points_from([*cube, (50.0, 50.0, 50.0)]))
    encoder = GeometricDescriptorEncoder(radius_policy(0.3))
    service = RepresentationService(
        source, encoder, run_id=PointRepresentationRunId("run-0001"), code_version="test"
    )
    middle, isolated = ref(62), ref(len(cube))

    outcomes = list(service.represent([middle, isolated]))

    assert all(isinstance(outcome, EncodedRepresentation) for outcome in outcomes)
    space_id = representation_space_fingerprint(encoder.representation_space())
    for outcome in outcomes:
        assert isinstance(outcome, EncodedRepresentation)
        assert outcome.representation.representation_space_id == space_id
        assert outcome.representation.shape == (14,)
        assert outcome.representation.dtype == "float64"
        assert outcome.representation.is_partial
    assert service.metrics.partial == 2

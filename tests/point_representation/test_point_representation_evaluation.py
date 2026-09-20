import dataclasses
import json
import math
import subprocess
import sys
from typing import Any

import pytest
from pointrep_builders import radius_policy
from pointrep_fakes import FakeEncoder, RejectingSmallSupportsEncoder
from pointrep_geometry import MAP_ID, LinearScanSource, points_from

from contextmap.evaluation import (
    GeometryVariation,
    RepresentationAblationReport,
    RepresentationArm,
    RepresentationArmReport,
    RepresentationArmRole,
    RepresentationEvaluationContext,
    RepresentationEvaluationError,
    compare_representation_arms,
    encode_representation_ablation_report,
    encode_representation_arm_report,
    evaluate_representation_arm,
    noise_variation,
    rotation_about_z_variation,
    subsample_variation,
    translation_variation,
)
from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.point_representation import (
    EncodedVector,
    PreparedSupport,
    RepresentationSpace,
    center_selection_id,
    representation_space_fingerprint,
)
from contextmap.point_representation.backends.geometric_descriptor import (
    GeometricDescriptorEncoder,
)

POLICY = radius_policy(0.3)
Vector = tuple[float, float, float]
CONTEXT = RepresentationEvaluationContext(
    evaluation_id="eval-0001",
    code_version="test-version",
    downstream_configuration_fingerprint="sha256:downstream-fixed",
)


def ref(index: int) -> GeometryReference:
    return GeometryReference(map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=index))


def lattice(count: int = 9, spacing_m: float = 0.125) -> list[Vector]:
    """A flat ``count x count`` lattice on z = 0; index is ``i * count + j``."""
    return [(i * spacing_m, j * spacing_m, 0.0) for i in range(count) for j in range(count)]


CENTERS = [ref(index) for index in (40, 30, 50, 22, 58)]


def make_source() -> LinearScanSource:
    return LinearScanSource(points_from(lattice()))


def descriptor_arm() -> RepresentationArm:
    return RepresentationArm(
        arm_id="geometric_descriptor",
        role=RepresentationArmRole.DETERMINISTIC_DESCRIPTOR,
        encoder=GeometricDescriptorEncoder(POLICY),
    )


def evaluate(arm: RepresentationArm, **kwargs: Any) -> RepresentationArmReport:
    return evaluate_representation_arm(
        arm, source=make_source(), centers=CENTERS, context=CONTEXT, **kwargs
    )


# --- Arms and context ----------------------------------------------------------------


def test_the_off_arm_has_no_encoder_and_every_other_arm_has_one() -> None:
    RepresentationArm(arm_id="off", role=RepresentationArmRole.OFF, encoder=None)
    with pytest.raises(ValueError, match="off"):
        RepresentationArm(arm_id="off", role=RepresentationArmRole.OFF, encoder=FakeEncoder(POLICY))
    with pytest.raises(ValueError, match="encoder"):
        RepresentationArm(arm_id="learned", role=RepresentationArmRole.LEARNED_3D, encoder=None)
    with pytest.raises(ValueError, match="arm_id"):
        RepresentationArm(arm_id="", role=RepresentationArmRole.OFF, encoder=None)


def test_ptv3_alone_cannot_be_labeled_a_pretrained_distilled_representation() -> None:
    ptv3_like = FakeEncoder(POLICY, family="ptv3", checkpoint="sha256:weights")

    RepresentationArm(arm_id="ptv3", role=RepresentationArmRole.LEARNED_3D, encoder=ptv3_like)
    with pytest.raises(ValueError, match="PTv3"):
        RepresentationArm(
            arm_id="ptv3-as-vernata",
            role=RepresentationArmRole.PRETRAINED_DISTILLED,
            encoder=ptv3_like,
        )


def test_a_pretrained_distilled_backend_fits_the_same_harness() -> None:
    distilled = RepresentationArm(
        arm_id="distilled",
        role=RepresentationArmRole.PRETRAINED_DISTILLED,
        encoder=FakeEncoder(POLICY, family="sonata_like_encoder", checkpoint="sha256:w"),
    )

    report = evaluate(distilled)

    assert report.coverage is not None
    assert report.coverage.represented == 5


@pytest.mark.parametrize(
    "field", ["evaluation_id", "code_version", "downstream_configuration_fingerprint"]
)
def test_the_context_needs_every_identity(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(CONTEXT, **{field: ""})


# --- The off arm -----------------------------------------------------------------------


def test_the_off_arm_reports_no_representation_metrics_only_the_downstream_baseline() -> None:
    off = RepresentationArm(arm_id="off", role=RepresentationArmRole.OFF, encoder=None)

    report = evaluate(off, downstream={"entity_resolution.false_merge_rate": 0.12})

    assert report.identity.encoder is None
    assert report.identity.representation_space_id is None
    assert report.identity.support_policy is None
    assert report.coverage is None
    assert report.repeatability is None
    assert report.distribution is None
    assert report.cost is None
    assert report.sensitivity == ()
    assert dict(report.downstream) == {"entity_resolution.false_merge_rate": 0.12}
    assert report.geometric_map_id == MAP_ID
    assert report.center_selection_id == center_selection_id(map_id=MAP_ID, centers=CENTERS)


# --- Representation-level validation ----------------------------------------------------


def test_a_descriptor_arm_is_fully_identified_and_covers_the_requested_centers() -> None:
    arm = descriptor_arm()
    assert arm.encoder is not None

    report = evaluate(arm)

    assert report.identity.encoder == arm.encoder.encoder_identity()
    assert report.identity.representation_space_id == representation_space_fingerprint(
        arm.encoder.representation_space()
    )
    assert report.identity.support_policy == POLICY
    assert (
        report.downstream_configuration_fingerprint == CONTEXT.downstream_configuration_fingerprint
    )
    assert report.coverage is not None
    assert (report.coverage.requested, report.coverage.represented) == (5, 5)
    assert report.coverage.partial == 5
    assert dict(report.coverage.failed_by_reason) == {}
    # Só o eixo principal (3 de 14 componentes) é indefinido em um patch simétrico.
    assert report.coverage.undefined_component_fraction == pytest.approx(3 / 14)


def test_repeatability_norms_and_cost_are_separate_sections() -> None:
    report = evaluate(descriptor_arm(), backend_diagnostics={"peak_memory_bytes": 4096})

    assert report.repeatability is not None
    assert report.repeatability.compared == 5
    assert report.repeatability.max_abs_difference == 0.0
    assert report.repeatability.repeatable is True
    assert report.distribution is not None
    assert report.distribution.count == 5
    assert report.distribution.minimum is not None and report.distribution.minimum > 0
    assert report.cost is not None
    assert report.cost.payload_bytes == 5 * 14 * 8
    assert report.cost.encoding_seconds >= 0.0
    assert report.cost.support_extraction_seconds >= 0.0
    assert report.cost.seconds_per_representation is not None
    assert dict(report.cost.backend_diagnostics) == {"peak_memory_bytes": 4096}


class DriftingEncoder(FakeEncoder):
    """Returns a different vector on every call, like a non-deterministic backend."""

    def __init__(self) -> None:
        super().__init__(POLICY)
        self._calls_made = 0

    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        self._calls_made += 1
        base = super().encode(prepared)
        return EncodedVector(values=tuple(v + 1e-3 * self._calls_made for v in base.values))


def test_non_repeatable_output_is_reported_not_hidden_and_tolerance_is_explicit() -> None:
    arm = RepresentationArm(
        arm_id="drifting", role=RepresentationArmRole.LEARNED_3D, encoder=DriftingEncoder()
    )

    strict = evaluate(arm)
    loose = evaluate(arm, repeatability_tolerance=1.0)

    assert strict.repeatability is not None and strict.repeatability.repeatable is False
    assert strict.repeatability.max_abs_difference > 0.0
    assert loose.repeatability is not None and loose.repeatability.repeatable is True
    assert loose.repeatability.tolerance == 1.0


def test_failures_are_reported_and_never_counted_as_representations() -> None:
    arm = RepresentationArm(
        arm_id="rejecting",
        role=RepresentationArmRole.LEARNED_3D,
        encoder=RejectingSmallSupportsEncoder(POLICY, minimum=99),
    )

    report = evaluate(arm, variations=[translation_variation((1.0, 0.0, 0.0))])

    assert report.coverage is not None
    assert (report.coverage.represented, report.coverage.requested) == (0, 5)
    assert dict(report.coverage.failed_by_reason) == {"unencodable_support": 5}
    assert report.distribution is None
    assert report.repeatability is not None
    assert report.repeatability.compared == 0
    assert report.repeatability.repeatable is None
    assert report.sensitivity[0].compared == 0
    assert report.sensitivity[0].unmatched == 5


# --- Sensitivity and spatial consistency ---------------------------------------------------


def test_a_rigid_translation_leaves_the_descriptor_unchanged() -> None:
    report = evaluate(descriptor_arm(), variations=[translation_variation((3.0, -2.0, 1.0))])

    (study,) = report.sensitivity
    assert (study.compared, study.unmatched) == (5, 0)
    assert study.relative_l2_change.maximum is not None
    assert study.relative_l2_change.maximum < 1e-9
    assert study.cosine_distance.maximum is not None
    assert study.cosine_distance.maximum < 1e-12
    assert study.component_mean_absolute_change is not None
    assert max(study.component_mean_absolute_change.values()) < 1e-9


def test_a_rotation_about_the_vertical_axis_leaves_a_flat_patch_unchanged() -> None:
    report = evaluate(descriptor_arm(), variations=[rotation_about_z_variation(30.0)])

    (study,) = report.sensitivity
    assert study.relative_l2_change.maximum is not None
    assert study.relative_l2_change.maximum < 1e-9


def test_sensitivity_to_noise_grows_with_the_perturbation() -> None:
    report = evaluate(
        descriptor_arm(),
        variations=[
            noise_variation(sigma_m=0.005, seed=1),
            noise_variation(sigma_m=0.03, seed=1),
        ],
    )

    small, large = report.sensitivity
    assert small.component_mean_absolute_change is not None
    assert large.component_mean_absolute_change is not None
    assert (
        large.component_mean_absolute_change["surface_variation"]
        > small.component_mean_absolute_change["surface_variation"]
        > 0.0
    )
    assert small.variation != large.variation


def test_sensitivity_to_density_changes_the_support_size() -> None:
    report = evaluate(descriptor_arm(), variations=[subsample_variation(0.6, seed=2)])

    (study,) = report.sensitivity
    assert (study.compared, study.unmatched) == (5, 0)
    assert study.component_mean_absolute_change is not None
    assert study.component_mean_absolute_change["support_size"] > 0.0


class OpaqueEncoder(FakeEncoder):
    """Like a learned encoder: its components have no interpretable names."""

    def representation_space(self) -> RepresentationSpace:
        return dataclasses.replace(super().representation_space(), feature_names=())


def test_components_are_only_named_for_interpretable_descriptors() -> None:
    opaque = RepresentationArm(
        arm_id="opaque", role=RepresentationArmRole.LEARNED_3D, encoder=OpaqueEncoder(POLICY)
    )

    opaque_report = evaluate(opaque, variations=[translation_variation((1.0, 0.0, 0.0))])
    named_report = evaluate(descriptor_arm(), variations=[translation_variation((1.0, 0.0, 0.0))])

    assert opaque_report.sensitivity[0].component_mean_absolute_change is None
    assert opaque_report.sensitivity[0].relative_l2_change.count == 5
    assert named_report.sensitivity[0].component_mean_absolute_change is not None


def test_a_variation_never_changes_the_evaluated_geometry() -> None:
    source = make_source()
    before = [point.coordinates_m for point in source.iter_geometry()]
    varied = noise_variation(sigma_m=0.05, seed=3).apply(source, CENTERS)

    assert [point.coordinates_m for point in source.iter_geometry()] == before
    assert varied.geometric_map.map_id == source.geometric_map.map_id
    assert [p.coordinates_m for p in varied.iter_geometry()] != before


def test_subsampling_always_keeps_the_centers() -> None:
    varied = subsample_variation(0.1, seed=4).apply(make_source(), CENTERS)

    kept = {point.reference for point in varied.iter_geometry()}
    assert set(CENTERS) <= kept
    assert varied.geometric_map.point_count < 81


@pytest.mark.parametrize(
    "build",
    [
        lambda: noise_variation(sigma_m=-0.1, seed=1),
        lambda: noise_variation(sigma_m=math.nan, seed=1),
        lambda: subsample_variation(0.0, seed=1),
        lambda: subsample_variation(1.5, seed=1),
        lambda: translation_variation((math.inf, 0.0, 0.0)),
        lambda: rotation_about_z_variation(math.nan),
    ],
)
def test_an_invalid_variation_is_rejected(build: Any) -> None:
    with pytest.raises(ValueError, match=r"variation|sigma|keep_fraction|finite|offset|degrees"):
        build()


def test_a_variation_is_a_named_and_described_recipe() -> None:
    variation = translation_variation((1.0, 0.0, 0.0))

    assert isinstance(variation, GeometryVariation)
    assert variation.name and variation.description


# --- Controlled comparison -----------------------------------------------------------------


def _arms_reports() -> list[RepresentationArmReport]:
    off = RepresentationArm(arm_id="off", role=RepresentationArmRole.OFF, encoder=None)
    learned = RepresentationArm(
        arm_id="ptv3",
        role=RepresentationArmRole.LEARNED_3D,
        encoder=FakeEncoder(POLICY, family="ptv3", checkpoint="sha256:weights"),
    )
    return [
        evaluate(off, downstream={"fusion.conflict_rate": 0.30}),
        evaluate(descriptor_arm(), downstream={"fusion.conflict_rate": 0.22}),
        evaluate(learned, downstream={"fusion.conflict_rate": 0.19}),
    ]


def test_arms_are_compared_side_by_side_under_the_same_controlled_variables() -> None:
    reports = _arms_reports()

    comparison = compare_representation_arms(reports)

    assert isinstance(comparison, RepresentationAblationReport)
    assert [arm.identity.arm_id for arm in comparison.arms] == [
        "off",
        "geometric_descriptor",
        "ptv3",
    ]
    assert comparison.geometric_map_id == MAP_ID
    assert comparison.center_selection_id == reports[0].center_selection_id
    assert comparison.downstream_configuration_fingerprint == (
        CONTEXT.downstream_configuration_fingerprint
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"geometric_map_id": MapId("other-map")}, "geometric map"),
        ({"center_selection_id": "sha256:other"}, "center selection"),
        ({"downstream_configuration_fingerprint": "sha256:drifted"}, "downstream"),
    ],
)
def test_a_drift_in_a_controlled_variable_is_rejected(
    override: dict[str, Any], message: str
) -> None:
    off, descriptor, learned = _arms_reports()

    with pytest.raises(RepresentationEvaluationError, match=message):
        compare_representation_arms([off, dataclasses.replace(descriptor, **override), learned])


def test_arms_must_be_distinct_and_at_least_one() -> None:
    off = _arms_reports()[0]

    with pytest.raises(RepresentationEvaluationError, match="duplicate"):
        compare_representation_arms([off, off])
    with pytest.raises(RepresentationEvaluationError, match="at least one"):
        compare_representation_arms([])


def test_there_is_no_single_winner_score_and_sections_stay_separate() -> None:
    comparison = compare_representation_arms(_arms_reports())
    forbidden = ("best", "winner", "score", "rank", "overall")

    for contract in (RepresentationAblationReport, RepresentationArmReport):
        names = {field.name for field in dataclasses.fields(contract)}
        assert not [n for n in names if any(word in n for word in forbidden)]
    record = encode_representation_ablation_report(comparison)
    arm = record["arms"][1]
    assert {"identity", "representation", "cost", "downstream"} <= set(arm)
    assert not any(any(word in key for word in forbidden) for key in _all_keys(record))


def _all_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(k) for k in value] + [key for v in value.values() for key in _all_keys(v)]
    if isinstance(value, list):
        return [key for item in value for key in _all_keys(item)]
    return []


def test_a_report_encodes_to_json_with_every_identity_needed_to_reproduce_it() -> None:
    report = evaluate(descriptor_arm(), variations=[translation_variation((1.0, 0.0, 0.0))])

    record = json.loads(json.dumps(encode_representation_arm_report(report)))

    assert record["identity"]["arm_id"] == "geometric_descriptor"
    assert record["identity"]["role"] == "deterministic_descriptor"
    assert record["identity"]["encoder"]["backend_id"] == "geometric_descriptor"
    assert record["identity"]["representation_space_id"].startswith("sha256:")
    assert record["identity"]["support_policy"]["radius_m"] == 0.3
    assert record["geometric_map_id"] == MAP_ID
    assert record["center_selection_id"].startswith("sha256:")
    assert record["representation"]["sensitivity"][0]["variation"]
    assert record["cost"]["payload_bytes"] == 5 * 14 * 8


def test_the_same_arm_and_inputs_reproduce_the_same_evidence() -> None:
    first = evaluate(descriptor_arm(), variations=[noise_variation(sigma_m=0.01, seed=5)])
    second = evaluate(descriptor_arm(), variations=[noise_variation(sigma_m=0.01, seed=5)])

    assert dataclasses.replace(first, cost=None) == dataclasses.replace(second, cost=None)


def test_the_harness_needs_no_gpu_or_model_runtime() -> None:
    code = (
        "import sys, contextmap.evaluation;"
        "bad = [m for m in ('numpy', 'torch', 'transformers') if m in sys.modules];"
        "assert not bad, bad"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr

import dataclasses
import math
import subprocess
import sys
from typing import Any

import pytest
from pointrep_builders import knn_policy, radius_policy
from pointrep_fakes import FakeEncoder
from pointrep_geometry import MAP_ID, LinearScanSource, line_of_points
from pointrep_ptv3_fakes import CHECKPOINT_HASH, FakePTv3Runtime, make_config

from contextmap.geometric_mapping import GeometryReference, geometry_id_for
from contextmap.point_representation import (
    EncodedRepresentation,
    FailedSupport,
    FailureReason,
    PointEncoder,
    PointRepresentationRunId,
    PreparedSupport,
    RepresentationService,
    RepresentationSpaceMismatchError,
    SupportExtractor,
    SupportType,
    UnencodableSupportError,
    ensure_compatible_representations,
    representation_space_fingerprint,
)
from contextmap.point_representation.backends.ptv3 import (
    PTv3Config,
    PTv3Inference,
    PTv3OutOfMemoryError,
    PTv3PointEncoder,
    PTv3RuntimeUnavailableError,
)

POLICY = radius_policy(0.6)


def ref(index: int) -> GeometryReference:
    return GeometryReference(map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=index))


def make_encoder(
    runtime: Any = None, *, policy: Any = POLICY, **config: object
) -> PTv3PointEncoder:
    return PTv3PointEncoder(
        config=make_config(**config),
        support_policy=policy,
        runtime=runtime if runtime is not None else FakePTv3Runtime(),
    )


def prepared_at(center_index: int = 5) -> PreparedSupport:
    return SupportExtractor(LinearScanSource(line_of_points(11)), POLICY).extract(ref(center_index))


# --- Configuration ------------------------------------------------------------


def test_a_valid_configuration_is_json_compatible_and_carries_no_secret() -> None:
    record = make_config().to_dict()

    assert record["checkpoint_hash"] == CHECKPOINT_HASH
    assert all(isinstance(value, str | int | float | bool) for value in record.values())
    assert not {"token", "api_key", "password", "url"} & set(record)


@pytest.mark.parametrize(
    "overrides",
    [
        {"variant": ""},
        {"checkpoint": " "},
        {"checkpoint_hash": "abc"},
        {"checkpoint_hash": "sha256:" + "z" * 64},
        {"device": ""},
        {"precision": "int8"},
        {"grid_size_m": 0.0},
        {"grid_size_m": math.nan},
        {"output_dimension": 0},
        {"pooling": "max"},
        {"normalization": "l3"},
        {"min_support_points": 0},
    ],
)
def test_an_invalid_configuration_is_rejected_before_any_model_is_built(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="PTv3"):
        make_config(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"variant": "sonata-ptv3"},
        {"checkpoint": "Vernata-distilled-v1"},
        {"variant": "PTv3-VERNATA"},
    ],
)
def test_a_pretrained_distilled_encoder_cannot_be_labeled_as_the_ptv3_baseline(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="separately identified"):
        make_config(**overrides)


# --- Identity ----------------------------------------------------------------------


def test_the_space_identifies_the_checkpoint_input_pooling_and_support() -> None:
    space = make_encoder().representation_space()

    assert space.family == "ptv3"
    assert "ptv3-base" in space.model
    assert "center" in space.model
    assert CHECKPOINT_HASH in (space.checkpoint or "")
    assert (space.dimension, space.dtype, space.normalization) == (8, "float32", "none")
    assert "xyz-local-prepared" in space.input_definition
    assert "0.05" in space.input_definition
    assert "float32" in space.input_definition
    assert space.support_semantics == POLICY
    assert space.feature_names == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"variant": "ptv3-small"},
        {"checkpoint": "ptv3-other"},
        {"checkpoint_hash": "sha256:" + "cd" * 32},
        {"precision": "float16"},
        {"grid_size_m": 0.1},
        {"output_dimension": 16},
        {"pooling": "mean"},
        {"normalization": "l2"},
    ],
)
def test_every_setting_that_changes_the_vector_changes_the_space_fingerprint(
    overrides: dict[str, Any],
) -> None:
    base = representation_space_fingerprint(make_encoder().representation_space())

    assert (
        representation_space_fingerprint(make_encoder(**overrides).representation_space()) != base
    )


def test_the_support_policy_changes_the_space_fingerprint() -> None:
    assert representation_space_fingerprint(
        make_encoder(policy=knn_policy(8)).representation_space()
    ) != representation_space_fingerprint(make_encoder().representation_space())


def test_the_device_is_configuration_identity_but_not_space_identity() -> None:
    cuda = make_encoder(device="cuda:0")
    cpu = make_encoder(device="cpu")

    assert representation_space_fingerprint(cuda.representation_space()) == (
        representation_space_fingerprint(cpu.representation_space())
    )
    assert (
        cuda.encoder_identity().configuration_fingerprint
        != cpu.encoder_identity().configuration_fingerprint
    )


def test_the_encoder_identity_carries_the_checkpoint_hash() -> None:
    identity = make_encoder().encoder_identity()

    assert identity.backend_id == "ptv3"
    assert identity.checkpoint_hash == CHECKPOINT_HASH
    assert identity.configuration_fingerprint.startswith("sha256:")
    assert identity == make_encoder().encoder_identity()


def test_the_adapter_satisfies_the_encoder_port() -> None:
    assert isinstance(make_encoder(), PointEncoder)


def test_a_point_support_policy_is_rejected() -> None:
    point_policy = dataclasses.replace(
        POLICY, support_type=SupportType.POINT, method=None, radius_m=None
    )

    with pytest.raises(ValueError, match="neighborhood"):
        make_encoder(policy=point_policy)


# --- Encoding through the runtime boundary ------------------------------------------


def test_the_runtime_receives_the_prepared_coordinates_and_the_center_index() -> None:
    runtime = FakePTv3Runtime()
    prepared = prepared_at()

    encoded = make_encoder(runtime).encode(prepared)

    ((coordinates, center_index, config),) = runtime.calls
    assert coordinates == prepared.local_coordinates_m
    assert center_index == 0
    assert config == make_config()
    assert len(encoded.values) == 8
    assert encoded.undefined_components == ()


def test_the_center_index_follows_the_support_order() -> None:
    runtime = FakePTv3Runtime()
    prepared = prepared_at()
    order = [2, 0, 1, 4, 3]
    reordered = PreparedSupport(
        support=dataclasses.replace(
            prepared.support,
            geometry_refs=tuple(prepared.support.geometry_refs[i] for i in order),
        ),
        local_coordinates_m=tuple(prepared.local_coordinates_m[i] for i in order),
    )

    make_encoder(runtime).encode(reordered)

    assert runtime.calls[0][1] == order.index(0)


def test_l2_normalization_is_applied_outside_the_runtime() -> None:
    encoded = make_encoder(normalization="l2").encode(prepared_at())

    assert math.hypot(*encoded.values) == pytest.approx(1.0)


class ZeroRuntime(FakePTv3Runtime):
    def infer(self, *, coordinates_m: Any, center_index: int, config: PTv3Config) -> PTv3Inference:
        return PTv3Inference(vector=(0.0,) * config.output_dimension)


def test_a_zero_vector_cannot_be_normalized_and_is_not_invented() -> None:
    with pytest.raises(UnencodableSupportError, match="zero"):
        make_encoder(ZeroRuntime(), normalization="l2").encode(prepared_at())


def test_a_support_below_the_minimum_never_reaches_the_runtime() -> None:
    runtime = FakePTv3Runtime()
    tiny = SupportExtractor(LinearScanSource(line_of_points(11)), radius_policy(0.2)).extract(
        ref(5)
    )

    with pytest.raises(UnencodableSupportError, match="at least 3"):
        make_encoder(runtime, policy=radius_policy(0.2)).encode(tiny)

    assert runtime.calls == []


def test_a_support_prepared_under_another_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="policy"):
        make_encoder(policy=radius_policy(2.0)).encode(prepared_at())


class WrongDimensionRuntime(FakePTv3Runtime):
    def infer(self, *, coordinates_m: Any, center_index: int, config: PTv3Config) -> PTv3Inference:
        return PTv3Inference(vector=(1.0, 2.0))


def test_a_vector_of_the_wrong_shape_is_an_explicit_failure() -> None:
    with pytest.raises(ValueError, match="dimension"):
        make_encoder(WrongDimensionRuntime()).encode(prepared_at())


class OutOfMemoryRuntime(FakePTv3Runtime):
    def infer(self, *, coordinates_m: Any, center_index: int, config: PTv3Config) -> PTv3Inference:
        if len(coordinates_m) > 3:
            raise PTv3OutOfMemoryError("CUDA out of memory (tried to allocate 2.00 GiB)")
        return super().infer(coordinates_m=coordinates_m, center_index=center_index, config=config)


def test_running_out_of_device_memory_is_a_failed_support_never_a_vector() -> None:
    encoder = make_encoder(OutOfMemoryRuntime())
    service = RepresentationService(
        LinearScanSource(line_of_points(11)),
        encoder,
        run_id=PointRepresentationRunId("run-0001"),
        code_version="test",
    )

    outcomes = list(service.represent([ref(5), ref(0)]))

    failed, ok = outcomes
    assert isinstance(failed, FailedSupport)
    assert failed.reason is FailureReason.UNENCODABLE_SUPPORT
    assert "out of memory" in failed.detail
    assert isinstance(ok, EncodedRepresentation)
    assert encoder.telemetry.out_of_memory == 1
    assert encoder.telemetry.encoded == 1


class NonFiniteRuntime(FakePTv3Runtime):
    def infer(self, *, coordinates_m: Any, center_index: int, config: PTv3Config) -> PTv3Inference:
        return PTv3Inference(vector=(1.0, math.nan) + (0.0,) * (config.output_dimension - 2))


def test_a_non_finite_result_is_reported_by_the_service_and_not_replaced() -> None:
    service = RepresentationService(
        LinearScanSource(line_of_points(11)),
        make_encoder(NonFiniteRuntime()),
        run_id=PointRepresentationRunId("run-0001"),
        code_version="test",
    )

    (outcome,) = service.represent([ref(5)])

    assert isinstance(outcome, FailedSupport)
    assert outcome.reason is FailureReason.NON_FINITE_OUTPUT


class UnavailableRuntime(FakePTv3Runtime):
    def infer(self, *, coordinates_m: Any, center_index: int, config: PTv3Config) -> PTv3Inference:
        raise PTv3RuntimeUnavailableError("torch is not installed")


def test_a_missing_runtime_stops_the_run_with_no_silent_fallback() -> None:
    service = RepresentationService(
        LinearScanSource(line_of_points(11)),
        make_encoder(UnavailableRuntime()),
        run_id=PointRepresentationRunId("run-0001"),
        code_version="test",
    )

    with pytest.raises(PTv3RuntimeUnavailableError, match="torch"):
        list(service.represent([ref(5)]))


# --- Telemetry --------------------------------------------------------------------


def test_runtime_and_peak_device_memory_are_recorded() -> None:
    class Growing(FakePTv3Runtime):
        def __init__(self) -> None:
            super().__init__()
            self._peaks = iter([100, 900, 300])

        def infer(
            self, *, coordinates_m: Any, center_index: int, config: PTv3Config
        ) -> PTv3Inference:
            base = super().infer(
                coordinates_m=coordinates_m, center_index=center_index, config=config
            )
            return PTv3Inference(vector=base.vector, peak_memory_bytes=next(self._peaks))

    encoder = make_encoder(Growing())

    for _ in range(3):
        encoder.encode(prepared_at())

    assert encoder.telemetry.encoded == 3
    assert encoder.telemetry.peak_memory_bytes == 900
    assert encoder.telemetry.inference_seconds >= 0.0


def test_an_unmeasurable_peak_stays_unknown_instead_of_zero() -> None:
    encoder = make_encoder(FakePTv3Runtime(peak_memory_bytes=None))

    encoder.encode(prepared_at())

    assert encoder.telemetry.peak_memory_bytes is None


# --- Selection through composition ---------------------------------------------------


def test_ptv3_and_the_baseline_flow_through_the_same_service_and_never_mix() -> None:
    source = LinearScanSource(line_of_points(11))
    run_id = PointRepresentationRunId("run-0001")
    learned_service = RepresentationService(
        source, make_encoder(), run_id=run_id, code_version="test"
    )
    baseline_service = RepresentationService(
        source, FakeEncoder(POLICY), run_id=run_id, code_version="test"
    )

    (learned,) = learned_service.represent([ref(5)])
    (baseline,) = baseline_service.represent([ref(5)])

    assert isinstance(learned, EncodedRepresentation)
    assert isinstance(baseline, EncodedRepresentation)
    assert learned.representation.encoder_identity.backend_id == "ptv3"
    assert learned.representation.shape == (8,)
    with pytest.raises(RepresentationSpaceMismatchError):
        ensure_compatible_representations(learned.representation, baseline.representation)


def test_the_backend_is_optional_and_needs_no_model_library_to_import() -> None:
    code = (
        "import sys, contextmap.point_representation;"
        "assert 'contextmap.point_representation.backends.ptv3' not in sys.modules;"
        "import contextmap.point_representation.backends.ptv3;"
        "bad = [m for m in ('numpy', 'torch', 'transformers') if m in sys.modules];"
        "assert not bad, bad"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr

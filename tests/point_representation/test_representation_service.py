import math
from collections.abc import Sequence

import pytest
from pointrep_builders import knn_policy, radius_policy
from pointrep_fakes import FakeEncoder
from pointrep_geometry import MAP_ID, LinearScanSource, line_of_points

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.point_representation import (
    EncodedRepresentation,
    EncodedVector,
    FailedSupport,
    FailureReason,
    PointEncoder,
    PointRepresentationRunId,
    PreparedSupport,
    RepresentationService,
    RepresentationSpaceMismatchError,
    UnencodableSupportError,
    ensure_compatible_representations,
    representation_id_for,
    representation_space_fingerprint,
)
from contextmap.point_representation.support import SupportExtractor

RUN_ID = PointRepresentationRunId("run-0001")
Outcome = EncodedRepresentation | FailedSupport


def ref(index: int, *, map_id: MapId = MAP_ID) -> GeometryReference:
    return GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))


def make_service(
    encoder: PointEncoder | None = None, *, source: LinearScanSource | None = None
) -> RepresentationService:
    return RepresentationService(
        source if source is not None else LinearScanSource(line_of_points(11)),
        encoder if encoder is not None else FakeEncoder(),
        run_id=RUN_ID,
        code_version="test-version",
    )


def run(service: RepresentationService, centers: Sequence[GeometryReference]) -> list[Outcome]:
    return list(service.represent(centers))


def represented(outcomes: Sequence[Outcome]) -> list[EncodedRepresentation]:
    return [outcome for outcome in outcomes if isinstance(outcome, EncodedRepresentation)]


# --- The port and its vector ---------------------------------------------------


def test_an_encoder_is_recognized_by_its_capability_not_by_inheritance() -> None:
    assert isinstance(FakeEncoder(), PointEncoder)
    assert not isinstance(object(), PointEncoder)


def test_an_encoded_vector_needs_values() -> None:
    with pytest.raises(ValueError, match="values"):
        EncodedVector(values=())


def test_a_failed_support_names_its_reason() -> None:
    support = (
        SupportExtractor(LinearScanSource(line_of_points(3)), radius_policy(0.6))
        .extract(ref(1))
        .support
    )

    assert FailedSupport(
        support=support, reason=FailureReason.UNENCODABLE_SUPPORT, detail="too few points"
    ).support.center == ref(1)
    with pytest.raises(ValueError, match="detail"):
        FailedSupport(support=support, reason=FailureReason.UNENCODABLE_SUPPORT, detail="")


# --- Assembling canonical representations --------------------------------------


def test_the_service_publishes_canonical_representations_in_request_order() -> None:
    encoder = FakeEncoder()
    outcomes = run(make_service(encoder), [ref(5), ref(0)])

    first, second = represented(outcomes)
    space = encoder.representation_space()
    assert first.representation.representation_id == representation_id_for(run_id=RUN_ID, index=0)
    assert second.representation.representation_id == representation_id_for(run_id=RUN_ID, index=1)
    assert first.representation.geometry_reference == ref(5)
    assert first.representation.support.geometry_refs == (ref(5), ref(4), ref(6), ref(3), ref(7))
    assert first.values == (5.0, 0.0, 0.0)
    assert second.values == (3.0, 0.25, 0.0)
    for outcome in (first, second):
        assert outcome.representation.representation_space_id == representation_space_fingerprint(
            space
        )
        assert outcome.representation.shape == (3,)
        assert outcome.representation.dtype == "float32"
        assert outcome.representation.normalization == "none"
        assert outcome.representation.encoder_identity == encoder.encoder_identity()
        assert outcome.representation.provenance.code_version == "test-version"
        assert outcome.representation.payload_reference is None
        assert not outcome.representation.is_partial


def test_the_support_policy_comes_from_the_encoder_representation_space() -> None:
    outcomes = run(make_service(FakeEncoder(knn_policy(3))), [ref(5)])

    (only,) = represented(outcomes)
    assert only.representation.support.policy == knn_policy(3)
    assert only.representation.support.geometry_refs == (ref(5), ref(4), ref(6))


def test_the_encoder_receives_the_prepared_support_and_nothing_else() -> None:
    encoder = FakeEncoder()

    run(make_service(encoder), [ref(5)])

    (received,) = encoder.calls
    assert received.support.center == ref(5)
    assert received.local_coordinates_m[0] == (0.0, 0.0, 0.0)


def test_the_same_request_gives_identical_results() -> None:
    service = make_service()

    assert run(service, [ref(5), ref(0), ref(9)]) == run(service, [ref(5), ref(0), ref(9)])


def test_published_values_are_plain_floats() -> None:
    (only,) = represented(run(make_service(), [ref(5)]))

    assert isinstance(only.values, tuple)
    assert all(type(value) is float for value in only.values)


def test_representations_of_two_encoders_flow_through_the_same_service_but_never_mix() -> None:
    descriptor = FakeEncoder(family="geometric_descriptor")
    learned = FakeEncoder(family="ptv3", checkpoint="sha256:weights")

    (from_descriptor,) = represented(run(make_service(descriptor), [ref(5)]))
    (from_learned,) = represented(run(make_service(learned), [ref(5)]))

    assert type(from_descriptor) is type(from_learned)
    with pytest.raises(RepresentationSpaceMismatchError):
        ensure_compatible_representations(
            from_descriptor.representation, from_learned.representation
        )


# --- Validation before any backend work ---------------------------------------


def test_a_center_of_another_map_is_rejected_before_the_encoder_runs() -> None:
    encoder = FakeEncoder()
    service = make_service(encoder)

    with pytest.raises(ValueError, match="map"):
        service.represent([ref(5), ref(1, map_id=MapId("another-map"))])

    assert encoder.calls == []


def test_a_repeated_center_is_rejected_before_the_encoder_runs() -> None:
    encoder = FakeEncoder()
    service = make_service(encoder)

    with pytest.raises(ValueError, match="twice"):
        service.represent([ref(5), ref(0), ref(5)])

    assert encoder.calls == []


def test_a_center_missing_from_the_map_is_a_caller_error_not_a_failed_support() -> None:
    with pytest.raises(KeyError):
        run(make_service(), [ref(99)])


def test_an_empty_request_produces_nothing() -> None:
    service = make_service()

    assert run(service, []) == []
    assert service.metrics.requested == 0


# --- Explicit failures ----------------------------------------------------------


class RejectingSmallSupports(FakeEncoder):
    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        if len(prepared.support.geometry_refs) < 5:
            raise UnencodableSupportError("fewer than 5 supporting points")
        return super().encode(prepared)


def test_an_unencodable_support_is_reported_and_the_run_continues() -> None:
    outcomes = run(make_service(RejectingSmallSupports()), [ref(0), ref(5), ref(10)])

    failed_first, ok, failed_last = outcomes
    assert isinstance(failed_first, FailedSupport)
    assert failed_first.reason is FailureReason.UNENCODABLE_SUPPORT
    assert failed_first.detail == "fewer than 5 supporting points"
    assert failed_first.support.center == ref(0)
    assert isinstance(ok, EncodedRepresentation)
    assert isinstance(failed_last, FailedSupport)
    # Um suporte falho deixa uma lacuna no índice em vez de reaproveitar a identidade.
    assert ok.representation.representation_id == representation_id_for(run_id=RUN_ID, index=1)


class NonFiniteEncoder(FakeEncoder):
    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        return EncodedVector(values=(3.0, math.nan, math.inf))


def test_a_non_finite_result_is_a_failed_support_never_a_default_vector() -> None:
    (outcome,) = run(make_service(NonFiniteEncoder()), [ref(5)])

    assert isinstance(outcome, FailedSupport)
    assert outcome.reason is FailureReason.NON_FINITE_OUTPUT


class UndefinedComponentEncoder(FakeEncoder):
    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        return EncodedVector(values=(3.0, math.nan, 0.5), undefined_components=(1,))


def test_undefined_components_are_declared_and_stored_as_placeholders() -> None:
    (outcome,) = represented(run(make_service(UndefinedComponentEncoder()), [ref(5)]))

    assert outcome.representation.undefined_components == (1,)
    assert outcome.representation.is_partial
    assert outcome.values == (3.0, 0.0, 0.5)


class WrongDimensionEncoder(FakeEncoder):
    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        return EncodedVector(values=(1.0, 2.0))


def test_an_encoder_breaking_its_declared_dimension_fails_the_run() -> None:
    with pytest.raises(ValueError, match="dimension"):
        run(make_service(WrongDimensionEncoder()), [ref(5)])


class OutOfRangeUndefinedEncoder(FakeEncoder):
    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        return EncodedVector(values=(1.0, 2.0, 3.0), undefined_components=(7,))


def test_undefined_components_outside_the_dimension_fail_the_run() -> None:
    with pytest.raises(ValueError, match="undefined_components"):
        run(make_service(OutOfRangeUndefinedEncoder()), [ref(5)])


class CrashingEncoder(FakeEncoder):
    def encode(self, prepared: PreparedSupport) -> EncodedVector:
        if self.calls:
            raise RuntimeError("CUDA out of memory")
        return super().encode(prepared)


def test_a_backend_crash_is_never_turned_into_a_representation() -> None:
    stream = iter(make_service(CrashingEncoder()).represent([ref(5), ref(0), ref(2)]))

    assert isinstance(next(stream), EncodedRepresentation)
    with pytest.raises(RuntimeError, match="out of memory"):
        next(stream)


# --- Metrics ----------------------------------------------------------------------


def test_the_run_reports_counts_failures_and_timings() -> None:
    service = make_service(RejectingSmallSupports())

    run(service, [ref(0), ref(5), ref(10), ref(4)])

    metrics = service.metrics
    assert (metrics.requested, metrics.represented, metrics.partial, metrics.failed) == (4, 2, 0, 2)
    assert dict(metrics.failed_by_reason) == {FailureReason.UNENCODABLE_SUPPORT: 2}
    assert metrics.support_extraction_seconds >= 0.0
    assert metrics.encoding_seconds >= 0.0


def test_partial_representations_are_counted() -> None:
    service = make_service(UndefinedComponentEncoder())

    run(service, [ref(5), ref(4)])

    assert service.metrics.represented == 2
    assert service.metrics.partial == 2

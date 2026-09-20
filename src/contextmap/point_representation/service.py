"""Backend-agnostic execution of Point Representation.

:class:`RepresentationService` resolves each requested center into a prepared
support, hands it to one selected :class:`PointEncoder`, and assembles the
canonical :class:`PointRepresentation`. It knows no concrete backend: which
encoder runs is decided by whoever constructs the service, and there is no
fallback from one encoder to another when the selected one fails.

Failure is explicit. A support the encoder declares unencodable, or one whose
result is not finite, becomes a :class:`FailedSupport`; a contract violation or
a backend crash stops the run. Nothing is ever replaced by a zero or default
vector.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

from contextmap.geometric_mapping import GeometryReference, GeometrySource
from contextmap.point_representation.compatibility import representation_space_fingerprint
from contextmap.point_representation.models import (
    FailedSupport,
    FailureReason,
    PointRepresentation,
    PointRepresentationRunId,
    PointSupport,
    RepresentationProvenance,
    representation_id_for,
)
from contextmap.point_representation.ports import (
    EncodedVector,
    PointEncoder,
    UnencodableSupportError,
)
from contextmap.point_representation.support import SupportExtractor


@dataclass(frozen=True, kw_only=True)
class EncodedRepresentation:
    """A representation together with its vector, before the vector is persisted.

    Attributes:
        representation: The canonical metadata; ``payload_reference`` is
            ``None`` until a writer stores the vector and says where.
        values: The vector as plain floats; components listed in
            ``representation.undefined_components`` are zero placeholders.
    """

    representation: PointRepresentation
    values: tuple[float, ...]


@dataclass(frozen=True, kw_only=True)
class RepresentationMetrics:
    """Counts and timings of one :meth:`RepresentationService.represent` call.

    Attributes:
        requested: Centers asked for.
        represented: Representations produced, partial ones included.
        partial: Representations with undefined components.
        failed_by_reason: Failed supports per reason.
        support_extraction_seconds: Wall time spent extracting supports.
        encoding_seconds: Wall time spent inside the encoder.
    """

    requested: int
    represented: int
    partial: int
    failed_by_reason: Mapping[FailureReason, int]
    support_extraction_seconds: float
    encoding_seconds: float

    @property
    def failed(self) -> int:
        """Total failed supports."""
        return sum(self.failed_by_reason.values())


class RepresentationService:
    """Produces representations of geometry elements with one selected encoder."""

    def __init__(
        self,
        source: GeometrySource,
        encoder: PointEncoder,
        *,
        run_id: PointRepresentationRunId,
        code_version: str,
    ) -> None:
        """Bind the service to one map, one encoder and one run identity.

        Args:
            source: The persistent geometry to represent.
            encoder: The selected encoder. Its representation space decides the
                support policy: supports are extracted under
                ``encoder.representation_space().support_semantics``.
            run_id: Identity of the run that owns the produced representations.
            code_version: Identity of the code producing them.

        Raises:
            ValueError: If ``code_version`` is empty.
        """
        self._encoder = encoder
        self._run_id = run_id
        self._space = encoder.representation_space()
        self._space_id = representation_space_fingerprint(self._space)
        self._identity = encoder.encoder_identity()
        self._provenance = RepresentationProvenance(code_version=code_version)
        self._extractor = SupportExtractor(source, self._space.support_semantics)
        self._map_id = source.geometric_map.map_id
        self._reset_metrics(requested=0)

    @property
    def metrics(self) -> RepresentationMetrics:
        """Counts and timings of the latest :meth:`represent` call.

        Complete once its iterator is exhausted; partial if the run stopped early.
        """
        return RepresentationMetrics(
            requested=self._requested,
            represented=self._represented,
            partial=self._partial,
            failed_by_reason=dict(self._failed_by_reason),
            support_extraction_seconds=self._support_seconds,
            encoding_seconds=self._encoding_seconds,
        )

    def represent(
        self, centers: Sequence[GeometryReference]
    ) -> Iterator[EncodedRepresentation | FailedSupport]:
        """Represent each center, streaming one outcome per center.

        The request is validated before any support is extracted or encoder
        called; the work itself runs lazily as the iterator is consumed.
        Outcomes follow the request order, and a representation's identity uses
        its position in ``centers``, so a failed support leaves a gap instead
        of shifting the identities after it.

        Args:
            centers: The geometry elements to represent, each at most once.

        Returns:
            For each center, an :class:`EncodedRepresentation`, or a
            :class:`FailedSupport` when the encoder cannot represent its support.

        Raises:
            ValueError: Immediately, if a center belongs to another map or is
                requested twice; while iterating, if the encoder returns the
                wrong number of values or invalid undefined components.
            KeyError: While iterating, if a center is not in the map.
        """
        requested = tuple(centers)
        self._require_valid_request(requested)
        self._reset_metrics(requested=len(requested))
        return self._stream(requested)

    def _stream(
        self, centers: Sequence[GeometryReference]
    ) -> Iterator[EncodedRepresentation | FailedSupport]:
        for index, center in enumerate(centers):
            started = perf_counter()
            prepared = self._extractor.extract(center)
            self._support_seconds += perf_counter() - started
            started = perf_counter()
            try:
                encoded = self._encoder.encode(prepared)
            except UnencodableSupportError as error:
                self._encoding_seconds += perf_counter() - started
                yield self._fail(
                    prepared.support,
                    FailureReason.UNENCODABLE_SUPPORT,
                    str(error) or "the encoder rejected the support without giving a reason",
                )
                continue
            self._encoding_seconds += perf_counter() - started
            yield self._assemble(index, prepared.support, encoded)

    def _assemble(
        self, index: int, support: PointSupport, encoded: EncodedVector
    ) -> EncodedRepresentation | FailedSupport:
        values = tuple(float(value) for value in encoded.values)
        if len(values) != self._space.dimension:
            raise ValueError(
                f"encoder {self._identity.backend_id!r} returned {len(values)} values for a "
                f"{self._space.dimension}-dimensional representation space"
            )
        undefined = set(encoded.undefined_components)
        if any(
            not math.isfinite(value)
            for position, value in enumerate(values)
            if position not in undefined
        ):
            return self._fail(
                support,
                FailureReason.NON_FINITE_OUTPUT,
                "the encoder returned a non-finite value in a component it did not declare "
                "undefined",
            )
        representation = PointRepresentation(
            representation_id=representation_id_for(run_id=self._run_id, index=index),
            geometry_reference=support.center,
            support=support,
            representation_space_id=self._space_id,
            shape=(self._space.dimension,),
            dtype=self._space.dtype,
            normalization=self._space.normalization,
            payload_reference=None,
            encoder_identity=self._identity,
            provenance=self._provenance,
            undefined_components=encoded.undefined_components,
        )
        self._represented += 1
        if representation.is_partial:
            self._partial += 1
        # Componentes indefinidos guardam um placeholder nulo, nunca o valor do encoder.
        stored = tuple(
            0.0 if position in undefined else value for position, value in enumerate(values)
        )
        return EncodedRepresentation(representation=representation, values=stored)

    def _fail(self, support: PointSupport, reason: FailureReason, detail: str) -> FailedSupport:
        self._failed_by_reason[reason] = self._failed_by_reason.get(reason, 0) + 1
        return FailedSupport(support=support, reason=reason, detail=detail)

    def _require_valid_request(self, centers: Sequence[GeometryReference]) -> None:
        seen: set[GeometryReference] = set()
        for center in centers:
            if center.map_id != self._map_id:
                raise ValueError(
                    f"center {center.geometry_id!r} belongs to map {center.map_id!r}, but this "
                    f"service represents map {self._map_id!r}"
                )
            if center in seen:
                raise ValueError(f"center {center.geometry_id!r} is requested twice")
            seen.add(center)

    def _reset_metrics(self, *, requested: int) -> None:
        self._requested = requested
        self._represented = 0
        self._partial = 0
        self._failed_by_reason: dict[FailureReason, int] = {}
        self._support_seconds = 0.0
        self._encoding_seconds = 0.0

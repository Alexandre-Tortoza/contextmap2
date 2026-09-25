"""Prompt-conditioned region grounding: explicit query requests and auditable executions.

Automatic Region Discovery (:class:`~contextmap.visual_perception.ports.RegionDiscovery`)
proposes regions from an image alone. Grounding answers an explicit language query about
one prepared image, so the query is a scientific inference input, never adapter state: it
travels inside a :class:`RegionGroundingRequest`, takes part in the request identity, is
validated against the backend's declared :class:`RegionGroundingCapabilities` before any
model runs, and is persisted with every execution.

A grounding execution keeps four things apart: the request, the exact prompt the backend
rendered from it, the verbatim raw response, and the parsed outputs. Only box outputs
become canonical :class:`~contextmap.visual_perception.models.Region2D` evidence; a point
output stays a point, because inventing a box around it would fabricate geometry the model
never produced. A label the backend associates with an output (for example the category it
was asked for) is kept verbatim as part of the evidence and never becomes a semantic claim,
an entity, or a confidence. See ``docs/region-grounding.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, NewType, cast

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.identity import grounding_region_id_for
from contextmap.visual_perception.models import (
    BackendProvenance,
    BoundingBox2D,
    PerceptionResult,
    PerceptionResultId,
    PreparedImage,
    Region2D,
    TransformationRecord,
)
from contextmap.visual_perception.region_models import ArtifactReference, JsonScalar
from contextmap.visual_perception.serialization import (
    decode_bounding_box,
    decode_provenance,
    encode_bounding_box,
    encode_provenance,
)

GroundingRequestId = NewType("GroundingRequestId", str)
"""Content identity of one grounding request: a digest of its inference inputs."""

GROUNDING_CAPABILITY = "region_grounding"
"""Capability name every grounding backend reports in its ``BackendProvenance``."""

_EDGE_TOLERANCE_PX = 1e-6
"""Tolerância, em pixels, para a borda direita/inferior de uma caixa convertida.

Coordenadas normalizadas viram pixels por multiplicação em ponto flutuante, e
``x + width`` pode exceder a largura da imagem por um ulp quando a caixa toca a borda.
Um micro-pixel absorve esse arredondamento sem aceitar uma caixa realmente fora da imagem.
"""


class GroundingTask(Enum):
    """What a grounding query asks the backend to localize.

    ``CATEGORY_DETECTION`` asks for every instance of each category in an ordered set;
    ``PHRASE_GROUNDING`` asks for whatever a free-form phrase refers to.
    """

    CATEGORY_DETECTION = "category_detection"
    PHRASE_GROUNDING = "phrase_grounding"


class GroundingGeometry(Enum):
    """Geometry family a grounding query requests and a backend output carries."""

    BOX = "box"
    POINT = "point"


class GroundingRejectionReason(Enum):
    """Why one span of a raw grounding response did not become an output."""

    MALFORMED_GEOMETRY = "malformed_geometry"
    COORDINATE_OUT_OF_RANGE = "coordinate_out_of_range"
    INVERTED_GEOMETRY = "inverted_geometry"
    UNRECOGNIZED_TEXT = "unrecognized_text"


class GroundingRequestError(ValueError):
    """Raised when a grounding request cannot be served by the selected backend.

    It is raised before any model is loaded or run, so a caller can count request
    validation failures without confusing them with inference failures.
    """


@dataclass(frozen=True, kw_only=True)
class GroundingQuery:
    """What one grounding request asks, and under which versioned policy.

    Attributes:
        task: What is being localized.
        policy_id: Versioned identity of the rule by which the backend turns this query
            into model input (for example ``"locateanything.category-detection/1"``).
            Changing how a query is rendered requires a new identity.
        geometry: Geometry family requested.
        text: Free-form phrase; required by ``PHRASE_GROUNDING`` only. Kept verbatim.
        categories: Ordered category set; required by ``CATEGORY_DETECTION`` only. The
            order is part of the query: it is the order the backend receives.
    """

    task: GroundingTask
    policy_id: str
    geometry: GroundingGeometry
    text: str | None = None
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require the query shape its task defines, with no blank or repeated term."""
        if not self.policy_id.strip():
            raise ValueError("grounding query policy_id must not be empty")
        if self.task is GroundingTask.CATEGORY_DETECTION:
            if not self.categories:
                raise ValueError("category detection requires at least one category")
            if self.text is not None:
                raise ValueError("category detection takes categories, not free-form text")
            if any(not category.strip() for category in self.categories):
                raise ValueError("grounding categories must not be blank")
            if len(set(self.categories)) != len(self.categories):
                raise ValueError("grounding categories must be unique")
        else:
            if self.text is None or not self.text.strip():
                raise ValueError("phrase grounding requires non-blank text")
            if self.categories:
                raise ValueError("phrase grounding takes free-form text, not categories")


@dataclass(frozen=True, kw_only=True)
class GroundingQuerySet:
    """The ordered queries a run asks about every image it grounds.

    Each (image, query) pair becomes one request, so a repeated query would ask the same
    thing twice and produce two requests with one identity; it is refused instead.
    """

    queries: tuple[GroundingQuery, ...]

    def __post_init__(self) -> None:
        """Require at least one query and no repeated query."""
        if not self.queries:
            raise ValueError("a grounding query set needs at least one query")
        if len(set(self.queries)) != len(self.queries):
            raise ValueError("grounding queries must be unique within a query set")


@dataclass(frozen=True, kw_only=True)
class GroundingQueryPolicy:
    """One query policy a grounding backend implements: which task, which geometry."""

    policy_id: str
    task: GroundingTask
    geometry: GroundingGeometry

    def __post_init__(self) -> None:
        """Require a versioned policy identity."""
        if not self.policy_id.strip():
            raise ValueError("grounding query policy_id must not be empty")


@dataclass(frozen=True, kw_only=True)
class RegionGroundingCapabilities:
    """Declare the query policies a grounding backend can serve."""

    query_policies: tuple[GroundingQueryPolicy, ...]

    def __post_init__(self) -> None:
        """Require at least one policy and unique policy identities."""
        if not self.query_policies:
            raise ValueError("a grounding backend must declare at least one query policy")
        identities = [policy.policy_id for policy in self.query_policies]
        if len(set(identities)) != len(identities):
            raise ValueError("grounding query policy identities must be unique")


def validate_grounding_query(
    query: GroundingQuery, capabilities: RegionGroundingCapabilities
) -> None:
    """Check that a backend declares the exact policy, task, and geometry a query asks for.

    Args:
        query: The query to serve.
        capabilities: The backend's declaration.

    Raises:
        GroundingRequestError: If the policy is unknown to the backend, or declared for
            another task or geometry. Nothing substitutes another policy or geometry.
    """
    policies = {policy.policy_id: policy for policy in capabilities.query_policies}
    policy = policies.get(query.policy_id)
    if policy is None:
        supported_geometries = {item.geometry for item in capabilities.query_policies}
        if query.geometry not in supported_geometries:
            raise GroundingRequestError(
                f"grounding backend supports no {query.geometry.value} geometry; "
                f"supported query policies: {sorted(policies)}"
            )
        raise GroundingRequestError(
            f"grounding backend does not support query policy {query.policy_id!r}; "
            f"supported query policies: {sorted(policies)}"
        )
    if policy.task is not query.task:
        raise GroundingRequestError(
            f"query policy {policy.policy_id!r} serves task {policy.task.value}, "
            f"not {query.task.value}"
        )
    if policy.geometry is not query.geometry:
        raise GroundingRequestError(
            f"query policy {policy.policy_id!r} produces {policy.geometry.value} geometry, "
            f"not the requested {query.geometry.value} geometry"
        )


@dataclass(frozen=True, kw_only=True)
class RegionGroundingRequest:
    """One explicit, reproducible grounding request over one prepared image.

    Attributes:
        perception_result_id: Result that owns the evidence this request produces.
        image: The exact prepared image the query is evaluated on. It must be
            content-addressed (``payload_artifact``) because its hash is part of the
            request identity, and it must carry no valid/exclusion constraint, since
            grounding does not apply them yet and silently ignoring one would be wrong.
        query: What is asked, under which versioned policy and geometry.
        configuration_fingerprint: Fingerprint of the backend configuration the request
            is issued to; a backend refuses a request fingerprinted for another one.
    """

    perception_result_id: PerceptionResultId
    image: PreparedImage
    query: GroundingQuery
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        """Reject an unowned, unaddressed, or constrained request."""
        if not str(self.perception_result_id).strip():
            raise ValueError("grounding request perception_result_id must not be empty")
        if not self.configuration_fingerprint.strip():
            raise ValueError("grounding request configuration_fingerprint must not be empty")
        if self.image.payload_artifact is None:
            raise ValueError(
                "grounding request image needs a content-addressed payload_artifact: its "
                "hash is part of the request identity"
            )
        if self.image.valid_region is not None or self.image.exclusion_regions:
            raise ValueError(
                "grounding does not apply prepared-image valid/exclusion constraints yet; "
                "refusing a constrained image instead of ignoring the constraint"
            )

    @property
    def source_observation_id(self) -> SourceObservationId:
        """Return the physical observation the request is about."""
        return self.image.source_observation_id

    @property
    def request_id(self) -> GroundingRequestId:
        """Return the content identity of this request's inference inputs.

        It digests the observation, the prepared image content and dimensions, the full
        query (task, policy, geometry, text, ordered categories) and the configuration
        fingerprint. It deliberately excludes ``perception_result_id``: the same inputs
        have the same identity in any run, while the evidence they produce stays local to
        its result (see :func:`~contextmap.visual_perception.identity.grounding_region_id_for`).
        """
        artifact = cast(ArtifactReference, self.image.payload_artifact)
        document = {
            "source_observation_id": str(self.source_observation_id),
            "image_sha256": artifact.sha256,
            "image_width": self.image.width,
            "image_height": self.image.height,
            "query": _encode_query(self.query),
            "configuration_fingerprint": self.configuration_fingerprint,
        }
        canonical = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return GroundingRequestId(f"grounding-{digest}")


@dataclass(frozen=True, kw_only=True)
class GroundingPoint:
    """A point in prepared-image pixel coordinates (``PIXEL_XY_TOP_LEFT``)."""

    x: float
    y: float

    def __post_init__(self) -> None:
        """Reject non-finite and negative coordinates."""
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("grounding point coordinates must be finite")
        if self.x < 0 or self.y < 0:
            raise ValueError("grounding point coordinates must be non-negative")


@dataclass(frozen=True, kw_only=True)
class GroundingOutput:
    """One parsed geometry output of a grounding backend, in prepared-image pixels.

    Attributes:
        output_index: Position of this output in the parsed response, shared with
            :class:`RejectedGroundingOutput` so the response order is preserved.
        native_text: The exact response span that produced this output.
        label: Query term the backend associated with the output, verbatim, or ``None``.
            It is evidence of what the backend answered, never a semantic claim.
        box: Pixel-space box, for box outputs.
        point: Pixel-space point, for point outputs.
        native_diagnostics: Backend-native per-output diagnostics, verbatim (for example
            which decoder produced it). They are never a confidence.
    """

    output_index: int
    native_text: str
    label: str | None = None
    box: BoundingBox2D | None = None
    point: GroundingPoint | None = None
    native_diagnostics: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Require one geometry, an ordered position, and named native diagnostics."""
        if self.output_index < 0:
            raise ValueError("grounding output_index must be non-negative")
        if (self.box is None) == (self.point is None):
            raise ValueError("a grounding output carries exactly one geometry: box or point")
        _validate_named_scalars(self.native_diagnostics, "grounding output native diagnostic")

    @property
    def geometry(self) -> GroundingGeometry:
        """Return the geometry family this output carries."""
        return GroundingGeometry.BOX if self.box is not None else GroundingGeometry.POINT


@dataclass(frozen=True, kw_only=True)
class RejectedGroundingOutput:
    """A span of the raw response that was explicitly not accepted as an output."""

    output_index: int
    native_text: str
    reason: GroundingRejectionReason
    detail: str
    label: str | None = None

    def __post_init__(self) -> None:
        """Require an ordered position and a human-readable reason."""
        if self.output_index < 0:
            raise ValueError("rejected grounding output_index must be non-negative")
        if not self.detail.strip():
            raise ValueError("rejected grounding output detail must not be empty")


@dataclass(frozen=True, kw_only=True)
class GroundingDiagnostics:
    """Runtime diagnostics of one grounding call.

    Attributes:
        latency_ms: Wall-clock duration of the backend call.
        peak_memory_bytes: Peak accelerator memory during the call, when measured.
        warnings: Human-readable warnings, for example a possibly truncated response.
        native: Backend-native decoder/runtime diagnostics, with their native names and
            values kept verbatim. A missing value stays missing; none is a confidence.
    """

    latency_ms: float
    peak_memory_bytes: int | None = None
    warnings: tuple[str, ...] = ()
    native: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Reject negative measurements and unnamed native diagnostics."""
        if not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("grounding latency_ms must be finite and non-negative")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("grounding peak_memory_bytes must be non-negative")
        _validate_named_scalars(self.native, "grounding native diagnostic")


@dataclass(frozen=True, kw_only=True)
class RegionGroundingExecution:
    """Auditable record of one grounding call: request, prompt, raw text, parsed outputs.

    Attributes:
        request: The request that was served.
        provenance: The grounding backend that served it.
        rendered_prompt: The exact model input text the backend rendered from the query.
        raw_response: The backend's verbatim response.
        outputs: Accepted outputs, in response order.
        rejected_outputs: Response spans explicitly rejected, with their reason.
        no_match_labels: Query terms for which the backend explicitly reported no
            instance (``None`` for an unlabelled no-match marker).
        diagnostics: Latency, memory, warnings and native decoder diagnostics.
        effective_configuration: The backend configuration in effect.
        runtime_identity: Software/hardware identity of the runtime that executed the
            call (library versions, device), as reported by the backend; empty if unknown.
    """

    request: RegionGroundingRequest
    provenance: BackendProvenance
    rendered_prompt: str
    raw_response: str
    outputs: tuple[GroundingOutput, ...]
    diagnostics: GroundingDiagnostics
    effective_configuration: Mapping[str, JsonScalar]
    rejected_outputs: tuple[RejectedGroundingOutput, ...] = ()
    no_match_labels: tuple[str | None, ...] = ()
    runtime_identity: Mapping[str, JsonScalar] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Bind the record to its backend and keep every output ordered and in the image."""
        if self.provenance.capability != GROUNDING_CAPABILITY:
            raise ValueError(
                f"grounding execution provenance capability must be {GROUNDING_CAPABILITY}"
            )
        if self.provenance.configuration_fingerprint != self.request.configuration_fingerprint:
            raise ValueError(
                "grounding execution provenance configuration fingerprint does not match "
                "the request it served"
            )
        if not self.rendered_prompt.strip():
            raise ValueError("grounding execution rendered_prompt must not be empty")
        indices = [item.output_index for item in self.outputs]
        indices.extend(item.output_index for item in self.rejected_outputs)
        if len(set(indices)) != len(indices):
            raise ValueError("grounding output_index values must be unique across outputs")
        width, height = self.request.image.width, self.request.image.height
        for output in self.outputs:
            _validate_inside_image(output, width, height)

    @property
    def request_id(self) -> GroundingRequestId:
        """Return the served request's identity."""
        return self.request.request_id

    @property
    def raw_response_sha256(self) -> str:
        """Return the SHA-256 of the verbatim raw response."""
        return hashlib.sha256(self.raw_response.encode("utf-8")).hexdigest()

    @property
    def no_match(self) -> bool:
        """Whether the backend returned no geometry at all, valid or rejected."""
        return not self.outputs and not self.rejected_outputs

    @property
    def regions(self) -> tuple[Region2D, ...]:
        """Return the canonical ``Region2D`` evidence of every box output.

        Identities are local to the owning result and derived from the request identity
        and the output position, so they are reproducible and never collide with
        discovered regions or with another request's outputs.
        """
        request = self.request
        return tuple(
            Region2D(
                region_id=grounding_region_id_for(
                    result_id=request.perception_result_id,
                    request_id=str(request.request_id),
                    index=output.output_index,
                ),
                bounding_box=output.box,
                provenance=self.provenance,
                source_observation_id=request.source_observation_id,
                image_width=request.image.width,
                image_height=request.image.height,
                area_pixels=output.box.width * output.box.height,
            )
            for output in self.outputs
            if output.box is not None
        )

    @property
    def unsupported_outputs(self) -> tuple[GroundingOutput, ...]:
        """Return outputs the canonical ``Region2D`` contract cannot represent (points)."""
        return tuple(output for output in self.outputs if output.box is None)


def with_grounded_regions(
    result: PerceptionResult, executions: Sequence[RegionGroundingExecution]
) -> PerceptionResult:
    """Return ``result`` with the box evidence of its grounding executions appended.

    Discovered regions keep their position; the result itself is not mutated.

    Args:
        result: The result that owns the requests.
        executions: Grounding executions issued for that result.

    Returns:
        A new result whose ``regions`` end with every grounded box region.

    Raises:
        ValueError: If an execution belongs to another result or observation.
    """
    grounded: list[Region2D] = []
    for execution in executions:
        request = execution.request
        if request.perception_result_id != result.result_id:
            raise ValueError(
                "grounding execution perception_result_id does not match the result: "
                f"{request.perception_result_id!r} != {result.result_id!r}"
            )
        if request.source_observation_id != result.source_observation_id:
            raise ValueError("grounding execution source_observation_id does not match")
        grounded.extend(execution.regions)
    if not grounded:
        return result
    return replace(result, regions=(*result.regions, *grounded))


# --- serialization ---------------------------------------------------------------------


def encode_region_grounding_request(request: RegionGroundingRequest) -> dict[str, Any]:
    """Encode a grounding request, including its query verbatim and its identity."""
    image = request.image
    artifact = cast(ArtifactReference, image.payload_artifact)
    return {
        "request_id": str(request.request_id),
        "perception_result_id": str(request.perception_result_id),
        "image": {
            "source_observation_id": str(image.source_observation_id),
            "payload_reference": image.payload_reference,
            "payload_artifact": artifact.to_dict(),
            "width": image.width,
            "height": image.height,
            "transformations": [record.to_dict() for record in image.transformations],
        },
        "query": _encode_query(request.query),
        "configuration_fingerprint": request.configuration_fingerprint,
    }


def decode_region_grounding_request(record: Mapping[str, Any]) -> RegionGroundingRequest:
    """Decode a grounding request and verify its recorded identity.

    Raises:
        ValueError: If the recorded ``request_id`` is not the digest of the decoded inputs.
    """
    raw_image = record["image"]
    request = RegionGroundingRequest(
        perception_result_id=PerceptionResultId(record["perception_result_id"]),
        image=PreparedImage(
            source_observation_id=SourceObservationId(raw_image["source_observation_id"]),
            payload_reference=raw_image["payload_reference"],
            payload_artifact=ArtifactReference.from_dict(raw_image["payload_artifact"]),
            width=raw_image["width"],
            height=raw_image["height"],
            transformations=tuple(
                _decode_transformation(item) for item in raw_image["transformations"]
            ),
        ),
        query=_decode_query(record["query"]),
        configuration_fingerprint=record["configuration_fingerprint"],
    )
    if record["request_id"] != request.request_id:
        raise ValueError(
            "grounding request_id does not match the digest of its inputs: "
            f"{record['request_id']!r} != {request.request_id!r}"
        )
    return request


def encode_region_grounding_execution(
    execution: RegionGroundingExecution, *, raw_response_reference: str
) -> dict[str, Any]:
    """Encode an execution; the raw response text is referenced, not embedded.

    Args:
        execution: The execution to encode.
        raw_response_reference: Where the verbatim raw response is persisted.

    Returns:
        A JSON-compatible record carrying the raw response's reference and SHA-256.
    """
    return {
        "request": encode_region_grounding_request(execution.request),
        "provenance": encode_provenance(execution.provenance),
        "rendered_prompt": execution.rendered_prompt,
        "raw_response_reference": raw_response_reference,
        "raw_response_sha256": execution.raw_response_sha256,
        "outputs": [_encode_output(output) for output in execution.outputs],
        "rejected_outputs": [
            {
                "output_index": item.output_index,
                "native_text": item.native_text,
                "reason": item.reason.value,
                "detail": item.detail,
                "label": item.label,
            }
            for item in execution.rejected_outputs
        ],
        "no_match_labels": list(execution.no_match_labels),
        "diagnostics": {
            "latency_ms": execution.diagnostics.latency_ms,
            "peak_memory_bytes": execution.diagnostics.peak_memory_bytes,
            "warnings": list(execution.diagnostics.warnings),
            "native": _encode_named_scalars(execution.diagnostics.native),
        },
        "effective_configuration": dict(execution.effective_configuration),
        "runtime_identity": dict(execution.runtime_identity),
    }


def decode_region_grounding_execution(
    record: Mapping[str, Any], *, raw_response: str
) -> RegionGroundingExecution:
    """Decode an execution, verifying the separately persisted raw response.

    Args:
        record: A record written by :func:`encode_region_grounding_execution`.
        raw_response: The raw response text read from ``raw_response_reference``.

    Raises:
        ValueError: If the raw response does not match the recorded SHA-256, or the
            record's request identity does not match its inputs.
    """
    digest = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
    if digest != record["raw_response_sha256"]:
        raise ValueError("grounding raw response does not match its recorded sha256")
    raw_diagnostics = record["diagnostics"]
    return RegionGroundingExecution(
        request=decode_region_grounding_request(record["request"]),
        provenance=decode_provenance(record["provenance"]),
        rendered_prompt=record["rendered_prompt"],
        raw_response=raw_response,
        outputs=tuple(_decode_output(item) for item in record["outputs"]),
        rejected_outputs=tuple(
            RejectedGroundingOutput(
                output_index=item["output_index"],
                native_text=item["native_text"],
                reason=GroundingRejectionReason(item["reason"]),
                detail=item["detail"],
                label=item["label"],
            )
            for item in record["rejected_outputs"]
        ),
        no_match_labels=tuple(record["no_match_labels"]),
        diagnostics=GroundingDiagnostics(
            latency_ms=raw_diagnostics["latency_ms"],
            peak_memory_bytes=raw_diagnostics["peak_memory_bytes"],
            warnings=tuple(raw_diagnostics["warnings"]),
            native=_decode_named_scalars(raw_diagnostics["native"]),
        ),
        effective_configuration=MappingProxyType(dict(record["effective_configuration"])),
        runtime_identity=MappingProxyType(dict(record["runtime_identity"])),
    )


def _encode_query(query: GroundingQuery) -> dict[str, Any]:
    return {
        "task": query.task.value,
        "policy_id": query.policy_id,
        "geometry": query.geometry.value,
        "text": query.text,
        "categories": list(query.categories),
    }


def _decode_query(record: Mapping[str, Any]) -> GroundingQuery:
    return GroundingQuery(
        task=GroundingTask(record["task"]),
        policy_id=record["policy_id"],
        geometry=GroundingGeometry(record["geometry"]),
        text=record["text"],
        categories=tuple(record["categories"]),
    )


def _encode_output(output: GroundingOutput) -> dict[str, Any]:
    return {
        "output_index": output.output_index,
        "native_text": output.native_text,
        "label": output.label,
        "box": None if output.box is None else encode_bounding_box(output.box),
        "point": None if output.point is None else {"x": output.point.x, "y": output.point.y},
        "native_diagnostics": _encode_named_scalars(output.native_diagnostics),
    }


def _decode_output(record: Mapping[str, Any]) -> GroundingOutput:
    raw_box = record["box"]
    raw_point = record["point"]
    return GroundingOutput(
        output_index=record["output_index"],
        native_text=record["native_text"],
        label=record["label"],
        box=None if raw_box is None else decode_bounding_box(raw_box),
        point=None if raw_point is None else GroundingPoint(x=raw_point["x"], y=raw_point["y"]),
        native_diagnostics=_decode_named_scalars(record["native_diagnostics"]),
    )


def _decode_transformation(record: Mapping[str, Any]) -> TransformationRecord:
    return TransformationRecord(
        operation=record["operation"],
        provenance_source=record["provenance_source"],
        parameters=tuple((item["name"], item["value"]) for item in record["parameters"]),
        input_dimensions=(record["input_dimensions"][0], record["input_dimensions"][1]),
        output_dimensions=(record["output_dimensions"][0], record["output_dimensions"][1]),
        output_image=ArtifactReference.from_dict(record["output_image"]),
        warnings=tuple(record["warnings"]),
    )


def _encode_named_scalars(values: tuple[tuple[str, JsonScalar], ...]) -> list[dict[str, Any]]:
    # Lista ordenada de pares, não um objeto: a ordem nativa e nomes repetidos sobrevivem.
    return [{"name": name, "value": value} for name, value in values]


def _decode_named_scalars(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, JsonScalar], ...]:
    return tuple((item["name"], cast(JsonScalar, item["value"])) for item in records)


def _validate_named_scalars(values: tuple[tuple[str, JsonScalar], ...], name: str) -> None:
    for key, value in values:
        if not key:
            raise ValueError(f"{name} names must not be empty")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"{name} values must be JSON scalar values")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"{name} values must be finite: {key!r}")


def _validate_inside_image(output: GroundingOutput, width: int, height: int) -> None:
    if output.box is not None:
        box = output.box
        if (
            box.x_min >= width
            or box.y_min >= height
            or box.x_max > width + _EDGE_TOLERANCE_PX
            or box.y_max > height + _EDGE_TOLERANCE_PX
        ):
            raise ValueError(
                f"grounding output {output.output_index} box lies outside the "
                f"{width}x{height} image"
            )
    elif output.point is not None and (output.point.x > width or output.point.y > height):
        raise ValueError(
            f"grounding output {output.output_index} point lies outside the {width}x{height} image"
        )

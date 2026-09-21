"""Public ingestion application service: preflight and execution behind one boundary.

A CLI, a TUI and the stage DAG all need to run canonical ingestion: construct the configured
source adapter, read canonical observations, validate them, synchronize, and publish an
immutable :class:`~contextmap.ingestion.SequenceArtifact`. This module is that one path, so no
frontend recomposes it. It owns construction, lifecycle, progress, cancellation and the
effective request identity; **every** rule about sources, validation, synchronization,
calibration and artifact layout stays in :mod:`contextmap.ingestion`.

It imports only the ingestion capability's public root, never an adapter: the adapter family
arrives through a factory built by the composition root, so a request for another family than
the configured one is refused, never substituted.

Observations are streamed into the writer's temporary artifact as they are read, and only
payload-free metadata is kept for the cross-observation checks and the synchronization, so a
recording larger than memory can be ingested. A failed or cancelled run aborts the writer and
publishes nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from contextmap.ingestion import (
    MODALITY_NAMES,
    CalibrationError,
    CalibrationSet,
    ImageObservation,
    LidarObservation,
    MissingRequiredTopicError,
    SequenceArtifactError,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SequenceProvenance,
    SourceAdapter,
    SourceAdapterConfig,
    SourceAdapterError,
    SourceObservation,
    SourceTopicMapping,
    SynchronizationConfig,
    compute_configuration_hash,
    compute_source_content_hash,
    current_code_version,
    observation_modality,
    synchronize,
    validate_frame_references,
    validate_image_observation,
    validate_lidar_observation,
    validate_timestamp_ordering,
)
from contextmap.runtime.artifacts import ArtifactRef
from contextmap.runtime.catalog import COMPONENTS
from contextmap.runtime.config import ConfigProblem
from contextmap.runtime.errors import CompositionError
from contextmap.runtime.lifecycle import (
    CancellationToken,
    EventEmitter,
    EventSink,
    StageFailure,
    utc_now,
)
from contextmap.runtime.pipeline import StageRequest

INGESTION_REQUEST_SCHEMA_VERSION = "0.1.0"
"""Version of the request document."""

SourceAdapterFactory = Callable[[SourceAdapterConfig], SourceAdapter]
"""Builds the configured source adapter for one request; the composition root provides it."""

_TOPIC_MODALITY = {"rgb": "image", "lidar": "lidar", "imu": "imu", "pose": "external_pose"}
_TOPIC_CAPABILITY = {
    "rgb": "rgb",
    "lidar": "lidar",
    "imu": "imu",
    "pose": "external_pose",
    "camera_info": "calibration",
}
_MODALITY_TOPIC = {modality: topic for topic, modality in _TOPIC_MODALITY.items()}

PHASES = (
    "planned",
    "reading-source",
    "validating",
    "synchronizing",
    "writing-artifact",
    "completed",
    "failed",
    "cancelled",
)
"""Lifecycle phases, emitted as ``ingestion.<phase>`` events (plus ``ingestion.progress``)."""


@dataclass(frozen=True, kw_only=True)
class ValidationPolicy:
    """What ingestion does with the structural problems it finds.

    Attributes:
        allow_duplicate_timestamps: Whether two observations of one clock may share a
            timestamp.
        on_problems: ``"fail"`` refuses the sequence when any problem is found (nothing is
            published); ``"warn"`` publishes it and records each problem as a warning.
    """

    allow_duplicate_timestamps: bool = True
    on_problems: Literal["fail", "warn"] = "fail"


@dataclass(frozen=True, kw_only=True)
class IngestionRequest:
    """One canonical ingestion, expressed with the ingestion capability's own types.

    Nothing here duplicates the source, topic or calibration schema: it is composed of
    :class:`~contextmap.ingestion.SourceTopicMapping`,
    :class:`~contextmap.ingestion.SynchronizationConfig` and
    :class:`~contextmap.ingestion.CalibrationSet`.

    Attributes:
        source_type: Adapter family identity, for example ``"ros1_bag"``.
        source_path: Path of the recorded source.
        sequence_name: Name of the sequence artifact to publish; a single path segment.
        workspace: Workspace that receives ``sequences/<sequence_name>/<artifact_id>``.
        topics: The topics or channels to read.
        synchronization: The synchronization policy applied before persisting.
        required_topics: Topic names that must exist in the source.
        timestamp_clock_id: Identity of the header clock shared by the topics; derived from
            the source when omitted.
        calibration: An externally supplied calibration merged with what the source holds.
        validation: What to do with structural problems.
        hash_source: Whether to hash the source bytes for its content identity. It is
            O(source size); turning it off is recorded in the provenance.
        config_identity: Digest of the runtime effective configuration this request belongs
            to, recorded for reproduction.
    """

    source_type: str
    source_path: str
    sequence_name: str
    workspace: str
    topics: SourceTopicMapping
    synchronization: SynchronizationConfig
    required_topics: frozenset[str] = frozenset()
    timestamp_clock_id: str | None = None
    calibration: CalibrationSet | None = None
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    hash_source: bool = True
    config_identity: str | None = None

    def __post_init__(self) -> None:
        """Validate the shape of the request; whether it can run is decided by preflight."""
        for name in ("source_type", "source_path", "sequence_name", "workspace"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if (
            self.sequence_name in {".", ".."}
            or "/" in self.sequence_name
            or "\\" in self.sequence_name
        ):
            raise ValueError(
                f"sequence_name must be a single path segment, got {self.sequence_name!r}"
            )

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form of the request, without secrets or the workspace."""
        return {
            "schema_version": INGESTION_REQUEST_SCHEMA_VERSION,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "sequence_name": self.sequence_name,
            "topics": dataclasses.asdict(self.topics),
            "required_topics": sorted(self.required_topics),
            "timestamp_clock_id": self.timestamp_clock_id,
            "synchronization": {
                "reference_modality": self.synchronization.reference_modality,
                "tolerance_nanoseconds": self.synchronization.tolerance_nanoseconds,
            },
            "validation": dataclasses.asdict(self.validation),
            "calibration": None
            if self.calibration is None
            else _calibration_identity(self.calibration),
            "hash_source": self.hash_source,
            "config_identity": self.config_identity,
        }

    @property
    def identity(self) -> str:
        """Return the deterministic identity of what this request asks for.

        The workspace is where the result is written, not part of what is ingested, so two
        requests that differ only in it have the same identity.
        """
        return _digest(self.to_document())

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], *, workspace: str, source_type: str | None = None
    ) -> IngestionRequest:
        """Build a request from a primitive document, as a CLI or a TUI form produces it.

        Args:
            document: A mapping with ``source_path``, ``sequence_name``, ``topics``,
                ``synchronization`` and optionally ``required_topics``, ``timestamp_clock_id``,
                ``validation``, ``hash_source`` and ``config_identity``.
            workspace: The output workspace.
            source_type: The adapter family, when the document does not carry it.

        Returns:
            The request.

        Raises:
            ValueError: If a field is missing or invalid.
        """
        unknown = sorted(set(document.get("topics", {})) - set(_TOPIC_CAPABILITY))
        if unknown:
            raise ValueError(f"unknown topic key(s) {unknown}; known: {sorted(_TOPIC_CAPABILITY)}")
        try:
            topics = SourceTopicMapping(**dict(document.get("topics", {})))
            synchronization = SynchronizationConfig(**dict(document["synchronization"]))
            validation = ValidationPolicy(**dict(document.get("validation", {})))
            return cls(
                source_type=document.get("source_type", source_type) or "",
                source_path=document["source_path"],
                sequence_name=document["sequence_name"],
                workspace=workspace,
                topics=topics,
                synchronization=synchronization,
                required_topics=frozenset(document.get("required_topics", ())),
                timestamp_clock_id=document.get("timestamp_clock_id"),
                validation=validation,
                hash_source=document.get("hash_source", True),
                config_identity=document.get("config_identity"),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid ingestion request: {error!r}") from error


@dataclass(frozen=True, kw_only=True)
class IngestionPreflight:
    """What preflight found before any observation was read.

    Attributes:
        identity: The request identity.
        problems: Everything that blocks the run, each with the path of the offending setting.
        warnings: What does not block but deserves attention.
        capabilities: The modalities the source actually provides, when it could be inspected.
    """

    identity: str
    problems: tuple[ConfigProblem, ...]
    warnings: tuple[str, ...]
    capabilities: Mapping[str, bool]

    @property
    def ok(self) -> bool:
        """Whether nothing blocks the run."""
        return not self.problems

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "ok": self.ok,
            "identity": self.identity,
            "problems": [{"path": p.path, "message": p.message} for p in self.problems],
            "warnings": list(self.warnings),
            "capabilities": dict(self.capabilities),
        }


@dataclass(frozen=True, kw_only=True)
class IngestionFailure:
    """Why an ingestion did not complete.

    Attributes:
        category: ``configuration``, ``dependency``, ``source``, ``validation``, ``output``,
            ``integrity`` or ``cancelled``.
        phase: The phase that was running.
        message: What went wrong, with secrets redacted.
        exception_type: Name of the exception class, when an exception caused it.
    """

    category: str
    phase: str
    message: str
    exception_type: str | None = None

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return dataclasses.asdict(self)


@dataclass(frozen=True, kw_only=True)
class IngestionMetrics:
    """Operational measurements of one ingestion, separate from data quality.

    Attributes:
        elapsed_s: Total wall time.
        phase_seconds: Wall time of each phase that ran.
        observations_read: Observations decoded from the source.
        observation_counts: Observations by modality.
        warnings: Warnings raised by the adapter and by validation.
        dropped_events: Events synchronization never selected.
        processing_observations: Synchronized groups produced.
        bytes_written: Size of the published files.
        files_written: Number of published files.
    """

    elapsed_s: float
    phase_seconds: Mapping[str, float]
    observations_read: int
    observation_counts: Mapping[str, int]
    warnings: int
    dropped_events: int
    processing_observations: int
    bytes_written: int
    files_written: int

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "elapsed_s": self.elapsed_s,
            "phase_seconds": dict(self.phase_seconds),
            "observations_read": self.observations_read,
            "observation_counts": dict(self.observation_counts),
            "warnings": self.warnings,
            "dropped_events": self.dropped_events,
            "processing_observations": self.processing_observations,
            "bytes_written": self.bytes_written,
            "files_written": self.files_written,
        }


@dataclass(frozen=True, kw_only=True)
class IngestionResult:
    """The outcome of one ingestion.

    The persisted :class:`~contextmap.ingestion.SequenceArtifact` stays authoritative:
    frontends reopen it through ``SequenceArtifactReader`` for anything beyond this summary.

    Attributes:
        status: ``"completed"``, ``"failed"`` or ``"cancelled"``.
        sequence_name: The requested sequence name.
        request_identity: The identity of the request that was run.
        artifact_id: The published artifact, or ``None`` when nothing was published.
        artifact_path: Where it was published.
        content_hash: Hash of the artifact's file inventory, for reuse identity.
        observation_counts: Observations by modality.
        warnings: Adapter and validation warnings.
        diagnostics: A summary of validation and synchronization.
        failure: Why it did not complete.
        metrics: Operational measurements.
    """

    status: Literal["completed", "failed", "cancelled"]
    sequence_name: str
    request_identity: str
    artifact_id: str | None
    artifact_path: str | None
    content_hash: str | None
    observation_counts: Mapping[str, int]
    warnings: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    failure: IngestionFailure | None
    metrics: IngestionMetrics

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {
            "status": self.status,
            "sequence_name": self.sequence_name,
            "request_identity": self.request_identity,
            "artifact_id": self.artifact_id,
            "artifact_path": self.artifact_path,
            "content_hash": self.content_hash,
            "observation_counts": dict(self.observation_counts),
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
            "failure": None if self.failure is None else self.failure.to_document(),
            "metrics": self.metrics.to_document(),
        }


class _Abort(Exception):
    """Internal: stops the run at a known phase, with the failure to report."""

    def __init__(self, failure: IngestionFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class IngestionService:
    """Runs canonical ingestion: ``preflight`` before, ``run`` for the whole path."""

    def __init__(
        self,
        adapter_factory: SourceAdapterFactory,
        *,
        clock: Callable[[], str] | None = None,
    ) -> None:
        """Create the service.

        Args:
            adapter_factory: Builds the configured source adapter for a request. It is the only
                way an adapter is obtained, and it refuses another family than the configured
                one, so there is no silent fallback.
            clock: Returns event timestamps; defaults to the current UTC time.
        """
        self._factory = adapter_factory
        self._clock = clock or utc_now

    def preflight(self, request: IngestionRequest) -> IngestionPreflight:
        """Check that a request can run, without reading the source's observations.

        Validates the topic and synchronization configuration, the source path, the adapter
        (constructed, so a missing optional module is reported now), which modalities the
        source really provides, and that the workspace can receive the artifact. It never
        changes the adapter, the synchronization policy or the calibration.

        Args:
            request: The request to check.

        Returns:
            Every problem found, all at once.
        """
        problems: list[ConfigProblem] = []
        warnings: list[str] = []
        capabilities: dict[str, bool] = {}
        self._check_topics(request, problems)
        self._check_output(request, problems)
        if not Path(request.source_path).exists():
            problems.append(_problem("source.path", f"{request.source_path} does not exist"))
        elif not os.access(request.source_path, os.R_OK):
            problems.append(_problem("source.path", f"{request.source_path} is not readable"))
        if not any(p.path.startswith("request.") for p in problems):
            adapter = self._build_adapter(request, problems)
            if adapter is not None and not any(p.path.startswith("source.path") for p in problems):
                self._check_source(request, adapter, problems, warnings, capabilities)
        return IngestionPreflight(
            identity=request.identity,
            problems=tuple(problems),
            warnings=tuple(warnings),
            capabilities=capabilities,
        )

    def run(
        self,
        request: IngestionRequest,
        *,
        event_sink: EventSink | None = None,
        cancellation: CancellationToken | None = None,
        redact: Callable[[str], str] | None = None,
        progress_interval: int = 1000,
    ) -> IngestionResult:
        """Run one canonical ingestion.

        The path is: preflight, construct the adapter, read observations (each one is validated
        and streamed into a temporary artifact), cross-observation validation, synchronization,
        provenance, atomic publication and an integrity check of what was published. Expected
        failures come back as a result with a category, the phase and the message; an
        unexpected exception is emitted as a ``failed`` event and re-raised. A failed or
        cancelled run never publishes anything.

        Args:
            request: What to ingest.
            event_sink: Receives ``ingestion.<phase>`` events and ``ingestion.progress``.
            cancellation: Cooperative cancellation, checked between observations and phases.
            redact: Replaces secret values inside a string, applied to events and messages.
            progress_interval: Emit a progress event every this many observations.

        Returns:
            The outcome.
        """
        emitter = EventEmitter([event_sink] if event_sink else [], clock=self._clock, redact=redact)
        run = _Run(request, emitter, cancellation, redact, progress_interval)
        emitter.emit(
            "ingestion.planned",
            request_identity=request.identity,
            sequence_name=request.sequence_name,
            source_type=request.source_type,
        )
        try:
            self._execute(run)
        except _Abort as abort:
            return run.result(failure=abort.failure)
        except KeyboardInterrupt:
            run.emit_terminal("cancelled", reason="interrupted")
            raise
        except Exception as error:
            run.emit_terminal("failed", category="unexpected", message=str(error))
            raise
        return run.result()

    # --- preflight --------------------------------------------------------------------

    def _check_topics(self, request: IngestionRequest, problems: list[ConfigProblem]) -> None:
        configured = {
            name for name, value in dataclasses.asdict(request.topics).items() if value is not None
        }
        if not configured & set(_TOPIC_MODALITY):
            problems.append(_problem("request.topics", "no data topic is configured"))
        for name in sorted(request.required_topics):
            if name not in _TOPIC_CAPABILITY:
                problems.append(
                    _problem(
                        "request.required_topics",
                        f"{name!r} is not a known topic; known: {sorted(_TOPIC_CAPABILITY)}",
                    )
                )
            elif name not in configured:
                problems.append(
                    _problem(
                        "request.required_topics",
                        f"required topic {name!r} has no topic configured in request.topics",
                    )
                )
        reference = _MODALITY_TOPIC.get(request.synchronization.reference_modality)
        if reference is None or reference not in configured:
            problems.append(
                _problem(
                    "request.synchronization",
                    f"the reference modality {request.synchronization.reference_modality!r} "
                    "has no topic configured",
                )
            )
        if request.timestamp_clock_id is not None and not request.timestamp_clock_id.strip():
            problems.append(_problem("request.timestamp_clock_id", "must not be empty"))

    def _check_output(self, request: IngestionRequest, problems: list[ConfigProblem]) -> None:
        workspace = Path(request.workspace)
        if workspace.exists() and not workspace.is_dir():
            problems.append(_problem("output.workspace", f"{workspace} is not a directory"))
            return
        nearest = workspace
        while not nearest.exists() and nearest != nearest.parent:
            nearest = nearest.parent
        if not os.access(nearest, os.W_OK):
            problems.append(
                _problem("output.workspace", f"{workspace} cannot be created or written")
            )

    def _build_adapter(
        self, request: IngestionRequest, problems: list[ConfigProblem]
    ) -> SourceAdapter | None:
        try:
            config = _adapter_config(request)
        except (ValueError, CalibrationError) as error:
            problems.append(_problem("request.adapter", str(error)))
            return None
        try:
            return self._factory(config)
        except ImportError as error:
            hint = _install_hint(request.source_type)
            module = f" {error.name!r}" if error.name else ""
            problems.append(
                _problem(
                    "adapter.dependency",
                    f"the {request.source_type!r} adapter needs the optional module{module}, "
                    f"which is not installed{hint}",
                )
            )
        except CompositionError as error:
            problems.append(_problem("adapter.selection", str(error)))
        return None

    def _check_source(
        self,
        request: IngestionRequest,
        adapter: SourceAdapter,
        problems: list[ConfigProblem],
        warnings: list[str],
        capabilities: dict[str, bool],
    ) -> None:
        try:
            found = adapter.capabilities()
        except Exception as error:
            # A fronteira do adapter decodifica arquivos arbitrários: qualquer falha ao abri-lo
            # é um problema de preflight, com o tipo da exceção para o diagnóstico.
            problems.append(
                _problem(
                    "source.read",
                    f"cannot inspect {request.source_path}: {type(error).__name__}: {error}",
                )
            )
            return
        capabilities.update(dataclasses.asdict(found))
        for name in sorted(request.required_topics):
            capability = _TOPIC_CAPABILITY.get(name)
            if capability is not None and not capabilities.get(capability, False):
                problems.append(
                    _problem(
                        "source.required_topics",
                        f"required topic {name!r} is not present in the source",
                    )
                )
        reference = _MODALITY_TOPIC.get(request.synchronization.reference_modality)
        capability = _TOPIC_CAPABILITY.get(reference or "")
        if capability is not None and not capabilities.get(capability, False):
            problems.append(
                _problem(
                    "source.synchronization",
                    f"the reference modality {request.synchronization.reference_modality!r} "
                    "is not present in the source",
                )
            )
        if not capabilities.get("calibration", False):
            warnings.append(
                "the source provides no calibration and none was configured: frame references "
                "cannot be validated"
            )

    # --- execution --------------------------------------------------------------------

    def _execute(self, run: _Run) -> None:
        request = run.request
        preflight = self.preflight(request)
        if not preflight.ok:
            first = preflight.problems[0]
            raise _Abort(
                IngestionFailure(
                    category=_category(first.path),
                    phase="planned",
                    message="; ".join(str(p) for p in preflight.problems),
                )
            )
        problems: list[ConfigProblem] = []
        adapter = self._build_adapter(request, problems)
        if adapter is None:  # inalcançável depois de um preflight ok; falha explícita mesmo assim
            raise _Abort(
                IngestionFailure(
                    category="dependency", phase="planned", message="the adapter cannot be built"
                )
            )

        # O serviço é o chamador do writer: ele escolhe a identidade e o diretório final. O writer
        # apenas grava onde lhe mandam e nunca aloca nada.
        artifact_id = SequenceArtifactId(uuid4().hex)
        directory = Path(request.workspace) / "sequences" / request.sequence_name / artifact_id
        with SequenceArtifactWriter(
            output_dir=directory, sequence_name=request.sequence_name, artifact_id=artifact_id
        ) as writer:
            calibration = self._read(run, adapter, writer)
            self._validate(run, calibration)
            diagnostics = self._synchronize(run)
            manifest = self._publish(run, adapter, writer, calibration, diagnostics)
        self._verify(run, manifest, directory)

    def _read(
        self, run: _Run, adapter: SourceAdapter, writer: SequenceArtifactWriter
    ) -> CalibrationSet | None:
        run.begin("reading-source")
        iterator = iter(_source_iterator(adapter, run))
        while True:
            run.check_cancelled()
            try:
                observation = next(iterator)
            except StopIteration:
                break
            run.content_problems.extend(_content_problems(observation))
            try:
                writer.add_observation(observation)
            except (SequenceArtifactError, OSError) as error:
                raise run.abort("output", error) from error
            run.metadata.append(_without_payload(observation))
            modality = observation_modality(observation)
            run.counts[modality] = run.counts.get(modality, 0) + 1
            run.read += 1
            if run.read % run.progress_interval == 0:
                run.emitter.emit("ingestion.progress", observations_read=run.read)
        try:
            calibration = adapter.read_calibration()
        except Exception as error:
            raise run.abort_source(error) from error
        run.adapter_warnings = tuple(warning.reason for warning in adapter.warnings())
        return calibration

    def _validate(self, run: _Run, calibration: CalibrationSet | None) -> None:
        run.check_cancelled()
        run.begin("validating")
        request = run.request
        problems = list(run.content_problems)
        problems += validate_timestamp_ordering(
            run.metadata, allow_duplicates=request.validation.allow_duplicate_timestamps
        )
        problems += validate_frame_references(run.metadata, calibration)
        if run.counts.get(request.synchronization.reference_modality, 0) == 0:
            problems.append(
                f"no {request.synchronization.reference_modality!r} observation was read: "
                "there is nothing to synchronize around"
            )
        run.validation_problems = tuple(problems)
        if problems and request.validation.on_problems == "fail":
            shown = "; ".join(problems[:3])
            more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
            raise _Abort(
                IngestionFailure(
                    category="validation",
                    phase="validating",
                    message=f"{len(problems)} structural problem(s): {shown}{more}",
                )
            )

    def _synchronize(self, run: _Run) -> Any:
        run.check_cancelled()
        run.begin("synchronizing")
        try:
            processing, diagnostics = synchronize(run.metadata, config=run.request.synchronization)
        except ValueError as error:
            raise run.abort("validation", error) from error
        run.processing = len(processing)
        run.dropped = len(diagnostics.dropped_events)
        run.policy = processing[0].policy if processing else None
        return diagnostics

    def _publish(
        self,
        run: _Run,
        adapter: SourceAdapter,
        writer: SequenceArtifactWriter,
        calibration: CalibrationSet | None,
        diagnostics: Any,
    ) -> Any:
        run.check_cancelled()
        run.begin("writing-artifact")
        request = run.request
        warnings = (*run.adapter_warnings, *run.validation_problems)
        run.warnings = warnings
        try:
            if calibration is not None:
                writer.set_calibration(calibration)
            writer.set_provenance(
                _provenance(request, calibration, run.policy, warnings, source_hash(request))
            )
            writer.set_diagnostics(warnings=warnings, synchronization=diagnostics)
            return writer.finalize()
        except CalibrationError as error:
            raise run.abort("validation", error) from error
        except (SequenceArtifactError, OSError) as error:
            raise run.abort("output", error) from error

    def _verify(self, run: _Run, manifest: Any, directory: Path) -> None:
        run.manifest = manifest
        run.artifact_path = directory
        problems = SequenceArtifactReader(directory).verify_integrity()
        if problems:
            raise _Abort(
                IngestionFailure(
                    category="integrity",
                    phase="writing-artifact",
                    message=(
                        f"the published artifact {directory} failed its integrity check: "
                        + "; ".join(problems[:3])
                    ),
                )
            )


class _Run:
    """The mutable state of one ingestion: counters, phase timing and the terminal event."""

    def __init__(
        self,
        request: IngestionRequest,
        emitter: EventEmitter,
        cancellation: CancellationToken | None,
        redact: Callable[[str], str] | None,
        progress_interval: int,
    ) -> None:
        self.request = request
        self.emitter = emitter
        self.cancellation = cancellation
        self.redact = redact
        self.progress_interval = max(1, progress_interval)
        self.started = time.monotonic()
        self.phase = "planned"
        self.phase_started = self.started
        self.phase_seconds: dict[str, float] = {}
        self.metadata: list[SourceObservation] = []
        self.counts: dict[str, int] = {}
        self.read = 0
        self.content_problems: list[str] = []
        self.validation_problems: tuple[str, ...] = ()
        self.adapter_warnings: tuple[str, ...] = ()
        self.warnings: tuple[str, ...] = ()
        self.processing = 0
        self.dropped = 0
        self.policy: str | None = None
        self.manifest: Any = None
        self.artifact_path: Path | None = None

    def begin(self, phase: str) -> None:
        now = time.monotonic()
        self.phase_seconds[self.phase] = round(
            self.phase_seconds.get(self.phase, 0.0) + now - self.phase_started, 6
        )
        self.phase, self.phase_started = phase, now
        self.emitter.emit(f"ingestion.{phase}")

    def check_cancelled(self) -> None:
        if self.cancellation is not None and self.cancellation.cancelled:
            raise _Abort(
                IngestionFailure(
                    category="cancelled", phase=self.phase, message=self.cancellation.reason
                )
            )

    def abort(self, category: str, error: BaseException) -> _Abort:
        return _Abort(
            IngestionFailure(
                category=category,
                phase=self.phase,
                message=self._clean(str(error)),
                exception_type=type(error).__name__,
            )
        )

    def abort_source(self, error: BaseException) -> _Abort:
        category = "dependency" if isinstance(error, ImportError) else "source"
        return self.abort(category, error)

    def emit_terminal(self, kind: str, **data: Any) -> None:
        self.emitter.emit(f"ingestion.{kind}", phase=self.phase, **data)

    def _clean(self, text: str) -> str:
        return text if self.redact is None else self.redact(text)

    def result(self, failure: IngestionFailure | None = None) -> IngestionResult:
        now = time.monotonic()
        self.phase_seconds[self.phase] = round(
            self.phase_seconds.get(self.phase, 0.0) + now - self.phase_started, 6
        )
        manifest = self.manifest if failure is None else None
        files = list(manifest.file_inventory) if manifest is not None else []
        status: Literal["completed", "failed", "cancelled"] = (
            "completed"
            if failure is None
            else ("cancelled" if failure.category == "cancelled" else "failed")
        )
        if failure is None:
            assert manifest is not None  # sem falha, a publicação terminou
            self.emit_terminal("completed", artifact_id=str(manifest.artifact_id))
        else:
            self.emit_terminal(
                "cancelled" if status == "cancelled" else "failed",
                category=failure.category,
                message=failure.message,
            )
        metrics = IngestionMetrics(
            elapsed_s=round(now - self.started, 6),
            phase_seconds=dict(self.phase_seconds),
            observations_read=self.read,
            observation_counts={name: self.counts.get(name, 0) for name in sorted(MODALITY_NAMES)},
            warnings=len(self.warnings or self.adapter_warnings),
            dropped_events=self.dropped,
            processing_observations=self.processing,
            bytes_written=sum(entry.size_bytes for entry in files),
            files_written=len(files),
        )
        return IngestionResult(
            status=status,
            sequence_name=self.request.sequence_name,
            request_identity=self.request.identity,
            artifact_id=None if manifest is None else str(manifest.artifact_id),
            artifact_path=None if manifest is None else str(self.artifact_path),
            content_hash=None if manifest is None else _inventory_hash(manifest),
            observation_counts=metrics.observation_counts,
            warnings=self.warnings or self.adapter_warnings,
            diagnostics={
                "processing_observations": self.processing,
                "dropped_events": self.dropped,
                "validation_problems": list(self.validation_problems),
                "synchronization_policy": self.policy,
            },
            failure=failure,
            metrics=metrics,
        )


class IngestionStageExecutor:
    """Runs canonical ingestion as the ``ingestion`` stage of the runtime DAG.

    The stage has no upstream artifact; what to ingest comes from the request given here.
    A failed or cancelled ingestion becomes a :class:`~contextmap.runtime.lifecycle.StageFailure`
    carrying the ingestion's own failure category, so the run record says why.
    """

    def __init__(
        self,
        service: IngestionService,
        request: IngestionRequest,
        *,
        event_sink: EventSink | None = None,
        cancellation: CancellationToken | None = None,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        """Bind the executor to a service and the request it runs."""
        self._service = service
        self._request = request
        self._event_sink = event_sink
        self._cancellation = cancellation
        self._redact = redact

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Ingest and return the published sequence artifact.

        Args:
            request: The stage request; ingestion has no upstream inputs to read from it.

        Returns:
            A reference to the published artifact, identified by the hash of its inventory.

        Raises:
            StageFailure: If the ingestion failed or was cancelled.
        """
        result = self._service.run(
            self._request,
            event_sink=self._event_sink,
            cancellation=self._cancellation,
            redact=self._redact,
        )
        if result.status != "completed" or result.artifact_id is None:
            failure = result.failure
            raise StageFailure(
                failure.message if failure else "ingestion did not complete",
                category=failure.category if failure else "execution",
            )
        return ArtifactRef(
            stage_id="ingestion",
            contract="SequenceArtifact",
            artifact_id=result.artifact_id,
            content_hash=result.content_hash,
        )


# --- helpers ----------------------------------------------------------------------------


def _problem(path: str, message: str) -> ConfigProblem:
    return ConfigProblem(path=path, message=message)


def _category(path: str) -> str:
    if path.startswith("adapter.dependency"):
        return "dependency"
    if path.startswith("source."):
        return "source"
    if path.startswith("output."):
        return "output"
    return "configuration"


def _install_hint(source_type: str) -> str:
    spec = COMPONENTS["ingestion.source_adapter"].backends.get(source_type)
    return f" ({spec.install_hint})" if spec is not None and spec.install_hint else ""


def _adapter_config(request: IngestionRequest) -> SourceAdapterConfig:
    return SourceAdapterConfig(
        source_type=request.source_type,
        path=request.source_path,
        topics=request.topics,
        timestamp_clock_id=request.timestamp_clock_id,
        calibration=request.calibration,
        required_topics=request.required_topics,
    )


def _source_iterator(adapter: SourceAdapter, run: _Run) -> Any:
    """Iterate the adapter's observations, translating what it raises into an abort."""
    try:
        yield from adapter.read_observations()
    except MissingRequiredTopicError as error:
        raise run.abort("source", error) from error
    except SourceAdapterError as error:
        raise run.abort("source", error) from error
    except ImportError as error:
        raise run.abort("dependency", error) from error
    except _Abort:
        raise
    except Exception as error:
        # A fronteira do adapter decodifica arquivos arbitrários e o conjunto de exceções do SDK
        # de origem não é enumerável aqui: toda falha de decodificação é uma falha da fonte.
        raise run.abort("source", error) from error


def _content_problems(observation: SourceObservation) -> list[str]:
    if isinstance(observation, ImageObservation):
        return validate_image_observation(observation)
    if isinstance(observation, LidarObservation):
        return validate_lidar_observation(observation)
    return []


def _without_payload(observation: SourceObservation) -> SourceObservation:
    """Keep what synchronization and ordering checks read, and drop the payload bytes."""
    if isinstance(observation, ImageObservation | LidarObservation):
        return dataclasses.replace(observation, data=b"")
    return observation


def source_hash(request: IngestionRequest) -> str | None:
    """Hash the source's bytes, when the request asks for it."""
    return compute_source_content_hash(Path(request.source_path)) if request.hash_source else None


def _provenance(
    request: IngestionRequest,
    calibration: CalibrationSet | None,
    policy: str | None,
    warnings: Sequence[str],
    content_hash: str | None,
) -> SequenceProvenance:
    config: dict[str, object] = {
        "topics": dataclasses.asdict(request.topics),
        "required_topics": sorted(request.required_topics),
        "timestamp_clock_id": request.timestamp_clock_id,
        "synchronization": request.to_document()["synchronization"],
        "validation": dataclasses.asdict(request.validation),
        "hash_source": request.hash_source,
        "request_identity": request.identity,
        "config_identity": request.config_identity,
    }
    return SequenceProvenance(
        source_type=request.source_type,
        source_path=request.source_path,
        source_content_hash=content_hash,
        ingestion_config=config,
        configuration_hash=compute_configuration_hash(config),
        adapter_type=request.source_type,
        code_version=current_code_version(),
        calibration_source_hash=None if calibration is None else _calibration_identity(calibration),
        synchronization_policy=policy,
        warnings=tuple(warnings),
    )


def _calibration_identity(calibration: CalibrationSet) -> str:
    """Identify a calibration by the content hashes of its entries and its static transforms."""
    document = {
        "entries": sorted(
            (str(key), entry.content_hash) for key, entry in calibration.entries.items()
        ),
        "transforms": [
            dataclasses.asdict(transform) for transform in calibration.static_transforms
        ],
    }
    return _digest(document)


def _inventory_hash(manifest: Any) -> str:
    inventory = sorted(
        (entry.path, entry.size_bytes, entry.content_hash) for entry in manifest.file_inventory
    )
    return _digest(inventory)


def _digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

"""Persisted, immutable, self-describing Visual Perception run artifacts.

A ``PerceptionRunArtifact`` is a first-class, immutable directory holding
the results of one configured :class:`~contextmap.visual_perception.models.PerceptionRun`.
It must be openable and inspectable on its own, at the directory the caller
chose to write it. See
``src/contextmap/visual_perception/docs/run_artifact.md`` for the on-disk
layout and the trade-offs made for v0 (notably: JSON Lines outputs
instead of Parquet, and a smaller debug layout than the issue's full
candidate structure).

Writing is atomic, the same pattern as
:mod:`contextmap.ingestion.sequence_artifact`: :class:`PerceptionRunWriter`
builds the run in a temporary sibling of the ``output_dir`` its caller chose
and only makes it visible there after an internal integrity check succeeds.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.dense_region_association import (
    DenseFeatureMap,
    DenseFeatureSampling,
)
from contextmap.visual_perception.embedding_space import embedding_space_fingerprint
from contextmap.visual_perception.feature_diagnostics import (
    DenseFeatureDiagnostic,
    FeatureDebugLevel,
    FeatureDiagnosticPreview,
    FeatureExtractionDiagnostic,
    write_feature_diagnostics,
)
from contextmap.visual_perception.feature_store import (
    FEATURE_INDEX_FILENAME,
    FeatureStoreReader,
    FeatureStoreWriter,
    write_feature_index,
)
from contextmap.visual_perception.mask_store import (
    MASK_INDEX_FILENAME,
    MaskStoreReader,
    MaskStoreWriter,
    write_mask_index,
)
from contextmap.visual_perception.models import (
    FeatureId,
    PerceptionResult,
    PerceptionRunId,
    Region2D,
    RegionId,
    VisualFeature,
)
from contextmap.visual_perception.pipeline import decode_pipeline_preset, encode_pipeline_preset
from contextmap.visual_perception.semantic_audit import (
    SemanticDebugLevel,
    write_semantic_audit,
)
from contextmap.visual_perception.semantic_backend import (
    FailedSemanticInterpretation,
    SemanticInterpretationExecution,
    decode_failed_semantic_interpretation,
    decode_semantic_execution,
    encode_failed_semantic_interpretation,
    encode_semantic_execution,
)
from contextmap.visual_perception.semantic_requests import SemanticVisualView
from contextmap.visual_perception.serialization import (
    decode_perception_result,
    encode_perception_result,
)
from contextmap.visual_perception.service import StageOutcome, StageStatus

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from contextmap.visual_perception.pipeline import PipelinePreset

SCHEMA_VERSION = "0.5.0"
"""Perception run artifact schema version written and understood by this module.

Bumped to ``0.5.0`` when a region's mask stopped being inlined as a JSON
pixel array in ``outputs/results.jsonl`` and moved to a compact,
lazily-loaded ``outputs/masks/`` store referenced by ``mask_reference``
(#378; see ``docs/run_artifact.md``). Bumped to ``0.4.0`` when canonical
semantic evidence and execution audit records changed the persisted
result and output contracts. This is a pre-1.0 schema, so no
compatibility reader for historical manifests is kept.
"""

_MANIFEST_FILENAME = "manifest.json"
_README_FILENAME = "README.md"
_RESULTS_FILENAME = "outputs/results.jsonl"
_METRICS_FILENAME = "metrics/stage-timings.jsonl"
_SEMANTIC_EXECUTIONS_FILENAME = "outputs/semantic-interpretations.jsonl"
_SEMANTIC_FAILURES_FILENAME = "outputs/semantic-interpretation-failures.jsonl"
_SEMANTIC_DEBUG_ROOT = "debug/40-semantic-interpretation"
_FEATURES_DIRNAME = "outputs/features"
_MASKS_DIRNAME = "outputs/masks"


class RunArtifactError(Exception):
    """Base class for perception run artifact read/write failures."""


class IncompleteRunArtifactError(RunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


@dataclass(frozen=True, kw_only=True)
class RunArtifactFileEntry:
    """One entry in a run artifact's file inventory.

    Attributes:
        path: Path relative to the run's directory, using ``/`` separators.
        size_bytes: Size of the file in bytes.
        content_hash: Content hash as ``"sha256:<hex digest>"``.
    """

    path: str
    size_bytes: int
    content_hash: str


@dataclass(frozen=True, kw_only=True)
class RunArtifactManifest:
    """Authoritative metadata for a persisted perception run.

    Attributes:
        run_id: Identity of the run, supplied by the caller.
        run_index: Ordinal of the run among the caller's runs of this
            sequence, supplied by the caller; never a global identity.
        sequence_name: Name of the processed sequence.
        sequence_artifact_id: Canonical sequence artifact processed.
        selection_id: Deterministic identity of the sequence selection
            processed.
        enabled_capabilities: Capability names enabled for this run.
        pipeline_preset: The versioned
            :class:`~contextmap.visual_perception.pipeline.PipelinePreset`
            resolved for this run (see
            :func:`~contextmap.visual_perception.pipeline.encode_pipeline_preset`),
            so the exact stage graph and backend identities this run
            used are inspectable without recomputing them.
        configuration_digest: Deterministic
            :meth:`~contextmap.visual_perception.pipeline.ResolvedPipeline.configuration_digest`
            for this run's resolved pipeline — changes whenever the
            preset content or a resolved backend's identity changes.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        result_count: Number of ``PerceptionResult``s in this run.
        region_count: Total regions across every result.
        feature_count: Total features across every result.
        claim_count: Total claims across every result.
        file_inventory: Every file this run artifact contains, excluding
            ``manifest.json`` and ``README.md`` themselves.
    """

    run_id: PerceptionRunId
    run_index: int
    sequence_name: str
    sequence_artifact_id: str
    selection_id: str
    enabled_capabilities: frozenset[str]
    pipeline_preset: PipelinePreset
    configuration_digest: str
    schema_version: str
    created_at: str
    result_count: int
    region_count: int
    feature_count: int
    claim_count: int
    file_inventory: Sequence[RunArtifactFileEntry]


class PerceptionRunWriter:
    """Builds an immutable perception run artifact on the local filesystem."""

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        run_id: PerceptionRunId,
        run_index: int,
        sequence_artifact_id: str,
        selection_id: str,
        enabled_capabilities: frozenset[str],
        pipeline_preset: PipelinePreset,
        configuration_digest: str,
        feature_debug_level: FeatureDebugLevel = FeatureDebugLevel.NONE,
        semantic_debug_level: SemanticDebugLevel = SemanticDebugLevel.FULL,
    ) -> None:
        """Create a writer for a new perception run artifact.

        Args:
            output_dir: The final directory of the artifact. The caller chooses
                it (in the runtime, ``<workspace>/<dataset>/<run>/visual_perception``);
                the writer computes no path, builds the run in a temporary
                sibling of ``output_dir`` and refuses to replace a directory
                that already exists.
            sequence_name: Name of the sequence this run processed.
            run_id: Identity of the run, supplied by the caller and never
                allocated here.
            run_index: Ordinal of this run among the caller's runs of the
                same sequence, supplied by the caller and recorded as given.
            sequence_artifact_id: Canonical sequence artifact processed.
            selection_id: Deterministic identity of the sequence
                selection processed.
            enabled_capabilities: Capability names enabled for this run.
            pipeline_preset: The
                :class:`~contextmap.visual_perception.pipeline.PipelinePreset`
                resolved for this run.
            configuration_digest: This run's resolved pipeline
                configuration digest (see
                :meth:`~contextmap.visual_perception.pipeline.ResolvedPipeline.configuration_digest`).
            feature_debug_level: Amount of non-contractual Feature Extraction
                debug evidence to persist. Required metrics are independent of
                this level.
            semantic_debug_level: Amount of human-oriented Semantic
                Interpretation evidence to persist. Canonical outputs and
                response hashes remain independent of this level.
        """
        self._run_id = run_id
        self._run_index = run_index
        self._sequence_name = sequence_name
        self._sequence_artifact_id = sequence_artifact_id
        self._selection_id = selection_id
        self._enabled_capabilities = enabled_capabilities
        self._pipeline_preset = pipeline_preset
        self._configuration_digest = configuration_digest
        self._feature_debug_level = feature_debug_level
        self._semantic_debug_level = semantic_debug_level
        self._final_dir = output_dir
        self._tmp_dir = output_dir.parent / f".tmp-{output_dir.name}-{uuid4().hex[:8]}"
        self._staging_created = False
        self._mask_store: MaskStoreWriter | None = None
        self._results: list[PerceptionResult] = []
        self._source_observation_ids: set[SourceObservationId] = set()
        # Só o registro leve (stage_id/status/duration_ms/error) fica retido: o `output` de
        # region_discovery é a mesma tupla de Region2D com máscaras que add_result() recebe.
        self._stage_outcome_records: list[dict[str, Any]] = []
        self._semantic_executions: list[SemanticInterpretationExecution] = []
        self._semantic_request_ids: set[str] = set()
        self._semantic_failures: list[FailedSemanticInterpretation] = []
        # reference -> (sha256, size_bytes); os bytes vao direto para o disco ao serem
        # adicionados, entao o writer nunca retem as imagens enviadas ao backend.
        self._semantic_view_payloads: dict[str, tuple[str, int]] = {}
        self._feature_store: FeatureStoreWriter | None = None
        self._feature_payloads: list[tuple[VisualFeature, SourceObservationId]] = []
        self._feature_diagnostics: list[FeatureExtractionDiagnostic] = []
        self._feature_previews: list[FeatureDiagnosticPreview] = []
        self._finalized = False

    def add_result(self, result: PerceptionResult) -> None:
        """Queue a result to be written by :meth:`finalize`.

        Args:
            result: A result produced by this run.

        Raises:
            RunArtifactError: If called after :meth:`finalize`, if the
                result belongs to another run or sequence artifact, or
                if this writer already contains a result for the same
                source observation.
        """
        if self._finalized:
            raise RunArtifactError("cannot add results after finalize()")
        if result.run_id != self._run_id:
            raise RunArtifactError(
                f"result run_id {result.run_id!r} does not match writer run_id {self._run_id!r}"
            )
        if result.sequence_artifact_id != self._sequence_artifact_id:
            raise RunArtifactError(
                "result sequence_artifact_id "
                f"{result.sequence_artifact_id!r} does not match writer "
                f"sequence_artifact_id {self._sequence_artifact_id!r}"
            )
        if result.source_observation_id in self._source_observation_ids:
            raise RunArtifactError(
                "duplicate source_observation_id in perception run: "
                f"{result.source_observation_id!r}"
            )
        self._results.append(self._persist_result_masks(result))
        self._source_observation_ids.add(result.source_observation_id)

    def _ensure_staging(self) -> Path:
        """Create the temporary run directory on first use and return it."""
        if not self._staging_created:
            self._tmp_dir.mkdir(parents=True, exist_ok=False)
            self._staging_created = True
        return self._tmp_dir

    def _discard_staging(self) -> None:
        """Remove the temporary directory, so an unpublished run leaves nothing behind."""
        if self._staging_created:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._staging_created = False

    def _persist_result_masks(self, result: PerceptionResult) -> PerceptionResult:
        """Persist this result's masks now and return it carrying references instead of pixels.

        A 640x480 ``InlineMask`` was a tuple of 307200 pointers (~2.36 MB measured), so
        buffering every frame's masks until :meth:`finalize` cost ~18 GB on a real 360-frame
        run; at one byte per pixel since #593 it would still be ~2.4 GB. Writing them here lets
        the caller release the pixels as soon as this returns,
        exactly as :class:`~contextmap.ingestion.sequence_artifact.SequenceArtifactWriter`
        does for observation payloads.
        """
        if all(region.mask is None for region in result.regions):
            return result
        try:
            if self._mask_store is None:
                self._mask_store = MaskStoreWriter(self._ensure_staging() / _MASKS_DIRNAME)
            references: dict[tuple[SourceObservationId, RegionId], str] = {}
            for region in result.regions:
                if region.mask is None:
                    continue
                entry = self._mask_store.write(
                    region_id=region.region_id,
                    source_observation_id=result.source_observation_id,
                    mask=region.mask,
                )
                references[(result.source_observation_id, region.region_id)] = (
                    f"{_MASKS_DIRNAME}/{entry.payload_reference}"
                )
        except BaseException:
            self._discard_staging()
            raise
        return _with_persisted_masks(result, references)

    def add_stage_outcomes(self, outcomes: Sequence[StageOutcome]) -> None:
        """Record stage outcomes (timings/status) to be written by :meth:`finalize`.

        Args:
            outcomes: Outcomes from one or more
                :func:`~contextmap.visual_perception.service.execute_stage_graph`
                calls.

        Raises:
            RunArtifactError: If called after :meth:`finalize`.
        """
        if self._finalized:
            raise RunArtifactError("cannot add stage outcomes after finalize()")
        self._stage_outcome_records.extend(_encode_stage_outcome(outcome) for outcome in outcomes)
        for outcome in outcomes:
            if outcome.status is StageStatus.SUCCEEDED and isinstance(
                outcome.output, SemanticInterpretationExecution
            ):
                request_id = str(outcome.output.request.request_id)
                if request_id in self._semantic_request_ids:
                    raise RunArtifactError(
                        f"duplicate semantic request_id in perception run: {request_id!r}"
                    )
                self._semantic_executions.append(outcome.output)
                self._semantic_request_ids.add(request_id)

    def add_semantic_view_payload(self, view: SemanticVisualView, payload: bytes) -> None:
        """Queue the exact visual payload supplied to one semantic request.

        Args:
            view: Canonical view whose artifact-relative reference and SHA-256
                identify the payload.
            payload: Exact encoded image bytes supplied to the backend.

        Raises:
            RunArtifactError: If finalized, the reference is outside the
                semantic-view output namespace, or the payload hash conflicts
                with the view contract or another queued payload.
        """
        if self._finalized:
            raise RunArtifactError("cannot add semantic view payloads after finalize()")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != view.sha256:
            raise RunArtifactError(
                "semantic view payload hash does not match SemanticVisualView.sha256: "
                f"expected {view.sha256!r}, found {digest!r}"
            )
        existing = self._semantic_view_payloads.get(view.payload_reference)
        if existing is not None:
            if existing[0] != digest:
                raise RunArtifactError(
                    f"conflicting semantic view payload for {view.payload_reference!r}"
                )
            return
        try:
            payload_path = self._ensure_staging() / view.payload_reference
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_bytes(payload)
        except BaseException:
            self._discard_staging()
            raise
        self._semantic_view_payloads[view.payload_reference] = (digest, len(payload))

    def add_failed_semantic_interpretation(self, failed: FailedSemanticInterpretation) -> None:
        """Record one real backend call whose response was observed but never materialized.

        The response is evidence even though the parser rejected it, so it is persisted in its
        own contractual stream (``outputs/semantic-interpretation-failures.jsonl``) instead of
        being reduced to an error string. Keeping it separate from
        ``outputs/semantic-interpretations.jsonl`` leaves the success contract, and every
        artifact already written under this schema version, readable.

        The attempt identity is shared: a reader reconciles ``attempted`` against ``parsed``
        and ``parse_failed`` through ``request.request_id``, which never appears in both
        streams.

        Args:
            failed: The observed-but-unparsed interpretation to persist.

        Raises:
            RunArtifactError: If called after :meth:`finalize`, or if this run already carries
                an execution or a failure for the same ``request_id``.
        """
        if self._finalized:
            raise RunArtifactError("cannot add failed semantic interpretations after finalize()")
        request_id = str(failed.request.request_id)
        if request_id in self._semantic_request_ids:
            raise RunArtifactError(
                f"duplicate semantic request_id in perception run: {request_id!r}"
            )
        self._semantic_request_ids.add(request_id)
        # Guardado como objeto, nao codificado: finalize() aplica as mesmas invariantes de
        # identidade e de evidencia de entrada que valem para as execucoes bem-sucedidas.
        self._semantic_failures.append(failed)

    def add_feature_payload(
        self,
        feature: VisualFeature,
        source_observation_id: SourceObservationId,
        array: NDArray[Any],
    ) -> None:
        """Queue a feature's numerical payload to be persisted by :meth:`finalize`.

        Persisting a feature's payload is opt-in per feature: a
        ``VisualFeature`` whose payload was never queued here still
        appears in ``outputs/results.jsonl`` with its metadata, just
        without a loadable array in this run's feature store (see
        :mod:`contextmap.visual_perception.feature_store`).

        Args:
            feature: The feature this array belongs to. Its ``shape``
                and ``dtype`` must match ``array`` exactly (validated at
                :meth:`finalize` time).
            source_observation_id: The physical observation the feature
                was produced for.
            array: The array to persist.

        Raises:
            RunArtifactError: If called after :meth:`finalize`.
        """
        if self._finalized:
            raise RunArtifactError("cannot add feature payloads after finalize()")
        try:
            if self._feature_store is None:
                self._feature_store = FeatureStoreWriter(self._ensure_staging() / _FEATURES_DIRNAME)
            self._feature_store.write(feature, source_observation_id, array)
        except BaseException:
            self._discard_staging()
            raise
        # Só a metadata fica retida: as validações de finalize() nunca leem o array.
        self._feature_payloads.append((feature, source_observation_id))

    def add_feature_diagnostic(self, diagnostic: FeatureExtractionDiagnostic) -> None:
        """Queue one structured Feature Extraction audit event.

        Raises:
            RunArtifactError: If called after :meth:`finalize`.
        """
        if self._finalized:
            raise RunArtifactError("cannot add feature diagnostics after finalize()")
        self._feature_diagnostics.append(diagnostic)

    def add_feature_preview(self, preview: FeatureDiagnosticPreview) -> None:
        """Queue one small human-only preview controlled by the debug level.

        Raises:
            RunArtifactError: If called after :meth:`finalize`.
        """
        if self._finalized:
            raise RunArtifactError("cannot add feature previews after finalize()")
        self._feature_previews.append(preview)

    def finalize(self) -> RunArtifactManifest:
        """Write every queued result/outcome and finalize the run atomically.

        Returns:
            The manifest of the finalized run.

        Raises:
            RunArtifactError: If already finalized, if a run already
                exists at ``output_dir``, if a queued feature payload or a
                ``SUCCEEDED``/``WARNING`` feature diagnostic does not describe
                exactly one feature of this run's results, or if writing fails.
        """
        if self._finalized:
            raise RunArtifactError("writer already finalized")
        try:
            if self._final_dir.exists():
                raise RunArtifactError(f"perception run artifact already exists: {self._final_dir}")
            self._validate_feature_payload_references()
            self._validate_feature_diagnostics()
            self._validate_semantic_execution_materialization()
            self._validate_failed_semantic_interpretations()
            self._validate_semantic_view_payloads()

            self._ensure_staging()
            manifest = self._write_contents()
            problems = _check_file_inventory(self._tmp_dir, manifest)
            if problems:
                raise RunArtifactError(
                    f"internal consistency check failed before finalize: {problems}"
                )
            self._tmp_dir.rename(self._final_dir)
            self._staging_created = False
        except BaseException:
            self._discard_staging()
            raise

        self._finalized = True
        return manifest

    def _result_features_by_key(
        self,
    ) -> dict[tuple[SourceObservationId, FeatureId], list[VisualFeature]]:
        """Index every result feature by ``(source_observation_id, feature_id)``."""
        features_by_key: dict[tuple[SourceObservationId, FeatureId], list[VisualFeature]] = {}
        for result in self._results:
            for feature in result.features:
                key = (result.source_observation_id, feature.feature_id)
                features_by_key.setdefault(key, []).append(feature)
        return features_by_key

    def _validate_feature_payload_references(self) -> None:
        """Require each queued payload to match one feature in its owning result."""
        features_by_key = self._result_features_by_key()

        metadata_fields = (
            "scope",
            "embedding_space_id",
            "shape",
            "dtype",
            "normalization",
            "payload_reference",
            "region_id",
            "provenance",
        )
        for feature, source_observation_id in self._feature_payloads:
            key = (source_observation_id, feature.feature_id)
            matches = features_by_key.get(key, [])
            if len(matches) != 1:
                raise RunArtifactError(
                    "queued feature payload does not resolve to exactly one result feature: "
                    f"source_observation_id={source_observation_id!r}, "
                    f"feature_id={feature.feature_id!r}, matches={len(matches)}"
                )
            result_feature = matches[0]
            for field_name in metadata_fields:
                queued_value = getattr(feature, field_name)
                result_value = getattr(result_feature, field_name)
                if queued_value != result_value:
                    raise RunArtifactError(
                        "queued feature payload metadata disagrees with result feature: "
                        f"{field_name} {queued_value!r} != {result_value!r} for "
                        f"source_observation_id={source_observation_id!r}, "
                        f"feature_id={feature.feature_id!r}"
                    )

    def _validate_feature_diagnostics(self) -> None:
        """Require each produced feature diagnostic to describe one persisted result feature.

        ``metrics/feature-extraction.jsonl`` is contractual: its ``dense`` record rebuilds the
        feature-to-image mapping after the run is reopened. A record that only hashes correctly
        could belong to another payload, so every ``SUCCEEDED``/``WARNING`` diagnostic is bound
        to exactly one ``VisualFeature`` of this run (same observation and feature identity),
        and to at most one diagnostic per feature so the geometry is unambiguous.
        """
        features_by_key = self._result_features_by_key()
        described: set[tuple[SourceObservationId, FeatureId]] = set()
        for diagnostic in self._feature_diagnostics:
            feature_id = diagnostic.feature_id
            # FeatureExtractionDiagnostic garante feature_id nos eventos produzidos; o teste
            # de None só estreita o tipo para o índice.
            if not diagnostic.produced_feature or feature_id is None:
                continue
            key = (diagnostic.source_observation_id, feature_id)
            matches = features_by_key.get(key, [])
            if len(matches) != 1:
                raise RunArtifactError(
                    "feature diagnostic does not resolve to exactly one result feature: "
                    f"event_id={diagnostic.event_id!r}, "
                    f"source_observation_id={diagnostic.source_observation_id!r}, "
                    f"feature_id={feature_id!r}, matches={len(matches)}"
                )
            if key in described:
                raise RunArtifactError(
                    "more than one feature diagnostic describes the same result feature: "
                    f"source_observation_id={diagnostic.source_observation_id!r}, "
                    f"feature_id={feature_id!r}"
                )
            described.add(key)
            self._validate_diagnostic_matches_feature(diagnostic, matches[0])

    def _validate_diagnostic_matches_feature(
        self, diagnostic: FeatureExtractionDiagnostic, feature: VisualFeature
    ) -> None:
        """Compare a produced diagnostic with its result feature and, for dense, its geometry."""
        space = diagnostic.embedding_space
        comparisons = (
            ("scope", diagnostic.scope, feature.scope),
            ("output_shape", diagnostic.output_shape, feature.shape),
            ("dtype", diagnostic.dtype, feature.dtype),
            ("normalization", diagnostic.normalization, feature.normalization),
            ("payload_reference", diagnostic.payload_reference, feature.payload_reference),
            ("backend provenance", diagnostic.backend, feature.provenance),
            (
                "embedding_space fingerprint",
                embedding_space_fingerprint(space) if space is not None else None,
                feature.embedding_space_id,
            ),
        )
        context = (
            f"event_id={diagnostic.event_id!r}, "
            f"source_observation_id={diagnostic.source_observation_id!r}, "
            f"feature_id={feature.feature_id!r}"
        )
        for name, declared, persisted in comparisons:
            if declared != persisted:
                raise RunArtifactError(
                    "feature diagnostic disagrees with its result feature: "
                    f"{name} {declared!r} != {persisted!r} for {context}"
                )
        region = diagnostic.region
        if region is not None and region.region_id != feature.region_id:
            raise RunArtifactError(
                "feature diagnostic disagrees with its result feature: "
                f"region_id {region.region_id!r} != {feature.region_id!r} for {context}"
            )
        if diagnostic.dense is not None:
            self._validate_dense_geometry(diagnostic, diagnostic.dense, feature, context)

    def _validate_dense_geometry(
        self,
        diagnostic: FeatureExtractionDiagnostic,
        dense: DenseFeatureDiagnostic,
        feature: VisualFeature,
        context: str,
    ) -> None:
        """Require the persisted dense geometry to rebuild the feature's ``DenseFeatureMap``."""
        if dense.source_artifact_id != str(self._run_id):
            raise RunArtifactError(
                "dense feature diagnostic source_artifact_id does not point to the run that "
                f"owns the feature: {dense.source_artifact_id!r} != {str(self._run_id)!r} "
                f"for {context}"
            )
        # Reconstrói o mapa como um consumidor de metrics/ o faria: assim shape 3-D e
        # (grid_height, grid_width) == feature.shape[:2] são as mesmas regras de DenseFeatureMap.
        try:
            DenseFeatureMap(
                feature=feature,
                sampling=DenseFeatureSampling(
                    grid_width=dense.grid_width,
                    grid_height=dense.grid_height,
                    source_image_width=diagnostic.source_image_width,
                    source_image_height=diagnostic.source_image_height,
                    origin_x=dense.origin_x,
                    origin_y=dense.origin_y,
                    stride_x=dense.stride_x,
                    stride_y=dense.stride_y,
                    support_width=dense.support_width,
                    support_height=dense.support_height,
                    coordinate_transform_id=dense.coordinate_transform_id,
                ),
                source_artifact_id=dense.source_artifact_id,
            )
        except ValueError as error:
            raise RunArtifactError(
                f"dense feature diagnostic does not rebuild a valid DenseFeatureMap: {error} "
                f"for {context}"
            ) from error

    def _validate_semantic_request_evidence(self, request: Any) -> PerceptionResult:
        """Require one semantic request to name evidence this run actually persisted.

        Shared by both streams: whether the response parsed or not, the request had to be
        issued against a real result, a real region, real features whose payloads exist and a
        real scene context. Only claim materialization differs, since a rejected response
        produced no claims.

        Returns:
            The single result the request resolves to.

        Raises:
            RunArtifactError: If any referenced evidence does not resolve exactly.
        """
        matches = [
            result
            for result in self._results
            if result.result_id == request.perception_result_id
            and result.source_observation_id == request.source_observation_id
        ]
        if len(matches) != 1:
            raise RunArtifactError(
                "semantic request does not resolve to exactly one result: "
                f"request_id={request.request_id!r}, matches={len(matches)}"
            )
        result = matches[0]
        if True:
            if request.region_id is not None and all(
                region.region_id != request.region_id for region in result.regions
            ):
                raise RunArtifactError(
                    "semantic execution region_id does not resolve in its result: "
                    f"{request.region_id!r}"
                )
            payload_keys = {
                (source_observation_id, feature.feature_id)
                for feature, source_observation_id in self._feature_payloads
            }
            for feature_reference in request.visual_features:
                feature_matches = [
                    feature
                    for feature in result.features
                    if feature.feature_id == feature_reference.feature_id
                    and feature.embedding_space_id == feature_reference.embedding_space_id
                    and feature.scope is feature_reference.scope
                    and feature.region_id == feature_reference.region_id
                ]
                if len(feature_matches) != 1:
                    raise RunArtifactError(
                        "semantic visual feature does not resolve exactly in its result: "
                        f"{feature_reference.feature_id!r}"
                    )
                payload_key = (result.source_observation_id, feature_reference.feature_id)
                if payload_key not in payload_keys:
                    raise RunArtifactError(
                        "semantic visual feature payload was not persisted: "
                        f"{feature_reference.feature_id!r}"
                    )
            context_reference = request.scene_context_reference
            if context_reference is not None:
                context_matches = [
                    candidate
                    for candidate in self._results
                    if str(candidate.result_id) == context_reference.evidence_id
                    and candidate.source_observation_id == request.source_observation_id
                    and candidate.scene_context is not None
                ]
                if len(context_matches) != 1:
                    raise RunArtifactError(
                        "semantic scene_context_reference does not resolve exactly: "
                        f"{context_reference.evidence_id!r}"
                    )
        return result

    def _validate_failed_semantic_interpretations(self) -> None:
        """Hold failures to the same identity and input-evidence invariants as successes.

        A contractual stream must not accept a record naming a result, region, feature or
        scene context that does not belong to this run. Claim materialization is deliberately
        excluded: a rejected response produced none.
        """
        for failed in self._semantic_failures:
            self._validate_semantic_request_evidence(failed.request)

    def _validate_semantic_execution_materialization(self) -> None:
        """Require every semantic execution to match canonical persisted evidence."""
        for execution in self._semantic_executions:
            result = self._validate_semantic_request_evidence(execution.request)
            missing_claims = [
                claim.claim_id for claim in execution.parsed.claims if claim not in result.claims
            ]
            if missing_claims:
                raise RunArtifactError(
                    "semantic execution claims were not materialized in its result: "
                    f"{missing_claims!r}"
                )
            parsed_context = execution.parsed.scene_context
            if parsed_context is not None and parsed_context != result.scene_context:
                raise RunArtifactError(
                    "semantic execution scene_context was not materialized in its result"
                )

    def _validate_semantic_view_payloads(self) -> None:
        """Require every referenced semantic view to be content-addressed in this run.

        Both streams count. The view of a rejected response is the exact visual evidence that
        produced it, so a failure naming a payload this run never persisted leaves the artifact
        citing evidence nobody can recover.
        """
        referenced: dict[str, str] = {}
        requests = [execution.request for execution in self._semantic_executions]
        requests.extend(failed.request for failed in self._semantic_failures)
        for request in requests:
            for view in request.visual_views:
                previous_hash = referenced.setdefault(view.payload_reference, view.sha256)
                if previous_hash != view.sha256:
                    raise RunArtifactError(
                        "semantic view reference has conflicting hashes: "
                        f"{view.payload_reference!r}"
                    )
        missing = sorted(set(referenced) - self._semantic_view_payloads.keys())
        if missing:
            raise RunArtifactError(f"missing semantic view payloads: {missing!r}")
        unused = sorted(self._semantic_view_payloads.keys() - set(referenced))
        if unused:
            raise RunArtifactError(f"semantic view payloads are not referenced: {unused!r}")

    def _write_mask_index(self, file_entries: list[RunArtifactFileEntry]) -> None:
        """Index the masks already persisted by :meth:`_persist_result_masks`.

        Masks are never opt-in the way feature payloads are (:meth:`add_feature_payload`):
        every region that arrives with a materialized ``mask`` through :meth:`add_result`
        is persisted right there (#378), keyed by ``(source_observation_id, region_id)``
        since ``RegionId`` is local to one result. Only the index and the inventory
        entries are left for finalize, so this costs nothing per frame.
        """
        if self._mask_store is None:
            return

        mask_store_root = self._tmp_dir / _MASKS_DIRNAME
        write_mask_index(mask_store_root, self._mask_store.entries())
        for entry in self._mask_store.entries():
            file_entries.append(
                RunArtifactFileEntry(
                    path=f"{_MASKS_DIRNAME}/{entry.payload_reference}",
                    size_bytes=entry.size_bytes,
                    content_hash=entry.content_hash,
                )
            )
        index_path = mask_store_root / MASK_INDEX_FILENAME
        file_entries.append(
            _file_entry(f"{_MASKS_DIRNAME}/{MASK_INDEX_FILENAME}", index_path.read_bytes())
        )

    def _write_contents(self) -> RunArtifactManifest:
        file_entries: list[RunArtifactFileEntry] = []

        self._write_mask_index(file_entries)

        results_content = "".join(
            f"{json.dumps(encode_perception_result(result), sort_keys=True)}\n"
            for result in self._results
        )
        results_path = self._tmp_dir / _RESULTS_FILENAME
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(results_content, encoding="utf-8")
        file_entries.append(_file_entry(_RESULTS_FILENAME, results_content.encode("utf-8")))

        metrics_content = "".join(
            f"{json.dumps(record, sort_keys=True)}\n" for record in self._stage_outcome_records
        )
        metrics_path = self._tmp_dir / _METRICS_FILENAME
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(metrics_content, encoding="utf-8")
        file_entries.append(_file_entry(_METRICS_FILENAME, metrics_content.encode("utf-8")))

        # Os bytes ja foram gravados em add_semantic_view_payload(); so falta inventaria-los.
        # Independente de haver execucao bem-sucedida: uma view pode pertencer so a uma failure.
        for payload_reference, (digest, size_bytes) in sorted(self._semantic_view_payloads.items()):
            file_entries.append(
                RunArtifactFileEntry(
                    path=payload_reference,
                    size_bytes=size_bytes,
                    content_hash=f"sha256:{digest}",
                )
            )

        if self._semantic_executions:
            semantic_records: list[dict[str, Any]] = []
            for execution in self._semantic_executions:
                raw_reference = _semantic_raw_response_reference(execution)
                _validate_semantic_raw_response_reference(execution, raw_reference)
                audit_paths = write_semantic_audit(
                    run_root=self._tmp_dir,
                    execution=execution,
                    debug_level=self._semantic_debug_level,
                )
                for relative_path in audit_paths:
                    file_entries.append(
                        _file_entry(relative_path, (self._tmp_dir / relative_path).read_bytes())
                    )
                semantic_records.append(
                    encode_semantic_execution(
                        execution,
                        raw_response_reference=raw_reference,
                    )
                )
            semantic_content = "".join(
                f"{json.dumps(record, sort_keys=True)}\n" for record in semantic_records
            )
            semantic_path = self._tmp_dir / _SEMANTIC_EXECUTIONS_FILENAME
            semantic_path.parent.mkdir(parents=True, exist_ok=True)
            semantic_path.write_text(semantic_content, encoding="utf-8")
            file_entries.append(
                _file_entry(_SEMANTIC_EXECUTIONS_FILENAME, semantic_content.encode("utf-8"))
            )

        for failed in self._semantic_failures:
            # Mesmo diretorio por request das execucoes bem-sucedidas, mesmo writer.
            for relative_path in write_semantic_audit(
                run_root=self._tmp_dir,
                execution=failed,
                debug_level=self._semantic_debug_level,
            ):
                file_entries.append(
                    _file_entry(relative_path, (self._tmp_dir / relative_path).read_bytes())
                )

        if True:
            # Escrito SEMPRE, mesmo vazio e mesmo sem nenhuma tentativa semantica: assim a
            # ausencia do arquivo significa inequivocamente "artifact anterior ao tracking",
            # sem heuristica. Emiti-lo so quando havia tentativa deixava um buraco: um artifact
            # legado em que toda tentativa falhou tambem tem zero execucoes e nenhuma stream,
            # e passaria por completo.
            #
            # Stream contratual proprio: a resposta invalida continua sendo evidencia observada,
            # e mante-la fora de semantic-interpretations.jsonl preserva o contrato de sucesso
            # (e todo artifact ja escrito sob esta versao de schema).
            failures_content = "".join(
                f"{json.dumps(encode_failed_semantic_interpretation(failed), sort_keys=True)}\n"
                for failed in self._semantic_failures
            )
            failures_path = self._tmp_dir / _SEMANTIC_FAILURES_FILENAME
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            failures_path.write_text(failures_content, encoding="utf-8")
            file_entries.append(
                _file_entry(_SEMANTIC_FAILURES_FILENAME, failures_content.encode("utf-8"))
            )

        if self._feature_store is not None:
            feature_store_root = self._tmp_dir / _FEATURES_DIRNAME
            # Os arrays ja foram gravados em add_feature_payload(); aqui so resta o indice.
            write_feature_index(feature_store_root, self._feature_store.entries())
            for entry in self._feature_store.entries():
                file_entries.append(
                    RunArtifactFileEntry(
                        path=f"{_FEATURES_DIRNAME}/{entry.payload_reference}",
                        size_bytes=entry.size_bytes,
                        content_hash=entry.content_hash,
                    )
                )
            index_path = feature_store_root / FEATURE_INDEX_FILENAME
            file_entries.append(
                _file_entry(
                    f"{_FEATURES_DIRNAME}/{FEATURE_INDEX_FILENAME}", index_path.read_bytes()
                )
            )

        diagnostic_paths = write_feature_diagnostics(
            run_root=self._tmp_dir,
            diagnostics=self._feature_diagnostics,
            previews=self._feature_previews,
            debug_level=self._feature_debug_level,
        )
        for relative_path in diagnostic_paths:
            file_entries.append(
                _file_entry(relative_path, (self._tmp_dir / relative_path).read_bytes())
            )

        manifest = RunArtifactManifest(
            run_id=self._run_id,
            run_index=self._run_index,
            sequence_name=self._sequence_name,
            sequence_artifact_id=self._sequence_artifact_id,
            selection_id=self._selection_id,
            enabled_capabilities=self._enabled_capabilities,
            pipeline_preset=self._pipeline_preset,
            configuration_digest=self._configuration_digest,
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            result_count=len(self._results),
            region_count=sum(len(result.regions) for result in self._results),
            feature_count=sum(len(result.features) for result in self._results),
            claim_count=sum(
                len(result.claims)
                + (0 if result.scene_context is None else len(result.scene_context.claims))
                for result in self._results
            ),
            file_inventory=tuple(sorted(file_entries, key=lambda entry: entry.path)),
        )

        manifest_path = self._tmp_dir / _MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True), encoding="utf-8"
        )

        readme_path = self._tmp_dir / _README_FILENAME
        readme_path.write_text(_render_readme(manifest), encoding="utf-8")

        return manifest


class PerceptionRunReader:
    """Reads a finalized perception run artifact from the local filesystem."""

    def __init__(self, run_dir: Path) -> None:
        """Open a perception run for reading.

        Args:
            run_dir: Path to the run's directory.

        Raises:
            IncompleteRunArtifactError: If ``manifest.json`` is missing.
            RunArtifactError: If the manifest's schema version is not
                understood by this module.
        """
        self._root = run_dir
        self._manifest = _load_manifest(run_dir)

    @property
    def manifest(self) -> RunArtifactManifest:
        """The run's manifest."""
        return self._manifest

    def feature_store(self) -> FeatureStoreReader:
        """Open this run's feature payload store, without loading any array.

        Returns:
            A reader over ``outputs/features/`` — empty (no
            ``feature_keys()``) when this run never persisted a feature
            payload, since persisting is opt-in per feature (see
            :meth:`PerceptionRunWriter.add_feature_payload`).
        """
        return FeatureStoreReader.open(self._root / _FEATURES_DIRNAME)

    def mask_store(self) -> MaskStoreReader:
        """Open this run's region-mask payload store, without loading any pixels.

        ``list_results()`` never materializes mask pixels (#378): a
        decoded ``Region2D.mask`` is always ``None``. A consumer that
        needs the actual pixels for a region whose ``mask_reference`` is
        set (e.g. Sensor Association mask membership) loads and
        hash-verifies them on demand through this reader.

        Returns:
            A reader over ``outputs/masks/`` — empty (no
            ``region_keys()``) when no region of this run carried a mask.
        """
        return MaskStoreReader.open(self._root / _MASKS_DIRNAME)

    def iter_results(self) -> Iterator[PerceptionResult]:
        """Yield each result in turn, decoding one line of ``outputs/results.jsonl`` at a time.

        Prefer this to :meth:`list_results` whenever the caller processes results one by
        one: a run's resident cost then does not grow with its frame count. The same
        ``Region2D.mask`` rule applies — pixels are never materialized here, only
        ``mask_reference`` (see :meth:`mask_store`).
        """
        results_path = self._root / _RESULTS_FILENAME
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield decode_perception_result(json.loads(stripped))

    def list_results(self) -> list[PerceptionResult]:
        """Return every result in this run.

        Returns:
            All results, decoded from ``outputs/results.jsonl``. This holds the whole run
            in memory; :meth:`iter_results` streams it instead.
        """
        return list(self.iter_results())

    def iter_semantic_executions(self) -> Iterator[SemanticInterpretationExecution]:
        """Yield each persisted semantic execution in turn.

        Every record carries its full ``raw_response`` text, so a run's executions are
        markedly heavier than its results; stream them unless all are needed at once.
        """
        executions_path = self._root / _SEMANTIC_EXECUTIONS_FILENAME
        if not executions_path.is_file():
            return
        with executions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                raw_reference = record["raw_response_reference"]
                try:
                    yield decode_semantic_execution(record)
                except (ValueError, KeyError, TypeError) as error:
                    raise RunArtifactError(
                        f"invalid semantic execution record for {raw_reference!r}: {error}"
                    ) from error

    def tracks_semantic_failures(self) -> bool:
        """Whether this run recorded its rejected interpretations at all.

        Decided by the manifest, not by what happens to be on disk. The writer always
        inventories the stream, so its absence from ``file_inventory`` can only mean the
        artifact predates tracking. A consumer needs that to tell "zero failures, fully
        tracked" from "never tracked", which otherwise both read as zero.

        Raises:
            RunArtifactError: If the manifest inventories the stream but the file is gone.
                That is a corrupted artifact, and silently reporting it as a legacy one would
                turn corruption into backward compatibility.
        """
        inventoried = any(
            entry.path == _SEMANTIC_FAILURES_FILENAME for entry in self._manifest.file_inventory
        )
        if not inventoried:
            return False
        if not (self._root / _SEMANTIC_FAILURES_FILENAME).is_file():
            raise RunArtifactError(
                "run inventoried its semantic failures stream but the file is missing: "
                f"{_SEMANTIC_FAILURES_FILENAME}"
            )
        return True

    def iter_failed_semantic_interpretations(self) -> Iterator[FailedSemanticInterpretation]:
        """Yield each real backend call whose response was observed but never materialized.

        Empty when the run recorded none, including every run written before this stream
        existed: its absence is not an error, so artifacts frozen under the same schema
        version stay readable.
        """
        failures_path = self._root / _SEMANTIC_FAILURES_FILENAME
        if not failures_path.is_file():
            return
        with failures_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield decode_failed_semantic_interpretation(json.loads(stripped))
                except (ValueError, KeyError, TypeError) as error:
                    raise RunArtifactError(
                        f"invalid failed semantic interpretation record: {error}"
                    ) from error

    def list_failed_semantic_interpretations(self) -> list[FailedSemanticInterpretation]:
        """Return every observed-but-unparsed interpretation of this run."""
        return list(self.iter_failed_semantic_interpretations())

    def list_semantic_executions(self) -> list[SemanticInterpretationExecution]:
        """Return persisted semantic requests, prompts, responses, and diagnostics."""
        return list(self.iter_semantic_executions())

    def result(self, source_observation_id: SourceObservationId) -> PerceptionResult:
        """Return this run's result for one physical observation.

        Args:
            source_observation_id: The observation to look up.

        Returns:
            The matching result.

        Raises:
            RunArtifactError: If this run has no result for that
                observation.
        """
        for result in self.iter_results():
            if result.source_observation_id == source_observation_id:
                return result
        raise RunArtifactError(f"no result for source_observation_id={source_observation_id!r}")

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            A list of human-readable problems; empty means no problem
            found.
        """
        return _check_file_inventory(self._root, self._manifest)


def _render_readme(manifest: RunArtifactManifest) -> str:
    capabilities = ", ".join(sorted(manifest.enabled_capabilities)) or "(none)"
    return (
        f"# Perception run {manifest.run_index:04d}\n"
        "\n"
        f"- Sequence: `{manifest.sequence_name}` (`{manifest.sequence_artifact_id}`)\n"
        f"- Selection: `{manifest.selection_id}`\n"
        f"- Run ID: `{manifest.run_id}`\n"
        f"- Enabled capabilities: {capabilities}\n"
        f"- Pipeline preset: `{manifest.pipeline_preset.preset_id}`\n"
        f"- Configuration digest: `{manifest.configuration_digest}`\n"
        f"- Created at: {manifest.created_at}\n"
        "\n"
        "## Results\n"
        "\n"
        f"- Processed observations: {manifest.result_count}\n"
        f"- Regions: {manifest.region_count}\n"
        f"- Features: {manifest.feature_count}\n"
        f"- Claims: {manifest.claim_count}\n"
    )


def _encode_stage_outcome(outcome: StageOutcome) -> dict[str, Any]:
    return {
        "stage_id": outcome.stage_id,
        "status": outcome.status.value,
        "duration_ms": outcome.duration_ms,
        "error": outcome.error,
    }


def _semantic_raw_response_reference(execution: SemanticInterpretationExecution) -> str:
    request_id = str(execution.request.request_id)
    request_path = PurePosixPath(request_id)
    if request_path.name != request_id or request_id in {".", ".."}:
        raise RunArtifactError(f"semantic request_id is not a safe path segment: {request_id!r}")
    return f"{_SEMANTIC_DEBUG_ROOT}/{request_id}/raw-response.txt"


def _validate_semantic_raw_response_reference(
    execution: SemanticInterpretationExecution,
    expected_reference: str,
) -> None:
    provenances = [claim.provenance for claim in execution.parsed.claims]
    if execution.parsed.scene_context is not None:
        provenances.append(execution.parsed.scene_context.provenance)
        provenances.extend(claim.provenance for claim in execution.parsed.scene_context.claims)
    mismatches = {
        provenance.raw_response_reference
        for provenance in provenances
        if provenance.raw_response_reference != expected_reference
    }
    if mismatches:
        raise RunArtifactError(
            "semantic provenance raw_response_reference does not match the materialized path: "
            f"expected {expected_reference!r}, found {sorted(mismatches, key=str)!r}"
        )


def _with_persisted_masks(
    result: PerceptionResult, mask_references: dict[tuple[SourceObservationId, RegionId], str]
) -> PerceptionResult:
    """Return ``result`` with each mask-bearing region's persisted reference attached.

    ``encode_region()`` never inlines pixel data (#378): it only ever
    encodes ``mask_reference``. This builds the exact ``Region2D`` view
    that reference belongs to, without mutating ``result`` or the
    original in-memory mask.
    """
    if not mask_references:
        return result
    patched_regions = tuple(
        _region_with_mask_reference(result.source_observation_id, region, mask_references)
        for region in result.regions
    )
    if patched_regions == result.regions:
        return result
    return replace(result, regions=patched_regions)


def _region_with_mask_reference(
    source_observation_id: SourceObservationId,
    region: Region2D,
    mask_references: dict[tuple[SourceObservationId, RegionId], str],
) -> Region2D:
    if region.mask is None:
        return region
    reference = mask_references.get((source_observation_id, region.region_id))
    if reference is None:
        return region
    # Os pixels ja estao no mask store; solta-los aqui e o que impede a RAM de crescer
    # com o numero de frames (uma mascara 640x480 custa ~2,36 MB viva).
    return replace(region, mask_reference=reference, mask=None)


def _file_entry(relative_path: str, data: bytes) -> RunArtifactFileEntry:
    digest = hashlib.sha256(data).hexdigest()
    return RunArtifactFileEntry(
        path=relative_path, size_bytes=len(data), content_hash=f"sha256:{digest}"
    )


def _manifest_to_dict(manifest: RunArtifactManifest) -> dict[str, Any]:
    return {
        "run_id": str(manifest.run_id),
        "run_index": manifest.run_index,
        "sequence_name": manifest.sequence_name,
        "sequence_artifact_id": manifest.sequence_artifact_id,
        "selection_id": manifest.selection_id,
        "enabled_capabilities": sorted(manifest.enabled_capabilities),
        "pipeline_preset": encode_pipeline_preset(manifest.pipeline_preset),
        "configuration_digest": manifest.configuration_digest,
        "schema_version": manifest.schema_version,
        "created_at": manifest.created_at,
        "result_count": manifest.result_count,
        "region_count": manifest.region_count,
        "feature_count": manifest.feature_count,
        "claim_count": manifest.claim_count,
        "file_inventory": [
            {"path": entry.path, "size_bytes": entry.size_bytes, "content_hash": entry.content_hash}
            for entry in manifest.file_inventory
        ],
    }


def _load_manifest(run_dir: Path) -> RunArtifactManifest:
    manifest_path = run_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise IncompleteRunArtifactError(f"missing {_MANIFEST_FILENAME} in {run_dir}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise RunArtifactError(f"unsupported run artifact schema_version: {schema_version!r}")

    return RunArtifactManifest(
        run_id=PerceptionRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        sequence_artifact_id=raw["sequence_artifact_id"],
        selection_id=raw["selection_id"],
        enabled_capabilities=frozenset(raw["enabled_capabilities"]),
        pipeline_preset=decode_pipeline_preset(raw["pipeline_preset"]),
        configuration_digest=raw["configuration_digest"],
        schema_version=schema_version,
        created_at=raw["created_at"],
        result_count=raw["result_count"],
        region_count=raw["region_count"],
        feature_count=raw["feature_count"],
        claim_count=raw["claim_count"],
        file_inventory=tuple(
            RunArtifactFileEntry(
                path=entry["path"],
                size_bytes=entry["size_bytes"],
                content_hash=entry["content_hash"],
            )
            for entry in raw["file_inventory"]
        ),
    )


def _check_file_inventory(root: Path, manifest: RunArtifactManifest) -> list[str]:
    problems: list[str] = []
    for entry in manifest.file_inventory:
        file_path = root / entry.path
        if not file_path.is_file():
            problems.append(f"missing file referenced by manifest: {entry.path}")
            continue
        data = file_path.read_bytes()
        if len(data) != entry.size_bytes:
            problems.append(
                f"size mismatch for {entry.path}: expected {entry.size_bytes}, found {len(data)}"
            )
            continue
        digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
        if digest != entry.content_hash:
            problems.append(f"content hash mismatch for {entry.path}")
    return problems

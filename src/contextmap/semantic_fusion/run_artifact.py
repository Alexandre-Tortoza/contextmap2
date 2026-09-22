"""Persisted, immutable, self-describing Semantic Fusion run artifacts.

A ``SemanticFusionRunArtifact`` is a directory holding the fusion supports of one run and
the evidence fused over each of them, with the lineage, metrics and skipped evidence needed
to trust them. It opens without NumPy, a perception runtime or a model library, and one
support or one contribution can be read by identity without loading the others. See
``src/contextmap/semantic_fusion/docs/artifact.md`` for the layout and the trade-offs of this
first schema.

Contractual data lives in ``outputs/`` and ``metrics/`` and is inventoried with size and
SHA-256 in the manifest; ``debug/`` is human evidence that is never inventoried, so removing it
cannot invalidate the run and Semantic Mapping may not depend on it. Writing follows
:class:`~contextmap.shared.AtomicRunDirectory`: an interrupted write never looks like a
finished run and a finished run is never modified.

Nothing upstream is duplicated: geometry is stored as positional deltas, and claims, scores,
features, quality and point representations are only referenced. Every hypothesis is kept with
all its alternatives, conflicts, abstentions and unscored evidence; the artifact never stores
just a primary hypothesis.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, NewType, TypeVar

from contextmap.geometric_mapping import MapId
from contextmap.point_representation import PointRepresentationRunId
from contextmap.semantic_fusion.models import (
    EvidenceContribution,
    EvidenceContributionId,
    EvidenceStance,
    FusedEvidence,
    FusionSupport,
    FusionSupportId,
    SupportSignalKind,
    UncertaintyKind,
    _require_canonical,
)
from contextmap.semantic_fusion.serialization import (
    decode_fused_evidence,
    decode_fusion_support,
    encode_fused_evidence,
    encode_fusion_support,
)
from contextmap.semantic_fusion.support import ExcludedObservation
from contextmap.sensor_association import SpatialObservationId
from contextmap.shared import (
    AtomicRunDirectory,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
    next_run_index,
    write_run_registry,
)
from contextmap.visual_perception import PerceptionRunId

SCHEMA_VERSION = "0.2.0"
"""Semantic Fusion run artifact schema version written and understood by this module.

Bumped to ``0.2.0`` when ``metrics/counts.json`` redefined ``inference_results``: ``0.1.0``
summed the results of every support, so a result whose regions fall in several supports was
counted once per support; ``0.2.0`` counts the distinct results of the whole run, like
``physical_observations``. The metrics are contractual and inventoried, so the same field with
two denominators must not share a version. This is a pre-1.0 schema: no compatibility reader
for ``0.1.0`` is kept and such a run is refused when opened.
"""

SemanticFusionRunId = NewType("SemanticFusionRunId", str)
"""Identity of one Semantic Fusion run."""

_MANIFEST = "manifest.json"
_SUPPORTS = "outputs/fusion-supports.jsonl"
_EVIDENCE = "outputs/fused-evidence.jsonl"
_SUPPORT_INDEX = "outputs/support-observation-index.jsonl"
_HYPOTHESIS_INDEX = "outputs/hypothesis-evidence-index.jsonl"
_GROUPS = "outputs/physical-observation-groups.jsonl"
_CONTRIBUTION_INDEX = "outputs/contribution-index.jsonl"
_EXCLUDED = "outputs/excluded-observations.jsonl"
_COUNTS = "metrics/counts.json"
_DISTRIBUTIONS = "metrics/distributions.json"
_PAYLOAD = "metrics/payload.json"
_RUNTIME = "metrics/runtime.json"

_DEBUG_SUPPORT_LIMIT = 20
"""Supports that get debug evidence at the ``standard`` level; ``full`` covers all of them."""

_T = TypeVar("_T")


class FusionRunArtifactError(Exception):
    """Base class for Semantic Fusion run artifact read/write failures."""


class IncompleteFusionRunArtifactError(FusionRunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class SemanticFusionDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and required metrics only.
        STANDARD: Adds a summary, the hypotheses, the conflicts and the physical observations
            of a sample of supports.
        FULL: The same for every support, plus a per-contribution trace and a geometry summary.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, kw_only=True)
class FusionRunLineage:
    """The upstream artifacts a fusion run consumed.

    Attributes:
        sequence_artifact_id: The canonical sequence everything was built from.
        geometric_map_id: The immutable geometric map the geometry belongs to.
        association_run_ids: The selected Sensor Association runs, sorted and unique.
        perception_run_ids: The selected perception runs whose claims were fused, sorted and
            unique.
        point_representation_run_ids: The Point Representation runs selected for the run, whose
            structure a declared channel may reference, sorted and unique; empty when none was
            selected. It records the selection, not the use: in a channel ablation every arm
            lists the same runs, so that the arms see identical upstream artifacts.
    """

    sequence_artifact_id: str
    geometric_map_id: MapId
    association_run_ids: tuple[str, ...]
    perception_run_ids: tuple[PerceptionRunId, ...]
    point_representation_run_ids: tuple[PointRepresentationRunId, ...] = ()

    def __post_init__(self) -> None:
        """Validate that every selection is explicit, sorted and unique.

        Raises:
            ValueError: If an identity is empty, a required selection is empty, or a
                selection is not sorted and unique.
        """
        if not self.sequence_artifact_id.strip() or not self.geometric_map_id.strip():
            raise ValueError("sequence_artifact_id and geometric_map_id must not be empty")
        for name in ("association_run_ids", "perception_run_ids"):
            items: tuple[str, ...] = getattr(self, name)
            if not items:
                raise ValueError(f"{name} must not be empty: upstream runs are selected explicitly")
        for name in ("association_run_ids", "perception_run_ids", "point_representation_run_ids"):
            _require_canonical(name, getattr(self, name), lambda item: (item,))


@dataclass(frozen=True, kw_only=True)
class FusionOutcome:
    """One support and the evidence fused over it.

    Attributes:
        support: The spatial support.
        evidence: The evidence fused over exactly that support.
    """

    support: FusionSupport
    evidence: FusedEvidence

    def __post_init__(self) -> None:
        """Require that the evidence belongs to the support.

        Raises:
            ValueError: If the two name different ``fusion_support_id`` values.
        """
        if self.support.fusion_support_id != self.evidence.fusion_support_id:
            raise ValueError(
                f"fusion_support_id {self.support.fusion_support_id!r} does not match the "
                f"evidence's {self.evidence.fusion_support_id!r}"
            )


@dataclass(frozen=True, kw_only=True)
class SemanticFusionRunManifest:
    """Authoritative metadata of a persisted Semantic Fusion run.

    Attributes:
        run_id: Identity of the run.
        run_index: Monotonic index within this sequence's semantic-fusion runs.
        sequence_name: Name of the processed sequence.
        lineage: The upstream artifacts the run consumed.
        grouping_policy_id: The physical-observation grouping policy, ``None`` for an empty run.
        support_policy_id: The support construction policy, ``None`` for an empty run.
        support_configuration_fingerprint: Hash of the support configuration.
        fusion_policy_id: The fusion policy, ``None`` for an empty run.
        fusion_configuration_fingerprint: Hash of the fusion configuration.
        evidence_identities: Per evidence channel, the identities (interpreters, scorers,
            embedding spaces, quality definitions, representation spaces, map) that fed it.
        code_version: Code revision that produced the run.
        support_count: Fusion supports persisted.
        fused_evidence_count: Fused evidence records persisted.
        contribution_count: Contributions across all supports.
        excluded_count: Observations left out of every support, persisted explicitly.
        warnings: Human-readable warnings of the run.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes the manifest, the
            README and ``debug/``.
    """

    run_id: SemanticFusionRunId
    run_index: int
    sequence_name: str
    lineage: FusionRunLineage
    grouping_policy_id: str | None
    support_policy_id: str | None
    support_configuration_fingerprint: str | None
    fusion_policy_id: str | None
    fusion_configuration_fingerprint: str | None
    evidence_identities: Mapping[str, tuple[str, ...]]
    code_version: str
    support_count: int
    fused_evidence_count: int
    contribution_count: int
    excluded_count: int
    warnings: tuple[str, ...]
    debug_level: str
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


def _sequence_dir(workspace_root: Path, sequence_name: str) -> Path:
    return workspace_root / "runs" / "semantic-fusion" / sequence_name


class SemanticFusionRunWriter:
    """Builds an immutable Semantic Fusion run artifact on the local filesystem.

    Outcomes are streamed to disk as they are produced, so the run is not bounded by memory,
    and the run is published atomically when the stream ends.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        sequence_name: str,
        run_id: SemanticFusionRunId,
        run_index: int,
        selection_label: str,
        policy_label: str,
        lineage: FusionRunLineage,
        code_version: str,
        debug_level: SemanticFusionDebugLevel = SemanticFusionDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            workspace_root: Root of the local workspace.
            sequence_name: Name of the sequence the run processed.
            run_id: Identity of the run.
            run_index: Monotonic index for this sequence's runs (see
                :func:`allocate_fusion_run_index`).
            selection_label: Short readable description of the selected upstream runs, for the
                directory name.
            policy_label: Short readable description of the fusion policy, for the directory
                name.
            lineage: The explicit upstream selection.
            code_version: Code revision that produced the run.
            debug_level: Amount of non-contractual debug evidence to persist.
        """
        self._workspace_root = workspace_root
        self._sequence_name = sequence_name
        self._run_id = run_id
        self._run_index = run_index
        self._lineage = lineage
        self._code_version = code_version
        self._debug_level = debug_level
        self._final_dir = _sequence_dir(workspace_root, sequence_name) / (
            f"run-{run_index:04d}__{selection_label}__{policy_label}"
        )

    def write(
        self,
        outcomes: Iterable[FusionOutcome],
        *,
        excluded: Iterable[ExcludedObservation] = (),
        warnings: Sequence[str] = (),
        runtime: Mapping[str, float | int | None] | None = None,
    ) -> SemanticFusionRunManifest:
        """Persist a run atomically, consuming the outcomes as a stream.

        Args:
            outcomes: The supports with their fused evidence, sorted by support identity.
            excluded: Observations that took part in no support, kept as skipped evidence.
            warnings: Human-readable warnings to record.
            runtime: Figures such as seconds and peak memory, recorded apart from every
                quality measure and only when the caller measured them.

        Returns:
            The manifest of the finalized run.

        Raises:
            FusionRunArtifactError: If a run already exists at the target path, the outcomes are
                not strictly sorted by support identity, a support and its evidence disagree, an
                outcome breaks the lineage or the run's single policy, or the write fails.
        """
        tally = _Tally(self._lineage)
        try:
            with AtomicRunDirectory(self._final_dir) as run:
                self._stream(run, outcomes, tally)
                excluded_records = [_encode_excluded(item) for item in excluded]
                self._write_tables(run, tally, excluded_records)
                self._write_metrics(run, tally, excluded_records, warnings, runtime)
                run.publish(
                    manifest=self._manifest_record(tally, excluded_records, warnings),
                    readme=self._render_readme(tally, excluded_records),
                )
        except RunDirectoryError as error:
            raise FusionRunArtifactError(str(error)) from error
        rebuild_fusion_run_registry(
            workspace_root=self._workspace_root, sequence_name=self._sequence_name
        )
        return _load_manifest(self._final_dir)

    def _stream(
        self, run: AtomicRunDirectory, outcomes: Iterable[FusionOutcome], tally: _Tally
    ) -> None:
        with (
            run.open_binary(_SUPPORTS) as supports_out,
            run.open_binary(_EVIDENCE) as evidence_out,
        ):
            support_offset = evidence_offset = 0
            for outcome in outcomes:
                tally.check(outcome)
                support_line = _line(encode_fusion_support(outcome.support)) + b"\n"
                evidence_line = _line(encode_fused_evidence(outcome.evidence)) + b"\n"
                # O stream não expõe posição: os deslocamentos são contados aqui.
                tally.add(
                    outcome,
                    support_offset=support_offset,
                    support_length=len(support_line) - 1,
                    evidence_offset=evidence_offset,
                    evidence_length=len(evidence_line) - 1,
                )
                supports_out.write(support_line)
                evidence_out.write(evidence_line)
                support_offset += len(support_line)
                evidence_offset += len(evidence_line)
                self._write_debug(run, outcome, tally.support_count)
        tally.supports_bytes = support_offset
        tally.evidence_bytes = evidence_offset
        tally.output_sizes[_SUPPORTS] = tally.supports_bytes
        tally.output_sizes[_EVIDENCE] = tally.evidence_bytes

    def _write_tables(
        self, run: AtomicRunDirectory, tally: _Tally, excluded: list[dict[str, Any]]
    ) -> None:
        tables = {
            _SUPPORT_INDEX: tally.support_rows,
            _HYPOTHESIS_INDEX: tally.hypothesis_rows,
            _GROUPS: tally.group_rows,
            _CONTRIBUTION_INDEX: tally.contribution_rows,
            _EXCLUDED: excluded,
        }
        for path, rows in tables.items():
            text = "".join(_line(row).decode() + "\n" for row in rows)
            run.write_text(path, text)
            tally.output_sizes[path] = len(text.encode())

    def _write_metrics(
        self,
        run: AtomicRunDirectory,
        tally: _Tally,
        excluded: list[dict[str, Any]],
        warnings: Sequence[str],
        runtime: Mapping[str, float | int | None] | None,
    ) -> None:
        run.write_text(_COUNTS, _json(tally.counts(len(excluded), len(warnings))))
        run.write_text(_DISTRIBUTIONS, _json(tally.distributions()))
        sizes = dict(sorted(tally.output_sizes.items()))
        run.write_text(_PAYLOAD, _json({"files": sizes, "total_bytes": sum(sizes.values())}))
        if runtime is not None:
            run.write_text(_RUNTIME, _json(dict(runtime)))

    def _write_debug(self, run: AtomicRunDirectory, outcome: FusionOutcome, number: int) -> None:
        if self._debug_level is SemanticFusionDebugLevel.NONE:
            return
        full = self._debug_level is SemanticFusionDebugLevel.FULL
        if not full and number > _DEBUG_SUPPORT_LIMIT:
            return
        base = f"debug/supports/{outcome.support.fusion_support_id}"
        evidence = outcome.evidence
        run.write_text(f"{base}/summary.json", _json(_debug_summary(outcome)), contractual=False)
        run.write_text(
            f"{base}/hypotheses.json",
            _json({"hypotheses": encode_fused_evidence(evidence)["hypotheses"]}),
            contractual=False,
        )
        run.write_text(
            f"{base}/conflicts.json",
            _json({"uncertainty": encode_fused_evidence(evidence)["uncertainty"]}),
            contractual=False,
        )
        run.write_text(
            f"{base}/physical-observations.json",
            _json({"groups": encode_fused_evidence(evidence)["physical_observation_groups"]}),
            contractual=False,
        )
        if full:
            run.write_text(
                f"{base}/contribution-trace.jsonl",
                "".join(_line(_trace(item)).decode() + "\n" for item in evidence.contributions),
                contractual=False,
            )
            run.write_text(
                f"{base}/geometry-summary.json",
                _json(_geometry_summary(outcome)),
                contractual=False,
            )

    def _manifest_record(
        self, tally: _Tally, excluded: list[dict[str, Any]], warnings: Sequence[str]
    ) -> dict[str, Any]:
        lineage = self._lineage
        return {
            "run_id": str(self._run_id),
            "run_index": self._run_index,
            "sequence_name": self._sequence_name,
            "lineage": {
                "sequence_artifact_id": lineage.sequence_artifact_id,
                "geometric_map_id": str(lineage.geometric_map_id),
                "association_run_ids": list(lineage.association_run_ids),
                "perception_run_ids": [str(item) for item in lineage.perception_run_ids],
                "point_representation_run_ids": [
                    str(item) for item in lineage.point_representation_run_ids
                ],
            },
            "policies": {
                "grouping_policy_id": tally.grouping_policy_id,
                "support_policy_id": tally.support_policy_id,
                "support_configuration_fingerprint": tally.support_fingerprint,
                "fusion_policy_id": tally.fusion_policy_id,
                "fusion_configuration_fingerprint": tally.fusion_fingerprint,
            },
            "evidence_identities": {
                channel: sorted(identities)
                for channel, identities in sorted(tally.identities.items())
            },
            "code_version": self._code_version,
            "counts": {
                "supports": tally.support_count,
                "fused_evidence": tally.support_count,
                "contributions": tally.contribution_count,
                "excluded_observations": len(excluded),
            },
            "warnings": list(warnings),
            "debug_level": self._debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _render_readme(self, tally: _Tally, excluded: list[dict[str, Any]]) -> str:
        return (
            f"# Semantic fusion run {self._run_index:04d}\n"
            "\n"
            f"- Run ID: `{self._run_id}`\n"
            f"- Geometric map: `{self._lineage.geometric_map_id}`\n"
            f"- Fusion policy: `{tally.fusion_policy_id}`\n"
            f"- Supports: {tally.support_count}, contributions: {tally.contribution_count}, "
            f"excluded observations: {len(excluded)}\n"
            "\n"
            "Contractual data is in `outputs/` and `metrics/`; `debug/` is human evidence and no "
            "downstream stage may depend on it. Fused evidence is belief in formation: every "
            "hypothesis, conflict, abstention and unscored claim is kept, and no entity identity "
            "exists yet.\n"
        )


class SemanticFusionRunReader:
    """Reads a finalized Semantic Fusion run from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file outside it is
                needed, and no perception or model runtime.

        Raises:
            IncompleteFusionRunArtifactError: If ``manifest.json`` is missing.
            FusionRunArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = _load_manifest(run_dir)
        self._rows: dict[FusionSupportId, dict[str, Any]] | None = None
        self._observation_support: dict[SpatialObservationId, FusionSupportId] | None = None
        self._contribution_support: dict[EvidenceContributionId, FusionSupportId] | None = None

    @property
    def manifest(self) -> SemanticFusionRunManifest:
        """The run's manifest."""
        return self._manifest

    def support_ids(self) -> list[FusionSupportId]:
        """The identities of every support, in persisted order."""
        return list(self._support_rows())

    def support(self, support_id: FusionSupportId) -> FusionSupport:
        """Read one support by identity without loading the others.

        Raises:
            FusionRunArtifactError: If the run has no such support or its record is malformed.
        """
        row = self._row(support_id)
        raw = self._read_at(_SUPPORTS, row["support_offset"], row["support_length"])
        return _decode(raw, decode_fusion_support, f"support {support_id!r}")

    def fused_evidence(self, support_id: FusionSupportId) -> FusedEvidence:
        """Read the evidence fused over one support without loading the others.

        Raises:
            FusionRunArtifactError: If the run has no such support or its record is malformed.
        """
        row = self._row(support_id)
        raw = self._read_at(_EVIDENCE, row["evidence_offset"], row["evidence_length"])
        return _decode(raw, decode_fused_evidence, f"fused evidence of {support_id!r}")

    def iter_outcomes(self) -> Iterator[FusionOutcome]:
        """Iterate every support with its fused evidence, in persisted order."""
        with (
            (self._root / _SUPPORTS).open("rb") as supports,
            (self._root / _EVIDENCE).open("rb") as evidence,
        ):
            for support_line, evidence_line in zip(supports, evidence, strict=True):
                support = _decode(support_line, decode_fusion_support, "support")
                fused = _decode(evidence_line, decode_fused_evidence, "fused evidence")
                yield FusionOutcome(support=support, evidence=fused)

    def support_of_observation(
        self, observation_id: SpatialObservationId
    ) -> FusionSupportId | None:
        """The support a spatial observation was accumulated in, or ``None`` if it was excluded."""
        if self._observation_support is None:
            self._observation_support = {
                SpatialObservationId(observation_id): support_id
                for support_id, row in self._support_rows().items()
                for observation_id in row["spatial_observation_ids"]
            }
        return self._observation_support.get(observation_id)

    def contribution(self, contribution_id: EvidenceContributionId) -> EvidenceContribution:
        """Resolve a contribution to its exact upstream references.

        Args:
            contribution_id: A contribution named by a hypothesis, a conflict or an index.

        Raises:
            FusionRunArtifactError: If the run has no such contribution.
        """
        if self._contribution_support is None:
            self._contribution_support = {
                EvidenceContributionId(row["contribution_id"]): FusionSupportId(
                    row["fusion_support_id"]
                )
                for row in self._read_rows(_CONTRIBUTION_INDEX)
            }
        support_id = self._contribution_support.get(contribution_id)
        if support_id is None:
            raise FusionRunArtifactError(f"unknown contribution in this run: {contribution_id!r}")
        for item in self.fused_evidence(support_id).contributions:
            if item.contribution_id == contribution_id:
                return item
        raise FusionRunArtifactError(
            f"contribution {contribution_id!r} is missing from its support"
        )

    def excluded_observations(self) -> list[ExcludedObservation]:
        """The observations that took part in no support, with why."""
        return [
            ExcludedObservation(
                spatial_observation_id=SpatialObservationId(row["spatial_observation_id"]),
                geometry_count=row["geometry_count"],
                minimum_geometry_count=row["minimum_geometry_count"],
            )
            for row in self._read_rows(_EXCLUDED)
        ]

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``outputs/`` or ``metrics/``.

        Raises:
            FusionRunArtifactError: If the path is not a JSON record of a contractual
                directory; ``debug/`` is never a valid source.
        """
        if not relative_path.endswith(".json") or not relative_path.startswith(
            ("outputs/", "metrics/")
        ):
            raise FusionRunArtifactError(f"not a contractual JSON record: {relative_path!r}")
        record: dict[str, Any] = json.loads(
            (self._root / relative_path).read_text(encoding="utf-8")
        )
        return record

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        return check_file_inventory(self._root, self._manifest.file_inventory)

    def _support_rows(self) -> dict[FusionSupportId, dict[str, Any]]:
        if self._rows is None:
            self._rows = {
                FusionSupportId(row["fusion_support_id"]): row
                for row in self._read_rows(_SUPPORT_INDEX)
            }
        return self._rows

    def _row(self, support_id: FusionSupportId) -> dict[str, Any]:
        row = self._support_rows().get(support_id)
        if row is None:
            raise FusionRunArtifactError(f"unknown support in this run: {support_id!r}")
        return row

    def _read_rows(self, relative_path: str) -> list[dict[str, Any]]:
        text = (self._root / relative_path).read_text(encoding="utf-8")
        try:
            return [json.loads(line) for line in text.splitlines() if line]
        except ValueError as error:
            raise FusionRunArtifactError(f"malformed index {relative_path}: {error}") from error

    def _read_at(self, relative_path: str, offset: int, length: int) -> bytes:
        with (self._root / relative_path).open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise FusionRunArtifactError(
                f"{relative_path} is truncated: {length} bytes expected at {offset}, "
                f"found {len(data)}"
            )
        return data


def allocate_fusion_run_index(*, workspace_root: Path, sequence_name: str) -> int:
    """Compute the next monotonic run index for a sequence's semantic-fusion runs.

    Scans the run directories, never the registry, so an interrupted or corrupted run is not
    counted.

    Args:
        workspace_root: Root of the local workspace.
        sequence_name: Name of the sequence.

    Returns:
        The next index, starting at ``1``.
    """
    return next_run_index(_sequence_dir(workspace_root, sequence_name), index_of=_valid_run_index)


def rebuild_fusion_run_registry(*, workspace_root: Path, sequence_name: str) -> None:
    """Rebuild a sequence's ``runs.json`` convenience registry from its valid runs.

    Args:
        workspace_root: Root of the local workspace.
        sequence_name: Name of the sequence.
    """
    write_run_registry(_sequence_dir(workspace_root, sequence_name), describe=_registry_record)


class _Tally:
    """Checks the outcomes as they stream and collects what the tables and metrics need."""

    def __init__(self, lineage: FusionRunLineage) -> None:
        self._lineage = lineage
        self.support_count = 0
        self.contribution_count = 0
        self.supports_bytes = 0
        self.evidence_bytes = 0
        self.output_sizes: dict[str, int] = {}
        self.support_rows: list[dict[str, Any]] = []
        self.hypothesis_rows: list[dict[str, Any]] = []
        self.group_rows: list[dict[str, Any]] = []
        self.contribution_rows: list[dict[str, Any]] = []
        self.identities: dict[str, set[str]] = {}
        self.grouping_policy_id: str | None = None
        self.support_policy_id: str | None = None
        self.support_fingerprint: str | None = None
        self.fusion_policy_id: str | None = None
        self.fusion_fingerprint: str | None = None
        self._previous: FusionSupportId | None = None
        self._physical: set[str] = set()
        self._results: set[str] = set()
        self._per_support: dict[str, list[int]] = {
            "physical_observations_per_support": [],
            "inference_results_per_support": [],
            "contributions_per_support": [],
            "hypotheses_per_support": [],
        }
        self._claims_total = 0
        self._claims_seen: dict[tuple[str, str], float | None] = {}
        self._abstaining: set[tuple[str, str]] = set()
        self._stances: Counter[str] = Counter()
        self._uncertainty: Counter[str] = Counter()
        self._hypotheses = 0
        self._scorer_signals = 0
        self._supports_with_uncertainty = 0

    def check(self, outcome: FusionOutcome) -> None:
        support, evidence = outcome.support, outcome.evidence
        if self._previous is not None and support.fusion_support_id <= self._previous:
            raise FusionRunArtifactError(
                f"supports must be strictly sorted by identity: {support.fusion_support_id!r} "
                f"follows {self._previous!r}"
            )
        self._previous = support.fusion_support_id
        contributed = sorted(item.spatial_observation_id for item in evidence.contributions)
        if list(support.spatial_observation_ids) != contributed:
            raise FusionRunArtifactError(
                f"support {support.fusion_support_id!r} lists spatial observations that differ "
                f"from the contributions of its evidence"
            )
        if support.geometric_map_id != self._lineage.geometric_map_id:
            raise FusionRunArtifactError(
                f"support {support.fusion_support_id!r} is over map {support.geometric_map_id!r}, "
                f"but the run's lineage names {self._lineage.geometric_map_id!r}"
            )
        for item in evidence.contributions:
            if item.perception_run_id not in self._lineage.perception_run_ids:
                raise FusionRunArtifactError(
                    f"contribution {item.contribution_id!r} comes from perception run "
                    f"{item.perception_run_id!r}, which the run's lineage does not list"
                )
        for ref in evidence.point_representation_refs:
            if ref.run_id not in self._lineage.point_representation_run_ids:
                raise FusionRunArtifactError(
                    f"support {support.fusion_support_id!r} references point representation run "
                    f"{ref.run_id!r}, which the run's lineage does not list"
                )
        self._single_policy(outcome)

    def _single_policy(self, outcome: FusionOutcome) -> None:
        support, evidence = outcome.support.provenance, outcome.evidence.provenance
        current = (
            evidence.grouping_policy_id,
            support.support_policy_id,
            support.configuration_fingerprint,
            evidence.fusion_policy_id,
            evidence.configuration_fingerprint,
        )
        first = (
            self.grouping_policy_id,
            self.support_policy_id,
            self.support_fingerprint,
            self.fusion_policy_id,
            self.fusion_fingerprint,
        )
        if self.support_count == 0:
            (
                self.grouping_policy_id,
                self.support_policy_id,
                self.support_fingerprint,
                self.fusion_policy_id,
                self.fusion_fingerprint,
            ) = current
        elif current != first:
            raise FusionRunArtifactError(
                f"support {outcome.support.fusion_support_id!r} was built or fused under a "
                f"different policy or configuration than the rest of the run: one run keeps one"
            )

    def add(
        self,
        outcome: FusionOutcome,
        *,
        support_offset: int,
        support_length: int,
        evidence_offset: int,
        evidence_length: int,
    ) -> None:
        support, evidence = outcome.support, outcome.evidence
        self.support_count += 1
        self.contribution_count += len(evidence.contributions)
        self._hypotheses += len(evidence.hypotheses)
        self._physical.update(
            str(g.physical_observation_id) for g in evidence.physical_observation_groups
        )
        # Um resultado de percepção tem várias regiões e cada uma pode cair em um suporte diferente:
        # somar `inference_result_count` por suporte contaria o mesmo resultado várias vezes e
        # deixaria de ser comparável com os frames físicos distintos acima.
        self._results.update(
            str(result)
            for g in evidence.physical_observation_groups
            for result in g.perception_result_ids
        )
        counts = self._per_support
        counts["physical_observations_per_support"].append(evidence.physical_observation_count)
        counts["inference_results_per_support"].append(evidence.inference_result_count)
        counts["contributions_per_support"].append(len(evidence.contributions))
        counts["hypotheses_per_support"].append(len(evidence.hypotheses))

        self.support_rows.append(
            {
                "fusion_support_id": str(support.fusion_support_id),
                "fused_evidence_id": str(evidence.fused_evidence_id),
                "support_offset": support_offset,
                "support_length": support_length,
                "evidence_offset": evidence_offset,
                "evidence_length": evidence_length,
                "spatial_observation_ids": [str(item) for item in support.spatial_observation_ids],
                "physical_observation_ids": [
                    str(g.physical_observation_id) for g in evidence.physical_observation_groups
                ],
            }
        )
        for group in evidence.physical_observation_groups:
            self.group_rows.append(
                {
                    "fusion_support_id": str(support.fusion_support_id),
                    "physical_observation_id": str(group.physical_observation_id),
                    "acquisition_timestamp": group.acquisition_timestamp.to_record(),
                    "spatial_observation_ids": [str(i) for i in group.spatial_observation_ids],
                    "perception_result_ids": [str(i) for i in group.perception_result_ids],
                    "perception_run_ids": [str(i) for i in group.perception_run_ids],
                }
            )
        for item in evidence.contributions:
            self._claims_total += len(item.claim_refs)
            self.contribution_rows.append(
                {
                    "contribution_id": str(item.contribution_id),
                    "fusion_support_id": str(support.fusion_support_id),
                    "spatial_observation_id": str(item.spatial_observation_id),
                    "physical_observation_id": str(item.physical_observation_id),
                    "perception_result_id": str(item.perception_result_id),
                    "perception_run_id": str(item.perception_run_id),
                    "region_id": str(item.region_id),
                    "claim_ids": [str(ref.claim_id) for ref in item.claim_refs],
                    "geometry_count": len(item.geometry_support),
                    "has_observation_quality": item.observation_quality is not None,
                }
            )
        self._add_hypotheses(outcome)
        self._add_uncertainty(evidence)
        if evidence.uncertainty:
            self._supports_with_uncertainty += 1
        for channel in evidence.channels:
            self.identities.setdefault(channel.channel.value, set()).update(channel.identities)

    def _add_hypotheses(self, outcome: FusionOutcome) -> None:
        support, evidence = outcome.support, outcome.evidence
        contributions = {item.contribution_id: item for item in evidence.contributions}
        for hypothesis in evidence.hypotheses:
            for item in hypothesis.evidence:
                contribution = contributions[item.contribution_id]
                self._stances[item.stance.value] += 1
                key = (str(item.contribution_id), str(item.claim_id))
                confidence = next(
                    (s.value for s in item.signals if s.kind is SupportSignalKind.CLAIM_CONFIDENCE),
                    None,
                )
                self._claims_seen.setdefault(key, confidence)
                self._scorer_signals += sum(
                    1 for s in item.signals if s.kind is SupportSignalKind.SCORER_SUPPORT
                )
                if item.stance is EvidenceStance.ABSTAINING:
                    self._abstaining.add(key)
                self.hypothesis_rows.append(
                    {
                        "fusion_support_id": str(support.fusion_support_id),
                        "fused_evidence_id": str(evidence.fused_evidence_id),
                        "hypothesis_id": str(hypothesis.hypothesis_id),
                        "label": hypothesis.label,
                        "contribution_id": str(item.contribution_id),
                        "claim_id": str(item.claim_id),
                        "stance": item.stance.value,
                        "role": item.role.value,
                        "physical_observation_id": str(contribution.physical_observation_id),
                        "perception_result_id": str(contribution.perception_result_id),
                        "perception_run_id": str(contribution.perception_run_id),
                        "spatial_observation_id": str(contribution.spatial_observation_id),
                    }
                )

    def _add_uncertainty(self, evidence: FusedEvidence) -> None:
        for record in evidence.uncertainty:
            self._uncertainty[record.kind.value] += 1
            if record.kind is UncertaintyKind.INSUFFICIENT_EVIDENCE:
                self._abstaining.update(
                    (str(ref.contribution_id), str(ref.claim_id))
                    for ref in record.evidence
                    if ref.claim_id is not None
                )

    def counts(self, excluded: int, warnings: int) -> dict[str, Any]:
        scored = sum(1 for value in self._claims_seen.values() if value is not None)
        return {
            "supports": self.support_count,
            "fused_evidence": self.support_count,
            "contributions": self.contribution_count,
            "hypotheses": self._hypotheses,
            "physical_observations": len(self._physical),
            "inference_results": len(self._results),
            "claims": {
                "total": self._claims_total,
                "scored": scored,
                "unscored": len(self._claims_seen) - scored,
                "without_hypothesis": self._claims_total - len(self._claims_seen),
            },
            "scorer_signals": self._scorer_signals,
            "abstaining_claims": len(self._abstaining),
            "evidence_stances": {
                stance.value: self._stances.get(stance.value, 0) for stance in EvidenceStance
            },
            "uncertainty": {
                kind.value: self._uncertainty.get(kind.value, 0)
                for kind in (
                    UncertaintyKind.CONTRADICTION,
                    UncertaintyKind.AMBIGUITY,
                    UncertaintyKind.NEAR_TIE,
                    UncertaintyKind.INSUFFICIENT_EVIDENCE,
                )
            },
            "supports_with_uncertainty": self._supports_with_uncertainty,
            "excluded_observations": excluded,
            "warnings": warnings,
        }

    def distributions(self) -> dict[str, Any]:
        return {name: _distribution(values) for name, values in self._per_support.items()}


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _encode_excluded(item: ExcludedObservation) -> dict[str, Any]:
    return {
        "spatial_observation_id": str(item.spatial_observation_id),
        "geometry_count": item.geometry_count,
        "minimum_geometry_count": item.minimum_geometry_count,
    }


def _debug_summary(outcome: FusionOutcome) -> dict[str, Any]:
    evidence = outcome.evidence
    contributions = {item.contribution_id: item for item in evidence.contributions}
    hypotheses = []
    for hypothesis in evidence.hypotheses:
        supporting = [
            contributions[item.contribution_id]
            for item in hypothesis.evidence
            if item.stance is EvidenceStance.SUPPORTING
        ]
        stances = Counter(item.stance.value for item in hypothesis.evidence)
        hypotheses.append(
            {
                "hypothesis_id": str(hypothesis.hypothesis_id),
                "label": hypothesis.label,
                "supporting_physical_observations": sorted(
                    {str(item.physical_observation_id) for item in supporting}
                ),
                "supporting_inference_results": sorted(
                    {str(item.perception_result_id) for item in supporting}
                ),
                "stances": dict(sorted(stances.items())),
                "unscored_claims": sum(
                    1
                    for item in hypothesis.evidence
                    if any(
                        s.kind is SupportSignalKind.CLAIM_CONFIDENCE and s.value is None
                        for s in item.signals
                    )
                ),
            }
        )
    return {
        "fusion_support_id": str(outcome.support.fusion_support_id),
        "fused_evidence_id": str(evidence.fused_evidence_id),
        "physical_observation_count": evidence.physical_observation_count,
        "inference_result_count": evidence.inference_result_count,
        "hypotheses": hypotheses,
        "active_channels": [item.channel.value for item in evidence.channels],
        "uncertainty": [
            {
                "kind": record.kind.value,
                "hypothesis_ids": [str(h) for h in record.hypothesis_ids],
                "evidence_count": len(record.evidence),
                "rule_id": record.rule_id,
            }
            for record in evidence.uncertainty
        ],
        "weighted": evidence.weighting is not None,
    }


def _trace(item: EvidenceContribution) -> dict[str, Any]:
    return {
        "contribution_id": str(item.contribution_id),
        "physical_observation_id": str(item.physical_observation_id),
        "perception_result_id": str(item.perception_result_id),
        "perception_run_id": str(item.perception_run_id),
        "spatial_observation_id": str(item.spatial_observation_id),
        "region_id": str(item.region_id),
        "claim_ids": [str(ref.claim_id) for ref in item.claim_refs],
        "score_refs": len(item.score_refs),
        "visual_feature_refs": [ref.embedding_space_id for ref in item.visual_feature_refs],
        "observation_quality": None
        if item.observation_quality is None
        else item.observation_quality.definitions_version,
        "geometry_count": len(item.geometry_support),
    }


def _geometry_summary(outcome: FusionOutcome) -> dict[str, Any]:
    support = outcome.support
    return {
        "geometric_map_id": str(support.geometric_map_id),
        "geometry_count": len(support.geometry_support),
        "bounds": {
            "frame_id": str(support.bounds.frame_id),
            "minimum_m": list(support.bounds.minimum_m),
            "maximum_m": list(support.bounds.maximum_m),
        },
        "centroid_m": list(support.centroid_m),
        "geometry_per_contribution": {
            str(item.contribution_id): len(item.geometry_support)
            for item in outcome.evidence.contributions
        },
    }


def _decode(raw: bytes, decoder: Callable[[Mapping[str, Any]], _T], what: str) -> _T:
    try:
        return decoder(json.loads(raw))
    except (ValueError, KeyError, TypeError) as error:
        raise FusionRunArtifactError(f"malformed {what}: {error}") from error


def _valid_run_index(run_dir: Path) -> int | None:
    # Um diretório ilegível ou com manifest malformado simplesmente não é um run válido.
    try:
        reader = SemanticFusionRunReader(run_dir)
    except (FusionRunArtifactError, ValueError, KeyError, OSError):
        return None
    return None if reader.verify_integrity() else reader.manifest.run_index


def _registry_record(run_dir: Path) -> dict[str, Any] | None:
    index = _valid_run_index(run_dir)
    if index is None:
        return None
    manifest = SemanticFusionRunReader(run_dir).manifest
    return {"run_index": index, "run_id": str(manifest.run_id), "directory": run_dir.name}


def _load_manifest(run_dir: Path) -> SemanticFusionRunManifest:
    manifest_path = run_dir / _MANIFEST
    if not manifest_path.is_file():
        raise IncompleteFusionRunArtifactError(f"missing {_MANIFEST} in {run_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise FusionRunArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}; "
            f"this reader understands {SCHEMA_VERSION!r}"
        )
    lineage = raw["lineage"]
    policies = raw["policies"]
    counts = raw["counts"]
    return SemanticFusionRunManifest(
        run_id=SemanticFusionRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        lineage=FusionRunLineage(
            sequence_artifact_id=lineage["sequence_artifact_id"],
            geometric_map_id=MapId(lineage["geometric_map_id"]),
            association_run_ids=tuple(lineage["association_run_ids"]),
            perception_run_ids=tuple(PerceptionRunId(i) for i in lineage["perception_run_ids"]),
            point_representation_run_ids=tuple(
                PointRepresentationRunId(i) for i in lineage["point_representation_run_ids"]
            ),
        ),
        grouping_policy_id=policies["grouping_policy_id"],
        support_policy_id=policies["support_policy_id"],
        support_configuration_fingerprint=policies["support_configuration_fingerprint"],
        fusion_policy_id=policies["fusion_policy_id"],
        fusion_configuration_fingerprint=policies["fusion_configuration_fingerprint"],
        evidence_identities={
            channel: tuple(identities) for channel, identities in raw["evidence_identities"].items()
        },
        code_version=raw["code_version"],
        support_count=counts["supports"],
        fused_evidence_count=counts["fused_evidence"],
        contribution_count=counts["contributions"],
        excluded_count=counts["excluded_observations"],
        warnings=tuple(raw["warnings"]),
        debug_level=raw["debug_level"],
        schema_version=raw["schema_version"],
        created_at=raw["created_at"],
        file_inventory=tuple(
            FileEntry(
                path=entry["path"],
                size_bytes=entry["size_bytes"],
                content_hash=entry["content_hash"],
            )
            for entry in raw["file_inventory"]
        ),
    )


def _line(record: Mapping[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"

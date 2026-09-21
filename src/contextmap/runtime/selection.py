"""Explicit run selection and lineage binding for multi-run experiments.

The same physical sequence can have many perception, state-estimation, representation or
fusion runs, so the runtime must never silently combine every available result. A
selection says exactly which upstream runs an execution starts from:

- an exact artifact id (or several, kept as distinct evidence);
- a named selection, a coherent set of exact ids defined in the configuration;
- ``"latest"``, an explicit opt-in to the newest run that is compatible with the rest.

A stage that is not listed is never selected implicitly, there is no "use all runs", and
selection only chooses inputs: it never fuses or reinterprets evidence.

Compatibility is judged from the lineage the artifacts declare (the identity of the
physical sequence, the observation selection, the calibration, the schema version and the
exact upstream artifacts each was built from), never from file names or sequence names.
An incompatible selection is reported before anything runs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from contextmap.runtime.artifacts import ArtifactRef
from contextmap.runtime.config import LATEST, NAMED_PREFIX, ConfigProblem, InputsConfig

if TYPE_CHECKING:
    from contextmap.runtime.pipeline import PipelinePlan


@dataclass(frozen=True, kw_only=True)
class Lineage:
    """What an artifact declares about where it came from.

    Every field is optional: an artifact states only what its manifest records, and a
    missing identity is never assumed equal to anything.

    Attributes:
        sequence: Identity of the physical sequence the artifact derives from. Two
            artifacts of the same sequence carry the same value even when the sequence was
            ingested more than once.
        selection: Identity of the observation selection it covers.
        calibration: Identity of the calibration it was built with.
        schema_version: Schema version of the artifact.
        upstream: The exact upstream artifact ids it was built from, by upstream stage.
    """

    sequence: str | None = None
    selection: str | None = None
    calibration: str | None = None
    schema_version: str | None = None
    upstream: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form persisted with a resolved selection."""
        return {
            "sequence": self.sequence,
            "selection": self.selection,
            "calibration": self.calibration,
            "schema_version": self.schema_version,
            "upstream": {stage: list(ids) for stage, ids in sorted(self.upstream.items())},
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Lineage:
        """Rebuild a lineage from its persisted form.

        Args:
            document: A mapping produced by :meth:`to_document`; every key is optional.

        Returns:
            The lineage.

        Raises:
            ValueError: If an identity is not text or ``upstream`` is not a mapping of
                stage to a list of artifact ids.
        """
        values: dict[str, str | None] = {}
        for name in ("sequence", "selection", "calibration", "schema_version"):
            value = document.get(name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"lineage {name} must be text or null")
            values[name] = value
        raw_upstream = document.get("upstream") or {}
        if not isinstance(raw_upstream, Mapping):
            raise ValueError("lineage upstream must be a mapping of stage to artifact ids")
        upstream: dict[str, tuple[str, ...]] = {}
        for stage, ids in raw_upstream.items():
            if not isinstance(ids, list | tuple) or not all(isinstance(i, str) for i in ids):
                raise ValueError(f"lineage upstream of {stage!r} must be a list of artifact ids")
            upstream[str(stage)] = tuple(ids)
        return cls(upstream=upstream, **values)


@dataclass(frozen=True, kw_only=True)
class CatalogEntry:
    """One available run of a stage, with the lineage it declares.

    Attributes:
        ref: The exact artifact.
        lineage: What it declares about where it came from.
        run_index: Monotonic index of the run among the runs of its stage, as its run
            index records it. It orders runs for ``"latest"``.
    """

    ref: ArtifactRef
    lineage: Lineage
    run_index: int

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form of the entry, as a catalog file lists it."""
        return {
            **self.ref.to_document(),
            "run_index": self.run_index,
            "lineage": self.lineage.to_document(),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> CatalogEntry:
        """Rebuild an entry from its persisted form.

        Args:
            document: A mapping produced by :meth:`to_document`.

        Returns:
            The entry.

        Raises:
            ValueError: If the artifact reference, the run index or the lineage is invalid.
        """
        if not isinstance(document, Mapping):
            raise ValueError("a catalog entry must be a mapping")
        run_index = document.get("run_index")
        if not isinstance(run_index, int) or isinstance(run_index, bool) or run_index < 0:
            raise ValueError("a catalog entry needs a non-negative integer run_index")
        lineage = document.get("lineage") or {}
        if not isinstance(lineage, Mapping):
            raise ValueError("a catalog entry's lineage must be a mapping")
        return cls(
            ref=ArtifactRef.from_document(document),
            lineage=Lineage.from_document(lineage),
            run_index=run_index,
        )


CATALOG_SCHEMA_VERSION = "0.1.0"
"""Version of the catalog file format."""


def load_catalog(path: str | os.PathLike[str]) -> StaticCatalog:
    """Load a catalog file listing the runs available to select from.

    The file is a JSON document ``{"schema_version": "0.1.0", "entries": [...]}`` whose
    entries are :meth:`CatalogEntry.to_document` mappings. It is an explicit input: nothing
    is discovered by scanning directories or matching names.

    Args:
        path: Path of the catalog file.

    Returns:
        The catalog.

    Raises:
        ValueError: If the file cannot be read, follows another schema version or holds an
            invalid entry.
    """
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read catalog {source}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"catalog {source} must follow schema_version {CATALOG_SCHEMA_VERSION!r}")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"catalog {source} needs an 'entries' list")
    loaded = []
    for index, raw in enumerate(entries):
        try:
            loaded.append(CatalogEntry.from_document(raw))
        except ValueError as error:
            raise ValueError(f"catalog {source}: entry {index} is invalid: {error}") from error
    return StaticCatalog(loaded)


class ArtifactCatalog(Protocol):
    """The runs available to select from, rebuilt from the capabilities' run indexes."""

    def get(self, artifact_id: str) -> CatalogEntry | None:
        """Return the run with this exact artifact id, if it exists."""
        ...

    def entries(self, stage_id: str) -> Sequence[CatalogEntry]:
        """Return the runs of a stage in ascending run-index order."""
        ...


class StaticCatalog:
    """A catalog held in memory, for callers that already know their runs.

    The order of the entries given is irrelevant: runs are ordered by run index, so
    ``"latest"`` is deterministic and does not depend on how the catalog was built.
    """

    def __init__(self, entries: Iterable[CatalogEntry]) -> None:
        """Index the entries, rejecting a repeated artifact id.

        Raises:
            ValueError: If two entries share an artifact id.
        """
        self._by_id: dict[str, CatalogEntry] = {}
        by_stage: dict[str, list[CatalogEntry]] = {}
        for entry in entries:
            if entry.ref.artifact_id in self._by_id:
                raise ValueError(f"artifact id {entry.ref.artifact_id!r} appears twice")
            self._by_id[entry.ref.artifact_id] = entry
            by_stage.setdefault(entry.ref.stage_id, []).append(entry)
        self._by_stage = {
            stage: tuple(sorted(items, key=lambda e: (e.run_index, e.ref.artifact_id)))
            for stage, items in by_stage.items()
        }

    def get(self, artifact_id: str) -> CatalogEntry | None:
        """Return the run with this exact artifact id, if it exists."""
        return self._by_id.get(artifact_id)

    def entries(self, stage_id: str) -> Sequence[CatalogEntry]:
        """Return the runs of a stage in ascending run-index order."""
        return self._by_stage.get(stage_id, ())


@dataclass(frozen=True, kw_only=True)
class SelectedRun:
    """One run an execution starts from, and how it was chosen.

    Attributes:
        entry: The run.
        origin: ``"exact"``, ``"named:<name>"`` or ``"latest"``.
    """

    entry: CatalogEntry
    origin: str


@dataclass(frozen=True, kw_only=True)
class ResolvedSelections:
    """The exact input set of an execution, resolved and checked.

    Attributes:
        sequence: The physical sequence identity the configuration expects, if any.
        runs: The selected runs by stage, in a deterministic order.
        problems: Everything that makes the selection unusable; preflight reports them.
    """

    sequence: str | None
    runs: Mapping[str, tuple[SelectedRun, ...]]
    problems: tuple[ConfigProblem, ...]

    @property
    def ok(self) -> bool:
        """Whether the selection is usable."""
        return not self.problems

    @property
    def provided(self) -> Mapping[str, tuple[ArtifactRef, ...]]:
        """The selected artifacts by stage, ready to feed an execution."""
        return {stage: tuple(run.entry.ref for run in runs) for stage, runs in self.runs.items()}

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible resolution persisted with the execution."""
        return {
            "sequence": self.sequence,
            "stages": {
                stage: [
                    {
                        "artifact_id": run.entry.ref.artifact_id,
                        "content_hash": run.entry.ref.content_hash,
                        "run_index": run.entry.run_index,
                        "origin": run.origin,
                        "lineage": run.entry.lineage.to_document(),
                    }
                    for run in runs
                ]
                for stage, runs in self.runs.items()
            },
        }


def resolve_selections(
    plan: PipelinePlan,
    inputs: InputsConfig,
    catalog: ArtifactCatalog,
    *,
    supported_schemas: Mapping[str, Collection[str]] | None = None,
) -> ResolvedSelections:
    """Resolve the configured selections against a catalog and check their lineage.

    Exact and named references are resolved first. Each ``"latest"`` is then resolved in
    topology order, so it sees every earlier choice: it takes the newest run of its stage
    that declares a lineage compatible with them. Finally the whole set is checked. Nothing
    is inferred: a stage without a selection is left out, and every problem is returned
    rather than raised, so preflight can report it with the rest.

    Args:
        plan: The topology being executed.
        inputs: The ``inputs`` section of the effective configuration.
        catalog: The runs available.
        supported_schemas: Schema versions each artifact kind may have; a kind not listed
            is not checked.

    Returns:
        The resolved selection, its problems included.
    """
    problems: list[ConfigProblem] = []
    order = plan.order or tuple(stage.stage_id for stage in plan.stages)
    in_plan = set(order)
    resolved: dict[str, list[SelectedRun]] = {}
    latest: list[str] = []

    for stage_id in inputs.selections:
        if stage_id not in in_plan:
            problems.append(
                ConfigProblem(
                    path=f"selections.{stage_id}",
                    message=f"{stage_id!r} is not a stage that takes part in this plan",
                )
            )
    for stage_id in order:
        references = inputs.selections.get(stage_id)
        if references is None:
            continue
        if references == (LATEST,):
            latest.append(stage_id)
            continue
        origin = "exact"
        ids = references
        if len(references) == 1 and references[0].startswith(NAMED_PREFIX):
            name = references[0].removeprefix(NAMED_PREFIX)
            origin = references[0]
            ids = inputs.named.get(name, {}).get(stage_id, ())
        for artifact_id in ids:
            entry = catalog.get(artifact_id)
            path = f"selections.{stage_id}"
            if entry is None:
                problems.append(
                    ConfigProblem(path=path, message=f"no artifact {artifact_id!r} in the catalog")
                )
            elif entry.ref.stage_id != stage_id:
                problems.append(
                    ConfigProblem(
                        path=path,
                        message=(
                            f"artifact {artifact_id!r} belongs to stage {entry.ref.stage_id!r}, "
                            f"not {stage_id!r}"
                        ),
                    )
                )
            elif entry.ref.contract != plan.stage(stage_id).output:
                problems.append(
                    ConfigProblem(
                        path=path,
                        message=(
                            f"artifact {artifact_id!r} is a {entry.ref.contract!r}; stage "
                            f"{stage_id!r} produces {plan.stage(stage_id).output!r}"
                        ),
                    )
                )
            else:
                resolved.setdefault(stage_id, []).append(SelectedRun(entry=entry, origin=origin))

    for stage_id in latest:
        current = [run.entry for runs in resolved.values() for run in runs]
        candidates = list(reversed(catalog.entries(stage_id)))
        chosen = next(
            (
                candidate
                for candidate in candidates
                if not _involving(current, candidate, supported_schemas)
            ),
            None,
        )
        if chosen is None:
            problems.append(
                ConfigProblem(
                    path=f"selections.{stage_id}",
                    message=(
                        f"no run of {stage_id!r} is compatible with the other selections "
                        f"({len(candidates)} considered, newest first)"
                    ),
                )
            )
        else:
            resolved[stage_id] = [SelectedRun(entry=chosen, origin=LATEST)]

    runs = {
        stage_id: tuple(
            sorted(resolved[stage_id], key=lambda r: (r.entry.run_index, r.entry.ref.artifact_id))
        )
        for stage_id in order
        if stage_id in resolved
    }
    problems.extend(_cardinality(plan, runs))
    entries = [run.entry for stage_runs in runs.values() for run in stage_runs]
    problems.extend(check_lineage(entries, supported_schemas))
    if inputs.sequence is not None:
        for entry in entries:
            declared = entry.lineage.sequence
            if declared is not None and declared != inputs.sequence:
                problems.append(
                    ConfigProblem(
                        path=f"selections.{entry.ref.stage_id}",
                        message=(
                            f"artifact {entry.ref.artifact_id!r} is of sequence {declared!r}; "
                            f"the configuration expects {inputs.sequence!r}"
                        ),
                    )
                )
    return ResolvedSelections(sequence=inputs.sequence, runs=runs, problems=tuple(problems))


def check_lineage(
    entries: Sequence[CatalogEntry],
    supported_schemas: Mapping[str, Collection[str]] | None = None,
) -> list[ConfigProblem]:
    """Check that selected runs can be used together, from the lineage they declare.

    Runs are compatible when they agree on the identity of the physical sequence, the
    observation selection and the calibration wherever they declare one; when each has a
    schema version its kind supports; and when every run built from an upstream stage was
    built from exactly the run of that stage that is selected. Repeated runs of one stage
    are checked the same way, so several inference runs over the same observations are
    accepted and runs over different ones are not.

    Args:
        entries: The selected runs.
        supported_schemas: Schema versions each artifact kind may have.

    Returns:
        Every disagreement found, all at once.
    """
    return [problem for problem, _ in _conflicts(entries, supported_schemas)]


def _involving(
    current: Sequence[CatalogEntry],
    candidate: CatalogEntry,
    supported_schemas: Mapping[str, Collection[str]] | None,
) -> list[ConfigProblem]:
    """List the conflicts a candidate run would introduce into the current selection."""
    artifact_id = candidate.ref.artifact_id
    return [
        problem
        for problem, involved in _conflicts([*current, candidate], supported_schemas)
        if artifact_id in involved
    ]


def _conflicts(
    entries: Sequence[CatalogEntry],
    supported_schemas: Mapping[str, Collection[str]] | None,
) -> list[tuple[ConfigProblem, frozenset[str]]]:
    found: list[tuple[ConfigProblem, frozenset[str]]] = []
    for attribute in ("sequence", "selection", "calibration"):
        first: tuple[str, CatalogEntry] | None = None
        for entry in entries:
            value = getattr(entry.lineage, attribute)
            if value is None:
                continue
            if first is None:
                first = (value, entry)
            elif value != first[0]:
                a, b = first[1].ref.artifact_id, entry.ref.artifact_id
                found.append(
                    (
                        ConfigProblem(
                            path=f"selections.{entry.ref.stage_id}",
                            message=(
                                f"{attribute} identity {value!r} of artifact {b!r} differs from "
                                f"{first[0]!r} of artifact {a!r}"
                            ),
                        ),
                        frozenset({a, b}),
                    )
                )
    if supported_schemas is not None:
        for entry in entries:
            supported = supported_schemas.get(entry.ref.contract)
            version = entry.lineage.schema_version
            if supported is not None and version not in supported:
                found.append(
                    (
                        ConfigProblem(
                            path=f"selections.{entry.ref.stage_id}",
                            message=(
                                f"artifact {entry.ref.artifact_id!r} has schema version "
                                f"{version!r}; {entry.ref.contract!r} supports "
                                f"{sorted(supported)}"
                            ),
                        ),
                        frozenset({entry.ref.artifact_id}),
                    )
                )
    by_stage: dict[str, list[CatalogEntry]] = {}
    for entry in entries:
        by_stage.setdefault(entry.ref.stage_id, []).append(entry)
    for built in entries:
        for stage, ids in built.lineage.upstream.items():
            for selected in by_stage.get(stage, ()):
                if selected.ref.artifact_id not in ids:
                    a, b = built.ref.artifact_id, selected.ref.artifact_id
                    found.append(
                        (
                            ConfigProblem(
                                path=f"selections.{selected.ref.stage_id}",
                                message=(
                                    f"artifact {a!r} was built from {stage} {list(ids)}, but "
                                    f"{b!r} is selected"
                                ),
                            ),
                            frozenset({a, b}),
                        )
                    )
    return found


def _cardinality(
    plan: PipelinePlan, runs: Mapping[str, tuple[SelectedRun, ...]]
) -> list[ConfigProblem]:
    """Refuse several runs of a stage where the consumer accepts exactly one."""
    problems = []
    for stage in plan.stages:
        for item in stage.inputs:
            selected = runs.get(item.source, ())
            if len(selected) > 1 and not item.multiple:
                problems.append(
                    ConfigProblem(
                        path=f"stages.{stage.stage_id}.{item.name}",
                        message=(
                            f"accepts exactly one run but {len(selected)} are selected from "
                            f"stage {item.source!r}"
                        ),
                    )
                )
    return problems

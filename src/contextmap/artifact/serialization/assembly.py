"""Deterministic assembly of a :class:`ContextMap` from already-decided upstream results.

Entity Resolution and Spatial Relations already decided everything a map composes: which source
entities are one physical object, what that object may be, and how resolved entities relate in
space. This module never resolves an entity and never decides a relation; it *translates* those
finished decisions into the map's own schema (:mod:`contextmap.artifact.composition`), the way
``structural_dependencies.py`` already reads the same two runs to *prove* a map's claims instead
of computing them. Geometry stays referenced: the geometric-map artifact is opened only for its
identity and size, never for its points.

Assembly is intentionally asymmetric between what it copies and what it defers:

* identity, resolution lineage (``member_entities``, ``resolution_decisions``,
  ``unresolved_neighbors``) and geometry references are copied verbatim from
  :class:`~contextmap.entity_resolution.ResolvedEntity`, because :mod:`composition` already
  requires the map's own copy of these fields to equal the upstream record exactly
  (``structural_dependencies.py``'s "prove content, not just existence");
* the semantic state is *translated*, not copied verbatim, because
  :class:`~contextmap.entity_resolution.ResolvedSemanticState` and
  :class:`~contextmap.artifact.ContextSemanticState` are different granularities of the same
  belief (see :func:`_translate_semantic_state`);
* every relation is translated the same way, endpoint by endpoint, through the entities this
  module already composed.

Provenance is deliberately kept apart by epistemic level, not just by upstream artifact:

* :class:`~contextmap.artifact.ContextEntity`'s own ``origin`` cites the *resolved entity* record
  of the Entity Resolution run this module opened, under ``DerivationKind.MULTIVIEW_FUSED``:
  identity resolution genuinely is a versioned rule (materialization) that accumulates several
  members' evidence into one belief, and Entity Resolution is the one run this module has
  actually opened to cite that derivation honestly;
* each ``LabelHypothesis.origin``, by contrast, cites the *Semantic Fusion* evidence the label
  actually came from (``ResolvedEntity.evidence.fused_evidence``, matched to each hypothesis by
  its own ``fused_evidence_id``), never the resolved-entity record: Entity Resolution resolves
  identity and aggregates pre-existing belief, it does not propose the visual hypothesis "chair".
  See ``src/contextmap/artifact/docs/assembly.md`` for the full rationale (issue #541).

This module lives under ``serialization/``, not beside :mod:`contextmap.artifact.composition`,
because it genuinely performs file I/O (opening the Entity Resolution, Spatial Relations and
Geometric Mapping run directories it is given, and reading their manifests to pin each one by
its content digest): exactly the same category of work ``structural_dependencies.py`` already
does from the same place, and exactly what the schema modules (``composition.py``, ``models.py``
and their siblings, checked by ``tests/artifact/test_context_map_invariants.py::
test_the_schema_package_performs_no_file_io_and_owns_no_format``) are not allowed to do. It is
still re-exported at the capability's public root (``contextmap.artifact.assemble_context_map``),
the same way ``ContextMapArtifactWriter`` and ``artifact_digest`` are.
"""

from __future__ import annotations

import re
import time
import tracemalloc
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from contextmap.artifact.composition import (
    AmbiguityStatus,
    ContextEntity,
    ContextRelation,
    ContextSemanticState,
    LabelHypothesis,
)
from contextmap.artifact.metadata import ContextMapMetadata, PolicyRef
from contextmap.artifact.models import ContextMap, GeometricMapLink
from contextmap.artifact.provenance import (
    ArtifactKind,
    DerivationKind,
    EvidenceOrigin,
    ProvenanceError,
    UpstreamArtifact,
)
from contextmap.artifact.references import (
    ContextEntityId,
    ContextEntityReference,
    ContextMapId,
    ContextRelationId,
    UpstreamRecordRef,
)
from contextmap.artifact.serialization.dependencies import artifact_digest
from contextmap.artifact.serialization.errors import UnresolvedReferenceError, UpstreamArtifactError
from contextmap.artifact.serialization.manifest import ContextMapArtifactManifest
from contextmap.artifact.serialization.writer import ContextMapArtifactWriter
from contextmap.artifact.versioning import CONTEXT_MAP_SCHEMA_VERSION
from contextmap.entity_resolution import (
    EntityResolutionRunReader,
    ResolutionRunLineage,
    ResolvedEntityReference,
    ResolvedEntitySet,
    ResolvedSemanticState,
)
from contextmap.entity_resolution import PolicyRef as ResolutionPolicyRef
from contextmap.geometric_mapping import (
    GeometricMapArtifactManifest,
    GeometricMapArtifactReader,
    MapId,
)
from contextmap.semantic_mapping import EntityEvidenceLinks, EntityHypothesis, FusedEvidenceRef
from contextmap.spatial_relations import (
    Relation,
    RelationProvenance,
    SpatialRelationsRunId,
    SpatialRelationsRunReader,
)

__all__ = [
    "AssemblyMetrics",
    "AssemblyResult",
    "assemble_context_map",
    "assemble_context_map_with_metrics",
]

LABEL_HYPOTHESIS_MERGE_POLICY_ID = "label-hypothesis-merge-by-text-v1"
"""Versioned identity of the rule that merges same-label hypotheses into one ``LabelHypothesis``.

Unlike the Entity Resolution and Spatial Relations policies this module translates, this one is
not borrowed from another capability: it is assembly's own rule (see
:func:`_translate_semantic_state`'s docstring for why merging by label text is not a flattening
of ambiguity), so it is versioned the same conventional way every other policy identity in this
codebase is (a ``-v<N>`` suffix; see :func:`_split_policy_id`) and owned directly by this module.
"""

_POLICY_VERSION_SUFFIX = re.compile(r"-(v\d+)$")


def assemble_context_map(
    *,
    context_map_id: ContextMapId,
    metadata: ContextMapMetadata,
    geometric_map_location: Path,
    entity_resolution_location: Path | None = None,
    spatial_relations_location: Path | None = None,
    additional_lineage: tuple[UpstreamArtifact, ...] = (),
) -> ContextMap:
    """Compose a validated :class:`ContextMap` from real upstream runs, by reference.

    This is pure composition: it opens each run's own public reader, translates what it already
    decided into the map's schema and pins every located artifact by the digest of its manifest
    (:func:`~contextmap.artifact.artifact_digest`). It never performs entity resolution or
    relation inference, and it never copies geometry: ``geometric_map_location`` is opened only
    for its identity and point count.

    Args:
        context_map_id: Identity of the map being assembled; unique per immutable artifact.
        metadata: What the map is, where it came from and how it was created — including the
            explicit, versioned assembly policy (``metadata.creation.assembly_policy``). Nothing
            here is derived from Entity Resolution or Spatial Relations, so this function never
            invents a frame, a time window or a declared capability on the caller's behalf; a
            declaration that disagrees with what is composed is rejected by
            :class:`~contextmap.artifact.ContextMap` itself, and one that disagrees with the
            geometric map this function actually opens is rejected here
            (:func:`_require_metadata_matches_geometry`).
        geometric_map_location: Directory of the immutable geometric-map artifact the map
            references. Required: every map has geometry.
        entity_resolution_location: Directory of the Entity Resolution run whose resolved
            entities become this map's entities; ``None`` for a geometry-only map.
        spatial_relations_location: Directory of the Spatial Relations run whose relations
            become this map's relations; ``None`` when the map declares no relations. Requires
            ``entity_resolution_location``, since a relation names entities.
        additional_lineage: Upstream artifacts this function itself never opens but that
            ``metadata`` still requires in the lineage. Every ``SEQUENCE`` artifact named by
            ``metadata.source_sequences`` is the one this always needs: pinning it needs the
            ingestion artifact's own digest, and ``contextmap.artifact`` has no dependency on
            ``contextmap.ingestion`` to compute one (see ``tests/architecture/test_boundaries.
            py``). An optional evidence artifact (a perception run, a point-representation run, a
            human-annotation set) that some other ``origin`` this function did not build might
            cite is another. A *structural* kind (``ArtifactKind.is_structural``: the geometric
            map, an Entity Resolution run, a Spatial Relations run) and ``SEMANTIC_MAP`` are never
            accepted here: this function always derives those itself, from the real runs it opens
            (or computes them for free from a manifest it already opened), and accepting one from
            the caller instead would let a map cite a run, and declare the capability it backs,
            without this function ever verifying that run actually produced what the map claims
            (see issue #541). Merged with the geometry, entity-resolution, semantic-map,
            semantic-fusion and spatial-relations entries this function does compute; a duplicate
            ``artifact_id`` is rejected by :class:`~contextmap.artifact.ContextMap` itself, never
            silently overridden.

    Returns:
        A :class:`ContextMap` whose own construction has already validated every reference,
        the provenance closure and the declared capabilities. Not yet written to disk: persisting
        it is :class:`~contextmap.artifact.ContextMapArtifactWriter`'s job.

    Raises:
        ValueError: If ``spatial_relations_location`` is given without
            ``entity_resolution_location``, or ``additional_lineage`` carries a structural or
            ``SEMANTIC_MAP`` artifact.
        UpstreamArtifactError: If the Entity Resolution or Spatial Relations run was built over a
            different geometric map than ``geometric_map_location``, the Spatial Relations run
            was built over a different Entity Resolution run than ``entity_resolution_location``
            (a dangling or swapped structural dependency), or ``metadata`` disagrees with the
            sequence, selection, frame or time window the opened geometric map actually reports.
        UnresolvedReferenceError: If a relation names a resolved entity that is not among the
            entities this run actually has (defensive: :meth:`SpatialRelationsRunReader.
            validate_resolution` already reports this as an upstream-mismatch problem first).
    """
    return ContextMap(
        **_assemble_fields(
            context_map_id=context_map_id,
            metadata=metadata,
            geometric_map_location=geometric_map_location,
            entity_resolution_location=entity_resolution_location,
            spatial_relations_location=spatial_relations_location,
            additional_lineage=additional_lineage,
        )
    )


@dataclass(frozen=True, kw_only=True)
class AssemblyMetrics:
    """Runtime, memory and size measurements of one assembly execution (issue #537).

    Attributes:
        translation_duration_seconds: Wall time opening the upstream runs and translating their
            records into the map's schema, up to (not including) constructing the ``ContextMap``.
        validation_duration_seconds: Wall time constructing the ``ContextMap`` itself, which is
            exactly when every invariant (reference integrity, provenance closure, declared
            capabilities) is checked: :mod:`contextmap.artifact.composition` and the lineage
            checks run entirely inside ``ContextMap.__post_init__``, so this is the map's real
            validation cost, not a separate pass this function invents. This function never
            writes the map to disk, so there is no write duration here; see
            :func:`write_context_map_with_metrics` for that separate measurement (issue #537
            asks for the two to be reported separately, not for this function to also write).
        peak_memory_bytes: Peak Python-allocated memory observed during the call
            (:mod:`tracemalloc`), from the moment measurement started.
        entity_count: Entities the assembled map actually carries.
        relation_count: Relations the assembled map actually carries.
        lineage_count: Upstream artifacts the assembled map actually cites.
    """

    translation_duration_seconds: float
    validation_duration_seconds: float
    peak_memory_bytes: int
    entity_count: int
    relation_count: int
    lineage_count: int


@dataclass(frozen=True, kw_only=True)
class AssemblyResult:
    """The assembled map together with the measurements issue #537 asks assembly to report.

    Attributes:
        context_map: The assembled, already-validated map; identical to what
            :func:`assemble_context_map` would return for the same arguments.
        metrics: How long assembly took, its peak memory, and the size of what it produced.
    """

    context_map: ContextMap
    metrics: AssemblyMetrics


def assemble_context_map_with_metrics(
    *,
    context_map_id: ContextMapId,
    metadata: ContextMapMetadata,
    geometric_map_location: Path,
    entity_resolution_location: Path | None = None,
    spatial_relations_location: Path | None = None,
    additional_lineage: tuple[UpstreamArtifact, ...] = (),
) -> AssemblyResult:
    """Assemble a :class:`ContextMap` exactly like :func:`assemble_context_map`, measuring the run.

    Issue #537 asks assembly runtime, peak memory and entity/relation/lineage counts to be
    measured and reported; this wraps the pure translation with that instrumentation instead of
    changing :func:`assemble_context_map`'s own return type, so every existing caller of the
    plain function is unaffected. See :class:`AssemblyMetrics` for what is measured.

    Args:
        context_map_id: See :func:`assemble_context_map`.
        metadata: See :func:`assemble_context_map`.
        geometric_map_location: See :func:`assemble_context_map`.
        entity_resolution_location: See :func:`assemble_context_map`.
        spatial_relations_location: See :func:`assemble_context_map`.
        additional_lineage: See :func:`assemble_context_map`.

    Returns:
        The assembled map and its metrics.

    Raises:
        Same as :func:`assemble_context_map`.
    """
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    try:
        translation_started = time.perf_counter()
        fields = _assemble_fields(
            context_map_id=context_map_id,
            metadata=metadata,
            geometric_map_location=geometric_map_location,
            entity_resolution_location=entity_resolution_location,
            spatial_relations_location=spatial_relations_location,
            additional_lineage=additional_lineage,
        )
        translation_ended = time.perf_counter()
        context_map = ContextMap(**fields)
        validation_ended = time.perf_counter()
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()
    metrics = AssemblyMetrics(
        translation_duration_seconds=translation_ended - translation_started,
        validation_duration_seconds=validation_ended - translation_ended,
        peak_memory_bytes=peak_memory_bytes,
        entity_count=len(context_map.entities),
        relation_count=len(context_map.relations),
        lineage_count=len(context_map.lineage),
    )
    return AssemblyResult(context_map=context_map, metrics=metrics)


@dataclass(frozen=True, kw_only=True)
class WriteMetrics:
    """Wall time spent persisting an already-assembled map (issue #537).

    Attributes:
        write_duration_seconds: Wall time of :meth:`~contextmap.artifact.serialization.writer.
            ContextMapArtifactWriter.write` itself: opening and verifying the upstream artifacts
            it cites, encoding every record table and publishing the directory atomically.
    """

    write_duration_seconds: float


def write_context_map_with_metrics(
    context_map: ContextMap,
    *,
    output_dir: Path,
    upstream_locations: Mapping[str, Path],
    written_at: datetime | None = None,
) -> tuple[ContextMapArtifactManifest, WriteMetrics]:
    """Persist ``context_map`` exactly like :class:`ContextMapArtifactWriter`, measuring the write.

    Kept separate from :func:`assemble_context_map_with_metrics`: persisting an already-assembled
    map is the writer's own responsibility, not assembly's (:class:`ContextMapArtifactWriter`'s
    own docstring), and issue #537 asks validation and write duration to be reported as two
    separate measurements, not folded into one "assembly" number. This adds only the wall-clock
    measurement around an otherwise unmodified call to the writer.

    Args:
        context_map: The map to persist; typically the ``context_map`` of an
            :class:`AssemblyResult`.
        output_dir: See :meth:`ContextMapArtifactWriter.__init__`.
        upstream_locations: See :meth:`ContextMapArtifactWriter.write`.
        written_at: See :meth:`ContextMapArtifactWriter.__init__`.

    Returns:
        The published manifest and how long the write itself took.

    Raises:
        Same as :meth:`ContextMapArtifactWriter.write`.
    """
    writer = ContextMapArtifactWriter(output_dir=output_dir, written_at=written_at)
    write_started = time.perf_counter()
    manifest = writer.write(context_map, upstream_locations=upstream_locations)
    write_ended = time.perf_counter()
    return manifest, WriteMetrics(write_duration_seconds=write_ended - write_started)


def _assemble_fields(
    *,
    context_map_id: ContextMapId,
    metadata: ContextMapMetadata,
    geometric_map_location: Path,
    entity_resolution_location: Path | None,
    spatial_relations_location: Path | None,
    additional_lineage: tuple[UpstreamArtifact, ...],
) -> dict[str, Any]:
    """Do the actual translation, returning ``ContextMap``'s constructor arguments.

    Split from :func:`assemble_context_map` so :func:`assemble_context_map_with_metrics` can time
    "translation" (this function) separately from "validation" (constructing the ``ContextMap``
    itself) without duplicating the translation logic.
    """
    if spatial_relations_location is not None and entity_resolution_location is None:
        raise ValueError(
            "spatial_relations_location requires entity_resolution_location: a relation names "
            "entities, and a map cannot declare relations without entities"
        )
    _reject_structural_additional_lineage(additional_lineage)

    with GeometricMapArtifactReader(geometric_map_location) as geometry_reader:
        geometry_manifest = geometry_reader.manifest
    _require_metadata_matches_geometry(metadata, geometry_manifest)
    geometry_ref = GeometricMapLink(
        map_id=geometry_manifest.map_id, point_count=geometry_manifest.point_count
    )
    lineage = [
        _geometry_upstream_artifact(geometry_manifest, geometric_map_location),
        *additional_lineage,
    ]

    entities: tuple[ContextEntity, ...] = ()
    relations: tuple[ContextRelation, ...] = ()
    if entity_resolution_location is not None:
        resolution_reader = EntityResolutionRunReader(entity_resolution_location)
        _require_same_geometric_map(
            declared=resolution_reader.manifest.lineage.geometric_map_id,
            referenced=geometry_manifest.map_id,
            owner=f"entity resolution run {resolution_reader.run_id!r}",
        )
        materialization = resolution_reader.materialization()
        entities, identities, fusion_lineage = _translate_entities(
            context_map_id, resolution_reader.resolved_entities(), materialization.policy
        )
        lineage.append(_resolution_upstream_artifact(resolution_reader, entity_resolution_location))
        lineage.extend(_semantic_map_upstream_artifacts(resolution_reader.manifest.lineage))
        lineage.extend(fusion_lineage)

        if spatial_relations_location is not None:
            relations_reader = SpatialRelationsRunReader(spatial_relations_location)
            _require_same_geometric_map(
                declared=relations_reader.manifest.lineage.geometric_map_id,
                referenced=geometry_manifest.map_id,
                owner=f"spatial relations run {relations_reader.manifest.run_id!r}",
            )
            problems = relations_reader.validate_resolution(resolution_reader)
            if problems:
                raise UpstreamArtifactError(
                    f"spatial relations run {relations_reader.manifest.run_id!r} does not match "
                    f"entity resolution run {resolution_reader.run_id!r} as its own lineage "
                    "records: " + "; ".join(problems)
                )
            relations = _translate_relations(relations_reader, identities, geometry_manifest.map_id)
            lineage.append(
                _relations_upstream_artifact(relations_reader, spatial_relations_location)
            )

    return {
        "context_map_id": context_map_id,
        "schema_version": CONTEXT_MAP_SCHEMA_VERSION,
        "metadata": metadata,
        "geometry_ref": geometry_ref,
        "entities": entities,
        "relations": relations,
        "lineage": tuple(sorted(lineage, key=lambda item: item.artifact_id)),
    }


def _reject_structural_additional_lineage(additional_lineage: tuple[UpstreamArtifact, ...]) -> None:
    """Refuse an ``additional_lineage`` entry this function must always derive itself.

    ``additional_lineage`` exists only to pin what this function structurally cannot derive on
    its own (a ``SEQUENCE``, or optional evidence some ``origin`` this function did not build
    cites — see this function's own docstring). A structural kind or ``SEMANTIC_MAP`` is always
    either opened directly here or computed for free from a manifest already opened; accepting
    one from the caller instead would let a map claim to be grounded in a run — citing it in
    lineage, declaring the capability it backs — without this function ever opening that run to
    verify it (issue #541, blocker 1: a caller could otherwise pass an ``ENTITY_RESOLUTION_RUN``
    here, never supply ``entity_resolution_location``, declare ``ENTITIES`` and get back a map
    with ``entities=()``).

    Raises:
        ValueError: If an entry's kind is structural (``ArtifactKind.is_structural``) or
            ``SEMANTIC_MAP``.
    """
    for item in additional_lineage:
        if item.kind.is_structural or item.kind is ArtifactKind.SEMANTIC_MAP:
            raise ValueError(
                f"additional_lineage cannot carry a {item.kind.name} artifact "
                f"({item.artifact_id!r}): assembly always derives every structural dependency "
                "(GEOMETRIC_MAP, ENTITY_RESOLUTION_RUN, SPATIAL_RELATIONS_RUN) and SEMANTIC_MAP "
                "itself, from the real runs it opens; passing one here would let a caller claim "
                "a run's lineage without this function ever verifying it"
            )


def _require_metadata_matches_geometry(
    metadata: ContextMapMetadata, manifest: GeometricMapArtifactManifest
) -> None:
    """Reject metadata that disagrees with the geometric map this function actually opened.

    ``metadata`` is caller-supplied and never derived from the runs this function opens; without
    this check, a caller could assemble a map that *declares* one sequence, selection or frame
    while the geometry underneath actually came from another (issue #541, should-strengthen 5).
    This checks every identity the opened manifest itself can prove: the sequence and selection
    the map was built from, the frame every coordinate is expressed in, and, when the two share a
    clock domain, the observation window.

    Raises:
        UpstreamArtifactError: If the geometric map's sequence/selection is not among
            ``metadata.source_sequences``, its frame disagrees with ``metadata.frame.frame_id``,
            or its time range disagrees with ``metadata.time_bounds`` on a shared clock.
    """
    declared_sequences = {
        (item.sequence_artifact_id, item.selection_id) for item in metadata.source_sequences
    }
    geometry_sequence = (str(manifest.sequence_artifact_id), manifest.selection_id)
    if geometry_sequence not in declared_sequences:
        raise UpstreamArtifactError(
            f"geometric map {manifest.map_id!r} was built from sequence "
            f"{geometry_sequence[0]!r} selection {geometry_sequence[1]!r}, which "
            f"metadata.source_sequences does not declare ({sorted(declared_sequences)!r})"
        )
    if metadata.frame.frame_id != str(manifest.map_frame):
        raise UpstreamArtifactError(
            f"geometric map {manifest.map_id!r} is in frame {manifest.map_frame!r}, not the "
            f"declared metadata.frame.frame_id {metadata.frame.frame_id!r}"
        )
    window = metadata.time_bounds
    if window.start.clock_id == manifest.clock_id:
        declared = (window.start.total_nanoseconds(), window.end.total_nanoseconds())
        actual = (manifest.start_time_ns, manifest.end_time_ns)
        if declared != actual:
            raise UpstreamArtifactError(
                f"geometric map {manifest.map_id!r} covers time [{actual[0]}, {actual[1]}] ns "
                f"on clock {manifest.clock_id!r}, not the declared metadata.time_bounds "
                f"[{declared[0]}, {declared[1]}] ns"
            )


def _require_same_geometric_map(*, declared: object, referenced: object, owner: str) -> None:
    """Reject a run built over a geometric map other than the one this map references.

    Um `ResolvedEntity`/`Relation` que aponta para outro mapa geométrico nunca é composto
    silenciosamente: a montagem falha aqui, antes de qualquer referência de geometria ser
    copiada para uma entidade.
    """
    if declared != referenced:
        raise UpstreamArtifactError(
            f"{owner} was built over geometric map {declared!r}, not the referenced {referenced!r}"
        )


def _geometry_upstream_artifact(
    manifest: GeometricMapArtifactManifest, location: Path
) -> UpstreamArtifact:
    return UpstreamArtifact(
        artifact_id=str(manifest.map_id),
        kind=ArtifactKind.GEOMETRIC_MAP,
        content_identity=artifact_digest(location),
        configuration_fingerprint=manifest.configuration_fingerprint,
        code_version=manifest.code_version,
        model_identities=(),
    )


def _resolution_upstream_artifact(
    reader: EntityResolutionRunReader, location: Path
) -> UpstreamArtifact:
    """The ``ENTITY_RESOLUTION_RUN`` lineage entry.

    ``UpstreamArtifact.configuration_fingerprint`` documents "hash of the effective
    configuration that produced it" — the whole run, not one of its roles. An Entity Resolution
    run has several independently-configured roles (materialization, candidate retrieval,
    comparison gates, ...), each with its own fingerprint; assembly only opens the
    materialization role, so citing its fingerprint here would misrepresent it as the run's
    single effective configuration (review of issue #541, second round). This stays ``None``
    until the capability exposes one fingerprint for the whole run; the materialization
    fingerprint itself is still available, unrenamed, wherever
    ``EntityResolutionRunReader.materialization().policy.configuration_fingerprint`` is reachable.
    """
    return UpstreamArtifact(
        artifact_id=str(reader.run_id),
        kind=ArtifactKind.ENTITY_RESOLUTION_RUN,
        content_identity=artifact_digest(location),
        configuration_fingerprint=None,
        code_version=reader.manifest.code_version,
        model_identities=(),
    )


def _semantic_map_upstream_artifacts(lineage: ResolutionRunLineage) -> tuple[UpstreamArtifact, ...]:
    """Derive the ``SEMANTIC_MAP`` lineage entry of every semantic map an entity may reference.

    Unlike ``SEQUENCE``, this one is free: the Entity Resolution manifest this module already
    opened names the exact semantic-mapping run it was built from and that run's own pinned
    digest (:func:`~contextmap.entity_resolution.mapping_artifact_digest`, computed once by
    Semantic Mapping's writer and carried unchanged since). No extra reader is opened.

    A resolution run built the ordinary way (:func:`~contextmap.entity_resolution.
    lineage_from_mapping_manifest`) names exactly one semantic-mapping run, so
    ``semantic_map_ids`` has exactly one entry in practice and every entry gets the same digest
    honestly; ``ResolutionRunLineage`` has no way to record a different digest per semantic map,
    so this function cannot invent one either if a lineage were ever built with more than one.
    """
    return tuple(
        UpstreamArtifact(
            artifact_id=str(semantic_map_id),
            kind=ArtifactKind.SEMANTIC_MAP,
            content_identity=lineage.semantic_mapping_artifact_digest,
            configuration_fingerprint=None,
            code_version=None,
            model_identities=(),
        )
        for semantic_map_id in lineage.semantic_map_ids
    )


def _relations_upstream_artifact(
    reader: SpatialRelationsRunReader, location: Path
) -> UpstreamArtifact:
    """The ``SPATIAL_RELATIONS_RUN`` lineage entry.

    Same reasoning as :func:`_resolution_upstream_artifact`: a Spatial Relations run has several
    independently-configured policies (``frame_conventions``, ``candidate``,
    ``geometry_summary``, ``geometric``, ``contact``). A single relation's own
    ``RelationProvenance.configuration_fingerprint`` covers only the decision-policy role, not
    the run's other policies, so citing it here would misrepresent it as the whole run's
    effective configuration (review of issue #541, second round). This stays ``None`` until the
    capability exposes one fingerprint for the whole run.
    """
    return UpstreamArtifact(
        artifact_id=str(reader.manifest.run_id),
        kind=ArtifactKind.SPATIAL_RELATIONS_RUN,
        content_identity=artifact_digest(location),
        configuration_fingerprint=None,
        code_version=reader.manifest.code_version,
        model_identities=(),
    )


def _split_policy_id(policy_id: str) -> tuple[str, str]:
    """Split a ``<rule>-v<N>`` policy id into its rule identity and its version.

    Entity Resolution, Spatial Relations and this module's own rules all encode a policy's
    version as a numbered suffix of the policy id by convention (for example
    ``connected-components-materialization-v1``); this recovers both parts so
    ``contextmap.artifact.PolicyRef``, which requires an explicit ``version`` distinct from the
    rule identity ("a different version is a different rule"), gets neither an invented value nor
    an unrelated identity conflated into it, such as an execution's configuration fingerprint
    (issue #541, blocker 3) or a vocabulary's taxonomy version.

    Raises:
        ValueError: If the policy id does not end with the conventional ``-v<N>`` suffix.
    """
    match = _POLICY_VERSION_SUFFIX.search(policy_id)
    if match is None:
        raise ValueError(
            f"policy id {policy_id!r} does not end with the conventional '-v<N>' version suffix "
            "this translation relies on to recover an explicit version"
        )
    return policy_id[: match.start()], match.group(1)


def _policy_ref(policy_id: str) -> PolicyRef:
    """Build the map's ``PolicyRef`` from a ``<rule>-v<N>`` identity (:func:`_split_policy_id`)."""
    rule_id, version = _split_policy_id(policy_id)
    return PolicyRef(policy_id=rule_id, version=version)


def _translate_resolution_policy(policy: ResolutionPolicyRef) -> PolicyRef:
    """Translate Entity Resolution's materialization policy into the map's ``PolicyRef``.

    Entity Resolution's own :class:`~contextmap.entity_resolution.PolicyRef` is
    ``(policy_id, configuration_fingerprint)`` with no separate ``version`` field, and its own
    docstring is explicit that the two are "never confused": ``policy_id`` (for example
    ``connected-components-materialization-v1``) is the versioned rule identity, and
    ``configuration_fingerprint`` is one execution's effective parameters. Two different
    configurations of the *same* rule version are two different executions, not two different
    rules, so ``configuration_fingerprint`` must not become this ``PolicyRef``'s ``version``
    (issue #541, blocker 3) — it is cited instead on the run's own lineage entry, see
    :func:`_resolution_upstream_artifact`.
    """
    return _policy_ref(policy.policy_id)


def _translate_relation_policy(provenance: RelationProvenance) -> PolicyRef:
    """Translate Spatial Relations' decision policy into the map's ``PolicyRef``.

    Same root issue as Entity Resolution's translation, and the same fix: ``decision_policy_id``
    (for example ``conservative-relation-decision-v1``) already encodes the rule's version as a
    ``-v<N>`` suffix by this codebase's convention (:func:`_split_policy_id`). ``taxonomy_version``
    is a different identity again — it versions the *predicate vocabulary*
    (``RelationPredicate``'s own definitions), not the decision policy's rule: two runs under the
    exact same decision-policy version can use different taxonomy versions, and the reverse, so
    it is not this ``PolicyRef``'s ``version`` either. It is not represented in the map's
    ``PolicyRef`` at all: like the attributes and detailed uncertainty ``docs/assembly.md``
    already documents as not copied, it stays reachable, unchanged, on the very ``Relation``
    record ``ContextRelation.source_relation_id`` cites in the Spatial Relations run.
    """
    return _policy_ref(provenance.decision_policy_id)


def _translate_entities(
    context_map_id: ContextMapId,
    resolved: ResolvedEntitySet,
    materialization_policy: ResolutionPolicyRef,
) -> tuple[
    tuple[ContextEntity, ...],
    dict[ResolvedEntityReference, ContextEntityReference],
    tuple[UpstreamArtifact, ...],
]:
    """Translate every resolved entity of a run into a ``ContextEntity``, in canonical order.

    ``resolved.entities`` is already sorted and unique by ``resolved_entity_id``
    (:class:`~contextmap.entity_resolution.ResolvedEntitySet` requires it at construction), a
    hash of the members and therefore independent of how the run's own table happens to be
    stored or iterated. Local identities are assigned in that order, so two assemblies of the
    same run always number entities the same way.

    Returns:
        The entities, the map from resolved-entity reference to local entity reference (used to
        translate relation endpoints), and the ``SEMANTIC_FUSION_RUN`` lineage entries the
        entities' own ``LabelHypothesis`` origins actually cite (see :func:`_translate_semantic_
        state`), deduplicated across every entity.
    """
    policy = _translate_resolution_policy(materialization_policy)
    width = max(4, len(str(len(resolved.entities))))
    entities: list[ContextEntity] = []
    identities: dict[ResolvedEntityReference, ContextEntityReference] = {}
    fusion_artifacts: dict[str, UpstreamArtifact] = {}
    for index, item in enumerate(resolved.entities, start=1):
        local_id = ContextEntityId(f"entity-{index:0{width}d}")
        record = UpstreamRecordRef(
            artifact_id=str(item.resolution_run_id), record_id=str(item.resolved_entity_id)
        )
        origin = EvidenceOrigin(
            kind=DerivationKind.MULTIVIEW_FUSED, derived_from=(record,), policy=policy
        )
        semantic_state, used_fusion_artifacts = _translate_semantic_state(
            item.semantic_state, evidence=item.evidence
        )
        for artifact in used_fusion_artifacts:
            _add_fusion_artifact(fusion_artifacts, artifact)
        entities.append(
            ContextEntity(
                entity_id=local_id,
                source=item.reference,
                member_entities=item.member_entity_refs,
                resolution_decisions=item.resolution_decision_refs,
                unresolved_neighbors=item.unresolved_neighbor_refs,
                geometry_refs=item.geometry.geometry_refs,
                semantic_state=semantic_state,
                origin=origin,
            )
        )
        identities[item.reference] = ContextEntityReference(
            context_map_id=context_map_id, entity_id=local_id
        )
    fusion_lineage = tuple(
        sorted(fusion_artifacts.values(), key=lambda artifact: artifact.artifact_id)
    )
    return tuple(entities), identities, fusion_lineage


def _add_fusion_artifact(seen: dict[str, UpstreamArtifact], artifact: UpstreamArtifact) -> None:
    """Add a derived ``SEMANTIC_FUSION_RUN`` lineage entry once, across every entity.

    Raises:
        ProvenanceError: If the same fusion run id was already derived with a different content
            identity — genuinely two different artifacts sharing one id, never silently resolved.
    """
    existing = seen.get(artifact.artifact_id)
    if existing is not None and existing.content_identity != artifact.content_identity:
        raise ProvenanceError(
            f"semantic fusion run {artifact.artifact_id!r} was derived with two different "
            f"content identities: {existing.content_identity!r} and {artifact.content_identity!r}"
        )
    seen[artifact.artifact_id] = artifact


def _translate_semantic_state(
    state: ResolvedSemanticState, *, evidence: EntityEvidenceLinks
) -> tuple[ContextSemanticState, tuple[UpstreamArtifact, ...]]:
    """Translate a resolved entity's belief into the map's semantic-state schema.

    This is the one genuinely lossy step of assembly, and it is lossy by the target schema's own
    design, not by an assembly shortcut: ``ResolvedSemanticState.hypotheses`` is keyed by
    ``(fused_evidence_id, hypothesis_id)`` and may hold two entries with the *same* label text
    (independent evidence that happens to agree), while :class:`LabelHypothesis` is keyed by
    label text alone (``ContextSemanticState`` refuses two hypotheses with the same label). Two
    items that agree on a label are reinforcing evidence for one candidate, not two competing
    ones, so collapsing them by label is not a flattening of *ambiguity* — ambiguity is exactly
    the presence of more than one *distinct* label, which this function still preserves in full:
    every distinct label proposed by any member survives as its own hypothesis, and
    ``ambiguity_state`` is carried over unchanged (``AmbiguityState`` and ``AmbiguityStatus`` are
    the same four values). Neither a score nor a hypothesis count is invented for the merge: the
    schema has no field for either, and the evidence behind each label stays reachable through
    ``origin``.

    Each merged label's own ``origin`` cites the Semantic Fusion evidence it actually rests on —
    every distinct ``(fusion_run_id, fused_evidence_id)`` of the ``EntityHypothesis`` records that
    proposed it, found by matching each hypothesis's own ``fused_evidence_id`` against ``evidence.
    fused_evidence`` — never the resolved-entity record ``ContextEntity.origin`` cites (issue
    #541, blocker 2): Entity Resolution resolves identity and aggregates pre-existing belief, it
    does not propose the visual hypothesis a label is.

    ``attributes`` and the detailed ``uncertainty`` records of ``ResolvedSemanticState`` have no
    corresponding field in ``ContextSemanticState`` (composition.py's own module docstring:
    "Support values are deliberately not copied; the evidence behind a result stays reachable
    through the upstream artifact"), so they are not copied here either; a consumer that needs
    them opens the entity-resolution run this entity's ``origin`` and ``source`` both cite.

    Returns:
        The translated semantic state, and the ``SEMANTIC_FUSION_RUN`` lineage entries its
        hypotheses' origins actually cite, deduplicated by fusion run id.

    Raises:
        ProvenanceError: If a hypothesis cites a ``fused_evidence_id`` this entity's own
            ``evidence.fused_evidence`` does not have — a resolved entity whose semantic state and
            evidence links disagree about their own members, which materialization's own
            invariants should already prevent.
    """
    fused_by_id: dict[str, FusedEvidenceRef] = {
        str(ref.fused_evidence_id): ref for ref in evidence.fused_evidence
    }
    by_label: dict[str, list[EntityHypothesis]] = {}
    for item in state.hypotheses:
        by_label.setdefault(item.label, []).append(item)

    hypotheses: list[LabelHypothesis] = []
    fusion_artifacts: dict[str, UpstreamArtifact] = {}
    for label in sorted(by_label):
        origin, used_fusion_artifacts = _label_hypothesis_origin(
            label, by_label[label], fused_by_id
        )
        hypotheses.append(LabelHypothesis(label=label, origin=origin))
        for artifact in used_fusion_artifacts:
            _add_fusion_artifact(fusion_artifacts, artifact)

    status = AmbiguityStatus(state.ambiguity_state.value)
    semantic_state = ContextSemanticState(status=status, hypotheses=tuple(hypotheses))
    return semantic_state, tuple(fusion_artifacts.values())


def _label_hypothesis_origin(
    label: str,
    hypotheses: list[EntityHypothesis],
    fused_by_id: Mapping[str, FusedEvidenceRef],
) -> tuple[EvidenceOrigin, tuple[UpstreamArtifact, ...]]:
    """The evidence-level origin of one merged ``LabelHypothesis``.

    ``kind`` stays ``MULTIVIEW_FUSED``: the label was genuinely accumulated from one or more
    evidence contributions by a versioned rule, which is exactly what that category describes —
    the bug this fixes was never the category, only which record ``derived_from`` cited. The
    policy this cites is deliberately *not* Entity Resolution's materialization policy: merging
    the (possibly several) fused-evidence entries that proposed this exact label text into one
    ``LabelHypothesis`` is this module's own rule (:data:`LABEL_HYPOTHESIS_MERGE_POLICY_ID`), a
    genuine derivation distinct from identity resolution — reusing the materialization policy here
    would just move the same epistemic-level conflation one field over.

    Raises:
        ProvenanceError: If a hypothesis's ``fused_evidence_id`` is not in ``fused_by_id``.
    """
    records: set[UpstreamRecordRef] = set()
    artifacts: dict[str, UpstreamArtifact] = {}
    for item in hypotheses:
        fused = fused_by_id.get(str(item.fused_evidence_id))
        if fused is None:
            raise ProvenanceError(
                f"hypothesis {item.hypothesis_id!r} of label {label!r} cites fused evidence "
                f"{item.fused_evidence_id!r}, which this entity's own evidence links do not have"
            )
        records.add(
            UpstreamRecordRef(
                artifact_id=str(fused.fusion_run_id), record_id=str(fused.fused_evidence_id)
            )
        )
        artifacts[str(fused.fusion_run_id)] = UpstreamArtifact(
            artifact_id=str(fused.fusion_run_id),
            kind=ArtifactKind.SEMANTIC_FUSION_RUN,
            content_identity=fused.fusion_artifact_digest,
            configuration_fingerprint=None,
            code_version=None,
            model_identities=(),
        )
    origin = EvidenceOrigin(
        kind=DerivationKind.MULTIVIEW_FUSED,
        derived_from=tuple(sorted(records)),
        policy=_policy_ref(LABEL_HYPOTHESIS_MERGE_POLICY_ID),
    )
    return origin, tuple(artifacts.values())


def _translate_relations(
    reader: SpatialRelationsRunReader,
    identities: Mapping[ResolvedEntityReference, ContextEntityReference],
    geometry_map_id: MapId,
) -> tuple[ContextRelation, ...]:
    """Translate every relation of a run into a ``ContextRelation``, in canonical order.

    ``iter_relations()`` yields relations in the run's own canonical order (subject, predicate,
    object), which has no relationship to ``relation_id`` (a digest of the same triple): sorting
    by ``relation_id`` here is what actually makes the map's own numbering independent of that
    reader order, not a no-op.
    """
    ordered = sorted(reader.iter_relations(), key=lambda item: str(item.relation_id))
    width = max(4, len(str(len(ordered))))
    run_id = reader.manifest.run_id
    relations: list[ContextRelation] = []
    for index, relation in enumerate(ordered, start=1):
        try:
            subject = identities[relation.subject_entity_ref]
            object_ = identities[relation.object_entity_ref]
        except KeyError as error:
            raise UnresolvedReferenceError(
                f"relation {relation.relation_id!r} of run {run_id!r} references resolved "
                f"entity {error.args[0]!r}, which entity resolution does not have"
            ) from None
        relations.append(
            ContextRelation(
                relation_id=ContextRelationId(f"relation-{index:0{width}d}"),
                source_run_id=run_id,
                source_relation_id=relation.relation_id,
                subject=subject,
                predicate=relation.predicate,
                object=object_,
                state=relation.state,
                uncertainty_kinds=tuple(
                    sorted(
                        {item.kind for item in relation.uncertainty}, key=lambda kind: kind.value
                    )
                ),
                origin=_relation_origin(reader, run_id, geometry_map_id, relation),
            )
        )
    return tuple(relations)


def _relation_origin(
    reader: SpatialRelationsRunReader,
    run_id: SpatialRelationsRunId,
    geometry_map_id: MapId,
    relation: Relation,
) -> EvidenceOrigin:
    """The evidence-level origin of one translated relation.

    ``origin.derived_from`` cites the relation's own evidence (``relation_evidence_refs``) and the
    geometry that evidence was measured over — reached by reading the evidence records this same
    already-open run's reader owns (``evidence_of``), never a second artifact — plus, for a
    derived relation (an inverse or symmetric twin), the relation it was generated from
    (``Relation.derived_from``). It never cites the relation's own record as its evidence (issue
    #541, should-fix 4): ``ContextRelation.source_run_id``/``source_relation_id`` already name
    that record structurally (see ``docs/lineage.md``), so repeating it here would make an
    ``UNRESOLVED`` relation with no evidence at all indistinguishable from one whose evidence
    happens to be exactly its own record. The relation record is cited only as a last resort, when
    there is genuinely nothing else: an ``UNRESOLVED`` relation may have no evidence and no
    ``derived_from`` at all, and ``EvidenceOrigin`` still requires at least one citation.
    """
    records: set[UpstreamRecordRef] = set()
    for evidence_id in relation.relation_evidence_refs:
        records.add(UpstreamRecordRef(artifact_id=str(run_id), record_id=str(evidence_id)))
    for evidence in reader.evidence_of(relation):
        for measured in evidence.geometry:
            for geometry_ref in measured.geometry_refs:
                records.add(
                    UpstreamRecordRef(
                        artifact_id=str(geometry_map_id), record_id=str(geometry_ref.geometry_id)
                    )
                )
    if relation.derived_from is not None:
        records.add(
            UpstreamRecordRef(artifact_id=str(run_id), record_id=str(relation.derived_from))
        )
    if not records:
        # Uma relação UNRESOLVED por falta de evidência pode não ter nenhuma evidência nem
        # derived_from: EvidenceOrigin exige ao menos uma citação, e não há nada além do próprio
        # registro da relação para citar com honestidade nesse caso extremo.
        records.add(UpstreamRecordRef(artifact_id=str(run_id), record_id=str(relation.relation_id)))
    return EvidenceOrigin(
        kind=DerivationKind.GEOMETRY_DERIVED,
        derived_from=tuple(sorted(records)),
        policy=_translate_relation_policy(relation.provenance),
    )

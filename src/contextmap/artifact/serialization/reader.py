"""Lightweight, read-only reader of a ContextMapArtifact.

The reader opens an artifact from its own directory and answers what a consumer needs to
understand the map: its metadata and bounds, its entities and relations as the schema's own types,
the relations an entity takes part in, and the geometry its references point at. Entities and
relations are read one at a time through the byte-offset indexes, and the geometry is opened only
when it is asked for, as a memory map through the existing ``GeometricMapArtifactReader``;
nothing large is loaded.

The reader is a data reader, not a query engine. It has no search, no language, no planning, no
navigation and no inference: it loads, validates, resolves and round-trips the public artifact.
It never writes, never repairs, never falls back to anything (a ``debug/`` directory does not
exist for it) and needs nothing beyond the standard library, NumPy and the geometric-mapping
contracts, so no model, ROS or robotics runtime is imported.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

from contextmap.artifact.composition import ContextEntity, ContextRelation
from contextmap.artifact.metadata import ContextMapMetadata
from contextmap.artifact.models import ContextMap, GeometricMapLink
from contextmap.artifact.provenance import UpstreamArtifact
from contextmap.artifact.records import ContextMapRecordError, context_map_from_record
from contextmap.artifact.references import (
    ContextEntityReference,
    ForeignContextEntityReferenceError,
    UnknownContextEntityError,
)
from contextmap.artifact.serialization.decoding import (
    decode_entity,
    decode_geometry_link,
    decode_metadata,
    decode_relation,
    decode_upstream_artifacts,
)
from contextmap.artifact.serialization.dependencies import (
    DependencyResolution,
    DependencyStatus,
    read_inventory,
    resolve_dependency,
    verify_inventory,
)
from contextmap.artifact.serialization.directory import check_files_present, load_manifest
from contextmap.artifact.serialization.errors import (
    BrokenIndexError,
    ContextMapArtifactError,
    DependencyMismatchError,
    MissingDependencyError,
    RecordNotFoundError,
    UnresolvedReferenceError,
    UpstreamArtifactError,
)
from contextmap.artifact.serialization.layout import (
    ENTITIES,
    ENTITY_INDEX,
    ENTITY_RELATION_INDEX,
    GEOMETRY_REFERENCE,
    LINEAGE,
    MAP_METADATA,
    RELATION_INDEX,
    RELATIONS,
)
from contextmap.artifact.serialization.manifest import ContextMapArtifactManifest
from contextmap.artifact.serialization.tables import RecordTable
from contextmap.geometric_mapping import (
    Bounds3D,
    GeometricMapArtifactReader,
    GeometryPoint,
    GeometryReference,
    GeometrySource,
    MapArtifactError,
    geometry_index_of,
)

_TRAVERSAL_FIELDS = frozenset({"key", "as_subject", "as_object"})


class ContextMapArtifactReader:
    """Read-only access to one finished ContextMapArtifact.

    Use :meth:`open` and, when the geometry was read, leave the ``with`` block (or call
    :meth:`close`) so the memory map of the geometry payload is released.
    """

    def __init__(
        self,
        root: Path,
        manifest: ContextMapArtifactManifest,
        dependency_paths: Mapping[str, Path],
        verify_hashes: bool,
    ) -> None:
        """Wrap an already checked directory; use :meth:`open` instead."""
        self._root = root
        self._manifest = manifest
        self._dependency_paths = dict(dependency_paths)
        self._verify_hashes = verify_hashes
        self._context_map: ContextMap | None = None
        self._metadata: ContextMapMetadata | None = None
        self._link: GeometricMapLink | None = None
        self._entities: RecordTable | None = None
        self._relations: RecordTable | None = None
        self._traversal: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None
        self._geometry: GeometricMapArtifactReader | None = None
        self._closed = False

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        dependency_paths: Mapping[str, Path] | None = None,
        verify_hashes: bool = False,
    ) -> ContextMapArtifactReader:
        """Open an artifact directory after the checks that need no parsing.

        Opening reads the manifest, checks its format and schema versions and its content
        identity, and checks that every inventoried file is present with the recorded size. It
        parses no record and touches no geometry.

        Args:
            path: The artifact directory.
            dependency_paths: Where upstream artifacts are now, keyed by artifact id. Without an
                entry, the relative hint recorded in the manifest is tried. A path given here is
                the only place looked at for that artifact.
            verify_hashes: Also hash every file of this artifact, and the files of the geometric
                map when it is opened. A change that keeps a file's size is only seen this way,
                at the cost of reading every byte.

        Returns:
            The reader.

        Raises:
            IncompleteContextMapArtifactError: If ``path`` is not a finished artifact.
            UnsupportedFormatVersionError: If the format version is not supported.
            UnsupportedArtifactSchemaError: If the schema version cannot be read.
            ManifestError: If the manifest is malformed.
            MissingPayloadError: If an inventoried file is missing.
            ArtifactIntegrityError: If the manifest or a file does not match its identity, size
                or, with ``verify_hashes``, hash.
        """
        manifest = load_manifest(path)
        check_files_present(path, manifest, verify_hashes=verify_hashes)
        return cls(
            path,
            manifest,
            dependency_paths if dependency_paths is not None else {},
            verify_hashes,
        )

    def __enter__(self) -> ContextMapArtifactReader:
        """Return the reader for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the geometry payload."""
        self.close()

    @property
    def manifest(self) -> ContextMapArtifactManifest:
        """The manifest of the artifact: identities, inventory, payloads and dependencies."""
        return self._manifest

    def close(self) -> None:
        """Release the geometry; the reader refuses to read afterwards. Safe to call twice."""
        if self._geometry is not None:
            self._geometry.close()
            self._geometry = None
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContextMapArtifactError("the reader is closed")

    def _document(self, relative_path: str) -> dict[str, Any]:
        try:
            record = json.loads((self._root / relative_path).read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise ContextMapArtifactError(f"{relative_path} is not valid JSON ({error})") from error
        if not isinstance(record, dict):
            raise ContextMapArtifactError(f"{relative_path} must hold a JSON object")
        return record

    # -- the map ---------------------------------------------------------------------------

    def context_map(self) -> ContextMap:
        """Rebuild the whole schema object from the artifact.

        This reads every entity and relation; use the per-record methods to read less.

        Returns:
            The map, revalidated in full by the schema's own strict decoding.

        Raises:
            ContextMapArtifactError: If the stored records do not form a valid map.
        """
        self._ensure_open()
        if self._context_map is None:
            record = {
                "context_map_id": self._manifest.context_map_id,
                "schema_version": self._manifest.schema_version,
                "metadata": self._document(MAP_METADATA),
                "geometry_ref": self._document(GEOMETRY_REFERENCE),
                "entities": [line["record"] for line in self._entity_table().iter_lines()],
                "relations": [line["record"] for line in self._relation_table().iter_lines()],
                "lineage": self._document(LINEAGE).get("upstream_artifacts"),
            }
            try:
                self._context_map = context_map_from_record(record)
            except (ContextMapRecordError, ValueError) as error:
                raise ContextMapArtifactError(
                    f"the stored records do not form a valid map: {error}"
                ) from error
        return self._context_map

    def metadata(self) -> ContextMapMetadata:
        """What the map is: creation, sources, frame, bounds, time and declared capabilities.

        Only the metadata document is read; no entity or relation is.
        """
        self._ensure_open()
        if self._metadata is None:
            try:
                self._metadata = decode_metadata(self._document(MAP_METADATA))
            except (ContextMapRecordError, ValueError) as error:
                raise ContextMapArtifactError(
                    f"{MAP_METADATA} is not valid metadata: {error}"
                ) from error
        return self._metadata

    def map_bounds(self) -> Bounds3D:
        """The spatial extent of the map, expressed in the map frame it declares."""
        return self.metadata().bounds

    def geometry_link(self) -> GeometricMapLink:
        """The geometric map the map refers to, by identity and number of elements."""
        self._ensure_open()
        if self._link is None:
            try:
                self._link = decode_geometry_link(self._document(GEOMETRY_REFERENCE))
            except (ContextMapRecordError, ValueError) as error:
                raise ContextMapArtifactError(
                    f"{GEOMETRY_REFERENCE} is not a valid geometry reference: {error}"
                ) from error
        return self._link

    def lineage(self) -> tuple[UpstreamArtifact, ...]:
        """Every upstream artifact the map cites, with the identities needed to audit it."""
        self._ensure_open()
        try:
            return decode_upstream_artifacts(self._document(LINEAGE).get("upstream_artifacts"))
        except (ContextMapRecordError, ValueError) as error:
            raise ContextMapArtifactError(f"{LINEAGE} is not a valid lineage: {error}") from error

    # -- entities and relations -------------------------------------------------------------

    def _entity_table(self) -> RecordTable:
        self._ensure_open()
        if self._entities is None:
            self._entities = RecordTable(
                self._root / ENTITIES,
                self._root / ENTITY_INDEX,
                record_count=self._manifest.entity_count,
            )
        return self._entities

    def _relation_table(self) -> RecordTable:
        self._ensure_open()
        if self._relations is None:
            self._relations = RecordTable(
                self._root / RELATIONS,
                self._root / RELATION_INDEX,
                record_count=self._manifest.relation_count,
            )
        return self._relations

    def _own(self, reference: ContextEntityReference) -> str:
        if str(reference.context_map_id) != self._manifest.context_map_id:
            raise ForeignContextEntityReferenceError(
                f"the reference names the map {reference.context_map_id!r} but this artifact "
                f"holds {self._manifest.context_map_id!r}"
            )
        return str(reference.entity_id)

    def entity_ids(self) -> tuple[str, ...]:
        """The id of every entity, in file order."""
        return self._entity_table().keys

    def entity(self, reference: ContextEntityReference) -> ContextEntity:
        """Read one entity without reading the others.

        Args:
            reference: The entity, as ``(context_map_id, entity_id)``.

        Returns:
            The entity as the schema defines it, revalidated by the schema's decoding.

        Raises:
            ForeignContextEntityReferenceError: If the reference names another map.
            UnknownContextEntityError: If the map has no such entity.
            BrokenIndexError: If the index does not lead to that entity's line.
            ContextMapArtifactError: If the stored record is not a valid entity.
        """
        key = self._own(reference)
        try:
            line = self._entity_table().read(key)
        except RecordNotFoundError as error:
            raise UnknownContextEntityError(f"the map has no entity {key!r}") from error
        return self._entity_of(line, key)

    def entities(self) -> Iterator[ContextEntity]:
        """Stream every entity in id order, one line at a time."""
        for line in self._entity_table().iter_lines():
            yield self._entity_of(line, line["key"])

    def _entity_of(self, line: Mapping[str, Any], key: str) -> ContextEntity:
        try:
            entity = decode_entity(line)
        except (ContextMapRecordError, ValueError) as error:
            raise ContextMapArtifactError(
                f"the stored entity {key!r} is not a valid entity: {error}"
            ) from error
        if str(entity.entity_id) != key:
            raise BrokenIndexError(f"the line of {key!r} holds the entity {entity.entity_id!r}")
        return entity

    def relation_ids(self) -> tuple[str, ...]:
        """The id of every relation, in file order."""
        return self._relation_table().keys

    def relation(self, relation_id: str) -> ContextRelation:
        """Read one relation without reading the others.

        Args:
            relation_id: The relation id.

        Returns:
            The relation as the schema defines it.

        Raises:
            RecordNotFoundError: If the map has no such relation.
            BrokenIndexError: If the index does not lead to that relation's line.
            ContextMapArtifactError: If the stored record is not a valid relation.
        """
        return self._relation_of(self._relation_table().read(relation_id), relation_id)

    def relations(self) -> Iterator[ContextRelation]:
        """Stream every relation in id order, one line at a time."""
        for line in self._relation_table().iter_lines():
            yield self._relation_of(line, line["key"])

    def _relation_of(self, line: Mapping[str, Any], key: str) -> ContextRelation:
        try:
            relation = decode_relation(line)
        except (ContextMapRecordError, ValueError) as error:
            raise ContextMapArtifactError(
                f"the stored relation {key!r} is not a valid relation: {error}"
            ) from error
        if str(relation.relation_id) != key:
            raise BrokenIndexError(
                f"the line of {key!r} holds the relation {relation.relation_id!r}"
            )
        return relation

    def relations_for(self, reference: ContextEntityReference) -> tuple[ContextRelation, ...]:
        """List the relations in which an entity is the subject or the object.

        This is traversal through the derived index, not a query: it reads the relations the
        index names and interprets none of them.

        Args:
            reference: The entity, as ``(context_map_id, entity_id)``.

        Returns:
            The relations, ordered by id; empty when the entity takes part in none.

        Raises:
            ForeignContextEntityReferenceError: If the reference names another map.
            UnknownContextEntityError: If the map has no such entity.
            BrokenIndexError: If the index names a relation that does not exist.
        """
        key = self._own(reference)
        if key not in self._entity_table():
            raise UnknownContextEntityError(f"the map has no entity {key!r}")
        as_subject, as_object = self._traversal_index().get(key, ((), ()))
        relations = []
        for relation_id in sorted({*as_subject, *as_object}):
            try:
                relations.append(self.relation(relation_id))
            except RecordNotFoundError as error:
                raise BrokenIndexError(
                    f"{ENTITY_RELATION_INDEX} names the relation {relation_id!r}, which the map "
                    "does not have"
                ) from error
        return tuple(relations)

    def _traversal_index(self) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
        if self._traversal is None:
            index: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
            raw = (self._root / ENTITY_RELATION_INDEX).read_bytes()
            for number, text in enumerate(raw.split(b"\n")[:-1]):
                try:
                    line = json.loads(text)
                except ValueError as error:
                    raise BrokenIndexError(
                        f"line {number} of {ENTITY_RELATION_INDEX} is not valid JSON ({error})"
                    ) from error
                if not isinstance(line, dict) or line.keys() != _TRAVERSAL_FIELDS:
                    raise BrokenIndexError(
                        f"line {number} of {ENTITY_RELATION_INDEX} must have exactly the "
                        f"fields {sorted(_TRAVERSAL_FIELDS)}"
                    )
                index[line["key"]] = (tuple(line["as_subject"]), tuple(line["as_object"]))
            self._traversal = index
        return self._traversal

    # -- geometry and dependencies ----------------------------------------------------------

    def validate_reference(self, reference: GeometryReference) -> None:
        """Check that a geometry reference belongs to this map and lies inside its geometry.

        The check uses only the map's own declaration (its geometric map and point count), so it
        does not open the geometry.

        Args:
            reference: The reference to check.

        Raises:
            UnresolvedReferenceError: If the reference names another geometric map, is not the
                canonical identity of an element, or is outside the range of the map.
        """
        link = self.geometry_link()
        if reference.map_id != link.map_id:
            raise UnresolvedReferenceError(
                f"the reference is into the geometric map {reference.map_id!r} but this map "
                f"uses {link.map_id!r}"
            )
        try:
            index = geometry_index_of(map_id=link.map_id, geometry_id=reference.geometry_id)
        except ValueError as error:
            raise UnresolvedReferenceError(
                f"{reference.geometry_id!r} is not the canonical identity of a geometry element "
                f"of {link.map_id!r}"
            ) from error
        if index >= link.point_count:
            raise UnresolvedReferenceError(
                f"the geometry element {index} is outside the map, which has {link.point_count}"
            )

    def geometry_source(self) -> GeometrySource:
        """Open the referenced geometry as a :class:`GeometrySource`, on first use.

        The geometry is not read: it is memory-mapped through the geometric-map reader, and the
        directory found must be exactly the one the manifest recorded.

        Returns:
            The geometry, backed by the upstream artifact.

        Raises:
            MissingDependencyError: If the geometric map cannot be found.
            DependencyMismatchError: If what was found is not the recorded artifact.
            UpstreamArtifactError: If the geometric map cannot be opened.
        """
        self._ensure_open()
        reader = self._geometry_reader()
        try:
            return reader.geometry()
        except MapArtifactError as error:
            raise UpstreamArtifactError(f"the geometric map cannot be read: {error}") from error

    def geometry(self, reference: GeometryReference) -> GeometryPoint:
        """Resolve one geometry reference to its authoritative geometry.

        Args:
            reference: A reference into this map's geometry.

        Returns:
            The point with its coordinates and lineage, as the geometric-map artifact holds it.

        Raises:
            UnresolvedReferenceError: If the reference does not belong to this map.
            MissingDependencyError: If the geometric map cannot be found.
            DependencyMismatchError: If what was found is not the recorded artifact.
        """
        self.validate_reference(reference)
        return self.geometry_source().get(reference)

    def dependency_location(self, artifact_id: str) -> Path:
        """Locate one recorded upstream artifact and check that it is the recorded one.

        Args:
            artifact_id: The identity of the upstream artifact, as the lineage names it.

        Returns:
            The directory that holds it.

        Raises:
            RecordNotFoundError: If the manifest records no such dependency.
            MissingDependencyError: If it cannot be found.
            DependencyMismatchError: If what was found is not the recorded artifact.
        """
        self._ensure_open()
        for record in self._manifest.dependencies:
            if record.artifact_id == artifact_id:
                return self._require_found(
                    resolve_dependency(
                        record,
                        artifact_root=self._root,
                        dependency_paths=self._dependency_paths,
                    )
                )
        raise RecordNotFoundError(f"the manifest records no dependency {artifact_id!r}")

    @staticmethod
    def _require_found(resolution: DependencyResolution) -> Path:
        if resolution.status is DependencyStatus.MISSING:
            raise MissingDependencyError(resolution.detail)
        if resolution.status is DependencyStatus.MISMATCH:
            raise DependencyMismatchError(resolution.detail)
        assert resolution.location is not None
        return resolution.location

    def _geometry_reader(self) -> GeometricMapArtifactReader:
        if self._geometry is None:
            link = self.geometry_link()
            location = self.dependency_location(str(link.map_id))
            try:
                if self._verify_hashes:
                    verify_inventory(location, read_inventory(location))
                self._geometry = GeometricMapArtifactReader(location)
            except (UpstreamArtifactError, MapArtifactError) as error:
                raise UpstreamArtifactError(
                    f"the geometric map {link.map_id!r} cannot be opened: {error}"
                ) from error
        return self._geometry

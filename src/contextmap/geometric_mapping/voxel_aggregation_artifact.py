"""Persisted, immutable inter-scan voxel aggregations derived from a raw map artifact.

A ``VoxelAggregationArtifact`` is a directory, separate from the raw
``GeometricMapArtifact`` it summarizes, that holds one aggregate per occupied voxel,
the per-scan lineage back to the raw points, the policy that produced it and the exact
identity of the raw artifact it was derived from. Deriving it only reads the raw
artifact: the raw geometry stays the canonical evidence and is never rewritten.

It is a derived, experimental representation (#623/#624): no canonical stage consumes
it. Contractual data (``outputs/``, ``metrics/``, ``lineage.json``, ``config.json``,
``environment.json``) is inventoried with size and SHA-256 in the manifest, and writing
follows :class:`~contextmap.shared.AtomicRunDirectory`. See
``src/contextmap/geometric_mapping/docs/inter-scan-aggregation.md``.
"""

from __future__ import annotations

import platform
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NewType

from contextmap.geometric_mapping.bulk_geometry import DEFAULT_BLOCK_POINTS
from contextmap.geometric_mapping.models import GeometricMap, MapId
from contextmap.geometric_mapping.run_artifact import (
    GeometricMapArtifactManifest,
    GeometricMapArtifactReader,
    IncompleteMapArtifactError,
    MapArtifactError,
    _installed_version,
    _json,
    _read_json,
)
from contextmap.geometric_mapping.serialization import decode_geometric_map, encode_geometric_map
from contextmap.geometric_mapping.voxel_aggregation import (
    INTER_SCAN_VOXEL_POLICY_ID,
    INTER_SCAN_VOXEL_POLICY_VERSION,
    VOXEL_INDEXING,
    AggregatedGeometry,
    InterScanVoxelPolicy,
    VoxelAggregation,
    VoxelAggregationError,
    VoxelGridSpec,
    aggregate_geometry,
)
from contextmap.ingestion import FrameId
from contextmap.shared import AtomicRunDirectory, FileEntry, RunDirectoryError, check_file_inventory

SCHEMA_VERSION = "0.1.0"
"""Voxel aggregation artifact schema version written and understood by this module."""

VoxelAggregationRunId = NewType("VoxelAggregationRunId", str)
"""Identity of one derivation, supplied by the caller."""

AGGREGATE_RECORD = struct.Struct("<3q3d3d3d3dqqIII")
"""One aggregate: key, centroid, offset sum, minimum, maximum, first/last ns, three counts."""

AGGREGATE_DTYPE_FIELDS = [
    ("key", "<i8", (3,)),
    ("centroid", "<f8", (3,)),
    ("offset_sum", "<f8", (3,)),
    ("minimum", "<f8", (3,)),
    ("maximum", "<f8", (3,)),
    ("first_observed_ns", "<i8"),
    ("last_observed_ns", "<i8"),
    ("point_count", "<u4"),
    ("scan_count", "<u4"),
    ("observation_count", "<u4"),
]
"""NumPy spelling of :data:`AGGREGATE_RECORD`."""

CONTRIBUTION_RECORD = struct.Struct("<II")
"""One contribution: scan ordinal in the raw map, raw points of that scan in the voxel."""

CONTRIBUTION_DTYPE_FIELDS = [("scan_ordinal", "<u4"), ("point_count", "<u4")]
"""NumPy spelling of :data:`CONTRIBUTION_RECORD`."""

_MANIFEST = "manifest.json"
_LINEAGE = "lineage.json"
_CONFIG = "config.json"
_ENVIRONMENT = "environment.json"
_AGGREGATES = "outputs/aggregates.bin"
_CONTRIBUTIONS = "outputs/contributions.bin"
_MAP_METADATA = "outputs/map-metadata.json"
_AGGREGATION = "outputs/aggregation.json"
_METRICS = "metrics/aggregation.json"
_RAW_GEOMETRY = "outputs/geometry.bin"

_CONTRACTUAL_RECORDS = frozenset({_LINEAGE, _CONFIG, _ENVIRONMENT})
_CONTRACTUAL_DIRECTORIES = ("outputs/", "metrics/")
_MAX_U32 = 2**32 - 1


@dataclass(frozen=True, kw_only=True)
class VoxelAggregationManifest:
    """Authoritative metadata of a persisted voxel aggregation.

    Attributes:
        run_id: Identity of the derivation.
        derived_map_id: Identity of the derived map its references name.
        source_map_id: The raw map summarized.
        source_run_id: The raw Geometric Mapping run summarized.
        policy_fingerprint: Hash of the aggregation policy.
        aggregation_rule: The rule the derived map declares.
        grid: Frame, origin, resolution and indexing of the grid.
        aggregate_count: Occupied voxels.
        source_point_count: Raw points summarized; every one in exactly one aggregate.
        contribution_count: ``(aggregate, scan)`` lineage entries.
        scan_count: Raw scans that contributed.
        observation_count: Physical source observations that contributed.
        clock_id: Clock domain of the observation times.
        code_version: Code revision that derived the aggregation, when known.
        schema_version: Artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash.
    """

    run_id: VoxelAggregationRunId
    derived_map_id: MapId
    source_map_id: MapId
    source_run_id: str
    policy_fingerprint: str
    aggregation_rule: str
    grid: dict[str, Any]
    aggregate_count: int
    source_point_count: int
    contribution_count: int
    scan_count: int
    observation_count: int
    clock_id: str
    code_version: str | None
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


class VoxelAggregationArtifactWriter:
    """Derives and persists an immutable voxel aggregation of a raw map artifact."""

    def __init__(self, *, output_dir: Path, run_id: VoxelAggregationRunId) -> None:
        """Create a writer for a new derivation.

        Args:
            output_dir: The final directory; created atomically, never replaced, and never
                inside the raw artifact.
            run_id: Identity of the derivation, supplied by the caller.
        """
        self._final_dir = output_dir
        self._run_id = run_id

    def finalize(
        self,
        *,
        source_dir: Path,
        policy: InterScanVoxelPolicy,
        code_version: str | None,
        block_points: int = DEFAULT_BLOCK_POINTS,
    ) -> VoxelAggregationManifest:
        """Aggregate a raw map artifact and persist the result atomically.

        Args:
            source_dir: The raw ``GeometricMapArtifact``; only read.
            policy: The aggregation policy.
            code_version: Code revision that derives the aggregation, when known.
            block_points: Rows read per block. Recorded as an execution parameter: it
                changes the result only within the documented tolerance, so it is not
                part of the policy identity.

        Returns:
            The manifest of the finalized artifact.

        Raises:
            MapArtifactError: If the target is inside the raw artifact, already exists,
                or the raw map cannot be aggregated.
        """
        if self._final_dir.resolve().is_relative_to(source_dir.resolve()):
            raise MapArtifactError(
                f"{self._final_dir} is inside the raw artifact {source_dir}; a derived "
                "representation never writes into the evidence it summarizes"
            )
        if self._final_dir.exists():
            raise MapArtifactError(f"run directory already exists: {self._final_dir}")
        with GeometricMapArtifactReader(source_dir) as source:
            geometry = source.geometry()
            try:
                aggregation = aggregate_geometry(geometry, policy, block_points=block_points)
                derived = AggregatedGeometry.derive(
                    aggregation=aggregation,
                    source_map=geometry.geometric_map,
                    map_id=MapId(f"{geometry.geometric_map.map_id}--{self._run_id}"),
                    code_version=code_version,
                )
            except VoxelAggregationError as error:
                raise MapArtifactError(str(error)) from error
            observation_count = len(
                {
                    geometry.scans[int(ordinal)].observation_id
                    for ordinal in set(aggregation.contribution_scan_ordinals.tolist())
                }
            )
            try:
                with AtomicRunDirectory(self._final_dir) as run:
                    run.write_bytes(_AGGREGATES, _pack_aggregates(aggregation))
                    run.write_bytes(_CONTRIBUTIONS, _pack_contributions(aggregation))
                    run.write_text(
                        _MAP_METADATA, _json(encode_geometric_map(derived.geometric_map))
                    )
                    run.write_text(
                        _AGGREGATION,
                        _json(
                            {
                                "source_map_id": str(aggregation.source_map_id),
                                "source_point_count": aggregation.source_point_count,
                                "clock_id": aggregation.clock_id,
                                "aggregate_record": AGGREGATE_RECORD.format,
                                "contribution_record": CONTRIBUTION_RECORD.format,
                            }
                        ),
                    )
                    run.write_text(_LINEAGE, _json(_lineage(source.manifest, code_version)))
                    run.write_text(
                        _CONFIG,
                        _json(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "policy": policy.to_record(),
                                "policy_fingerprint": policy.fingerprint(),
                                "execution": {"block_points": block_points},
                            }
                        ),
                    )
                    run.write_text(_ENVIRONMENT, _json(_environment(code_version)))
                    run.write_text(_METRICS, _json(_metrics(aggregation)))
                    run.publish(
                        manifest={
                            "run_id": str(self._run_id),
                            "derived_map_id": str(derived.geometric_map.map_id),
                            "source_map_id": str(aggregation.source_map_id),
                            "source_run_id": str(source.manifest.run_id),
                            "policy_fingerprint": policy.fingerprint(),
                            "aggregation_rule": policy.rule,
                            "grid": policy.grid.to_record(),
                            "aggregate_count": len(aggregation),
                            "source_point_count": aggregation.source_point_count,
                            "contribution_count": int(aggregation.scan_counts.sum()),
                            "scan_count": len(set(aggregation.contribution_scan_ordinals.tolist())),
                            "observation_count": observation_count,
                            "clock_id": aggregation.clock_id,
                            "code_version": code_version,
                            "schema_version": SCHEMA_VERSION,
                            "created_at": datetime.now(UTC).isoformat(),
                        },
                        readme=_readme(self._run_id, derived.geometric_map, aggregation),
                    )
            except RunDirectoryError as error:
                raise MapArtifactError(str(error)) from error
        return _load_manifest(self._final_dir)


class VoxelAggregationArtifactReader:
    """Reads a finalized voxel aggregation from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a derivation.

        Raises:
            IncompleteMapArtifactError: If ``manifest.json`` is missing.
            MapArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = _load_manifest(run_dir)
        self._aggregation: VoxelAggregation | None = None

    @property
    def manifest(self) -> VoxelAggregationManifest:
        """The derivation's manifest."""
        return self._manifest

    def policy(self) -> InterScanVoxelPolicy:
        """Decode the policy that produced the aggregation.

        Raises:
            MapArtifactError: If the policy identity, version, representative or grid
                indexing is not the one this module implements.
        """
        return _decode_policy(self.read_record(_CONFIG)["policy"])

    def aggregation(self) -> VoxelAggregation:
        """Load the aggregates and their lineage into memory, validated.

        Raises:
            MapArtifactError: If a payload is missing, has a partial record, or the
                decoded aggregation breaks one of its invariants.
        """
        if self._aggregation is not None:
            return self._aggregation
        import numpy as np

        policy = self.policy()
        header = self.read_record(_AGGREGATION)
        aggregates = self._payload(_AGGREGATES, AGGREGATE_DTYPE_FIELDS)
        contributions = self._payload(_CONTRIBUTIONS, CONTRIBUTION_DTYPE_FIELDS)
        try:
            self._aggregation = VoxelAggregation(
                policy=policy,
                source_map_id=MapId(header["source_map_id"]),
                source_point_count=int(header["source_point_count"]),
                clock_id=header["clock_id"],
                keys=np.array(aggregates["key"], dtype=np.int64),
                centroids_m=np.array(aggregates["centroid"], dtype=np.float64),
                offset_sums_m=np.array(aggregates["offset_sum"], dtype=np.float64),
                minimum_m=np.array(aggregates["minimum"], dtype=np.float64),
                maximum_m=np.array(aggregates["maximum"], dtype=np.float64),
                point_counts=np.array(aggregates["point_count"], dtype=np.int64),
                scan_counts=np.array(aggregates["scan_count"], dtype=np.int64),
                observation_counts=np.array(aggregates["observation_count"], dtype=np.int64),
                first_observed_ns=np.array(aggregates["first_observed_ns"], dtype=np.int64),
                last_observed_ns=np.array(aggregates["last_observed_ns"], dtype=np.int64),
                contribution_scan_ordinals=np.array(contributions["scan_ordinal"], dtype=np.int64),
                contribution_point_counts=np.array(contributions["point_count"], dtype=np.int64),
            )
        except VoxelAggregationError as error:
            raise MapArtifactError(f"the persisted aggregation is inconsistent: {error}") from error
        return self._aggregation

    def geometry(self) -> AggregatedGeometry:
        """Open the aggregates as the block source of the derived map.

        Raises:
            MapArtifactError: If the aggregation or the derived map cannot be read.
        """
        geometric_map = decode_geometric_map(self.read_record(_MAP_METADATA))
        try:
            return AggregatedGeometry(aggregation=self.aggregation(), geometric_map=geometric_map)
        except VoxelAggregationError as error:
            raise MapArtifactError(str(error)) from error

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a contractual JSON record.

        Raises:
            MapArtifactError: If the path is not a contractual JSON record.
        """
        contractual = relative_path in _CONTRACTUAL_RECORDS or (
            relative_path.endswith(".json") and relative_path.startswith(_CONTRACTUAL_DIRECTORIES)
        )
        if not contractual or ".." in relative_path:
            raise MapArtifactError(f"not a contractual JSON record: {relative_path!r}")
        path = self._root / relative_path
        if not path.is_file():
            raise MapArtifactError(f"missing {relative_path} in {self._root}")
        return _read_json(path)

    def verify_integrity(self, *, source: GeometricMapArtifactReader | None = None) -> list[str]:
        """Check the inventory, the aggregation's invariants and, optionally, its lineage.

        Args:
            source: The raw artifact the derivation claims to summarize. When given, its
                identity and geometry hash must be the recorded ones, and the per-scan
                lineage is re-derived from its geometry; this reads the whole raw map.

        Returns:
            Human-readable problems; empty means the derivation is intact.
        """
        problems = check_file_inventory(self._root, self._manifest.file_inventory)
        if problems:
            return problems
        try:
            aggregation = self.aggregation()
            geometry = self.geometry()
        except MapArtifactError as error:
            return [str(error)]
        if geometry.geometric_map.map_id != self._manifest.derived_map_id:
            problems.append("the derived map metadata names another map than the manifest")
        if len(aggregation) != self._manifest.aggregate_count:
            problems.append("the manifest counts another number of aggregates than the payload")
        if source is None:
            return problems
        recorded = self.read_record(_LINEAGE)["source"]
        actual = _source_identity(source.manifest)
        mismatched = sorted(key for key in recorded if recorded[key] != actual.get(key))
        if mismatched:
            problems.append(f"the raw artifact differs from the recorded source in {mismatched}")
            return problems
        problems.extend(aggregation.verify_lineage(source.geometry()))
        return problems

    def _payload(self, relative_path: str, fields: list[Any]) -> Any:
        import numpy as np

        path = self._root / relative_path
        if not path.is_file():
            raise MapArtifactError(f"missing {relative_path} in {self._root}")
        dtype = np.dtype(fields)
        data = path.read_bytes()
        if len(data) % dtype.itemsize:
            raise MapArtifactError(
                f"{relative_path} has {len(data)} bytes, not a whole number of "
                f"{dtype.itemsize}-byte records"
            )
        return np.frombuffer(data, dtype=dtype)


def _pack_aggregates(aggregation: VoxelAggregation) -> bytes:
    import numpy as np

    for name in ("point_counts", "scan_counts", "observation_counts"):
        if len(aggregation) and int(getattr(aggregation, name).max()) > _MAX_U32:
            raise MapArtifactError(f"{name} exceed the 32-bit record field")
    records = np.empty(len(aggregation), dtype=np.dtype(AGGREGATE_DTYPE_FIELDS))
    records["key"] = aggregation.keys
    records["centroid"] = aggregation.centroids_m
    records["offset_sum"] = aggregation.offset_sums_m
    records["minimum"] = aggregation.minimum_m
    records["maximum"] = aggregation.maximum_m
    records["first_observed_ns"] = aggregation.first_observed_ns
    records["last_observed_ns"] = aggregation.last_observed_ns
    records["point_count"] = aggregation.point_counts
    records["scan_count"] = aggregation.scan_counts
    records["observation_count"] = aggregation.observation_counts
    payload = records.tobytes()
    if len(payload) != len(aggregation) * AGGREGATE_RECORD.size:
        raise MapArtifactError("the aggregate record layout disagrees with the documented one")
    return payload


def _pack_contributions(aggregation: VoxelAggregation) -> bytes:
    import numpy as np

    records = np.empty(
        len(aggregation.contribution_scan_ordinals), dtype=np.dtype(CONTRIBUTION_DTYPE_FIELDS)
    )
    records["scan_ordinal"] = aggregation.contribution_scan_ordinals
    records["point_count"] = aggregation.contribution_point_counts
    return records.tobytes()


def _source_identity(manifest: GeometricMapArtifactManifest) -> dict[str, Any]:
    """The identity of the raw artifact, down to the hash of its geometry payload."""
    geometry_hash = next(
        (entry.content_hash for entry in manifest.file_inventory if entry.path == _RAW_GEOMETRY),
        None,
    )
    return {
        "run_id": str(manifest.run_id),
        "map_id": str(manifest.map_id),
        "schema_version": manifest.schema_version,
        "configuration_fingerprint": manifest.configuration_fingerprint,
        "geometry_content_hash": geometry_hash,
        "aggregation_rule": manifest.aggregation_rule,
        "sequence_artifact_id": str(manifest.sequence_artifact_id),
        "selection_id": manifest.selection_id,
        "trajectory_id": str(manifest.trajectory_id),
        "calibration_identity": manifest.calibration_identity,
    }


def _lineage(source: GeometricMapArtifactManifest, code_version: str | None) -> dict[str, Any]:
    return {"source": _source_identity(source), "code_version": code_version}


def _environment(code_version: str | None) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _installed_version("numpy"),
        "code_version": code_version,
    }


def _metrics(aggregation: VoxelAggregation) -> dict[str, Any]:
    """Scale and provenance cost of the derivation, apart from any quality measure."""
    contributions = int(aggregation.scan_counts.sum())
    count = len(aggregation)
    return {
        "source_point_count": aggregation.source_point_count,
        "aggregate_count": count,
        "reduction_ratio": count / aggregation.source_point_count,
        "contribution_count": contributions,
        "aggregate_bytes": count * AGGREGATE_RECORD.size,
        "lineage_bytes": contributions * CONTRIBUTION_RECORD.size,
        "lineage_bytes_per_aggregate": contributions * CONTRIBUTION_RECORD.size / count,
        # O custo de listar explicitamente o índice int64 de cada ponto bruto, a alternativa
        # que a lineage por scan evita: comparável, nunca escrito.
        "explicit_point_lineage_bytes": aggregation.source_point_count * 8,
        "points_per_aggregate": _order_statistics(aggregation.point_counts),
        "scans_per_aggregate": _order_statistics(aggregation.scan_counts),
        "observations_per_aggregate": _order_statistics(aggregation.observation_counts),
    }


def _order_statistics(values: Any) -> dict[str, float]:
    """Minimum, median, 95th percentile by nearest rank and maximum of integer counts."""
    import math

    import numpy as np

    ordered = np.sort(np.asarray(values))
    rank = math.ceil(0.95 * len(ordered))
    return {
        "minimum": int(ordered[0]),
        "median": float(np.median(ordered)),
        "p95": int(ordered[rank - 1]),
        "maximum": int(ordered[-1]),
    }


def _decode_policy(record: dict[str, Any]) -> InterScanVoxelPolicy:
    grid = record["grid"]
    if grid.get("indexing") != VOXEL_INDEXING:
        raise MapArtifactError(
            f"unsupported grid indexing {grid.get('indexing')!r}; this reader implements "
            f"{VOXEL_INDEXING!r}"
        )
    if (record.get("policy_id"), record.get("policy_version")) != (
        INTER_SCAN_VOXEL_POLICY_ID,
        INTER_SCAN_VOXEL_POLICY_VERSION,
    ):
        raise MapArtifactError(
            f"unsupported policy {record.get('policy_id')!r} {record.get('policy_version')!r}"
        )
    x, y, z = grid["origin_m"]
    policy = InterScanVoxelPolicy(
        grid=VoxelGridSpec(
            frame_id=FrameId(grid["frame_id"]),
            origin_m=(float(x), float(y), float(z)),
            cell_m=float(grid["cell_m"]),
        )
    )
    if policy.to_record() != record:
        raise MapArtifactError("the persisted policy is not the one this version computes")
    return policy


def _load_manifest(run_dir: Path) -> VoxelAggregationManifest:
    path = run_dir / _MANIFEST
    if not path.is_file():
        raise IncompleteMapArtifactError(f"missing {_MANIFEST} in {run_dir}")
    raw = _read_json(path)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise MapArtifactError(
            f"unsupported voxel aggregation schema_version: {raw.get('schema_version')!r}"
        )
    return VoxelAggregationManifest(
        run_id=VoxelAggregationRunId(raw["run_id"]),
        derived_map_id=MapId(raw["derived_map_id"]),
        source_map_id=MapId(raw["source_map_id"]),
        source_run_id=raw["source_run_id"],
        policy_fingerprint=raw["policy_fingerprint"],
        aggregation_rule=raw["aggregation_rule"],
        grid=raw["grid"],
        aggregate_count=raw["aggregate_count"],
        source_point_count=raw["source_point_count"],
        contribution_count=raw["contribution_count"],
        scan_count=raw["scan_count"],
        observation_count=raw["observation_count"],
        clock_id=raw["clock_id"],
        code_version=raw["code_version"],
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


def _readme(
    run_id: VoxelAggregationRunId, derived: GeometricMap, aggregation: VoxelAggregation
) -> str:
    return (
        f"# Inter-scan voxel aggregation `{run_id}`\n"
        "\n"
        f"- Derived map: `{derived.map_id}` in frame `{derived.frame_id}`\n"
        f"- Raw map summarized: `{aggregation.source_map_id}`\n"
        f"- Rule: `{aggregation.policy.rule}`\n"
        f"- Aggregates: {len(aggregation)} from {aggregation.source_point_count} raw points\n"
        "\n"
        "A derived representation: the raw map stays the canonical geometry. Contractual data "
        "is in `outputs/`, `metrics/`, `lineage.json`, `config.json` and `environment.json`.\n"
    )

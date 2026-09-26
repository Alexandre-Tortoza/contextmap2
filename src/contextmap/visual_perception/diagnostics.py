"""Atomic contractual output and optional Region Discovery diagnostics writer."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from html import escape
from pathlib import Path
from uuid import uuid4

from .discovery import BackendDiagnostics, DiscoveryRunResult
from .models import BoundingBox2D, PreparedImage, Region2D
from .normalization import NormalizationResult
from .region_models import BoundingBox, InlineMask, JsonScalar, RegionCandidate


class DebugLevel(StrEnum):
    """Amount of optional human diagnostics persisted for one frame."""

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class DiscoveryAuditRecord:
    """All canonical evidence required to write one discovery-stage artifact."""

    prepared_image: PreparedImage
    backend_id: str
    backend_config: tuple[tuple[str, JsonScalar], ...]
    discovery: DiscoveryRunResult
    normalization: NormalizationResult
    normalization_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate backend identity, config, and timing."""
        if not self.backend_id:
            raise ValueError("backend_id must not be empty")
        keys = [key for key, _ in self.backend_config]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("backend config names must be non-empty and unique")
        if self.normalization_duration_ms < 0:
            raise ValueError("normalization_duration_ms must be non-negative")
        if len(self.discovery.passes) != len(self.discovery.diagnostics):
            raise ValueError("each discovery pass must have one diagnostic record")


@dataclass(frozen=True, slots=True)
class WrittenDiscoveryEvidence:
    """Paths of the finalized immutable discovery-stage artifact."""

    stage_directory: Path
    manifest_path: Path
    metrics_path: Path


class RegionDiscoveryEvidenceWriter:
    """Persist contractual outputs and optional diagnostics with atomic finalize."""

    def write(
        self,
        stage_directory: Path,
        record: DiscoveryAuditRecord,
        debug_level: DebugLevel,
    ) -> WrittenDiscoveryEvidence:
        """Write and atomically finalize one immutable frame-stage directory.

        Args:
            stage_directory: Final ``20-region-discovery`` directory.
            record: Canonical discovery and normalization evidence.
            debug_level: Optional diagnostic detail to persist.

        Returns:
            Paths inside the finalized stage directory.

        Raises:
            FileExistsError: If the requested artifact already exists.
        """
        if stage_directory.exists():
            raise FileExistsError(f"discovery stage already finalized: {stage_directory}")
        stage_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = stage_directory.parent / f".{stage_directory.name}.tmp-{uuid4().hex}"
        temporary.mkdir()
        try:
            self._write_contents(temporary, record, debug_level)
            os.replace(temporary, stage_directory)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return WrittenDiscoveryEvidence(
            stage_directory=stage_directory,
            manifest_path=stage_directory / "manifest.json",
            metrics_path=stage_directory / "outputs" / "metrics.json",
        )

    def _write_contents(
        self, root: Path, record: DiscoveryAuditRecord, debug_level: DebugLevel
    ) -> None:
        outputs = root / "outputs"
        outputs.mkdir()
        _write_jsonl(
            outputs / "regions.jsonl",
            [region.to_dict() for region in record.normalization.regions],
        )
        _write_json(outputs / "metrics.json", _metrics(record))

        if debug_level is not DebugLevel.NONE:
            self._write_debug(root / "debug", record, debug_level)

        files = [
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]
        _write_json(
            root / "manifest.json",
            {
                "schema": "contextmap.region-discovery-stage/v1",
                "source_observation_id": record.prepared_image.source_observation_id,
                "backend_id": record.backend_id,
                "normalization_config_digest": record.normalization.config_digest,
                "debug_level": debug_level.value,
                "files": files,
            },
        )

    def _write_debug(
        self, debug: Path, record: DiscoveryAuditRecord, debug_level: DebugLevel
    ) -> None:
        debug.mkdir()
        _write_json(debug / "prepared-image.json", record.prepared_image.to_dict())
        _write_json(debug / "backend-config.json", dict(record.backend_config))
        _write_json(
            debug / "passes.json",
            [
                {
                    **discovery_pass.to_dict(),
                    "diagnostics": diagnostics.to_dict(),
                }
                for discovery_pass, diagnostics in zip(
                    record.discovery.passes, record.discovery.diagnostics, strict=True
                )
            ],
        )
        _write_jsonl(
            debug / "candidates.jsonl",
            [candidate.to_dict() for candidate in record.discovery.candidates],
        )
        _write_jsonl(
            debug / "accepted-regions.jsonl",
            [region.to_dict() for region in record.normalization.regions],
        )
        rejections = (*record.discovery.rejected, *record.normalization.rejected)
        _write_jsonl(
            debug / "rejected-regions.jsonl",
            [rejection.to_dict() for rejection in rejections],
        )
        _write_jsonl(
            debug / "merge-decisions.jsonl",
            [decision.to_dict() for decision in record.normalization.merge_decisions],
        )
        (debug / "candidates-overlay.svg").write_text(
            _candidate_overlay(
                record.prepared_image, record.discovery.candidates, "Raw candidates", "#ffb000"
            ),
            encoding="utf-8",
        )
        (debug / "accepted-regions-overlay.svg").write_text(
            _region_overlay(
                record.prepared_image,
                record.normalization.regions,
                "Accepted regions",
                "#00a878",
            ),
            encoding="utf-8",
        )
        (debug / "rejected-regions-overlay.svg").write_text(
            _rejection_overlay(record, "Rejected candidates", "#e63946"),
            encoding="utf-8",
        )

        if debug_level is DebugLevel.FULL:
            for region in record.normalization.regions:
                region_directory = debug / "regions" / str(region.region_id)
                region_directory.mkdir(parents=True)
                _write_json(region_directory / "region.json", region.to_dict())
                if isinstance(region.mask, InlineMask):
                    (region_directory / "mask.pbm").write_text(
                        _portable_bitmap(region.mask), encoding="ascii"
                    )


def _metrics(record: DiscoveryAuditRecord) -> dict[str, object]:
    raw_count = sum(
        _raw_proposal_count(diagnostics) for diagnostics in record.discovery.diagnostics
    )
    rejections = (*record.discovery.rejected, *record.normalization.rejected)
    areas = [
        region.area_pixels
        for region in record.normalization.regions
        if region.area_pixels is not None
    ]
    durations = [diagnostics.duration_ms for diagnostics in record.discovery.diagnostics]
    warnings = [
        warning for diagnostics in record.discovery.diagnostics for warning in diagnostics.warnings
    ]
    peak_memory = [
        value
        for diagnostics in record.discovery.diagnostics
        for key, value in diagnostics.metadata
        if key == "peak_memory_mb" and isinstance(value, (int, float))
    ]
    return {
        "schema": "contextmap.region-discovery-metrics/v1",
        "backend_id": record.backend_id,
        "backend_config": dict(record.backend_config),
        "raw_candidate_count": raw_count,
        "remapped_candidate_count": len(record.discovery.candidates),
        "accepted_region_count": len(record.normalization.regions),
        "rejected_candidate_count": len(rejections),
        "merge_count": len(record.normalization.merge_decisions),
        "duplicate_merge_ratio": (
            len(record.normalization.merge_decisions) / raw_count if raw_count else 0.0
        ),
        "invalid_geometry_count": sum(
            rejection.reason.value == "invalid_geometry" for rejection in rejections
        ),
        "constraint_violation_count": sum(
            rejection.reason.value in {"outside_valid_region", "exclusion_overlap"}
            for rejection in rejections
        ),
        "area_pixels": {
            "values": areas,
            "minimum": min(areas) if areas else None,
            "maximum": max(areas) if areas else None,
            "mean": sum(areas) / len(areas) if areas else None,
        },
        "per_pass_duration_ms": durations,
        "total_backend_duration_ms": sum(durations),
        "normalization_duration_ms": record.normalization_duration_ms,
        "peak_memory_mb": max(peak_memory) if peak_memory else None,
        "warnings": warnings,
        "normalization_config_digest": record.normalization.config_digest,
    }


def _raw_proposal_count(diagnostics: BackendDiagnostics) -> int:
    for key, value in diagnostics.metadata:
        if key == "raw_proposal_count" and type(value) is int:
            return int(value)
    return diagnostics.proposal_count


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    lines = [json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values]
    path.write_text("" if not lines else "\n".join(lines) + "\n", encoding="utf-8")


def _svg_start(image: PreparedImage, title: str) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {image.width} '
            f'{image.height}" role="img" aria-label="{escape(title)}">'
        ),
        f'<rect width="{image.width}" height="{image.height}" fill="#1f2937"/>',
    ]


def _candidate_overlay(
    image: PreparedImage,
    candidates: tuple[RegionCandidate, ...],
    title: str,
    color: str,
) -> str:
    lines = _svg_start(image, title)
    for candidate in candidates:
        if candidate.bounding_box is not None:
            lines.extend(_svg_box(candidate.bounding_box, candidate.candidate_id, color))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _region_overlay(
    image: PreparedImage,
    regions: tuple[Region2D, ...],
    title: str,
    color: str,
) -> str:
    lines = _svg_start(image, title)
    for region in regions:
        lines.extend(_svg_box(region.bounding_box, str(region.region_id), color))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _rejection_overlay(record: DiscoveryAuditRecord, title: str, color: str) -> str:
    lines = _svg_start(record.prepared_image, title)
    by_id = {candidate.candidate_id: candidate for candidate in record.discovery.candidates}
    rejections = (*record.discovery.rejected, *record.normalization.rejected)
    listed: list[str] = []
    for rejection in rejections:
        candidate = by_id.get(rejection.candidate_id)
        label = f"{rejection.candidate_id}: {rejection.reason.value}"
        if candidate is not None and candidate.bounding_box is not None:
            lines.extend(_svg_box(candidate.bounding_box, label, color))
        else:
            listed.append(label)
    for index, label in enumerate(listed):
        lines.append(
            f'<text x="0" y="{index + 1}" font-size="1" fill="{color}">{escape(label)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _svg_box(box: BoundingBox | BoundingBox2D, label: str, color: str) -> list[str]:
    return [
        (
            f'<rect x="{box.x_min}" y="{box.y_min}" width="{box.width}" '
            f'height="{box.height}" fill="none" stroke="{color}" stroke-width="0.3"/>'
        ),
        (
            f'<text x="{box.x_min}" y="{max(0.8, box.y_min + 0.8)}" font-size="0.8" '
            f'fill="{color}">{escape(label)}</text>'
        ),
    ]


def _portable_bitmap(mask: InlineMask) -> str:
    """Render a mask as plain PBM: one row of space-separated ``0``/``1`` per image row."""
    import numpy as np

    # Cada linha tem 2 x width caracteres: dígitos nas posições pares, espaços nas ímpares e a
    # quebra de linha no lugar do último espaço.
    text = np.full((mask.height, 2 * mask.width), ord(" "), dtype=np.uint8)
    text[:, 0::2] = mask.as_array() + ord("0")
    text[:, -1] = ord("\n")
    return f"P1\n{mask.width} {mask.height}\n" + text.tobytes().decode("ascii")

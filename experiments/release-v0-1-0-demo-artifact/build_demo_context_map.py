"""Generate the committed synthetic demo ContextMapArtifact of the v0.1.0 release (issue #188).

The demo artifact has to be small, redistributable and **real in format**: written by the
capability's own writer, validating under ``ValidationLevel.FULL``, and exercising entities,
geometry references, semantic alternatives and relations rather than being an empty smoke
artifact. Its *content* is synthetic: no recorded sensor data, no model runtime, no dataset.

This script reuses the serialization test builders instead of restating them. They already write
the upstream world with each capability's real writer -- a GeometricMapArtifact, an
EntityResolutionRunArtifact and a SpatialRelationsRunArtifact over a tiny scene -- and assemble
the map from those two runs' real identities and geometry references. Duplicating ~775 lines of
that here to avoid importing from ``tests/`` would be worse: the example would drift from the
format it claims to demonstrate.

The demo ships **with** its upstream world, because a ContextMapArtifact alone does not
validate: three of its dependencies are required, and without them ``validate`` reports
``dependency.required_missing`` and the artifact is invalid. That is correct behavior, not a
defect -- the map references geometry instead of copying it -- so the bundle carries the
geometric map, the entity-resolution run and the spatial-relations run next to it. The whole
bundle is about 170 KB. The layout is the one the artifact was written against, so the relative
hints recorded in its manifest resolve with no extra flags.

Usage:
    python build_demo_context_map.py <repository-root>/examples/v0.1.0/demo
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tests" / "artifact"))

from context_map_serialization_builders import (  # noqa: E402
    make_context_map,
    make_world,
)

from contextmap.artifact import (  # noqa: E402
    ContextMapArtifactReader,
    ContextMapArtifactWriter,
    Severity,
    ValidationLevel,
    ValidationStatus,
    validate_context_map_artifact,
)

# Determinístico: o manifesto grava este instante como provenance, então o artifact commitado
# é byte a byte o mesmo a cada geração.
WRITTEN_AT = datetime.fromisoformat("2026-09-25T00:00:00+00:00")
# A cena é a que os builders declaram no metadata do mapa (POINT_COUNT = 10 x 100): o writer
# recusa um mapa cujo metadata não corresponda ao GeometricMapArtifact citado, e essa checagem
# é exatamente o que se quer demonstrar.
SCANS = 10
POINTS_PER_SCAN = 100


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    destination = Path(sys.argv[1])
    workspace = Path(tempfile.mkdtemp()) / "demo"
    workspace.mkdir(parents=True)

    world = make_world(workspace, geometry_scans=SCANS, points_per_scan=POINTS_PER_SCAN)
    context_map = make_context_map(world)
    # O writer público, não o atalho dos testes: o caminho de saída é escolhido aqui para o
    # pacote publicado ter um layout legível, e as dicas relativas do manifesto seguem dele.
    artifact_dir = workspace / "context-map"
    cited = {item.artifact_id for item in context_map.lineage}
    manifest = ContextMapArtifactWriter(output_dir=artifact_dir, written_at=WRITTEN_AT).write(
        context_map,
        upstream_locations={
            artifact_id: location
            for artifact_id, location in world.locations.items()
            if artifact_id in cited
        },
    )

    # O artifact é validado com o mundo ainda no lugar, para provar que o formato está correto
    # antes de ele ser separado da própria linhagem.
    located = validate_context_map_artifact(
        artifact_dir, level=ValidationLevel.FULL, dependency_paths=world.locations
    )
    errors = [finding for finding in located.findings if finding.severity is Severity.ERROR]
    if located.status is ValidationStatus.INVALID or errors:
        raise SystemExit(f"the demo artifact did not validate with its world: {errors}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # O pacote inteiro é copiado com a mesma disposição relativa em que foi escrito.
    shutil.copytree(workspace, destination)
    shutil.rmtree(workspace.parent)
    published = destination / artifact_dir.relative_to(workspace)

    # E revalidado como será distribuído: sem passar caminho nenhum, só pelas dicas do manifesto.
    alone = validate_context_map_artifact(published, level=ValidationLevel.FULL)
    reader = ContextMapArtifactReader.open(published, verify_hashes=True)
    entities = list(reader.entities())
    summary = {
        "context_map_id": str(manifest.context_map_id),
        "schema_version": manifest.schema_version,
        "entity_count": len(entities),
        "relation_count": sum(1 for _ in reader.relations()),
        # "Alternativas semânticas" do #188: uma entidade com mais de uma hipótese, ou marcada
        # como ambígua. O mapa preserva a ambiguidade em vez de escolher um rótulo.
        "entities_with_multiple_hypotheses": sum(
            1 for entity in entities if len(entity.semantic_state.hypotheses) > 1
        ),
        "ambiguity_statuses": sorted(
            {entity.semantic_state.status.value for entity in entities}
        ),
        "entities_with_geometry_reference": sum(
            1 for entity in entities if entity.geometry_refs
        ),
        "status_with_world": located.status.value,
        "artifact_path": str(published.relative_to(destination.parent)),
        "status_from_recorded_hints": alone.status.value,
        "findings_from_recorded_hints": [f"{f.severity.value}:{f.code}" for f in alone.findings],
        "bytes": sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()),
        "largest_file_bytes": max(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        ),
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

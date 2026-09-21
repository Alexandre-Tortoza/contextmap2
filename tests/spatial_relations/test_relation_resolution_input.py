"""Reading an Entity Resolution run: lineage, resolved geometry and reference validation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from relation_resolution_fixture import ER_RUN, er_scene_source, write_er_run
from relation_scene import CONNECTED_POLICY

from contextmap.entity_resolution import (
    EntityResolutionRunReader,
    ResolvedEntityId,
    ResolvedEntityReference,
    ResolvedEntitySet,
    resolution_artifact_digest,
)
from contextmap.geometric_mapping import MapId
from contextmap.semantic_mapping import EntityGeometry
from contextmap.spatial_relations import (
    CandidatePolicy,
    FrameConventions,
    GeometricPredicatePolicy,
    RelationPredicate,
    RelationsRunLineage,
    RelationsRunPolicies,
    SpatialRelationsRunId,
    SpatialRelationsRunReader,
    SpatialRelationsRunWriter,
    decide_relations,
    evaluate_geometric_candidates,
    generate_relation_candidates,
    lineage_from_resolution_manifest,
    resolved_entity_geometries,
)

CONVENTIONS = FrameConventions(map_frame="map")
GEOMETRIC = GeometricPredicatePolicy(
    boundary_tolerance_m=0.02,
    next_to_max_gap_m=0.6,
    adjacent_penetration_m=0.05,
    containment_slack_m=0.05,
    directional_overlap_fraction=0.5,
)
CANDIDATES = CandidatePolicy(
    predicates=(RelationPredicate.NEXT_TO,), proximity_radius_m=0.6, directional_radius_m=1.0
)
POLICIES = RelationsRunPolicies(
    frame_conventions=CONVENTIONS, candidate=CANDIDATES, geometric=GEOMETRIC
)


@pytest.fixture(scope="module")
def resolution_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("resolution") / "run"
    write_er_run(directory)
    return directory


def _geometries(reader: EntityResolutionRunReader) -> dict[ResolvedEntityReference, EntityGeometry]:
    return resolved_entity_geometries(
        reader.resolved_entities(), source=er_scene_source(), policy=CONNECTED_POLICY
    )


# --- lineage from the manifest ---


def test_the_lineage_is_derived_from_the_resolution_manifest(resolution_dir: Path) -> None:
    manifest = EntityResolutionRunReader(resolution_dir).manifest
    lineage = lineage_from_resolution_manifest(manifest)
    assert lineage == RelationsRunLineage(
        entity_resolution_run_id=ER_RUN,
        entity_resolution_schema_version=manifest.schema_version,
        entity_resolution_artifact_digest=resolution_artifact_digest(manifest),
        geometric_map_id=manifest.lineage.geometric_map_id,
    )
    assert lineage.geometric_map_id == MapId("map-0001")


# --- the geometry of resolved entities ---


def test_resolved_entities_are_given_the_geometry_the_resolution_persisted(
    resolution_dir: Path,
) -> None:
    resolved = EntityResolutionRunReader(resolution_dir).resolved_entities()
    geometries = _geometries(EntityResolutionRunReader(resolution_dir))
    assert set(geometries) == {
        ResolvedEntityReference(
            resolution_run_id=ER_RUN, resolved_entity_id=item.resolved_entity_id
        )
        for item in resolved.entities
    }
    for item in resolved.entities:
        reference = ResolvedEntityReference(
            resolution_run_id=ER_RUN, resolved_entity_id=item.resolved_entity_id
        )
        geometry = geometries[reference]
        assert geometry.geometry_refs == item.geometry.geometry_refs
        assert geometry.bounds == item.geometry.bounds
        assert geometry.map_frame == item.geometry.map_frame


def test_geometry_that_differs_from_what_the_resolution_persisted_is_refused(
    resolution_dir: Path,
) -> None:
    resolved = EntityResolutionRunReader(resolution_dir).resolved_entities()
    shifted = ResolvedEntitySet(
        resolution_run_id=resolved.resolution_run_id,
        entities=tuple(
            dataclasses.replace(
                item,
                geometry=dataclasses.replace(
                    item.geometry,
                    bounds=dataclasses.replace(item.geometry.bounds, maximum_m=(99.0, 99.0, 99.0)),
                ),
            )
            for item in resolved.entities
        ),
    )
    with pytest.raises(ValueError, match="bounds"):
        resolved_entity_geometries(shifted, source=er_scene_source(), policy=CONNECTED_POLICY)


# --- the run reads its resolution ---


def _spatial_run(
    reader: EntityResolutionRunReader,
    directory: Path,
    *,
    ghost: bool = False,
    lineage_changes: dict[str, str] | None = None,
) -> SpatialRelationsRunReader:
    geometries = _geometries(reader)
    if ghost:
        template = next(iter(geometries.values()))
        geometries[
            ResolvedEntityReference(
                resolution_run_id=ER_RUN, resolved_entity_id=ResolvedEntityId("resolved--ghost")
            )
        ] = template
    candidates = generate_relation_candidates(
        geometries, policy=CANDIDATES, conventions=CONVENTIONS
    )
    evidence = evaluate_geometric_candidates(
        candidates, entities=geometries, policy=GEOMETRIC, conventions=CONVENTIONS
    )
    decisions = decide_relations(candidates, evidence)
    derived = lineage_from_resolution_manifest(reader.manifest)
    lineage = dataclasses.replace(derived, **(lineage_changes or {}))  # type: ignore[arg-type]
    SpatialRelationsRunWriter(
        output_dir=directory,
        run_id=SpatialRelationsRunId("relations-run-0001"),
        lineage=lineage,
        policies=POLICIES,
        code_version="test",
    ).write(candidates=candidates, evidence=evidence, decisions=decisions)
    return SpatialRelationsRunReader(directory)


def test_a_run_over_the_resolution_it_names_validates_clean(
    resolution_dir: Path, tmp_path: Path
) -> None:
    resolution = EntityResolutionRunReader(resolution_dir)
    run = _spatial_run(resolution, tmp_path / "relations")
    assert list(run.iter_relations())
    assert run.validate_resolution(resolution) == ()
    assert run.manifest.lineage == lineage_from_resolution_manifest(resolution.manifest)


def test_a_change_of_the_upstream_artifact_is_detected(
    resolution_dir: Path, tmp_path: Path
) -> None:
    resolution = EntityResolutionRunReader(resolution_dir)
    cases = {
        "entity_resolution_artifact_digest": ("sha256:other", "digest"),
        "entity_resolution_schema_version": ("9.9.9", "schema"),
    }
    for index, (field, (value, word)) in enumerate(cases.items()):
        run = _spatial_run(
            resolution, tmp_path / f"relations-{index}", lineage_changes={field: value}
        )
        issues = run.validate_resolution(resolution)
        assert len(issues) == 1 and word in issues[0], issues


def test_a_reference_to_an_entity_the_resolution_does_not_have_is_reported(
    resolution_dir: Path, tmp_path: Path
) -> None:
    resolution = EntityResolutionRunReader(resolution_dir)
    run = _spatial_run(resolution, tmp_path / "relations", ghost=True)
    issues = run.validate_resolution(resolution)
    assert len(issues) == 1
    assert "resolved--ghost" in issues[0]


def test_a_resolution_run_that_is_not_the_one_named_is_reported(
    resolution_dir: Path, tmp_path: Path
) -> None:
    resolution = EntityResolutionRunReader(resolution_dir)
    run = _spatial_run(resolution, tmp_path / "relations")
    other_dir = tmp_path / "other-resolution"
    write_er_run(other_dir)
    manifest = json.loads((other_dir / "manifest.json").read_text())
    manifest["run_id"] = "resolution-run-0002"
    (other_dir / "manifest.json").write_text(json.dumps(manifest))
    issues = run.validate_resolution(EntityResolutionRunReader(other_dir))
    assert any("resolution-run-0002" in item for item in issues)

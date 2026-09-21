"""The synthetic Solution 1 chain: stage artifacts written and read back through public readers.

Contract evidence only (fake/contract): the geometry, poses and projections come from formulas and
the perception evidence is canned. The chain fills the gap the CI fixture catalogue declares as
"multi-view fusion not available": every stage runs its real code and persists its real artifact.
"""

from __future__ import annotations

from pathlib import Path

from chain import SyntheticChain, synthetic_chain


def test_every_stage_of_the_chain_persists_an_artifact_that_verifies(tmp_path: Path) -> None:
    with synthetic_chain(tmp_path) as chain:
        assert chain.sequence.verify_integrity() == []
        assert chain.trajectory.verify_integrity() == []
        assert chain.geometry.verify_integrity() == []
        for _, association in chain.associations:
            assert association.verify_integrity() == []
        assert chain.fusion.verify_integrity() == []
        assert chain.geometry.manifest.point_count == 12
        # Duas runs de percepção: a run-b repete o frame-0000 (3 frames físicos, 4 vistas).
        assert len(chain.spatial_observations) == 8
        assert len(chain.fusion_outcomes) == 2
        assert chain.fusion_excluded == ()


def test_the_fused_evidence_keeps_the_disagreement_between_repeated_inferences(
    tmp_path: Path,
) -> None:
    with synthetic_chain(tmp_path) as chain:
        pallet, shelf = (outcome.evidence for outcome in chain.fusion_outcomes)

        labels = {hypothesis.label for hypothesis in pallet.hypotheses}
        assert labels == {"pallet", "crate"}
        assert [item.kind.value for item in pallet.uncertainty] == ["contradiction"]
        assert {hypothesis.label for hypothesis in shelf.hypotheses} == {"shelf post"}
        assert len(pallet.contributions) == 4
        assert len(pallet.physical_observation_groups) == 3


def _inventory(manifest: object) -> dict[str, str]:
    return {entry.path: entry.content_hash for entry in manifest.file_inventory}  # type: ignore[attr-defined]


def _manifests(chain: SyntheticChain) -> list[object]:
    return [
        chain.sequence.manifest,
        chain.trajectory.manifest,
        chain.geometry.manifest,
        *(reader.manifest for _, reader in chain.associations),
        chain.fusion.manifest,
    ]


def test_rerunning_the_chain_reproduces_every_contractual_file_hash(tmp_path: Path) -> None:
    with (
        synthetic_chain(tmp_path / "first") as first,
        synthetic_chain(tmp_path / "second") as second,
    ):
        for left, right in zip(_manifests(first), _manifests(second), strict=True):
            assert _inventory(left) == _inventory(right)

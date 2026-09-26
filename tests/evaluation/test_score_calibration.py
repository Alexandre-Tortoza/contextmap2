"""Calibration/evaluation split guard for native scores (#573), on toy reference sets."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from reference_set_builders import sample_id, write_valid_reference_set

from contextmap.evaluation import (
    CalibrationSplitError,
    CalibrationSplitPair,
    ReferenceSetIntegrityError,
    SelectionBinding,
    require_valid_reference_set,
)
from contextmap.evaluation.reference_set import (
    ReferenceSplit,
    SplitRole,
    SplitScheme,
    SplitUnit,
)


def _validated(tmp_path: Path, scheme: SplitScheme | None = None):  # type: ignore[no-untyped-def]
    overrides = {} if scheme is None else {"split_schemes": (scheme,)}
    manifest = write_valid_reference_set(tmp_path, **overrides)
    return require_valid_reference_set(manifest, tmp_path)


def _three_way_scheme() -> SplitScheme:
    return SplitScheme(
        scheme_id="grounding-by-sequence",
        task="grounding_calibration",
        unit=SplitUnit.SEQUENCE,
        rationale="frames of one sequence are temporally correlated",
        splits=(
            ReferenceSplit(
                name="calibration",
                role=SplitRole.DEVELOPMENT,
                sample_ids=(sample_id(0), sample_id(1)),
            ),
            ReferenceSplit(
                name="held-out", role=SplitRole.TEST, sample_ids=(sample_id(2), sample_id(3))
            ),
        ),
    )


def test_a_disjoint_calibration_and_held_out_pair_is_accepted(tmp_path: Path) -> None:
    validated = _validated(tmp_path)

    pair = CalibrationSplitPair.from_reference_set(
        validated,
        scheme_id="regions-by-sequence",
        calibration_split="tuning",
        evaluation_split="test",
    )

    assert pair.calibration.role is SplitRole.TUNING
    assert pair.evaluation.role is SplitRole.TEST
    assert set(pair.calibration.sample_ids).isdisjoint(pair.evaluation.sample_ids)
    assert CalibrationSplitPair.from_record(pair.to_record()) == pair


def test_a_development_split_may_calibrate(tmp_path: Path) -> None:
    validated = _validated(tmp_path, _three_way_scheme())

    pair = CalibrationSplitPair.from_reference_set(
        validated,
        scheme_id="grounding-by-sequence",
        calibration_split="calibration",
        evaluation_split="held-out",
    )

    assert pair.calibration.role is SplitRole.DEVELOPMENT


@pytest.mark.parametrize(
    ("calibration", "evaluation", "message"),
    [
        ("test", "tuning", "held-out"),
        ("tuning", "tuning", "different"),
        ("test", "test", "different"),
    ],
)
def test_calibrating_on_the_held_out_split_or_evaluating_off_it_is_refused(
    tmp_path: Path, calibration: str, evaluation: str, message: str
) -> None:
    validated = _validated(tmp_path)

    with pytest.raises(CalibrationSplitError, match=message):
        CalibrationSplitPair.from_reference_set(
            validated,
            scheme_id="regions-by-sequence",
            calibration_split=calibration,
            evaluation_split=evaluation,
        )


def test_shared_samples_between_calibration_and_evaluation_are_leakage(tmp_path: Path) -> None:
    validated = _validated(tmp_path)
    manifest = validated.manifest
    calibration = SelectionBinding.for_split(manifest, "regions-by-sequence", "tuning")
    evaluation = SelectionBinding.for_split(manifest, "regions-by-sequence", "test")
    leaking = replace(evaluation, sample_ids=(*evaluation.sample_ids, calibration.sample_ids[0]))

    with pytest.raises(CalibrationSplitError, match="leak"):
        CalibrationSplitPair(calibration=calibration, evaluation=leaking)


def test_a_pair_across_schemes_or_reference_sets_is_refused(tmp_path: Path) -> None:
    validated = _validated(tmp_path)
    calibration = SelectionBinding.for_split(validated.manifest, "regions-by-sequence", "tuning")
    evaluation = SelectionBinding.for_split(validated.manifest, "regions-by-sequence", "test")

    with pytest.raises(CalibrationSplitError, match="scheme"):
        CalibrationSplitPair(
            calibration=calibration, evaluation=replace(evaluation, scheme_id="other")
        )
    with pytest.raises(CalibrationSplitError, match="reference set"):
        CalibrationSplitPair(
            calibration=calibration,
            evaluation=replace(
                evaluation, reference_set=replace(evaluation.reference_set, version="2.0.0")
            ),
        )


def test_a_reference_set_whose_splits_leak_can_never_feed_a_calibration(tmp_path: Path) -> None:
    straddling = SplitScheme(
        scheme_id="regions-by-sequence",
        task="grounding_calibration",
        unit=SplitUnit.SEQUENCE,
        rationale="frames of one sequence are temporally correlated",
        splits=(
            ReferenceSplit(
                name="tuning",
                role=SplitRole.TUNING,
                sample_ids=(sample_id(0), sample_id(1), sample_id(2)),
            ),
            ReferenceSplit(name="test", role=SplitRole.TEST, sample_ids=(sample_id(3),)),
        ),
    )

    with pytest.raises(ReferenceSetIntegrityError, match="split-leakage"):
        _validated(tmp_path, straddling)

"""Held-out split guard for calibrating model-native scores into probabilities.

A model-native score (LocateAnything decoder statistics, SAM2's predicted IoU, ...) is not a
probability that an output is correct. It may only become one through a calibration fitted
on one split of a validated reference set and evaluated on a *different*, held-out ``TEST``
split of the same scheme. :class:`CalibrationSplitPair` is that pair, and it refuses every
way the two could leak into each other. It deliberately carries no calibrator and no
calibrated value: those need real annotated data (#573) and do not exist yet, so nothing in
the repository can hold a calibrated score today. See
``src/contextmap/evaluation/docs/score-calibration.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from contextmap.evaluation.experiments import SelectionBinding
from contextmap.evaluation.reference_integrity import ValidatedReferenceSet
from contextmap.evaluation.reference_set import SplitRole


class CalibrationSplitError(ValueError):
    """Raised when a calibration and its evaluation are not a disjoint held-out pair."""


@dataclass(frozen=True, kw_only=True)
class CalibrationSplitPair:
    """The split a score calibration is fitted on and the held-out split it is judged on.

    Attributes:
        calibration: Samples a calibrator may be fitted on; never a ``TEST`` split.
        evaluation: Held-out samples the calibration is evaluated on; always ``TEST``.
    """

    calibration: SelectionBinding
    evaluation: SelectionBinding

    def __post_init__(self) -> None:
        """Refuse a pair from different reference sets or schemes, or one that can leak."""
        calibration, evaluation = self.calibration, self.evaluation
        if calibration.reference_set != evaluation.reference_set:
            raise CalibrationSplitError(
                "calibration and evaluation must come from the same reference set version"
            )
        if calibration.scheme_id != evaluation.scheme_id:
            raise CalibrationSplitError(
                "calibration and evaluation must be splits of the same split scheme"
            )
        if calibration.split == evaluation.split:
            raise CalibrationSplitError("calibration and evaluation must be different splits")
        if calibration.role is SplitRole.TEST:
            raise CalibrationSplitError(
                f"split {calibration.split!r} is the held-out TEST split; a calibrator is never "
                "fitted on it"
            )
        if evaluation.role is not SplitRole.TEST:
            raise CalibrationSplitError(
                f"split {evaluation.split!r} is {evaluation.role.value}; a calibration is judged "
                "only on a held-out TEST split"
            )
        shared = set(calibration.sample_ids) & set(evaluation.sample_ids)
        if shared:
            raise CalibrationSplitError(
                f"calibration and evaluation leak through shared samples: {sorted(shared)}"
            )

    @classmethod
    def from_reference_set(
        cls,
        validated: ValidatedReferenceSet,
        *,
        scheme_id: str,
        calibration_split: str,
        evaluation_split: str,
    ) -> CalibrationSplitPair:
        """Bind a pair of splits of a validated reference set.

        A :class:`ValidatedReferenceSet` already refuses a scheme whose splits share a
        sample, a physical observation, a grouping unit or an adjacent time span, so the
        pair inherits those guarantees; this adds the calibration-specific rules.

        Raises:
            ReferenceSetError: If the scheme or a split does not exist.
            CalibrationSplitError: If the pair is not a disjoint held-out pair.
        """
        manifest = validated.manifest
        return cls(
            calibration=SelectionBinding.for_split(manifest, scheme_id, calibration_split),
            evaluation=SelectionBinding.for_split(manifest, scheme_id, evaluation_split),
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "calibration": self.calibration.to_record(),
            "evaluation": self.evaluation.to_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CalibrationSplitPair:
        """Rebuild a pair from :meth:`to_record` output, re-checking every rule."""
        return cls(
            calibration=SelectionBinding.from_record(record["calibration"]),
            evaluation=SelectionBinding.from_record(record["evaluation"]),
        )

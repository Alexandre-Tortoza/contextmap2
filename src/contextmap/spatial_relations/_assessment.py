"""The decision machinery shared by every evidence channel of Spatial Relations.

The geometric and the contact evaluators decide a predicate the same way: a list of *conditions*,
each compared with a threshold under an explicit tolerance band, is turned into a status, and the
conditions that could not be decided become caveats. That rule is one domain rule with one owner, so
it lives here once and the channels only differ in the measurements and conditions they produce.

A comparison is decided only when it holds (or fails) for every value within the tolerance of the
measurement. The status follows from the conditions:

* any condition that fails makes the record ``CONFLICTS``;
* otherwise, an input the rule needs but did not have makes it ``UNAVAILABLE``;
* otherwise, when every condition meets its threshold it is ``SUPPORTS``;
* otherwise it is ``AMBIGUOUS``.

A sparse or disconnected support is never decisive, because its bounds may not describe the object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from contextmap.semantic_mapping import (
    EntityGeometry,
    GeometryDiagnosticKind,
    geometry_set_digest,
)
from contextmap.spatial_relations.evidence import (
    EvidenceCaveat,
    EvidenceCaveatKind,
    MeasuredGeometry,
    Quantity,
    RelationEvidenceStatus,
)


class Verdict(Enum):
    """The outcome of one comparison against one threshold."""

    MEETS = "meets"
    FAILS = "fails"
    WITHIN_TOLERANCE = "within_tolerance"


@dataclass(frozen=True)
class Check:
    """One condition of a predicate, decided under a tolerance.

    Attributes:
        name: The measurement the condition is about.
        verdict: Whether the condition meets, fails or is within tolerance of its threshold.
        value: The measurement.
        threshold: The threshold it was compared with.
        tolerance: The tolerance band used.
        unit: The unit of the measurement, for the caveat text.
        caveat: An explanation to record instead of the default one when the condition is not
            decided.
    """

    name: str
    verdict: Verdict
    value: float
    threshold: float
    tolerance: float
    unit: str = "m"
    caveat: EvidenceCaveat | None = None

    def explanation(self) -> EvidenceCaveat:
        """The caveat that says why this condition was not decided."""
        if self.caveat is not None:
            return self.caveat
        return EvidenceCaveat(
            kind=EvidenceCaveatKind.WITHIN_TOLERANCE,
            detail=(
                f"{self.name} {self.value:.6g} is within {self.tolerance:.6g} {self.unit} of its "
                f"threshold {self.threshold:.6g}"
            ),
        )


def at_least(name: str, value: float, threshold: float, tolerance: float, unit: str = "m") -> Check:
    """Decide ``value >= threshold`` under a tolerance band."""
    if value - tolerance >= threshold:
        verdict = Verdict.MEETS
    elif value + tolerance < threshold:
        verdict = Verdict.FAILS
    else:
        verdict = Verdict.WITHIN_TOLERANCE
    return Check(name, verdict, value, threshold, tolerance, unit)


def at_most(name: str, value: float, threshold: float, tolerance: float, unit: str = "m") -> Check:
    """Decide ``value <= threshold`` under a tolerance band."""
    if value + tolerance <= threshold:
        verdict = Verdict.MEETS
    elif value - tolerance > threshold:
        verdict = Verdict.FAILS
    else:
        verdict = Verdict.WITHIN_TOLERANCE
    return Check(name, verdict, value, threshold, tolerance, unit)


@dataclass(frozen=True)
class Assessment:
    """The numbers and conditions of one predicate over one pair, before they become evidence.

    Attributes:
        measurements: The measured numbers.
        thresholds: The thresholds and tolerances the conditions used.
        checks: The conditions, decided.
        measures_penetration_depth: Whether the predicate reads an interpenetration depth, which a
            flat support cannot state.
        missing_inputs: Caveats for inputs the rule needed and did not have; when no condition
            fails they make the record unavailable instead of decided.
        measured_geometry: Geometry subsets measured on top of the two whole supports.
    """

    measurements: list[Quantity]
    thresholds: list[Quantity]
    checks: list[Check]
    measures_penetration_depth: bool = False
    missing_inputs: list[EvidenceCaveat] = field(default_factory=list)
    measured_geometry: list[MeasuredGeometry] = field(default_factory=list)


def decide(
    assessment: Assessment, subject: EntityGeometry, obj: EntityGeometry
) -> tuple[RelationEvidenceStatus, list[EvidenceCaveat]]:
    """Turn the conditions of an assessment into a status and the caveats behind it.

    Args:
        assessment: The measured conditions.
        subject: The subject's geometry, for the reliability of its support.
        obj: The object's geometry.

    Returns:
        The status and its caveats, unsorted.
    """
    verdicts = [check.verdict for check in assessment.checks]
    caveats = [
        check.explanation()
        for check in assessment.checks
        if check.verdict is Verdict.WITHIN_TOLERANCE
    ]
    if Verdict.FAILS in verdicts:
        status = RelationEvidenceStatus.CONFLICTS
    elif assessment.missing_inputs:
        status = RelationEvidenceStatus.UNAVAILABLE
        caveats.extend(assessment.missing_inputs)
    elif all(verdict is Verdict.MEETS for verdict in verdicts):
        status = RelationEvidenceStatus.SUPPORTS
    else:
        status = RelationEvidenceStatus.AMBIGUOUS
    unreliable = reliability_caveats(subject, obj, assessment.measures_penetration_depth)
    if unreliable:
        # Limites de um suporte esparso, desconexo ou plano podem não descrever o objeto: nunca
        # decisivos, mas as medições e a razão continuam registradas.
        if status in (RelationEvidenceStatus.SUPPORTS, RelationEvidenceStatus.CONFLICTS):
            status = RelationEvidenceStatus.AMBIGUOUS
        caveats.extend(unreliable)
    return status, caveats


def reliability_caveats(
    subject: EntityGeometry, obj: EntityGeometry, measures_penetration_depth: bool
) -> list[EvidenceCaveat]:
    """State why the support of an entity may not describe the object, when it may not.

    Args:
        subject: The subject's geometry.
        obj: The object's geometry.
        measures_penetration_depth: Whether a flat support is a problem for the predicate.

    Returns:
        One caveat per problem, per entity; empty when both supports are reliable.
    """
    caveats: list[EvidenceCaveat] = []
    for role, geometry in (("subject", subject), ("object", obj)):
        kinds = geometry.diagnostic_kinds()
        if GeometryDiagnosticKind.SPARSE_SUPPORT in kinds:
            caveats.append(
                EvidenceCaveat(
                    kind=EvidenceCaveatKind.UNRELIABLE_GEOMETRY,
                    detail=(
                        f"the {role} support is sparse ({geometry.statistics.point_count} "
                        f"points): its bounds may not describe the object"
                    ),
                )
            )
        if GeometryDiagnosticKind.DISCONNECTED_SUPPORT in kinds:
            caveats.append(
                EvidenceCaveat(
                    kind=EvidenceCaveatKind.UNRELIABLE_GEOMETRY,
                    detail=(
                        f"the {role} support is disconnected "
                        f"({geometry.statistics.component_count} components): its bounds may not "
                        f"describe one object"
                    ),
                )
            )
        if measures_penetration_depth and GeometryDiagnosticKind.DEGENERATE_EXTENT in kinds:
            caveats.append(
                EvidenceCaveat(
                    kind=EvidenceCaveatKind.DEGENERATE_GEOMETRY,
                    detail=(
                        f"the {role} support is flat on some axis, so an interpenetration depth "
                        f"cannot be measured along it"
                    ),
                )
            )
    return caveats


def measured_geometry(role: str, geometry: EntityGeometry) -> MeasuredGeometry:
    """Describe a whole entity support by the digest of its geometry identities."""
    return MeasuredGeometry(
        role=role,
        point_count=geometry.statistics.point_count,
        geometry_digest=geometry_set_digest(geometry.geometry_refs),
    )


def quantity(name: str, value: float, unit: str) -> Quantity:
    """Build a named, finite, unit-carrying measurement."""
    return Quantity(name=name, value=value, unit=unit)

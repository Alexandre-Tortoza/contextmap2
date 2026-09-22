"""Optional 3D structural comparison of two entities from their point representations.

Point Representation is an optional evidence channel: an entity may reference the numerical
description of the local 3D structure around its geometry, and this channel compares those
descriptions. Resolution works without it, and it never depends on one backend: a deterministic
descriptor and a learned encoder (PTv3 or any other) cross the same boundary.

* **only within one representation space.** Two representations are comparable when their
  representation-space fingerprints are identical, never because their dimensions match; different
  checkpoints, descriptor versions, normalization or support semantics are different spaces. The
  policy declares the space. When an entity has no representation in that space the channel is
  ``unavailable`` (``incompatible_domain``) and when it has none at all it is ``unavailable``
  (``missing_evidence``): missing structure is never a zero score and never evidence for
  ``DISTINCT``. A store that *contradicts* the space a reference declares is corrupt evidence and
  fails loudly with :class:`~contextmap.point_representation.RepresentationSpaceMismatchError`;
* **undefined components are never interpreted.** An encoder may be unable to define some
  component for a support (a curvature of a degenerate support): its stored value is a placeholder.
  Such a component is left out of the comparison instead of being read as zero, the measurement
  records how many components were compared, and the channel is unavailable when the two entities
  have too few components defined in common;
* **static structure, not views.** Representations are anchored to geometry, not to camera
  observations, so unlike appearance there is no per-observation grouping: each support counts once.

The baseline aggregation, ``support-prototype-v1``, is a prototype: each component of the entity
prototype is the mean of that component over the entity's representations that define it (and is
undefined when none does). The similarity is the cosine of the two prototypes over the components
defined in both; the smallest and largest cosine between one representation of each side (over the
components those two share) stay on the measurement. Cosine similarity is a score with no
probabilistic meaning and is never combined with another channel's score; for a descriptor whose
components have very different scales it favors the large ones, so a different metric would be a
new policy version.

Nothing here classifies, concatenates a 3D vector with a visual or semantic value, or requires
PTv3.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from contextmap.entity_resolution import _vectors
from contextmap.entity_resolution.channels import (
    EvidenceStatus,
    Finding,
    PointRepresentationEvidence,
    RepresentationMeasurement,
    RepresentationRef,
    Unavailability,
    UnavailableReason,
)
from contextmap.entity_resolution.models import PolicyRef, reference_order
from contextmap.point_representation import (
    PointRepresentation,
    PointRepresentationRunId,
    PointRepresentationRunReader,
    RepresentationSpaceMismatchError,
    ensure_compatible_representations,
)
from contextmap.semantic_mapping import Entity, EntityReference

REPRESENTATION_COMPARISON_POLICY_ID = "entity-representation-comparison-v1"
"""Versioned identity of the representation metric and rules described in this module."""

REPRESENTATION_AGGREGATION_ID = "support-prototype-v1"
"""Versioned identity of the aggregation of many representations into one comparison."""

_METRIC = "cosine-similarity"

_UNAVAILABLE_PRIORITY = (
    UnavailableReason.MISSING_EVIDENCE,
    UnavailableReason.INCOMPATIBLE_DOMAIN,
    UnavailableReason.INSUFFICIENT_EVIDENCE,
)


@dataclass(frozen=True, kw_only=True)
class LoadedRepresentation:
    """A point representation and its vector, as a source loaded them.

    Attributes:
        representation: The representation's metadata, including the space its vector really
            lives in and the components it could not define.
        values: The vector, one float per component; undefined components hold placeholders.
    """

    representation: PointRepresentation
    values: tuple[float, ...]


class RepresentationVectorSource(Protocol):
    """Where the vectors of the representations an entity references are loaded from.

    It isolates the persisted Point Representation runs from the comparison: the run-reader adapter
    and test doubles are implementations.
    """

    def load(self, representation: RepresentationRef) -> LoadedRepresentation:
        """Load one representation.

        Args:
            representation: The reference an entity holds, as a resolution contract.

        Returns:
            The representation's metadata and vector.
        """
        ...


class RunReaderRepresentationSource:
    """A :class:`RepresentationVectorSource` over persisted Point Representation runs."""

    def __init__(
        self, readers: Mapping[PointRepresentationRunId, PointRepresentationRunReader]
    ) -> None:
        """Wrap the runs.

        Args:
            readers: The opened run of each Point Representation run the entities reference.
        """
        self._readers = readers

    def load(self, representation: RepresentationRef) -> LoadedRepresentation:
        """Load one representation from its run, without loading the others.

        Args:
            representation: The reference an entity holds, as a resolution contract.

        Returns:
            The representation's metadata, as the run persisted it, and its vector.

        Raises:
            ValueError: If no run was given for the reference's run.
            RunArtifactError: If the run has no such representation or its payload is malformed.
        """
        reader = self._readers.get(representation.run_id)
        if reader is None:
            raise ValueError(f"no representation run was given for run {representation.run_id!r}")
        return LoadedRepresentation(
            representation=reader.representation(representation.representation_id),
            values=reader.vector(representation.representation_id),
        )


@dataclass(frozen=True, kw_only=True)
class RepresentationComparisonPolicy:
    """Explicit declaration of the space to compare in and how to read the similarity.

    There are no defaults: a cosine similarity is a score of one representation space with no
    probabilistic meaning, so what counts as similar is a choice a profile makes per space and
    justifies with validation data.

    Attributes:
        representation_space_id: Fingerprint of the representation space to compare in.
        min_supporting_similarity: Cosine similarity, in ``(-1, 1]``, from which structure supports
            the match.
        max_conflicting_similarity: Cosine similarity, in ``[-1, 1)`` and below the supporting one,
            at or under which structure argues against the match; ``None`` never argues against it.
        min_defined_component_fraction: Share of the components, in ``(0, 1]``, that both entities
            must define for the similarity to be trusted; below it the channel is unavailable.
    """

    representation_space_id: str
    min_supporting_similarity: float
    max_conflicting_similarity: float | None
    min_defined_component_fraction: float

    def __post_init__(self) -> None:
        """Validate the space and the thresholds.

        Raises:
            ValueError: If the space is empty, a threshold is not finite or is outside its range,
                or the conflicting threshold is not below the supporting one.
        """
        if not self.representation_space_id.strip():
            raise ValueError("representation_space_id must not be empty")
        supporting = self.min_supporting_similarity
        if not (math.isfinite(supporting) and -1.0 < supporting <= 1.0):
            raise ValueError(
                f"min_supporting_similarity must be within (-1, 1], got {supporting!r}"
            )
        conflicting = self.max_conflicting_similarity
        if conflicting is not None:
            if not (math.isfinite(conflicting) and -1.0 <= conflicting < 1.0):
                raise ValueError(
                    f"max_conflicting_similarity must be within [-1, 1), got {conflicting!r}"
                )
            if conflicting >= supporting:
                raise ValueError(
                    "max_conflicting_similarity must be below min_supporting_similarity"
                )
        fraction = self.min_defined_component_fraction
        if not (math.isfinite(fraction) and 0.0 < fraction <= 1.0):
            raise ValueError(
                f"min_defined_component_fraction must be within (0, 1], got {fraction!r}"
            )

    def fingerprint(self) -> str:
        """Hash the policy identity, aggregation, metric and thresholds, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "policy_id": REPRESENTATION_COMPARISON_POLICY_ID,
                "aggregation_id": REPRESENTATION_AGGREGATION_ID,
                "metric": _METRIC,
                "representation_space_id": self.representation_space_id,
                "min_supporting_similarity": self.min_supporting_similarity,
                "max_conflicting_similarity": self.max_conflicting_similarity,
                "min_defined_component_fraction": self.min_defined_component_fraction,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def ref(self) -> PolicyRef:
        """The policy identity and configuration, as recorded on the evidence."""
        return PolicyRef(
            policy_id=REPRESENTATION_COMPARISON_POLICY_ID,
            configuration_fingerprint=self.fingerprint(),
        )


@dataclass(frozen=True)
class _Vector:
    """One representation's vector and which components it defines."""

    values: tuple[float, ...]
    defined: tuple[bool, ...]


@dataclass(frozen=True)
class _Profile:
    """An entity's prototype: what is compared, computed once per entity."""

    refs: tuple[RepresentationRef, ...]
    representations: tuple[PointRepresentation, ...]
    vectors: tuple[_Vector, ...]
    prototype: _Vector


class RepresentationComparator:
    """Compares the 3D structure of entity pairs under one policy, loading each entity once.

    The comparator keeps the prototype of every entity it has seen, so an entity that takes part in
    several candidate pairs has its vectors loaded and aggregated once. Use one comparator per run.
    """

    def __init__(
        self, source: RepresentationVectorSource, policy: RepresentationComparisonPolicy
    ) -> None:
        """Create a comparator.

        Args:
            source: Where the vectors are loaded from.
            policy: The representation space to compare in and the thresholds.
        """
        self._source = source
        self._policy = policy
        self._profiles: dict[EntityReference, _Profile] = {}

    @property
    def policy(self) -> RepresentationComparisonPolicy:
        """The policy this comparator applies."""
        return self._policy

    def compare(self, entity_a: Entity, entity_b: Entity) -> PointRepresentationEvidence:
        """Compare the 3D structure of two entities.

        The result does not depend on the order of the arguments: the pair is put in canonical
        order and the measurements refer to it.

        Args:
            entity_a: One entity.
            entity_b: The other entity.

        Returns:
            The representation evidence, or an unavailable channel when an entity has no
            representation, has none in the declared space, or the two share too few defined
            components to be compared.

        Raises:
            ValueError: If a defined component is not finite or vectors of the compared space
                differ in dimension.
            RepresentationSpaceMismatchError: If the store reports another representation space
                than the one a reference declares.
        """
        first, second = sorted(
            (entity_a, entity_b), key=lambda item: reference_order(item.reference)
        )
        selected_a, selected_b = self._select(first), self._select(second)
        if isinstance(selected_a, Unavailability) or isinstance(selected_b, Unavailability):
            problems = [
                item for item in (selected_a, selected_b) if isinstance(item, Unavailability)
            ]
            return PointRepresentationEvidence(
                policy=self._policy.ref(), unavailable=_combine(problems)
            )
        profile_a, profile_b = self._profile(first, selected_a), self._profile(second, selected_b)
        ensure_compatible_representations(
            profile_a.representations[0], profile_b.representations[0]
        )
        dimension = len(profile_a.prototype.values)
        if dimension != len(profile_b.prototype.values):
            raise ValueError(
                f"representations of space {self._policy.representation_space_id!r} differ in "
                f"dimension: {dimension} and {len(profile_b.prototype.values)}"
            )
        common = _common(profile_a.prototype, profile_b.prototype)
        share = len(common) / dimension
        if share < self._policy.min_defined_component_fraction:
            return self._unavailable(
                f"the entities define {len(common)} of {dimension} components in common, "
                f"fewer than the {self._policy.min_defined_component_fraction} share the policy "
                f"requires; undefined components are never read as zero"
            )
        similarity = _restricted_cosine(profile_a.prototype, profile_b.prototype, common)
        if similarity is None:
            return self._unavailable(
                "the commonly defined components of an entity have no direction"
            )
        pairs = [
            value
            for left in profile_a.vectors
            for right in profile_b.vectors
            if (value := _restricted_cosine(left, right, _common(left, right))) is not None
        ]
        measurement = RepresentationMeasurement(
            representation_space_id=self._policy.representation_space_id,
            metric=_METRIC,
            aggregation_id=REPRESENTATION_AGGREGATION_ID,
            similarity=similarity,
            pair_similarity_min=min(pairs, default=similarity),
            pair_similarity_max=max(pairs, default=similarity),
            representations_a=profile_a.refs,
            representations_b=profile_b.refs,
            dimension=dimension,
            compared_components=len(common),
        )
        return PointRepresentationEvidence(
            policy=self._policy.ref(),
            measurement=measurement,
            findings=(_similarity_finding(measurement, self._policy),),
        )

    def _select(self, entity: Entity) -> tuple[RepresentationRef, ...] | Unavailability:
        """Select the entity's references in the declared space, without any I/O."""
        references = entity.evidence.point_representation_refs
        if not references:
            return Unavailability(
                reason=UnavailableReason.MISSING_EVIDENCE,
                detail=f"{entity.entity_id!r} has no point representation",
            )
        selected = tuple(
            RepresentationRef(
                run_id=ref.run_id,
                representation_id=ref.representation_id,
                representation_space_id=ref.representation_space_id,
                geometry_reference=ref.geometry_reference,
            )
            for ref in references
            if ref.representation_space_id == self._policy.representation_space_id
        )
        if not selected:
            spaces = sorted({ref.representation_space_id for ref in references})
            return Unavailability(
                reason=UnavailableReason.INCOMPATIBLE_DOMAIN,
                detail=(
                    f"{entity.entity_id!r} has representations only in space(s) {spaces!r}, but "
                    f"the policy compares within {self._policy.representation_space_id!r}"
                ),
            )
        return selected

    def _unavailable(self, detail: str) -> PointRepresentationEvidence:
        return PointRepresentationEvidence(
            policy=self._policy.ref(),
            unavailable=Unavailability(
                reason=UnavailableReason.INSUFFICIENT_EVIDENCE, detail=detail
            ),
        )

    def _profile(self, entity: Entity, references: tuple[RepresentationRef, ...]) -> _Profile:
        cached = self._profiles.get(entity.reference)
        if cached is not None:
            return cached
        representations: list[PointRepresentation] = []
        vectors: list[_Vector] = []
        for reference in references:
            loaded = self._source.load(reference)
            if loaded.representation.representation_space_id != reference.representation_space_id:
                raise RepresentationSpaceMismatchError(
                    f"representation {reference.representation_id!r} is referenced in space "
                    f"{reference.representation_space_id!r} but its run reports "
                    f"{loaded.representation.representation_space_id!r}"
                )
            undefined = set(loaded.representation.undefined_components)
            defined = tuple(index not in undefined for index in range(len(loaded.values)))
            if any(
                not math.isfinite(value)
                for value, is_defined in zip(loaded.values, defined, strict=True)
                if is_defined
            ):
                raise ValueError(
                    f"representation {reference.representation_id!r} has a defined component "
                    f"that is not finite"
                )
            representations.append(loaded.representation)
            vectors.append(_Vector(values=loaded.values, defined=defined))
        if len({len(item.values) for item in vectors}) != 1:
            raise ValueError(
                f"the representations of entity {entity.entity_id!r} differ in dimension"
            )
        profile = _Profile(
            refs=references,
            representations=tuple(representations),
            vectors=tuple(vectors),
            prototype=_prototype(vectors),
        )
        self._profiles[entity.reference] = profile
        return profile


def _prototype(vectors: list[_Vector]) -> _Vector:
    """Per component, the mean over the vectors that define it; undefined when none does."""
    values: list[float] = []
    defined: list[bool] = []
    for index in range(len(vectors[0].values)):
        present = [item.values[index] for item in vectors if item.defined[index]]
        values.append(math.fsum(present) / len(present) if present else 0.0)
        defined.append(bool(present))
    return _Vector(values=tuple(values), defined=tuple(defined))


def _common(first: _Vector, second: _Vector) -> list[int]:
    return [
        index
        for index, (left, right) in enumerate(zip(first.defined, second.defined, strict=True))
        if left and right
    ]


def _restricted_cosine(first: _Vector, second: _Vector, common: list[int]) -> float | None:
    """Cosine over the given components, or ``None`` when either side has no direction there."""
    left = [first.values[index] for index in common]
    right = [second.values[index] for index in common]
    if not left or _vectors.norm(left) == 0.0 or _vectors.norm(right) == 0.0:
        return None
    return _vectors.cosine(left, right)


def _combine(problems: list[Unavailability]) -> Unavailability:
    """One unavailability for a pair: the most basic obstacle, with what each entity lacks."""
    reason = min((item.reason for item in problems), key=_UNAVAILABLE_PRIORITY.index)
    return Unavailability(reason=reason, detail="; ".join(item.detail for item in problems))


def _similarity_finding(
    measurement: RepresentationMeasurement, policy: RepresentationComparisonPolicy
) -> Finding:
    similarity = measurement.similarity
    conflicting = policy.max_conflicting_similarity
    if similarity >= policy.min_supporting_similarity:
        status, threshold = EvidenceStatus.SUPPORTING, policy.min_supporting_similarity
    elif conflicting is not None and similarity <= conflicting:
        status, threshold = EvidenceStatus.CONFLICTING, conflicting
    else:
        status, threshold = EvidenceStatus.NEUTRAL, policy.min_supporting_similarity
    return Finding(
        rule_id="representation-similarity",
        status=status,
        detail=(
            f"cosine similarity {similarity:.3f} between the prototypes of "
            f"{len(measurement.representations_a)} and {len(measurement.representations_b)} "
            f"representation(s) over {measurement.compared_components} of "
            f"{measurement.dimension} components in space {measurement.representation_space_id!r}"
        ),
        metric="similarity",
        observed=similarity,
        threshold=threshold,
    )

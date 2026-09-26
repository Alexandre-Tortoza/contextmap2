"""Appearance comparison of two entities from their visual features: one channel, one space.

Entities reference visual features; the vectors live in Visual Perception's feature store. This
channel compares them, but only where a comparison means something:

* **only within one embedding space.** Two features are comparable when their embedding-space
  fingerprints are identical, never because their dimensions match: a DINOv3 and a CLIP vector of
  the same size are not comparable. The policy declares the space to compare in. When an entity has
  no feature in that space the channel is ``unavailable`` (``incompatible_domain``), and when it has
  no region feature at all it is ``unavailable`` (``missing_evidence``): missing appearance is never
  a zero score and never evidence for ``DISTINCT``. A store that *contradicts* the reference of a
  feature (it reports another space than the entity's reference) is corrupt evidence and fails
  loudly with :class:`~contextmap.visual_perception.EmbeddingSpaceMismatchError`;
* **only region-scoped features** describe an entity: a global or dense feature describes the whole
  image, not the entity's region;
* **one vote per physical observation.** Repeated inference over one frame yields several
  correlated features; they are grouped under that frame, so they are never independent appearance
  evidence.

The baseline aggregation, ``physical-observation-prototype-v1``, is a prototype: each feature is
scaled to unit length, the features of one physical observation are averaged into that
observation's prototype (and scaled to unit length again), and the entity prototype is the mean of
its observation prototypes, each observation counting once. The similarity is the cosine of the two
entity prototypes; the smallest and largest cosine between one observation of each side stay on the
measurement, so a pair whose views disagree is visible. Cosine similarity is a score with no
probabilistic meaning and is never combined with another channel's score.

Nothing here classifies, projects across spaces, or concatenates a visual vector with anything
else, and appearance alone never identifies an object: it is one finding among the others.
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
    AppearanceEvidence,
    AppearanceMeasurement,
    EvidenceStatus,
    FeatureContribution,
    Finding,
    Unavailability,
    UnavailableReason,
)
from contextmap.entity_resolution.models import PolicyRef, reference_order
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_mapping import Entity, EntityFeatureRef, EntityReference
from contextmap.visual_perception import (
    EmbeddingSpaceMismatchError,
    FeatureScope,
    FeatureStoreReader,
    PerceptionRunId,
    VisualFeature,
    ensure_compatible_features,
    perception_result_id_for,
)

APPEARANCE_COMPARISON_POLICY_ID = "entity-appearance-comparison-v1"
"""Versioned identity of the appearance metric and rules described in this module."""

APPEARANCE_AGGREGATION_ID = "physical-observation-prototype-v1"
"""Versioned identity of the aggregation of many features into one comparison."""

_METRIC = "cosine-similarity"

_UNAVAILABLE_PRIORITY = (
    UnavailableReason.MISSING_EVIDENCE,
    UnavailableReason.INCOMPATIBLE_DOMAIN,
    UnavailableReason.INSUFFICIENT_EVIDENCE,
)


@dataclass(frozen=True, kw_only=True)
class LoadedFeature:
    """A visual feature and its vector, as a source loaded them.

    Attributes:
        feature: The feature's metadata, including the embedding space the vector really lives in.
        values: The vector, one float per component.
    """

    feature: VisualFeature
    values: tuple[float, ...]


class FeatureVectorSource(Protocol):
    """Where the vectors of the features an entity references are loaded from.

    It isolates the feature store (files, hashes, NumPy) from the comparison: the store adapter
    and an in-memory fake used by tests are two real implementations.
    """

    def load(
        self, feature: EntityFeatureRef, physical_observation_id: SourceObservationId
    ) -> LoadedFeature:
        """Load one feature of one physical observation.

        Args:
            feature: The reference an entity holds.
            physical_observation_id: The physical observation the feature was extracted from.

        Returns:
            The feature's metadata and vector.
        """
        ...


class FeatureStoreVectorSource:
    """A :class:`FeatureVectorSource` over the persisted feature stores of perception runs."""

    def __init__(self, stores: Mapping[PerceptionRunId, FeatureStoreReader]) -> None:
        """Wrap the stores.

        Args:
            stores: The opened feature store of each perception run the entities reference.
        """
        self._stores = stores

    def load(
        self, feature: EntityFeatureRef, physical_observation_id: SourceObservationId
    ) -> LoadedFeature:
        """Load one feature from its run's store, verifying its integrity.

        Args:
            feature: The reference an entity holds.
            physical_observation_id: The physical observation the feature was extracted from.

        Returns:
            The feature's metadata, as the store indexed it, and its vector.

        Raises:
            ValueError: If no store was given for the run, the store's scope differs from the
                reference's, or the payload is not a vector.
            FeatureStoreError: If the store has no such feature or a payload is missing.
            FeaturePayloadIntegrityError: If a payload does not match its index.
        """
        reader = self._stores.get(feature.perception_run_id)
        if reader is None:
            raise ValueError(
                f"no feature store was given for perception run {feature.perception_run_id!r}"
            )
        entry = reader.entry(physical_observation_id, feature.feature_id)
        if entry.scope is not feature.scope:
            raise ValueError(
                f"feature {feature.feature_id!r} is {entry.scope.value}-scoped in the store but "
                f"{feature.scope.value}-scoped in the entity"
            )
        array = reader.load(physical_observation_id, feature.feature_id)
        if array.ndim != 1:
            raise ValueError(
                f"feature {feature.feature_id!r} must be a vector, got shape {tuple(array.shape)}"
            )
        return LoadedFeature(
            feature=VisualFeature(
                feature_id=feature.feature_id,
                scope=entry.scope,
                embedding_space_id=entry.embedding_space_id,
                shape=entry.shape,
                dtype=entry.dtype,
                payload_reference=entry.payload_reference,
                provenance=entry.provenance,
                region_id=feature.region_id,
                normalization=entry.normalization,
            ),
            values=tuple(float(value) for value in array.tolist()),
        )


@dataclass(frozen=True, kw_only=True)
class AppearanceComparisonPolicy:
    """Explicit declaration of the space to compare in and how to read the similarity.

    There are no defaults: a cosine similarity is a score of one embedding space with no
    probabilistic meaning, so what counts as similar is a choice a profile makes per space and
    justifies with validation data.

    Attributes:
        embedding_space_id: Fingerprint of the embedding space to compare in.
        min_supporting_similarity: Cosine similarity, in ``(-1, 1]``, from which appearance
            supports the match.
        max_conflicting_similarity: Cosine similarity, in ``[-1, 1)`` and below the supporting one,
            at or under which appearance argues against the match; ``None`` never argues against
            it, because a different viewpoint can look different.
    """

    embedding_space_id: str
    min_supporting_similarity: float
    max_conflicting_similarity: float | None = None

    def __post_init__(self) -> None:
        """Validate the space and the thresholds.

        Raises:
            ValueError: If the space is empty, a threshold is not finite or is outside its range,
                or the conflicting threshold is not below the supporting one.
        """
        if not self.embedding_space_id.strip():
            raise ValueError("embedding_space_id must not be empty")
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

    def fingerprint(self) -> str:
        """Hash the policy identity, aggregation, metric and thresholds, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {
                "policy_id": APPEARANCE_COMPARISON_POLICY_ID,
                "aggregation_id": APPEARANCE_AGGREGATION_ID,
                "metric": _METRIC,
                "embedding_space_id": self.embedding_space_id,
                "min_supporting_similarity": self.min_supporting_similarity,
                "max_conflicting_similarity": self.max_conflicting_similarity,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def ref(self) -> PolicyRef:
        """The policy identity and configuration, as recorded on the evidence."""
        return PolicyRef(
            policy_id=APPEARANCE_COMPARISON_POLICY_ID, configuration_fingerprint=self.fingerprint()
        )


@dataclass(frozen=True)
class _Plan:
    """The region features of an entity in the declared space, grouped by physical observation."""

    observations: tuple[tuple[SourceObservationId, tuple[EntityFeatureRef, ...]], ...]


@dataclass(frozen=True)
class _Profile:
    """An entity's prototype: what is compared, computed once per entity."""

    contributions: tuple[FeatureContribution, ...]
    features: tuple[VisualFeature, ...]
    observation_prototypes: tuple[tuple[float, ...], ...]
    prototype: tuple[float, ...]


class AppearanceComparator:
    """Compares the appearance of entity pairs under one policy, loading each entity once.

    The comparator keeps the prototype of every entity it has seen, so an entity that takes part in
    several candidate pairs has its vectors loaded and aggregated once. Use one comparator per run.
    """

    def __init__(self, source: FeatureVectorSource, policy: AppearanceComparisonPolicy) -> None:
        """Create a comparator.

        Args:
            source: Where the vectors are loaded from.
            policy: The embedding space to compare in and the thresholds.
        """
        self._source = source
        self._policy = policy
        self._profiles: dict[EntityReference, _Profile] = {}

    @property
    def policy(self) -> AppearanceComparisonPolicy:
        """The policy this comparator applies."""
        return self._policy

    def compare(self, entity_a: Entity, entity_b: Entity) -> AppearanceEvidence:
        """Compare the appearance of two entities.

        The result does not depend on the order of the arguments: the pair is put in canonical
        order and the measurements refer to it.

        Args:
            entity_a: One entity.
            entity_b: The other entity.

        Returns:
            The appearance evidence, or an unavailable channel when an entity has no region
            feature, has none in the declared embedding space, or its prototype has no direction.

        Raises:
            ValueError: If a feature belongs to no physical observation of its entity, a vector is
                degenerate (zero or not finite), or vectors of the compared space differ in
                dimension.
            EmbeddingSpaceMismatchError: If the store reports another embedding space than the
                one an entity's reference declares.
        """
        first, second = sorted(
            (entity_a, entity_b), key=lambda item: reference_order(item.reference)
        )
        plan_a, plan_b = self._plan(first), self._plan(second)
        if isinstance(plan_a, Unavailability) or isinstance(plan_b, Unavailability):
            problems = [plan for plan in (plan_a, plan_b) if isinstance(plan, Unavailability)]
            return AppearanceEvidence(policy=self._policy.ref(), unavailable=_combine(problems))
        profile_a, profile_b = self._profile(first, plan_a), self._profile(second, plan_b)
        ensure_compatible_features(profile_a.features[0], profile_b.features[0])
        if len(profile_a.prototype) != len(profile_b.prototype):
            raise ValueError(
                f"features of embedding space {self._policy.embedding_space_id!r} differ in "
                f"dimension: {len(profile_a.prototype)} and {len(profile_b.prototype)}"
            )
        if _vectors.norm(profile_a.prototype) == 0.0 or _vectors.norm(profile_b.prototype) == 0.0:
            return AppearanceEvidence(
                policy=self._policy.ref(),
                unavailable=Unavailability(
                    reason=UnavailableReason.INSUFFICIENT_EVIDENCE,
                    detail="the observation prototypes of an entity cancel out: no direction",
                ),
            )
        # Protótipos são unitários, então o produto interno já é o cosseno; o clamp em [-1, 1]
        # só corrige o arredondamento (dot(u, u) pode passar de 1.0), como faz _vectors.cosine.
        pairs = [
            max(-1.0, min(1.0, _vectors.dot(left, right)))
            for left in profile_a.observation_prototypes
            for right in profile_b.observation_prototypes
        ]
        measurement = AppearanceMeasurement(
            embedding_space_id=self._policy.embedding_space_id,
            metric=_METRIC,
            aggregation_id=APPEARANCE_AGGREGATION_ID,
            similarity=_vectors.cosine(profile_a.prototype, profile_b.prototype),
            pair_similarity_min=min(pairs),
            pair_similarity_max=max(pairs),
            contributions_a=profile_a.contributions,
            contributions_b=profile_b.contributions,
        )
        return AppearanceEvidence(
            policy=self._policy.ref(),
            measurement=measurement,
            findings=(_similarity_finding(measurement, self._policy),),
        )

    def _plan(self, entity: Entity) -> _Plan | Unavailability:
        """Select the entity's region features in the declared space, without any I/O."""
        regions = [
            ref for ref in entity.evidence.visual_feature_refs if ref.scope is FeatureScope.REGION
        ]
        if not regions:
            return Unavailability(
                reason=UnavailableReason.MISSING_EVIDENCE,
                detail=f"{entity.entity_id!r} has no region-scoped visual feature",
            )
        selected = [
            ref for ref in regions if ref.embedding_space_id == self._policy.embedding_space_id
        ]
        if not selected:
            spaces = sorted({ref.embedding_space_id for ref in regions})
            return Unavailability(
                reason=UnavailableReason.INCOMPATIBLE_DOMAIN,
                detail=(
                    f"{entity.entity_id!r} has features only in embedding space(s) {spaces!r}, "
                    f"but the policy compares within {self._policy.embedding_space_id!r}"
                ),
            )
        by_result = {
            perception_result_id_for(run_id=ref.perception_run_id, source_observation_id=obs): obs
            for ref in selected
            for obs in entity.evidence.physical_observation_ids
        }
        grouped: dict[SourceObservationId, list[EntityFeatureRef]] = {}
        for ref in selected:
            observation = by_result.get(ref.perception_result_id)
            if observation is None:
                raise ValueError(
                    f"feature {ref.feature_id!r} of result {ref.perception_result_id!r} belongs "
                    f"to no physical observation of entity {entity.entity_id!r}"
                )
            grouped.setdefault(observation, []).append(ref)
        return _Plan(
            observations=tuple((obs, tuple(refs)) for obs, refs in sorted(grouped.items()))
        )

    def _profile(self, entity: Entity, plan: _Plan) -> _Profile:
        cached = self._profiles.get(entity.reference)
        if cached is not None:
            return cached
        contributions: list[FeatureContribution] = []
        loaded_features: list[VisualFeature] = []
        prototypes: list[tuple[float, ...]] = []
        for observation, refs in plan.observations:
            vectors: list[tuple[float, ...]] = []
            for ref in refs:
                loaded = self._source.load(ref, observation)
                if loaded.feature.embedding_space_id != self._policy.embedding_space_id:
                    raise EmbeddingSpaceMismatchError(
                        f"feature {ref.feature_id!r} is referenced in embedding space "
                        f"{self._policy.embedding_space_id!r} but its store reports "
                        f"{loaded.feature.embedding_space_id!r}"
                    )
                loaded_features.append(loaded.feature)
                vectors.append(_vectors.unit(loaded.values, what=f"feature {ref.feature_id!r}"))
            prototypes.append(
                _vectors.unit(
                    _vectors.mean(vectors), what=f"the features of observation {observation!r}"
                )
            )
            contributions.append(
                FeatureContribution(
                    physical_observation_id=observation,
                    feature_refs=tuple(
                        sorted(refs, key=lambda ref: (ref.perception_result_id, ref.feature_id))
                    ),
                )
            )
        profile = _Profile(
            contributions=tuple(contributions),
            features=tuple(loaded_features),
            observation_prototypes=tuple(prototypes),
            prototype=_vectors.mean(prototypes),
        )
        self._profiles[entity.reference] = profile
        return profile


def _combine(problems: list[Unavailability]) -> Unavailability:
    """One unavailability for a pair: the most basic obstacle, with what each entity lacks."""
    reason = min((item.reason for item in problems), key=_UNAVAILABLE_PRIORITY.index)
    return Unavailability(reason=reason, detail="; ".join(item.detail for item in problems))


def _similarity_finding(
    measurement: AppearanceMeasurement, policy: AppearanceComparisonPolicy
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
        rule_id="appearance-similarity",
        status=status,
        detail=(
            f"cosine similarity {similarity:.3f} between the prototypes of "
            f"{len(measurement.contributions_a)} and {len(measurement.contributions_b)} physical "
            f"observation(s) in embedding space {measurement.embedding_space_id!r}"
        ),
        metric="similarity",
        observed=similarity,
        threshold=threshold,
    )

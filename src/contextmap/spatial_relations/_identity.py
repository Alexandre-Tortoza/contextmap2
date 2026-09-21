"""Identity rules shared by relations and their evidence.

A relation and the evidence about it are both about one *directed candidate*: a subject, a
predicate and an object. This module holds the two rules that every record about a candidate
shares, so that they cannot drift apart: what makes a pair of entities relatable, and how the
candidate is hashed into a deterministic identity.
"""

from __future__ import annotations

import hashlib

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.spatial_relations.taxonomy import RelationPredicate

_SEPARATOR = "\x1f"
_DIGEST_HEX_LENGTH = 16


def require_relatable_pair(subject: ResolvedEntityReference, obj: ResolvedEntityReference) -> None:
    """Require two distinct resolved entities of one resolution artifact.

    Args:
        subject: The entity the statement is about.
        obj: The entity it is related to.

    Raises:
        ValueError: If both references name the same entity, or they come from different
            resolution artifacts: an id is only meaningful inside the artifact that allocated it,
            so entities of two artifacts are never assumed to be comparable.
    """
    if subject == obj:
        raise ValueError(f"an entity is never related to itself: {subject.resolved_entity_id!r}")
    if subject.resolution_run_id != obj.resolution_run_id:
        raise ValueError(
            f"subject and object come from different resolution artifacts, "
            f"{subject.resolution_run_id!r} and {obj.resolution_run_id!r}"
        )


def reference_key(reference: ResolvedEntityReference) -> tuple[str, str]:
    """Order references canonically: by resolution artifact, then by resolved entity.

    Args:
        reference: A resolved-entity reference.

    Returns:
        The sort key; equal references have equal keys and different ones never do.
    """
    return (reference.resolution_run_id, reference.resolved_entity_id)


def candidate_digest(
    subject: ResolvedEntityReference,
    predicate: RelationPredicate,
    obj: ResolvedEntityReference,
) -> str:
    """Hash a directed candidate into a short, deterministic digest.

    The hash covers the full reference of both entities, so the same local ids in two resolution
    artifacts never collide, and the order matters: ``a PREDICATE b`` and ``b PREDICATE a`` are
    different candidates.

    Args:
        subject: The entity the statement is about.
        predicate: The predicate.
        obj: The entity it is related to.

    Returns:
        A lowercase hexadecimal digest.
    """
    joined = _SEPARATOR.join(
        (
            subject.resolution_run_id,
            subject.resolved_entity_id,
            predicate.value,
            obj.resolution_run_id,
            obj.resolved_entity_id,
        )
    )
    return hashlib.sha256(joined.encode()).hexdigest()[:_DIGEST_HEX_LENGTH]

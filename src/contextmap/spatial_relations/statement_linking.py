"""Linking upstream relational statements to resolved entities, through explicit evidence only.

An upstream statement names things the way perception saw them: as regions in a frame, projected
into the map as spatial observations. It does not know which resolved entity those observations
ended up in. That knowledge lives in Entity Resolution: a resolved entity carries the spatial
observations of *every* one of its members in its evidence, and lists the members it merged, and the
run reader answers which resolved entities a spatial observation supports. This module derives the
link from exactly that answer, and from nothing else:

* an end is linked to the one resolved entity the reader says its spatial observation supports;
* an observation that no resolved entity has, or that several have, is **not** linked and is
  reported with the reason, never guessed and never resolved by proximity or by label;
* a statement whose two ends land in the same resolved entity cannot be a relation between two
  entities, and is reported as such.

The result feeds :func:`~contextmap.spatial_relations.observation_evidence_from_statements`. A
statement that cannot be linked is neutral: it produces no evidence, and it is listed so that the
loss is visible.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from contextmap.entity_resolution import EntityResolutionRunReader, ResolvedEntityReference
from contextmap.spatial_relations._checks import require_present
from contextmap.spatial_relations.statements import (
    EndpointLink,
    ObservationRelationStatement,
    StatementPolarity,
    UpstreamStatementRef,
)


class LinkFailure(Enum):
    """Why a statement could not be linked to two distinct resolved entities.

    Attributes:
        NOT_IN_ANY_ENTITY: A spatial observation of the statement is in the evidence of no
            resolved entity.
        IN_SEVERAL_ENTITIES: A spatial observation is in the evidence of more than one resolved
            entity, so the end is ambiguous.
        BOTH_ENDS_IN_ONE_ENTITY: The two ends are in the same resolved entity.
    """

    NOT_IN_ANY_ENTITY = "not_in_any_entity"
    IN_SEVERAL_ENTITIES = "in_several_entities"
    BOTH_ENDS_IN_ONE_ENTITY = "both_ends_in_one_entity"


@dataclass(frozen=True, kw_only=True)
class UpstreamRelationStatement:
    """A relational statement as upstream made it, before it is tied to any resolved entity.

    Attributes:
        source: The exact upstream statement.
        subject_spatial_observation_id: The spatial observation the statement is about.
        predicate_text: The predicate exactly as the upstream statement worded it.
        object_spatial_observation_id: The spatial observation it is related to.
        polarity: Whether the statement asserts or denies the predicate.
    """

    source: UpstreamStatementRef
    subject_spatial_observation_id: str
    predicate_text: str
    object_spatial_observation_id: str
    polarity: StatementPolarity

    def __post_init__(self) -> None:
        """Require the wording and two distinct spatial observations.

        Raises:
            ValueError: If the wording or an observation is empty, or both ends are the same
                observation.
        """
        require_present(
            self,
            "subject_spatial_observation_id",
            "predicate_text",
            "object_spatial_observation_id",
        )
        if self.subject_spatial_observation_id == self.object_spatial_observation_id:
            raise ValueError("a statement cannot relate a spatial observation to itself")

    @property
    def sort_key(self) -> tuple[str, str, str, str, str]:
        """A total order that makes a set of statements canonical."""
        return (
            self.source.source_run_id,
            self.source.statement_id,
            self.subject_spatial_observation_id,
            self.predicate_text,
            self.object_spatial_observation_id,
        )


@dataclass(frozen=True, kw_only=True)
class UnlinkedStatement:
    """A statement that could not be linked, and why.

    Attributes:
        statement: The statement, untouched.
        failures: The distinct reasons, sorted.
        detail: A deterministic, human-readable explanation naming the observations concerned.
    """

    statement: UpstreamRelationStatement
    failures: tuple[LinkFailure, ...]
    detail: str


@dataclass(frozen=True, kw_only=True)
class LinkedStatements:
    """The outcome of linking a set of statements.

    Attributes:
        linked: The statements tied to two distinct resolved entities, ready to become
            observation evidence, in canonical order.
        unlinked: The statements that could not be tied, with the reasons, in canonical order.
    """

    linked: tuple[ObservationRelationStatement, ...]
    unlinked: tuple[UnlinkedStatement, ...]


def link_statements(
    statements: Iterable[UpstreamRelationStatement], *, resolution: EntityResolutionRunReader
) -> LinkedStatements:
    """Tie the ends of upstream statements to the resolved entities that hold them.

    Args:
        statements: The upstream statements, in any order.
        resolution: The reader of one Entity Resolution run. Its ``resolved_of_spatial_observation``
            says which resolved entities a spatial observation supports and never picks one: zero
            or several are reported here, not chosen.

    Returns:
        The linked statements and the unlinked ones with their reasons. The same statements and
        run always give the same result, whatever the order of the statements.
    """
    members: dict[ResolvedEntityReference, str] = {}
    linked: list[ObservationRelationStatement] = []
    unlinked: list[UnlinkedStatement] = []
    for statement in sorted(statements, key=lambda item: item.sort_key):
        subject, subject_failure = _holder(statement.subject_spatial_observation_id, resolution)
        obj, object_failure = _holder(statement.object_spatial_observation_id, resolution)
        failures = {item for item in (subject_failure, object_failure) if item is not None}
        if subject is not None and subject == obj:
            failures.add(LinkFailure.BOTH_ENDS_IN_ONE_ENTITY)
        if failures or subject is None or obj is None:
            unlinked.append(
                UnlinkedStatement(
                    statement=statement,
                    failures=tuple(sorted(failures, key=lambda item: item.value)),
                    detail=_explain(statement, subject_failure, object_failure, subject),
                )
            )
            continue
        linked.append(
            ObservationRelationStatement(
                source=statement.source,
                subject=_link(
                    statement.subject_spatial_observation_id, subject, resolution, members
                ),
                predicate_text=statement.predicate_text,
                object=_link(statement.object_spatial_observation_id, obj, resolution, members),
                polarity=statement.polarity,
            )
        )
    return LinkedStatements(linked=tuple(linked), unlinked=tuple(unlinked))


def _holder(
    observation: str, resolution: EntityResolutionRunReader
) -> tuple[ResolvedEntityReference | None, LinkFailure | None]:
    found = resolution.resolved_of_spatial_observation(observation)
    if not found:
        return None, LinkFailure.NOT_IN_ANY_ENTITY
    if len(found) > 1:
        return None, LinkFailure.IN_SEVERAL_ENTITIES
    return found[0], None


def _link(
    observation: str,
    entity: ResolvedEntityReference,
    resolution: EntityResolutionRunReader,
    members: dict[ResolvedEntityReference, str],
) -> EndpointLink:
    if entity not in members:
        merged = resolution.resolved_entity(entity)
        members[entity] = ", ".join(
            sorted(str(member.entity_ref.entity_id) for member in merged.members)
        )
    return EndpointLink(
        upstream_ref=observation,
        entity_ref=entity,
        linked_through=(
            f"spatial observation {observation} is in the evidence of resolved entity "
            f"{entity.resolved_entity_id} (members: {members[entity]})"
        ),
    )


def _explain(
    statement: UpstreamRelationStatement,
    subject_failure: LinkFailure | None,
    object_failure: LinkFailure | None,
    subject: ResolvedEntityReference | None,
) -> str:
    parts: list[str] = []
    for observation, failure in (
        (statement.subject_spatial_observation_id, subject_failure),
        (statement.object_spatial_observation_id, object_failure),
    ):
        if failure is LinkFailure.NOT_IN_ANY_ENTITY:
            parts.append(
                f"spatial observation {observation} is in the evidence of no resolved entity"
            )
        elif failure is LinkFailure.IN_SEVERAL_ENTITIES:
            parts.append(
                f"spatial observation {observation} is in the evidence of several resolved entities"
            )
    if not parts and subject is not None:
        parts.append(
            f"both spatial observations are in the evidence of resolved entity "
            f"{subject.resolved_entity_id}, and an entity is never related to itself"
        )
    return "; ".join(parts)

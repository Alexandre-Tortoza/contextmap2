"""Upstream relational statements, held by reference and never interpreted.

A visual perception or language model may say something relational about a scene: "the crate is on
the table". Nothing upstream defines a relational claim, so this capability defines the smallest
contract that can carry one *as evidence*: who said it (the run and the statement identity), which
physical observation it was about, what it says (the predicate exactly as stated and whether it
asserts or denies it), and how each of its two ends was tied to a resolved entity.

That last part is the point. A statement names things in upstream terms (a region, a claim), not
resolved entities. It only becomes evidence about two resolved entities through an **explicit
link** per end: the upstream identity that was named, the resolved entity it was tied to, and the
evidence the tie went through. Nothing here guesses a link, and a statement whose end has no link to
the selected entity set is refused rather than approximated.

The raw statement never becomes a relation. It is one evidence channel, kept apart from geometry,
and the decision policy never lets it establish or override a measured relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.spatial_relations._checks import require_present


class StatementPolarity(Enum):
    """What an upstream statement says about the predicate it names.

    Attributes:
        ASSERTS: The statement says the predicate holds.
        DENIES: The statement says the predicate does not hold.
    """

    ASSERTS = "asserts"
    DENIES = "denies"


@dataclass(frozen=True, kw_only=True)
class UpstreamStatementRef:
    """The exact upstream statement, by identity.

    Attributes:
        source_run_id: The upstream run that produced the statement, such as a perception run.
        statement_id: The statement, unique inside ``source_run_id``.
        physical_observation_id: The physical observation, such as a frame, the statement is
            about; the same frame seen by two inference runs is still one observation.
        producer: The backend, model and prompt version that produced it, as an opaque text, so
            statements of different producers stay distinguishable.
    """

    source_run_id: str
    statement_id: str
    physical_observation_id: str
    producer: str

    def __post_init__(self) -> None:
        """Require every identity.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(
            self, "source_run_id", "statement_id", "physical_observation_id", "producer"
        )


@dataclass(frozen=True, kw_only=True)
class EndpointLink:
    """The explicit tie of one end of a statement to a resolved entity.

    Attributes:
        upstream_ref: What the statement named, in upstream terms, such as a region or a claim.
        entity_ref: The resolved entity it was tied to.
        linked_through: The evidence the tie went through, such as the fused evidence or the
            spatial observation that connects the upstream reference to the entity.
    """

    upstream_ref: str
    entity_ref: ResolvedEntityReference
    linked_through: str

    def __post_init__(self) -> None:
        """Require the upstream reference and the evidence of the tie.

        Raises:
            ValueError: If either is empty: a link without its evidence is a guess.
        """
        require_present(self, "upstream_ref", "linked_through")


@dataclass(frozen=True, kw_only=True)
class ObservationRelationStatement:
    """One upstream relational statement about two entities, linked and unmapped.

    Attributes:
        source: The exact upstream statement.
        subject: The link of the end the statement is about.
        predicate_text: The predicate exactly as the upstream statement worded it.
        object: The link of the end it is related to.
        polarity: Whether the statement asserts or denies the predicate.
    """

    source: UpstreamStatementRef
    subject: EndpointLink
    predicate_text: str
    object: EndpointLink
    polarity: StatementPolarity

    def __post_init__(self) -> None:
        """Require the wording and two distinct ends.

        Raises:
            ValueError: If the wording is empty, or both ends are tied to the same entity.
        """
        require_present(self, "predicate_text")
        if self.subject.entity_ref == self.object.entity_ref:
            raise ValueError("a statement cannot relate an entity to itself")

    @property
    def sort_key(self) -> tuple[str, str, str, str, str]:
        """A total order that makes a set of statements canonical."""
        return (
            self.source.source_run_id,
            self.source.statement_id,
            self.subject.upstream_ref,
            self.predicate_text,
            self.object.upstream_ref,
        )

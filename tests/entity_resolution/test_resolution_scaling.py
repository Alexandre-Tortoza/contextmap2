"""Scale gate for materialization and lookups: about 10^4 entities and 10^4 decisions.

Small fixtures hid the cost this gate exists for (#599): every member scanned every ``MATCH``
link of the run, and every lookup scanned the whole resolved set, so a run of 10^4 entities took
seconds where it takes a fraction of one. CI keeps to counters and a broad envelope; the measured
wall-clock numbers are in ``src/contextmap/entity_resolution/docs/resolved-entities.md``.
"""

from __future__ import annotations

import dataclasses
from itertools import pairwise
from time import perf_counter
from typing import Any

import pytest
from resolution_builders import decision_between
from resolution_entity_builders import entity_at

from contextmap.entity_resolution import (
    EntityResolutionRunId,
    ResolutionDecision,
    ResolutionOutcome,
    materialization,
    materialize_resolved_entities,
)
from contextmap.semantic_mapping import Entity

ENTITIES = 10_000
GROUP = 4
# Envelope largo: dez vezes o medido depois da #599, abaixo do medido antes (ver o docstring).
ENVELOPE_S = 10.0


class ScanCountingLinks(dict[Any, Any]):
    """A link table that counts every full scan."""

    scans = 0

    def items(self) -> Any:
        ScanCountingLinks.scans += 1
        return super().items()


def chains(entities: list[Entity]) -> list[ResolutionDecision]:
    """Chains of ``GROUP`` matched entities, each chain distinct from the next."""
    decisions = []
    for index, (first, second) in enumerate(pairwise(entities)):
        outcome = (
            ResolutionOutcome.DISTINCT if index % GROUP == GROUP - 1 else ResolutionOutcome.MATCH
        )
        decisions.append(decision_between(first.reference, second.reference, outcome))
    return decisions


def test_ten_thousand_entities_materialize_and_resolve_without_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_graph = materialization._match_graph

    def counting_graph(decisions: Any) -> Any:
        graph = build_graph(decisions)
        return dataclasses.replace(graph, link=ScanCountingLinks(graph.link))

    monkeypatch.setattr(materialization, "_match_graph", counting_graph)
    ScanCountingLinks.scans = 0
    entities = [
        entity_at(f"e{index:05d}", (index * 5.0, 0.0, 0.0), support_number=index + 1)
        for index in range(ENTITIES)
    ]
    decisions = chains(entities)

    started = perf_counter()
    result = materialize_resolved_entities(
        entities, decisions, resolution_run_id=EntityResolutionRunId("resolution-run-scale")
    )
    owners = {result.resolved.resolved_of(entity.reference) for entity in entities}
    elapsed = perf_counter() - started

    assert sum(item.decision is ResolutionOutcome.MATCH for item in decisions) == 7_500
    assert len(result.resolved.entities) == len(owners) == ENTITIES // GROUP
    assert ScanCountingLinks.scans == 0
    assert elapsed < ENVELOPE_S

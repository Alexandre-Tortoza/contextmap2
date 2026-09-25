# Entity resolution run

- Run ID: `entity-resolution--run-0001`
- Semantic mapping run: `mapping-run-0001`
- Source entities: 4, candidate pairs compared: 3
- MATCH: 1, DISTINCT: 1, UNRESOLVED: 1
- Resolved entities: 3, contradictions: 0

Contractual data is in `outputs/` and `metrics/`; `debug/` is human evidence and no downstream stage may depend on it. A resolved entity id is local to this run. Source entities are referenced, never copied or modified, and every source entity belongs to exactly one resolved entity.

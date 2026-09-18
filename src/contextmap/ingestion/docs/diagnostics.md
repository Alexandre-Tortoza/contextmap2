# Diagnósticos e sumário legível

Este documento descreve `src/contextmap/ingestion/diagnostics.py` e a integração opcional com `SequenceArtifactWriter`/`Reader` (issue #39).

## Debug, não contrato

Diagnósticos são evidência de depuração, nunca dependência contratual (`docs/ARTIFACTS.md`: "`outputs/` é contratual, `debug/` não é"). Um `SequenceArtifactReader` funciona plenamente sem nunca chamar nada deste módulo — `read_diagnostics()` retorna `None` quando `set_diagnostics()` nunca foi chamado no writer.

## `summarize_observations()`

Calcula um sumário legível por humano de um conjunto de observações: contagem por modalidade, intervalo de tempo normalizado (`first_timestamp_seconds`/`last_timestamp_seconds`, apenas comparável quando `clock_ids` tem uma única entrada — múltiplos `clock_id` na mesma modalidade sinalizam que o intervalo mistura domínios de clock e não deve ser lido como um único range), resoluções de imagem distintas vistas, e layouts distintos de campos de point cloud vistos. Não requer artefato persistido — funciona sobre qualquer `Sequence[SourceObservation]`, incluindo diretamente a saída de um `SourceAdapter` (#43) antes mesmo de persistir.

## Persistência: `diagnostics/summary.json` + `diagnostics/warnings.jsonl`

`writer.set_diagnostics(warnings=[...])` antes de `finalize()` ativa a escrita de diagnósticos: o sumário é computado automaticamente a partir das observações já adicionadas ao writer; o chamador só fornece a lista de warnings (texto legível). `reader.read_diagnostics()` retorna `SequenceDiagnostics(summary=..., warnings=...)` ou `None`.

### Por que só dois arquivos, não a estrutura completa sugerida pela issue

A issue cita `summary.json`, `warnings.jsonl`, `synchronization.jsonl`, `dropped-events.jsonl`, `frame-graph.json` como candidatos, mas também deixa explícito: "Exact names/formats may differ". O v0 entrega `summary.json` + `warnings.jsonl`; diagnósticos de sincronização (`SynchronizationDiagnostics.dropped_events` da issue #40) podem ser formatados como texto pelo chamador e passados na mesma lista `warnings` — um arquivo dedicado por tipo de diagnóstico fica adiado até que exista um consumidor real que precise parsear esses dados estruturadamente, em vez de apenas lê-los como texto. `frame-graph.json` (grafo de frames de calibração) também não é gerado no v0 — `validate_frame_references()` (`docs/validation.md`) já cobre a checagem que motivaria esse arquivo.

## Integração com `validation.py`

Um pipeline de ingestão típico chamaria `validate_observations()` (`docs/validation.md`) antes de `writer.finalize()`, formatando cada problema encontrado como uma entrada da lista `warnings` passada a `set_diagnostics()` — ou decidindo abortar a ingestão se os problemas forem graves o suficiente. Esta issue não impõe essa decisão; fornece os dois primitivos (`validate_observations()`, `set_diagnostics()`) para que o chamador decida a política.

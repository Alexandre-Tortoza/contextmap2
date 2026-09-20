# Evidência multi-run

Este documento descreve `src/contextmap/visual_perception/evidence_set.py`: como várias execuções de percepção, já persistidas (issue #52), podem ser lidas juntas sem fundir sua evidência.

## Seleção sempre explícita

`PerceptionEvidenceSet.open(run_dirs)` nunca infere "use todos os runs encontrados" ou "prefira o run mais recente". Um run conhecido-defeituoso é simplesmente excluído por não estar na lista — seu artefato nunca é lido, modificado ou apagado.

```mermaid
flowchart LR
    D1["run_dir A"] --> R1["PerceptionRunReader A"]
    D2["run_dir B"] --> R2["PerceptionRunReader B"]
    D3["run_dir C"] --> R3["PerceptionRunReader C"]
    R1 --> VALID["validar mesma sequence_artifact_id<br/>e run_id único"]
    R2 --> VALID
    R3 --> VALID
    VALID --> INDEX["índice em memória por<br/>source_observation_id"]
    INDEX --> VIEW["ObservationEvidence<br/>results_by_run"]
    VIEW -. nenhuma fusão .-> DOWN["Semantic Fusion / Entity Resolution"]
```
## Agrupamento por observação física, nunca fusão

`evidence_for(source_observation_id)` retorna um `ObservationEvidence` com `results_by_run: Mapping[PerceptionRunId, PerceptionResult]` — um `PerceptionResult` por run selecionado que produziu resultado para aquela observação. Esta view **nunca**:

- escolhe um rótulo vencedor entre runs;
- combina confidences;
- decide que rótulos de runs diferentes são equivalentes;
- associa regiões entre runs como o mesmo objeto;
- cria entidades persistentes;
- trata inferência repetida como evidência física independente.

O agrupamento correlacionado e a acumulação já pertencem a Semantic Fusion;
identidade persistente continua planejada para Entity Resolution. Esta view não
executa nenhuma das duas responsabilidades.

## Seleções sobrepostas

```mermaid
flowchart LR
    R1["run-0001<br/>frames 0000..1000"] --> F200["frame-0200"]
    R1 --> F450["frame-0450"]
    R5["run-0005<br/>frames 0430..0480"] --> F450
    R7["run-0007<br/>frames 0430..0480"] --> F450
    F200 --> E200["results_by_run<br/>{run-0001}"]
    F450 --> E450["results_by_run<br/>{run-0001, run-0005, run-0007}"]
```

Nenhum payload é copiado ou reescrito para construir essa view — `PerceptionEvidenceSet` apenas lê cada `PerceptionRunReader` já existente e monta um índice em memória por `source_observation_id`.

## Sequências incompatíveis

`open()`/`__init__` levantam `EvidenceSetError` se os runs selecionados referenciarem mais de um `sequence_artifact_id` — misturar evidência de sequências diferentes sob uma única observação não faz sentido semântico.

A seleção também é rejeitada quando dois diretórios declaram o mesmo `run_id`, pois a chave deixaria de identificar univocamente a origem da evidência. Para cada resultado lido, a view confirma ainda que `PerceptionResult.run_id` coincide com o `run_id` do manifest que o possui, evitando atribuição silenciosa a outro run.

## Sem `runs.json`

Cada run é aberto via `PerceptionRunReader` (issue #52), que nunca requer `runs.json` — a view multi-run funciona igual quando o registro de conveniência está ausente.

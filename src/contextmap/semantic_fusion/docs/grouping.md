# Agrupamento por observação física

`group_by_physical_observation` separa **quantos frames físicos** existem de **quantas inferências** os interpretaram. É o primeiro passo de Semantic Fusion e existe para que inferência repetida não vire confiança artificial.

## Regra de correlação

Três execuções de percepção sobre `frame-0120` são **uma** observação física com **três** resultados de inferência correlacionados:

```text
SourceObservation frame-0120
├── run A -> resultado, claims
├── run B -> resultado, claims
└── run C -> resultado, claims

physical_observation_count = 1
inference_result_count     = 3
```

Nenhuma inferência é descartada e nenhuma é contada como voto independente. A função também não assume independência estatística entre backends.

## Entrada

```python
grouping = group_by_physical_observation(
    observations,  # SpatialObservation, em qualquer ordem
    selected_runs=[run_a, run_b],  # PerceptionRun escolhidas explicitamente
    acquisition_timestamps=timestamps,  # SourceObservationId -> SourceTimestamp
)
```

- **Seleção explícita.** Só as runs em `selected_runs` participam. Uma observação de qualquer outra run é erro: nunca se mescla tudo o que existe para uma sequência.
- **Timestamps.** A aquisição de cada frame vem do chamador, normalmente da sequência canônica; um frame sem timestamp é erro.
- Uma `SpatialObservation` é uma região de um frame sob uma run. Várias regiões do mesmo resultado continuam sendo **um** resultado de inferência.

## Saída

`PhysicalObservationGrouping`:

| Campo / propriedade | Significado |
| --- | --- |
| `grouping_policy_id` | `physical-observation-grouping-v1`. |
| `selected_run_ids` | As runs recebidas, ordenadas. |
| `groups` | Um `PhysicalObservationGroup` por frame, ordenado por frame. |
| `physical_observation_count` | Frames físicos distintos. |
| `inference_result_count` | Resultados de inferência, correlacionados dentro de cada frame. |
| `perception_run_count` | Runs que produziram evidência. |
| `inference_variant_count` | Identidades de backend distintas entre essas runs. |

Duas runs com o mesmo backend e a mesma configuração são **repetição de inferência**, não variantes: `perception_run_count` conta as duas, `inference_variant_count` conta uma. Trocar o modelo, a versão ou o fingerprint de configuração cria uma variante nova. A variante é a identidade dos backends da run (`PerceptionRun.backend_provenance`), não o id da run nem o `code_version`.

O resultado não depende da ordem das observações nem da ordem das runs.

## Erros

O agrupamento falha cedo, com mensagem acionável, quando:

- nenhuma run é selecionada, uma run é selecionada duas vezes ou as runs processam sequências diferentes;
- uma observação vem de uma run não selecionada ou de outra sequência que a da sua run;
- a mesma `SpatialObservation` aparece duas vezes;
- uma run tem dois resultados para o mesmo frame, ou um resultado descreve dois frames;
- um frame não tem timestamp de aquisição.

Sem observações, o resultado é um agrupamento vazio (contagens zero), não um erro.

## O que não faz

- não escolhe hipótese, não decide vencedor nem cria identidade de entidade;
- não constrói `FusionSupport` (isso é a próxima etapa) nem `EvidenceContribution`;
- não lê claims nem scores: só a identidade das observações espaciais.

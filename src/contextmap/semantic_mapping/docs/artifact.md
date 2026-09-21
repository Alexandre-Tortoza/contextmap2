# `SemanticMappingRunArtifact`

Este documento descreve `src/contextmap/semantic_mapping/run_artifact.py` e `serialization.py`. As regras gerais de artifacts (imutabilidade, atomicidade, inventário, índice de run) estão em [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md) e são implementadas uma única vez em `contextmap.shared.run_directory`.

Um run persiste as **entidades semânticas** com os índices, a linhagem, as métricas e os candidatos rejeitados. É imutável, autodescritivo e **abre sem NumPy, sem runtime de percepção, sem runtime de fusão e sem biblioteca de modelo**: Entity Resolution e Spatial Relations consomem as saídas contratuais sem depender de `debug/`. Nada a montante é duplicado: evidência fundida, observações, features e representações 3D são só referenciadas, e a geometria é guardada como deltas posicionais.

## O que nunca se perde

- **Todas as hipóteses**, com alternativas, conflitos, abstenções e sinais sem score (`None`): o artifact nunca guarda só um label.
- **Frames físicos e resultados de inferência** continuam distintos, com o histórico de observações de cada entidade.
- **Vínculos de evidência** com a identidade, a versão e o digest do artifact de fusão: sobrevivem à reabertura e são validáveis contra o run de origem.
- **Geometria** como referência exata e os resumos derivados, com sua proveniência.
- **Não há estado de merge, split ou resolução**: isso pertence a Entity Resolution.

## Layout

```text
workspace/runs/semantic-mapping/<sequence>/
├── runs.json                                   # registry reconstruível
└── run-000N__<selection>__<policy>/
    ├── README.md
    ├── manifest.json                           # identidade, linhagem, política e inventário
    ├── outputs/                                # contratual
    │   ├── entities.jsonl                      # uma Entity canônica por linha (autocontida, autoritativa)
    │   ├── entity-index.jsonl                  # entidade → deslocamento e tamanho em entities.jsonl
    │   ├── entity-geometry-index.jsonl         # entidade → mapa, frame, pontos, limites, centroide, diagnósticos
    │   ├── entity-evidence-index.jsonl         # entidade → evidência fundida (run, evidência, suporte) e contagens
    │   ├── entity-observation-index.jsonl      # uma linha por (entidade, frame físico): instante e inferências
    │   ├── entity-semantic-state.jsonl         # entidade → ambiguidade, primária, labels, atributos, incerteza
    │   ├── entity-temporal-state.jsonl         # entidade → first/last seen, contagens, ciclo de vida
    │   └── rejected-candidates.jsonl           # candidatos que não viraram entidade, com o motivo
    ├── metrics/                                # contratual
    │   ├── counts.json
    │   ├── distributions.json
    │   ├── payload.json
    │   └── runtime.json                        # somente quando medido
    └── debug/                                  # nunca contratual
        └── entities/<entity-id>/{summary,geometry-summary,semantic-state,evidence-trace,temporal-history}.json
```

Não existem `config.yaml`, `lineage.json`, `environment.json` nem `events.jsonl` separados: a política e a linhagem ficam no `manifest.json`, e os candidatos rejeitados em `rejected-candidates.jsonl`. Criá-los sem produtor real violaria YAGNI (mesma decisão de Semantic Fusion, Sensor Association e Point Representation).

`entities.jsonl` é a fonte autoritativa; os índices são resumos derivados para acesso direto e consulta sem carregar cada entidade.

## Escrita

`SemanticMappingRunWriter.write(entities, rejections=..., warnings=..., runtime=...)` consome as entidades **como fluxo**, gravando cada registro assim que ele chega (`open_binary`, com hash durante a escrita); o run é publicado de forma atômica ao fim do fluxo.

O writer recusa, com `MappingRunArtifactError` e sem deixar run visível, quando:

- as entidades não chegam estritamente ordenadas por identidade (o que também recusa duplicatas);
- uma entidade pertence a outro semantic map que o do run;
- uma entidade é sobre outro mapa geométrico que o da linhagem, ou referencia um run de fusão, uma run de percepção ou uma run de Point Representation que a linhagem não lista;
- entidades usam política ou configuração de materialização diferentes: **um run guarda uma política**;
- um candidato é ao mesmo tempo entidade e rejeição, ou as rejeições não estão ordenadas por suporte;
- já existe um run no caminho (um run finalizado nunca é sobrescrito; reexecutar cria outro índice).

## Linhagem (manifest)

`MappingRunLineage` é a seleção **explícita** do run de fusão de origem, com a sua identidade, versão do schema e digest, o mapa geométrico, a sequência e, pela linhagem do run de fusão, as runs de Sensor Association, de percepção e de Point Representation. `lineage_from_fusion_manifest(manifest)` a deriva do manifest de fusão. O manifest também registra a política e o id de alocação (com o fingerprint da configuração), as contagens, os avisos, o código, o nível de debug e o inventário com hash de cada arquivo contratual. Um run sem entidades é válido e explícito (as políticas ficam `null`).

## Métricas

Separadas, sem um escalar único:

- `counts.json`: entidades; rejeitados (total e por motivo); entidades por estado de ambiguidade; com hipótese primária; com conflitos; **sem features visuais** e **sem representações 3D** (canais opcionais ausentes); frames físicos distintos e resultados de inferência (contados à parte); pontos de geometria; avisos;
- `distributions.json`: por entidade, pontos de geometria, frames físicos, resultados de inferência e hipóteses (contagem, mínimo, mediana, máximo);
- `payload.json`: o tamanho de cada arquivo contratual;
- `runtime.json`: só quando o chamador mediu (segundos, memória), à parte de qualquer medida de qualidade.

## Leitura

`SemanticMappingRunReader(run_dir)` abre um run **só pelo próprio diretório**: nenhum registry, nenhum outro arquivo e nenhum runtime de percepção, fusão ou modelo.

- `entity(reference)` resolve uma `EntityReference` **sem carregar as demais**, pelo deslocamento do `entity-index`. Devolve exatamente a entidade canônica escrita. Uma referência de **outro** semantic map levanta `ForeignEntityReferenceError`; uma entidade inexistente, `UnknownEntityError`.
- `iter_entities()`, `entities()` (um `EntitySet`), `entity_ids()`, `rejected_candidates()`.
- `entities_of_observation(physical_observation_id)`: as entidades a que um frame físico contribuiu.
- `read_table(...)` e `read_record(...)` só aceitam `outputs/` e `metrics/`; **`debug/` nunca é uma fonte válida**.

## Integridade e referências

- `verify_integrity()` compara o inventário com o disco e detecta arquivo ausente, tamanho ou hash diferente. Um registro truncado ou malformado é um erro explícito (`MappingRunArtifactError`), nunca um valor confiado: a decodificação revalida todas as invariantes.
- `validate_references(fusion_runs=..., geometry=...)` confere que a evidência que cada entidade referencia continua lá e inalterada, relatando artifact ausente, divergente, desatualizado ou corrompido, evidência inexistente, linhagem incompatível e geometria não resolvível ([`evidence.md`](evidence.md)). Cada run de fusão é verificado uma única vez, por maior que seja o número de entidades.

## Debug

`MappingDebugLevel.NONE` (padrão) não escreve nada; `STANDARD` escreve um resumo, a geometria, o estado semântico, os vínculos de evidência e o histórico temporal de uma amostra de entidades; `FULL` cobre todas. Nunca é inventariado, então removê-lo não invalida o run, e nenhum estágio a jusante pode depender dele.

## Índices de run

Cada run tem um índice monotônico por sequência, calculado por `allocate_mapping_run_index` a partir dos runs **válidos** no disco (um run corrompido ou incompleto não conta). `runs.json` é só uma conveniência reconstruível (`rebuild_mapping_run_registry`).

## Validação

`tests/semantic_mapping/test_semantic_mapping_run_artifact.py` escreve e reabre runs reais a partir de um run de fusão real: round-trip de referência para entidade, sobrevivência de geometria, alternativas, conflitos, evidência e tempo, leitura de uma entidade sem carregar as outras, abertura em um processo sem runtimes pesados, layout e manifest, índices, referências de outro mapa, corrupção (conteúdo, arquivo ausente, truncamento, tabela e registro malformados, schema não suportado), debug, métricas, recusas do writer, atomicidade e índices de run.

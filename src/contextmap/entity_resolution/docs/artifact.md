# `EntityResolutionRunArtifact`

Diretório imutável, atômico e autodescritivo com tudo o que uma execução de resolução decidiu e por quê: candidatos, evidência de comparação, **todas** as decisões (`MATCH`, `DISTINCT` e `UNRESOLVED`), entidades resolvidas com linhagem de fusão, contradições de transitividade, candidatos a divisão opcionais e métricas. Reabre sem NumPy, sem runtime de percepção, de representação ou de modelo (há teste que abre o artifact num subprocesso e confere que nenhum deles foi importado), e uma entidade resolvida é lida por referência sem carregar as outras.

## Layout

```text
<output_dir>/
├── manifest.json
├── README.md
├── outputs/                                (contratual, inventariado)
│   ├── candidate-sets.jsonl                um EntityCandidateSet por linha
│   ├── match-evidence.jsonl                a evidência tipada de cada par comparado
│   ├── resolution-decisions.jsonl          MATCH, DISTINCT e UNRESOLVED
│   ├── resolved-entities.jsonl             uma ResolvedEntity por linha
│   ├── resolved-entity-index.jsonl         id resolvido -> deslocamento e tamanho (leitura preguiçosa)
│   ├── source-to-resolved-index.jsonl      entidade de origem -> entidade resolvida
│   ├── merge-lineage.jsonl                 membros, decisões e contradições por entidade resolvida
│   ├── unresolved-entities.jsonl           entidades de origem que uma decisão deixou em aberto
│   ├── transitivity-contradictions.jsonl
│   └── split-candidates.jsonl              vazio se a detecção não rodou
├── metrics/                                (contratual, inventariado)
│   ├── counts.json
│   ├── distributions.json
│   ├── payload.json
│   └── runtime.json                        só se quem chamou mediu
└── debug/                                  (nunca contratual)
    └── decisions/<decision-id>/...
```

O escritor recebe o **diretório final** do artifact (`output_dir`): não aloca índice de execução, não mantém `runs.json` e não calcula caminho de workspace. A identidade da execução (`run_id`, o escopo de todo id resolvido) é dada por quem chama. O diretório é finalizado com `AtomicRunDirectory`: uma escrita interrompida nunca parece um run pronto, e um run pronto nunca é reescrito (`output_dir` existente é recusado: reexecutar cria outro artifact).

### Desvio das sugestões da issue

A issue sugere `config.yaml`, `lineage.json`, `environment.json` e `events.jsonl`. Seguindo os artifacts irmãos já entregues (Semantic Fusion, Semantic Mapping, Sensor Association, Point Representation), a **política e a linhagem ficam em `manifest.json`** e não há esses quatro arquivos. O código é registrado como `code_version` (revisão) e a configuração como fingerprint por política; não há digest do código.

## Manifest e linhagem

`EntityResolutionRunManifest`: `run_id`, `schema_version` (`0.1.0`), `created_at`, `code_version`, a **linhagem**, as **políticas**, as contagens, os avisos, o nível de debug e o inventário (tamanho e SHA-256 de cada arquivo contratual; o manifest, o README e o `debug/` ficam de fora).

`ResolutionRunLineage`: sequência, mapa geométrico, run de Semantic Mapping (identidade, versão do schema e **digest** de identidade e inventário, como `fusion_artifact_digest`), mapas semânticos resolvidos e os runs de percepção e de Point Representation por trás. `lineage_from_mapping_manifest(manifest)` deriva tudo do manifest do run de Semantic Mapping (`mapping_artifact_digest`).

As **políticas não são passadas ao escritor: são derivadas dos registros**, uma por papel: `candidate_retrieval`, `resolution`, `materialization`, `split_detection` e `channel_<canal>` (geometria, semântica, aparência, temporal, representação 3D). Dois fingerprints para o mesmo papel são recusados, então o manifest não pode discordar dos dados.

## Consistência checada antes de escrever

O escritor recusa (`RunArtifactError`), sem deixar nada no disco: par comparado que **não é candidato** (só candidatos são comparados, nunca all-pairs), par resolvido duas vezes, mais de uma política para um papel, entidades resolvidas de outro run, entidade fora dos mapas da linhagem ou sobre outro mapa geométrico, entidade resolvida que cita uma decisão que não foi escrita e contradição de transitividade cuja decisão `DISTINCT` ou cuja cadeia de `MATCH` não foi escrita.

## Leitura (`EntityResolutionRunReader`)

`candidate_sets()`, `match_evidence()`, `decisions()`, `contradictions()`, `split_candidates()`, `unresolved_entities()`, `resolved_entities()`, `materialization()`, e por referência: `resolved_entity(ResolvedEntityReference)` (lê só o registro, por deslocamento; referência de outro run levanta `ForeignResolvedEntityReferenceError`, id inexistente `UnknownResolvedEntityError`) e `resolved_of(EntityReference)` (o índice origem para resolvida). Todo registro é reconstruído pelo construtor do contrato, então uma linha adulterada é recusada e não confiada (`RunArtifactError`). `verify_integrity()` confere o inventário: arquivo faltando, tamanho ou hash diferentes. Um diretório sem `manifest.json` ou com schema desconhecido é recusado.

`read_table` e `read_record` só aceitam `outputs/*.jsonl` e `metrics/*.json`: **`debug/` nunca é uma fonte válida**. Spatial Relations consome as entidades resolvidas só dos arquivos contratuais, e apagar `debug/` não invalida o run nem muda o que se lê.

## O que quem consome fixa e segue

Um artifact posterior (Spatial Relations) precisa de três valores deste run, todos pela API pública de `contextmap.entity_resolution`:

- **versão do schema e digest do run:** `reader.manifest.schema_version` e `resolution_artifact_digest(reader.manifest)`. O digest é calculado aqui, pelo dono do artifact, com a mesma regra de `mapping_artifact_digest`: identidade do run, versão do schema e hash de cada arquivo contratual. Ignora `debug/` e o instante da escrita, então o mesmo conteúdo contratual tem o mesmo digest, e uma mudança posterior no run é detectável;
- **elo de uma afirmação a montante até a entidade resolvida:** uma afirmação sobre uma região só chega a uma entidade resolvida pela observação espacial que ligou a região ao 3D. `reader.resolved_of_spatial_observation(spatial_observation_id)` devolve as entidades resolvidas cuja evidência inclui essa observação. Zero significa que nenhuma entidade a usou; mais de uma significa que a observação sustenta entidades que a resolução manteve separadas. Os dois casos são devolvidos como são, nunca reduzidos a um palpite: quem consome recusa ou decide de forma explícita. Já `resolved_of(EntityReference)` leva de uma entidade de origem à sua entidade resolvida;
- **identidade de cada entidade resolvida**, para avaliar saídas construídas sobre elas: vem da avaliação de Entity Resolution (`IdentityEvaluation.identity_of_resolved_entity`, em [`evaluation`](../../evaluation/docs/entity_resolution.md)), porque só a avaliação tem a referência anotada. O run em si não conhece identidades físicas.

## Métricas (separadas, nenhum score composto)

`counts.json`: entidades de origem, conjuntos e pares candidatos, pares comparados e bloqueados, decisões por resultado, entidades resolvidas e fundidas, contradições, candidatos a divisão e, por canal, quantas comparações o mediram, tiveram o canal indisponível ou não o avaliaram. `distributions.json`: histograma do tamanho dos grupos de fusão, candidatos e entidades examinadas por entidade. `payload.json`: tamanho por arquivo e total. `runtime.json` (tempo e memória) só existe quando quem chamou mediu, e fica à parte de toda medida de qualidade.

## Debug (não contratual)

`ResolutionDebugLevel.NONE` (padrão) não escreve nada; `STANDARD` traça uma amostra (20) das decisões; `FULL` traça todas. Cada `debug/decisions/<decision-id>/` tem `decision.json`, `entity-a.json`, `entity-b.json`, `candidate-retrieval.json`, `merge-lineage.json` e um `<canal>-evidence.json` **só para os canais aplicáveis**. Nunca entra no inventário, então apagá-lo não invalida o run.

## O que não duplica

Nenhuma geometria, imagem, feature ou payload de upstream: as entidades de origem, as referências de geometria, as features e as representações são só referenciadas por identidade. A identidade de cada entidade de origem continua a de seu semantic map, e nenhuma delas é alterada.

## Limites

Só dados sintéticos: não há run de Semantic Mapping real, então nenhuma resolução real foi gravada. Os campos do manifest (contagens, papéis de política) são a primeira versão do schema (`0.1.0`). A validação real da linhagem contra um run de Semantic Mapping em disco é da integração da runtime; aqui a linhagem é derivada do manifest, que é um dataclass público.

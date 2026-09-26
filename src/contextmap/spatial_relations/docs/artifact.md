# SpatialRelationsRunArtifact

Um run de Spatial Relations é persistido como um diretório **imutável e autodescritivo**, para que a montagem do Context Map reutilize as relações sem recalcular nenhuma avaliação de par de entidades, e para que toda relação continue rastreável até as entidades resolvidas, as medições, os canais de evidência e a política de decisão. Abre sem NumPy, sem runtime de percepção e sem runtime de modelo.

## Onde é gravado

O escritor recebe **`output_dir`**, o diretório final do artifact. Não existe contador de runs, `runs.json` nem caminho `run-NNNN` calculado internamente: a identidade do run (`SpatialRelationsRunId`) é dada por quem chama. A gravação usa `AtomicRunDirectory`: um run interrompido nunca parece concluído, um run concluído nunca é alterado e um `output_dir` que já existe é recusado (reexecutar cria outro artifact). O layout de workspace da execução decide o caminho; este módulo não o conhece.

```text
<output_dir>/
├── manifest.json                    # linhagem, políticas, contagens, inventário com hash
├── README.md
├── outputs/                         # dados contratuais
│   ├── relations.jsonl              # Relation (todos os estados), em ordem canônica
│   ├── relation-evidence.jsonl      # RelationEvidence de todos os canais, por identidade
│   ├── relation-candidates.jsonl    # candidatos, exclusões, resumos de exclusões e predicados pulados
│   ├── relation-decisions.jsonl     # uma RelationDecision por relação
│   └── entity-relation-index.jsonl  # por entidade resolvida: relações como sujeito e como objeto
├── metrics/
│   ├── counts.json                  # contagens por estado, predicado, canal, regra, incerteza e exclusões
│   └── runtime.json                 # só quando quem chamou mediu
└── debug/                           # opcional; nunca contratual
    └── relations/<relation_id>.json
```

**Desvio da issue, seguindo os artifacts irmãos:** a linhagem, as políticas e a configuração efetiva ficam dentro de `manifest.json`, e não em `config.yaml`, `lineage.json`, `environment.json` e `events.jsonl` separados.

## Manifest

| Campo | Significado |
| --- | --- |
| `run_id`, `schema_version`, `created_at`, `code_version` | Identidade, versão do schema, instante UTC e revisão do código. |
| `lineage` | `RelationsRunLineage`: run de Entity Resolution selecionado (`entity_resolution_run_id`, `_schema_version`, `_artifact_digest`) e o mapa geométrico (`geometric_map_id`). É **derivada do manifest de Entity Resolution** por `lineage_from_resolution_manifest`, e o digest é o `resolution_artifact_digest` **do próprio Entity Resolution**, dono do artifact (identidade, versão do schema e hash de cada arquivo contratual, como nos artifacts irmãos): ele torna detectável uma mudança posterior do artifact upstream, sem duplicar a regra aqui. |
| `taxonomy_version` | Versão do vocabulário de predicados. |
| `policies` | A política de cada etapa com id, fingerprint e parâmetros: `candidate`, `geometric`, `contact` (quando há esse canal), `frame_conventions` (eixos declarados), `observation` e `decision`. |
| `counts` | Resumo: relações, evidência, candidatos e relações por estado. |
| `warnings`, `debug_level` | Avisos do run e o nível de debug pedido. |
| `file_inventory` | Cada arquivo contratual com tamanho e SHA-256; exclui o manifest, o README e `debug/`. |

O manifest não tem caminho absoluto nem segredo, então o run é **portável**: mover o diretório não o invalida.

## O que é validado na escrita

Antes do escritor, `RelationsRunPolicies` recusa (`ValueError`) já na construção uma política de candidatos cujo `proximity_radius_m` não cobre as tolerâncias dos avaliadores declarados, então um run com políticas incoerentes nunca chega a ser publicado ([coerência entre alcance e tolerâncias](candidates.md#coerência-entre-alcance-e-tolerâncias-dos-avaliadores)).

`SpatialRelationsRunWriter.write` recusa (`RelationsRunArtifactError`) antes de gravar qualquer coisa:

- relação sobre entidade de **outro** run de resolução que o selecionado na linhagem;
- candidatos ou evidência de outro **mapa geométrico** que o declarado, ou candidatos gerados sob outras convenções de frame ou outra política de candidatos;
- evidência de um canal **sem a política declarada**, ou com fingerprint diferente do declarado;
- evidência repetida ou que não é sobre um candidato do run;
- relação ou decisão que cita evidência que não foi entregue, ou relação decidida por uma política diferente da declarada;
- avaliadas e candidatos que não correspondem um a um.

## Leitura

`SpatialRelationsRunReader(run_dir)` só precisa do diretório do run:

- `iter_relations()`, `relation(id)`, `iter_evidence()`, `evidence(id)`, `evidence_of(relation)`, `iter_decisions()`, `decision(relation_id)`;
- `relation(id)` lê e decodifica **só o registro pedido** (`seek`/`read`), a partir dos deslocamentos das linhas de `relations.jsonl`, obtidos uma vez por leitor numa passada que lê a identidade de cada linha sem decodificar nenhuma relação (#601);
- `relations_of(entidade, as_subject=, as_object=)` usa o índice de entidades, carregado uma vez por leitor, e decodifica só as relações da entidade, na ordem canônica da tabela;
- `iter_relations()` lê a tabela linha a linha e não guarda as relações no leitor; `evidence(id)`, `evidence_of` e `decision` ainda carregam a própria tabela inteira na primeira consulta;
- `candidate_set()` reconstrói os candidatos, as exclusões com a razão, os resumos dos grupos de exclusões acima do teto e os predicados pulados;
- `read_table` e `read_record` só aceitam `outputs/` e `metrics/`: `debug/` nunca é uma fonte válida;
- `verify_integrity()` compara o inventário com o disco e detecta arquivo ausente, tamanho ou hash diferente;
- `validate_resolution(leitor_de_resolução)` confere a linhagem contra o `EntityResolutionRunReader` do run de resolução (identidade, versão do schema e digest) e que **toda entidade resolvida** que as relações citam existe nele (`resolved_entity`, sem carregar as demais); devolve os problemas, e vazio significa que tudo bate. Um run que não é o nomeado é reportado sozinho.

Toda decodificação reconstrói os contratos pelos construtores, então uma linha adulterada é recusada em vez de aceita. Como `relation(id)` só decodifica o registro pedido, uma linha adulterada de outra relação só é detectada quando for lida (ou por `verify_integrity()`); uma linha que nem é JSON válido é recusada já na passada dos deslocamentos.

Os deslocamentos não são persistidos: os runs gravados antes (entre eles o de `examples/v0.1.0/`) não teriam o índice e precisam continuar legíveis, então o leitor o deriva, sem mudar o layout nem a `schema_version`. Medido numa máquina de desenvolvimento, num run de 11 646 relações: a primeira `relation(id)` passou de 11 646 relações decodificadas (14,5 MiB retidos no leitor) para 1 (2,7 MiB, só os deslocamentos), e `relations_of` num leitor novo, de 456 ms e 11 646 decodificadas para 91 ms e as 270 da entidade. Consultar **todas** as relações uma a uma (como a validação `FULL` do Context Map) ficou mais lento, 0,42 s → 0,67 s, porque cada consulta abre o arquivo e decodifica o próprio registro em vez de achar a relação já decodificada em memória; para ler todas, `iter_relations()` é o caminho (0,49 s → 0,36 s).

## Entidades resolvidas de um run de Entity Resolution

`resolved_entity_geometries(resolvidas, source=..., policy=...)` produz, para cada entidade resolvida, o `EntityGeometry` que candidatos e avaliadores leem: o resumo espacial (estatísticas, diagnósticos e orientação opcional) do suporte **união** dos membros, com o mesmo algoritmo de Semantic Mapping. Ele confere o resultado com o que a resolução persistiu: as referências de geometria são as que Entity Resolution nomeou e os limites e o frame recalculados a partir do mapa são **exatamente** os que ela registrou; senão levanta `ValueError`, porque o mapa não é aquele em que a entidade foi resolvida. Uma relação nunca é medida sobre uma aproximação da geometria da entidade resolvida.

## Estados e rastro

`SUPPORTED`, `REJECTED` e `UNRESOLVED` são todos persistidos e permanecem distinguíveis (`metrics/counts.json` conta por estado e por predicado × estado). Toda relação lista a evidência em que se apoia, e a decisão diz quais registros decidiram e quais foram ignorados, então cada relação `SUPPORTED` rastreia até as entidades resolvidas exatas e as medições. Nada é copiado: as entidades são referenciadas por `ResolvedEntityReference` e a geometria só é identificada por digest.

## Debug

`RelationsRunDebugLevel.NONE` não grava nada além do contratual; `STANDARD` grava, para uma amostra de relações, a relação, a decisão e a evidência completa com todas as medições; `FULL` grava para todas. O conteúdo do `debug/` nunca entra no inventário, então removê-lo não invalida o run, e nenhum consumidor pode depender dele.

## Versões do schema

A `schema_version` escrita é **`0.2.0`**; o leitor abre `0.1.0` e `0.2.0` e recusa qualquer outra.

| Versão | O que muda |
| --- | --- |
| `0.1.0` | `relation-candidates.jsonl` lista **toda** exclusão de todo par enumerado; `counts.json` → `exclusions` é o número delas. |
| `0.2.0` | Um grupo `(predicado, razão)` com mais de 32 exclusões lista só as 32 mais próximas, dentro de um registro `exclusion_summary` (`predicate`, `reason`, `count`, `min_gap_m`, `max_gap_m`, `digest` e `nearest`, as listadas em ordem de distância) que resume todas ([teto de exclusões](candidates.md#teto-de-exclusões)); os grupos menores continuam como registros `exclusion`, byte a byte como em `0.1.0`. `counts.json` → `exclusions` conta todas, listadas ou não, e `unlisted_exclusions` as que não estão listadas. |

Um run `0.1.0` (como o de `examples/v0.1.0/`) é lido como um conjunto sem nenhum grupo resumido; nada mais difere entre os dois layouts, então essa é a única ramificação de compatibilidade. Um consumidor que precisa saber se uma exclusão ficou de fora do registro lê os resumos, como faz o avaliador de relações (`excluded_unlisted`).

## Limites conhecidos

- Uma única escrita em memória: as relações de um mapa cabem em memória, ao contrário das entidades de Semantic Mapping (não há escrita em fluxo).

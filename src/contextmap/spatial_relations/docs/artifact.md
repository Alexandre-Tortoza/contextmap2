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
│   ├── relation-candidates.jsonl    # conjunto de candidatos, exclusões e predicados pulados
│   ├── relation-decisions.jsonl     # uma RelationDecision por relação
│   └── entity-relation-index.jsonl  # por entidade resolvida: relações como sujeito e como objeto
├── metrics/
│   ├── counts.json                  # contagens por estado, predicado, canal, regra e incerteza
│   └── runtime.json                 # só quando quem chamou mediu
└── debug/                           # opcional; nunca contratual
    └── relations/<relation_id>.json
```

**Desvio da issue, seguindo os artifacts irmãos:** a linhagem, as políticas e a configuração efetiva ficam dentro de `manifest.json`, e não em `config.yaml`, `lineage.json`, `environment.json` e `events.jsonl` separados.

## Manifest

| Campo | Significado |
| --- | --- |
| `run_id`, `schema_version`, `created_at`, `code_version` | Identidade, versão do schema, instante UTC e revisão do código. |
| `lineage` | `RelationsRunLineage`: run de Entity Resolution selecionado (`entity_resolution_run_id`, `_schema_version`, `_artifact_digest`) e o mapa geométrico (`geometric_map_id`). O digest é calculado por Entity Resolution, para que uma mudança posterior do artifact upstream seja detectável. |
| `taxonomy_version` | Versão do vocabulário de predicados. |
| `policies` | A política de cada etapa com id, fingerprint e parâmetros: `candidate`, `geometric`, `contact` (quando há esse canal), `frame_conventions` (eixos declarados), `observation` e `decision`. |
| `counts` | Resumo: relações, evidência, candidatos e relações por estado. |
| `warnings`, `debug_level` | Avisos do run e o nível de debug pedido. |
| `file_inventory` | Cada arquivo contratual com tamanho e SHA-256; exclui o manifest, o README e `debug/`. |

O manifest não tem caminho absoluto nem segredo, então o run é **portável**: mover o diretório não o invalida.

## O que é validado na escrita

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
- `relations_of(entidade, as_subject=, as_object=)` usa o índice de entidades, sem varrer os pares;
- `candidate_set()` reconstrói os candidatos, as exclusões com a razão e os predicados pulados;
- `read_table` e `read_record` só aceitam `outputs/` e `metrics/`: `debug/` nunca é uma fonte válida;
- `verify_integrity()` compara o inventário com o disco e detecta arquivo ausente, tamanho ou hash diferente;
- `validate_references(entidades)` devolve as entidades citadas pelas relações que **não** estão no conjunto de entidades resolvidas do run de resolução (vazio significa que tudo resolve).

Toda decodificação reconstrói os contratos pelos construtores, então uma linha adulterada é recusada em vez de aceita.

## Estados e rastro

`SUPPORTED`, `REJECTED` e `UNRESOLVED` são todos persistidos e permanecem distinguíveis (`metrics/counts.json` conta por estado e por predicado × estado). Toda relação lista a evidência em que se apoia, e a decisão diz quais registros decidiram e quais foram ignorados, então cada relação `SUPPORTED` rastreia até as entidades resolvidas exatas e as medições. Nada é copiado: as entidades são referenciadas por `ResolvedEntityReference` e a geometria só é identificada por digest.

## Debug

`RelationsRunDebugLevel.NONE` não grava nada além do contratual; `STANDARD` grava, para uma amostra de relações, a relação, a decisão e a evidência completa com todas as medições; `FULL` grava para todas. O conteúdo do `debug/` nunca entra no inventário, então removê-lo não invalida o run, e nenhum consumidor pode depender dele.

## Limites conhecidos

- **Linhagem de Entity Resolution como entrada.** `RelationsRunLineage` recebe a versão de schema e o digest do artifact de resolução como valores; derivá-los do manifest de Entity Resolution e checar as referências contra o seu leitor precisa dos contratos `ResolvedEntity` e do leitor daquela milestone, ainda não disponíveis nesta branch. `validate_references` já aceita o conjunto de entidades resolvidas e serve de costura.
- Uma única escrita em memória: as relações de um mapa cabem em memória, ao contrário das entidades de Semantic Mapping (não há escrita em fluxo).

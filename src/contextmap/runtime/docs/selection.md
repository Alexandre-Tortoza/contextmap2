# Seleção de runs e lineage

A mesma sequência física pode ter vários runs de percepção, estimação de estado, representação ou fusão. O runtime **nunca combina em silêncio todos os resultados disponíveis**: uma execução parte de uma seleção explícita, resolvida e validada antes de qualquer estágio rodar.

Seleção só **escolhe entradas**. Ela não funde nem reinterpreta evidência.

## Gramática (`inputs.selections`)

Por estágio, em `inputs.selections.<estágio>`:

| Valor | Significado |
|---|---|
| `"seq-1"` | o artifact exato com esse id |
| `["perc-1", "perc-2"]` | vários runs distintos do mesmo estágio, mantidos como evidência separada |
| `"named:<nome>"` | a seleção nomeada `inputs.named.<nome>.<estágio>`, um conjunto coerente de ids exatos definido na configuração |
| `"latest"` | **opt-in explícito** ao run mais recente **compatível** com o restante da seleção; deve ser a única referência do estágio |

Um estágio que **não** está listado nunca é selecionado implicitamente; não existe "usar todos os runs". Uma seleção nomeada lista apenas ids exatos (não aceita `latest` nem outra seleção nomeada). Estágio desconhecido, seleção nomeada inexistente, lista vazia ou repetida e mistura de `latest` com ids são rejeitados na validação da configuração, com o caminho do problema.

```toml
[inputs]
sequence = "S1"                                   # expectativa explícita, conferida contra a linhagem

[inputs.named.canonical-v1]
ingestion = "seq-1"
state_estimation = "traj-1"

[inputs.selections]
ingestion = "named:canonical-v1"
state_estimation = "named:canonical-v1"
visual_perception = ["perc-1", "perc-2"]          # dois runs de inferência, evidência distinta
sensor_association = "latest"
```

## Catálogo

`resolve_selections(plan, inputs, catalog)` resolve a seleção contra um `ArtifactCatalog`, que devolve os runs disponíveis de um estágio **em ordem de índice de run** (`CatalogEntry.run_index`). Por isso `latest` é determinístico e reconstruível a partir dos índices de run das capabilities: independe da ordem de inserção, do nome e da data do arquivo. `StaticCatalog` mantém as entradas em memória.

Cada `CatalogEntry` traz o `ArtifactRef` exato (com hash de conteúdo) e a `Lineage` que o artifact **declara**.

## Linhagem e compatibilidade

Compatibilidade **nunca** é inferida do nome de arquivo nem do nome da sequência: só vale o que o artifact declara na linhagem.

| Campo da `Lineage` | Regra |
|---|---|
| `sequence` | identidade da sequência **física**; todos os runs selecionados que a declaram concordam |
| `selection` | seleção de observações; concordam |
| `calibration` | identidade da calibração; concordam |
| `schema_version` | precisa estar entre as versões suportadas do tipo do artifact (`supported_schemas`) |
| `upstream` | os ids exatos de que o artifact foi construído; se o estágio a montante também está selecionado, tem de ser exatamente esse run |

A cadeia de `upstream` cobre também a identidade do mapa geométrico: uma associação construída sobre `map-1` não combina com `map-9`. Uma identidade não declarada nunca é assumida igual a outra. `inputs.sequence`, quando informada, é uma asserção explícita conferida contra a linhagem.

Os runs repetidos de um mesmo estágio são checados do mesmo jeito: vários runs de inferência sobre as **mesmas observações** são aceitos e continuam distintos, cada um com sua identidade de conteúdo e a identidade física comum; runs de outra sequência são recusados.

## `latest` compatível

Os `latest` são resolvidos em ordem topológica, depois de todas as referências exatas e nomeadas, então cada um enxerga as escolhas anteriores. Ele toma o run **mais recente cuja linhagem é compatível** com a seleção corrente: um run mais recente de outra sequência, ou construído sobre um upstream diferente do selecionado, é ignorado. Sem run compatível, é um problema, não um fallback.

## Cardinalidade

Só uma entrada declarada `multiple` aceita vários runs do estágio de origem (hoje: a percepção e a associação consumidas por `sensor_association` e `semantic_fusion`). Selecionar vários runs para uma entrada que aceita exatamente um é um problema de preflight.

## Integração com o DAG

`plan.scope(targets=..., selections=resolved)` alimenta a execução com os artifacts selecionados (`provided` e `selections` são exclusivos). Todo problema da seleção chega ao `preflight()` junto com os demais, e uma seleção incompatível **falha antes de qualquer estágio rodar**.

`StageRequest.inputs` passa a ser, para cada entrada, uma **tupla de runs** (um run para uma entrada comum; vários, em ordem determinística, para uma entrada que aceita vários). O `ExecutionRecord` persiste em `execution.json`:

- `selections`: os ids exatos, o `origin` de cada um (`exact`, `named:<nome>` ou `latest`), o índice de run e a linhagem declarada, incluindo a identidade física comum;
- as entradas exatas de cada estágio, para que um run posterior exponha os ids dos artifacts de que partiu.

O reuso (ver [`reuse.md`](reuse.md)) combina os hashes de conteúdo de vários runs de uma entrada como um conjunto: o hash de um único run não muda, e a ordem dos runs não altera a chave.

## Arquivo de catálogo

`load_catalog(caminho)` lê um catálogo explícito (é o que a CLI usa em `--catalog`); nada é descoberto por varredura de diretório nem por nome:

```json
{
  "schema_version": "0.1.0",
  "entries": [
    {
      "artifact_id": "perc-1", "stage_id": "visual_perception",
      "contract": "PerceptionRunArtifact", "content_hash": "sha256:...",
      "run_index": 1,
      "lineage": {"sequence": "S1", "selection": null, "calibration": "cal-1",
                  "schema_version": "0.1.0", "upstream": {"ingestion": ["seq-1"]}}
    }
  ]
}
```

Entrada inválida, versão de schema diferente ou arquivo ilegível é um erro explícito com o índice da entrada.

## Lacunas conhecidas

- **Catálogo sobre os índices reais.** O protocolo `ArtifactCatalog` e o `StaticCatalog` estão prontos; a implementação que lê os índices de run e os manifests de cada capability (cujos campos de linhagem diferem) acompanha os executores reais, na validação end-to-end (#177). Sem ela, quem chama fornece o catálogo.
- **Seleção por sequência.** `inputs.sequence` é uma asserção; a escolha automática de um run de ingestão a partir dela não existe, de propósito: seria inferência.
- **Persistência da seleção só no registro de execução.** A seleção resolvida vai em `execution.json`; `plan.json` guarda apenas a topologia.

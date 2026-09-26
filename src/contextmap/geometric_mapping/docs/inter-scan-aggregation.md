# Agregação espacial inter-scan (derivada, com provenance)

Este documento descreve `src/contextmap/geometric_mapping/voxel_aggregation.py` e `voxel_aggregation_artifact.py`, e registra a decisão de contrato da issue #623. A medição que decide o que acontece depois está na issue #624 e no experimento [`experiments/geometric-aggregation-raw-vs-voxel-20260926/`](../../../../experiments/geometric-aggregation-raw-vs-voxel-20260926/README.md).

## Baseline: nada é agregado entre scans

O mapa canônico preserva cada ponto aceito como uma medição crua. O baseline é caracterizado pelo próprio artifact, sem medição adicional:

| Campo do `GeometricMapArtifact` | Valor no baseline | Significado |
|---|---|---|
| `manifest.aggregation_rule` | `null` | nenhum ponto é agregado |
| `metrics/mapping.json` → `reduction_ratio` | `1.0` | `point_count == source_point_count` |
| registro do payload → `count` | `1` em todo ponto | cada registro é uma medição, com o seu `source_point_index` |

A redundância entre scans sobrepostos permanece: o mapa cresce com os pontos acumulados e uma plataforma parada persiste duplicatas indefinidamente (limitação assumida em [`accumulation.md`](accumulation.md)).

## Dois níveis que não se confundem

| | `ScanVoxelPolicy` | `InterScanVoxelPolicy` |
|---|---|---|
| Onde roda | dentro do mapeamento, no `MapAccumulator` | offline, sobre um `GeometricMapArtifact` bruto já finalizado |
| O que une | pontos **do mesmo scan** | pontos de **scans diferentes** |
| Efeito no mapa canônico | substitui os pontos do scan no próprio `GeometricMap` | nenhum: o mapa bruto é só lido |
| Saída | `GeometryPoint` com `origin = AGGREGATED` | `VoxelAggregate`, em um artifact derivado separado |
| Regra declarada | `scan-voxel-centroid-<cell>m` | `inter-scan-voxel-centroid-<cell>m` |
| Entrada aceita | scans transformados | só um mapa bruto (`aggregation_rule = null`) |

A agregação inter-scan **recusa** um mapa já agregado por `ScanVoxelPolicy`: cada registro de entrada tem de ser exatamente uma medição, senão `point_count` deixaria de contar medições. A voxelização interna de modelos (PTv3/Vernata em `point_representation`) é outra coisa: é um detalhe de extração de features e nunca é reutilizada como contrato de mapa.

## Contrato

### Identidade da grade (`VoxelGridSpec`)

- `frame_id`: o frame do mapa. Uma grade em outro frame é recusada; frames nunca são reinterpretados.
- `origin_m`: o canto do voxel `(0, 0, 0)`.
- `cell_m`: a aresta do voxel, em metros.
- `indexing`: fixo em `floor-half-open`. O voxel `k` cobre `[origin + k·cell, origin + (k+1)·cell)` em cada eixo, ou seja, `k = floor((p − origin) / cell)`.

Uma chave só é aceita quando o quociente cabe exatamente na mantissa do float64 (`|k| < 2^53`). Fora disso a chave deixaria de ser exata, e a agregação é recusada em vez de arredondar em silêncio.

### Política versionada (`InterScanVoxelPolicy`)

`policy_id = inter-scan-voxel-centroid`, `policy_version = 0.1.0`. Nesta versão só a grade varia. Representante (centróide), estatísticas e granularidade da lineage são fixos. `fingerprint()` é o SHA-256 do registro completo (`to_record()`): mudar o frame, a origem ou a resolução muda o fingerprint (teste). O tamanho de bloco da leitura é um parâmetro de execução gravado em `config.json`, fora do fingerprint, porque não muda a representação além da tolerância documentada abaixo.

### Um agregado (`VoxelAggregate`)

| Campo | Semântica |
|---|---|
| `key` | chave inteira do voxel |
| `centroid_m` | o **representante**: a média dos pontos brutos do voxel, no frame do mapa |
| `offset_sum_m` | `Σ (p − anchor)`, onde `anchor = origin + key·cell` é o canto inferior do voxel |
| `minimum_m`, `maximum_m` | mínimo e máximo por eixo dos pontos brutos |
| `point_count` | **medições brutas** resumidas |
| `scan_count` | **scans** distintos (entradas do índice de origem) que contribuíram |
| `observation_count` | **observações físicas** distintas que contribuíram |
| `first_observed_at`, `last_observed_at` | aquisição do primeiro e do último scan contribuinte, no clock do mapa |
| `contributions` | um `(scan_ordinal, point_count)` por scan contribuinte |

As três contagens têm semânticas distintas e são calculadas de forma independente: `point_count` soma medições, `scan_count` conta ordinais de scan, `observation_count` conta `SourceObservationId` distintos pela tabela de scans. Em um único mapa bruto um scan é uma observação (o `MapAccumulator` recusa a mesma observação duas vezes), então as duas últimas coincidem ali. O contrato não assume essa bijeção. Duas observações da mesma superfície no mesmo voxel continuam sendo **duas contribuições físicas** (teste).

`offset_sum_m` é a estatística suficiente que a #623 chama de `sum_xyz`, só que expressa relativa ao canto do voxel. Somar offsets, que ficam dentro de um voxel, em vez de coordenadas absolutas limita o erro de arredondamento pelo tamanho do voxel, e não pela extensão do mapa. O centróide é sempre `anchor + offset_sum / point_count`; o reader confere essa igualdade bit a bit.

### Lineage agregado → contribuintes

A lineage persistida é **por scan**: `(scan_ordinal, point_count)`, 8 bytes por par `(voxel, scan)`, em vez de um índice por ponto bruto (8 bytes por ponto). Quais pontos são não precisa ser guardado: é função pura das coordenadas brutas e da grade. Por isso a lineage é compacta e **recuperável**:

- `VoxelAggregation.members_of(i, raw)` lê só os scans contribuintes e devolve as `GeometryReference` brutas exatas do agregado `i`, conferindo a contagem de cada scan;
- `VoxelAggregation.membership(raw)` devolve, para cada ponto bruto, o agregado que o resume;
- `VoxelAggregation.verify_lineage(raw)` re-deriva todas as contribuições a partir do mapa bruto e as compara com as gravadas.

A soma das contagens de contribuição é exatamente o número de pontos brutos, e nenhuma contribuição cruza voxel: `keys_of(minimum)` e `keys_of(maximum)` são a própria chave do agregado, o que é verificado ao abrir.

### Atualização incremental e invariância

`InterScanVoxelAggregator` recebe blocos em qualquer ordem e tamanho, e cada ponto bruto exatamente uma vez (ponto repetido ou faltante é recusado). O estado é uma tabela de estatísticas parciais por par `(voxel, scan)`, consolidada conforme cresce, então a memória segue os pares ocupados e não os pontos.

- **Exatos** sob qualquer ordem ou chunking: chaves, as três contagens, mínimo, máximo, tempos e lineage.
- **Dentro de tolerância**: somas e centróides, porque só a ordem das `n` adições muda. Cada offset é calculado igual em qualquer ordem e fica dentro de um voxel, então duas ordens diferem em no máximo `n²·ε·cell` na soma e `n·ε·cell` na média, mais um arredondamento de `anchor + média` de cada lado. A tolerância por eixo é `2·n·ε·cell + 2·ε·|centroid|`, com `ε = 2⁻⁵²` (`centroid_tolerance_m`), e `equivalence_problems` aplica exatamente essa regra.
- A mesma entrada com o mesmo chunking reproduz os mesmos bits (teste).

### O artifact derivado

```text
<dir>/                                  # separado do GeometricMapArtifact; nunca dentro dele
├── README.md
├── manifest.json                       # identidade, grade, fingerprint, contagens, inventário
├── lineage.json                        # identidade exata do artifact bruto: digest do inventário e hashes lidos
├── config.json                         # política completa, fingerprint e execution.block_points
├── environment.json
├── outputs/
│   ├── aggregates.bin                  # <3q3d3d3d3dqqIII: 148 bytes por agregado, por chave
│   ├── contributions.bin               # <II: 8 bytes por (agregado, scan), na mesma ordem
│   ├── aggregation.json                # mapa bruto resumido, pontos brutos, clock, formatos
│   └── map-metadata.json               # GeometricMap derivado
└── metrics/aggregation.json            # escala e custo de provenance
```

`aggregates.bin`, na ordem: chave, centróide, soma dos offsets, mínimo, máximo, primeiro e último ns, `point_count`, `scan_count`, `observation_count`. As contribuições de cada agregado são as `scan_count` entradas seguintes de `contributions.bin`, então não há offset gravado. A escrita segue `AtomicRunDirectory`: um diretório existente nunca é substituído, e escrever dentro do artifact bruto é recusado. Os bytes do artifact bruto permanecem idênticos (teste).

**Identidade do artifact bruto.** A lineage depende de `geometry.bin` (coordenadas), mas também de `source-index.jsonl` (observação, tempo, cadeia de transforms e intervalo de cada scan) e de `map-metadata.json`. `lineage.json` grava por isso duas coisas:

- `consumed_files`: o SHA-256 de cada um desses três arquivos;
- `contractual_inventory_digest`: o SHA-256 do inventário contratual inteiro do bruto (caminho, tamanho e hash de cada arquivo), que identifica o bruto exato mesmo que alguém o regrave com um manifest coerente.

O `GeometricMapArtifactReader` não verifica o inventário ao abrir. Por isso o writer confere o bruto contra o próprio inventário **antes** de derivar, e recusa qualquer arquivo alterado. `verify_integrity(source=raw)` confere de novo o inventário do bruto, compara a identidade gravada campo a campo e só então re-deriva a lineage. Um `source-index.jsonl` adulterado é recusado antes da derivação e detectado depois dela, inclusive quando o manifest do bruto foi reescrito para concordar com ele (testes). O índice derivado de limites do bruto não é recomputado nessas verificações: a derivação não o lê, e o inventário já cobre o hash de todo arquivo.

`metrics/aggregation.json` separa escala e custo de provenance de qualquer medida de qualidade: `reduction_ratio`, `aggregate_bytes`, `lineage_bytes`, `lineage_bytes_per_aggregate`, `explicit_point_lineage_bytes` (quanto custaria listar um índice por ponto bruto: comparável, nunca escrito) e distribuições de pontos, scans e observações por agregado.

### Como um consumidor lê os agregados

`VoxelAggregationArtifactReader.geometry()` devolve um `AggregatedGeometry`, que implementa `GeometryBlockSource`: um consumidor de blocos, como Sensor Association, roda **sem mudança** sobre os centróides. O `GeometricMap` derivado tem identidade própria (`<mapa bruto>--<run_id>`, nunca a do bruto), declara `inter-scan-voxel-centroid-<cell>m`, mantém a linhagem upstream do bruto (sequência, seleção, trajetória, calibração, janela de tempo) e usa o fingerprint da política como `configuration_fingerprint`. A linha `i` de um bloco é o agregado `i`. Um agregado **nunca** vira `GeometryPoint`: não tem uma observação de origem, uma coordenada de sensor nem uma cadeia de transform únicas, e fingir essas coisas colapsaria evidência em crença. Uma `GeometryReference` do mapa derivado se resolve pela lineage.

### Sem semântica

Nenhum label, claim, embedding, hipótese semântica ou identidade de instância existe no agregado (teste). A agregação é evidência sobre **onde** medições brutas estavam, não uma crença sobre **o que** existe ali.

## Decisão registrada (#623)

Decidido agora, antes do schema canônico:

1. **A geometria bruta continua sendo a evidência canônica.** `GeometricMap`, `MapAccumulator`, o `GeometricMapArtifact` e o runtime não mudam.
2. **A agregação inter-scan existe só como artifact derivado e separado**, com lineage para o artifact bruto exato (digest do inventário contratual e hashes dos arquivos lidos). Nenhum estágio canônico o consome.
3. **A semântica da provenance agregado → contribuintes está definida**: lineage por scan, recuperável por recomputação, verificada contra o bruto. Contagens de pontos, scans e observações ficam separadas.
4. **A grade (frame, origem, resolução, convenção de indexação) faz parte da identidade e do fingerprint.**
5. **Nenhuma voxelização interna de modelo ML é reutilizada como contrato de mapa.**

Pendente da medição real da #624: escolher entre (1) manter só a geometria bruta, (2) persistir bruto + representação derivada no pipeline ou (3) adotar a agregação como geometria canônica. A opção (3) só é possível se os contratos de provenance e dos consumidores forem satisfeitos explicitamente. Nenhuma resolução default é escolhida sem um limite de erro geométrico e de associação justificado pelo uso downstream. A issue de implementação canônica, com schema e migração, só é aberta depois dessa decisão.

### O que veio do V1, o que foi adaptado e o que é original

- **V1 (`contextual-3d-mapping`)**: voxel downsampling clássico, com um ponto no centróide de cada voxel ocupado e relação com os pontos originais. É a referência, não a especificação.
- **Adaptado**: o centróide por voxel como primeiro representante.
- **Original do ContextMap2**: representação derivada em vez de substituição; lineage por scan recuperável em vez de listas de pontos; contagens de pontos, scans e observações separadas; offsets relativos ao voxel com tolerância de invariância documentada; grade como identidade; consumo pelos consumidores existentes através de `GeometryBlockSource`, sem fabricar `GeometryPoint`.

### Alternativas ainda não implementadas

Representative point (uma medição real em vez da média), multi-resolução/LOD derivado do bruto e TSDF/occupancy/reconstrução de superfície. Ficam fora até a comparação com a primeira política: as três últimas exigem modelo de sensor, espaço livre ou reconstrução de superfície, além do problema atual.

## Limitações

- A agregação carrega o mapa agregado inteiro em memória. No benchmark sintético de 26,46 M pontos uniformes (945 scans), a 0,05 m foram ~109 s e ~10 GB de pico de RSS, e a 0,10 m ~90 s e ~10 GB. É o pior caso, sem pares repetidos dentro do scan. O número real está no relatório da #624.
- A consulta por caixa sobre os agregados é um filtro linear sobre os centróides; não há índice espacial derivado.
- Um único mapa bruto por agregação. Agregar vários runs exige definir identidade de scan entre mapas, o que ainda não tem requisito.

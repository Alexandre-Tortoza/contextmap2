# Geometria bruta × agregação voxel inter-scan (issue #624)

Este experimento mede o trade-off de uma representação voxel-agregada **derivada** da mesma geometria bruta, antes de qualquer promoção para o caminho canônico. A decisão de contrato e o que já está decidido estão na issue #623 e em [`inter-scan-aggregation.md`](../../src/contextmap/geometric_mapping/docs/inter-scan-aggregation.md). As métricas foram pré-registradas em [`voxel_aggregation.md`](../../src/contextmap/evaluation/docs/voxel_aggregation.md).

> **Estado: preparado, ainda não executado em dado real.** Os drivers, o protótipo e as métricas estão prontos e testados, mas este diretório ainda **não** tem relatório de dado real. Os artifacts reais (`outputs/`, fora do Git) não estavam disponíveis no ambiente em que foi preparado. Não há número real a citar até o run abaixo ser executado e registrado.

## Hipótese

Uma representação voxel-agregada derivada da mesma geometria bruta pode reduzir substancialmente o custo de armazenamento e de consulta sem introduzir erro geométrico ou perda de associação incompatíveis com os usos downstream, desde que a resolução seja escolhida por medição e a lineage dos contribuintes seja preservada.

**Pode ser rejeitada.** Se a perda geométrica, a instabilidade 2D→3D ou o custo de provenance superar o benefício, a geometria bruta continua sendo a representação persistida e a redução fica restrita a representações derivadas ou LOD.

## Braços

| Braço | Representação | Resolução | O que muda |
|---|---|---|---|
| **A** | `GeometricMapArtifact` bruto, como está no disco | — | baseline: lido, verificado e cronometrado, nunca reescrito |
| **B** | voxel centroid derivado | r1 = 0,05 m | resolução |
| **C** | voxel centroid derivado | r2 = 0,10 m | resolução |
| **D** (opcional) | voxel centroid derivado | r3 = 0,20 m | só se a curva qualidade × redução pedir |

Entre B, C e D **só a resolução muda**. Origem da grade `(0, 0, 0)` no frame do mapa, convenção `floor-half-open`, representante (centróide), estatísticas e lineage são idênticos. As resoluções amostram a curva; nenhuma delas é proposta como default. O braço A é o comportamento atual do V2 byte a byte: o mesmo artifact, sem cópia nem regravação, e `tests/geometric_mapping/test_geometric_mapping_voxel_aggregation_artifact.py` prova que derivar não muda nenhum byte dele.

## População

- Mapa bruto: o `GeometricMapArtifact` do run canônico `corridor-245-90s` (`corridor-02--ef41c6a06c0cf7dc774faef2f677cca1`, 26.630.193 pontos, 945 scans), o mesmo do experimento #564. Um run menor reproduzível entra como segundo mapa, se houver.
- Associação 2D→3D: a **janela congelada** de 15 imagens do `corridor-245-90s` (`camera_1_image_raw-000168` a `-000252`, passo 6) e as mesmas políticas de oclusão, pose e tolerância do [#564](../sensor-association-scaling-20260925/README.md), reusadas e não reinventadas. O candidate policy é `max_range_m = 20` em todos os braços.
- A fixture sintética de 1.000 pontos do viewer (`examples/v0.1.0/demo/`) **não** é evidência. Só foi usada como smoke do comando.

O relatório registra o commit, os ids do artifact de entrada (map, run, fingerprint de configuração, sequência, seleção, trajetória, calibração), a política de correção de movimento, o hardware e a configuração de cada braço (`report.json`).

## Métricas

As definições exatas estão em [`voxel_aggregation.md`](../../src/contextmap/evaluation/docs/voxel_aggregation.md).

- **Escala**: pontos de entrada e agregados de saída, `reduction_ratio`, bytes em disco, tempo de construção, pico de RSS (um processo novo por passo, `RUSAGE_CHILDREN`, o mesmo método do #181 e do #564), tempo de leitura integral, tempo de 16 consultas por caixa fixas e bytes de lineage por agregado, comparados com o custo de uma lista explícita de índices por ponto.
- **Fidelidade geométrica**: distância de cada ponto bruto ao representante do seu agregado (média, mediana, p95, máximo), a mesma distância por faixa de alcance ao sensor, e preservação do limite do mapa.
- **Contagens**: `point_count`, `scan_count` e `observation_count` reportados separadamente, por agregado e no artifact.
- **Estabilidade 2D→3D**: retenção de região, retenção de suporte por voxel, desacordo (1 − Jaccard), pontos de suporte por região e regiões finas (`|R| ≤ 20`).
- **Efeito downstream adicional**: deslocamento do centróide, mudança de extensão e IoU da caixa do suporte de cada região, que é a geometria que entidades e predicados de distância e contato recebem. Materializar entidades e relações sobre o agregado fica registrado como não medido, com a justificativa em `voxel_aggregation.md`.
- **Invariância**: lineage verificada contra o mapa bruto, e chunking de `1_000_000` × `4_099` pontos equivalente dentro da tolerância documentada, sobre o dado real.

## Como executar

Da raiz do repositório, com os artifacts reais em `outputs/`:

```bash
python experiments/geometric-aggregation-raw-vs-voxel-20260926/scripts/run_all.py \
    outputs/geometric-aggregation-20260926 \
    --raw-map outputs/corridor-245-90s/run-0001/geometric_mapping \
    --cells 0.05 0.10 \
    --sequence outputs/ingest-real/sequences/corridor-02/d8ef485b87af4452b224c9611ba0c621 \
    --trajectory outputs/corridor-245-90s/run-0001/state_estimation \
    --perception outputs/corridor-245-90s/visual_perception/workspace/corridor-02/run-0001/visual_perception
```

Sem `--sequence`, `--trajectory` e `--perception`, só a parte geométrica roda. Cada braço deixa em `<saída>/<braço>/` o seu artifact derivado (`voxel_aggregation/`), `arm-report.json`, o run de associação (`sensor_association/`) e `association-stability.json`. `report.json` reúne tudo. Os artifacts e configs de cada braço ficam retidos para reinspeção.

| Script | Faz |
|---|---|
| `scripts/run_all.py` | orquestra os braços, um processo novo por passo, e monta `report.json` |
| `scripts/_run_one.py` | roda um passo e mede o próprio wall time e o pico de RSS |
| `scripts/arm.py` | braço A: lê, verifica e cronometra o bruto. B/C/D: derivam o artifact, medem escala e fidelidade e verificam lineage e chunking |
| `scripts/associate.py` | Sensor Association sobre a geometria de um braço, na janela congelada |
| `scripts/compare.py` | estabilidade 2D→3D de um braço contra o braço A |

## Automação e CI

Nenhum workflow novo. O caminho inteiro já roda na CI existente (`pytest`) sobre dados pequenos e redistribuíveis gerados por código: o corredor simulado (`tests/evaluation/`) e a cadeia sintética de CI com o `SensorAssociationService` real sobre os agregados (`tests/end_to_end/test_voxel_aggregation_pairing.py`). Runs reais grandes ficam fora da CI padrão. O comando acima é o registro reproduzível.

## Regra de decisão

O experimento **não** escolhe um default universal. O relatório mostra a curva de trade-off e dá à #623 evidência para decidir entre:

1. manter só a geometria bruta;
2. persistir bruto + representação agregada derivada;
3. adotar a agregação como representação geométrica canônica, somente se os contratos de provenance e dos consumidores downstream forem satisfeitos explicitamente.

Qualquer resolução default exige justificar um limite de erro geométrico e de associação a partir do uso downstream, não só maximizar `reduction_ratio`.

## Ao registrar o run real

1. Rodar o comando acima em uma árvore limpa, no commit que será citado.
2. Copiar para este diretório `report.json`, os `arm-report.json` e os `association-stability.json` (sem caminhos pessoais), e escrever o `manifest.json` do bundle com o SHA-256 completo de cada arquivo executado, como pede [`docs/ARTIFACTS.md`](../../docs/ARTIFACTS.md#experiments). A partir daí, `tests/evaluation/test_experiment_bundles.py` passa a conferi-lo.
3. Resumir aqui as tabelas de escala, fidelidade e estabilidade, e referenciar o resultado na #623.

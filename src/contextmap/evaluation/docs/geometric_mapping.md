# Validação de Geometric Mapping

Este documento descreve `src/contextmap/evaluation/geometric_mapping.py`.

Um mapa não é aceito só porque foi persistido: seus pontos precisam ser numericamente válidos, a cadeia que os colocou precisa reproduzir suas coordenadas, scans que se sobrepõem precisam concordar entre si, e reabrir o artefato não pode mudar uma coordenada nem uma referência. O harness mede isso **antes** de a associação RGB↔3D fazer um erro de geometria parecer uma falha de percepção. Projeção de câmera e correção semântica ficam fora.

```text
GeometricMapArtifactReader ──► evaluate_geometric_mapping(reader, protocol, ...) ──► GeometricMappingEvaluationReport
                                                                                         └─► encode_geometric_mapping_report (JSON)
```

Nada aqui altera um artefato. Qualidade e custo ficam em seções separadas.

## O protocolo não tem valores padrão

`GeometricMappingProtocol` reúne **todas** as decisões de amostragem e todos os limiares, fornecidos pelo perfil de referência; nenhum campo científico tem padrão, então nenhum limiar é herdado de um corredor ou de um dataset. Os campos opcionais (`max_plausible_range_m`, `expected_bounds`) só se aplicam quando informados.

## Camadas do relatório

| Seção | O que mede |
| --- | --- |
| `structure` | contagens, finitude das coordenadas de mapa e de origem, frames explícitos (nenhum ponto fora do frame do mapa), medições cruas vs. agregadas, linhagens que conectam o frame do mapa ao frame do sensor, pontos descartados por não serem finitos e os problemas de integridade do artefato (inventário e índice derivado) |
| `transform_trace` | para uma amostra de pontos espalhada pelo mapa, o erro de **composição** (a cadeia reproduz a coordenada de mapa gravada?) e o erro de **round trip** (a cadeia inversa devolve a coordenada de origem?), contra uma tolerância do protocolo, com a referência da pose e da calibração de cada ponto |
| `expected_points` | pontos de origem cuja posição global é conhecida **independentemente do mapeamento** (fixtures ou referência) contra a coordenada produzida, com o erro numérico |
| `density` | ocupação por voxels (tamanho do protocolo): pontos por voxel ocupado e por scan |
| `sensor_range` | distância dos pontos ao sensor, a partir das coordenadas originais, e quantos passam do alcance plausível do perfil |
| `bounds` | limites do mapa e quantos pontos ficam fora da caixa esperada pelo perfil |
| `overlap` | concordância entre scans temporalmente adjacentes (ver abaixo), inclusive dividida pelo estado de correção de movimento |
| `reference` | comparação com uma nuvem **declarada** referência, sem alinhamento |
| `cost` | tamanho, tempo e memória de pico registrados pelo run; separado de toda medida de qualidade |

Os contadores por origem e as checagens de finitude e de frame cobrem os pontos **verificados** (um a cada `structure_point_stride`); `density`, `sensor_range` e `bounds` usam os mesmos pontos. Com o passo 1 verificam-se todos.

### Concordância entre scans (`overlap`)

Para pares `(i, i + lag)` espalhados pelo mapa, amostra-se pontos do scan anterior e mede-se a distância **ponto-plano** de cada um a um plano ajustado pelos `plane_neighbour_count` pontos mais próximos do scan posterior. Um ponto sem vizinhança plana (borda, canto, `max_plane_curvature`) ou com vizinhos além do raio de correspondência não tem contraparte e não contribui com resíduo. `ghost_residual_m` marca a fração de resíduos grandes como **indicador** de superfície duplicada ou fantasma, não como prova.

A distância ponto-ponto ao vizinho mais próximo não serve para LiDAR de poucos anéis: ela é dominada pelo espaçamento da amostragem (a resolução vertical de um Velodyne de 16 anéis é de dezenas de centímetros a 10 m) e não mudou nem para um erro de montagem de 10° (medida real abaixo). O resíduo ponto-plano não depende de quão esparsa é a superfície.

**Limitações de sensibilidade.** O resíduo é cego a deslizar ao longo de uma superfície. Um erro que desloca todos os scans do mesmo jeito, ou que aponta ao longo do corredor (um erro de tempo com velocidade constante), não aparece aqui; os pontos esperados e uma referência confiável o pegam. Ler o resíduo **contra outro run da mesma sequência**, nunca como acurácia absoluta. Scans não corrigidos para o movimento carregam a distorção do movimento durante a varredura, que entra no resíduo.

## Referência de geometria

`ReferenceGeometry` exige a identidade do perfil que a declara, o papel (`EVALUATION_REFERENCE` ou `UNVERIFIED_PRODUCT`) e o frame. **Um `.pcd` nunca é referência pelo nome ou pelo formato**: só o papel declarado conta, e `UNVERIFIED_PRODUCT` é recusado. A nuvem precisa já estar no frame do mapa (outro frame é recusado, nenhum é assumido alinhado) e **nenhum alinhamento é aplicado**: o relatório registra `alignment: "none"` porque alinhar esconderia exatamente o erro de pose ou de frame que se mede. A exatidão é a distância de cada amostra do mapa ao ponto de referência mais próximo, e a completude a fração da referência com um ponto do mapa dentro do raio. As nuvens são amostradas (até 5 000 pontos) para a busca ser tratável, e as contagens amostradas são reportadas.

## Reprodutibilidade e persistência

- `evaluate_round_trip(expected, reopened, probe_boxes)` compara ponto a ponto um mapa construído em memória com o mesmo mapa reaberto do artefato: coordenadas, identidade, linhagem e proveniência, os metadados do mapa, os limites e o resultado de caixas de consulta. Deriva de coordenada ou de referência é **contada**, nunca ignorada.
- `compare_contractual_inventories(first, second)` compara os hashes dos arquivos contratuais de dois runs: entradas e configuração idênticas reproduzem os mesmos arquivos (o `run_id` e a data de criação não entram no inventário).
- O `debug/` nunca é lido: a avaliação só usa `outputs/`, `metrics/` e os registros contratuais.

## Comparação controlada

`compare_geometric_mapping_reports` alinha os números principais de dois ou mais runs e exige que **só o mapeamento** varie: rejeita diferença de versão do avaliador, sequência, seleção, calibração ou protocolo. Cada entrada preserva o `run_id`, a trajetória e o fingerprint de configuração. Serve para ablações de uma variável por vez (política de lookup, agregação, trajetória).

## Fixtures sintéticas e regressões (CI)

Um corredor sintético (duas paredes e um piso) é observado de dez poses de uma plataforma que se move, guina e desvia. **Cada scan é simulado a partir da verdade**, então um mapeamento correto põe o mesmo ponto do mundo na mesma coordenada de mapa em todos os scans. Os testes garantem:

- pontos conhecidos caem na coordenada esperada com erro abaixo de 1e-9 m, e a cadeia inversa devolve a origem;
- **erros comuns de direção e de tempo falham**: extrinsecos com valores invertidos, poses invertidas e um erro de tempo, todos com erro absoluto nos pontos esperados; os dois primeiros também quebram a concordância entre scans;
- um erro de tempo que desloca todos os scans do mesmo jeito é visto primeiro pelos pontos esperados, não pela concordância (a limitação acima, testada);
- adulteração da cadeia gravada, do índice de origem ou do índice derivado é detectada;
- referências são estáveis (mesmos bytes e mesmo inventário), limites e persistência sobrevivem a fechar e reabrir, e o protocolo recusa valores impossíveis e não tem padrões.

## Execução real de referência (ad hoc)

Não há trajetória FAST-LIO real nesta milestone; a execução usa a trajetória de referência do dataset como entrada `ExternalPose` (corpo = IMU, extrínseco `laser_to_imu` do próprio dataset, mesmo clock de cabeçalho dos scans). Ela **não** é uma verdade absoluta do mapa: o que se mede é a consistência interna e a numérica. Dois trechos de `corridor-02` (120 scans cada), lookup interpolado com tolerância de 400 ms:

| | Trecho reto (170–182 s, ≈ 1,4 m/s) | Trecho com curva (366–378 s, 141° em 12 s) |
| --- | --- | --- |
| scans aceitos / recusados | 106 / 14 (lacunas de 403 ms na referência) | 120 / 0 |
| pontos persistidos | 3 053 550 (195 MB) | 2 136 127 (137 MB) |
| montar (transformar + acumular + gravar) | 0,6 s | 0,5 s |
| integridade | sem problemas | sem problemas |
| erro de composição / round trip | 2e-16 m / 1e-14 m | 2e-14 m / 5e-14 m |
| resíduo ponto-plano, lag 5 (mediana / p95) | 1,4 cm / 15,5 cm | 4,2 cm / 19,8 cm |

Sensibilidade do resíduo a erros **injetados** (mediana com lag 5; sem erro entre parênteses):

| Erro injetado | Reto (1,4 cm) | Curva (4,2 cm) |
| --- | --- | --- |
| montagem +1° em yaw | 2,1 cm | 4,3 cm |
| montagem +3° em yaw | 4,3 cm | 4,4 cm |
| montagem +10° em yaw | 14,2 cm | 5,0 cm |
| montagem +0,3 m em x | 1,9 cm | 4,8 cm |
| tempo +100 ms | 1,6 cm | 6,7 cm |
| tempo +250 ms | 2,0 cm | 9,6 cm |
| tempo −250 ms | 1,7 cm | 7,7 cm |

Leitura: o trecho **reto** revela erros de montagem (as paredes laterais enviesam) e quase não vê erros de tempo; a **curva** revela erros de tempo e vê pouco os de montagem. Os dois trechos são complementares e, ainda assim, nenhum pega um erro pequeno de montagem (1–3°) na curva: a métrica interna é um indicador, não um substituto de referência. O resíduo nominal (mediana de 1 a 4 cm, p95 de 16 a 20 cm) reúne ruído do sensor, imprecisão da referência de pose e a distorção de movimento dos scans não corrigidos (na curva o giro é de ≈ 12°/s, ≈ 1,2° por varredura de 0,1 s, ≈ 20 cm a 10 m: ordem de grandeza compatível com o p95); esta execução não separa as três contribuições. O campo `time` por ponto do payload permitirá medir o ganho de um deskew real.

O `voxel-centroid` de 5 cm por scan reduz o trecho reto de 3,05 M para 1,03 M pontos (razão 0,34) e o resíduo mediano vai de 1,4 para 1,8 cm.

Avaliar o trecho reto (3 milhões de pontos) leva ≈ 31 s com `structure_point_stride=1` e ≈ 18 s com `structure_point_stride=20`; o restante é a integridade do artefato (hash e índice derivado) e a concordância entre scans.

## Testes com o dataset real

`test_a_real_segment_maps_persists_and_validates_end_to_end` roda um trecho curto de `corridor-02` de ponta a ponta e é **pulado quando o dataset não existe** (ele não é versionado; `CONTEXTMAP_CORRIDOR02_DIR` aponta para outro diretório). Não afirma números científicos, só integridade, finitude, cadeia e a presença da concordância.

## Limitações

- O harness não tem CLI; a integração pelo `runtime` vem na milestone de Runtime.
- O custo de verificação é linear no número de pontos (Python puro por ponto); `structure_point_stride` evita ler os pontos que pula.
- Nenhuma referência de geometria confiável está declarada para `corridor-02` (`corridor-02.pcd` não tem papel nem frame declarados), então a comparação com referência foi exercitada só com fixtures sintéticas.
- Uma referência real do FAST-LIO e a medida do ganho de um deskew dependem de trabalho fora desta milestone.

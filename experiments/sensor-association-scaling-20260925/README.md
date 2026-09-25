# Escala e equivalência do Sensor Association (issue #564)

Este experimento mede as duas otimizações de Sensor Association — seleção de candidatos por
frame (#562) e persistência em streaming (#563) — **separadamente**, sobre os mesmos artifacts
reais congelados, e verifica que elas não mudam as associações 2D↔3D válidas.

O motivo de medir as duas juntas é que fixtures pequenas escondiam a falha: o perfil real do
#181 mostrou `sensor_association` com **24,96 GB de pico** e 177,8 s de wall time no mapa de
26,63 M pontos, o maior consumo de memória de qualquer estágio só-CPU do pipeline. O motivo de
medir a equivalência junto com o recurso é que culling pode **descartar geometria válida em
silêncio** e ainda produzir um mapa plausível: só a comparação por identidade persistente
distingue uma otimização de uma regressão científica.

## Braços

Um braço é um par (política de candidatos, modo de persistência) executado sobre os mesmos
artifacts upstream e a mesma janela congelada de frames, em um diretório de saída próprio.
Nenhum braço toca um artifact existente.

| Braço | Candidatos | Persistência | Revisão | Isola |
|---|---|---|---|---|
| **A** | mapa inteiro | batch (retém todos os frames) | `40b1820` | o baseline |
| **D** | mapa inteiro | streaming | `c8bf518` | **#563** (A → D) |
| **C** | esfera de 20 m | streaming | `c8bf518` | **#562** (D → C) |

O braço "culling + batch" que o #564 lista **não existe**: a retenção em batch foi removida pelo
#563 e não foi ressuscitada só para medir. `A → D` isola o streaming e `D → C` isola o culling,
o que dá atribuição independente completa com três braços em vez de quatro.

O braço A roda de um worktree em `40b1820` com `PYTHONPATH` apontando para o `src/` daquela
revisão. `scripts/arm.py` não recebe um flag de persistência: ele detecta qual API a revisão
tem (`writer.finalize(outcome)` versus `writer.transaction()` + `sink=`) e reporta o que usou,
por isso o mesmo arquivo serve aos três braços.

## População

A **janela congelada do run canônico `corridor-245-90s`**: as mesmas 15 imagens físicas
(`camera_1_image_raw-000168` a `-000252`, passo 6) que
`outputs/corridor-245-90s/run-0001/sensor_association` associou — 11 aceitas e 4 rejeitadas
pela política de pose. Reusada, não reinventada.

| Entrada | Identidade |
|---|---|
| Sequência (bag real) | `d8ef485b87af4452b224c9611ba0c621` |
| Trajetória | `8f0bb680be8fc76d6f2bc7c7f983f279` |
| Mapa geométrico | `corridor-02--ef41c6a06c0cf7dc774faef2f677cca1`, **26.630.193 pontos**, 945 scans |
| Run de percepção | `vp-sam2-corridor245`, 360 resultados |

Políticas idênticas às que o run canônico registrou no seu manifest: oclusão
`cell_size_px=4, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02`;
pose interpolada com `max_interpolation_gap_ns=450 ms`; tolerâncias
`max_pose_time_delta_ns=250 ms, max_map_window_offset_ns=1 s`.

## Por que `max_range_m = 20`

O valor não foi copiado de repositório antigo; ele vem de medir o próprio run. Percorrendo
`outputs/geometry-support.u32` do run canônico e calculando a distância de cada ponto associado
ao centro óptico interpolado do seu frame:

| | alcance |
|---|---|
| p50 | 1,42 m |
| p95 | 2,13 m |
| p99 | 2,71 m |
| p99,9 | 4,51 m |
| **máximo** | **5,49 m** |
| acima de 8 m | **0 pontos** (de 172.219) |

A razão é física, não estatística: é um corredor fechado, e a oclusão já descarta tudo atrás da
primeira superfície muito antes do alcance. 20 m dá margem de **3,6×** sobre o máximo observado
e corta os candidatos para ~22% do mapa **nesta janela**.

> Os ~22% são específicos desta janela, que caiu numa parte pouco densa do percurso. Sobre a
> trajetória inteira a fração mediana é **35%** e a máxima **48%** — ver
> [`full-sequence.md`](full-sequence.md). O ganho do culling sobre a sequência é ~2,9×, não os
> ~4,5× que a janela sugere. (A medição usa a translação do corpo como centro
óptico; o extrínseco body→câmera é de poucos centímetros, o que não muda um limite arredondado
ao metro.)

## Recursos, por braço

Máquina: 16 CPUs lógicas, **39,1 GiB de RAM**, RTX 3060 8 GB, Linux 6.x, Python 3.12.14.
Wall time e peak RSS medidos com `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` em um processo
pai novo por braço — obrigatório, porque `RUSAGE_CHILDREN` é um máximo corrente sobre todos os
filhos que um processo já gerou, e medir vários braços de um pai só vazaria o pico de um no
outro. Mesmo método do #181, para os números serem comparáveis com ele.

| Braço | Wall time | Peak RSS | Frames | Rejeitados | Observações | Integridade |
|---|---|---|---|---|---|---|
| **A** mapa inteiro + batch | 274,7 s | **22.939 MB** | 11 | 4 | 307 | limpa |
| **D** mapa inteiro + streaming | 120,1 s | 6.629 MB | 11 | 4 | 307 | limpa |
| **C** esfera 20 m + streaming | **33,2 s** | **2.307 MB** | 11 | 4 | 307 | limpa |

### Atribuição independente

| Mudança | Memória | Tempo |
|---|---|---|
| **#563** streaming (A → D) | 22.939 → 6.629 MB (**3,5×**) | 274,7 → 120,1 s (**2,3×**) |
| **#562** culling (D → C) | 6.629 → 2.307 MB (**2,9×**) | 120,1 → 33,2 s (**3,6×**) |
| **Combinado** (A → C) | **9,9× menos memória** | **8,3× mais rápido** |

O ganho de tempo do streaming não era esperado e vale registrar: ele não reduz trabalho
aritmético algum. A explicação provável é pressão de alocador e de page fault — o braço A
mantém ~1,2 GB de arrays vivos por frame, então cada novo frame aloca contra um heap que só
cresce. **Isto é hipótese, não medição**: nenhuma contagem de page fault ou de `mmap` foi
coletada para confirmá-la.

## Equivalência científica

Comparada por **identidade persistente de geometria**, nunca por linha de array. Índices locais
de candidatos não são identidades — a coincidência entre os dois era exatamente o acoplamento
que o #562 removeu.

Os braços foram escritos por **schemas diferentes** (`0.1.0` no braço A, antes do schema de
candidatos; `0.2.0` nos outros), então `scripts/equivalence.py` lê os arquivos persistidos
diretamente em vez de pelo reader atual, que recusaria o manifest antigo. Os formatos que ele lê
são os mesmos nas duas versões: `observation-index.jsonl` localiza cada observação em
`spatial-observations.jsonl`, cujo `support` nomeia uma corrida de `uint32` little-endian em
`geometry-support.u32`. No `0.1.0` esses índices eram linhas de candidatos sobre o mapa inteiro,
que para um braço de mapa inteiro **são** exatamente os índices globais — e é essa coincidência
que o #562 eliminou.

| Comparação | Veredito | Suporte geométrico idêntico |
|---|---|---|
| A vs D | **`equivalent`** (0 findings) | **sim** |
| D vs C | `declared-difference` | **sim** |
| A vs C | `declared-difference` | **sim** |

**As 307 observações espaciais têm suporte geométrico idêntico nos três braços**, observação por
observação, referência por referência.

### O que o culling mudou, e só isso

Por frame, com a população avaliada caindo de 26.630.193 para ~5,5–6,0 M (20,6%–22,4%):

| Frame | avaliados (D) | avaliados (C) | fração | visíveis (D) | visíveis (C) | associados (D) | associados (C) |
|---|---|---|---|---|---|---|---|
| `-000168` | 26.630.193 | 5.471.837 | 20,6% | 6.231 | **6.231** | 3.822 | **3.822** |
| `-000174` | 26.630.193 | 5.509.496 | 20,7% | 7.160 | **7.160** | 4.118 | **4.118** |
| `-000180` | 26.630.193 | 5.572.111 | 20,9% | 6.732 | **6.732** | 5.197 | **5.197** |
| `-000186` | 26.630.193 | 5.628.721 | 21,1% | 7.514 | **7.514** | 5.329 | **5.329** |
| `-000192` | 26.630.193 | 5.681.354 | 21,3% | 6.657 | **6.657** | 5.295 | **5.295** |
| `-000198` | 26.630.193 | 5.734.291 | 21,5% | 8.370 | **8.370** | 5.206 | **5.206** |
| `-000216` | 26.630.193 | 5.885.234 | 22,1% | 37.631 | **37.631** | 36.310 | **36.310** |
| `-000222` | 26.630.193 | 5.951.214 | 22,4% | 30.382 | **30.382** | 29.150 | **29.150** |
| `-000228` | 26.630.193 | 5.965.723 | 22,4% | 28.472 | **28.472** | 25.300 | **25.300** |
| `-000234` | 26.630.193 | 5.966.091 | 22,4% | 30.299 | **30.299** | 26.858 | **26.858** |
| `-000252` | 26.630.193 | 5.963.962 | 22,4% | 27.710 | **27.710** | 25.634 | **25.634** |

**As contagens de visíveis e de associados são idênticas em todos os 11 frames.** As únicas 11
diferenças encontradas (uma por frame) são a contagem de `occluded`: **24,6 M** avaliando o mapa
inteiro contra **3,5 M** avaliando os candidatos. Todas classificadas como `declared`.

Isso é a confirmação empírica do invariante que a esfera compra. A região de candidatos é uma
**esfera centrada no centro óptico**, nunca a caixa alinhada aos eixos com que a consulta é
expressa: todo elemento excluído está estritamente **mais longe** da câmera que todo elemento
retido, então um excluído nunca poderia ter sido o suporte de profundidade mais próximo de um
retido. No mapa real, o corte a 20 m removeu **apenas geometria ocluída** — nenhum ponto visível,
nenhum ponto associado, nenhuma referência de suporte. Uma caixa não teria essa propriedade (um
ponto logo fora de uma face está mais perto que um dentro de um canto), e é por isso que a caixa
só decide o que é **lido**.

## Gates de complexidade em CI

Medição de hardware fica neste relatório; CI usa contadores, invariantes de retenção e
envelopes largos, como o #564 pede. `tests/sensor_association/test_association_scaling.py`:

- crescer o mapa global de 4 para 4.004 pontos com a cena local **fixa** não muda a população
  avaliada nem a forma dos arrays por frame — a alocação é `O(candidatos)`, não `O(mapa)`;
- a fração de candidatos cai com o crescimento do mapa, e o braço de mapa inteiro cresce com ele,
  que é a falha sendo corrigida;
- **exatamente uma `VisibilityResolution` viva** enquanto cada frame é aceito, para 1, 3 e 5
  frames, com e sem persistência: nada de `0..K-1` sobrevive;
- o `SensorAssociationOutcome` tem o mesmo tamanho para 1 e para 5 frames;
- equivalência de artifact entre braços: mesmos ids de observação, mesmo suporte geométrico,
  mesma qualidade, mesmas regiões por ponto persistente, mesmas amostras densas, mesma linhagem;
- a única diferença declarada é a população avaliada, e `in_support` é idêntico.

`tests/sensor_association/test_candidate_equivalence.py` cobre o mesmo em nível de projeção para
**pinhole, fisheye e MEI**, incluindo bordas e cantos da imagem, o ponto exatamente no limite do
alcance, e a propriedade "todo descartado está mais longe que todo retido".

## Reprodutibilidade

| | |
|---|---|
| Commit do baseline | `40b1820` |
| Commit otimizado | `c8bf518` |
| Seleção | janela congelada de 15 frames do `corridor-245-90s` (acima) |
| Políticas | idênticas ao manifest do run canônico (acima) |
| Scripts | `scripts/arm.py`, `scripts/_run_one.py`, `scripts/run_all.py`, `scripts/equivalence.py` |
| Métricas brutas | `report.json`, `equivalence-A-vs-D.json`, `equivalence-D-vs-C.json`, `equivalence-A-vs-C.json` |

Os três braços publicaram artifacts com `verify_integrity()` limpo. Nenhum braço falhou e
nenhum atingiu OOM nesta janela; o braço A a 22,9 GB é **59% da RAM da máquina**, reportado como
risco de margem observado, não como falha induzida.

## Sequência completa

Ver [`full-sequence.md`](full-sequence.md).

# Viabilidade de sequência completa (issue #564)

A janela congelada de 15 frames responde à equivalência e à atribuição, mas não responde a
"cabe rodar a sequência inteira?". Esta parte mede passagens longas sobre os **mesmos** artifacts
reais e reporta o que resta de gargalo, honestamente.

> **Uma correção.** A primeira versão deste documento atribuía o pico do braço C-full a
> "páginas do mapa + o maior frame". Isso estava **errado**: a review apontou que o executor —
> e o driver deste experimento — materializavam todos os payloads de imagem antes de começar, e
> a medição direta deu **2.192,9 MB** para as 2.495 imagens do bag (900 KB cada, `bgr8`
> 640×480), um terço do pico. Depois de gerar os frames preguiçosamente (`b654762`), os braços
> longos foram remedidos. Os números abaixo são os remedidos.

## O que foi medido

Máquina: 16 CPUs lógicas, **39,1 GiB de RAM**, RTX 3060 8 GB, Python 3.12.14. Um processo pai
novo por braço, `RUSAGE_CHILDREN.ru_maxrss` — obrigatório, porque é um máximo corrente sobre
todos os filhos que um processo já gerou. Mesmo método do #181.

| Braço | Candidatos | Frames sel. | Assoc. | Rej. | Observações | Wall time | **s/frame** | Peak RSS |
|---|---|---|---|---|---|---|---|---|
| **C-full** | esfera 20 m | 360 | 350 | 10 | 7.935 | 1.357,1 s (22,6 min) | **3,88** | **4.340 MB** |
| **D-partial** | mapa inteiro | 30 | 26 | 4 | 730 | 283,9 s | **10,89** | 6.547 MB |

Ambos com `verify_integrity()` limpo. Sobre a mesma trajetória, o culling a 20 m é **2,81× mais
rápido** por frame que avaliar o mapa inteiro.

### O efeito de soltar a entrada, isolado

| Braço | imagens carregadas | Peak antes | Peak depois | Δ |
|---|---|---|---|---|
| **C-full** | 2.495 | 6.592,2 MB | **4.339,8 MB** | **−2.252,4 MB** |
| **D-partial** | 2.495 | 8.758,9 MB | **6.547,1 MB** | **−2.211,8 MB** |
| D (janela) | 15 | 6.628,7 MB | 6.545,9 MB | −82,8 MB |
| C (janela) | 15 | 2.306,7 MB | 2.293,4 MB | −13,3 MB |

A queda nos braços longos casa com o payload medido (2.192,9 MB) **dentro de 3%**, e os braços
de janela quase não se movem porque só carregavam 15 imagens. A atribuição fica fechada por
medição, não por aritmética.

## O pico não escala com o número de frames

| | frames | máx. candidatos em um frame | Peak RSS |
|---|---|---|---|
| Janela (braço C) | 11 | 5.966.091 | 2.293 MB |
| Sequência (braço C-full) | **350** (31,8×) | 12.822.696 (2,15×) | **4.340 MB** (1,89×) |

Se o pico ainda escalasse com o número de frames, 350 frames custariam ~31,8× os 2.293 MB da
janela, isto é **~73 GB** — quase o dobro da RAM da máquina. Ele custou 1,89×, e esse fator fica
**abaixo** do crescimento do maior frame isolado (2,15×). O modelo do #563 se sustenta:

```
pico  =  páginas do mapa residentes (mmap, ~1,7 GB, recuperáveis)
       + o maior frame de candidatos/projeção/visibilidade (12,8 M pontos)
       + um AssociationFrameInput
       + buffers do writer (pequenos)
```

A decomposição dos ~2,6 GB restantes em ~206 B por candidato é **aritmética sobre tamanhos de
array**, não medição por componente: nenhum contador de page fault foi coletado.

O total **não** é estritamente constante no número de frames, e não se deve afirmar `O(1)`: a
transação acumula um registro de tempo por frame (quatro primitivos, ~567 B) porque o #562 pede
os tempos por frame e `metrics/runtime.json` é um documento único, e o serviço guarda o conjunto
de observações já vistas (~66 B por frame) para recusar um frame repetido em um único passe. São
**~1,9 MB em 3.096 frames**, escalares e não estado perceptual ou geométrico — quatro ordens de
magnitude abaixo do pico. O termo linear existe, é pequeno e está nomeado.

## Um resultado que corrige a manchete da janela

A janela de 15 frames tinha 20,6%–22,4% de candidatos. Sobre a trajetória inteira:

| | fração de candidatos | pontos |
|---|---|---|
| mínimo | 20,6% | 5.471.837 |
| **p50** | **35,0%** | 9.318.524 |
| p95 | 47,6% | 12.676.116 |
| **máximo** | **48,2%** | 12.822.696 |

A janela caiu numa parte pouco densa do percurso. Onde o robô revisita o mesmo corredor, o mapa
acumulado fica localmente muito mais denso e um recorte de 20 m alcança quase metade dele. **O
ganho real do culling sobre a sequência é ~2,9× em candidatos (1/0,35), não os ~4,5× que a janela
sugeria.** Reportar só o número da janela seria selecionar o caso favorável.

Isso não afeta a equivalência: o culling continua removendo apenas geometria ocluída, e para esta
câmera (MEI, `RAY_RANGE`) a preservação é exata — ver `README.md`.

## Onde o tempo vai, no braço com culling

| Etapa | Total | Fração | p50 por frame |
|---|---|---|---|
| Consulta de candidatos | 402,2 s | **30%** | ~1,15 s |
| Projeção exata | 381,1 s | 28% | — |
| Visibilidade, pertencimento, máscaras, qualidade, diagnóstico, persistência, decodificação | 573,0 s | 42% | — |

No braço de mapa inteiro a consulta cai para 6% (não há caixa a consultar) e a projeção sobe
para 35% — o trabalho apenas se move de lugar, e o total quase triplica.

## Extrapolação para o bag completo, e o que ela **não** garante

O bag completo é da ordem de **3.096 frames processados** e um mapa de **~229 M pontos** (8,6× o
deste experimento). Escalando **só pelo número de frames**, a partir de taxas medidas:

| Braço | s/frame medido | 3.096 frames |
|---|---|---|
| culling + streaming | 3,88 | **~3,3 h** |
| mapa inteiro + streaming | 10,89 | ~9,4 h |
| mapa inteiro + batch (baseline) | — | **OOM**: 3.096 frames × ~1,2 GB retidos |

**Isto é extrapolação, não medição.** Dois termos podem piorar com um mapa 8,6× maior e o
experimento não os mediu:

1. **A contagem de candidatos.** Governada pela densidade *local* em torno da câmera, e um mapa
   maior cobre mais *área* em vez de adensar a mesma área — deve permanecer na mesma ordem
   (~13 M no pior caso). É o que o gate de complexidade verifica sinteticamente, mas **não** foi
   verificado no mapa de 229 M pontos.
2. **A consulta de candidatos (30% do tempo).** O índice espacial do Geometric Mapping é o
   *bounding box por scan*, e esses boxes são enormes: extensão mediana de **104 × 64 × 3,7 m**,
   porque cada varredura do LiDAR alcança ~100 m. Neste mapa, 540 de 945 scans intersectam a
   caixa de 40 m, ou seja **56% do payload é varrido por frame**. Num mapa 8,6× maior a *fração*
   cai, mas o número absoluto de scans intersectados pode crescer. Se crescer proporcionalmente,
   a consulta passa a dominar.

**Este é o gargalo remanescente, e o dono dele é Geometric Mapping (#94), não Sensor
Association**: um índice mais fino (grid ou octree em vez de box por scan) atacaria exatamente o
termo que pode crescer. Fica registrado, não implementado — fora do escopo de #562/#563/#564.

## E o pipeline como um todo

Depois destas mudanças, `sensor_association` **deixa de ser o gargalo**. Pelo perfil do #181,
`visual_perception` gasta 656,7 s de geração Qwen3-VL-4B para as imagens da janela; na ordem de
3.096 frames isso é da ordem de **dezenas de horas de GPU**, uma ordem de magnitude acima da
associação. Um bag de 15 minutos é viável no sentido de que a RAM não é mais o limite e a
associação não é mais o item crítico; o próximo alvo é a percepção.

## Reprodutibilidade

```bash
# janela congelada, três braços atribuíveis
python scripts/run_all.py <out> --baseline-src <worktree-40b1820>/src \
    --baseline-script <worktree-40b1820>/arm_baseline.py --report report.json

# sequência completa e a taxa de mapa inteiro
python scripts/_run_one.py python scripts/arm.py C-full <out>/arm-C-full --max-range-m 20 --window all
python scripts/_run_one.py python scripts/arm.py D-partial <out>/arm-D-partial --window all --frames 30
```

Baseline `40b1820`; braços otimizados remedidos em `b654762`. Métricas brutas em `report.json`,
com `peak_before_mb` / `peak_after_mb` para a correção da retenção de entrada.

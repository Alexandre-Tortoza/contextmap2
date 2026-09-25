# Viabilidade de sequência completa (issue #564)

A janela congelada de 15 frames responde à equivalência e à atribuição, mas não responde a
"cabe rodar a sequência inteira?". Esta parte mede passagens longas sobre os **mesmos** artifacts
reais e reporta o que resta de gargalo, honestamente.

## O que foi medido

Máquina: 16 CPUs lógicas, **39,1 GiB de RAM**, RTX 3060 8 GB, Python 3.12.14. Mesmo método de
medição da janela: um processo pai novo por braço, `RUSAGE_CHILDREN.ru_maxrss`.

| Braço | Candidatos | Frames selecionados | Associados | Rejeitados | Observações | Wall time | **s/frame** | Peak RSS |
|---|---|---|---|---|---|---|---|---|
| **C-full** | esfera 20 m | 360 | 350 | 10 | 7.935 | 1.416,7 s (23,6 min) | **4,05** | 6.592 MB |
| **D-partial** | mapa inteiro | 30 | 26 | 4 | 730 | 281,0 s | **10,81** | 8.759 MB |

Ambos com `verify_integrity()` limpo. Sobre a mesma trajetória, o culling a 20 m é **2,7× mais
rápido** por frame que avaliar o mapa inteiro.

## A memória não escala com o número de frames — e a prova é o contraste

| | frames | máx. candidatos em um frame | Peak RSS |
|---|---|---|---|
| Janela (braço C) | 11 | 5.966.091 | 2.307 MB |
| Sequência (braço C-full) | **350** (31,8×) | 12.822.696 (2,15×) | **6.592 MB** (2,86×) |

Se o pico ainda escalasse com o número de frames, 350 frames custariam ~31,8× os 2.307 MB da
janela, isto é **~73 GB** — quase o dobro da RAM da máquina. Ele custou 2,86×, e esse fator
acompanha o **maior frame isolado** (2,15× mais candidatos), não a contagem de frames.

O modelo que o #563 previu se confirma quantitativamente:

```
pico  =  payload do mapa residente (mmap, ~1,7 GB, recuperável)
       + o maior frame de candidatos/projeção/visibilidade (~12,8 M pontos)
       + buffers do writer (pequenos)
```

A ~12,8 M candidatos, os arrays contratuais por frame somam ~1,5 GB, e os temporários da
transformação e da projeção multiplicam isso por ~3 — daí ~4,5 GB, mais ~1,7 GB de páginas do
mapa que a consulta de candidatos acaba tocando por inteiro ao longo de 350 frames. **~6,2 GB
previsto contra 6,59 GB medido.** A decomposição é aritmética sobre os tamanhos de array, não
uma medição por componente: nenhum contador de page fault foi coletado.

## Um resultado que corrige a manchete da janela

A janela de 15 frames tinha 20,6%–22,4% de candidatos. Sobre a trajetória inteira a distribuição
é bem pior:

| | fração de candidatos | pontos |
|---|---|---|
| mínimo | 20,6% | 5.471.837 |
| **p50** | **35,0%** | 9.318.524 |
| p95 | 47,6% | 12.676.116 |
| **máximo** | **48,2%** | 12.822.696 |

A janela caiu numa parte pouco densa do percurso. Onde o robô revisita o mesmo corredor, o mapa
acumulado fica localmente muito mais denso e um recorte de 20 m alcança quase metade dele. **O
ganho real do culling sobre a sequência é ~2,9× (1/0,35), não os ~4,5× que a janela sugeria.**
Reportar só o número da janela seria selecionar o caso favorável.

Isso não afeta a equivalência: o culling continua removendo apenas geometria ocluída (ver
`README.md`). Afeta a expectativa de desempenho, e por isso está aqui.

## Onde o tempo vai, no braço com culling

| Etapa | Total | Fração | p50 por frame |
|---|---|---|---|
| Consulta de candidatos | 410,6 s | **29%** | 1,193 s |
| Projeção exata | 380,6 s | 27% | — |
| Visibilidade, pertencimento, máscaras, qualidade, diagnóstico, persistência, decodificação de imagem | 625,5 s | 44% | — |

No braço de mapa inteiro a consulta cai para 5% (não há caixa a consultar) e a projeção sobe
para 35% — o trabalho apenas se move de lugar, e o total quase triplica.

## Extrapolação para o bag completo, e o que ela **não** garante

O bag completo é da ordem de **3.096 frames processados** e um mapa de **~229 M pontos**
(8,6× o deste experimento). Escalando **só pelo número de frames**, a partir de taxas medidas:

| Braço | s/frame medido | 3.096 frames |
|---|---|---|
| culling + streaming | 4,05 | **~3,5 h** |
| mapa inteiro + streaming | 10,81 | ~9,3 h |
| mapa inteiro + batch (baseline) | — | **OOM**: 3.096 frames × ~1,2 GB retidos |

**Isto é extrapolação, não medição.** Dois termos podem piorar com um mapa 8,6× maior e o
experimento não os mediu:

1. **A contagem de candidatos.** Ela é governada pela densidade *local* em torno da câmera, e um
   mapa maior cobre mais *área* em vez de adensar a mesma área — então ela deve permanecer na
   mesma ordem (~13 M no pior caso). É o que o gate de complexidade em CI verifica
   sinteticamente, mas **não** foi verificado no mapa de 229 M pontos.
2. **A consulta de candidatos (29% do tempo).** O índice espacial do Geometric Mapping é o
   *bounding box por scan*, e esses boxes são enormes: extensão mediana de **104 × 64 × 3,7 m**,
   porque cada varredura do LiDAR alcança ~100 m. Neste mapa, 540 de 945 scans intersectam a
   caixa de 40 m, ou seja **56% do payload é varrido por frame**. Num mapa 8,6× maior a *fração*
   cai, mas o número absoluto de scans intersectados pode crescer. Se crescer proporcionalmente,
   a consulta passa a dominar.

**Este é o gargalo remanescente, e o dono dele é Geometric Mapping (#94), não Sensor
Association**: um índice mais fino (grid ou octree em vez de box por scan) atacaria exatamente o
termo que pode crescer. Fica registrado, não implementado — está fora do escopo de #562/#563/#564.

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

Commit do baseline `40b1820`; commit otimizado `c8bf518`. Métricas brutas em `report.json`.

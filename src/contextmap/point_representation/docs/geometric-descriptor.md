# Descritor geométrico determinístico

Este documento descreve `src/contextmap/point_representation/backends/geometric_descriptor.py`.

## Por que existe

O descritor é a **condição de controle** das ablações de representação: geometria apenas (descritor determinístico) contra uma representação 3D aprendida contra nenhuma representação. Ele dá ao pipeline um canal geométrico reproduzível de estrutura local antes de qualquer backbone aprendido, e é o ponto de comparação que um encoder aprendido precisa superar para se justificar.

Ele é derivado **somente** das coordenadas do suporte preparado. Não usa RGB, feature visual, label, modelo neural, fusão entre observações nem classificação semântica: é evidência geométrica, não um classificador.

## Uso

`GeometricDescriptorEncoder(support_policy)` implementa `PointEncoder`. A política de suporte faz parte da identidade do espaço; uma política de ponto único é rejeitada porque não há estrutura local a descrever. O backend não é exportado pela raiz de `contextmap.point_representation`: a composição em `runtime` o importa de `contextmap.point_representation.backends.geometric_descriptor`.

## Espaço de representação (versão `1`)

`family = "geometric_descriptor"`, `model = "local-covariance-shape"`, `version = "1"`, sem checkpoint, `dimension = 14`, `dtype = "float64"`, `normalization = "none"`, `input_definition = "xyz-local-prepared"`. O fingerprint cobre a versão, a **lista ordenada de features**, a normalização, a política de suporte e a dimensão/`dtype`: mudar a definição de uma feature exige nova versão.

Sejam `λ1 ≥ λ2 ≥ λ3 ≥ 0` os autovalores da covariância populacional dos membros do suporte, nas coordenadas preparadas (metros, a menos que a política normalize a escala).

| Índice | Feature | Definição |
| --- | --- | --- |
| 0 | `support_size` | número de membros |
| 1 | `density_per_m3` | `n / (4/3 π r³)`, com `r` a distância do membro mais afastado ao centro, em metros |
| 2 | `rms_radius` | `√(λ1 + λ2 + λ3)` |
| 3 | `max_radius` | maior distância de um membro ao centroide |
| 4 | `linearity` | `(λ1 − λ2) / λ1` |
| 5 | `planarity` | `(λ2 − λ3) / λ1` |
| 6 | `scattering` | `λ3 / λ1` |
| 7 | `surface_variation` | `λ3 / (λ1 + λ2 + λ3)` |
| 8–10 | `normal_abs_x/y/z` | `\|componente\|` do autovetor de `λ3` |
| 11–13 | `principal_axis_abs_x/y/z` | `\|componente\|` do autovetor de `λ1` |

`linearity`, `planarity` e `scattering` somam 1 (Weinmann et al., 2014, autovalores da covariância local); o descritor aplica uma definição própria, sem ser uma reprodução do artigo. As features de forma são invariantes a rotação, translação e escala; `rms_radius` e `max_radius` carregam a escala, e `density_per_m3` usa as estatísticas em metros do suporte, então também independe da normalização de escala.

**Ambiguidade de sinal.** Um autovetor não tem sinal, então as direções são publicadas em **valor absoluto**: isso elimina a ambiguidade sem uma convenção arbitrária e sem assumir que o eixo z do frame do mapa é vertical. A contrapartida é que `(1, 1, 0)` e `(1, −1, 0)` têm o mesmo valor.

## Comportamento numérico

Tudo é aritmética em Python puro sobre doubles: somas com `math.fsum` (exatas e independentes da ordem dos membros) e uma iteração de Jacobi cíclica com limite fixo de varreduras para os autovalores, precisa também nas matrizes singulares e de autovalores repetidos que suportes degenerados produzem. Nenhuma biblioteca numérica ou de modelo é importada.

**Tolerância de reprodutibilidade.** A mesma entrada gera o mesmo vetor, bit a bit, na mesma plataforma; entre plataformas o esperado é concordância em torno de `1e-12`, e os testes usam `1e-9`. A ordem dos membros não altera o vetor.

### Componentes indefinidos

Um componente que não pode ser definido é declarado em `undefined_components`, com placeholder zero. **Nada é inventado para ele.**

| Situação | Indefinido |
| --- | --- |
| o membro mais afastado está à distância zero (ponto único ou pontos coincidentes) | densidade |
| `rms_radius ≤ 1e-9` (`MIN_SUPPORT_EXTENT`) | features de forma, normais e eixos |
| `λ2 − λ3 ≤ 1e-6·λ1` (`EIGEN_GAP_TOLERANCE`): uma reta, ou uma dispersão sem normal distinta | normal |
| `λ1 − λ2 ≤ 1e-6·λ1`: um plano ou volume isotrópico | eixo principal |

Uma direção só é publicada quando seu autovalor está separado do vizinho, então ela é numericamente única; o quão confiável ela é como geometria é o que `linearity` e `planarity` dizem. Coordenadas inválidas (`NaN`/infinito) já são rejeitadas pelo contrato `PreparedSupport`, então nunca chegam ao descritor. Um suporte preparado sob uma política diferente da declarada no espaço é rejeitado com `ValueError`, porque rotularia o vetor com o espaço errado.

## Validação sintética

Os testes usam suportes sintéticos exatos: um plano (`planarity = 1`, normal ao longo do eixo, sem eixo principal), um retângulo (eixo principal ao longo do lado maior), uma reta (`linearity = 1`, sem normal) e um volume (`scattering = 1`, `surface_variation = 1/3`, sem direções), além de um plano ruidoso, rotações e um cruzamento com uma decomposição independente em NumPy (25 sementes com rotação aleatória) quando NumPy está instalado.

## Linha de base de tempo e memória

Medição ad hoc (não versionada), Python 3.14 em Linux x86_64, um único thread, Python puro:

| Suporte (n membros) | `encode` |
| --- | --- |
| 10 | 29 µs |
| 100 | 95 µs |
| 350 | 289 µs |
| 1 000 | 799 µs |

Pelo serviço, sobre 100 mil pontos sintéticos de corredor com fonte em grade e suporte por raio de 0,5 m (mediana de 336 pontos, 400 centros): **1,91 ms por representação**, dos quais 1,59 ms são a extração do suporte e 0,29 ms o descritor. O pico de memória rastreada em streaming foi de cerca de 336 KiB, e cada vetor armazenado ocupa 112 bytes (14 × `float64`). Como cada representação custa milissegundos, representar todos os pontos de um mapa grande é caro em um único thread; o custo é dominado pela consulta espacial, não pelo descritor. Sem GPU, sem runtime de modelo.

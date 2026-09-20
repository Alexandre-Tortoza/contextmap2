# Extração de suporte local

Este documento descreve `src/contextmap/point_representation/support.py`.

`SupportExtractor(source, policy)` seleciona e prepara o suporte local de um elemento de geometria: `extract(center)` devolve um `PreparedSupport` (o `PointSupport` mais uma coordenada preparada por membro). Ele lê a geometria **somente** pela porta pública `GeometrySource` de Geometric Mapping, então um encoder e o extrator não dependem de como o mapa é armazenado ou indexado, e a extração não usa modelo, label nem evidência visual.

## Ordem determinística

Os membros são ordenados **centro primeiro**, depois por distância euclidiana crescente ao centro, com empate por `geometry_id` crescente. A ordem não depende da ordem em que a fonte devolve os pontos: o mesmo mapa e a mesma política produzem o mesmo suporte, as mesmas coordenadas preparadas e os mesmos metadados. Elementos distintos com coordenadas idênticas (pontos coincidentes) são todos mantidos.

O centro é sempre membro do próprio suporte, mesmo que a fonte não o devolva na consulta, e nunca é descartado por `max_neighbors`.

## Métodos

| Política | Consulta | `query_method` |
| --- | --- | --- |
| `POINT` | nenhuma: o centro é resolvido por referência | `reference-lookup` |
| `NEIGHBORHOOD` + `RADIUS` | uma consulta de caixa, depois filtro exato de distância `≤ radius_m` (fronteira inclusiva) | `bounds-query+euclidean-radius` |
| `NEIGHBORHOOD` + `K_NEAREST` | consultas de caixa com raio dobrando até `k` elementos caberem no raio | `bounds-query-doubling+euclidean-k-nearest` |

A caixa de pré-filtro tem uma folga relativa de `1e-9`, porque subtrair e somar o raio arredonda e um ponto cuja distância exata está dentro do raio poderia cair um ulp fora de uma caixa justa; o filtro de distância é quem decide. No k-nearest, o raio inicial é uma estimativa pela densidade do mapa e serve só como ponto de partida: quando `k` elementos estão dentro do raio, nenhum elemento fora dele pode ser mais próximo que o `k`-ésimo, então o resultado é exato e independe da estimativa. A busca também para quando a caixa cobre todo o mapa, então um mapa com menos de `k` pontos devolve todos (a contagem menor fica visível em `statistics.count`).

`max_neighbors` corta um suporte por raio mantendo os mais próximos; `statistics.candidate_count` guarda quantos elementos qualificaram antes do corte, então um suporte truncado nunca parece completo. Para k-nearest, `candidate_count` é `None`.

## Frames e erros

Toda coordenada usada é a `coordinates_m` autoritativa, em metros, no frame do mapa; a medição no frame do sensor nunca entra. O extrator falha cedo com `ValueError` quando a fonte devolve um ponto de outro mapa, um ponto em outro frame que o do mapa ou o mesmo elemento duas vezes na mesma consulta, e com `KeyError` (da própria porta) quando o centro não pertence ao mapa.

## Perto dos limites do mapa

`statistics.near_map_bounds` indica que a bola que o suporte pode cobrir ultrapassa os limites do mapa em algum eixo (raio da política para um suporte por raio, distância ao membro mais distante nos demais): um suporte cortado pela borda do mapa não é comparável a um do meio dele. Tocar o limite exatamente não conta.

## Preparação das coordenadas

`PreparedSupport.local_coordinates_m` traz uma coordenada por membro, na mesma ordem de `geometry_refs`:

- `centering = NONE`: coordenadas do frame do mapa;
- `CENTER`: relativas ao centro (o centro fica na origem);
- `CENTROID`: relativas à média dos membros (soma exata com `math.fsum`);
- `scale_normalization = SUPPORT_RADIUS`: divididas pelo raio da política, então os membros ficam em uma bola unitária. `PointSupport.applied_scale_m` informa o divisor.

A preparação usada fica registrada em `support.policy.preparation`; as referências continuam dizendo qual geometria persistente formou as coordenadas. Amostragem/downsampling explícito e canonicalização de orientação **não** existem: `max_neighbors` é um teto determinístico, não uma amostragem.

## Linha de base de desempenho

Medição ad hoc (não versionada) sobre geometria sintética de corredor de 20 m × 3 m × 3 m com ruído de 5 mm, uma fonte em memória em Python puro, um único thread, Python 3.14 em Linux x86_64. Mediana por suporte, centros aleatórios:

| N pontos | Fonte | raio 0,25 m | raio 0,5 m | k = 16 |
| --- | --- | --- | --- | --- |
| 10 000 | varredura linear | 5,1 ms | 5,1 ms | 6,0 ms |
| 10 000 | índice em grade (0,5 m) | 0,06 ms | 0,13 ms | 1,9 ms |
| 100 000 | varredura linear | 52 ms | 54 ms | 56 ms |
| 100 000 | índice em grade (0,5 m) | 0,5 ms | 1,4 ms | 4,6 ms |

Interpretação: o custo é dominado pela consulta de caixa da fonte, não pelo extrator. Com varredura linear o custo cresce com N (52 ms por suporte a 100 mil pontos) e extrair o suporte de todos os pontos seria inviável; com um índice espacial fica em ordem de milissegundo. Os números descrevem fontes sintéticas: a fonte real de Geometric Mapping ainda não foi medida aqui, e nenhuma implementação de KD-tree ou voxel foi forçada no contrato público. Nenhuma otimização foi feita: os números não mostram uma necessidade que o índice da fonte não resolva.

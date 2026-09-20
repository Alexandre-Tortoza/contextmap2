# Acesso espacial: limites, lookup e índice

Este documento descreve a leitura de `src/contextmap/geometric_mapping/geometry_storage.py` (`PackedGeometry`) e a fronteira `GeometrySource` de `ports.py`.

Capabilities posteriores recuperam geometria **por referência** e **por extensão espacial** sem depender de como o mapa é armazenado. A fronteira é pequena e não tem semântica:

| Método | Significado |
| --- | --- |
| `geometric_map` | identidade, frame, limites (`bounds`), observações de origem e proveniência do mapa |
| `get(reference)` | resolve a referência para a **mesma** geometria autoritativa, sempre (`KeyError` se não pertence ao mapa) |
| `iter_geometry()` | itera toda a geometria em ordem determinística (a ordem de acumulação) |
| `query_bounds(bounds)` | itera a geometria dentro de uma caixa, na ordem de índice |

Não há label, consulta semântica, "entidade mais próxima" nem projeção em imagens, e não existe um motor de consulta genérico.

## Semântica dos limites

`Bounds3D` declara o frame dos seus extremos, e a consulta usa o frame global do mapa. Um `Bounds3D` em **outro** frame é recusado (`ValueError`), nunca reinterpretado; nada é inferido do nome do mapa nem do chamador. As **faces são inclusivas**: um ponto sobre uma face é devolvido, e uma caixa degenerada (um ponto) funciona. `GeometricMap.bounds` é o envelope justo da geometria persistida.

## O índice

O índice inicial é `scan_bounds` (registrado em `GeometricMap.spatial_index` com `is_derived=True`, sem parâmetros): cada `ScanRecord` guarda o envelope justo da geometria que o scan contribuiu. Uma consulta:

1. ignora os scans cujos limites não intersectam a caixa;
2. devolve todos os pontos de um scan **inteiramente dentro** da caixa sem testá-los um a um;
3. testa cada ponto apenas dos scans que cruzam a caixa.

O índice só decide **o que é lido**, nunca **o que a geometria é**: o resultado é exatamente o que um filtro completo de `iter_geometry()` devolveria (um teste compara 60 caixas aleatórias contra a força bruta, e outro roda a mesma consulta contra uma fonte sem índice algum). O chamador não vê qual estrutura espacial existe: trocá-la por uma grade de voxels ou uma árvore não muda a fronteira.

Os limites por scan são a parte do índice de origem que é **derivada**: não são autoritativos. A geometria persistida é a única autoridade.

## Reconstrução e integridade

- `rebuild_scan_bounds()` recalcula os limites de todos os scans a partir do payload, sem tocar em nenhuma identidade nem coordenada;
- `verify_index()` recalcula e compara: devolve um problema por scan cujos limites registrados não são o envelope da sua geometria e outro se `GeometricMap.bounds` não é o envelope do mapa. Um índice derivado corrompido é, portanto, **detectado** em vez de redefinir silenciosamente o que uma consulta encontra; `get` e `iter_geometry` nunca dependem dele.

Não há arquivo de índice persistido separado nesta etapa: os limites por scan vivem na tabela de origem, e a integridade do arquivo é coberta pelo inventário do artefato.

## Linha de base de desempenho

Medida sintética, uma execução, mapa de corredor com **3 milhões de pontos** (300 scans de 10 mil pontos, 192 MB empacotados; nuvem de ±12 m em torno de cada pose, poses a 1 m):

| Operação | Tempo |
| --- | --- |
| montar o mapa (transformar + acumular) | 0,34 s (≈ 8,8 Mpts/s) |
| consulta por caixa de 1 m (2 894 pontos) com o índice | 66 ms |
| consulta por caixa de 5 m (14 470 pontos) com o índice | 129 ms |
| mesmo filtro linear sobre todos os registros (referência, sem montar `GeometryPoint`) | ≈ 630 ms |
| `verify_index()` | 1,4 s |

Uma consulta que devolve muitos pontos é dominada pela construção de cada `GeometryPoint` (cerca de 5 µs cada). O ganho do índice cresce com o número de scans: ele evita ler os que não encostam na caixa.

## Limitações

- Dentro de um scan que cruza a caixa a busca é linear. Uma grade de voxels por scan ou global só entra se medições justificarem.
- `iter_geometry` e `query_bounds` constroem objetos Python por ponto; para percorrer o mapa inteiro em escala, a consulta por caixa deve ser preferida.

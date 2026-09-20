# Contratos de Geometric Mapping

Este documento descreve `src/contextmap/geometric_mapping/models.py`, `ports.py` e `serialization.py`.

## Autoridade da coordenada

- `GeometryPoint.coordinates_m` no **frame global do mapa** é a coordenada autoritativa e persistente, em metros;
- `source_coordinates_m` guarda a medição original no frame do sensor e nunca é confundida com a anterior;
- coordenadas relativas a um segmento ou submapa só podem existir como **visão derivada explícita** (`P_segment = T_segment_map · P_map`); nunca substituem nem redefinem a coordenada do mapa;
- processar apenas um segmento da sequência não cria um segundo sistema de coordenadas global.

Unidades e frames são explícitos em todo contrato: `Bounds3D` declara o frame de seus extremos, e nada é inferido pelo nome do mapa ou pelo contexto do chamador.

## Identidade

`GeometryReference` é `(map_id, geometry_id)` e é **local ao artifact de mapa imutável** identificado por `map_id`. Referências continuam válidas depois de fechar e reabrir o artifact. `geometry_id_for(map_id=..., index=...)` gera identidades como função pura das entradas, então não há registry de identidade, e `geometry_index_of(...)` a inverte: só a grafia canônica é aceita.

Downstream mantém referências em vez de duplicar XYZ ou proveniência; quando precisa da coordenada, resolve a referência por `GeometrySource.get`.

## `GeometryPoint`

| Campo | Significado |
| --- | --- |
| `geometry_id` / `map_id` | identidade do ponto e do mapa dono |
| `map_frame` / `coordinates_m` | frame global do mapa e posição autoritativa nele |
| `source_frame` / `source_coordinates_m` | frame do sensor e medição original |
| `source_observation_id` | observação física de origem |
| `source_point_index` | índice dentro da observação; `None` para um ponto agregado |
| `acquisition_timestamp` | quando a observação de origem foi adquirida (`SourceTimestamp`, com `clock_id`) |
| `transform_lineage` | cadeia de transforms que levou `source_coordinates_m` a `coordinates_m` |
| `provenance` | como o ponto foi produzido (`GeometryPointProvenance`) |

Validação em construção: identidades e frames não vazios, coordenadas finitas, índice não negativo e uma linhagem que **conecta** o frame do mapa ao frame da fonte. Um ponto sem passos de transform só é válido quando os dois frames coincidem.

## `TransformLineage`

`P_map = T_0 · T_1 · … · P_source`, onde `T_i` é `steps[i]`: o primeiro passo tem o frame do mapa como pai e o frame filho de cada passo é o pai do seguinte. Para um ponto de LiDAR, a cadeia é `T_map_body(t) · T_body_lidar`.

Cada `TransformStep` declara `kind` (`STATIC_CALIBRATION` ou `DYNAMIC_POSE`), `parent_frame`, `child_frame`, `reference` (identidade da calibração ou da pose usada) e, para um passo dinâmico, os `source_estimate_ids`: a pose estimada, ou as duas poses quando ela foi interpolada. Um passo estático não carrega poses. Uma linhagem é compartilhada por todos os pontos de uma mesma observação; não é copiada por ponto.

## `GeometryPointProvenance` e agregação

Um ponto é `MEASURED` (uma medição do sensor, transformada) ou `AGGREGATED` (produzido por uma regra explícita de deduplicação ou downsampling). Um ponto agregado **nunca finge ser uma medição crua**: nomeia a `aggregation_rule`, informa `contributing_point_count` (pelo menos dois) e não carrega `source_point_index`.

Cada ponto também carrega `motion_correction` (`RAW`, `CORRECTED` ou `UNKNOWN`, padrão `UNKNOWN`): o estado de correção de movimento do scan de origem, nunca inferido. Ver [`motion-correction.md`](motion-correction.md).

## `Bounds3D`

Caixa alinhada aos eixos, com `frame_id`, `minimum_m` e `maximum_m`. Fronteiras são **inclusivas**: um ponto sobre uma face está contido e duas caixas que se tocam se intersectam. `contains` e `intersects` exigem o mesmo frame e falham caso contrário. `Bounds3D.enclosing` constrói a caixa justa de um conjunto de pontos.

## `GeometricMap`

Identidade e metadados de um mapa persistente: `map_id`, `frame_id` (o frame global), `point_count`, `bounds` (no mesmo frame), `source_observation_ids` (únicas), `time_bounds`, `spatial_index` opcional e `provenance` (`GeometricMapProvenance`: sequência, seleção, trajetória e run de State Estimation usados, identidade da calibração, política de lookup de pose, fingerprint de configuração e versão do código).

`aggregation_rule` registra a regra explícita pela qual medições foram unidas em pontos agregados; `None` quando todo ponto é uma medição crua.

O objeto **não embute os pontos**: eles vivem no storage do artifact e são alcançados por `GeometrySource`. `SpatialIndexMetadata` registra o índice quando ele afeta o comportamento de consulta e distingue um índice derivado (`is_derived`) da geometria autoritativa: um índice derivado corrompido nunca redefine coordenadas.

## `GeometrySource`

Porta de leitura que consumidores usam sem depender de como o mapa é armazenado:

- `geometric_map`: metadados do mapa;
- `get(reference)`: resolve a referência para a mesma geometria autoritativa sempre;
- `iter_geometry()`: itera todos os elementos em ordem determinística;
- `query_bounds(bounds)`: itera os elementos dentro de uma caixa (fronteiras incluídas), que deve estar no frame do mapa.

A porta não tem label, feature, entidade nem consulta semântica, e não projeta em imagens.

## Serialização

`serialization.py` converte os contratos para registros com apenas primitivas JSON, legíveis sem ROS, biblioteca de nuvem de pontos ou NumPy, e revalida os contratos ao decodificar (uma linhagem que não conecta os frames falha em vez de produzir um ponto inválido). Os timestamps usam `SourceTimestamp.to_record()`/`from_record()` de `contextmap.shared`.

# Transformação fonte → mapa

Este documento descreve `src/contextmap/geometric_mapping/transformation.py`.

Um ponto de LiDAR só é geometria autoritativa depois de chegar ao frame global do mapa de forma reprodutível:

```text
P_map = T_map_body(t) · T_body_source · P_source
```

`T_map_body(t)` é a pose resolvida na montagem dos inputs ([`inputs.md`](inputs.md)); `T_body_source` é o extrínseco estático da calibração canônica. Ambos seguem a convenção `T_parent_child` (`p_parent = R · p_child + t`), quaternion `(x, y, z, w)` e metros. Cada passo mantém a direção e a referência (a pose ou a calibração usada), então qualquer ponto pode ser rastreado até a origem exata.

## Resultado

`transform_scan(item, calibration_identity=...)` devolve um `TransformedScan`; `transform_scans(plan)` faz o mesmo para todos os inputs de um plano, **um scan por vez** (iterador preguiçoso, então só as colunas de um scan ficam vivas).

| Campo | Significado |
| --- | --- |
| `observation_id`, `payload_hash`, `acquisition_timestamp` | identidade e instante do scan de origem |
| `source_frame` / `map_frame` | frames das coordenadas originais e das autoritativas |
| `source_point_indices` | índice, dentro do scan, de cada ponto mantido |
| `source_coordinates_m` | `(x, y, z)` **originais** de cada ponto mantido, em metros |
| `map_coordinates_m` | `(x, y, z)` **autoritativos** no frame do mapa, em metros |
| `transform_chain` | os fatores aplicados, com os números e a referência de cada um |
| `motion_correction` | estado de correção de movimento do scan (ver [`motion-correction.md`](motion-correction.md)) |

As colunas são `array.array` da biblioteca padrão (`(x, y, z)` por ponto), não objetos NumPy nem uma lista de objetos Python; trate-as como somente leitura. A coordenada de origem é o valor do payload promovido a `float64` sem perda; **nunca é recomputada a partir do mapa**. `geometry_point(position, map_id=..., geometry_index=...)` materializa um ponto como `GeometryPoint`, com as duas coordenadas, a identidade da observação, o índice original e a linhagem.

## Linhagem

A linhagem é a mesma para todos os pontos do scan, então não é copiada por ponto:

- passo dinâmico (`DYNAMIC_POSE`): `map ← body`, `reference` é a identidade da pose usada e `source_estimate_ids` as poses estimadas (duas quando a pose foi interpolada; a `reference` é então a pose derivada, distinta das duas);
- passo estático (`STATIC_CALIBRATION`): `body ← frame do scan`, `reference` é a identidade da calibração; ausente quando o scan já está no frame do corpo.

## Validação

Falham com `GeometryTransformError`, nunca com um palpite:

- frames da cadeia que não se conectam (o pai do extrínseco não é o corpo da pose, ou o filho não é o frame do scan; um scan fora do frame do corpo sem transform estático);
- um transform estático sem a identidade da calibração de onde veio;
- uma rotação que não é um quaternion unitário, ou uma translação que não é finita;
- uma coordenada de mapa que não é finita.

Pontos cujas coordenadas de origem **não são finitas** (`NaN`/`inf`, comuns em scans não densos) são descartados e contados (`dropped_non_finite_count`); não são "consertados". O índice original dos pontos mantidos é preservado. Não há filtro de distância nem de ruído aqui.

## Precisão numérica

A aritmética é `float64`, com ordem de avaliação fixa (sem BLAS), então o resultado não depende do build do NumPy. Para coordenadas de LiDAR abaixo de 1e3 m o erro de arredondamento é da ordem de 1e-12 m; os testes usam a tolerância absoluta de **1e-9 m**. A quantização do payload `float32` (~1e-4 m a 1e3 m) vem do sensor, não da transformação. Um scan de 100 mil pontos leva cerca de 13 ms.

O NumPy é importado apenas quando um scan é transformado (`import contextmap.geometric_mapping` não o importa; há um teste). Ele está nos extras `dev` e `ros1`, não nas dependências base do pacote.

## Traces de transformação

`scan.trace_point(position)` devolve um `TransformTrace` equivalente a:

```text
source observation: lidar-frame-01824
source point index: 713
P_lidar:            [4.21, -0.71, 0.32]
T_body_lidar:       <referência da calibração> + números
T_map_body(t):      <referência da pose> + números
P_map:              [18.41, 3.82, 1.24]
```

`verify_transform_trace(trace)` reconstrói `P_map` a partir da própria cadeia e devolve os problemas encontrados (cadeia não contígua, ou `P_map` a mais de 1e-9 m do registrado); `transform_trace_residual_m(trace)` devolve essa distância. Um ponto agregado tem `source_point_index = None`. `encode_transform_trace`/`decode_transform_trace` (em `serialization.py`) persistem uma amostra de traces em JSON puro **sem duplicar a matriz por ponto**: a cadeia aparece uma vez por trace. Escolher e gravar a amostra é responsabilidade do artefato de mapa.

## Limitações

- A correção de movimento não é aplicada: todos os pontos de um scan usam a pose de **um** instante. O estado do scan (`RAW`/`CORRECTED`/`UNKNOWN`) é registrado em cada ponto, mas nenhum deskew é feito aqui.
- Sem projeção em câmera e sem dado semântico.

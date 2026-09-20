# Contratos de Sensor Association

Este documento descreve `src/contextmap/sensor_association/models.py` e `serialization.py`.

## Evidência, não crença

Uma `SpatialObservation` registra que **esta região, neste frame, enxerga esta geometria**. Ela é evidência produzida em uma execução específica:

- não há `label`, `embedding`, `entity_id`, `FusedEvidence` nem identidade final: estes níveis semânticos pertencem a outras capabilities;
- regiões sobrepostas podem compartilhar a mesma geometria: nenhuma observação é dona exclusiva de um ponto, e nada aqui escolhe um vencedor;
- geometria, features e claims são **referências**; a coordenada é resolvida por `GeometrySource.get`, e o vetor de uma feature ou o texto de uma claim, por Visual Perception.

## `SpatialObservation`

| Campo | Significado |
| --- | --- |
| `spatial_observation_id` | identidade; `spatial_observation_id_for(perception_result_id=, region_id=)` a calcula como função pura |
| `source_observation_id` | o frame de câmera em que a região foi observada |
| `perception_result_id` / `region_id` | o resultado de percepção e a região observada |
| `geometry_support` | `GeometryReference` da geometria visível dentro da região |
| `projection_summary` | fatos da projeção no nível do frame (`ProjectionSummary`) |
| `visual_feature_refs` | features do mesmo resultado que descrevem a região |
| `semantic_claim_refs` | claims do mesmo resultado sobre a região |
| `calibration_ref` / `pose_ref` | calibração e pose usadas |
| `visibility` | por que candidatos entraram ou não (`VisibilityDiagnostics`) |
| `provenance` | mapa, run de percepção, sequência, políticas, configuração e código |

Invariantes validadas na construção (e revalidadas ao decodificar):

- identidades não vazias;
- `geometry_support` é **ordenado por `geometry_id`, sem repetição** e pertence a **um único mapa**, o mesmo de `provenance.geometric_map_id`; a ordem canônica torna o registro determinístico;
- a contagem `ASSOCIATED` de `visibility` é igual ao tamanho de `geometry_support`;
- os diagnósticos não excedem o que a projeção considerou (`visibility.total <= considered_count`) nem o que ficou visível (`len(geometry_support) <= visible_count`);
- uma feature de escopo `REGION` pertence à região observada.

Uma região **sem suporte** é representável: o suporte é vazio e `visibility` explica o porquê (por exemplo, tudo ocluído). Ausência de ponto nunca é silenciosa.

## `VisibilityState` e `VisibilityDiagnostics`

| Estado | Significado |
| --- | --- |
| `BEHIND_CAMERA` | o modelo de câmera não consegue projetar o ponto |
| `OUTSIDE_IMAGE` | projeta fora da imagem preparada |
| `OUTSIDE_VALID_SUPPORT` | cai na imagem preparada, mas fora do suporte válido ou dentro de uma região de exclusão |
| `OCCLUDED` | uma superfície mais próxima o esconde no mesmo raio |
| `VISIBLE_UNASSIGNED` | visível, mas em nenhuma máscara de região |
| `ASSOCIATED` | visível e dentro da máscara da região |

`VisibilityDiagnostics.counts` guarda apenas contagens positivas, na ordem de declaração do enum, então diagnósticos iguais sempre codificam para o mesmo registro. Contagem negativa é rejeitada.

## `PointCorrespondence`

Registro de baixo nível do que aconteceu com **um** elemento do mapa em **um** frame: `geometry`, `visibility`, `camera_range_m`, `raw_pixel`, `prepared_pixel` e `support_range_m`. Serve a diagnósticos e a verificações de reprojeção; consumidores downstream usam `SpatialObservation`.

Regras por estado:

- `OUTSIDE_VALID_SUPPORT`, `OCCLUDED`, `VISIBLE_UNASSIGNED` e `ASSOCIATED` chegaram a um pixel da imagem preparada e **carregam `prepared_pixel`**;
- `BEHIND_CAMERA` não tem pixel algum nem suporte;
- `OUTSIDE_IMAGE` pode guardar o pixel que o modelo produziu, mas não tem suporte;
- `OCCLUDED` nomeia o `support_range_m` da superfície que o esconde, estritamente menor que o seu `camera_range_m`;
- todo valor é finito e nenhum alcance é negativo.

## Convenções

- **Alcance (`range`)**: distância, em metros, do centro óptico da câmera ao ponto, **ao longo do raio de visada**. Usa-se alcance e não profundidade `z` porque um modelo com campo de visão acima de 180° enxerga pontos com `z <= 0`; o alcance é definido e positivo em qualquer modelo.
- **Pixel**: `(u, v)` contínuos, com o **centro** do pixel no inteiro. Aparece em dois sistemas: `raw_pixel` (imagem crua) e `prepared_pixel` (depois da transformação identificada por `ProjectionSummary.image_transform_id`).
- **Frames e unidades**: `CalibrationRef.camera_frame` declara o frame óptico em que a projeção foi feita; distâncias em metros.

## Referências e proveniência

- `VisualFeatureRef` guarda `feature_id`, `embedding_space_id` (para nunca misturar espaços por acidente), `scope` e, para escopo `REGION`, a `region_id`. Escopos `DENSE` e `GLOBAL` não têm região.
- `SemanticClaimRef` guarda apenas o `claim_id`.
- `CalibrationRef`: `calibration_identity` (hash do conjunto de calibração), `camera_calibration_id`, `camera_model_kind` e `camera_frame`.
- `PoseRef` espelha a evidência de um lookup de State Estimation sem embutir a pose: trajetória, run, `source_estimate_ids` (uma pose, ou as duas vizinhas de uma interpolação), `lookup_outcome`, `time_delta_ns` e `interpolation_fraction`. Uma pose interpolada exige duas estimativas e uma fração em `[0, 1]`; uma pose direta exige uma estimativa e nenhuma fração.
- `AssociationProvenance`: mapa geométrico, run de percepção, sequência canônica, `visibility_policy_id`, `membership_policy_id`, fingerprint de configuração e versão do código. Políticas são versionadas; trocar a regra troca o identificador.

## Serialização

`serialization.py` converte os contratos para registros com apenas primitivas JSON, legíveis sem ROS, NumPy ou SDK de modelo, e revalida os contratos ao decodificar: um suporte fora de ordem, ou uma contagem `ASSOCIATED` inconsistente, falha em vez de produzir uma observação inválida. `visibility` é codificado como `{estado: contagem}` e pixels ausentes ficam como `null` explícito.

# Montagem dos inputs de geometria

Este documento descreve `src/contextmap/geometric_mapping/inputs.py`.

Geometric Mapping consome **observações canônicas**, nunca arquivos de dataset, tópicos ROS ou nomes de fabricante. A montagem pareia os scans LiDAR de um trecho selecionado da sequência canônica com tudo o que a transformação para o frame do mapa vai precisar, e recusa adivinhar quando algo falta. **Nenhum ponto é transformado aqui.**

```text
SequenceArtifact (seleção) ─┐
Calibração estática ────────┼─► assemble_geometry_inputs ─► GeometryInputPlan
Trajectory (State Estimation)┘        (sem transformar pontos)
```

## Seleção

A seleção é a do Ingestion (`SequenceSelection`): sequência inteira, intervalo de frames, intervalo de timestamps (em um clock) ou ids explícitos. Ela é resolvida por `resolve_selection`, o mesmo caminho de replay de todas as seleções, e a ordem canônica da sequência é a ordem dos inputs. Duas montagens com as mesmas entradas produzem o mesmo plano.

`assemble_geometry_inputs_from_artifacts(sequence=..., selection=..., run=...)` lê o `SequenceArtifactReader` e o `StateEstimationRunReader`; `assemble_geometry_inputs(...)` trabalha sobre objetos de domínio já lidos.

## O que cada input declara

`GeometryInput` reúne, por scan, antes de qualquer transformação:

| Campo | Significado |
| --- | --- |
| `observation` | o scan de origem, sem alteração |
| `source_frame` / `timestamp` | frame do scan e instante de aquisição, com o `clock_id` |
| `payload_hash` | `sha256:<hex>` dos bytes do payload |
| `layout` | onde `x`, `y`, `z` estão em cada registro de ponto (`PointCloudLayout`) |
| `pose` | `T_map_body(t)` no instante do scan, com como foi resolvida (`ResolvedPose`) |
| `static_transform` | `T_body_source` da calibração; `None` só quando o scan já está no frame do corpo |
| `motion_correction` | estado de correção declarado e a decisão da política (ver [`motion-correction.md`](motion-correction.md)) |

A cadeia é `P_map = T_map_body(t) · T_body_source · P_source`: `pose` é o primeiro fator e `static_transform` o segundo. A pose é resolvida agora, pela política de lookup explícita, para que a **cobertura da trajetória** seja verificada antes de tocar em um ponto; a transformação consome esse resultado em vez de repetir o lookup.

O extrínseco vem do grafo de frames estático da calibração canônica: o caminho `body → frame do scan` é composto com direção explícita. Nunca é inferido pelo nome de um frame.

## Layout do payload e unidades

`resolve_point_cloud_layout` só aceita `x`, `y`, `z` como **um mesmo tipo de ponto flutuante** (`float32` ou `float64`), com um elemento cada, sem sobreposição e dentro do registro, e um payload com o tamanho que declara. Outros campos (intensidade, anel) são permitidos e ignorados. Um layout que exigiria um palpite (`x` inteiro, `z` ausente, tipos diferentes) é recusado com o motivo. Os bytes são little-endian, como o Ingestion normaliza.

O contrato canônico de `LidarObservation` **não tem campo de unidade**: Geometric Mapping trata `x`, `y`, `z` como **metros** (a convenção ROS, a mesma de `ExternalPoseMeasurement.translation`). Uma fonte em outra unidade precisa ser convertida no Ingestion; a montagem não consegue detectá-la por scan.

## Scans recusados e erros da montagem

Um scan inutilizável vira `GeometryInputRejection(observation_id, reason, detail)`, em vez de sumir ou derrubar a execução. O primeiro teste que falha, nesta ordem, decide o motivo:

| Motivo | Quando |
| --- | --- |
| `MISSING_SOURCE_FRAME` | o scan não declara frame |
| `MISSING_CLOCK_IDENTITY` | o timestamp não nomeia o clock |
| `MISSING_PAYLOAD` | não há pontos |
| `UNSUPPORTED_LAYOUT` | as coordenadas não podem ser decodificadas sem palpite |
| `INCONSISTENT_MOTION_CORRECTION` | o estado declarado contradiz o scan |
| `MOTION_CORRECTION_REJECTED` | a política da execução rejeita o estado de correção |
| `NO_STATIC_TRANSFORM` | a calibração não liga o corpo ao frame do scan (ou não existe calibração) |
| `CLOCK_DOMAIN_MISMATCH` | o scan está em outro clock que a trajetória; clocks nunca são comparados implicitamente |
| `POSE_LOOKUP_REJECTED` | a política de lookup não aceita pose para o instante (fora da trajetória, gap, tolerância) |

Um problema que torna a montagem inteira sem sentido levanta `GeometryInputError`: trajetória estimada sobre **outra sequência**; trajetória estimada com **outra calibração** que a dos extrínsecos (um backend que não usou calibração não declara identidade, então não há o que comparar); **identidades de observação repetidas**; ou uma seleção explícita que nomeia uma observação que **não é um scan LiDAR**.

Observações da seleção que não são fonte de geometria (imagem, IMU, pose externa) não são descartadas em silêncio: entram em `ignored_observation_counts`, por modalidade.

## Fontes de geometria

Apenas `LidarObservation` é fonte de geometria hoje. Point clouds derivados de depth **ainda não são suportados**: depth não é uma modalidade canônica do Ingestion, e nenhuma abstração de payload sem tipo entra no lugar. Quando existir, ela entra como um tipo explícito com o seu teste.

## Independência de dataset

Nenhum código desta capability nomeia dataset, arquivo, tópico ou fabricante; um teste varre os módulos por esses nomes. `source_topic` e `source_path` continuam apenas na proveniência da observação, dentro do Ingestion.

## Limitações

- O Ingestion carrega todas as observações da sequência na memória (`list_observations()` é eager); o plano guarda os objetos dos scans selecionados. Trechos longos exigem memória proporcional ao payload selecionado. Um leitor por streaming é uma mudança do Ingestion, fora deste escopo.
- A unidade das coordenadas é uma convenção, não um campo verificável (ver acima).

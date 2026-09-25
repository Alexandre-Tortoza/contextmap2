# Diagnósticos de calibração, reprojeção e alinhamento temporal

Este documento descreve `src/contextmap/sensor_association/diagnostics.py`, versão `association-diagnostics-v2`.

A `v2` (#562) muda três definições: a referência confiável de um frame passa a carregar um `ReprojectionAttempt` com um `ReprojectionOutcome` explícito (`no_reference`, `not_evaluated`, `none_projectable`, `measured`), de modo que a **ausência** de residual é um dos três fatos contados em vez de um `None` único; o aviso `REFERENCE_NOT_EVALUATED` separa uma política de candidatos que não avaliou a geometria da referência de uma câmera que não consegue projetá-la; e a taxa de inválidos passa a ser medida sobre a população **avaliada**, não sobre a declarada. A identidade muda porque as definições mudaram: dois runs cujos diagnósticos significam coisas diferentes nunca podem compartilhá-la.

A associação só é tão confiável quanto a calibração, o alinhamento temporal da pose e o mapa que ela projeta. Estes diagnósticos tornam essas condições **observáveis**, por frame, sem virar confiança semântica: guardam as medidas brutas para que uma política a jusante decida se e como usá-las, nunca convertem uma medida em probabilidade e nunca alteram uma claim.

## O que é sempre registrado

`diagnose_frame(resolution, tolerances=, membership=, correspondences=, dense_samples=)` devolve um `FrameDiagnostics` com o que o frame usou:

| Grupo | Campos |
| --- | --- |
| observação | `source_observation_id`, `image_timestamp` (o timestamp RGB, com o relógio) |
| geometria | `map_id`, `map_time_bounds` (a janela de aquisição da geometria) e `map_window_offset_ns` (a distância com sinal do frame à janela, `0` dentro dela) |
| calibração e câmera | `calibration_ref`, `camera` (identidade, hash, frame, modelo e tamanho) e `extrinsic` |
| pose | `pose_ref`: trajetória, estimativas de origem, desfecho do lookup e `time_delta_ns` |
| imagem | `image_transform_id`, `prepared_image_size` |
| visibilidade | política (`policy_id` e fingerprint), `depth_metric`, contagens por estado, `visible_count` e o resumo de profundidade dos visíveis (a faixa de alcance) |
| pertencimento | o resumo de `MembershipStatistics`, quando avaliado |

`to_record()` devolve o relatório em primitivas JSON, legível por máquina e sem dependência de ROS ou modelos.

## Reprojeção, só contra referência confiável

O resíduo de reprojeção só existe contra **correspondências de referência confiáveis** (`TrustedCorrespondences`): pares de geometria do mapa e do pixel cru em que ela foi realmente observada, vindos de uma fonte confiável (marcos medidos, um alvo de calibração). Nada aqui as cria, e nenhuma reprojeção é estimada a partir da própria associação.

`reprojection_statistics(frame, correspondences)` mede, por correspondência, a distância em pixels entre onde o frame projeta a geometria e onde ela foi observada, e devolve `ReprojectionStatistics`: contagens, média, mediana, **p95** (com interpolação linear entre estatísticas de ordem) e máximo. Uma correspondência que o modelo de câmera não projeta é **inválida**: conta em `invalid_count` e não contribui com resíduo. Se nenhuma projeta, o resultado é `None`.

Sem referência, `reprojection` é `None` e `reprojection_unavailable_reason` diz por quê; nunca é zero.

## Achados explícitos

`DiagnosticTolerances` **não tem valores padrão**: uma verificação só é tão significativa quanto o perfil de que vem, e `None` a desliga. Um valor fora da tolerância vira um `DiagnosticFinding` com código, severidade, mensagem, o valor observado e a tolerância:

| Código | Severidade | Quando |
| --- | --- | --- |
| `POSE_TIME_DELTA_EXCEEDS_TOLERANCE` | aviso | a pose usada está mais longe no tempo do frame que a tolerância |
| `FRAME_OUTSIDE_MAP_TIME_WINDOW` | aviso | o frame está fora da janela de aquisição da geometria além da tolerância (antes ou depois) |
| `REPROJECTION_RESIDUAL_EXCEEDS_TOLERANCE` | aviso | o p95 do resíduo passa da tolerância |
| `REPROJECTION_INVALID_RATE_EXCEEDS_TOLERANCE` | aviso | a parcela de correspondências não projetáveis passa da tolerância |
| `NOTHING_VISIBLE_IN_IMAGE` | falha | nenhum ponto do mapa está visível no suporte da imagem preparada |
| `NO_REFERENCE_CORRESPONDENCE_PROJECTS` | falha | há referência, mas nenhuma correspondência projeta |

`FrameDiagnostics.failed` indica se algum achado é falha. Uma calibração inválida não chega aqui: `camera_projection_for` a recusa antes (`CalibrationError`), e o `FrameProjector`, um frame com calibração ou pose incoerentes (`AssociationInputError`).

## Varredura de deslocamento temporal (só diagnóstico)

`time_offset_sweep(projector, observation, prepared_image, correspondences, offsets_ns)` re-projeta a geometria de referência com a pose buscada no timestamp deslocado e devolve, por deslocamento, as estatísticas de reprojeção (ou o motivo pelo qual o lookup recusou o instante). Ela **nunca** altera a calibração nem um timestamp: a observação é imutável e a varredura só relata. Agir sobre o resultado é uma decisão explícita e separada. Não há otimização automática de calibração.

## Caminhos de features nativo e melhorado

A parte de geometria e calibração do relatório independe do caminho de features densas, então uma execução nativa e uma melhorada sobre as mesmas entradas relatam valores idênticos ali. O relatório acrescenta um `DenseSamplingSummary` por mapa denso amostrado (feature, artefato, espaço, política de interpolação, geometria de amostragem, se é melhorado e de qual feature nativa, e as contagens elegíveis, amostradas e fora do suporte), o que permite compará-las sob o mesmo diagnóstico.

## Sobreposições

A geração de imagens de sobreposição (RGB, projeção, profundidade, visibilidade e regiões) é evidência de depuração do artifact de run (#103): este módulo fornece os dados por ponto (pixel preparado, estado, profundidade e apoio), e nada a jusante depende desses arquivos.

## Como é verificado

Cenas sintéticas de resposta conhecida: registro de todas as fontes, resíduo contra uma referência com deslocamentos conhecidos (média, mediana, p95 e máximo, com um resíduo em ambos os eixos), correspondências não projetáveis, referência ausente e inteiramente inválida, um **fixture de calibração desalinhada** (o extrínseco com 2° de erro de guinada) sinalizado pelo resíduo, timing desalinhado e frame fora da janela do mapa (antes e depois), varredura com o deslocamento que alinha a referência e um instante recusado, imutabilidade, comparabilidade nativo × melhorado e o relatório em JSON. Mutações que medem só um eixo, trocam o p95 pelo máximo, deixam de contar inválidos, removem a checagem de timing, ignoram o deslocamento na varredura ou zeram frames anteriores à janela fazem testes falharem.

# Correção de movimento (deskew) em Geometric Mapping

Este documento descreve `src/contextmap/geometric_mapping/motion_correction.py`.

Um scan de LiDAR é adquirido ao longo de um intervalo de tempo não nulo enquanto a plataforma pode estar em movimento. Tratar todos os pontos como capturados em um único instante pode deformar o mapa. Por outro lado, nenhum consumidor pode **assumir** que um scan foi corrigido só porque um certo estimador foi usado.

## Os três estados

| Estado | Significado |
| --- | --- |
| `RAW` | declarado como **não** corrigido para o movimento da plataforma |
| `CORRECTED` | corrigido, com evidência de quem corrigiu, com qual trajetória, configuração e payload |
| `UNKNOWN` | nada foi declarado; é o **padrão** e nunca é inferido |

`UNKNOWN` é honesto: não é `RAW` nem `CORRECTED`. Nenhum campo de `SourceObservation`, tipo de fonte, sensor ou estimador muda o estado. Em particular, **não existe** a suposição `FAST-LIO ⇒ deskewed=true`: um scan cuja proveniência diz `source_type="fast_lio"` continua `UNKNOWN` até que uma declaração explícita o mude. `unknown_motion_correction(observation)` e `declared_raw_motion_correction(observation)` criam os registros; só o segundo afirma algo, e só para fontes que o operador sabe entregar scans crus.

## `MotionCorrectionRecord`

Registro por scan, com identidade preservada (`observation_id` é a observação física de origem; a observação **nunca** é reescrita).

| Campo | Significado |
| --- | --- |
| `observation_id` | observação de origem |
| `state` | `RAW`, `CORRECTED` ou `UNKNOWN` |
| `acquisition_start` / `acquisition_end` | intervalo de aquisição, quando conhecido (ambos ou nenhum, mesmo clock, fim não anterior ao início) |
| `per_point_timing_available` | se o payload carrega tempo por ponto |
| `evidence` | suporte de uma afirmação `CORRECTED`; ausente nos demais estados |

Regras validadas na construção:

- um scan `CORRECTED` **exige** evidência e exige declarar o tempo que usou (intervalo de aquisição ou tempo por ponto). O tempo **nunca é fabricado**: sem ele, a afirmação é recusada;
- evidência em um scan `RAW` ou `UNKNOWN` é recusada: não se pode carregar um argumento de correção em um scan que não a tem;
- o intervalo de aquisição deve ser completo, ordenado e em um único clock.

`verify_motion_correction(record, observation)` confere o registro contra o scan e devolve uma lista de problemas legíveis: observação diferente, clock do scan diferente do clock do intervalo, timestamp do scan fora do intervalo declarado e, para um registro `CORRECTED`, `payload_hash` da evidência diferente do SHA-256 do payload realmente entregue pelo scan (#595; a mensagem cita os dois hashes). Lista vazia significa consistente.

## `MotionCorrectionEvidence`

O que sustenta uma afirmação de correção, para que ela seja rastreável até a origem exata:

- `producer`: implementação/backend que corrigiu (não vazio);
- `trajectory_id`: trajetória (fonte de estado) usada;
- `payload_hash`: `sha256:<64 hex>` do payload corrigido;
- `configuration_fingerprint`: hash da configuração de correção, quando existe.

A evidência não impõe um estimador: o produtor pode ser um backend de State Estimation ou uma implementação dedicada de preparação de geometria. O contrato só define o estado canônico e as regras de integração.

## Política de scans não corrigidos

`MotionCorrectionPolicy(raw=..., unknown=...)` decide, por estado, o que uma execução faz com o scan:

| `ScanDisposition` | Efeito |
| --- | --- |
| `ACCEPT` | usa o scan |
| `WARN` | usa o scan e registra um aviso |
| `REJECT` | deixa o scan de fora e registra o motivo |

Um scan `CORRECTED` é sempre aceito. `raw` e `unknown` **não têm valor padrão**: uma execução não deixa entrar um scan cru ou desconhecido por omissão. `apply_motion_correction_policy(record, policy)` devolve um `MotionCorrectionVerdict` com a disposição e, quando não é `ACCEPT`, uma mensagem que nomeia o scan e o estado.

A disposição "corrigir quando suportado" citada na issue **não existe**: nenhum corretor está implementado neste repositório, e criar a opção sem consumidor violaria YAGNI. Quando existir um corretor, ela entra como uma nova disposição com o seu teste.

## Estado por ponto e serialização

`GeometryPointProvenance.motion_correction` carrega o estado do scan de origem em **cada ponto**, com `UNKNOWN` como padrão, então um leitor a jusante distingue geometria crua de geometria corrigida sem consultar outra estrutura. O campo é aditivo: os contratos de #89 continuam válidos.

`serialization.py` codifica e decodifica o registro (`encode_motion_correction_record`/`decode_motion_correction_record`), a política (`encode_motion_correction_policy`/`decode_motion_correction_policy`) e o estado por ponto (`provenance.motion_correction`), com apenas primitivas JSON. A decodificação revalida as regras acima. Registrar a política e um registro por scan mapeado no manifesto de execução é responsabilidade do `GeometricMapArtifact`, que ainda **não existe**: este módulo só fornece as formas serializáveis.

## Limitações

- Nenhum corretor de movimento é implementado: o módulo define o estado, a evidência e a política. Um scan só é `CORRECTED` quando alguém declara e fornece a evidência.
- O tempo por ponto é apenas um indicador (`per_point_timing_available`); a extração desse tempo do payload é responsabilidade de quem corrige.

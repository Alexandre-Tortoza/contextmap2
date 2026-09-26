# Sincronização e agrupamento temporal

Este documento descreve `synchronize()` (`src/contextmap/ingestion/synchronization.py`): como observações de modalidades diferentes, com taxas e timestamps distintos, são agrupadas em `ProcessingObservation`s determinísticas e auditáveis.

## Timestamp de origem vs. normalizado

Cada `SourceObservation` já carrega seu `timestamp: SourceTimestamp` original (segundos + nanossegundos + `clock_id`), definido pela issue #38 — esse valor nunca é modificado pela sincronização. Comparações e offsets usam `timestamp.total_nanoseconds()`, um inteiro exato comparável **somente entre observações que compartilham o mesmo `clock_id`**. Isso evita perda de precisão de ponto flutuante em timestamps epoch grandes.

## Limitação conhecida do v0: um domínio de clock por comparação

O v0 **não realiza reconciliação entre domínios de clock diferentes** (ex.: converter `ros1_bag:/camera` para o mesmo eixo de `ros1_bag:/imu` quando eles não usam o mesmo `clock_id`). Isso é uma decisão deliberada, não uma omissão: a documentação global (`docs/shared-primitives.md`) é explícita — "nunca assuma que dois números de timestamp compartilham um clock". Assumir implicitamente que dois `clock_id` diferentes são comparáveis seria exatamente o erro que essa regra proíbe.

Nos adapters ROS, todos os `header.stamp` da mesma fonte usam o `timestamp_clock_id` compartilhado configurado. O timestamp de gravação do bag é outra evidência e fica em `SourceProvenance.raw_metadata`; ele não define automaticamente o clock do sensor. Quando um candidato tem `clock_id` diferente do anchor, ele nunca é selecionado e aparece em diagnostics com status/motivo `clock_id_mismatch` — nunca é comparado silenciosamente. Reconciliação explícita entre múltiplos domínios de clock (offset conhecido, clock skew) não é escopo desta issue; se necessária no futuro, deve ser uma decisão própria, versionada, não um efeito colateral da sincronização.



## Fluxo da decisão de associação

```mermaid
flowchart TD
    A[Anchor da reference_modality] --> C[Candidatos de outra modalidade]
    C --> K{clock_id compatível?}
    K -- não --> X[Sem associação: clock_id_mismatch]
    K -- sim --> N[Escolher menor abs(offset_nanoseconds)]
    N --> T{offset dentro da tolerance?}
    T -- sim --> M[matched]
    T -- não --> O[Sem associação: outside_tolerance]

    X --> D[SynchronizationDiagnostics]
    M --> D
    O --> D
    M --> G[ProcessingObservation]
    X --> G
    O --> G
```

A decisão é reproduzível a partir do anchor, do domínio temporal, da tolerância e dos candidatos. Não há conversão implícita entre clocks nem interpolação escondida.

## Política v0: nearest-within-tolerance

Única política implementada nesta issue (`policy="nearest_within_tolerance"` em `ProcessingObservation`):

1. As observações da modalidade `reference_modality` (configurada) são ordenadas por `(total_nanoseconds, observation_id)` — ordem determinística mesmo com timestamps duplicados.
2. Cada observação da modalidade de referência vira um **anchor** e recebe um `frame_index` sequencial (0, 1, 2, …).
3. Para cada outra modalidade, seleciona-se o candidato com menor `|offset|` em relação ao anchor, **entre os que compartilham o `clock_id` do anchor** e cujo `|offset| <= tolerance_nanoseconds`. Não havendo candidato, a associação é `ModalityAssociation(observation=None, offset_nanoseconds=None)` — ausência explícita, nunca um valor sentinela.
   Empates de `|offset|` vão para o menor `observation_id`. Os candidatos de cada modalidade são ordenados **uma vez** por domínio de clock, por `(total_nanoseconds, observation_id)`, e cada anchor os consulta por bisseção: só o primeiro candidato da sequência de timestamps iguais imediatamente anterior e o da imediatamente posterior disputam, o que dá o mesmo resultado da varredura completa em O((A + C) log C) em vez de O(A × C).
4. Um candidato pode ser selecionado por mais de um anchor (seleção não é exclusiva); isso é intencional e corresponde ao caso comum de uma modalidade de alta frequência (ex.: IMU) sendo consultada por vários anchors de baixa frequência (ex.: câmera).
5. **Nenhuma interpolação é realizada.** Uma associação é sempre "este candidato, neste offset" ou "nenhum candidato" — nunca um valor interpolado. Uma política de interpolação futura precisaria de um `policy` próprio e documentar explicitamente onde a decisão de interpolar fica registrada; o v0 não a implementa.

## Diagnostics

`SynchronizationDiagnostics.decisions` registra uma decisão por `(anchor, modalidade)`, incluindo o candidato escolhido, `offset_nanoseconds` e status (`matched`, `no_candidate`, `clock_id_mismatch` ou `outside_tolerance`). `dropped_events` lista toda observação de uma modalidade não-referência que nunca foi selecionada por nenhum anchor, com o motivo:

- `"clock_id_mismatch"` — nunca compartilhou `clock_id` com nenhum anchor;
- `"no_anchor_within_tolerance"` — compartilhou `clock_id` com ao menos um anchor, mas nunca foi o candidato mais próximo dentro da tolerância configurada.

## Exemplo

```text
reference_modality = "image"
tolerance_nanoseconds = 50_000_000

images:  t=1.00  t=2.00
imu:     t=0.99  t=1.98  t=3.50

frame_index=0  anchor=image@1.00  imu → imu@0.99 (offset=-10_000_000 ns)
frame_index=1  anchor=image@2.00  imu → imu@1.98 (offset=-20_000_000 ns)

dropped: imu@3.50 (reason="no_anchor_within_tolerance", nenhum anchor a menos de 0.05s)
```

## O que esta issue não define

- Persistência do resultado de sincronização no artefato de sequência (`SequenceArtifactWriter` continua indexando `SourceObservation`s individuais; agrupar/persistir `ProcessingObservation`s é decisão de uso, não desta issue).
- Seleção/replay por frame/tempo — issue #42.
- Calibração/frames de coordenadas — issue #41.

# Política de normalização de timestamp por dataset

Este documento descreve `src/contextmap/ingestion/timestamp_policy.py` (issue #554): como um dataset específico pode declarar uma correção explícita de clock sem que isso vaze para o adapter de fonte, para outro dataset, ou para ingestão ao vivo/streaming.

## O problema

Algumas fontes gravadas têm um clock de header/sensor cuja progressão relativa é utilizável para sincronização mesmo com o **epoch absoluto errado** — por exemplo, um driver de câmera cujo relógio de sistema nunca foi ajustado, carimbando cada mensagem em algum ponto de 2001, enquanto a gravação (`recording_time`) é de 2026. Corrigir isso a partir do tipo de fonte (ex.: "é ROS 1, então..."), ou por heurística (ex.: "o ano parece errado, usar o ano da gravação"), esconderia uma decisão específica de dataset dentro de código genérico de adapter — exatamente o que este módulo e `docs/adapters.md` rejeitam.

## Quatro noções sempre separadas

- **tempo de gravação bruto** — o clock de captura do próprio bag, já carregado em `SourceProvenance.raw_metadata["bag_timestamp_nanoseconds"]` para os adapters de bag; este módulo nunca toca nisso.
- **timestamp de fonte/sensor bruto** — o valor que um `SourceAdapter` decodificou em `timestamp` antes de qualquer política deste módulo rodar.
- **tempo de evento normalizado/derivado** — o mesmo campo, depois de `apply_timestamp_policy()` substituí-lo; o valor bruto substituído é preservado em `provenance.raw_metadata`, nunca descartado.
- **clock usado para resolver janelas temporais** — sempre o próprio clock de gravação da fonte hoje (`SourceAdapterConfig.resolved_window_clock_id()`, ver [`adapters.md`](adapters.md)), nomeado explicitamente por `TimestampPolicy.window_clock` para viajar junto do resto da política num único registro auditável em vez de ficar implícito.

## `TimestampPolicy`

Selecionada por `IngestionRequest.timestamp_policy` (issue #554), nunca inferida do adapter. O padrão (`DEFAULT_TIMESTAMP_POLICY`) é equivalente a nenhuma correção — uma fonte não configurada, ou uma ingestão ao vivo/streaming, se comporta exatamente como antes deste módulo existir:

```python
from contextmap.ingestion import ConstantOffsetCorrection, TimestampPolicy

# Uma fonte com epoch de header errado em ~25 anos, offset derivado de âncoras conhecidas.
policy = TimestampPolicy(
    correction=ConstantOffsetCorrection.from_anchors(
        source_time=wrong_epoch_reading,  # SourceTimestamp lido do clock afetado
        reference_time=known_correct_reading,  # SourceTimestamp correto para o mesmo instante
    )
)
```

## O único modelo de correção suportado: offset constante

```text
event_time = source_time + offset
```

Nenhum termo afim, compensação de deriva ou estimativa online — a issue #554 rejeita explicitamente isso sem evidência de que um offset constante seja insuficiente (AGENTS.md, YAGNI). O offset é calculado em nanossegundos exatos (`SourceTimestamp.total_nanoseconds()`), nunca em ponto flutuante de segundos, para que um offset de anos não acumule erro.

`ConstantOffsetCorrection.from_offset_seconds(...)` aceita um offset informado diretamente; `ConstantOffsetCorrection.from_anchors(source_time=..., reference_time=...)` deriva o offset exato de um par (leitura errada, leitura correta) do mesmo instante físico.

## Onde a correção é aplicada

`apply_timestamp_policy()` roda em `IngestionService._read()` (`contextmap.runtime.ingestion_service`), depois que o adapter decodifica cada `SourceObservation` e antes de qualquer validação, sincronização ou escrita — o `timestamp` publicado, sincronizado e validado já é o tempo de evento normalizado. O adapter em si nunca vê a política: continua fiel à fonte, sem heurística de reparo de epoch/ano, satisfazendo a fronteira descrita em [`adapters.md`](adapters.md).

O valor bruto nunca é descoberto silenciosamente: `apply_timestamp_policy()` grava `source_time_before_correction_seconds`/`source_time_before_correction_nanoseconds` em `provenance.raw_metadata` antes de substituir `timestamp`. Sem correção configurada, a observação retorna inalterada (mesma instância) — o caminho padrão continua byte-idêntico ao que era antes desta issue.

A identidade do clock (`timestamp.clock_id`) não muda: só o valor é normalizado. Duas fontes distintas continuam distinguíveis pelo próprio `clock_id` derivado de `source_type`/`path`; uma correção de uma não pode, por construção, alcançar a outra.

## Diagnóstico antes de aceitar uma correção

`diagnose_source_clock()` computa, sobre as observações **brutas** (antes ou depois de `apply_timestamp_policy()` — o resultado é o mesmo, já que a função reconstrói o valor bruto a partir da proveniência quando uma correção já foi aplicada):

- timestamps de fonte ausentes (convenção ROS de "não definido": `seconds == 0 and nanoseconds == 0`);
- não-monotonicidade do clock bruto (reset/retrocesso);
- distribuição de `recording_time - source_time` (média/desvio padrão), quando `recording_time` está disponível;
- resíduo de `(recording_time - source_time) - offset_configurado`, quando uma correção é dada.

Nada disso decide silenciosamente reinterpretar um clock: `TimestampCorrectionDiagnostics.warnings()` produz avisos legíveis, sempre visíveis na proveniência publicada (`SequenceProvenance.warnings`) — nunca escondidos atrás de uma correção aplicada sem ressalva. A não-monotonicidade do clock corrigido também continua sujeita à checagem genérica já existente, `validate_timestamp_ordering()` (ver [`validation.md`](validation.md)), que por padrão (`ValidationPolicy.on_problems == "fail"`) recusa publicar a sequência.

## Proveniência e reprodutibilidade

`encode_timestamp_policy()`/`decode_timestamp_policy()` serializam a política inteira (clocks nomeados + tipo/parâmetros da correção + âncoras, quando houver) dentro de `SequenceProvenance.ingestion_config["timestamp_policy"]` — o mesmo mecanismo genérico já usado para tópicos, janela e sincronização, não um campo novo no schema do manifest. Como `ingestion_config` já alimenta `configuration_hash`/`compute_content_identity()`, uma mudança na política muda a identidade de conteúdo da sequência, exatamente como uma mudança de calibração ou sincronização já mudava antes desta issue: mesma fonte + mesma configuração produz sempre o mesmo tempo normalizado e a mesma identidade de seleção.

## Janela e evento são clocks independentes

Uma janela (`SourceWindow`, ver [`adapters.md`](adapters.md)) continua resolvida inteiramente pelo próprio clock de gravação da fonte, dentro do adapter — nunca afetada por `TimestampPolicy`. Um dataset pode, portanto, usar `recording_time` para selecionar um trecho estável da gravação e, ao mesmo tempo, ter seu `header_stamp` corrigido como clock de evento/sincronização: as duas configurações coexistem no mesmo `IngestionRequest` sem interferência, cada uma resolvida pelo mecanismo que já era responsável por ela.

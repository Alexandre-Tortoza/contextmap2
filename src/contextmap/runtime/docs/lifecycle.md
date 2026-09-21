# Ciclo de vida de um run

Execuções de pesquisa longas falham por dependência opcional ausente, falta de memória, entrada corrompida, bloqueio de calibração ou interrupção. O runtime dá a cada run um **ciclo de vida explícito** e um **rastro estruturado**, e permite retomar sem nunca mutar um artifact científico já publicado.

## Estados

| Estado | Significado |
|---|---|
| `planned` | o run existe e tem plano; nada começou |
| `running` | estágios executando (ou o processo morreu sem registrar o fim; ver *interrompido*) |
| `completed` | todos os estágios terminaram e `execution.json` foi gravado |
| `failed` | um estágio ou o runner falhou; o registro de falha diz onde e por quê |
| `blocked` | o preflight recusou iniciar; nada rodou |
| `cancelled` | cancelamento cooperativo ou interrupção (Ctrl+C) parou o run |

Um run **interrompido** é um run que nunca chegou a um estado terminal e cujo processo dono já não existe (o que sobra de um `kill -9`): `status.json` diz `running`, o lock aponta para um pid morto, e `RunSummary.interrupted` é verdadeiro. Um run que ainda está vivo (lock de um pid vivo) não é interrompido e não pode ser retomado.

## O diretório do run

`<workspace>/runtime/run-NNNN/`, alocado atomicamente (nunca dois runs no mesmo diretório):

| Arquivo | Conteúdo |
|---|---|
| `effective_config.json`, `plan.json` | o que foi pedido e como foi resolvido (o plano é gravado quando não tem problema estrutural) |
| `events.jsonl` | eventos estruturados, **append-only**, um objeto JSON por linha, gravados e sincronizados em disco antes de o status mudar |
| `status.json` | o estado atual, **substituído atomicamente** a cada transição, com o registro de falha, o ambiente e a identidade do código |
| `execution.json` | entradas e saídas exatas; **só existe** em um run concluído |
| `run.lock` | o pid do processo dono, enquanto o run não terminou |

Um run `failed`, `cancelled` ou `blocked` é **histórico**: nunca é modificado depois. Nenhum run sobrescreve outro e uma falha nunca toca em um artifact publicado por um run anterior.

## Eventos

Cada evento tem `sequence` (a partir de 1, sem lacunas), `time` (UTC), `kind`, `stage_id` e `data`:

| `kind` | Quando | `data` |
|---|---|---|
| `run_planned` | início | digests do plano e da configuração, estágios a rodar, artifacts fornecidos |
| `run_resumed` | ao retomar | run retomado, seu estado, estágios que ele havia concluído |
| `run_blocked` | preflight recusou | os problemas (caminho e mensagem) |
| `run_started` | preflight passou | — |
| `stage_started` | antes de executar | ids exatos das entradas, digest da configuração do estágio |
| `stage_reused` | artifact anterior reutilizado | o artifact exato e a decisão de reuso |
| `stage_completed` | estágio terminou | o artifact produzido, a decisão de reuso, `elapsed_s` |
| `stage_failed` | estágio falhou | categoria, mensagem, tipo da exceção, `elapsed_s` |
| `run_failed` | o run falhou | categoria, mensagem, tipo, estágios concluídos (o `stage_id` do evento é o estágio) |
| `run_cancelled` | cancelamento | motivo, estágios concluídos |
| `run_completed` | fim | ordem, `elapsed_s`, relatório de retomada |

`run_plan` emite os eventos para o `journal` e para um `events` (qualquer `EventSink`, por exemplo uma TUI); um sink não muda o que o run faz. Um estágio é executado **uma única vez**: não há retry escondido, e nada troca de backend ou de dispositivo em silêncio.

## Registro de falha

`FailureRecord` (em `status.json` e em `run_failed`): estágio, categoria, mensagem, tipo da exceção e estágios já concluídos. A categoria vem da **exceção**, nunca de um palpite sobre o texto:

| Categoria | Origem |
|---|---|
| `dependency` | `ImportError` |
| `resource` | `MemoryError` |
| `contract` | o estágio devolveu um artifact de outro tipo/estágio |
| `execution` | qualquer outra exceção do estágio |
| `unexpected` | o próprio runner falhou fora de um estágio (por exemplo o índice de reuso ilegível) |
| a do executor | um executor que levanta `StageFailure(mensagem, category="...")` define a sua |

Uma exceção inesperada fora de um estágio também vira `run_failed` (categoria `unexpected`) antes de ser relevantada, então o run nunca fica em `planned`/`running` por causa de um erro do runner. Interrupção (`KeyboardInterrupt`) vira `run_cancelled` (motivo `interrupted`) e é relevantada.

## Cancelamento cooperativo

`CancellationToken.cancel(motivo)` é checado **antes de cada estágio**: o runner nunca interrompe um estágio pelo meio, então um run cancelado para numa fronteira em que todo artifact produzido está completo. Levanta `RunCancelledError` com o estágio que ia começar e os concluídos.

## Segredos

Nada de segredo entra nos eventos, no status ou na mensagem de erro: `run_plan(redact=...)` passa toda string de todo evento (e a mensagem do `StageExecutionError`) por `ResolvedSecrets.redact`, que troca cada valor mantido por `***` sem nunca expô-lo. A CLI o liga automaticamente.

## Ambiente para reprodução

`status.json` registra `capture_environment`: versão do `contextmap`, Python, sistema, arquitetura, número de CPUs, o dispositivo e a **versão instalada** dos pacotes opcionais que os backends selecionados exigem (lida dos metadados, sem importar nada). Sem nome de host, usuário nem variável de ambiente. Junto vêm a identidade do código e, em cada estágio, o `elapsed_s`.

## Leitura e integridade

`read_run(diretório)` devolve um `RunSummary` (estado, `interrupted`, falha, problemas de bloqueio, estágios concluídos, eventos, ambiente) e confere a coerência:

- o `status.json` é a autoridade e o log é o rastro; uma **última linha parcial** (o processo morreu ao gravá-la) é ignorada e reportada (`truncated`);
- linha corrompida em qualquer outro ponto, **lacuna na numeração**, status ilegível ou outra versão de schema são `RunRecordError`;
- incoerências que não impedem a leitura viram `notes` (run concluído sem `execution.json`, log terminal com status não terminal).

## Retomada

`resume_plan(run_anterior, execução, executores, reuse=...)` retoma um run `failed`, `cancelled` ou interrompido **como um run novo**, sem alterar o anterior:

- só é aceito o **mesmo pedido**: a mesma topologia, a mesma configuração efetiva, os mesmos estágios a rodar e os mesmos artifacts fornecidos. Qualquer mudança é `ResumeError` (um run novo ainda reutiliza tudo o que não mudou pelo índice);
- recusa-se um run `completed` (nada a retomar), `planned`/`blocked` (nada rodou) e um run ainda vivo;
- exige uma `ReusePolicy`: os estágios concluídos só são reutilizados se **as checagens normais de reuso passam** (artifact de mesma identidade indexado e ainda verificável). O que falha uma checagem é **recomputado e reportado**;
- uma saída parcial ou temporária nunca é indexada, então **nunca** é confundida com um estágio concluído: o estágio interrompido no meio é sempre recomputado;
- o registro de execução do novo run traz `resume`: qual run, e quais dos estágios concluídos antes foram `reused` e quais `recomputed`.

Resumir é reuso de orquestração, não mutação de artifacts científicos.

## CLI

`run`/`stage` criam o diretório do run antes de executar e o journal grava tudo; falhas, bloqueios e cancelamentos deixam registro (a mensagem de erro aponta `run record: <diretório>`). `--reuse-index DIR`, `--code-identity ID`, `--force ESTÁGIO` e `--resume RUN` expõem o reuso e a retomada; o `verify` do índice é fornecido a `main(verifier=...)` por quem possui os executores. `inspect run DIR [--events]` mostra o ciclo de vida e `validate DIR` confere um registro de run. Interrupção sai com `130`. Detalhes em [`cli.md`](cli.md).

## Lacunas conhecidas

- **Executores e verificador reais.** O `verify` do índice e o `content_hash` dos artifacts vêm dos leitores e manifests de cada capability e acompanham os executores reais (#177); sem eles a retomada real só é exercitada com estágios de teste.
- **Sem serviço de longa duração.** O lock é um arquivo com o pid: detecta um dono morto no mesmo host; em outro host ou sistema de arquivos compartilhado ele não é confiável.
- **Sem recuperação automática.** Não há retry, fallback de backend ou de dispositivo; quem retoma decide, explicitamente.

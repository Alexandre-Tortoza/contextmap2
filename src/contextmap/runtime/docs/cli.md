# Linha de comando

A CLI é uma camada **fina** sobre o runtime: ela interpreta argumentos, chama os serviços e imprime o que voltou. Não há ramo por backend, lógica científica, parsing de dataset, matemática de projeção nem regra de fusão, entidade ou relação. Cada flag vira um **override de configuração**, então a CLI nunca é um segundo lugar que decide o que roda.

```text
contextmap run     [opções] [--stage ESTÁGIO]... [--dry-run]
contextmap stage   ESTÁGIO [opções] [--dry-run]
contextmap inspect config | plan   [opções]
contextmap inspect artifact CAMINHO
contextmap validate CAMINHO
```

Também funciona como `python -m contextmap ...`. O ponto de entrada de console é `contextmap = contextmap.runtime.cli:main`.

## Opções de configuração

| Flag | Vira |
|---|---|
| `-c/--config ARQUIVO` (repetível) | camada de arquivo `.json`/`.toml`; a última vence |
| `--profile ID` | perfil base (padrão `canonical/1`) |
| `--set CAMINHO=VALOR` (repetível) | override direto |
| `--sequence ID` | `inputs.sequence` |
| `--select ESTÁGIO=REF[,REF...]` | `inputs.selections.<estágio>`: ids exatos, `latest` ou `named:<nome>` |
| `--catalog ARQUIVO` | catálogo dos runs disponíveis para selecionar |
| `--workspace DIR` | `resources.workspace` |
| `--device DEVICE` | `resources.device` |
| `--debug-level {none,standard,full}` | `policies.debug_level` |
| `--json` | saída legível por máquina |

Precedência: perfil < arquivos < `--set` < flags. A seleção continua explícita: um estágio sem `--select` nunca é escolhido implicitamente, e `--select` exige `--catalog` (o arquivo é uma entrada explícita; nada é descoberto por varredura de diretório). O formato do catálogo está em [`selection.md`](selection.md).

## Comandos

### `run` e `stage`

Resolvem a configuração, derivam o plano, escopam (o pipeline completo, ou os alvos de `--stage`; `stage` é exatamente um alvo), validam o preflight e executam.

- **`--dry-run`**: mostra a configuração efetiva com o digest, a topologia resolvida (entradas, saídas e backends de cada estágio, o que roda, o que é fornecido por seleção, o que é indisponível e por quê), a seleção resolvida, o preflight e a cobertura de executores. **Não carrega modelo, não executa e não escreve nada.** Sai com `0` se o preflight passa e `1` se está bloqueado.
- **execução real**: exige `--workspace` (ou `resources.workspace`), porque persiste seus registros. Cria `<workspace>/runtime/run-NNNN/` **antes** de executar e o journal grava ali o plano, os eventos, o status e, ao concluir, `execution.json` ([`lifecycle.md`](lifecycle.md)). Cada run tem seu próprio diretório e nunca sobrescreve outro. Um preflight bloqueado, um estágio que falha ou uma interrupção deixam um registro inspecionável (`run record: <diretório>` na mensagem de erro); Ctrl+C sai com `130`.
- **reuso e retomada** (`--reuse-index DIR`, `--code-identity ID`, `--force ESTÁGIO`, `--resume RUN`): o reuso combina a identidade exata dos estágios; `--resume` retoma um run falhado, cancelado ou interrompido como um run novo. Exigem um verificador de artifacts que o dono dos executores fornece a `main(verifier=...)`: a CLI não sabe se um artifact indexado ainda existe.

### `inspect`

- `inspect config`: a configuração efetiva, o digest e as camadas que a produziram;
- `inspect plan`: a topologia resolvida, sem executar nada;
- `inspect run DIR [--events]`: o ciclo de vida de um run (estado, falha, problemas de bloqueio, estágios concluídos, eventos, ambiente);
- `inspect artifact CAMINHO`: um resumo do `manifest.json` de um diretório de artifact (id, schema, criação, número e tamanho dos arquivos) e o resultado da checagem de integridade, ou um documento do runtime.

### `validate CAMINHO`

Verifica a integridade: em um registro de run (`status.json`), o log de eventos (linha corrompida, lacuna na numeração), a coerência com o status e o digest de cada documento; em um diretório de artifact, cada arquivo do `file_inventory` (existência, tamanho e SHA-256; um caminho que sai do artifact é recusado); em `effective_config.json`, `plan.json` e `execution.json`, a versão de schema e o digest. Sai com `1` se algo falha e nomeia o arquivo ou o problema. Usa só a biblioteca padrão: não importa NumPy, modelo nem ROS.

## Erros acionáveis

Todo problema sai com o caminho e o que fazer: perfil desconhecido (lista os conhecidos), configuração inválida (todos os problemas de uma vez), módulo opcional ausente (com a dica de instalação), segredo ausente (pelo nome), seleção incompatível (o que diverge, com os ids), estágio sem capability, estágio sem executor. Com `--json`, uma falha sai como `{"ok": false, "error": ..., "problems": [...]}` na saída padrão.

Códigos de saída: `0` sucesso; `1` pedido entendido, mas não atendível (configuração inválida, preflight bloqueado, estágio que falha, integridade); `2` uso incorreto da linha de comando (por exemplo, execução real sem workspace).

## Fronteira

A CLI não faz rede, não abre dashboard e não usa "todos os runs". Importar `contextmap.runtime.cli` não carrega SDK pesado (torch, transformers, rosbags, NumPy); um teste garante isso.

## Executores

Os executores dos estágios reais **não** estão empacotados: `main(argv, executors=...)` os recebe de quem chama (os testes usam estágios falsos). Sem um executor, o preflight de uma execução real bloqueia com `no executor is registered for it` e nada roda; um dry-run não precisa de executores e informa quais faltam.

## Lacunas conhecidas

- **Execução real do canônico.** O comando existe e é testado de ponta a ponta com executores falsos, mas os executores das capabilities reais (que precisam das políticas de Geometric Mapping e Sensor Association e dos hashes de conteúdo dos manifests) acompanham a validação end-to-end (#177). Além disso, o pipeline completo continua bloqueado pelos estágios das milestones #12–#15; hoje o caminho executável é um subgrafo (`--stage`).
- **`ingest`.** O comando de ingestão chega com o serviço público de ingestion (issue #263), que a CLI deve chamar em vez de recompor adapters, validação e sincronização.
- **`export`.** Não há artifact a exportar antes das milestones #15/#16 (`ContextMapArtifact`).
- **Verificador de artifacts.** As flags de reuso e retomada existem, mas o `verify` do índice vem do dono dos executores reais (#177); a CLI recusa `--reuse-index` sem ele.
- **API pública.** A CLI chama as funções do runtime diretamente; a fachada pública frontend-neutra (#264) passa a ser o que ela consome.

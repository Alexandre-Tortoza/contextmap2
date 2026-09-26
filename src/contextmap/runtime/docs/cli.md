# Linha de comando

A CLI é uma camada **fina** sobre o runtime: ela interpreta argumentos, chama os serviços e imprime o que voltou. Não há ramo por backend, lógica científica, parsing de dataset, matemática de projeção nem regra de fusão, entidade ou relação. Cada flag vira um **override de configuração**, então a CLI nunca é um segundo lugar que decide o que roda.

```text
contextmap ingest  --source PATH --sequence-name NOME --topic CHAVE=TÓPICO ... [--preflight]
contextmap run     [opções] [--stage ESTÁGIO]... [--dry-run]
contextmap stage   ESTÁGIO [opções] [--dry-run]
contextmap inspect config | plan   [opções]
contextmap inspect artifact CAMINHO
contextmap validate CAMINHO [--full]
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

### `ingest`

Ingestion canônica pelo [serviço público de ingestion](ingestion-service.md): `--preflight` só confere o pedido; sem ele lê, valida, sincroniza e publica um `SequenceArtifact` imutável em `--output-dir DIR` (o diretório final, que não pode existir; num run, o estágio `ingestion` publica em `<run>/ingestion` pelo mesmo serviço). O adapter **não** é uma flag: vem do backend selecionado em `components.ingestion.source_adapter.backend` e é composto pela composition root (módulo opcional ausente falha com a dica de instalação). O progresso sai em stderr e o resultado em stdout (ou JSON, com os eventos); Ctrl+C sai com `130` sem publicar nada.

### `run` e `stage`

Resolvem a configuração, derivam o plano, escopam (o pipeline completo, ou os alvos de `--stage`; `stage` é exatamente um alvo), validam o preflight e executam.

- **`--dry-run`**: mostra a configuração efetiva com o digest, a topologia resolvida (entradas, saídas e backends de cada estágio, o que roda, o que é fornecido por seleção, o que é indisponível e por quê), a seleção resolvida, o preflight e a cobertura de executores. **Não carrega modelo, não executa e não escreve nada.** Sai com `0` se o preflight passa e `1` se está bloqueado.
- **execução real**: exige `--workspace` (ou `resources.workspace`), porque persiste seus registros. Exige `inputs.sequence` (o dataset) e cria `<workspace>/<dataset>/run-NNNN/` **antes** de executar e o journal grava ali o plano, os eventos, o status e, ao concluir, `execution.json` ([`lifecycle.md`](lifecycle.md)). Cada run tem seu próprio diretório e nunca sobrescreve outro. Um preflight bloqueado, um estágio que falha ou uma interrupção deixam um registro inspecionável (`run record: <diretório>` na mensagem de erro); Ctrl+C sai com `130`.
- **reuso e retomada** (`--reuse-index DIR`, `--code-identity ID`, `--force ESTÁGIO`, `--resume RUN`): o reuso combina a identidade exata dos estágios; `--resume` retoma um run falhado, cancelado ou interrompido como um run novo. Exigem um verificador de artifacts que o dono dos executores fornece a `main(verifier=...)`: a CLI não sabe se um artifact indexado ainda existe.

### `inspect`

- `inspect config`: a configuração efetiva, o digest e as camadas que a produziram;
- `inspect plan`: a topologia resolvida, sem executar nada;
- `inspect run DIR [--events]`: o ciclo de vida de um run (estado, falha, problemas de bloqueio, estágios concluídos, eventos, ambiente);
- `inspect artifact CAMINHO`: um resumo do `manifest.json` de um diretório de artifact (id, schema, criação, número e tamanho dos arquivos; num `ContextMapArtifact`, também `context_map_id` e `content_identity`) e o resultado da checagem de integridade (a mesma do `validate`, no nível estrutural), ou um documento do runtime.

### `validate CAMINHO [--full]`

Verifica a integridade: em um registro de run (`status.json`), o log de eventos (linha corrompida, lacuna na numeração), a coerência com o status e o digest de cada documento; em um diretório de artifact, cada arquivo do `file_inventory` (existência, tamanho e SHA-256; um caminho que sai do artifact é recusado); em `effective_config.json`, `plan.json` e `execution.json`, a versão de schema e o digest. Sai com `1` se algo falha e nomeia o arquivo ou o problema.

Um `ContextMapArtifact` (`artifact_type` `context_map` no manifest) passa, além disso, pelo validador da própria capability `artifact` ([`validate_context_map_artifact`](../../artifact/docs/integrity-validation.md)), que é quem sabe o que torna o mapa válido: um manifest editado à mão com o inventário coerente, por exemplo, é recusado com `manifest.identity_mismatch`. O nível padrão é o `STRUCTURAL` (manifest e versões, um `stat` por arquivo, descritores, estrutura dos índices, documentos pequenos e onde estão as dependências); `--full` roda o `FULL` (também cada registro, as referências, os índices reconstruídos e os arquivos das dependências a montante). Cada achado de severidade `error` entra em `integrity.problems` como `<código>: <mensagem>` e faz o comando sair com `1`; um aviso (`warning`, como uma dependência opcional ausente) não invalida o mapa pelo contrato do validador, então não entra em `integrity.problems` nem muda o código de saída (o relatório completo, com os avisos, é o que `validate_context_map_artifact` devolve). As dependências são procuradas pelos hints relativos do manifest. `--full` não muda nada para os demais caminhos, que têm um único nível de verificação.

Só esse ramo carrega uma capability (`contextmap.artifact`, importada de forma tardia); os demais tipos de artifact, os registros de run e os documentos do runtime usam só a biblioteca padrão: não importam NumPy, modelo nem ROS.

## Erros acionáveis

Todo problema sai com o caminho e o que fazer: perfil desconhecido (lista os conhecidos), configuração inválida (todos os problemas de uma vez), módulo opcional ausente (com a dica de instalação), segredo ausente (pelo nome), seleção incompatível (o que diverge, com os ids), estágio sem capability, estágio sem executor. Com `--json`, uma falha sai como `{"ok": false, "error": ..., "problems": [...]}` na saída padrão.

Códigos de saída: `0` sucesso; `1` pedido entendido, mas não atendível (configuração inválida, preflight bloqueado, estágio que falha, integridade); `2` uso incorreto da linha de comando (por exemplo, execução real sem workspace).

## Fronteira

A CLI não faz rede, não abre dashboard e não usa "todos os runs". Importar `contextmap.runtime.cli` não carrega SDK pesado (torch, transformers, rosbags, NumPy); um teste garante isso. A única capability que a CLI importa é a raiz pública de `contextmap.artifact`, e só dentro do ramo do `ContextMapArtifact` de `validate` e `inspect artifact`. A exceção é da função `_context_map_findings`, não do módulo: o mesmo import em qualquer outro ponto da CLI é violação (`tests/architecture/test_runtime_boundaries.py`).

## Executores

Para `run`, `stage` e seus dry runs, a CLI monta automaticamente os executores que [`compose_executors`](composition.md) consegue construir só a partir da configuração resolvida — hoje `state_estimation`, `geometric_mapping`, `sensor_association`, `semantic_fusion`, `entity_resolution`, `spatial_relations` e, quando cada backend selecionado que não tem loader empacotado (`sam2`/`sam3`/`florence2` em region discovery, `qwen`/`gemini`/`florence2` em semantic interpretation) tem um `RuntimeProvider` disponível, também `visual_perception`. Esse provider chega por **dois** caminhos, e a CLI não inventa um terceiro: `main(argv, providers=...)` (só um chamador Python: um teste, uma API embutida) ou `resources.providers` **dentro da própria configuração resolvida** (`"<capability>.<slot>" -> "módulo:atributo"`, resolvido por `resolve_provider`; ver [`composition.md`](composition.md#providers-declarados-em-configuração-resourcesproviders) e [`configuration.md`](configuration.md#resourcesproviders-runtime-de-modelo-declarado-em-configuração)). O segundo é o que fecha a lacuna real: o binário `contextmap` instalado, chamado como `contextmap run ...`/`contextmap stage ...`, nunca invoca `main()` com um `providers=` Python — só o primeiro caminho nunca bastaria para ele. Um `providers=` explícito, quando presente para o mesmo componente, ainda vence sobre um alvo declarado. O resultado composto se mescla com o que `main(argv, executors=...)` recebeu de quem chama; o que vem de `main()` sempre vence (um teste, um estágio que a composição não sabe montar como `ingestion`, ou uma substituição explícita). Isso é o que deixa o binário `contextmap` instalado compor essas capabilities sem um wrapper Python que monte o executor **ou o provider** à mão.

`ingestion` continua sem ser composto automaticamente: o `IngestionStageExecutor` precisa de um `IngestionRequest` concreto (fonte, tópicos, sincronização) que não é parte de nenhuma configuração — é exatamente o que os flags do comando `ingest` constroem. O caminho canônico sem wrapper é `contextmap ingest` seguido de `contextmap run --select ingestion=<artifact-id> --catalog ...`, que trata a saída da ingestion como entrada fornecida em vez de reexecutá-la. `point_representation` continua sem executor real (depende de GPU/modelo); sem uma injeção, o preflight de uma execução real bloqueia com `no executor is registered for it` para esses estágios, e nada roda; um dry-run não precisa de executores e informa quais faltam. Um estágio que a composição tentou e não conseguiu montar bloqueia a execução real com o motivo, no componente em que falhou (`components.<capability>.<slot>`: parâmetro recusado pelo backend, runtime de modelo ausente, alvo `resources.providers` inválido), em vez do genérico (issue #602).

## Lacunas conhecidas

- **`ingestion` dentro de `run`/`stage`.** Não é composto automaticamente (ver acima); precisa de injeção explícita ou de rodar como um `contextmap ingest` separado e ser fornecido/selecionado.
- **`visual_perception` sem `providers`.** Seus backends de region discovery e semantic interpretation não têm loader empacotado; sem `main(providers=...)` para cada um, a composição fica honestamente ausente e um run real fica bloqueado no preflight.
- **`point_representation`.** Sem executor real ainda; um run que o inclua precisa de um executor injetado (por exemplo um teste) ou fica bloqueado no preflight, explicitamente.
- **Calibração externa em `ingest`.** A CLI ainda não carrega um arquivo de calibração (o decoder não é API pública de `contextmap.ingestion`); fontes com `camera_info` trazem a calibração pelo adapter.
- **`export`.** Não há artifact a exportar antes da etapa de montagem do `ContextMapArtifact`; o parser não declara esse comando até que exista algo real para exportar.
- **Verificador de artifacts.** As flags de reuso e retomada existem, mas o `verify` do índice vem do dono dos executores reais; a CLI recusa `--reuse-index` sem ele.
- **API pública.** A CLI chama as funções do runtime diretamente; a fachada pública frontend-neutra (`Runtime`, em `api.py`) passa a ser o que ela consome.

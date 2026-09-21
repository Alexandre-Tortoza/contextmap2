# Testes do runtime

O runtime controla **quais artifacts, backends e configurações exatos** uma execução usa. Um erro aqui pode invalidar em silêncio uma comparação científica mesmo com todas as capabilities corretas, então ele tem cobertura própria, determinística e sem GPU, modelo ou rede.

## Estratégia

- **Estágios falsos.** `tests/runtime/runtime_worlds.py` define um `World` cujos estágios são funções puras da entrada e da configuração: o conteúdo (hash) de cada artifact deriva do que o estágio consumiu e de como foi configurado, e cada execução é um run distinto com id próprio. Assim reuso, invalidação, seleção e retomada são exercitados de verdade; só a ciência é substituída.
- **Não duplicar os testes científicos.** Nenhum teste do runtime reexecuta a lógica de uma capability; as capabilities continuam testáveis sem construir o runtime (`tests/architecture` garante que nenhuma importa o runtime).
- **Identidade, não caminho.** Os testes de cache usam identidades e conteúdo de artifacts: o mesmo nome com outro conteúdo nunca é reutilizado, e outro nome com o mesmo conteúdo é.
- **Sem rede nem APIs externas.** Nada é obrigatório em CI além do próprio repositório.

## Mapa da cobertura

| Arquivo | O que protege |
|---|---|
| `test_runtime_config.py`, `test_runtime_catalog.py` | precedência, digest determinístico, escopo por backend, segredos, incompatibilidades, persistência da configuração efetiva |
| `test_runtime_coercion.py`, `test_runtime_composition.py` | parâmetros JSON → configuração da própria capability, combinações inválidas, falhas explícitas, construção sem carregar modelo |
| `test_runtime_pipeline.py` | ordem e dependências do DAG, estágios opcionais, ciclos, contratos, escopo/subgrafo, preflight, execução e falha |
| `test_runtime_reuse.py` | chave por conteúdo, invalidação seletiva, índice, decisões, braços de experimento |
| `test_runtime_selection.py` | seleção explícita, `latest` determinístico, linhagem, cardinalidade, catálogo |
| `test_runtime_lifecycle.py` | estados, eventos, registro de falha, cancelamento, run morto/log corrompido, segredos, retomada |
| `test_runtime_cli.py` | a CLI como camada fina: dry-run, execução, seleção, artifacts, run records, reuso/retomada, códigos de saída |
| `test_runtime_ingestion_service.py` | o serviço de ingestion com um adapter falso roteirizado: preflight, identidade determinística, execução até um `SequenceArtifact` real, avisos, falhas por categoria, cancelamento, nada publicado em falha, segredos, o estágio do DAG e o reuso |
| `test_runtime_ingestion_integration.py` | `contextmap ingest` com o adapter ROS 1 **real** composto e um bag sintético (pulado sem `rosbags`) |
| `test_runtime_api.py` | a API pública `Runtime` com o mundo falso: descoberta com backends disponíveis e indisponíveis (e nenhum SDK pesado importado), configuração e topologia determinísticas, edições permitidas, estágio/backend não suportado explícito, preflight (sucesso, todos os problemas, avisos, previsão de reuso), execução (sucesso, falha, bloqueio, cancelamento, interrupção), ordem dos eventos, sink que levanta, segredos, reuso/retomada visíveis, lista e detalhe de runs a partir do registro persistido sem inferir ausências, consultas repetidas equivalentes |
| `test_runtime_api_consumer.py` | um frontend escrito **só com imports públicos** de `contextmap.runtime` (descoberta → configuração → preflight → execução → inspeção → reuso → cancelamento) e um teste que lê o próprio arquivo e prova que só `contextmap.runtime` foi importado |
| `test_runtime_regressions.py` | erros comuns de configuração, seleção, cache e segredos, cada um com um teste nomeado; dry-run = plano executado; execuções equivalentes geram registros equivalentes |
| `test_runtime_end_to_end.py` | o caminho canônico, do arquivo de configuração aos registros, com estágios falsos |
| `tests/architecture/test_runtime_boundaries.py` | só a composition root conhece capabilities e backends, e só dentro de factories; importar o runtime não carrega capability nem backend; a API pública só depende da biblioteca padrão e do runtime; nenhum módulo do runtime depende de biblioteca de UI; os contratos públicos não expõem tipo ROS nem de backend |

## Invariantes exercitadas de ponta a ponta

- o **dry-run** resolve o mesmo plano (mesmo digest, mesmo `plan.json`) que a execução real do mesmo pedido, sem carregar modelo nem escrever nada;
- a **linhagem entre estágios** é consistente: a entrada de cada estágio é exatamente o artifact que o estágio produtor emitiu, em toda execução, reutilizada ou retomada;
- o **mesmo pedido** (mesmo arquivo, mesmo workspace) deixa `effective_config.json`, `plan.json` e `execution.json` idênticos byte a byte e a mesma trilha de eventos (exceto tempos);
- os **digests** são iguais entre processos e sementes de hash diferentes, e o ambiente do processo nunca os altera;
- a **retomada** só reutiliza o que passa nas checagens de reuso, nunca promove uma saída parcial e nunca toca o run anterior;
- **nenhum segredo** aparece em saída, eventos, status, plano, configuração efetiva ou ambiente registrado, nem em uma execução bem-sucedida.

- a **API pública** devolve, para consultas repetidas (status, capacidades, plano, preflight, lista e detalhe de runs), documentos equivalentes; toda inspeção de run vem do registro persistido e uma ausência (entradas de um estágio reaproveitado, configuração persistida) é `None` com uma nota, nunca deduzida.

## Como o teste foi validado

A API pública foi validada por mutação: sink que levantar quebrando o run, workspace da configuração diferente aceito, ordem de runs lexicográfica, seleção sem catálogo aceita em silêncio, todo backend "disponível", segredos sem redação, identidade de código de reuso não registrada, entradas do estágio descartadas do registro, retomada impossível não recusada antes de criar o run e executor exigido de um estágio que será reaproveitado: cada quebra é pega por pelo menos um teste.

Além de passar, os guardas centrais foram verificados por **mutação**: quebrar deliberadamente a regra (índice que nunca registra, índice que nunca verifica, retomada que aceita topologia alterada, mensagem de falha sem redação, import de capability fora da composition root) faz a suíte falhar. Uma dessas mutações revelou a ausência de um teste de retomada com a **topologia** alterada e a mesma configuração efetiva, hoje coberto.

## Lacunas conhecidas

- Os testes de composição usam runtimes falsos: a construção real de SAM, DINO, VLMs e FAST-LIO com modelos é smoke/integração separada, fora da CI padrão.
- Não há teste de concorrência real entre dois processos alocando o mesmo diretório de run; a alocação usa `mkdir` exclusivo.
- A execução real do canônico com executores das capabilities pertence à validação end-to-end da milestone #19.

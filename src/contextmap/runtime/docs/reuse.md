# Reuso de estágios

Um artifact de estágio é imutável, então seu resultado pode ser reutilizado sempre que tudo de que ele depende é idêntico. O runtime decide isso estágio a estágio, em ordem de dependência, para que alterar um estágio invalide exatamente ele e quem depende de fato do que ele produz.

O reuso **nunca** usa nome de diretório nem nome legível de run como chave, e nunca esconde o artifact anterior atrás de um "cache hit" genérico: a decisão nomeia o artifact exato e o motivo.

## Chave de reuso

`ReuseKey` reúne tudo de que o resultado de um estágio depende:

| Componente | Origem |
|---|---|
| estágio e contrato de saída | `PlannedStage` |
| configuração própria do estágio | `PlannedStage.config_digest` (contrato declarado, backend e parâmetros dos componentes) |
| entradas | o tipo e o **hash de conteúdo** de cada artifact de entrada (`ArtifactRef.content_hash`), não seu nome nem seu id de run |
| código e políticas | `ReusePolicy.code_identity`, **sem default**: reusar entre versões de código é uma decisão |
| identidades extras | `ReusePolicy.identities[estágio]`: calibração, mapa, seleção de observações, transformações opcionais |

`ReuseKey.digest` é o SHA-256 da forma canônica. Como a chave usa o **conteúdo** das entradas, dois artifacts com o mesmo hash de conteúdo são intercambiáveis, e um estágio recomputado que reproduz o mesmo conteúdo não invalida seus dependentes.

O `content_hash` é declarado pelo executor que produz o artifact (por exemplo, o digest do inventário do manifest). Um artifact sem ele pode ser consumido, mas nunca reutilizado nem indexado: sem identidade de conteúdo, a chave dos dependentes não pode ser construída e eles são recomputados, com o motivo registrado.

## Índice

`FileArtifactStore` guarda uma entrada JSON pequena e imutável por identidade; não há serviço de cache global mutável.

- A entrada é escrita **somente depois** que o estágio conclui, de forma atômica e sem substituir uma entrada válida. Uma falha, uma interrupção ou uma saída parcial não deixam nada que satisfaça uma consulta; arquivos temporários (`.tmp-*`) nunca são entradas.
- `find()` revalida a entrada contra a chave, checa a forma e pergunta ao callback `verify` se o artifact ainda existe e está íntegro (por exemplo, abrindo-o com o leitor da capability). O `verify` **não tem default**: um índice não sabe se o artifact por trás de uma entrada ainda existe. Entrada ilegível, divergente, de outro estágio ou que falha na verificação é tratada como ausente.
- `record()` nunca substitui uma entrada válida (o primeiro artifact concluído de uma identidade permanece indexado) e repara uma entrada inválida com o novo artifact.

## Decisão por estágio

`ReuseDecision` acompanha cada estágio no `ExecutionRecord`:

- `reused`: o artifact exato reutilizado (`reused_from`) e o digest da chave;
- `recomputed`: o motivo. Sem artifact anterior com essa identidade; entrada ilegível ou divergente; artifact anterior que falhou na verificação; recomputação forçada; entrada sem hash de conteúdo (sem chave); e, quando aplicável, saída sem hash de conteúdo (não indexada) ou artifact anterior que permanece indexado.

Sem `ReusePolicy`, o runner sempre executa e não registra decisão.

## Invalidação sensível ao DAG

Os estágios são decididos em ordem topológica, com a chave calculada a partir das entradas reais (reutilizadas ou recém-produzidas):

- alterar uma política de Semantic Fusion recomputa só a fusão; ingestion, percepção, estado, geometria e associação são reutilizados;
- alterar a configuração da percepção recomputa a percepção e o que consome seu **novo conteúdo**; estado e geometria são reutilizados;
- se uma recomputação reproduz o mesmo conteúdo, os dependentes seguem reutilizáveis;
- mudar `code_identity` invalida tudo, e uma identidade extra invalida o estágio para o qual foi declarada.

## Braços de um experimento

Um DAG alternativo continua identificável (seu plano tem outro digest) e compartilha o upstream idêntico. Com a inserção de um estágio opcional entre a extração densa e a associação, o braço com enhancement reutiliza os artifacts exatos de ingestão e extração do braço nativo, executa só o enhancement e a associação, e o braço nativo permanece reutilizável.

## Previsão e preflight

`predict_reuse()` diz, sem executar nada, o que seria reutilizado. A previsão é **conservadora**: um estágio a jusante de outro que será recomputado aparece como recomputado (suas entradas ainda não existem), embora a execução possa reutilizá-lo se a recomputação reproduzir o mesmo conteúdo; o registro da execução é a autoridade.

No `preflight()`, um estágio que **certamente** será reutilizado dispensa executor, módulos opcionais e segredos: nada dele vai rodar (a configuração continua precisando estar completa). Forçar um estágio fora da execução é um problema.

## Recomputação forçada

`ReusePolicy.force_recompute` obriga um estágio a rodar mesmo com a identidade indexada. A decisão registra `forced recomputation requested`; o novo artifact vale para a própria execução e **não** desloca o indexado.

## Lacunas conhecidas

- Os executores reais precisam calcular o `content_hash` a partir dos manifests das capabilities e fornecer o `verify` com seus leitores; isso acompanha os executores da validação end-to-end (#177). Os testes usam estágios puros que derivam o conteúdo de suas entradas e configuração.
- O índice não tem coleta de lixo nem é distribuído: fora do escopo do v0.1.0.
- A retomada de um run interrompido usa exatamente estas checagens de reuso ([`lifecycle.md`](lifecycle.md)); a seleção de runs e a validação de linhagem estão em [`selection.md`](selection.md).

# Avaliação de Semantic Interpretation

A avaliação semântica consome `SemanticInterpretationExecution` sem alterar
claims, escolher winners ou aplicar Semantic Fusion. O mesmo report schema
aceita Qwen, Gemini e Florence-2.

O `SemanticEvaluationContext` torna obrigatórios reference-set, seleção, run,
artifact, pipeline/configuration digest e evaluator version (na comparação entre
backends, quais deles precisam coincidir está em
[Comparações controladas](#comparações-controladas)). Cada
`SemanticEvaluationInput` associa uma execution a um `evidence_variant_id` e,
opcionalmente, a uma anotação, permitindo ablações controladas como masked
subject, tight crop, contextual crop e with/without scene context.

## Convenções de correspondência e anotação

O baseline `casefold-exact/1` normaliza case e whitespace, mas não afirma
equivalência semântica entre sinônimos. Outro matcher open-vocabulary precisa
usar uma policy versionada diferente.

Anotações são **parciais**: a ausência de anotação significa "não anotado", nunca
verdade negativa. Por isso:

- sem anotação, as claims são **não avaliadas** (`acceptable_claim_count=None`) e as
  taxas de correção ficam `None`, em vez de contar tudo como `unsupported`;
- `unsupported` é uma claim fora de `acceptable_hypotheses` (talvez um conceito
  legítimo que ninguém anotou); só `rejected_hypotheses` é verdade negativa, e só
  ela conta como alucinação (`rejected_claim_rate`);
- `abstention_expected` representa "o anotador não soube dizer": abstenção é a
  resposta certa e a correção de qualquer claim não se aplica
  (`correct_abstention_count`, `unexpected_abstention_count`);
- `SceneContextAnnotation` lista valores aceitáveis por campo de `SceneContext`;
  só os campos anotados são pontuados, como `correct`, `incorrect` ou `missing`;
- `visibility_stratum` estratifica o report; estratos derivados dos dados (por
  exemplo, área da região) entram como `StratumSource.DERIVED` e nunca se
  confundem com anotação de visibilidade.

Os nomes seguem a família `semantics` do reference-set (`concepts`,
`rejected_concepts`, status `UNKNOWN`, visibilidade), de modo que o conversor
daquela família possa repassá-los sem que este módulo importe seu schema.

## Blocos do report

O report mantém blocos separados:

- **qualidade** (só run primário): claims aceitáveis/unsupported/rejeitadas,
  preservação de ambiguidade, abstenção esperada, campos de cena, duplicatas e
  concentração de rótulos (`distinct_hypothesis_count`, hipótese dominante e seu
  share, para expor domínio de rótulos estruturais ou triviais);
- **custo** (todas as execuções): latência p50/p95, no método nearest-rank
  (`percentile_method`), por modo e no total; retries; tokens; e o pico de memória
  de GPU que o backend reportou, que para modelo local inclui os pesos;
- **outcomes** por modo: interpretadas, abstidas, falhas de parser e falhas de
  backend, sempre distinguíveis; falha de parser preserva a resposta bruta
  (`SemanticResponseParseError.raw_response`), o que separa truncamento ou desvio de
  schema de um modelo que respondeu errado;
- **estabilidade entre repetições** (`repeat_index`): quantas requests repetidas
  devolveram o mesmo texto bruto e a mesma resposta normalizada, e quais divergem;
- **estratos**: qualidade e falhas por estrato.

Repetições sobre o mesmo frame são amostras de estabilidade e custo, não novas
evidências físicas: a qualidade conta somente `repeat_index=0`.

## Comparações controladas

`compare_semantic_backends()` alinha os reports pelas requests **tentadas**
(sucesso ou falha), para que uma falha seja um resultado e não desapareça do
alinhamento. Lista o resultado por request e backend e o acordo entre as
hipóteses primárias. Acordo entre backends não é correção.

A comparação só vale se os backends interpretaram as **mesmas entradas físicas**.
`request_id` e variante de evidência não bastam: dois reports podem reutilizar os
mesmos ids para outros frames ou regiões. Por isso a chave de alinhamento de cada
request é `(request_id, evidence_variant_id, source_observation_id, region_id,
mode)`, e a comparação é rejeitada (`SemanticEvaluationError`) quando os
conjuntos de chaves diferem em qualquer componente; a mensagem nomeia a primeira
request que cada lado tentou e o outro não. Cada `SemanticRequestOutcomes` grava
a observação, a região e o modo em que os reports foram alinhados.

Uma falha é uma tentativa como as outras e precisa carregar a mesma identidade:
`SemanticEvaluationFailure` continua aceitando `mode`, `source_observation_id` e
`region_id` opcionais (um report isolado não precisa deles, e
`SemanticEvaluationFailure.from_error` os preenche a partir do request), mas a
comparação rejeita uma falha sem `mode` e `source_observation_id`, uma falha de
região sem `region_id` e uma falha de cena com `region_id`.

O contexto de avaliação também precisa ser compatível. A comparação exige o mesmo
`reference_set_version`, `selection_id`, `perception_run_id` (um `region_id` só
tem sentido dentro do run de percepção que o produziu), `evaluator_version` e
`matching_policy`. `evaluation_id`, `artifact_id` e `pipeline_configuration_digest`
ficam de fora de propósito: identificam a avaliação, o artifact e o pipeline de
cada backend e por isso diferem por construção. Os relatórios reais desta
milestone (cena, região com tight crop e nf4 versus int8) continuam comparáveis
sob essa regra e reproduzem as mesmas contagens.

`compare_evidence_variants()` é o gancho de ablação: pareia a mesma observação e
região entre variantes de evidência e mede cada uma contra uma variante de
referência (taxas, abstenções, falhas, latência e deltas). Sem anotação, a
sensibilidade à evidência ainda é mensurável: quantas regiões mantêm a mesma
hipótese primária da variante de referência (`primary_agreement_count` de
`primary_comparable_count`). Os canais de evidência
de cada variante (tipos de view, features, scene context) são lidos dos requests,
não do rótulo da variante, e os `prompt_template_id` de cada lado são listados,
então uma ablação de prompt também é observável. `with/without SceneContext`
exige um backend que aceite `scene_context_reference`; nenhum adapter atual o
aceita.

`encode_semantic_evaluation_report()`, `encode_semantic_backend_comparison()` e
`encode_evidence_variant_comparison()` devolvem primitivas JSON com todas as
identidades.

## Estado de validação e limitações

A CI exercita o schema do relatório, as convenções acima, falhas, repetições,
estratos e as comparações com execuções canônicas construídas em teste
(fake/contract). Isso valida a aritmética do avaliador, não a qualidade de um
backend.

Não existe reference set com anotações semânticas humanas para a amostra
corridor-02 de 20 frames. Os relatórios reais dessa amostra são, portanto,
**sem anotação**: correção, alucinação, abstenção esperada, campos de cena e
visibilidade são N/A, e só contagens, resultados por request, estabilidade,
concentração de rótulos e custo são medidos. Essas medidas não sustentam
conclusão sobre qual backend tem melhor qualidade semântica.

## Execução real de referência (2026-09-21, sem anotações)

Todos os números abaixo são **reais** (GPU RTX 3060 compartilhada, decoding
determinístico, uma amostra pequena) e não são correção semântica. Amostra: os
20 frames de `outputs/validation/2026-09-21/selection.json`; requests de cena
sobre o frame completo e requests de região sobre as 5 maiores regiões SAM2
aceitas (`vp-sam2-rerun`, área ≥ 1500 px), 100 regiões, com o tight crop como
evidência primária. Contagens da execução primária (`repeat_index=0`); latência
em nearest-rank; o pico de memória é o do processo, com pesos.

| Backend e configuração | Modo | Requests | Interpretadas | Falha de parser | Falha de backend | Latência p50 / p95 | Pico de GPU |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B, nf4 | scene | 20 | 17 | 3 | 0 | 7,9 s / 22,7 s | 3246 MiB |
| Qwen3-VL-4B, nf4 | region | 100 | 53 | 47 | 0 | 4,7 s / 6,3 s | 3144 MiB |
| Florence-2 `<DETAILED_CAPTION>` | scene | 20 | 20 | 0 | 0 | 0,50 s / 0,59 s | 1819 MiB |
| Florence-2 `<REGION_TO_CATEGORY>` | region | 100 | 100 | 0 | 0 | 0,23 s / 0,25 s | 1819 MiB |
| Florence-2 `<REGION_TO_DESCRIPTION>` | region | 100 | 100 | 0 | 0 | 0,31 s / 0,39 s | 1819 MiB |

- **Estabilidade.** Com decoding guloso ou determinístico, as repetições foram
  idênticas: 120 requests do Qwen (2 repetições) e as do Florence-2 devolveram o
  mesmo texto bruto, inclusive as falhas de parser.
- **Falhas de parser do Qwen.** Nenhuma é falha do runtime: a maior parte é o
  modelo omitir `confidence` (o schema exige `null`) ou pôr atributos não
  escalares; poucas coincidem com truncamento. O relatório preserva o texto
  rejeitado de cada uma.
- **Concentração de rótulos.** O Florence-2 `<REGION_TO_CATEGORY>` produziu 24
  rótulos distintos em 200 respostas, o mais frequente (`poster`) com 17,5%. As
  hipóteses do Qwen são frases descritivas quase todas distintas, e por isso o
  acordo exato entre backends (`primary_agreement_count`) foi 0 de 17 (cena) e 0
  de 53 (região): comparar frase com rótulo por `casefold-exact/1` não mede
  concordância semântica.
- **Ablação de evidência** (mesma região e backend, pareadas por observação e
  região). Florence-2 `<REGION_TO_CATEGORY>`, 100 regiões: mascarar o fundo
  (`masked_subject`) manteve o rótulo do tight crop em apenas 11 regiões,
  então o canal de evidência muda a resposta na maior parte dos casos, e sem
  anotação não há como dizer qual variante está certa. Qwen3-VL-4B nf4, 50
  regiões dos 10 primeiros frames, 1 repetição: respostas parseadas em 27
  (tight crop), 26 (contextual crop) e 45 (masked subject) de 50. É uma medida de
  aderência ao schema, não de qualidade, e a diferença não tem causa
  demonstrada.
- **Quantização** (5 primeiros frames, 30 requests, tight crop). nf4 interpretou
  19 de 30 (região 14/25, cena 5/5) com latência p50 4,7 s e pico de 3246 MiB;
  int8 interpretou 21 de 30 (região 19/25, cena 2/5) com p50 12,8 s e pico de
  4938 MiB, porque o LLM.int8 é cerca de 2,7 vezes mais lento e ocupa mais VRAM.
  Amostra pequena, sem anotação.

Os relatórios completos (`*.report.json`, comparações e variantes), as
execuções brutas com a resposta bruta de cada request e o driver ficam fora do
git em
`workspace/corridor-02/validation-semantic-interpretation-20260921/visual_perception/`.
O Gemini não tem relatório real: não há credencial nem consentimento para enviar
frames a um serviço externo.

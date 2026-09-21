# Avaliação de Semantic Interpretation

A avaliação semântica consome `SemanticInterpretationExecution` sem alterar
claims, escolher winners ou aplicar Semantic Fusion. O mesmo report schema
aceita Qwen, Gemini e Florence-2.

O `SemanticEvaluationContext` torna obrigatórios reference-set, seleção, run,
artifact, pipeline/configuration digest e evaluator version. Cada
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
(sucesso ou falha) e rejeita reports que não cubram os mesmos pares
request/evidence variant, para que uma falha seja um resultado e não desapareça
do alinhamento. Lista o resultado por request e backend e o acordo entre as
hipóteses primárias. Acordo entre backends não é correção.

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

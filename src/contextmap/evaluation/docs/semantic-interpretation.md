# Avaliação de Semantic Interpretation

A avaliação semântica consome `SemanticInterpretationExecution` sem alterar
claims, escolher winners ou aplicar Semantic Fusion. O mesmo report schema
aceita Qwen, Gemini e Florence-2.

O `SemanticEvaluationContext` torna obrigatórios reference-set, seleção, run,
artifact, pipeline/configuration digest e evaluator version. Cada
`SemanticEvaluationInput` associa uma execution a uma anotação e a um
`evidence_variant_id`, permitindo ablações controladas como masked subject,
tight crop, contextual crop e with/without scene context.

O baseline `casefold-exact/1` normaliza case e whitespace, mas não afirma
equivalência semântica entre sinônimos. Outro matcher open-vocabulary precisa
usar uma policy versionada diferente.

O report mantém blocos separados:

- qualidade: claims aceitáveis/unsupported, preservação de ambiguidade,
  abstention e duplicatas;
- custo: latência, retries, tokens e memória;
- falhas: `parser` e `backend` permanecem distinguíveis.

`compare_semantic_backends()` rejeita reports que não cubram exatamente os
mesmos pares request/evidence variant. Repetições sobre o mesmo frame continuam
amostras de estabilidade, não novas evidências físicas.

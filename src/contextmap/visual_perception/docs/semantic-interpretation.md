# Semantic Interpretation

Semantic Interpretation transforma evidência visual selecionada em hipóteses
semânticas de uma única inferência. As saídas continuam sendo evidência de
frame/run; não são entidades persistentes, fusão entre observações ou truth do
mapa.

## Request canônico

`SemanticInterpretationRequest` registra a seleção exata entregue ao backend:

```text
request_id
source_observation_id
perception_result_id
mode = SCENE | REGION
region_id?
visual_views[]
visual_features[]
scene_context_reference?
supporting_metadata[]
prompt_template_id
requested_output_schema
configuration_fingerprint
```

`SemanticVisualView` distingue full frame, masked subject, tight crop e
contextual crop. Cada view referencia o payload materializado, a observação de
origem e, quando aplicável, a região congelada. `SemanticFeatureReference`
registra `feature_id`, scope e `embedding_space_id`; o vetor não é embutido no
request. `SceneContext` e metadata são opcionais e permanecem explícitos.

Uma requisição de região pode usar apenas pixels/views. Features DINO/CLIP ou
contexto de cena nunca são dependências ocultas nem obrigatórias do contrato.

## Validação antes do backend

O construtor rejeita:

- modo de cena com `region_id` ou modo de região sem `region_id`;
- requests sem view visual;
- view de outra observação ou região;
- feature de região incompatível;
- ids duplicados de views/features e metadata ambígua;
- identidades vazias de prompt, schema ou configuração.

`SemanticInterpreterCapabilities` declara modes e tipos de view suportados, se
o backend aceita features/contexto e quais views são obrigatórias.
`validate_semantic_request()` compara o request com essa declaração antes de
qualquer chamada local ou remota. Evidência não suportada causa erro explícito;
ela não é descartada silenciosamente.

## Rastreabilidade

`evidence_references()` deriva referências canônicas para cada view, feature e
contexto selecionados. Com as identidades de prompt, schema e configuração,
isso permite reconstruir o limite exato da chamada a partir do artifact do run,
sem persistir objetos nativos de Qwen, Gemini ou Florence-2.

O template e o parser versionados são definidos separadamente; este contrato
apenas torna explícita a entrada que eles recebem.

## Prompt e parsing versionados

`SemanticPromptTemplate` identifica de forma inseparável o texto de instrução,
o modo (`SCENE`/`REGION`) e o schema de saída. Os defaults `scene/v1` e
`region/v1` produzem um `RenderedSemanticPrompt` determinístico, incluindo um
fingerprint SHA-256 do texto efetivo. Template, modo e schema devem coincidir
com o request antes da renderização.

`parse_semantic_response()` aceita somente o objeto JSON do schema
`semantic-response/1`. Ele preserva `PRIMARY`/`ALTERNATIVE`, ausência real de
confidence, atributos escalares e campos estruturados de cena. Uma resposta de
região precisa conter exatamente uma hipótese primária; abstention é explícita
e não pode carregar claims escondidas.

Campos inesperados, tipos inválidos, JSON malformado, confidence não finita ou
fora de `[0, 1]` e conteúdo obrigatório ausente causam
`SemanticResponseParseError`. A única reparação v1 é remover uma code fence JSON
externa bem-formada; a decisão aparece em `SemanticParseDiagnostic`. O parser
nunca completa labels, confidence ou atributos ausentes e registra o hash da
resposta bruta separadamente dos outputs canônicos.

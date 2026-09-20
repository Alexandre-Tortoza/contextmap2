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
origem, um SHA-256 obrigatório e, quando aplicável, a região congelada. O run
artifact inventaria os bytes exatos abaixo de `outputs/semantic-views/` e
valida seu hash antes da finalização. `SemanticFeatureReference`
registra `feature_id`, scope e `embedding_space_id`; o vetor não é embutido no
request. `SceneContext` e metadata são opcionais e permanecem explícitos.
`supporting_metadata` integra o request serializado e o prompt renderizado;
portanto, o registro auditável coincide com o conteúdo textual efetivamente
entregue ao modelo.

Uma requisição de região pode usar apenas pixels/views. Features DINO/CLIP ou
contexto de cena nunca são dependências ocultas nem obrigatórias do contrato.

## Validação antes do backend

O construtor rejeita:

- modo de cena com `region_id` ou modo de região sem `region_id`;
- requests sem view visual;
- `payload_reference` fora do namespace seguro `outputs/semantic-views/`;
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

Na finalização, referências a features precisam resolver por identidade,
embedding space, scope e região no `PerceptionResult`, e o payload numérico
correspondente deixa de ser opt-in: se a feature foi consumida semanticamente,
ela precisa estar no feature store. `region_id` precisa existir nas regiões do
resultado mesmo em abstention. `scene_context_reference` usa o
`PerceptionResultId` proprietário como `evidence_id` e precisa resolver para um
`SceneContext` persistido da mesma observação.

O template e o parser versionados são definidos separadamente; este contrato
apenas torna explícita a entrada que eles recebem.

## Prompt e parsing versionados

`SemanticPromptTemplate` identifica de forma inseparável o texto de instrução,
o modo (`SCENE`/`REGION`) e o schema de saída. Os defaults `scene/v1` e
`region/v1` produzem um `RenderedSemanticPrompt` determinístico, incluindo um
fingerprint SHA-256 do texto efetivo. Template, modo e schema devem coincidir
com o request antes da renderização.

`parse_semantic_response()` aceita somente o objeto JSON do schema
`semantic-response/1`. O schema renderizado é específico ao modo: REGION exige
`scene_context=null` e ao menos uma claim com exatamente uma primária; SCENE
exige um objeto `scene_context` e permite `claims=[]` quando os campos
estruturados contêm ao menos um valor não nulo. Como tolerância de parsing
documentada em #340, uma resposta REGION que omite a chave `scene_context` é
normalizada para `null` e essa normalização entra nos diagnostics; um valor
não nulo continua inválido. O modo SCENE permanece estrito e continua rejeitando
a ausência da chave. Uma resposta SCENE não abstida é rejeitada quando não
possui claim nem campo de contexto significativo. Abstention é explícita e
não pode carregar saída semântica escondida.

Campos inesperados, tipos inválidos, JSON malformado e conteúdo obrigatório
ausente causam
`SemanticResponseParseError`. A única reparação v1 é remover uma code fence JSON
externa bem-formada; a decisão aparece em `SemanticParseDiagnostic`. O parser
`SemanticConfidencePolicy` torna a semântica de score explícita no boundary do
prompt/parser. Qwen e Gemini usam `UNSCORED_ONLY`, apresentam apenas `null` no
schema e rejeitam números auto-relatados pelo VLM. Um backend que possua uma
fonte realmente medida ou calibrada pode selecionar `MEASURED`, preservando um
número finito em `[0, 1]` sem mudar o contrato canônico. O hash da resposta
bruta é registrado separadamente dos outputs canônicos.

## Materialização e persistência

`assemble_perception_result()` recebe explicitamente os ids dos stages que
produziram `SemanticInterpretationExecution` e materializa
`execution.parsed.claims`/`scene_context` no `PerceptionResult`, validando as
identidades da observação e do resultado. Ao receber os mesmos outcomes,
`PerceptionRunWriter` persiste a execução em
`outputs/semantic-interpretations.jsonl` e materializa a resposta bruta no path
de debug declarado pela proveniência. Assim, execução, evidência canônica e
artifact permanecem ligados pelo mesmo request id.

`SemanticDebugLevel` controla apenas o conteúdo humano em
`debug/40-semantic-interpretation/<request_id>/`. `NONE` não grava debug,
`STANDARD` grava request, prompt, parsing, outputs finais e diagnostics, e
`FULL` acrescenta a resposta bruta. Os outputs canônicos, hashes, métricas e
views content-addressed continuam válidos em qualquer nível. Antes da
serialização, campos de credencial conhecidos são redigidos recursivamente;
contadores como `input_tokens`/`output_tokens` não são confundidos com secrets.

## Scoring semântico

`SemanticScore` registra separadamente o suporte de uma feature visual a uma
claim: ids do score/claim/feature, tipo e valor do score, embedding space,
observação, resultado e provenance completa do scorer. O campo opcional
`calibrated_probability` permanece `None` nos adapters atuais. Uma claim sem
score continua sendo evidência válida; ausência de record nunca é serializada
como suporte zero.

`ClipSemanticScorer` compara claims de cena com `VisualFeature` global.
`AlphaClipSemanticScorer` compara claims regionais somente com a feature da
mesma região congelada. Ambos exigem espaço de embedding idêntico entre texto e
imagem, vetores declarados e verificados como L2-normalized, payload
unidimensional finito e provenance de modelo/configuração. O valor persistido é
cosine similarity em `[-1, 1]`, sem remapeamento ou comparação implícita entre
as escalas CLIP e AlphaCLIP.

O encoder de texto e o carregador lazy de payload são seams internos
injetáveis; tensores/objetos nativos não entram no contrato público. O
compilador do DAG conhece `semantic_scorer` com inputs `claims` e `features`, e
`assemble_perception_result()` materializa os scores dos estágios selecionados.
O preset canônico não escolhe um scorer automaticamente.

## Adapter Qwen

`QwenSemanticInterpreter` é o adapter local substituível. Ele recebe apenas o
request canônico, valida as capacidades e o fingerprint de configuração,
renderiza o template compartilhado, delega a geração a `QwenRuntime` e usa o
parser canônico. Modelo, device, precision, quantização, token limit e
temperature formam o fingerprint e permanecem disponíveis na configuração
efetiva da execução.

O seam de runtime mantém Transformers/Torch e objetos Qwen fora dos contratos.
Falha ou indisponibilidade de Qwen é propagada; não existe fallback implícito.
Métricas de tokens, latência, memória e warnings são registradas quando o
runtime consegue medi-las. A cobertura CI usa runtime fake determinístico; uma
execução de referência com pesos reais continua exigindo ambiente compatível e
deve ser registrada pelo protocolo de avaliação, nunca simulada como evidência
real.

## Adapter Gemini

`GeminiSemanticInterpreter` usa o mesmo request, template e parser do adapter
local. `GeminiSemanticConfig` contém somente model, timeout, retries e settings
de geração/raciocínio; credenciais pertencem ao `GeminiClient` injetado e nunca
entram no fingerprint, outputs ou debug. Falhas transitórias possuem retries
limitados e contados; resposta vazia/bloqueada e retries esgotados terminam com
erro explícito, sem substituição por outro backend. Usage, latência, warnings e
identidade do provider permanecem auditáveis.

## Adapter Florence-2

`Florence2SemanticInterpreter` é separado de `Florence2RegionDiscovery` mesmo
quando ambos compartilham lifecycle/modelo no composition root. Sua
`Florence2SemanticConfig` fixa checkpoint, revisão imutável, task, modes
suportados, device, precision e geração. A task e o mode entram em
`task_identity`; checkpoint, revisão e configuração entram na provenance e no
fingerprint. O runtime retorna somente texto/diagnostics SDK-neutral, e a saída
passa pelo mesmo prompt/parser canônico com `UNSCORED_ONLY`.

## Avaliação

`contextmap.evaluation.semantic_interpretation` fornece um report comum para
Qwen, Gemini e Florence-2. O contexto registra reference-set, seleção, run,
artifact, pipeline digest e versão do evaluator. Cada amostra preserva request,
região, evidence variant, backend/model/config, prompt e métricas. Qualidade e
custo permanecem em blocos distintos. O baseline usa a policy versionada
`casefold-exact/1`.


## Estado do milestone

O branch de integração materializa:

- `SemanticClaim`/`SceneContext`, request/evidence e prompt/parser possuem
  contratos canônicos, provenance e incerteza explícita;
- requests e executions são persistidos com views exatas content-addressed,
  features consumidas materializadas no feature store e contexto de cena
  resolvível;
- Qwen e Gemini implementam o mesmo boundary `SemanticInterpreter`, usando
  `UNSCORED_ONLY` para não promover confidence auto-relatada pelo VLM;
- Florence-2 implementa o mesmo boundary por adapter separado de Region
  Discovery;
- auditoria possui níveis explícitos e redaction de secrets;
- o harness de avaliação compara qualidade e custo sem Semantic Fusion;
- testes determinísticos cobrem parsing, abstention, retries, materialização no
  `PerceptionResult` e reabertura do run artifact.

Execuções reais controladas ainda dependem de pesos/runtime local para Qwen e
Florence-2 e de credenciais/acesso para Gemini. O ambiente de CI valida seams,
contratos, parsing, provenance, falhas e report schema com doubles
determinísticos; isso não é registrado como evidência experimental real.

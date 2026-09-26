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
o backend aceita features/contexto, quais views são obrigatórias e, quando
existe, o número máximo de views por request (`max_visual_views`).
`validate_semantic_request()` compara o request com essa declaração antes de
qualquer chamada local ou remota. Evidência não suportada causa erro explícito;
ela não é descartada silenciosamente.

## Política de views de evidência (#524)

`SemanticViewPolicy` é a política versionada que diz quais views um request de
região carrega, em que ordem, e como cada uma é cortada da imagem preparada.
`materialize_region_views()` e `materialize_scene_view()` a aplicam sobre os
pixels RGB da imagem preparada e a região congelada, sem alterar a geometria da
região. Uma ablação de views é só uma mudança de configuração: nenhum código de
request é escrito à mão.

- **Views declaradas, em ordem.** `region_views` é uma tupla ordenada e sem
  repetição de `VisualViewKind`, com ao menos uma view presa à região
  (`MASKED_SUBJECT`, `TIGHT_CROP` ou `CONTEXTUAL_CROP`). O request de região
  carrega exatamente essas views, nessa ordem, e o runtime de Qwen/Gemini as
  recebe nessa ordem antes do prompt. Nenhum frame inteiro é acrescentado sem
  que a política o declare; quando declarado, o `FULL_FRAME` da região é a mesma
  view (mesmo `view_id`, bytes e SHA-256) do request de cena do frame. O request
  de cena carrega sempre exatamente um `FULL_FRAME`.
- **Variantes.** Qualquer combinação válida é expressável; as do experimento
  (#521) são `masked`, `tight`, `contextual`, `full+tight`, `masked+tight`,
  `masked+contextual`, `tight+contextual`, `masked+tight+contextual` e
  `full+masked+tight+contextual`.
- **Recorte.** `TIGHT_CROP` e `MASKED_SUBJECT` cobrem a caixa da região
  arredondada para fora (piso da borda mínima, teto da máxima) e limitada à
  imagem. `CONTEXTUAL_CROP` expande a caixa por `context_margin_ratio` vezes a
  largura (esquerda/direita) e a altura (cima/baixo) da própria caixa, com o
  mesmo arredondamento; na borda da imagem o recorte é truncado, nunca
  preenchido, então o sujeito pode ficar descentralizado.
- **Máscara.** `MASKED_SUBJECT` mantém os pixels da máscara inline da região e
  pinta o resto com `mask_fill_rgb`. Uma região só com caixa (sem máscara
  inline) é recusada com erro explícito; a caixa nunca substitui a máscara em
  silêncio.
- **Contorno.** `context_boundary` (`rgb`, `width_px`) desenha, no recorte
  contextual, um anel logo fora da caixa da região, sem pintar pixels do
  sujeito; `None` não desenha nada.
- **Parâmetros vivos.** Um parâmetro só existe para a view que ele molda:
  `mask_fill_rgb` é obrigatório exatamente quando `MASKED_SUBJECT` é declarado,
  `context_margin_ratio` exatamente quando `CONTEXTUAL_CROP` é declarado, e
  `context_boundary` só é aceito com o recorte contextual. Assim duas políticas
  que constroem as mesmas views têm a mesma identidade.
- **Identidade.** `to_document()` registra todas as regras, inclusive as fixas
  da versão `semantic-views/1` (arredondamento, sem redimensionamento — as views
  mantêm a grade de pixels da imagem preparada; o orçamento de resolução do
  modelo é de #526 — e codificação PNG RGB8, filtro 0, zlib 9), e
  `fingerprint()` é o SHA-256 desse documento. Mudar qualquer regra fixa é uma
  versão nova.
- **Bytes e linhagem.** Cada view é um PNG determinístico: pixels, região e
  política idênticos produzem bytes idênticos, identificados pelo `sha256` da
  view. `SemanticVisualView.construction` (`SemanticViewConstruction`) registra
  o fingerprint da política, o SHA-256 da imagem de origem e a janela de pixels
  `(x_min, y_min, x_max, y_max)` usada; junto com `source_observation_id` e
  `region_id`, isso reconstrói de onde cada byte veio. Esse registro é
  persistido com o request e não entra no prompt renderizado.
- **Preflight.** `check_view_policy_supported()` compara a política com as
  capacidades do intérprete antes de qualquer inferência (tipos de view, número
  máximo e views obrigatórias, só nos modos que o intérprete suporta). O
  Florence-2 declara `max_visual_views=1` e só views que a região preenche, então
  uma política com duas views, recorte contextual ou frame inteiro falha na
  composição do runtime.
- **Seleção por configuração.** O runtime lê a política do grupo reservado
  obrigatório `view_policy` do backend semântico (ver
  [composição do runtime](../../runtime/docs/composition.md#política-de-views-semânticas-524));
  ela entra na configuração efetiva e na identidade do estágio
  `visual_perception`.

## Integridade das views na inferência

O `sha256` de cada `SemanticVisualView` identifica os bytes exatos que o request
declara ter fornecido ao modelo. Validar só a forma do request não basta: se o
arquivo mudar depois que o request foi construído, o backend inferiria sobre
bytes diferentes dos registrados. A verificação por `add_semantic_view_payload()`
acontece na persistência do run, depois da inferência, e continua existindo como
segunda barreira do artifact; ela não substitui a verificação abaixo.

Os seams internos de runtime/client (`QwenRuntime`, `Florence2SemanticRuntime`,
`EagleRuntime` e `GeminiClient`) recebem a **identidade completa das views**
(`visual_views: tuple[SemanticVisualView, ...]`) em vez de apenas
`payload_reference`. A leitura e a validação dos bytes são centralizadas em
`read_view_payload(view_root, view)` (`backends/_semantic_views.py`, interno à
capability, fora da API pública):

1. resolve a referência dentro de `view_root`, seguindo links simbólicos, e
   rejeita o que escapa dele (`ValueError`) ou não existe (`FileNotFoundError`);
2. lê o arquivo **uma única vez** e calcula o SHA-256 dos bytes lidos;
3. rejeita, com `ValueError` que mostra o hash esperado e o encontrado, um payload
   cujo hash difere de `SemanticVisualView.sha256`;
4. devolve os próprios bytes verificados.

Cada runtime decodifica ou transmite **somente esses bytes**: Qwen, Florence-2 e
Eagle 2.5 abrem a imagem por `Image.open(BytesIO(bytes))` e o Gemini envia os bytes como
parte inline. Assim, o que foi verificado é exatamente o que é consumido, sem
janela entre a checagem e o uso, e a verificação precede a abertura da imagem e
o envio ao provider. No Gemini, todas as views são verificadas antes da primeira
chamada de rede, então um payload divergente nunca sai da máquina.

Um payload divergente é uma falha explícita e terminal (`QwenInferenceError`,
`Florence2InferenceError`, `EagleInferenceError` ou `GeminiSemanticError`): não há retry, fallback nem
inferência parcial. Quem implementa esses seams com outro runtime, gateway ou
fake precisa usar `read_view_payload` (ou uma checagem equivalente); o contrato
está registrado nas docstrings dos protocolos.

A troca de `visual_payload_references` por `visual_views` altera apenas esses
seams internos, que não são exportados por `contextmap.visual_perception`; os
contratos públicos (`SemanticVisualView`, request, execution) não mudam. Em
`v0.x` não há consumidor externo dos seams, então não há camada de compatibilidade.

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
o modo (`SCENE`/`REGION`) e o schema de saída. As políticas canônicas `scene/v1`
e `region/v1` produzem um `RenderedSemanticPrompt` determinístico, incluindo um
fingerprint SHA-256 do texto efetivo. Template, modo e schema devem coincidir
com o request antes da renderização.

### Política de prompt explícita (#542)

A política de prompt é uma entrada da execução, selecionada antes da inferência;
nenhum backend escolhe o prompt. Isso permite uma ablação controlada (mesma
observação, região, views, modelo e configuração; só a política muda) sem editar
código nem criar comportamento específico de backend.

- **Catálogo versionado.** `SEMANTIC_PROMPT_TEMPLATES` é o catálogo fechado e
  imutável de templates de instrução, indexado pela identidade de cada um.
  `scene/v1` e `region/v1` são as políticas canônicas. `region-abstention/v1` é
  uma alternativa **não canônica e não avaliada**: varia só a instrução de
  abstenção em relação a `region/v1` e existe para que a seleção seja exercitável
  ponta a ponta; as famílias reais de prompt são avaliadas em #525. Uma identidade
  nomeia exatamente um texto de instrução e um schema de saída: uma política nova
  é uma entrada nova, nunca a edição de uma existente.
- **O request seleciona.** `SemanticInterpretationRequest.prompt_template_id` é a
  seleção. Qwen, Gemini e Eagle 2.5 renderizam exatamente o template do catálogo com
  essa identidade e entregam ao modelo exatamente esse texto. Não existe mais
  `SemanticPromptTemplate.default_for()` nem padrão interno de backend.
  Identidade desconhecida, modo divergente ou `requested_output_schema` diferente
  do schema do template falham com `ValueError` antes de qualquer chamada ao
  modelo ou provider.
- **Schema explícito.** `SemanticPromptTemplate` só aceita
  `output_schema_version="semantic-response/1"`, o único schema que o renderer e
  o parser implementam: um template não pode prometer ao modelo um contrato que
  ninguém analisa. Prompt e parser continuam separados; o parser não conhece a
  política que produziu a resposta.
- **Evidência coerente.** `RenderedSemanticPrompt` recusa um `fingerprint` que não
  seja o SHA-256 do próprio `text`, e `SemanticInterpretationExecution`/
  `FailedSemanticInterpretation` recusam um prompt renderizado cujo `template_id`
  ou `output_schema_version` difira do que o request selecionou. Nenhum adapter,
  atual ou futuro, consegue publicar evidência que afirme a política pedida tendo
  renderizado outra, e o prompt consumido é reconstruível do próprio registro da
  execução (`rendered_prompt.text`, também quando o parser rejeita a resposta).
- **Seleção por configuração.** `SemanticPromptPolicy(scene=..., region=...)` é a
  seleção declarativa por modo que uma execução configura para um interpretador
  que segue instruções; cada identidade é validada contra o catálogo e o modo. O
  runtime a lê do grupo reservado obrigatório `prompt_policy` da configuração do
  backend (ver [composição do runtime](../../runtime/docs/composition.md#política-de-prompt-semântico-542)),
  então ela entra na configuração efetiva, no seu digest e na identidade do
  estágio `visual_perception`, independentemente do backend, das configurações de
  geração e das views.
- **Florence-2 é nativo da task.** Ele não consome templates livres; sua política
  é o prompt da task ([abaixo](#decisão-de-design-task-token-versus-json-canônico)).

Renderizar a mesma política para o mesmo request é determinístico (JSON com
chaves ordenadas), e `region/v1`/`scene/v1` continuam produzindo exatamente os
mesmos bytes de antes de se tornarem políticas explícitas: um teste fixa os
fingerprints renderizados.

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
`SemanticResponseParseError`. As únicas normalizações v1 são não semânticas e sempre registradas em
`SemanticParseDiagnostic`: remover uma code fence JSON externa bem-formada (`removed_code_fence`) e,
no modo REGION, tratar a chave `scene_context` omitida como `null`
(`defaulted_null_scene_context`). No modo REGION essa chave só pode ser `null`, então sua ausência
carrega a mesma informação; um valor não nulo continua rejeitado, e no modo SCENE a chave continua
obrigatória. O prompt e as versões de template/schema (`region/v1`, `semantic-response/1`) não mudam,
e qualquer outra chave ausente ou inesperada continua rejeitando a resposta inteira. A tolerância
existe porque modelos reais descrevem corretamente a região mas omitem a chave nula (Qwen3-VL-4B: 3
de 3 respostas de região, issue #340).

`SemanticConfidencePolicy` torna a semântica de score explícita no boundary do
prompt/parser. Qwen, Gemini e Eagle 2.5 usam `UNSCORED_ONLY`, apresentam apenas `null` no
schema e rejeitam números auto-relatados pelo VLM. Um backend que possua uma
fonte realmente medida ou calibrada pode selecionar `MEASURED`, preservando um
número finito em `[0, 1]` sem mudar o contrato canônico. O hash da resposta
bruta é registrado separadamente dos outputs canônicos.

## Condicionamento por contexto de cena (#529)

Um request de região pode ser condicionado ao `SceneContext` que o request de cena
da mesma observação produziu. O contexto continua sendo evidência/hipótese de uma
inferência anterior, nunca truth.

- **Contexto nomeado e carregado.** `scene_context_reference` nomeia o contexto e
  `scene_context` carrega exatamente o `SceneContext` nomeado; os dois vêm juntos
  ou nenhum vem. A referência precisa nomear o `perception_result_id` do contexto
  carregado, e o contexto precisa ser da mesma observação do request. O run
  artifact recusa, na finalização, um request cujo contexto carregado difira do
  `SceneContext` persistido para aquele resultado: o contexto renderizado é o
  contexto citado.
- **Sem ciclo.** Um request de cena nunca é condicionado a contexto de cena, e só
  um template de região pode renderizar contexto; a dependência é sempre
  cena → região, nunca o contrário.
- **Renderização versionada.** `region-scene-context/v1` tem as instruções de
  `region/v1` mais `scene_context_instructions`. O prompt ganha uma seção
  `Scene context (scene-context/1)`: as instruções de enquadramento ("hipótese
  anterior, não truth; use só para desambiguar a região") seguidas do contexto em
  JSON canônico (chaves ordenadas): os seis campos estruturados e, por claim,
  hipótese, role, categoria, region kind e atributos. A confiança nunca é
  renderizada. `region/v1` continua byte a byte igual: não tem seção de contexto.
- **Desligável para uma ablação pareada.** `SemanticPromptPolicy.region_scene_context`
  liga o condicionamento (exige um template de região que renderize contexto).
  Desligado com o mesmo template, o request não carrega contexto e a seção diz
  explicitamente que nenhum foi fornecido; os dois braços usam o mesmo template e
  diferem só no insumo de contexto. Com `region/v1` não há contexto nenhum.
- **Nunca descartado em silêncio.** Um template que não renderiza contexto recusa
  um request que carrega um, antes da inferência.
- **Adapters.** Qwen e Gemini declaram `accepts_scene_context=True` desde que esse
  caminho existe e está testado; Florence-2 continua sem suporte (não aceita
  `prompt_policy`, e sua declaração segue `False`). O runtime recusa, na
  composição, `region_scene_context` para um intérprete que não aceita contexto.
- **Runtime.** A bridge guarda só o `SceneContext` do frame corrente, produzido
  pelo request de cena (que roda antes das regiões do mesmo frame na ordem
  topológica determinística do preset). Se o frame não tem contexto (cena falhou,
  abstenção ou não executada), cada request de região condicionado falha com erro
  explícito em vez de seguir sem o contexto pedido.
- **Avaliação.** O gancho com/sem contexto de `compare_evidence_variants()` roda
  sobre execuções reais de Qwen e Gemini (runtimes/clients fake na CI): o canal
  `scene_context` vem do request, e o template é o mesmo nos dois braços.

## Política de request resolvida (#544)

`SemanticRequestPolicy` reúne o que molda todos os requests semânticos de um run: o
prompt de cada modo interpretado (`SemanticModePrompt`: `template_id` e schema de saída;
um modo sem prompt não é interpretado e nunca recebe um padrão), a `SemanticViewPolicy`
e o interruptor `region_scene_context`. `SemanticRequestPolicy.from_prompt_policy()` a
resolve a partir da `SemanticPromptPolicy` e da política de views de um intérprete que
segue instruções; o Florence-2 recebe o prompt nativo da task só no modo que ela serve.
`to_document()` registra todas as escolhas (inclusive as regras das views) sob a versão
`semantic-request-policy/1`, e `fingerprint()` é o SHA-256 desse documento: seleções
equivalentes têm a mesma identidade, e mudar prompt, views, ordem das views ou contexto a
muda. `check_request_policy_supported()` recusa, antes de qualquer inferência, uma política
que o intérprete declara não consumir. O backend, o modelo e o orçamento visual continuam
configuração do backend, fora da política.

## Materialização e persistência

`assemble_perception_result()` recebe explicitamente os ids dos stages que
produziram `SemanticInterpretationExecution` e materializa
`execution.parsed.claims`/`scene_context` no `PerceptionResult`, validando as
identidades da observação e do resultado. Ao receber os mesmos outcomes,
`PerceptionRunWriter` persiste a execução em
`outputs/semantic-interpretations.jsonl` e materializa a resposta bruta no path
de debug declarado pela proveniência. Assim, execução, evidência canônica e
artifact permanecem ligados pelo mesmo request id. Os diagnostics persistidos
incluem `visual_inputs` (`null` quando o backend não mede o que o processor
fez de cada view); um registro gravado antes do #526, sem a chave, é lido como
não medido, e artifacts do schema 0.5.0 continuam legíveis.

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
parser canônico. Modelo, device, precision, quantização, token limit,
temperature e, quando configurado, o orçamento de entrada visual
(`min_pixels`/`max_pixels`) formam o fingerprint e permanecem disponíveis na
configuração efetiva da execução.

O seam de runtime mantém Transformers/Torch e objetos Qwen fora dos contratos.
Falha ou indisponibilidade de Qwen é propagada; não existe fallback implícito.
Métricas de tokens, latência, memória, warnings e o que cada view virou na
entrada do modelo (`visual_inputs`) são registradas quando o runtime consegue
medi-las. A cobertura CI usa runtime fake determinístico. Um
diagnóstico com Qwen3-VL-4B real em três requests REGION motivou a tolerância
registrada para `scene_context` omitido (#340).

### Runtime Transformers (`HuggingFaceQwenRuntime`)

`HuggingFaceQwenRuntime` implementa `QwenRuntime` com `AutoProcessor` e
`AutoModelForImageTextToText`, de modo que a mesma classe carrega Qwen2.5-VL e
Qwen3-VL. As importações de `torch`, `transformers` e `PIL` são lazy: sem elas o
runtime falha com `QwenDependencyError`, nunca com fallback para outro backend.

- **Identidade imutável.** O runtime exige `QwenSemanticConfig.revision`, um SHA
  de commit completo do Hugging Face. O campo é opcional no seam (fakes e
  gateways não precisam dele), mas entra no fingerprint quando presente, e o
  runtime recusa `revision=None`. Por padrão só lê o cache local
  (`local_files_only=True`).
- **Quantização realmente aplicada.** `quantization="4bit"` carrega com
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=<precision>)` e `"8bit"` com `load_in_8bit=True`
  (LLM.int8). Depois do load, o runtime confere que o modelo reporta
  `quantization_config`; um modelo que ignorou o pedido causa
  `QwenModelLoadError` em vez de rodar silenciosamente em precisão plena.
  Quantização exige device CUDA. Como ela altera saída e custo, faz parte da
  configuração efetiva e do fingerprint.
- **Decoding explícito.** `temperature=0` usa decoding guloso e anula
  `top_p`/`top_k` herdados do `generation_config` do checkpoint; um valor
  positivo amostra com essa temperatura e usa os defaults do checkpoint (fixado
  pela revisão).
- **Orçamento de entrada visual (#526).** `min_pixels`/`max_pixels` limitam a
  contagem de pixels (altura × largura) de **cada** view depois do resize do
  processor, que preserva o aspecto e arredonda cada lado para múltiplo de
  `patch_size × merge_size` (28 px no Qwen2.5-VL, 32 px no Qwen3-VL); cada
  quadrado com esse lado é um token visual. O orçamento não limita o total do
  request: com N views, o custo visual é a soma das N. Os dois limites são
  configurados juntos ou nenhum, porque um orçamento parcial deixaria o outro
  limite no default não registrado do checkpoint. Configurado, o orçamento é
  passado a `AutoProcessor.from_pretrained(min_pixels=..., max_pixels=...)`, o
  caminho documentado pelo Qwen2.5-VL: no transformers 5.x essas chaves
  substituem as do `preprocessor_config.json` e viram
  `image_processor.size = {shortest_edge, longest_edge}`, também no Qwen3-VL
  (passar só `size` seria sobrescrito pelas chaves do Qwen2.5-VL). Antes de
  carregar os pesos, o runtime confere em `image_processor.size` que o orçamento
  foi aplicado; um processor que o ignorou, ou um `max_pixels` menor que
  `(patch_size × merge_size)²` (o menor tamanho que o processor produz), é
  `QwenModelLoadError`. Sem orçamento, nada é passado ao processor e vale o
  default do checkpoint na revisão fixada (Qwen2.5-VL: 3 136 a 12 845 056 px;
  Qwen3-VL-4B: 65 536 a 16 777 216 px, isto é, até 16 384 tokens visuais por
  imagem); as chaves ficam fora da configuração efetiva, e o fingerprint de uma
  configuração anterior ao #526 não muda. O resize do processor é o único: o
  runtime entrega a imagem decodificada dos bytes verificados, sem resize
  próprio. Não existe limite de tokens visuais por request, porque o processor
  não o aplica e impô-lo exigiria uma política de resize do próprio runtime.
- **Evidência e prompt.** As views chegam como imagens, na ordem do request,
  seguidas do prompt renderizado da política que o request seleciona
  (`region/v1`, `scene/v1` ou outra entrada do catálogo). O runtime não acrescenta
  instrução própria. As referências são resolvidas dentro de
  `view_root` e não podem escapar dele, e cada imagem é decodificada dos bytes
  cujo SHA-256 foi verificado contra `SemanticVisualView.sha256`
  ([Integridade das views](#integridade-das-views-na-inferência)).
- **Diagnóstico.** Cada resposta traz tokens de entrada/saída e
  `peak_memory_bytes`, o pico de memória alocada na GPU pelo processo (pesos
  mais ativações, não a memória de outras sessões). Atingir `max_new_tokens`
  gera um warning, porque o JSON provavelmente foi truncado e essa falha de
  parsing não é falha semântica do modelo. `load()` permite carregar antes de
  medir latência, para que o load único não seja atribuído à primeira request.
- **Entrada visual medida.** `SemanticBackendDiagnostics.visual_inputs` traz,
  para cada view e na ordem do request, altura e largura em pixels depois do
  resize e os tokens visuais, medidos do `image_grid_thw` do processor
  (`h·patch_size × w·patch_size` px e `t·h·w / merge_size²` tokens);
  `input_tokens` já inclui esses tokens. Um processor que não devolve
  exatamente um grid por view é `QwenInferenceError`: nenhuma imagem é
  descartada ou acrescentada em silêncio. As medições são diagnóstico da
  execução, nunca identidade. Assim, cada registro de execução (e de falha de
  parsing) guarda juntos o número de views (o request), o orçamento
  (configuração efetiva) e o que cada view virou (diagnostics).
- **Falta de memória.** `torch.cuda.OutOfMemoryError` durante a request vira
  `QwenOutOfMemoryError`, subclasse de `QwenInferenceError`; não há retry,
  resize nem fallback. Como o estágio que falha guarda só a mensagem, ela nomeia
  o número de views, o orçamento em vigor (o configurado ou o default do
  checkpoint, lido do processor carregado), o que cada view mediu,
  `input_tokens` e o pico de memória: um arm que estoura memória é atribuível a
  um orçamento registrado, não a um default desconhecido. Falta de memória em
  CPU não tem tipo próprio no torch e continua `QwenInferenceError`.

O runtime não corrige nem reinterpreta a resposta: o texto gerado segue para o
parser canônico, e uma resposta fora do schema continua sendo falha explícita de
parsing.

## Adapter Gemini

`GeminiSemanticInterpreter` usa o mesmo request, a mesma política de prompt
selecionada pelo request e o mesmo parser do adapter local. `GeminiSemanticConfig` contém somente model, timeout, retries e settings
de geração/raciocínio; credenciais pertencem ao `GeminiClient` injetado e nunca
entram no fingerprint, outputs ou debug. Falhas transitórias possuem retries
limitados e contados; resposta vazia/bloqueada e retries esgotados terminam com
erro explícito, sem substituição por outro backend. Usage, latência, warnings e
identidade do provider permanecem auditáveis.

`GeminiSemanticConfig` também registra `structured_output` (pede
`response_mime_type=application/json`; o schema continua no prompt renderizado
da política versionada, porque a API aceita só um subconjunto de JSON Schema e não há como
validá-lo sem chamada real) e `retry_backoff_s`, a base do backoff exponencial
entre tentativas (tentativa `n` espera `retry_backoff_s * 2**(n-1)`, limitada a
60 s). Sem `retry_wait` injetado, o adapter dorme esse tempo, para que um 429
não seja repetido imediatamente.

### Cliente `google-genai` (`GoogleGenAIGeminiClient`)

`GoogleGenAIGeminiClient` implementa `GeminiClient` com o SDK oficial
`google-genai`, importado de forma lazy (`GeminiDependencyError` se ausente; não
há extra em `pyproject.toml`, seguindo o padrão de torch/transformers). Construí-lo
não contata o serviço; **`generate` envia os bytes das views e o prompt para a
API do Google**, então quem o chama precisa ter decidido que aqueles frames
podem sair da máquina.

- **Credencial.** Chave explícita ou variável `GEMINI_API_KEY` (nome
  configurável). Sem chave, `GeminiCredentialError` na construção. A chave é
  privada, não aparece em `repr`, e é removida de toda mensagem levantada; as
  exceções do provedor não são encadeadas (`from None`), pois o texto delas
  poderia ecoar a chave. Nada da credencial entra no fingerprint, na
  configuração efetiva, nos outputs nem no debug.
- **Requisição.** As views seguem em ordem como partes inline (`png`, `jpeg` ou
  `webp`, pelo sufixo do payload), depois o prompt renderizado. `temperature`,
  `thinking_budget`, `structured_output` e o timeout por tentativa
  (`timeout_s`, em milissegundos no SDK) vêm da configuração.
- **Views verificadas antes do envio.** O cliente lê e confere o SHA-256 de todas
  as views contra `SemanticVisualView.sha256` antes de entregar o primeiro byte
  ao SDK. Um payload divergente, ausente, fora de `view_root` ou de formato não
  aceito é `GeminiSemanticError` terminal e nenhuma requisição é feita: bytes
  enviados a um serviço remoto não podem ser recolhidos
  ([Integridade das views](#integridade-das-views-na-inferência)).
- **Falhas.** Timeout, falha de transporte, HTTP 408/429 e 5xx são
  `GeminiTransientError` e entram nos retries do adapter. Demais erros HTTP
  (400/401/403/404), exceções inesperadas, prompt bloqueado, resposta sem
  candidatos, `finish_reason` diferente de `STOP`/`MAX_TOKENS` e texto vazio são
  `GeminiSemanticError` terminais e nunca são repetidos nem substituídos por outro
  backend. `MAX_TOKENS` devolve o texto com warning de truncamento. Saída
  malformada continua sendo `SemanticResponseParseError` do parser
  compartilhado, sem retry.
- **Usage.** `input_tokens` vem de `prompt_token_count`; `output_tokens` soma
  `candidates_token_count` e `thoughts_token_count`, porque os tokens de
  raciocínio são faturados como saída. Ausência de metadata vira `None`, não zero.

Não existe execução real do Gemini: `GEMINI_API_KEY` não está configurada e
enviar frames a um serviço externo exige consentimento explícito. A validação é
fake/contract: testes com módulos SDK falsos (rodam na CI) e testes que usam o
SDK real com `httpx.MockTransport` (pulados se o SDK não está instalado), que
fixam o formato da requisição e o mapeamento dos erros reais sem rede.

## Adapter Eagle 2.5

`EagleSemanticInterpreter` executa o VLM Eagle 2.5 (NVlabs) pelo mesmo request, a mesma
política de prompt selecionada pelo request e o mesmo parser `UNSCORED_ONLY` de Qwen e Gemini,
sem modelo de evidência próprio. Declara os modos `SCENE`/`REGION` e todos os tipos de view,
sem features nem contexto de cena. Várias views viram várias imagens de uma mesma mensagem, na
ordem do request. `EagleSemanticConfig` exige, além de modelo, revisão, device, precisão e
geração, o orçamento visual do processor `eagle_2_5_vl` (`max_dynamic_tiles`, mais
`min_dynamic_tiles` e `use_thumbnail`), que entra no fingerprint; o runtime transformers
(`HuggingFaceEagleRuntime`, `trust_remote_code` só na revisão fixada e só do cache local) envia
esse orçamento explicitamente ao processor e recusa, antes da geração, tiles que o excedam.
Detalhes, o que vem do upstream e o que é adaptado, e a licença não comercial dos pesos estão em
[`eagle2_5.md`](eagle2_5.md). Não há execução real registrada.

## Adapter Florence-2

`Florence2SemanticInterpreter` é separado de `Florence2RegionDiscovery` mesmo
quando ambos compartilham lifecycle/modelo no composition root. Sua
`Florence2SemanticConfig` fixa checkpoint, revisão imutável, task, modes
suportados, device, precision e geração. A task e o mode entram em
`task_identity`; checkpoint, revisão e configuração entram na provenance e no
fingerprint. A task também é a política de prompt (`prompt_template_id`
`florence2-task-prompt/1:<task>`). A saída passa pelo mesmo parser canônico com
`UNSCORED_ONLY`.

### Decisão de design: task token versus JSON canônico

Florence-2 é dirigido por *task tokens* e responde texto puro (por exemplo,
`<REGION_TO_CATEGORY>` devolve `door`). O boundary canônico exige JSON
`semantic-response/1`, e o prompt canônico (instruções mais JSON Schema) não é
algo que o modelo entenda. A decisão foi:

1. **O adapter é dono do mapeamento, não o runtime.** O `Florence2SemanticRuntime`
   devolve o texto nativo da task, após o parser oficial do processor. A regra
   `florence2-task-envelope/1` (`_canonical_response_json`) é uma função pura,
   testável sem transformers: o texto vira **exatamente uma claim `primary`**,
   com `hypothesis` igual ao texto, sem `category`, `region_kind`, atributos nem
   confidence (`null`). Nunca há alternativas. No modo `scene`, a claim fica em
   um `scene_context` vazio, porque nenhum campo de cena (tipo, ambiente,
   iluminação, navegabilidade) pode ser derivado do texto sem heurística. Texto
   vazio vira `abstained=true`, uma abstenção explícita com warning, e nunca uma
   claim inventada.
2. **O texto só é publicado como claim depois do parser compartilhado.** O JSON
   intermediário é serializado com `json.dumps`, então um texto que pareça JSON
   não consegue acrescentar claims, alternativas ou campos, e passa por
   `parse_semantic_response` como qualquer outro backend.
3. **O raw response é o texto do modelo.** `execution.raw_response` e o hash
   `raw_response_sha256` referem-se ao texto nativo da task; o envelope é
   reconstruível pela política e sua aplicação fica registrada no diagnostic
   `wrapped_task_text` do parsing.
4. **O prompt registrado é o prompt consumido (#542).** A política de prompt do
   Florence-2 é nativa da task (`florence2-task-prompt/1`): o input do modelo é o
   task token configurado, mais a caixa da view inteira nas tasks de região. O
   adapter expõe essa identidade em `prompt_template_id`
   (`florence2-task-prompt/1:<task>`, por exemplo
   `florence2-task-prompt/1:<REGION_TO_CATEGORY>`); todo request precisa nomeá-la,
   e o `RenderedSemanticPrompt` da execução registra exatamente o texto entregue ao
   runtime, com o fingerprint desse texto. Um request que nomeie um template livre
   (`region/v1`, `region-abstention/v1`, ...) é recusado antes da inferência, em
   vez de o template ser registrado como se o Florence-2 o tivesse visto; o
   runtime também recusa `prompt_policy` na configuração do Florence-2. Antes de
   #542 o adapter renderizava `region/v1`/`scene/v1` só como contrato de resposta,
   com um warning constante dizendo que aquilo não era input do modelo: numa
   ablação de prompt, o Florence-2 pareceria variar um input que nunca recebe.
5. **Tasks declaradas.** `FLORENCE2_SEMANTIC_TASKS` lista as tasks de texto:
   `<CAPTION>`, `<DETAILED_CAPTION>` e `<MORE_DETAILED_CAPTION>` (modo `scene`,
   view `FULL_FRAME`) e `<REGION_TO_CATEGORY>` e `<REGION_TO_DESCRIPTION>` (modo
   `region`). As tasks que produzem geometria (`<OD>`, `<REGION_PROPOSAL>`, ...)
   ficam de fora de propósito: pertencem a `Florence2RegionDiscovery`, e seus
   rótulos não são claims semânticas. Uma task serve um único modo, e
   `supported_modes` precisa coincidir com ele.
6. **Views aceitas.** Uma task de região recebe a view inteira como região
   (`<loc_0><loc_0><loc_999><loc_999>`), pois o request não carrega a caixa da
   região dentro de um frame completo ou de um crop contextual. Por isso só
   `TIGHT_CROP` e `MASKED_SUBJECT` são aceitos; qualquer outra view é rejeitada
   antes do modelo, assim como requests com mais de uma view (limite declarado em
   `max_visual_views=1`, que o preflight da política de views também verifica).

**Trade-offs aceitos.**

- Nada é fabricado e o mapeamento é determinístico e auditável, ao custo de
  claims pobres: uma legenda vira uma frase em `hypothesis`, não um conceito, e
  `casefold-exact/1` quase nunca a casa com um conceito anotado. O relatório
  deve ler isso como limitação do output do Florence-2, não como alucinação.
- Sem alternativas, a preservação de ambiguidade é impossível para este backend.
  A abstenção só acontece por texto vazio.
- `scene_context` não é estruturado, então as métricas de campos de cena do
  Florence-2 são vazias por construção.
- Alternativas rejeitadas: extrair substantivos ou atributos da legenda
  (fabricação por NLP ad hoc); usar `<OD>` para claims de cena (mistura Region
  Discovery); pedir JSON ao modelo (não suportado); devolver o JSON no runtime
  (mistura regra de domínio com o SDK e impede testar sem transformers).

### Runtime Transformers (`HuggingFaceFlorence2SemanticRuntime`)

Carrega o port transformers-nativo (`florence-community/Florence-2-*`) na
revisão fixada, com `Florence2ForConditionalGeneration` e `AutoProcessor`, sem
`trust_remote_code` e somente do cache local por padrão. Aplica o parser oficial
`post_process_generation` da task e remove os tokens `<loc_*>` que ecoam a caixa
de entrada nas tasks de região, pois repetem o input e não fazem parte da
resposta. Registra tokens, pico de memória de GPU e o mesmo `load()` explícito
do runtime Qwen. A imagem é decodificada dos bytes cujo SHA-256 foi verificado
contra `SemanticVisualView.sha256`, e um payload divergente é
`Florence2InferenceError` antes da inferência
([Integridade das views](#integridade-das-views-na-inferência)). Sem SDK, device
ou checkpoint disponível, falha com erro explícito.

## Avaliação

`contextmap.evaluation.semantic_interpretation` fornece um report comum para
Qwen, Gemini, Florence-2 e Eagle 2.5. O contexto registra reference-set, seleção, run,
artifact, pipeline digest e versão do evaluator. Cada amostra preserva request,
região, evidence variant, backend/model/config, prompt e métricas. Qualidade e
custo permanecem em blocos distintos. O baseline usa a policy versionada
`casefold-exact/1`. A comparação entre backends alinha os reports pela
identidade física de cada request (observação, região, modo e variante), não só
pelo `request_id`; ver
[Avaliação de Semantic Interpretation](../../evaluation/docs/semantic-interpretation.md).


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
- Eagle 2.5 implementa o mesmo boundary com orçamento visual explícito e
  verificado, ainda sem execução real;
- os runtimes reais de Qwen e Florence-2 e o cliente do Gemini verificam o
  SHA-256 de cada view antes de abrir a imagem ou enviar bytes ao provider;
- auditoria possui níveis explícitos e redaction de secrets;
- o harness de avaliação compara qualidade e custo sem Semantic Fusion, e só
  compara backends que interpretaram as mesmas observações, regiões, modos e
  variantes de evidência;
- testes determinísticos cobrem parsing, abstention, retries, materialização no
  `PerceptionResult` e reabertura do run artifact.

### Validação real e fake/contract

- **Real.** Qwen3-VL-4B (nf4 e int8, aplicados e verificados) e Florence-2
  (`florence-community/Florence-2-large`, tasks `<DETAILED_CAPTION>`,
  `<REGION_TO_CATEGORY>` e `<REGION_TO_DESCRIPTION>`) foram executados com os
  runtimes transformers do repositório sobre os 20 frames de corridor-02 (uma
  request de cena por frame e as 5 maiores regiões SAM2), com repetições. Os
  números estão em
  [Avaliação de Semantic Interpretation](../../evaluation/docs/semantic-interpretation.md):
  por exemplo, o Qwen3-VL-4B nf4 interpretou 17 de 20 cenas e 53 de 100 regiões
  (as demais falharam no parser, principalmente por omitir `confidence`), e o
  Florence-2 interpretou todas, com decoding determinístico e repetições
  idênticas.
- **Fake/contract.** O Gemini tem cliente `google-genai` validado apenas com
  transporte simulado; não há credencial nem consentimento para enviar frames.
  Runtime e adapters também têm testes com módulos SDK falsos, e o harness de
  avaliação usa execuções canônicas construídas em teste. Isso valida contratos,
  mapeamentos, falhas, redação de segredos e a aritmética do avaliador, não a
  qualidade de um backend. O orçamento de entrada visual do Qwen (#526) também
  só foi validado com processor e modelo falsos: o perfil real (1 a 4 views,
  vários orçamentos, nf4 e um arm não nf4, com tokens visuais, latência, pico
  de VRAM, taxa de OOM e qualidade do parser) exige GPU e continua pendente.

Não há anotações semânticas humanas para a amostra, então correção,
alucinação, abstenção esperada, campos de cena e visibilidade são N/A e nenhum
resultado sustenta que um backend é melhor que outro. A execução de referência
do Gemini e a avaliação com anotações continuam pendentes.

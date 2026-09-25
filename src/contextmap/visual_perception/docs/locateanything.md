# LocateAnything (Region Grounding)

Este documento descreve `src/contextmap/visual_perception/backends/locateanything.py`, o
adapter do NVIDIA LocateAnything atrás do port `RegionGrounding` (issue #567). O contrato
de grounding está em [`region-grounding.md`](region-grounding.md).

Referências upstream: [NVlabs/Eagle `Embodied/`](https://github.com/NVlabs/Eagle/tree/main/Embodied)
(README e `locateanything_worker.py`) e o model card
[`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B).

## O que o upstream propõe e o que adaptamos

| Upstream (LocateAnything) | ContextMap2 |
|---|---|
| VLM (MoonViT + Qwen2.5-3B) que gera texto com blocos atômicos de coordenadas (Parallel Box Decoding) | o texto gerado é a **resposta bruta**, persistida verbatim e separada do parsing |
| coordenadas inteiras normalizadas em `[0, 1000]`; `parse_boxes()` faz `x / 1000 * largura` | mesma conversão, na imagem preparada que o modelo recebeu; `0` e `1000` caem exatamente nas bordas |
| templates de prompt por tarefa no worker (`detect`, `ground_multi`, `point`, ...) | três políticas versionadas com os **mesmos** textos (ver abaixo) |
| modos `fast` (MTP), `slow` (NTP/AR) e `hybrid` (MTP com fallback para AR) | `generation_mode` explícito na configuração, parte do fingerprint |
| `verbose=True` devolve histórico de amostragem e um texto de estatísticas (`switch_to_ar=N`, ...) | guardados como diagnósticos **nativos brutos**, nunca como confiança |

O que é original do ContextMap2: a identidade de request por conteúdo, a validação de
capacidade antes da inferência, o parser total com rejeições explícitas, a regra de que um
ponto nunca vira caixa e a persistência auditável no `PerceptionRunArtifact`.

## Políticas de query

| `policy_id` | task | geometria | prompt renderizado (template upstream) |
|---|---|---|---|
| `locateanything.category-detection/1` | `CATEGORY_DETECTION` | `BOX` | `Locate all the instances that matches the following description: {c1}</c>{c2}...` |
| `locateanything.phrase-grounding/1` | `PHRASE_GROUNDING` | `BOX` | `Locate all the instances that match the following description: {frase}.` |
| `locateanything.pointing/1` | `PHRASE_GROUNDING` | `POINT` | `Point to: {frase}.` |

As categorias são unidas **na ordem da query** com o separador upstream `</c>`; uma
categoria que contenha `</c>` é recusada antes da inferência (seria dividida em duas). O
texto dos templates repete o do worker oficial, inclusive o "matches" de `detect()`: foi com
ele que o modelo foi treinado. Mudar um template exige outra identidade de política.

Um pedido que o adapter não declara (por exemplo `CATEGORY_DETECTION` com geometria
`POINT`) e um request cujo `configuration_fingerprint` não é o desta configuração falham com
`GroundingRequestError` **antes** de o runtime ser chamado.

## Gramática da resposta e parsing

`parse_locateanything_response()` é total: cada trecho da resposta vira uma saída, um
no-match explícito ou uma rejeição explícita com motivo. Nada é descartado ou "consertado".

| Trecho | Resultado |
|---|---|
| `<ref>rótulo</ref>` | define o rótulo das geometrias seguintes (guardado verbatim, nunca claim) |
| `<box><x1><y1><x2><y2></box>` | saída `BOX` → `BoundingBox2D` em pixels → `Region2D` |
| `<box><x><y></box>` | saída `POINT` → `GroundingPoint`; **nunca** uma caixa |
| `<box>none</box>` | no-match explícito do rótulo corrente (`no_match_labels`) |
| `<\|im_end\|>`, `<\|endoftext\|>` | fim da resposta (`terminated`) |
| coordenadas que não são tokens inteiros (`<nan>`, `<-1>`, `<1.5>`, texto), 1, 3 ou 5+ coordenadas, `<box>` sem fechamento | rejeição `MALFORMED_GEOMETRY` |
| coordenada acima de 1000 | rejeição `COORDINATE_OUT_OF_RANGE` |
| `x2 <= x1` ou `y2 <= y1` (invertida ou degenerada) | rejeição `INVERTED_GEOMETRY` |
| qualquer outro texto, ou qualquer conteúdo depois do fim | rejeição `UNRECOGNIZED_TEXT` |

Saídas e rejeições compartilham o mesmo `output_index`, então a ordem da resposta sobrevive.
A gramática é estrita como o `parse_boxes()` oficial (sem espaços dentro do bloco). No modo
`fast` o upstream deixa passar blocos irregulares; eles chegam aqui e são rejeitados
explicitamente.

O adapter ainda registra warnings, sem mudar a evidência:

- resposta sem token de fim: possível truncamento em `max_new_tokens`;
- geometria diferente da pedida (ponto em pedido de caixa, ou o contrário): mantida como
  veio e contada em `unsupported_outputs` quando é ponto;
- histórico de decodificação que não reconstrói a resposta.

## Evidência, proveniência e diagnósticos

- `BackendProvenance`: `backend_id="locateanything"`, `capability="region_grounding"`,
  `provider="nvidia"`, `model` = id do modelo, `version` = revisão (commit) fixada e
  `configuration_fingerprint` = digest da configuração efetiva e de `PARSER_VERSION`.
- `effective_configuration` da execução registra todos os parâmetros e `parser_version`;
  `runtime_identity` registra o que o runtime reporta (versões de bibliotecas, dispositivo).
- Diagnósticos nativos da execução: `stats.raw` (texto de estatísticas verbatim) e, quando
  ele tem o formato do `generate()` oficial, cada par `stats.<nome>` com nome e valor
  nativos (`stats.switch_to_ar` é o número de fallbacks MTP → AR do modo `hybrid`).
- Diagnóstico nativo por saída: `decoder` = `mtp`, `ar` ou `mtp+ar` (bloco iniciado em MTP e
  refeito em AR), derivado do histórico de amostragem quando ele reconstrói a resposta.
- Nenhum desses valores é confiança, e nenhum é inventado quando ausente. O worker público
  não expõe probabilidade por caixa; calibração é trabalho separado (#573).

## Runtime e configuração reproduzível (#569)

O adapter só conhece o seam `LocateAnythingRuntime.generate(image, prompt, config) ->
LocateAnythingGeneration`. Torch, Transformers e o código remoto do modelo ficam atrás dele,
então a CI usa runtimes fake determinísticos e nunca baixa pesos.

### `LocateAnythingConfig`

| Parâmetro | Regra |
|---|---|
| `model` | id Hugging Face (`nvidia/LocateAnything-3B`) ou diretório local **absoluto** que seja o snapshot da revisão (`.../snapshots/<revision>`) |
| `revision` | **obrigatório**, commit SHA completo (40 hex); branch/tag são recusados — não existe perfil sem revisão fixada |
| `device` | `cpu`, `cuda` ou `cuda:<n>` |
| `dtype` | `bfloat16` (dtype da release), `float16` (não em CPU) ou `float32` |
| `generation_mode` | **obrigatório**: `fast`, `slow` ou `hybrid` |
| `max_new_tokens`, `temperature` | **obrigatórios**; o model card sugere 8192 tokens; `temperature=0` decodifica de forma gulosa |
| `top_p`, `top_k`, `repetition_penalty` | padrões do worker upstream (0,9; 0 = desligado; 1,1) |
| `text_attention` | **obrigatório**: `sdpa`, `eager`, `magi` (Hopper/Blackwell) e, só no runtime batch, `la_flash` |
| `vision_attention` | **obrigatório**: `sdpa`, `eager` ou `flash_attention_2` |
| `runtime` | `standard` (caminho do worker: `AutoModel` + `generate()` remoto) ou `batch` (`batch_utils` da release) |
| `scheduler`, `group_size` | só no runtime batch, e obrigatórios nele (`eager`, `hold_ar`, `ar_first`, `pipeline`, `adaptive`; `group_size >= 0`, 0 = o upstream escolhe) |
| `local_files_only` | `true` por padrão: só o cache local, sem download implícito |

Combinações recusadas na construção, antes de qualquer import: atenção desconhecida ou
`auto`; `la_flash` fora do runtime batch; atenção só-CUDA (`flash_attention_2`, `la_flash`,
`magi`) em CPU; `scheduler`/`group_size` fora do runtime batch; runtime batch sem CUDA, fora
do modo `hybrid` (o upstream só suporta esse) ou fora de `bfloat16` (o runtime batch não
expõe controle de dtype). O upstream resolveria as duas atenções com fallback silencioso
para SDPA; aqui elas são sempre explícitas.

O `fingerprint` cobre todos os parâmetros e `PARSER_VERSION`, exceto `local_files_only`:
com a revisão fixada, ele decide de onde vêm os arquivos, não quais nem como rodam. A
configuração efetiva completa, inclusive `local_files_only`, vai para cada execução.

### `TransformersLocateAnythingRuntime`

Runtime empacotado, lazy e ligado a uma configuração e à raiz das imagens preparadas:

- construir não importa nada; o primeiro `generate()` (ou `load()` explícito) importa `torch`,
  `transformers` e `Pillow`, e carrega o modelo **uma vez** — a composition root cria um
  runtime por run, então o modelo carrega uma vez por run resolvido;
- antes de carregar: device CUDA disponível; `flash_attn` instalado para
  `flash_attention_2`/`la_flash`; `magi_attention` instalado e GPU com compute capability
  ≥ 9.0 para `magi`;
- runtime `standard`: `AutoConfig`/`AutoTokenizer`/`AutoProcessor`/`AutoModel` com
  `revision`, `local_files_only` e `trust_remote_code=True` (o código do modelo vem do próprio
  repositório, na revisão fixada); a atenção configurada é escrita em `text_config` e
  `vision_config` e **conferida depois do load** — se o código remoto trocou de atenção, o
  load falha em vez de seguir com outra; commit carregado (`_commit_hash`) diferente da
  revisão e dtype diferente do pedido também falham;
- runtime `batch`: resolve o snapshot da revisão (`huggingface_hub.snapshot_download`, local
  por padrão), exige que o diretório seja o daquela revisão, põe o snapshot no `sys.path`,
  configura as variáveis `LA_FLASH_*` (efeito de processo, documentado) sempre com
  `LA_FLASH_STRICT_ATTN=1` (falhar em vez de cair para SDPA) e chama `batch_utils.load()`.
  Cada request é uma chamada de lote unitário; agrupar requests em lote fica para quando o
  harness de avaliação medir o ganho;
- cada imagem é lida da raiz das imagens preparadas e tem o SHA-256 conferido contra o
  `payload_artifact` do request antes de ser decodificada;
- `runtime_identity`: runtime, Python, `torch` (e CUDA), `transformers`, `flash_attn`/
  `magi_attention` quando usados, `huggingface_hub` no batch, nome do device e commit do
  modelo; `runtime.cold_load_ms` aparece nos diagnósticos nativos da chamada que carregou o
  modelo, e o pico de memória CUDA vai em `peak_memory_bytes`.

Falhas são explícitas e tipadas (`LocateAnythingDependencyError`,
`LocateAnythingDeviceError`, `LocateAnythingModelLoadError`,
`LocateAnythingInferenceError`) e nunca trocam de modelo, atenção ou runtime.

### Seleção pelo runtime

`visual_perception.region_grounding` é um ponto de variação **opcional** do estágio
`visual_perception`, com o backend `locateanything` e o grupo reservado `query_set`
(obrigatório). Ver [`composition.md`](../../runtime/docs/composition.md) e
[`executors.md`](../../runtime/docs/executors.md): as queries são validadas na composição,
antes de qualquer modelo, e o executor faz cada query a cada imagem como um request próprio.

## Licenças: código vs. pesos

São licenças **diferentes** e não podem ser confundidas:

| Artefato | Licença | Consequência |
|---|---|---|
| repositório `NVlabs/Eagle` (código do GitHub) | Apache-2.0 (`LICENSE` do repositório) | uso e redistribuição do código conforme Apache-2.0; note que alguns arquivos de `Embodied/` (ex.: `locateanything_worker.py`) trazem um cabeçalho proprietário da NVIDIA, então a licença de cada arquivo precisa ser conferida antes de copiá-lo |
| pesos `nvidia/LocateAnything-3B` e o código remoto publicado no repositório do modelo (`modeling_*.py`, `generate_utils.py`, `batch_utils/`, `kernel_utils/`) | NVIDIA License (`LICENSE` do repositório do modelo; `LICENSE_MODEL` no GitHub) | **somente uso não comercial**: pesquisa ou avaliação; uso comercial não é permitido (exceto pela NVIDIA e afiliadas); redistribuição precisa manter a licença e os avisos |
| componentes de terceiros do modelo | Qwen2.5-3B-Instruct (Qwen Research License); MoonViT-SO-400M (MIT) | as restrições de cada componente se somam às da NVIDIA License |

O ContextMap2 não vendoriza código nem pesos do LocateAnything: o adapter reimplementa o
parsing e reutiliza apenas os textos curtos dos prompts upstream, necessários para
interoperar com o modelo, e o runtime carrega pesos e código remoto do cache local do
usuário na revisão fixada. Todo experimento que execute o LocateAnything herda a restrição
não comercial dos pesos e deve registrá-la na metadata do experimento; a execução continua
sendo uma decisão de quem implanta.

## Limites atuais

- O runtime real nunca foi executado aqui (sem GPU e sem pesos): o mapeamento para a API
  upstream segue o worker, o `generate()` remoto e o `batch_infer.py` publicados, e é
  verificado só com SDKs fake. Tempo de carga a frio, VRAM de pico/estável, RAM, latência por
  request, throughput, escala do lote, falhas/OOM e contagem de fallback híbrido têm onde ser
  registrados (`runtime.cold_load_ms`, `peak_memory_bytes`, `latency_ms`, `stats.*`), mas
  ainda não foram medidos.
- Não há avaliação real: recall/cobertura de localização, IoU, falsos positivos,
  fragmentação, taxa de falha do parser, taxa de no-match, latência p50/p95, boxes/s,
  contagem de fallback híbrido e pico de memória exigem o slice de referência anotado e GPU
  (#528/#576).
- Refinamento de caixa em máscara (#568), associação 3D e identidade persistente estão fora
  do escopo.
- LocateAnything não é o backend default nem substitui Region Discovery.

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

## Runtime

O adapter só conhece o seam `LocateAnythingRuntime.generate(image, prompt, config) ->
LocateAnythingGeneration`. Torch, Transformers e o código remoto do modelo ficam atrás dele,
então a CI usa runtimes fake determinísticos e nunca baixa pesos.

## Limites atuais

- Não há avaliação real: recall/cobertura de localização, IoU, falsos positivos,
  fragmentação, taxa de falha do parser, taxa de no-match, latência p50/p95, boxes/s,
  contagem de fallback híbrido e pico de memória exigem o slice de referência anotado e GPU
  (#528/#576).
- Refinamento de caixa em máscara (#568), associação 3D e identidade persistente estão fora
  do escopo.
- LocateAnything não é o backend default nem substitui Region Discovery.

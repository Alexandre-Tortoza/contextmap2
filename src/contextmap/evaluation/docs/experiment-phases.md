# Fases de experimento e composições selecionadas (#527)

`contextmap.evaluation.experiment_phases` expande as fases nomeadas do experimento de percepção (#521) em **configurações comuns do runtime**, sem importar nem carregar nenhum modelo. Ele se apoia no manifesto de experimento ([`experiments.md`](experiments.md)), nos braços pareados do #545 e na matriz de capacidades do #522 ([`capability-matrix.md`](capability-matrix.md)). O runtime não pode importar `evaluation`; por isso a camada fica aqui e só chama as funções públicas de resolução do runtime.

## Declaração

| Tipo | Conteúdo |
|---|---|
| `PhaseSpec` | experimento, fase (`PhaseKind`), versão, propósito, estágio avaliado, seleção congelada do reference set, configuração base (overrides do perfil canônico), capacidades e fiação de base, fatores, estágios-alvo, artifacts pinados, composições selecionadas, parâmetros casados, métricas e controles fixos |
| `Factor` | nome, `VariationKind`, níveis e nível baseline |
| `FactorLevel` | overrides `components.*` e `pipeline.stages.*` da configuração do runtime, capacidades que nenhum componente seleciona (um scorer, uma fonte 3D bloqueada) e arestas produtor → consumidor |
| `MatchedParameters` | parâmetros que todo arm executável dá ao backend selecionado de um componente (política de prompt e de views de Qwen × Eagle 2.5, por exemplo) |

Os nomes dos campos da configuração são os do runtime: `components.visual_perception.semantic_interpretation.<backend>.prompt_policy` (com `region_scene_context` para Qwen e Gemini, #529), `...<backend>.view_policy` (`region_views` e os parâmetros de cada view, #524), `...qwen.min_pixels`/`max_pixels` e `components.visual_perception.region_grounding.locateanything.query_set`. A camada não conhece nenhum campo específico: uma política nova entra como override sem mudar o código.

## As onze fases

| `PhaseKind` | Fatores permitidos | Células |
|---|---|---|
| `backend_grid` | `backend` (vários) | um nível por vez |
| `native_task` | `configuration` (um) | um nível por vez |
| `grounding_refinement` | `topology` (um) | um nível por vez |
| `feature_backend` | `backend` ou `evidence_channels` (um) | um nível por vez |
| `semantic_interpreter` | `backend` (um) | um nível por vez |
| `semantic_prompt` | `policy` (um) | um nível por vez |
| `visual_view`, `scene_context` | `evidence_channels` (um) | um nível por vez |
| `visual_budget` | `configuration` (um) | um nível por vez |
| `pipeline_composition`, `downstream_confirmation` | qualquer tipo | só o baseline e as composições **selecionadas**, nunca o produto cartesiano |

Mudança de topologia (inserir o refino, ligar o grounding) é sempre um fator `topology`, distinto de uma troca de backend.

## Expansão

`expand_phase(spec)`:

1. resolve cada arm pelo runtime: `resolve_effective_config` (base + overrides dos níveis), `resolve_plan` (problemas estruturais) e `check_selection` nos componentes dos estágios do arm. Um arm que não resolve **falha a fase**. Os parâmetros de cada backend só são validados na composição, que carrega o adapter; por isso não entram aqui;
2. quando o arm interpreta semântica, resolve sua política de requisição com `resolve_semantic_request_policy` (#544), sem compor backend, e registra `request_policy_fingerprint`; uma política inválida (template de região que não renderiza contexto de cena, `view_policy` ausente) **falha a fase**;
3. monta a topologia do arm: os alvos e todos os estágios a montante, com **um nó por ponto de variação selecionado** (`visual_perception.region_refinement`, ...), cuja configuração é o bloco resolvido do componente; um nó depende dos nós dos estágios de entrada, e os estágios pinados levam o artifact imutável;
4. mapeia cada seleção inventariada pela matriz para suas capacidades (o `task` do Florence-2, a `strategy` do SAM3, o `policy_id` de cada query do LocateAnything). Uma seleção que a matriz não reconhece (SAM3 `automatic`, por exemplo) **falha a fase**;
5. confere a fiação: uma aresta cujo extremo não está no arm, não declarada, `incompatible` ou `not_scientifically_comparable` **falha a fase**; uma capacidade ou aresta `planned`/`blocked` mantém o arm, `blocked`, com o motivo exato da matriz e a issue (scorers semânticos #527, prompt visual do LocateAnything #574, LocateAnything3D #575);
6. deduplica de forma determinística: arms com o mesmo digest de configuração efetiva, a mesma topologia e a mesma fiação ficam `skipped`, com `duplicate_of` apontando o primeiro na ordem das células;
7. confere os parâmetros casados e que uma substituição **só de backend** mantém a tarefa: cada capacidade trocada precisa de uma contraparte do mesmo grupo de comparação (Qwen × Florence-2 `<REGION_TO_CATEGORY>` é recusado);
8. com dois ou mais arms executáveis, monta o `ExperimentManifest` em modo `selected`, cuja construção aplica o #545: qualquer diferença não declarada entre um arm e o baseline falha a fase. Os campos de configuração declarados vêm dos overrides de cada fator.

Um baseline bloqueado falha a fase: sem ele nada se compara.

## Manifesto da fase

`write_phase_manifest(root, manifest)` publica `phase.json` (`contextmap.experiment-phase/v1`), imutável e com digest. Ele guarda a declaração inteira (experimento, fase, fatores e níveis, células, composições, restrições: alvos, pinados, parâmetros casados), o digest da configuração base, a identidade do reference set, a versão da matriz, a identidade do `ExperimentManifest` e, por arm: id, atribuição dos fatores, estado e motivo, issues, digest da configuração efetiva, digest da topologia, identidade da política de requisição semântica, capacidades, fiação e `duplicate_of`. Reexpandir a mesma declaração produz o mesmo documento.

| `ArmState` | Quando |
|---|---|
| `planned` | resolvido e executável |
| `blocked` | usa capacidade ou aresta planejada ou bloqueada; motivo e issues registrados |
| `skipped` | duplicata de outro arm, ou backend indisponível na execução |
| `executed` / `failed` | resultado de `record_phase_outcomes(manifest, run)` depois de `run_experiment`; erro e OOM ficam em `failed` com tipo e mensagem |

`record_phase_outcomes` devolve um novo manifesto (outra identidade) e nunca remove um arm; ele recusa um run de outro experimento.

## Limites

- A validação dos parâmetros de cada backend só ocorre na composição; a expansão garante a resolução, a topologia e a seleção.
- Ainda não existe executor real (#528); a topologia por ponto de variação faz o executor reportar um artifact por nó, mesmo quando vários nós saem do mesmo run de percepção.
- A fiação dentro de um estágio (grounding → refino) é registrada e verificada contra a matriz, mas não vira dependência da topologia: o grounding é recomputado no mesmo run e não pode ser pinado à parte.

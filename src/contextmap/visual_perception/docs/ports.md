# Capability ports

Este documento descreve `src/contextmap/visual_perception/ports.py`: os cinco `Protocol`s que backends concretos de percepção visual podem implementar.

## Mapeamento planejado capability/backend

```text
RegionDiscovery
├── SAM2
├── SAM3
└── Florence-2 (adapter de region discovery)

FeatureExtractor
├── DINOv2
├── DINOv3
├── CLIP
└── AlphaCLIP

FeatureResolutionEnhancement
└── backend aprendido futuro (opcional; nenhum default nesta milestone)

SemanticInterpreter
├── Qwen
├── Gemini
└── Florence-2 (adapter de semantic interpretation)

SemanticScorer
├── CLIP
└── AlphaCLIP
```

Um mesmo modelo pode satisfazer mais de uma capability através de adapters **distintos** — Florence-2 como `RegionDiscovery` e Florence-2 como `SemanticInterpreter` continuam sendo dois adapters de capability separados, mesmo compartilhando o runtime do modelo internamente. O mesmo vale para CLIP como `FeatureExtractor` vs. CLIP como `SemanticScorer`.

## Nenhuma sequência obrigatória

Os ports não codificam `RegionDiscovery -> FeatureExtractor -> SemanticInterpreter` como uma sequência fixa. Cada capability declara o que exige/produz; o grafo de execução resolvido (`docs/architecture.md`/issue #50/#55) decide a ordenação real. Exemplos:

```text
DINOv3 (dense)         requires: PreparedImage                    provides: VisualFeature[] (dense)
resolution enhancement requires: DenseFeatureMap                  provides: DenseFeatureMap
AlphaCLIP (region)     requires: PreparedImage + Region2D[]       provides: VisualFeature[] (region)
Gemini (region)        requires: PreparedImage + Region2D[]       provides: SemanticClaim[]
CLIP (scorer)          requires: SemanticClaim[] + PreparedImage  provides: SemanticSupport[]
```

## `PreparedImage`

Contrato canônico de "imagem pronta para os backends consumirem" (`contextmap.visual_perception.models.PreparedImage`) — qualquer implementação de preparação de imagem (resize, rectification, crop, máscara de área válida...) deve produzir esse formato. Os parâmetros específicos de cada transformação não fazem parte deste contrato central; apenas um registro legível (`transformations: Sequence[str]`) de quais passos foram aplicados.

## `FeatureExtractor.required_scope()`

Em vez de multiplicar tipos de port por escopo (dense/global/region), um único `FeatureExtractor` declara seu escopo via `required_scope()`. Um extrator dense/global só recebe `image`; um extrator region-scoped também recebe `regions`. Isso evita forçar uma assinatura mandatória `PreparedImage + Region2D[]` em extratores que não precisam de regiões (ex.: DINOv3 dense).

## `FeatureResolutionEnhancement`

É um port separado porque possui input/output de artifact, custo, falha e ativação próprios. Recebe um `DenseFeatureMap` já produzido e devolve outro `DenseFeatureMap`, com lineage completa em `FeatureResolutionEnhancementProvenance`. Não é um modo interno do DINO e não aparece no caminho canônico quando ausente da configuração.

## `SemanticScorer` nunca muta uma claim

`score()` retorna `SemanticSupport[]` — um julgamento de suporte separado, referenciando `claim_id` — nunca modifica a `SemanticClaim` original. Isso preserva a claim como evidência imutável.

## Metadata de backend

Todo port expõe `backend_provenance() -> BackendProvenance` (issue #48) — identidade do backend, capability satisfeita, provider/model/version, e um fingerprint de configuração opcional. Nenhum port expõe objetos nativos do SDK do modelo.

## Substituibilidade

Qualquer classe que implemente os métodos de um port satisfaz esse port (`Protocol` com `@runtime_checkable`) — código de orquestração nunca precisa de `isinstance(backend, Sam2RegionDiscovery)` ou qualquer branch por modelo. `tests/visual_perception/test_ports.py` demonstra isso com duas implementações fake de `RegionDiscovery` sendo usadas de forma intercambiável pelo mesmo código consumidor.

## O que este contrato não define

Instanciação de backends concretos (pertence à composition root/`runtime`), configuração específica de modelo, e a ordem real de execução (issues #50/#55) ficam fora do escopo desta issue.

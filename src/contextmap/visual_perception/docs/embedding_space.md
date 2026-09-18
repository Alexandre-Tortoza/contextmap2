# Espaço de embedding e regra de compatibilidade

Este documento descreve `src/contextmap/visual_perception/embedding_space.py`: o que a referência opaca `VisualFeature.embedding_space_id` (`contracts.md`, issue #48) realmente identifica, e como a comparabilidade entre features é validada explicitamente.

## `EmbeddingSpace`

Identidade do espaço vetorial de onde vem o payload de um `VisualFeature`: `family` (ex. `"dinov3"`, `"clip"`), `model`, `version`, `checkpoint` opcional, `layer`/projeção opcional, `dimension`, `normalization` opcional. Nenhum objeto de SDK de modelo ou tensor aparece aqui — apenas identidade e metadados primitivos, serializáveis.

## Regra central: fingerprint exato, nunca dimensão

Duas features só são comparáveis (similaridade de cosseno, média, indexação, scoring) quando seus espaços de embedding têm o **mesmo fingerprint exato** — nunca apenas porque compartilham `dimension`.

```text
DINOv3 (dimension=1024) != CLIP (dimension=1024)         # famílias diferentes
CLIP checkpoint A != CLIP checkpoint B automaticamente    # checkpoints diferentes
```

`embedding_space_fingerprint(space)` computa um hash determinístico (`"sha256:<hex>"`) sobre todos os campos de `EmbeddingSpace` — dois espaços produzem o mesmo fingerprint se e somente se todo campo é igual. Este é o valor que um backend real grava em `VisualFeature.embedding_space_id`.

## Duas formas de validar compatibilidade

- `ensure_compatible_embedding_spaces(a, b)` — quando os dois `EmbeddingSpace` completos estão disponíveis (ex.: ao configurar dois backends para uma operação combinada).
- `ensure_compatible_features(feature_a, feature_b)` — o caso comum downstream: apenas os `VisualFeature`s persistidos estão disponíveis, com seu `embedding_space_id` opaco; compara os ids diretamente sem precisar resolver o `EmbeddingSpace` completo de volta.

Ambas levantam `EmbeddingSpaceMismatchError` em caso de incompatibilidade — nunca uma comparação silenciosa.

## O que este módulo não faz

- Não mantém um registro/resolvedor que reconstrua um `EmbeddingSpace` completo a partir de um `embedding_space_id` — nenhum consumidor real dessa operação existe ainda neste milestone (YAGNI); cada backend concreto (#68-71) conhece seu próprio `EmbeddingSpace` no momento em que produz uma feature.
- Não decide automaticamente que dimensões iguais implicam compatibilidade — essa é exatamente a falha que este módulo existe para prevenir.
- Não embute o `EmbeddingSpace` inteiro dentro de cada `VisualFeature` — o mesmo padrão de referência opaca que Ingestion usa para `calibration_id` (`docs/calibration.md`), evitando duplicar metadados em cada feature individual.

# Avaliação de técnicas opcionais

Duas técnicas são opcionais e só podem influenciar o perfil canônico com evidência **da nossa própria pipeline e dos nossos cenários de referência**; métricas de artigos externos são contexto, nunca evidência de aceite:

1. **Resolução de features** entre a extração densa (DINO) e Sensor Association: features nativas × o mesmo artifact de extração passado por `FeatureResolutionEnhancement`.
2. **Fusão semântica ciente de qualidade**: fusão uniforme × fusão ponderada pelo `ObservationQuality` canônico, sobre exatamente o mesmo artifact de Sensor Association.

Esta avaliação **não altera o perfil canônico/default**. Executada sobre dados reais, ela produz evidência que pode sustentar uma decisão de configuração explícita e posterior, e a re-validação pelo fluxo de aceite E2E. **Hoje só existe o protocolo e o harness**: nenhuma execução real foi feita, então não há evidência nem decisão.

Os módulos são `contextmap.evaluation.technique_protocols` (definição dos experimentos) e `contextmap.evaluation.technique_evidence` (evidência e registro de decisão). Eles se apoiam nos manifestos e na execução controlada de [`experiments.md`](experiments.md), no registro de métricas de [`metrics.md`](metrics.md) e no reference set de [`reference-set.md`](reference-set.md).

## Protocolos

Um `TechniqueProtocol` é um conjunto de `ExperimentManifest` que **só diferem no estágio avaliado**. Um relatório mede um estágio, então há um experimento por estágio medido, todos comparando as mesmas topologias resolvidas e a mesma seleção de amostras. Cada experimento já é uma comparação controlada: só a variável declarada varia e o trecho variado consome artifacts imutáveis pinados. Portanto os critérios de aceite abaixo valem **por construção**:

| Critério | Como é garantido |
|---|---|
| features nativas e melhoradas compartilham a **mesma identidade de artifact DINO** | o estágio `dense_feature_extraction` é upstream de um estágio tocado, logo é pinado; um arm com outro artifact é `undeclared change` |
| fusão uniforme e ciente de qualidade compartilham a **mesma evidência de Association/FusionSupport** | `sensor_association` pinado com o mesmo artifact nos dois arms; só a configuração de `semantic_fusion` (variável `policy`) difere |
| nenhum parâmetro ajustado escondido por arm | qualquer mudança de backend, configuração, dependência ou artifact fora do declarado recusa o manifesto |
| falha ou indisponibilidade do backend de enhancement fica explícita | ver "Evidência", abaixo |

### Experimento A — `build_feature_resolution_protocol()`

```text
baseline: dense_feature_extraction (artifact X) ─────────────────────────► sensor_association ─► fusão ─► entidades ─► relações
variant:  dense_feature_extraction (artifact X) ─► feature_resolution_enhancement ─► sensor_association ─► fusão ─► entidades ─► relações
```

Variável `feature_resolution` (`topology`): toca `feature_resolution_enhancement` e `sensor_association`. Fixos e pinados: observações de origem, imagens preparadas, calibração, pose, mapa geométrico e o artifact de extração; as políticas a jusante não mudam, só as consequências do mapa de features alterado.

### Experimento B — `build_quality_aware_fusion_protocol()`

```text
sensor_association (artifact A, pinado) ─► semantic_fusion [uniforme | ciente de qualidade] ─► entidades ─► relações
```

Variável `fusion_policy` (`policy`): toca só `semantic_fusion`. As métricas de Association não entram (o artifact é o mesmo nos dois arms).

## Métricas: separadas, nunca um score único

Cada estágio mede suas métricas (registro v2, ver [`metrics.md`](metrics.md)); qualidade e custo ficam separados, e não há score geral.

| Estágio | Métricas de qualidade (A) | (B) |
|---|---|---|
| `sensor_association` | `association.reprojection_error.median`, `association.visible_support.ratio`, `association.feature_anchoring.rate` (diagnóstico de ancoragem/amostragem de features densas) | — |
| `semantic_fusion` | `fusion.reference_recovery.rate` (correção), `fusion.view_consistency.rate` (consistência), `fusion.ambiguity_retention.rate` | idem |
| `entity_resolution` | `entity.false_merge.rate`, `entity.duplicate.rate`, `entity.semantic_accuracy.rate` | idem |
| `spatial_relations` | `relations.f1`, `relations.negative_violation.rate` | idem |

Custos, capturados em todo experimento: `runtime.wall_time`, `runtime.peak_memory` (por `device=cpu|gpu`), `runtime.storage_size` (por `artifact_role=intermediate|final`), `runtime.throughput` e `runtime.failure_rate` (falhas, inclusive OOM).

Os avaliadores de Entity Resolution e Spatial Relations dependem dos contratos das milestones 12–14 e ainda não existem: os experimentos e as definições de métrica estão prontos, e a evidência marca esses estágios como **não avaliados** até que haja comparação (nunca como neutros).

## Estratos

`stratification_factors` nomeia os estratos do protocolo:

- **A:** `range_band` (profundidade do suporte), `visibility` (oclusão), `support_density` (densidade de suporte LiDAR), `valid_region` (região válida da imagem/fisheye) e `projected_size` (tamanho projetado do suporte);
- **B:** um fator por componente canônico de `ObservationQuality` (`quality.support_depth`, `quality.occluded_fraction`, … — um por `QualityComponent`), mais `ambiguity_level` e `conflict_level`.

Só se usam **anotações e estratos explicitamente versionados**: os fatores precisam estar em `stratum_definitions` do reference set. `TechniqueProtocol.undeclared_factors(reference_set)` lista os que faltam, e a evidência os marca como não declarados; nenhum estrato é inventado. Conceitos open-vocabulary não anotados permanecem *N/A*, não errados. Inferência repetida sobre um mesmo frame físico não é uma amostra nova (`repetitions_per_sample` separado de `physical_sample_count`).

## Evidência

`build_technique_evidence(protocol, comparisons, registry=…, reference_set=…, policy=…)` transforma as comparações de #174 em um `TechniqueEvidence`:

- **Por métrica e por estrato.** Cada métrica tem o efeito na população inteira (`whole_population`) e em cada estrato: `improved`, `regressed`, `unchanged`, `changed` (métrica sem direção melhor) ou `not_comparable`. Um ganho global não esconde uma regressão em um estrato; `quality_regressions` as lista pelo local (`estágio:métrica/versão@fator=valor`).
- **Política explícita.** `EffectPolicy` traz a tolerância **por unidade** (sem valores padrão; uma unidade sem tolerância é erro) e o mínimo de amostras por estrato. A direção vem do registro (`higher_is_better`/`lower_is_better`).
- **Não comparável nunca vira zero.** Não aplicável, não suportado, estrato reportado por só um arm, populações diferentes ou amostras abaixo do mínimo dão `not_comparable` com o motivo, sem delta.
- **Estratos declarados.** Estrato de qualidade fora de `stratum_definitions` é recusado; custos só podem ser divididos por `device` e `artifact_role`.
- **Custos separados.** Qualidade em `quality`, custos em `costs`, com `cost_regressions` à parte.
- **Prova de upstream compartilhado.** `shared_artifacts` de cada estágio traz os artifacts idênticos nos dois arms (o DINO X, ou a Association A).
- **Faltas explícitas.** Um estágio do protocolo sem comparação vai para `unevaluated_stages`; um arm indisponível ou com falha (por exemplo, o backend de enhancement) deixa o estágio `incomplete` com o arm e o motivo, sem efeitos; `factors` diz, por fator, se foi declarado pelo reference set e se foi de fato reportado. `complete` só é verdadeiro quando tudo foi avaliado com os dois arms.
- **Amarração.** Cada comparação precisa ser do experimento exato do protocolo, do mesmo reference set e do mesmo registro; comparações de estágios fora do protocolo são recusadas.

O relatório é reproduzível (mesmas entradas, mesmo digest) e não tem score, ranking nem vencedor.

## Decisão: manter, adiar ou mudar o default

`record_technique_decision()` registra uma decisão **humana** (`decided_by` e `rationale` obrigatórios) presa ao digest da evidência exata, e recusa a que a evidência não sustenta:

| Decisão | Exige |
|---|---|
| `defer` | nada: sempre possível, inclusive com evidência incompleta |
| `keep_optional` | ao menos um estágio avaliado |
| `change_default` | evidência **completa**, ao menos uma melhora de qualidade, e **toda regressão de qualidade reconhecida** pela chave (`acknowledged_regressions`); chaves inexistentes são recusadas |

Uma decisão `change_default` é só uma **proposta**: `requires_e2e_revalidation` é verdadeiro e `modifies_default_profile` é sempre falso. Nada aqui altera configuração; a mudança do perfil e a re-validação seguem o fluxo de aceite E2E.

## Limitações e lacunas

- **Sem runtime nem executores reais.** Como em #174, as topologias resolvidas e o executor de arms são fornecidos; a conexão com o runtime e com o pipeline real virá com a milestone 17. Nenhuma execução real (GPU, backend de enhancement, cenários reais) está registrada: o harness está pronto, mas a evidência científica ainda não existe.
- **Escopo de #197.** Este módulo entrega o protocolo e o harness. Os resultados de ganhos e regressões sobre o reference set real e a evidência que sustente manter, adiar ou mudar o default dependem da execução real e continuam em aberto: a issue #197 não está concluída.
- Os estratos de qualidade dependem de o reference set versionado declará-los e de os avaliadores por estágio os produzirem; hoje há definições e validação, não dados reais.
- Entity Resolution e Spatial Relations aguardam as milestones 12–14.

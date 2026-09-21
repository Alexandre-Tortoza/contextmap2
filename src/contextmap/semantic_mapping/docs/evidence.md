# Vínculo de evidência e proveniência

Este documento descreve `src/contextmap/semantic_mapping/evidence.py` (os contratos e o construtor dos vínculos) e `evidence_integrity.py` (validação de integridade e travessia da proveniência).

Uma entidade é um registro compacto que **referencia** a evidência que a criou. Ela nunca duplica imagens, máscaras, embeddings nem payloads de fusão, mas continua totalmente auditável.

## Cadeias de proveniência

```text
Entity
→ FusedEvidence
→ SpatialObservation
→ PerceptionResult / Region2D / SemanticClaim
→ SourceObservation

Entity
→ EntityGeometry
→ GeometryReference
→ GeometricMapArtifact
→ observação LiDAR de origem / linhagem de transformações
```

Ambas são percorridas só por identidades, sem carregar percepção, fusão nem modelo, e sem reexecutá-los.

## `EntityEvidenceLinks`

| Campo | Significado |
| --- | --- |
| `fused_evidence` | `FusedEvidenceRef` da evidência fundida de origem; **nunca vazio**. |
| `spatial_observation_ids` | As `SpatialObservation` que contribuíram, ordenadas e únicas. |
| `physical_observation_ids` | Os frames físicos (`SourceObservation`) que contribuíram: o fim da cadeia, quantas execuções de inferência os tenham interpretado. |
| `visual_feature_refs` | `EntityFeatureRef`: features de região ou globais das vistas contribuintes; vazio se o canal está ausente. |
| `point_representation_refs` | `PointRepresentationRef` das representações 3D do suporte; vazio se o canal está ausente. |

Todas as listas são ordenadas e sem duplicatas (o contrato recusa uma referência duplicada). Um canal opcional ausente **não invalida** uma entidade: as listas ficam vazias.

### `FusedEvidenceRef`

Identifica a evidência dentro de um `SemanticFusionRunArtifact` e carrega a **identidade e a versão** do artifact: `fusion_run_id`, `fusion_schema_version`, `fused_evidence_id`, `fusion_support_id` e `fusion_artifact_digest`, o digest de `fusion_artifact_digest(manifest)` (identidade, versão do schema e tamanho e hash de cada arquivo contratual). O digest independe do layout interno do run e muda se qualquer coisa de que a entidade foi materializada mudar.

O digest **não cobre a linhagem** do manifest. Por isso a referência carrega também `sequence_artifact_id`, a sequência canônica sobre a qual o run foi construído quando a entidade foi materializada, e `validate_entity_evidence` a compara explicitamente com `manifest.lineage.sequence_artifact_id` (o mapa geométrico é comparado com o da geometria da entidade). Trocar só a sequência do run, mantendo mapa, identidade e inventário, é portanto um `incompatible_lineage` explícito e não passa despercebido.

### `EntityFeatureRef`

`perception_run_id`, `perception_result_id`, `feature_id`, `embedding_space_id`, `scope` e `region_id` (presente exatamente para features de região). O vetor continua no feature store de Visual Perception; só a identidade viaja, junto do espaço de embedding, para que espaços diferentes nunca sejam misturados por acidente.

A identidade da feature é a tripla **`(perception_run_id, perception_result_id, feature_id)`**, nessa ordem: um `PerceptionResultId` é local ao `PerceptionRun` e um `FeatureId` é local ao resultado, então dois runs podem reutilizar o mesmo par e só o run os distingue. A deduplicação em `feature_refs_of`, a ordem canônica de `visual_feature_refs` e a recusa de duplicatas usam essa tripla; uma referência nunca sobrescreve a de outro run.

### Construção

`evidence_links_from_fused_evidence(evidence, manifest=...)` cria os vínculos a partir da evidência fundida e do manifest do run que a possui. `feature_refs_of(evidence)` lista as features das vistas contribuintes.

Uma representação 3D ancorada **fora** do suporte geométrico da entidade é recusada pelo contrato de `Entity`.

## Integridade de referências

`validate_entity_evidence(entity, fusion_runs={run_id: fonte}, geometry=None)` checa cada referência contra os artifacts que ela nomeia e **relata** (nunca levanta) o que estiver errado:

| Tipo | Quando |
| --- | --- |
| `missing_artifact` | Um run referenciado não foi oferecido. |
| `artifact_mismatch` | O run oferecido tem outra identidade ou versão de schema. |
| `stale_reference` | O conteúdo do run mudou desde que a entidade foi materializada (digest). |
| `corrupt_artifact` | O run falha na própria verificação de integridade. |
| `missing_reference` | A evidência fundida não está no run, ou o suporte guarda outra. |
| `incompatible_lineage` | O run foi construído sobre outro mapa geométrico que o da entidade, ou sobre outra sequência que a de que a entidade foi materializada (`sequence_artifact_id` da referência × linhagem do manifest). |
| `observation_mismatch` | As observações listadas diferem das que a evidência fundida traz. |
| `feature_mismatch`, `point_representation_mismatch` | As features ou representações 3D listadas diferem das que a evidência fundida traz. |
| `geometry_outside_support` | A evidência fundida enxerga geometria fora do suporte da entidade. |
| `missing_geometry` | Uma referência de geometria da entidade não resolve no mapa (só com `geometry`). |

Falhas que impedem ler a evidência (artifact ausente, divergente, desatualizado ou evidência inexistente) bloqueiam a comparação das listas: comparar sem os fatos só geraria falsos positivos.

A fonte é a porta estrutural `FusedEvidenceSource` (`manifest`, `fused_evidence(support_id)`, `verify_integrity()`), satisfeita por `SemanticFusionRunReader`. Só os runs que a entidade referencia são lidos.

## Travessia

- `trace_entity_evidence(entity, fusion_runs=...)` percorre a entidade até as vistas: cada `ContributionTrace` traz a observação espacial, o run e o resultado de percepção, a região, as claims e o frame físico. `EntityEvidenceTrace.physical_observation_ids` lista os frames distintos. Levanta `EvidenceTraceError` se o run ou a evidência não podem ser lidos.
- `trace_geometry_sources(geometry, source=...)` percorre a geometria até as observações LiDAR de origem: uma `GeometrySourceTrace` por observação, com a contagem de pontos e o intervalo de aquisição.

## Sobrevive à reabertura

O vínculo é só identidade e digest, então persiste no registro JSON da entidade e continua validando contra o run reaberto do disco (`tests/semantic_mapping/test_entity_evidence.py`, que usa um run de fusão real, escrito e reaberto).

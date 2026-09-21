# Entity Resolution

## Responsabilidade

Decidir, a partir de evidência tipada e preservada por canal, se entidades semânticas de um mapa descrevem o **mesmo objeto físico**, **objetos distintos** ou se **não é possível saber** (`UNRESOLVED`), e materializar as entidades resolvidas que decisões explícitas justificam, dentro de um artifact imutável.

```mermaid
flowchart LR
    SMA["SemanticMappingRunArtifact<br/>(Entity, EntitySet)"] --> ER["Entity Resolution"]
    PTR["PointRepresentation<br/>(opcional)"] -. evidência 3D .-> ER
    VF["VisualFeature<br/>(FeatureStore)"] -. evidência de aparência .-> ER
    ER --> RES["ResolvedEntity<br/>+ ResolutionDecision"]
    RES -. próximo boundary .-> SR["Spatial Relations<br/>(planejado)"]
```

## O que este módulo explicitamente não possui

- criação de entidades a partir de evidência fundida: Semantic Mapping;
- relações espaciais entre entidades: Spatial Relations;
- geometria e suas coordenadas: Geometric Mapping (aqui só há `GeometryReference`);
- features visuais e representações 3D: Visual Perception e Point Representation (aqui só há referências e a comparação delas);
- rastreamento de objetos dinâmicos e re-identificação após movimento;
- identidade permanente entre mapas construídos de forma independente.

## Estado implementado

Existem os **contratos** de identidade e de comparação:

- `EntityResolutionRunId`, `ResolvedEntityId` e `ResolvedEntityReference`, com o codec JSON que revalida a referência;
- `EntityMatchEvidence`: a evidência de **uma comparação**, com a evidência tipada de cada canal (geometria, semântica, aparência, temporal e, opcional, representação 3D) mantida separada e sem nenhum score que a resuma, mais os resultados dos gates duros de validade (`evaluate_comparison_gates`);
- `ResolutionDecision`: o veredito de uma política versionada, `MATCH`, `DISTINCT` ou `UNRESOLVED`, com a evidência de origem, as regras que dispararam, os canais usados e ignorados e o motivo quando não resolve;
- as **entidades resolvidas** (`materialize_resolved_entities`, `ResolvedEntity`, `ResolvedEntitySet`): a materialização determinística dos grupos de decisões `MATCH`, com linhagem de fusão, contradições de transitividade expostas (o componente não é fundido) e agregação exata que nunca inventa confiança ([`resolved-entities.md`](resolved-entities.md));
- a **política de resolução baseline** (`decide`, `ConservativeResolutionPolicy`, `MatchEvidenceBuilder`, `resolve_candidate_pairs`): estágios explícitos (gates, elegibilidade, evidência, regras, decisão) sobre o status dos canais, sem soma ponderada, que prefere `UNRESOLVED` e tem um caminho só de geometria válido ([`resolution-policy.md`](resolution-policy.md));
- a **comparação de representações 3D**, opcional (`RepresentationComparator`, `RunReaderRepresentationSource`): estrutura 3D comparada só dentro de um espaço de representação compatível, sem nunca interpretar componentes indefinidos, com ausência como `unavailable` ([`representation-comparison.md`](representation-comparison.md));
- a **comparação de aparência** (`AppearanceComparator`, `FeatureStoreVectorSource`): features visuais comparadas só dentro de um espaço de embedding compatível, um voto por observação física, com ausência como `unavailable` ([`appearance-comparison.md`](appearance-comparison.md));
- a **comparação semântica e temporal** (`compare_semantics`, `compare_temporal`): compatibilidade de labels, hipóteses e atributos, e do histórico de observações, sem ontologia, sem senso comum e sem tratar ausência como evidência negativa ([`semantic-temporal-comparison.md`](semantic-temporal-comparison.md));
- a **comparação geométrica** (`compare_geometry`, `GeometryComparisonPolicy`): o canal de geometria, com sinais próprios e regras versionadas, sem semântica e sem decisão ([`geometry-comparison.md`](geometry-comparison.md));
- a **recuperação de candidatos** (`retrieve_candidate_sets`, `EntitySpatialIndex`): para cada entidade, os alvos plausíveis de comparação, de forma permissiva e determinística, sem all-pairs e sem decidir nada ([`candidate-retrieval.md`](candidate-retrieval.md)).

Um canal é sempre **medido ou indisponível**: a falta de evidência nunca vira zero nem voto por `DISTINCT`. O codec é estrito e reflexivo (`_codec.py`). Os demais contratos (política, entidade resolvida, artifact e avaliação) chegam nas issues seguintes da milestone.

## Escopo de identidade

Um `ResolvedEntityId` é único **dentro de um** artifact de resolução. A mesma string em dois artifacts nomeia dois registros diferentes: uma nova execução cria um novo artifact e novos ids, então nada aqui implica permanência entre mapas reconstruídos de forma independente. Por isso a referência estável é `ResolvedEntityReference(resolution_run_id, resolved_entity_id)`, nunca um id nu, no mesmo desenho de `EntityReference(semantic_map_id, entity_id)`.

Uma entidade resolvida **nunca substitui** os membros: as entidades de origem mantêm a própria `EntityReference` e continuam endereçáveis individualmente.

## Contratos públicos

- `EntityResolutionRunId`, `ResolvedEntityId`, `ResolvedEntityReference` — identidade escopada ao artifact de resolução e o handle estável `(resolution_run_id, resolved_entity_id)`.
- `EntityMatchEvidence`, `ComparisonId`, `comparison_id_for`, `MatchEvidenceProvenance`, `GateResult`, `evaluate_comparison_gates`, `COMPARISON_GATES_POLICY_ID`, `reference_order` — a comparação de um par e seus gates.
- `MatchChannel`, `EvidenceStatus`, `UnavailableReason`, `Unavailability`, `Finding`, `ChannelEvidence` — o vocabulário comum dos canais.
- `GeometryEvidence`, `GeometryMeasurement`, `SupportDistance`; `SemanticEvidence`, `SemanticMeasurement`, `LabelComparison`, `LabelRelation`, `AttributeComparison`; `AppearanceEvidence`, `AppearanceMeasurement`, `FeatureContribution`; `TemporalEvidence`, `TemporalMeasurement`; `PointRepresentationEvidence`, `RepresentationMeasurement`, `RepresentationRef` — a evidência tipada de cada canal.
- `ResolutionDecision`, `ResolutionDecisionId`, `ResolutionOutcome`, `UnresolvedReason`, `PolicyStage`, `TriggeredRule`, `DecisionProvenance`, `decision_id_for` — a decisão.
- `CandidateRetrievalPolicy`, `CANDIDATE_RETRIEVAL_POLICY_ID`, `retrieve_candidate_sets`, `EntitySpatialIndex`, `EntityCandidateSet`, `EntityCandidate`, `RetrievalDiagnostics`, `RetrievalReason`, `ExclusionReason`, `CandidacyAssessment`, `explain_candidacy`, `candidate_pairs` — a recuperação de candidatos.
- `SemanticCompatibilityPolicy`, `SEMANTIC_COMPATIBILITY_POLICY_ID`, `BASELINE_REFINEMENT_MODIFIERS`, `compare_semantics`, `compare_labels`, `normalize_label`; `TemporalCompatibilityPolicy`, `TEMPORAL_COMPATIBILITY_POLICY_ID`, `compare_temporal` — os canais semântico e temporal.
- `GeometryComparisonPolicy`, `SupportDistancePolicy`, `GEOMETRY_COMPARISON_POLICY_ID`, `compare_geometry` — o canal de geometria.
- `AppearanceComparisonPolicy`, `APPEARANCE_COMPARISON_POLICY_ID`, `APPEARANCE_AGGREGATION_ID`, `AppearanceComparator`, `FeatureVectorSource`, `LoadedFeature`, `FeatureStoreVectorSource` — o canal de aparência e a fronteira de carregamento de vetores.
- `RepresentationComparisonPolicy`, `REPRESENTATION_COMPARISON_POLICY_ID`, `REPRESENTATION_AGGREGATION_ID`, `RepresentationComparator`, `RepresentationVectorSource`, `LoadedRepresentation`, `RunReaderRepresentationSource` — o canal opcional de representação 3D e sua fronteira de carregamento.
- `materialize_resolved_entities`, `ResolvedEntityMaterialization`, `ResolvedEntitySet`, `ResolvedEntity`, `ResolvedMember`, `ResolvedGeometry`, `ResolvedSemanticState`, `MemberAmbiguity`, `ResolvedEntityProvenance`, `TransitivityContradiction`, `MaterializationError`, `materialization_policy`, `resolved_entity_id_for`, `contradiction_id_for`, `derive_resolved_ambiguity`, `ForeignResolvedEntityReferenceError`, `UnknownResolvedEntityError` e as constantes de regra — a entidade resolvida e sua materialização.
- `ConservativeResolutionPolicy`, `CONSERVATIVE_RESOLUTION_POLICY_ID`, `decide`, `ComparisonChannels`, `MatchEvidenceBuilder`, `PairResolution`, `resolve_candidate_pairs` — a política baseline, a coleta de evidência e o serviço de execução.
- `PolicyRef` — política versionada e fingerprint da configuração, comum a canais, recuperação e políticas.
- `encode_*` e `decode_*` de referência resolvida, evidência de comparação, decisão e conjunto de candidatos — o codec JSON, que revalida todas as invariantes.

Ver [`contracts.md`](contracts.md) para a referência de campos e as invariantes.

## Módulos consumidos

- `contextmap.semantic_mapping`: `Entity`, `EntityGeometry`, `EntityReference`, `EntityFeatureRef`, `EntitySemanticState`, `AmbiguityState`, `AttributeOrigin`, `CLASS_ATTRIBUTE_DERIVATION_ID` e `resolve_geometry`, o que é comparado e referenciado.
- `contextmap.geometric_mapping`: `GeometryReference`, `MapId`, `Bounds3D` e `GeometrySource`, o suporte 3D referenciado, as caixas e a leitura opcional dos pontos.
- `contextmap.visual_perception`: `VisualFeature`, `FeatureStoreReader`, `FeatureScope`, `EmbeddingSpaceMismatchError`, `ensure_compatible_features` e `perception_result_id_for`, o carregamento e a compatibilidade das features.
- `contextmap.ingestion`: `SourceObservationId`, só como identidade dos frames físicos que observaram uma entidade e que indexam o feature store; nenhuma lógica de ingestion é usada.
- `contextmap.point_representation`: `PointRepresentation`, `PointRepresentationId`, `PointRepresentationRunId`, `PointRepresentationRunReader`, `RepresentationSpaceMismatchError` e `ensure_compatible_representations`, as representações referenciadas e a compatibilidade.

As dependências estão declaradas em `tests/architecture/test_boundaries.py` e são sempre feitas pela API pública.

## Onde estão os documentos detalhados

- [`contracts.md`](contracts.md) — contratos, escopo de identidade e invariantes.
- [`resolved-entities.md`](resolved-entities.md) — agrupamento, identidade, contradições, agregação exata e linhagem.
- [`resolution-policy.md`](resolution-policy.md) — estágios, regras de decisão, configuração e execução.
- [`representation-comparison.md`](representation-comparison.md) — espaço de representação, componentes indefinidos e agregação.
- [`appearance-comparison.md`](appearance-comparison.md) — espaço de embedding, agregação por observação física e fonte de vetores.
- [`semantic-temporal-comparison.md`](semantic-temporal-comparison.md) — regras de label, atributos e histórico de observações.
- [`geometry-comparison.md`](geometry-comparison.md) — sinais geométricos, regras, casos explícitos e custo.
- [`candidate-retrieval.md`](candidate-retrieval.md) — política de recuperação, diagnóstico de exclusão, índice espacial e linha de base de desempenho.

# Validação de Semantic Mapping

Este documento descreve `src/contextmap/evaluation/semantic_mapping.py`: a validação determinística de um `SemanticMappingRunArtifact` persistido contra a evidência de que foi materializado.

Semantic Mapping é validado como um estágio de **materialização**, e as falhas precisam continuar visíveis. O harness lê o run de mapeamento pelo leitor público, junto do run de fusão e do mapa geométrico de origem, e só lê: nada é alterado nem reparado, e `debug/` nunca é lido. Não avalia acurácia de mesmo objeto nem de entity resolution, não usa extração de relações para validar entidades e não faz merge ou split automático.

## Uso

```python
report = evaluate_semantic_mapping(
    SemanticMappingRunReader(mapping_run_dir),
    fusion=SemanticFusionRunReader(fusion_run_dir),
    geometry=geometry_source,
    policy=EntityMaterializationPolicy(geometry=GeometrySummaryPolicy(...)),
)
report.passed  # todas as checagens passaram
report.failed_checks  # o que falhou, com cada falha descrita
encode_semantic_mapping_report(report)  # JSON com todas as identidades
```

`SemanticMappingEvaluationError` recusa a avaliação inteira quando o run de fusão ou o mapa geométrico oferecidos não são os que a linhagem do run de mapeamento nomeia (identidade, versão do schema e digest do inventário), porque as checagens comparariam coisas sem relação.

## Relatório

Cada `SemanticMappingValidationCheck` traz a camada, um `check_id` estável, quantos itens examinou e as falhas exatas (vazio significa que passou). **Não há score composto**: uma checagem passa ou lista por que não passa.

`SemanticMappingEvaluationLineage` registra tudo o que torna o relatório reproduzível e comparável:

- o `SemanticFusionRunArtifact` (identidade, versão do schema e digest do inventário);
- o `GeometricMapArtifact` (identidade do mapa);
- a política de materialização e sua versão, a política de identidade e o fingerprint da configuração;
- a versão do schema das entidades (do artifact);
- a versão do código;
- a versão do avaliador (`EVALUATOR_VERSION`, incrementada sempre que a definição de uma checagem muda).

## Camadas

### Contrato e invariantes (`contract`)

- `entity_ids_unique_and_counted`: ids únicos e locais ao artifact, contagem igual à do manifest, ordem do índice.
- `semantic_map_identity_explicit`: toda entidade declara o semantic map do run.
- `entity_references_resolve`: cada `EntityReference` resolve para a mesma entidade, e uma referência de outro mapa é recusada.
- `geometry_valid_and_authoritative`: geometria não vazia, no frame do mapa, que resolve no mapa geométrico e cujos resumos são os que a política produz (`verify_geometry_summary`).
- `semantic_state_invariants`: o estado de ambiguidade é o que os registros implicam, nenhuma primária esconde alternativas, atributos citam evidência.
- `provenance_complete`: política, fingerprint, versão de código e referências ao run de fusão da linhagem.

### Preservação semântica (`semantic_preservation`)

Comparada com a evidência fundida real: `hypotheses_preserved`, `uncertainty_preserved` (conflitos e ambiguidade), `abstention_preserved`, `unscored_signals_preserved`, `no_forced_single_label` e `semantic_state_is_the_mapping_of_the_evidence`. Nenhuma evidência some só porque existe uma hipótese primária conveniente.

### Linhagem de evidência (`evidence_lineage`)

`references_resolve` (integridade de referências, [`evidence.md`](../../semantic_mapping/docs/evidence.md)), `entity_to_source_observation_traversal` (entidade → evidência fundida → observação espacial → frame físico) e `entity_geometry_to_source_traversal` (entidade → geometria → observações LiDAR de origem).

### Comportamento temporal (`temporal`)

`first_and_last_seen_reproducible` (a partir da evidência exata e do resumo temporal da fusão) e `physical_observations_not_inflated` (as contagens de frames físicos e de resultados de inferência batem com a evidência fundida, inclusive quando várias execuções interpretam um único frame).

### Fronteira de materialização (`materialization_boundary`)

`one_support_one_entity`, `identity_is_a_function_of_the_support`, `known_materialization_policy` e `rematerialization_reproduces_the_entities`: re-materializar a mesma seleção, com a mesma configuração, reproduz **exatamente** as entidades e as rejeições persistidas. Dois suportes separados continuam duas entidades.

### Round-trip do artifact (`artifact_round_trip`)

`run_integrity` (inventário contra o disco), `entities_reopen_unchanged` (codificar e decodificar e reabrir por referência não altera o registro) e `derived_indexes_agree_with_the_entities`.

## Robustez

Uma evidência fundida que não pode ser lida, um payload de fusão corrompido ou um suporte inexistente são **relatados** como falhas por entidade, nunca levantados: o relatório continua completo.

## Validação

`tests/semantic_mapping/test_semantic_mapping_evaluation.py` valida um run limpo (todas as checagens de todas as camadas passam, determinismo e codificação sem score composto) e, sobretudo, que cada camada **detecta defeitos reais**: estado semântico que perdeu alternativas e forçou um label, evidência sem score perdida, estado temporal incompleto, vínculos com um observador estranho, um suporte que virou duas entidades ou identidade renomeada, resumo geométrico adulterado, política diferente da usada, evidência fundida ilegível, corrupção do run persistido e upstream errado.

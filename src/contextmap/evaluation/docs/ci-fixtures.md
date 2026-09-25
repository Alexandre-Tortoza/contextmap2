# Subconjunto de fixtures para CI

A avaliação com dados reais exige arquivos grandes, modelos em GPU ou APIs externas. A CI precisa de um subconjunto **pequeno, estável e sem GPU, rede ou download de modelo** que exercite os contratos e os casos-limite conhecidos das capabilities implementadas. Ele é gerado por `contextmap.evaluation.ci_fixtures` e commitado em `tests/fixtures/ci_subset/<versão>/` para revisão.

O subconjunto protege contra regressões. **Não é evidência de qualidade no mundo real**, e a avaliação com dados reais continua separada: o sucesso deste subconjunto nunca substitui uma avaliação sobre o reference set real.

## Conteúdo

```text
tests/fixtures/ci_subset/1.0.1/
├── manifest.json          # ReferenceSetManifest (ver reference-set.md)
├── catalogue.json         # casos, valores esperados, tolerâncias, cobertura
└── annotations/           # uma família de anotação por arquivo (ver annotations.md)
```

- **Sequência sintética** (`build_synthetic_sequence()`): três frames de RGB 16×12, LiDAR e pose externa, com calibração pinhole e extrínsecas. O robô avança 1 m/s em +x com orientação identidade. Tudo sai de fórmulas: sem números aleatórios, sem relógio, sem rede, sem modelo. A sequência **não é commitada**: os testes a materializam por código, escrevem-na como um sequence artifact de ingestion e a leem de volta.
- **Marcos 3D com projeção conhecida analiticamente**: dois visíveis, um que começa atrás da câmera e um que cai fora da imagem.
- **Reference set** (`build_reference_set()`): fonte `synthetic_fixture`, licença e redistribuição declaradas, anotações `trusted_ground_truth` de proveniência `synthetic_generation`, um único split de teste (o subconjunto nunca é usado para tuning). Passa a validação de integridade sem nenhum achado.
- **Catálogo** (`FixtureCatalogue`): cada `FixtureCase` tem id estável, capability, casos-limite nomeados, entradas, saídas esperadas, tolerâncias, gerador, licença, redistribuição e hash de conteúdo. O catálogo tem digest próprio e cita a identidade do reference set.

## Casos

| Caso | Capability | Casos-limite explícitos |
|---|---|---|
| `ingestion-canonical-sequence` | ingestion | domínio de clock único, layout float32 XYZ, payload RGB8 |
| `projection-known-landmarks` | sensor_association | atrás da câmera, fora da imagem, projeção que depende da pose |
| `trajectory-external-pose` | state_estimation | orientação identidade, velocidade constante |
| `regions-overlapping-masks` | visual_perception | regiões sobrepostas, área de exclusão e área válida, cobertura parcial |
| `semantic-claims-variants` | visual_perception | alternativa semântica, abstenção, claim sem score, claim com confiança medida, inferência repetida sobre uma observação física |
| `identity-duplicate-and-distinct` | entity_resolution | mesmo rótulo com objetos distintos, um objeto em várias observações |
| `relations-symmetry-and-negatives` | spatial_relations | predicado simétrico, negativo explícito, relação ambígua |
| `visibility-and-context-strata` | evaluation | visibilidade desconhecida, fração ocluída, contexto por amostra |
| `reference-set-integrity` | evaluation | proveniência sintética, split único de teste |

As respostas de modelo do caso `semantic-claims-variants` são **enlatadas**: o objeto do teste é o parser público, não um modelo.

## Cobertura explícita

O catálogo traz a matriz de cobertura, verificada em CI, com o que **não** está coberto e por quê:

| Requisito | Status | Observação |
|---|---|---|
| ingestão canônica RGB/LiDAR/pose/calibração | coberto | |
| transformações/projeções 3D conhecidas | coberto | modelo de câmera e composição de extrínsecas; o `FrameProjector` completo não roda |
| máscaras/regiões 2D e regiões sobrepostas | coberto | |
| alternativas/abstenção/claims sem score | coberto | respostas enlatadas pelo parser público |
| inferência repetida sobre uma observação física | coberto | uma observação, duas inferências |
| **fusão multi-vista** | **não disponível** | exige saídas sintéticas de Sensor Association (`SpatialObservation` com referências de geometria); Semantic Fusion é coberto pelos testes da própria capability |
| duplicata/distinção de entidades | nível de anotação | os contratos `Entity`/Entity Resolution ainda não existem |
| relações espaciais | nível de anotação | os contratos `Relation` ainda não existem |
| **round-trip do `ContextMapArtifact` final** | **não disponível** | schema e serialização do artifact ainda não existem |

## Regressão entre módulos

`tests/evaluation/test_ci_fixtures.py` roda, sem GPU nem rede:

- ingestion → sequence artifact → leitura, com ids, timestamps e bytes de payload iguais;
- calibração do subconjunto → `camera_projection_for` e composição de extrínsecas pelas primitivas de `contextmap.shared`, contra a projeção analítica do catálogo;
- poses externas → `ExternalPoseEstimator` → trajetória esperada;
- máscaras anotadas → interseção/união/IoU do catálogo;
- respostas enlatadas → `parse_semantic_response` → claims esperados, e avaliação contra a anotação semântica;
- o reference set do subconjunto → validação de integridade sem achados.

**Escopo (#172).** Esta regressão cobre só os elos acima; a cadeia entre módulos "da ingestão até o artifact final" é uma regressão separada, `tests/end_to_end/chain.py`/`acceptance.py` (`synthetic_chain()`): ela roda o código real de todo estágio (ingestão, percepção enlatada, state estimation, geometric mapping, sensor association, semantic fusion, semantic mapping, entity resolution, spatial relations e a montagem do `ContextMapArtifact`) sobre o subconjunto sintético e valida o round-trip completo, sem GPU nem rede. Nenhuma decisão geométrica/de contato é fabricada: sem avaliador selecionado, todo candidato de relação fica honestamente `UNRESOLVED`.

## Versionamento e revisão

- A versão é o diretório (`ci_subset/1.0.1/`) e está em `manifest.json` e `catalogue.json`. **Qualquer** mudança em um arquivo gerado exige uma nova versão.
- O teste `test_the_committed_subset_is_exactly_what_the_generator_produces` regenera o subconjunto e exige que cada arquivo commitado seja idêntico byte a byte: alterar o gerador sem regenerar (ou o contrário) falha a CI, e a mudança aparece inteira no diff.
- Para atualizar: aumente `CI_FIXTURE_VERSION`, gere a nova versão e commite-a junto com o código:

```bash
python -c "from pathlib import Path; from contextmap.evaluation import generate_ci_fixture_subset; generate_ci_fixture_subset(Path('tests/fixtures/ci_subset/<nova-versão>'))"
```

A escrita é imutável: o gerador recusa sobrescrever um diretório existente.

### Histórico de versões

- **1.0.1** — a relação `rel-supports-ambiguous` passou a ser ancorada em `frame-0001`, onde as duas identidades aparecem. A versão 1.0.0 a ancorava em `frame-0002`, onde `pallet-2` não é visto: o defeito foi encontrado pelo QA das anotações ([`annotation-qa.md`](annotation-qa.md)), que agora roda sobre o próprio subconjunto.
- **1.0.0** — versão inicial (substituída; nunca saiu da branch da milestone).

## Licença e redistribuição

Os dados são gerados por código do repositório, sob a licença do repositório (`LICENSE`), e todos os casos são redistribuíveis. Cada fonte e cada caso registram licença e `redistributable`.

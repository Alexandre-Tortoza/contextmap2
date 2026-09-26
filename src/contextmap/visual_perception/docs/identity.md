# Identidade e proveniência

Este documento descreve `src/contextmap/visual_perception/identity.py` e a cadeia completa de rastreabilidade de evidência visual.

## Cadeia de rastreabilidade

Dado qualquer `SemanticClaim`, `VisualFeature` ou `Region2D`, é possível rastrear até:

```mermaid
flowchart LR
    E["Region2D / VisualFeature / SemanticClaim"] --> R["PerceptionResult<br/>result_id"]
    E --> B["BackendProvenance<br/>backend/model/config"]
    R --> RUN["PerceptionRun<br/>run_id"]
    R --> OBS["SourceObservation<br/>source_observation_id"]
    RUN --> SEQ["SequenceArtifact<br/>sequence_artifact_id"]
    RUN --> SEL["Selection<br/>selection_id"]

    subgraph ING[Ingestion ownership]
        OBS
        SEQ
        SEL
    end
```

Nenhum desses IDs implica identidade de entidade 3D persistente — são todos identidades locais de evidência de inferência (ver `contracts.md`).

### Escopos de identidade

```mermaid
flowchart TB
    PHYS["SourceObservationId<br/>identidade da observação física"]
    RUNID["PerceptionRunId<br/>identidade da execução"]
    RID["PerceptionResultId<br/>run + observação"]
    LOCAL["RegionId / FeatureId / ClaimId<br/>locais ao resultado"]
    PERSIST["Entidade 3D persistente<br/>não definida aqui"]

    PHYS --> RID
    RUNID --> RID
    RID --> LOCAL
    LOCAL -. associação downstream explícita .-> PERSIST
```

A seta tracejada não representa uma conversão automática. Ela marca apenas a fronteira onde uma capability posterior pode associar evidência local a uma entidade persistente.

## Por que identidades determinísticas

`perception_result_id_for()`, `region_id_for()`, `feature_id_for()`, `claim_id_for()` são funções puras: a mesma entrada sempre produz a mesma identidade, em vez de um valor aleatório (UUID). Isso significa:

- a identidade de um `PerceptionResult` é sempre `f"{run_id}--{source_observation_id}"` — não precisa de um registro externo para saber qual resultado pertence a qual run+frame;
- reconstruir os mesmos objetos a partir dos mesmos dados brutos produz os mesmos IDs — útil para testes e para comparar re-execuções determinísticas;
- a identidade sobrevive a um round-trip de serialização (é só uma string, não um estado externo).

`feature_id_for()` exige também um `producer_id` estável. O helper incorpora o
SHA-256 desse identificador no `FeatureId`, portanto dois stages podem usar o
mesmo índice dentro do mesmo resultado sem colisão:

```text
feature_id_for(result_id=rid, producer_id="dense_feature_extraction", index=0)
feature_id_for(result_id=rid, producer_id="global_feature_extraction", index=0)
```

`grounding_region_id_for()` identifica uma região produzida por grounding. O
namespace é o `request_id` do grounding — um digest das entradas da inferência
(observação, conteúdo da imagem, query completa, política, geometria e fingerprint
de configuração) — mais a posição da saída na resposta:

```text
grounding_region_id_for(result_id=rid, request_id="grounding-<sha256>", index=0)
    → "<rid>--grounding-<sha256>-region-0000"
```

Uma região de grounding nunca colide com uma região de discovery nem com a saída de
outro request, e mudar só a query muda só as identidades que dependem dela. Ver
[`region-grounding.md`](region-grounding.md).

## Runs repetidos permanecem distintos

Duas execuções (`run_id` diferente) sobre a **mesma** `SourceObservation` produzem `PerceptionResultId`s diferentes (o `run_id` faz parte da chave), mas ambas preservam o mesmo `source_observation_id` — exatamente a distinção central da issue #48/#50: reprocessamento nunca é uma nova observação física.

```text
perception_result_id_for(run_id="run-0001", source_observation_id="frame-0124")
    → "run-0001--frame-0124"
perception_result_id_for(run_id="run-0002", source_observation_id="frame-0124")
    → "run-0002--frame-0124"
```

## Seleções sobrepostas

Quando duas seleções de sequência se sobrepõem (ex.: `run-0001` processa frames 0-1000, `run-0005` processa frames 430-480), cada run continua produzindo seus próprios `PerceptionResult`s com identidade própria por frame — não há fusão implícita. Ver `tests/visual_perception/test_identity.py` para o teste com seleções sobrepostas.

## Escopo local, não global

Um `RegionId`/`FeatureId`/`ClaimId` é único **dentro de um `PerceptionResult`**.
Regiões e claims incluem o `result_id` e o índice; features incluem também o
`producer_id`, pois múltiplos stages de extração podem contribuir para o mesmo
resultado. Esses IDs nunca são globais nem comparáveis entre resultados sem
associação explícita posterior por uma capability downstream.

# Política baseline de acumulação de evidência

`accumulate_baseline_evidence` transforma as observações espaciais de um `FusionSupport` em um `FusedEvidence`. É a política **canônica de controle**: transparente, determinística e deliberadamente conservadora, o ponto de comparação das políticas posteriores (por exemplo, a ciente de qualidade).

Identidade da política: `baseline-evidence-accumulation-v1`. A regra de chave de label faz parte dessa versão: mudá-la exige nova versão.

## O que faz

1. **Uma contribuição por observação espacial.** Cada `SpatialObservation` do suporte vira uma `EvidenceContribution` com referências às suas claims, aos scores de scorer, às features visuais e, quando fornecida, à qualidade da observação. A identidade da contribuição é `evidence_contribution_id_for(suporte, observação)`.
2. **Grupos por observação física.** Os grupos vêm do agrupamento (`PhysicalObservationGrouping`), que fornece o timestamp de aquisição de cada frame.
3. **Hipóteses por chave de label.** As claims são agrupadas pela *chave de label*: o texto sob normalização Unicode NFKC, em minúsculas (`casefold`) e com espaços colapsados. É uma regra tipográfica: `"Pallet"` e `"pallet "` são uma hipótese; `"pallet"` e `"wooden pallet"` são duas. Nenhum sinônimo, plural ou taxonomia é assumido. O label exibido é a grafia (com espaços colapsados) mais comum, com empate resolvido pela menor.
4. **Stance de cada claim para cada hipótese.**
   - `SUPPORTING`: a claim propõe a hipótese;
   - `AMBIGUOUS`: qualquer outra claim que vem da mesma contribuição de uma claim que sustenta a hipótese (alternativas de uma só interpretação) ou que tem papel `ALTERNATIVE`;
   - `CONFLICTING`: qualquer outra claim.
5. **Sinais tipados por claim.** A confiança da própria claim (`CLAIM_CONFIDENCE`, com `None` quando não pontuada, nunca zero) e um `SCORER_SUPPORT` por saída de scorer. Nada é somado, ponderado ou combinado.
6. **Contagem de evidência independente.** Hipóteses são numeradas pela chave de label (`hypothesis-0001`, …), então a ordem não sugere ranking. A contagem de suporte independente é o número de **observações físicas distintas** (`FusedEvidence.supporting_physical_observations`): inferência repetida sobre um frame e uma claim sobre muita geometria contam uma vez.

## Entrada

```python
fused = accumulate_baseline_evidence(
    support,
    observations=observations,  # SpatialObservation por identidade
    grouping=grouping,  # PhysicalObservationGrouping das mesmas observações
    perception_results=results,  # PerceptionResult por identidade (dono das claims)
    semantic_scores=scores,  # opcional: SemanticSupport por resultado
    observation_quality_refs=qualities,  # opcional: ObservationQualityRef por observação
    point_representation_refs=structure,  # opcional: PointRepresentationRef
)
```

Os canais são **opt-in**: scores, qualidade e estrutura só participam quando o chamador os passa. Um score de uma claim que não está anexada a uma observação do suporte é ignorado. Só as referências de estrutura ancoradas dentro do suporte são mantidas, cada uma uma vez, em `FusedEvidence.point_representation_refs`.

O resultado não depende da ordem de nenhuma entrada: observações, resultados, scores e referências são ordenados por identidade.

## Fronteira de qualidade

O baseline **ignora** a qualidade de observação como peso, mas **preserva** a referência em cada contribuição. Nada aqui pondera por distância, visibilidade ou reprojeção só porque esses campos existem; isso é a política ciente de qualidade, opcional e comparada contra este baseline.

## Erros

Falha cedo, com mensagem acionável, quando: uma observação do suporte não está nas entradas, pertence a outro mapa ou tem geometria fora do suporte; um `PerceptionResult` falta ou não contém uma claim referenciada; o agrupamento não cobre um frame ou não lista uma observação sob o seu frame; ou um scorer pontuou a mesma claim duas vezes.

## O que não faz

- não pondera por qualidade de observação;
- não trata `unknown`/abstenção de forma especial nem constrói `UncertaintyRecord` (preservação de ambiguidade, conflito e abstenção é a issue #118): hoje um label como `unknown` seria só mais uma hipótese;
- não escolhe vencedor, não calcula probabilidade nem combina confiança, similaridade e qualidade;
- não cria identidade de entidade nem aplica conhecimento prévio.

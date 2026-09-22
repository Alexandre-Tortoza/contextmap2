# Comparação de aparência

Um canal opcional do par de entidades: compara as **features visuais** que as duas entidades referenciam (`AppearanceComparator`, `AppearanceEvidence`). A aparência nunca é o único critério de identidade, nunca classifica e nunca é concatenada com geometria, semântica ou outro vetor. Pode ser ligada ou desligada de forma independente na política de resolução (um canal `None` no `EntityMatchEvidence` é "não avaliado").

## Só onde a comparação faz sentido

| Regra | Efeito |
| --- | --- |
| **Só dentro de um espaço de embedding.** Duas features só são comparáveis se o fingerprint do espaço for idêntico, nunca porque a dimensão coincide: um vetor DINOv3 e um CLIP do mesmo tamanho não são comparáveis. | A política declara o `embedding_space_id`. Entidade sem feature nesse espaço: canal `unavailable(incompatible_domain)`, sem carregar nenhum vetor. |
| **Só features de região.** Uma feature global ou densa descreve a imagem toda, não a região da entidade. | Entidade sem feature de região: `unavailable(missing_evidence)`. |
| **Ausência não é zero nem `DISTINCT`.** | Nunca vira score; a política de resolução lê `unavailable` como falta de evidência. |
| **Um voto por observação física.** | A inferência repetida sobre o mesmo frame gera features correlacionadas, agrupadas sob o frame (`FeatureContribution`), e nunca evidência independente. |
| **Contradição vira erro.** | Um store que informa outro espaço que a referência da entidade é evidência corrompida: `EmbeddingSpaceMismatchError`, nunca comparação numérica nem `unavailable` silencioso. |

`ensure_compatible_features` (Visual Perception) é chamado antes de comparar dois vetores, sobre as features **carregadas** pela fonte. As entidades só guardam referências (`EntityFeatureRef`), então a seleção por espaço, que precisa vir antes de qualquer I/O, usa o mesmo critério (fingerprint idêntico) sobre a referência.

## Agregação `physical-observation-prototype-v1`

1. cada feature é escalada para norma 1;
2. as features de **uma** observação física são médias e escaladas de novo: o protótipo da observação;
3. o protótipo da entidade é a média dos protótipos das suas observações, cada observação contando uma vez;
4. `similarity` é o cosseno entre os dois protótipos de entidade; `pair_similarity_min/max` são o menor e o maior cosseno entre uma observação de cada lado, para que pontos de vista que discordam fiquem visíveis.

Vetor zero ou com componente não finita é evidência corrompida e levanta `ValueError` que nomeia a feature; dimensões diferentes dentro do mesmo espaço também. Se os protótipos de observações de uma entidade se anulam (sem direção), o canal fica `unavailable(insufficient_evidence)`.

O cosseno **não é probabilidade** e não é comparável com nenhum outro canal.

## Política `entity-appearance-comparison-v1`

`AppearanceComparisonPolicy(embedding_space_id, min_supporting_similarity, max_conflicting_similarity=None)` **não tem valores padrão**: o que é "parecido" depende do espaço e se justifica com dados de validação (#140). O achado `appearance-similarity` (métrica `similarity`, com o valor e o limiar) é `supporting` se a similaridade for ≥ `min_supporting_similarity`, `conflicting` se `max_conflicting_similarity` existir e a similaridade for ≤ a ele, e `neutral` senão. Sem limiar de conflito, uma similaridade baixa nunca é evidência contra: um ponto de vista diferente pode parecer diferente.

## Fonte dos vetores

`FeatureVectorSource` isola o feature store (arquivos, hashes, NumPy) da comparação, com duas implementações reais: `FeatureStoreVectorSource`, sobre os `FeatureStoreReader` dos runs de percepção (verifica hash, shape e dtype de cada payload), e um fake em memória nos testes. O comparador guarda o protótipo de cada entidade já vista, então uma entidade que participa de vários pares tem os vetores carregados e agregados uma vez: use um comparador por execução.

O frame físico de uma feature é recuperado sem interpretar strings: para cada observação física da entidade, `perception_result_id_for(run, observação)` precisa ser o `perception_result_id` da referência; uma feature que não pertence a nenhuma observação da entidade é proveniência corrompida (`ValueError`).

Os vetores atravessam a fronteira como tuplas de `float`, nunca como tensores, e a aritmética é Python puro em ordem fixa, então o resultado é reproduzível e ler ou gravar resoluções não exige NumPy.

## O que este módulo não faz

Nenhuma classificação semântica, nenhuma projeção ou alinhamento entre espaços, nenhuma decisão `MATCH`/`DISTINCT` e nenhuma inferência de aparência nova: só lê features já extraídas.

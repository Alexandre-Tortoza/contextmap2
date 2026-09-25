# Comparação de representações 3D

Um canal **opcional** do par de entidades: compara as representações de estrutura 3D local (`PointRepresentation`) que as duas entidades referenciam (`RepresentationComparator`, `PointRepresentationEvidence`). A resolução funciona sem ele: com Point Representation desligada, indisponível ou incompatível, os outros canais continuam valendo, e a ausência é `unavailable`, nunca evidência para `DISTINCT`. Não depende de PTv3: um descritor determinístico e um encoder aprendido cruzam a mesma fronteira.

## Só onde a comparação faz sentido

| Regra | Efeito |
| --- | --- |
| **Só dentro de um espaço de representação.** Duas representações só são comparáveis se o fingerprint do `RepresentationSpace` for idêntico, nunca porque a dimensão coincide: outro checkpoint, versão de descritor, normalização ou semântica de suporte é outro espaço. | A política declara o `representation_space_id`. Entidade sem representação nesse espaço: `unavailable(incompatible_domain)`, sem carregar vetor algum. |
| **Ausência não é zero nem `DISTINCT`.** | Entidade sem nenhuma representação: `unavailable(missing_evidence)`. |
| **Contradição vira erro.** | Um run que informa outro espaço que a referência da entidade é evidência corrompida: `RepresentationSpaceMismatchError`, nunca comparação numérica. `ensure_compatible_representations` (Point Representation) é aplicado às representações **carregadas**. |
| **Componentes indefinidos nunca são interpretados.** | Ver abaixo. |
| **Estrutura estática, não vistas.** | Representações estão ancoradas à geometria, não a observações de câmera: não há agrupamento por observação física, cada suporte conta uma vez. |

As entidades só guardam referências (`PointRepresentationRef`), então a seleção por espaço, que precisa vir antes de qualquer I/O, usa o mesmo critério (fingerprint idêntico) sobre a referência.

## Componentes indefinidos

Um encoder pode não conseguir definir algum componente para um suporte (uma curvatura de suporte degenerado): o valor guardado é um **placeholder** que nunca deve ser interpretado. O canal:

- deixa o componente **fora** da comparação em vez de lê-lo como zero (um exemplo: `(1, 1, 0, ?)` e `(1, 1, 0, 1)` dão cosseno 1 nos três componentes definidos; lido como zero, daria 0,816);
- registra em `RepresentationMeasurement.compared_components` quantos dos `dimension` componentes foram comparados;
- fica `unavailable(insufficient_evidence)` quando as duas entidades definem em comum menos componentes que `min_defined_component_fraction`, ou quando os componentes em comum de uma entidade não têm direção (norma zero).

O descritor geométrico determinístico é o caso real: ele não define todos os componentes de um suporte em linha reta, e um teste usa esse encoder de verdade.

## Agregação `support-prototype-v1`

O protótipo da entidade tem, em cada componente, a **média do componente sobre as representações da entidade que o definem** (indefinido se nenhuma o define). `similarity` é o cosseno dos dois protótipos sobre os componentes definidos nos dois; `pair_similarity_min/max` são o menor e o maior cosseno entre uma representação de cada lado (sobre os componentes que aquele par tem em comum). O cosseno **não é probabilidade** e não é comparável com nenhum outro canal.

Ressalva de escala: num descritor cujos componentes têm escalas muito diferentes (`normalization: none`), o cosseno sobre valores brutos favorece os componentes grandes. Trocar a métrica seria uma nova versão da política, decidida pela avaliação (#140).

## Política `entity-representation-comparison-v1`

`RepresentationComparisonPolicy(representation_space_id, min_supporting_similarity, max_conflicting_similarity, min_defined_component_fraction)` **não tem valores padrão**: o que é "parecido" depende do espaço e se justifica com dados de validação. O achado `representation-similarity` (métrica `similarity`, com o valor e o limiar) é `supporting` se a similaridade for ≥ `min_supporting_similarity`, `conflicting` se `max_conflicting_similarity` existir e a similaridade for ≤ a ele, e `neutral` senão. Com `max_conflicting_similarity=None`, uma similaridade baixa nunca é evidência contra.

## Fonte dos vetores

`RepresentationVectorSource` isola os runs persistidos da comparação. `RunReaderRepresentationSource` lê, de cada `PointRepresentationRunReader`, uma representação e seu vetor sem carregar as outras (verificação de integridade continua sendo do run). Os vetores atravessam a fronteira como tuplas de `float`; a aritmética é Python puro em ordem fixa (reproduzível, sem NumPy). O comparador guarda o protótipo de cada entidade já vista: use um comparador por execução.

## O que este módulo não faz

Nenhuma classificação semântica a partir da estrutura 3D, nenhuma concatenação com vetores visuais ou semânticos, nenhuma comparação entre espaços e nenhuma decisão `MATCH`/`DISTINCT`. A justificativa de custo do canal (Point Representation é opcional) é uma ablação da milestone #140.

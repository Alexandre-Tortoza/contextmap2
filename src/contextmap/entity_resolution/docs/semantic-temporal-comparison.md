# Comparação semântica e temporal

Dois canais opcionais do par de entidades, ambos independentes e inspecionáveis pela política de resolução: `compare_semantics` (`SemanticEvidence`) e `compare_temporal` (`TemporalEvidence`). Nenhum decide um merge, e a **falta** de evidência semântica ou temporal nunca é evidência negativa. Como os demais canais, cada um é *medido* ou *indisponível*, com status `supporting`, `conflicting`, `neutral` ou `unavailable`.

## Semântica (`entity-semantic-compatibility-v1`)

Uma entidade guarda toda hipótese, alternativa, conflito, abstenção e atributo. O canal compara **tudo isso**, não um label por entidade, e a linha de base é conservadora e **sem ontologia**.

### Relação entre labels

| Relação | Regra |
| --- | --- |
| `same` | iguais depois da normalização (só caixa, pontuação e espaçamento: `Wooden-Pallet` = `wooden pallet`); nada de stemming, lematização ou tradução |
| `refinement` | um label é o outro mais palavras à esquerda que a **política declara** como modificadores, com o núcleo do nome intacto: `wooden pallet` refina `pallet` se `wooden` foi declarado; `pallet rack` **não** é um `pallet` e `toy pallet` não é refinamento se `toy` não foi declarado |
| `related` | só quando a política declara o par; nada é inferido por senso comum |
| `different` | qualquer outro caso |

`SemanticCompatibilityPolicy(refinement_modifiers, related_label_pairs)` não tem valores padrão: quais palavras são modificadores e quais labels são relacionados são escolha do perfil, validada (normalizados, ordenados e únicos) e entra no fingerprint. `BASELINE_REFINEMENT_MODIFIERS` é um conjunto inicial de cores, materiais e tamanhos que um perfil pode passar; é convenção, não ontologia. `compare_labels` expõe a relação e a regra versionada que a decidiu.

`SemanticMeasurement` guarda o estado de ambiguidade e o número de hipóteses de cada lado, a relação de **cada par** de labels (primária ou alternativa, com a regra) e as comparações de atributos. Nenhuma alternativa é descartada.

### Regras

| Regra | `supporting` | `conflicting` | `neutral` |
| --- | --- | --- | --- |
| `label-compatibility` | algum par é `same` ou `refinement` (o par primária-primária tem precedência no detalhe) | os dois estados são `unambiguous` e nenhum par é compatível nem relacionado (`pallet` ↔ `person`) | só há pares `related`; ou nenhum par compatível mas uma entidade é ambígua ou conflitante: as alternativas continuam abertas e **não é um falso conflito** |
| `attribute-compatibility` (só se há atributo com o mesmo nome dos dois lados) | nunca | algum valor difere | os valores concordam: concordar num material ou numa cor **não** é evidência de identidade |

Um conflito semântico não é prova de identidade distinta: a semântica upstream pode estar errada. Uma entidade **sem hipótese suportada** (desconhecida ou abstida) torna o canal `unavailable(missing_evidence)`, e o detalhe nomeia a entidade.

Os **scores das hipóteses nunca são lidos**: são sinais heterogêneos sem semântica comum definida, e uma hipótese só existe porque ao menos uma claim a apoia. O atributo `class` (derivado, que só repete a hipótese primária) e o conhecimento externo (nunca é evidência) ficam de fora. Cada comparação de atributo guarda a origem (`observed` ou `derived`) dos dois valores.

## Temporal (`entity-temporal-compatibility-v1`)

Compara o histórico de observações das duas entidades. Frames físicos e resultados de inferência são contados **à parte**: várias inferências sobre o mesmo frame são correlacionadas, nunca vistas independentes.

`TemporalMeasurement` guarda o domínio de relógio, a sobreposição **ou** a lacuna entre os intervalos (em ns; intervalos que se tocam têm as duas em 0), os frames físicos e os resultados de inferência de cada lado, os frames compartilhados e a união de frames distintos. Repetir a inferência não aumenta a união.

| Regra | `conflicting` | `neutral` |
| --- | --- | --- |
| `static-scene-co-observation` | a política declara `static_scene=True` **e** as entidades compartilham pelo menos um frame físico: numa cena estática, um objeto não aparece duas vezes num frame | senão (inclusive sem cena estática declarada) |

`TemporalCompatibilityPolicy(static_scene)` **não tem padrão**: a suposição de cena estática precisa ser justificada pelo perfil, porque uma segmentação excessiva pode dividir um objeto em duas regiões do mesmo frame. O achado guarda sempre a métrica (`shared_physical_observation_count`) e, com cena estática, o limiar.

O tempo **nunca infere movimento, re-identificação ou desaparecimento**: uma lacuna grande é medida, não julgada, e a premissa é cena estática ou quase estática. Domínios de relógio diferentes tornam o canal `unavailable(incompatible_domain)`, porque os instantes não são comparáveis. Um timestamp ausente não ocorre: uma entidade sem histórico não é uma entidade válida.

## O que estes módulos não fazem

Nenhuma decisão `MATCH`/`DISTINCT`, nenhum LLM ou senso comum, nenhum rastreamento de objetos dinâmicos e nenhuma alteração do estado semântico da entidade: as alternativas e os conflitos permanecem na entidade de origem.

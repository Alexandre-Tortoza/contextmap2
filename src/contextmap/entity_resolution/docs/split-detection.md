# Detecção de candidatos a divisão

Uma capacidade **opcional e só de diagnóstico**. Semantic Mapping materializa uma entidade por suporte fundido, e um suporte pode cobrir mais de um objeto físico, produzindo uma entidade sobre-fundida antes da resolução. Dividir é mais difícil que fundir, então `detect_split_candidates` apenas **reporta** `SplitCandidate`: nunca divide uma entidade, nunca roda dentro da resolução par a par e nunca muta uma entidade de origem. Nada aqui materializa uma divisão, então a resolução baseline é idêntica com ou sem a detecção (há teste comparando as duas).

## O que faz (`entity-split-detection-v1`)

Resolve as referências de geometria exatas da entidade no `GeometrySource`, liga pontos mais próximos que `connectivity_radius_m` (grade + union-find, determinístico) e toma os componentes conexos como **partições candidatas**, cada uma guardada como as **referências exatas** que a compõem (subconjunto autoritativo do suporte, nunca cópia de coordenadas) mais a caixa. Uma entidade de suporte conexo não é candidata. Uma entidade desconexa **não** é dividida só por ser desconexa: os sinais decidem.

| Sinal | `supporting` | `conflicting` |
| --- | --- | --- |
| `disconnected-components` | pelo menos duas partições **significativas** (`min_partition_points` pontos e `min_partition_fraction` do suporte) | (neutro se há menos de duas: um suporte esparso cujos pedaços são ruído não qualifica) |
| `partition-gap` | a menor lacuna entre as caixas de duas partições significativas é ≥ `min_gap_m` | menor: os pedaços podem ser um objeto esparso |
| `dominant-partition` | (neutro) | uma partição tem mais que `max_dominant_fraction` do suporte: o resto parece ruído ao redor de um objeto |

## Estado (`SplitStatus`)

| Estado | Condição |
| --- | --- |
| `suggested` | há o sinal de componentes e nenhum sinal conflitante |
| `unresolved` | há o sinal de componentes, mas outro sinal argumenta contra |
| `rejected` | não há duas partições significativas |

`SplitDetectionPolicy` não tem valores padrão: os limiares dependem da escala e da densidade da cena e se justificam por avaliação. O `SplitCandidate` guarda a entidade (`EntityReference`), as partições ordenadas e disjuntas, os sinais (`Finding` com métrica, valor e limiar), o estado (validado contra os sinais), a política e o fingerprint. Uma referência de suporte que não resolve levanta `GeometryResolutionError`, sem pular.

## O que não faz

- **Não materializa divisão.** A issue permite materializar só casos determinísticos de alta confiança "se a validação sustentar"; não há validação, então nenhuma divisão automática existe. Qualquer materialização futura precisaria de configuração, versão e avaliação próprias, e desligada por padrão.
- Não usa o label semântico como critério de divisão e não implementa descontinuidade semântica nem de aparência, nem clustering além da conectividade: nenhum tem consumidor hoje.
- Não rastreia objetos dinâmicos.

O custo é linear no número de pontos do suporte (mais a resolução da geometria no mapa), sem a checagem de conectividade quadrática do entity model.

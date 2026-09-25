# QA das anotações e reprodutibilidade dos evaluators

A infraestrutura de avaliação também precisa ser validada: anotações ruins, identidades inconsistentes, verdade de relação contraditória ou um evaluator não determinístico invalidam todo resultado que dependa deles. Este módulo cobre duas verificações: o **QA do conteúdo das anotações** (`contextmap.evaluation.annotation_qa`) e a **reprodutibilidade dos evaluators** (`contextmap.evaluation.evaluator_reproducibility`).

A integridade do manifesto (unicidade, splits, vazamento, proveniência, hashes) é a de [`reference-integrity.md`](reference-integrity.md). O QA a **inclui** no seu relatório e verifica o que ela não vê: o conteúdo dos arquivos de anotação.

## Relatório

`check_annotation_quality(manifest, root, policy=…)` nunca lança para um reference set defeituoso: todo problema vira um achado. O `AnnotationQaReport` (schema `contextmap.annotation-qa/v1`) separa:

| Grupo | Significado |
|---|---|
| **blockers** | o reference set não pode ser usado até serem corrigidos |
| **warnings** | suspeito, merece revisão, não desqualifica |
| **permissible** | ambiguidade, desconhecido e cobertura parcial: informação sobre os dados, **não defeito** |
| **disagreements** | diferenças entre anotadores; visíveis e versionadas, **não** blockers |
| **integrity** | o relatório de integridade completo, incluindo os splits |

`is_valid` exige integridade válida **e** nenhum blocker de QA. Arquivos ausentes, alterados ou ilegíveis são pulados pelo QA e aparecem como blockers no relatório de integridade embutido.

`AnnotationQaPolicy` traz as tolerâncias **sem valores padrão** e versionadas (`policy_id`): distância em pixels e em metros abaixo da qual duas correspondências são a mesma, e o IoU mínimo para dois anotadores terem marcado a mesma região.

## Verificações por família

| Família | Blockers | Warnings |
|---|---|---|
| `regions` | máscara ou área com tamanho diferente da imagem; máscara vazia; caixa fora da imagem; máscara fora da própria caixa; geometria duplicada; região inteira em área de exclusão ou inteira fora da área válida | região que sobrepõe parcialmente exclusão ou se estende além da área válida |
| `geometry` | calibração que o reference set não declara; pixel fora da imagem; pontos do mesmo scan em frames diferentes; mesmo pixel com pontos 3D distintos | correspondência repetida |
| `identity` | ocorrência em região que não existe; uma região reivindicada por duas identidades | identidade repetida na observação; ocorrência fora do escopo declarado; conceitos disjuntos em observações diferentes |
| `relations` | identidade inexistente; âncora onde a identidade não é vista; violação de **simetria** ou de **inverso** (`HOLDS` × `DOES_NOT_HOLD`); mesma relação com status conflitantes | relação duplicada |
| `semantics` | região inexistente | — |
| `visibility` | alvo inexistente; nível incompatível com a fração ocluída; identidade `NOT_VISIBLE` onde tem região anotada | — |

Fração ocluída por nível: `fully_visible` só aceita 0; `not_visible` só 1; `unknown` nenhuma; os níveis parciais exigem valor em (0, 1).

## Ambiguidade e desconhecido não são defeitos

`AMBIGUOUS` e `UNKNOWN` (semântica, relação e visibilidade) e a cobertura parcial (regiões e identidade) vão para `permissible`, com a família, o arquivo e o alvo. Uma anotação `UNKNOWN` significa que o anotador não soube dizer: isso é verdade sobre o dado, não erro a corrigir.

## Anotadores diferentes: divergência visível, sem escolha silenciosa

Quando há mais de um arquivo da mesma família:

- **anotadores diferentes** geram um `DisagreementSummary` por par: quantos alvos foram comparados, quantos concordam e, para cada divergência, **o que cada arquivo diz** (`positions`). Semântica compara status e conceitos sob a normalização declarada (`status-differs`, `disjoint-concepts`, `partial-concept-overlap`); relações e visibilidade comparam status e nível; regiões casam por IoU (`region-sets-differ`, com as regiões sem par). Nada marca um lado como certo, e o relatório não tem campo de resolução ou consenso;
- o **mesmo anotador** duas vezes não é divergência: registros idênticos são `duplicate-annotation-record` (warning) e registros diferentes `conflicting-annotation-records` (blocker);
- normalizações diferentes (`normalization-differs`, warning) não são comparadas em silêncio: o par fica sem resumo.

Identidade, geometria e contexto de cena não têm resumo de divergência entre anotadores nesta versão (lacuna registrada).

## Saída de modelo nunca corrige a verdade de referência

Isso é garantido na integridade do manifesto: uma anotação de origem `model_inference` só pode ser `diagnostic_only`, e uma anotação semeada por artifact de modelo sem revisão explícita é blocker. O QA embute esse relatório; nenhuma verificação de QA altera uma anotação.

## Certificação

`certify_reference_set(root, policy=…)` abre o reference set e exige integridade **e** QA sem blockers, devolvendo um `CertifiedReferenceSet` (com o `ValidatedReferenceSet` e o relatório de QA). Levanta `ReferenceSetIntegrityError` ou `AnnotationQaError`. A integridade é pré-requisito das execuções oficiais de avaliação: `run_experiment()` ([`experiments.md`](experiments.md)) só aceita um `ValidatedReferenceSet`.

## Reprodutibilidade dos evaluators

`check_evaluator_reproducibility(evaluate, repetitions=…, nondeterministic=…)` roda o evaluator repetidamente sobre as **mesmas** entradas (artifacts e reference set fechados pelo `evaluate`) e compara os relatórios. Dois relatórios são equivalentes quando tudo coincide, exceto os **valores** das medições de recursos: metadados de reprodutibilidade, cada métrica de qualidade, o status e o `sample_count` de cada métrica e o relatório do estágio.

- **Recursos.** Tempo, memória e storage variam por natureza; os valores são excluídos e o resultado registra `performance_values_excluded` e o motivo. A presença, o status e o `sample_count` continuam comparados.
- **Não determinismo inevitável precisa ser declarado.** `NondeterministicField(location, reason)` com `quality_metric:<métrica>` ou `stage_report:<chave.pontilhada>` e um **motivo obrigatório** (por exemplo, redução em ponto flutuante não ordenada em GPU). Só o campo declarado é excusado; qualquer outra diferença é falha. O resultado lista o que foi declarado.
- `require_reproducible()` levanta `EvaluatorNondeterminismError` com os locais que diferiram, e `compare_evaluation_reports()` compara dois relatórios diretamente.

## O que este módulo não faz

- não corrige nem escolhe anotações: relata;
- não interpreta a qualidade científica do rótulo (se "pallet" é o conceito certo), só a consistência entre registros;
- máscaras usam `InlineMask` e as comparações são por pixel; para máscaras grandes será preciso um formato compacto.

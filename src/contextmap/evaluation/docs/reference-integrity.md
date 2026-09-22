# Integridade do reference set

Uma avaliação científica é inválida se amostras de tuning e de avaliação se sobrepõem sem querer, se frames relacionados de uma mesma sequência física caem dos dois lados de um split ou se uma anotação carrega saída de modelo sem que isso apareça. A validação de integridade (`contextmap.evaluation.reference_integrity`) detecta isso no manifesto ([`reference-set.md`](reference-set.md)) e nos arquivos de anotação ([`annotations.md`](annotations.md)).

## Relatório

`validate_reference_set(manifest, root=None)` nunca lança para um reference set inválido: todo problema vira um `IntegrityFinding` (`code` estável, mensagem e `subjects`) de severidade **blocker** ou **warning**. O `ReferenceSetIntegrityReport` traz a identidade (`id`, `version`, `digest`), se os arquivos foram checados (`files_checked`), os achados em ordem determinística e a auditoria de proveniência. `to_record()` o serializa em JSON.

Um **blocker** recusa o reference set; um **warning** pede atenção sem recusá-lo.

## Entradas para as ferramentas de avaliação

| Função | Uso |
|---|---|
| `validate_reference_set()` | diagnóstico: devolve o relatório, mesmo inválido |
| `require_valid_reference_set(manifest, root)` | lança `ReferenceSetIntegrityError` se houver blocker; devolve `ValidatedReferenceSet` |
| `open_validated_reference_set(root)` | lê o manifesto (digest verificado) e o valida por completo |

`ValidatedReferenceSet` não aceita um relatório com blocker, de outro reference set ou sem a checagem dos arquivos. As ferramentas de avaliação que o exigem **recusam por padrão** um manifesto inválido; não existe modo "ignorar blockers".

## Política de split

Cada `SplitScheme` declara a **tarefa**, a **unidade** (`sequence`, `scene`, `time_segment`, `physical_object` ou `other`) e a **justificativa** de por que essa unidade evita vazamento. Evite splits aleatórios por frame quando frames adjacentes gerariam vazamento óbvio.

- A chave de grupo de uma amostra para a unidade do scheme vem de `ReferenceSample.groups`; para `sequence` é a própria fonte.
- `adjacency_window_ns` (opcional) declara que amostras da mesma fonte com intervalos de tempo mais próximos que a janela contam como adjacentes e devem ficar no mesmo split.

## Catálogo de checagens

**Blockers**

| Código | Quando |
|---|---|
| `duplicate-sample-content` | amostras com o mesmo `content_hash` |
| `calibration-source-mismatch` | a calibração da amostra pertence a outra fonte |
| `no-split-scheme` / `empty-scheme` | não há scheme, ou o scheme não tem splits |
| `split-overlap` | uma amostra em mais de um split do mesmo scheme |
| `split-leakage` | uma unidade (sequência, cena, …) atravessa splits do scheme |
| `missing-group-key` | a amostra não tem chave de grupo para a unidade do scheme |
| `split-observation-overlap` | a mesma observação física em splits diferentes |
| `adjacent-split-leakage` | amostras adjacentes (dentro da janela) em splits diferentes |
| `missing-time-span` | há janela de adjacência e a amostra não tem intervalo de tempo |
| `duplicate-annotation-path` | dois registros apontam para o mesmo arquivo |
| `unknown-annotation-schema` | schema de anotação não suportado |
| `unreviewed-model-seeding` | anotação semeada por artifact de modelo, sem revisão explícita, declarada com trust acima de `diagnostic_only` |
| `annotation-file-missing` / `-hash-mismatch` / `-outside-root` | arquivo ausente, alterado ou fora da raiz |
| `annotation-unreadable` / `annotation-schema-mismatch` | arquivo ilegível, ou de outra família que a declarada |
| `annotation-unknown-sample` / `-sample-not-declared` / `-unknown-observation` | vínculo do conteúdo com amostras e observações do manifesto |
| `annotation-calibration-mismatch` | correspondência geométrica usa calibração que não é da amostra |

**Warnings**

| Código | Quando |
|---|---|
| `shared-observation` | uma observação física em mais de uma amostra |
| `sequence-shared-across-splits` | a unidade não é a sequência e uma sequência alimenta mais de um split |
| `no-test-split`, `empty-split`, `multiple-schemes-for-task` | política de split incompleta ou ambígua |
| `sample-not-in-any-split`, `sample-without-annotation` | amostra sem split ou sem anotação |
| `duplicate-annotation-content`, `annotation-without-samples`, `unused-provenance` | higiene do manifesto |
| `model-seeded-reviewed` | anotação semeada por modelo, mas revisada explicitamente |
| `diagnostic-annotation` | anotação `diagnostic_only`: não sustenta métricas de aceite |

## Proveniência auditável independentemente de saídas de modelo

Cada arquivo de anotação gera um `ProvenanceAuditEntry` com trust, origem, anotador, método, artifacts de semeadura, se houve revisão e `independent_of_model_output` — verdadeiro apenas quando a anotação nem vem de inferência de modelo nem foi semeada por um artifact de modelo. Uma anotação `model_inference` já não constrói como referência confiável (ver o manifesto); a validação cobre também o caso de anotação manual semeada por modelo sem revisão.

## O que esta validação não faz

- não inspeciona a qualidade do conteúdo das anotações (tamanho de máscara vs imagem, consistência de identidade, relações): é a verificação de qualidade das anotações;
- não decide se um split é cientificamente adequado: exige que a unidade e a justificativa sejam explícitas e que a unidade declarada não vaze.

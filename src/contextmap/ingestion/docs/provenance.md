# Provenance, integridade e identidade de conteúdo

Este documento descreve `src/contextmap/ingestion/sequence_provenance.py` e as extensões de integridade em `sequence_artifact.py` (issue #39). Para a estrutura geral do artefato, ver [`artifact.md`](artifact.md); para as convenções globais, ver [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md).

## Por que uma issue separada da #39

A issue #39 já registra hash + tamanho por arquivo no `manifest.json` — suficiente para detectar corrupção/arquivo ausente. O que faltava, e esta issue define, é: **de onde os dados vieram**, **como foram normalizados** e **quando dois resultados de ingestão contam como "a mesma coisa"**. `artifact.md` já reservava `provenance/` como diretório não criado pelo writer v0, adiado explicitamente para esta issue.

## Identidade de conteúdo

`compute_content_identity(provenance)` é determinística sobre exatamente três campos: `source_type`, `source_content_hash`, `configuration_hash`. **`source_path` é deliberadamente excluído.**

Consequências, correspondendo diretamente aos requisitos da issue:

| Situação | `source_content_hash` | `configuration_hash` | Identidade |
| --- | --- | --- | --- |
| mesma fonte, mesma config | igual | igual | **igual** |
| mesma fonte, config de sync/calibração mudou | igual | diferente | **diferente** |
| fonte modificada sob o mesmo path | diferente (recalculado do conteúdo) | igual | **diferente** |
| mesmo conteúdo, path diferente | igual | igual | **igual** |

A última linha é intencional: "não assumir que igualdade de path significa igualdade de conteúdo" vale nos dois sentidos — path igual não garante conteúdo igual, e path diferente não impede conteúdo igual ser reconhecido como a mesma identidade.

## `compute_source_content_hash()`

Hasheia a fonte bruta inteira: um arquivo (bag ROS 1) é hasheado diretamente; um diretório (bag ROS 2) é hasheado sobre a lista ordenada de paths relativos + tamanho + hash de cada arquivo — nunca apenas path/mtime, que não detectaria conteúdo trocado sob o mesmo nome.

**Trade-off deliberado**: isso é O(tamanho da fonte) — lê cada byte. A função não decide quando rodar; quem ingere decide (tipicamente uma vez, no momento da ingestão, não a cada leitura do artefato já persistido). Isso segue a orientação explícita da issue: "não torne o rehash de todo payload grande caro em toda leitura se uma estratégia mais barata preserva integridade" — a estratégia mais barata aqui é: hash pesado uma vez na escrita, hash leve (já feito pela issue #39) na leitura.

## Metadados de `SequenceProvenance`

Ver docstring de `SequenceProvenance` para a lista completa de campos. Nenhum é obrigatório além de `source_type`/`source_path` — os demais (`configuration_hash`, `adapter_type`, `code_version`, `calibration_source_hash`, `synchronization_policy`, `warnings`) ficam `None`/vazios quando não aplicáveis, nunca com valor sentinela.

`current_code_version()` retorna `contextmap.__version__` quando o pacote está instalado com uma versão real (não o fallback `"0.0.0"` do `setuptools_scm` fora de um checkout git) — quem persiste a sequência decide se chama essa função ou passa outra identidade (ex.: commit SHA explícito).

## Persistência no artefato

`writer.set_provenance(provenance)` antes de `finalize()` grava `provenance/provenance.json` (incluindo `content_identity` já calculado, para inspeção sem precisar recomputar). `reader.read_provenance()` retorna `None` para artefatos finalizados antes desta issue — mesma compatibilidade retroativa de `read_calibration()` (#41).

## Extensão de `verify_integrity()`

Além das checagens já existentes da issue #39 (arquivo ausente, tamanho/hash incorretos), `verify_integrity()` agora também detecta **cross-references inválidas entre `index.jsonl` e o inventário de arquivos**: um registro do índice cujo `payload_path` não corresponde a nenhuma entrada em `manifest.file_inventory`. Essa checagem roda tanto na leitura (`SequenceArtifactReader.verify_integrity()`) quanto internamente antes de `finalize()` — um índice corrompido nunca chega a virar um artefato "válido" no path final.

Schema incompatível/desconhecido continua sendo detectado por `_load_manifest()` (levanta `SequenceArtifactError`/`IncompleteSequenceArtifactError` antes de qualquer outra leitura ser tentada) — um reader que não entende o schema não pode interpretar com segurança mais nada do artefato, então essa checagem não faz parte da lista de problemas de `verify_integrity()`; ela impede o reader de sequer abrir.

## O que continua fora do escopo

Verificação em lote de múltiplos artefatos, um comando de linha de comando dedicado, e diagnósticos estruturados (`diagnostics/`) ficam para a issue #47 (fixtures determinísticas e diagnóstico de debug) — esta issue entrega a metadata e a função de identidade/verificação como API Python, não uma ferramenta de CLI.

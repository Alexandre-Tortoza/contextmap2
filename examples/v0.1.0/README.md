# Exemplos do v0.1.0

Tudo aqui é verificado por `tests/packaging/test_release_examples.py`, que roda na instalação base (só NumPy) no job `lightweight-install` da CI. Se um comando abaixo deixar de funcionar, a CI falha.

## Procedência, sem ambiguidade

| Arquivo | O que é |
| --- | --- |
| [`demo/`](demo/) | **sintético.** Formato real, conteúdo inventado: escrito pelos writers reais das capabilities, mas sem nenhum dado de sensor gravado, modelo ou dataset. **Não é benchmark** e não diz nada sobre qualidade no mundo real |
| [`canonical-config.toml`](canonical-config.toml) | a configuração canônica congelada do v0.1.0, com os caminhos de entrada como **placeholders**: o bag do `corridor-02` não é redistribuível ([third-party-licenses.md](../../docs/third-party-licenses.md)) |
| [`effective-config.json`](effective-config.json) | a configuração **efetiva** resolvida a partir do arquivo acima, com o digest e as fontes. Gerada, não escrita à mão |
| [`resolved-plan.txt`](resolved-plan.txt) | a topologia resolvida dos 12 estágios, sem executar nada. Gerada |

A evidência sobre dados **reais** não está aqui: está em [`v0.1.0-release-contract-20260925`](../../src/contextmap/evaluation/docs/validation/v0.1.0-release-contract-20260925.md) e no registro da campanha em [`experiments/e2e-real-canonical-run-20260923/`](../../experiments/e2e-real-canonical-run-20260923/).

## Abrir e validar o artifact de demonstração

Só a instalação base é necessária — sem ROS, sem Torch, sem GPU:

```bash
pip install contextmap
contextmap validate examples/v0.1.0/demo/context-map
contextmap inspect artifact examples/v0.1.0/demo/context-map
```

`validate` responde `ok`. Em Python:

```python
from contextmap.artifact import ContextMapArtifactReader, ValidationLevel, validate_context_map_artifact

report = validate_context_map_artifact("examples/v0.1.0/demo/context-map", level=ValidationLevel.FULL)
print(report.status.value)  # verified

reader = ContextMapArtifactReader.open("examples/v0.1.0/demo/context-map")
for entity in reader.entities():
    print(entity.entity_id, entity.semantic_state.status.value, len(entity.geometry_refs))
```

## Por que o pacote leva a linhagem junto

Um `ContextMapArtifact` **referencia** geometria em vez de copiá-la, e três das suas dependências são obrigatórias. Um mapa distribuído sozinho é **inválido**: `validate` reporta `dependency.required_missing`. Isso é o contrato funcionando, não um defeito.

Por isso `demo/` carrega, ao lado do mapa, o `GeometricMapArtifact`, o `EntityResolutionRunArtifact` e o `SpatialRelationsRunArtifact` que ele cita, na mesma disposição relativa em que foi escrito — então as dicas gravadas no manifesto resolvem sem nenhuma flag. O pacote inteiro tem ~166 KB.

Os três avisos `dependency.optional_missing` que aparecem são esperados: a sequência, o run de percepção e o mapa semântico são dependências **opcionais** e não vão no pacote.

## O que o artifact de demonstração exercita

Não é um smoke vazio. Ele tem 3 entidades e 10 relações, e cobre:

- **referências de geometria**: as 3 entidades apontam para elementos do `GeometricMapArtifact` citado, por índice global;
- **alternativas semânticas**: uma entidade carrega mais de uma hipótese, e os três estados de ambiguidade aparecem (`unambiguous`, `ambiguous`, `insufficient_evidence`). O mapa **preserva** a ambiguidade em vez de escolher um rótulo;
- **relações** com as identidades e o estado reais do run de spatial relations;
- **linhagem** com o digest de cada artifact de origem.

## Resolver a configuração e o plano

```bash
contextmap inspect config -c examples/v0.1.0/canonical-config.toml --json
contextmap inspect plan   -c examples/v0.1.0/canonical-config.toml
```

Nenhum dos dois executa estágio algum. Para um run real, troque `inputs.sequence` pela identidade do seu `SequenceArtifact` já ingerido e `resources.device` por `cuda` (os backends de visão exigem o extra `vision`).

A configuração seleciona um backend **por ponto de variação, explicitamente**: o runtime nunca escolhe no lugar de quem chama, mesmo quando a política tem uma única opção.

## Um aviso sobre o backend semântico

`components.visual_perception.semantic_interpretation` usa o Qwen3-VL, que é **experimental** no v0.1.0. Ele roda no pipeline canônico e sua evidência chega ao mapa final, mas a equivalência exata de rerun das claims que ele gera **não faz parte** do contrato de release: duas execuções reais independentes da configuração idêntica concordaram em 29/90 (32,2%) das claims canônicas. Ver [#556](https://github.com/Alexandre-Tortoza/contextmap2/issues/556) e a seção 3 de [release-v0.1.0.md](../../docs/release-v0.1.0.md).

## Regenerar

```bash
python experiments/release-v0-1-0-demo-artifact/build_demo_context_map.py examples/v0.1.0/demo
contextmap inspect config -c examples/v0.1.0/canonical-config.toml --json > examples/v0.1.0/effective-config.json
contextmap inspect plan   -c examples/v0.1.0/canonical-config.toml       > examples/v0.1.0/resolved-plan.txt
```

O gerador é determinístico: o instante gravado como provenance é fixo, então o artifact sai byte a byte igual.

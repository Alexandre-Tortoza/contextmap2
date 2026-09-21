# Validação real dos estágios novos pelo layout da runtime (2026-09-21)

Registro do primeiro run **real** que atravessa a runtime (`run_plan` + `RunJournal`) com os executores de [`runtime/docs/executors.md`](../../../runtime/docs/executors.md) e grava no layout `<workspace>/<dataset>/<run>/<estágio>/`. Ele exercita só o que é **novo**: Semantic Mapping, Entity Resolution e Spatial Relations. Tudo a montante é reaproveitado por referência do [run parcial de 2026-09-21](e2e-real-sample-20260921.md).

**Classe de evidência: `real`, parcial.** Não cumpre nenhum gate do cenário [`solution-1-canonical` 1.0.0](../end-to-end.md): não é um run canônico, não passa por ingestion, percepção, State Estimation, Geometric Mapping, Sensor Association nem Semantic Fusion, e não chega ao `ContextMapArtifact` (não existe o passo de montagem).

## O que rodou

| Item | Valor |
|---|---|
| Dispositivo | CPU; nenhum modelo, nenhuma chamada externa, nenhum quadro fora da máquina |
| Reaproveitado (somente leitura, por `ArtifactRef` com `location`) | mapa geométrico `run-0001__ts-w336-90s__voxel20cm` (4 313 593 pontos, voxel 0,20 m) e run de fusão baseline da mesma validação |
| Produzido | `semantic_mapping`, `entity_resolution`, `spatial_relations`, cada um em `workspace/corridor-02/run-NNNN/<estágio>/` (não versionado) |
| Escopo | `plan.scope(targets=["spatial_relations"], provided={geometric_mapping, semantic_fusion})` |
| Políticas | **exploratórias e declaradas**, sem calibração com anotações: resumo de geometria (mín. 5 pontos, raio de conectividade 0,3 m), recuperação por centroide (3,0 m, margem 0,3 m), comparação só por **geometria** (o run de fusão tem **0 hipóteses**: as entidades não têm estado semântico e não há canal semântico), predicados geométricos e de contato com os valores dos testes |

Ressalva de interpretação: o canal de contato usa distância de 0,05 m sobre um mapa em voxel de 0,20 m; o resultado de `touching`, `on_top_of` e `leaning_against` não é interpretável como contato físico.

## Fatia pequena (3 imagens, 41 suportes): verde

Fatia de desenvolvimento, a menor que já existe da validação (o run de fusão de 3 imagens). Três execuções da mesma configuração:

| Run | O que foi | Tempo | Armazenamento |
|---|---|---|---|
| `run-0001` | executa os três estágios (com política de reuso) | 9,7 s | 7,57 MB (mapeamento 0,26 MB, resolução 2,99 MB, relações 4,29 MB) |
| `run-0002` | mesma execução: os três estágios **reutilizados por referência** ao `run-0001` | 0,02 s | 21 KB (só o diário; nenhuma pasta de estágio) |
| `run-0003` | mesma execução **sem reuso** | 9,7 s | 7,57 MB |

Pico de RSS do processo: **204 MB**. Resultados:

- **Reprodutibilidade dos três estágios:** o `run-0003` produz o mesmo `artifact_id` e o mesmo digest de inventário que o `run-0001`, estágio a estágio (`semantic_mapping`, `entity_resolution`, `spatial_relations`). Vale para estes três estágios e **não** satisfaz `reproducibility.rerun_equivalence`, que exige o run canônico.
- **Reuso por referência:** o `run-0002` aponta para `corridor-02/run-0001/<estágio>`, sem cópia; os artifacts têm integridade verificada pelos leitores.
- **Conteúdo (para dimensionar, não para julgar qualidade):** 41 entidades geradas, 223 pares comparados (6 `match`, 136 `distinct`, 81 `unresolved`), 35 entidades resolvidas; Spatial Relations com 864 candidatos, 864 evidências e 1 297 relações (66 `supported`, 258 `rejected`, 973 `unresolved`).

## Fatia de aceitação (20 imagens, 155 suportes): falha real em Entity Resolution

O `run-0004` executou `semantic_mapping` (155 entidades, 1,46 MB, integridade verificada) e **falhou em `entity_resolution`**, sem deixar pasta do estágio nem `.tmp-*`, e com o diário marcado como `failed`:

```text
ValueError: contradiction 'contradiction--0a1ef4caf041dbbd' is not recorded on the entity
'entity--support-000001' it withholds
```

**Causa (defeito da capability, não da runtime):** em `entity_resolution.materialize_resolved_entities`, o mapa `withheld` associa **uma** contradição a cada entidade (`{referência: id}` sobrescrito a cada contradição). Quando um mesmo componente de `match` contém **duas ou mais** decisões `distinct` (por exemplo `a~b~c~d` com `a != c` e `b != d`), cada entidade guarda só a última, e a validação de `ResolvedEntityMaterialization` exige que toda contradição esteja registrada em toda entidade do seu componente. Reproduzível sem dados reais: com uma contradição a materialização funciona, com duas levanta `ValueError`. A correção é guardar o **conjunto** de contradições por entidade (`contradiction_ids` já é uma tupla). Esta milestone não altera o código de Entity Resolution (pertence à milestone `entity-resolution`); o achado fica para o dono da capability, com o teste de regressão do caso de duas contradições.

Consequência: Spatial Relations **não foi executada** sobre as 20 imagens, e o run de aceitação de 90 s / 20 imagens fica **sem resultado**, não aprovado.

## O que não foi rodado, de propósito

- Ingestion, percepção, State Estimation, Geometric Mapping, Sensor Association e Semantic Fusion: reaproveitados (regra de custo; ver [`end-to-end.md`](../end-to-end.md)); o run canônico completo continua bloqueado (sem passo de montagem do `ContextMapArtifact`, sem janela de seleção na ingestion, sequência pinada sem poses nem modelo de câmera).
- Recuperação de interrupção por `--resume` em dados reais: fica como evidência **`fake_contract`** (`tests/end_to_end/test_runtime_chain.py`); injetar uma falha real não acrescenta informação sobre o contrato.
- GPU, o bag inteiro de 24 GB e qualquer chamada externa.

## Recursos e armazenamento (#181)

Só CPU e só estes três estágios; não é o perfil de custo do #181 (não separa E/S de cálculo e não cobre os estágios a montante). Crescimento medido em `workspace/`: 7,57 MB por run que executa os três estágios sobre 41 suportes, 21 KB por run que só reutiliza.

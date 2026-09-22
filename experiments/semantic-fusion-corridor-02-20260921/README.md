# Bundle de reprodução: Semantic Fusion sobre `corridor-02` (20 frames)

Registro **leve e versionado** da primeira execução real de Semantic Fusion (issue #121): a seleção dos 20 frames, o manifest do experimento com as identidades e os hashes completos das runs de entrada, a configuração dos sete braços, os drivers que rodaram e os relatórios finais. A leitura dos resultados está em [`src/contextmap/evaluation/docs/semantic_fusion.md`](../../src/contextmap/evaluation/docs/semantic_fusion.md).

Nada aqui é contrato do pipeline. O bundle **não** contém o dataset nem artifacts grandes: fica abaixo de 1 MB, e `tests/evaluation/test_experiment_bundles.py` confere isso, além dos hashes, dos arquivos listados, da ausência de caminhos pessoais e de segredos.

## O que está no bundle

| Caminho | O que é |
| --- | --- |
| `manifest.json` | Identidade do experimento: versões de código, arquivos do dataset (hash), `SequenceArtifact`, seleção (identidade e os 20 frames), runs de entrada (`run_id`, hashes, digest dos outputs), parâmetros declarados antes de olhar as saídas, os sete braços e a tabela de arquivos com SHA-256 completo |
| `selection.json` | A seleção exata: janela de 90 s, 892 scans LiDAR, os 20 frames de imagem e a `selection_identity`. Idêntica byte a byte à usada na execução |
| `scripts/` | Os drivers que rodaram (`s01` a `s05`, `fusion_inputs.py`, `common.py`), os de leitura dos resultados (`s06`, `s07`, `qstats.py`) e `verify_inputs.py` (confere as entradas locais com o manifest) |
| `reports/semantic_fusion/report.json` | O relatório completo dos sete braços: avaliação por braço, comparação controlada, determinismo, repetição de inferência, sensibilidade estrutural e custo |
| `reports/semantic_fusion/visual_consistency.json` | A análise visual (proxy sem anotação, medida ad hoc do driver, **não** é métrica do harness) |
| `reports/state_estimation/`, `geometric_mapping/`, `sensor_association/` | Os resumos de cada etapa a montante, com `run_id`, contagens, tempo e memória |
| `reports/trial_sensor_association_2frames/` | O trial de 2 frames de cuja marginal saíram as rampas de qualidade, declaradas antes de olhar qualquer saída de fusão |

Os drivers `s08_final_numbers.py` e `probe_pr.py` do diretório original não estão aqui: o primeiro só copiava arquivos e gerava o `HASHES.json` (o manifest o substitui) e o segundo era uma sonda de tempo sem número citado.

## O que não está no Git e precisa existir localmente

| Insumo | Identidade (no manifest) | Como obter |
| --- | --- | --- |
| Dataset `corridor-02` em `datasets/corridor-02/` | SHA-256 de `corridor-02-gt.txt` e de `corridor-02-Intrinsics.yaml` | Não versionado (o bag tem 24 GB). O bag não é lido por estes drivers |
| `SequenceArtifact` em `outputs/ingest-full/sequences/corridor-02/<id>` | `artifact_id` e SHA-256 do `manifest.json` | Produzido pela Ingestion sobre o bag; fora deste bundle |
| 3 runs de percepção em `outputs/validation/2026-09-21/visual_perception/workspace/runs/visual-perception/corridor-02` | `run_id`, digest dos outputs e contagens (`claim_count` é 0 nas três) | Produzidas antes por SAM2/SAM3 + DINOv2 + CLIP em GPU; **este bundle não as regenera** |
| State Estimation, Geometric Mapping, Sensor Association | `run_id` e digest dos outputs | Regeneradas por `s01`, `s02` e `s03` (sem GPU): 1 s, 13 s (mapa de 924 MB) e de 214 a 260 s por run com pico de 18 a 21 GB de memória |
| Point Representation e os sete braços de fusão | `run_id` e digest dos outputs | Regenerados por `s04` (cerca de 9 minutos, sem GPU, pico de 3,5 GB) |

## Como reproduzir

```bash
# Raiz do checkout que contém datasets/ e outputs/ (padrão: a raiz deste checkout).
export CONTEXTMAP_INPUT_ROOT=/caminho/do/checkout
# Onde as etapas gravam (padrão: workspace/corridor-02/validation-semantic-fusion-20260921).
export CONTEXTMAP_VALIDATION_DIR=/caminho/de/saida

pip install -e '.[dev]' pyyaml     # o common.py usa PyYAML para ler os intrínsecos
cd experiments/semantic-fusion-corridor-02-20260921/scripts

python s01_state_estimation.py                       # ExternalPose a partir do GT do dataset
python s02_geometric_mapping.py all-points           # mapa LiDAR de 14 422 535 pontos
python s03_sensor_association.py \
    run-0001__frames-08026-10078__sam2-dinov2-clip \
    run-0005__frames-08026-10078__sam2-dinov2-clip \
    run-0006__frames-08026-10078__sam3-dinov2-clip   # uma run de associação por run de percepção
python verify_inputs.py                              # entradas locais x manifest
python s04_fusion.py                                 # Point Representation, sete braços, avaliação
python s05_visual_consistency.py                     # visual_consistency.json
python s06_summary.py; python s07_digest.py          # os números citados na documentação
```

Cada etapa lê os artifacts das anteriores em `CONTEXTMAP_VALIDATION_DIR`, o `SequenceArtifact`, as runs de percepção e a seleção versionada neste bundle. `s04_fusion.py` aceita `--association-root` e `--stage` para apontar outra origem de associação e outro diretório de saída.

### Como conferir

- **Entradas:** `verify_inputs.py` procura cada run pelo `run_id`, recalcula o digest dos outputs a partir do `manifest.json` local e compara com o manifest do bundle, além dos hashes dos dois arquivos do dataset. O digest é o SHA-256 do texto `<caminho><tab><content_hash><nova linha>` das entradas de `outputs/`, ordenadas por caminho. `metrics/` fica de fora porque guarda tempos. A conferência é entre inventários; que os arquivos de uma run ainda batem com o inventário dela é `verify_integrity()` do leitor da capability.
- **Saída:** `reports/semantic_fusion/report.json` é o resultado registrado. Uma reexecução é comparada campo a campo com ele; os campos que **devem** mudar estão na próxima seção.

## Reexecução no head desta PR

`s03_sensor_association.py` e `s04_fusion.py` (Point Representation e os sete braços) foram reexecutados sobre as mesmas runs de State Estimation, Geometric Mapping e Sensor Association já gravadas (lidas read-only), no commit `2aff1b5` (a base desta correção). O `reports/semantic_fusion/report.json` comparado campo a campo com o reexecutado:

- **4 857 dos 4 906 campos folha são idênticos byte a byte**; os 49 diferentes são só tempo, memória, `code_sha`/`code_version` (o commit mudou) e `schema_version` (`0.1.0` → `0.2.0`, o bump desta correção) — nenhum deles é uma grandeza de resultado;
- **as 49 identidades de resultado batem em todos os sete braços**: `support_count`, `physical_observation_count`, `inference_result_count`, `evidence_base_id`, `hypothesis_labels_id`, `hypothesis_stances_id` e o fingerprint da configuração de fusão;
- **determinismo confirmado de novo**: os 7 arquivos contratuais byte a byte iguais entre a reexecução com ordem original e com ordem embaralhada, e a repetição de inferência (SAM2 + `vp-sam2-rerun`) segue dobrando exatamente contribuições e resultados em 179 de 179 suportes;
- Point Representation muda de conteúdo só no `provenance.code_version` de cada vetor (o commit do checkout): as 189 representações, a geometria de suporte e as estatísticas são idênticas — por isso `verify_inputs.py` confere esse grupo pela identidade (`representation_count`, `representation_space_id`, `dimension`), não por um digest de arquivo, que mudaria a cada commit sem que o conteúdo geométrico mudasse.

O relatório completo dessa comparação está em `manifest.json` → `reexecution`.

## Sobre os arquivos versionados e os executados

O manifest registra, para cada arquivo, o SHA-256 do que está no Git (`sha256`) e, para o que rodou, o do arquivo local que foi executado (`executed_sha256`). São idênticos, exceto:

- os relatórios JSON, onde caminhos absolutos pessoais foram trocados por `<VALIDATION_DIR>` (a pasta de saída das etapas) e `<REPO_ROOT>` (a raiz do checkout dos insumos), por substituição de texto: o resto do arquivo é byte a byte o mesmo;
- `scripts/common.py` e `scripts/s04_fusion.py`, cujas únicas mudanças são as constantes de caminho (`MAIN`, `VAL`, a seleção lida do bundle) e a leitura do `CODE_SHA`, que antes apontavam para um checkout e um worktree fixos. Os demais drivers são idênticos ao executado.

`README.md` e `scripts/verify_inputs.py` foram escritos para o bundle e não têm contrapartida executada.

## Limitações

- Os sete runs de fusão registrados foram gravados com `schema_version` `0.1.0`; o schema do `SemanticFusionRunArtifact` está em `0.2.0` e o leitor atual **recusa** `0.1.0`. Reexecutar `s04_fusion.py` os regenera em `0.2.0`.
- A execução usa a trajetória de referência do dataset (não FAST-LIO), uma calibração MEI derivada do YAML de intrínsecos (o `SequenceArtifact` persistido não guarda o modelo MEI), um único trecho de 90 s e nenhuma run de percepção tem claims: **correção, contradição e efeito dos canais são N/A**. Ver a seção de limites em `semantic_fusion.md`.
- As entradas de percepção vêm de uma validação anterior em GPU que este bundle não reproduz; uma regeneração delas pode não ser bit a bit igual e, então, o digest muda e o resto da cadeia precisa ser refeito.

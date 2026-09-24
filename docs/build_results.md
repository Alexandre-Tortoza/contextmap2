# Resultados reais do run canônico end-to-end (corridor-02, milestone #19)

Este documento resume os resultados **reais** (não sintéticos) da primeira execução completa do
pipeline canônico do Solution 1 sobre dados reais (`corridor-02`), produzida pela milestone
[End-to-End Validation (#19)](https://github.com/Alexandre-Tortoza/contextmap2/milestone/19) e
pelas cinco rodadas de revisão real que se seguiram na PR #438. Os artifacts brutos que geraram
esses números viviam em `outputs/` (fora do controle de versão, ~39 GB) e foram removidos após
este resumo ser escrito; os scripts que os produziram e os relatórios de aceitação continuam no
repositório em `experiments/e2e-real-canonical-run-20260923/`.

## O que foi validado

Pipeline canônico completo, sobre uma sequência real (ROS 1 bag) e uma sequência de pose real
(`corridor-02-gt.txt`), do dado bruto ao `ContextMapArtifact`:

```text
ingestion (bag + pose) -> state_estimation -> geometric_mapping -> visual_perception
    -> sensor_association -> semantic_fusion -> semantic_mapping -> entity_resolution
    -> spatial_relations -> context_map
```

Backends reais usados: `ExternalPoseEstimator` (trajetória = `corridor-02-gt.txt`), SAM2.1-hiera-
tiny (region discovery), DINOv2-base + CLIP-vit-large-patch14 (features), Qwen3-VL-4B-Instruct
nf4 (interpretação semântica), políticas geométricas/conservadoras reais para
`entity_resolution`/`spatial_relations`.

## Resultado científico final (estável em toda a campanha)

- **170 entidades**, **10.852 registros de relação** no `ContextMapArtifact` final.
- Das relações: 914 (8,4%) `SUPPORTED`, 8.668 (79,9%) `UNRESOLVED`, 1.270 (11,7%) `REJECTED` --
  ~5,4 relações confirmadas por entidade, número normal para uma cena real de corredor; geração
  de candidatos é conservadora (1,4%/6,2% dos pares possíveis dentro dos raios configurados).
- Esse resultado científico foi **reproduzido de forma idêntica, bit a bit** (via comparação
  canônica de geometria real + status + estado de relação) em 3 execuções independentes
  downstream (dado o mesmo input de percepção): a cadeia original, uma re-execução manual e uma
  re-execução via `run_plan()`/`resume_plan()` real, incluindo um teste real de interrupção/
  recuperação (falha injetada depois do estágio de 25 GB de RAM `sensor_association`, retomada
  sem recomputar nada já concluído).

## As cinco rodadas de revisão real da PR #438

Cada rodada revisou o head real do código/evidência e encontrou um problema real, não hipotético:

1. **Lineage da pose auxiliar (#555)**: `StateEstimationRunArtifact` não registrava qual artifact
   produziu a trajetória quando uma pose auxiliar era mesclada. Corrigido: schema `0.2.0` agora
   nomeia `auxiliary_sequence_artifact_id`; `cross_stage.lineage_closure` passou a verificar essa
   sequência (32 assertions, 0 findings). Também corrigidos nessa rodada: `VisualPerceptionExecutor`
   perdia a evidência bruta do VLM na runtime composta (agora preservada); `ContextMapExecutor`
   não verificava se a sequência recebida batia com a que a geometria realmente usou (agora
   verifica); ordem de checagem da salvaguarda de ground-truth; nomenclatura honesta da checagem
   de compatibilidade de clock (plausibilidade, não prova).
2. **Ambiente stale**: o `code_version` gravado nos artifacts apontava para um commit 23 commits
   atrás do código real, porque o `setuptools` do venv de desenvolvimento tinha sumido e
   `pip install -e` parava de atualizar a metadata silenciosamente. Corrigido reinstalando
   `setuptools`; o gerador do relatório passou a ler `contextmap.__version__` ao vivo.
3. **`ContextMapExecutor.configuration_fingerprint` hardcoded em `None`**: um gap de contrato
   real, não só de evidência -- o `ContextMapArtifact` nunca poderia satisfazer
   `runtime.provenance_identity` não importa quão bem a auditoria checasse. Corrigido com TDD:
   agora hasheia `assembly_policy` + `up_direction`.
4. **Auditoria de provenance incompleta**: faltavam `semantic_fusion.fusion_configuration_
   fingerprint`, `PolicyRef.configuration_fingerprint` de `entity_resolution`/`spatial_relations`,
   e identidade real por backend em `visual_perception` (só a existência de um hash agregado era
   checada). Corrigido; a auditoria agora cobre os 11 artifacts reais do run sistematicamente.
5. **Comparador de reprodutibilidade incompleto**: a canonicalização de entidades/relações
   ignorava `semantic_state.status` e `RelationState`/`uncertainty_kinds` -- uma relação mudando
   de `SUPPORTED` para `UNRESOLVED` entre duas execuções ainda seria reportada como equivalente.
   Corrigido. **Essa correção revelou o achado mais importante de toda a campanha** (próxima
   seção).

## O achado real mais importante: `reproducibility.rerun_equivalence` = FAILED

Completar esse gate de acordo com o texto do cenário 1.0.4 ("repeated canonical runs ... yield
equivalent artifacts, metric reports and final map") exigiu uma segunda execução real e
independente do Visual Perception (SAM2 + DINOv2 + CLIP + Qwen3-VL-4B) sobre os mesmos 20 frames
reais -- algo que nenhuma rodada anterior tinha de fato feito (toda "re-execução" anterior
reaproveitava o único `PerceptionRunArtifact` real e só recomputava os estágios determinísticos
downstream).

Resultado real, medido:

| | run-0001 (original) | run-0002 (segunda execução independente) |
|---|---|---|
| Descoberta de região (SAM2) | baseline | **0/20 frames com diferença** -- bounding boxes, `region_kind`, `is_accepted` idênticos |
| Execuções semânticas bem-sucedidas | 76/120 | 78/120 |
| Claims | 59 | 60 |
| Scene contexts | 17 | 18 |
| Concordância exata de claims canônicas | -- | **29/90 (32,2%)** |

A descoberta de região é perfeitamente reprodutível. A taxa de sucesso da interpretação semântica
é parecida (63% vs 65%), mas o **conteúdo** das claims (hypothesis, role, category, confidence) só
concorda exatamente em 32,2% dos casos entre as duas execuções independentes, apesar de
configuração idêntica (`temperature=0.0`, decodificação gulosa).

Downstream do Visual Perception, dado o **mesmo** input de percepção, os 8 estágios seguintes
(`state_estimation` até `context_map`) são exatamente reprodutíveis (relatórios de métrica e
conteúdo final idênticos entre execuções independentes). Mas essa evidência, por si só, não fecha
a reprodutibilidade do pipeline inteiro: o requisito do cenário 1.0.4 é conjuntivo sobre o run
canônico completo, e Visual Perception é um artifact intermediário obrigatório desse mesmo run.

**O mecanismo não foi isolado.** Não-determinismo de ponto flutuante em GPU se propagando pela
geração autoregressiva é tecnicamente plausível (kernels CUDA não são determinísticos por padrão
sem configuração explícita, que este pipeline não usa), mas atribuir causalidade exigiria um
experimento controlado (seeds fixas, `torch.use_deterministic_algorithms`,
`CUBLAS_WORKSPACE_CONFIG`, hardware/driver fixos) que esta rodada não executou. Isso fica
registrado como pergunta em aberto, não como fato.

## Estado final da milestone (25 gates, cenário 1.0.4)

`unmet_required_gates(scenario, report, kinds={INVARIANT})` retorna **um** item:
`reproducibility.rerun_equivalence`.

| Status | Contagem |
|---|---|
| `passed` (real) | 16 |
| `failed` (real) | 1 -- `reproducibility.rerun_equivalence` |
| `blocked` (sem reference set anotado) | 5 |
| `not_evaluated` | 3 |

**Solution 1 não está pronta para v0.1.0 sob a política de release desta campanha.** Isso é um
resultado real e valioso, não uma falha de processo: a validação end-to-end fez exatamente o que
deveria -- encontrou uma propriedade real e antes desconhecida da configuração canônica atual.

## Próximos passos recomendados

1. Abrir uma issue de follow-up para isolar o mecanismo: não-determinismo real de execução em
   GPU, sensibilidade do parser/contrato do Qwen a variações mínimas de texto, ou definição de
   equivalência semântica canônica excessivamente estrita? Um experimento controlado (flags de
   determinismo, seeds fixas, hardware fixo) consegue distinguir essas hipóteses.
2. Dependendo da resposta: tornar a execução canônica do Visual Perception determinística, ou
   definir e versionar uma tolerância de equivalência semântica explícita para o cenário (uma
   nova versão, não uma reinterpretação silenciosa da 1.0.4).
3. Não mesclar a PR #438 nem declarar a v0.1.0 pronta citando este cenário até
   `reproducibility.rerun_equivalence` ser resolvido -- corrigido ou formalmente redefinido e
   reevidenciado.

## Onde encontrar os detalhes completos

Todo o código, scripts e relatórios de aceitação (JSON + Markdown) de cada uma das cinco rodadas
ficam versionados em `experiments/e2e-real-canonical-run-20260923/`:

- `README.md` -- histórico completo das 5 rodadas e o que cada uma superseder.
- `acceptance-report-run0005.{json,md}` -- relatório final, autoritativo.
- `scripts-run5/` -- scripts da rodada final, incluindo os dois comparadores reais
  (`14_metric_report_equivalence.py`, `15_visual_perception_content_equivalence.py`) e o gerador
  do relatório (`16_assemble_acceptance_report.py`).
- `scripts/`, `scripts-run2/`, `scripts-run3/`, `scripts-run4/` -- rodadas anteriores, mantidas
  para histórico (não citar como evidência de lineage/reprodutibilidade/provenance atual).

Os artifacts reais brutos que esses scripts produziram (sequências, PerceptionRunArtifacts,
ContextMapArtifacts, etc.) não estão mais no disco de desenvolvimento -- eram gerados sob
`outputs/`, fora do controle de versão, e foram removidos depois deste resumo ser escrito. Para
reproduzir qualquer resultado aqui descrito, os scripts em `experiments/` são o ponto de partida
real: eles documentam exatamente os artifacts de entrada (sequência bag, sequência de pose,
`PerceptionRunArtifact`) e a ordem de execução usada.

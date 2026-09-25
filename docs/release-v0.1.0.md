# Escopo congelado e checklist de aceitação do v0.1.0

> **Status: congelado** em 2026-09-25, contra o cenário de aceitação `solution-1-canonical` **1.0.5** e o relatório [`v0.1.0-release-contract-20260925`](../src/contextmap/evaluation/docs/validation/v0.1.0-release-contract-20260925.md). Depois deste congelamento, mudar qualquer versão de schema da tabela abaixo faz `tests/packaging/test_release_scope.py` falhar — de propósito: uma release não muda de contrato sem revisão explícita de escopo e de versão.

## Regra

O v0.1.0 **empacota a Solution 1 validada**; não introduz capability científica nova. Um gate end-to-end obrigatório que falhe bloqueia a release, a menos que o escopo ou a versão sejam **revistos e revalidados**. Foi exatamente o que aconteceu aqui, e a seção 3 registra como.

O produto do release é o `ContextMapArtifact` e o pacote que o produz e o lê. Visualização, busca semântica, interpretação de comandos em linguagem natural, planejamento e navegação, agentes e dashboards web são **consumidores externos e ficam fora** do v0.1.0.

## 1. Escopo congelado

| Dimensão | v0.1.0 |
| --- | --- |
| Fonte de entrada | bag ROS 1 (`Ros1BagSourceAdapter`, extra `ros1`), validada sobre o `corridor-02`. O adapter ROS 2 lê o formato pelo mesmo `rosbags`, mas **não** tem execução real registrada: suportado por contrato, não por evidência |
| Perfil de runtime canônico | `canonical/1` (`CANONICAL_PROFILE_ID`), topologia única de ponta a ponta |
| Estágios do run canônico | `ingestion`, `pose_ingestion`, `visual_perception`, `state_estimation`, `geometric_mapping`, `sensor_association`, `point_representation`, `semantic_fusion`, `semantic_mapping`, `entity_resolution`, `spatial_relations`, `context_map` — todos declarados disponíveis no catálogo e todos executados sobre dados reais |
| Estágio opcional | `point_representation` com o descritor geométrico determinístico. O runtime PTv3 existe e foi medido sobre geometria real do `corridor-02` **sem vantagem demonstrada** de discriminação de lugar sobre o descritor, e com custo maior; permanece opcional e nunca substitui o baseline por fallback |
| Backends com execução real registrada | SAM2 e SAM3 (region discovery), DINOv2, DINOv3 e CLIP (features), Qwen3-VL-4B e Florence-2 (interpretação semântica, sobre 20 frames reais, sem anotações), `ExternalPose` (trajetória), PTv3 (point representation) |
| Backends **experimentais** (sem execução real registrada) | AlphaCLIP (checkpoints de origem e licença não verificadas), Gemini (cliente `google-genai` validado só com transporte simulado, sem credencial nem consentimento para enviar frames), backend FAST-LIO (o wrapper roda no container do fornecedor; só testado com processo substituto) |
| Backend canônico com **propriedade não garantida** | a interpretação semântica Qwen3-VL roda no pipeline canônico e sua evidência chega ao mapa final, mas a **equivalência exata de rerun das claims não faz parte do contrato** — ver seção 3 |
| Política de fusão | baseline de acumulação. A política ciente de qualidade continua opcional |
| Leitor e validador leves | a instalação base (só NumPy) importa os contratos públicos, abre e valida os artifacts persistidos, inclusive o `ContextMapArtifact` |
| Extras opcionais | `ros1`, `ros2`, `vision`, `gemini` ([installation.md](installation.md)) |
| Cenário de referência validado | `corridor-02`, janela congelada de 90 s do cenário 1.0.5, mais o artifact de demonstração de #188 |
| Qualidade científica | **não há reference set anotado** para o `corridor-02`. Os cinco gates de qualidade ficam bloqueados, não pontuados. O v0.1.0 não afirma qualidade semântica, de entidades ou de relações |
| Recursos de hardware | a instalação base e a CI não usam GPU. As execuções reais usaram uma RTX 3060 8 GB e uma máquina de 39 GiB de RAM; o perfil de recursos por estágio está em [`resource-profile.json`](../experiments/e2e-real-canonical-run-20260923/resource-profile.json) |
| Publicação | wheel, sdist e `SHA256SUMS` em uma GitHub Release (pre-release, série `v0.x`); sem PyPI durante a validação ([versioning.md](versioning.md)) |

### Versões de schema e artifact

Congeladas e verificadas por `tests/packaging/test_release_scope.py`.

| Artifact ou documento | Versão |
| --- | --- |
| `SequenceArtifact` / `CalibrationSet` / provenance / diagnósticos de ingestion | 0.2.0 / 0.1.0 / 0.1.0 / 0.2.0 |
| `PerceptionRunArtifact` (pipeline, espaço de embedding, índice de features, índice de máscaras) | 0.5.0 (0.2.0, 0.1.0, 0.1.0, 0.1.0) |
| `StateEstimationRunArtifact` | 0.2.0 |
| `GeometricMapArtifact` | 0.1.0 |
| `SensorAssociationRunArtifact` | 0.2.0 |
| `PointRepresentationRunArtifact` | 0.1.0 |
| `SemanticFusionRunArtifact` | 0.2.0 |
| `SemanticEntityArtifact` (serialização de entidades) | 0.1.0 (0.1.0) |
| `EntityResolutionRunArtifact` | 0.1.0 |
| `SpatialRelationsRunArtifact` | 0.1.0 |
| **`ContextMapArtifact`** | **0.1.0** |
| Configuração de runtime: configuração, plano, run, catálogo, reuso, pedido de ingestão | 0.1.0 cada |
| Cenário de aceitação | `solution-1-canonical` **1.0.5** |

## 2. Checklist de aceitação

Estados: **feito** (com evidência), **preparado** (existe, falta só a ação externa), **bloqueado** (depende de algo fora deste escopo).

### A. Pacote, automação e higiene (milestone #20)

| # | Item | Evidência | Estado |
| --- | --- | --- | --- |
| A1 | A instalação base importa os contratos e lê/valida os artifacts, inclusive o `ContextMapArtifact`, sem ROS, Torch ou modelos | `tests/packaging/`, job `lightweight-install` (a suíte inteira contra a wheel só com NumPy) | feito |
| A2 | Os extras instalam o que anunciam | job `lightweight-install`, laço por extra (`ros1`, `ros2`, `gemini`), cada um com o smoke e os testes que ele habilita | feito para `ros1`, `ros2` e `gemini`; `vision` não é instalado na CI (Torch), e a checagem dele é a de declaração |
| A3 | O comando `contextmap` existe no pacote instalado | `[project.scripts]`, `tests/packaging/test_release_metadata.py`, smoke em venv novo | feito |
| A4 | O pacote reporta a versão da tag | smoke `--expect-version` no workflow `Release` | preparado; só a primeira release comprova |
| A5 | CI e release endurecidos | #186 | feito; o reuso em uma release real ainda não foi exercido |
| A6 | Higiene, licenças e segurança | #189, [third-party-licenses.md](third-party-licenses.md), [repository-settings.md](repository-settings.md) | feito |
| A7 | Existe `main`, que recebe a promoção `dev` -> `main` | seção 4 | **bloqueado: `main` não existe (verificado em 2026-09-25); é ação do mantenedor** |
| A8 | Documentação pública final | #184 | feito |
| A9 | Configuração canônica, manifest de referência e artifact de demonstração | #188 | feito |
| A10 | Changelog e notas de release finais | #190 | preparado; a tag e a release são do mantenedor |
| A11 | O escopo congelado é verificado por teste | `tests/packaging/test_release_scope.py` | feito |

### B. Validação end-to-end (milestone #19)

A evidência é o relatório [`v0.1.0-release-contract-20260925`](../src/contextmap/evaluation/docs/validation/v0.1.0-release-contract-20260925.md), decidido contra o cenário 1.0.5: **18 gates cumpridos com evidência real, 0 reprovados, 5 bloqueados, 3 não avaliados**, e `unmet_required_gates(kinds={INVARIANT})` vazio.

| # | Item | Issue | Estado |
| --- | --- | --- | --- |
| B1 | Cenário canônico e matriz de aceitação | #176 | feito (1.0.4 congelado; 1.0.5 é o contrato de release) |
| B2 | Execução do pipeline canônico da fonte registrada ao `ContextMapArtifact` | #177 | feito (run real; `entity_count=170`, `relation_count=10852`) |
| B3 | Linhagem entre estágios, consistência de coordenadas e preservação de evidência | #178 | feito (4 gates `cross_stage.*` cumpridos) |
| B4 | Qualidade semântica, de entidades e de relações no conjunto de referência | #179 | **fora do v0.1.0** (milestone Post-v0.1.0): não há reference set anotado; os gates ficam bloqueados, não pontuados |
| B5 | Ablações controladas de backends e canais de evidência | #180 | **fora do v0.1.0** (milestone de experimento de ablação) |
| B6 | Custo de tempo, memória, armazenamento e modelos externos | #181 | feito (`runtime.resource_reporting` cumprido) |
| B7 | Reprodutibilidade, reuso, recuperação de interrupção e relatório final | #182 | feito sob o contrato 1.0.5 — ver seção 3 |

### C. Capabilities integradas em `dev`

Todas as milestones de capability da Solution 1 (#1 a #19) estão fechadas e mescladas em `dev`: runtime e configuração, semantic mapping, entity resolution, spatial relations, schema e serialização do `ContextMapArtifact`, conjunto de referência e avaliação, e validação end-to-end.

## 3. O gate que falhou, e como o escopo foi revisto

A regra da seção "Regra" foi acionada de verdade. A campanha de aceitação da milestone #19 encontrou **um** gate invariante obrigatório reprovado, com evidência real e medida: `reproducibility.rerun_equivalence`, no cenário 1.0.4.

Duas execuções reais e independentes da configuração canônica idêntica de Visual Perception (SAM2.1-hiera-tiny + DINOv2-base + CLIP-ViT-L/14 + Qwen3-VL-4B-Instruct nf4, greedy, `temperature=0.0`) sobre os mesmos 20 frames reais concordaram em **29 de 90 claims canônicas comparadas (32,2%)**. Region discovery divergiu em **0/20 frames**, e, a partir do **mesmo `PerceptionRunArtifact`**, os oito estágios a jusante foram exatamente reprodutíveis.

O que foi feito, na ordem:

1. **A 1.0.4 permanece imutável e reprovada.** É o registro autoritativo do que aquela campanha mediu. Nenhum limiar, gate ou interpretação dela foi alterado, e seus snapshots não foram editados.
2. **A 1.0.5 foi criada como contrato de release.** Ela estreita `reproducibility.rerun_equivalence` para a propriedade que a evidência sustenta — runs repetidos **a partir do mesmo `PerceptionRunArtifact`** — e acrescenta `reproducibility.semantic_rerun_agreement`, um gate de **relatório** que registra a concordância medida com denominador explícito e sem limiar.
3. **O backend Qwen3-VL foi declarado experimental** (`ExperimentalComponent`), nomeando a propriedade não garantida, a medição com denominador e a issue de investigação.
4. **O relatório foi decidido contra a 1.0.5**, não reinterpretado a partir da 1.0.4: 24 gates carregados literalmente, dois decididos de novo sob as definições novas.
5. **A #556 continua aberta**, fora do bloqueio de release, para isolar o mecanismo.

O v0.1.0 **não afirma** que "sensores registrados -> `ContextMapArtifact`" é reprodutível de ponta a ponta. Afirma algo mais estreito e demonstrado: dado o mesmo `PerceptionRunArtifact`, o pipeline a jusante é exatamente reprodutível.

### Limite honesto desta evidência

Os artifacts reais da campanha foram **podados** de `outputs/` depois dela. O relatório do contrato de release carrega as identidades de evidência que o run-0005 registrou e **declara em suas próprias limitações** que nada nele foi re-medido nem teve hash re-verificado. O registro primário continua sendo [`acceptance-report-run0005.json`](../experiments/e2e-real-canonical-run-20260923/acceptance-report-run0005.json). Uma campanha futura que re-execute o pipeline precisará re-fixar o sujeito, porque `artifact_id` é fornecido pelo chamador e o manifesto carrega `created_at`: uma re-ingestão não reproduz o digest fixado.

## 4. Decisões e ações que dependem do mantenedor

Estado externo **verificado em 2026-09-25** pela API do GitHub, não assumido:

| Item | Estado verificado |
| --- | --- |
| Branch `main` | **não existe** (37 branches; `dev` é a padrão) |
| Proteção de branch / rulesets | **nenhuma**: 0 rulesets e `dev` não protegida (`Branch not protected`) |
| Métodos de merge | merge commit, squash e rebase **todos habilitados** |
| Descrição e tópicos do repositório | descrição vazia, 0 tópicos |
| Tags e releases | **0 tags, 0 releases** |

Ações pendentes, todas do mantenedor:

- criar a branch `main` e decidir se ela passa a ser a padrão (A7);
- criar a tag `v0.1.0` no commit validado e publicar a release (A10; nenhum workflow cria tags, e o workflow `Release` recusa uma tag que não esteja em `main`);
- aplicar as configurações documentadas em [repository-settings.md](repository-settings.md), ou atualizar aquele documento para descrever o estado real;
- confirmar `AGPL-3.0-only` e o classifier de estágio de desenvolvimento para o v0.1.0.

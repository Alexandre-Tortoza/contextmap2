# Escopo e checklist de aceitação do v0.1.0

> **Status: rascunho.** O escopo só é congelado depois do relatório de aceitação end-to-end da milestone #19 (issues #176 a #182), que ainda não existe. Este documento fixa a regra do congelamento, propõe o escopo com base no que `origin/dev` realmente contém em 2026-09-21 e lista cada item de aceitação com a evidência que o sustenta. Um item sem evidência não está pronto.

## Regra

O v0.1.0 **empacota a Solution 1 validada**; não introduz capability científica nova. Depois do congelamento, nenhuma capability, backend ou schema entra sem revisão explícita do escopo e da versão e sem revalidação. Um gate end-to-end obrigatório que falhe bloqueia a release, a menos que o escopo ou a versão sejam revistos e revalidados.

O produto do release é o `ContextMapArtifact` e o pacote que o produz e o lê. Visualização, busca semântica, interpretação de comandos em linguagem natural, planejamento e navegação, agentes e dashboards web são **consumidores externos e ficam fora** do v0.1.0.

## 1. Escopo proposto

"Estado em `origin/dev`" é o que existe no código em 2026-09-21; "proposta" é o que este rascunho sugere congelar e depende de #176 e da aprovação do mantenedor.

| Dimensão | Proposta para o v0.1.0 | Estado em `origin/dev` |
| --- | --- | --- |
| Fonte de entrada | bag ROS 1 (`Ros1BagSourceAdapter`, extra `ros1`); confirmar em #176 que o formato do `corridor-02` é esse | adapters ROS 1 e ROS 2 implementados; o ROS 2 não tem validação real registrada, então fica como suportado por contrato, sem evidência |
| Perfil de runtime canônico | `canonical/1` (`CANONICAL_PROFILE_ID`) | a runtime e o perfil estão integrados a esta branch (PR #387, ainda aberta para `dev`); os estágios `semantic_mapping`, `entity_resolution`, `spatial_relations` e `context_map` seguem declarados indisponíveis no catálogo da runtime, e os executores das capabilities reais ainda não estão ligados |
| Estágios obrigatórios | ingestion, visual_perception, state_estimation, geometric_mapping, sensor_association, semantic_fusion, semantic_mapping, entity_resolution, spatial_relations, context_map | os seis primeiros existem em `dev`; os quatro últimos dependem das milestones de semantic mapping (PR #367), entity resolution, spatial relations e context map |
| Estágio opcional | `point_representation`, desligado por padrão no perfil canônico | implementado com o descritor geométrico determinístico; o PTv3 é só fronteira; a justificativa por avaliação controlada não existe |
| Backends com evidência real registrada | SAM2/SAM3 (region discovery), DINOv2 e CLIP (features), Qwen (interpretação), `ExternalPose` (trajetória do dataset) | execuções reais de 2026-09-20 com escopos distintos, descritos nos documentos de cada adapter; não há avaliação científica comparativa comum |
| Backends experimentais (sem execução real) | DINOv3 (repositório *gated*), AlphaCLIP (checkpoints ausentes), Gemini (sem chave e sem consentimento para enviar frames), Florence-2 semântico, PTv3, backend FAST-LIO (só testado com processo substituto) | ficam fora do caminho validado até haver execução e avaliação registradas; um caminho experimental não substitui o baseline |
| Política de fusão | baseline de acumulação | a política ciente de qualidade continua opcional: nenhuma decisão foi tomada porque não há run de fusão sobre dados reais |
| Leitor e validador leves | a instalação base (só NumPy) abre e valida os artifacts persistidos | verificado para os artifacts atuais (CI `lightweight-install`); o leitor e o validador do `ContextMapArtifact` dependem das milestones de schema e serialização |
| Cenário de referência validado | `corridor-02`, em amostra definida por #176, mais o artifact de demonstração sintético de #188 | não há relatório de aceitação nem artifact de demonstração ainda |
| Recursos de hardware | a instalação base e a CI não usam GPU; as execuções reais registradas usaram uma GPU RTX 3060 | requisitos de tempo, VRAM, RAM e armazenamento dos backends reais **não foram medidos de forma sistemática**; entram com #181 |
| Publicação | wheel, sdist e `SHA256SUMS` em uma GitHub Release (pre-release, série `v0.x`) | sem publicação no PyPI durante a fase de validação ([versioning.md](versioning.md)) |

### Versões de schema e artifact

Derivadas das constantes do código em `origin/dev`. O congelamento registra a versão de cada uma; mudar uma delas depois exige revisão do escopo.

| Artifact ou documento | Versão de schema |
| --- | --- |
| `SequenceArtifact` | 0.2.0 |
| `CalibrationSet` / provenance da sequência / diagnósticos de ingestion | 0.1.0 / 0.1.0 / 0.2.0 |
| `PerceptionRunArtifact` (pipeline, espaço de embedding, índice de features) | 0.4.0 (0.2.0, 0.1.0, 0.1.0) |
| `StateEstimationRunArtifact` | 0.1.0 |
| `GeometricMapArtifact` | 0.1.0 |
| `SensorAssociationRunArtifact` | 0.1.0 |
| `PointRepresentationRunArtifact` | 0.1.0 |
| `SemanticFusionRunArtifact` | 0.1.0 |
| `ContextMapArtifact` | **pendente** (milestones de schema e serialização) |
| Configuração de runtime (`canonical/1`): configuração, plano, run, catálogo, reuso, pedido de ingestão | 0.1.0 cada (`CONFIG_SCHEMA_VERSION`, `PLAN_SCHEMA_VERSION`, `RUN_SCHEMA_VERSION`, `CATALOG_SCHEMA_VERSION`, `REUSE_SCHEMA_VERSION`, `INGESTION_REQUEST_SCHEMA_VERSION`) |

## 2. Checklist de aceitação

Estados: **feito** (com evidência), **preparado** (existe, falta comprovação ou integração), **pendente**, **bloqueado** (depende de algo que este PR não controla).

### A. Pacote, automação e higiene (milestone #20)

| # | Item obrigatório | Evidência | Estado |
| --- | --- | --- | --- |
| A1 | A instalação base importa os contratos e lê os artifacts atuais sem ROS, Torch ou modelos | `tests/packaging/`, jobs `package` e `lightweight-install` da CI ([installation.md](installation.md)) | feito, para os artifacts atuais |
| A2 | Os extras instalam o que anunciam | jobs `lightweight-install` (`ros1`, `ros2`); resolução de `vision` por `pip install --dry-run` | feito; `vision` só foi verificado por resolução, não por instalação completa |
| A3 | O comando `contextmap` existe no pacote instalado | `[project.scripts]` no `pyproject.toml`, `tests/packaging/test_release_metadata.py` e o smoke em venv novo (`contextmap --help` e `--version` saem com 0; 140 módulos importam sem extras) | feito; a runtime chega a `dev` pela PR #387 |
| A4 | O pacote reporta a versão da tag | smoke `--expect-version` no workflow `Release` | preparado; só é comprovado pela primeira release |
| A5 | CI e release endurecidos | #186 (PR #396) | feito; falta comprovar o reuso do `ci.yml` em uma release real |
| A6 | Higiene, licenças e segurança | #189 (PR #403), [third-party-licenses.md](third-party-licenses.md), [repository-settings.md](repository-settings.md) | feito, com decisões do mantenedor pendentes |
| A7 | Existe `main`, que recebe a promoção `dev` -> `main` | seção "Estado verificado" de [repository-settings.md](repository-settings.md) | **bloqueado: `main` não existe no remoto; é ação do mantenedor** |
| A8 | Documentação pública final | #184 | bloqueado até o pipeline final |
| A9 | Configuração canônica, manifest de referência e artifact de demonstração validado | #188 | bloqueado até schema, writer e runtime |
| A10 | Changelog e notas de release validados | #190 | preparado como rascunho; a tag e a release são do mantenedor |

### B. Validação end-to-end (milestone #19)

Cada item exige o relatório ou teste correspondente referenciado aqui antes de ser marcado como feito.

| # | Item obrigatório | Issue | Estado |
| --- | --- | --- | --- |
| B1 | Cenário canônico e matriz de aceitação | #176 | pendente |
| B2 | Execução do pipeline canônico da fonte registrada ao `ContextMapArtifact` | #177 | bloqueado (executores reais e capabilities finais) |
| B3 | Linhagem entre estágios, consistência de coordenadas e preservação de evidência | #178 | pendente |
| B4 | Qualidade semântica, de entidades e de relações no conjunto de referência | #179 | bloqueado (anotações de referência, PR #377) |
| B5 | Ablações controladas de backends e canais de evidência | #180 | pendente |
| B6 | Custo de tempo, memória, armazenamento e modelos externos | #181 | pendente |
| B7 | Reprodutibilidade, reuso, recuperação de interrupção e relatório final de aceitação | #182 | pendente |

### C. Capabilities que o v0.1.0 precisa ter integrado em `dev`

Cada linha exige a PR mesclada e a evidência de aceitação da própria milestone.

| # | Capability | Situação em 2026-09-21 |
| --- | --- | --- |
| C1 | Runtime e configuração | PR #387 aberta; integrada a esta branch, com o CLI empacotado e testado |
| C2 | Semantic mapping e modelo de entidades | PR #367 aberta |
| C3 | Conjunto de referência e avaliação | PR #377 aberta |
| C4 | Entity resolution, spatial relations | em desenvolvimento em branches de milestone |
| C5 | Schema e serialização do `ContextMapArtifact` | em desenvolvimento em branches de milestone |
| C6 | Execuções reais de feature extraction, interpretação semântica, state estimation, point representation e fusão | em milestones separadas; o que ficar sem execução real entra como experimental |

## 3. Gate

- Todo item obrigatório referencia uma issue, teste, relatório ou artifact; um item sem link não é marcado como feito.
- O relatório de aceitação de #182 é a evidência de B1 a B7. Se algum gate obrigatório falhar, a release é bloqueada até o escopo ou a versão serem revistos e revalidados.
- No congelamento, as versões da tabela de schemas passam a constantes de um teste que falha quando um schema muda; antes disso ele bloquearia as milestones que ainda estão alterando contratos, por isso não existe neste rascunho.
- O commit tagueado é o commit validado: só se cria a tag a partir de `main`, depois da promoção normal ([versioning.md](versioning.md)).

## 4. Decisões que dependem do mantenedor

- Criar a branch `main` e decidir se ela passa a ser a padrão (A7).
- Confirmar os backends obrigatórios da tabela de escopo e o que fica experimental (DINOv3, AlphaCLIP, Gemini, Florence-2 semântico, PTv3, FAST-LIO).
- Confirmar `AGPL-3.0-only` (em vez de `-or-later`) e o classifier de desenvolvimento (`Pre-Alpha` ou `Alpha`) para o v0.1.0.
- Criar a tag `v0.1.0` e publicar a release: nenhum workflow cria tags.

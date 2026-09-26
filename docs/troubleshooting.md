# Diagnóstico de falhas comuns

Este documento cobre as falhas que aparecem de verdade ao rodar o pipeline canônico: calibração, backends de modelo, recursos e publicação. Ele descreve o comportamento **implementado**, com os erros que o código realmente levanta.

Um princípio vale para tudo abaixo: o ContextMap2 **não tem fallback silencioso**. Um backend indisponível é recusado com uma mensagem explícita, nunca substituído por outro; um estágio sem executor é reportado no preflight, nunca preenchido com um substituto. Se algo parece ter "funcionado mesmo assim", provavelmente você não rodou o que pensa ter rodado.

## Calibração e geometria

| Sintoma | Causa provável | O que fazer |
| --- | --- | --- |
| `CalibrationError` ao construir a projeção de uma câmera | a entrada de calibração não descreve um modelo de câmera projetável: intrínsecos ausentes, `camera_model=None`, ou parâmetros incoerentes com o modelo declarado | `camera_projection_for` recusa **antes** de qualquer projeção, de propósito. Confira se o `SequenceArtifact` que você está usando carrega o modelo de câmera; um artifact antigo pode ter sido ingerido sem ele |
| `AssociationInputError: the trajectory is expressed in frame X but the map is in frame Y` | o mapa e a trajetória estão em frames diferentes | não force: um deles foi construído com outra convenção. Reconstrua o mapa com a trajetória correta |
| `AssociationInputError: the map was built from trajectory A but the association uses trajectory B` | o run de associação aponta para uma trajetória que não é a que gerou o mapa | selecione o run de state estimation que o mapa cita na sua linhagem |
| `AssociationInputError: the map comes from sequence A but the trajectory from B` | linhagem cruzada entre sequências diferentes | o mesmo: a linhagem é verificada porque associar sobre entradas incoerentes produz um mapa plausível e errado |
| Nada visível na imagem (`NOTHING_VISIBLE_IN_IMAGE`) | extrínseco com erro, timestamp desalinhado, ou a câmera aponta para fora do mapa | os diagnósticos por frame trazem `pose_ref.time_delta_ns`, `map_window_offset_ns` e o resumo de profundidade dos visíveis. Um extrínseco com poucos graus de erro é detectável pelo resíduo de reprojeção, se você tiver correspondências de referência confiáveis |
| Resíduo de reprojeção alto | calibração desalinhada ou pose deslocada no tempo | `time_offset_sweep()` re-projeta com a pose buscada em instantes deslocados e **só relata**: ela nunca altera calibração nem timestamp. Agir sobre o resultado é decisão separada e explícita |

Uma nota de honestidade: o pipeline não tem otimização automática de calibração, e a correção da associação sobre o `corridor-02` **não foi medida**, porque não existem correspondências 3D↔pixel de referência para ele.

## Backends de modelo e dependências opcionais

A instalação base depende só de NumPy. Todo backend pesado está atrás de um extra ou de um runtime externo, e cada um falha com um erro que **nomeia o que falta**.

| Erro | O que instalar |
| --- | --- |
| `DinoV2DependencyError`, `DinoV3DependencyError`, `ClipDependencyError` — "requires torch, transformers, and Pillow (plus torchvision with transformers 5.x)" | `pip install 'contextmap[vision]'` |
| `GeminiDependencyError` — "Gemini requires the google-genai package" | `pip install 'contextmap[gemini]'`. O `GeminiSemanticInterpreter` também aceita qualquer cliente pelo seu `Protocol`, e nesse caminho não há dependência alguma |
| `AlphaClipDependencyError` | `torch`, `Pillow` e o módulo `alpha_clip`, que **não está no PyPI**: vem do repositório `SunzeY/AlphaCLIP`. A origem e a licença dos checkpoints não foram verificadas |
| `ModuleNotFoundError: sam2` | SAM 2 vem do repositório `facebookresearch/sam2`, não do PyPI |
| `ModuleNotFoundError: rosbags` ao ler um bag | `pip install 'contextmap[ros1]'` (ou `[ros2]`; os dois instalam o mesmo `rosbags` e leem os dois formatos sem instalar o ROS) |
| Erro ao carregar DINOv3 ou SAM 3 | são repositórios *gated* no Hugging Face: o acesso é seu, e precisa estar autenticado |
| Runtime de PTv3 ausente | exige Pointcept, spconv e torch-scatter, que não existem no PyPI como um pacote único |
| `rospy` / `nav_msgs` no wrapper do FAST-LIO | vêm da distribuição do ROS **dentro do container** do FAST-LIO, não do PyPI. O wrapper é deliberadamente independente e nem importa `contextmap` |

Detalhes e verificações de cada extra: [installation.md](installation.md). Licenças e origens: [third-party-licenses.md](third-party-licenses.md).

## Recursos: memória, VRAM e disco

Os números abaixo são de **uma** máquina (16 CPUs, 39 GiB de RAM, RTX 3060 8 GB) e de **uma** amostra. Não são promessa de desempenho; são ordem de grandeza para dimensionar.

**RAM em Sensor Association.** Era o maior consumidor do pipeline: 24,96 GB de pico (~64% daquela máquina) antes de #562/#563, com risco real de OOM em máquinas menores. Hoje o estágio seleciona candidatos por frame antes de projetar e grava cada frame em fluxo. Se você ainda vê pico alto:

- confirme que a política de candidatos tem um alcance configurado (`policies.association_max_range_m`); com `None`, o estágio avalia o mapa inteiro a cada frame, que é o comportamento antigo;
- o pico escala com o **maior frame** e com as páginas residentes do mapa, não com o número de frames. Se ele cresce com a duração do bag, algo está retendo por frame;
- a consulta espacial ainda é o gargalo remanescente, e o dono dele é Geometric Mapping (#94): o índice é o *bounding box por scan*, e no mapa medido 56% do payload é varrido por frame.

**VRAM.** A configuração canônica cabe em 8 GB com o Qwen3-VL-4B em nf4. Trocar a quantização, o modelo ou o tamanho do batch muda isso; o pipeline não reduz precisão sozinho para caber.

**Disco.** O run canônico completo grava ~1,3 GB, dominado por Geometric Mapping (~1,05 GB). `outputs/`, `runs/`, `workspace/` e `artifacts/` são ignorados pelo git de propósito: dados de pesquisa nunca entram num release.

## Orquestração e runtime

| Sintoma | Causa | O que fazer |
| --- | --- | --- |
| Preflight recusa com `no executor is registered for it` | o estágio não pôde ser composto da configuração | é informação, não defeito: `compose_executors()` deixa de fora um estágio cujos pontos de variação não estão todos selecionados, cujo backend recusa os próprios parâmetros, ou que não tem executor. Veja a tabela de estado em [PIPELINE.md](PIPELINE.md) |
| `ingestion` não aparece nos executores compostos | `IngestionStageExecutor` precisa de um `IngestionRequest` concreto (caminho da fonte, tópicos, tolerância de sincronização), que é entrada da invocação e não valor de configuração | rode `contextmap ingest` primeiro e passe o artifact publicado para `run`/`stage` |
| `semantic_mapping` não compõe | ele também exige `semantic_map_id` e `code_digest`, que não são valor de configuração | forneça os dois explicitamente |
| `point_representation` não compõe | ainda não há executor real (depende de backend com modelo/GPU) | forneça o artifact ou injete o executor |
| `semantic_fusion` composto mas a política quality-aware não roda | o executor global suporta `baseline-evidence-accumulation-v1` | a política existe na capability; ela não é executada por esse executor, e não há substituição silenciosa |
| `ConfigurationError` ao resolver a configuração | perfil desconhecido, arquivo ilegível, ou combinação incompatível | `contextmap inspect config --json` lista cada problema com o caminho exato no documento. Nada é lido do ambiente |

## Leitura e validação de artifacts

| Achado ou erro | Significado |
| --- | --- |
| `dependency.required_missing` (validação **inválida**) | o `ContextMapArtifact` foi separado dos artifacts de origem que ele cita. Um mapa **referencia** geometria em vez de copiá-la, e três dependências são obrigatórias. Distribua o mapa com a linhagem, como faz [`examples/v0.1.0/demo/`](../examples/v0.1.0/demo/), ou passe `dependency_paths` |
| `dependency.optional_missing` (**aviso**) | a sequência, o run de percepção ou o mapa semântico não foram localizados. São opcionais; o mapa continua válido |
| `dependency.upstream_damaged` | o artifact de origem foi localizado mas seus arquivos não conferem: faltando, truncado ou com hash diferente |
| `ArtifactExistsError` | o diretório de saída já existe. Um artifact finalizado é imutável: gere outro run com linhagem correta, nunca edite o anterior |
| `UnsupportedFormatVersionError`, `UnsupportedArtifactSchemaError` | o artifact é de uma versão de schema que este código não suporta. Durante o `v0.x` não há camada de compatibilidade, de propósito |
| Validação lenta | `ValidationLevel.FULL` lê todos os bytes e reconstrói os índices. `STRUCTURAL` faz um `stat` por arquivo e não verifica conteúdo |

## Publicação do release

| Sintoma | Causa |
| --- | --- |
| O workflow `Release` recusa a tag | ele exige que a tag esteja em `main`, com a CI verde e a versão da wheel igual à da tag. Hoje **`main` não existe** no remoto ([repository-settings.md](repository-settings.md)) |
| O gate recusa as notas | `CHANGELOG.md` precisa da entrada datada `## [X.Y.Z] - AAAA-MM-DD` e `docs/releases/vX.Y.Z.md` não pode continuar marcado como rascunho |
| `__version__` reporta `0.0.0` | o build não tinha metadata de git: o `setuptools-scm` caiu no fallback. Construa de um clone com histórico |

## Reprodutibilidade semântica

Se duas execuções da mesma configuração canônica produzem claims diferentes, **isso é conhecido e medido**, não um defeito da sua instalação: o backend Qwen3-VL é experimental no v0.1.0 e duas execuções reais independentes concordaram em 29/90 (32,2%) das claims canônicas. Region discovery é reprodutível (0/20 frames divergentes) e, dado o **mesmo** `PerceptionRunArtifact`, os estágios a jusante são exatamente reprodutíveis. O mecanismo não está isolado; a investigação é a [#556](https://github.com/Alexandre-Tortoza/contextmap2/issues/556). Ver [releases/v0.1.0.md](releases/v0.1.0.md).

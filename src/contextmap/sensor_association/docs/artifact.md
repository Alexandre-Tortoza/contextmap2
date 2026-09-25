# SensorAssociationRunArtifact e serviço de associação

Este documento descreve `src/contextmap/sensor_association/service.py`, `run_artifact.py` e `debug_evidence.py`. As regras gerais de artifacts (imutabilidade, atomicidade, inventário, índice de run) estão em [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md) e são implementadas uma única vez em `contextmap.shared.run_directory`.

## O serviço

`SensorAssociationService().run(request)` compõe os passos da capability para uma execução inteira: para cada frame de câmera projeta o mapa (cadeia de pose e de imagem), resolve visibilidade, associa as máscaras congeladas, monta os `SpatialObservation` e sua qualidade, amostra os canais de features densas declarados e diagnostica o frame. Um frame cuja pose a política de lookup rejeita é **reportado** em `outcome.rejected`, não escondido, e a execução continua.

O serviço não tem regra científica própria: cada passo vive no seu módulo, e o runtime fornece as entradas concretas.

### Canais de features densas

Mapas densos são declarados como **canais** (`DenseChannel`: identidade e política de interpolação). Um mapa nativo e um mapa melhorado por `FeatureResolutionEnhancement` são dois canais de evidência **distintos**; nunca são combinados automaticamente. Cada frame deve trazer o `DenseFeatureMap` de todo canal declarado, e identidades de canal ou de frame repetidas são recusadas.

A `configuration_fingerprint` da execução é um hash da política de oclusão, da política de pose, das tolerâncias, dos canais e das versões das definições; ela entra em cada `SpatialObservation`.

## Por que existe o artifact

Quando uma associação sai errada, é preciso descobrir se a causa está no alcance, na visibilidade, na amostragem espacial, na calibração ou no tempo. O run persiste as observações espaciais com os índices e a linhagem exata, mais métricas e diagnósticos legíveis por máquina, e **abre sem ROS, sem modelos e sem NumPy**.

## Layout

## Escrita em streaming

O run é persistido **frame a frame**, não a partir de um resultado inteiro em memória. `SensorAssociationRunWriter.transaction()` abre a transação, que é um `FrameSink`:

```python
with writer.transaction() as run:
    outcome = SensorAssociationService().run(request, sink=run)
    manifest = run.finalize(outcome)
```

O serviço projeta um frame, resolve, associa, mede, diagnostica, entrega ao sink e **solta** suas referências; a transação acrescenta o payload do frame a streams já abertos (`AtomicRunDirectory.open_binary`, que hasheia enquanto escreve) e não retém **nada** do frame — nem array, nem observação, nem projeção. O pico é `estado estático do mapa + um frame de candidatos/projeção/visibilidade + buffers do writer`. Antes, a associação retinha ~1,17 GB por frame até o fim do run (#563).

A entrada também é streamada: `SensorAssociationRequest.frames` é um `Iterable` consumido **uma vez**, com validação por frame, então um chamador que produz um frame por vez nunca mantém todos os payloads de imagem vivos — o que custava 2,19 GB no corridor-02.

**O termo linear que sobra, nomeado.** O pico **não** é estritamente constante no número de frames, e não se deve afirmar `O(1)`. A transação acumula um registro de tempo por frame, porque o #562 pede os tempos de consulta e de projeção por frame e `metrics/runtime.json` é um documento único, escrito só quando o chamador mediu. São quatro primitivos, ~567 B, ou seja **menos de 1,7 MB em 3.096 frames** — quatro ordens de magnitude abaixo do pico de 4,3 GB do run real de 350 frames. O serviço também guarda o conjunto de observações já vistas (~66 B por frame) para recusar um frame repetido em um único passe. A regra que esses termos precisam continuar obedecendo é serem **escalares**, nunca um frame ou um array.

`SensorAssociationOutcome` é, por isso, a identidade, as políticas, os fingerprints, os frames rejeitados e `frame_count` do run — não os frames. Um consumidor lê os frames do artifact, que é onde eles estão.

Sair do `with` sem um `finalize()` bem-sucedido — normalmente ou por exceção — descarta tudo o que foi escrito: um run interrompido nunca deixa artifact publicável. `finalize()` recusa um `outcome` cujo `frame_count` não seja o número de frames que a transação de fato persistiu.

O writer grava o artifact **exatamente** no `output_dir` que o chamador entrega; ele não calcula caminho, não aloca índice e não mantém registro. No runtime, `output_dir` é `<workspace>/<dataset>/<run>/sensor_association/` ([`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md)).

```text
<output_dir>/
├── README.md
├── manifest.json
├── outputs/                               # contratual
│   ├── spatial-observations.jsonl         # um SpatialObservation por linha
│   ├── observation-index.jsonl            # id, frame, região, offsets em observações e qualidade
│   ├── geometry-support.u32               # região → geometria, tabela colunar de uint32
│   ├── observation-quality.jsonl          # ObservationQuality por observação
│   ├── projection-records.jsonl           # por frame: câmera, pose, extrínseco, cadeia de imagem
│   ├── visibility-records.jsonl           # por frame: política, contagens, pertencimento, regiões
│   ├── dense-feature-associations.jsonl   # por frame e canal: proveniência e offsets (se houver canais)
│   └── dense-feature-cells.bin            # índices e pesos das células (se houver canais)
├── metrics/                               # contratual
│   ├── frame-diagnostics.jsonl            # FrameDiagnostics por frame, com os achados
│   ├── summary.json                       # agregados e frames rejeitados
│   └── runtime.json                       # somente quando o tempo foi medido
└── debug/                                 # nunca contratual
```

Não existem `config.yaml`, `lineage.json`, `environment.json` nem `events.jsonl` separados: a configuração efetiva e a linhagem ficam no `manifest.json`, e os achados, em `metrics/frame-diagnostics.jsonl`. Criar arquivos sem produtor real violaria YAGNI, o critério dos outros artifacts.

## Tabelas compactas

O run não repete XYZ nem vetores de embedding:

- **região → geometria**: `geometry-support.u32` guarda, por observação, as posições ordenadas da geometria como `uint32` little-endian; a identidade do mapa é posicional (`geometry_id_for`), então a referência se reconstrói sem guardá-la. `spatial-observations.jsonl` guarda só `support: {offset, count}`. O leitor devolve o `SpatialObservation` completo, revalidado pelo contrato;
- **geometria → regiões**: derivada da tabela anterior por `regions_of(frame, referência)`, que preserva toda região sobreposta; não há uma segunda cópia;
- **associação densa**: por frame e canal, o índice JSON traz a proveniência e o offset, e `dense-feature-cells.bin` traz cinco seções sequenciais: pontos elegíveis, amostrados, linhas, colunas e pesos das células. Nenhum vetor de feature é persistido; ele é lido do payload do artefato de percepção quando preciso.

O leitor decodifica esses arquivos com a biblioteca padrão (`array`, `json`).

## `manifest.json`

Identifica o run, o que ele consumiu e quem o produziu: `run_id`, `run_index`, `sequence_name`, `sequence_artifact_id`, `selection_id`, `geometric_map_id`, `trajectory_id`, `state_estimation_run_id`, `perception_run_ids`, `calibration_identity`, a política de candidatos e a de visibilidade (cada uma com o seu fingerprint), a política de pertencimento, as versões das definições, a política de pose, as tolerâncias, `configuration_fingerprint`, `code_version`, contagens (frames, rejeitados, com falha, observações), `finding_counts`, `debug_level`, `schema_version`, `created_at` e o `file_inventory` (tamanho e SHA-256 de cada arquivo contratual, sem o manifest, o README e o `debug/`).

`dense_channels` lista cada canal com sua interpolação e as **fontes de features** exatas que consumiu: artefato de origem, espaço de embedding, transformação e fingerprint da geometria de amostragem, extrator e, se houver, a melhoria de resolução (backend, espaços de entrada e saída e grades), com o número de frames de cada fonte. A proveniência de features nunca é inferida do nome de um arquivo.

## Nativo e melhorado

Uma execução nativa e uma melhorada podem compartilhar os mesmos artifacts de geometria, trajetória, calibração e percepção e continuar **identificáveis de forma independente**: o `run_id`, a `configuration_fingerprint`, o rótulo do canal no diretório e as `dense_channels` do manifest as distinguem, e o `geometric_map_id`, o `sequence_artifact_id`, a `calibration_identity` e os `perception_run_ids` são os mesmos.

## Acesso

`SensorAssociationRunReader` abre um run somente pelo seu diretório:

- `observation(id)` lê uma observação sem carregar as demais; `observations()` itera; `observations_of_frame`;
- `quality(id)`, `geometry_support(id)`, `regions_of(frame, referência)`;
- `dense_association(frame, canal)`: índices e pesos como `array` da biblioteca padrão, mais a proveniência;
- `read_record`/`read_records` leem JSON e JSONL de `outputs/` e `metrics/` e **recusam** `debug/`, o manifest e qualquer outro caminho, para que nenhum estágio a jusante dependa dele por engano.

## Níveis de debug

| Nível | Conteúdo em `debug/` |
| --- | --- |
| `none` | nada; os outputs contratuais, a linhagem e as métricas continuam completos |
| `standard` | por frame, `samples.csv` (estado, pixel preparado, profundidade, apoio e regiões de cada ponto) e `distributions.json` (contagens por estado; profundidade e histograma dos associados; distribuição por região da imagem em 3 × 3; densidade de suporte por região) |
| `full` | acrescenta `overlay.png` (projeção colorida por estado, gerada só com a biblioteca padrão), `dense-sampling-<canal>.csv` (coordenadas de amostragem) e `feature-sources.json` (a fonte exata de cada canal e frame) |

Arquivos de debug são escritos mas nunca entram no inventário, então removê-los não invalida o run, e nada a jusante pode depender deles.

## Integridade, imutabilidade e identidade

- a escrita acontece em um diretório temporário e o run só aparece no caminho final depois de a checagem de inventário passar; uma escrita interrompida não pode parecer um run válido;
- um run finalizado nunca é sobrescrito: o writer recusa um `output_dir` que já exista, e reexecutar grava em outro diretório;
- o writer confere que a geometria de cada observação coincide com o seu pertencimento antes de persistir; uma observação inconsistente nunca é gravada;
- um run interrompido no meio de um frame não publica nada, e `finalize()` recusa um `outcome` que conte outros frames que os que chegaram à transação;
- `verify_integrity()` detecta arquivo ausente, tamanho diferente e hash diferente; um schema desconhecido levanta `RunArtifactError` e um diretório sem manifest, `IncompleteRunArtifactError`.

O `schema_version` corrente é **`0.2.0`**, o schema de seleção de candidatos (#562): o manifest ganhou `candidate_policy` (obrigatório), cada registro de projeção ganhou `candidates` e `stage_counts`, o `point_count` único **saiu** (ele significaria o tamanho do mapa no `0.1.0` e a população avaliada aqui, e um leitor não teria como distinguir — os dois conceitos agora são `candidates.map_point_count` e `candidates.candidate_count`), e `geometry-support.u32` e os `eligible_indices` densos guardam índices globais de geometria explicitamente, não linhas de candidatos que por acaso coincidiam com eles. Um artifact `0.1.0` é **recusado** com erro claro, não migrado: em `v0.x` um run é reexecutado, nunca reescrito.

`run_id` e `run_index` são entregues pelo chamador e gravados como recebidos; o writer nunca os aloca. O `run_index` é um ordinal legível, mas não substitui identidade nem hash.

## Como é verificado

Layout e linhagem no manifest; ida e volta das observações com a geometria resolvida da tabela colunar; leitura de uma observação por identidade; o índice geometria → regiões com sobreposição; ida e volta da qualidade e das associações densas (índices e pesos); os registros por frame; o resumo com frames rejeitados; nativo e melhorado com artifacts compartilhados e identidades independentes; imutabilidade, identidade gravada como recebida e diretório de saída exato, sem registro; escrita interrompida sem run visível; integridade (ausente, tamanho, hash), schema e manifest; recusa de `debug/` a jusante; cada nível de debug, inclusive a decodificação do PNG; e a abertura do artifact em um subprocesso sem NumPy, ROS ou bibliotecas de modelo.

# `PointRepresentationRunArtifact`

Este documento descreve `src/contextmap/point_representation/run_artifact.py`.

Um run persiste as representações que um encoder produziu sobre um mapa geométrico, com a linhagem, as métricas e os suportes que falharam. É **imutável**, **auto-descritivo** e abre sem NumPy, biblioteca de modelo nem o próprio mapa geométrico. Uma representação isolada, ou só o seu vetor, pode ser lida por identidade ou pela geometria à qual está ancorada, sem carregar as demais nem qualquer dado de debug.

## Layout

O writer grava o artifact **exatamente** no `output_dir` que o chamador entrega; ele não calcula caminho, não aloca índice e não mantém registro. No runtime, `output_dir` é `<workspace>/<dataset>/<run>/point_representation/` ([`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md)).

```text
<output_dir>/
├── manifest.json                       # identidade, linhagem e inventário (tamanho + sha256)
├── README.md
├── outputs/
│   ├── representations.jsonl           # PointRepresentation canônica, uma por linha
│   ├── representation-index.jsonl      # representation_id → deslocamento e tamanho na linha
│   ├── representation-spaces.json      # o RepresentationSpace do run + fingerprint
│   ├── geometry-representation-index.jsonl   # geometria → representação ancorada nela
│   ├── failed-supports.jsonl           # suportes que não produziram representação
│   └── payloads/vectors.f32|f64        # vetores, uma linha de tamanho fixo por representação
├── metrics/
│   ├── counts.json  support-size.json  norms.json  runtime.json
└── debug/                              # somente standard/full; nunca inventariado
```

`manifest.json` traz a linhagem e o inventário dos arquivos contratuais, então não há `config.yaml`, `lineage.json`, `environment.json` nem `events.jsonl` separados: ainda não existem produtores reais para eles, e criá-los vazios violaria YAGNI (`docs/ARTIFACTS.md`).

## Payload

Os vetores ficam em um arquivo binário de linhas de tamanho fixo (`dimension × itemsize` bytes, **little-endian** explícito, `float32` ou `float64` conforme o espaço), e cada representação guarda `payload_reference = "outputs/payloads/vectors.f32#<linha>"`. O leitor calcula o deslocamento, lê só aquela linha e valida que a referência aponta para o payload do próprio run e que ele não está truncado. O hash do arquivo está no inventário. O escritor **recusa** um valor que não cabe no `dtype` (em vez de gravar infinito), um vetor com valor não finito e um de dimensão errada. Componentes indefinidos ficam com o placeholder zero e listados em `undefined_components`.

## Linhagem (manifest)

- **Geometria:** `geometric_map_id` (o `GeometricMapArtifact` consumido), frame, contagem de pontos, `sequence_artifact_id`, seleção e trajetória do mapa. O mapa **não é duplicado**: só referências e linhagem (um teste garante que nenhum arquivo contém `coordinates_m`).
- **Seleção de centros:** `center_selection_id`, o hash dos centros pedidos, independente da ordem.
- **Suporte:** `support_policy` (raio ou k, teto, e a preparação de coordenadas) e o `RepresentationSpace` completo, com o fingerprint.
- **Encoder:** backend, versão, fingerprint da configuração e, se aprendido, o hash do checkpoint. Nunca segredos.
- **Código:** `code_version`.
- **Contexto de associação (opcional):** `association_context_id`, a identidade de um run de associação usado **explicitamente** como seleção/contexto. Nunca é fusão oculta de features, e é `None` quando não há.
- **Resultado:** contagens de representações, parciais e falhas por motivo, e o inventário com o hash de cada arquivo contratual.

## Falhas explícitas e run completo

Cada suporte que não produziu representação vai para `failed-supports.jsonl` com o suporte, o motivo e o detalhe do encoder; um run só de falhas é um run válido e explícito. O escritor recebe as `RepresentationMetrics` do serviço em `finalize` e **recusa** persistir um run cujos resultados não batem com elas, então um fluxo interrompido nunca é gravado como completo. Um centro só pode ter um resultado, e uma representação de outro mapa, espaço ou encoder é rejeitada.

## Integridade, imutabilidade e identidade

- O inventário detecta arquivo ausente, alterado ou de tamanho diferente (`verify_integrity`); um schema desconhecido é recusado.
- A escrita usa `shared.AtomicRunDirectory(output_dir)`: um run interrompido nunca parece finalizado e não deixa nada para trás, e um run finalizado nunca é sobrescrito: o writer recusa um `output_dir` que já exista, e reexecutar grava em outro diretório.
- `run_id` e `run_index` são entregues pelo chamador e gravados como recebidos; o writer nunca os aloca. O `run_index` é um ordinal legível, mas não substitui identidade nem hash.

## Níveis de debug

`debug/` nunca é inventariado nem contratual: removê-lo não invalida o run e `read_record` só lê `outputs/` e `metrics/`.

- `none`: só saídas contratuais, linhagem e métricas;
- `standard`: `support-traces.jsonl` (até 20 suportes espaçados, com estatísticas de distância) e `representation-norms.csv` (uma norma por representação);
- `full`: além disso, `component-statistics.json` (mínimo, máximo e média por componente sobre os valores definidos).

## Métricas

`counts.json` (pedidos, representados, parciais, falhas por motivo), `support-size.json` e `norms.json` (contagem, mínimo, mediana, p95 e máximo; a norma usa só os componentes definidos, e `non_finite_values` é sempre 0 porque um vetor não finito nunca é armazenado) e `runtime.json` (tempo de extração de suporte e de codificação do serviço, e os números específicos do backend, como o pico de memória de GPU, à parte de qualquer medida de qualidade).

## Decisões e limites

- **Sem `support-index` invertido.** A issue sugeria um índice de suporte; cada representação já carrega o seu suporte exato, alcançável por identidade pelo índice de representações. Um índice invertido (geometria → representações que ela suportou) custaria armazenamento proporcional a `N × tamanho do suporte` sem um consumidor hoje, então não foi materializado.
- **O suporte domina o tamanho.** Medição ad hoc (não versionada), 1 985 representações do descritor sobre um corredor sintético de 100 mil pontos (raio de 0,5 m, mediana de 336 membros): 21,8 MB no total, ou 10,7 KiB por representação, dos quais `representations.jsonl` é 20,7 MB (cada linha lista os ids dos ~336 membros) e o payload de vetores, só 217 KiB. Por isso o suporte é escrito de forma compacta (o mapa uma vez, os membros por `geometry_id`). Um run com todos os pontos de um mapa de 100 mil pontos pesaria cerca de 1 GB; escolher centros por subamostragem é o uso esperado.
- **O escritor acumula em memória.** `shared.AtomicRunDirectory` grava arquivos inteiros, então o tamanho de um run é limitado pela memória (23,8 MB de pico rastreado para os 1 985 acima). Uma escrita em streaming exigiria estender o módulo compartilhado; não foi feita aqui.
- **Tempo** (mesma medição, sem rastreamento, Python 3.14, um único thread): `represent` + `add` levou 3,7 s para os 1 985 centros (1,9 ms por representação), `finalize` 56 ms, `verify_integrity` 13 ms e `vector()` 0,27 ms por chamada (inclui a leitura do registro).

# Contratos de Point Representation

Este documento descreve `src/contextmap/point_representation/models.py`, `compatibility.py` e `serialization.py`.

## Um vetor descreve um suporte, não um ponto

Um vetor local nunca é calculado só a partir do centro. `PointRepresentation.support` (`PointSupport`) lista **exatamente quais** `GeometryReference` formaram o insumo, sob qual `SupportPolicy`, e `geometry_reference` é sempre o centro desse suporte. Um consumidor sempre consegue reconstruir o que o vetor descreve resolvendo as referências em `GeometrySource.get`.

## `SupportPolicy`

| Campo | Significado |
| --- | --- |
| `support_type` | `POINT` (só o centro) ou `NEIGHBORHOOD` |
| `method` | `RADIUS` ou `K_NEAREST`; `None` exatamente para `POINT` |
| `radius_m` | raio euclidiano em metros (só `RADIUS`) |
| `k` | número de membros, centro incluído (só `K_NEAREST`) |
| `max_neighbors` | teto determinístico sobre um suporte por raio, mantendo os mais próximos |
| `preparation` | `CoordinatePreparation` aplicada às coordenadas |

Em ambos os métodos o **centro é membro do próprio suporte** e conta para o limite. Parâmetros faltando ou sobrando para o tipo/método, raio não finito ou não positivo, `k < 1` e `max_neighbors < 1` falham na construção. Suporte voxel/célula **não** existe: não há requisito atual que o justifique.

## `CoordinatePreparation`

Toda normalização aplicada às coordenadas é registrada:

- `centering`: `NONE` (frame do mapa), `CENTER` (relativo ao centro; padrão) ou `CENTROID` (relativo à média dos membros);
- `scale_normalization`: `NONE` (metros) ou `SUPPORT_RADIUS` (dividido pelo raio da política, então o suporte ocupa uma bola unitária; só definido para o método por raio).

Uma canonicalização de orientação **não é oferecida** até ter uma definição explícita. `PointSupport.applied_scale_m` deriva da política: o divisor em metros, ou `None` quando não há escala.

## `PointSupport` e `SupportStatistics`

`PointSupport` guarda `policy`, `center`, `geometry_refs` (ordem = ordem das coordenadas preparadas), `map_frame`, `statistics` e `query_method` (identidade da consulta espacial que selecionou os candidatos).

Invariantes validadas na construção: o centro está em `geometry_refs`; não há referência repetida; todas pertencem ao mapa do centro; `statistics.count` é o número de referências; um suporte `POINT` tem só o centro; o suporte não excede `k`, `max_neighbors` nem o raio da política.

`SupportStatistics` resume distâncias **euclidianas, em metros, no frame do mapa, a partir do centro**: `count`, `min/max/mean_distance_m` (com `0 ≤ min ≤ mean ≤ max`), `near_map_bounds` (o suporte pode estar truncado pela borda do mapa) e `candidate_count` (quantos candidatos qualificaram antes de um teto; `None` quando não é rastreado).

## `PreparedSupport`

`PreparedSupport` acompanha o suporte com `local_coordinates_m`, uma coordenada finita por referência, na mesma ordem, depois do centering e da escala. As coordenadas locais **nunca substituem** as referências: o suporte continua dizendo qual geometria persistente as originou. Quem produz um `PreparedSupport` é o `SupportExtractor` ([`support-extraction.md`](support-extraction.md)).

## `RepresentationSpace` e comparabilidade

`RepresentationSpace` identifica o espaço vetorial: `family`, `model`, `version`, `checkpoint` (`None` para descritor determinístico), `dimension`, `dtype` (`float32` ou `float64`), `normalization`, `input_definition`, `support_semantics` e `feature_names` (significado de cada componente de um descritor interpretável; vazio para um encoder aprendido).

A **política de suporte faz parte da identidade**: o mesmo encoder sobre 0,25 m e sobre 1 m produz espaços diferentes.

`representation_space_fingerprint()` devolve `"sha256:<hex>"` do JSON canônico do espaço; é o valor gravado em `PointRepresentation.representation_space_id`. Duas representações só são comparáveis (distância, similaridade, média, indexação) quando o fingerprint é **idêntico**. **Dimensão igual nunca implica compatibilidade**: um descritor e um encoder aprendido de mesma dimensão são espaços distintos. `ensure_compatible_representation_spaces` e `ensure_compatible_representations` levantam `RepresentationSpaceMismatchError` quando os fingerprints diferem.

## `PointRepresentation`

| Campo | Significado |
| --- | --- |
| `representation_id` | identidade local ao run (`representation_id_for(run_id=..., index=...)`) |
| `geometry_reference` | geometria à qual está ancorada; é o centro do `support` |
| `support` | exatamente quais elementos formaram o insumo |
| `representation_space_id` | fingerprint do `RepresentationSpace` |
| `shape` / `dtype` / `normalization` | descrição do vetor armazenado |
| `payload_reference` | referência relativa ao run (`payloads/...`); `None` quando o payload não foi armazenado |
| `encoder_identity` | `EncoderIdentity`: backend, versão, fingerprint da configuração, hash do checkpoint |
| `provenance` | `RepresentationProvenance`: versão do código |
| `undefined_components` | índices ascendentes de componentes que o encoder não pôde definir |

O vetor **não é embutido**: só é referenciado. Uma referência de payload vazia, absoluta ou com `..` é rejeitada. `undefined_components` torna explícito que, por exemplo, uma curvatura de um suporte degenerado é indefinida: o valor armazenado é um placeholder que nunca deve ser interpretado (`is_partial`). A linhagem de nível de run (artifacts de entrada, configuração efetiva) fica no run artifact, não repetida por vetor.

Uma representação **não é verdade semântica**: não carrega label, claim, entidade nem feature visual, e nenhum contrato concatena canais.

## `FailedSupport`

Um suporte que não pôde ser representado é um resultado explícito, nunca um vetor nulo ou padrão: `support` (qual geometria estava envolvida), `reason` (`FailureReason`: `UNENCODABLE_SUPPORT` ou `NON_FINITE_OUTPUT`) e `detail` (a explicação do encoder, obrigatória). Ver [`execution.md`](execution.md).

## Serialização

`serialization.py` converte os contratos para registros com apenas primitivas JSON, legíveis sem NumPy ou biblioteca de modelo, e **revalida os contratos ao decodificar**: um registro adulterado (por exemplo, a âncora deixando de ser o centro do suporte) falha em vez de produzir uma representação inválida. Um registro de representação nunca contém o vetor numérico. O suporte é codificado de forma compacta: o mapa (único por invariante) e o `center_geometry_id` uma vez e os membros como `geometry_ids`, porque um run persiste um suporte por representação e repetir o mapa por membro dominaria seu tamanho.

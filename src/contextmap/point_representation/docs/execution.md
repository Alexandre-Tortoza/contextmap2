# Porta `PointEncoder` e serviço de execução

Este documento descreve `src/contextmap/point_representation/ports.py` e `service.py`.

## Por que existe uma porta

Um descritor geométrico determinístico e um encoder 3D aprendido (por exemplo, PTv3) são produtores diferentes do **mesmo contrato público**: ambos recebem o suporte preparado e publicam um vetor para `PointRepresentation`. A porta existe porque essa variação é real; não é um sistema de plugins. Qual encoder roda é decidido por quem constrói o serviço (a composição em `runtime`), nunca pelo serviço, e não há registry.

## `PointEncoder`

| Método | Papel |
| --- | --- |
| `encoder_identity()` | `EncoderIdentity`: backend, versão, fingerprint da configuração efetiva (sem segredos) e hash do checkpoint |
| `representation_space()` | `RepresentationSpace` dos vetores produzidos, **incluindo a política de suporte** sob a qual o serviço deve extrair |
| `encode(prepared)` | um `PreparedSupport` → um `EncodedVector` |

`EncodedVector` traz `values` (um `float` Python por componente, nunca um tensor de framework) e `undefined_components` (índices ascendentes que o encoder não conseguiu definir). Um encoder que não consegue representar um suporte levanta `UnencodableSupportError`; ele **nunca** devolve um vetor nulo ou padrão no lugar de uma falha. O encoder só enxerga o `PreparedSupport`: não lê geometria de outra forma e não depende do índice do mapa.

## `RepresentationService`

`RepresentationService(source, encoder, *, run_id, code_version)` resolve cada centro pedido em um suporte preparado (`SupportExtractor`, sob a política do espaço do encoder), chama o encoder e monta a `PointRepresentation` canônica com o fingerprint do espaço, `shape`, `dtype`, `normalization`, `EncoderIdentity` e `RepresentationProvenance`. O serviço não conhece nenhum backend concreto, não ramifica por backend e não tem fallback de um encoder para outro.

`represent(centers)` devolve um iterador com um resultado por centro, na ordem do pedido:

- `EncodedRepresentation(representation, values)`: metadados canônicos e o vetor. `payload_reference` é `None` até um writer gravar o vetor e dizer onde;
- `FailedSupport(support, reason, detail)`: o suporte não pôde ser representado.

A identidade de uma representação usa a **posição do centro no pedido**: um suporte falho deixa uma lacuna em vez de deslocar as identidades seguintes.

### Validação antes do trabalho pesado

Antes de extrair qualquer suporte ou chamar o encoder, `represent` rejeita com `ValueError` um centro de outro mapa e um centro repetido. O restante roda de forma preguiçosa, conforme o iterador é consumido.

### Falhas explícitas

| Situação | Resultado |
| --- | --- |
| encoder levanta `UnencodableSupportError` | `FailedSupport(UNENCODABLE_SUPPORT, detail=...)`; a execução continua |
| valor não finito em componente não declarado indefinido | `FailedSupport(NON_FINITE_OUTPUT)`; a execução continua |
| número de valores diferente da dimensão do espaço | `ValueError`; a execução para |
| `undefined_components` fora da dimensão ou fora de ordem | `ValueError`; a execução para |
| qualquer outra exceção do backend (por exemplo, falta de memória de GPU) | propaga; a execução para |
| centro ausente do mapa | `KeyError` da porta `GeometrySource`; erro do chamador |

Nada disso vira um vetor nulo ou padrão. Componentes declarados indefinidos são armazenados como placeholder zero e listados em `undefined_components`; nunca devem ser interpretados.

### Métricas

`service.metrics` (`RepresentationMetrics`) reflete a última chamada de `represent`, completa quando o iterador termina: `requested`, `represented`, `partial`, `failed_by_reason` (e `failed`), `support_extraction_seconds` e `encoding_seconds`. Distribuições de tamanho de suporte e de normas ficam com quem persiste os resultados.

## O que a porta não faz

Sem fusão semântica, sem concatenação multimodal, sem ramificação por PTv3 no núcleo, sem objetos de framework nas saídas públicas e sem alterar o mapa geométrico para guardar estado privado do backend. A construção de um backend e a verificação de requisitos de dispositivo e de checkpoint acontecem na composição (`runtime`) e na construção do próprio backend, não no serviço.

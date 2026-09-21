# Invariantes, integridade de referências e fixture representativa

Este documento descreve a validação do **schema em si**, feita em memória sobre os contratos públicos e sobre a visão canônica em registros. Ela independe de serializador, de layout de arquivos, de ROS e de bibliotecas de modelo, então continua válida qualquer que seja o serializador do artifact. A validação de arquivos (inventário, hashes, payloads ausentes, índices em disco) pertence ao validador do `ContextMapArtifact` (milestone Context Map Serialization).

## Onde cada invariante é aplicada

Uma invariante é validada **na fronteira mais próxima que a possui** e falha na construção, nunca depois de o dado circular: cada registro valida o que só ele precisa (identidade, ordem canônica, coerência interna) e o `ContextMap` valida o que envolve mais de um registro.

| Invariante | Onde | Erro |
| --- | --- | --- |
| identidade presente e que não é caminho; ids únicos e ordem canônica | cada registro; `ContextMap` para entidades, relações e linhagem | `ValueError` |
| versão do schema legível | `ContextMap` e `context_map_from_record` (antes de qualquer campo) | `UnsupportedSchemaVersionError` |
| frame, unidade, âncora e `up_direction` completos e coerentes; bounds no frame do mapa; janela em um relógio | `MapFrame`, `MapAnchor`, `ContextMapMetadata`, `ObservationWindow` | `ValueError` |
| geometria da entidade pertence ao mapa referenciado, com identidade canônica e dentro do intervalo | `ContextMap` | `ReferenceIntegrityError` |
| sujeito e objeto de uma relação resolvem a entidades do mesmo mapa e são distintos | `ContextRelation`, `ContextMap` | `ValueError`, `ReferenceIntegrityError` |
| identidade de origem mapeada uma única vez | `ContextMap` | `ReferenceIntegrityError` |
| estado semântico concorda com as hipóteses (incerteza não colapsa) | `ContextSemanticState` | `ValueError` |
| capacidades declaradas concordam com o conteúdo e com a linhagem | `ContextMap` | `ValueError` |
| fechamento de proveniência: todo artifact citado está na linhagem com o tipo certo | `ContextMap` | `ReferenceIntegrityError` |
| a categoria de derivação é sustentada pela evidência citada | `EvidenceOrigin`, `ContextMap` | `ProvenanceError` |
| forma do registro: campo ausente, desconhecido ou de tipo errado | `context_map_from_record` | `ContextMapRecordError` |

Nada é reparado em silêncio: um registro inválido **nunca** vira um mapa "corrigido".

## Cobertura

`tests/artifact/test_context_map_invariants.py` cobre cada linha da tabela sobre a fixture, com uma matriz de mutações do registro. Cada caso aplica **uma** mutação ao registro válido e exige o erro exato:

- versão: maior e menor não suportados, versão malformada;
- frame, unidades, âncora e extensão: unidade desconhecida, frame vazio, `up_direction` que não é unitário, origem local que reivindica referência externa, origem externa sem referência, bounds em outro frame, janela com relógios distintos, nenhuma sequência de origem;
- referências: mapa geométrico vazio, geometria de outro mapa, geometria além do intervalo, mapa geométrico ausente da linhagem;
- entidades: id duplicado, ordem, duas entidades para o mesmo registro de origem, entidade sem geometria, entidade resolvida a partir de nada;
- relações: entidade inexistente, entidade de outro mapa, relação consigo mesma, predicado vazio, artifact de origem do tipo errado;
- estado: estado inequívoco com duas hipóteses, conflito reduzido a um rótulo, abstenção com hipótese;
- capacidades: entidades, relações e evidência de representação 3D fora da declaração, tipos de relação divergentes do conteúdo;
- linhagem e proveniência: artifact citado fora da linhagem, tipo errado, linhagem fora de ordem, identidade de conteúdo que não é digest, resultado fundido sem política, origem sem evidência, saída de VLM marcada como observada, relação marcada como observação direta, categoria desconhecida;
- forma do registro: campo desconhecido, campo ausente e tipo errado.

Também há testes de **ausência de conteúdo opcional** (mapa só com geometria, entidades sem relações, relações declaradas porém vazias versus ausentes, evidência opcional ausente) e um teste que garante que o pacote do schema **não faz E/S de arquivo nem possui formato** (nenhum import de `os`, `pathlib`, `json`, `pickle`, `io`, `shutil`, `tempfile`, `glob` ou `sqlite3`).

## Fixture representativa

`tests/fixtures/context_map/corridor.json` é um mapa pequeno e legível a olho: geometria por referência, quatro entidades, três relações e a proveniência completa. Foi gerada **só a partir de contratos públicos** por `tests/artifact/context_map_fixture_builder.py`, sem backend de runtime, e todas as identidades e digests são sintéticos e determinísticos.

| Parte | O que demonstra |
| --- | --- |
| `entity-0001` (`chair`, `UNAMBIGUOUS`) | entidade **resolvida a partir de duas entidades de origem** (`member_entities`) com uma decisão de resolução; origem `MULTIVIEW_FUSED` citando fusão, **observações físicas** e evidência 3D opcional; hipótese `MODEL_INFERRED` |
| `entity-0002` (`desk` / `table`, `AMBIGUOUS`) | hipóteses concorrentes preservadas, sem ranking |
| `entity-0003` (`bin` / `box`, `CONFLICTING`) | conflito preservado: as duas hipóteses continuam no mapa |
| `entity-0004` (`INSUFFICIENT_EVIDENCE`) | abstenção: nenhuma hipótese afirmada, o que não é evidência negativa |
| `relation-0001` (`next_to`, `SUPPORTED`) | relação confirmada, `GEOMETRY_DERIVED` com a política versionada |
| `relation-0002` (`on`, `UNRESOLVED`) | relação candidata que **não** vira confirmada |
| `relation-0003` (`near`, `CONFLICTING`) | evidência que apoia e contradiz |
| `metadata.frame` | frame local de estimador com `up_direction` **desconhecida** (`null` explícito) |
| `lineage` | sequência, percepção (com modelos), fusão, mapa semântico, mapa geométrico, representação 3D, resolução e relações, cada um com digest de conteúdo |

Os testes verificam que a fixture é válida, que reconstrói exatamente o mesmo mapa **qualquer que seja o layout do JSON** (compacto, indentado, chaves ordenadas), que a travessia relação → entidade → geometria → evidência usa apenas tipos públicos e que o arquivo versionado é exatamente a renderização atual dos contratos: se o contrato mudar, o teste falha e a mensagem indica como regenerar (`python tests/artifact/context_map_fixture_builder.py`).

## Fora do escopo

- integridade de arquivos, hashes e payloads: validador do artifact;
- verificação contra o `GeometricMap` real (frame, tamanho e bounds do artifact de geometria): feita por quem abre os dois artifacts, como o validador;
- `entity_resolution` e `spatial_relations` reais: enquanto seus contratos não existem na `dev`, a fixture usa `UpstreamRecordRef` para as identidades de origem.

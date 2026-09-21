# Versionamento, compatibilidade e evolução do schema

Este documento define como o schema do `ContextMap` evolui e como um leitor decide se consegue interpretar um mapa. O código está em `src/contextmap/artifact/versioning.py` (versão e regra de leitura) e `schema_identity.py` (impressão digital estrutural).

## O que a versão descreve

`schema_version` descreve a **semântica dos dados**: quais campos existem, o que significam e quais invariantes valem. Não é a versão do pacote Python nem a de um serializador, e as três mudam de forma independente:

| Identidade | O que identifica | Onde vive |
| --- | --- | --- |
| `schema_version` | semântica dos dados do `ContextMap` | `ContextMap.schema_version`, `CONTEXT_MAP_SCHEMA_VERSION` |
| versão do código/pacote | qual revisão do software montou o mapa | `MapCreation.code_version` e as tags `vMAJOR.MINOR.PATCH` ([`docs/versioning.md`](../../../../docs/versioning.md)) |
| versão do formato de armazenamento | como os registros chegam ao disco | manifesto do serializador (fora deste schema) |

Um pacote `v0.1.0` pode escrever um schema `0.1.0`, mas nada obriga os dois a andarem juntos: reescrever o serializador sem mudar a semântica não muda a `schema_version`, e mudar a semântica a muda mesmo sem nova release do pacote.

## Formato e fases

`MAJOR.MINOR.PATCH`, sem prefixo, pré-release nem zeros à esquerda; qualquer outra grafia é `UnsupportedSchemaVersionError`. O schema nasce em `0.1.0`.

- **`PATCH`**: esclarecimento que não muda dado nem validação;
- **`MINOR`**: adição compatível;
- **`MAJOR`**: qualquer mudança incompatível.

Enquanto o `MAJOR` for zero (fase de validação, [`docs/versioning.md`](../../../../docs/versioning.md)), seguimos o SemVer para `0.y.z`: um `MINOR` pode ser incompatível. Por isso o leitor só lê a **mesma `MAJOR.MINOR`** e nunca há janela de compatibilidade entre `0.1.x` e `0.2.x`. A janela aditiva abaixo passa a valer quando o schema chegar a `1.0.0`, isto é, quando existir um contrato externo estável.

## Classificação de mudanças

Cada linha é um exemplo executável: `tests/artifact/test_context_map_versioning.py` verifica que ele aparece aqui com a classificação declarada e que a versão resultante tem as consequências prometidas para leitores.

| Mudança | Classificação | Por quê |
| --- | --- | --- |
| novo campo opcional com padrão explícito | MINOR | um leitor antigo o ignora sem reinterpretar nada; a ausência significa "não presente" |
| nova capacidade opcional em MapCapability | MINOR | conteúdo a mais, declarado; quem não a conhece a ignora e a reporta |
| novo predicado de relação | MINOR | predicados são texto de uma taxonomia versionada; um predicado novo não altera os existentes |
| novo ArtifactKind apenas de evidência | MINOR | evidência opcional não é necessária para resolver o mapa |
| novo DerivationKind | MINOR | é metadado de proveniência; um leitor que não o conhece o reporta como origem não reconhecida e nunca o trata como uma categoria conhecida |
| esclarecimento de documentação | PATCH | não muda dado nem validação |
| correção de mensagem de erro | PATCH | não muda o que é válido |
| remover ou renomear um campo | MAJOR | um leitor antigo perderia ou reinterpretaria o dado |
| novo campo obrigatório | MAJOR | mapas antigos deixariam de ser válidos e leitores antigos não o produzem |
| mudar o significado, a unidade ou o frame de um campo | MAJOR | o mesmo valor passaria a significar outra coisa em silêncio |
| novo membro de LengthUnit, Handedness, AnchorKind, AmbiguityStatus ou RelationState | MAJOR | um leitor que ramifica sobre o enum interpretaria errado um valor que não conhece (por exemplo, não saberia se uma relação está confirmada) |
| novo ArtifactKind estrutural | MAJOR | um artifact necessário para resolver o mapa que o leitor antigo não sabe resolver |
| mudar o escopo de uma identidade | MAJOR | referências antigas passariam a apontar para outra coisa |
| invalidar um mapa que era válido | MAJOR | endurecer uma invariante quebra dados existentes |

Uma correção de validação que só rejeita dados que **já eram inválidos pelo contrato** é `PATCH`; a linha "invalidar um mapa que era válido" cobre o caso contrário.

## Obrigatório versus opcional

- um campo **obrigatório** só é adicionado ou removido em `MAJOR`;
- um campo **opcional** nasce com padrão explícito (`None` ou vazio) e sua ausência significa "não presente", nunca um valor implícito diferente;
- o registro escreve **todos** os campos, inclusive os opcionais ausentes como `None`: metadado que falta é explícito ([`metadata.md`](metadata.md));
- um campo nunca é reaproveitado com semântica nova: reinterpretar em silêncio um campo antigo é sempre `MAJOR`.

## Evolução de enums, predicados e capacidades

| Vocabulário | Regra |
| --- | --- |
| `MapCapability` (aberto) | membro novo é `MINOR`; um leitor que não o conhece o ignora e o reporta como capacidade não reconhecida |
| `relation_predicates` (texto) | predicado novo é `MINOR`; mudar direção, simetria ou significado de um existente é `MAJOR` (a versão da taxonomia fica na política da origem) |
| `DerivationKind` (aberto) | membro novo é `MINOR`, com a regra de origem não reconhecida acima |
| `ArtifactKind` | evidência opcional é `MINOR`; um tipo estrutural é `MAJOR` |
| `LengthUnit`, `Handedness`, `AnchorKind`, `AmbiguityStatus`, `RelationState` (fechados) | membro novo é sempre `MAJOR` |

Um valor **removido** de qualquer enum é `MAJOR`.

## Campos desconhecidos

- **Fase `0.x` (hoje):** a decodificação é **estrita**. Um campo desconhecido, ausente ou de tipo errado é `ContextMapRecordError`, e uma versão diferente de `0.1.x` é recusada antes de qualquer campo ser lido.
- **A partir de `1.0.0`:** para um documento do mesmo `MAJOR` com `MINOR` maior que o do leitor, o leitor lê o núcleo, **ignora** campos opcionais e capacidades que não conhece e os reporta; um documento do mesmo `MINOR` com campo desconhecido está corrompido e é recusado. Esse comportamento **não está implementado**: não há schema `1.x` que o exija e o código da tolerância seria especulativo (YAGNI). Ele entra junto com a primeira release `1.0.0`, com seus testes.

## Negociação de versão e janela do leitor

O leitor declara a versão que implementa (`CONTEXT_MAP_SCHEMA_VERSION`) e aplica `SchemaVersion.is_readable_by`:

| Documento | Leitor | Legível |
| --- | --- | --- |
| `0.1.0` … `0.1.9` | `0.1.x` | sim (só o `PATCH` varia) |
| `0.2.0` | `0.1.x` | não |
| `1.0.5` | `1.2.0` | sim (versão mais antiga, mesmo `MAJOR`) |
| `1.3.0` | `1.1.0` | sim (o núcleo; o conteúdo novo é ignorado) |
| `2.0.0` | `1.9.9` | não |

Uma versão fora da janela é recusada **explicitamente**, com as duas versões na mensagem (`UnsupportedSchemaVersionError`), antes de qualquer outra leitura. Nada é interpretado parcialmente.

## Descontinuação

- **`0.x`:** não há ciclo de descontinuação. Como determina a regra de v0.x do projeto, não se mantêm wrappers, migração nem fallback só para preservar formatos antigos sem consumidor real: a mudança é direta e a versão sobe;
- **a partir de `1.0.0`:** um campo ou capacidade é marcado como descontinuado em um `MINOR` (continua sendo escrito e lido, com a nota na documentação) e só é removido no `MAJOR` seguinte. Descontinuar nunca muda o significado.

## Migração

Um artifact finalizado é imutável e **nunca é reescrito** para "atualizar" o schema. A v0.1.0 **não exige migração automática** de versões históricas. Migrar um mapa significa gerar **outro** `ContextMap`, com outra identidade, a partir dos artifacts a montante (reexecutando a montagem). Se um dia uma ferramenta de migração for necessária, ela é explícita, produz um artifact novo cuja linhagem aponta o antigo e pertence à release que introduz o `MAJOR`. Fixtures de cada versão prometida como legível são mantidas pelos testes de serialização.

## Impressão digital estrutural

`schema_fingerprint()` devolve `sha256:<hex>` da descrição canônica (`describe_schema()`): nomes, tipos e valores de enum do contrato, incluindo todo tipo alcançável a partir de `ContextMap`, mesmo os reutilizados de outras capabilities (`GeometryReference`, `Bounds3D`, `SourceTimestamp`). A descrição ignora a ordem de declaração e o módulo onde o tipo vive: reordenar campos ou mover uma classe não é mudança de schema; renomear um tipo é.

Ela **não vê significado**: dois schemas com a mesma estrutura e invariantes diferentes têm a mesma impressão. Classificar a mudança continua sendo uma decisão registrada aqui. A impressão protege da deriva silenciosa: uma mudança estrutural sem uma decisão de versão consciente.

- o manifesto do artifact registra `schema_version` **e** `schema_fingerprint`; um leitor que encontra a mesma versão com outra impressão sabe que o código e o dado divergem;
- `tests/artifact/test_context_map_versioning.py` fixa a impressão de cada versão prometida como legível e **falha de propósito** quando a estrutura muda.

## Como mudar o schema

1. altere o contrato e rode `tests/artifact/test_context_map_versioning.py`: o teste da impressão falha e mostra a nova;
2. classifique a mudança na tabela acima (`PATCH`, `MINOR` ou `MAJOR`);
3. atualize `CONTEXT_MAP_SCHEMA_VERSION` conforme a classificação e registre a nova impressão no teste;
4. atualize a documentação do campo afetado e adicione ou ajuste as fixtures;
5. declare o impacto no PR (`Alteração compatível` ou `incompatível`) e, para `MAJOR`, use `BREAKING CHANGE:` no commit.

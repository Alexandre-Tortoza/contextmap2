# Taxonomia de predicados

O vocabulário de Solution 1 é **pequeno, fechado e versionado** (`TAXONOMY_VERSION = "spatial-relation-taxonomy-v1"`). Um predicado só entra quando a sua geometria pode ser definida e validada; acrescentar um é uma nova versão da taxonomia, nunca uma extensão silenciosa. Direção, simetria, inverso e pressupostos de frame vivem numa única tabela (`PREDICATE_SPECS`), lida por candidatos, avaliadores, política de decisão e artifact.

A sentença é sempre `sujeito PREDICADO objeto`.

## Tabela

| Predicado | Família | Simétrico | Inverso | Avaliado diretamente | Frame exigido |
| --- | --- | --- | --- | --- | --- |
| `NEXT_TO` | proximidade | sim | ele mesmo | sim | `MAP_FRAME` |
| `ABOVE` | direcional | não | `BELOW` | sim | `UP_AXIS` |
| `BELOW` | direcional | não | `ABOVE` | **não (derivado)** | `UP_AXIS` |
| `IN_FRONT_OF` | direcional | não | `BEHIND` | sim | `UP_AND_FORWARD_AXES` |
| `BEHIND` | direcional | não | `IN_FRONT_OF` | **não (derivado)** | `UP_AND_FORWARD_AXES` |
| `INSIDE` | topológica | não | `CONTAINS` | sim | `MAP_FRAME` |
| `CONTAINS` | topológica | não | `INSIDE` | **não (derivado)** | `MAP_FRAME` |
| `INTERSECTS` | topológica | sim | ele mesmo | sim | `MAP_FRAME` |
| `TOUCHING` | suporte/contato | sim | ele mesmo | sim | `MAP_FRAME` |
| `ON_TOP_OF` | suporte/contato | não | nenhum | sim | `UP_AXIS` |
| `LEANING_AGAINST` | suporte/contato | não | nenhum | sim | `UP_AXIS` |

As famílias separam quatro tipos de significado: **proximidade** (distância, sem direção), **direcional** (posição relativa ao longo de um eixo declarado), **topológica** (como as extensões se relacionam como regiões) e **suporte/contato** (contato físico, apoio e inclinação, que exigem mais do que uma distância).

## Direção, simetria e inverso

- **Simétrico** (`NEXT_TO`, `INTERSECTS`, `TOUCHING`): `a P b` e `b P a` são o mesmo fato; o inverso do predicado é ele mesmo. A avaliação acontece **uma vez** por par não ordenado, e a relação do sentido oposto é gerada como gêmea simétrica.
- **Com inverso** (`ABOVE`/`BELOW`, `IN_FRONT_OF`/`BEHIND`, `INSIDE`/`CONTAINS`): as duas palavras dizem o mesmo fato com sujeito e objeto trocados. Só **um** membro de cada par é avaliado diretamente (`ABOVE`, `IN_FRONT_OF`, `INSIDE`); o outro (`is_derived`) é gerado pela inversão da relação avaliada. Assim não há duas medições que possam discordar da mesma geometria.
- **Sem inverso** (`ON_TOP_OF`, `LEANING_AGAINST`): o converso (`SUPPORTS`) seria vocabulário novo, e a taxonomia não expande a ontologia em silêncio.

Invariantes verificadas por teste: o inverso de um simétrico é ele mesmo; o inverso é uma involução; exatamente um membro de cada par inverso é avaliado diretamente; os dois membros de um par compartilham o requisito de frame.

## Convenções de frame

Nada a montante declara qual direção é "para cima": o frame de um mapa geométrico é escolhido pela estimação de estado e um frame de SLAM só é alinhado à gravidade quando algo o alinhou. Por isso predicados verticais e de profundidade são **sem significado** até que a execução **declare** os eixos do frame do mapa em `FrameConventions`, uma política versionada (`map-frame-conventions-v1`) com fingerprint.

| Campo | Significado |
| --- | --- |
| `map_frame` | O frame descrito; toda geometria avaliada precisa estar expressa nele. |
| `up_axis` | A direção oposta à gravidade, ou `None` quando não declarada. |
| `forward_axis` | A direção de referência **horizontal** de `IN_FRONT_OF`/`BEHIND`, ou `None`. Pertence ao frame do mapa, não à orientação de nenhuma entidade. |

Regras:

- **Nenhum eixo é assumido** nem inferido da geometria. Não há valor padrão.
- Os eixos são os **eixos principais com sinal** do frame (`AxisDirection`: `+x`, `-x`, `+y`, `-y`, `+z`, `-z`). O resumo espacial de uma entidade é uma caixa alinhada aos eixos do frame, então um "para cima" inclinado em relação às faces da caixa não seria avaliado com exatidão. Uma convenção inclinada seria uma nova versão da política, não uma aproximação silenciosa.
- `forward_axis` só existe junto de `up_axis` e precisa ser perpendicular a ele.
- Nada é específico de um dataset: quem sabe qual eixo é vertical no frame de uma execução é a configuração dela.

### Falha explícita

Antes de medir qualquer coisa, os avaliadores chamam `FrameConventions.require_evaluable(requisito, *geometrias)`:

- `IncompatibleFrameError` — a geometria está expressa em outro frame que o das convenções, ou as geometrias pertencem a **mapas geométricos diferentes**: nomes de frame iguais em dois mapas não tornam as coordenadas comparáveis;
- `UndeclaredAxisError` — o predicado exige um eixo que a execução não declarou.

Ambos herdam de `FrameConventionError` (`ValueError`). Um predicado cujo eixo nunca foi declarado não produz evidência que só *parece* significativa: chamá-lo levanta. `FrameConventions.supports(requisito)` permite consultar, sem levantar, se os eixos declarados bastam para um predicado.

## Versionamento

`TAXONOMY_VERSION` identifica o vocabulário **e** a semântica descrita aqui. Mudar o significado de um predicado, acrescentar ou remover um, ou trocar direção, simetria ou inverso exige uma nova versão; a versão vai na proveniência de toda relação e de toda evidência.

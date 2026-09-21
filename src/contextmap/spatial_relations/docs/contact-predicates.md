# Contato e apoio

Uma distância entre centroides ou entre caixas não distingue tocar de estar apenas perto, nem apoiar de pairar. `TOUCHING`, `ON_TOP_OF` e `LEANING_AGAINST` exigem medições **por ponto**, então os avaliadores resolvem os pontos das duas entidades a partir do mapa geométrico (`GeometrySource`) e os medem diretamente. O resultado é `RelationEvidence` do canal `CONTACT`, com as medições exatas, os limiares e os subconjuntos de geometria usados, listados por identidade.

```python
evaluate_contact_predicate(
    predicate,
    subject_entity_ref=...,
    subject_geometry=...,
    object_entity_ref=...,
    object_geometry=...,
    geometry_source=...,
    policy=ContactPredicatePolicy(...),
    conventions=FrameConventions(...),
)
evaluate_contact_candidates(
    candidates, entities=..., geometry_source=..., policy=..., conventions=...
)
```

`evaluate_contact_candidates` avalia, na ordem dos candidatos, os do canal de contato, resolve os pontos de cada entidade uma única vez e recusa candidatos gerados sob outras convenções de frame. Os candidatos geométricos ficam para `evaluate_geometric_candidates`.

Nenhum label semântico, nenhuma simulação de física e nenhuma gravidade inferida: o eixo vertical é o que a execução declarou. Não existem normais de superfície a montante, então nenhuma é usada.

## Regra de decisão

A mesma da [avaliação geométrica](geometric-predicates.md): uma comparação só é decidida quando vale (ou falha) em toda a faixa de tolerância em torno da medição. O status é `CONFLICTS` se alguma condição falha; senão `UNAVAILABLE` se uma entrada exigida não existe; senão `SUPPORTS` se todas cumprem; senão `AMBIGUOUS`. Suporte esparso ou desconexo nunca é decisivo.

## `ContactPredicatePolicy`

Sem valores padrão e sem nada específico de dataset; a política tem versão (`CONTACT_POLICY_ID = "point-contact-predicates-v1"`), fingerprint, e cada predicado tem sua regra versionada (`point-contact-touching-v1`, ...).

| Campo | Significado |
| --- | --- |
| `contact_distance_m` | Distância (m) até a qual dois pontos estão em contato. |
| `contact_tolerance_m` | Incerteza da posição de um ponto (m); é a faixa de ambiguidade. |
| `min_contact_points` | Pontos que cada entidade precisa ter em contato para que uma área de contato seja mensurável. |
| `support_height_tolerance_m` | Quão longe a face mais baixa do sujeito pode estar da mais alta do objeto (`ON_TOP_OF`). |
| `support_footprint_fraction` | Fração, em `(0, 1]`, da extensão do sujeito que o objeto deve sustentar em cada eixo horizontal (`ON_TOP_OF`). |
| `leaning_min_tilt_deg`, `leaning_max_tilt_deg` | Inclinação do sujeito em relação ao eixo vertical que é "encostado" e não "de pé" nem "deitado" (`0 < min < max < 90`). |
| `tilt_tolerance_deg` | Incerteza da inclinação medida (graus). |
| `leaning_min_vertical_overlap_m` | Quanto as alturas de sujeito e objeto devem se sobrepor (m) para o sujeito se apoiar na lateral do objeto. |

## Medições e condições

`bounds_gap` e `pairs_within_search_radius` (pares de pontos dentro de `contact_distance + contact_tolerance`, o raio de busca) sempre são registrados. A menor distância entre os conjuntos de pontos (`nearest_support_distance`) só é registrada quando algum par cai no raio de busca: sem par, a distância mínima é maior que o contato mais a tolerância, o que basta para falhar a condição sem inventar um valor exato (o `bounds_gap` é então um limite inferior).

| Predicado | Condições (todas sob a faixa de tolerância) |
| --- | --- |
| `TOUCHING` (simétrico) | `nearest_support_distance <= contact_distance`; e, se a distância cumpre, `contact_points_subject` e `contact_points_object` `>= min_contact_points` |
| `ON_TOP_OF` | contato como acima; `abs(support_height_error) <= support_height_tolerance`, com `support_height_error = S.baixo - O.alto` ao longo do `up_axis`; em cada eixo horizontal, `support_footprint_overlap >= support_footprint_fraction x extensão do sujeito` |
| `LEANING_AGAINST` | contato como acima; `vertical_overlap >= leaning_min_vertical_overlap`; `leaning_min_tilt <= tilt_from_up <= leaning_max_tilt` |

- `contact_points_subject` / `contact_points_object` são os pontos de cada entidade a menos de `contact_distance` de algum ponto da outra: a **área de contato** mensurável. Poucos pontos tornam o contato `AMBIGUOUS` com ressalva `UNRELIABLE_GEOMETRY` ("poucos para medir uma área de contato"), em vez de decidir sobre um único ponto ruidoso.
- `tilt_from_up` é o ângulo, em `[0, 90]` graus, entre o eixo dominante do sujeito (o primeiro eixo principal de `EntityGeometry.orientation`) e o `up_axis`; o sinal de um eixo principal é arbitrário, então só a inclinação conta.
- **`LEANING_AGAINST` exige contato e orientação compatível.** Sem `orientation` no sujeito o registro é `UNAVAILABLE` com `MISSING_INPUT`, nunca adivinhado, a menos que o contato já falhe (então é `CONFLICTS`, pois a inclinação não mudaria o resultado). Um rótulo semântico sozinho nunca estabelece a relação: não há entrada por onde ele chegue.

## Geometria preservada

Além dos suportes inteiros (`subject`, `object`, por contagem e digest), o registro traz os pontos em contato (`subject_contact_points`, `object_contact_points`) com a contagem e o digest da identidade exata. Até 256 pontos o subconjunto é listado por `GeometryReference`; acima disso, só contagem e digest, para o registro não crescer com a densidade da nuvem. Os limiares usados e a proveniência (regra, fingerprint da política, frame, mapa e convenções) acompanham o registro.

## Escala

A busca de pares usa uma grade uniforme com célula do tamanho do raio de busca: cada ponto do sujeito só encontra os pontos do objeto das 27 células ao redor, então o custo é proporcional aos pares realmente próximos, sem varrer todos os pares de pontos. O resultado é exato e independe da ordem dos pontos.

## Frame e erros

`TOUCHING` só exige o frame do mapa; `ON_TOP_OF` e `LEANING_AGAINST` exigem `up_axis` (`UndeclaredAxisError` se faltar). Geometria ou fonte em outro frame, ou entidades de mapas diferentes: `IncompatibleFrameError`. Pontos que não resolvem no mapa entregue: `GeometryResolutionError`. Pedir um predicado fora do canal levanta `ValueError`.

## Limites conhecidos

- A área de contato é uma contagem de pontos, não uma área em m²; com nuvens de densidades muito diferentes o limiar `min_contact_points` precisa ser escolhido com isso em mente.
- A inclinação usa o eixo dominante do sujeito: objetos cujos eixos principais são mal definidos (cubos) não têm orientação e `LEANING_AGAINST` fica indisponível para eles.
- `ON_TOP_OF` e `LEANING_AGAINST` não têm inverso no vocabulário (`SUPPORTS` seria vocabulário novo).
- Sem validação em dado real: os testes são contratos com nuvens sintéticas (caixotes, mesa, parede e ripa inclinada).

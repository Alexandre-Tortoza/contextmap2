# Predicados geométricos

Os avaliadores de `NEXT_TO`, `ABOVE`, `IN_FRONT_OF`, `INSIDE` e `INTERSECTS` produzem `RelationEvidence` do canal `GEOMETRY` a partir de `EntityGeometry`, de forma determinística. Eles leem só os limites alinhados aos eixos, os diagnósticos do suporte e as convenções de frame declaradas: nenhum label semântico chega à aritmética e nenhum sinal é fundido num score. O `status`, as medições e os limiares ficam inspecionáveis sem nenhuma reconciliação posterior.

```python
evaluate_geometric_predicate(
    predicate,
    subject_entity_ref=...,
    subject_geometry=...,
    object_entity_ref=...,
    object_geometry=...,
    policy=GeometricPredicatePolicy(...),
    conventions=FrameConventions(...),
)
evaluate_geometric_candidates(candidates, entities=..., policy=..., conventions=...)
```

`evaluate_geometric_candidates` avalia, na ordem dos candidatos, os candidatos do canal geométrico e deixa os de contato e apoio (#144) para o avaliador deles; recusa candidatos gerados sob outras convenções de frame.

## Regra única de comparação

`boundary_tolerance_m` é a incerteza de uma face dos limites. Uma comparação só é **decidida** quando vale para todo valor dentro dessa tolerância em torno da medição:

- vale em toda a faixa: a comparação **cumpre** o limiar;
- falha em toda a faixa: a comparação **falha**;
- caso contrário a medição está **dentro da tolerância** do limiar: o veredito é ambíguo.

O registro é `SUPPORTS` quando toda condição cumpre, `CONFLICTS` quando alguma falha e `AMBIGUOUS` nos demais casos, com uma ressalva `WITHIN_TOLERANCE` (que cita a medição, o valor e o limiar) para cada condição não decidida. Duas regras evitam que geometria pouco confiável pareça decisiva:

- **suporte esparso ou desconexo nunca é decisivo** (`UNRELIABLE_GEOMETRY`): os limites podem não descrever o objeto, então o registro é `AMBIGUOUS` e as medições continuam registradas;
- **profundidade de interpenetração não é mensurável ao longo de um eixo plano**: `NEXT_TO` e `INTERSECTS` sobre suporte plano são `AMBIGUOUS` com `DEGENERATE_GEOMETRY`.

## `GeometricPredicatePolicy`

Sem valores padrão e sem nada específico de dataset; a política tem versão (`GEOMETRIC_POLICY_ID = "bounds-geometric-predicates-v1"`) e fingerprint, e cada predicado tem sua regra versionada (`bounds-next-to-v1`, `bounds-above-v1`, ...) na proveniência.

| Campo | Significado |
| --- | --- |
| `boundary_tolerance_m` | Incerteza de uma face dos limites (m). |
| `next_to_max_gap_m` | Maior distância entre os limites que ainda é "ao lado" (m). |
| `adjacent_penetration_m` | Maior interpenetração que ainda é "adjacente" e não "interseção" (m); também limita quanto o sujeito pode afundar no objeto e ainda estar acima dele. |
| `containment_slack_m` | Quanto uma face do sujeito pode ultrapassar a do objeto e ainda estar dentro (m); absorve ruído em limites parcialmente observados. |
| `directional_overlap_fraction` | Fração, em `(0, 1]`, da menor extensão que as seções transversais devem compartilhar em cada eixo perpendicular à direção. |

## Definições

Todas as grandezas estão em metros no frame do mapa. `S` é o sujeito e `O` o objeto.

| Predicado | Medições | Condições |
| --- | --- | --- |
| `NEXT_TO` (simétrico) | `bounds_gap`, `min_axis_overlap` | `bounds_gap <= next_to_max_gap` **e** `min_axis_overlap <= adjacent_penetration` |
| `INTERSECTS` (simétrico) | `axis_overlap_x/y/z`, `min_axis_overlap` | `min_axis_overlap >= adjacent_penetration` |
| `INSIDE` | `containment_margin` | `containment_margin >= -containment_slack` |
| `ABOVE` | `vertical_clearance`, `footprint_overlap_<eixo>`, `footprint_overlap_fraction_<eixo>` | `vertical_clearance >= -adjacent_penetration` **e**, em cada eixo perpendicular ao `up_axis`, `footprint_overlap >= directional_overlap_fraction x menor extensão` |
| `IN_FRONT_OF` | `forward_clearance`, `footprint_overlap_<eixo>`, `footprint_overlap_fraction_<eixo>` | igual, ao longo do `forward_axis` |

- `bounds_gap` é a distância entre as faces mais próximas (`0` se as caixas se sobrepõem ou se tocam).
- `min_axis_overlap` é a menor sobreposição de intervalos entre os três eixos, com sinal: positiva é a profundidade de interpenetração, negativa é a separação no eixo mais próximo.
- `containment_margin` é a menor folga, entre as seis faces, do sujeito para dentro das faces do objeto; negativa significa que o sujeito ultrapassa o objeto.
- `vertical_clearance` e `forward_clearance` são `S.baixo - O.alto` lidos ao longo do eixo declarado com sinal (positivo é folga livre, negativo é afundamento).
- Os limiares derivados (`required_footprint_overlap_<eixo>` = fração x menor extensão) ficam nos limiares do registro, junto de `adjacent_penetration`, `boundary_tolerance` e dos demais valores da política usados por aquele predicado.

### Por que `NEXT_TO` e `INTERSECTS` nunca valem juntos

Os dois dividem o mesmo limiar `adjacent_penetration_m` com faixas simétricas: `NEXT_TO` exige profundidade abaixo dele e `INTERSECTS` acima. Perto do limiar ambos ficam ambíguos, e nunca ambos suportados. Caixas que apenas se tocam são `NEXT_TO`, não `INTERSECTS` ("faces que se tocam não se interceptam").

### Inverso e simetria

Só um membro de cada par inverso é avaliado (`ABOVE`, `IN_FRONT_OF`, `INSIDE`); `BELOW`, `BEHIND` e `CONTAINS` são **gerados** por inversão do registro avaliado (pela política de decisão, #146), então duas medições não podem discordar sobre a mesma geometria. Pedir um predicado derivado levanta `ValueError` que nomeia o inverso. Os simétricos dão o mesmo veredito e os mesmos números em qualquer ordem. Um predicado direcional nunca é `SUPPORTS` nos dois sentidos (verificado sobre uma grade de posições relativas), exceto `INSIDE` para caixas idênticas, que estão uma dentro da outra: a política de decisão reporta esse par como estrutura inconsistente em vez de o avaliador decidir.

## Frame e eixos

`ABOVE` exige `up_axis` e `IN_FRONT_OF` exige `up_axis` e `forward_axis` em `FrameConventions`; sem eles `UndeclaredAxisError`. Geometria em outro frame ou de outro mapa: `IncompatibleFrameError`. Inverter `up_axis` inverte a resposta, e uma cena empilhada em `y` só é "acima" quando `+y` é declarado o eixo vertical. Os predicados sem eixo (`NEXT_TO`, `INSIDE`, `INTERSECTS`) não registram o fingerprint das convenções.

## Evidência emitida

Todo registro traz `evidence_id` determinístico, canal `GEOMETRY`, o candidato, o status, as medições e limiares ordenados por nome, a geometria medida (`subject` e `object`, com contagem de pontos e digest exato dos ids de geometria, sem copiar coordenadas), as ressalvas e a proveniência (regra, fingerprint da política, versão da taxonomia, frame, mapa geométrico e, quando há eixos, fingerprint das convenções).

## Limites conhecidos

- Só limites alinhados aos eixos: objetos alongados na diagonal têm caixas folgadas, o que favorece `NEXT_TO`/`INTERSECTS` e prejudica `INSIDE` estrito. Uma convenção de eixo inclinada seria uma nova versão da política.
- O eixo `forward` é uma referência do frame do mapa, não a orientação de nenhuma entidade: "à frente de" não é "à frente do ponto de vista do objeto".
- Sem validação em dado real: todos os testes são contratos sobre geometria sintética resumida pelo algoritmo real de Semantic Mapping.

# Construção de `FusionSupport`

`build_fusion_supports` agrupa observações espaciais que enxergam **geometria sobreposta** em unidades de suporte sobre as quais a evidência pode ser acumulada. Fundir em cada ponto criaria milhões de estados semânticos redundantes; criar entidades aqui violaria a fronteira com Semantic Mapping. Um `FusionSupport` afirma apenas:

> estas observações espaciais se referem a suporte espacial suficientemente sobreposto sob a política escolhida.

Ele **não** implica mesmo objeto, mesma classe semântica nem identidade persistente entre versões do mapa.

## Política baseline `geometry-jaccard-support-v1`

| Aspecto | Regra |
| --- | --- |
| Medida de sobreposição | Índice de Jaccard entre os conjuntos de `GeometryReference` de duas observações: `\|A ∩ B\| / \|A ∪ B\|`. Só a identidade da geometria conta: nunca label, claim ou score. |
| Ligação | Duas observações com sobreposição `>= min_overlap` (inclusivo) estão ligadas. |
| Suportes | As componentes conexas dessas ligações. A ligação é **transitiva**: uma cadeia de vistas parcialmente sobrepostas pode reunir vistas que não se sobrepõem diretamente. |
| Divisão | Uma observação pertence a exatamente um suporte e um suporte nunca é dividido. Geometria disjunta nunca é unida: claims iguais não juntam lugares diferentes. |
| Suporte aninhado | Jaccard é simétrico: uma região pequena dentro de uma grande tem sobreposição baixa e continua um suporte separado. |
| Suporte mínimo | Uma observação com menos de `min_geometry_count` elementos não entra em nenhum suporte e é listada em `excluded`, nunca descartada em silêncio. |
| Geometria desconexa | A geometria de uma região não é analisada quanto à conectividade: uma região com pontos distantes fica em um suporte e seus limites cobrem tudo. |
| Ordem e identidade | Observações ordenadas por identidade; suportes pela primeira observação; `support-000001`, `support-000002`, … seguem essa ordem. A mesma entrada e configuração reconstroem sempre os mesmos suportes. |

### Por que Jaccard e não contenção

Se um `door` e um `handle` se sobrepusessem por contenção, cairiam no mesmo suporte e apareceriam como hipóteses concorrentes, um conflito espúrio: o baseline não tem ontologia para saber que uma é parte da outra. Uma **falsa junção** cria conflitos e contamina hipóteses; uma **falsa divisão** apenas acumula menos evidência, e Entity Resolution pode refinar depois sem alterar a evidência fundida. O custo é que duas vistas do mesmo objeto em escalas muito diferentes (uma muito menor que a outra) podem não se unir. O `min_overlap` é explícito e a avaliação (issue #121) mede essa troca.

### Configuração

`GeometryOverlapSupportPolicy(min_geometry_count, min_overlap)` **não tem valores padrão**: os limiares são escolhas científicas que um perfil declara. `fingerprint()` devolve `sha256:` da identidade da política e dos limiares e entra na proveniência de cada suporte.

## Entrada e saída

```python
build = build_fusion_supports(
    observations,  # SpatialObservation, em qualquer ordem
    geometry=geometry_source,  # GeometrySource do mapa referenciado
    acquisition_timestamps=timestamps,  # SourceObservationId -> SourceTimestamp
    policy=policy,
)
```

`FusionSupportBuild` traz `supports` (ordenados) e `excluded` (observações com pouca geometria, com a contagem e o mínimo). Toda observação de entrada está em exatamente um suporte ou em `excluded`. `support_id_of()` indexa observação → suporte; o índice suporte → observações e suporte → geometria são os campos `spatial_observation_ids` e `geometry_support` de cada `FusionSupport`.

Cada suporte resume:

- `geometry_support`: a união da geometria das observações, ordenada e única;
- `bounds` (`Bounds3D` no frame do mapa) e `centroid_m` (média das coordenadas autoritativas, em metros);
- `time_bounds`: o intervalo de aquisição das observações, em um único domínio de relógio;
- `provenance`: política, fingerprint da configuração e `code_version`.

As coordenadas vêm de `GeometrySource.get`, cada elemento resolvido uma única vez, mesmo visto por muitas observações.

## Erros

Falha cedo, com mensagem acionável, quando: uma observação se repete, ou pertence a outro mapa que o da `GeometrySource`; uma observação referencia geometria que o mapa não contém; um frame suportado não tem timestamp de aquisição; os timestamps de um suporte cobrem mais de um domínio de relógio; ou a política é impossível (`min_geometry_count < 1`, `min_overlap` fora de `(0, 1]`).

## Custo

A sobreposição usa um conjunto de bits por observação sobre a geometria que pelo menos duas observações referenciam. O custo é quadrático no número de observações e linear na geometria compartilhada por par. Medição ad hoc (não versionada): 400 observações de 2 000 pontos sobre um mapa de 100 000 pontos, ~1 s.

## O que não faz

- não cria `Entity` nem identidade persistente;
- não usa label, claim ou score como chave de agrupamento;
- não usa `PointRepresentation` para definir a identidade do suporte;
- não altera geometria nem `SpatialObservation`;
- não constrói contribuições nem acumula evidência (issue #117).

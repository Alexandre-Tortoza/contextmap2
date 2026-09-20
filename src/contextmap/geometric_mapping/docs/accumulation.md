# Acumulação do mapa e referências espaciais

Este documento descreve `src/contextmap/geometric_mapping/accumulation.py` e `geometry_storage.py`.

Os scans transformados ([`transformation.md`](transformation.md)) são acumulados em **um** mapa persistente, com referências estáveis que as capabilities seguintes seguram no lugar de copiar XYZ ou proveniência.

```text
TransformedScan ──► MapAccumulator ──► payload empacotado (sink)  + AccumulatedMap
                                                                     (GeometricMap + índice de origem)
payload + AccumulatedMap ──► PackedGeometry  (GeometrySource)
```

## Um único frame global

`MapAccumulator` recebe o frame do mapa e **recusa** qualquer scan expresso em outro frame: processar só um trecho da sequência continua produzindo coordenadas no mesmo frame global, e um frame de segmento ou submapa só pode existir como uma visão derivada com transform explícito. Um teste confere que as coordenadas de um scan são idênticas quando ele é processado em um segmento ou na execução inteira.

Também são recusados: o mesmo scan duas vezes, scans em domínios de clock diferentes e um mapa sem nenhuma geometria.

## Formato do payload

A geometria autoritativa é um arquivo plano de **registros de tamanho fixo, little-endian**, um por elemento, na ordem de acumulação:

```text
<3d3dIqI   x y z do mapa (m) | x y z de origem (m) | ordinal do scan | índice do ponto no scan | contagem
```

64 bytes por ponto. O índice do ponto no scan é `-1` para um ponto agregado; a contagem é o número de medições atrás do ponto (`1` para uma medição crua). O que é comum aos pontos de um scan (observação, frame, timestamp, cadeia de transforms com os números, estado de correção) fica **uma vez** na tabela de scans, não em cada ponto. O payload é lido com a biblioteca padrão (`struct`, `mmap`, `memoryview`): abrir um mapa não exige NumPy, ROS nem biblioteca de nuvem de pontos. Só a escrita usa NumPy, importado ao acumular.

A escrita é em **fluxo**: cada scan vai direto para o `sink`, então um mapa longo não fica na memória.

## Referências estáveis

O registro na posição `i` é a geometria de identidade `geometry_id_for(map_id, i)`. Resolver uma referência é aritmética sobre a identidade (`geometry_index_of`) e um acesso aleatório ao payload; não há tabela de identidade. Só a grafia canônica é aceita, então cada geometria tem exatamente uma identidade. Uma referência de outro mapa, malformada, não canônica ou além do fim é um `KeyError`.

Entradas iguais reproduzem os mesmos bytes e as mesmas referências (teste). As referências são locais ao artefato imutável; uma política de identidade mais forte só entra se for projetada explicitamente.

## Índice de origem

`ScanRecord` (uma entrada por scan, em `AccumulatedMap.scans` e `PackedGeometry.scans`) guarda a observação, o frame, o timestamp, o `payload_hash`, o estado de correção, a cadeia de transforms, quantos pontos o scan tinha e quantos foram descartados por não serem finitos, o intervalo de geometria (`first_geometry_index`, `geometry_count`) e os limites do scan. `PackedGeometry.references_for(observation_id)` lista as referências de uma observação e `scan_record(observation_id)` devolve a entrada. Um scan aceito sem nenhum ponto finito fica no índice com `geometry_count == 0` e não conta em `source_observation_ids`.

A abertura valida que o payload tem exatamente um registro por ponto e que o índice de origem ladrilha a geometria; um registro que aponta para o intervalo de outro scan, ou um ponto agregado num mapa que não declara regra de agregação, é detectado no acesso.

## Rastreio de um ponto persistido

`PackedGeometry.trace(reference)` reconstrói o `TransformTrace` de um ponto persistido a partir do payload e do índice de origem, sem o scan original: mostra a pose e a calibração que o colocaram na posição global. `verify_transform_trace` o confere. `close()` libera a visão do payload.

## Limites e tempo

Os limites do mapa (`GeometricMap.bounds`) e de cada scan são o envelope justo das coordenadas persistidas, no frame do mapa. `time_bounds` cobre os scans que contribuíram com geometria, em um único clock.

## Agregação explícita

Por padrão **cada ponto transformado é persistido como uma medição crua**. A política opcional `ScanVoxelPolicy(cell_m=...)` une os pontos **de um mesmo scan** que caem no mesmo voxel (grade alinhada à origem do frame do mapa) no seu centroide:

- o ponto agregado declara a regra (`scan-voxel-centroid-<cell_m>m`), a contagem de medições e **não** carrega índice de ponto; `origin = AGGREGATED`;
- as coordenadas de origem do agregado são o centroide das coordenadas de origem dos membros (a cadeia é rígida, logo afim: o centroide de origem leva ao centroide do mapa);
- um voxel com um só ponto continua uma medição crua, com o seu índice;
- pontos de **scans diferentes nunca são unidos**, então nenhuma agregação colapsa observações de origem em silêncio;
- a ordem dos pontos agregados é determinística (por voxel);
- `GeometricMap.aggregation_rule` registra a regra, e `AccumulatedMap.source_point_count` junto de `point_count` dá a razão de redução.

Não há agregação entre scans: a redundância entre scans sobrepostos permanece. Isso é uma limitação assumida, não um descuido; uma política entre scans exige estado global e uma decisão explícita sobre o que preservar da origem.

## Sem semântica

Nenhum campo de label, `SemanticClaim`, embedding, entidade ou relação existe na geometria. A evidência visual ou semântica referencia a geometria persistente por `GeometryReference`; nunca se anexa um scan colorido como geometria duplicada.

## Limitações

- A consulta espacial e o índice derivado estão em [`spatial-access.md`](spatial-access.md).
- A escrita exige NumPy (extras `dev` e `ros1`, não as dependências base).

# Avaliação de Point Representation

Este documento descreve `src/contextmap/evaluation/point_representation.py`.

Point Representation é **opcional** e precisa justificar seu custo por avaliação controlada. O harness roda a mesma geometria e os mesmos centros por cada **braço** de uma ablação e relata o que cada um produziu, sem transformar isso em veredito.

| Braço | Papel (`RepresentationArmRole`) |
| --- | --- |
| A. `off` | `OFF`: sem representação 3D; a linha de base para a comparação downstream |
| B. descritor determinístico | `DETERMINISTIC_DESCRIPTOR`: o controle só de geometria |
| C. PTv3 | `LEARNED_3D`: inferência com um encoder 3D genérico |
| D. pré-treinado/destilado | `PRETRAINED_DISTILLED`: um backend identificado separadamente (estilo Sonata/Vernata); opcional, fora do caminho canônico v0.1.0 |

`RepresentationArm(arm_id, role, encoder)` recebe o encoder **já construído**
(a composição hoje é responsabilidade do chamador; `contextmap.runtime` ainda
está planejado); `OFF` não tem encoder e os demais exigem um. PTv3 sozinho
**não** pode ser rotulado como `PRETRAINED_DISTILLED`: o backend `ptv3` com esse
papel é rejeitado. Um backend futuro do braço D usa o mesmo harness sem mudar
nenhum contrato downstream.

## O que o relatório de um braço contém

`evaluate_representation_arm(arm, source=..., centers=..., context=...)` devolve um `RepresentationArmReport` com seções **separadas**:

- **Identidade** (`RepresentationArmIdentity`): backend, versão, fingerprint da configuração, hash do checkpoint, fingerprint do espaço e política de suporte, além de `evaluation_id`, `code_version`, o mapa geométrico, o `center_selection_id` (a mesma regra do manifest do run artifact) e a configuração downstream mantida fixa. Com isso o braço é reproduzível a partir de identidades exatas.
- **Cobertura**: pedidos, representados, parciais, falhas por motivo e a fração de componentes indefinidos. Uma falha nunca conta como representação (comportamento de falha/OOM).
- **Repetibilidade**: o mesmo insumo é executado duas vezes; relata a maior diferença absoluta entre componentes, se a cobertura foi idêntica e se `max_abs_difference ≤ repeatability_tolerance`. A tolerância é explícita e o padrão `0.0` exige saída idêntica, o que um encoder determinístico deve cumprir; um encoder que não repete é **relatado**, não escondido.
- **Distribuição**: contagem, mínimo, mediana, p95 e máximo das normas L2 sobre os componentes definidos.
- **Sensibilidade** (uma por variação): quanto as representações se movem sob uma variação controlada da geometria, sempre **dentro do próprio espaço do braço**, comparando só componentes definidos dos dois lados. Mede a mudança L2 relativa e a distância cosseno por centro e, para um descritor interpretável, a mudança absoluta média **por componente nomeado**; um espaço opaco (aprendido) não tem esse detalhe. Como as escalas dos componentes de um descritor são heterogêneas (densidade e tamanho dominam a norma), a visão por componente evita que um componente grande esconda os pequenos.
- **Custo**: tempo de extração de suporte, tempo dentro do encoder, tempo por representação, bytes de payload (`RepresentationSpace.vector_bytes` por representação) e os números do backend fornecidos por quem chama (por exemplo, pico de memória de GPU). O custo nunca entra em nenhuma medida de qualidade.
- **Downstream**: as métricas que o chamador mediu para o braço (consistência da fusão, falsas fusões, duplicatas, taxa de não resolvidas) são **carregadas como dadas** e nunca combinadas. Fusão e resolução de entidades pertencem a capabilities posteriores: quando existirem, o experimento as executa com percepção, associação, geometria e claims fixos e passa aqui o resultado por braço.

O braço `off` só traz identidade, variáveis fixas e o downstream.

## Variações controladas

`GeometryVariation(name, description, apply)` é uma receita nomeada que produz uma **cópia** variada da geometria (a original nunca é modificada; o índice espacial do mapa original é descartado na cópia):

- `translation_variation(offset_m)`: translação rígida (consistência espacial: uma representação consistente não deve mudar);
- `rotation_about_z_variation(degrees)`: rotação em torno do eixo z do mapa; formas locais não mudam, mas as **direções** absolutas de um descritor giram com o mapa;
- `noise_variation(sigma_m, seed)`: ruído gaussiano reproduzível (sensibilidade a ruído do sensor e à fronteira do suporte);
- `subsample_variation(keep_fraction, seed)`: mantém uma fração aleatória, sempre com os centros (sensibilidade à densidade).

Sensibilidade a raio e a `k` compara **espaços diferentes** (a política de suporte faz parte do espaço) e por isso não é feita aqui: exige braços separados, um por política, comparados como braços.

## Comparação controlada

`compare_representation_arms(reports)` exige que todos os braços tenham o mesmo **mapa geométrico**, o mesmo **conjunto de centros** e a mesma **configuração downstream mantida fixa** (o drift em qualquer um é rejeitado), que os `arm_id` sejam distintos e que haja ao menos um relatório. O resultado guarda os relatórios lado a lado e **não calcula nada entre eles**: não há score, ranking nem vencedor. Quem decide se mantém a capacidade opcional, adota um backend ou adia lê as seções separadas. Não se escolhe um backend aprendido por uma única métrica, e a qualidade da representação e os efeitos ponta a ponta são relatados à parte do custo computacional.

`encode_representation_arm_report()`/`encode_representation_ablation_report()` produzem dados JSON-compatíveis (`identity`, `representation`, `cost`, `downstream`) para artifacts de avaliação futuros.

## Execução real de amostra (descritor, corredor sintético)

Medição ad hoc (não versionada): 149 centros sobre um corredor sintético de 20 mil pontos (raio de 0,5 m), braços `off` e descritor, Python 3.14, um único thread; a avaliação completa com seis variações levou 9,3 s.

- **Cobertura**: 149 de 149 representadas, 0 parciais, 0 falhas. **Repetibilidade**: diferença máxima `0.0` (repetível).
- **Norma L2**: mediana 147, p95 185. **Custo**: 0,4 ms por representação, 16,7 KB de payload.
- **Translação rígida**: mudança L2 relativa mediana de `4e-16` (invariante, como esperado).
- **Rotação de 30° em z**: mediana de `3,5e-3`: as formas e a densidade não mudam, mas os componentes de direção (`principal_axis_abs_*`) giram com o mapa.
- **Ruído** de 5 mm e de 2 cm: mediana de `1,7e-2` e `3,0e-2`; `surface_variation` (mudança média) sobe de `0,0011` para `0,0052` e `normal_abs_z` de `0,0044` para `0,013`.
- **Subamostragem** que mantém 70% e 40% dos pontos: mediana de `0,29` e `0,58`, dominada pelos componentes de densidade e tamanho do suporte (`density_per_m3`, `support_size`), enquanto os de forma mudam pouco.

Esses números mostram o que o relatório reporta para o **controle geométrico**; eles não dizem nada sobre um encoder aprendido, que continua sem execução real.

## Execução real de amostra (geometria real do corredor-02)

Diferente da seção sintética acima, esta usa **geometria real**: o `GeometricMap` que o Geometric Mapping produziu a partir da trajetória `ExternalPose` de `datasets/corridor-02/corridor-02-gt.txt` na janela de 90 s da validação (14 422 535 pontos, 860 varreduras LiDAR; run `geometric-mapping/corridor-02/run-0001__ts-w336-90s__external-pose-all-points`, lido sem modificação e com integridade verificada). São **reais** a geometria, o descritor e o PTv3 (o backbone do Pointcept pelo `PointceptPTv3Runtime`, ver [`ptv3.md`](../../point_representation/docs/ptv3.md)). **Não há downstream** nesta seção.

**Por que um recorte.** Avaliar suportes sobre o mapa inteiro não foi tentado: o harness materializa em memória uma cópia de toda a geometria por variação (14 milhões de objetos `GeometryPoint`), e com 860 varreduras sobrepostas o índice `scan_bounds` tende a selecionar muitas delas por consulta (não medido aqui). A geometria avaliada é um **recorte** do mapa real: os pontos numa caixa de ±4 m em x e y (e de -1,5 a +3,5 m em z) em torno da pose do meio da janela, reduzidos a **um ponto real por voxel de 10 cm** (o de menor índice de geometria). Cada ponto mantido é um `GeometryPoint` real do mapa, com identidade e proveniência próprias, então as referências continuam válidas no mapa. O recorte tem 28 787 pontos (de 1 458 162 na caixa) e o SHA-256 do conjunto de índices mantidos é `sha256:fcf58d10369514dabbba1313f7fb25fda98f51fbfd672ac40e4d2967893a2d4b`.

**Protocolo.** 100 centros reais, sorteados com semente 0 entre os pontos a pelo menos 0,5 m das faces da caixa (para não medir truncamento de suporte); suporte por raio de 0,5 m com centralização; as seis variações da seção sintética. Braços `off`, descritor e PTv3 (`ptv3m1-base`, checkpoint `nuscenes-semseg-pt-v3m1-0-base` com `sha256:e2774d2fa1dd33e640a514afe0bd1e5af94eb08be19d08d1f9402f20dfd6db94`, `float32`, `grid_size_m = 0,05`, pooling `center`, sem normalização). Não há configuração downstream fixada, e o fingerprint dela carrega a identidade do recorte. O relatório completo (`report_with-ptv3.json`, `sha256:92567d8c0f45f0d7f921e681b92515f628999bb0df8aeda0e48cdbb9b9d64682`) e os run artifacts dos dois encoders ficam em `workspace/corridor-02/validation-point-representation-20260921/point_representation/`, fora do controle de versão; o driver é `scripts/10_evaluate_real_geometry.py` (`sha256:87794c75570c4a71f372a6f2778bd755f8208e05fe9e47d9232bcb251fac0b18`), executado no commit `42d2e8e`, que não alterou o harness. Espaços: descritor `sha256:ed35aeda…`, PTv3 `sha256:dd1486da…`.

### Braços lado a lado (as seções não são combinadas)

| | Descritor (controle) | PTv3 (aprendido) |
| --- | --- | --- |
| Cobertura | 100/100, 0 falhas | 100/100, 0 falhas |
| Repetibilidade (tolerância `0.0`) | idêntica (diferença máxima `0.0`) | **não** idêntica: `2,9e-6` (GPU `float32`; relativa de `2,6e-7`) |
| Norma L2 (mediana; p95) | 641; 779 | 11,0; 12,1 |
| Translação rígida (mudança L2 relativa mediana) | `0` | `3,7e-7` |
| Rotação de 30° em z | `7,7e-4` | `0,32` (p95 `0,67`) |
| Ruído de 5 mm | `8,0e-3` (p95 `0,026`) | `0,28` (p95 `0,57`) |
| Ruído de 2 cm | `0,016` (p95 `0,067`) | `0,42` (p95 `0,76`) |
| Subamostragem a 70% | `0,30` (p95 `0,34`) | `0,29` (p95 `0,52`) |
| Subamostragem a 40% | `0,59` (p95 `0,64`) | `0,51` (p95 `0,73`) |
| Separação entre lugares (mediana; p95) | `0,133`; `0,61` | `0,607`; `0,906` |
| Tempo por representação | 1,5 a 2,9 ms | 32,4 ms |
| Payload por representação | 112 bytes | 256 bytes |
| Pico de memória de tensores | CPU | 193 MiB (GPU) |

A separação entre lugares é a distância L2 relativa entre pares de centros distintos do mesmo recorte (diagnóstico do driver, não do harness). Ela dá a escala contra a qual uma mudança deve ser lida: uma variação que move o vetor mais que a diferença típica entre dois lugares diferentes destrói a capacidade de distinguir lugares.

- **Descritor.** É repetível, invariante a translação, quase invariante à rotação (`7,7e-4`) e pouco sensível a ruído, mas a remoção de 30% dos pontos o move cerca de **2,2 vezes** a diferença típica entre dois lugares (e a de 60%, cerca de 4,4): os componentes de densidade e tamanho do suporte dominam o vetor, e a densidade de um LiDAR real varia com a distância. É uma limitação da linha de base, medida.
- **PTv3.** É invariante a translação até a precisão numérica, mas **não** é invariante a rotação em torno de z (mudança mediana de 0,32) e é sensível a ruído de 5 mm (0,28). Em relação à separação entre lugares, a razão mudança/separação com 30% dos pontos removidos é **0,47** (contra 2,2 do descritor), com ruído de 5 mm **0,46** (contra 0,06) e com rotação de 30° **0,53** (contra 0,006). Ou seja, é bem mais robusto à densidade e bem mais frágil a ruído e rotação. Nenhum braço domina o outro.
- **Acordo entre os espaços.** Sobre os mesmos 100 centros, o CKA linear entre o descritor e o PTv3 é 0,22 e a sobreposição dos 10 vizinhos mais próximos é 0,18 (o acaso daria cerca de 0,10): os dois espaços organizam os lugares de modo diferente, e o PTv3 não é uma recodificação do descritor.

### Consistência de reobservação (real, sem rótulos de identidade)

Como não há anotações de identidade, o valor de discriminação foi medido por uma proxy com correspondência por **coordenadas do mapa**, não por rótulos inventados. As varreduras que contribuem para o recorte (414) foram divididas na mediana (ordinal 432): as **anteriores** (o robô se aproximando do lugar; 23 900 pontos reais no recorte) e as **posteriores** (afastando-se; 19 970 pontos), com pontos de vista, alcances e densidades diferentes. Cada centro real do primeiro conjunto é pareado ao ponto mais próximo do segundo (no máximo 10 cm; 100 pares, semente 0) e o vetor de um é comparado com os dos 100 pontos pareados do outro. O relatório é `report_reobservation.json` (`sha256:e377264ed07c1bb8c3764a1e570693ac7df0993809ad97a78f769ea4f5ad2995`), do driver `scripts/30_reobservation_consistency.py` (`sha256:cb01915bebdf4db4fb019a47c8aed3c65f7af8b76c1f32adbea7db2d6779d5e5`).

| Medida (L2 bruta; padronizada por componente) | Descritor | PTv3 | Acaso |
| --- | --- | --- | --- |
| AUC: o mesmo lugar fica mais perto que outro lugar | 0,510; 0,529 | 0,541; 0,547 | 0,5 |
| Recall@1 exato | 0,02; 0,00 | 0,01; 0,01 | 0,01 |
| Recall@5 exato | 0,06; 0,04 | 0,09; 0,12 | 0,05 |
| Posto mediano do par verdadeiro (de 100) | 53; 44 | 45; 42 | ~50 |
| Top-1 tolerante (contraparte a até 1 m do lugar) | 0,04; 0,08 | 0,06; 0,08 | 0,067 |
| Mudança L2 relativa do mesmo lugar (L2 bruta: mediana; p95) | 0,24; 0,45 | 0,53; 0,77 | |

**Leitura.** Os dois encoders ficam **perto do acaso** nesta tarefa. A diferença de AUC entre o PTv3 e o descritor (cerca de 0,03) está dentro do erro esperado com 100 consultas (erro-padrão da AUC de aproximadamente 0,03, por estimativa), então **não demonstra** vantagem do PTv3. Um corredor uniforme é autossimilar, e um suporte local de 0,5 m sobre uma parede plana é quase o mesmo em qualquer ponto dela: para qualquer descritor puramente local de geometria, baixa discriminação é esperada neste ambiente. O resultado diz que o PTv3 não **demonstra** valor aqui, não que não possa tê-lo em outro ambiente ou com um backend adaptado ao domínio (o checkpoint é de segmentação de LiDAR veicular do nuScenes e o canal de intensidade é preenchido com zeros).

### Decisão baseada em evidência

Com o que foi medido, a decisão sustentada é **manter Point Representation opcional e adiar a adoção do PTv3 como backend do caminho canônico**:

1. o custo é cerca de 20 vezes o do descritor por representação (32 ms contra 1,5 a 2,9 ms), exige GPU e um checkpoint de 554 MB de outro domínio;
2. não há vantagem demonstrada de discriminação de lugar sob reobservação (ambos perto do acaso);
3. ele troca robustez à densidade por fragilidade a ruído e rotação, o que não é uma melhoria uniforme;
4. o PTv3 só foi avaliado como baseline genérico; um backend pré-treinado ou destilado (braço D, estilo Sonata/Vernata) caberia no mesmo harness e não foi avaliado.

Isso é uma decisão sobre **qualidade da representação e custo, reportados separadamente**; não há um score único nem vencedor.

## Estado e pendências

- **Ablação downstream não realizada, e não simulada.** Entity Resolution existe hoje só no branch `milestone/entity-resolution` (ainda não integrado à `dev`, empilhado sobre `milestone/semantic-mapping-entity-model`), com um canal opcional de estrutura 3D (`RepresentationComparator`). Comparar as decisões de resolução com e sem o canal PTv3 sobre entidades derivadas da geometria real exigiria (a) esse branch, cerca de 46 mil linhas ainda não revisadas, e (b) **anotações de identidade** do corredor-02, que não existem (o esquema `IdentityAnnotationSet` existe em `evaluation-reference-set`, mas não há dado anotado). Sem rótulos, o **valor** downstream (falsas fusões, duplicatas, taxa de não resolvidas) não pode ser pontuado, e inventá-los não é aceitável. A amostra real da validação tem 20 quadros, poucas entidades para uma comparação que, sem rótulos, não pode ser pontuada. Fica para quando Entity Resolution estiver na `dev` e houver anotações de identidade.
- **Anotações de estrutura em dado real**: não há rótulos de estrutura do corredor, então a capacidade de um espaço distinguir tipos de estrutura não é medida; só estabilidade, invariância, separação entre lugares, consistência de reobservação e custo.
- **Sensibilidade a raio e a `k`** compara espaços diferentes e exige braços separados por política; não foi executada nesta amostra.
- **Braço D** (pré-treinado/destilado) continua opcional e fora do caminho canônico v0.1.0.

## Restrições

Sem entradas visuais ou semânticas ocultas nas comparações só de geometria, sem NumPy nem runtime de modelo, sem mutar nenhum artifact avaliado e sem tratar codificações repetidas da mesma geometria como observações físicas independentes.

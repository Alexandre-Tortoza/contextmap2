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

## Execução real de amostra (descritor, geometria real do corredor-02)

Diferente da seção sintética acima, esta usa **geometria real**: o `GeometricMap` que o Geometric Mapping produziu a partir da trajetória `ExternalPose` de `datasets/corridor-02/corridor-02-gt.txt` na janela de 90 s da validação (14 422 535 pontos, 860 varreduras LiDAR; run `geometric-mapping/corridor-02/run-0001__ts-w336-90s__external-pose-all-points`, lido sem modificação e com integridade verificada). Real, e não simulada: a geometria e o descritor. **Não há PTv3 nem downstream nesta seção.**

**Por que um recorte.** Avaliar suportes sobre o mapa inteiro não foi tentado: o harness materializa em memória uma cópia de toda a geometria por variação (14 milhões de objetos `GeometryPoint`), e com 860 varreduras sobrepostas o índice `scan_bounds` tende a selecionar muitas delas por consulta (não medido aqui). A geometria avaliada é um **recorte** do mapa real: os pontos numa caixa de ±4 m em x e y (e de -1,5 a +3,5 m em z) em torno da pose do meio da janela, reduzidos a **um ponto real por voxel de 10 cm** (o de menor índice de geometria). Cada ponto mantido é um `GeometryPoint` real do mapa, com identidade e proveniência próprias, então as referências continuam válidas no mapa. O recorte tem 28 787 pontos (de 1 458 162 na caixa) e o SHA-256 do conjunto de índices mantidos é `sha256:fcf58d10369514dabbba1313f7fb25fda98f51fbfd672ac40e4d2967893a2d4b`.

**Protocolo.** 100 centros reais, sorteados com semente 0 entre os pontos a pelo menos 0,5 m das faces da caixa (para não medir truncamento de suporte); suporte por raio de 0,5 m com centralização; as seis variações da seção sintética. Braços `off` e `descriptor`. Não há configuração downstream fixada, e o fingerprint dela carrega a identidade do recorte. O relatório completo (`report_cpu-arms.json`, `sha256:eeaf10b8d19e2c9b81e9b8fe4db1fdd7417ee91d1fd58ea9149364c7e6a40a76`) e o run artifact do descritor ficam em `workspace/corridor-02/validation-point-representation-20260921/point_representation/`, fora do controle de versão; o driver é `scripts/10_evaluate_real_geometry.py` (`sha256:95938f0bc36b33e135fffde9d85e42843621386a1627e7ad5957b502f7fe504c`), executado no commit `b495e3a`, que não alterou o harness.

- **Cobertura**: 100 de 100 representadas, 0 parciais, 0 falhas. **Repetibilidade**: diferença máxima `0.0`.
- **Norma L2**: mediana 641, p95 779 (mínimo 113, máximo 792). **Custo**: 1,5 ms por representação (0,13 s de extração de suporte e 0,02 s no encoder para os 100 centros) e 112 bytes de payload por representação.
- **Translação rígida**: mudança L2 relativa `0` (invariante).
- **Rotação de 30° em z**: mediana de `7,7e-4` (p95 `1,1e-3`).
- **Ruído** de 5 mm e de 2 cm: mediana de `8,0e-3` (p95 `0,026`) e de `0,016` (p95 `0,067`).
- **Subamostragem** que mantém 70% e 40% dos pontos: mediana de `0,30` (p95 `0,34`) e de `0,59` (p95 `0,64`).
- **Separação entre lugares** (diagnóstico do driver, não do harness): a distância L2 relativa entre pares de centros distintos tem mediana `0,133` (p95 `0,61`). Uma remoção de 30% dos pontos move o descritor cerca de `2,2` vezes mais que a diferença típica entre dois lugares diferentes do corredor, e uma de 60% cerca de `4,4` vezes: o descritor é dominado pelos componentes de densidade e tamanho do suporte, então **não é robusto à densidade**, propriedade que num LiDAR real varia com a distância. É uma limitação da linha de base, medida, e não uma falha do harness.

Isso caracteriza o **controle geométrico** em geometria real. Nada aqui diz algo sobre um encoder aprendido nem sobre o efeito downstream.

## Estado e pendências

- **Braço C (PTv3) sem execução real registrada no harness**: o runtime real existe (`backends/ptv3_pointcept.py`, ver [`ptv3.md`](../../point_representation/docs/ptv3.md)), mas a passagem real dele sobre este recorte, com tempo e pico de VRAM, ainda não foi registrada. O comando do driver para isso é `10_evaluate_real_geometry.py --arms off,descriptor,ptv3` no venv do modelo, sob o lock da GPU. Até lá a comparação do descritor com um PTv3 real e as medidas de acordo entre os dois espaços continuam pendentes.
- **Ablação downstream pendente**: Semantic Fusion já existe e aceita Point
  Representation como canal opcional, mas nenhum resultado downstream foi
  medido; Entity Resolution ainda não existe, então falsas fusões, duplicatas e
  taxa de não resolvidas não podem ser medidas nem simuladas. O harness já
  carrega e compara as condições, sem transformar essa capacidade estrutural em
  evidência de ganho.
- **Decisão baseada em evidência**: com o que existe, a única conclusão sustentada é manter Point Representation **opcional** e adiar a adoção de um backend aprendido; a decisão de adotar ou descartar o PTv3 exige o braço C real e o efeito downstream.
- **Anotações de estrutura em dado real**: não há rótulos de estrutura do corredor, então a capacidade de um espaço distinguir tipos de estrutura não é medida; só estabilidade, invariância, separação entre lugares e custo.

## Restrições

Sem entradas visuais ou semânticas ocultas nas comparações só de geometria, sem NumPy nem runtime de modelo, sem mutar nenhum artifact avaliado e sem tratar codificações repetidas da mesma geometria como observações físicas independentes.

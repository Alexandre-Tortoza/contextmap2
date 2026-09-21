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

## Estado e pendências

- **Braço C (PTv3) sem execução real**: não há torch nem pesos no ambiente, então a comparação contra `off` e o descritor com um PTv3 real está pendente.
- **Ablação downstream pendente**: Semantic Fusion já existe e aceita Point
  Representation como canal opcional, mas nenhum resultado downstream foi
  medido; Entity Resolution ainda não existe. O harness já carrega e compara as
  condições, sem transformar essa capacidade estrutural em evidência de ganho.
- **Checagem em dados reais parcial**: a validação deste harness usa geometria sintética. O descritor determinístico teve uma execução real sobre o mapa LiDAR de `corridor-02` (189 âncoras, todas representadas), como insumo do canal 3D da fusão (ver [Semantic Fusion](semantic_fusion.md)); nenhuma comparação contra `off` nem ablação downstream sobre dados reais foi registrada.

## Restrições

Sem entradas visuais ou semânticas ocultas nas comparações só de geometria, sem NumPy nem runtime de modelo, sem mutar nenhum artifact avaliado e sem tratar codificações repetidas da mesma geometria como observações físicas independentes.

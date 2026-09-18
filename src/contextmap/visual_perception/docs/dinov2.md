# Backend de features densas DINOv2

Este documento descreve `src/contextmap/visual_perception/backends/dinov2.py`, o adapter concreto que transforma patch tokens DINOv2 em `DenseFeatureMap` canônico sem expor tensores PyTorch ou objetos do Transformers.

## Fronteira do backend

`DinoV2DenseFeatureBackend` implementa o port `FeatureExtractor` com `required_scope() == FeatureScope.DENSE`. A execução rica `extract_dense()` produz:

- `VisualFeature` denso com shape, dtype, normalização, `EmbeddingSpace` e referência de payload;
- `DenseFeatureSampling` completo, em coordenadas da imagem preparada;
- payload NumPy na resolução nativa dos patch tokens;
- `EmbeddingSpace` completo para persistência/registro pelo runtime.

O payload é enviado a um `FeaturePayloadSink` estruturalmente compatível com `PerceptionRunWriter.add_feature_payload()`. Assim, `extract()` satisfaz o port existente, retorna somente metadata canônica e ainda enfileira o array para a finalização atômica do artefato. `extract_dense()` expõe o mesmo resultado, sem executar inferência uma segunda vez, para composition roots que também precisam encaminhar imediatamente o `DenseFeatureMap` e o array a pooling ou outro estágio compatível.

## Configuração e identidade

`DinoV2Config` torna explícitos checkpoint, revisão, device, precisão, dimensões de entrada, política de download, normalização L2 opcional, prefixo de payload e versão do adapter. A proveniência usa um fingerprint determinístico da configuração efetiva.

O `EmbeddingSpace` usa:

```text
family        = dinov2
model         = <checkpoint>
checkpoint    = <checkpoint>@<revision>
layer         = last_hidden_state.patch_tokens
dimension     = canais do output real
normalization = none | l2
```

Mesmo dimensionamento não torna este espaço compatível com DINOv3, CLIP ou outro checkpoint DINOv2.

## Preprocessamento e transformação espacial

O runtime Hugging Face faz resize direto e determinístico para `input_width × input_height`, sem center crop. O processor continua responsável pela conversão RGB, rescale e normalização esperados pelo checkpoint. O patch size vem de `model.config.patch_size`; o grid é validado contra `model_input // patch_size`.

Cada patch é mapeado de volta à imagem preparada por escala independente nos eixos X/Y. Origem, stride e suporte são persistidos em pixels da imagem preparada. A identidade da transformação inclui dimensões e transformações da `PreparedImage`, dimensões da entrada do modelo, política de resize, patch size e fingerprint da configuração.

Resize direto pode distorcer aspect ratio quando a configuração não preserva a proporção da imagem preparada. Essa decisão é explícita e reproduzível; o runtime deve configurar dimensões coerentes com a preparação de imagem usada pelo experimento. O backend não faz upsampling do grid de features. Aumento de resolução pertence ao estágio opcional da issue #194.

## Carregamento lazy e falhas explícitas

Importar `contextmap.visual_perception` ou o módulo do backend não importa PyTorch, Transformers, Pillow ou NumPy. Os SDKs e o checkpoint só são carregados na primeira inferência.

`local_files_only=True` é o default: nenhuma execução baixa pesos implicitamente. Na máquina de inferência, o usuário deve instalar `torch`, `transformers` e `Pillow` e disponibilizar antecipadamente o checkpoint/revision no cache local (ou configurar conscientemente `local_files_only=False`).

Falhas não acionam fallback:

- `DinoV2DependencyError` — SDK opcional ausente;
- `DinoV2DeviceError` — device/precisão indisponível;
- `DinoV2ModelLoadError` — checkpoint/revision não carregável;
- `DinoV2InferenceError` — payload, preprocessamento, inferência ou shape nativo inválido.

## Validação desta implementação

Os testes de contrato usam um runtime injetado e determinístico. Eles cobrem metadata, persistência, mapeamento espacial, identidade de embedding, normalização, determinismo, port, pooling comum e falhas explícitas sem download, GPU ou inferência real.

Uma execução controlada com pesos reais não foi realizada neste ambiente por decisão explícita do usuário. Portanto, equivalência numérica real para um checkpoint específico ainda precisa ser validada na máquina de inferência antes de encerrar a issue #68.

## O que este backend não faz

- não produz labels ou `SemanticClaim`;
- não projeta LiDAR nem cria associação 2D↔3D;
- não aumenta a resolução nativa dos patch tokens;
- não compara espaços de embedding;
- não oferece fallback de checkpoint, device ou backend;
- não torna PyTorch/Transformers parte dos contratos públicos.

Ver [`dense_region_association.md`](dense_region_association.md) para consumo do mapa, [`embedding_space.md`](embedding_space.md) para compatibilidade e [`feature_store.md`](feature_store.md) para persistência.

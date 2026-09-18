# Region Discovery

Region Discovery propõe geometria 2D para uma execução de Visual Perception. A capability
preserva evidência geométrica e provenance, sem decidir identidade persistente, label final ou
suporte 3D.

## Contratos canônicos

`RegionCandidate` representa uma proposta antes de validação, merge e normalização. A proposta
carrega identidade da observação física, run, resultado, pass e proposta nativa. Sua geometria pode
ser uma bounding box, uma máscara inline ou uma referência imutável para uma máscara persistida.

`Region2D` representa geometria aceita e congelada. Sua identidade é composta por
`perception_run_id`, `perception_result_id` e `region_id`; portanto, dois resultados podem usar o
mesmo `region_id` sem sugerir que representam o mesmo objeto físico. A identidade não pode ser
usada como entity ID do mapa.

`RejectedRegionCandidate` registra uma rejeição com motivo legível por máquina, detalhe e pass de
origem. Rejeições e propostas incorporadas por merge permanecem disponíveis para auditoria.

## Espaço de coordenadas

A convenção inicial é `pixel_xy_top_left`:

- origem no canto superior esquerdo;
- `x` cresce para a direita e `y` cresce para baixo;
- bounding boxes são intervalos semiabertos `[x_min, x_max)` e `[y_min, y_max)`;
- largura e altura descrevem o espaço da imagem preparada;
- máscaras inline usam ordem row-major e exatamente `width * height` valores.

Geometria fora dos limites da imagem é inválida. Remapeamentos de crop, resize ou tile devem
ocorrer antes da criação da região canônica e permanecer registrados em provenance.

## Scores

`BackendScore` exige nome, valor e semântica. Um `predicted_iou` do SAM não é tratado como
probabilidade nem comparado diretamente com scores de Florence-2. Ausência de score permanece
`None`; ela não é convertida em zero ou um.

## Normalização de backends

Adapters convertem apenas dados serializáveis para `RegionCandidate`:

```text
SAM2/SAM3 mask + box + native scores -> RegionCandidate
Florence-2 box ou mask + parser data  -> RegionCandidate
fake deterministic proposal          -> RegionCandidate
```

Tensors, objetos de SDK e handles de modelo ficam dentro do adapter. Prompt ou texto usado para
descobrir uma região pode aparecer na provenance, mas não cria automaticamente um
`SemanticClaim`.

## Preparação de imagem e constraints

`prepare_image` recebe uma `SourceImage` imutável e uma sequência explícita de operações. Sem
configuração, o resultado referencia o payload original e registra `transformations = []`. Resize,
crop, rectification e normalization recebem a referência do payload materializado pelo adapter de
imagem e produzem um `TransformationRecord` ordenado com dimensões de entrada/saída e parâmetros.

`ValidRegion` restringe os pixels elegíveis e `ExclusionRegion` remove áreas nomeadas. Ambos são
opcionais, precisam corresponder ao espaço de coordenadas final e registram motivo e origem da
configuração. A API não contém defaults para câmera fisheye, veículo, drone, rig ou dataset. Uma
necessidade desse tipo deve ser declarada pelo preset/source config que criou a constraint.

O contrato não pinta pixels excluídos de preto nem altera a observação física. Backends recebem a
imagem preparada e as constraints separadamente, evitando que uma alteração visual silenciosa seja
confundida com evidência do sensor.

## Passes e tiling

O baseline executa um único pass `full_frame`. `DiscoveryPassConfig` pode adicionar um ou mais
grids de tiles com tamanho, overlap, escala e budget por pass explícitos. As janelas são geradas em
ordem row-major e o último tile de cada eixo é alinhado ao limite da imagem para garantir cobertura
sem produzir coordenadas fora do espaço preparado.

O port `RegionDiscovery` recebe `DiscoveryInput` e devolve `RegionCandidate` mais diagnostics. A
orquestração não contém branches para SAM2, SAM3 ou Florence-2. Propostas locais de tiles são
remapeadas para a imagem preparada, recebem ID prefixado pelo pass e preservam o ID nativo em
provenance. Máscaras inline são expandidas no espaço global; uma máscara externa de tile sem
geometria decodificada é rejeitada porque não pode ser remapeada de forma verificável.

`BorderPolicy.KEEP` mantém propostas que tocam bordas internas. A política
`REJECT_INTERNAL_BORDER` registra `tile_border_truncation` sem apagar a proposta dos diagnostics.
Deduplicação entre passes não ocorre aqui; ela pertence à normalização geométrica.

## Backend SAM2

`Sam2RegionDiscovery` implementa o mesmo port usado pelos demais backends. O adapter recebe
`Sam2Config` validada e um runtime injetado que isola carregamento de checkpoint, Torch e objetos
do SDK. Apenas boxes, máscaras booleanas e scores escalares atravessam essa fronteira interna.

A configuração efetiva inclui checkpoint, versão, device, precision, thresholds e parâmetros do
automatic mask generator. Seu digest determinístico acompanha cada proposta. `predicted_iou` e
`stability_score` mantêm seus nomes e significados SAM2; nenhum deles vira confidence universal.
O runtime recebe o `DiscoveryInput` completo, incluindo constraints explícitas, pass e tile. Erros
de shape ou runtime interrompem a execução, sem fallback silencioso para outro backend.

## Backend SAM3

`Sam3RegionDiscovery` é o adapter planejado para o baseline da Solution 1 e continua substituível
pelo mesmo port. `Sam3Config` torna checkpoint, versão, device, precision, thresholds e estratégia
parte do digest da execução. Estratégias `automatic`, `text_prompt`, `point_grid`, `tracker` e `pcs`
são distintas; `text_prompt` exige prompt, enquanto `automatic` rejeita prompt oculto.

O runtime retorna propostas escalares, warnings e métricas próprias. O adapter preserva query ID,
prompt aplicável e nome/semântica do score em provenance. Texto usado para obter a máscara não é
publicado como `SemanticClaim`. Falha do runtime ou estratégia configurada é propagada; não existe
fallback silencioso para outra estratégia ou backend.

## Backend Florence-2

`Florence2RegionDiscovery` atende somente ao port de descoberta de regiões. Sua configuração
registra checkpoint, versão, device, precision, task, prompt opcional, threshold e generation
settings. O task é obrigatório porque diferentes modos de Florence-2 têm semânticas de proposta
distintas.

O parser interno pode produzir box, máscara opcional, score opcional, texto parseado e diagnostics.
O adapter transforma apenas a geometria em `RegionCandidate`. Task, prompt e texto parseado ficam
como provenance/metadata de descoberta; não geram `SemanticClaim`. Um futuro adapter Florence-2
para interpretação semântica deve implementar outro port, mesmo que compartilhe o runtime carregado.

## Normalização, merge e geometry freeze

`normalize_regions` aplica a mesma política a propostas de SAM2, SAM3, Florence-2 e fakes. A ordem
é: validar geometria, aplicar limites de área, verificar valid/exclusion masks declaradas, detectar
duplicatas por IoU ou containment, aplicar budget e criar `Region2D` imutável. Nenhuma regra usa
label semântico ou compara scores de backends diferentes.

A ordem canônica é pelo `candidate_id`, tornando IDs `region-0001`, `region-0002` e decisões de
budget reproduzíveis. No merge, a primeira geometria canônica permanece como representante e todas
as propostas contribuintes e respectivas provenances são preservadas. A proposta incorporada gera
tanto `MergeDecision` quanto uma rejeição `merged_duplicate`, portanto não desaparece dos
diagnostics.

Os thresholds e budgets vivem em `NormalizationConfig`; seu digest acompanha o resultado. Máscaras
inline são avaliadas pixel a pixel. Uma máscara persistida sem box inspecionável é rejeitada em vez
de receber área ou overlap inventados. Constraints só são aplicadas quando a `PreparedImage` as
declara explicitamente.

## Evidência persistida e diagnostics

`RegionDiscoveryEvidenceWriter` finaliza atomicamente `20-region-discovery/` e recusa sobrescrever
um estágio existente. `outputs/regions.jsonl`, `outputs/metrics.json` e `manifest.json` são
contratuais. O manifest registra schema, backend, digest da política, nível de debug e hash de cada
payload. Consumidores downstream não leem `debug/`.

Os níveis são:

- `none`: somente regiões, métricas e manifest;
- `standard`: prepared-image reference, configuração efetiva, passes/timings, candidates,
  accepted/rejected, merge decisions e overlays SVG;
- `full`: conteúdo standard mais `region.json` e máscara PBM por região inline.

Os overlays usam IDs canônicos e coordenadas da imagem preparada. O formato vetorial mantém a
inspeção disponível sem introduzir uma biblioteca de imagem no domínio. Métricas preservam counts,
motivos de rejeição, distribuição de área, merge ratio, duração por pass, warnings e memória quando
o runtime a reporta.

## Geometry freeze

Depois da normalização, `Region2D` é um value object imutável. Feature extraction, semantic
interpretation, scoring e audit podem referenciar a região, mas não podem alterar seu ID, máscara
ou bounding box. Uma geometria diferente exige outro resultado de percepção ou outra evidência
derivada com identidade própria.

# Validação real parcial do corridor-02 (2026-09-21)

Registro da primeira execução **real** das partes do cenário [`solution-1-canonical` 1.0.0](../end-to-end.md) que não dependem de capability ausente: geometria, associação sensor-mapa e fusão semântica sobre a amostra pequena (janela de 90 s, 20 imagens), mais as checagens entre estágios sobre esses artifacts reais. O relatório de aceitação completo está em [`e2e-real-sample-20260921.acceptance-report.json`](e2e-real-sample-20260921.acceptance-report.json).

**Classe de evidência: `real`, parcial.** Não é a validação da Solution 1: 22 dos 25 gates continuam sem cumprir, e cada um nomeia o motivo.

## O que rodou e o que foi reaproveitado

Estágios reaproveitados **por referência** (somente leitura, nunca copiados) da validação de 2026-09-21; identidades e digests do manifesto estão no relatório:

| Estágio | Origem | Identidade |
|---|---|---|
| ingestion | `SequenceArtifact` pinado no cenário | `e145f73f8d894f18b96ef1f55ca308c2` (manifesto `sha256:74d39984…`, igual ao pino) |
| state_estimation | run `ExternalPose` sobre `corridor-02-gt.txt` (entrada de pose declarada) | `se-corridor-02-extpose-w336-attempt1`, 684 poses, verifica sem problemas |
| visual_perception | run SAM2 + DINOv2 + CLIP sobre as 20 imagens | `run-0005`, **sem interpretação semântica (0 claims)** |

Estágios **produzidos aqui**, em CPU (GPU não usada), em `workspace/corridor-02/validation-e2e-20260921/` (não versionado):

| Estágio | Configuração | Resultado |
|---|---|---|
| geometric_mapping | scans da janela (892; 32 rejeitados explicitamente pelo plano), agregação por scan em voxel de 0,20 m | 4 313 593 pontos, integridade sem problemas |
| sensor_association | 20 imagens, modelo MEI, política de oclusão exploratória, sem canal de features densas | 19 imagens associadas, **1 rejeitada** (`interpolation_gap`), 410 observações espaciais, 332 com geometria |
| semantic_fusion | política baseline, suporte por sobreposição de geometria (mín. 5 elementos, Jaccard 0,3) | 155 suportes, 166 observações excluídas por pouca geometria, **0 hipóteses** |
| cross_stage | `check_cross_stage` sobre os artifacts acima | 17 checagens de lineage, 700 524 de coordenadas, 2 349 de rastreabilidade, 554 de identidade física; **2 achados** |

Contagem de visibilidade do run de associação (19 imagens × 4 313 593 pontos = 81 958 267, e a soma fecha exatamente): 22 046 096 atrás da câmera, 25 177 662 ocluídos, 34 212 346 fora da imagem, 522 163 visíveis.

## Recursos (CPU, sem GPU)

| Estágio | Tempo | Pico de RSS do processo (cumulativo) |
|---|---|---|
| geometric_mapping | ~3 s (construção; reaproveitada depois) | 657 MB |
| sensor_association | 355,5 s | 7 086 MB |
| semantic_fusion | 2,3 s | 7 086 MB |
| cross_stage | 3,1 s | 7 086 MB |

Armazenamento produzido: mapa 265 MB, associação 4,0 MB, fusão 2,8 MB. Não é o perfil de custo do #181: não separa carga de modelo, E/S e inferência, não mede GPU e não cobre os estágios reaproveitados.

## Resultado por gate

| Status | Gates |
|---|---|
| `passed` (evidência real) | `state_estimation.trajectory_coverage`, `sensor_association.projection_validity`, `semantic_fusion.evidence_preservation` |
| `failed` | `cross_stage.lineage_closure` (capabilities: `geometric_mapping`, `sensor_association`) |
| `blocked` | percepção sem claims; gates que exigem reference set anotado (`region_quality`, `semantic_quality`, `reference_recovery`); Entity Resolution, Spatial Relations, artifact final; os três gates `cross_stage.*` restantes, bloqueados por essas mesmas fronteiras |
| `not_evaluated` | integridade completa da sequência (24 GB, não rodada), acurácia da trajetória (não aplicável), frame do mapa (checagem parcial), qualidade de geometria e de associação, custo, reprodutibilidade |

O `passed` da fusão é **vazio para semântica**: sem claims, alternativas e conflitos não foram exercitados; as checagens de rastreabilidade e de identidade física sobre regiões e geometria não são vazias.

## Achados

1. **O `SequenceArtifact` pinado não tem modelo de câmera.** Foi ingerido antes de o MEI ser canônico; a entrada `cal-camera_1_image_raw` tem `camera_model = null`. A associação exige um modelo. Para seguir, o driver construiu, **de forma declarada e hasheada**, um override MEI a partir de `corridor-02-Intrinsics.yaml` (`sha256:8d47e98a…`) e construiu o mapa com ele (a associação recusa uma calibração diferente da do mapa, o que já protege a fronteira). O resultado: a calibração usada pelo mapa e pela associação (`sha256:6b3f6bf0…`) **não é a do artifact de sequência** (`sha256:e62cb707…`), e o gate de lineage falha, corretamente. **Ação:** reingerir a calibração do `corridor-02` com o modelo MEI, o que cria um novo `SequenceArtifact` e exige uma nova versão do cenário (o pino de identidade muda). Até lá, nenhum run canônico real é possível sem essa quebra.
2. **A regra de seleção e a configuração do estimador discordam.** A seleção admite imagens cuja lacuna de pose ao redor é de até 450 ms; o `ExternalPose` declara lacunas acima de 350 ms como lacunas da trajetória, e a interpolação é recusada. Resultado: 1 das 20 imagens (`camera_1_image_raw-009106`) é rejeitada explicitamente na associação (`interpolation_gap`); a seleção registra uma lacuna de 403 ms depois da pose `1646000109628678000`, que cobre essa imagem. O cenário 1.0.0 não fixou essa escolha.
3. **A trajetória `ExternalPose` não grava identidade de calibração** (`calibration_identity = null`), então a linhagem de calibração só é comparável entre mapa e associação.
4. **Falta interpretação semântica canônica.** O run de percepção reaproveitado não tem claims; a fusão real produz suportes sem hipóteses. O runtime Qwen é da milestone de Semantic Interpretation.
5. **Custo da associação.** 20 imagens sobre o mapa agregado de 4,3 M pontos levaram ~6 min e picaram em ~7 GB de RSS do processo (incluindo as máscaras inline do run de percepção, ~20 MB por imagem). A decomposição por causa não foi medida, e a associação sobre o mapa sem agregação (14,4 M pontos, o das validações anteriores) **não foi tentada**; relevante para o #181.
6. **Defeito conhecido (#374).** `SequenceArtifactReader` carrega a sequência inteira; o driver decodifica só a janela com o decodificador privado, uma solução de driver que **não** entra em `src`.
7. **Valores exploratórios.** A agregação (voxel de 0,20 m por scan), a política de oclusão e os limiares de suporte da fusão não estão congelados pelo cenário 1.0.0; uma versão futura deve fixá-los.

## Como reproduzir

Entradas hasheadas: manifesto da sequência, `selection.json` da validação (`sha256:0fe5c5a4…`), `corridor-02-Intrinsics.yaml`, `corridor-02-gt.txt` (`sha256:cddb6739…`) e os manifestos dos runs reaproveitados. Os drivers (`real_chain.py`, `real_report.py`) ficam no workspace local: eles dependem do decodificador privado do defeito #374 e de caminhos absolutos do dataset, então não entram no repositório. O procedimento é: (1) ler a janela pelos registros do índice; (2) montar o plano de geometria com o override declarado e gravar o mapa; (3) associar as imagens ao mapa com a trajetória e o run de percepção reaproveitados; (4) fundir; (5) executar `check_cross_stage` com a identidade de calibração do artifact de sequência; (6) montar o `AcceptanceReport` com `assemble_acceptance_report()`.

Nenhum segredo, token ou quadro saiu da máquina: não houve chamada de API externa.

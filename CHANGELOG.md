# Changelog

As mudanças relevantes do ContextMap2 são registradas aqui, no formato do [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/). As versões seguem [docs/versioning.md](docs/versioning.md); durante a fase de validação a série é `0.x.y`.

Uma entrada só recebe data quando a release é criada. O workflow `Release` recusa uma tag cuja versão não tenha, neste arquivo, a entrada `## [X.Y.Z] - AAAA-MM-DD` nem um `docs/releases/vX.Y.Z.md` que já não seja rascunho.

## [Não lançado]

Estado de `origin/dev` em 2026-09-21. O conteúdo do v0.1.0 ainda não está congelado ([docs/release-v0.1.0.md](docs/release-v0.1.0.md)), e o que outras milestones integrarem antes da release será acrescentado aqui.

### Adicionado

- Ingestion: adapters de bag ROS 1 e ROS 2 (`rosbags`), observações canônicas, calibração (pinhole, fisheye e MEI), sincronização, seleção e replay, provenance, validação e `SequenceArtifact`.
- Visual perception: core de execução, preset canônico versionado, Region Discovery (SAM2, SAM3, Florence-2), Feature Extraction (DINOv2, DINOv3, CLIP, AlphaCLIP), Semantic Interpretation (Qwen, Gemini, Florence-2 atrás de seams injetáveis) e `PerceptionRunArtifact`.
- State estimation: `PoseEstimate` e `Trajectory`, lookup temporal auditável, backends `ExternalPose` e FAST-LIO e `StateEstimationRunArtifact`.
- Geometric mapping: `GeometricMap`, correção de movimento explícita, acumulação com referências estáveis, acesso espacial e `GeometricMapArtifact`.
- Sensor association: `SpatialObservation`, modelos de câmera calibrados, visibilidade e oclusão, features densas e `SensorAssociationRunArtifact`.
- Point representation (opcional): descritor geométrico determinístico, fronteira do PTv3 e `PointRepresentationRunArtifact`.
- Semantic fusion: `FusionSupport`, acumulação baseline e ciente de qualidade e `SemanticFusionRunArtifact`.
- Evaluation: protocolos determinísticos para cada capability acima.
- Empacotamento: extras `ros1`, `ros2` e `vision`; licença como expressão SPDX (`AGPL-3.0-only`); smoke de instalação em ambiente novo; jobs de CI para compatibilidade de Python, pacote e instalação leve.
- Release: workflow com gate de ancestralidade em `main`, reuso da CI, conferência da versão da wheel, checksums e notas validadas.
- Documentação: instalação, licenças de terceiros, estado verificado das configurações do repositório e escopo do v0.1.0.

### Alterado

- O NumPy passou a ser dependência base (`numpy<2.4`); antes só existia nos extras.
- O workflow de release deixou de usar uma action de terceiros para criar a release e passou a publicar somente o que a CI construiu e testou.

### Pendente para o v0.1.0

Runtime e CLI, semantic mapping, entity resolution, spatial relations, `ContextMapArtifact` (schema, escrita, leitura e validação), execução end-to-end e o relatório de aceitação. Veja o checklist em [docs/release-v0.1.0.md](docs/release-v0.1.0.md).

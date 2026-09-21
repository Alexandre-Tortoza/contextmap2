# Runtime

## Responsabilidade

Resolver a configuração de uma execução, selecionar e construir as implementações concretas das capabilities, orquestrar os stages e registrar o que foi executado. **Compõe e executa; não decide ciência.** Segmentação, projeção, fusão e as demais regras de domínio continuam nas capabilities que as possuem.

```mermaid
flowchart LR
    FILES["Arquivos .json / .toml"] --> RES["resolve_effective_config()"]
    PROFILE["Perfil canonical/1"] --> RES
    OVR["Overrides<br/>dotted.path=valor"] --> RES
    RES --> EFF["EffectiveConfig<br/>config + digest + sources"]
    EFF --> CHK["check_selection()<br/>check_availability()"]
    ENV["Ambiente"] -. só segredos .-> SEC["resolve_secrets()"]
    EFF -. planejado .-> COMP["Composition root"]
    COMP -. planejado .-> DAG["DAG, reuse, lifecycle, CLI"]
```

## O que este módulo explicitamente não possui

- lógica de segmentação, interpretação semântica, projeção, fusão, resolução de entidades ou relações espaciais;
- limiares científicos de outra capability: eles vêm da configuração do usuário e são validados por quem os possui;
- o formato do `ContextMapArtifact`: a versão do schema de configuração é independente da versão do schema do mapa.

## Estado implementado

Existe a **configuração versionada e a resolução da configuração efetiva** (issue #161): o catálogo estático de stages, pontos de variação e backends; o perfil `canonical/1`; a precedência perfil < arquivos < overrides; o digest determinístico; a persistência atômica de `effective_config.json`; a resolução de segredos somente a partir do ambiente; e as verificações de completude e de disponibilidade que rodam antes de qualquer execução pesada.

A composition root, o DAG, o reuse, a seleção de runs, a CLI e o lifecycle são as demais issues da milestone #17 e ainda não existem. Detalhes em [`configuration.md`](configuration.md).

## Contratos públicos

- `resolve_effective_config()`, `EffectiveConfig`, `ConfigurationSource` — a configuração efetiva, seu digest e as camadas que a produziram.
- `RuntimeConfig`, `PipelineConfig`, `ComponentConfig`, `InputsConfig`, `ResourcesConfig`, `PoliciesConfig` — o modelo resolvido, separado por preocupação.
- `parse_override()` — lê um override `dotted.path=valor`.
- `check_selection()`, `check_availability()`, `ConfigProblem`, `ConfigurationError` — problemas reportados de forma explícita, todos de uma vez.
- `resolve_secrets()`, `ResolvedSecrets` — segredos em memória, nunca persistidos nem impressos.
- `write_effective_config()`, `read_effective_config()`, `EFFECTIVE_CONFIG_FILENAME` — persistência e verificação do documento.
- `CANONICAL_PROFILE_ID`, `CONFIG_SCHEMA_VERSION`, `DEBUG_LEVELS` — identidades e constantes.
- `BackendSpec`, `ComponentSpec`, `StageDeclaration`, `RuntimePreset` — o catálogo estático.

## Módulos consumidos

Nenhum em produção: o catálogo é dado puro e não importa backend algum. Os testes verificam que as identidades de política e o canal de evidência que ele nomeia ainda existem em `contextmap.semantic_fusion`.

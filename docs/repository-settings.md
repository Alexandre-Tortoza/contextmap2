# Configuração do repositório

Este documento registra as configurações esperadas do GitHub para que possam ser auditadas junto ao código.

## Metadados

Descrição:

```text
Persistent 3D contextual map generation from synchronized robotic sensor data.
```

Tópicos sugeridos:

```text
robotics
3d-mapping
semantic-mapping
lidar
computer-vision
ros2
open-vocabulary
vision-language-models
scene-graphs
research
```

## Política de merge

Configuração recomendada:

- branch padrão: `main`;
- squash merge: habilitado;
- rebase merge: habilitado;
- merge commit: desabilitado;
- exclusão automática da branch de origem após merge: habilitada;
- auto-merge: opcional depois que os checks obrigatórios estiverem estáveis.

## Política de branches

O fluxo oficial é:

```text
<type>/<issue-number>-<slug>
    ↓
milestone/<milestone-slug>
    ↓
dev
    ↓
main
```

### `main`

- recebe mudanças apenas por pull request;
- origem esperada: `dev`;
- CI obrigatório antes do merge;
- conversas de review devem estar resolvidas;
- force push bloqueado;
- exclusão bloqueada.

### `dev`

- integra milestones concluídas;
- origem esperada: `milestone/*`;
- CI obrigatório antes do merge;
- force push bloqueado;
- exclusão bloqueada quando possível.

### `milestone/*`

- criada a partir de `dev`;
- integra somente branches das issues pertencentes à milestone;
- recebe PRs no formato `<type>/<issue-number>-<slug>`;
- não recebe implementação direta;
- deve passar CI antes de ser promovida para `dev`.

### Branches de issue

Branches de issue são criadas a partir da branch da milestone e seguem:

```text
<type>/<issue-number>-<slug>
```

Tipos permitidos:

```text
feat
fix
research
experiment
refactor
test
docs
ci
chore
```

A branch deve abrir PR para a branch da milestone correspondente, nunca diretamente para `dev` ou `main`.

### `qa`

A branch `qa` pode permanecer temporariamente no repositório por histórico ou compatibilidade, mas não faz parte do fluxo padrão e não deve ser usada como etapa de promoção.

## Branch policy automatizada

O workflow `Branch policy` deve validar pelo menos:

```text
issue branch -> milestone/*
milestone/*  -> dev
dev          -> main
```

Também deve rejeitar branch de issue que não siga o padrão:

```text
<type>/<issue-number>-<slug>
```

## Checks obrigatórios

Quando regras de proteção estiverem habilitadas, os checks mínimos devem incluir:

```text
CI / quality
Branch policy / validate
```

CodeQL deve permanecer habilitado para `main`. Durante a fase inicial de pesquisa, ele não precisa bloquear toda mudança intermediária, a menos que a política de segurança exija.

## Pull requests

PRs devem usar Conventional Commits no título e o template do repositório.

O template deve exigir:

- issue relacionada;
- milestone e branch de destino;
- resumo e motivação;
- validação;
- impacto em artefatos e contratos;
- impacto arquitetural e documental;
- checklist de qualidade.

## Releases

Somente `main` recebe tags de release.

Tags seguem:

```text
vMAJOR.MINOR.PATCH
```

A série `v0.x.y` permanece como série de validação enquanto os contratos externos ainda estiverem evoluindo.

## Segurança

Recursos recomendados:

- Dependabot alerts e security updates;
- secret scanning quando disponível;
- private vulnerability reporting;
- CodeQL scanning;
- regras de proteção para `main`, `dev` e, quando viável, padrão `milestone/*`.

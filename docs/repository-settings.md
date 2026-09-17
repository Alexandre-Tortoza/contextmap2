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

O fluxo oficial para desenvolvimento humano é:

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
- origem normal: `milestone/*`;
- aceita PRs automáticos `dependabot/*` somente quando o ator é `dependabot[bot]`;
- CI obrigatório antes do merge;
- force push bloqueado;
- exclusão bloqueada quando possível.

A exceção do Dependabot existe apenas para manutenção automatizada de dependências. Ela não autoriza branches humanas a ignorarem milestones e nunca permite promoção direta para `main`.

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

### Dependabot

O arquivo `.github/dependabot.yml` configura `dev` como `target-branch` para atualizações de Python e GitHub Actions.

O caminho automatizado é:

```text
dependabot/*
    ↓
dev
    ↓
main
```

Esse caminho continua sujeito a CI e demais checks aplicáveis. A branch policy valida também o ator do PR para impedir que uma branch humana com prefixo `dependabot/` utilize a exceção.

### `qa`

A branch `qa` pode permanecer temporariamente no repositório por histórico ou compatibilidade, mas não faz parte do fluxo padrão e não deve ser usada como etapa de promoção.

## Políticas automatizadas

O workflow `Branch policy` valida:

```text
issue branch -> milestone/*
milestone/*  -> dev
dependabot/* -> dev   # somente dependabot[bot]
dev          -> main
```

Ele também rejeita branches de issue fora do padrão:

```text
<type>/<issue-number>-<slug>
```

O workflow `Commit policy` valida todos os commits de um pull request e exige:

- subject compatível com Conventional Commits;
- corpo não vazio com contexto da alteração.

O workflow `CI` executa quality gates em pull requests e pushes de `main`, `dev` e `milestone/*`.

## Checks obrigatórios

Quando regras de proteção estiverem habilitadas, os checks mínimos devem incluir:

```text
CI / quality
Branch policy / validate
Commit policy / validate
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

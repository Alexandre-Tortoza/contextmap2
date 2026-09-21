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

O workflow `CI` executa em pull requests e pushes de `main`, `dev` e `milestone/*` e também é reutilizado pelo workflow de release. Seus jobs são `quality` (lint, formatação, mypy e testes em Python 3.11), `python-compatibility` (testes em 3.12 a 3.14), `package` (sdist e wheel, `twine check` e smoke de instalação em ambientes novos) e `lightweight-install` (suíte contra a wheel instalada só com NumPy e os extras `ros1`/`ros2`). Nenhum job novo altera o nome de `quality`.

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

## Estado verificado em 2026-09-21

Este documento descreve as configurações **esperadas**. A tabela abaixo confronta cada uma com o que a API do GitHub devolveu em 2026-09-21, por leitura (`GET`) com as credenciais do mantenedor; nenhuma configuração foi alterada. Uma configuração que a API não permite ler com esse acesso é marcada como não verificável, e nenhuma é dada como ativa sem evidência.

| Configuração | Esperado | Observado | Situação |
| --- | --- | --- | --- |
| Branch `main` | existe, é o padrão e origem das tags | **não existe** no remoto; o padrão é `dev` | divergente, e bloqueia a release: o gate do workflow `Release` exige a tag em `main` |
| Proteção de `main` e `dev` | PR obrigatório, CI obrigatório, sem force push | `dev` sem proteção (a API responde "Branch not protected"); nenhum ruleset (`[]`) | **não ativa** |
| Checks obrigatórios (`CI / quality`, `Branch policy / validate`, `Commit policy / validate`) | exigidos por regra de proteção | sem regra de proteção, nada os exige | **não ativos**; os workflows existem e rodam |
| Métodos de merge | squash e rebase; merge commit desabilitado | squash, rebase **e** merge commit habilitados | divergente; o fluxo de milestones usa merge commit, então o documento ou a configuração precisa mudar |
| Excluir branch após o merge | habilitado | desabilitado | divergente |
| Auto-merge | opcional | desabilitado | consistente |
| Descrição e tópicos | descrição e tópicos sugeridos acima | descrição vazia; nenhum tópico | divergente |
| Tags | somente em `main` | nenhuma tag existe | consistente com "sem release ainda" |
| Secret scanning e push protection | recomendados | habilitados | consistente |
| Dependabot alerts e security updates | recomendados | security updates desabilitado; o endpoint de alertas responde 404 (não habilitado); a atualização de versões por `.github/dependabot.yml` está ativa | divergente |
| Private vulnerability reporting | recomendado | desabilitado | divergente |
| CodeQL | habilitado | workflow ativo, análises existentes, nenhum alerta aberto | consistente |
| Actions | permissões mínimas | Actions habilitadas, todas as actions permitidas, token padrão somente leitura, sem fixação obrigatória por SHA | parcial |
| Licença detectada | AGPL-3.0 | AGPL-3.0 | consistente |

Ações que dependem do mantenedor e que este repositório não executa: criar `main` (e decidir se ela passa a ser a branch padrão), ativar proteção ou rulesets com os checks acima, decidir os métodos de merge, preencher descrição e tópicos, habilitar o private vulnerability reporting e os Dependabot alerts.

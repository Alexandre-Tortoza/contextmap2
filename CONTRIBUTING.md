# Contribuindo

O ContextMap2 está validando o canonical pipeline. Contribuições devem priorizar reprodutibilidade, qualidade mensurável do mapa, rastreabilidade e mudanças pequenas o suficiente para serem avaliadas de forma independente.

As regras completas de desenvolvimento estão em [docs/development.md](docs/development.md).

## Fluxo obrigatório

Toda alteração deve estar associada a uma issue e à milestone responsável pelo trabalho.

```text
issue
  ↓
<type>/<issue-number>-<slug>
  ↓
milestone/<milestone-slug>
  ↓
dev
  ↓
main
```

Regras principais:

- a branch da issue nasce da branch da milestone;
- o PR da issue aponta para `milestone/<milestone-slug>`;
- a branch da milestone nasce de `dev`;
- a milestone concluída abre PR para `dev`;
- `dev` abre PR para `main` após validação integrada;
- `qa` não faz parte do fluxo padrão;
- não fazer push direto em `main`, `dev` ou branches de milestone.

Branches de issue usam:

```text
<type>/<issue-number>-<slug>
```

Prefixos aceitos: `feat/`, `fix/`, `research/`, `experiment/`, `refactor/`, `test/`, `docs/`, `ci/` e `chore/`.

## Commits

Todos os commits seguem Conventional Commits e devem possuir corpo descritivo.

```text
<type>(<scope>): <summary>

<contexto, motivo e impacto da alteração>

Refs: #<issue>
```

Use `BREAKING CHANGE:` quando houver incompatibilidade de contrato.

## Pull requests

Todo PR deve registrar:

- issue relacionada;
- milestone e branch de destino;
- problema ou hipótese;
- abordagem adotada;
- validação realizada;
- impacto arquitetural, de contrato ou de documentação;
- métricas ou artefatos quando houver impacto científico;
- limitações conhecidas quando aplicável.

O título do PR também segue Conventional Commits.

## Quality gates

Antes de solicitar revisão:

```bash
make check
```

## Python

- identificadores e APIs em inglês;
- docstrings em inglês, Google style;
- módulos, classes, funções e métodos públicos devem possuir docstring;
- privados não triviais devem documentar contratos, invariantes, efeitos ou exceções relevantes;
- comentários explicativos ficam em PT-BR e devem explicar principalmente o porquê;
- documentação Markdown fica em PT-BR.

O projeto aplica Clean Code, SOLID, KISS, DRY e YAGNI de forma pragmática. KISS e YAGNI devem evitar abstrações especulativas; DRY não deve forçar generalizações prematuras; SOLID não deve criar camadas sem responsabilidade real.

## Documentação

A documentação é integrada em duas camadas:

- `docs/`, decisões e convenções globais;
- `src/contextmap/<module>/docs/`, documentação específica de domínio e módulo.

`docs/README.md` é o índice global. Cada módulo documentado deve possuir um `docs/README.md` próprio e ser referenciado pelo índice global. Conteúdo transversal deve ser linkado, não copiado.

## Pesquisa

Mudanças de pesquisa devem declarar hipótese, baseline, métrica, configuração e evidência de validação. Um caminho experimental não substitui silenciosamente o baseline validado.

## Releases

Tags são produzidas somente a partir de `main` e seguem [docs/versioning.md](docs/versioning.md).

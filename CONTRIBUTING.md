# Contribuindo

O ContextMap2 está validando a Solution 1. Contribuições devem priorizar reprodutibilidade, qualidade mensurável do mapa, rastreabilidade e mudanças pequenas o suficiente para serem avaliadas de forma independente.

## Fluxo de desenvolvimento

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

Regras obrigatórias:

- a branch de uma issue é criada a partir da branch da milestone;
- o PR da issue sempre aponta para `milestone/<milestone-slug>`;
- a branch da milestone é criada a partir de `dev`;
- somente uma milestone concluída é promovida para `dev`;
- somente `dev` é promovida para `main`;
- `qa` não faz parte do fluxo padrão;
- não fazer push direto em `main`, `dev` ou branches de milestone;
- uma branch de issue não deve agrupar issues independentes.

### Branches de issue

Formato obrigatório:

```text
<type>/<issue-number>-<slug>
```

Exemplos:

```text
feat/38-canonical-sensor-contract
fix/112-preserve-frame-provenance
research/145-temporal-fusion-ablation
docs/187-development-workflow
```

Tipos aceitos:

- `feat/`, novo comportamento ou capacidade;
- `fix/`, correção de defeito;
- `research/`, implementação orientada por hipótese de pesquisa;
- `experiment/`, experimento temporário ou comparativo;
- `refactor/`, mudança estrutural sem alteração intencional de comportamento;
- `test/`, testes, fixtures ou benchmarks;
- `docs/`, documentação;
- `ci/`, automação e integração contínua;
- `chore/`, manutenção do repositório.

Branches de milestone usam:

```text
milestone/<milestone-slug>
```

## Commits

Todos os commits do projeto seguem Conventional Commits.

Formato:

```text
<type>(<scope>): <resumo>

<contexto e motivação da alteração>
<impacto ou decisão relevante>

Refs: #<issue>
```

O corpo do commit é obrigatório. Ele deve registrar contexto suficiente para que a decisão continue compreensível sem depender apenas do diff ou da descrição do PR.

Exemplo:

```text
feat(ingestion): add canonical observation contract

Define backend-agnostic sensor observation models so downstream modules
can consume ROS 1, ROS 2, and dataset inputs through the same boundary.

Refs: #38
```

Use `!` e o footer `BREAKING CHANGE:` quando houver uma alteração incompatível de contrato.

Scopes devem representar o domínio ou módulo afetado, por exemplo `ingestion`, `visual-perception`, `semantic-fusion`, `artifact`, `runtime` ou `docs`.

## Pull requests

PRs de issue devem ser pequenos, focados e apontar para a branch da milestone correspondente.

Todo PR deve registrar:

- issue relacionada;
- milestone e branch de destino;
- problema ou hipótese;
- abordagem adotada;
- como a alteração foi validada;
- métricas ou artefatos quando houver impacto na qualidade do mapa;
- impacto arquitetural, de contrato ou de documentação;
- limitações conhecidas e trabalho posterior quando aplicável.

O título do PR também segue Conventional Commits.

## Quality gates

Antes de solicitar revisão, execute:

```bash
make check
```

A validação padrão inclui formatação, lint, tipagem estática e testes.

## Padrões de código Python

### Idioma

- identificadores, APIs públicas e nomes de símbolos ficam em inglês;
- docstrings ficam em inglês;
- comentários explicativos no código ficam em PT-BR;
- documentação Markdown fica em PT-BR.

### Docstrings

Docstrings são o equivalente adotado pelo projeto para documentação estruturada de código Python.

O estilo obrigatório é Google style.

Devem possuir docstring:

- módulos públicos;
- classes públicas;
- funções públicas;
- métodos públicos;
- funções privadas não triviais quando contrato, invariantes, efeitos colaterais ou exceções não forem evidentes.

Docstrings devem documentar contrato e intenção. Quando aplicável, use seções `Args`, `Returns`, `Raises`, `Yields` e `Examples`.

Exemplo:

```python
def associate_observation(frame: FrameBundle, tolerance_ms: float) -> SpatialObservation:
    """Associate visual evidence with spatial support.

    Args:
        frame: Canonical synchronized sensor observation.
        tolerance_ms: Maximum temporal offset accepted during association.

    Returns:
        Spatial observation preserving source provenance.

    Raises:
        AssociationError: If no spatial support satisfies the configured tolerance.
    """
```

### Comentários

Comentários não devem repetir o que o código já expressa. Use comentários para registrar principalmente:

- por que uma decisão existe;
- uma invariante que não pode ser quebrada;
- um trade-off arquitetural ou científico;
- uma limitação relevante do algoritmo, sensor, dataset ou modelo;
- o motivo de uma solução aparentemente não óbvia.

Quando um contexto maior for necessário antes de uma função ou bloco, um comentário em PT-BR é aceitável, desde que complemente e não duplique a docstring.

## Princípios de design

O projeto aplica Clean Code, SOLID, KISS, DRY e YAGNI de forma pragmática.

- **Clean Code**, nomes devem comunicar intenção e funções devem manter responsabilidade clara.
- **SOLID**, preservar responsabilidade, interfaces pequenas, substituibilidade e direção correta das dependências.
- **KISS**, preferir a solução mais simples que satisfaça o contrato e os requisitos científicos atuais.
- **DRY**, remover duplicação conceitual real, sem criar abstrações prematuras apenas para evitar repetição superficial.
- **YAGNI**, não implementar extensões, camadas ou generalizações sem um requisito ou experimento atual que as justifique.

KISS e YAGNI devem impedir abstrações especulativas. SOLID e DRY não justificam criar cerimônia arquitetural sem benefício mensurável.

Os limites de módulo e seus contratos públicos têm precedência sobre conveniência local.

## Documentação

A documentação possui duas camadas integradas.

### Documentação global

```text
docs/
├── README.md
├── architecture.md
├── development.md
├── repository-settings.md
└── versioning.md
```

Ela registra decisões transversais, arquitetura global, fluxo de desenvolvimento, configuração do repositório e versionamento.

### Documentação por módulo

Cada capability ou domínio implementado deve manter sua documentação específica dentro do próprio módulo:

```text
src/contextmap/<module>/
└── docs/
    ├── README.md
    ├── architecture.md
    ├── contracts.md
    └── pipeline.md
```

Somente arquivos necessários devem existir. Não criar documentos vazios apenas para completar a estrutura.

Regras:

- `docs/README.md` é o índice global e integra a documentação dos módulos;
- `src/contextmap/<module>/docs/README.md` é a entrada da documentação daquele domínio;
- decisões globais não devem ser duplicadas em cada módulo;
- documentação de módulo deve apontar para contratos e documentos globais relacionados;
- dependências documentais entre módulos devem ser feitas por links, não por cópia de conteúdo;
- uma alteração de contrato, arquitetura, pipeline ou comportamento público atualiza código e documentação no mesmo PR;
- diagramas Mermaid devem ser preferidos quando ajudam a representar fluxo, dependências ou estados.

## Mudanças de pesquisa

Mudanças orientadas por pesquisa não devem substituir silenciosamente um baseline validado.

Ao testar um novo método:

1. declare a hipótese ou melhoria esperada;
2. preserve um baseline reproduzível quando aplicável;
3. defina a métrica que pode confirmar ou rejeitar a hipótese;
4. registre configuração, modelo, dataset, seleção e provenance usados;
5. mantenha workarounds específicos de dataset fora do caminho geral até que seu valor seja demonstrado.

## Dependências

Evite dependências pesadas de runtime até que sejam necessárias para um caminho de implementação validado. Dependências específicas de modelos, provedores ou middlewares devem permanecer isoladas dos contratos de domínio.

## Releases

Não criar tags de release para estados experimentais não revisados. Tags são produzidas somente a partir de `main` e seguem a política descrita em [docs/versioning.md](docs/versioning.md).
